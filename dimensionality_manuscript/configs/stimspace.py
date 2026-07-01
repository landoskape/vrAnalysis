"""StimSpaceConfig — stimulus-space cross-validated shared variance analysis."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import torch

from vrAnalysis.processors.placefields import get_placefield, FrameBehavior, Placefield
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.metrics import FractionActive
from vrAnalysis.helpers import reliability_loo

from ..pipeline.base import AnalysisConfigBase
from ..registry import PopulationRegistry, get_activity_parameters, Population
from ..regression_models.hyperparameters import PlaceFieldHyperparameters
from .regression import VALID_SPKS_TYPES


@dataclass(frozen=True)
class StimSpaceConfig(AnalysisConfigBase):
    """Configuration for stimulus-space cross-validated shared variance analysis.

    Parameters
    ----------
    normalize : bool
        Whether to normalize placefield matrices by per-neuron peak response.
    use_fast_sampling : bool
        Whether to use fast sampling for placefield computation.
    reliability_threshold : float or None
        Minimum leave-one-out reliability for neuron inclusion.
    fraction_active_threshold : float or None
        Minimum fraction active for neuron inclusion.
    num_bins : int
        Number of spatial bins for place field computation.
    smooth_width : float or None
        Gaussian smoothing width for place fields.
    spks_type : SpksTypes
        Spike type to use.
    """

    schema_version: str = "v5"
    data_config_name: str = "even"

    activity_parameters_name: str = "raw"
    reliability_fraction_active_thresholds: Optional[tuple[float, float]] = (None, None)
    num_bins: int = 100
    smooth_width: Optional[float] = None
    spks_type: SpksTypes = "sigrebase"
    display_name: ClassVar[str] = "stimspace"

    @staticmethod
    def _param_grid() -> dict:
        return {
            # "spks_type": list(VALID_SPKS_TYPES), # now only use sigrebase! oasis is bad bad bad
            "activity_parameters_name": ["raw", "default"],
            # "reliability_fraction_active_thresholds": [(None, None), (0.2, 0.05)],  # Why didn't I think of this before?
            "smooth_width": [None, 5.0],
        }

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"ap={self.activity_parameters_name}",
            f"rel={self.reliability_fraction_active_thresholds[0]}",
            f"frac={self.reliability_fraction_active_thresholds[1]}",
            f"bins={self.num_bins}",
            f"smooth={self.smooth_width}",
            self.schema_version,
        ]
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Run stimulus-space shared variance analysis on a session."""
        from ..subspace_analysis.stimspace import StimSpaceSubspace

        hyps = PlaceFieldHyperparameters(num_bins=self.num_bins, smooth_width=self.smooth_width)
        ap = get_activity_parameters(self.activity_parameters_name)
        model = StimSpaceSubspace(
            registry,
            activity_parameters=ap,
            reliability_threshold=self.reliability_fraction_active_thresholds[0],
            fraction_active_threshold=self.reliability_fraction_active_thresholds[1],
        )
        metrics = model.get_score(session, spks_type=self.spks_type, hyperparameters=hyps)
        cv_variance_scale = model.compute_cv_variance_scale(session, spks_type=self.spks_type, hyperparameters=hyps)
        for key, val in cv_variance_scale.items():
            metrics[key] = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
        return metrics


# ---------------------------------------------------------------------------
# Helpers for StimSpaceSpectraConfig
# ---------------------------------------------------------------------------


def _to_g(x: torch.Tensor) -> torch.Tensor:
    """Center rows and scale so x @ x.T equals the sample covariance. (N, K) -> (N, K)."""
    return (x - x.mean(dim=1, keepdim=True)) / (x.shape[1] - 1) ** 0.5


def _cvsvd(stim_train: torch.Tensor, stim_test: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Cross-validated singular values of stim_train.T @ proj, scored on stim_test.T @ proj.

    Inputs should be pre-normalized via _to_g.
    """
    cross_train = (stim_train.T @ proj).numpy()
    cross_test = (stim_test.T @ proj).numpy()
    U, _, Vt = np.linalg.svd(cross_train, full_matrices=False)
    return torch.from_numpy(np.sum(U * (cross_test @ Vt.T), axis=0).copy())


def _direct_svd(A: torch.Tensor, B: torch.Tensor, n_components: int | None = None) -> torch.Tensor:
    """Singular values of A.T @ B. Pass n_components to use randomized SVD (large FF matrices)."""
    cross = (A.T @ B).numpy()
    if n_components is not None:
        from sklearn.utils.extmath import randomized_svd as _rsvd

        return torch.from_numpy(_rsvd(cross, n_components=n_components)[1].copy())
    return torch.from_numpy(np.linalg.svd(cross, full_matrices=False, compute_uv=False).copy())


def _pf_position_filter(placefields: list[Placefield]) -> tuple[np.ndarray, np.ndarray]:
    """Intersect valid environments and position bins across all Placefield folds.

    Returns
    -------
    valid_envs : (E,) int array of environment ids shared by every fold.
    valid_bins : (B,) bool mask of bins with no NaN in any fold/env/neuron.
    """
    env_sets = [set(np.asarray(pf.environment, dtype=int).tolist()) for pf in placefields]
    valid_envs = np.array(sorted(set.intersection(*env_sets)), dtype=int)
    if len(valid_envs) == 0:
        raise ValueError("No environments are shared across all placefield folds.")
    envs = tuple(int(e) for e in valid_envs)
    num_bins = placefields[0].count.shape[1]
    valid_bins = np.ones(num_bins, dtype=bool)
    for pf in placefields:
        filtered = pf.filter_by_environment(envs)
        valid_bins &= ~np.any(np.isnan(filtered.placefield), axis=(0, 2))
    if not np.any(valid_bins):
        raise ValueError("No valid position bins across all placefield folds.")
    return valid_envs, valid_bins


def _pf_to_matrix(placefield: Placefield, valid_envs: np.ndarray, valid_bins: np.ndarray) -> torch.Tensor:
    """Extract an aligned (N, envs*bins) float32 tensor from a Placefield fold."""
    rows = []
    for env in valid_envs:
        idx = placefield.environment == env
        rows.append(placefield.placefield[idx][:, valid_bins, :])  # (1, valid_bins, N)
    aligned = np.concatenate(rows, axis=0)  # (envs, valid_bins, N)
    return torch.tensor(aligned.reshape(-1, aligned.shape[2]).T, dtype=torch.float32)  # (N, envs*bins)


def _select_rois(
    session: B2Session,
    registry: PopulationRegistry,
    population: Population,
    frame_behavior: FrameBehavior,
    neuron_data: np.ndarray,
    dist_edges: np.ndarray,
    rel_thresh: float | None,
    frac_thresh: float | None,
) -> np.ndarray:
    """Select reliable and active ROIs via leave-one-out reliability and fraction active."""
    idx_keep = np.ones(neuron_data.shape[0], dtype=bool)
    if rel_thresh is None and frac_thresh is None:
        return idx_keep

    full_split = registry.time_split["full"]
    full_data = population.apply_split(neuron_data, full_split, prefiltered=False)
    full_fb = frame_behavior.filter(population.get_split_times(full_split, within_idx_samples=False))
    _all_trials = get_placefield(
        full_data.T.numpy(),
        full_fb,
        dist_edges=dist_edges,
        average=False,
        use_fast_sampling=True,
        session=session,
    )
    _by_env = [_all_trials.filter_by_environment(env) for env in session.environments]
    _pf_data = [np.transpose(pf.placefield, (2, 0, 1)) for pf in _by_env]  # list of (N, trials, bins)
    if rel_thresh is not None:
        idx_keep &= np.logical_or.reduce([reliability_loo(d) > rel_thresh for d in _pf_data])
    if frac_thresh is not None:
        idx_keep &= np.logical_or.reduce(
            [
                FractionActive.compute(
                    d,
                    activity_axis=2,
                    fraction_axis=1,
                    activity_method="rms",
                    fraction_method="participation",
                )
                > frac_thresh
                for d in _pf_data
            ]
        )
    return idx_keep


@dataclass(frozen=True)
class StimSpaceSpectraConfig(AnalysisConfigBase):
    """SS / SF / FF cross-covariance spectra from 4 independent session draws.

    Computes five spectral estimators using the 4 equal time-splits:
    - ss_cv, sf_cv   : cross-validated singular values (4 combinations of 3 draws)
    - ss_direct, sf_direct, ff : direct singular values (all 6 draw pairs)

    Parameters
    ----------
    smooth_train : float or None
        Gaussian smooth width for placefield averages defining CV subspace directions.
    smooth_test : float or None
        Gaussian smooth width for all other placefield inputs (CV scoring side,
        both sides of direct estimators).
    """

    schema_version: str = "v1"
    data_config_name: str = "even"
    activity_parameters_name: str = "raw"
    reliability_fraction_active_thresholds: Optional[tuple[float, float]] = (None, None)
    num_bins: int = 100
    smooth_widths: tuple[Optional[float], Optional[float]] = (None, None)
    spks_type: SpksTypes = "sigrebase"
    display_name: ClassVar[str] = "stimspace_spectra"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "activity_parameters_name": ["raw", "default"],
            "smooth_widths": [(None, None), (5.0, 5.0), (5.0, None)],
        }

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"ap={self.activity_parameters_name}",
            f"rel={self.reliability_fraction_active_thresholds[0]}",
            f"frac={self.reliability_fraction_active_thresholds[1]}",
            f"bins={self.num_bins}",
            f"strain={self.smooth_widths[0]}",
            f"stest={self.smooth_widths[1]}",
            self.schema_version,
        ]
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        population, frame_behavior = registry.get_population(session, spks_type=self.spks_type)
        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)
        ap = get_activity_parameters(self.activity_parameters_name)
        smooth_train, smooth_test = self.smooth_widths

        neuron_data = population.data[population.idx_neurons]  # (N_all, T_total)

        rel_thresh, frac_thresh = self.reliability_fraction_active_thresholds
        idx_keep = _select_rois(
            session,
            registry,
            population,
            frame_behavior,
            neuron_data,
            dist_edges,
            rel_thresh,
            frac_thresh,
        )

        neuron_data = neuron_data[idx_keep]
        n_neurons = neuron_data.shape[0]

        activities, pf_train_list, pf_test_list = [], [], []
        for time_idx in range(4):
            data = population.apply_split(
                neuron_data,
                time_idx,
                prefiltered=False,
                scale=ap.scale,
                scale_type=ap.scale_type,
                pre_split=ap.presplit,
            )
            fb_split = frame_behavior.filter(population.get_split_times(time_idx, within_idx_samples=False))
            spks_np = data.T.numpy()
            activities.append(data)
            pf_train_list.append(
                get_placefield(
                    spks_np,
                    fb_split,
                    dist_edges,
                    average=True,
                    smooth_width=smooth_train,
                    zero_to_nan=True,
                    use_fast_sampling=True,
                    session=session,
                )
            )
            pf_test_list.append(
                get_placefield(
                    spks_np,
                    fb_split,
                    dist_edges,
                    average=True,
                    smooth_width=smooth_test,
                    zero_to_nan=True,
                    use_fast_sampling=True,
                    session=session,
                )
            )

        valid_envs, valid_bins = _pf_position_filter(pf_train_list + pf_test_list)

        sm_train = [_to_g(_pf_to_matrix(pf, valid_envs, valid_bins)) for pf in pf_train_list]
        sm_test = [_to_g(_pf_to_matrix(pf, valid_envs, valid_bins)) for pf in pf_test_list]
        g_data = [_to_g(da) for da in activities]

        combos3 = list(itertools.combinations(range(4), 3))  # 4 combos for CV estimators
        pairs = list(itertools.combinations(range(4), 2))  # 6 pairs for direct estimators

        # Randomized SVD n_components is capped at min(T_i, T_j) for each pair's cross matrix.
        # Time splits can differ by a sample or two, so cap by the smallest split up front to
        # keep every ff entry the same length (otherwise torch.stack fails below).
        min_samples = min(g.shape[1] for g in g_data)
        ff_components = min(n_neurons, min_samples)

        ss_cv_terms = [_cvsvd(sm_train[i], sm_test[j], sm_test[k]) for i, j, k in combos3]
        sf_cv_terms = [_cvsvd(sm_train[i], sm_test[j], g_data[k]) for i, j, k in combos3]
        ss_dir_terms = [_direct_svd(sm_test[i], sm_test[j]) for i, j in pairs]
        sf_dir_terms = [_direct_svd(sm_test[i], g_data[j]) for i, j in pairs]
        ff_terms = [_direct_svd(g_data[i], g_data[j], ff_components) for i, j in pairs]

        ss_cv = torch.mean(torch.stack(ss_cv_terms), dim=0)
        sf_cv = torch.mean(torch.stack(sf_cv_terms), dim=0)
        ss_dir = torch.mean(torch.stack(ss_dir_terms), dim=0)
        sf_dir = torch.mean(torch.stack(sf_dir_terms), dim=0)
        ff = torch.mean(torch.stack(ff_terms), dim=0)

        return {
            "ss_cv": ss_cv.cpu().numpy(),
            "sf_cv": sf_cv.cpu().numpy(),
            "ss_direct": ss_dir.cpu().numpy(),
            "sf_direct": sf_dir.cpu().numpy(),
            "ff": ff.cpu().numpy(),
        }
