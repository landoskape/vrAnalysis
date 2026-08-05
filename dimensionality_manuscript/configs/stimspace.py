"""StimSpaceConfig — stimulus-space cross-validated shared variance analysis."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import ClassVar, Optional

import numpy as np
import torch

from dimilibi import make_time_splits, PCA, SVCA
from dimilibi.helpers import vector_correlation

from vrAnalysis.processors.placefields import get_placefield, get_placefield_prediction, FrameBehavior, Placefield
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.metrics import FractionActive
from vrAnalysis.helpers import reliability_loo, stable_hash

from ..pipeline.base import AnalysisConfigBase
from ..registry import PopulationRegistry, get_activity_parameters, Population
from ..regression_models.hyperparameters import PlaceFieldHyperparameters
from ..env_order import load_env_order, MAX_ENV_SLOTS
from ..subspace_analysis.subspaces import _paired_placefield_matrices, _residualize_activity
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
            "activity_parameters_name": ["raw", "default", "std"],
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


def _cvpca_svd(pf_train: torch.Tensor, pf_test1: torch.Tensor, pf_test2: torch.Tensor) -> torch.Tensor:
    """cvPCA-style cross-validated spectrum matching ``CVPCA(on_stimuli=True)``.

    Spatial eigenbasis from one (smoothed) train fold, covariance across neurons of two
    disjoint (unsmoothed) test folds projected onto that basis. All inputs are raw aligned
    (N, K) placefield matrices (i.e. NOT passed through ``_to_g``): this estimator centers
    over neurons (per-position mean removed), which is the centering axis cvpca uses, rather
    than the per-neuron centering of ``_to_g``.
    """

    def _center_neurons(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=0, keepdim=True)

    train = _center_neurons(pf_train)  # (N, K)
    _, _, Vt = torch.linalg.svd(train, full_matrices=False)
    v = Vt.transpose(0, 1)  # (K, k) spatial eigenvectors
    proj1 = _center_neurons(pf_test1) @ v  # (N, k) per-neuron scores
    proj2 = _center_neurons(pf_test2) @ v
    return vector_correlation(proj1, proj2, covariance=True, dim=0, center=True)


def _direct_svd(A: torch.Tensor, B: torch.Tensor, n_components: int | None = None) -> torch.Tensor:
    """Singular values of A.T @ B. Pass n_components to use randomized SVD (large FF matrices)."""
    cross = (A.T @ B).numpy()
    if n_components is not None:
        from sklearn.utils.extmath import randomized_svd as _rsvd

        return torch.from_numpy(_rsvd(cross, n_components=n_components)[1].copy())
    return torch.from_numpy(np.linalg.svd(cross, full_matrices=False, compute_uv=False).copy())


@dataclass(frozen=True)
class _ResidualFold:
    """One held-out fold after subtracting a PF prediction fit on the complementary folds."""

    vr: torch.Tensor
    """Residual VR activity, containing only frames with a valid PF prediction (N, T_vr)."""

    func: torch.Tensor
    """Residual func activity (N, T_func): ``vr`` followed by unchanged ITI frames when requested."""

    frame_behavior: FrameBehavior
    """Frame behavior aligned row-for-column with ``vr``."""


def _concat_frame_behavior(parts: list[FrameBehavior]) -> FrameBehavior:
    """Concatenate disjoint fold behaviors while preserving their original full-session indices."""
    return FrameBehavior(
        position=np.concatenate([part.position for part in parts]),
        speed=np.concatenate([part.speed for part in parts]),
        environment=np.concatenate([part.environment for part in parts]),
        trial=np.concatenate([part.trial for part in parts]),
        reward_delivery=np.concatenate([part.reward_delivery for part in parts]),
        reward_omitted=np.concatenate([part.reward_omitted for part in parts]),
        idx=np.concatenate([part.idx for part in parts]),
    )


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


def _pf_valid_bins_for_env(placefields: list[Placefield], env: int) -> np.ndarray:
    """Bins with no NaN in any fold for a single environment.

    Single-environment analog of the ``valid_bins`` produced by ``_pf_position_filter``:
    a bin is dropped only when it is missing for *this* environment in some fold, not when
    another environment lacks it. ``env`` must be present in every fold (guaranteed when it
    comes from ``_pf_position_filter``'s ``valid_envs``).
    """
    num_bins = placefields[0].count.shape[1]
    valid_bins = np.ones(num_bins, dtype=bool)
    for pf in placefields:
        filtered = pf.filter_by_environment((int(env),))
        valid_bins &= ~np.any(np.isnan(filtered.placefield), axis=(0, 2))
    return valid_bins


def _select_rois(
    session: B2Session,
    registry: PopulationRegistry,
    population: Population,
    frame_behavior: FrameBehavior,
    neuron_data: np.ndarray,
    dist_edges: np.ndarray,
    rel_thresh: float | None,
    frac_thresh: float | None,
    filter_by_env: int | None = None,
) -> np.ndarray:
    """Select reliable and active ROIs via leave-one-out reliability and fraction active.

    When ``filter_by_env`` is given, reliability / fraction-active are computed for that
    single environment only (rather than OR-ed across every environment), so the returned
    mask keeps neurons reliable/active in that environment. With no thresholds set the mask
    is all-True regardless of ``filter_by_env``.
    """
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
    _envs = [int(filter_by_env)] if filter_by_env is not None else list(session.environments)
    _by_env = [_all_trials.filter_by_environment(env) for env in _envs]
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

    Computes seven spectral estimators using the 4 equal time-splits:
    - ss_cv, sf_cv   : cross-validated singular values (4 combinations of 3 draws)
    - ss_cvpca       : cvPCA-style spectrum (neuron-centered, single-fold spatial basis,
                       two disjoint test folds), matching CVPCA(on_stimuli=True)
    - ss_direct, sf_direct, ff : direct singular values (all 6 draw pairs)
    - ffres          : direct singular values of held-out residual activity (all 6 draw pairs);
                       for each pair, its PF prediction is fit jointly on the other two folds

    Parameters
    ----------
    smooth_train : float or None
        Gaussian smooth width for placefield averages defining CV subspace directions.
    smooth_test : float or None
        Gaussian smooth width for all other placefield inputs (CV scoring side,
        both sides of direct estimators).
    include_iti : bool
        If False (default), the func side (sf / ff estimators) uses only the VR-running
        frames (``idx_samples``), matching the placefield frames. If True, each func fold
        additionally includes that fold's share of the ITI / non-``idx_samples`` frames,
        stacked at the end. Placefields (and therefore all ss estimators) always use
        VR-only frames.
    """

    schema_version: str = "v5"
    data_config_name: str = "even"
    activity_parameters_name: str = "raw"
    reliability_fraction_active_thresholds: Optional[tuple[float, float]] = (None, None)
    num_bins: int = 100
    smooth_widths: tuple[Optional[float], Optional[float]] = (None, None)
    spks_type: SpksTypes = "sigrebase"
    include_iti: bool = False
    display_name: ClassVar[str] = "stimspace_spectra"

    _result_handling = {
        "added_frames": "skip",
        "original_frames": "skip",
    }

    @staticmethod
    def _param_grid() -> dict:
        return {
            "activity_parameters_name": ["raw", "default", "std"],
            "smooth_widths": [(None, None), (5.0, 5.0), (5.0, None)],
            "reliability_fraction_active_thresholds": [(None, None), (0.3, 0.1)],
            "include_iti": [False, True],
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
            f"iti={self.include_iti}",
            self.schema_version,
        ]
        return "_".join(parts)

    @staticmethod
    def _residualize_fold(
        activity: torch.Tensor,
        func_fold: torch.Tensor,
        frame_behavior: FrameBehavior,
        placefield: Placefield,
    ) -> _ResidualFold:
        """Subtract an out-of-fold PF prediction from one held-out activity fold.

        VR frames without a valid prediction (an unseen environment/bin in the two training folds)
        are excluded rather than retained with their PF variance intact. When ``include_iti`` is
        active, the func fold's appended ITI frames have no PF prediction and are retained unchanged.
        """
        eligible = np.isin(frame_behavior.environment, placefield.environment)
        eligible_idx = np.flatnonzero(eligible)
        if eligible_idx.size:
            eligible_fb = frame_behavior.filter(eligible)
            prediction, extras = get_placefield_prediction(placefield, eligible_fb)
            valid_within_eligible = np.asarray(extras["idx_valid"], dtype=bool)
            valid_idx = eligible_idx[valid_within_eligible]
            prediction = prediction[valid_within_eligible]
        else:
            valid_idx = np.array([], dtype=int)
            prediction = np.empty((0, activity.shape[0]), dtype=float)

        prediction_t = torch.as_tensor(prediction.T, dtype=activity.dtype, device=activity.device)
        residual_vr = activity[:, valid_idx] - prediction_t
        residual_fb = frame_behavior.filter(valid_idx)

        # _build_func_folds always leaves the original VR fold first and appends ITI columns.
        iti = func_fold[:, activity.shape[1] :]
        residual_func = torch.cat([residual_vr, iti], dim=1)
        return _ResidualFold(vr=residual_vr, func=residual_func, frame_behavior=residual_fb)

    def _build_residual_pairs(
        self,
        session: B2Session,
        dist_edges: np.ndarray,
        smooth_width: float | None,
        activities: list[torch.Tensor],
        fb_splits: list[FrameBehavior],
        func_folds: list[torch.Tensor],
        pairs: list[tuple[int, int]],
    ) -> dict[tuple[int, int], tuple[_ResidualFold, _ResidualFold]]:
        """Build leakage-free residual folds for every direct-SVD test pair.

        For test folds ``(i, j)``, one PF map is estimated from the concatenated activity and
        behavior of the complementary folds. That same independent map is subtracted from both
        held-out folds before their residual cross-spectrum is measured.
        """
        residual_pairs: dict[tuple[int, int], tuple[_ResidualFold, _ResidualFold]] = {}
        all_folds = set(range(len(activities)))
        for i, j in pairs:
            train_folds = sorted(all_folds - {i, j})
            train_activity = torch.cat([activities[k] for k in train_folds], dim=1)
            train_fb = _concat_frame_behavior([fb_splits[k] for k in train_folds])
            train_pf = get_placefield(
                train_activity.T.numpy(),
                train_fb,
                dist_edges,
                average=True,
                smooth_width=smooth_width,
                zero_to_nan=True,
                use_fast_sampling=True,
                session=session,
            )
            residual_pairs[(i, j)] = (
                self._residualize_fold(activities[i], func_folds[i], fb_splits[i], train_pf),
                self._residualize_fold(activities[j], func_folds[j], fb_splits[j], train_pf),
            )
        return residual_pairs

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        population, frame_behavior = registry.get_population(session, spks_type=self.spks_type)
        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)
        ap = get_activity_parameters(self.activity_parameters_name)
        smooth_train, smooth_test = self.smooth_widths

        neuron_data_full = population.data[population.idx_neurons]  # (N_all, T_total)

        rel_thresh, frac_thresh = self.reliability_fraction_active_thresholds
        idx_keep = _select_rois(
            session,
            registry,
            population,
            frame_behavior,
            neuron_data_full,
            dist_edges,
            rel_thresh,
            frac_thresh,
        )

        neuron_data = neuron_data_full[idx_keep]
        n_neurons = neuron_data.shape[0]

        activities, pf_train_list, pf_test_list, fb_splits = [], [], [], []
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
            fb_splits.append(fb_split)
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

        raw_train = [_pf_to_matrix(pf, valid_envs, valid_bins) for pf in pf_train_list]
        raw_test = [_pf_to_matrix(pf, valid_envs, valid_bins) for pf in pf_test_list]
        sm_train = [_to_g(m) for m in raw_train]
        sm_test = [_to_g(m) for m in raw_test]
        func_folds = self._build_func_folds(session, registry, population, neuron_data, activities, ap)
        added_frames = [fd.shape[1] - activity.shape[1] for fd, activity in zip(func_folds, activities)]
        original_frames = [activity.shape[1] for activity in activities]
        g_data = [_to_g(fd) for fd in func_folds]

        combos3 = list(itertools.combinations(range(4), 3))  # 4 combos for CV estimators
        pairs = list(itertools.combinations(range(4), 2))  # 6 pairs for direct estimators
        residual_pairs = self._build_residual_pairs(
            session,
            dist_edges,
            smooth_train,
            activities,
            fb_splits,
            func_folds,
            pairs,
        )
        min_residual_samples = min(fold.func.shape[1] for folds in residual_pairs.values() for fold in folds)
        if min_residual_samples < 2:
            raise ValueError("Fewer than two valid samples remain in a residual activity fold.")
        g_residual_pairs = {pair: (_to_g(fold_i.func), _to_g(fold_j.func)) for pair, (fold_i, fold_j) in residual_pairs.items()}

        # Randomized SVD n_components is capped at min(T_i, T_j) for each pair's cross matrix.
        # Time splits can differ by a sample or two, so cap by the smallest split up front to
        # keep every ff entry the same length (otherwise torch.stack fails below).
        min_samples = min(g.shape[1] for g in g_data)
        ff_components = min(n_neurons, min_samples)
        ffres_components = min(n_neurons, min_residual_samples)

        ss_cv_terms = [_cvsvd(sm_train[i], sm_test[j], sm_test[k]) for i, j, k in combos3]
        ss_cvpca_terms = [_cvpca_svd(raw_train[i], raw_test[j], raw_test[k]) for i, j, k in combos3]
        sf_cv_terms = [_cvsvd(sm_train[i], sm_test[j], g_data[k]) for i, j, k in combos3]
        ss_dir_terms = [_direct_svd(sm_test[i], sm_test[j]) for i, j in pairs]
        sf_dir_terms = [_direct_svd(sm_test[i], g_data[j]) for i, j in pairs]
        ff_terms = [_direct_svd(g_data[i], g_data[j], ff_components) for i, j in pairs]
        ffres_terms = [_direct_svd(g_residual_pairs[pair][0], g_residual_pairs[pair][1], ffres_components) for pair in pairs]

        ss_cv = torch.mean(torch.stack(ss_cv_terms), dim=0)
        ss_cvpca = torch.mean(torch.stack(ss_cvpca_terms), dim=0)
        sf_cv = torch.mean(torch.stack(sf_cv_terms), dim=0)
        ss_dir = torch.mean(torch.stack(ss_dir_terms), dim=0)
        sf_dir = torch.mean(torch.stack(sf_dir_terms), dim=0)
        ff = torch.mean(torch.stack(ff_terms), dim=0)
        ffres = torch.mean(torch.stack(ffres_terms), dim=0)

        result = {
            "ss_cv": ss_cv.cpu().numpy(),
            "ss_cvpca": ss_cvpca.cpu().numpy(),
            "sf_cv": sf_cv.cpu().numpy(),
            "ss_direct": ss_dir.cpu().numpy(),
            "sf_direct": sf_dir.cpu().numpy(),
            "ff": ff.cpu().numpy(),
            "ffres": ffres.cpu().numpy(),
            "added_frames": added_frames,
            "original_frames": original_frames,
        }
        result.update(
            self._per_env_spectra(
                session,
                registry,
                population,
                frame_behavior,
                ap,
                dist_edges,
                neuron_data_full,
                idx_keep,
                rel_thresh,
                frac_thresh,
                pf_train_list,
                pf_test_list,
                activities,
                fb_splits,
                g_data,
                residual_pairs,
                valid_envs,
                combos3,
                pairs,
            )
        )
        return result

    def _per_env_spectra(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        population: Population,
        frame_behavior: FrameBehavior,
        ap,
        dist_edges: np.ndarray,
        neuron_data_full: np.ndarray,
        idx_keep: np.ndarray,
        rel_thresh: float | None,
        frac_thresh: float | None,
        pf_train_list: list[Placefield],
        pf_test_list: list[Placefield],
        activities: list[torch.Tensor],
        fb_splits: list[FrameBehavior],
        g_data: list[torch.Tensor],
        residual_pairs: dict[tuple[int, int], tuple[_ResidualFold, _ResidualFold]],
        valid_envs: np.ndarray,
        combos3: list[tuple[int, int, int]],
        pairs: list[tuple[int, int]],
    ) -> dict:
        """Per-environment SS / SF / FF / FF-residual spectra, keyed by experience-order slot.

        For each shared environment the place-field (stim) side comes from that single
        environment, and the func (full) side is built two ways: ``full1`` = frames from
        that environment only (VR, no ITI), ``fullall`` = all-environment frames reusing
        the yoked func folds (``g_data``, ITI per ``include_iti``). Every returned key is a
        fixed ``(MAX_ENV_SLOTS, L)`` array with slot 0 = the first environment the mouse
        experienced; unfilled slots and the padded tail are NaN. ``env_slot_ids`` maps each
        slot to its real environment id for this mouse.
        """
        order = load_env_order()
        mouse_order = order.get(session.mouse_name)

        keys = [
            "ss_cv_env",
            "ss_cvpca_env",
            "ss_direct_env",
            "sf_cv_env_full1",
            "sf_cv_env_fullall",
            "sf_direct_env_full1",
            "sf_direct_env_fullall",
            "ff_env_full1",
            "ff_env_full1_fullall",
            "ffres_env_full1",
            "ffres_env_full1_fullall",
        ]
        per_env: dict[str, dict[int, np.ndarray]] = {k: {} for k in keys}

        env_slot_ids = np.full(MAX_ENV_SLOTS, np.nan)
        if mouse_order is not None:
            env_slot_ids[: len(mouse_order)] = mouse_order

        pf_folds = pf_train_list + pf_test_list
        n_reduced = int(idx_keep.sum())

        for env in valid_envs:
            env = int(env)
            if env < 0 or mouse_order is None or env not in mouse_order:
                continue
            slot = mouse_order.index(env)

            # Per-env neuron sub-mask within the reduced (idx_keep) set.
            if rel_thresh is None and frac_thresh is None:
                sub = np.ones(n_reduced, dtype=bool)
            else:
                idx_keep_e = _select_rois(
                    session,
                    registry,
                    population,
                    frame_behavior,
                    neuron_data_full,
                    dist_edges,
                    rel_thresh,
                    frac_thresh,
                    filter_by_env=env,
                )
                sub = idx_keep_e[idx_keep]
            n_e = int(sub.sum())
            if n_e < 2:
                continue

            valid_bins_e = _pf_valid_bins_for_env(pf_folds, env)
            bins_e = int(valid_bins_e.sum())
            if bins_e < 2:
                continue

            env_arr = np.array([env])
            raw_train_e = [_pf_to_matrix(pf, env_arr, valid_bins_e)[sub] for pf in pf_train_list]
            raw_test_e = [_pf_to_matrix(pf, env_arr, valid_bins_e)[sub] for pf in pf_test_list]
            sm_train_e = [_to_g(m) for m in raw_train_e]
            sm_test_e = [_to_g(m) for m in raw_test_e]

            # full1 func side: env-only VR frames. Skip the env if any fold has fewer frames
            # than valid bins (keeps the cross-validated SVD spectra the same length so they
            # stack across folds; a sub-bin-count fold is degenerate anyway).
            f1_cols = [fb.environment == env for fb in fb_splits]
            if min(int(c.sum()) for c in f1_cols) < bins_e:
                continue
            g_f1 = [_to_g(activities[i][sub][:, f1_cols[i]]) for i in range(4)]
            # fullall func side: reuse yoked func folds, row-sliced to env neurons
            # (row-slice commutes with the per-row centering in _to_g).
            g_fall = [g_data[i][sub] for i in range(4)]

            min_f1 = min(g.shape[1] for g in g_f1)
            min_fall = min(g.shape[1] for g in g_fall)
            ff_comp_1 = min(n_e, min_f1)
            ff_comp_mix = min(n_e, min_f1, min_fall)

            ss_cv_e = torch.mean(torch.stack([_cvsvd(sm_train_e[i], sm_test_e[j], sm_test_e[k]) for i, j, k in combos3]), dim=0)
            ss_cvpca_e = torch.mean(torch.stack([_cvpca_svd(raw_train_e[i], raw_test_e[j], raw_test_e[k]) for i, j, k in combos3]), dim=0)
            ss_dir_e = torch.mean(torch.stack([_direct_svd(sm_test_e[i], sm_test_e[j]) for i, j in pairs]), dim=0)
            sf_cv_1_e = torch.mean(torch.stack([_cvsvd(sm_train_e[i], sm_test_e[j], g_f1[k]) for i, j, k in combos3]), dim=0)
            sf_cv_all_e = torch.mean(torch.stack([_cvsvd(sm_train_e[i], sm_test_e[j], g_fall[k]) for i, j, k in combos3]), dim=0)
            sf_dir_1_e = torch.mean(torch.stack([_direct_svd(sm_test_e[i], g_f1[j]) for i, j in pairs]), dim=0)
            sf_dir_all_e = torch.mean(torch.stack([_direct_svd(sm_test_e[i], g_fall[j]) for i, j in pairs]), dim=0)
            ff_1_e = torch.mean(torch.stack([_direct_svd(g_f1[i], g_f1[j], ff_comp_1) for i, j in pairs]), dim=0)
            ff_1_all_e = torch.mean(torch.stack([_direct_svd(g_f1[i], g_fall[j], ff_comp_mix) for i, j in pairs]), dim=0)

            # Pair-specific residual spectra. Each pair was residualized using a PF map fit on
            # exactly the other two folds, so both the env-only and all-env func sides remain
            # independent of their PF estimate. Invalid-prediction VR frames were already dropped;
            # leave these result slots empty only if that leaves a degenerate residual fold.
            residual_counts = []
            residual_inputs = {}
            for pair, (res_i, res_j) in residual_pairs.items():
                idx_env_i = res_i.frame_behavior.environment == env
                idx_env_j = res_j.frame_behavior.environment == env
                r1_i = res_i.vr[sub][:, idx_env_i]
                r1_j = res_j.vr[sub][:, idx_env_j]
                rall_i = res_i.func[sub]
                rall_j = res_j.func[sub]
                residual_inputs[pair] = (r1_i, r1_j, rall_i, rall_j)
                residual_counts.extend([r1_i.shape[1], r1_j.shape[1], rall_i.shape[1], rall_j.shape[1]])

            ffres_1_e = None
            ffres_1_all_e = None
            if residual_counts and min(residual_counts) >= 2:
                g_residual_inputs = {pair: tuple(_to_g(value) for value in values) for pair, values in residual_inputs.items()}
                min_res_1 = min(values[idx].shape[1] for values in g_residual_inputs.values() for idx in (0, 1))
                min_res_all = min(values[idx].shape[1] for values in g_residual_inputs.values() for idx in (2, 3))
                ffres_comp_1 = min(n_e, min_res_1)
                ffres_comp_mix = min(n_e, min_res_1, min_res_all)
                ffres_1_e = torch.mean(
                    torch.stack([_direct_svd(g_residual_inputs[pair][0], g_residual_inputs[pair][1], ffres_comp_1) for pair in pairs]),
                    dim=0,
                )
                ffres_1_all_e = torch.mean(
                    torch.stack([_direct_svd(g_residual_inputs[pair][0], g_residual_inputs[pair][3], ffres_comp_mix) for pair in pairs]),
                    dim=0,
                )

            per_env["ss_cv_env"][slot] = ss_cv_e.cpu().numpy()
            per_env["ss_cvpca_env"][slot] = ss_cvpca_e.cpu().numpy()
            per_env["ss_direct_env"][slot] = ss_dir_e.cpu().numpy()
            per_env["sf_cv_env_full1"][slot] = sf_cv_1_e.cpu().numpy()
            per_env["sf_cv_env_fullall"][slot] = sf_cv_all_e.cpu().numpy()
            per_env["sf_direct_env_full1"][slot] = sf_dir_1_e.cpu().numpy()
            per_env["sf_direct_env_fullall"][slot] = sf_dir_all_e.cpu().numpy()
            per_env["ff_env_full1"][slot] = ff_1_e.cpu().numpy()
            per_env["ff_env_full1_fullall"][slot] = ff_1_all_e.cpu().numpy()
            if ffres_1_e is not None:
                per_env["ffres_env_full1"][slot] = ffres_1_e.cpu().numpy()
                per_env["ffres_env_full1_fullall"][slot] = ffres_1_all_e.cpu().numpy()

        out: dict = {}
        for key, slot_map in per_env.items():
            max_len = max((v.shape[0] for v in slot_map.values()), default=1)
            arr = np.full((MAX_ENV_SLOTS, max_len), np.nan)
            for slot, v in slot_map.items():
                arr[slot, : v.shape[0]] = v
            out[key] = arr
        out["env_slot_ids"] = env_slot_ids
        return out

    def _build_func_folds(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        population: Population,
        neuron_data: np.ndarray,
        activities: list[torch.Tensor],
        ap,
    ) -> list[torch.Tensor]:
        """Build the func-side activity folds (the sf / ff inputs).

        When ``include_iti`` is False, returns ``activities`` unchanged (VR-only frames).
        When True, appends each fold's share of the ITI / non-``idx_samples`` frames to the
        end of that fold's VR activity. ITI frames are the complement of ``population.idx_samples``,
        grouped into contiguous chunks and greedily balanced across the folds. When the session
        has a spontaneous window (``session.has_spontaneous()``), that trailing block is one large
        contiguous chunk; it is first split into ``num_folds`` contiguous sub-chunks so it spreads
        across folds instead of landing entirely in one.

        Parameters
        ----------
        session : B2Session
            Session being processed (used to seed the ITI split reproducibly).
        registry : PopulationRegistry
            Registry providing the chunking parameters for the ITI split.
        population : Population
            Population holding the full activity and ``idx_samples``.
        neuron_data : np.ndarray
            The (N, total_timepoints) selected-neuron activity (full session).
        activities : list of torch.Tensor
            The 4 VR-only activity folds, already scaled per ``ap``.
        ap : ActivityParameters
            Activity scaling parameters, applied consistently to the ITI columns.

        Returns
        -------
        list of torch.Tensor
            One (N, samples) tensor per fold. When ``include_iti`` is True each has its VR
            columns first and its ITI columns stacked at the end.
        """
        if not self.include_iti:
            return activities

        # ITI / other frames = complement of idx_samples over the full timeline.
        mask = torch.ones(population.total_timepoints, dtype=torch.bool)
        mask[population.idx_samples] = False
        iti_abs = torch.nonzero(mask, as_tuple=False).squeeze(1)  # sorted ascending

        # Assign each sample to a pre-existing chunk within iti_abs
        current_fold = 0
        iti_fold = torch.full((len(iti_abs),), current_fold, dtype=torch.long)
        for sample in range(1, len(iti_abs)):
            if iti_abs[sample] - iti_abs[sample - 1] > 1:
                current_fold += 1
            iti_fold[sample] = current_fold

        num_folds = len(activities)

        # A genuine spontaneous window is one large contiguous block of frames at the very end
        # of the session, so it is always the final ITI chunk. Left intact it lands entirely in a
        # single fold, which then dominates the func-side folds unevenly. Split it into num_folds
        # contiguous sub-chunks (relabelled as new chunks); these are pre-assigned one per fold
        # below so every func fold carries an equal share of spontaneous frames. Sessions without a
        # spontaneous window are chunked exactly as before.
        spont_chunk_ids: list[int] = []
        if session.has_spontaneous():
            spont_pos = torch.nonzero(iti_fold == current_fold, as_tuple=False).squeeze(1)  # positions of the spontaneous block in iti_abs
            sub_parts = torch.tensor_split(spont_pos, num_folds)  # contiguous, ~equal sub-chunks
            base_chunk = current_fold
            for offset, part in enumerate(sub_parts[1:], start=1):
                iti_fold[part] = base_chunk + offset  # first sub-chunk keeps base_chunk; the rest become new chunks
            current_fold = base_chunk + len(sub_parts) - 1
            spont_chunk_ids = [base_chunk + i for i in range(len(sub_parts))]

        # Assign chunks to fold with minimum current size to make it even (not all chunks are same size)
        num_chunks = current_fold + 1
        chunk_sizes = torch.bincount(iti_fold, minlength=num_chunks)  # size per chunk label

        # Split the ITI frames into num_folds chunked folds. Seed deterministically per
        # session (the VR splits are cached, but this one is regenerated each call) and
        # restore the global torch RNG state afterward so nothing else is perturbed.
        seed = int(stable_hash(".".join(session.session_name)), 16) % (2**31)
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        chunk_order = torch.randperm(num_chunks)
        fold_totals = torch.zeros(num_folds, dtype=torch.long)
        fold_chunk_ids: list[list[int]] = [[] for _ in range(num_folds)]
        # Pre-assign the spontaneous sub-chunks one per fold so each func fold gets an equal share.
        for fold, cid in enumerate(spont_chunk_ids):
            fold_chunk_ids[fold].append(cid)
            fold_totals[fold] += chunk_sizes[cid]
        # Greedily balance the remaining (interior ITI) chunks by current fold size.
        spont_set = set(spont_chunk_ids)
        for cid in chunk_order.tolist():
            if cid in spont_set:
                continue
            fold = int(fold_totals.argmin())
            fold_chunk_ids[fold].append(cid)
            fold_totals[fold] += chunk_sizes[cid]
        iti_folds = [torch.nonzero(torch.isin(iti_fold, torch.tensor(ids, dtype=torch.long)), as_tuple=False).squeeze(1) for ids in fold_chunk_ids]
        torch.random.set_rng_state(rng_state)

        # Scale the full session once so the ITI columns share the VR columns' scaling.
        scaled_full = population.apply_split(
            neuron_data,
            None,
            scale=ap.scale,
            scale_type=ap.scale_type,
            pre_split=ap.presplit,
            prefiltered=False,
        )

        func_folds = []
        for fold, iti_fold in zip(activities, iti_folds):
            iti_cols = iti_abs[iti_fold]
            func_folds.append(torch.cat([fold, scaled_full[:, iti_cols]], dim=1))
        return func_folds


def _pca_cross(pca_a: "PCA", pca_b: "PCA", keep_components: int) -> np.ndarray:
    """Cross-projection of two PCA component bases, truncated to ``keep_components`` each."""
    components_a = pca_a.get_components()[:, :keep_components]
    components_b = pca_b.get_components()[:, :keep_components]
    return (components_a.T @ components_b).cpu().numpy()


def _residual_activity(
    activity: torch.Tensor,
    frame_behavior: FrameBehavior,
    placefield: Placefield,
) -> Optional[torch.Tensor]:
    """Activity with its (in-sample) place field prediction subtracted.

    Same prediction/masking mechanics as ``StimSpaceSpectraConfig._residualize_fold``: frames
    whose position bin has no prediction (unvisited / NaN'd out by ``zero_to_nan=True``) are
    dropped via ``idx_valid``. Returns ``None`` if fewer than two valid frames remain.
    """
    prediction, extras = get_placefield_prediction(placefield, frame_behavior)
    idx_valid = np.asarray(extras["idx_valid"], dtype=bool)
    if idx_valid.sum() < 2:
        return None
    prediction_t = torch.as_tensor(prediction[idx_valid].T, dtype=activity.dtype, device=activity.device)
    return activity[:, idx_valid] - prediction_t


@dataclass(frozen=True)
class StimSpaceEnvPCAConfig(AnalysisConfigBase):
    """Per-environment PCA of place fields and of full VR activity, plus their unfiltered (all-env) fits.

    For every environment the mouse experienced, fits a ``dimilibi`` ``PCA`` (matching
    ``CovCovSubspace``: uncentered gram matrix by default) on that environment's place field
    matrix, and another on that environment's full activity -- both restricted to the ``full``
    time split's VR-running frames, so ITI and spontaneous-window frames are excluded exactly as
    ``_select_rois`` already excludes them for reliability/fraction-active screening. Neuron
    selection and place field computation otherwise match ``StimSpaceSpectraConfig`` exactly.

    Also fits the same pair (placefield, full activity) once over all environments combined
    (unfiltered), the same way ``CovCovSubspace`` fits its subspaces.

    For each of the per-env pairs, and for the unfiltered pair, additionally measures the cross
    projection of the placefield PCA components onto the full-activity PCA components (see
    ``SubspaceConfig.process`` in ``subspace.py``), truncated to ``keep_components`` before the
    cross so its size doesn't depend on the neuron count.

    Also fits PCA on the residual activity left after subtracting each placefield's own (in-sample)
    prediction -- same prediction/``idx_valid`` masking as ``StimSpaceSpectraConfig._residualize_fold``
    -- both unfiltered and per-env, with the matching cross projection against the placefield PCA.

    Results are packed onto the mouse's experience-order slot axis (see ``env_order``): fixed
    ``MAX_ENV_SLOTS``-length lists of ``PCA`` objects / cross arrays (or ``None`` where the mouse
    never experienced that many distinct environments, or an environment was dropped for having
    too few neurons/frames/bins).
    """

    schema_version: str = "v1"
    data_config_name: str = "even"
    activity_parameters_name: str = "raw"
    reliability_fraction_active_thresholds: Optional[tuple[float, float]] = (None, None)
    num_bins: int = 100
    smooth_width: Optional[float] = None
    spks_type: SpksTypes = "sigrebase"
    centered: bool = False
    display_name: ClassVar[str] = "stimspace_env_pca"

    _result_handling = {
        "pca_placefield": "skip",
        "pca_full": "skip",
        "pca_full_res": "skip",
        "pca_placefield_env": "skip",
        "pca_full_env": "skip",
        "cross_env": "skip",
        "pca_full_res_env": "skip",
        "cross_res_env": "skip",
    }

    @staticmethod
    def _param_grid() -> dict:
        return {
            "activity_parameters_name": ["raw", "default", "std"],
            "smooth_width": [5.0, None],
            "reliability_fraction_active_thresholds": [(None, None), (0.3, 0.1)],
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
        keep_components = 1000
        population, frame_behavior = registry.get_population(session, spks_type=self.spks_type)
        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)
        ap = get_activity_parameters(self.activity_parameters_name)

        neuron_data_full = population.data[population.idx_neurons]  # (N_all, T_total)

        rel_thresh, frac_thresh = self.reliability_fraction_active_thresholds
        idx_keep = _select_rois(
            session,
            registry,
            population,
            frame_behavior,
            neuron_data_full,
            dist_edges,
            rel_thresh,
            frac_thresh,
        )
        neuron_data = neuron_data_full[idx_keep]

        # "full" split, VR-only frames -- excludes ITI / spontaneous windows, same as _select_rois.
        full_split = registry.time_split["full"]
        full_data = population.apply_split(
            neuron_data,
            full_split,
            prefiltered=False,
            scale=ap.scale,
            scale_type=ap.scale_type,
            pre_split=ap.presplit,
        )
        full_fb = frame_behavior.filter(population.get_split_times(full_split, within_idx_samples=False))

        placefield = get_placefield(
            full_data.T.numpy(),
            full_fb,
            dist_edges,
            average=True,
            smooth_width=self.smooth_width,
            zero_to_nan=True,
            use_fast_sampling=True,
            session=session,
        )
        environments = np.asarray(full_fb.environment)

        # Unfiltered (all environments combined), matching CovCovSubspace.fit exactly.
        placefield_extended = torch.tensor(placefield.flattened()).T.contiguous()
        valid_bins = ~torch.any(torch.isnan(placefield_extended), dim=0)
        placefield_extended = placefield_extended[:, valid_bins]
        pca_placefield = PCA(center=self.centered).fit(placefield_extended)
        pca_full = PCA(center=self.centered).fit(full_data)
        cross = _pca_cross(pca_placefield, pca_full, keep_components)

        residual_full = _residual_activity(full_data, full_fb, placefield)
        pca_full_res = PCA(center=self.centered).fit(residual_full) if residual_full is not None else None
        cross_res = _pca_cross(pca_placefield, pca_full_res, keep_components) if pca_full_res is not None else None

        order = load_env_order()
        mouse_order = order.get(session.mouse_name)
        env_slot_ids = np.full(MAX_ENV_SLOTS, np.nan)
        if mouse_order is not None:
            env_slot_ids[: len(mouse_order)] = mouse_order

        pca_placefield_env: list[Optional[PCA]] = [None] * MAX_ENV_SLOTS
        pca_full_env: list[Optional[PCA]] = [None] * MAX_ENV_SLOTS
        cross_env: list[Optional[np.ndarray]] = [None] * MAX_ENV_SLOTS
        pca_full_res_env: list[Optional[PCA]] = [None] * MAX_ENV_SLOTS
        cross_res_env: list[Optional[np.ndarray]] = [None] * MAX_ENV_SLOTS

        if mouse_order is not None:
            for env in np.unique(environments):
                env = int(env)
                if env < 0 or env not in mouse_order:
                    continue
                slot = mouse_order.index(env)
                if slot >= MAX_ENV_SLOTS:
                    continue

                placefield_env = placefield.filter_by_environment(env)
                if len(placefield_env) == 0:
                    continue
                placefield_env_extended = torch.tensor(placefield_env.flattened()).T.contiguous()
                valid_bins_e = ~torch.any(torch.isnan(placefield_env_extended), dim=0)
                placefield_env_extended = placefield_env_extended[:, valid_bins_e]
                if placefield_env_extended.shape[1] < 2:
                    continue

                idx_env = environments == env
                if int(idx_env.sum()) < 2:
                    continue
                full_data_env = full_data[:, idx_env]
                full_fb_env = full_fb.filter(idx_env)

                pca_placefield_env[slot] = PCA(center=self.centered).fit(placefield_env_extended)
                pca_full_env[slot] = PCA(center=self.centered).fit(full_data_env)
                cross_env[slot] = _pca_cross(pca_placefield_env[slot], pca_full_env[slot], keep_components)

                residual_env = _residual_activity(full_data_env, full_fb_env, placefield_env)
                if residual_env is not None:
                    pca_full_res_env[slot] = PCA(center=self.centered).fit(residual_env)
                    cross_res_env[slot] = _pca_cross(pca_placefield_env[slot], pca_full_res_env[slot], keep_components)

        return {
            "pca_placefield": pca_placefield,
            "pca_full": pca_full,
            "cross": cross,
            "pca_full_res": pca_full_res,
            "cross_res": cross_res,
            "pca_placefield_env": pca_placefield_env,
            "pca_full_env": pca_full_env,
            "cross_env": cross_env,
            "pca_full_res_env": pca_full_res_env,
            "cross_res_env": cross_res_env,
            "env_slot_ids": env_slot_ids,
        }


# ---------------------------------------------------------------------------
# Helpers for StimspaceSVCAConfig
# ---------------------------------------------------------------------------


def _filter_placefield_neurons(placefield: Placefield, keep: np.ndarray) -> Placefield:
    """Restrict a ``Placefield`` to a neuron boolean mask along its last (neuron) axis.

    ``count`` is occupancy (envs, bins), shared across neurons, so it is untouched.
    """
    return replace(placefield, placefield=placefield.placefield[:, :, keep])


def _svca_score(
    source_train: torch.Tensor,
    target_train: torch.Tensor,
    source_test: torch.Tensor,
    target_test: torch.Tensor,
    centered: bool,
    num_components: int,
) -> torch.Tensor:
    """Fit SVCA on one fold, score its shared-variance spectrum on another."""
    model = SVCA(centered=centered, num_components=num_components).fit(source_train, target_train)
    return model.score(source_test, target_test)[0]


def _svca_fold_metrics(
    sources: list[torch.Tensor],
    targets: list[torch.Tensor],
    fbs: list[FrameBehavior],
    pf_source_train: list[Placefield],
    pf_target_train: list[Placefield],
    pf_source_test: list[Placefield],
    pf_target_test: list[Placefield],
    n_source: int,
    n_target: int,
    centered: bool,
    max_components: int,
) -> dict[str, torch.Tensor]:
    """Cross-validated SVCA spectra (``ff``/``sf``/``ss``/``ff_res``/``ss_pred``) over fold combinations.

    ``sources``/``targets``/``fbs``/``pf_*`` are one entry per fold (4 folds for the global spectra,
    fewer for a per-environment call whenever a fold has too few environment frames -- callers must
    drop degenerate folds themselves since fold combinatorics need at least 2 (3 for ``ff_res``)
    folds to be meaningful.

    - ``ff``, ``sf``, ``ss`` : every fold pair, fold ``i`` fit / fold ``j`` scored, averaged.
    - ``ff_res`` : every fold triple ``(i, j, k)``; fold ``i``'s place field predicts folds ``j``
      and ``k`` (both held out of the place field fit), residual SVCA fit on ``j``, scored on ``k``.
    - ``ss_pred`` : every fold pair; each fold's own (in-sample) place field prediction is used for
      both the fit and score side, so no third fold is needed.
    """
    num_folds = len(sources)
    pairs = list(itertools.combinations(range(num_folds), 2))
    combos3 = list(itertools.combinations(range(num_folds), 3))

    pf_pair_train = [
        _paired_placefield_matrices(pf_source_train[k], pf_target_train[k], n_source, n_target, nan_safe=False) for k in range(num_folds)
    ]
    pf_pair_test = [_paired_placefield_matrices(pf_source_test[k], pf_target_test[k], n_source, n_target, nan_safe=False) for k in range(num_folds)]

    min_T = min(s.shape[1] for s in sources)
    min_bins_train = min(p[0].shape[1] for p in pf_pair_train)
    min_bins_test = min(p[0].shape[1] for p in pf_pair_test)

    ff_components = min(max_components, n_source, n_target, min_T)
    ff = torch.mean(torch.stack([_svca_score(sources[i], targets[i], sources[j], targets[j], centered, ff_components) for i, j in pairs]), dim=0)

    sf_components = min(max_components, n_source, n_target, min_bins_train, min_T)
    sf = torch.mean(
        torch.stack([_svca_score(pf_pair_train[i][0], pf_pair_train[i][1], sources[j], targets[j], centered, sf_components) for i, j in pairs]),
        dim=0,
    )

    ss_components = min(max_components, n_source, n_target, min_bins_train, min_bins_test)
    ss = torch.mean(
        torch.stack(
            [_svca_score(pf_pair_train[i][0], pf_pair_train[i][1], pf_pair_test[j][0], pf_pair_test[j][1], centered, ss_components) for i, j in pairs]
        ),
        dim=0,
    )

    residual_terms = []
    for i, j, k in combos3:
        (res_source_j, res_target_j), _, _ = _residualize_activity((sources[j], targets[j]), fbs[j], (pf_source_train[i], pf_target_train[i]))
        (res_source_k, res_target_k), _, _ = _residualize_activity((sources[k], targets[k]), fbs[k], (pf_source_train[i], pf_target_train[i]))
        residual_terms.append((res_source_j, res_target_j, res_source_k, res_target_k))
    min_res_T = min(min(t.shape[1] for t in term) for term in residual_terms)
    ffres_components = min(max_components, n_source, n_target, min_res_T)
    ff_res = torch.mean(torch.stack([_svca_score(rsj, rtj, rsk, rtk, centered, ffres_components) for rsj, rtj, rsk, rtk in residual_terms]), dim=0)

    prediction_by_fold = {}
    for k in range(num_folds):
        _, (pred_source_k, pred_target_k), _ = _residualize_activity((sources[k], targets[k]), fbs[k], (pf_source_train[k], pf_target_train[k]))
        prediction_by_fold[k] = (pred_source_k, pred_target_k)
    min_pred_T = min(p[0].shape[1] for p in prediction_by_fold.values())
    sspred_components = min(max_components, n_source, n_target, min_pred_T)
    ss_pred = torch.mean(
        torch.stack(
            [
                _svca_score(
                    prediction_by_fold[i][0],
                    prediction_by_fold[i][1],
                    prediction_by_fold[j][0],
                    prediction_by_fold[j][1],
                    centered,
                    sspred_components,
                )
                for i, j in pairs
            ]
        ),
        dim=0,
    )

    return dict(ff=ff, sf=sf, ss=ss, ff_res=ff_res, ss_pred=ss_pred)


@dataclass(frozen=True)
class StimspaceSVCAConfig(AnalysisConfigBase):
    """Fold cross-validated SVCA (source/target neuron split) shared-variance spectra.

    Reimplements the SVCA analysis previously done via ``SVCASubspace`` (see
    ``subspace_analysis/subspaces.py``) as free functions over the same 4-fold time-split
    combinatorics used by :class:`StimSpaceSpectraConfig`, so every quantity is fit and scored on
    genuinely disjoint data instead of reslicing the output of one globally-fit subspace:

    - ``ff``      : SVCA fit on one fold's raw activity, scored on another fold's raw activity.
                    All 4-choose-2 fold pairs, averaged.
    - ``sf``      : SVCA fit on one fold's place field map, scored on another fold's raw activity.
                    All 4-choose-2 fold pairs, averaged.
    - ``ss``      : SVCA fit on one fold's place field map, scored on another fold's place field map.
                    All 4-choose-2 fold pairs, averaged.
    - ``ff_res``  : SVCA fit on one fold's residual activity (place field prediction from a third,
                    disjoint fold subtracted), scored on another fold's residual activity (same place
                    field). All 4-choose-3 fold triples, averaged.
    - ``ss_pred`` : SVCA fit on one fold's own (in-sample) place field prediction, scored on another
                    fold's own place field prediction. All 4-choose-2 fold pairs, averaged.

    Each has a per-environment counterpart (``*_env``) computed the same way restricted to that
    environment's frames, packed onto the mouse's experience-order slot axis (see ``env_order``).
    """

    schema_version: str = "v1"
    data_config_name: str = "even"
    activity_parameters_name: str = "raw"
    reliability_fraction_active_thresholds: Optional[tuple[float, float]] = (None, None)
    num_bins: int = 100
    smooth_widths: tuple[Optional[float], Optional[float]] = (None, None)
    spks_type: SpksTypes = "sigrebase"
    centered: bool = True
    max_components: int = 300
    display_name: ClassVar[str] = "stimspace_svca"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "activity_parameters_name": ["raw", "default", "std"],
            "smooth_widths": [(None, None), (5.0, 5.0), (5.0, None)],
            "reliability_fraction_active_thresholds": [(None, None), (0.3, 0.1)],
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
            f"centered={self.centered}",
            self.schema_version,
        ]
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        population, frame_behavior = registry.get_population(session, spks_type=self.spks_type)
        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)
        ap = get_activity_parameters(self.activity_parameters_name)
        smooth_train, smooth_test = self.smooth_widths

        neuron_data_full = population.data[population.idx_neurons]
        rel_thresh, frac_thresh = self.reliability_fraction_active_thresholds
        idx_keep = _select_rois(
            session,
            registry,
            population,
            frame_behavior,
            neuron_data_full,
            dist_edges,
            rel_thresh,
            frac_thresh,
        )
        source_idx = population.cell_split_indices[0].numpy()
        target_idx = population.cell_split_indices[1].numpy()
        source_keep = idx_keep[source_idx]
        target_keep = idx_keep[target_idx]
        n_source = int(source_keep.sum())
        n_target = int(target_keep.sum())
        if n_source < 2 or n_target < 2:
            raise ValueError(f"Too few reliable source/target neurons ({n_source}/{n_target}) for session {session.session_name}.")

        pop_kwargs = dict(scale=ap.scale, scale_type=ap.scale_type, pre_split=ap.presplit)

        sources, targets, fbs = [], [], []
        pf_source_train, pf_target_train, pf_source_test, pf_target_test = [], [], [], []
        for time_idx in range(4):
            source_data, target_data = population.get_split_data(time_idx, **pop_kwargs)
            source_data = source_data[source_keep]
            target_data = target_data[target_keep]
            fb = frame_behavior.filter(population.get_split_times(time_idx, within_idx_samples=False))

            sources.append(source_data)
            targets.append(target_data)
            fbs.append(fb)

            pf_kwargs = dict(dist_edges=dist_edges, average=True, zero_to_nan=True, use_fast_sampling=True, session=session)
            pf_source_train.append(get_placefield(source_data.T.numpy(), fb, smooth_width=smooth_train, **pf_kwargs))
            pf_target_train.append(get_placefield(target_data.T.numpy(), fb, smooth_width=smooth_train, **pf_kwargs))
            pf_source_test.append(get_placefield(source_data.T.numpy(), fb, smooth_width=smooth_test, **pf_kwargs))
            pf_target_test.append(get_placefield(target_data.T.numpy(), fb, smooth_width=smooth_test, **pf_kwargs))

        metrics = _svca_fold_metrics(
            sources,
            targets,
            fbs,
            pf_source_train,
            pf_target_train,
            pf_source_test,
            pf_target_test,
            n_source,
            n_target,
            self.centered,
            self.max_components,
        )
        result = {key: value.cpu().numpy() for key, value in metrics.items()}

        result.update(
            self._per_env_svca(
                session,
                registry,
                population,
                frame_behavior,
                dist_edges,
                neuron_data_full,
                idx_keep,
                rel_thresh,
                frac_thresh,
                sources,
                targets,
                fbs,
                pf_source_train,
                pf_target_train,
                pf_source_test,
                pf_target_test,
                source_idx,
                target_idx,
                source_keep,
                target_keep,
            )
        )
        return result

    def _per_env_svca(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        population: Population,
        frame_behavior: FrameBehavior,
        dist_edges: np.ndarray,
        neuron_data_full: np.ndarray,
        idx_keep: np.ndarray,
        rel_thresh: float | None,
        frac_thresh: float | None,
        sources: list[torch.Tensor],
        targets: list[torch.Tensor],
        fbs: list[FrameBehavior],
        pf_source_train: list[Placefield],
        pf_target_train: list[Placefield],
        pf_source_test: list[Placefield],
        pf_target_test: list[Placefield],
        source_idx: np.ndarray,
        target_idx: np.ndarray,
        source_keep: np.ndarray,
        target_keep: np.ndarray,
    ) -> dict:
        """Per-environment ``ff``/``sf``/``ss``/``ff_res``/``ss_pred``, keyed by experience-order slot.

        Same fold combinatorics as the global spectra, restricted to one environment's frames at a
        time; a fold contributes only if it has at least 2 frames (and, for the place-field sides,
        at least 2 valid bins) in that environment, and an environment is skipped entirely for a
        given metric if fewer than 2 (``ff_res``: 3) folds qualify.
        """
        order = load_env_order()
        mouse_order = order.get(session.mouse_name)
        env_slot_ids = np.full(MAX_ENV_SLOTS, np.nan)
        if mouse_order is not None:
            env_slot_ids[: len(mouse_order)] = mouse_order

        keys = ["ff_env", "sf_env", "ss_env", "ff_res_env", "ss_pred_env"]
        per_env: dict[str, dict[int, np.ndarray]] = {k: {} for k in keys}

        if mouse_order is None:
            out: dict = {key: np.full((MAX_ENV_SLOTS, 1), np.nan) for key in keys}
            out["env_slot_ids"] = env_slot_ids
            return out

        environments = np.unique(np.concatenate([np.asarray(fb.environment) for fb in fbs]))
        for env in environments:
            env = int(env)
            if env < 0 or env not in mouse_order:
                continue
            slot = mouse_order.index(env)
            if slot >= MAX_ENV_SLOTS:
                continue

            idx_keep_e = _select_rois(
                session,
                registry,
                population,
                frame_behavior,
                neuron_data_full,
                dist_edges,
                rel_thresh,
                frac_thresh,
                filter_by_env=env,
            )
            # idx_keep_e is a mask over the *raw* (pre-global-filter) neurons; sources[k]/targets[k]
            # and the pf_*_train/test Placefields were already row-filtered by source_keep/target_keep
            # in process(), so re-index the per-env mask through that same filter to align.
            source_keep_e = idx_keep_e[source_idx][source_keep]
            target_keep_e = idx_keep_e[target_idx][target_keep]
            n_source_e = int(source_keep_e.sum())
            n_target_e = int(target_keep_e.sum())
            if n_source_e < 2 or n_target_e < 2:
                continue

            fold_idx = []
            sources_e, targets_e, fbs_e = [], [], []
            pf_source_train_e, pf_target_train_e, pf_source_test_e, pf_target_test_e = [], [], [], []
            for k in range(len(sources)):
                idx_env_k = np.asarray(fbs[k].environment) == env
                if int(idx_env_k.sum()) < 2:
                    continue
                pf_st = _filter_placefield_neurons(pf_source_train[k].filter_by_environment(env), source_keep_e)
                pf_tt = _filter_placefield_neurons(pf_target_train[k].filter_by_environment(env), target_keep_e)
                pf_ss = _filter_placefield_neurons(pf_source_test[k].filter_by_environment(env), source_keep_e)
                pf_ts = _filter_placefield_neurons(pf_target_test[k].filter_by_environment(env), target_keep_e)
                if len(pf_st) == 0 or len(pf_tt) == 0 or len(pf_ss) == 0 or len(pf_ts) == 0:
                    continue

                fold_idx.append(k)
                sources_e.append(sources[k][:, idx_env_k][source_keep_e])
                targets_e.append(targets[k][:, idx_env_k][target_keep_e])
                fbs_e.append(fbs[k].filter(idx_env_k))
                pf_source_train_e.append(pf_st)
                pf_target_train_e.append(pf_tt)
                pf_source_test_e.append(pf_ss)
                pf_target_test_e.append(pf_ts)

            if len(fold_idx) < 3:
                # ff_res needs 3 folds; fewer than that also makes ff/sf/ss/ss_pred combinatorics
                # too thin to be worth reporting for this environment.
                continue

            try:
                metrics_e = _svca_fold_metrics(
                    sources_e,
                    targets_e,
                    fbs_e,
                    pf_source_train_e,
                    pf_target_train_e,
                    pf_source_test_e,
                    pf_target_test_e,
                    n_source_e,
                    n_target_e,
                    self.centered,
                    self.max_components,
                )
            except ValueError:
                continue

            for key, value in metrics_e.items():
                per_env[f"{key}_env"][slot] = value.cpu().numpy()

        out: dict = {}
        for key, slot_map in per_env.items():
            max_len = max((v.shape[0] for v in slot_map.values()), default=1)
            arr = np.full((MAX_ENV_SLOTS, max_len), np.nan)
            for slot, v in slot_map.items():
                arr[slot, : v.shape[0]] = v
            out[key] = arr
        out["env_slot_ids"] = env_slot_ids
        return out
