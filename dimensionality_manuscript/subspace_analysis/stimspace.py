"""StimSpaceSubspace — stimulus-space cross-validated shared variance analysis.

Implements the G_A(B) kernel formulation from shared_variance.md. Works in
stimulus space (spatial bins) rather than neural space, computing a
cross-validated bilinear kernel for an unbiased estimator of shared variance.

fit  (train0 / train1):  K = G1.T @ cov_PF0 @ G1  →  eigenvectors U
score (validation / test): w[i] = (G2 @ u_i).T @ cov_PF0_train @ (G3 @ u_i)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import torch

from vrAnalysis.helpers import reliability_loo
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.placefields import FrameBehavior, Placefield, get_placefield
from vrAnalysis.sessions import B2Session, SpksTypes
from dimilibi import PCA

from ..regression_models.base import ActivityParameters
from ..regression_models.hyperparameters import PlaceFieldHyperparameters
from .base import SubspaceModel, Subspace, _eigh_numpy, _eigvalsh_numpy, _svd_numpy

if TYPE_CHECKING:
    from ..registry import SplitName


def _make_G(pf: torch.Tensor) -> torch.Tensor:
    """Center over bins and scale so G @ G.T = cov(pf). Shape (N, S) → (N, S)."""
    S = pf.shape[1]
    return (pf - pf.mean(dim=1, keepdim=True)) / (S - 1) ** 0.5


@dataclass(frozen=True)
class StimSpaceFoldSpec:
    """Logical fold name mapped to a registry split and placefield smoothing."""

    name: str
    registry_split: "SplitName"
    smooth_width: Optional[float]


@dataclass
class StimSpacePrepState:
    """Preparation state produced at fit and replayed at score."""

    idx_keep: np.ndarray
    valid_environments: np.ndarray
    valid_positions: np.ndarray

    def to_extras(self) -> dict:
        """Return only preparation keys for ``Subspace.extras``."""
        return {
            "idx_keep": self.idx_keep,
            "valid_environments": self.valid_environments,
            "valid_positions": self.valid_positions,
        }

    @classmethod
    def from_extras(cls, extras: dict) -> "StimSpacePrepState":
        """Reconstruct preparation state from ``Subspace.extras``."""
        return cls(
            idx_keep=extras["idx_keep"],
            valid_environments=extras["valid_environments"],
            valid_positions=extras["valid_positions"],
        )


@dataclass
class StimSpaceFolds:
    """Processed activity, placefields, and aligned PF matrices per logical fold."""

    activity: dict[str, torch.Tensor]
    placefields: dict[str, Placefield]
    pf_matrices: dict[str, torch.Tensor]


class StimSpaceSubspace(SubspaceModel):
    """Stimulus-space cross-validated subspace model.

    Parameters
    ----------
    registry : PopulationRegistry
    hyperparameters : PlaceFieldHyperparameters
    autosave : bool
    normalize : bool
        Normalize placefield matrices by per-neuron peak response across
        both folds before computing G matrices.
    reliability_threshold : float or None
        Exclude neurons with leave-one-out reliability below this value.
        Computed from the full data split.
    fraction_active_threshold : float or None
        Exclude neurons with fraction-active below this value.
        Computed from the full data split.
    """

    def __init__(
        self,
        registry,
        autosave: bool = True,
        activity_parameters: ActivityParameters = ActivityParameters(center=False, scale=False, scale_type="none"),
        reliability_threshold: Optional[float] = None,
        fraction_active_threshold: Optional[float] = None,
    ) -> None:
        super().__init__(
            registry,
            autosave=autosave,
            activity_parameters=activity_parameters,
        )
        self.reliability_threshold = reliability_threshold
        self.fraction_active_threshold = fraction_active_threshold

    def _fit_fold_specs(self, hyperparameters: PlaceFieldHyperparameters) -> list[StimSpaceFoldSpec]:
        """Fold definitions used during fit (includes train and all CV folds).
        We need all cv folds - even though not used in fit - because we check which positions are valid
        across all folds and it needs to be done "looking ahead". Probably shouldn't have done this in
        fit, but it works this way and is easily justified.
        """
        smooth_width = hyperparameters.smooth_width
        return [
            StimSpaceFoldSpec("train", "train0", smooth_width),
            StimSpaceFoldSpec("cv1", "train1", None),
            StimSpaceFoldSpec("cv2", "validation", None),
            StimSpaceFoldSpec("test", "test", smooth_width),
        ]

    def _score_fold_specs(self, hyperparameters: PlaceFieldHyperparameters) -> list[StimSpaceFoldSpec]:
        """Fold definitions used during score."""
        smooth_width = hyperparameters.smooth_width
        return [
            StimSpaceFoldSpec("cv1", "train1", None),
            StimSpaceFoldSpec("cv2", "validation", None),
            StimSpaceFoldSpec("test", "test", smooth_width),
        ]

    def _neuron_filter(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        num_neurons: int,
        dist_edges: np.ndarray,
    ) -> np.ndarray:
        """Return boolean mask (num_neurons,) of neurons to keep."""
        idx_keep = np.ones(num_neurons, dtype=bool)
        if self.reliability_threshold is None and self.fraction_active_threshold is None:
            return idx_keep

        full_data, full_fb, _ = self.get_session_data(session, spks_type, "full", use_cell_split=False)
        _all_trials = get_placefield(
            full_data.T.numpy(),
            full_fb,
            dist_edges=dist_edges,
            average=False,
            use_fast_sampling=True,
            session=session,
        )
        _all_by_env = [_all_trials.filter_by_environment(env) for env in session.environments]
        _pf_data = [np.transpose(pf.placefield, (2, 0, 1)) for pf in _all_by_env]  # list of (N, num_trials, num_bins)

        if self.reliability_threshold is not None:
            _idx_reliable = [reliability_loo(_pfd) > self.reliability_threshold for _pfd in _pf_data]
            idx_keep &= np.logical_or.reduce(_idx_reliable)  # keep if reliable in any environment

        if self.fraction_active_threshold is not None:
            _idx_active = []
            for _pfd in _pf_data:
                _fa = FractionActive.compute(
                    _pfd,
                    activity_axis=2,
                    fraction_axis=1,
                    activity_method="rms",
                    fraction_method="participation",
                )
                _idx_active.append(_fa > self.fraction_active_threshold)

            idx_keep &= np.logical_or.reduce(_idx_active)  # keep if active in any environment

        return idx_keep

    def _preprocess_data(
        self,
        data: torch.Tensor,
        idx_keep: np.ndarray,
    ) -> torch.Tensor:
        """Apply neuron filtering."""
        return data[idx_keep]

    def _compute_placefield_folds(
        self,
        fold_specs: dict[str, tuple[torch.Tensor, "FrameBehavior", Optional[float]]],
        dist_edges: np.ndarray,
        session: B2Session,
    ) -> dict[str, Placefield]:
        """Compute placefields for all folds used by the stimulus-space estimator.

        Parameters
        ----------
        fold_specs : dict
            Mapping from fold name to ``(data, frame_behavior, smooth_width)``.
            ``data`` has shape ``(num_neurons, num_samples)``.
        dist_edges : np.ndarray
            Spatial bin edges for placefield computation.
        session : B2Session
            Session used by fast-sampling placefield computation.

        Returns
        -------
        dict
            Mapping from fold name to a ``Placefield`` with zero-count bins set
            to NaN.
        """
        placefields = {}
        for name, (data, frame_behavior, smooth_width) in fold_specs.items():
            placefields[name] = get_placefield(
                data.T.numpy(),
                frame_behavior,
                dist_edges=dist_edges,
                average=True,
                smooth_width=smooth_width,
                zero_to_nan=True,
                use_fast_sampling=True,
                session=session,
            )
        return placefields

    def _position_filter(self, placefields: dict[str, Placefield]) -> tuple[np.ndarray, np.ndarray]:
        """Find environments and positions valid in every placefield fold.

        Parameters
        ----------
        placefields : dict
            Placefield folds whose flattened stimulus axes must be aligned.

        Returns
        -------
        valid_environments : np.ndarray
            Environments present in every fold.
        valid_positions : np.ndarray
            Boolean mask over original position bins. A bin is valid only if no
            retained fold has NaNs for any retained environment or neuron.
        """
        environment_sets = [set(np.asarray(placefield.environment, dtype=int).tolist()) for placefield in placefields.values()]
        valid_environments = np.array(sorted(set.intersection(*environment_sets)), dtype=int)
        if len(valid_environments) == 0:
            raise ValueError("No environments are shared across all StimSpace placefield folds.")

        num_positions = next(iter(placefields.values())).count.shape[1]
        valid_positions = np.ones(num_positions, dtype=bool)
        environments = tuple(int(env) for env in valid_environments)
        for placefield in placefields.values():
            filtered = placefield.filter_by_environment(environments)
            valid_positions &= ~np.any(np.isnan(filtered.placefield), axis=(0, 2))

        if not np.any(valid_positions):
            raise ValueError("No positions are valid across all StimSpace placefield folds.")

        return valid_environments, valid_positions

    def _placefield_matrix(
        self,
        placefield: Placefield,
        valid_environments: np.ndarray,
        valid_positions: np.ndarray,
    ) -> torch.Tensor:
        """Return an aligned ``(neurons, environments * positions)`` matrix."""
        rows = []
        for environment in valid_environments:
            idx_environment = placefield.environment == environment
            if np.sum(idx_environment) != 1:
                raise ValueError(f"Expected exactly one row for environment {environment}, " f"found {np.sum(idx_environment)}.")
            rows.append(placefield.placefield[idx_environment][:, valid_positions, :])

        aligned = np.concatenate(rows, axis=0)
        if np.any(np.isnan(aligned)):
            raise ValueError("Filtered placefield matrix still contains NaNs.")
        return torch.tensor(
            aligned.reshape(-1, aligned.shape[2]).T,
            dtype=torch.float32,
        )

    def get_processed_folds(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        fold_specs: Sequence[StimSpaceFoldSpec],
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        *,
        prep_state: Optional[StimSpacePrepState] = None,
    ) -> tuple[StimSpaceFolds, StimSpacePrepState]:
        """Load, preprocess, and align activity and placefield data for StimSpace folds.

        Parameters
        ----------
        session : B2Session
        spks_type : SpksTypes
        fold_specs : sequence of StimSpaceFoldSpec
            Logical folds to load.
        hyperparameters : PlaceFieldHyperparameters
        prep_state : StimSpacePrepState, optional
            When provided (score path), reuses neuron mask and position mask from fit.
            When ``None`` (fit path), computes them.

        Returns
        -------
        folds : StimSpaceFolds
        prep_state : StimSpacePrepState
            Preparation state for reuse at score (new state when ``prep_state`` was ``None``).
        """
        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        fold_specs = list(fold_specs)

        if prep_state is None:
            _first_data, _, num_neurons = self.get_session_data(session, spks_type, fold_specs[0].registry_split, use_cell_split=False)
            idx_keep = self._neuron_filter(session, spks_type, num_neurons, dist_edges)
            valid_environments = None
            valid_positions = None
        else:
            idx_keep = prep_state.idx_keep
            valid_environments = prep_state.valid_environments
            valid_positions = prep_state.valid_positions

        activity: dict[str, torch.Tensor] = {}
        frame_behaviors: dict[str, FrameBehavior] = {}
        smooth_widths: dict[str, Optional[float]] = {}

        for spec in fold_specs:
            data, frame_behavior, _ = self.get_session_data(session, spks_type, spec.registry_split, use_cell_split=False)
            activity[spec.name] = self._preprocess_data(data, idx_keep)
            frame_behaviors[spec.name] = frame_behavior
            smooth_widths[spec.name] = spec.smooth_width

        pf_fold_names = [spec.name for spec in fold_specs]
        pf_input_specs = {name: (activity[name], frame_behaviors[name], smooth_widths[name]) for name in pf_fold_names}
        placefields = self._compute_placefield_folds(pf_input_specs, dist_edges, session)

        if valid_environments is None or valid_positions is None:
            valid_environments, valid_positions = self._position_filter(placefields)

        pf_matrices = {name: self._placefield_matrix(placefields[name], valid_environments, valid_positions) for name in pf_fold_names}

        out_prep = StimSpacePrepState(
            idx_keep=idx_keep,
            valid_environments=valid_environments,
            valid_positions=valid_positions,
        )
        return StimSpaceFolds(activity=activity, placefields=placefields, pf_matrices=pf_matrices), out_prep

    def fit(
        self,
        session: B2Session,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "train",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
    ) -> Subspace:
        """
        Fit the stimulus-space cross-validated subspace model.

        Parameters
        ----------
        session : B2Session
        spks_type : SpksTypes
        split : SplitName
            Not used, kept for consistency with other subspace models.
        hyperparameters : PlaceFieldHyperparameters
        nan_safe : bool
            If True, allow NaNs in placefield matrices (bins not visited in a fold).
        """
        folds, prep = self.get_processed_folds(
            session,
            spks_type,
            self._fit_fold_specs(hyperparameters),
            hyperparameters,
        )

        pf_mat_train = folds.pf_matrices["train"]
        pf_mat_test = folds.pf_matrices["test"]
        data_train = folds.activity["train"]
        data_test = folds.activity["test"]

        S = pf_mat_test.shape[1]
        if S < 2:
            raise ValueError(f"Too few valid bins ({S}) after NaN filtering in fit.")

        G_train = _make_G(pf_mat_train)  # (N, S)

        cov_pf_test = torch.cov(pf_mat_test)
        cov_data_test = torch.cov(data_test)

        pca_activity = PCA(center=True).fit(data_train)
        pca_placefields = PCA(center=True).fit(pf_mat_train)

        pf_pf_kernel = G_train.T @ cov_pf_test @ G_train  # (S, S)
        pf_full_kernel = G_train.T @ cov_data_test @ G_train  # (S, S)
        u_pf_full = torch.fliplr(_eigh_numpy(pf_full_kernel)[1])
        u_pf_pf = torch.fliplr(_eigh_numpy(G_train.T @ G_train)[1])

        return Subspace(
            subspace_activity=pca_activity,
            subspace_placefields=pca_placefields,
            extras={
                **prep.to_extras(),
                "pf_pf_kernel": pf_pf_kernel,
                "pf_full_kernel": pf_full_kernel,
                "u_pf_pf": u_pf_pf,
                "u_pf_full": u_pf_full,
            },
        )

    def score(
        self,
        session: B2Session,
        subspace: Subspace,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "not_train",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
    ) -> dict:
        """
        Score the stimulus-space cross-validated subspace model.

        split isn't used!
        """
        prep = StimSpacePrepState.from_extras(subspace.extras)
        folds, _ = self.get_processed_folds(
            session,
            spks_type,
            self._score_fold_specs(hyperparameters),
            hyperparameters,
            prep_state=prep,
        )

        test_data = folds.activity["test"]
        test_data_cov = torch.cov(test_data)

        precov_cv1_pf = _make_G(folds.pf_matrices["cv1"])
        precov_cv2_pf = _make_G(folds.pf_matrices["cv2"])
        test_placefield_extended = folds.pf_matrices["test"]
        test_placefield_cov = torch.cov(test_placefield_extended)

        K_pf1_full3_pf2 = precov_cv1_pf.T @ test_data_cov @ precov_cv2_pf
        K_pf1_pf3_pf2 = precov_cv1_pf.T @ test_placefield_cov @ precov_cv2_pf

        u_pf_full: torch.Tensor = subspace.extras["u_pf_full"]
        u_pf_pf: torch.Tensor = subspace.extras["u_pf_pf"]
        cv_variance_placefields = torch.sum(u_pf_full * (K_pf1_full3_pf2 @ u_pf_full), dim=0)
        cv_variance_placefield_placefield = torch.sum(u_pf_pf * (K_pf1_pf3_pf2 @ u_pf_pf), dim=0)

        train_evecs_activity = subspace.subspace_activity.get_components()
        train_eval_activity_root = torch.diag(torch.sqrt(subspace.subspace_activity.get_eigenvalues()))
        train_evecs_placefields = subspace.subspace_placefields.get_components()
        train_eval_placefields_root = torch.diag(torch.sqrt(subspace.subspace_placefields.get_eigenvalues()))

        inner_block_activity = train_eval_activity_root @ train_evecs_activity.T @ test_data_cov @ train_evecs_activity @ train_eval_activity_root
        inner_block_placefields = (
            train_eval_placefields_root @ train_evecs_placefields.T @ test_data_cov @ train_evecs_placefields @ train_eval_placefields_root
        )
        inner_block_pfpf = (
            train_eval_placefields_root @ train_evecs_placefields.T @ test_placefield_cov @ train_evecs_placefields @ train_eval_placefields_root
        )

        variance_activity = torch.sqrt(torch.clamp_min(torch.flipud(_eigvalsh_numpy(inner_block_activity)), 0.0))
        variance_placefields = torch.sqrt(torch.clamp_min(torch.flipud(_eigvalsh_numpy(inner_block_placefields)), 0.0))
        variance_placefield_placefield = torch.sqrt(torch.clamp_min(torch.flipud(_eigvalsh_numpy(inner_block_pfpf)), 0.0))

        return dict(
            variance_activity=variance_activity,
            variance_placefields=variance_placefields,
            variance_placefield_placefield=variance_placefield_placefield,
            cv_variance_squared_placefields=cv_variance_placefields,
            cv_variance_squared_placefield_placefield=cv_variance_placefield_placefield,
        )

    def compute_cv_variance_scale(
        self,
        session: B2Session,
        spks_type: SpksTypes = "oasis",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
    ) -> dict:
        """Cross-validated variance-scale shared-variance estimator.

        Self-contained companion to ``fit``/``score``: uses the 4 real time
        splits (``train0``, ``train1``, ``validation``, ``test``) as 4
        interchangeable draws, all with the same placefield smoothing
        (``hyperparameters.smooth_width``), and mirrors
        ``_stim_full_cv_variance_scale_result``/``cvsvd_stimfull`` from
        ``simulations/shared_variance.py``. Computes its own neuron and
        position filtering independent of any ``Subspace`` from ``fit()``.

        Has no reference of its own; pair the returned candidate modes with
        ``variance_activity`` from ``score()``.

        Returns
        -------
        dict
            ``{"cv_variance_scale_placefields": torch.Tensor}`` averaged over
            all 3-of-4 draw combinations.
        """

        def _compute_scores(stim_list_train, stim_list_test, data_list_reference, double_cross_val: bool = True):
            num_draws = len(stim_list_train)
            if num_draws != len(stim_list_test) or num_draws != len(data_list_reference):
                raise ValueError("All lists must have the same number of draws.")
            combo_scores = []
            for i, j, k in itertools.combinations(range(num_draws), 3):
                sf_cross_train = stim_list[i].T @ data_list[k]
                if double_cross_val:
                    sf_cross_test = stim_list_test[j].T @ data_list[k]
                else:
                    sf_cross_test = stim_list[i].T @ data_list[k]

                U_train, _, Vt_train = _svd_numpy(sf_cross_train)
                score = torch.sum(U_train * (sf_cross_test @ Vt_train.T), dim=0)

                # We normalize the SVD because I want it to be similar to PCA eigenvalues that we'd get from
                # data @ data.T for example - where we'd divide by the number of timepoints - 1 to get a neuron x neuron covariance
                # Here, we have stim.T @ data --- where we multiply through the neural dimension - but it's a way to recover a neuron
                # x neuron covariance so that's how we normalize it! (It's compared to other neuron x neuron variances from other metrics).
                norm = ((stim_list[i].shape[1] - 1) * (data_list[k].shape[1] - 1)) ** 0.5
                combo_scores.append(score / norm)
            return torch.mean(torch.stack(combo_scores, dim=0), dim=0)

        draw_splits: tuple["SplitName", ...] = ("train0", "train1", "validation", "test")
        smoothed_specs = [StimSpaceFoldSpec(f"draw{i}", split, hyperparameters.smooth_width) for i, split in enumerate(draw_splits)]

        if hyperparameters.smooth_width is None:
            folds, _ = self.get_processed_folds(session, spks_type, smoothed_specs, hyperparameters)
            stim_list = [folds.pf_matrices[f"draw{i}"] for i in range(4)]
            data_list = [folds.activity[f"draw{i}"] for i in range(4)]
            stim_list = [stim_list[i] - stim_list[i].mean(dim=1, keepdim=True) for i in range(4)]
            data_list = [data_list[i] - data_list[i].mean(dim=1, keepdim=True) for i in range(4)]
            output = {"cv_variance_scale_placefields": _compute_scores(stim_list, stim_list, data_list)}
            output["variance_scale_placefields"] = _compute_scores(stim_list, stim_list, data_list, double_cross_val=False)

        # Fetch smoothed and raw (unsmoothed) draws in one call so the NaN-based
        # position filter intersects validity across both, keeping bin counts aligned.
        raw_specs = [StimSpaceFoldSpec(f"draw{i}_raw", split, None) for i, split in enumerate(draw_splits)]
        folds, _ = self.get_processed_folds(session, spks_type, smoothed_specs + raw_specs, hyperparameters)

        stim_list = [folds.pf_matrices[f"draw{i}"] for i in range(4)]
        data_list = [folds.activity[f"draw{i}"] for i in range(4)]
        stim_list = [stim_list[i] - stim_list[i].mean(dim=1, keepdim=True) for i in range(4)]
        data_list = [data_list[i] - data_list[i].mean(dim=1, keepdim=True) for i in range(4)]

        output = {
            "cv_variance_scale_placefields": _compute_scores(stim_list, stim_list, data_list),
        }

        stim_list_test = [folds.pf_matrices[f"draw{i}_raw"] for i in range(4)]
        stim_list_test = [stim_list_test[i] - stim_list_test[i].mean(dim=1, keepdim=True) for i in range(4)]

        output["cv_variance_scale_placefields_raw_test"] = _compute_scores(stim_list, stim_list_test, data_list)
        output["variance_scale_placefields_raw_test"] = _compute_scores(stim_list, stim_list_test, data_list, double_cross_val=False)
        return output

    def _get_model_name(self) -> str:
        base = "stimspace_subspace"
        if self.reliability_threshold is not None:
            base += f"_rel{self.reliability_threshold}"
        if self.fraction_active_threshold is not None:
            base += f"_frac{self.fraction_active_threshold}"
        return base
