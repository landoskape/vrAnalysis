"""StimSpaceSubspace — stimulus-space cross-validated shared variance analysis.

Implements the G_A(B) kernel formulation from shared_variance.md. Works in
stimulus space (spatial bins) rather than neural space, computing a
cross-validated bilinear kernel for an unbiased estimator of shared variance.

fit  (train0 / train1):  K = G1.T @ cov_PF0 @ G1  →  eigenvectors U
score (validation / test): w[i] = (G2 @ u_i).T @ cov_PF0_train @ (G3 @ u_i)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from vrAnalysis.helpers import reliability_loo
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.placefields import FrameBehavior, Placefield, get_placefield
from vrAnalysis.sessions import B2Session, SpksTypes
from dimilibi import PCA

from ..regression_models.hyperparameters import PlaceFieldHyperparameters
from .base import SubspaceModel, Subspace

if TYPE_CHECKING:
    from ..registry import SplitName


def _make_G(pf: torch.Tensor) -> torch.Tensor:
    """Center over bins and scale so G @ G.T = cov(pf). Shape (N, S) → (N, S)."""
    S = pf.shape[1]
    return (pf - pf.mean(dim=1, keepdim=True)) / (S - 1) ** 0.5


class StimSpaceSubspace(SubspaceModel):
    """Stimulus-space cross-validated subspace model.

    Parameters
    ----------
    registry : PopulationRegistry
    centered : bool
    correlation : bool
    hyperparameters : PlaceFieldHyperparameters
    max_components : int
    autosave : bool
    normalize : bool
        Normalize placefield matrices by per-neuron peak response across
        both folds before computing G matrices.
    use_fast_sampling : bool
        Passed to ``get_placefield``.
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
        centered: bool = False,
        correlation: bool = False,
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        max_components: int = 300,
        match_dimensions: bool = False,
        autosave: bool = True,
        normalize: bool = True,
        use_fast_sampling: bool = True,
        reliability_threshold: Optional[float] = None,
        fraction_active_threshold: Optional[float] = None,
        directions_from_placefield_only: bool = False,
        cross_validated_placefield_kernel: bool = False,
    ) -> None:
        # Note, match dimensions present but always set to False!!!
        super().__init__(
            registry,
            centered=centered,
            correlation=correlation,
            hyperparameters=hyperparameters,
            max_components=max_components,
            match_dimensions=False,
            autosave=autosave,
        )
        self.normalize = normalize
        self.use_fast_sampling = use_fast_sampling
        self.reliability_threshold = reliability_threshold
        self.fraction_active_threshold = fraction_active_threshold
        self.directions_from_placefield_only = directions_from_placefield_only
        self.cross_validated_placefield_kernel = cross_validated_placefield_kernel

    def _best_env_idx(self, session: B2Session) -> int:
        num_per_env = {i: int(np.sum(session.trial_environment == i)) for i in session.environments}
        best_env = max(num_per_env, key=num_per_env.get)
        return int(np.where(session.environments == best_env)[0][0])

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
            use_fast_sampling=self.use_fast_sampling,
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

    def _get_norm_values(self, session: B2Session, spks_type: SpksTypes, idx_keep: np.ndarray, min_norm: float = 1.0) -> torch.Tensor:
        """
        Get a global norm value for each neuron computed by taking the 99.5% percentile value of the spks
        """
        if not self.normalize:
            return torch.ones(len(idx_keep), dtype=torch.float32)

        data = self.get_session_data(session, spks_type, "full")[0]
        data = data[idx_keep]
        norm_values = torch.quantile(data, torch.tensor(0.995), dim=1, keepdim=True)
        norm_values[norm_values < min_norm] = min_norm
        return norm_values

    def _preprocess_data(
        self,
        data: torch.Tensor,
        idx_keep: np.ndarray,
        norm_values: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply neuron filtering, optional normalization, and optional centering."""
        data = data[idx_keep]
        if norm_values is not None:
            data = data / norm_values
        return self._center_data(data, self.centered)

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
                use_fast_sampling=self.use_fast_sampling,
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

    def fit(
        self,
        session: B2Session,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "train",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
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
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)

        # Load sub-splits used by fit and score so position filtering can align
        # every stimulus-space matrix before flattening.
        train_split, cv1_split, cv2_split, test_split = "train0", "train1", "validation", "test"
        data_train, fb_train, _ = self.get_session_data(session, spks_type, train_split, use_cell_split=False)
        data_cv1, fb_cv1, _ = self.get_session_data(session, spks_type, cv1_split, use_cell_split=False)
        data_cv2, fb_cv2, _ = self.get_session_data(session, spks_type, cv2_split, use_cell_split=False)
        data_test, fb_test, num_neurons = self.get_session_data(session, spks_type, test_split, use_cell_split=False)

        # Neuron filtering by reliability and/or fraction active
        # (computed from full data split, and using logical_or across environments)
        idx_keep = self._neuron_filter(session, spks_type, num_neurons, dist_edges)
        norm_values = self._get_norm_values(session, spks_type, idx_keep) if self.normalize else None
        data_train = self._preprocess_data(data_train, idx_keep, norm_values)
        data_cv1 = self._preprocess_data(data_cv1, idx_keep, norm_values)
        data_cv2 = self._preprocess_data(data_cv2, idx_keep, norm_values)
        data_test = self._preprocess_data(data_test, idx_keep, norm_values)

        num_test_samples = data_test.shape[1]
        num_test_samples_split0 = num_test_samples // 2
        test_perm = torch.randperm(num_test_samples)
        idx_test_split0 = torch.sort(test_perm[:num_test_samples_split0]).values
        idx_test_split1 = torch.sort(test_perm[num_test_samples_split0:]).values

        fold_specs = {
            "train": (data_train, fb_train, hyperparameters.smooth_width),
            "cv1": (data_cv1, fb_cv1, None),
            "cv2": (data_cv2, fb_cv2, None),
            "test": (data_test, fb_test, hyperparameters.smooth_width),
        }
        if self.cross_validated_placefield_kernel:
            fold_specs["test0"] = (
                data_test[:, idx_test_split0],
                fb_test.filter(idx_test_split0),
                hyperparameters.smooth_width,
            )
            fold_specs["test1"] = (
                data_test[:, idx_test_split1],
                fb_test.filter(idx_test_split1),
                hyperparameters.smooth_width,
            )

        placefields = self._compute_placefield_folds(fold_specs, dist_edges, session)
        valid_environments, valid_positions = self._position_filter(placefields)
        pf_mat_train = self._placefield_matrix(placefields["train"], valid_environments, valid_positions)
        pf_mat_test = self._placefield_matrix(placefields["test"], valid_environments, valid_positions)

        S = pf_mat_test.shape[1]
        if S < 2:
            raise ValueError(f"Too few valid bins ({S}) after NaN filtering in fit.")

        G_train = _make_G(pf_mat_train)  # (N, S)

        # Use covariance from test fold in kernel to estimate directions
        cov_pf_test = torch.cov(pf_mat_test)
        cov_data_test = torch.cov(data_test)

        # Learn pca on placefields train and full data train
        pca_activity = PCA(center=True).fit(data_train)
        pca_placefields = PCA(center=True).fit(pf_mat_train)

        # Now we need to learn directions on the "stimulus" kernels
        # K_pf(PF) to get the potentially cross-validatable eigenvectors of our PF-PF kernel
        #          - this is so that we can do a covcov version of cvPCA
        # K_pf(Full) to prepare the numerator of cross-validatable shared variance ratio
        pf_pf_kernel = G_train.T @ cov_pf_test @ G_train  # (S, S)
        pf_full_kernel = G_train.T @ cov_data_test @ G_train  # (S, S)
        u_pf_pf = torch.fliplr(torch.linalg.eigh(pf_pf_kernel)[1])  # (S, S) eigenvectors of pf_pf_kernel
        u_pf_full = torch.fliplr(torch.linalg.eigh(pf_full_kernel)[1])  # (S, S) eigenvectors of pf_full_kernel

        if self.directions_from_placefield_only:
            u_pf_pf = torch.fliplr(torch.linalg.eigh(G_train.T @ G_train)[1])

        return Subspace(
            subspace_activity=pca_activity,
            subspace_placefields=pca_placefields,
            extras={
                "pf_pf_kernel": pf_pf_kernel,
                "pf_full_kernel": pf_full_kernel,
                "u_pf_pf": u_pf_pf,
                "u_pf_full": u_pf_full,
                "idx_keep": idx_keep,
                "valid_environments": valid_environments,
                "valid_positions": valid_positions,
                "idx_test_split0": idx_test_split0,
                "idx_test_split1": idx_test_split1,
            },
        )

    def score(
        self,
        session: B2Session,
        subspace: Subspace,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "not_train",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
        nan_safe: bool = False,
    ) -> dict:
        """
        Score the stimulus-space cross-validated subspace model.

        split isn't used!
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)

        # These names are grandfathered from the TimeSplit class in registry.py
        # this model is connected to the "even" data split config (see configs/stimspace)
        # so each split name is just an even non-overlapping split of data
        cv1_split, cv2_split, test_split = "train1", "validation", "test"
        cv1_data, cv1_fb, _ = self.get_session_data(session, spks_type, cv1_split, use_cell_split=False)
        cv2_data, cv2_fb, _ = self.get_session_data(session, spks_type, cv2_split, use_cell_split=False)
        test_data, fb_test, num_neurons = self.get_session_data(session, spks_type, test_split, use_cell_split=False)
        idx_keep = subspace.extras["idx_keep"]
        valid_environments = subspace.extras["valid_environments"]
        valid_positions = subspace.extras["valid_positions"]

        norm_values = self._get_norm_values(session, spks_type, idx_keep) if self.normalize else None
        cv1_data = self._preprocess_data(cv1_data, idx_keep, norm_values)
        cv2_data = self._preprocess_data(cv2_data, idx_keep, norm_values)
        test_data = self._preprocess_data(test_data, idx_keep, norm_values)

        # Compute covariance of test data
        test_data_cov = torch.cov(test_data)

        # Also measure covariance of test placefield data for a placefield-placefield comparison as a control
        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)

        fold_specs = {
            "cv1": (cv1_data, cv1_fb, None),
            "cv2": (cv2_data, cv2_fb, None),
            "test": (test_data, fb_test, hyperparameters.smooth_width),
        }
        if self.cross_validated_placefield_kernel:
            idx_test_split0 = subspace.extras["idx_test_split0"]
            idx_test_split1 = subspace.extras["idx_test_split1"]
            fold_specs["test0"] = (
                test_data[:, idx_test_split0],
                fb_test.filter(idx_test_split0),
                hyperparameters.smooth_width,
            )
            fold_specs["test1"] = (
                test_data[:, idx_test_split1],
                fb_test.filter(idx_test_split1),
                hyperparameters.smooth_width,
            )

        placefields = self._compute_placefield_folds(fold_specs, dist_edges, session)
        cv1_placefield_extended = self._placefield_matrix(placefields["cv1"], valid_environments, valid_positions)
        cv2_placefield_extended = self._placefield_matrix(placefields["cv2"], valid_environments, valid_positions)
        test_placefield_extended = self._placefield_matrix(placefields["test"], valid_environments, valid_positions)

        # Get pre-cov matrices
        precov_cv1_pf = _make_G(cv1_placefield_extended)
        precov_cv2_pf = _make_G(cv2_placefield_extended)

        test_placefield_cov = torch.cov(test_placefield_extended)

        # Now we can compute the cross-validated stimulus space kernels
        K_pf1_full3_pf2 = precov_cv1_pf.T @ test_data_cov @ precov_cv2_pf

        if not self.cross_validated_placefield_kernel:
            K_pf1_pf3_pf2 = precov_cv1_pf.T @ test_placefield_cov @ precov_cv2_pf

        else:
            test_placefield0_extended = self._placefield_matrix(placefields["test0"], valid_environments, valid_positions)
            test_placefield1_extended = self._placefield_matrix(placefields["test1"], valid_environments, valid_positions)
            precov_test0_pf = _make_G(test_placefield0_extended)
            precov_test1_pf = _make_G(test_placefield1_extended)
            center_kernel = precov_test0_pf @ precov_test1_pf.T
            K_pf1_pf3_pf2 = precov_cv1_pf.T @ center_kernel @ precov_cv2_pf

        # And we can now measure variance on learned directions of the symmetric kernel
        # using a separate fold
        u_pf_full: torch.Tensor = subspace.extras["u_pf_full"]
        u_pf_pf: torch.Tensor = subspace.extras["u_pf_pf"]
        cv_variance_placefields = torch.sum(u_pf_full * (K_pf1_full3_pf2 @ u_pf_full), dim=0)
        cv_variance_placefield_placefield = torch.sum(u_pf_pf * (K_pf1_pf3_pf2 @ u_pf_pf), dim=0)

        # We're looking for train_cov^{1/2} @ test_cov @ train_cov^{1/2}
        # We can use the eigenvalues of the inner block:
        # train_eval^{1/2} @ train_evecs.T @ test_cov @ train_evecs @ train_eval^{1/2}
        # The outer train_evecs don't affect the eigenvalues and make it a bigger matrix
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

        variance_activity = torch.sqrt(torch.clamp_min(torch.flipud(torch.linalg.eigvalsh(inner_block_activity)), 0.0))
        variance_placefields = torch.sqrt(torch.clamp_min(torch.flipud(torch.linalg.eigvalsh(inner_block_placefields)), 0.0))
        variance_placefield_placefield = torch.sqrt(torch.clamp_min(torch.flipud(torch.linalg.eigvalsh(inner_block_pfpf)), 0.0))

        return dict(
            variance_activity=variance_activity,
            variance_placefields=variance_placefields,
            variance_placefield_placefield=variance_placefield_placefield,
            cv_variance_squared_placefields=cv_variance_placefields,
            cv_variance_squared_placefield_placefield=cv_variance_placefield_placefield,
        )

    def _get_model_name(self) -> str:
        base = "stimspace_subspace"
        if self.normalize:
            base += "_norm"
        if self.use_fast_sampling:
            base += "_fast"
        if self.reliability_threshold is not None:
            base += f"_rel{self.reliability_threshold}"
        if self.fraction_active_threshold is not None:
            base += f"_frac{self.fraction_active_threshold}"
        return base
