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
from vrAnalysis.processors.placefields import get_placefield
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
    ):
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

        # Load sub-splits
        # - we use test as a reference for all kernels
        # - train0 is used for the "train" fold because we need two more folds for cross-validation
        train_split, ref_split = "train0", "test"
        data_train, fb_train, _ = self.get_session_data(session, spks_type, train_split, use_cell_split=False)
        data_ref, fb_ref, num_neurons = self.get_session_data(session, spks_type, ref_split, use_cell_split=False)

        # Neuron filtering by reliability and/or fraction active
        # (computed from full data split, and using logical_or across environments)
        idx_keep = self._neuron_filter(session, spks_type, num_neurons, dist_edges)
        data_train = data_train[idx_keep]
        data_ref = data_ref[idx_keep]

        if self.normalize:
            norm_values = self._get_norm_values(session, spks_type, idx_keep)
            data_train = data_train / norm_values
            data_ref = data_ref / norm_values

        # Center if requested (handled within _center_data)
        data_train = self._center_data(data_train, self.centered)
        data_ref = self._center_data(data_ref, self.centered)

        # Placefields for each sub-split, single best environment
        pf_kw = dict(
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=self.use_fast_sampling,
            session=session,
        )
        pf_train = get_placefield(data_train.T.numpy(), fb_train, **pf_kw)
        pf_ref = get_placefield(data_ref.T.numpy(), fb_ref, **pf_kw)

        # Output of flattened is np.ndarray (num_env * num_bins, num_neurons)
        pf_mat_train = torch.tensor(pf_train.flattened().T, dtype=torch.float32)
        pf_mat_ref = torch.tensor(pf_ref.flattened().T, dtype=torch.float32)  # (N, S)

        # Filter to bins present in both folds
        valid_bins = ~(torch.any(torch.isnan(pf_mat_ref), dim=0) | torch.any(torch.isnan(pf_mat_train), dim=0))
        pf_mat_train = pf_mat_train[:, valid_bins]
        pf_mat_ref = pf_mat_ref[:, valid_bins]

        S = pf_mat_ref.shape[1]
        if S < 2:
            raise ValueError(f"Too few valid bins ({S}) after NaN filtering in fit.")

        G_train = _make_G(pf_mat_train)  # (N, S)

        cov_pf_ref = torch.cov(pf_mat_ref)
        cov_data_ref = torch.cov(data_ref)

        # Learn pca on placefields train and full data train
        pca_activity = PCA(center=True).fit(data_train)
        pca_placefields = PCA(center=True).fit(pf_mat_train)

        # Now we need to learn directions on the "stimulus" kernels
        # K_pf(PF) to get the potentially cross-validatable eigenvectors of our PF-PF kernel
        #          - this is so that we can do a covcov version of cvPCA
        # K_pf(Full) to prepare the numerator of cross-validatable shared variance ratio
        pf_pf_kernel = G_train.T @ cov_pf_ref @ G_train  # (S, S)
        pf_full_kernel = G_train.T @ cov_data_ref @ G_train  # (S, S)
        u_pf_pf = torch.fliplr(torch.linalg.eigh(pf_pf_kernel)[1])  # (S, S) eigenvectors of pf_pf_kernel
        u_pf_full = torch.fliplr(torch.linalg.eigh(pf_full_kernel)[1])  # (S, S) eigenvectors of pf_full_kernel

        return Subspace(
            subspace_activity=pca_activity,
            subspace_placefields=pca_placefields,
            extras={
                "pf_pf_kernel": pf_pf_kernel,
                "pf_full_kernel": pf_full_kernel,
                "u_pf_pf": u_pf_pf,
                "u_pf_full": u_pf_full,
                "idx_keep": idx_keep,
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
        cv1_split, cv2_split, ref_split = "train1", "validation", "test"
        cv1_data, cv1_fb, _ = self.get_session_data(session, spks_type, cv1_split, use_cell_split=False)
        cv2_data, cv2_fb, _ = self.get_session_data(session, spks_type, cv2_split, use_cell_split=False)
        test_data, frame_behavior_test, num_neurons = self.get_session_data(session, spks_type, ref_split, use_cell_split=False)
        idx_keep = subspace.extras["idx_keep"]

        # filter data to include kept neurons only
        cv1_data = cv1_data[idx_keep]
        cv2_data = cv2_data[idx_keep]
        test_data = test_data[idx_keep]

        if self.normalize:
            norm_values = self._get_norm_values(session, spks_type, idx_keep)
            cv1_data = cv1_data / norm_values
            cv2_data = cv2_data / norm_values
            test_data = test_data / norm_values

        # Center if requested (handled within _center_data)
        cv1_data = self._center_data(cv1_data, self.centered)
        cv2_data = self._center_data(cv2_data, self.centered)
        test_data = self._center_data(test_data, self.centered)

        # Compute covariance of test data
        test_data_cov = torch.cov(test_data)

        # Also measure covariance of test placefield data for a placefield-placefield comparison as a control
        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)

        # Placefields for each sub-split, single best environment
        # Note: hard code smooth_width=None because these are the test trials - (similar to r-cvPCA)
        pf_kw = dict(
            dist_edges=dist_edges,
            average=True,
            smooth_width=None,  # hard coded!!!!!
            use_fast_sampling=self.use_fast_sampling,
            session=session,
        )
        cv1_placefield = get_placefield(cv1_data.T.numpy(), cv1_fb, **pf_kw)
        cv2_placefield = get_placefield(cv2_data.T.numpy(), cv2_fb, **pf_kw)
        test_placefield = get_placefield(test_data.T.numpy(), frame_behavior_test, **pf_kw)
        cv1_placefield_extended = torch.tensor(cv1_placefield.flattened()).T
        cv2_placefield_extended = torch.tensor(cv2_placefield.flattened()).T
        test_placefield_extended = torch.tensor(test_placefield.flattened()).T

        # Get pre-cov matrices
        precov_cv1_pf = _make_G(cv1_placefield_extended)
        precov_cv2_pf = _make_G(cv2_placefield_extended)

        # Check for NaNs and filter if needed
        test_placefield_extended, _ = self._check_and_filter_nans(test_placefield_extended, test_data, nan_safe=nan_safe)
        test_placefield_cov = torch.cov(test_placefield_extended)

        # Now we can compute the cross-validated stimulus space kernels
        K_pf1_full3_pf2 = precov_cv1_pf.T @ test_data_cov @ precov_cv2_pf
        K_pf1_pf3_pf2 = precov_cv1_pf.T @ test_placefield_cov @ precov_cv2_pf

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
