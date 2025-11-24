# class SubspaceAnalysis:
#     def fit(self, session, spks_type): ...  # Returns the fit components and extras (data, placefields, etc)
#     def score(self, session, fits, spks_type): ...  # Returns the variance in the test data
#     def reconstruction_score(
#         self, session, fits, spks_type
#     ): ...  # Returns the frobenius norm of the difference between the test data and the reconstructed data for each expanding subspace
#     def get_scores(): ...  # A similar cache method for getting scores without dealing with refitting which is slow

from typing import NamedTuple, Any, Union, TYPE_CHECKING, Optional
import torch
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_placefield
from dimilibi import PCA, SVCA
from .base import SubspaceModel
from ..regression_models.hyperparameters import PlaceFieldHyperparameters

if TYPE_CHECKING:
    from ..registry import SplitName


class Subspace(NamedTuple):
    subspace_activity: Union[PCA, SVCA]
    subspace_placefields: Union[PCA, SVCA]
    extras: dict[str, Any]


class PCASubspace(SubspaceModel):
    def fit(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        split: "SplitName",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
        nan_safe: bool = False,
    ):
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        train_data, frame_behavior_train, num_neurons = self.get_session_data(session, spks_type, split, use_cell_split=False)
        train_data = self._center_data(train_data, self.centered)

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        placefield = get_placefield(
            train_data.T.numpy(),
            frame_behavior_train,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
        )
        placefield_extended = torch.tensor(placefield.placefield).reshape(-1, num_neurons).T

        # Check for NaNs and filter if needed
        placefield_extended, train_data = self._check_and_filter_nans(placefield_extended, train_data, nan_safe=nan_safe)

        num_components = self._compute_num_components(self.max_components, train_data.shape, placefield_extended.shape)
        pca_activity = PCA(num_components=num_components).fit(train_data)
        pca_placefields = PCA(num_components=num_components).fit(placefield_extended)

        return Subspace(
            subspace_activity=pca_activity,
            subspace_placefields=pca_placefields,
            extras=dict(placefield=placefield),
        )

    def score(
        self,
        session: B2Session,
        subspace: Subspace,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "not_train",
    ):
        test_data, _, _ = self.get_session_data(session, spks_type, split, use_cell_split=False)
        test_data = self._center_data(test_data, self.centered)

        subspace_activity = subspace.subspace_activity.get_components()
        subspace_placefields = subspace.subspace_placefields.get_components()
        variance_activity = torch.var(test_data.T @ subspace_activity, dim=0)
        variance_placefields = torch.var(test_data.T @ subspace_placefields, dim=0)

        return dict(
            variance_activity=variance_activity,
            variance_placefields=variance_placefields,
        )

    def _get_model_name(self) -> str:
        """Get the name of the model."""
        return "pca_subspace"


class CVPCASubspace(SubspaceModel):
    def fit(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        split: "SplitName",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
        nan_safe: bool = False,
    ):
        """Fit the CVPCASubspace model.

        CVPCASubspace requires two independent training sets for cross-validation.
        When split="train", it automatically splits into "train0" and "train1".
        For any other split, it randomly splits the data into two halves.

        Parameters
        ----------
        session : B2Session
            The session to fit the model on.
        spks_type : SpksTypes
            The type of spike data to use.
        split : "SplitName"
            The split to use for fitting. If "train", automatically splits into "train0" and "train1".
            For any other split, randomly splits the data into two halves.
        hyperparameters : Optional[PlaceFieldHyperparameters]
            The hyperparameters to use. If None, uses self.hyperparameters.
        nan_safe : bool
            If True, will check for NaN values in placefield data and raise an error if found.
            If False, will filter out NaN samples from placefield data and corresponding activity data.
            Default is False.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        # CVPCASubspace requires two independent training sets for cross-validation.
        if split == "train":
            # When split="train", automatically split into "train0" and "train1".
            train0_data, frame_behavior_train0, num_neurons = self.get_session_data(session, spks_type, "train0", use_cell_split=False)
            train1_data, frame_behavior_train1, _ = self.get_session_data(session, spks_type, "train1", use_cell_split=False)
        else:
            # For any other split, get the data once and randomly split it into two halves.
            train_data, frame_behavior, num_neurons = self.get_session_data(session, spks_type, split, use_cell_split=False)

            # Randomly split the data into two halves
            num_samples = train_data.shape[1]
            num_samples_split0 = num_samples // 2
            perm = torch.randperm(num_samples)
            idx_split0 = torch.sort(perm[:num_samples_split0]).values
            idx_split1 = torch.sort(perm[num_samples_split0:]).values

            train0_data = train_data[:, idx_split0]
            train1_data = train_data[:, idx_split1]
            frame_behavior_train0 = frame_behavior.filter(idx_split0)
            frame_behavior_train1 = frame_behavior.filter(idx_split1)

        # Balance data lengths if needed (shouldn't be necessary for train0/train1, but keep for safety)
        if train0_data.shape[1] != train1_data.shape[1]:
            num_samples = min(train0_data.shape[1], train1_data.shape[1])
            idx_train0 = torch.sort(torch.randperm(train0_data.shape[1])[:num_samples]).values
            idx_train1 = torch.sort(torch.randperm(train1_data.shape[1])[:num_samples]).values
            train0_data = train0_data[:, idx_train0]
            train1_data = train1_data[:, idx_train1]
            frame_behavior_train0 = frame_behavior_train0.filter(idx_train0)
            frame_behavior_train1 = frame_behavior_train1.filter(idx_train1)

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        placefield0 = get_placefield(
            train0_data.T.numpy(),
            frame_behavior_train0,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
        )
        placefield1 = get_placefield(
            train1_data.T.numpy(),
            frame_behavior_train1,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
        )
        placefield0_extended = torch.tensor(placefield0.placefield).reshape(-1, num_neurons).T
        placefield1_extended = torch.tensor(placefield1.placefield).reshape(-1, num_neurons).T

        # Check for NaNs and filter if needed
        placefield0_extended, train0_data = self._check_and_filter_nans(placefield0_extended, train0_data, nan_safe=nan_safe)
        placefield1_extended, train1_data = self._check_and_filter_nans(placefield1_extended, train1_data, nan_safe=nan_safe)

        num_components = self._compute_num_components(
            self.max_components,
            train0_data.shape,
            train1_data.shape,
            placefield0_extended.shape,
            placefield1_extended.shape,
        )

        svca_activity = SVCA(centered=self.centered, num_components=num_components)
        svca_activity = svca_activity.fit(train0_data, train1_data)
        svca_placefields = SVCA(centered=self.centered, num_components=num_components)
        svca_placefields = svca_placefields.fit(placefield0_extended, placefield1_extended)

        return Subspace(
            subspace_activity=svca_activity,
            subspace_placefields=svca_placefields,
            extras=dict(placefield0=placefield0, placefield1=placefield1),
        )

    def score(
        self,
        session: B2Session,
        subspace: Subspace,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "not_train",
    ):
        test_data, _, _ = self.get_session_data(session, spks_type, split, use_cell_split=False)

        subspace_activity = subspace.subspace_activity.U
        subspace_placefields = subspace.subspace_placefields.U
        variance_activity = torch.var(test_data.T @ subspace_activity, dim=0)
        variance_placefields = torch.var(test_data.T @ subspace_placefields, dim=0)

        return dict(
            variance_activity=variance_activity,
            variance_placefields=variance_placefields,
        )

    def _get_model_name(self) -> str:
        """Get the name of the model."""
        return "cvpca_subspace"


class SVCASubspace(SubspaceModel):
    def fit(
        self,
        session: B2Session,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "train",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
        nan_safe: bool = False,
    ):
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        (train_source, train_target), frame_behavior_train, (num_source_neurons, num_target_neurons) = self.get_session_data(
            session,
            spks_type,
            split,
            use_cell_split=True,
        )

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        placefield_source = get_placefield(
            train_source.T.numpy(),
            frame_behavior_train,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
        )
        placefield_target = get_placefield(
            train_target.T.numpy(),
            frame_behavior_train,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
        )
        placefield_source_extended = torch.tensor(placefield_source.placefield).reshape(-1, num_source_neurons).T
        placefield_target_extended = torch.tensor(placefield_target.placefield).reshape(-1, num_target_neurons).T

        # Check for NaNs and filter if needed
        # Note: We need to filter both source and target together to keep them aligned
        # Check for NaNs in either placefield
        idx_nan_samples = torch.any(torch.isnan(placefield_source_extended), dim=0) | torch.any(torch.isnan(placefield_target_extended), dim=0)

        if nan_safe:
            if torch.any(idx_nan_samples):
                num_nan = torch.sum(idx_nan_samples).item()
                total = len(idx_nan_samples)
                raise ValueError(f"{num_nan} / {total} samples have NaN values in placefield data!")
            if torch.any(torch.isnan(train_source)) or torch.any(torch.isnan(train_target)):
                raise ValueError("NaN values in train_source or train_target!")
        else:
            # Filter out NaN samples from all data
            idx_valid = ~idx_nan_samples
            placefield_source_extended = placefield_source_extended[:, idx_valid]
            placefield_target_extended = placefield_target_extended[:, idx_valid]

        num_components = self._compute_num_components(
            self.max_components,
            train_source.shape,
            train_target.shape,
            placefield_source_extended.shape,
            placefield_target_extended.shape,
        )

        svca_activity = SVCA(centered=self.centered, num_components=num_components)
        svca_activity = svca_activity.fit(train_source, train_target)
        svca_placefields = SVCA(centered=self.centered, num_components=num_components)
        svca_placefields = svca_placefields.fit(placefield_source_extended, placefield_target_extended)

        return Subspace(
            subspace_activity=svca_activity,
            subspace_placefields=svca_placefields,
            extras=dict(placefield_source=placefield_source, placefield_target=placefield_target),
        )

    def score(
        self,
        session: B2Session,
        subspace: Subspace,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "not_train",
    ):
        (test_source, test_target), _, _ = self.get_session_data(session, spks_type, split, use_cell_split=True)
        variance_activity = subspace.subspace_activity.score(test_source, test_target)[0]
        variance_placefields = subspace.subspace_placefields.score(test_source, test_target)[0]

        return dict(
            variance_activity=variance_activity,
            variance_placefields=variance_placefields,
        )

    def _get_model_name(self) -> str:
        """Get the name of the model."""
        return "svca_subspace"
