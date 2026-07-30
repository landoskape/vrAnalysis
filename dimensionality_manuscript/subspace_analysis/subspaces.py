from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
import torch


def _cov_or_gram(X: torch.Tensor, centered: bool) -> torch.Tensor:
    """Covariance (centered) or gram matrix (uncentered), both with n-1 denominator."""
    if centered:
        return torch.cov(X)
    return X @ X.T / (X.shape[1] - 1)


def _log_abs_max(tensor: torch.Tensor, name: str, session: "B2Session") -> None:
    """Print the max absolute value, for spotting float32-overflow-prone magnitudes."""
    print(f"  [diag] {name} abs-max={tensor.abs().max().item():.6e} session={session.session_uid}")


def _assert_finite(tensor: torch.Tensor, name: str, session: "B2Session") -> None:
    """Raise with session/tensor context the moment a NaN or Inf first appears."""
    if not torch.isfinite(tensor).all():
        n_nan = torch.isnan(tensor).sum().item()
        n_inf = torch.isinf(tensor).sum().item()
        raise ValueError(
            f"{name} contains non-finite values for session {session.session_uid} " f"(nan={n_nan}, inf={n_inf}, shape={tuple(tensor.shape)})"
        )


from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_placefield, Placefield
from dimilibi import PCA, SVCA
from .base import SubspaceModel, Subspace, _eigvalsh_numpy
from ..env_order import load_env_order, MAX_ENV_SLOTS
from ..regression_models.hyperparameters import PlaceFieldHyperparameters

if TYPE_CHECKING:
    from ..registry import SplitName


_VARIANCE_KEYS = (
    "variance_activity",
    "variance_placefields",
    "variance_placefield_placefield",
)


def _paired_placefield_matrices(
    placefield_source: Placefield,
    placefield_target: Placefield,
    num_source_neurons: int,
    num_target_neurons: int,
    nan_safe: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten aligned source/target place fields and remove their shared NaN columns."""
    source = torch.tensor(placefield_source.flattened()).reshape(-1, num_source_neurons).T.contiguous()
    target = torch.tensor(placefield_target.flattened()).reshape(-1, num_target_neurons).T.contiguous()
    idx_nan_samples = torch.any(torch.isnan(source), dim=0) | torch.any(torch.isnan(target), dim=0)

    if nan_safe and torch.any(idx_nan_samples):
        num_nan = torch.sum(idx_nan_samples).item()
        total = len(idx_nan_samples)
        raise ValueError(f"{num_nan} / {total} samples have NaN values in placefield data!")
    if not nan_safe:
        idx_valid = ~idx_nan_samples
        source = source[:, idx_valid]
        target = target[:, idx_valid]
    return source, target


def _environment_slot_scores(
    session: B2Session,
    global_scores: dict[str, torch.Tensor],
    environments: np.ndarray,
    score_environment: Callable[[int], dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | np.ndarray]:
    """Pack per-environment score vectors onto the mouse's experience-order slot axis."""
    mouse_order = load_env_order().get(session.mouse_name)
    env_slot_ids = np.full(MAX_ENV_SLOTS, np.nan)
    if mouse_order is not None:
        env_slot_ids[: min(len(mouse_order), MAX_ENV_SLOTS)] = mouse_order[:MAX_ENV_SLOTS]

    output: dict[str, torch.Tensor | np.ndarray] = {
        f"{key}_env": torch.full(
            (MAX_ENV_SLOTS, value.numel()),
            torch.nan,
            dtype=value.dtype,
            device=value.device,
        )
        for key, value in global_scores.items()
        if key in _VARIANCE_KEYS
    }

    if mouse_order is not None:
        for env in np.unique(environments):
            env = int(env)
            if env < 0 or env not in mouse_order:
                continue
            slot = mouse_order.index(env)
            if slot >= MAX_ENV_SLOTS:
                continue
            for key, value in score_environment(env).items():
                env_key = f"{key}_env"
                if env_key not in output:
                    continue
                expected = output[env_key].shape[1]
                if value.numel() != expected:
                    raise ValueError(
                        f"Environment {env} returned {value.numel()} values for {key}; "
                        f"expected {expected} from the global score."
                    )
                output[env_key][slot] = value

    output["env_slot_ids"] = env_slot_ids
    return output


class PCASubspace(SubspaceModel):
    def fit(
        self,
        session: B2Session,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "train",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
    ):
        train_data, frame_behavior_train, num_neurons = self.get_session_data(session, spks_type, split, use_cell_split=False)

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        placefield = get_placefield(
            train_data.T.numpy(),
            frame_behavior_train,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        placefield_extended = torch.tensor(placefield.placefield).reshape(-1, num_neurons).T.contiguous()

        # Check for NaNs and filter if needed
        placefield_extended, train_data = self._check_and_filter_nans(placefield_extended, train_data, nan_safe=nan_safe)

        num_components = self._compute_num_components(self.max_components, train_data.shape, placefield_extended.shape)
        if self.match_dimensions:
            pca_activity = PCA(num_components=num_components, center=self.centered).fit(train_data)
        else:
            pca_activity = PCA(center=self.centered).fit(train_data)
        pca_placefields = PCA(num_components=num_components, center=self.centered).fit(placefield_extended)

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
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
    ):
        test_data, frame_behavior_test, num_neurons = self.get_session_data(session, spks_type, split, use_cell_split=False)

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        placefield = get_placefield(
            test_data.T.numpy(),
            frame_behavior_test,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        placefield_extended = torch.tensor(placefield.placefield).reshape(-1, num_neurons).T.contiguous()

        # Check for NaNs and filter if needed
        placefield_extended, _ = self._check_and_filter_nans(placefield_extended, test_data, nan_safe=nan_safe)

        subspace_activity = subspace.subspace_activity.get_components()
        subspace_placefields = subspace.subspace_placefields.get_components()
        # torch.var is invariant to mean shift, so centering doesn't affect these scores
        variance_activity = torch.var(test_data.T @ subspace_activity, dim=0)
        variance_placefields = torch.var(test_data.T @ subspace_placefields, dim=0)
        variance_placefield_placefield = torch.var(placefield_extended.T @ subspace_placefields, dim=0)

        return dict(
            variance_activity=variance_activity,
            variance_placefields=variance_placefields,
            variance_placefield_placefield=variance_placefield_placefield,
        )

    def _get_model_name(self) -> str:
        """Get the name of the model."""
        base_name = "pca_subspace"
        if self.centered:
            base_name += "_centered"
        if self.match_dimensions:
            base_name += "_with_match"
        return base_name


class SVCASubspace(SubspaceModel):
    score_schema_version = "env-v1"

    def fit(
        self,
        session: B2Session,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "train",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
    ):
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
            use_fast_sampling=True,
            session=session,
        )
        placefield_target = get_placefield(
            train_target.T.numpy(),
            frame_behavior_train,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        placefield_source_extended, placefield_target_extended = _paired_placefield_matrices(
            placefield_source,
            placefield_target,
            num_source_neurons,
            num_target_neurons,
            nan_safe,
        )
        if nan_safe and (torch.any(torch.isnan(train_source)) or torch.any(torch.isnan(train_target))):
            raise ValueError("NaN values in train_source or train_target!")

        num_components = self._compute_num_components(
            self.max_components,
            train_source.shape,
            train_target.shape,
            placefield_source_extended.shape,
            placefield_target_extended.shape,
        )

        if self.match_dimensions:
            svca_activity = SVCA(centered=self.centered, num_components=num_components)
        else:
            svca_activity = SVCA(centered=self.centered)
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
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
    ):
        (test_source, test_target), frame_behavior_test, (num_source_neurons, num_target_neurons) = self.get_session_data(
            session, spks_type, split, use_cell_split=True
        )

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        placefield_source = get_placefield(
            test_source.T.numpy(),
            frame_behavior_test,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        placefield_target = get_placefield(
            test_target.T.numpy(),
            frame_behavior_test,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        placefield_source_extended, placefield_target_extended = _paired_placefield_matrices(
            placefield_source,
            placefield_target,
            num_source_neurons,
            num_target_neurons,
            nan_safe,
        )
        if nan_safe and (torch.any(torch.isnan(test_source)) or torch.any(torch.isnan(test_target))):
            raise ValueError("NaN values in test_source or test_target!")

        result = {
            "variance_activity": subspace.subspace_activity.score(test_source, test_target)[0],
            "variance_placefields": subspace.subspace_placefields.score(test_source, test_target)[0],
            "variance_placefield_placefield": subspace.subspace_placefields.score(
                placefield_source_extended,
                placefield_target_extended,
            )[0],
        }

        def score_environment(env: int) -> dict[str, torch.Tensor]:
            env_result: dict[str, torch.Tensor] = {}
            idx_env = torch.as_tensor(np.asarray(frame_behavior_test.environment) == env)
            if int(idx_env.sum()) >= 2:
                source_env = test_source[:, idx_env]
                target_env = test_target[:, idx_env]
                env_result["variance_activity"] = subspace.subspace_activity.score(source_env, target_env)[0]
                env_result["variance_placefields"] = subspace.subspace_placefields.score(source_env, target_env)[0]

            pf_source_env = placefield_source.filter_by_environment(env)
            pf_target_env = placefield_target.filter_by_environment(env)
            if len(pf_source_env) and len(pf_target_env):
                source_pf_env, target_pf_env = _paired_placefield_matrices(
                    pf_source_env,
                    pf_target_env,
                    num_source_neurons,
                    num_target_neurons,
                    nan_safe,
                )
                if source_pf_env.shape[1] >= 2:
                    env_result["variance_placefield_placefield"] = subspace.subspace_placefields.score(
                        source_pf_env,
                        target_pf_env,
                    )[0]
            return env_result

        result.update(
            _environment_slot_scores(
                session,
                result,
                np.asarray(frame_behavior_test.environment),
                score_environment,
            )
        )
        return result

    def _get_model_name(self) -> str:
        """Get the name of the model."""
        base_name = "svca_subspace"
        if self.centered:
            base_name += "_centered"
        if self.match_dimensions:
            base_name += "_with_match"
        return base_name


class CovCovSubspace(SubspaceModel):
    score_schema_version = "env-v1"

    def fit(
        self,
        session: B2Session,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "train",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
    ):
        train_data, frame_behavior_train, num_neurons = self.get_session_data(session, spks_type, split, use_cell_split=False)

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        placefield = get_placefield(
            train_data.T.numpy(),
            frame_behavior_train,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        placefield_extended = torch.tensor(placefield.flattened()).T.contiguous()

        # Check for NaNs and filter if needed
        placefield_extended, train_data = self._check_and_filter_nans(placefield_extended, train_data, nan_safe=nan_safe)

        num_components = self._compute_num_components(self.max_components, train_data.shape, placefield_extended.shape)
        if self.match_dimensions:
            pca_activity = PCA(num_components=num_components, center=self.centered).fit(train_data)
        else:
            pca_activity = PCA(center=self.centered).fit(train_data)
        pca_placefields = PCA(num_components=num_components, center=self.centered).fit(placefield_extended)

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
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
    ):
        test_data, frame_behavior_test, num_neurons = self.get_session_data(session, spks_type, split, use_cell_split=False)

        # Also measure covariance of test placefield data for a placefield-placefield comparison as a control
        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        placefield = get_placefield(
            test_data.T.numpy(),
            frame_behavior_test,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        placefield_extended = torch.tensor(placefield.flattened()).T.contiguous()

        # Check for NaNs and filter if needed
        placefield_extended, _ = self._check_and_filter_nans(placefield_extended, test_data, nan_safe=nan_safe)
        # We're looking for train_cov^{1/2} @ test_cov @ train_cov^{1/2}
        # We can use the eigenvalues of the inner block:
        # train_eval^{1/2} @ train_evecs.T @ test_cov @ train_evecs @ train_eval^{1/2}
        # The outer train_evecs don't affect the eigenvalues and make it a bigger matrix
        train_evecs_activity = subspace.subspace_activity.get_components()
        train_eval_activity_root = torch.diag(torch.sqrt(subspace.subspace_activity.get_eigenvalues()))
        train_evecs_placefields = subspace.subspace_placefields.get_components()
        train_eval_placefields_root = torch.diag(torch.sqrt(subspace.subspace_placefields.get_eigenvalues()))

        def score_activity(data: torch.Tensor) -> dict[str, torch.Tensor]:
            data_cov = _cov_or_gram(data, self.centered)
            inner_block_activity = (
                train_eval_activity_root @ train_evecs_activity.T @ data_cov @ train_evecs_activity @ train_eval_activity_root
            )
            inner_block_placefields = (
                train_eval_placefields_root
                @ train_evecs_placefields.T
                @ data_cov
                @ train_evecs_placefields
                @ train_eval_placefields_root
            )
            return {
                "variance_activity": torch.sqrt(torch.clamp_min(torch.flipud(_eigvalsh_numpy(inner_block_activity)), 0.0)),
                "variance_placefields": torch.sqrt(torch.clamp_min(torch.flipud(_eigvalsh_numpy(inner_block_placefields)), 0.0)),
            }

        def score_placefield(placefield_data: torch.Tensor) -> torch.Tensor:
            placefield_cov = _cov_or_gram(placefield_data, self.centered)
            inner_block_pfpf = (
                train_eval_placefields_root
                @ train_evecs_placefields.T
                @ placefield_cov
                @ train_evecs_placefields
                @ train_eval_placefields_root
            )
            return torch.sqrt(torch.clamp_min(torch.flipud(_eigvalsh_numpy(inner_block_pfpf)), 0.0))

        result = score_activity(test_data)
        result["variance_placefield_placefield"] = score_placefield(placefield_extended)

        def score_environment(env: int) -> dict[str, torch.Tensor]:
            env_result: dict[str, torch.Tensor] = {}
            idx_env = torch.as_tensor(np.asarray(frame_behavior_test.environment) == env)
            if int(idx_env.sum()) >= 2:
                env_result.update(score_activity(test_data[:, idx_env]))

            placefield_env = placefield.filter_by_environment(env)
            if len(placefield_env):
                placefield_env_extended = torch.tensor(placefield_env.flattened()).T.contiguous()
                placefield_env_extended, _ = self._check_and_filter_nans(
                    placefield_env_extended,
                    test_data[:, idx_env],
                    nan_safe=nan_safe,
                )
                if placefield_env_extended.shape[1] >= 2:
                    env_result["variance_placefield_placefield"] = score_placefield(placefield_env_extended)
            return env_result

        result.update(
            _environment_slot_scores(
                session,
                result,
                np.asarray(frame_behavior_test.environment),
                score_environment,
            )
        )
        return result

    def _get_model_name(self) -> str:
        """Get the name of the model."""
        base_name = "covcov_subspace"
        if self.centered:
            base_name += "_centered"
        if self.match_dimensions:
            base_name += "_with_match"
        return base_name


class CovCovCrossvalidatedSubspace(SubspaceModel):
    def fit(
        self,
        session: B2Session,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "train",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
        split_train: bool = True,
    ):
        if split_train and split == "train":
            split0 = "train0"
            split1 = "train1"
            train0_data, frame_behavior_train0, num_neurons = self.get_session_data(session, spks_type, split0, use_cell_split=False)
            train1_data, _, _ = self.get_session_data(session, spks_type, split1, use_cell_split=False)
            _assert_finite(train0_data, "train0_data (raw sigrebase)", session)
            _assert_finite(train1_data, "train1_data (raw sigrebase)", session)
            _log_abs_max(train0_data, "train0_data (raw sigrebase)", session)
            _log_abs_max(train1_data, "train1_data (raw sigrebase)", session)

        else:
            # If any other split name or not split_train, we just divide the samples in half randomly
            train_data, frame_behavior_train, num_neurons = self.get_session_data(session, spks_type, split, use_cell_split=False)

            num_samples = train_data.shape[1]
            num_samples_split0 = num_samples // 2
            perm = torch.randperm(num_samples)
            idx_split0 = torch.sort(perm[:num_samples_split0]).values
            idx_split1 = torch.sort(perm[num_samples_split0:]).values
            train0_data = train_data[:, idx_split0]
            train1_data = train_data[:, idx_split1]
            frame_behavior_train0 = frame_behavior_train.filter(idx_split0)

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        placefield0 = get_placefield(
            train0_data.T.numpy(),
            frame_behavior_train0,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        placefield0_extended = torch.tensor(placefield0.placefield).reshape(-1, num_neurons).T.contiguous()

        # Check for NaNs and filter if needed
        placefield0_extended, train0_data = self._check_and_filter_nans(placefield0_extended, train0_data, nan_safe=nan_safe)

        num_components = self._compute_num_components(self.max_components, train0_data.shape, train1_data.shape, placefield0_extended.shape)

        # Get the root covariance matrices for activity in each split
        if self.match_dimensions:
            pca_activity0 = PCA(num_components=num_components, center=self.centered).fit(train0_data)
            pca_activity1 = PCA(num_components=num_components, center=self.centered).fit(train1_data)
        else:
            pca_activity0 = PCA(center=self.centered).fit(train0_data)
            pca_activity1 = PCA(center=self.centered).fit(train1_data)
        components0 = pca_activity0.get_components()
        components1 = pca_activity1.get_components()
        eigenvalues0 = torch.clamp_min(pca_activity0.get_eigenvalues(), 0.0)
        eigenvalues1 = torch.clamp_min(pca_activity1.get_eigenvalues(), 0.0)
        _assert_finite(eigenvalues0, "eigenvalues0 (PCA on activity split 0)", session)
        _assert_finite(eigenvalues1, "eigenvalues1 (PCA on activity split 1)", session)
        _log_abs_max(eigenvalues0, "eigenvalues0 (PCA on activity split 0)", session)
        _log_abs_max(eigenvalues1, "eigenvalues1 (PCA on activity split 1)", session)
        root_cov_activity0 = components0 @ torch.diag(torch.sqrt(eigenvalues0)) @ components0.T
        root_cov_activity1 = components1 @ torch.diag(torch.sqrt(eigenvalues1)) @ components1.T
        _assert_finite(root_cov_activity0, "root_cov_activity0", session)
        _assert_finite(root_cov_activity1, "root_cov_activity1", session)
        _log_abs_max(root_cov_activity0, "root_cov_activity0", session)
        _log_abs_max(root_cov_activity1, "root_cov_activity1", session)

        # Get the root covariance matrices for place fields in the first half split
        pca_placefields0 = PCA(num_components=num_components, center=self.centered).fit(placefield0_extended)
        pf_components0 = pca_placefields0.get_components()
        pf_eigenvalues0 = torch.clamp_min(pca_placefields0.get_eigenvalues(), 0.0)
        root_cov_placefields0 = pf_components0 @ torch.diag(torch.sqrt(pf_eigenvalues0)) @ pf_components0.T

        # Measure SVD on activity vs activity or PFs vs activity
        if self.match_dimensions:
            SVCA_activity = SVCA(centered=False, num_components=num_components).fit(root_cov_activity0, root_cov_activity1)
            SVCA_placefields = SVCA(centered=False, num_components=num_components).fit(root_cov_placefields0, root_cov_activity1)
        else:
            SVCA_activity = SVCA(centered=False).fit(root_cov_activity0, root_cov_activity1)
            SVCA_placefields = SVCA(centered=False).fit(root_cov_placefields0, root_cov_activity1)

        return Subspace(
            subspace_activity=SVCA_activity,
            subspace_placefields=SVCA_placefields,
            extras=dict(placefield0=placefield0),
        )

    def score(
        self,
        session: B2Session,
        subspace: Subspace,
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "not_train",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
        split_not_train: bool = True,
    ):
        if split_not_train and split == "not_train":
            split0 = "validation"
            split1 = "test"
            test0_data, frame_behavior_test0, num_neurons = self.get_session_data(session, spks_type, split0, use_cell_split=False)
            test1_data, frame_behavior_test1, _ = self.get_session_data(session, spks_type, split1, use_cell_split=False)
            _assert_finite(test0_data, "test0_data (raw sigrebase)", session)
            _assert_finite(test1_data, "test1_data (raw sigrebase)", session)
            _log_abs_max(test0_data, "test0_data (raw sigrebase)", session)
            _log_abs_max(test1_data, "test1_data (raw sigrebase)", session)

        else:
            # If any other split name or not split_train, we just divide the samples in half randomly
            test_data, frame_behavior_test, num_neurons = self.get_session_data(session, spks_type, split, use_cell_split=False)

            num_samples = test_data.shape[1]
            num_samples_split0 = num_samples // 2
            perm = torch.randperm(num_samples)
            idx_test0 = torch.sort(perm[:num_samples_split0]).values
            idx_test1 = torch.sort(perm[num_samples_split0:]).values
            test0_data = test_data[:, idx_test0]
            test1_data = test_data[:, idx_test1]
            frame_behavior_test0 = frame_behavior_test.filter(idx_test0)
            frame_behavior_test1 = frame_behavior_test.filter(idx_test1)

        dist_edges = self._get_placefield_dist_edges(session, hyperparameters)
        placefield0 = get_placefield(
            test0_data.T.numpy(),
            frame_behavior_test0,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        placefield0_extended = torch.tensor(placefield0.placefield).reshape(-1, num_neurons).T.contiguous()
        placefield1 = get_placefield(
            test1_data.T.numpy(),
            frame_behavior_test1,
            dist_edges=dist_edges,
            average=True,
            smooth_width=hyperparameters.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        placefield1_extended = torch.tensor(placefield1.placefield).reshape(-1, num_neurons).T.contiguous()

        # Check for NaNs and filter if needed
        placefield0_extended, test0_data = self._check_and_filter_nans(placefield0_extended, test0_data, nan_safe=nan_safe)
        placefield1_extended, test1_data = self._check_and_filter_nans(placefield1_extended, test1_data, nan_safe=nan_safe)

        num_components = self._compute_num_components(
            self.max_components, test0_data.shape, test1_data.shape, placefield0_extended.shape, placefield1_extended.shape
        )

        # Get the root covariance matrices for activity in each split
        if self.match_dimensions:
            pca_activity0 = PCA(num_components=num_components, center=self.centered).fit(test0_data)
            pca_activity1 = PCA(num_components=num_components, center=self.centered).fit(test1_data)
        else:
            pca_activity0 = PCA(center=self.centered).fit(test0_data)
            pca_activity1 = PCA(center=self.centered).fit(test1_data)
        components0 = pca_activity0.get_components()
        components1 = pca_activity1.get_components()
        eigenvalues0 = torch.clamp_min(pca_activity0.get_eigenvalues(), 0.0)
        eigenvalues1 = torch.clamp_min(pca_activity1.get_eigenvalues(), 0.0)
        _assert_finite(eigenvalues0, "eigenvalues0 (PCA on activity split 0)", session)
        _assert_finite(eigenvalues1, "eigenvalues1 (PCA on activity split 1)", session)
        _log_abs_max(eigenvalues0, "eigenvalues0 (PCA on activity split 0)", session)
        _log_abs_max(eigenvalues1, "eigenvalues1 (PCA on activity split 1)", session)
        root_cov_activity0 = components0 @ torch.diag(torch.sqrt(eigenvalues0)) @ components0.T
        root_cov_activity1 = components1 @ torch.diag(torch.sqrt(eigenvalues1)) @ components1.T
        _assert_finite(root_cov_activity0, "root_cov_activity0", session)
        _assert_finite(root_cov_activity1, "root_cov_activity1", session)
        _log_abs_max(root_cov_activity0, "root_cov_activity0", session)
        _log_abs_max(root_cov_activity1, "root_cov_activity1", session)

        # Get the root covariance matrices for place fields in each split
        pca_placefields0 = PCA(num_components=num_components, center=self.centered).fit(placefield0_extended)
        pf_components0 = pca_placefields0.get_components()
        pf_eigenvalues0 = torch.clamp_min(pca_placefields0.get_eigenvalues(), 0.0)
        root_cov_placefields0 = pf_components0 @ torch.diag(torch.sqrt(pf_eigenvalues0)) @ pf_components0.T
        pca_placefields1 = PCA(num_components=num_components, center=self.centered).fit(placefield1_extended)
        pf_components1 = pca_placefields1.get_components()
        pf_eigenvalues1 = torch.clamp_min(pca_placefields1.get_eigenvalues(), 0.0)
        root_cov_placefields1 = pf_components1 @ torch.diag(torch.sqrt(pf_eigenvalues1)) @ pf_components1.T

        # variance activity
        variance_activity = subspace.subspace_activity.score(root_cov_activity0, root_cov_activity1, normalize=False)[0]
        variance_placefields = subspace.subspace_placefields.score(root_cov_placefields0, root_cov_activity1, normalize=False)[0]
        variance_placefield_placefield = subspace.subspace_placefields.score(root_cov_placefields0, root_cov_placefields1, normalize=False)[0]

        return dict(
            variance_activity=variance_activity,
            variance_placefields=variance_placefields,
            variance_placefield_placefield=variance_placefield_placefield,
        )

    def _get_model_name(self) -> str:
        """Get the name of the model."""
        base_name = "covcov_crossvalidated_subspace"
        if self.centered:
            base_name += "_centered"
        if self.match_dimensions:
            base_name += "_with_match"
        return base_name
