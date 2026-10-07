"""Trial-stratified population registry for held-out regression evaluation.

This module deliberately lives beside, rather than inside, :mod:`.registry`.  The ordinary
``PopulationRegistry`` and ``RegistryParameters`` are part of the identity of long-lived
population, hyperparameter, and score caches.  Trial splitting is an independent extension with
its own parameter type and cache namespace, so introducing it cannot change any existing hash or
path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dimilibi import Population
from vrAnalysis.helpers import cross_validate_trials, stable_hash
from vrAnalysis.processors.placefields import FrameBehavior, get_frame_behavior
from vrAnalysis.sessions import B2Session

from .registry import PopulationRegistry


@dataclass(frozen=True)
class TrialTimeSplit:
    """Named indices for one 50/50 train/test split made from whole trials.

    There is intentionally no validation fold: :class:`TrialSplitRegressionConfig` inherits
    hyperparameters optimized by the ordinary ``even`` registry.  ``train0`` and ``train1`` are
    aliases of ``train`` only for API compatibility; this registry must not be used for models
    that require two independent training folds.
    """

    train: int = 0
    test: int = 1
    full: tuple[int, int] = (0, 1)
    train0: int = 0
    train1: int = 0
    half0: int = 0
    half1: int = 1
    not_train: int = 1

    def __getitem__(self, name: str):
        if name == "validation":
            raise ValueError("TrialTimeSplit has no validation fold; use inherited hyperparameters")
        if name not in self.__dataclass_fields__:
            raise KeyError(f"{self.__class__.__name__!r} has no split {name!r}")
        return getattr(self, name)


@dataclass(frozen=True)
class TrialValidationTimeSplit:
    """Named indices for a train/validation/test split made from whole trials."""

    train: int = 0
    validation: int = 1
    test: int = 2
    full: tuple[int, int, int] = (0, 1, 2)
    # The models used by the trial analyses do not need double cross-validation.  Aliasing the
    # two names to train keeps their common data-loading API usable without leaking validation.
    train0: int = 0
    train1: int = 0
    half0: tuple[int, int] = (0, 1)
    half1: int = 2
    not_train: tuple[int, int] = (1, 2)

    def __getitem__(self, name: str):
        if name not in self.__dataclass_fields__:
            raise KeyError(f"{self.__class__.__name__!r} has no split {name!r}")
        return getattr(self, name)


@dataclass(frozen=True)
class TrialRegistryParameters:
    """Identity of an isolated two- or three-fold environment-stratified trial split."""

    name: str = "even"
    relative_size: tuple[int, ...] = (1, 1)
    split_seed: int = 0
    speed_threshold: float = 1.0

    # RegressionModel.get_split_chunk_index checks this attribute before reconstructing chunks.
    # TrialRegistry advertises an explicit trial gain-unit mode, so the numerical value is only a
    # compatibility guard and does not define the split.
    time_split_num_buffer: int = 1

    def __post_init__(self):
        if self.name != "even":
            raise ValueError("TrialRegistry currently supports only the named 'even' split")
        if self.relative_size not in ((1, 1), (2, 1, 1)):
            raise ValueError("TrialRegistry supports relative_size=(1, 1) or (2, 1, 1)")
        if self.speed_threshold < 0:
            raise ValueError("speed_threshold must be non-negative")


@dataclass(frozen=True)
class EnvironmentTrialRegistryParameters(TrialRegistryParameters):
    """Cache identity for one environment-specific, explicitly filtered population."""

    relative_size: tuple[int, ...] = (2, 1, 1)
    environment: int = -1
    selected_roi_indices: tuple[int, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if self.environment < 0:
            raise ValueError("environment must be non-negative")
        if len(self.selected_roi_indices) < 4:
            raise ValueError("selected_roi_indices must contain at least four ROIs")


def split_trials_by_environment(
    trial_environment: np.ndarray,
    relative_size: tuple[int, ...] = (1, 1),
    seed: int = 0,
) -> tuple[np.ndarray, ...]:
    """Reproducibly divide whole trials within each environment.

    ``cross_validate_trials`` draws from NumPy's global random state, so the state is restored
    after the call.  Returned values are global trial indices aligned with ``trial_environment``.
    """
    trial_environment = np.asarray(trial_environment)
    if trial_environment.ndim != 1:
        raise ValueError("trial_environment must be one-dimensional")
    if len(relative_size) not in (2, 3) or any(size <= 0 for size in relative_size):
        raise ValueError("relative_size must contain two or three positive values")

    state = np.random.get_state()
    np.random.seed(seed % (2**32))
    try:
        folds = cross_validate_trials(trial_environment, list(relative_size))
    finally:
        np.random.set_state(state)

    result = tuple(np.sort(np.asarray(fold, dtype=np.int64)) for fold in folds)
    combined = np.concatenate(result)
    if combined.size != trial_environment.size or not np.array_equal(np.sort(combined), np.arange(trial_environment.size)):
        raise ValueError("trial folds must partition every trial exactly once")
    for environment in np.unique(trial_environment):
        if any(not np.any(trial_environment[fold] == environment) for fold in result):
            num_trials = int(np.sum(trial_environment == environment))
            raise ValueError(
                f"Environment {environment} has {num_trials} trial(s), too few to appear in all " f"{len(result)} folds of the requested trial split"
            )
    return result


class TrialRegistry(PopulationRegistry):
    """Population registry whose time folds contain disjoint whole trials.

    The base registry supplies the exact ROI population and source/target cell split.  Only sample
    selection and time folds are rebuilt.  Trial populations are cached below
    ``population-registry/trial`` and their identity includes both the trial parameters and the
    base registry parameters and exact base cell assignment, so they cannot collide with ordinary
    populations or silently outlive a manually regenerated base cell split.

    ``gain_unit_mode='trial'`` is consumed by ``RegressionModel.get_split_chunk_index``: structured
    gain therefore receives a constant chunk id and its existing ``chunk | trial`` grouping
    reduces exactly to one unit per trial.
    """

    gain_unit_mode: str = "trial"

    def __init__(
        self,
        base_registry: PopulationRegistry,
        registry_params: TrialRegistryParameters = TrialRegistryParameters(),
        autosave: bool = True,
    ):
        self.base_registry = base_registry
        super().__init__(
            registry_paths=base_registry.registry_paths,
            registry_params=registry_params,
            time_split=TrialTimeSplit() if len(registry_params.relative_size) == 2 else TrialValidationTimeSplit(),
            autosave=autosave,
        )
        self.trial_registry_path.mkdir(parents=True, exist_ok=True)

    @property
    def trial_registry_path(self) -> Path:
        """Dedicated population-cache directory for trial splits."""
        return self.registry_paths.registry_path / "trial"

    def _make_population(self, session: B2Session) -> tuple[Population, FrameBehavior]:
        """Reuse base cells and assign every valid fast-running frame by its whole trial."""
        base_population, _ = self.base_registry.get_population(session, session.params.spks_type)

        frame_behavior = get_frame_behavior(session, clear_one_cache=True)
        idx_valid = frame_behavior.valid_frames(full_check=True)
        idx_valid &= frame_behavior.speed >= self.registry_params.speed_threshold
        idx_samples = np.flatnonzero(idx_valid)
        frame_behavior = frame_behavior.filter(idx_valid)
        if len(frame_behavior) == 0:
            raise ValueError(f"No valid fast-running frames in {session.session_print()}")

        split_seed = int(stable_hash(session.session_uid, self.registry_params.split_seed, "trial-split"), 16)
        trial_folds = split_trials_by_environment(
            np.asarray(session.trial_environment),
            self.registry_params.relative_size,
            split_seed,
        )

        frame_trial = np.asarray(frame_behavior.trial, dtype=np.int64)
        time_split_indices = [torch.as_tensor(np.flatnonzero(np.isin(frame_trial, fold)), dtype=torch.long) for fold in trial_folds]
        if any(indices.numel() == 0 for indices in time_split_indices):
            raise ValueError(f"The even trial split left an empty train or test fold in {session.session_print()}")

        membership = np.zeros(len(frame_behavior), dtype=np.int8)
        for indices in time_split_indices:
            membership[np.asarray(indices)] += 1
        if not np.all(membership == 1):
            raise ValueError("Every retained frame must belong to exactly one trial fold")

        idx_neurons = base_population.idx_neurons.detach().cpu().numpy()
        population = Population(
            session.spks.T,
            generate_splits=False,
            idx_samples=idx_samples,
            idx_neurons=idx_neurons,
        )
        population.cell_split_indices = [indices.detach().cpu().clone() for indices in base_population.cell_split_indices]
        population.time_split_indices = time_split_indices
        return population, frame_behavior

    def _get_population_path(self, session: B2Session) -> Path:
        return self.trial_registry_path / f"{self._get_unique_id(session)}.joblib"

    def _get_params_path(self) -> Path:
        identity = stable_hash("trial-registry", self.registry_params, self.base_registry.registry_params)
        return self.trial_registry_path / f"params_{identity}.joblib"

    def _get_unique_id(self, session: B2Session) -> str:
        session_name = ".".join(session.session_name)
        base_population, _ = self.base_registry.get_population(session, session.params.spks_type)
        base_cells = (
            base_population.idx_neurons.detach().cpu().tolist(),
            [indices.detach().cpu().tolist() for indices in base_population.cell_split_indices],
        )
        identity = stable_hash(
            "trial-registry",
            self.registry_params,
            self.base_registry.registry_params,
            base_cells,
        )
        return f"{session_name}_{identity}"


class EnvironmentTrialRegistry(TrialRegistry):
    """Trial registry restricted to one environment and an explicit set of place cells.

    ``selected_roi_indices`` contains session-level ROI identities.  The base population's
    source/target assignment is retained exactly; filtering never repartitions neurons.
    """

    def __init__(
        self,
        base_registry: PopulationRegistry,
        environment: int,
        selected_roi_indices: np.ndarray,
        registry_params: TrialRegistryParameters = TrialRegistryParameters(relative_size=(2, 1, 1)),
        autosave: bool = True,
    ):
        self.environment = int(environment)
        self.selected_roi_indices = np.sort(np.asarray(selected_roi_indices, dtype=np.int64))
        identity_params = EnvironmentTrialRegistryParameters(
            name=registry_params.name,
            relative_size=registry_params.relative_size,
            split_seed=registry_params.split_seed,
            speed_threshold=registry_params.speed_threshold,
            time_split_num_buffer=registry_params.time_split_num_buffer,
            environment=self.environment,
            selected_roi_indices=tuple(int(roi) for roi in self.selected_roi_indices),
        )
        super().__init__(base_registry, registry_params=identity_params, autosave=autosave)

    def _make_population(self, session: B2Session) -> tuple[Population, FrameBehavior]:
        base_population, _ = self.base_registry.get_population(session, session.params.spks_type)
        base_neurons = np.asarray(base_population.idx_neurons.detach().cpu(), dtype=np.int64)
        selected = set(self.selected_roi_indices.tolist())

        split_rois = []
        for indices in base_population.cell_split_indices:
            absolute = base_neurons[np.asarray(indices.detach().cpu(), dtype=np.int64)]
            split_rois.append(absolute[np.isin(absolute, self.selected_roi_indices)])
        if any(rois.size < 2 for rois in split_rois):
            raise ValueError("Environment place-cell population requires at least two source and two target ROIs")
        if set(split_rois[0]).intersection(split_rois[1]):
            raise ValueError("Source and target ROI membership must remain disjoint")
        if set(np.concatenate(split_rois).tolist()) != selected:
            raise ValueError("Every selected ROI must belong to one base source/target split")

        frame_behavior = get_frame_behavior(session, clear_one_cache=True)
        idx_valid = frame_behavior.valid_frames(full_check=True)
        idx_valid &= frame_behavior.speed >= self.registry_params.speed_threshold
        idx_valid &= np.asarray(frame_behavior.environment) == self.environment
        idx_samples = np.flatnonzero(idx_valid)
        frame_behavior = frame_behavior.filter(idx_valid)
        if len(frame_behavior) == 0:
            raise ValueError(f"No valid frames for environment {self.environment} in {session.session_print()}")

        trial_environment = np.asarray(session.trial_environment)
        environment_trials = np.flatnonzero(trial_environment == self.environment)
        split_seed = int(stable_hash(session.session_uid, self.environment, self.registry_params.split_seed, "trial-split"), 16)
        local_folds = split_trials_by_environment(trial_environment[environment_trials], self.registry_params.relative_size, split_seed)
        trial_folds = tuple(environment_trials[fold] for fold in local_folds)
        frame_trial = np.asarray(frame_behavior.trial, dtype=np.int64)
        time_split_indices = [torch.as_tensor(np.flatnonzero(np.isin(frame_trial, fold)), dtype=torch.long) for fold in trial_folds]
        if any(indices.numel() == 0 for indices in time_split_indices):
            raise ValueError(f"Environment {self.environment} has an empty trial fold")

        idx_neurons = np.concatenate(split_rois)
        population = Population(
            session.spks.T,
            generate_splits=False,
            idx_samples=idx_samples,
            idx_neurons=idx_neurons,
        )
        population.cell_split_indices = [
            torch.arange(split_rois[0].size, dtype=torch.long),
            torch.arange(split_rois[0].size, idx_neurons.size, dtype=torch.long),
        ]
        population.time_split_indices = time_split_indices
        return population, frame_behavior

    def _get_unique_id(self, session: B2Session) -> str:
        base_id = super()._get_unique_id(session)
        identity = stable_hash("environment-placecells", self.environment, self.selected_roi_indices.tolist())
        return f"{base_id}_env{self.environment}_{identity}"
