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
class TrialRegistryParameters:
    """Identity of the trial registry's fixed ``even`` split.

    ``relative_size=(1, 1)`` is intentionally fixed: half of each environment's trials train the
    model and the other half test it.  Hyperparameters come from the matching ordinary ``even``
    population and therefore need no validation allocation here.
    """

    name: str = "even"
    relative_size: tuple[int, int] = (1, 1)
    split_seed: int = 0
    speed_threshold: float = 1.0

    # RegressionModel.get_split_chunk_index checks this attribute before reconstructing chunks.
    # TrialRegistry advertises an explicit trial gain-unit mode, so the numerical value is only a
    # compatibility guard and does not define the split.
    time_split_num_buffer: int = 1

    def __post_init__(self):
        if self.name != "even":
            raise ValueError("TrialRegistry currently supports only the named 'even' split")
        if self.relative_size != (1, 1):
            raise ValueError("TrialRegistry currently requires relative_size=(1, 1)")
        if self.speed_threshold < 0:
            raise ValueError("speed_threshold must be non-negative")


def split_trials_by_environment(
    trial_environment: np.ndarray,
    relative_size: tuple[int, int] = (1, 1),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproducibly divide whole trials within each environment.

    ``cross_validate_trials`` draws from NumPy's global random state, so the state is restored
    after the call.  Returned values are global trial indices aligned with ``trial_environment``.
    """
    trial_environment = np.asarray(trial_environment)
    if trial_environment.ndim != 1:
        raise ValueError("trial_environment must be one-dimensional")
    if len(relative_size) != 2 or any(size <= 0 for size in relative_size):
        raise ValueError("relative_size must contain two positive values")

    state = np.random.get_state()
    np.random.seed(seed % (2**32))
    try:
        folds = cross_validate_trials(trial_environment, list(relative_size))
    finally:
        np.random.set_state(state)

    train, test = (np.sort(np.asarray(fold, dtype=np.int64)) for fold in folds)
    combined = np.concatenate([train, test])
    if combined.size != trial_environment.size or not np.array_equal(np.sort(combined), np.arange(trial_environment.size)):
        raise ValueError("trial folds must partition every trial exactly once")
    for environment in np.unique(trial_environment):
        if not np.any(trial_environment[train] == environment) or not np.any(trial_environment[test] == environment):
            num_trials = int(np.sum(trial_environment == environment))
            raise ValueError(
                f"Environment {environment} has {num_trials} trial(s), too few to appear in both "
                "halves of an even trial split"
            )
    return train, test


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
            time_split=TrialTimeSplit(),
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
