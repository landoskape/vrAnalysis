from dataclasses import dataclass
from typing import Optional, Literal, overload
from pathlib import Path
from joblib import dump, load
import numpy as np
from vrAnalysis import files
from vrAnalysis.helpers import stable_hash
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_frame_behavior, FrameBehavior
from vrAnalysis.processors.support import convert_position_to_bins
from dimilibi import Population
from .base import RegressionModel
from .models import PlaceFieldModel, ReducedRankRegressionModel  # , RBFPosModel
from .hyperparameters import PlaceFieldHyperparameters, ReducedRankRegressionHyperparameters, HyperparametersBase

# Type alias for model names
ModelName = Literal[
    "external_placefield_1d",
    "internal_placefield_1d",
    "external_placefield_1d_gain",
    "internal_placefield_1d_gain",
    "rbfpos",
    "rrr",
]


@dataclass(frozen=True)
class RegistryPaths:
    """Paths to the files used in the regression model analysis."""

    manuscript_path: Path = files.local_data_path() / "dimensionality-manuscript"
    cache_path: Path = manuscript_path / "cache"
    registry_path: Path = cache_path / "population-registry"
    hyperparameter_path: Path = cache_path / "hyperparameters"
    score_path: Path = cache_path / "scores"
    error_path: Path = cache_path / "errors"

    def __post_init__(self):
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.hyperparameter_path.mkdir(parents=True, exist_ok=True)
        self.score_path.mkdir(parents=True, exist_ok=True)
        self.error_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RegistryParameters:
    speed_threshold: float = 1.0
    time_split_groups: int = 4
    time_split_relative_size: tuple[int] = (4, 4, 1, 1)
    time_split_chunks_per_group: int = 10
    time_split_num_buffer: int = 3
    cell_split_force_even: bool = False

    @property
    def time_split_prms(self):
        return dict(
            num_groups=self.time_split_groups,
            relative_size=self.time_split_relative_size,
            chunks_per_group=self.time_split_chunks_per_group,
            num_buffer=self.time_split_num_buffer,
        )

    @property
    def cell_split_prms(self):
        return dict(
            force_even=self.cell_split_force_even,
        )


@dataclass(frozen=True)
class TimeSplit:
    """Named time splits for the regression model analysis.

    We do train/val/test splits for cross-validated analyses and optimizing hyperparameters
    wherever relevant. However, because the RBF(POS) model needs two independent training
    sets to prevent non-spatial leakage between activity and position activity, we define
    two independent train sets which are typically combined for training in all other models.
    """

    train_0: int = 0
    train_1: int = 1
    validation: int = 2
    test: int = 3
    train: tuple[int, int] = (0, 1)
    full: tuple[int, int, int, int] = (0, 1, 2, 3)

    def __getitem__(self, name):
        # Enables access like time_split[split_name] to get the split index/tuple.
        # Accept both string keys ("full") and int keys (0, 1, etc.)
        if isinstance(name, str):
            if name in self.__dataclass_fields__:
                return getattr(self, name)
            raise KeyError(f"{self.__class__.__name__!r} object has no attribute {name!r}")
        else:
            raise KeyError(f"{self.__class__.__name__!r} indices must be str or int, not {type(name).__name__}")


class PopulationRegistry:
    """Registry for population objects for the regression model analysis.

    This registry is used to store population objects for each session in the analysis.
    It is used to avoid re-creating population objects for each session and to avoid
    loading the population object from disk every time it is needed.

    Attributes
    ----------
    registry_paths : RegistryPaths
        The paths to the registry directory.
    registry_params : RegistryParameters
        The parameters to use for the registry.
    time_split : TimeSplit
        The time split to use for the population.
    autosave : bool
        Whether to automatically save the population object to the registry after creating it.
    """

    def __init__(
        self,
        registry_paths: RegistryPaths = RegistryPaths(),
        registry_params: RegistryParameters = RegistryParameters(),
        time_split: TimeSplit = TimeSplit(),
        autosave: bool = True,
    ):
        self.registry_paths = registry_paths
        self.registry_params = registry_params
        self.time_split = time_split
        self.autosave = autosave

    def get_population(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        force_remake: bool = False,
    ) -> tuple[Population, FrameBehavior]:
        """Get the population object for a session.

        If the population object already exists, it will be loaded from the registry.
        Otherwise, a new population object will be created and saved to the registry.

        Parameters
        ----------
        session : B2Session
            The session to get the population for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session
            provided as input.
        force_remake : bool
            If True, will remake the population object even if it already exists.
            (default is False)

        Returns
        -------
        population : Population
            The population object for the session.
        frame_behavior : FrameBehavior
            The frame behavior object for the session.
        """
        # If a spks_type is provided, use it to set the session's spks_type
        if spks_type is not None:
            session.params.spks_type = spks_type
        if self._check_population_exists(session) and not force_remake:
            population, frame_behavior = self._load_population(session)
        else:
            population, frame_behavior = self._make_population(session)
            if self.autosave:
                self._save_population(session, population)
        return population, frame_behavior

    def clear_population(self, session: B2Session) -> None:
        """Remove the population object for a session from the registry.

        If the population object exists, it will be removed. If it doesn't exist,
        this method will return None without raising an error.

        Parameters
        ----------
        session : B2Session
            The session to clear the population for.
        """
        ppath = self._get_population_path(session)
        ppath.unlink(missing_ok=True)

    def _make_population(self, session: B2Session) -> tuple[Population, FrameBehavior]:
        """Create a new population object for a session.

        Parameters
        ----------
        session : B2Session
            The session to create a population for.

        Returns
        -------
        population : Population
            The population object for the session.
        frame_behavior : FrameBehavior
            The frame behavior object for the session.
        """
        # Calculate whether the frame is "fast" e.g. if the mouse is moving above the speed threshold
        frame_behavior = get_frame_behavior(session, clear_one_cache=True)
        fast_frame = frame_behavior.speed >= self.registry_params.speed_threshold
        idx_valid = frame_behavior.valid_frames() & fast_frame
        frame_behavior = frame_behavior.filter(idx_valid)

        # Check that each env/position combo is present in every split
        max_attempts = 10
        attempts = 0

        # Hard coded because this is the most high resolution place fields we are using in models.py
        max_num_bins = 100
        bin_edges = np.linspace(0, session.env_length[0], max_num_bins + 1)
        while True:
            # Can only do this so many times...
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(f"Failed to make a valid population after {max_attempts} attempts.")

            # Make a new population which defines the cell splits and time splits
            population = Population(
                session.spks.T,
                time_split_prms=self.registry_params.time_split_prms,
                cell_split_prms=self.registry_params.cell_split_prms,
                idx_samples=np.where(idx_valid)[0],
                idx_neurons=np.where(session.idx_rois)[0],
            )

            # Check that each env/position combo is present in every split
            environments = np.unique(frame_behavior.environment)

            splits = ["train_0", "train_1"]
            num_splits = len(splits)
            valid_split_positions = np.zeros((num_splits, len(environments), len(bin_edges) - 1), dtype=bool)
            for i, split in enumerate(splits):
                split_idx = self.time_split[split]
                idx = np.array(population.get_split_times(split_idx, within_idx_samples=False))
                split_behavior = frame_behavior.filter(idx)
                for ienv, env in enumerate(environments):
                    idx_env = split_behavior.environment == env
                    if np.any(idx_env):
                        included_bins = np.unique(convert_position_to_bins(split_behavior.position[idx_env], bin_edges))
                        valid_split_positions[i, ienv, included_bins] = True
                    else:
                        # This means an environment wasn't in at least one of the splits,
                        # try building the population splits again!
                        continue

            # Figure out the valid positions
            position_bins = convert_position_to_bins(frame_behavior.position, bin_edges)
            valid_positions = np.all(valid_split_positions, axis=0)
            valid_frames = np.zeros(len(frame_behavior), dtype=bool)
            for ienv, env in enumerate(environments):
                idx_env = frame_behavior.environment == env
                in_valid_position = valid_positions[ienv, position_bins]
                valid_frames[idx_env & in_valid_position] = True

            if np.sum(valid_frames) / len(valid_frames) < 0.9:
                # Too many invalid frames, try again!
                continue

            # Otherwise, filter the time splits
            time_split_indices = population.time_split_indices.copy()
            for itsi, time_split_index in enumerate(time_split_indices):
                _valid = valid_frames[time_split_index]
                time_split_indices[itsi] = time_split_index[_valid]

            # Save the new time splits
            population.time_split_indices = time_split_indices
            break

        return population, frame_behavior

    def _save_population(self, session: B2Session, population: Population) -> None:
        """Save a population object to the registry.

        Uses the population indices dictionary to save a lightweight summary of the population.
        Also saves the registry parameters to a file if it doesn't exist.

        Parameters
        ----------
        session : B2Session
            The session to save the population for.
        population : Population
            The population object to save.
        """
        indices_dict = population.get_indices_dict()
        ppath = self._get_population_path(session)
        dump(indices_dict, ppath)

        if not self._check_params_exists():
            dump(self.registry_params, self._get_params_path())

    def _load_population(self, session: B2Session) -> tuple[Population, FrameBehavior]:
        """Load a population object from the registry.

        Parameters
        ----------
        session : B2Session
            The session to load the population for.

        Returns
        -------
        population : Population
            The population object loaded from the registry.
        frame_behavior : FrameBehavior
            The frame behavior object for the session.
        """
        ppath = self._get_population_path(session)
        indices_dict = load(ppath)
        frame_behavior = get_frame_behavior(session, clear_one_cache=True).filter(indices_dict["idx_samples"])
        return Population.make_from_indices(indices_dict, session.spks.T), frame_behavior

    def _prepare_data(self, session: B2Session) -> tuple[FrameBehavior, np.ndarray, np.ndarray]:
        """Prepare the data for the population.

        Will use the frame_behavior for the session to filter valid samples and use
        the session object to filter valid neurons. Uses the idx_neurons, idx_samples
        functionality of Population objects to do the filtering.

        Parameters
        ----------
        session : B2Session
            The session to prepare the data for.

        Returns
        -------
        frame_behavior : FrameBehavior
            The frame behavior object for the session.
        idx_samples : np.ndarray
            The indices to the valid samples.
        idx_neurons : np.ndarray
            The indices to the valid neurons.
        """
        # Calculate whether the frame is "fast" e.g. if the mouse is moving above the speed threshold
        frame_behavior = get_frame_behavior(session, clear_one_cache=True)
        fast_frame = frame_behavior.speed >= self.registry_params.speed_threshold
        idx_valid = frame_behavior.valid_frames() & fast_frame
        frame_behavior = frame_behavior.filter(idx_valid)

        # Return frame_behavior and indices to valid samples and neurons
        return frame_behavior, np.where(idx_valid)[0], np.where(session.idx_rois)[0]

    def _check_population_exists(self, session: B2Session) -> bool:
        """Check if a population exists for a session.

        Parameters
        ----------
        session : B2Session
            The session to check for a population.

        Returns
        -------
        bool
            True if the population exists, False otherwise.
        """
        return self._get_population_path(session).exists()

    def _check_params_exists(self) -> bool:
        """Check if the parameters file exists.

        Returns
        -------
        bool
            True if the parameters file exists, False otherwise.
        """
        return self._get_params_path().exists()

    def _get_population_path(self, session: B2Session) -> Path:
        """Get the path to the population file for a session.

        Parameters
        ----------
        session : B2Session
            The session to get the population path for.

        Returns
        -------
        pathlib.Path
            The path to the population file for the session.
        """
        return self.registry_paths.registry_path / (self._get_unique_id(session) + ".joblib")

    def _get_params_path(self) -> Path:
        """Get the path to the parameters file.

        Returns
        -------
        pathlib.Path
            The path to the parameters file.
        """
        return self.registry_paths.registry_path / ("params_" + stable_hash(self.registry_params) + ".joblib")

    def _get_unique_id(self, session: B2Session):
        """Get a unique identifier for a population.

        This includes the session name and a hash of the registry parameters.
        This way each population that is stored is uniquely identified by the
        session and the way the population timesplits and cellsplits are done.

        Parameters
        ----------
        session : B2Session
            The session to get the unique id for.

        Returns
        -------
        str
            The unique id for the population.
        """
        session_name = ".".join(session.session_name)
        params_hash = stable_hash(RegistryParameters())
        return f"{session_name}_{params_hash}"


MODEL_NAMES: tuple[ModelName] = (
    "external_placefield_1d",
    "internal_placefield_1d",
    "external_placefield_1d_gain",
    "internal_placefield_1d_gain",
    "rbfpos",
    "rrr",
)


@overload
def get_model(
    model_name: Literal["external_placefield_1d", "internal_placefield_1d", "external_placefield_1d_gain", "internal_placefield_1d_gain"],
    population_registry: PopulationRegistry,
    hyperparameters: Optional[HyperparametersBase] = None,
) -> PlaceFieldModel: ...


@overload
def get_model(
    model_name: Literal["rbfpos"],
    population_registry: PopulationRegistry,
    hyperparameters: Optional[HyperparametersBase] = None,
) -> RegressionModel:  # Update to RBFPosModel when implemented!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ...


@overload
def get_model(
    model_name: Literal["rrr"],
    population_registry: PopulationRegistry,
    hyperparameters: Optional[HyperparametersBase] = None,
) -> ReducedRankRegressionModel: ...


def get_model(
    model_name: ModelName,
    population_registry: PopulationRegistry,
    hyperparameters: Optional[HyperparametersBase] = None,
) -> RegressionModel:
    """Get a model object for a model name.

    Parameters
    ----------
    model_name : ModelName
        The name of the model to get.
    population_registry : PopulationRegistry
        The population registry to use for the model.
    hyperparameters : Optional[HyperparametersBase]
        The hyperparameters to use for the model. If None, uses the default hyperparameters for the model.

    Returns
    -------
    model : RegressionModel
        The model object for the model name. The specific type depends on model_name.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Model {model_name} not found in registry.")

    if model_name == "external_placefield_1d":
        hyperparameters = hyperparameters or PlaceFieldHyperparameters()
        return PlaceFieldModel(population_registry, internal=False, gain=False, hyperparameters=PlaceFieldHyperparameters())
    if model_name == "internal_placefield_1d":
        hyperparameters = hyperparameters or PlaceFieldHyperparameters()
        return PlaceFieldModel(population_registry, internal=True, gain=False, hyperparameters=PlaceFieldHyperparameters())
    if model_name == "external_placefield_1d_gain":
        hyperparameters = hyperparameters or PlaceFieldHyperparameters()
        return PlaceFieldModel(population_registry, internal=False, gain=True, hyperparameters=PlaceFieldHyperparameters())
    if model_name == "internal_placefield_1d_gain":
        hyperparameters = hyperparameters or PlaceFieldHyperparameters()
        return PlaceFieldModel(population_registry, internal=True, gain=True, hyperparameters=PlaceFieldHyperparameters())
    # if model_name == "rbfpos":
    #     return RBFPosModel(population_registry)
    if model_name == "rrr":
        hyperparameters = hyperparameters or ReducedRankRegressionHyperparameters()
        return ReducedRankRegressionModel(population_registry, hyperparameters=hyperparameters)

    raise ValueError(f"Model {model_name} not found in registry.")
