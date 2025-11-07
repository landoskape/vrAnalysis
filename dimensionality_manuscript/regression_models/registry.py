from dataclasses import dataclass, field
from pathlib import Path
from joblib import dump, load
from vrAnalysis import files
from vrAnalysis.sessions import B2Session
from dimilibi import Population


@dataclass(frozen=True)
class RegistryPaths:
    """Paths to the files used in the regression model analysis."""

    manuscript_path: Path = files.local_data_path() / "dimensionality-manuscript"
    cache_path: Path = manuscript_path / "cache"
    registry_path: Path = cache_path / "population-registry"


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
    val: int = 2
    test: int = 3
    train: tuple[int, int] = (0, 1)


@dataclass(frozen=True)
class RegistryParameters:
    time_split_groups: int = 4
    time_split_relative_size: tuple[int] = field(default_factory=lambda: (4, 4, 1, 1))
    time_split_chunks_per_group: int = 10
    cell_split_force_even: bool = False

    @property
    def time_split_prms(self):
        return dict(
            num_groups=self.time_split_groups,
            relative_size=self.time_split_relative_size,
            chunks_per_group=self.time_split_chunks_per_group,
        )

    @property
    def cell_split_prms(self):
        return dict(
            force_even=self.cell_split_force_even,
        )


class PopulationRegistry:
    """Registry for population objects for the regression model analysis.

    This registry is used to store population objects for each session in the analysis.
    It is used to avoid re-creating population objects for each session and to avoid
    loading the population object from disk every time it is needed.

    Attributes
    ----------
    registry_path : Path
        The path to the registry directory.
    registry_params : RegistryParameters
        The parameters to use for the registry.
    """

    def __init__(self, registry_path: Path, registry_params: RegistryParameters):
        self.registry_path = registry_path
        self.registry_params = registry_params

    def get_population(self, session: B2Session) -> Population:
        """Get the population object for a session.

        If the population object already exists, it will be loaded from the registry.
        Otherwise, a new population object will be created and saved to the registry.

        Parameters
        ----------
        session : B2Session
            The session to get the population for.

        Returns
        -------
        npop : Population
            The population object for the session.
        """
        if self._check_population_exists(session):
            return self._load_population(session)
        else:
            npop = self._make_population(session)
            self._save_population(session, npop)
            return npop

    def _make_population(self, session: B2Session) -> Population:
        """Create a new population object for a session.

        Parameters
        ----------
        session : B2Session
            The session to create a population for.

        Returns
        -------
        npop : Population
            The population object for the session.
        """
        npop = Population(session.spks.T, time_split_prms=self.registry_params.time_split_prms)
        return npop

    def _save_population(self, session: B2Session, population: Population) -> None:
        """Save a population object to the registry.

        Uses the population indices dictionary to save a lightweight summary of the population.

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

    def _load_population(self, session: B2Session) -> Population:
        """Load a population object from the registry.

        Parameters
        ----------
        session : B2Session
            The session to load the population for.

        Returns
        -------
        population : Population
            The population object loaded from the registry.
        """
        ppath = self._get_population_path(session)
        indices_dict = load(ppath)
        return Population.make_from_indices(indices_dict, session.spks.T)

    def _check_population_exists(self, session: B2Session):
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

    def _get_population_path(self, session: B2Session):
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
        return self.registry_path / (self._get_unique_id(session) + ".joblib")

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
        params_hash = hash(RegistryParameters())
        return f"{session_name}_{params_hash}"
