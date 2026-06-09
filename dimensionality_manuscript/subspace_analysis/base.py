from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, NamedTuple, Union, Any
from joblib import dump, load
import numpy as np
import torch
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.helpers import stable_hash
from ..regression_models.hyperparameters import PlaceFieldHyperparameters
from ..regression_models.base import ActivityParameters

if TYPE_CHECKING:
    from ..registry import PopulationRegistry, SplitName
    from dimilibi import PCA, SVCA
    from vrAnalysis.processors.placefields import FrameBehavior


def _eigh_numpy(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition via numpy to avoid PyTorch LAPACK bugs (e.g. illegal SVD argument)."""
    vals, vecs = np.linalg.eigh(A.numpy())
    return torch.from_numpy(vals.copy()), torch.from_numpy(vecs.copy())


def _eigvalsh_numpy(A: torch.Tensor) -> torch.Tensor:
    """Eigenvalues via numpy to avoid PyTorch LAPACK bugs (e.g. illegal SVD argument)."""
    return torch.from_numpy(np.linalg.eigvalsh(A.numpy()).copy())


class Subspace(NamedTuple):
    subspace_activity: Union["PCA", "SVCA", "torch.Tensor"]
    subspace_placefields: Union["PCA", "SVCA", "torch.Tensor"]
    extras: dict[str, Any]


class SubspaceModel(ABC):
    """Base class for subspace models with common data loading and processing methods."""

    def __init__(
        self,
        registry: "PopulationRegistry",
        centered: bool = False,
        activity_parameters: ActivityParameters = ActivityParameters(center=False, scale=False, scale_type="none"),
        max_components: int = 300,
        match_dimensions: bool = False,
        autosave: bool = True,
    ):
        """Initialize the subspace model.

        Parameters
        ----------
        registry : PopulationRegistry
            The registry to use for getting population data.
        centered : bool
            Whether to center data before the linear algebra step. Default is False.
        activity_parameters : ActivityParameters
            Controls scaling of activity data. The ``center`` field is ignored here;
            use the ``centered`` parameter instead. Default is raw (no scaling).
        max_components : int
            Maximum number of components to use. Default is 300.
        match_dimensions : bool
            Whether to match the dimensions of the activity and placefields. Default is False.
        autosave : bool
            Whether to automatically save results to cache. Default is True.
        """
        self.registry = registry
        self.centered = centered
        self.activity_parameters = activity_parameters
        self.max_components = max_components
        self.match_dimensions = match_dimensions
        self.autosave = autosave

    def get_session_data(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        split: "SplitName",
        use_cell_split: bool = False,
    ) -> tuple[torch.Tensor, "FrameBehavior", int]:
        """Get the activity data and frame behavior for a session.

        Parameters
        ----------
        session : B2Session
            The session to get the activity data for.
        spks_type : SpksTypes
            The type of spike data to use.
        split : "SplitName"
            The split to use for the activity data.
        use_cell_split : bool
            If True, returns source and target data separately (for SVCASubspace).
            If False, returns all neurons together. Default is False.

        Returns
        -------
        tuple[torch.Tensor, FrameBehavior, int] or tuple[tuple[torch.Tensor, torch.Tensor], FrameBehavior, tuple[int, int]]
            If use_cell_split is False:
                - data: torch.Tensor of shape (num_neurons, num_timepoints)
                - frame_behavior: Filtered FrameBehavior object
                - num_neurons: int
            If use_cell_split is True:
                - (source_data, target_data): tuple of torch.Tensors
                - frame_behavior: Filtered FrameBehavior object
                - (num_source_neurons, num_target_neurons): tuple of ints
        """
        split_idx = self.registry.time_split[split]
        population, frame_behavior = self.registry.get_population(session, spks_type=spks_type)

        _population_kwargs = dict(
            scale=self.activity_parameters.scale,
            scale_type=self.activity_parameters.scale_type,
            pre_split=self.activity_parameters.presplit,
        )

        if use_cell_split:
            # For SVCASubspace - returns source and target separately
            source_data, target_data = population.get_split_data(split_idx, **_population_kwargs)
            frame_behavior_filtered = frame_behavior.filter(population.get_split_times(split_idx, within_idx_samples=False))
            num_source_neurons = len(population.cell_split_indices[0])
            num_target_neurons = len(population.cell_split_indices[1])
            return (source_data, target_data), frame_behavior_filtered, (num_source_neurons, num_target_neurons)
        else:
            # For PCASubspace and CVPCASubspace - returns all neurons together
            num_neurons = len(population.idx_neurons)
            data = population.apply_split(population.data[population.idx_neurons], split_idx, prefiltered=False, **_population_kwargs)
            frame_behavior_filtered = frame_behavior.filter(population.get_split_times(split_idx, within_idx_samples=False))
            return data, frame_behavior_filtered, num_neurons

    def _get_placefield_dist_edges(self, session: B2Session, hyperparameters: PlaceFieldHyperparameters) -> np.ndarray:
        """Compute distance edges for placefield calculation.

        Parameters
        ----------
        session : B2Session
            The session containing environment information.
        hyperparameters : PlaceFieldHyperparameters
            The hyperparameters containing num_bins.

        Returns
        -------
        np.ndarray
            Array of distance edges for placefield binning.
        """
        return np.linspace(0, session.env_length[0], hyperparameters.num_bins + 1)

    def _compute_num_components(self, max_components: int, *shapes: tuple[int, ...]) -> int:
        """Compute number of components given max and data shapes.

        Parameters
        ----------
        max_components : int
            Maximum number of components to use.
        *shapes : tuple[int, ...]
            Variable number of shape tuples (num_features, num_samples) to consider.

        Returns
        -------
        int
            The minimum of max_components and all shape dimensions.
        """
        all_dims = []
        for shape in shapes:
            all_dims.extend(shape)
        return min(max_components, *all_dims)

    @abstractmethod
    def fit(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        split: "SplitName",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        nan_safe: bool = False,
    ):
        """Fit the subspace model on the given session.

        Parameters
        ----------
        session : B2Session
            The session to fit the model on.
        spks_type : SpksTypes
            The type of spike data to use.
        split : "SplitName"
            The split to use for fitting.
        hyperparameters : PlaceFieldHyperparameters
            The hyperparameters to use.
        nan_safe : bool
            If True, will raise on NaN values in placefield data.
            If False, will filter out NaN samples. Default is False.
        """
        pass

    def _check_and_filter_nans(
        self,
        placefield_data: torch.Tensor,
        activity_data: torch.Tensor,
        nan_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check for NaNs in placefield data and optionally filter them.

        Parameters
        ----------
        placefield_data : torch.Tensor
            The placefield data to check for NaNs. Shape should be (num_neurons, num_samples_placefields).
        activity_data : torch.Tensor
            The corresponding activity data to filter if NaNs are found. Shape should be (num_neurons, num_samples_activity).
        nan_safe : bool
            If True, will raise an error if NaNs are found. If False, will filter out NaN samples.
            Default is False.

        Returns
        -------
        placefield_data_filtered : torch.Tensor
            The filtered placefield data (if nan_safe=False) or original data (if nan_safe=True and no NaNs).
        activity_data_filtered : torch.Tensor
            The filtered activity data (if nan_safe=False) or original data (if nan_safe=True and no NaNs).
        """
        # Check for NaN values in placefield data (any NaN in any neuron for a given sample)
        idx_nan_samples_placefields = torch.any(torch.isnan(placefield_data), dim=0)
        idx_nan_samples_activity = torch.any(torch.isnan(activity_data), dim=0)

        if nan_safe:
            if torch.any(idx_nan_samples_placefields):
                num_nan = torch.sum(idx_nan_samples_placefields).item()
                total = len(idx_nan_samples_placefields)
                raise ValueError(f"{num_nan} / {total} samples have NaN values in placefield data!")
            if torch.any(idx_nan_samples_activity):
                num_nan = torch.sum(idx_nan_samples_activity).item()
                total = len(idx_nan_samples_activity)
                raise ValueError(f"{num_nan} / {total} samples have NaN values in activity data!")
            return placefield_data, activity_data
        else:
            # Filter out NaN samples
            return placefield_data[:, ~idx_nan_samples_placefields], activity_data[:, ~idx_nan_samples_activity]

    @abstractmethod
    def score(
        self,
        session: B2Session,
        subspace: "Subspace",
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "not_train",
    ):
        """Score the subspace model on the given session.

        Parameters
        ----------
        session : B2Session
            The session to score the model on.
        subspace : Subspace
            The fitted subspace to score.
        spks_type : SpksTypes
            The type of spike data to use. Default is "oasis".
        split : "SplitName"
            The split to use for scoring. Default is "not_train".
        """
        pass

    def evaluate(self, variance: dict[str, torch.Tensor]) -> float:
        """Evaluate how well the position model explains the full variance.

        Variance is a dictionary with the keys "variance_activity" and "variance_placefields".
        We expect that variance_activity is typically larger than variance_placefields. To calculate
        the performance, we measure the cumulative difference between them, and try to minimize.
        For interpretability, we return the ratio w/r/t the total variance.

        Parameters
        ----------
        variance : dict[str, torch.Tensor]
            A dictionary with the keys "variance_activity" and "variance_placefields".
            This is the output of the score() method.

        Returns
        -------
        float
            The evaluation score.
        """
        variance_activity = variance["variance_activity"]
        variance_placefields = variance["variance_placefields"]
        if not self.match_dimensions:
            variance_activity = variance_activity[: len(variance_placefields)]
        return torch.sum(variance_activity - variance_placefields) / torch.sum(variance_activity)

    @abstractmethod
    def _get_model_name(self) -> str:
        """Get the name of the model.

        Returns
        -------
        str
            The name of the model (e.g., "pca_subspace", "svca_subspace").
        """
        pass

    def get_score(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        test_split: Optional["SplitName"] = "test",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        force_remake: bool = False,
    ) -> dict[str, float]:
        """Get the score for the model with explicitly provided hyperparameters.

        Parameters
        ----------
        session : B2Session
            The session to get the score for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses spks_type from session. Default is None.
        train_split : Optional["SplitName"]
            The split to use for training. Default is "train".
        test_split : Optional["SplitName"]
            The split to use for testing. Default is "test".
        hyperparameters : PlaceFieldHyperparameters
            The hyperparameters to use for fitting and scoring.
        force_remake : bool
            If True, re-computes the score even if a cached result exists. Default is False.

        Returns
        -------
        metrics : dict[str, float]
            A dictionary with the metrics as keys and the values as the metrics.
        """
        if spks_type is None:
            spks_type = session.params.spks_type
        cache_key = self._get_score_from_hyps_cache_key(session, spks_type, train_split, test_split, hyperparameters)
        return self._compute_and_cache_score(session, spks_type, train_split, test_split, hyperparameters, cache_key, force_remake)

    def check_existing_score_from_hyps(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        test_split: Optional["SplitName"] = "test",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
    ) -> bool:
        """Check if the score for the model exists in the score cache.

        Parameters
        ----------
        session : B2Session
            The session to check the score for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses spks_type from session. Default is None.
        train_split : Optional["SplitName"]
            The split to use for training. Default is "train".
        test_split : Optional["SplitName"]
            The split to use for testing. Default is "test".
        hyperparameters : PlaceFieldHyperparameters
            The hyperparameters to check for.

        Returns
        -------
        bool
            True if the score exists in the cache, False otherwise.
        """
        if spks_type is None:
            spks_type = session.params.spks_type
        cache_key = self._get_score_from_hyps_cache_key(session, spks_type, train_split, test_split, hyperparameters)
        cache_path = self.registry.registry_paths.subspace_score_path / f"{cache_key}.joblib"
        return cache_path.exists()

    def _get_score_from_hyps_cache_key(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: "SplitName",
        test_split: "SplitName",
        hyperparameters: PlaceFieldHyperparameters,
    ) -> str:
        """Get the cache key for the score based on explicit hyperparameters.

        The cache key is a string used to identify the score of the model in the score cache files
        when hyperparameters are explicitly provided. It is a combination of:
        - model_name
        - session_name (mouse/date/session_id)
        - spks_type
        - hash(registry_params) (determines the population object and cell/time splits)
        - train_split (which train set to use for training)
        - test_split (which test set to use for testing)
        - hash(hyperparameters) (the actual hyperparameter values)
        - hash(model_params) (centered, max_components)

        Parameters
        ----------
        session : B2Session
            The session to get the cache key for.
        spks_type : SpksTypes
            The type of spike data to use.
        train_split : "SplitName"
            The split to use for training.
        test_split : "SplitName"
            The split to use for testing.
        hyperparameters : PlaceFieldHyperparameters
            The hyperparameters to use for scoring.

        Returns
        -------
        str
            The cache key string for the score of the model.
        """
        model_name = self._get_model_name()
        session_name = ".".join(session.session_name)
        registry_params_hash = stable_hash(self.registry.registry_params)
        hyperparameters_hash = stable_hash(vars(hyperparameters))
        model_params_hash = stable_hash(
            (
                self.centered,
                self.max_components,
                self.match_dimensions,
                self.activity_parameters.scale,
                self.activity_parameters.scale_type,
                self.activity_parameters.presplit,
            )
        )
        cache_params = [
            model_name,
            session_name,
            spks_type,
            registry_params_hash,
            train_split,
            test_split,
            hyperparameters_hash,
            model_params_hash,
        ]
        return "_".join(cache_params)

    def _compute_and_cache_score(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: "SplitName",
        test_split: "SplitName",
        hyperparameters: PlaceFieldHyperparameters,
        cache_key: str,
        force_remake: bool = False,
    ) -> dict[str, float]:
        """Compute and cache the score for the model with given hyperparameters.

        Parameters
        ----------
        session : B2Session
            The session to score the model on.
        spks_type : SpksTypes
            The type of spike data to use.
        train_split : "SplitName"
            The split to use for training.
        test_split : "SplitName"
            The split to use for testing.
        hyperparameters : PlaceFieldHyperparameters
            The hyperparameters to use for scoring.
        cache_key : str
            The cache key to use for storing/retrieving the score.
        force_remake : bool
            If True, will re-measure the score and save the results even if it already exists.
            Default is False.

        Returns
        -------
        metrics : dict[str, float]
            A dictionary with the metrics as keys and the values as the metrics.
        """
        cache_path = self.registry.registry_paths.subspace_score_path / f"{cache_key}.joblib"

        if cache_path.exists() and not force_remake:
            metrics = load(cache_path)
        else:
            subspace = self.fit(session, spks_type=spks_type, split=train_split, hyperparameters=hyperparameters)
            variance = self.score(session, subspace, spks_type=spks_type, split=test_split)
            evaluation_score = self.evaluate(variance)
            metrics = {"evaluation_score": float(evaluation_score)}
            for key, val in variance.items():
                metrics[key] = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
            if self.autosave:
                dump(metrics, cache_path)

        return metrics

    def clear_cached_score_from_hyps(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        test_split: Optional["SplitName"] = "test",
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
    ) -> None:
        """Clear the cached score for the model scored with explicit hyperparameters.

        Parameters
        ----------
        session : B2Session
            The session to clear the cached score for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses spks_type from session.
        train_split : Optional["SplitName"]
            The split to use for training. Default is "train".
        test_split : Optional["SplitName"]
            The split to use for testing. Default is "test".
        hyperparameters : PlaceFieldHyperparameters
            The hyperparameters to use.
        """
        if spks_type is None:
            spks_type = session.params.spks_type
        cache_key = self._get_score_from_hyps_cache_key(session, spks_type, train_split, test_split, hyperparameters)
        cache_path = self.registry.registry_paths.subspace_score_path / f"{cache_key}.joblib"
        if cache_path.exists():
            cache_path.unlink()
