from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Literal, NamedTuple, Union, Any
from joblib import dump, load
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from tqdm.auto import tqdm as tqdm_async
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.helpers import stable_hash
from ..regression_models.hyperparameters import PlaceFieldHyperparameters

if TYPE_CHECKING:
    from ..registry import PopulationRegistry, SplitName
    from optuna import Trial
    from dimilibi import PCA, SVCA
    from vrAnalysis.processors.placefields import FrameBehavior

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
        correlation: bool = False,
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        max_components: int = 300,
        match_dimensions: bool = True,
        autosave: bool = True,
    ):
        """Initialize the subspace model.

        Parameters
        ----------
        registry : PopulationRegistry
            The registry to use for getting population data.
        centered : bool
            Whether to center the data. Default is False.
        correlation : bool
            Whether to use correlation instead of covariance. Default is False.
        hyperparameters : PlaceFieldHyperparameters
            The hyperparameters for placefield calculation. Default is PlaceFieldHyperparameters().
        max_components : int
            Maximum number of components to use. Default is 300.
        match_dimensions : bool
            Whether to match the dimensions of the activity and placefields. Default is True.
        autosave : bool
            Whether to automatically save optimization results to cache. Default is True.
        """
        self.registry = registry
        self.centered = centered
        self.correlation = correlation
        self.hyperparameters = hyperparameters
        self.max_components = max_components
        self.match_dimensions = match_dimensions
        self.autosave = autosave

        # How to scale the population data for the subspace analysis
        if correlation:
            self.population_parameters = {
                "scale": True,
                "scale_type": "max",
                "pre_split": True,
            }
        else:
            self.population_parameters = {
                "scale": False,
                "scale_type": None,
                "pre_split": False,
            }

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

        if use_cell_split:
            # For SVCASubspace - returns source and target separately
            source_data, target_data = population.get_split_data(split_idx, **self.population_parameters)
            frame_behavior_filtered = frame_behavior.filter(population.get_split_times(split_idx, within_idx_samples=False))
            num_source_neurons = len(population.cell_split_indices[0])
            num_target_neurons = len(population.cell_split_indices[1])
            return (source_data, target_data), frame_behavior_filtered, (num_source_neurons, num_target_neurons)
        else:
            # For PCASubspace and CVPCASubspace - returns all neurons together
            num_neurons = len(population.idx_neurons)
            data = population.apply_split(population.data[population.idx_neurons], split_idx, prefiltered=False, **self.population_parameters)
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

    def _center_data(self, data: torch.Tensor, centered: bool) -> torch.Tensor:
        """Conditionally center data if centered is True.

        Parameters
        ----------
        data : torch.Tensor
            The data to potentially center.
        centered : bool
            Whether to center the data.

        Returns
        -------
        torch.Tensor
            The (possibly centered) data.
        """
        if centered:
            return data - data.mean(dim=1, keepdim=True)
        return data

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
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
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
        hyperparameters : Optional[PlaceFieldHyperparameters]
            The hyperparameters to use. If None, uses self.hyperparameters.
        nan_safe : bool
            If True, will check for NaN values in placefield data and raise an error if found.
            If False, will filter out NaN samples from placefield data and corresponding activity data.
            Default is False.
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

    def reconstruction_score(
        self,
        session: B2Session,
        subspace: "Subspace",
        spks_type: SpksTypes = "oasis",
        split: "SplitName" = "not_train",
    ):
        """Compute reconstruction score for the subspace model.

        Parameters
        ----------
        session : B2Session
            The session to compute the reconstruction score on.
        subspace : Subspace
            The fitted subspace to score.
        spks_type : SpksTypes
            The type of spike data to use. Default is "oasis".
        split : "SplitName"
            The split to use for scoring. Default is "not_train".
        """
        raise NotImplementedError("Reconstruction score is not implemented for this subspace model. The ABC should be @abstractmethod when ready.")

    @abstractmethod
    def _get_model_name(self) -> str:
        """Get the name of the model.

        Returns
        -------
        str
            The name of the model (e.g., "pca_subspace", "cvpca_subspace", "svca_subspace").
        """
        pass

    def optimize(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
        method: Literal["grid", "optuna"] = "grid",
    ) -> tuple[dict, float, pd.DataFrame]:
        """Optimize the hyperparameters of the model.

        Parameters
        ----------
        session : B2Session
            The session to optimize the hyperparameters for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        validation_split : Optional["SplitName"]
            The split to use for the validation. If None, uses the split from the session provided as input. Default is "validation".
        method : Literal["grid", "optuna"]
            The method to use for hyperparameter optimization. If "grid", uses grid search. If "optuna", uses Optuna. Default is "grid".

        Returns
        -------
        best_params : dict
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            A DataFrame with all the results from the optimization.
        """
        if spks_type is None:
            spks_type = session.params.spks_type

        if method == "grid":
            return self._optimize_grid(session, spks_type, train_split, validation_split)
        elif method == "optuna":
            return self._optimize_optuna(session, spks_type, train_split, validation_split)
        else:
            raise ValueError(f"Invalid method: {method}. Must be one of ['grid', 'optuna'].")

    def _optimize_grid(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: "SplitName",
        validation_split: "SplitName",
    ) -> tuple[dict, float, pd.DataFrame]:
        """Optimize the hyperparameters of the model using grid search.

        Parameters
        ----------
        session : B2Session
            The session to optimize the hyperparameters for.
        spks_type : SpksTypes
            The type of spike data to use for the population.
        train_split : "SplitName"
            The split to use for the training.
        validation_split : "SplitName"
            The split to use for the validation.

        Returns
        -------
        best_params : dict
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            A DataFrame with all the results from the grid search.
        """
        HyperparameterClass = PlaceFieldHyperparameters
        search_space = HyperparameterClass.get_search_space()
        training_grid = HyperparameterClass.generate_grid(search_space)

        best_params = None
        best_score = float("inf")
        results = []

        # Make progress bar
        training_grid = tqdm(training_grid, desc="Grid search", leave=False)

        # Perform grid search
        for training_params in training_grid:
            hyperparameters = HyperparameterClass.from_dict(training_params)
            subspace = self.fit(session, spks_type=spks_type, split=train_split, hyperparameters=hyperparameters)
            variance = self.score(session, subspace, spks_type=spks_type, split=validation_split)
            score = self.evaluate(variance)
            final_params = vars(hyperparameters)
            result = final_params | dict(score=score)
            results.append(result)
            if score < best_score:
                best_score = score
                best_params = final_params

        # Create DataFrame with all results
        results_df = pd.DataFrame(results)
        return best_params, best_score, results_df

    def _optimize_optuna(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: "SplitName",
        validation_split: "SplitName",
        n_trials: int = 15,
        timeout: float | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
    ) -> tuple[dict, float, pd.DataFrame]:
        """Optimize the hyperparameters of the model using Optuna.

        Parameters
        ----------
        session : B2Session
            The session to optimize the hyperparameters for.
        spks_type : SpksTypes
            The type of spike data to use for the population.
        train_split : "SplitName"
            The split to use for the training.
        validation_split : "SplitName"
            The split to use for the validation.
        n_trials : int
            Number of Optuna trials.
        timeout : float | None
            Optional timeout in seconds (Optuna will stop once reached).
        sampler : optuna.samplers.BaseSampler | None
            Optional Optuna sampler; if None, defaults are used.

        Returns
        -------
        best_params : dict
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            A DataFrame with all the results from the Optuna optimization.
        """
        HyperparameterClass = PlaceFieldHyperparameters

        if sampler is None:
            sampler = TPESampler(seed=42)

        results: list[dict] = []

        def objective(trial: "Trial") -> float:
            training_params = HyperparameterClass.get_optuna_space(trial)
            hyperparameters = HyperparameterClass.from_dict(training_params)
            subspace = self.fit(session, spks_type=spks_type, split=train_split, hyperparameters=hyperparameters)
            variance = self.score(session, subspace, spks_type=spks_type, split=validation_split)
            score = self.evaluate(variance)

            if np.isnan(score):
                # Invalidate the trial if score is NaN
                score = float("inf")

            final_params = vars(hyperparameters)
            result = final_params | dict(score=score)
            results.append(result)

            # Attach params to the trial so we can recover them later
            trial.set_user_attr("best_params", final_params)
            trial.set_user_attr("score", float(score))

            return score

        study = optuna.create_study(
            study_name="subspace_model_optuna",
            direction="minimize",
            sampler=sampler,
        )

        with tqdm_async(total=n_trials, desc="Optuna search", leave=False) as pbar:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=[lambda study, trial: pbar.update(1)],
                gc_after_trial=True,
            )

        best_trial = study.best_trial
        best_params = best_trial.user_attrs["best_params"]
        best_score = best_trial.user_attrs["score"]
        results_df = pd.DataFrame(results)

        return best_params, best_score, results_df

    def get_best_hyperparameters(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
        method: Literal["grid", "optuna"] = "grid",
        force_remake: bool = False,
    ) -> tuple[PlaceFieldHyperparameters, float, pd.DataFrame]:
        """Get the best hyperparameters for the model.

        Parameters
        ----------
        session : B2Session
            The session to get the best hyperparameters for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        validation_split : Optional["SplitName"]
            The split to use for the validation. If None, uses the split from the session provided as input. Default is "validation".
        method : Literal["grid", "optuna"]
            The method to use for hyperparameter optimization. If "grid", uses grid search. If "optuna", uses Optuna. Default is "grid".
        force_remake: bool = False
            If True, will re-run optimization and save the results even if it already exists.
            (default is False)

        Returns
        -------
        best_params : PlaceFieldHyperparameters
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            The full optimization results.
        """
        if spks_type is None:
            spks_type = session.params.spks_type

        # First get the cache key to identify a potential cached result
        cache_key = self._get_hyperparameter_cache_key(session, spks_type, train_split, validation_split, method)
        cache_path = self.registry.registry_paths.subspace_hyperparameter_path / f"{cache_key}.joblib"

        if cache_path.exists() and not force_remake:
            # If a cache exists, load it and return the best hyperparameters
            full_optimization_results = load(cache_path)
            best_params = full_optimization_results["best_params"]
            best_score = full_optimization_results["best_score"]
            results_df = full_optimization_results["results_df"]
        else:
            # If no cache exists, optimize the hyperparameters and save the results to the cache
            best_params, best_score, results_df = self.optimize(session, spks_type, train_split, validation_split, method)
            full_optimization_results = dict(best_params=best_params, best_score=best_score, results_df=results_df)
            if self.autosave:
                dump(full_optimization_results, cache_path)

        hyperparameters = PlaceFieldHyperparameters.from_dict(best_params)
        return hyperparameters, best_score, results_df

    def get_best_score(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
        test_split: Optional["SplitName"] = "test",
        method: Literal["grid", "optuna", "best"] = "grid",
        force_remake: bool = False,
        force_reoptimize: bool = False,
    ) -> dict[str, float]:
        """Get the best score for the model.

        Parameters
        ----------
        session : B2Session
            The session to get the best score for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        validation_split : Optional["SplitName"]
            The split to use for the validation. If None, uses the split from the session provided as input. Default is "validation".
        test_split : Optional["SplitName"]
            The split to use for the testing. If None, uses the split from the session provided as input. Default is "test".
        method : Literal["grid", "optuna", "best"]
            The method to use for hyperparameter optimization. If "grid", uses grid search. If "optuna", uses Optuna. If "best", uses whichever method has the best score. Default is "grid".
        force_remake: bool = False
            If True, will re-measure the score and save the results even if it already exists.
            (default is False)
        force_reoptimize: bool = False
            If True, will re-optimize the hyperparameters and score the model even if it already exists.
            (default is False)

        Returns
        -------
        metrics : dict[str, float]
            A dictionary with the metrics as keys and the values as the metrics.
        """
        if spks_type is None:
            spks_type = session.params.spks_type

        # Handle "best" method by trying both grid and optuna
        if method == "best":
            grid_key = self._get_score_cache_key(session, spks_type, train_split, validation_split, test_split, "grid")
            optuna_key = self._get_score_cache_key(session, spks_type, train_split, validation_split, test_split, "optuna")
            grid_path = self.registry.registry_paths.subspace_score_path / f"{grid_key}.joblib"
            optuna_path = self.registry.registry_paths.subspace_score_path / f"{optuna_key}.joblib"

            # Check which exists and has better score
            if grid_path.exists() and optuna_path.exists() and not force_remake and not force_reoptimize:
                grid_metrics = load(grid_path)
                optuna_metrics = load(optuna_path)
                # Compare evaluation scores (lower is better)
                if grid_metrics.get("evaluation_score", float("inf")) <= optuna_metrics.get("evaluation_score", float("inf")):
                    return grid_metrics
                else:
                    return optuna_metrics
            elif grid_path.exists() and not force_remake and not force_reoptimize:
                return load(grid_path)
            elif optuna_path.exists() and not force_remake and not force_reoptimize:
                return load(optuna_path)
            else:
                # Try both and return the best
                grid_metrics = self.get_best_score(
                    session,
                    spks_type,
                    train_split,
                    validation_split,
                    test_split,
                    "grid",
                    force_remake,
                    force_reoptimize,
                )
                optuna_metrics = self.get_best_score(
                    session,
                    spks_type,
                    train_split,
                    validation_split,
                    test_split,
                    "optuna",
                    force_remake,
                    force_reoptimize,
                )
                if grid_metrics.get("evaluation_score", float("inf")) <= optuna_metrics.get("evaluation_score", float("inf")):
                    return grid_metrics
                else:
                    return optuna_metrics

        # Get the best hyperparameters from optimization
        hyperparameters = self.get_best_hyperparameters(
            session,
            spks_type,
            train_split,
            validation_split,
            method,
            force_remake=force_reoptimize,
        )[0]

        # Get cache key based on optimization method
        cache_key = self._get_score_cache_key(
            session,
            spks_type,
            train_split,
            validation_split,
            test_split,
            method,
        )

        # Use shared backend to compute and cache score
        # If force_reoptimize is True, we've already forced reoptimization above, so use force_remake for scoring
        return self._compute_and_cache_score(
            session,
            spks_type,
            train_split,
            test_split,
            hyperparameters,
            cache_key,
            force_remake,
        )

    def get_score(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        test_split: Optional["SplitName"] = "test",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
        source_model: Optional["SubspaceModel"] = None,
        source_method: Optional[Literal["grid", "optuna", "best"]] = None,
        source_validation_split: Optional["SplitName"] = None,
        force_remake: bool = False,
    ) -> dict[str, float]:
        """Get the score for the model with explicitly provided hyperparameters.

        This method allows scoring with hyperparameters that are either:
        1. Directly provided via the `hyperparameters` parameter, or
        2. Retrieved from another model's optimization via `source_model` and `source_method`.

        Parameters
        ----------
        session : B2Session
            The session to get the score for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        test_split : Optional["SplitName"]
            The split to use for the testing. If None, uses the split from the session provided as input. Default is "test".
        hyperparameters : Optional[PlaceFieldHyperparameters]
            The hyperparameters to use for scoring. If provided, these will be used directly.
        source_model : Optional[SubspaceModel]
            Another SubspaceModel instance to get hyperparameters from. Must be used with `source_method`.
            If provided, will retrieve the best hyperparameters from that model's optimization.
        source_method : Optional[Literal["grid", "optuna", "best"]]
            The optimization method to use when retrieving hyperparameters from `source_model`.
            Required if `source_model` is provided. If "best", will use whichever method (grid or optuna)
            has the better score from the source model.
        source_validation_split : Optional["SplitName"]
            The validation split that was used in the source model's optimization.
            Required if `source_model` is provided. Default is "validation".
        force_remake: bool = False
            If True, will re-measure the score and save the results even if it already exists.
            (default is False)

        Returns
        -------
        metrics : dict[str, float]
            A dictionary with the metrics as keys and the values as the metrics.

        Raises
        ------
        ValueError
            If both `hyperparameters` and `source_model` are provided, or if neither is provided,
            or if `source_model` is provided but `source_method` is not, or if the source model
            doesn't have cached hyperparameters.
        """
        if spks_type is None:
            spks_type = session.params.spks_type

        # Validate that exactly one of hyperparameters or source_model is provided
        if hyperparameters is not None and source_model is not None:
            raise ValueError("Cannot provide both 'hyperparameters' and 'source_model'. Please provide only one.")
        if hyperparameters is None and source_model is None:
            raise ValueError("Must provide either 'hyperparameters' or 'source_model' (with 'source_method').")

        # Get hyperparameters from source model if needed
        if source_model is not None:
            if source_method is None:
                raise ValueError("'source_method' is required when 'source_model' is provided.")
            if source_validation_split is None:
                source_validation_split = "validation"

            # Handle "best" method by trying both grid and optuna
            if source_method == "best":
                grid_exists = source_model.check_existing_hyperparameters(
                    session,
                    spks_type=spks_type,
                    train_split=train_split,
                    validation_split=source_validation_split,
                    method="grid",
                )
                optuna_exists = source_model.check_existing_hyperparameters(
                    session,
                    spks_type=spks_type,
                    train_split=train_split,
                    validation_split=source_validation_split,
                    method="optuna",
                )

                if not grid_exists and not optuna_exists:
                    raise ValueError(
                        f"Source model does not have cached hyperparameters for "
                        f"session={session.session_print()}, spks_type={spks_type}, "
                        f"train_split={train_split}, validation_split={source_validation_split} "
                        f"for either grid or optuna method. Please run optimization on the source model first."
                    )

                # Try to get scores for both methods and return the best one
                grid_score = None
                optuna_score = None

                if grid_exists:
                    grid_hyperparameters = source_model.get_best_hyperparameters(
                        session,
                        spks_type=spks_type,
                        train_split=train_split,
                        validation_split=source_validation_split,
                        method="grid",
                    )[0]
                    grid_cache_key = self._get_score_from_hyps_cache_key(
                        session,
                        spks_type,
                        train_split,
                        test_split,
                        grid_hyperparameters,
                    )
                    grid_cache_path = self.registry.registry_paths.subspace_score_path / f"{grid_cache_key}.joblib"
                    if grid_cache_path.exists() and not force_remake:
                        grid_score = load(grid_cache_path)
                    else:
                        # Compute score for grid hyperparameters
                        grid_score = self._compute_and_cache_score(
                            session,
                            spks_type,
                            train_split,
                            test_split,
                            grid_hyperparameters,
                            grid_cache_key,
                            force_remake,
                        )

                if optuna_exists:
                    optuna_hyperparameters = source_model.get_best_hyperparameters(
                        session,
                        spks_type=spks_type,
                        train_split=train_split,
                        validation_split=source_validation_split,
                        method="optuna",
                    )[0]
                    optuna_cache_key = self._get_score_from_hyps_cache_key(
                        session,
                        spks_type,
                        train_split,
                        test_split,
                        optuna_hyperparameters,
                    )
                    optuna_cache_path = self.registry.registry_paths.subspace_score_path / f"{optuna_cache_key}.joblib"
                    if optuna_cache_path.exists() and not force_remake:
                        optuna_score = load(optuna_cache_path)
                    else:
                        # Compute score for optuna hyperparameters
                        optuna_score = self._compute_and_cache_score(
                            session,
                            spks_type,
                            train_split,
                            test_split,
                            optuna_hyperparameters,
                            optuna_cache_key,
                            force_remake,
                        )

                # Return the best score (lower evaluation_score is better)
                if grid_score is not None and optuna_score is not None:
                    if grid_score.get("evaluation_score", float("inf")) <= optuna_score.get("evaluation_score", float("inf")):
                        return grid_score
                    else:
                        return optuna_score
                elif grid_score is not None:
                    return grid_score
                elif optuna_score is not None:
                    return optuna_score
                else:
                    # This shouldn't happen due to the check above, but just in case
                    raise ValueError("Could not retrieve scores from source model.")

            else:
                # Check if source model has cached hyperparameters
                if not source_model.check_existing_hyperparameters(
                    session,
                    spks_type=spks_type,
                    train_split=train_split,
                    validation_split=source_validation_split,
                    method=source_method,
                ):
                    raise ValueError(
                        f"Source model does not have cached hyperparameters for "
                        f"session={session.session_print()}, spks_type={spks_type}, "
                        f"train_split={train_split}, validation_split={source_validation_split}, method={source_method}. "
                        f"Please run optimization on the source model first."
                    )

                # Get hyperparameters from source model
                hyperparameters = source_model.get_best_hyperparameters(
                    session,
                    spks_type=spks_type,
                    train_split=train_split,
                    validation_split=source_validation_split,
                    method=source_method,
                )[0]

        # Get cache key based on hyperparameters
        cache_key = self._get_score_from_hyps_cache_key(
            session,
            spks_type,
            train_split,
            test_split,
            hyperparameters,
        )

        # Use shared backend to compute and cache score
        return self._compute_and_cache_score(
            session,
            spks_type,
            train_split,
            test_split,
            hyperparameters,
            cache_key,
            force_remake,
        )

    def check_existing_hyperparameters(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
        method: Literal["grid", "optuna"] = "grid",
    ) -> bool:
        """Check if the hyperparameters for the model exist in the hyperparameter cache."""
        if spks_type is None:
            spks_type = session.params.spks_type
        cache_key = self._get_hyperparameter_cache_key(session, spks_type, train_split, validation_split, method)
        cache_path = self.registry.registry_paths.subspace_hyperparameter_path / f"{cache_key}.joblib"
        return cache_path.exists()

    def check_existing_score(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
        test_split: Optional["SplitName"] = "test",
        method: Literal["grid", "optuna"] = "grid",
    ) -> bool:
        """Check if the score for the model exist in the score cache."""
        if spks_type is None:
            spks_type = session.params.spks_type
        cache_key = self._get_score_cache_key(session, spks_type, train_split, validation_split, test_split, method)
        cache_path = self.registry.registry_paths.subspace_score_path / f"{cache_key}.joblib"
        return cache_path.exists()

    def check_existing_score_from_hyps(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        test_split: Optional["SplitName"] = "test",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
        source_model: Optional["SubspaceModel"] = None,
        source_method: Optional[Literal["grid", "optuna", "best"]] = None,
        source_validation_split: Optional["SplitName"] = None,
    ) -> bool:
        """Check if the score for the model exists in the score cache.

        This method allows checking for cached scores with hyperparameters that are either:
        1. Directly provided via the `hyperparameters` parameter, or
        2. Retrieved from another model's optimization via `source_model` and `source_method`.

        Parameters
        ----------
        session : B2Session
            The session to check the score for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        test_split : Optional["SplitName"]
            The split to use for the testing. If None, uses the split from the session provided as input. Default is "test".
        hyperparameters : Optional[PlaceFieldHyperparameters]
            The hyperparameters to check for. If provided, these will be used directly.
        source_model : Optional[SubspaceModel]
            Another SubspaceModel instance to get hyperparameters from. Must be used with `source_method`.
            If provided, will retrieve the best hyperparameters from that model's optimization.
        source_method : Optional[Literal["grid", "optuna", "best"]]
            The optimization method to use when retrieving hyperparameters from `source_model`.
            Required if `source_model` is provided.
        source_validation_split : Optional["SplitName"]
            The validation split that was used in the source model's optimization.
            Required if `source_model` is provided. Default is "validation".

        Returns
        -------
        bool
            True if the score exists in the cache, False otherwise.

        Raises
        ------
        ValueError
            If both `hyperparameters` and `source_model` are provided, or if neither is provided,
            or if `source_model` is provided but `source_method` is not.
        """
        if spks_type is None:
            spks_type = session.params.spks_type

        # Validate that exactly one of hyperparameters or source_model is provided
        if hyperparameters is not None and source_model is not None:
            raise ValueError("Cannot provide both 'hyperparameters' and 'source_model'. Please provide only one.")
        if hyperparameters is None and source_model is None:
            raise ValueError("Must provide either 'hyperparameters' or 'source_model' (with 'source_method').")

        # Get hyperparameters from source model if needed
        if source_model is not None:
            if source_method is None:
                raise ValueError("'source_method' is required when 'source_model' is provided.")
            if source_validation_split is None:
                source_validation_split = "validation"

            # Handle "best" method by checking both grid and optuna
            if source_method == "best":
                # Check both grid and optuna methods
                grid_exists = source_model.check_existing_hyperparameters(
                    session,
                    spks_type=spks_type,
                    train_split=train_split,
                    validation_split=source_validation_split,
                    method="grid",
                )
                optuna_exists = source_model.check_existing_hyperparameters(
                    session,
                    spks_type=spks_type,
                    train_split=train_split,
                    validation_split=source_validation_split,
                    method="optuna",
                )

                # Check if score exists for either method
                if grid_exists:
                    grid_hyperparameters = source_model.get_best_hyperparameters(
                        session,
                        spks_type=spks_type,
                        train_split=train_split,
                        validation_split=source_validation_split,
                        method="grid",
                    )[0]
                    grid_cache_key = self._get_score_from_hyps_cache_key(
                        session,
                        spks_type,
                        train_split,
                        test_split,
                        grid_hyperparameters,
                    )
                    grid_cache_path = self.registry.registry_paths.subspace_score_path / f"{grid_cache_key}.joblib"
                    if grid_cache_path.exists():
                        return True

                if optuna_exists:
                    optuna_hyperparameters = source_model.get_best_hyperparameters(
                        session,
                        spks_type=spks_type,
                        train_split=train_split,
                        validation_split=source_validation_split,
                        method="optuna",
                    )[0]
                    optuna_cache_key = self._get_score_from_hyps_cache_key(
                        session,
                        spks_type,
                        train_split,
                        test_split,
                        optuna_hyperparameters,
                    )
                    optuna_cache_path = self.registry.registry_paths.subspace_score_path / f"{optuna_cache_key}.joblib"
                    if optuna_cache_path.exists():
                        return True

                # Neither exists
                return False
            else:
                # Check if source model has cached hyperparameters
                if not source_model.check_existing_hyperparameters(
                    session,
                    spks_type=spks_type,
                    train_split=train_split,
                    validation_split=source_validation_split,
                    method=source_method,
                ):
                    # If source model doesn't have cached hyperparameters, the score can't exist
                    return False

            # Get hyperparameters from source model
            hyperparameters = source_model.get_best_hyperparameters(
                session,
                spks_type=spks_type,
                train_split=train_split,
                validation_split=source_validation_split,
                method=source_method,
            )[0]

        # Get cache key based on hyperparameters
        cache_key = self._get_score_from_hyps_cache_key(
            session,
            spks_type,
            train_split,
            test_split,
            hyperparameters,
        )
        cache_path = self.registry.registry_paths.subspace_score_path / f"{cache_key}.joblib"
        return cache_path.exists()

    def _get_hyperparameter_cache_key(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: "SplitName",
        validation_split: "SplitName",
        method: Literal["grid", "optuna"],
    ) -> str:
        """Get the cache key for the hyperparameters of the model.

        The cache key is a string used to identify the model in the hyperparameter cache files.
        It is a combination of the following things:
        - model_name
        - session_name (mouse/date/session_id)
        - spks_type
        - hash(registry_params) (determines the population object and cell/time splits)
        - train_split (which train set to use for training)
        - validation_split (which validation set to use for validation)
        - optimization method (grid or optuna)
        - hash(model_params) (centered, max_components)

        Parameters
        ----------
        session : B2Session
            The session to get the cache key for.
        spks_type : SpksTypes
            The type of spike data to use.
        train_split : "SplitName"
            The split to use for training.
        validation_split : "SplitName"
            The split to use for validation.
        method : Literal["grid", "optuna"]
            The optimization method.

        Returns
        -------
        str
            The cache key string for the model.
        """
        # Get model name
        model_name = self._get_model_name()

        # Get session name (following pattern from registry._get_unique_id)
        session_name = ".".join(session.session_name)

        # Get registry params hash (following pattern from registry._get_unique_id)
        registry_params_hash = stable_hash(self.registry.registry_params)

        # Model params hash
        model_params_hash = stable_hash(
            (
                self.centered,
                self.max_components,
            )
        )

        # Combine all components into cache key
        cache_params = [
            model_name,
            session_name,
            spks_type,
            registry_params_hash,
            train_split,
            validation_split,
            method,
            model_params_hash,
        ]
        return "_".join(cache_params)

    def _get_score_cache_key(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: "SplitName",
        validation_split: "SplitName",
        test_split: "SplitName",
        method: Literal["grid", "optuna"],
    ) -> str:
        """Get the cache key for the score of the model.

        The cache key is a string used to identify the score of the model in the score cache files.
        It is a combination of the following things:
        - hyperparameter_cache_key (the key for the hyperparameters of the model)
        - test_split (which test set to use for testing)

        Parameters
        ----------
        session : B2Session
            The session to get the cache key for.
        spks_type : SpksTypes
            The type of spike data to use.
        train_split : "SplitName"
            The split to use for training.
        validation_split : "SplitName"
            The split to use for validation.
        test_split : "SplitName"
            The split to use for testing.
        method : Literal["grid", "optuna"]
            The method used for optimization.

        Returns
        -------
        str
            The cache key string for the score of the model.
        """
        hyperparameter_cache_key = self._get_hyperparameter_cache_key(session, spks_type, train_split, validation_split, method)
        cache_params = [
            hyperparameter_cache_key,
            test_split,
        ]
        return "_".join(cache_params)

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
        when hyperparameters are explicitly provided (not from optimization). It is a combination of:
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
        # Get model name
        model_name = self._get_model_name()

        # Get session name (following pattern from registry._get_unique_id)
        session_name = ".".join(session.session_name)

        # Get registry params hash (following pattern from registry._get_unique_id)
        registry_params_hash = stable_hash(self.registry.registry_params)

        # Hyperparameters hash
        hyperparameters_hash = stable_hash(vars(hyperparameters))

        # Model params hash
        model_params_hash = stable_hash(
            (
                self.centered,
                self.max_components,
            )
        )

        # Combine all components into cache key
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

        This is a shared backend method used by both get_best_score() and get_score().

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
            # If a cache exists, load it and return the score
            metrics = load(cache_path)
        else:
            # If no cache exists, fit the model and score it, then save the results to the cache
            subspace = self.fit(session, spks_type=spks_type, split=train_split, hyperparameters=hyperparameters)
            variance = self.score(session, subspace, spks_type=spks_type, split=test_split)
            evaluation_score = self.evaluate(variance)
            metrics = {
                "evaluation_score": float(evaluation_score),
                "variance_activity": (
                    variance["variance_activity"].cpu().numpy()
                    if isinstance(variance["variance_activity"], torch.Tensor)
                    else variance["variance_activity"]
                ),
                "variance_placefields": (
                    variance["variance_placefields"].cpu().numpy()
                    if isinstance(variance["variance_placefields"], torch.Tensor)
                    else variance["variance_placefields"]
                ),
            }
            if self.autosave:
                dump(metrics, cache_path)

        return metrics

    def clear_cached_hyperparameter(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
        method: Literal["grid", "optuna"] = "grid",
    ) -> None:
        """Clear the cached hyperparameter for the model.

        Parameters
        ----------
        session : B2Session
            The session to clear the cached hyperparameter for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        train_split : Optional["SplitName"]
            The split to use for training. Default is "train".
        validation_split : Optional["SplitName"]
            The split to use for validation. Default is "validation".
        method : Literal["grid", "optuna"]
            The method used for optimization. Default is "grid".
        """
        if spks_type is None:
            spks_type = session.params.spks_type
        cache_key = self._get_hyperparameter_cache_key(session, spks_type, train_split, validation_split, method)
        cache_path = self.registry.registry_paths.subspace_hyperparameter_path / f"{cache_key}.joblib"
        if cache_path.exists():
            cache_path.unlink()

    def clear_cached_score(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
        test_split: Optional["SplitName"] = "test",
        method: Literal["grid", "optuna"] = "grid",
    ) -> None:
        """Clear the cached score for the model.

        Parameters
        ----------
        session : B2Session
            The session to clear the cached score for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        train_split : Optional["SplitName"]
            The split to use for training. Default is "train".
        validation_split : Optional["SplitName"]
            The split to use for validation. Default is "validation".
        test_split : Optional["SplitName"]
            The split to use for testing. Default is "test".
        method : Literal["grid", "optuna"]
            The method used for optimization. Default is "grid".
        """
        if spks_type is None:
            spks_type = session.params.spks_type
        cache_key = self._get_score_cache_key(session, spks_type, train_split, validation_split, test_split, method)
        cache_path = self.registry.registry_paths.subspace_score_path / f"{cache_key}.joblib"
        if cache_path.exists():
            cache_path.unlink()
