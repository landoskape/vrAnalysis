from typing import Literal, Optional, TYPE_CHECKING, Any, Union, overload, TypeVar, Generic
from abc import ABC, abstractmethod
from itertools import product
from joblib import dump, load
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from vrAnalysis.helpers import stable_hash
from vrAnalysis.sessions import B2Session, SpksTypes
from dimilibi import measure_r2

if TYPE_CHECKING:
    from .registry import PopulationRegistry
    from optuna import Trial

SplitName = Literal["train", "train0", "train1", "validation", "test", "full"]

H = TypeVar("H", bound="HyperparametersBase")


class RegressionModel(ABC, Generic[H]):
    hyperparameters: H

    def __init__(
        self,
        registry: "PopulationRegistry",
        center: bool = False,
        scale: bool = True,
        scale_type: Optional[str] = "preserve",
        presplit: bool = True,
        autosave: bool = True,
    ):
        self.registry = registry
        self.center = center
        self.scale = scale
        self.scale_type = scale_type
        self.presplit = presplit
        self.autosave = autosave

    @abstractmethod
    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional[SplitName] = "train",
        hyperparameters: Optional[H] = None,
    ) -> Any:
        """Train the model on the given session.

        Parameters
        ----------
        session : B2Session
            The session to train on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the session's default.
        split : Optional[SplitName]
            The data split to use for training. Default is "train".
        hyperparameters : Optional[Hyperparameters]
            Model-specific hyperparameters. If None, uses the model's default.

        Returns
        -------
        Any
            Model-specific trained coefficients/model object. The exact type depends on the model subclass.
        """
        ...

    @abstractmethod
    def predict(
        self,
        session: B2Session,
        coefficients: Any,
        spks_type: Optional[SpksTypes] = None,
        split: Optional[SplitName] = "test",
        hyperparameters: Optional[H] = None,
    ) -> tuple[np.ndarray, dict]:
        """Predict target activity for a session.

        Parameters
        ----------
        session : B2Session
            The session to predict for.
        coefficients : Any
            Model-specific trained coefficients (output of train()).
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the session's default.
        split : Optional[SplitName]
            The data split to use for prediction. Default is "test".
        hyperparameters : Optional[Hyperparameters]
            Model-specific hyperparameters. If None, uses the model's default.

        Returns
        -------
        tuple[np.ndarray, dict]
            Prediction array and extras dictionary.
        """
        ...

    @property
    @abstractmethod
    def _model_hyperparameters(self) -> type[H]: ...
    @abstractmethod
    def _get_model_name(self) -> str: ...

    @overload
    def score(
        self,
        session: B2Session,
        reduce: Literal["mean"],
        spks_type: Optional[SpksTypes] = None,
        split: Optional[SplitName] = "test",
        hyperparameters: Optional[H] = None,
    ) -> float: ...
    @overload
    def score(
        self,
        session: B2Session,
        reduce: Literal["none"],
        spks_type: Optional[SpksTypes] = None,
        split: Optional[SplitName] = "test",
        hyperparameters: Optional[H] = None,
    ) -> np.ndarray: ...
    def score(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        reduce: Literal["mean", "none"] = "mean",
        train_split: Optional[SplitName] = "train",
        test_split: Optional[SplitName] = "test",
        hyperparameters: Optional[H] = None,
    ) -> Union[float, np.ndarray]:
        """Score the model on a session.

        Parameters
        ----------
        session : B2Session
            The session to score the model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session
            provided as input.
        reduce : Literal["mean", "none"]
            The reduction to apply to the r-squared value. If "mean", the mean of the r-squared value
            is returned. If "none", the r-squared value is returned without reduction. Default is mean.
        train_split: Optional[SplitName]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        test_split: Optional[SplitName]
            The split to use for the scoring. If None, uses the split from the session
            provided as input. Default is "test".
        hyperparameters: Optional[Hyperparameters]
            The hyperparameters to use for the model. If None, uses the default hyperparameters for the model. Must be the appropriate subclass for the model.

        Returns
        -------
        score : Union[float, np.ndarray]
            The score of the model on the session. If reduce is "mean", returns a float. If reduce is "none", returns a numpy array.
        """
        hyperparameters = hyperparameters or self.hyperparameters
        population = self.registry.get_population(session, spks_type)[0]
        target_data = population.get_split_data(
            self.registry.time_split[test_split],
            center=self.center,
            scale=self.scale,
            scale_type=self.scale_type,
            pre_split=self.presplit,
        )[1]
        trained_model = self.train(
            session,
            spks_type,
            split=train_split,
            hyperparameters=hyperparameters,
        )
        predicted_data, extras = self.predict(
            session,
            trained_model,
            spks_type,
            split=test_split,
            hyperparameters=hyperparameters,
        )
        idx_nan_samples = np.any(np.isnan(predicted_data), axis=0)
        if np.any(idx_nan_samples):
            raise ValueError(f"{np.sum(idx_nan_samples)} / {len(idx_nan_samples)} samples have nan predictions in {session.session_print()}!!!")
        r2 = measure_r2(predicted_data.T, target_data.T, reduce=reduce)
        if torch.any(torch.isnan(r2)):
            raise ValueError(f"NaN r-squared values in {session.session_print()}!!!")
        if reduce == "none":
            return np.array(r2)
        return r2

    def optimize(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        method: Literal["grid", "optuna"] = "grid",
    ) -> tuple[dict, float, pd.DataFrame]:
        """Optimize the hyperparameters of the model.

        Parameters
        ----------
        session : B2Session
            The session to optimize the hyperparameters for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : Optional[SplitName]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        validation_split : Optional[SplitName]
            The split to use for the validation. If None, uses the split from the session provided as input. Default is "validation".
        method : Literal["grid", "optuna"]
            The method to use for hyperparameter optimization. If "grid", uses grid search. If "optuna", uses Optuna. Default is "grid".

        Returns
        -------
        best_params : Hyperparameters
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            A DataFrame with all the results from the optimization.
        """
        # Use fixed hyperparameters for now which define the full search space
        hyperparameters = self._model_hyperparameters()

        if method == "grid":
            return self._optimize_grid(session, spks_type, train_split, validation_split, hyperparameters)
        elif method == "optuna":
            return self._optimize_optuna(session, spks_type, train_split, validation_split, hyperparameters)
        else:
            raise ValueError(f"Invalid method: {method}. Must be one of ['grid', 'optuna'].")

    def _optimize_grid(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: SplitName,
        validation_split: SplitName,
        hyperparameters: H,
    ) -> tuple[dict, float, pd.DataFrame]:
        """Optimize the hyperparameters of the model using grid search.

        Parameters
        ----------
        session : B2Session
            The session to optimize the hyperparameters for.
        spks_type : SpksTypes
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : SplitName
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        validation_split : SplitName
            The split to use for the validation. If None, uses the split from the session provided as input. Default is "validation".
        hyperparameters : Hyperparameters
            The hyperparameters to use for the model. If None, uses the default hyperparameters for the model.

        Returns
        -------
        best_params : dict
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            A DataFrame with all the results from the grid search.
        """
        hyperparameter_grid = hyperparameters.generate_grid()
        best_params = None
        best_score = -float("inf")
        results = []
        for params in tqdm(hyperparameter_grid, desc="Grid search", leave=False):
            score = self.score(
                session,
                spks_type,
                train_split=train_split,
                test_split=validation_split,
                hyperparameters=self._model_hyperparameters.from_dict(params),
            )
            result = params.copy()
            result["score"] = score
            results.append(result)

            if score > best_score:
                best_score = score
                best_params = params

        # Create DataFrame with all results
        results_df = pd.DataFrame(results)

        return best_params, best_score, results_df

    def _optimize_optuna(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: SplitName,
        validation_split: SplitName,
        hyperparameters: H,
    ) -> dict:
        raise NotImplementedError("Optuna optimization is not implemented yet.")

    @overload
    def get_best_hyperparameters(
        self,
        session: B2Session,
        full_results: Literal[False],
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        method: Literal["grid", "optuna"] = "grid",
        force_remake: bool = False,
    ) -> H: ...

    @overload
    def get_best_hyperparameters(
        self,
        session: B2Session,
        full_results: Literal[True],
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        method: Literal["grid", "optuna"] = "grid",
        force_remake: bool = False,
    ) -> tuple[H, float, pd.DataFrame]: ...

    def get_best_hyperparameters(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        method: Literal["grid", "optuna"] = "grid",
        full_results: bool = False,
        force_remake: bool = False,
    ) -> H:
        """Get the best hyperparameters for the model.

        Parameters
        ----------
        session : B2Session
            The session to get the best hyperparameters for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : Optional[SplitName]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        validation_split : Optional[SplitName]
            The split to use for the validation. If None, uses the split from the session provided as input. Default is "validation".
        method : Literal["grid", "optuna"]
            The method to use for hyperparameter optimization. If "grid", uses grid search. If "optuna", uses Optuna. Default is "grid".
        full_results: bool = False
            Whether to return the full optimization results. If True, returns a dictionary with the best hyperparameters, best score,
            and results DataFrame. If False, returns only the best hyperparameters. Default is False.
        force_remake: bool = False
            If True, will re-run optimization and save the results even if it already exists.
            (default is False)

        Returns
        -------
        best_params : Hyperparameters
            The best hyperparameters for the model.
        best_score : float
            The best score for the model. (Only returned if full_results is True.)
        results_df : pd.DataFrame
            The full optimization results. (Only returned if full_results is True.)
        """
        # First get the cache key to identify a potential cached result
        cache_key = self._get_hyperparameter_cache_key(session, spks_type, train_split, validation_split, method)
        cache_path = self.registry.registry_paths.hyperparameter_path / f"{cache_key}.joblib"

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

        hyperparameters = self._model_hyperparameters.from_dict(best_params)
        if full_results:
            return hyperparameters, best_score, results_df
        else:
            return hyperparameters

    def get_best_score(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        reduce: Literal["mean", "none"] = "mean",
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        test_split: Optional[SplitName] = "test",
        method: Literal["grid", "optuna"] = "grid",
        force_remake: bool = False,
        force_reoptimize: bool = False,
    ) -> float:
        """Get the best score for the model.

        Parameters
        ----------
        session : B2Session
            The session to get the best score for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        reduce : Literal["mean", "none"]
            The reduction to apply to the r-squared value. If "mean", the mean of the r-squared value
            is returned. If "none", the r-squared value is returned without reduction. Default is "mean".
        train_split : Optional[SplitName]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        validation_split : Optional[SplitName]
            The split to use for the validation. If None, uses the split from the session provided as input. Default is "validation".
        test_split : Optional[SplitName]
            The split to use for the testing. If None, uses the split from the session provided as input. Default is "test".
        method : Literal["grid", "optuna"]
            The method to use for hyperparameter optimization. If "grid", uses grid search. If "optuna", uses Optuna. Default is "grid".
        force_remake: bool = False
            If True, will re-measure the score and save the results even if it already exists.
            (default is False)
        force_reoptimize: bool = False
            If True, will re-optimize the hyperparameters and score the model even if it already exists.
            (default is False)

        Returns
        -------
        best_score : float
            The best score for the model.
        """
        # First get the cache key to identify a potential cached result
        cache_key = self._get_score_cache_key(session, spks_type, train_split, validation_split, test_split, method)
        cache_path = self.registry.registry_paths.score_path / f"{cache_key}.joblib"

        if cache_path.exists() and not force_remake and not force_reoptimize:
            # If a cache exists, load it and return the best score
            scores = load(cache_path)

        else:
            # If no cache exists, get the best hyperparameters and score the model and save the results to the cache
            hyperparameters = self.get_best_hyperparameters(session, spks_type, train_split, validation_split, method, force_remake=force_reoptimize)
            # We always use reduce="none" because we want to save the full score array
            scores = self.score(session, spks_type, reduce="none", train_split=train_split, test_split=test_split, hyperparameters=hyperparameters)
            if self.autosave:
                dump(scores, cache_path)

        if reduce == "mean":
            return np.mean(scores)
        elif reduce == "none":
            return scores
        else:
            raise ValueError(f"Invalid reduction: {reduce}. Must be one of ['mean', 'none'].")

    def check_existing_hyperparameters(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        method: Literal["grid", "optuna"] = "grid",
    ) -> bool:
        """Check if the hyperparameters for the model exist in the hyperparameter cache."""
        cache_key = self._get_hyperparameter_cache_key(session, spks_type, train_split, validation_split, method)
        cache_path = self.registry.registry_paths.hyperparameter_path / cache_key
        return cache_path.exists()

    def check_existing_score(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        test_split: Optional[SplitName] = "test",
        method: Literal["grid", "optuna"] = "grid",
    ) -> bool:
        """Check if the score for the model exist in the score cache."""
        cache_key = self._get_score_cache_key(session, spks_type, train_split, validation_split, test_split, method)
        cache_path = self.registry.registry_paths.score_path / cache_key
        return cache_path.exists()

    def _get_hyperparameter_cache_key(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        method: Literal["grid", "optuna"] = "grid",
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
        - hash(scaling_params) (determines the scaling parameters)

        Parameters
        ----------
        session : B2Session
            The session to get the cache key for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        train_split : Optional[SplitName]
            The split to use for training. Default is "train".
        validation_split : Optional[SplitName]
            The split to use for validation. Default is "validation".
        method : Literal["grid", "optuna"]
            The optimization method. Default is "grid".

        Returns
        -------
        str
            The cache key string for the model.
        """
        # Get model name from internal and gain attributes
        model_name = self._get_model_name()

        # Get session name (following pattern from registry._get_unique_id)
        session_name = ".".join(session.session_name)

        # Get spks_type (use from session if not provided)
        if spks_type is None:
            spks_type = session.params.spks_type

        # Get registry params hash (following pattern from registry._get_unique_id)
        registry_params_hash = stable_hash(self.registry.registry_params)

        # Scaling hash
        scaling_hash = stable_hash((self.center, self.scale, self.scale_type, self.presplit))

        # Combine all components into cache key
        cache_params = [
            model_name,
            session_name,
            spks_type,
            registry_params_hash,
            train_split,
            validation_split,
            method,
            scaling_hash,
        ]
        return "_".join(cache_params)

    def _get_score_cache_key(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        test_split: Optional[SplitName] = "test",
        method: Literal["grid", "optuna"] = "grid",
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
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        train_split : Optional[SplitName]
            The split to use for training. Default is "train".
        validation_split : Optional[SplitName]
            The split to use for validation. Default is "validation".
        test_split : Optional[SplitName]
            The split to use for testing. Default is "test".
        method : Literal["grid", "optuna"]
            The method used for optimization. Default is "grid".

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

    def clear_cached_hyperparameter(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        method: Literal["grid", "optuna"] = "grid",
    ) -> None:
        """Clear the cached hyperparameter for the model.

        Parameters
        ----------
        session : B2Session
            The session to clear the cached hyperparameter for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        train_split : Optional[SplitName]
            The split to use for training. Default is "train".
        validation_split : Optional[SplitName]
            The split to use for validation. Default is "validation".
        method : Literal["grid", "optuna"]
            The method used for optimization. Default is "grid".
        """
        cache_key = self._get_hyperparameter_cache_key(session, spks_type, train_split, validation_split, method)
        cache_path = self.registry.registry_paths.hyperparameter_path / cache_key
        if cache_path.exists():
            cache_path.unlink()

    def clear_cached_score(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional[SplitName] = "train",
        validation_split: Optional[SplitName] = "validation",
        test_split: Optional[SplitName] = "test",
        method: Literal["grid", "optuna"] = "grid",
    ) -> None:
        """Clear the cached score for the model.

        Parameters
        ----------
        session : B2Session
            The session to clear the cached score for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        train_split : Optional[SplitName]
            The split to use for training. Default is "train".
        validation_split : Optional[SplitName]
            The split to use for validation. Default is "validation".
        test_split : Optional[SplitName]
            The split to use for testing. Default is "test".
        method : Literal["grid", "optuna"]
            The method used for optimization. Default is "grid".
        """
        cache_key = self._get_score_cache_key(session, spks_type, train_split, validation_split, test_split, method)
        cache_path = self.registry.registry_paths.score_path / cache_key
        if cache_path.exists():
            cache_path.unlink()


class HyperparametersBase(ABC):
    """Base class for hyperparameters.

    This class structures the hyperparameters for each model. The model will have to provide a search space
    for the hyperparameters, with the keys being the hyperparameters and the values being the search space
    for the hyperparameter. Supports both grid search (for discrete spaces) and Optuna (for continuous
    or high-dimensional spaces).

    """

    @classmethod
    @abstractmethod
    def get_search_space(cls) -> dict[str, tuple[Any, ...]]:
        """Get the search space for the hyperparameters.

        This is different for each model, it needs to be implemented in the relevant subclass.
        The structure should be a dictionary with the hyperparameters as keys and the values as
        the search space for the hyperparameter.

        Example:
        {
            "prm1": [value1, value2, value3],
            "prm2": [value4, value5, value6],
            ...
        }

        Returns
        -------
        search_space : dict
            The search space for the hyperparameters.
        """
        raise NotImplementedError("This method must be implemented in the relevant subclass.")

    @classmethod
    @abstractmethod
    def get_optuna_space(cls, trial: "Trial") -> dict[str, Any]:
        """Get hyperparameters from Optuna trial.

        This is different for each model, it needs to be implemented in the relevant subclass.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object to suggest hyperparameters from.

        Returns
        -------
        params : dict[str, Any]
            Dictionary of hyperparameter values (not distributions) suggested by Optuna.
            These are the actual values to use for training/evaluation.

        Raises
        ------
        ImportError
            If Optuna is not installed.
        """
        raise NotImplementedError("This method must be implemented in the relevant subclass.")

    @classmethod
    def generate_grid(cls) -> list[dict]:
        """Generate all combinations for grid search.

        Returns
        -------
        grid : list[dict]
            List of all hyperparameter combinations for grid search.
        """
        search_space = cls.get_search_space()
        keys = search_space.keys()
        values = search_space.values()
        return [dict(zip(keys, combo)) for combo in product(*values)]

    @classmethod
    def from_dict(cls, params: dict):
        """Create instance from dictionary.

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameter values.

        Returns
        -------
        instance : HyperparametersBase
            Instance of the hyperparameters class with values from the dictionary.
        """
        return cls(**params)
