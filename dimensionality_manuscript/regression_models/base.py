from typing import Literal, Optional, TYPE_CHECKING, Any, TypeVar, Generic, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from itertools import product
from joblib import dump, load
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from tqdm.auto import tqdm as tqdm_async
from vrAnalysis.helpers import stable_hash
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import FrameBehavior
from dimilibi import measure_r2, mse

if TYPE_CHECKING:
    from ..registry import PopulationRegistry, SplitName
    from optuna import Trial

H = TypeVar("H", bound="HyperparametersBase")

# We've got beautiful nested Taqqadums and Optuna's like meeeeeeeeeeeeeee... nope.
optuna.logging.set_verbosity(optuna.logging.WARNING)

MINIMUM_NON_NAN_FRACTION: float = 0.9


@dataclass(frozen=True)
class ActivityParameters:
    """Parameters for the activity data.

    Parameters
    ----------
    center : bool
        Whether to center the activity data.
    scale : bool
        Whether to scale the activity data.
    scale_type : str
        The type of scaling to apply to the activity data.
    presplit : bool
        Whether to compute statistical modes on full data before splitting by timepoint.
    """

    center: bool = False
    scale: bool = True
    scale_type: str = "max"
    presplit: bool = True


class RegressionModel(ABC, Generic[H]):
    hyperparameters: H

    def __init__(
        self,
        registry: "PopulationRegistry",
        activity_parameters: ActivityParameters = ActivityParameters(),
        autosave: bool = True,
    ):
        self.registry = registry
        self.activity_parameters = activity_parameters
        self.autosave = autosave

    @abstractmethod
    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "train",
        hyperparameters: Optional[H] = None,
    ) -> Any:
        """Train the model on the given session.

        Parameters
        ----------
        session : B2Session
            The session to train on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the session's default.
        split : Optional["SplitName"]
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
        split: Optional["SplitName"] = "test",
        hyperparameters: Optional[H] = None,
        nan_safe: bool = False,
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
        split : Optional["SplitName"]
            The data split to use for prediction. Default is "test".
        hyperparameters : Optional[Hyperparameters]
            Model-specific hyperparameters. If None, uses the model's default.
        nan_safe : bool
            If True, will check for NaN values in predictions and raise an error if found.
            If False, will filter out NaN samples from predictions.

        Returns
        -------
        tuple[np.ndarray, dict]
            Prediction array and extras dictionary.
        """
        ...

    @property
    @abstractmethod
    def _model_hyperparameters(self) -> Type[H]: ...
    @abstractmethod
    def _get_model_name(self) -> str: ...

    def get_session_data(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "test",
    ) -> tuple[torch.Tensor, torch.Tensor, FrameBehavior]:
        """Get the activity data and frame behavior for a session.

        Parameters
        ----------
        session : B2Session
            The session to get the activity data for.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the session's default.
        split : Optional["SplitName"]
            The split to use for the activity data. If None, uses the split from the session provided as input. Default is "test".

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, FrameBehavior]
            The source and target activity data and frame behavior for the session.
            All data is filtered to the requested split.
        """
        population, frame_behavior = self.registry.get_population(session, spks_type)
        source_data, target_data = population.get_split_data(
            self.registry.time_split[split],
            center=self.activity_parameters.center,
            scale=self.activity_parameters.scale,
            scale_type=self.activity_parameters.scale_type,
            pre_split=self.activity_parameters.presplit,
        )
        idx = np.array(population.get_split_times(self.registry.time_split[split], within_idx_samples=False))
        frame_behavior = frame_behavior.filter(idx)
        return source_data, target_data, frame_behavior

    def score(
        self,
        session: B2Session,
        trained_model: Any,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "test",
        hyperparameters: Optional[H] = None,
        nan_safe: bool = False,
    ) -> float:
        """Score the model on the given session.

        Parameters
        ----------
        session : B2Session
            The session to score the model on.
        trained_model : Any
            The trained model to score.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split : Optional["SplitName"]
            The split to use for the scoring. If None, uses the split from the session provided as input. Default is "test".
        hyperparameters : Optional[Hyperparameters]
            The hyperparameters to use for the model. If None, uses the default hyperparameters for the model. Must be the appropriate subclass for the model.
        nan_safe : bool = True
            If True, will check for NaN values in predictions and metrics and raise errors if found. If False, will remove NaN samples from the prediction and target data.

        Returns
        -------
        score : float
            The MSE of the model on the session at the requested split.
        """
        predicted_data, extras = self.predict(
            session,
            trained_model,
            spks_type=spks_type,
            split=split,
            hyperparameters=hyperparameters,
            nan_safe=nan_safe,
        )
        target_data = self.get_session_data(session, spks_type, split)[1]

        # If nan_safe=False, filter target_data to match filtered predictions
        if not nan_safe and "idx_valid_predictions" in extras:
            if (len(extras["idx_valid_predictions"]) / len(target_data)) < MINIMUM_NON_NAN_FRACTION:
                raise ValueError(
                    f"Too many NaN values in predictions! {len(extras['idx_valid_predictions'])} / {len(target_data)} samples have NaN values in predictions!!!"
                )
            idx_valid = extras["idx_valid_predictions"]
            target_data = target_data[:, idx_valid]

        metrics = self.evaluate(predicted_data, target_data, nan_safe=nan_safe)
        return metrics["mse"]

    def evaluate(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        nan_safe: bool = False,
    ) -> dict[str, float]:
        """Evaluate the model on the given prediction and target data.

        Parameters
        ----------
        prediction : np.ndarray
            The predicted data.
        target : np.ndarray
            The target data.
        nan_safe : bool = False
            If True, will check for NaN values in predictions and metrics and raise errors if found. If False, will remove NaN samples from the prediction and target data.

        Returns
        -------
        metrics : dict[str, float]
            A dictionary with the metrics as keys and the values as the metrics.
        """
        # Check for NaN values in predictions
        idx_nan_samples = np.any(np.isnan(prediction), axis=0)
        if nan_safe:
            if np.any(idx_nan_samples):
                raise ValueError(f"{np.sum(idx_nan_samples)} / {len(idx_nan_samples)} samples have nan predictions!!!")
        else:
            prediction = prediction[:, ~idx_nan_samples]
            target = target[:, ~idx_nan_samples]

        # Evaluate with the desired metrics
        metrics = dict(
            mse=mse(prediction, target, reduce="mean", dim=None),
            r2=measure_r2(prediction, target, reduce="mean", dim=None),
        )

        if nan_safe:
            for metric in metrics.values():
                if np.any(np.isnan(metric)):
                    raise ValueError(f"NaN values in {metric}!!!")

        return metrics

    class Report(Generic[H]):
        """A namedtuple-like class containing a full report of the model.

        Parameters
        ----------
        metrics : dict[str, float]
            A dictionary with the metrics as keys and the values as the metrics.
        predicted_data : np.ndarray
            The predicted data.
        target_data : np.ndarray
            The target data.
        trained_model : Any
            The trained model.
        hyperparameters : H
            The hyperparameters used for the model.
        extras : dict[str, Any]
            Any extras returned by the predict method.
        """

        metrics: dict[str, float]
        predicted_data: np.ndarray
        target_data: np.ndarray
        trained_model: Any
        hyperparameters: H
        extras: dict[str, Any]

        def __init__(
            self,
            metrics: dict[str, float],
            predicted_data: np.ndarray,
            target_data: np.ndarray,
            trained_model: Any,
            hyperparameters: H,
            extras: dict[str, Any],
        ):
            self.metrics = metrics
            self.predicted_data = predicted_data
            self.target_data = target_data
            self.trained_model = trained_model
            self.hyperparameters = hyperparameters
            self.extras = extras

        def __repr__(self) -> str:
            return (
                f"Report(metrics={self.metrics!r}, predicted_data=..., "
                f"target_data=..., trained_model=..., "
                f"hyperparameters={self.hyperparameters!r}, extras={self.extras!r})"
            )

        def _asdict(self) -> dict:
            """Return a dict representation of the Report."""
            return {
                "metrics": self.metrics,
                "predicted_data": self.predicted_data,
                "target_data": self.target_data,
                "trained_model": self.trained_model,
                "hyperparameters": self.hyperparameters,
                "extras": self.extras,
            }

    def process(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        test_split: Optional["SplitName"] = "test",
        hyperparameters: Optional[H] = None,
        nan_safe: bool = False,
    ) -> Report[H]:
        """Process the model on a session and return a full report.

        Parameters
        ----------
        session : B2Session
            The session to score the model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session
            provided as input.
        train_split: Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        test_split: Optional["SplitName"]
            The split to use for the scoring. If None, uses the split from the session
            provided as input. Default is "test".
        hyperparameters: Optional[Hyperparameters]
            The hyperparameters to use for the model. If None, uses the default hyperparameters for the model. Must be the appropriate subclass for the model.
        nan_safe: bool = False
            If True, will check for NaN values in predictions and metrics and raise errors if found.
            If False, will skip all NaN checks and allow NaN values to pass through.

        Returns
        -------
        report : Report
            A Report namedtuple containing a full report of the model.
        """
        # Train model and predict target activity
        hyperparameters = hyperparameters or self.hyperparameters
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
            nan_safe=nan_safe,
        )
        target_data = self.get_session_data(session, spks_type, test_split)[1]

        # If nan_safe=False, filter target_data to match filtered predictions
        if not nan_safe and "idx_valid_predictions" in extras:
            idx_valid = extras["idx_valid_predictions"]
            target_data = target_data[:, idx_valid]

        metrics = self.evaluate(predicted_data, target_data, nan_safe=nan_safe)

        report = self.Report(
            metrics=metrics,
            predicted_data=predicted_data,
            target_data=target_data,
            trained_model=trained_model,
            hyperparameters=hyperparameters,
            extras=extras,
        )

        return report

    def optimize(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
        method: Literal["grid", "optuna"] = "grid",
        nan_safe: bool = False,
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
        nan_safe: bool = False
            If True, will check for NaN values in predictions and metrics and raise errors if found.
            If False, will skip all NaN checks and allow NaN values to pass through.

        Returns
        -------
        best_params : dict
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            A DataFrame with all the results from the optimization.
        """
        if method == "grid":
            return self._optimize_grid(session, spks_type, train_split, validation_split, nan_safe=nan_safe)
        elif method == "optuna":
            return self._optimize_optuna(session, spks_type, train_split, validation_split, nan_safe=nan_safe)
        else:
            raise ValueError(f"Invalid method: {method}. Must be one of ['grid', 'optuna'].")

    def _optimize_grid(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: "SplitName",
        validation_split: "SplitName",
        nan_safe: bool = False,
    ) -> tuple[dict, float, pd.DataFrame]:
        """Optimize the hyperparameters of the model using grid search.

        Parameters
        ----------
        session : B2Session
            The session to optimize the hyperparameters for.
        spks_type : SpksTypes
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : "SplitName"
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        validation_split : "SplitName"
            The split to use for the validation. If None, uses the split from the session provided as input. Default is "validation".
        nan_safe: bool = False
            If True, will check for NaN values in predictions and metrics and raise errors if found.
            If False, will skip all NaN checks and allow NaN values to pass through.

        Returns
        -------
        best_params : dict
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            A DataFrame with all the results from the grid search.
        """
        HyperparameterClass = self._model_hyperparameters()
        independent_optimization = HyperparameterClass.independent_optimization
        if independent_optimization:
            search_space = HyperparameterClass.get_search_space()
            training_grid = HyperparameterClass.generate_grid(search_space["training"])
            prediction_grid = HyperparameterClass.generate_grid(search_space["prediction"])
        else:
            # If not independent, the search space will simply return the full space.
            # So we make a training_grid to iterate in the outer loop, and then we make an
            # empty prediction_grid so our inner loop just passes through the hyperparameters and runs once.
            search_space = HyperparameterClass.get_search_space()
            training_grid = HyperparameterClass.generate_grid(search_space)
            prediction_grid = [{}]

        best_params = None
        best_score = float("inf")
        results = []

        # Check overlap between training and prediction hyperparameters
        for training_params in training_grid:
            for prediction_params in prediction_grid:
                if set(prediction_params) & set(training_params):
                    raise ValueError(
                        f"Overlap between training and prediction hyperparameters is not allowed: "
                        f"choose where to put: {set(prediction_params) & set(training_params)}"
                    )

        # Make progress bars
        if independent_optimization:
            training_grid = tqdm(training_grid, desc="Grid search (training)", leave=False)
            prediction_grid = tqdm(prediction_grid, desc="Grid search (prediction)", leave=False)
        else:
            training_grid = tqdm(training_grid, desc="Grid search", leave=False)

        # Perform grid search
        for training_params in training_grid:
            hyperparameters = HyperparameterClass.from_dict(training_params)
            trained_model = self.train(session, spks_type=spks_type, split=train_split, hyperparameters=hyperparameters)
            for prediction_params in prediction_grid:
                hyperparameters.update_from_dict(prediction_params)
                score = self.score(
                    session,
                    trained_model,
                    spks_type=spks_type,
                    split=validation_split,
                    hyperparameters=hyperparameters,
                    nan_safe=nan_safe,
                )
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
        n_trials: int = 50,
        timeout: float | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        nan_safe: bool = False,
    ) -> tuple[dict, float, pd.DataFrame]:
        """Optimize the hyperparameters of the model using Optuna.

        Parameters
        ----------
        session : B2Session
            The session to optimize the hyperparameters for.
        spks_type : SpksTypes
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : "SplitName"
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        validation_split : "SplitName"
            The split to use for the validation. If None, uses the split from the session provided as input. Default is "validation".
        n_trials : int
            Number of Optuna trials.
        timeout : float | None
            Optional timeout in seconds (Optuna will stop once reached).
        sampler : optuna.samplers.BaseSampler | None
            Optional Optuna sampler; if None, defaults are used.
        nan_safe: bool = False
            If True, will check for NaN values in predictions and metrics and raise errors if found.
            If False, will skip all NaN checks and allow NaN values to pass through.

        Returns
        -------
        best_params : dict
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            A DataFrame with all the results from the Optuna optimization.
        """
        HyperparameterClass = self._model_hyperparameters
        independent_optimization = HyperparameterClass.independent_optimization

        if sampler is None:
            sampler = TPESampler(seed=42)

        results: list[dict] = []

        def objective(trial: optuna.Trial) -> float:
            if independent_optimization:
                full_params = HyperparameterClass.get_optuna_space(trial)
                training_params = full_params["training"]
                prediction_space = full_params["prediction"]  # e.g. a dict for grid generation
                prediction_grid = HyperparameterClass.generate_grid(prediction_space)
            else:
                training_params = HyperparameterClass.get_optuna_space(trial)
                prediction_grid = [{}]

            hyperparameters = HyperparameterClass.from_dict(training_params)
            trained_model = self.train(
                session,
                spks_type=spks_type,
                split=train_split,
                hyperparameters=hyperparameters,
            )

            # Run a grid over prediction hyperparameters (if exists, otherwise will just run once with general suggestion)
            best_score = float("inf")
            best_params = None
            for prediction_params in prediction_grid:
                if set(prediction_params) & set(training_params):
                    raise ValueError(
                        f"Overlap between training and prediction hyperparameters is not allowed: "
                        f"choose where to put: {set(prediction_params) & set(training_params)}"
                    )
                hyperparameters.update_from_dict(prediction_params)
                score = self.score(
                    session,
                    trained_model,
                    spks_type=spks_type,
                    split=validation_split,
                    hyperparameters=hyperparameters,
                    nan_safe=nan_safe,
                )
                if np.isnan(score):
                    # Invalidate the trial with infinite score just to be sure
                    score = float("inf")

                final_params = vars(hyperparameters)
                result = final_params | dict(score=score)
                results.append(result)
                if score < best_score:
                    best_score = score
                    best_params = final_params

            # Attach best params to the trial so we can recover them later
            trial.set_user_attr("best_params", best_params)

            # Optuna only sees a single scalar per trial: the best score for this training config
            trial.set_user_attr("score", float(best_score))
            return best_score

        study = optuna.create_study(
            study_name="my_model_optuna",
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
    ) -> tuple[H, float, pd.DataFrame]:
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
            grid_path = self.registry.registry_paths.score_path / f"{grid_key}.joblib"
            optuna_path = self.registry.registry_paths.score_path / f"{optuna_key}.joblib"

            # Check which exists and has better score
            if grid_path.exists() and optuna_path.exists() and not force_remake and not force_reoptimize:
                grid_metrics = load(grid_path)
                optuna_metrics = load(optuna_path)
                # Compare MSE scores (lower is better)
                if grid_metrics.get("mse", float("inf")) <= optuna_metrics.get("mse", float("inf")):
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
                if grid_metrics.get("mse", float("inf")) <= optuna_metrics.get("mse", float("inf")):
                    return grid_metrics
                else:
                    return optuna_metrics

        # First get the cache key to identify a potential cached result
        cache_key = self._get_score_cache_key(session, spks_type, train_split, validation_split, test_split, method)
        cache_path = self.registry.registry_paths.score_path / f"{cache_key}.joblib"

        if cache_path.exists() and not force_remake and not force_reoptimize:
            # If a cache exists, load it and return the best score
            metrics = load(cache_path)

        else:
            # If no cache exists, get the best hyperparameters and score the model and save the results to the cache
            hyperparameters = self.get_best_hyperparameters(
                session,
                spks_type,
                train_split,
                validation_split,
                method,
                force_remake=force_reoptimize,
            )[0]
            report = self.process(
                session,
                spks_type,
                train_split=train_split,
                test_split=test_split,
                hyperparameters=hyperparameters,
            )
            mse_roi = mse(report.predicted_data, report.target_data, reduce="none", dim=1)
            r2_roi = measure_r2(report.predicted_data, report.target_data, reduce="none", dim=1)
            report.metrics["mse_roi"] = np.array(mse_roi)
            report.metrics["r2_roi"] = np.array(r2_roi)
            metrics = report.metrics
            if self.autosave:
                dump(metrics, cache_path)

        return metrics

    def check_existing_hyperparameters(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
        method: Literal["grid", "optuna"] = "grid",
    ) -> bool:
        """Check if the hyperparameters for the model exist in the hyperparameter cache."""
        cache_key = self._get_hyperparameter_cache_key(session, spks_type, train_split, validation_split, method)
        cache_path = self.registry.registry_paths.hyperparameter_path / f"{cache_key}.joblib"
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
        cache_key = self._get_score_cache_key(session, spks_type, train_split, validation_split, test_split, method)
        cache_path = self.registry.registry_paths.score_path / f"{cache_key}.joblib"
        return cache_path.exists()

    def _get_hyperparameter_cache_key(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
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
        train_split : Optional["SplitName"]
            The split to use for training. Default is "train".
        validation_split : Optional["SplitName"]
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
        scaling_hash = stable_hash(
            (
                self.activity_parameters.center,
                self.activity_parameters.scale,
                self.activity_parameters.scale_type,
                self.activity_parameters.presplit,
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
            scaling_hash,
        ]
        return "_".join(cache_params)

    def _get_score_cache_key(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
        test_split: Optional["SplitName"] = "test",
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
        train_split : Optional["SplitName"]
            The split to use for training. Default is "train".
        validation_split : Optional["SplitName"]
            The split to use for validation. Default is "validation".
        test_split : Optional["SplitName"]
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
        cache_key = self._get_hyperparameter_cache_key(session, spks_type, train_split, validation_split, method)
        cache_path = self.registry.registry_paths.hyperparameter_path / f"{cache_key}.joblib"
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
        cache_key = self._get_score_cache_key(session, spks_type, train_split, validation_split, test_split, method)
        cache_path = self.registry.registry_paths.score_path / f"{cache_key}.joblib"
        if cache_path.exists():
            cache_path.unlink()


class HyperparametersBase(ABC):
    """Base class for hyperparameters.

    This class structures the hyperparameters for each model. The model will have to provide a search space
    for the hyperparameters, with the keys being the hyperparameters and the values being the search space
    for the hyperparameter. Supports both grid search (for discrete spaces) and Optuna (for continuous
    or high-dimensional spaces).

    Attributes
    ----------
    independent_optimization: bool
        Whether to optimize hyperparameters for training and prediction separately.
        If True, the get_search_space and get_optuna_space methods should return a dictionary with two keys: "training" and "prediction".
        If False, the get_search_space and get_optuna_space methods should return a single dictionary with the hyperparameters as keys and the search space for the hyperparameter as values.
    """

    independent_optimization: bool = False

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
    def generate_grid(cls, search_space: dict[str, tuple[Any, ...]]) -> list[dict]:
        """Generate all combinations for grid search.

        Parameters
        ----------
        search_space : dict[str, tuple[Any, ...]]
            The search space for the hyperparameters.

        Returns
        -------
        grid : list[dict]
            List of all hyperparameter combinations for grid search.
        """
        keys = search_space.keys()
        values = search_space.values()
        return [dict(zip(keys, combo)) for combo in product(*values)]

    @classmethod
    def from_dict(cls, params: dict) -> "HyperparametersBase":
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
        if "independent_optimization" in params:
            params.pop("independent_optimization")
        return cls(**params)

    def update_from_dict(self, params: dict) -> None:
        """Update the hyperparameters from a dictionary.

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameter values.
        """
        self.__dict__.update(params)

    @classmethod
    def from_optuna(cls, params: dict) -> "HyperparametersBase":
        """Create instance from Optuna parameters.

        Overwrite in cases where the Optuna parameters do not directly match
        the hyperparameters (like use_smoothing for PlaceFieldHyperparameters).

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameter values.
        """
        return params
