"""Environment-specific place-cell model performance on held-out whole trials."""

import numpy as np
import warnings

from dimensionality_manuscript.pipeline import ResultsAggregator, average_array_by_mouse
from dimensionality_manuscript.registry import ModelName
from dimensionality_manuscript.figure_scripts.panels import data_selection

from .performance import ModelPerformanceViewer


TRIAL_PERFORMANCE_MODEL_NAMES: tuple[ModelName, ...] = (
    "external_placefield_1d",
    "external_placefield_1d_gain",
    "internal_placefield_1d",
    "internal_placefield_1d_gain",
    "internal_placefield_1d_structured_gain",
    "rrr",
)
TRIAL_PERFORMANCE_MODEL_LABELS = (
    "External\nPF",
    "External\nGlobal Gain",
    "Internal\nPF",
    "Internal\nGlobal Gain",
    "Trial\nGain",
    "Peer\nPrediction",
)
TRIAL_PERFORMANCE_MODEL_COLORS = (
    "#666666",
    "#e07070",
    "#000000",
    "#c00000",
    "#008000",
    "#0000cd",
)
TRIAL_PERFORMANCE_METRICS = {"r2": r"R$^2$", "mse": "MSE"}
ENVIRONMENT_OPTIONS = ("Average", "Slot 1", "Slot 2", "Slot 3")


def trial_performance_scores(results, model_names, metric, selection, environment, avg_by_mouse):
    """Reduce environment slots within session before optionally reducing sessions by mouse."""
    per_model = []
    for model_name in model_names:
        selected = results.sel(model_name=model_name, avg_by_mouse=False, **selection)
        if metric not in selected:
            raise ValueError(
                f"No {metric!r} results are available for {model_name!r}. "
                "TrialModelPerformanceViewer requires results aggregated from "
                "TrialPlacecellRegressionConfig; also check that transferred results use its current schema."
            )
        values = np.asarray(selected[metric], dtype=float)
        if values.ndim != 2:
            raise ValueError(f"{metric!r} must have shape (sessions, environment slots), got {values.shape}")
        if environment == "Average":
            with np.errstate(invalid="ignore"):
                session_values = np.nanmean(values, axis=1)
        else:
            slot = int(environment.removeprefix("Slot ")) - 1
            session_values = values[:, slot] if slot < values.shape[1] else np.full(values.shape[0], np.nan)
        if avg_by_mouse:
            # A mouse can have no data for a later environment slot; that is a meaningful NaN,
            # not a condition worth surfacing as a runtime warning in an interactive viewer.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                session_values = average_array_by_mouse(session_values, results.mouse_names)
        per_model.append(session_values)
    return np.stack(per_model, axis=0)


class TrialModelPerformanceViewer(ModelPerformanceViewer):
    """Compare six environment-specific PC models on temporal held-out-trial predictions."""

    base_model_names = TRIAL_PERFORMANCE_MODEL_NAMES
    base_model_labels = TRIAL_PERFORMANCE_MODEL_LABELS
    base_model_colors = TRIAL_PERFORMANCE_MODEL_COLORS
    performance_metrics = TRIAL_PERFORMANCE_METRICS

    def __init__(self, results: ResultsAggregator, *, metric: str = "r2", environment: str = "Average", **kwargs):
        if environment not in ENVIRONMENT_OPTIONS:
            raise ValueError(f"environment must be one of {ENVIRONMENT_OPTIONS}")
        analysis_name = getattr(getattr(results, "config_class", None), "display_name", None)
        if analysis_name is not None and analysis_name != "trial_placecell_regression":
            raise ValueError(
                "TrialModelPerformanceViewer requires a ResultsAggregator built from " f"TrialPlacecellRegressionConfig, not {analysis_name!r}."
            )
        self._initial_environment = environment
        super().__init__(results, metric=metric, **kwargs)
        self.add_selection("environment", value=environment, options=list(ENVIRONMENT_OPTIONS))
        self.on_change("environment", self.refresh_data)

    def refresh_data(self, state):
        environment = state.get("environment", getattr(self, "_initial_environment", "Average"))
        self._scores = trial_performance_scores(
            self.results,
            self.model_names,
            state["metric"],
            data_selection(state, self.results, self.selection_names),
            environment,
            state["avg_by_mouse"],
        )
