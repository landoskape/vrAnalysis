"""Environment-specific place-cell model performance on held-out whole trials."""

import warnings

import numpy as np

from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import ModelName
from dimensionality_manuscript.figure_scripts.panels import data_selection

from .performance import ModelPerformanceViewer

TRIAL_PERFORMANCE_MODEL_NAMES: tuple[ModelName, ...] = (
    "internal_placefield_1d",
    "internal_placefield_1d_gain",
    "internal_placefield_1d_structured_gain",
    "rrr",
)
TRIAL_PERFORMANCE_MODEL_LABELS = (
    "Placefield",
    "Global\nGain",
    "Trial\nGain",
    "Peer\nPrediction",
)
TRIAL_PERFORMANCE_MODEL_COLORS = (
    "#000000",
    "#c00000",
    "#008000",
    "#0000cd",
)
TRIAL_PERFORMANCE_METRICS = {"r2": r"R$^2$", "mse": "MSE"}
ENVIRONMENT_OPTIONS = ("Average", "Slot 1", "Slot 2", "Slot 3")
AGGREGATION_OPTIONS = ("mean", "median")


def _finite_reduce(values: np.ndarray, axis: int, aggregation: str) -> np.ndarray:
    """Reduce finite values without warning when an entire slice is missing."""
    reducer = np.nanmean if aggregation == "mean" else np.nanmedian
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return reducer(values, axis=axis)


def _reduce_sessions_by_mouse(values: np.ndarray, mouse_names, aggregation: str) -> np.ndarray:
    """Reduce session values within each mouse, retaining first-seen mouse order."""
    mouse_names = np.asarray(mouse_names)
    mice = list(dict.fromkeys(mouse_names))
    return np.asarray([_finite_reduce(values[mouse_names == mouse], axis=0, aggregation=aggregation) for mouse in mice])


def trial_performance_scores(results, model_names, metric, selection, environment, avg_by_mouse, aggregation="mean"):
    """Reduce environment slots within session before optionally reducing sessions by mouse."""
    if aggregation not in AGGREGATION_OPTIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATION_OPTIONS}, got {aggregation!r}")
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
            session_values = _finite_reduce(values, axis=1, aggregation=aggregation)
        else:
            slot = int(environment.removeprefix("Slot ")) - 1
            session_values = values[:, slot] if slot < values.shape[1] else np.full(values.shape[0], np.nan)
        if avg_by_mouse:
            session_values = _reduce_sessions_by_mouse(session_values, results.mouse_names, aggregation)
        per_model.append(session_values)
    return np.stack(per_model, axis=0)


class TrialModelPerformanceViewer(ModelPerformanceViewer):
    """Compare environment-specific PC models on temporal held-out-trial predictions.

    External PF is optional via ``include_external_placefield`` (default False). The main
    axis subtracts internal PF, or external PF when included. The inset shows absolute scores.
    Style parameters match :class:`ModelPerformanceViewer`; ``selection_defaults`` seeds
    data-selection widgets from the results aggregator.

    ``aggregation`` controls every nested summary: environment slots within each session,
    sessions within each mouse, and the across-subject center drawn on both axes.
    """

    base_model_names = TRIAL_PERFORMANCE_MODEL_NAMES
    base_model_labels = TRIAL_PERFORMANCE_MODEL_LABELS
    base_model_colors = TRIAL_PERFORMANCE_MODEL_COLORS
    performance_metrics = TRIAL_PERFORMANCE_METRICS

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        metric: str = "r2",
        environment: str = "Average",
        aggregation: str = "mean",
        avg_by_mouse: bool = True,
        include_structured_additive: bool = False,
        include_external_placefield: bool = False,
        fontsize: float = 12.0,
        xtick_rotation: float = 45.0,
        ytick_max: float = 0.1,
        ylim: tuple[float, float] = (-0.03, 0.12),
        show_zero_line: bool = True,
        zero_line_pad: float = 0.5,
        figsize: tuple[float, float] = (7.0, 5.0),
        markersize: float = 5.0,
        mean_linewidth: float = 1.5,
        subject_linewidth: float = 0.5,
        subject_alpha: float = 0.3,
        show_inset: bool = True,
        inset_x: float = 0.1,
        inset_y: float = 0.5,
        inset_width: float = 0.35,
        inset_height: float = 0.45,
        inset_markersize: float = 3.0,
        inset_fontsize: float = 10.0,
        inset_ytick_max: float = 0.2,
        inset_ylim: tuple[float, float] = (-0.02, 0.25),
        **selection_defaults,
    ):
        if environment not in ENVIRONMENT_OPTIONS:
            raise ValueError(f"environment must be one of {ENVIRONMENT_OPTIONS}")
        if aggregation not in AGGREGATION_OPTIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATION_OPTIONS}")
        analysis_name = getattr(getattr(results, "config_class", None), "display_name", None)
        if analysis_name is not None and analysis_name != "trial_placecell_regression":
            raise ValueError(
                "TrialModelPerformanceViewer requires a ResultsAggregator built from " f"TrialPlacecellRegressionConfig, not {analysis_name!r}."
            )
        self._initial_environment = environment
        self._initial_aggregation = aggregation
        super().__init__(
            results,
            metric=metric,
            avg_by_mouse=avg_by_mouse,
            include_structured_additive=include_structured_additive,
            include_external_placefield=include_external_placefield,
            fontsize=fontsize,
            xtick_rotation=xtick_rotation,
            ytick_max=ytick_max,
            ylim=ylim,
            show_zero_line=show_zero_line,
            zero_line_pad=zero_line_pad,
            figsize=figsize,
            markersize=markersize,
            mean_linewidth=mean_linewidth,
            subject_linewidth=subject_linewidth,
            subject_alpha=subject_alpha,
            show_inset=show_inset,
            inset_x=inset_x,
            inset_y=inset_y,
            inset_width=inset_width,
            inset_height=inset_height,
            inset_markersize=inset_markersize,
            inset_fontsize=inset_fontsize,
            inset_ytick_max=inset_ytick_max,
            inset_ylim=inset_ylim,
            **selection_defaults,
        )
        self.add_selection("environment", value=environment, options=list(ENVIRONMENT_OPTIONS))
        self.add_selection("aggregation", value=aggregation, options=list(AGGREGATION_OPTIONS))
        self.on_change("environment", self.refresh_data)
        self.on_change("aggregation", self.refresh_data)

    def refresh_data(self, state):
        environment = state.get("environment", getattr(self, "_initial_environment", "Average"))
        aggregation = state.get("aggregation", getattr(self, "_initial_aggregation", "mean"))
        self._scores = trial_performance_scores(
            self.results,
            self.model_names,
            state["metric"],
            data_selection(state, self.results, self.selection_names),
            environment,
            state["avg_by_mouse"],
            aggregation,
        )
