"""Regression performance versus dimensionality for the three figure-2 models."""

import numpy as np

from vrAnalysis.helpers.plotting import errorPlot, format_spines
from dimensionality_manuscript.configs.regression import RANK_VALUES
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import ModelName
from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, apply_legend, update_legend_widgets
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)

from .performance import (
    PERFORMANCE_MODEL_COLORS,
    PERFORMANCE_MODEL_LABELS,
    PERFORMANCE_MODEL_NAMES,
    STRUCTURED_ADDITIVE_MODEL_COLOR,
)

# RegressionDimensionalitySweepConfig.process() (configs/regression.py) sweeps the same quantity
# for every model type: the rank of the projection applied to a single fitted model's prediction.
# Every model therefore writes one set of rank_{values,dim,mse,r2} keys.
SWEEP_PREFIX: str = "rank"

DIM_SWEEP_METRIC_LABELS = {"r2": r"$R^2$", "mse": "MSE"}
DIM_SWEEP_MODEL_NAMES: tuple[ModelName, ...] = tuple(PERFORMANCE_MODEL_NAMES)
DIM_SWEEP_MODEL_LABELS = tuple(PERFORMANCE_MODEL_LABELS)
DIM_SWEEP_MODEL_COLORS = tuple(PERFORMANCE_MODEL_COLORS)

RANK_SWEEP_STRUCTURED_ADDITIVE_MODELS: dict[str, ModelName] = {
    "external": "external_placefield_1d_structured_additive",
    "internal": "internal_placefield_1d_structured_additive",
}
RANK_SWEEP_STRUCTURED_ADDITIVE_LABELS = {
    "external": "Ext. Shared Residual",
    "internal": "Shared Residual",
}
RANK_SWEEP_RRR_MODEL: ModelName = "rrr"
RANK_SWEEP_RRR_LABEL = "Peer Prediction"
RANK_SWEEP_MODEL_COLORS = (STRUCTURED_ADDITIVE_MODEL_COLOR, PERFORMANCE_MODEL_COLORS[2])


def dim_sweep_curve(
    results: ResultsAggregator,
    model_name: ModelName,
    metric: str,
    selection: dict,
    *,
    allowed_dimensions: np.ndarray | None = None,
    min_included: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Projection ranks and NaN-aligned per-mouse metric means.

    Within each mouse, flatten all session/sweep observations and average metric values
    sharing the same finite dimensionality. The returned y rows are mice and its columns
    are the sorted union of dimensionalities in x; combinations a mouse did not sample stay
    NaN so population error bars use mice, rather than sessions, as observations. When supplied,
    ``allowed_dimensions`` restricts that union before aggregation and ``min_included`` removes
    ranks supported by fewer than that many mice.
    """
    prefix = SWEEP_PREFIX
    sel = results.sel(model_name=model_name, squeeze_ones=False, **selection)
    dim = np.atleast_2d(sel[f"{prefix}_dim"])
    y = np.atleast_2d(sel[f"{prefix}_{metric}"])
    if dim.shape != y.shape:
        raise ValueError(f"dimensionality and {metric} arrays must match, got {dim.shape} and {y.shape}")
    if dim.shape[0] != len(results.mouse_names):
        raise ValueError(
            "selected sweep rows must match results.mouse_names, got " f"{dim.shape[0]} rows and {len(results.mouse_names)} mouse labels"
        )

    if min_included < 1:
        raise ValueError(f"min_included must be at least 1, got {min_included}")

    x = np.unique(dim[np.isfinite(dim)])
    if allowed_dimensions is not None:
        x = x[np.isin(x, np.asarray(allowed_dimensions))]
    mouse_names = np.asarray(results.mouse_names)
    mice = list(dict.fromkeys(mouse_names.tolist()))
    mouse_means = np.full((len(mice), len(x)), np.nan, dtype=float)

    for mouse_idx, mouse in enumerate(mice):
        mouse_dim = dim[mouse_names == mouse].reshape(-1)
        mouse_y = y[mouse_names == mouse].reshape(-1)
        for dim_idx, dimensionality in enumerate(x):
            values = mouse_y[mouse_dim == dimensionality]
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                mouse_means[mouse_idx, dim_idx] = np.mean(finite_values)

    included = np.sum(np.isfinite(mouse_means), axis=0) >= min_included
    return x[included], mouse_means[:, included]


def draw_dim_sweep_panel(
    ax,
    results: ResultsAggregator,
    metric: str,
    selection: dict,
    *,
    plot_style: str,
    se: bool,
    xlog: bool,
    linewidth: float,
    fill_alpha: float,
    fontsize: float,
    legend_state: dict,
    model_names: tuple[ModelName, ...] = DIM_SWEEP_MODEL_NAMES,
    model_labels: tuple[str, ...] = DIM_SWEEP_MODEL_LABELS,
    model_colors: tuple[str, ...] = DIM_SWEEP_MODEL_COLORS,
    allowed_dimensions: np.ndarray | None = None,
    min_included: int = 1,
) -> None:
    """Draw performance-vs-dimensionality curves and apply the shared sweep styling."""
    if not (len(model_names) == len(model_labels) == len(model_colors)):
        raise ValueError("model_names, model_labels, and model_colors must have equal lengths")
    for model_name, label, color in zip(model_names, model_labels, model_colors):
        x, y = dim_sweep_curve(
            results,
            model_name,
            metric,
            selection,
            allowed_dimensions=allowed_dimensions,
            min_included=min_included,
        )
        if plot_style == "errorPlot":
            errorPlot(
                x,
                y,
                axis=0,
                se=se,
                ax=ax,
                color=color,
                linestyle="-",
                linewidth=linewidth,
                alpha=fill_alpha,
                label=label,
            )
        elif plot_style == "each":
            ax.plot(x, y.T, color=color, alpha=0.3, linewidth=0.5)
            valid_count = np.sum(np.isfinite(y), axis=0)
            mean = np.divide(
                np.nansum(y, axis=0),
                valid_count,
                out=np.full(len(x), np.nan),
                where=valid_count > 0,
            )
            ax.plot(x, mean, color=color, linewidth=linewidth, label=label)
        else:
            raise ValueError(f"Unknown plot_style {plot_style!r}. Options: ['each', 'errorPlot']")

    if xlog:
        ax.set_xscale("log")
    natural_xlim = ax.get_xlim()
    natural_ylim = ax.get_ylim()
    xlim = (1.0, natural_xlim[1])
    ylim = (0.0, natural_ylim[1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    xticks = [t for t in (1, 10, 100, 1000, 10000) if t <= xlim[1]]
    # Start at zero and keep only exact tenths below matplotlib's data-driven upper limit.
    # Restoring ``ylim`` after setting ticks prevents floating-point rounding from expanding it.
    last_tenth = int(np.floor(ylim[1] * 10 + 1e-9))
    yticks = np.arange(0, last_tenth + 1, dtype=float) / 10
    ax.set_xlabel("Model Rank", fontsize=fontsize)
    ax.set_ylabel(DIM_SWEEP_METRIC_LABELS[metric], fontsize=fontsize, labelpad=-10)
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        xbounds=xlim,
        ybounds=ylim,
        xticks=xticks,
        yticks=yticks,
        tick_fontsize=fontsize,
        spines_visible=["left", "bottom"],
    )
    # Explicitly restore the chosen limits after setting ticks; matplotlib otherwise permits
    # fixed ticks to enlarge an axis when floating-point rounding puts one just outside its range.
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Apply this after format_spines, which calls tick_params itself. Include minor ticks
    # so logarithmic-axis labels follow the viewer fontsize as well.
    ax.tick_params(axis="both", which="both", labelsize=fontsize)
    apply_legend(ax, legend_state, fontsize, auto_loc="upper left")


def add_dim_sweep_widgets(
    viewer,
    *,
    metric: str = "r2",
    plot_style: str = "errorPlot",
    se: bool = True,
    xlog: bool = True,
    linewidth: float = 2.0,
    fill_alpha: float = 0.12,
    legend_options: dict | None = None,
) -> None:
    """Add the sweep panel's metric, curve, and standard legend knobs to ``viewer``."""
    if metric not in DIM_SWEEP_METRIC_LABELS:
        raise ValueError(f"metric must be one of {list(DIM_SWEEP_METRIC_LABELS)}, got {metric!r}")
    if plot_style not in ("each", "errorPlot"):
        raise ValueError(f"plot_style must be 'each' or 'errorPlot', got {plot_style!r}")
    viewer.add_selection("metric", value=metric, options=list(DIM_SWEEP_METRIC_LABELS))
    viewer.add_selection("plot_style", value=plot_style, options=["each", "errorPlot"])
    viewer.add_boolean("se", value=se)
    viewer.add_boolean("xlog", value=xlog)
    viewer.add_float("linewidth", value=linewidth, min=0.5, max=6.0)
    viewer.add_float("fill_alpha", value=fill_alpha, min=0.0, max=1.0)
    add_legend_widgets(viewer)
    update_legend_widgets(viewer, legend_options or {})


class RegressionDimSweepViewer(FigureViewer):
    """Mean +/- SE performance-vs-dimensionality curves for PF, PF+Gain, and Peer.

    Every model sweeps the same quantity -- the rank of the projection applied to one fitted
    model's prediction -- so the three curves share an x-axis. The highest rank a session reaches
    still depends on its data (number of target cells, number of environments), so sessions don't
    share an exact grid. Session/sweep observations are grouped by their rank and averaged within
    mouse. The resulting NaN-aligned mouse curves can be drawn individually with their mean,
    or as the across-mouse mean +/- SE with
    :func:`~vrAnalysis.helpers.plotting.errorPlot`.

    The model set is fixed to the same comparison used by the other figure-2 score panels:
    plain place fields (black), place fields with gain (red), and peer prediction/RRR (blue).
    All three curves use solid lines.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionDimensionalitySweepConfig`` results, with ``model_name`` as a
        ``model_name`` param axis covering PF, PF+Gain, and RRR.
    metric : {"r2", "mse"}
        Fit metric to plot on the y-axis.
    plot_style : {"each", "errorPlot"}
        Draw every mouse faintly plus the across-mouse mean, or draw the mean and error band.
    se : bool
        Standard error (True) or standard deviation (False) shading.
    xlog : bool
        Log-scale the dimensionality axis.
    linewidth : float
        Mean-curve line width.
    fill_alpha : float
        Opacity of the +/- SE/SD fill band.
    fontsize : float
        Font size for axis labels, tick labels, and the legend.
    figsize : tuple[float, float]
        Figure size in inches.
    legend_options : dict or None
        Legend knobs forwarded to :mod:`~dimensionality_manuscript.figure_scripts.legends`
        (``{"loc": ..., "ncols": ...}``); ``{"loc": "none"}`` hides it.
    **selection_defaults
        Starting values for the data-selection widgets built from the aggregator's param axes.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        metric: str = "r2",
        plot_style: str = "errorPlot",
        se: bool = True,
        xlog: bool = True,
        linewidth: float = 2.0,
        fill_alpha: float = 0.12,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (6.0, 4.5),
        legend_options: dict | None = None,
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.selection_names = add_data_selection_widgets(self, results, skip=("model_name",), defaults=selection_defaults)
        add_dim_sweep_widgets(
            self,
            metric=metric,
            plot_style=plot_style,
            se=se,
            xlog=xlog,
            linewidth=linewidth,
            fill_alpha=fill_alpha,
            legend_options=legend_options,
        )
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

    def plot(self, state):
        fig, ax = self.new_subplots(figsize=self.figsize, layout="constrained")
        draw_dim_sweep_panel(
            ax,
            self.results,
            state["metric"],
            data_selection(state, self.results, self.selection_names),
            plot_style=state["plot_style"],
            se=state["se"],
            xlog=state["xlog"],
            linewidth=state["linewidth"],
            fill_alpha=state["fill_alpha"],
            fontsize=state["fontsize"],
            legend_state=state,
        )
        return fig


class RankModelsSweepViewer(FigureViewer):
    """Regression-rank curves for one structured-additive model and RRR.

    This viewer reads :class:`~dimensionality_manuscript.configs.regression.RankModelsSweepConfig`
    results.  The ``structured_additive`` selection switches the orange curve between the
    external and internal structured-additive models; the blue RRR/Peer curve remains fixed.
    Like :class:`RegressionDimSweepViewer`, observations are first averaged within mouse at
    every measured rank and may be shown as individual mouse curves plus their mean or as an
    across-mouse mean +/- SE/SD band.

    Only ranks from the config's shared :data:`~dimensionality_manuscript.configs.regression.RANK_VALUES`
    grid are shown; session-specific ``full_rank`` endpoints are discarded. Parameters are the
    same display and data-selection controls as :class:`RegressionDimSweepViewer`, with the
    additions of ``structured_additive`` and ``min_included``.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RankModelsSweepConfig`` results.
    structured_additive : {"external", "internal"}
        Structured-additive model to compare with RRR.
    min_included : int
        Minimum number of mice with finite data required to retain a rank in a model's plot.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        structured_additive: str = "external",
        min_included: int = 2,
        metric: str = "r2",
        plot_style: str = "errorPlot",
        se: bool = True,
        xlog: bool = True,
        linewidth: float = 2.0,
        fill_alpha: float = 0.12,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (6.0, 4.5),
        legend_options: dict | None = None,
        **selection_defaults,
    ):
        if structured_additive not in RANK_SWEEP_STRUCTURED_ADDITIVE_MODELS:
            raise ValueError("structured_additive must be one of " f"{list(RANK_SWEEP_STRUCTURED_ADDITIVE_MODELS)}, got {structured_additive!r}")
        self.results = results
        self.figsize = figsize
        self.selection_names = add_data_selection_widgets(self, results, skip=("model_name",), defaults=selection_defaults)
        self.add_selection(
            "structured_additive",
            value=structured_additive,
            options=list(RANK_SWEEP_STRUCTURED_ADDITIVE_MODELS),
        )
        num_mice = len(set(np.asarray(results.mouse_names).tolist()))
        if not 1 <= min_included <= max(num_mice, 1):
            raise ValueError(f"min_included must be between 1 and {max(num_mice, 1)}, got {min_included}")
        self.add_integer("min_included", value=min_included, min=1, max=max(num_mice, 1))
        add_dim_sweep_widgets(
            self,
            metric=metric,
            plot_style=plot_style,
            se=se,
            xlog=xlog,
            linewidth=linewidth,
            fill_alpha=fill_alpha,
            legend_options=legend_options,
        )
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

    def plot(self, state):
        structured_additive = state["structured_additive"]
        model_names = (
            RANK_SWEEP_STRUCTURED_ADDITIVE_MODELS[structured_additive],
            RANK_SWEEP_RRR_MODEL,
        )
        model_labels = (
            RANK_SWEEP_STRUCTURED_ADDITIVE_LABELS[structured_additive],
            RANK_SWEEP_RRR_LABEL,
        )
        fig, ax = self.new_subplots(figsize=self.figsize, layout="constrained")
        draw_dim_sweep_panel(
            ax,
            self.results,
            state["metric"],
            data_selection(state, self.results, self.selection_names),
            plot_style=state["plot_style"],
            se=state["se"],
            xlog=state["xlog"],
            linewidth=state["linewidth"],
            fill_alpha=state["fill_alpha"],
            fontsize=state["fontsize"],
            legend_state=state,
            model_names=model_names,
            model_labels=model_labels,
            model_colors=RANK_SWEEP_MODEL_COLORS,
            allowed_dimensions=RANK_VALUES,
            min_included=state["min_included"],
        )
        return fig
