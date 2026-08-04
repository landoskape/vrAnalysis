"""Regression performance versus dimensionality for the three figure-2 models."""

import numpy as np

from vrAnalysis.helpers.plotting import errorPlot, format_spines
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import ModelName
from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, apply_legend, update_legend_widgets
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)

from .performance import PERFORMANCE_MODEL_COLORS, PERFORMANCE_MODEL_LABELS, PERFORMANCE_MODEL_NAMES

# RegressionDimensionalitySweepConfig.process() (configs/regression.py) sweeps a different
# hyperparameter per model type: placefield models vary num_bins, RBFPos/pos_speed/full-regressor
# models vary num_basis, and RRR varies rank. Each produces its own {prefix}_{values,dim,mse,r2}
# result keys, so a model's sweep data must be read from its own prefix.
SWEEP_KEY_BY_MODEL: dict[ModelName, str] = {
    "internal_placefield_1d": "num_bins",
    "internal_placefield_1d_gain": "num_bins",
    "rrr": "rank",
}

DIM_SWEEP_METRIC_LABELS = {"r2": r"$R^2$", "mse": "MSE"}
DIM_SWEEP_MODEL_NAMES: tuple[ModelName, ...] = tuple(PERFORMANCE_MODEL_NAMES)
DIM_SWEEP_MODEL_LABELS = tuple(PERFORMANCE_MODEL_LABELS)
DIM_SWEEP_MODEL_COLORS = tuple(PERFORMANCE_MODEL_COLORS)


def dim_sweep_curve(
    results: ResultsAggregator,
    model_name: ModelName,
    metric: str,
    selection: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Across-session x (mean dim, sorted) and y (metric) arrays: shapes (n_pts,), (n_sess, n_pts)."""
    prefix = SWEEP_KEY_BY_MODEL[model_name]
    sel = results.sel(model_name=model_name, squeeze_ones=False, **selection)
    dim = np.atleast_2d(sel[f"{prefix}_dim"])
    y = np.atleast_2d(sel[f"{prefix}_{metric}"])
    x = np.nanmean(dim, axis=0)
    idx_sort = np.argsort(x)
    return x[idx_sort], y[:, idx_sort]


def draw_dim_sweep_panel(
    ax,
    results: ResultsAggregator,
    metric: str,
    selection: dict,
    *,
    se: bool,
    xlog: bool,
    linewidth: float,
    fill_alpha: float,
    fontsize: float,
    legend_state: dict,
) -> None:
    """Draw fixed PF, PF+Gain, and Peer performance-vs-dimensionality curves."""
    for model_name, label, color in zip(DIM_SWEEP_MODEL_NAMES, DIM_SWEEP_MODEL_LABELS, DIM_SWEEP_MODEL_COLORS):
        x, y = dim_sweep_curve(results, model_name, metric, selection)
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
    ax.set_xlabel("Model Dimensionality", fontsize=fontsize)
    ax.set_ylabel(DIM_SWEEP_METRIC_LABELS[metric], fontsize=fontsize)
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
    se: bool = True,
    xlog: bool = True,
    linewidth: float = 2.0,
    fill_alpha: float = 0.12,
    legend_options: dict | None = None,
) -> None:
    """Add the sweep panel's metric, curve, and standard legend knobs to ``viewer``."""
    if metric not in DIM_SWEEP_METRIC_LABELS:
        raise ValueError(f"metric must be one of {list(DIM_SWEEP_METRIC_LABELS)}, got {metric!r}")
    viewer.add_selection("metric", value=metric, options=list(DIM_SWEEP_METRIC_LABELS))
    viewer.add_boolean("se", value=se)
    viewer.add_boolean("xlog", value=xlog)
    viewer.add_float("linewidth", value=linewidth, min=0.5, max=6.0)
    viewer.add_float("fill_alpha", value=fill_alpha, min=0.0, max=1.0)
    add_legend_widgets(viewer)
    update_legend_widgets(viewer, legend_options or {})


class RegressionDimSweepViewer(FigureViewer):
    """Mean +/- SE performance-vs-dimensionality curves for PF, PF+Gain, and Peer.

    Each model sweeps a different hyperparameter (num_bins / num_basis / rank; see
    ``SWEEP_KEY_BY_MODEL``), and the resulting regressor dimensionality also depends on a
    session's number of environments, so sessions don't share an exact dimensionality grid.
    The x position at sweep index ``i`` is therefore the across-session mean dimensionality at
    that index, and the curve is the across-session mean +/- SE of the metric at that index,
    drawn with :func:`~vrAnalysis.helpers.plotting.errorPlot`.

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
            se=state["se"],
            xlog=state["xlog"],
            linewidth=state["linewidth"],
            fill_alpha=state["fill_alpha"],
            fontsize=state["fontsize"],
            legend_state=state,
        )
        return fig
