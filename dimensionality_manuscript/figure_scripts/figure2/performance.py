"""Performance of the PF, gain, and peer models, relative to PF."""

import numpy as np

from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import ModelName
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    add_score_inset_widgets,
    add_trace_style_widgets,
    data_selection,
    draw_score_inset,
    style_model_axis,
)

from ._scores import draw_subject_traces, performance_scores

# The panel's model set is fixed by design: it is the slice of the zoo this figure argues about,
# laid out along the x-axis in a deliberate order. The labels and colors are panel-specific and
# deliberately do not inherit the obsolete external/internal/neural role taxonomy.
PERFORMANCE_MODEL_NAMES: list[ModelName] = [
    "internal_placefield_1d",
    "internal_placefield_1d_gain",
    "rrr",
]
PERFORMANCE_MODEL_LABELS = ("Placefield", "Global\nGain", "Peer\nPrediction")
PERFORMANCE_MODEL_COLORS = ("#000000", "#c00000", "#0000cd")

STRUCTURED_ADDITIVE_MODEL_NAME: ModelName = "internal_placefield_1d_structured_additive"
STRUCTURED_ADDITIVE_MODEL_LABEL = "Shared\nResidual"
STRUCTURED_ADDITIVE_MODEL_COLOR = "#c06000"

# Axis label for each selectable metric. Both are plotted in their stored units -- R^2 as a
# fraction (<= 1), not a percentage -- so there is no display rescaling anywhere in this panel.
PERFORMANCE_METRICS: dict[str, str] = {
    "r2": r"R$^2$",
    "mse": "MSE",
}


class ModelPerformanceViewer(FigureViewer):
    """One score each for PF, PF+Gain, optionally StructuredAdditive, and Peer, relative to PF.

    Each model contributes a single number per subject (a mouse when ``avg_by_mouse``, else a
    session): its optimized-hyperparameter test score from ``RegressionConfig``. The main axis
    plots that score minus the PF score, so the panel reads as "what each extra ingredient buys
    over a plain placefield model", with one faint line per subject behind the across-subject
    mean. PF, PF+Gain, and Peer are consistently colored black, red, and blue; the optional
    StructuredAdditive point is orange.

    An inset repeats the same traces without the PF subtraction, so the absolute scale of the
    scores stays visible next to the differences.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionConfig`` results, with ``model_name`` as a param axis covering
        PF, PF+Gain, and Peer, plus StructuredAdditive when requested.
    metric : {"r2", "mse"}
        Fit metric, plotted in its stored units: R^2 as a fraction (<= 1), MSE in its own units.
        Every y knob below is in those units too.
    avg_by_mouse : bool
        Average sessions within a mouse before plotting, so one trace is one mouse.
    include_structured_additive : bool
        Insert the internal structured-additive model between PF+Gain and Peer.
    fontsize : float
        Font size for the main axis's tick labels and y label.
    xtick_rotation : float
        Rotation of the model-name tick labels, in degrees.
    ytick_max : float
        Upper y tick on the main axis (the lower one is 0).
    ylim : tuple[float, float]
        Main-axis y limits. The offset left spine spans exactly this range, so it also sets how
        far the spine extends past the ticks.
    show_zero_line : bool
        Draw the dashed y = 0 reference line on the main axis.
    zero_line_pad : float
        How far the zero line extends past the outermost ticks, in tick units, on both sides.
    figsize : tuple[float, float]
        Figure size in inches.
    **style
        Trace and inset knobs, forwarded to
        :func:`~dimensionality_manuscript.figure_scripts.panels.add_trace_style_widgets`,
        and :func:`~dimensionality_manuscript.figure_scripts.panels.add_score_inset_widgets`.
        Data-selection widgets (``spks_type``, ``activity_parameters_name``, ...) come from the
        aggregator's own param axes and can be seeded here by name.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        metric: str = "r2",
        avg_by_mouse: bool = True,
        include_structured_additive: bool = False,
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
        if metric not in PERFORMANCE_METRICS:
            raise ValueError(f"metric must be one of {list(PERFORMANCE_METRICS)}, got {metric!r}")

        self.results = results
        self.figsize = figsize
        self.include_structured_additive = include_structured_additive
        self.model_names = list(PERFORMANCE_MODEL_NAMES)
        self.model_labels = list(PERFORMANCE_MODEL_LABELS)
        self.model_colors = list(PERFORMANCE_MODEL_COLORS)
        if include_structured_additive:
            self.model_names.insert(2, STRUCTURED_ADDITIVE_MODEL_NAME)
            self.model_labels.insert(2, STRUCTURED_ADDITIVE_MODEL_LABEL)
            self.model_colors.insert(2, STRUCTURED_ADDITIVE_MODEL_COLOR)
        self._scores = np.empty((len(self.model_names), 0))

        # --- data selection (model_name is fixed by self.model_names) ---
        self.selection_names = add_data_selection_widgets(self, results, skip=("model_name",), defaults=selection_defaults)
        self.add_selection("metric", value=metric, options=list(PERFORMANCE_METRICS))
        self.add_boolean("avg_by_mouse", value=avg_by_mouse)

        # --- main axis style ---
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        # y knobs are in metric units (R^2 as a fraction), with explicit steps: syd rounds a value
        # to its parameter's step, and the default 0.01 is too coarse for an axis spanning ~0.1.
        self.add_float("ytick_max", value=ytick_max, min=0.001, max=2.0, step=0.001)
        self.add_float_range("ylim", value=ylim, min=-1.0, max=2.0, step=0.001)
        self.add_boolean("show_zero_line", value=show_zero_line)
        self.add_float("zero_line_pad", value=zero_line_pad, min=0.0, max=2.0)
        add_trace_style_widgets(
            self,
            markersize=markersize,
            mean_linewidth=mean_linewidth,
            subject_linewidth=subject_linewidth,
            subject_alpha=subject_alpha,
        )
        add_score_inset_widgets(
            self,
            show_inset=show_inset,
            inset_x=inset_x,
            inset_y=inset_y,
            inset_width=inset_width,
            inset_height=inset_height,
            inset_markersize=inset_markersize,
            inset_fontsize=inset_fontsize,
            inset_ytick_max=inset_ytick_max,
            inset_ylim=inset_ylim,
        )

        for name in (*self.selection_names, "metric", "avg_by_mouse"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select scores for the current data-selection, metric, and averaging."""
        self._scores = performance_scores(
            self.results,
            self.model_names,
            state["metric"],
            data_selection(state, self.results, self.selection_names),
            state["avg_by_mouse"],
        )

    def plot(self, state):
        fontsize = state["fontsize"]
        metric_label = PERFORMANCE_METRICS[state["metric"]]

        xvals = np.arange(len(self.model_names), dtype=float)
        scores = self._scores
        relative_scores = scores - scores[0]

        fig, ax = self.new_subplots(1, 1, figsize=self.figsize, layout="constrained")

        if state["show_zero_line"]:
            pad = state["zero_line_pad"]
            ax.plot([xvals[0] - pad, xvals[-1] + pad], [0, 0], color="k", linewidth=0.5, linestyle="--")
        draw_subject_traces(ax, xvals, relative_scores, self.model_colors, state)

        # Fix ylim before styling: format_spines positions the offset spines relative to the
        # current limits. The x limits are left to matplotlib (the dashed zero line sets the
        # padding), while the bottom spine spans the models themselves.
        ax.set_ylim(state["ylim"])
        style_model_axis(
            ax,
            fontsize=fontsize,
            xvals=xvals,
            labels=self.model_labels,
            xbounds=[xvals[0], xvals[-1]],
            ybounds=state["ylim"],
            yticks=[0, state["ytick_max"]],
            xtick_rotation=0,
            xha="center",
        )
        for tick_label, color in zip(ax.get_xticklabels(), self.model_colors):
            tick_label.set_color(color)
        ax.set_ylabel(rf"$\Delta$ {metric_label}", labelpad=-10, fontsize=fontsize)

        draw_score_inset(
            ax,
            state,
            lambda inset: draw_subject_traces(
                inset,
                xvals,
                scores,
                self.model_colors,
                state,
                markersize=state["inset_markersize"],
            ),
            metric_label,
        )
        return fig
