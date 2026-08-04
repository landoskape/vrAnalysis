"""Place-field residual RMS, within and outside the target place field.

Two panels over the same models: one summarizing each mouse with a single number
(:class:`ModelPlacefieldResidualViewer`), one following each mouse across its chronological
sessions (:class:`ModelPlacefieldResidualFamiliarityViewer`).
"""

import numpy as np

from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import ModelName
from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, apply_legend, update_legend_widgets
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    add_trace_style_widgets,
    data_selection,
    style_model_axis,
)

from ._scores import (
    SESSION_XLABEL,
    add_familiarity_curve_widgets,
    draw_familiarity_series,
    draw_subject_traces,
    mouse_session_curves,
)
from .model_style import ROLE_COLOR
from .performance import PERFORMANCE_MODEL_COLORS, PERFORMANCE_MODEL_LABELS

PF_RESIDUAL_MODEL_NAMES: list[ModelName] = [
    "internal_placefield_1d",
    "internal_placefield_1d_gain",
    "rrr",
]
PF_RESIDUAL_METRICS = ("within_pf_rms", "outside_pf_rms")
PF_RESIDUAL_SUBSETS = ("all", "quality", "not quality")
PF_RESIDUAL_LABELS = {
    "within_pf_rms": "Within-PF\nResidual RMS",
    "outside_pf_rms": "Outside-PF\nResidual RMS",
}

# The familiarity panel names its three curves by role rather than by the models' own roles: the
# two placefield variants are the comparison, so the plain one takes the external color.
PF_RESIDUAL_FAMILIARITY_COLORS = {
    "internal_placefield_1d": ROLE_COLOR["external"],
    "internal_placefield_1d_gain": ROLE_COLOR["internal"],
    "rrr": ROLE_COLOR["neural"],
}
PF_RESIDUAL_FAMILIARITY_LABELS = {
    "internal_placefield_1d": "PF",
    "internal_placefield_1d_gain": "PF+Gain",
    "rrr": "Peer Prediction",
}


def _residual_key_prefix(subset: bool | str, normalized: bool) -> str:
    """Result-key prefix for the requested precomputed scalar summary."""
    prefix = "mean_"
    # Keep accepting the familiarity viewer's boolean quality selection while the
    # summary viewer exposes all three available ROI subsets.
    if subset is True or subset == "quality":
        prefix += "quality_filtered_"
    elif subset == "not quality":
        prefix += "notquality_filtered_"
    elif subset is not False and subset != "all":
        raise ValueError(f"Unknown residual subset: {subset!r}")
    if normalized:
        prefix += "normalized_"
    return prefix


def _residual_ylabel(metric: str, normalized: bool, relative: bool) -> str:
    """Y label for one residual metric under the current display options."""
    label = PF_RESIDUAL_LABELS[metric]
    if normalized:
        label = f"Normalized {label.lower()}"
    if relative:
        label = rf"$\Delta$ {label}"
    return label


class ModelPlacefieldResidualViewer(FigureViewer):
    """Within- (ax[0]) and outside-place-field (ax[1]) residual RMS for the residual models.

    Results come from ``RegressionPlacefieldResidualConfig``. Each circular point is one mouse
    after averaging the session-level, across-ROI finite mean within that mouse; horizontal
    colored marks show the across-mouse mean for each model.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionPlacefieldResidualConfig`` results.
    model_names : list[ModelName]
        Models to draw, in x order.
    normalized : bool
        Select residual RMS divided by each ROI's held-out target standard deviation.
    main_show, inset_show : {"all", "quality", "not quality"}
        ROI subset displayed in the main panels and inset panels, respectively. ``inset_show``
        has no visual effect when ``include_inset`` is False.
    sharey : bool
        Give the two panels a common y-axis scale.
    relative : bool
        Subtract each mouse's first-model value from all of that mouse's values, independently
        on each panel.
    fontsize, markersize, mean_linewidth, subject_linewidth, subject_alpha : float
        Style knobs for the per-mouse traces and the across-mouse mean trace and dots.
    within_text_x, within_text_y, outside_text_x, outside_text_y : float
        Axes-fraction coordinates for the ``Within PF`` and ``Outside PF`` panel text.
    inset_x, inset_y, inset_width, inset_height : float
        Shared axes-fraction bounds for the inset in each panel.
    placecells_x, placecells_y : float
        Shared axes-fraction position for the inset's ``placecells`` label.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the data-selection widgets built from the aggregator's param axes.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        model_names: list[ModelName] = PF_RESIDUAL_MODEL_NAMES,
        normalized: bool = False,
        main_show: str = "all",
        inset_show: str = "quality",
        sharey: bool = True,
        relative: bool = False,
        fontsize: float = 12.0,
        markersize: float = 5.0,
        mean_linewidth: float = 1.5,
        subject_linewidth: float = 0.5,
        subject_alpha: float = 0.3,
        within_text_x: float = 0.05,
        within_text_y: float = 0.9,
        outside_text_x: float = 0.05,
        outside_text_y: float = 0.9,
        include_inset: bool = True,
        inset_x: float = 0.6,
        inset_y: float = 0.55,
        inset_width: float = 0.35,
        inset_height: float = 0.35,
        placecells_x: float = 0.05,
        placecells_y: float = 0.05,
        figsize: tuple[float, float] = (7.0, 3.5),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.model_names = list(model_names)
        self._scores: dict[str, np.ndarray] = {}
        self._inset_scores: dict[str, np.ndarray] = {}

        self.selection_names = add_data_selection_widgets(self, results, skip=("model_name",), defaults=selection_defaults)
        self.add_boolean("normalized", value=normalized)
        self.add_selection("main_show", value=main_show, options=list(PF_RESIDUAL_SUBSETS))
        self.add_selection("inset_show", value=inset_show, options=list(PF_RESIDUAL_SUBSETS))
        self.add_boolean("sharey", value=sharey)
        self.add_boolean("relative", value=relative)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        add_trace_style_widgets(
            self,
            markersize=markersize,
            mean_linewidth=mean_linewidth,
            subject_linewidth=subject_linewidth,
            subject_alpha=subject_alpha,
        )
        self.add_float("within_text_x", value=within_text_x, min=0.0, max=1.0, step=0.01)
        self.add_float("within_text_y", value=within_text_y, min=0.0, max=1.0, step=0.01)
        self.add_float("outside_text_x", value=outside_text_x, min=0.0, max=1.0, step=0.01)
        self.add_float("outside_text_y", value=outside_text_y, min=0.0, max=1.0, step=0.01)
        self.add_float("inset_x", value=inset_x, min=0.0, max=1.0, step=0.01)
        self.add_float("inset_y", value=inset_y, min=0.0, max=1.0, step=0.01)
        self.add_float("inset_width", value=inset_width, min=0.05, max=1.0, step=0.01)
        self.add_float("inset_height", value=inset_height, min=0.05, max=1.0, step=0.01)
        self.add_float("placecells_x", value=placecells_x, min=0.0, max=1.0, step=0.01)
        self.add_float("placecells_y", value=placecells_y, min=0.0, max=1.0, step=0.01)
        self.add_boolean("include_inset", value=include_inset)
        for name in (*self.selection_names, "normalized", "main_show", "inset_show"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Load the selected main and inset ROI subsets, averaged within mouse."""
        selection = data_selection(state, self.results, self.selection_names)

        def load_scores(subset):
            prefix = _residual_key_prefix(subset, state["normalized"])
            return {
                metric: np.stack(
                    [
                        self.results.sel(model_name=model_name, keys=[prefix + metric], avg_by_mouse=True, **selection)[prefix + metric]
                        for model_name in self.model_names
                    ],
                    axis=0,
                )
                for metric in PF_RESIDUAL_METRICS
            }

        self._scores = load_scores(state["main_show"])
        self._inset_scores = load_scores(state["inset_show"])

    def plot(self, state):
        fontsize = state["fontsize"]
        xvals = np.arange(len(self.model_names), dtype=float)

        fig, ax = self.new_subplots(1, 2, figsize=self.figsize, layout="constrained", sharey=state["sharey"])
        for axis, metric, text, text_x, text_y in zip(
            ax,
            PF_RESIDUAL_METRICS,
            ("Within PF", "Outside PF"),
            (state["within_text_x"], state["outside_text_x"]),
            (state["within_text_y"], state["outside_text_y"]),
        ):
            values = self._scores[metric]
            if state["relative"]:
                values = values - values[0]
                axis.axhline(0.0, color="k", linewidth=0.5, linestyle="--")
            draw_subject_traces(axis, xvals, values, PERFORMANCE_MODEL_COLORS, state)
            axis.text(
                text_x,
                text_y,
                text,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=fontsize,
            )

            if state["include_inset"]:
                inset = axis.inset_axes([state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]])
                inset_values = self._inset_scores[metric]
                if state["relative"]:
                    inset_values = inset_values - inset_values[0]
                    inset.axhline(0.0, color="k", linewidth=0.5, linestyle="--")
                draw_subject_traces(inset, xvals, inset_values, PERFORMANCE_MODEL_COLORS, state, markersize=3.0)

                # Matplotlib's autoscale can leave the relative baseline almost flush with the
                # inset boundary when every other value is negative. Add explicit breathing room
                # before choosing ticks so the y=0 points remain fully visible.
                inset_ymin, inset_ymax = inset.get_ylim()
                inset_ypad = 0.08 * (inset_ymax - inset_ymin)
                inset.set_ylim(inset_ymin - inset_ypad, inset_ymax + inset_ypad)
                inset_ymin, inset_ymax = inset.get_ylim()
                inset_ytick_min = np.ceil(inset_ymin * 10) / 10
                inset_ytick_max = np.floor(inset_ymax * 10) / 10
                if inset_ytick_min <= inset_ytick_max:
                    inset_yticks = np.arange(inset_ytick_min, inset_ytick_max + 0.05, 0.1)
                else:
                    # Preserve visible ticks even when the entire inset spans less than 0.1.
                    inset_yticks = np.linspace(inset_ymin, inset_ymax, 3)
                style_model_axis(
                    inset,
                    fontsize=fontsize,
                    xvals=xvals,
                    labels=("",) * len(xvals),
                    xbounds=[xvals[0], xvals[-1]],
                    xtick_rotation=0,
                    yticks=inset_yticks,
                    xha="center",
                )
                inset.yaxis.set_visible(True)
                inset.tick_params(axis="y", which="both", left=True, labelleft=True)
                inset.tick_params(axis="x", which="major", length=2.0)
                inset.text(
                    state["placecells_x"],
                    state["placecells_y"],
                    "placecells",
                    transform=inset.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=fontsize,
                )

        # With a shared y-axis, both panels must contribute to autoscaling before format_spines
        # converts its fractional bottom-spine offset to data coordinates. Formatting ax[0] while
        # ax[1] is still empty leaves ax[0]'s spine at a stale y value.
        for axis_index, axis in enumerate(ax):
            shared_secondary = state["sharey"] and axis_index == 1
            ymin, ymax = axis.get_ylim()
            yticks = np.arange(np.fix(ymin * 10) / 10, np.fix(ymax * 10) / 10 + 0.05, 0.1)
            style_model_axis(
                axis,
                fontsize=fontsize,
                xvals=xvals,
                labels=PERFORMANCE_MODEL_LABELS,
                xbounds=[xvals[0], xvals[-1]],
                xtick_rotation=0,
                yticks=yticks,
                xha="center",
                spines_visible=("bottom",) if shared_secondary else ("left", "bottom"),
            )
            for tick_label, color in zip(axis.get_xticklabels(), PERFORMANCE_MODEL_COLORS):
                tick_label.set_color(color)
            if shared_secondary:
                axis.yaxis.set_visible(False)
            else:
                ylabel = r"$\Delta$ Residual RMS" if state["relative"] else "Residual RMS"
                axis.set_ylabel(ylabel, fontsize=fontsize)
        return fig


class ModelPlacefieldResidualFamiliarityViewer(FigureViewer):
    """Within- and outside-PF residual RMS over each mouse's chronological session number.

    Each model curve orders a mouse's sessions chronologically, renumbers them from 0, and
    averages the resulting curves across mice. With ``relative``, each model is first expressed
    relative to that same mouse-session's first-model value.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionPlacefieldResidualConfig`` results.
    model_names : list[ModelName]
        Models to draw, one curve each.
    normalized, quality, sharey, relative : bool
        As in :class:`ModelPlacefieldResidualViewer`.
    within_only : bool
        Draw only the within-place-field metric as a single-axis figure.
    auto_ylim : bool
        Fit the y limits and outer y ticks to the drawn data, ignoring ``ylim`` and
        ``ytick_max``.
    ylim : tuple[float, float]
        Y limits when ``auto_ylim`` is False, spanned by the offset left spine.
    ytick_max : float
        Upper y tick when ``auto_ylim`` is False (the lower one is 0).
    show_zero_line : bool
        Draw the dashed y = 0 reference line. Only applies when ``relative`` is True.
    fontsize : float
        Font size for axis labels, tick labels, and the legend.
    figsize : tuple[float, float]
        Figure size in inches.
    **curve_and_selection
        Curve-rendering knobs (``style``, ``se``, ``min_mice``, ``linewidth``,
        ``subject_linewidth``, ``subject_alpha``, ``fill_alpha``) and starting values for the
        data-selection widgets.
    """

    _CURVE_KNOBS = ("style", "se", "min_mice", "linewidth", "subject_linewidth", "subject_alpha", "fill_alpha")

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        model_names: list[ModelName] = PF_RESIDUAL_MODEL_NAMES,
        normalized: bool = False,
        quality: bool = True,
        sharey: bool = True,
        relative: bool = False,
        within_only: bool = False,
        auto_ylim: bool = True,
        ylim: tuple[float, float] = (0.0, 1.0),
        ytick_max: float = 1.0,
        show_zero_line: bool = True,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (7.0, 3.0),
        legend_options: dict | None = None,
        **curve_and_selection,
    ):
        self.results = results
        self.figsize = figsize
        self.model_names = list(model_names)
        self._scores: dict[str, np.ndarray] = {}

        curve_kwargs = {name: curve_and_selection.pop(name) for name in self._CURVE_KNOBS if name in curve_and_selection}
        self.selection_names = add_data_selection_widgets(self, results, skip=("model_name",), defaults=curve_and_selection)
        self.add_boolean("normalized", value=normalized)
        self.add_boolean("quality", value=quality)
        self.add_boolean("sharey", value=sharey)
        self.add_boolean("relative", value=relative)
        self.add_boolean("within_only", value=within_only)
        self.add_boolean("auto_ylim", value=auto_ylim)
        self.add_float_range("ylim", value=ylim, min=-2.0, max=5.0, step=0.01)
        self.add_float("ytick_max", value=ytick_max, min=0.01, max=5.0, step=0.01)
        self.add_boolean("show_zero_line", value=show_zero_line)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        add_familiarity_curve_widgets(self, results, **curve_kwargs)
        add_legend_widgets(self)
        update_legend_widgets(self, legend_options or {})

        for name in (*self.selection_names, "normalized", "quality"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Select session-level finite ROI means for every metric and model."""
        prefix = _residual_key_prefix(state["quality"], state["normalized"])
        selection = data_selection(state, self.results, self.selection_names)
        self._scores = {
            metric: np.stack(
                [
                    self.results.sel(model_name=model_name, keys=[prefix + metric], avg_by_mouse=False, **selection)[prefix + metric]
                    for model_name in self.model_names
                ],
                axis=0,
            )
            for metric in PF_RESIDUAL_METRICS
        }

    def plot(self, state):
        fontsize = state["fontsize"]
        metrics = PF_RESIDUAL_METRICS[:1] if state["within_only"] else PF_RESIDUAL_METRICS
        fig, axes = self.new_subplots(
            1,
            len(metrics),
            figsize=self.figsize,
            layout="constrained",
            sharey=state["sharey"] and len(metrics) > 1,
            squeeze=False,
        )
        axes = axes.ravel()

        extents = []
        visible_by_metric = []
        for axis, metric in zip(axes, metrics):
            scores = self._scores[metric]
            model_names = self.model_names
            if state["relative"]:
                scores = scores - scores[0]
                scores = scores[1:]  # Drop the first-model baseline from the relative curves.
                model_names = model_names[1:]  # Drop the first-model baseline from the relative curves.
                if state["show_zero_line"]:
                    axis.axhline(0.0, color="k", linewidth=0.5, linestyle="--")

            visible_values = []
            for model_name, values in zip(model_names, scores):
                extent, visible = draw_familiarity_series(
                    axis,
                    mouse_session_curves(self.results, values, single_env_only=False),
                    PF_RESIDUAL_FAMILIARITY_COLORS[model_name],
                    PF_RESIDUAL_FAMILIARITY_LABELS[model_name],
                    state,
                )
                extents.append(extent)
                visible_values.append(visible[np.isfinite(visible)])
            visible_by_metric.append(np.concatenate(visible_values) if visible_values else np.array([]))

        # Defer the spine styling until both panels have populated the shared y-axis; otherwise
        # format_spines' fractional bottom-spine offset is converted using stale shared limits.
        xmax = max(max(extents, default=1) - 1, 1)
        if state["sharey"] and len(metrics) > 1:
            finite_groups = [values for values in visible_by_metric if values.size]
            shared_values = np.concatenate(finite_groups) if finite_groups else np.array([])
            visible_by_metric = [shared_values] * len(metrics)

        for axis, metric, visible in zip(axes, metrics, visible_by_metric):
            if state["auto_ylim"] and visible.size:
                low, high = min(float(visible.min()), 0.0), max(float(visible.max()), 0.0)
                pad = 0.05 * (high - low) if high > low else 0.05
                axis_ylim = (low - pad, high + pad)
                if high > 0:
                    yticks = [0, round(high, 2)]
                else:
                    yticks = [round(low, 2), 0]
            else:
                axis_ylim = state["ylim"]
                yticks = [0, state["ytick_max"]]

            axis.set_ylim(axis_ylim)
            style_model_axis(
                axis,
                fontsize=fontsize,
                xbounds=[0, xmax],
                ybounds=axis_ylim,
                yticks=yticks,
            )
            axis.set_xlabel(SESSION_XLABEL, fontsize=fontsize)
            axis.set_ylabel(
                _residual_ylabel(metric, state["normalized"], state["relative"]),
                labelpad=-10,
                fontsize=fontsize,
            )
        apply_legend(axes[0], state, fontsize, auto_loc="lower right")
        return fig
