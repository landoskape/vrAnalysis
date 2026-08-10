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
PF_RESIDUAL_REGIONS = ("within", "outside")
# Which membership place field localized the residual: ``xval`` measured it on the training
# frames, ``infold`` on the held-out frames themselves.
PF_RESIDUAL_FOLDS = ("xval", "infold")
PF_RESIDUAL_METRIC_OPTIONS = ("rms", "normalized_rms", "r2_weighted", "r2_shared")
# Across-ROI reduction. These distributions are right-skewed, so mean and median differ noticeably
# for every metric; for the R² metrics the mean is unusable and "median" is the one to read.
PF_RESIDUAL_STATISTICS = ("mean", "median")
PF_RESIDUAL_SUBSETS = ("all", "quality", "not quality")
PF_RESIDUAL_SUBSET_LABELS = {
    "all": None,
    "quality": "placecells",
    "not quality": "non placecells",
}
PF_RESIDUAL_REGION_LABELS = {
    "within": "Within-PF",
    "outside": "Outside-PF",
}
# Lowercase forms compose after a region name ("Within-PF residual RMS").
PF_RESIDUAL_METRIC_LABELS = {
    "rms": "residual RMS",
    "normalized_rms": "normalized residual RMS",
    "r2_weighted": "weighted $R^2$",
    "r2_shared": "shared-variance $R^2$",
}
# Standalone forms label an axis whose region is named by panel text instead.
PF_RESIDUAL_METRIC_AXIS_LABELS = {
    "rms": "Residual RMS",
    "normalized_rms": "Normalized residual RMS",
    "r2_weighted": "Weighted $R^2$",
    "r2_shared": "Shared-variance $R^2$",
}
# Two-line variants for the cramped inset axes.
PF_RESIDUAL_METRIC_SHORT_LABELS = {
    "rms": "Res.\nRMS",
    "normalized_rms": "Norm.\nRes. RMS",
    "r2_weighted": "Weighted\n$R^2$",
    "r2_shared": "Shared\n$R^2$",
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


def _residual_key_prefix(subset: bool | str, fold: str, statistic: str = "mean") -> str:
    """Result-key prefix for the requested precomputed scalar summary."""
    if fold not in PF_RESIDUAL_FOLDS:
        raise ValueError(f"Unknown residual fold: {fold!r}")
    if statistic not in PF_RESIDUAL_STATISTICS:
        raise ValueError(f"Unknown residual statistic: {statistic!r}")
    prefix = f"{statistic}_"
    # Keep accepting the familiarity viewer's boolean quality selection while the
    # summary viewer exposes all three available ROI subsets.
    if subset is True or subset == "quality":
        prefix += "quality_filtered_"
    elif subset == "not quality":
        prefix += "notquality_filtered_"
    elif subset is not False and subset != "all":
        raise ValueError(f"Unknown residual subset: {subset!r}")
    return f"{prefix}{fold}_"


def _residual_metric_key(region: str, metric: str) -> str:
    """Result-key suffix naming one region's metric, e.g. ``within_pf_r2_weighted``."""
    return f"{region}_pf_{metric}"


def _residual_ylabel(region: str, metric: str, relative: bool) -> str:
    """Y label for one region's metric under the current display options."""
    label = f"{PF_RESIDUAL_REGION_LABELS[region]} {PF_RESIDUAL_METRIC_LABELS[metric]}"
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
    fold : {"xval", "infold"}
        Membership place field that localized the residual: estimated on the training frames
        (``xval``) or on the held-out frames themselves (``infold``).
    metric : {"rms", "normalized_rms", "r2_weighted", "r2_shared"}
        Residual metric to draw. ``normalized_rms`` divides by each ROI's held-out target standard
        deviation; ``r2_weighted`` measures against the weighted mean of the target, so its within-
        and outside-field values do not share a baseline; ``r2_shared`` normalizes by the ROI's
        total held-out variance and is exactly ``1 - normalized_rms**2``.
    statistic : {"mean", "median"}
        Across-ROI reduction of the per-session values. These distributions are right-skewed, so
        the two differ noticeably for every metric; for the R² metrics the mean is unusable (an ROI
        with near-zero within-field variance produces an R² in the thousands) and the median is the
        one to read.
    main_show, inset_show : {"all", "quality", "not quality"}
        ROI subset displayed in the main panels and inset panels, respectively. ``inset_show``
        has no visual effect when ``include_inset`` is False.
    main_sharey, inset_sharey : bool
        Give the two main panels or the two inset panels, respectively, a common y-axis scale.
    main_relative, inset_relative : bool
        Subtract each mouse's first-model value from all of that mouse's values in the main or
        inset panels, respectively.
    sharey, relative : bool, optional
        Backward-compatible aliases. ``sharey`` sets ``main_sharey``; ``relative`` sets both
        relative controls unless the corresponding new keyword is supplied explicitly.
    fontsize, markersize, mean_linewidth, subject_linewidth, subject_alpha : float
        Style knobs for the per-mouse traces and the across-mouse mean trace and dots.
    within_text_x, within_text_y, outside_text_x, outside_text_y : float
        Axes-fraction coordinates for the ``Within PF`` and ``Outside PF`` panel text.
    inset_x, inset_y, inset_width, inset_height : float
        Shared axes-fraction bounds for the inset in each panel.
    placecells_x, placecells_y : float
        Shared axes-fraction position for the inset's ROI-subset label.
    inset_ylabel_fontsize_scale : float
        Inset y-label font size as a multiple of ``fontsize``.
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
        fold: str = "xval",
        metric: str = "rms",
        statistic: str = "mean",
        main_show: str = "all",
        inset_show: str = "quality",
        main_sharey: bool | None = None,
        inset_sharey: bool = False,
        main_relative: bool | None = None,
        inset_relative: bool | None = None,
        sharey: bool | None = None,
        relative: bool | None = None,
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
        inset_ylabel_fontsize_scale: float = 1.0,
        figsize: tuple[float, float] = (7.0, 3.5),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.model_names = list(model_names)
        self._scores: dict[str, np.ndarray] = {}
        self._inset_scores: dict[str, np.ndarray] = {}

        if main_sharey is None:
            main_sharey = True if sharey is None else sharey
        if main_relative is None:
            main_relative = False if relative is None else relative
        if inset_relative is None:
            inset_relative = False if relative is None else relative

        self.selection_names = add_data_selection_widgets(self, results, skip=("model_name",), defaults=selection_defaults)
        self.add_selection("fold", value=fold, options=list(PF_RESIDUAL_FOLDS))
        self.add_selection("metric", value=metric, options=list(PF_RESIDUAL_METRIC_OPTIONS))
        self.add_selection("statistic", value=statistic, options=list(PF_RESIDUAL_STATISTICS))
        self.add_selection("main_show", value=main_show, options=list(PF_RESIDUAL_SUBSETS))
        self.add_selection("inset_show", value=inset_show, options=list(PF_RESIDUAL_SUBSETS))
        self.add_boolean("main_sharey", value=main_sharey)
        self.add_boolean("inset_sharey", value=inset_sharey)
        self.add_boolean("main_relative", value=main_relative)
        self.add_boolean("inset_relative", value=inset_relative)
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
        self.add_float("inset_ylabel_fontsize_scale", value=inset_ylabel_fontsize_scale, min=0.1, max=3.0, step=0.05)
        self.add_boolean("include_inset", value=include_inset)
        for name in (*self.selection_names, "fold", "metric", "statistic", "main_show", "inset_show"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Load the selected main and inset ROI subsets, averaged within mouse."""
        selection = data_selection(state, self.results, self.selection_names)

        def load_scores(subset):
            prefix = _residual_key_prefix(subset, state["fold"], state["statistic"])
            scores = {}
            for region in PF_RESIDUAL_REGIONS:
                key = prefix + _residual_metric_key(region, state["metric"])
                scores[region] = np.stack(
                    [
                        self.results.sel(model_name=model_name, keys=[key], avg_by_mouse=True, **selection)[key]
                        for model_name in self.model_names
                    ],
                    axis=0,
                )
            return scores

        self._scores = load_scores(state["main_show"])
        self._inset_scores = load_scores(state["inset_show"])

    def plot(self, state):
        fontsize = state["fontsize"]
        xvals = np.arange(len(self.model_names), dtype=float)

        fig, ax = self.new_subplots(1, 2, figsize=self.figsize, layout="constrained", sharey=state["main_sharey"])
        insets = []
        for axis, region, text, text_x, text_y in zip(
            ax,
            PF_RESIDUAL_REGIONS,
            ("Within PF", "Outside PF"),
            (state["within_text_x"], state["outside_text_x"]),
            (state["within_text_y"], state["outside_text_y"]),
        ):
            values = self._scores[region]
            if state["main_relative"]:
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
                inset = axis.inset_axes(
                    [state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]],
                    sharey=insets[0] if state["inset_sharey"] and insets else None,
                )
                insets.append(inset)
                inset_values = self._inset_scores[region]
                if state["inset_relative"]:
                    inset_values = inset_values - inset_values[0]
                    inset.axhline(0.0, color="k", linewidth=0.5, linestyle="--")
                draw_subject_traces(inset, xvals, inset_values, PERFORMANCE_MODEL_COLORS, state, markersize=3.0)
                inset_label = PF_RESIDUAL_SUBSET_LABELS[state["inset_show"]]
                if inset_label is not None:
                    inset.text(
                        state["placecells_x"],
                        state["placecells_y"],
                        inset_label,
                        transform=inset.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=fontsize,
                    )

        # Style the inset pair only after both have contributed to a potentially shared y-axis.
        # Otherwise the first inset's spine and ticks are based on stale limits.
        if insets:
            # Matplotlib's autoscale can leave the relative baseline almost flush with the
            # inset boundary when every other value is negative. Add explicit breathing room
            # before choosing ticks so the y=0 points remain fully visible. A shared pair only
            # needs this once because changing either member changes both.
            inset_groups = insets[:1] if state["inset_sharey"] else insets
            for inset in inset_groups:
                inset_ymin, inset_ymax = inset.get_ylim()
                inset_ypad = 0.08 * (inset_ymax - inset_ymin)
                inset.set_ylim(inset_ymin - inset_ypad, inset_ymax + inset_ypad)

            # Always retain the zero reference in view, for both absolute and relative scores.
            for inset in inset_groups:
                inset_ymin, inset_ymax = inset.get_ylim()
                inset.set_ylim(min(inset_ymin, 0.0), max(inset_ymax, 0.0))

        for inset in insets:
            inset_ymin, inset_ymax = inset.get_ylim()
            inset_ytick_step = 0.1 if state["inset_relative"] else 1.0
            inset_ytick_min = np.ceil(inset_ymin / inset_ytick_step) * inset_ytick_step
            inset_ytick_max = np.floor(inset_ymax / inset_ytick_step) * inset_ytick_step
            if inset_ytick_min <= inset_ytick_max:
                inset_yticks = np.arange(
                    inset_ytick_min,
                    inset_ytick_max + inset_ytick_step / 2,
                    inset_ytick_step,
                )
            else:
                # Preserve visible ticks even when the entire inset spans less than one step.
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
            inset.tick_params(axis="x", which="major", length=2.0)
            # Keep both inset y-axes visible even when their limits are shared. Unlike the main
            # panels, the insets are visually separated enough that suppressing the second axis
            # makes their shared scale hard to recognize.
            inset.yaxis.set_visible(True)
            inset.tick_params(axis="y", which="both", left=True, labelleft=True, length=2.0)
            inset_ylabel = PF_RESIDUAL_METRIC_SHORT_LABELS[state["metric"]]
            if state["inset_relative"]:
                inset_ylabel = f"$\\Delta$ {inset_ylabel}"
            inset.set_ylabel(
                inset_ylabel,
                fontsize=fontsize * state["inset_ylabel_fontsize_scale"],
            )

        # With a shared y-axis, both panels must contribute to autoscaling before format_spines
        # converts its fractional bottom-spine offset to data coordinates. Formatting ax[0] while
        # ax[1] is still empty leaves ax[0]'s spine at a stale y value.
        main_groups = ax[:1] if state["main_sharey"] else ax
        for axis in main_groups:
            ymin, ymax = axis.get_ylim()
            axis.set_ylim(min(ymin, 0.0), max(ymax, 0.0))

        for axis_index, axis in enumerate(ax):
            shared_secondary = state["main_sharey"] and axis_index == 1
            ymin, ymax = axis.get_ylim()
            ytick_step = 0.1 if state["main_relative"] else 1.0
            ytick_min = np.ceil(ymin / ytick_step) * ytick_step
            ytick_max = np.floor(ymax / ytick_step) * ytick_step
            if ytick_min <= ytick_max:
                yticks = np.arange(ytick_min, ytick_max + ytick_step / 2, ytick_step)
            else:
                yticks = np.linspace(ymin, ymax, 3)
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
                ylabel = PF_RESIDUAL_METRIC_AXIS_LABELS[state["metric"]]
                if state["main_relative"]:
                    ylabel = rf"$\Delta$ {ylabel}"
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
    fold, metric, statistic : str
        As in :class:`ModelPlacefieldResidualViewer`.
    quality, sharey, relative : bool
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
        fold: str = "xval",
        metric: str = "rms",
        statistic: str = "mean",
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
        self.add_selection("fold", value=fold, options=list(PF_RESIDUAL_FOLDS))
        self.add_selection("metric", value=metric, options=list(PF_RESIDUAL_METRIC_OPTIONS))
        self.add_selection("statistic", value=statistic, options=list(PF_RESIDUAL_STATISTICS))
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

        for name in (*self.selection_names, "fold", "metric", "statistic", "quality"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Select session-level finite ROI means for every region and model."""
        prefix = _residual_key_prefix(state["quality"], state["fold"], state["statistic"])
        selection = data_selection(state, self.results, self.selection_names)
        self._scores = {}
        for region in PF_RESIDUAL_REGIONS:
            key = prefix + _residual_metric_key(region, state["metric"])
            self._scores[region] = np.stack(
                [
                    self.results.sel(model_name=model_name, keys=[key], avg_by_mouse=False, **selection)[key]
                    for model_name in self.model_names
                ],
                axis=0,
            )

    def plot(self, state):
        fontsize = state["fontsize"]
        regions = PF_RESIDUAL_REGIONS[:1] if state["within_only"] else PF_RESIDUAL_REGIONS
        fig, axes = self.new_subplots(
            1,
            len(regions),
            figsize=self.figsize,
            layout="constrained",
            sharey=state["sharey"] and len(regions) > 1,
            squeeze=False,
        )
        axes = axes.ravel()

        extents = []
        visible_by_region = []
        for axis, region in zip(axes, regions):
            scores = self._scores[region]
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
            visible_by_region.append(np.concatenate(visible_values) if visible_values else np.array([]))

        # Defer the spine styling until both panels have populated the shared y-axis; otherwise
        # format_spines' fractional bottom-spine offset is converted using stale shared limits.
        xmax = max(max(extents, default=1) - 1, 1)
        if state["sharey"] and len(regions) > 1:
            finite_groups = [values for values in visible_by_region if values.size]
            shared_values = np.concatenate(finite_groups) if finite_groups else np.array([])
            visible_by_region = [shared_values] * len(regions)

        for axis, region, visible in zip(axes, regions, visible_by_region):
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
                _residual_ylabel(region, state["metric"], state["relative"]),
                labelpad=-10,
                fontsize=fontsize,
            )
        apply_legend(axes[0], state, fontsize, auto_loc="lower right")
        return fig
