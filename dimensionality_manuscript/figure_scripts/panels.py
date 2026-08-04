"""Shared Syd widget bundles and axis styling for the manuscript figure panels.

Three kinds of thing live here, all figure-agnostic:

- :class:`FigureViewer`, a :class:`syd.Viewer` that closes only the figure it made last
  instead of every open figure.
- ``add_*_widgets`` helpers, each registering one recurring group of knobs on a viewer.
  They follow the same shape as :mod:`dimensionality_manuscript.figure_scripts.legends`:
  the caller passes its own defaults through, so a panel states each default exactly once.
- Drawing helpers that consume those widgets (:func:`draw_dot_legend`,
  :func:`draw_score_inset`) plus :func:`style_model_axis`, which packages the
  ``format_spines`` -> ``set_xticks`` -> ``tick_params`` sequence every categorical panel
  repeats.

Data-selection widgets are derived from a :class:`ResultsAggregator`'s ``param_axes``
rather than from hard-coded option lists, so a panel offers exactly the variations its
results were stored over. See :func:`add_data_selection_widgets`.
"""

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from syd import Viewer

from vrAnalysis.helpers.plotting import errorPlot, format_spines

# Param-axis value types a Syd selection can hold directly. ``None`` is included because several
# configs use it as a real option (e.g. ``smooth_width: [5.0, None]`` for "no smoothing"), not as
# a sentinel for "unset". Axes whose values are tuples (e.g. RRRToExternalLatentsConfig's
# model_ext_int_pair) need a label encoding, so the panel that uses them supplies its own widget
# and names the axis in ``skip``.
_SCALAR_TYPES = (str, int, float, bool, type(None))


def _config_value(results, name):
    """The value ``results``' config fixes for ``name``, or None if it defines no such field.

    An :class:`AnalysisConfigBase` declares a field for every parameter it takes, but
    ``_param_grid()`` sweeps only some of them -- ``RegressionConfig`` still has a
    ``spks_type`` field after dropping it from the grid. So the config, not the grid, is what
    says which value a panel is actually looking at.
    """
    return getattr(getattr(results, "config_class", None), name, None)


class FigureViewer(Viewer):
    """Viewer that closes only its own previous figure when it re-plots.

    Syd re-runs ``plot`` on every widget change, so a viewer that never closes anything
    leaks figures. Closing ``"all"`` is the other extreme: it also discards figures the
    notebook made elsewhere. Building through :meth:`new_figure` / :meth:`new_subplots`
    closes just the one this viewer made last.
    """

    _previous_figure = None

    def _close_previous(self) -> None:
        if self._previous_figure is not None:
            plt.close(self._previous_figure)
            self._previous_figure = None

    def new_figure(self, **kwargs):
        """``plt.figure(**kwargs)``, after closing this viewer's previous figure."""
        self._close_previous()
        fig = plt.figure(**kwargs)
        self._previous_figure = fig
        return fig

    def new_subplots(self, *args, **kwargs):
        """``plt.subplots(*args, **kwargs)``, after closing this viewer's previous figure."""
        self._close_previous()
        fig, ax = plt.subplots(*args, **kwargs)
        self._previous_figure = fig
        return fig, ax


# ======================================================================================
# Data selection (driven by ResultsAggregator.param_axes)
# ======================================================================================


def add_data_selection_widgets(viewer: Viewer, *results, skip=(), defaults=None, require=()) -> tuple[str, ...]:
    """Add one selection widget per param axis across ``results``, and return their names.

    The options come from each aggregator's ``param_axes``, so the widget offers exactly the
    values the results were stored over. Several aggregators can be passed at once (a panel
    that reads two configs); shared axis names are added once, and the returned names cover
    their union.

    An axis is left out when it is named in ``skip`` -- either because the panel fixes it
    (``model_name``, when the panel already knows which models it draws) or because its values
    aren't scalars and the panel supplies its own label-encoded widget. Non-scalar axes are
    skipped automatically for the same reason.

    ``require`` is for parameters a panel needs a concrete value for whether or not the results
    sweep them -- typically because it passes them to something other than ``results.sel``
    (``get_model_predictions``, ``registry.get_population``). ``RegressionConfig`` still has a
    ``spks_type`` field after dropping it from ``_param_grid()``, so an unswept name gets a
    widget pinned to the config's own value. The panel can then read ``state[name]`` and
    register ``on_change(name, ...)`` without knowing which parameters are currently swept.

    Each widget starts at ``defaults[name]`` if the caller gave one, else at the value the
    config fixes for it, else at the axis's first option.

    Parameters
    ----------
    viewer : Viewer
        Viewer to register the widgets on.
    *results : ResultsAggregator
        Aggregators whose ``param_axes`` define the axes and their options, and whose
        ``config_class`` supplies the starting values.
    skip : iterable of str
        Axis names to leave out. Takes precedence over ``require``.
    defaults : dict or None
        ``{axis: value}`` starting values, overriding the config's.
    require : iterable of str
        Names that must end up registered (see above). A required name that is neither swept,
        defined by a config, nor given a default raises.

    Returns
    -------
    tuple[str, ...]
        The widget names, in the order they were added -- pass these to
        :func:`data_selection` and to ``viewer.on_change`` for the panel's refresh callback.
    """
    defaults = defaults or {}
    names: list[str] = []

    def starting_value(name, options=None):
        """Caller default, else the config's value (when selectable), else the first option."""
        if name in defaults:
            return defaults[name]
        for aggregator in results:
            value = _config_value(aggregator, name)
            if value is not None and (options is None or value in options):
                return value
        return options[0] if options else None

    for aggregator in results:
        for axis, values in aggregator.param_axes.items():
            if axis in skip or axis in names:
                continue
            if not all(isinstance(value, _SCALAR_TYPES) for value in values):
                continue
            value = starting_value(axis, values)
            # A two-valued boolean axis reads better as a checkbox than as a True/False dropdown.
            if len(values) == 2 and all(isinstance(v, bool) for v in values):
                viewer.add_boolean(axis, value=bool(value))
            else:
                viewer.add_selection(axis, value=value, options=list(values))
            names.append(axis)

    for name in require:
        if name in names or name in skip:
            continue
        value = starting_value(name)
        if value is None:
            raise ValueError(f"{name!r} is required but is not swept, not defined by any config, and has no default")
        # Pinned to one option: the panel needs the value, but these results offer no choice.
        viewer.add_selection(name, value=value, options=[value])
        names.append(name)
    return tuple(names)


def data_selection(state, results, names, **extra) -> dict:
    """``results.sel`` kwargs for the data-selection widgets ``results`` actually has.

    Panels reading two aggregators share one set of widgets (see
    :func:`add_data_selection_widgets`), but the aggregators need not share every axis, so
    each axis is only forwarded to an aggregator that declares it.

    Parameters
    ----------
    state : dict
        Current viewer state.
    results : ResultsAggregator
        The aggregator the returned kwargs are destined for.
    names : iterable of str
        Widget names from :func:`add_data_selection_widgets`.
    **extra
        Axis values the panel fixes itself (e.g. ``model_name=...``), passed through as-is.

    Returns
    -------
    dict
        Keyword arguments for ``results.sel``.
    """
    selection = {name: state[name] for name in names if name in results.param_axes}
    selection.update(extra)
    return selection


# ======================================================================================
# Recurring widget bundles
# ======================================================================================


def add_trace_style_widgets(
    viewer: Viewer,
    *,
    markersize: float = 5.0,
    mean_linewidth: float = 1.5,
    subject_linewidth: float = 0.5,
    subject_alpha: float = 0.3,
) -> None:
    """Add the per-subject / across-subject trace style knobs shared by the score panels."""
    viewer.add_float("markersize", value=markersize, min=1.0, max=15.0)
    viewer.add_float("mean_linewidth", value=mean_linewidth, min=0.25, max=5.0)
    viewer.add_float("subject_linewidth", value=subject_linewidth, min=0.0, max=3.0)
    viewer.add_float("subject_alpha", value=subject_alpha, min=0.0, max=1.0)


def add_dot_legend_widgets(
    viewer: Viewer,
    *,
    show_legend: bool = True,
    legend_x: float = 5.0,
    legend_y: float = 0.085,
    legend_dy: float = -0.01,
    legend_markersize: float = 5.0,
    legend_text_dx: float = 0.25,
) -> None:
    """Add the hand-placed dot-legend knobs (positions are in the main axis's data coordinates).

    The legend entries explain marker colors, so they are drawn as the markers themselves
    rather than as proxy handles -- hence data coordinates rather than a matplotlib ``loc``.
    See :func:`draw_dot_legend`.
    """
    viewer.add_boolean("show_legend", value=show_legend)
    viewer.add_float("legend_x", value=legend_x, min=-1.0, max=12.0)
    viewer.add_float("legend_y", value=legend_y, min=-0.5, max=1.0, step=0.005)
    viewer.add_float("legend_dy", value=legend_dy, min=-0.1, max=0.1, step=0.005)
    viewer.add_float("legend_markersize", value=legend_markersize, min=1.0, max=15.0)
    viewer.add_float("legend_text_dx", value=legend_text_dx, min=0.0, max=1.0)


def draw_dot_legend(ax, state, entries, fontsize: float) -> None:
    """Draw the ``add_dot_legend_widgets`` legend: one colored dot and label per entry.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    state : dict
        Viewer state carrying the ``legend_*`` widgets.
    entries : iterable of tuple[str, str]
        ``(label, color)`` per entry, top to bottom.
    fontsize : float
        Label font size.
    """
    if not state["show_legend"]:
        return
    for i, (label, color) in enumerate(entries):
        y = state["legend_y"] + i * state["legend_dy"]
        ax.plot(state["legend_x"], y, color=color, marker="o", markersize=state["legend_markersize"], linestyle="none")
        # center_baseline, not center: "center" centers the text's full bbox, which sits
        # visibly high next to a dot for labels with no descender ("Ext", "Neural Latent").
        ax.text(
            state["legend_x"] + state["legend_text_dx"],
            y,
            label,
            color=color,
            ha="left",
            va="center_baseline",
            fontsize=fontsize,
        )


def add_score_inset_widgets(
    viewer: Viewer,
    *,
    show_inset: bool = True,
    inset_x: float = 0.1,
    inset_y: float = 0.5,
    inset_width: float = 0.35,
    inset_height: float = 0.45,
    inset_markersize: float = 3.0,
    inset_fontsize: float = 10.0,
    inset_ytick_max: float = 0.2,
    inset_ylim: tuple[float, float] = (-0.02, 0.25),
) -> None:
    """Add the knobs for an absolute-score inset drawn inside a relative-score main axis.

    The y knobs are in metric units (R^2 as a fraction), with an explicit step: syd rounds a
    value to its parameter's step, and the default 0.01 is coarse for an axis spanning ~0.1.
    """
    viewer.add_boolean("show_inset", value=show_inset)
    viewer.add_float("inset_x", value=inset_x, min=0.0, max=1.0)
    viewer.add_float("inset_y", value=inset_y, min=0.0, max=1.0)
    viewer.add_float("inset_width", value=inset_width, min=0.05, max=1.0)
    viewer.add_float("inset_height", value=inset_height, min=0.05, max=1.0)
    viewer.add_float("inset_markersize", value=inset_markersize, min=1.0, max=12.0)
    viewer.add_float("inset_fontsize", value=inset_fontsize, min=4.0, max=24.0)
    viewer.add_float("inset_ytick_max", value=inset_ytick_max, min=0.001, max=2.0, step=0.001)
    viewer.add_float_range("inset_ylim", value=inset_ylim, min=-1.0, max=2.0, step=0.001)


def draw_score_inset(ax, state, draw, ylabel: str):
    """Create the ``add_score_inset_widgets`` inset, let ``draw`` fill it, then style it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Main axis the inset is placed inside.
    state : dict
        Viewer state carrying the ``inset_*`` widgets.
    draw : callable
        Called as ``draw(inset)`` to plot into the new axis.
    ylabel : str
        Inset y label (the metric in its absolute units).

    Returns
    -------
    matplotlib.axes.Axes or None
        The inset, or None when ``show_inset`` is off.
    """
    if not state["show_inset"]:
        return None
    fontsize = state["inset_fontsize"]
    inset = ax.inset_axes([state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]])
    draw(inset)
    inset.set_ylim(state["inset_ylim"])
    format_spines(
        inset,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left"],
        ybounds=state["inset_ylim"],
        yticks=[0, state["inset_ytick_max"]],
        tick_fontsize=fontsize,
    )
    inset.set_xticks([])
    inset.tick_params(axis="y", which="both", labelsize=fontsize)
    inset.set_ylabel(ylabel, labelpad=-10, fontsize=fontsize)
    return inset


# ======================================================================================
# Curve groups (a set of same-shaped curves drawn as one visual unit)
# ======================================================================================


def render_curve_group(
    ax,
    xvals: np.ndarray,
    data: np.ndarray,
    color,
    plot_style: str,
    *,
    label: str | None = None,
    hide_error: bool = False,
    min_support: int = 1,
    line_alpha: float = 0.3,
    linewidth: float = 2.0,
) -> None:
    """Render one group of ``(n_series, n_x)`` curves as individual+mean lines or an errorPlot band.

    ``"each"`` draws every row thin plus a solid mean; ``"errorPlot"`` draws mean +/- std as a
    band, or just the mean when ``hide_error``. Columns with fewer than ``min_support`` finite
    rows are dropped from the mean/band (not from the thin lines) -- a "mean" backed by one
    series is that series, not an average.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    xvals : np.ndarray
        Shared x values, one per column of ``data``.
    data : np.ndarray
        ``(n_series, n_x)`` curves. NaN marks a series that has no value at that x.
    color : color-like
        Color of the whole group.
    plot_style : {"each", "errorPlot"}
    label : str or None
        Legend label of the group's mean line / band.
    hide_error : bool
        With ``"errorPlot"``, draw the mean without the band.
    min_support : int
        Minimum number of finite series a column needs to contribute to the mean/band.
    line_alpha : float
        Opacity of the individual series under ``"each"``.
    linewidth : float
        Line width of the mean curve.
    """
    valid_count = np.sum(np.isfinite(data), axis=0)
    value_sum = np.nansum(data, axis=0)
    mean_curve = np.divide(value_sum, valid_count, out=np.full_like(value_sum, np.nan, dtype=float), where=valid_count > 0)
    mean_curve[valid_count < min_support] = np.nan

    if plot_style == "each":
        ax.plot(xvals, data.T, color=(color, line_alpha), linewidth=0.5)
        ax.plot(xvals, mean_curve, color=color, linewidth=linewidth, label=label)
    elif plot_style == "errorPlot":
        if hide_error:
            ax.plot(xvals, mean_curve, color=color, linewidth=linewidth, label=label)
        else:
            masked = np.where(valid_count[None, :] >= min_support, data, np.nan)
            errorPlot(xvals, masked, axis=0, ax=ax, color=color, linewidth=linewidth, alpha=0.2, label=label)
    else:
        raise ValueError(f"Unknown plot_style {plot_style!r}. Options: ['each', 'errorPlot']")


def curve_group_bounds(data: np.ndarray, plot_style: str) -> tuple[float, float] | None:
    """Visible ``(low, high)`` of a group drawn by :func:`render_curve_group`, or None if empty.

    Under ``"errorPlot"`` it is the band (mean +/- std) that is on screen, not the individual
    series, so the two styles do not have the same extent for the same data.
    """
    if plot_style == "each":
        visible = data
    else:
        spread = np.nanstd(data, axis=0)
        mean_curve = np.nanmean(data, axis=0)
        visible = np.stack([mean_curve - spread, mean_curve + spread])
    visible = visible[np.isfinite(visible)]
    if not visible.size:
        return None
    return float(np.min(visible)), float(np.max(visible))


def draw_session_colorbar_inset(ax, first_label, last_label, cmap_name: str, fontsize: float, bounds) -> None:
    """Draw a small horizontal "session #" colorbar inset on ``ax``.

    ``bounds`` is ``[x, y, width, height]`` in axes-fraction coordinates; values outside
    ``[0, 1]`` place the bar outside the axes.
    """
    cmap_data = mpl.colormaps[cmap_name](np.linspace(0, 1, 255))[np.newaxis, :, :]
    inset = ax.inset_axes(list(bounds))
    inset.imshow(cmap_data, aspect="auto")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.text(0.02, 0.5, f"{first_label}", transform=inset.transAxes, ha="left", va="center", color="white", fontsize=fontsize)
    inset.text(0.98, 0.5, f"{last_label}", transform=inset.transAxes, ha="right", va="center", color="white", fontsize=fontsize)
    inset.text(0.5, 0.5, "session #", transform=inset.transAxes, ha="center", va="center", color="black", fontsize=fontsize)


# ======================================================================================
# Axis styling
# ======================================================================================


def style_model_axis(
    ax,
    *,
    fontsize: float,
    xvals=None,
    labels=None,
    xbounds=None,
    ybounds=None,
    yticks=None,
    xticks=None,
    xtick_rotation: float = 45.0,
    spines_visible=("left", "bottom"),
    x_pos: float = -0.02,
    y_pos: float = -0.02,
    xha: str = "right",
) -> None:
    """Offset the spines, set rotated categorical x tick labels, and size every tick label.

    Packages the sequence every categorical panel repeats. The ordering matters:
    ``format_spines`` positions the offset spines from the axis's *current* limits (so the
    caller must fix ``set_ylim`` first) and calls ``tick_params`` itself, so the font size is
    applied afterwards -- including minor ticks, which log axes rely on.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    fontsize : float
        Font size for every tick label on the axis.
    xvals, labels : array-like or None
        Tick positions and their labels. Both must be given to set categorical ticks; pass
        neither to leave the x ticks alone (``xticks`` still applies).
    xbounds, ybounds : tuple or None
        Data range each offset spine spans.
    yticks, xticks : list or None
        Explicit tick positions, forwarded to ``format_spines``.
    xtick_rotation : float
        Rotation of the categorical tick labels, in degrees.
    spines_visible : iterable of str
        Which spines to draw.
    x_pos, y_pos : float
        Fractional spine offsets, forwarded to ``format_spines``.
    xha : str
        Horizontal alignment of the categorical tick labels, forwarded to ``set_xticks``.
    """
    format_spines(
        ax,
        x_pos=x_pos,
        y_pos=y_pos,
        spines_visible=list(spines_visible),
        xbounds=xbounds,
        ybounds=ybounds,
        xticks=xticks,
        yticks=yticks,
        tick_fontsize=fontsize,
    )
    if xvals is not None and labels is not None:
        ax.set_xticks(
            xvals,
            labels=list(labels),
            rotation=xtick_rotation,
            rotation_mode="anchor",
            ha=xha,
            fontsize=fontsize,
        )
    ax.tick_params(axis="both", which="both", labelsize=fontsize)
