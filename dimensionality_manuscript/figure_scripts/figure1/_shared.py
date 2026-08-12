"""Cross-panel helpers for figure 1: colorscales, axis styling, and session-curve bundles.

Everything here is used by more than one panel of this figure. Helpers that are genuinely
figure-agnostic (``render_curve_group``, ``draw_session_colorbar_inset``, the data-selection
widgets) live in :mod:`dimensionality_manuscript.figure_scripts.panels` instead.
"""

import matplotlib as mpl
import numpy as np
import seaborn as sns
from syd import Viewer

from vrAnalysis.helpers.plotting import format_spines

from dimensionality_manuscript.env_order import ENV_SLOT_COLORS
from dimensionality_manuscript.figure_scripts.panels import (
    curve_group_bounds,
    draw_session_colorbar_inset,
    render_curve_group,
)

# Axes-fraction rectangle (x0, y0, width, height) of a horizontal colorscale inset.
COLORSCALE_INSET_RECT: tuple[float, float, float, float] = (0.72, 0.10, 0.25, 0.2)


def env_slot_color(slot: int):
    """Experience-slot color, shared by the VR schematic, the timeline, and the ``by_env`` panels.

    Environments are indexed by the order the mouse first saw them, not by environment id: the
    two cohorts run disjoint environment indices, so only the slot is comparable across them.
    """
    return ENV_SLOT_COLORS[int(slot) % len(ENV_SLOT_COLORS)]


# ======================================================================================
# Axis styling
# ======================================================================================


def style_axis(ax, *, fontsize: float, spines_visible=("left", "bottom"), x_pos: float = -0.02, y_pos: float = -0.02, **kwargs) -> None:
    """``format_spines`` with this figure's offsets and tick length, sized to ``fontsize``.

    Every panel of figure 1 offsets its spines by the same fraction and uses the same tick
    length; stating that once here keeps the call sites down to what actually differs (the
    bounds, ticks, and which spines are drawn).

    ``format_spines`` positions the offset spines from the axis's *current* limits, so
    ``set_xlim`` / ``set_ylim`` must already have been called.
    """
    format_spines(
        ax,
        x_pos=x_pos,
        y_pos=y_pos,
        spines_visible=list(spines_visible),
        tick_length=2,
        tick_fontsize=fontsize,
        **kwargs,
    )


def hide_spines(ax) -> None:
    """Strip every spine and tick from an axes -- the bare-image panels of this figure."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def round_up_labels(ticks) -> list[str]:
    """Tick labels rounded up to a single decimal place (``-0.0`` is normalized to ``0.0``)."""
    labels = []
    for tick in ticks:
        value = float(np.ceil(float(tick) * 10) / 10)
        labels.append(f"{value if value != 0 else 0.0:.1f}")
    return labels


def decimal_yticks(low: float, high: float, step: float = 0.1) -> list[float]:
    """Multiples of ``step`` inside ``[low, high]``.

    Used instead of matplotlib's locator so several panels of the same quantity tick at exactly
    the same values, and so 0 is always ticked whenever it is on the axis.
    """
    # Nudged tolerances: the bounds are usually themselves arithmetic on multiples of the step,
    # so an exact endpoint can land a hair outside it in floating point.
    first = int(np.ceil(low / step - 1e-9))
    last = int(np.floor(high / step + 1e-9))
    return [round(index * step, 10) for index in range(first, last + 1)]


def fit_square_panels(fig, axs, tolerance: float = 0.005, max_iter: int = 4) -> None:
    """
    Resize a figure's height until its panels come out square, leaving the width alone.

    The panels of a constrained-layout figure are whatever is left of it once the decorations
    (tick labels, axis labels, layout padding, the row's own margins) have taken their share, and
    that share is set in points: it does not move when the figure is resized. So the height the
    panels are short of square is exactly the height the *figure* is short of square, and one pass
    lands on it -- the loop only exists to catch a decoration that does rescale (a legend placed in
    figure coordinates, say).

    Only equal-width panels can all be square, since a row of axes shares one height. Call this on
    a figure built with equal ``width_ratios``: constrained layout equalizes the margins across the
    columns of a gridspec, so equal ratios give equal panel widths even when the panels carry
    different decorations. With unequal ratios the widest panel is made square and the rest end up
    taller than they are wide.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Fully drawn figure -- everything that takes up space must already be on it.
    axs : sequence of matplotlib.axes.Axes
        The panels to square up.
    tolerance : float
        Stop once every panel is within this many inches of square.
    max_iter : int
        Give up after this many resizes.
    """
    for _ in range(max_iter):
        # Runs the layout engine without rasterizing, which is what puts the axes where the
        # decorations leave room for them.
        fig.draw_without_rendering()
        width, height = fig.get_size_inches()
        positions = [ax.get_position() for ax in axs]
        deficit = max(pos.width * width - pos.height * height for pos in positions)
        if abs(deficit) < tolerance:
            return
        fig.set_figheight(height + deficit)


# ======================================================================================
# Colorscales
# ======================================================================================


def add_colorscale_widgets(
    viewer: Viewer,
    *,
    colorscale_text_y: float = 0.5,
    colorscale_inset_rect: tuple[float, float, float, float] = COLORSCALE_INSET_RECT,
) -> None:
    """Add the placement knobs of a horizontal colorscale inset (see :func:`draw_colorscale_inset`).

    The steps are 0.001 rather than syd's default 0.01: these are axes fractions on a panel
    that is usually an inch or two wide, where 0.01 is a visible jump.
    """
    viewer.add_float("colorscale_text_y", value=colorscale_text_y, min=0.0, max=1.0, step=0.001)
    viewer.add_float("colorscale_inset_x", value=colorscale_inset_rect[0], min=0.0, max=1.0, step=0.001)
    viewer.add_float("colorscale_inset_y", value=colorscale_inset_rect[1], min=0.0, max=1.0, step=0.001)
    viewer.add_float("colorscale_inset_width", value=colorscale_inset_rect[2], min=0.001, max=1.0, step=0.001)
    viewer.add_float("colorscale_inset_height", value=colorscale_inset_rect[3], min=0.001, max=1.0, step=0.001)


def colorscale_inset_rect(state) -> list[float]:
    """``[x, y, width, height]`` from the :func:`add_colorscale_widgets` knobs."""
    return [
        state["colorscale_inset_x"],
        state["colorscale_inset_y"],
        state["colorscale_inset_width"],
        state["colorscale_inset_height"],
    ]


def draw_colorscale_inset(
    ax,
    cmap_name: str,
    *,
    right_label: str,
    left_label: str | None = None,
    right_color="w",
    left_color="k",
    fontsize: float = 8.0,
    text_y: float = 0.5,
    inset_rect=COLORSCALE_INSET_RECT,
) -> None:
    """Add a horizontal colorscale inset to an axes, labeled at its two ends.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to place the inset on.
    cmap_name : str
        Name of the colormap to sample (255 colors, rendered as a (1, 255, 4) image).
    right_label : str
        Text drawn at the right (high) end of the colorscale.
    left_label : str or None
        Text drawn at the left (low) end. If None, no left label is drawn.
    right_color, left_color : color-like
        Text colors for the right and left labels.
    fontsize : float
        Font size of the end labels.
    text_y : float
        Vertical position of the end labels in inset-axes coordinates.
    inset_rect : tuple of float
        ``(x, y, width, height)`` of the colorscale in parent-axes coordinates.
    """
    colors = mpl.colormaps[cmap_name](np.linspace(0, 1, 255))[None, :, :]  # (1, 255, 4)
    axins = ax.inset_axes(list(inset_rect))
    axins.imshow(colors, aspect="auto")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.text(0.97, text_y, right_label, transform=axins.transAxes, ha="right", va="center", color=right_color, fontsize=fontsize)
    if left_label is not None:
        axins.text(0.03, text_y, left_label, transform=axins.transAxes, ha="left", va="center", color=left_color, fontsize=fontsize)


def draw_vertical_colorscale(
    ax,
    cmap_name: str | mpl.colors.Colormap,
    *,
    low_label: str,
    high_label: str,
    fontsize: float,
    ylabel: str | None = None,
    center_label: str | None = None,
    low_color="k",
    high_color="w",
) -> None:
    """Fill ``ax`` with a vertical colorscale, labeled at its two ends.

    Used both on an ``inset_axes`` of a raster and as a figure's own leading column. Pass
    ``center_label`` instead of ``ylabel`` when the strip is wide enough to carry its name
    inside it, which is what a standalone column usually wants.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes the strip fills entirely.
    cmap_name : str or matplotlib.colors.Colormap
        Colormap name or object to sample (255 colors, low at the bottom).
    low_label, high_label : str
        Text at the bottom and top of the strip.
    fontsize : float
        Font size of every label.
    ylabel : str or None
        Axis label naming the quantity, drawn beside the strip.
    center_label : str or None
        Same, but rotated inside the strip.
    low_color, high_color : color-like
        Text colors of the two end labels, chosen against the colormap's own ends.
    """
    cmap = mpl.colormaps[cmap_name] if isinstance(cmap_name, str) else cmap_name
    colors = cmap(np.linspace(0, 1, 255))[:, None, :]  # (255, 1, 4)
    ax.imshow(colors, aspect="auto", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.02, low_label, transform=ax.transAxes, ha="center", va="bottom", color=low_color, fontsize=fontsize)
    ax.text(0.5, 0.98, high_label, transform=ax.transAxes, ha="center", va="top", color=high_color, fontsize=fontsize)
    if center_label is not None:
        ax.text(0.5, 0.5, center_label, transform=ax.transAxes, ha="center", va="center", rotation=90, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ======================================================================================
# The example-ROI scatter panel (data vs place-field prediction)
# ======================================================================================


def draw_example_roi_panel(
    ax,
    data: np.ndarray,
    prediction: np.ndarray,
    *,
    style: str = "points",
    alpha: float = 0.1,
    fontsize: float = 7.0,
    ylabel: str = "PF",
) -> None:
    """
    Draw one ROI's activity against its place-field prediction, with a unity line.

    Both axes run from 0 to the larger of the two maxima and are ticked only at the ends. The
    units are arbitrary, so the upper tick is labeled ``1`` rather than a number that would eat
    horizontal space -- this panel is usually small.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data, prediction : np.ndarray
        The ROI's measured and predicted activity, frame by frame.
    style : {"points", "density"}
        A translucent point cloud, or a filled 2D KDE of the same points.
    alpha : float
        Opacity of the point cloud. Ignored by ``"density"``.
    fontsize : float
        Font size of the axis labels and tick labels.
    ylabel : str
        Label of the prediction axis. Short by default, since the panel usually is.
    """
    axmax = float(np.max([np.nanmax(data), np.nanmax(prediction)]))
    if style == "points":
        ax.plot(
            data,
            prediction,
            markerfacecolor="k",
            markeredgecolor="none",
            marker=".",
            markersize=2.5,
            linestyle="none",
            alpha=alpha,
        )
    elif style == "density":
        valid = np.isfinite(data) & np.isfinite(prediction)
        sns.kdeplot(
            x=data[valid],
            y=prediction[valid],
            ax=ax,
            fill=True,
            levels=8,
            logscale=True,
            thresh=0.05,
            cmap="Greys",
            zorder=1,
        )
    else:
        raise ValueError(f"Invalid style: {style!r}")
    ax.plot([0, axmax], [0, axmax], color="0.6", linestyle="--", linewidth=0.75, zorder=5)
    ax.set_xlabel("Observed Activity", fontsize=fontsize, labelpad=-8)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=-8)
    style_axis(
        ax,
        fontsize=fontsize,
        x_pos=-0.05,
        y_pos=-0.05,
        xbounds=[0, axmax],
        ybounds=[0, axmax],
        xticks=[0, axmax],
        yticks=[0, axmax],
        xlabels=["0", "1"],
        ylabels=["0", "1"],
    )


# ======================================================================================
# Population curves indexed by session number
# ======================================================================================


def add_session_curve_widgets(
    viewer: Viewer,
    *,
    num_sessions: int,
    curve_mode: str = "by_session",
    plot_style: str = "each",
    hide_error: bool = False,
    skip_sessions: int = 0,
    inset_position: tuple[float, float, float, float] = (0.35, 0.85, 0.6, 0.075),
) -> None:
    """Add the knobs of a population panel that groups every mouse's curves by session number.

    The panel either averages each mouse's sessions into one curve per mouse
    (``curve_mode="average"``) or draws one group per session number
    (``"by_session"``), colored by session number with a colorbar inset. See
    :func:`draw_session_curve_groups`.
    """
    viewer.add_selection("curve_mode", value=curve_mode, options=["average", "by_session"])
    viewer.add_selection("plot_style", value=plot_style, options=["each", "errorPlot"])
    viewer.add_boolean("hide_error", value=hide_error)
    viewer.add_integer("skip_sessions", value=skip_sessions, min=0, max=max(num_sessions, 1))
    # Axes-fraction placement of the session-number colorbar inset. Steps below syd's 0.01
    # default: the inset is a few percent of the panel tall, so 0.01 is a coarse nudge.
    viewer.add_float("inset_x", value=inset_position[0], min=-1.0, max=1.0, step=0.01)
    viewer.add_float("inset_y", value=inset_position[1], min=-1.0, max=1.0, step=0.01)
    viewer.add_float("inset_width", value=inset_position[2], min=0.01, max=1.5, step=0.01)
    viewer.add_float("inset_height", value=inset_position[3], min=0.01, max=1.0, step=0.005)


def by_session_curves(curves_by_mouse: list[list[np.ndarray | None]], grid_size: int) -> np.ndarray:
    """Stack per-mouse curve lists into ``(mice, max_sessions, grid_size)``, NaN-padded.

    Mice with shorter histories are padded, so column j is "everyone's j-th session".
    """
    max_sessions = max((len(curves) for curves in curves_by_mouse), default=0)
    stacked = np.full((len(curves_by_mouse), max_sessions, grid_size), np.nan)
    for imouse, curves in enumerate(curves_by_mouse):
        for isession, curve in enumerate(curves):
            if curve is not None:
                stacked[imouse, isession] = curve
    return stacked


def draw_session_curve_groups(ax, xvals: np.ndarray, by_session: np.ndarray, state, fontsize: float):
    """Draw the population panel of :func:`add_session_curve_widgets`, returning the mean curves.

    Under ``curve_mode="by_session"`` each session number with at least two mice becomes one
    group -- a group backed by a single mouse is that mouse's curve, not a population average --
    colored across the kept session numbers and thinned by ``skip_sessions`` (first and last
    always kept). Under ``"average"`` every mouse contributes one curve, its own sessions
    averaged, and the whole thing is drawn as a single black group.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    xvals : np.ndarray
        The grid every curve is evaluated on.
    by_session : np.ndarray
        ``(mice, sessions, grid)`` curves, from :func:`by_session_curves`.
    state : dict
        Viewer state carrying the :func:`add_session_curve_widgets` knobs plus ``cmap``.
    fontsize : float
        Font size of the colorbar inset's labels.

    Returns
    -------
    list of np.ndarray
        The mean curve of each drawn group, for a caller fitting its y limits to what is drawn.
    """
    cmap_name = state["cmap"]
    cmap = mpl.colormaps[cmap_name]
    drawn_means = []

    if state["curve_mode"] == "by_session":
        support = np.array([np.sum(np.isfinite(by_session[:, j, :]).any(axis=1)) for j in range(by_session.shape[1])])
        kept_js = np.where(support > 1)[0]
        # Color over the kept session numbers, so the full colormap range is used even when
        # trailing sparse session numbers are excluded.
        session_colors = cmap(np.linspace(0, 1, max(len(kept_js), 1)))

        # Thin out which kept sessions are drawn (always first + last), so dense session counts
        # don't overplot the panel. Colors still index the full kept_js range.
        n_kept = len(kept_js)
        step = state["skip_sessions"] + 1
        if n_kept <= 2 or step <= 1:
            show_idx = np.arange(n_kept)
        else:
            n_points = max(2, int(round((n_kept - 1) / step)) + 1)
            show_idx = np.unique(np.round(np.linspace(0, n_kept - 1, n_points)).astype(int))

        for color_idx in show_idx:
            session_data = by_session[:, kept_js[color_idx], :]
            render_curve_group(
                ax,
                xvals,
                session_data,
                session_colors[color_idx],
                state["plot_style"],
                hide_error=state["hide_error"],
                linewidth=1.5,
            )
            drawn_means.append(np.nanmean(session_data, axis=0))

        if n_kept:
            inset_bounds = [state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]]
            draw_session_colorbar_inset(ax, kept_js[0] + 1, kept_js[-1] + 1, cmap_name, fontsize, inset_bounds)
    else:
        # One curve per mouse (its own sessions averaged), rendered as a single group.
        mouse_average = np.stack(
            [np.nanmean(curves, axis=0) if np.any(np.isfinite(curves)) else np.full(by_session.shape[2], np.nan) for curves in by_session]
        )
        render_curve_group(ax, xvals, mouse_average, "k", state["plot_style"], hide_error=state["hide_error"], linewidth=2.0)
        drawn_means.append(np.nanmean(mouse_average, axis=0))

    return drawn_means


# ======================================================================================
# Per-slot experience curves
# ======================================================================================


def pad_stack(curves: list[np.ndarray]) -> np.ndarray:
    """Stack ragged per-mouse 1-D curves into a NaN-padded ``(n_mice, max_len)`` array."""
    max_len = max((len(values) for values in curves), default=0)
    stack = np.full((len(curves), max_len), np.nan)
    for i, values in enumerate(curves):
        stack[i, : len(values)] = values
    return stack


def support_length(stack: np.ndarray, min_support: int = 1) -> int:
    """Number of leading columns where more than ``min_support`` mice have finite data."""
    support = np.sum(np.isfinite(stack), axis=0)
    valid = np.where(support > min_support)[0]
    return int(valid[-1] + 1) if valid.size else 0


def ordinal(value: float) -> str:
    """English ordinal of a percentile level: ``90`` -> ``90th``, ``2`` -> ``2nd``, ``2.5`` -> ``2.5th``.

    Fractional levels (``2.5``, ``97.5``) take a plain ``th``, which is how percentiles are
    conventionally written.
    """
    if value != int(value):
        return f"{value:g}th"
    number = int(value)
    # 11th/12th/13th are the exceptions to the 1st/2nd/3rd pattern.
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


__all__ = [
    "COLORSCALE_INSET_RECT",
    "add_colorscale_widgets",
    "add_session_curve_widgets",
    "by_session_curves",
    "colorscale_inset_rect",
    "curve_group_bounds",
    "decimal_yticks",
    "draw_colorscale_inset",
    "draw_example_roi_panel",
    "draw_session_colorbar_inset",
    "draw_session_curve_groups",
    "draw_vertical_colorscale",
    "env_slot_color",
    "fit_square_panels",
    "hide_spines",
    "ordinal",
    "pad_stack",
    "render_curve_group",
    "round_up_labels",
    "style_axis",
    "support_length",
]
