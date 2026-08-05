"""Normalized ``sf_cv`` / ``ff`` spectra and their cumulative-variance-ratio beeswarms.

The data prep (:func:`ratios_arrays`) and all three drawing helpers are shared by the standalone
ratios panel and the composite figure, which differ only in how the beeswarms are laid out: two
side-by-side axes there, one axis with spacing-carried grouping here.
"""

import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

from vrAnalysis.helpers.plotting import beeswarm, format_spines
from dimensionality_manuscript import average_by_mouse
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.figure_scripts.figure4._spectrum_math import _clip_at_first_negative, _smooth_spectrum

from ._familiarity import CONDITION_COLORS, select_by_env_spectra

# The spectrum panel's x axis is cut just past the last dimension still drawn above the y floor:
# both spectra fall out of view long before the last shared dimension, so autoscaling to the full
# array width leaves a long empty tail.
SPECTRUM_XMAX_PAD = 1.1

# Fractional spine offset used across the composite figure's panels. The combined beeswarm's
# segmented x spine is drawn by hand at the same offset (and format_spines' default spine width)
# so its segments land exactly where a continuous bottom spine would have.
COMPOSITE_SPINE_OFFSET = -0.02
COMPOSITE_SPINE_LINEWIDTH = 1.0

# Condition order within each beeswarm group, and the array keys holding each group's values.
RATIOS_GROUP_LABELS = ["1st 10", "All"]
RATIOS_GROUP_KEYS = [
    ("sf_cv_total_10", "sf_cv_total_iti_10", "sf_cv_total_spont_10"),
    ("sf_cv_total", "sf_cv_total_iti", "sf_cv_total_spont"),
]


def _average_available_env_slots(spectra: np.ndarray) -> np.ndarray:
    """Average a session's available environment spectra, preserving all-NaN dimensions."""
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim == 2:
        return spectra
    if spectra.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D spectrum array, got shape {spectra.shape}")
    count = np.sum(np.isfinite(spectra), axis=1)
    total = np.nansum(spectra, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def ratios_arrays(
    results: ResultsAggregator,
    sel_params: dict,
    *,
    spectrum_mode: str = "all",
    env_full_scope: str = "within_env",
    full_within_env: bool = False,
) -> dict[str, np.ndarray]:
    """Mouse-averaged spectra plus per-mouse cumulative-variance-ratio arrays for the ratios figure.

    Shared data prep for :func:`plot_ratios_spectrum`, :func:`plot_ratios_beeswarms` and
    :func:`plot_ratios_beeswarms_combined`.
    """
    out = results.sel(keys=["sf_cv", "ff"], **sel_params, avg_by_mouse=False, include_iti=False)
    out_iti = results.sel(keys=["sf_cv", "ff"], **sel_params, avg_by_mouse=False, include_iti=True)

    if spectrum_mode == "all":
        # Retain the unnormalized behaving spectra for optional display-only processing. All SVR
        # calculations below continue to use their own untouched raw inputs.
        sf_cv_sessions_raw = np.asarray(out["sf_cv"], dtype=float)
        ff_sessions_raw = np.asarray(out["ff"], dtype=float)
        spectrum_mouse_names = results.mouse_names
    elif spectrum_mode == "avg_env":
        sf_env, ff_env, session_mask = select_by_env_spectra(
            results,
            sel_params,
            env_full_scope,
            full_within_env,
        )
        sf_cv_sessions_raw = _average_available_env_slots(sf_env)[session_mask]
        ff_sessions_raw = _average_available_env_slots(ff_env)[session_mask]
        spectrum_mouse_names = results.mouse_names[session_mask]
    else:
        raise ValueError(f"Unknown spectrum_mode {spectrum_mode!r}. Options: ['all', 'avg_env']")

    full_sum = np.nansum(out["ff"], axis=1, keepdims=True)
    full_sum_iti = np.nansum(out_iti["ff"], axis=1, keepdims=True)
    sf_cv = out["sf_cv"] / full_sum
    sf_cv_iti = out_iti["sf_cv"] / full_sum_iti
    ff = out["ff"] / full_sum

    full_sum_10 = np.nansum(out["ff"][:, :10], axis=1, keepdims=True)
    full_sum_iti_10 = np.nansum(out_iti["ff"][:, :10], axis=1, keepdims=True)
    sf_cv_10 = out["sf_cv"][:, :10] / full_sum_10
    sf_cv_iti_10 = out_iti["sf_cv"][:, :10] / full_sum_iti_10

    session_has_spontaneous = np.array([session.has_spontaneous() for session in results.sessions])

    # Measure cumulative variance after normalizing
    sf_cv_total = average_by_mouse(np.nansum(sf_cv, axis=1), results.mouse_names)
    sf_cv_total_10 = average_by_mouse(np.nansum(sf_cv_10, axis=1), results.mouse_names)
    sf_cv_total_iti = average_by_mouse(np.nansum(sf_cv_iti[~session_has_spontaneous], axis=1), results.mouse_names[~session_has_spontaneous])
    sf_cv_total_iti_10 = average_by_mouse(np.nansum(sf_cv_iti_10[~session_has_spontaneous], axis=1), results.mouse_names[~session_has_spontaneous])
    sf_cv_total_spont = average_by_mouse(np.nansum(sf_cv_iti[session_has_spontaneous], axis=1), results.mouse_names[session_has_spontaneous])
    sf_cv_total_spont_10 = average_by_mouse(np.nansum(sf_cv_iti_10[session_has_spontaneous], axis=1), results.mouse_names[session_has_spontaneous])

    # Average the unmodified curves by mouse for viewers that do not request display processing.
    sf_cv = average_by_mouse(sf_cv, results.mouse_names)
    ff = average_by_mouse(ff, results.mouse_names)

    return dict(
        sf_cv=sf_cv,
        ff=ff,
        sf_cv_total=sf_cv_total,
        sf_cv_total_10=sf_cv_total_10,
        sf_cv_total_iti=sf_cv_total_iti,
        sf_cv_total_iti_10=sf_cv_total_iti_10,
        sf_cv_total_spont=sf_cv_total_spont,
        sf_cv_total_spont_10=sf_cv_total_spont_10,
        _sf_cv_sessions_raw=sf_cv_sessions_raw,
        _ff_sessions_raw=ff_sessions_raw,
        _mouse_names=spectrum_mouse_names,
    )


def spectrum_display_arrays(
    arrays: dict[str, np.ndarray],
    pf_smooth_method: str,
    pf_smooth_width: float,
    full_smooth_method: str,
    full_smooth_width: float,
    clip_negative: bool = True,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Copy ``arrays`` with display-only clipping/smoothing applied to its two spectra.

    The cumulative SVR arrays are deliberately left untouched. This lets a viewer calculate all
    ratios from raw spectra while using Figure 4's first-negative clipping and geometric-mean
    rank smoothing only for the curves it draws. When ``normalize`` is true, both processed
    spectra share the same denominator: the processed Full-CA1 spectrum's sum for that mouse.
    """

    def _prepare(session_key: str, averaged_key: str, method: str, width: float) -> np.ndarray:
        spectra = arrays.get(session_key, arrays[averaged_key])
        if clip_negative:
            spectra = _clip_at_first_negative(spectra)
        if session_key in arrays and "_mouse_names" in arrays:
            spectra = average_by_mouse(spectra, arrays["_mouse_names"])
        return _smooth_spectrum(spectra, method, width)

    display = dict(arrays)
    sf_cv = _prepare("_sf_cv_sessions_raw", "sf_cv", pf_smooth_method, pf_smooth_width)
    ff = _prepare("_ff_sessions_raw", "ff", full_smooth_method, full_smooth_width)
    if normalize:
        denominator = np.nansum(ff, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            sf_cv = sf_cv / denominator
            ff = ff / denominator
    display["sf_cv"] = sf_cv
    display["ff"] = ff
    return display


def last_visible_dimension(ymin: float, *curve_stacks: np.ndarray) -> int:
    """Largest 1-based shared dimension at which any per-mouse curve is still above ``ymin``.

    Only the per-mouse curves need checking: their mouse-average can't be above ``ymin`` at a
    dimension where every mouse is below it.
    """
    last = 1
    for values in curve_stacks:
        columns = np.where((values > ymin).any(axis=0))[0]
        if columns.size:
            last = max(last, int(columns[-1]) + 1)
    return last


def plot_ratios_spectrum(
    ax,
    arrays: dict[str, np.ndarray],
    fontsize: float,
    standard_log_yticklabels: bool = False,
    ylim: tuple[float, float] | None = None,
    show_first10_indicator: bool = True,
    include_legend: bool = True,
) -> None:
    """Mouse-averaged normalized ``sf_cv`` / ``ff`` spectra (log-log).

    The x axis is pinned to start at exactly 1 (so no minor decade ticks appear below the first
    shared dimension) and to end :data:`SPECTRUM_XMAX_PAD` past the last dimension still drawn
    above the y floor, with major ticks on the decades inside that range and no major or minor
    tick beyond it.
    """
    sf_cv, ff = arrays["sf_cv"], arrays["ff"]
    ylim = (10**-5.5, 10**-0.8) if ylim is None else ylim
    ylim_min, ylim_max = np.log10(ylim)
    yline = ylim_min + 0.25
    sf_color = CONDITION_COLORS["behaving"]
    ff_color = "black"
    each_alpha = 0.3

    def xvals(x):
        return np.arange(x.shape[1]) + 1

    ax.plot(xvals(sf_cv), sf_cv.T, color=sf_color, alpha=each_alpha, linewidth=1.0)
    ax.plot(xvals(ff), ff.T, color=ff_color, alpha=each_alpha, linewidth=1.0)
    ax.plot(xvals(sf_cv), np.nanmean(sf_cv, axis=0), color=sf_color, label=r"PF$\times$CA1", linewidth=2.0)
    ax.plot(xvals(ff), np.nanmean(ff, axis=0), color=ff_color, label=r"CA1$\times$CA1", linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if standard_log_yticklabels:
        ax.tick_params(axis="y", labelsize=fontsize)
    else:
        yticks = ax.get_yticks()
        ytick_power = [np.log10(yt) for yt in yticks]
        ax.set_yticks(yticks, labels=ytick_power, fontsize=fontsize)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Shared Modes", fontsize=fontsize)
    ax.set_ylabel("Shared Variance", fontsize=fontsize)
    # Set before format_spines, which freezes its x_pos fraction into a data coordinate.
    xmax = SPECTRUM_XMAX_PAD * last_visible_dimension(ylim[0], sf_cv, ff)
    ax.set_xlim(1, xmax)
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=[1, xmax],
        ybounds=[ylim[0], ylim[1]],
        tick_fontsize=fontsize,
    )
    if include_legend:
        ax.legend(loc="upper right", fontsize=fontsize, frameon=False)
    # Ticks are placed explicitly rather than filtered after the fact: matplotlib draws only what
    # falls inside the view interval, so pinning xlim to [1, xmax] is what keeps the sub-decade
    # minor ticks from spilling below 1 or past the end of the data.
    decades = 10.0 ** np.arange(0, np.floor(np.log10(xmax)) + 1)
    ax.set_xticks(decades, labels=[f"{int(decade)}" for decade in decades], fontsize=fontsize)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if show_first10_indicator:
        ax.annotate(
            "",
            xy=(10, 10**yline),
            xytext=(1, 10**yline),
            arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0),
            annotation_clip=False,
        )
        ax.text(np.sqrt(10), 10 ** (yline + 0.1), "1st 10", fontsize=fontsize, ha="center", va="bottom")


def plot_ratios_beeswarms(ax1, ax2, arrays: dict[str, np.ndarray], fontsize: float) -> None:
    """Beeswarm of cumulative variance ratio in the first 10 dims (``ax1``) and all dims (``ax2``)."""
    xticks = [0, 1, 2]
    xticklabels = ["Behaving", "w/ ITIs", "w/ Spont"]

    beewidth = 0.2
    alpha = 0.3
    line_extent = np.array([-0.25, 0.25])
    np1 = np.array([1, 1])
    linewidth = 2.0
    color_behaving = CONDITION_COLORS["behaving"]
    color_itis = CONDITION_COLORS["itis"]
    color_spontaneous = CONDITION_COLORS["spontaneous"]

    def _swarm(ax, x, values, color):
        ax.plot(
            x + beewidth * beeswarm(values),
            values,
            color=color,
            linestyle="none",
            linewidth=0.5,
            marker="o",
            markersize=3,
            alpha=alpha,
        )

    _swarm(ax1, xticks[0], arrays["sf_cv_total_10"], color_behaving)
    _swarm(ax1, xticks[1], arrays["sf_cv_total_iti_10"], color_itis)
    _swarm(ax1, xticks[2], arrays["sf_cv_total_spont_10"], color_spontaneous)
    _swarm(ax2, xticks[0], arrays["sf_cv_total"], color_behaving)
    _swarm(ax2, xticks[1], arrays["sf_cv_total_iti"], color_itis)
    _swarm(ax2, xticks[2], arrays["sf_cv_total_spont"], color_spontaneous)

    ax1.plot(xticks[0] + line_extent, np1 * np.nanmean(arrays["sf_cv_total_10"]), color=color_behaving, linewidth=linewidth)
    ax1.plot(xticks[1] + line_extent, np1 * np.nanmean(arrays["sf_cv_total_iti_10"]), color=color_itis, linewidth=linewidth)
    ax1.plot(xticks[2] + line_extent, np1 * np.nanmean(arrays["sf_cv_total_spont_10"]), color=color_spontaneous, linewidth=linewidth)
    ax2.plot(xticks[0] + line_extent, np1 * np.nanmean(arrays["sf_cv_total"]), color=color_behaving, linewidth=linewidth)
    ax2.plot(xticks[1] + line_extent, np1 * np.nanmean(arrays["sf_cv_total_iti"]), color=color_itis, linewidth=linewidth)
    ax2.plot(xticks[2] + line_extent, np1 * np.nanmean(arrays["sf_cv_total_spont"]), color=color_spontaneous, linewidth=linewidth)
    ax1.set_ylim(-0.12, 1.10)
    ax1.set_xlim(-0.5, max(xticks) + 0.5)
    ax1.set_ylabel("Variance Ratio", fontsize=fontsize)
    yline_ratio = -0.10
    ax1.annotate(
        "",
        xy=(xticks[0], yline_ratio),
        xytext=(xticks[2], yline_ratio),
        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0),
        annotation_clip=False,
    )
    ax1.text(np.mean(xticks), yline_ratio + 0.02, "1st 10", fontsize=fontsize, ha="center", va="bottom")
    ax2.annotate(
        "",
        xy=(xticks[0], yline_ratio),
        xytext=(xticks[2], yline_ratio),
        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0),
        annotation_clip=False,
    )
    ax2.text(np.mean(xticks), yline_ratio + 0.02, "All", fontsize=fontsize, ha="center", va="bottom")
    format_spines(
        ax1,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=[0, max(xticks)],
        ybounds=[0, 1],
        yticks=[0, 0.5, 1.0],
        tick_fontsize=fontsize,
    )
    format_spines(
        ax2,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["bottom"],
        xbounds=[0, max(xticks)],
        tick_fontsize=fontsize,
    )
    ax2.tick_params(axis="y", left=False, labelleft=False)
    ax1.set_xticks(xticks, labels=xticklabels, rotation=45, ha="right", fontsize=fontsize)
    ax2.set_xticks(xticks, labels=xticklabels, rotation=45, ha="right", fontsize=fontsize)


def ratios_group_positions(within_xspace: float, between_xspace: float) -> np.ndarray:
    """x positions of every beeswarm, as ``(n_groups, n_conditions)``, built up from 0.

    Consecutive conditions within a group are ``within_xspace`` apart; each group boundary adds a
    further ``between_xspace`` on top of that step.
    """
    positions = []
    x = 0.0
    for igroup, keys in enumerate(RATIOS_GROUP_KEYS):
        if igroup:
            x += between_xspace
        for ikey in range(len(keys)):
            if ikey:
                x += within_xspace
            positions.append(x)
    return np.array(positions).reshape(len(RATIOS_GROUP_KEYS), -1)


def plot_ratios_beeswarms_combined(
    ax,
    arrays: dict[str, np.ndarray],
    fontsize: float,
    within_xspace: float = 0.8,
    between_xspace: float = 1.6,
) -> None:
    """Both beeswarm groups (1st 10 dims, all dims) on one axis, for the composite figure.

    Each group holds the three conditions (Behaving/w ITIs/w Spont) ``within_xspace`` apart, and the
    two groups are pushed a further ``between_xspace`` apart, so the grouping is carried by the
    spacing itself. The x spine is drawn as one segment per group, spanning only that group's outer
    swarm *centers* -- the swarm spread and mean lines deliberately spill past both ends, and
    ``xlim`` pads out from those drawn points rather than from the segments. Only the group centers
    get tick labels ("1st 10" / "All"), with the tick marks dropped since they would sit under the
    middle of each segment; the per-color condition labels aren't repeated here since the
    neighboring familiarity legend already maps these same colors to Behaving/w ITIs/w Spont.

    The swarm spread and mean-line width scale with ``within_xspace`` so that tightening the
    within-group spacing brings the conditions closer without making them overlap.
    """
    alpha = 0.3
    linewidth = 2.0
    beewidth = 0.2 * within_xspace
    line_extent = np.array([-0.25, 0.25]) * within_xspace
    np1 = np.array([1, 1])
    colors = (CONDITION_COLORS["behaving"], CONDITION_COLORS["itis"], CONDITION_COLORS["spontaneous"])

    positions = ratios_group_positions(within_xspace, between_xspace)

    # Tracked across every drawn artist, since the swarm's spread is data-dependent and the mean
    # lines stick out past the outermost swarm centers.
    xdata_min, xdata_max = np.inf, -np.inf
    for group_positions, group_keys in zip(positions, RATIOS_GROUP_KEYS):
        for x, key, color in zip(group_positions, group_keys, colors):
            values = arrays[key]
            x_swarm = x + beewidth * beeswarm(values)
            ax.plot(x_swarm, values, color=color, linestyle="none", linewidth=0.5, marker="o", markersize=3, alpha=alpha)
            ax.plot(x + line_extent, np1 * np.nanmean(values), color=color, linewidth=linewidth)
            xdata_min = min(xdata_min, np.nanmin(x_swarm), x + line_extent[0])
            xdata_max = max(xdata_max, np.nanmax(x_swarm), x + line_extent[1])

    ax.set_ylabel("Variance Ratio", fontsize=fontsize)
    # Before format_spines, which freezes the left spine's x_pos fraction into a data coordinate.
    xpad = 0.05 * (xdata_max - xdata_min)
    ax.set_xlim(xdata_min - xpad, xdata_max + xpad)
    format_spines(
        ax,
        x_pos=COMPOSITE_SPINE_OFFSET,
        y_pos=COMPOSITE_SPINE_OFFSET,
        spines_visible=["left"],
        ybounds=[0, 1],
        yticks=[0, 0.5, 1.0],
        tick_fontsize=fontsize,
    )
    ax.set_xticks(positions[:, positions.shape[1] // 2], labels=RATIOS_GROUP_LABELS, fontsize=fontsize)
    ax.tick_params(axis="x", length=0)

    # Drawn at the y the hidden bottom spine would take once the composite's shared [0, 1] ylim is
    # applied -- which is outside that ylim, hence clip_on=False.
    for group_positions in positions:
        ax.plot(
            [group_positions[0], group_positions[-1]],
            [COMPOSITE_SPINE_OFFSET] * 2,
            color="k",
            linewidth=COMPOSITE_SPINE_LINEWIDTH,
            solid_capstyle="butt",
            clip_on=False,
            zorder=3,
        )


def plot_ratios_beeswarms_concise(
    ax,
    arrays: dict[str, np.ndarray],
    fontsize: float,
    include_first10: bool = True,
) -> None:
    """Behaving-only variance ratios for all dimensions, optionally preceded by the first 10."""
    keys = ("sf_cv_total_10", "sf_cv_total") if include_first10 else ("sf_cv_total",)
    labels = RATIOS_GROUP_LABELS if include_first10 else ["All"]
    xticks = np.arange(len(keys), dtype=float)
    color = CONDITION_COLORS["behaving"]
    alpha = 0.3
    beewidth = 0.2
    line_extent = np.array([-0.25, 0.25])

    for x, key in zip(xticks, keys):
        values = arrays[key]
        ax.plot(
            x + beewidth * beeswarm(values),
            values,
            color=color,
            linestyle="none",
            linewidth=0.5,
            marker="o",
            markersize=3,
            alpha=alpha,
        )
        ax.plot(
            x + line_extent,
            np.repeat(np.nanmean(values), 2),
            color=color,
            linewidth=2.0,
            label="All" if key == "sf_cv_total" else None,
        )

    ax.set_xlim(-0.4, xticks[-1] + 0.4)
    ax.set_ylabel("Variance Ratio", fontsize=fontsize)
    format_spines(
        ax,
        x_pos=COMPOSITE_SPINE_OFFSET,
        y_pos=COMPOSITE_SPINE_OFFSET,
        spines_visible=["left", "bottom"] if include_first10 else ["left"],
        xbounds=xticks if include_first10 else None,
        ybounds=[0, 1],
        yticks=[0, 0.5, 1.0],
        tick_fontsize=fontsize,
    )
    if include_first10:
        ax.set_xticks(xticks, labels=labels, fontsize=fontsize)
    else:
        ax.set_xticks([])
