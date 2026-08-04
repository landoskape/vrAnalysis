"""Cross-panel statistics and beeswarm/decay-comparison drawing helpers.

These are figure4-specific drawing routines (unlike :mod:`..panels`, which only holds
figure-agnostic bundles): they know about the decay-model comparison (power law vs exponential)
and the per-mouse beeswarm conventions this figure repeats across panels.
"""

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from vrAnalysis.helpers.plotting import beeswarm, errorPlot, format_spines


def _zero_to_max_ticks(data_list, step: float = 5.0) -> tuple[tuple[float, float], list[float]]:
    """``(ybounds, yticks)`` for a non-negative statistic: the range ``[0, max]`` over ``data_list``.

    Ticks are ``0`` and the largest multiple of ``step`` at or below the data maximum (just ``[0]``
    when the maximum falls below one step). ``data_list`` is any collection of arrays; non-finite
    entries are ignored.
    """
    finite = np.concatenate([np.asarray(d, dtype=float).ravel() for d in data_list]) if len(data_list) else np.array([])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0), [0.0]
    ymax = float(np.max(finite))
    top_tick = step * np.floor(ymax / step)
    return (0.0, ymax), [0.0] if top_tick <= 0 else [0.0, float(top_tick)]


def _format_stat_spines(ax, xbounds, ybounds, yticks) -> None:
    """:func:`format_spines` for a decay-stat panel, honoring an explicit y range / ticks when given.

    With ``ybounds`` the axis bottom is pinned there (the top keeps matplotlib's padded auto limit) and
    the y spine is bounded to it; without it the spine spans the current limits, as before.
    """
    if ybounds is not None:
        ax.set_ylim(bottom=ybounds[0])
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=list(xbounds),
        ybounds=list(ybounds) if ybounds is not None else list(ax.get_ylim()),
        **({} if yticks is None else {"yticks": list(yticks)}),
    )


def _decay_stat_panel(
    ax,
    data_list,
    colors,
    labels,
    display: str,
    beewidth: float,
    fontsize: float,
    xtick_labels,
    ybounds: tuple[float, float] | None = None,
    yticks=None,
) -> None:
    """One panel of a per-curve statistic at the two decay-model x-positions (power law, exponential).

    ``data_list`` holds one ``(n_mice, 2)`` array per curve option (its two columns are the power-law
    and exponential values). NaN mice are dropped by every mode.

    - ``display == "each"``: one faint per-mouse line across x=``[0, 1]`` per curve, plus a bold
      across-mouse mean line.
    - ``display == "errorPlot"``: the across-mouse mean +/- SE band per curve (via
      :func:`~vrAnalysis.helpers.plotting.errorPlot`).
    - ``display == "swarm"``: no per-mouse connections -- one beeswarm column per (decay model, curve)
      at ``x = model_index * n_curves + curve_index`` (blocks ``[0 .. n_curves-1]`` for the power law,
      then ``[n_curves .. 2*n_curves-1]`` for the exponential), each with a short horizontal mean line.
      ``beewidth`` sets the point spread; the reduced-equation label sits under each block.

    ``ybounds``/``yticks`` override the y range and ticks (see :func:`_format_stat_spines`); by default
    the y spine spans the automatic limits and keeps matplotlib's ticks.
    """
    if display == "swarm":
        n_curves = len(data_list)
        line_extent = np.array([-0.25, 0.25])
        for j in range(2):  # decay model column: 0 = power law, 1 = exponential
            for k, (data, color) in enumerate(zip(data_list, colors)):
                vals = np.asarray(data, dtype=float)[:, j]
                x = j * n_curves + k
                offsets = np.zeros_like(vals)
                finite = np.isfinite(vals)
                if finite.any():
                    offsets[finite] = beeswarm(vals[finite])
                ax.plot(
                    x + beewidth * offsets,
                    vals,
                    color=color,
                    linestyle="none",
                    marker="o",
                    markersize=3,
                    alpha=0.3,
                    label=labels[k] if j == 0 else None,
                )
                ax.plot(x + line_extent, [np.nanmean(vals)] * 2, color=color, linewidth=2.0)
        n_pos = 2 * n_curves
        ax.set_xlim(-0.5, n_pos - 0.5)
        centers = [(n_curves - 1) / 2.0, n_curves + (n_curves - 1) / 2.0]
        _format_stat_spines(ax, [0, n_pos - 1], ybounds, yticks)
        ax.set_xticks(centers, labels=xtick_labels, fontsize=fontsize)
        return

    x = np.array([0.0, 1.0])
    for data, color, label in zip(data_list, colors, labels):
        data = np.atleast_2d(np.asarray(data, dtype=float))
        if display == "errorPlot":
            errorPlot(x, data, axis=0, se=True, ax=ax, color=color, alpha=0.25, label=label, linewidth=2.0)
        else:  # "each"
            ax.plot(x, data.T, color=color, alpha=0.3, linewidth=0.8)
            ax.plot(x, np.nanmean(data, axis=0), color=color, linewidth=2.0, label=label)
    ax.set_xlim(-0.3, 1.3)
    _format_stat_spines(ax, [0, 1], ybounds, yticks)
    ax.set_xticks([0, 1], labels=xtick_labels, fontsize=fontsize)


def _paired_pvalue(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Two-sided paired-sample p-value of ``a`` vs ``b`` over the mice finite in both.

    ``method`` is ``"ttest"`` (:func:`scipy.stats.ttest_rel`) or ``"wilcoxon"``
    (:func:`scipy.stats.wilcoxon`, signed-rank). Returns NaN with fewer than two paired
    observations, and ``1.0`` when the two are numerically identical (no difference to test).
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return np.nan
    a, b = a[mask], b[mask]
    if np.allclose(a, b):
        return 1.0
    if method == "wilcoxon":
        return float(wilcoxon(a, b, alternative="two-sided").pvalue)
    return float(ttest_rel(a, b).pvalue)


def _significance_stars(p: float) -> str:
    """Tiered significance label: ``***`` p<0.001, ``**`` p<0.01, ``*`` p<0.05, else ``ns``."""
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _beeswarm_panel(
    ax,
    values_list,
    colors,
    labels,
    fontsize,
    beewidth: float = 0.2,
    each_alpha: float = 0.3,
    yscale: str = "linear",
    markersize: float = 3.0,
    mean_linewidth: float = 2.0,
) -> None:
    """Per-mouse beeswarm (points + bold mean line) at integer x-positions ``0, 1, ...``."""
    line_extent = np.array([-0.25, 0.25])
    for x, (vals, color) in enumerate(zip(values_list, colors)):
        vals = np.asarray(vals, dtype=float)
        offsets = np.zeros_like(vals)
        finite = np.isfinite(vals)
        if finite.any():
            offsets[finite] = beeswarm(vals[finite])
        ax.plot(x + beewidth * offsets, vals, color=color, linestyle="none", marker="o", markersize=markersize, alpha=each_alpha)
        ax.plot(x + line_extent, [np.nanmean(vals)] * 2, color=color, linewidth=mean_linewidth)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_yscale(yscale)
    ymin = 0 if yscale == "linear" else 1
    _ax_ylim = ax.get_ylim()
    ax.set_ylim(ymin, _ax_ylim[1])

    xticks = range(len(labels))
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=[0, max(xticks)],
        ybounds=[ymin, _ax_ylim[1]],
    )
    if len(labels) > 2:
        ax.set_xticks(xticks, labels=labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_xticks(xticks, labels=labels, fontsize=fontsize)


def _horizontal_beeswarm_panel(
    ax,
    values_list,
    colors,
    beewidth: float = 0.15,
    each_alpha: float = 0.6,
    markersize: float = 3.0,
    mean_linewidth: float = 2.0,
) -> None:
    """Draw several horizontal swarms around one shared, unlabeled y-position.

    The x-coordinate is the statistic itself. Swarm offsets are computed in log space because
    this helper is used below a log-rank spectrum axis. When groups contain the same number of
    mice, later groups reuse the PF offsets so corresponding CA1/PF points are y-aligned; otherwise
    the group gets its own centered offsets. Group identity is intentionally left to the spectrum
    legend in the shared panel above.
    """
    mean_extent = np.array([-0.22, 0.22])
    reference_offsets = None
    for vals, color in zip(values_list, colors):
        vals = np.asarray(vals, dtype=float)
        finite = np.isfinite(vals) & (vals > 0)
        offsets = np.zeros_like(vals)
        if reference_offsets is not None and len(reference_offsets) == len(vals):
            offsets = reference_offsets.copy()
        elif finite.any():
            offsets[finite] = beeswarm(np.log10(vals[finite]))
        if reference_offsets is None:
            reference_offsets = offsets.copy()
        ax.plot(
            vals,
            beewidth * offsets,
            color=color,
            linestyle="none",
            marker="o",
            markersize=markersize,
            alpha=each_alpha,
        )
        if finite.any():
            mean = np.nanmean(vals[finite])
            ax.plot([mean, mean], mean_extent, color=color, linewidth=mean_linewidth)

    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, right=False, labelleft=False)
