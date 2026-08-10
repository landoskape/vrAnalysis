"""Place-field peak amplitude: one example mouse, the population, and the environment slots."""

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from vrAnalysis.helpers import edge2center

from dimensionality_manuscript.env_order import MAX_ENV_SLOTS, _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    curve_group_bounds,
    data_selection,
    render_curve_group,
)
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import (
    add_session_curve_widgets,
    by_session_curves,
    draw_session_curve_groups,
    env_slot_color,
    pad_stack,
    style_axis,
    support_length,
)

RELIABILITY_THRESHOLD = 0.3


def _nanmax_over_slots(slot_peaks: np.ndarray) -> np.ndarray:
    """Largest peak across environment slots for each ROI, NaN where an ROI has none.

    Plain ``np.nanmax`` would warn on the all-NaN columns that ROI padding and unrun environments
    both produce, so the reduction is restricted to the columns that have data.
    """
    values = np.full(slot_peaks.shape[1], np.nan)
    has_data = np.any(np.isfinite(slot_peaks), axis=0)
    if np.any(has_data):
        values[has_data] = np.nanmax(slot_peaks[:, has_data], axis=0)
    return values


class PlacefieldPeakAmplitude(FigureViewer):
    """Place-field peak amplitude: one example mouse, the population, and the environment slots.

    A peak is the max over position of an ROI's trial-averaged place field, in units of that ROI's
    standard deviation in time. Everything is read from the ``PFPredQualityConfig`` aggregator --
    ``pf_peak_hist`` (counts per session and environment slot, precomputed on
    ``pf_peak_hist_edges``), ``pf_peak_n`` (how many ROIs went into each), and ``pf_peak`` (the
    per-ROI peaks themselves) -- so nothing is measured here and the figure is cheap to redraw.
    The aggregator's slot-aligned ``reliability_slot`` values optionally restrict every panel to
    ROIs with reliability greater than 0.3 in the environment contributing that peak.

    - ``ax[0]``: every session of one mouse, one histogram per session colored by session number
      (early sessions cool and late warm). A drift in how strongly cells are driven at their
      preferred position shows up as an ordered fan of curves.
    - ``ax[1]``: the same curves for the whole population -- mouse-averaged overall
      (``curve_mode="average"``) or grouped by session number (``"by_session"``). Shares ``ax[0]``'s
      y-axis.
    - ``ax[2]``: the distribution collapsed to one number per session (``summary_stat``), against
      how many sessions of that environment the mouse has had, one curve per experience slot.
    - An optional fourth panel repeats that summary for the example mouse alone.

    Environments are indexed by experience-order slot, so slot 0 is the first environment that
    mouse ever saw; ``env_slot_ids`` carries the underlying environment index, which differs
    between the two cohorts.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``PFPredQualityConfig`` results.
    mouse : str or None
        Example mouse for ``ax[0]`` (and the optional 4th panel). None takes the first
        alphabetically.
    env_mode : {"pooled", "best", "slot"}
        Which environments contribute to ``ax[0]`` and ``ax[1]``. ``"pooled"`` sums the per-slot
        histograms (every (ROI, environment) pair is a sample), ``"best"`` rebins each ROI's
        largest peak across slots, ``"slot"`` takes a single experience slot. ``ax[2]`` always
        splits by slot, so this does not affect it.
    env_slot : int
        Experience-order slot used when ``env_mode="slot"``.
    filter_by_reliability : bool
        Keep only peaks whose same-environment spatial reliability is greater than 0.3. Both the
        unfiltered and filtered histograms are cached when the aggregator selection is loaded, so
        toggling this option does not reload or recompute session data.
    xrange : tuple[float, float] or None
        X limits of ``ax[0]``/``ax[1]``. None shows the full stored range. This only crops the
        view -- the bins are fixed by the config that wrote the results.
    density : bool
        Normalize each session's histogram by its ROI count and bin width. False plots raw counts,
        which then also reflect how many ROIs the session had.
    cumulative : bool
        Plot cumulative distributions instead of histograms.
    log_y : bool
        Log-scale the (shared) y axis of ``ax[0]`` and ``ax[1]``.
    summary_stat : {"median", "mean"}
        How ``ax[2]`` (and the optional 4th panel) collapses a session's peak distribution.
    slot_style : {"each", "errorPlot"}
        ``ax[2]``'s rendering.
    show_slot_legend : bool
        Draw the env-slot legend on ``ax[2]``.
    style : {"line", "step", "fill"}
        ``ax[0]``'s histogram rendering.
    cmap : str
        Colormap sampled across sessions, first session at 0 and last at 1.
    linewidth, alpha : float
        Line width and opacity of ``ax[0]``'s curves.
    show_colorbar : bool
        Draw ``ax[0]``'s session-number colorbar.
    show_env_label : bool
        Label which environments ``ax[0]``/``ax[1]`` came from, in the corner of ``ax[0]``.
    show_summary : bool
        Add a fourth panel: the example mouse's per-session summary stat against session number.
    fontsize : float
        Base font size in points.
    figsize : tuple[float, float]
        Figure size in inches.
    curve_mode, plot_style, hide_error, skip_sessions, inset_position
        ``ax[1]``'s knobs, forwarded to :func:`~._shared.add_session_curve_widgets`.
    **selection_defaults
        Starting values for the aggregator's own param axes, by name.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        env_mode: str = "pooled",
        env_slot: int = 0,
        filter_by_reliability: bool = False,
        xrange: tuple[float, float] | None = None,
        density: bool = True,
        cumulative: bool = False,
        log_y: bool = False,
        curve_mode: str = "by_session",
        plot_style: str = "each",
        hide_error: bool = False,
        skip_sessions: int = 0,
        inset_position: tuple[float, float, float, float] = (0.35, 0.85, 0.6, 0.075),
        summary_stat: str = "median",
        slot_style: str = "errorPlot",
        show_slot_legend: bool = True,
        style: str = "line",
        cmap: str = "coolwarm",
        linewidth: float = 1.0,
        alpha: float = 0.9,
        show_colorbar: bool = True,
        show_env_label: bool = True,
        show_summary: bool = False,
        fontsize: float = 8.0,
        figsize: tuple[float, float] = (8.0, 2.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize

        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self._load_arrays(self.state)

        # Rows of every array above are aggregator sessions; a mouse's rows are ordered
        # chronologically here so "session number" means what it says on the color axis.
        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(name) for name in mouse_names})
        self._rows_by_mouse = {
            name: np.array(sorted(np.flatnonzero(mouse_names == name), key=lambda row: _session_sort_key(results.sessions[row])))
            for name in self.mice
        }

        # --- what goes into a curve ---
        self.add_selection("mouse", value=mouse if mouse is not None else self.mice[0], options=self.mice)
        self.add_selection("env_mode", value=env_mode, options=["pooled", "best", "slot"])
        self.add_integer("env_slot", value=env_slot, min=0, max=MAX_ENV_SLOTS - 1)
        self.add_boolean("filter_by_reliability", value=filter_by_reliability)
        # The bins come from the config that wrote the results, so this crops the view rather than
        # rebinning -- hence the step of one bin width.
        self.add_float_range(
            "xrange",
            value=tuple(xrange) if xrange is not None else (float(self.bin_edges[0]), float(self.bin_edges[-1])),
            min=float(self.bin_edges[0]),
            max=float(self.bin_edges[-1]),
            step=self.bin_width,
        )
        self.add_boolean("density", value=density)
        self.add_boolean("cumulative", value=cumulative)
        self.add_boolean("log_y", value=log_y)

        # --- ax[1]: population curves ---
        add_session_curve_widgets(
            self,
            num_sessions=len(results.sessions),
            curve_mode=curve_mode,
            plot_style=plot_style,
            hide_error=hide_error,
            skip_sessions=skip_sessions,
            inset_position=inset_position,
        )

        # --- ax[2]: summary stat against experience, one curve per environment slot ---
        self.add_selection("summary_stat", value=summary_stat, options=["median", "mean"])
        self.add_selection("slot_style", value=slot_style, options=["each", "errorPlot"])
        self.add_boolean("show_slot_legend", value=show_slot_legend)

        # --- style ---
        self.add_selection("style", value=style, options=["line", "step", "fill"])
        self.add_text("cmap", value=cmap)
        self.add_float("linewidth", value=linewidth, min=0.25, max=4.0)
        self.add_float("alpha", value=alpha, min=0.05, max=1.0)
        self.add_float("fontsize", value=fontsize, min=3.0, max=30.0)
        self.add_boolean("show_colorbar", value=show_colorbar)
        self.add_boolean("show_env_label", value=show_env_label)
        self.add_boolean("show_summary", value=show_summary)

        self.on_change(list(self.selection_names), self.reload_arrays)
        self.on_change("mouse", self.update_slot_bounds)
        self.on_change(
            ["mouse", "env_mode", "env_slot", "filter_by_reliability", "density", "cumulative", "summary_stat"],
            self.refresh_data,
        )
        self.update_slot_bounds(self.state)
        self.refresh_data(self.state)

    # ---------------------------------------------------------------- data selection --

    def _load_arrays(self, state) -> None:
        """Pull the per-ROI peaks, their precomputed histograms, and the bin edges."""
        keys = ["pf_peak", "pf_peak_hist", "pf_peak_n", "pf_peak_hist_edges", "reliability_slot", "env_slot_ids"]
        out = self.results.sel(keys=keys, squeeze_ones=False, **data_selection(state, self.results, self.selection_names))
        self.pf_peak = np.asarray(out["pf_peak"], dtype=float)  # (sessions, slots, max_rois)
        self.pf_peak_hist = np.asarray(out["pf_peak_hist"], dtype=float)  # (sessions, slots, bins)
        self.pf_peak_n = np.asarray(out["pf_peak_n"], dtype=float)  # (sessions, slots)
        self.reliability_slot = np.asarray(out["reliability_slot"], dtype=float)  # (sessions, slots, max_rois)
        self.env_slot_ids = np.asarray(out["env_slot_ids"], dtype=float)  # (sessions, slots)
        # Every session stores the same edges (they come from the config), so take the first
        # session that actually has them.
        edges = np.asarray(out["pf_peak_hist_edges"], dtype=float)
        idx_edges = np.flatnonzero(np.all(np.isfinite(edges), axis=1))
        if idx_edges.size == 0:
            raise ValueError("No session in the aggregator has pf_peak_hist_edges -- rerun the pfpred_quality sweep.")
        self.bin_edges = edges[idx_edges[0]]
        self.bin_centers = edge2center(self.bin_edges)
        self.bin_width = float(self.bin_edges[1] - self.bin_edges[0])
        self.num_slots = self.pf_peak_n.shape[1]

        # Cache the reliability-filtered counterparts once per aggregator selection. The mask is
        # slot-local: a peak only survives when that ROI is reliable in the environment whose
        # place field supplied the peak. NaN padding naturally fails the comparison.
        reliable_peaks = np.where(self.reliability_slot > RELIABILITY_THRESHOLD, self.pf_peak, np.nan)
        reliable_hist = np.full_like(self.pf_peak_hist, np.nan)
        reliable_n = np.full_like(self.pf_peak_n, np.nan)
        for row in range(reliable_peaks.shape[0]):
            for slot in range(reliable_peaks.shape[1]):
                # Preserve missing/unrun slots as NaN rather than turning them into real zeros.
                if not np.isfinite(self.pf_peak_n[row, slot]):
                    continue
                values = reliable_peaks[row, slot]
                values = values[np.isfinite(values)]
                reliable_hist[row, slot] = np.histogram(values, bins=self.bin_edges)[0]
                reliable_n[row, slot] = values.size
        self._peak_cache = {False: self.pf_peak, True: reliable_peaks}
        self._hist_cache = {False: self.pf_peak_hist, True: reliable_hist}
        self._n_cache = {False: self.pf_peak_n, True: reliable_n}

    def reload_arrays(self, state):
        """Re-pull the arrays after a data-selection change, then re-derive every curve."""
        self._load_arrays(state)
        self.update_slot_bounds(state)
        self.refresh_data(self.state)

    def update_slot_bounds(self, state):
        """Limit the slot selector to the environments this mouse actually ran."""
        rows = self._rows_by_mouse[state["mouse"]]
        num_slots = int(np.sum(np.any(np.isfinite(self.pf_peak_n[rows]), axis=0)))
        self.update_integer("env_slot", max=max(num_slots - 1, 0))

    # ------------------------------------------------------------------ derived curves --

    def _env_mode_label(self, state) -> str:
        """Corner label naming the environments the curves were built from.

        In slot mode it also quotes the underlying environment index, which is what the label
        means for *this* mouse -- the two cohorts run disjoint environment indices, so the slot
        number is the only thing comparable across them.
        """
        reliability_label = f"\nrel > {RELIABILITY_THRESHOLD:g}" if state["filter_by_reliability"] else ""
        if state["env_mode"] == "pooled":
            return "all envs" + reliability_label
        if state["env_mode"] == "best":
            return "best env" + reliability_label
        slot = state["env_slot"]
        slot_ids = self.env_slot_ids[self._rows_by_mouse[state["mouse"]], slot]
        slot_ids = slot_ids[np.isfinite(slot_ids)]
        if slot_ids.size == 0:
            return f"env #{slot + 1}" + reliability_label
        return f"env #{slot + 1} (id {int(slot_ids[0])})" + reliability_label

    def _row_peaks(self, row: int, state) -> np.ndarray:
        """Per-ROI peaks of one session under the current env mode (empty if it ran nothing)."""
        slot_peaks = self._peak_cache[state["filter_by_reliability"]][row]  # (slots, max_rois)
        env_mode = state["env_mode"]
        if env_mode == "pooled":
            # Every (roi, environment) pair is its own sample.
            values = slot_peaks.reshape(-1)
        elif env_mode == "best":
            values = _nanmax_over_slots(slot_peaks)
        elif env_mode == "slot":
            values = slot_peaks[state["env_slot"]]
        else:
            raise ValueError(f"Invalid env_mode: {env_mode!r}")
        return values[np.isfinite(values)]

    def _row_counts(self, row: int, state) -> tuple[np.ndarray, float]:
        """Histogram of one session on ``bin_edges`` as ``(counts, n)``; ``n`` is 0 if it ran nothing.

        ``n`` counts every finite peak, including any above the last edge, so a density is
        ``counts / (n * bin_width)`` and integrates to the fraction that fell in range.
        """
        env_mode = state["env_mode"]
        hist = self._hist_cache[state["filter_by_reliability"]]
        n = self._n_cache[state["filter_by_reliability"]]
        if env_mode == "best":
            # Not derivable from the per-slot histograms -- rebin the stored per-ROI peaks.
            values = self._row_peaks(row, state)
            return np.histogram(values, bins=self.bin_edges)[0].astype(float), float(values.size)
        if env_mode == "pooled":
            valid = np.isfinite(n[row])
            if not np.any(valid):
                return np.zeros(len(self.bin_centers)), 0.0
            return np.sum(hist[row][valid], axis=0), float(np.sum(n[row][valid]))
        slot = state["env_slot"]
        n_slot = n[row, slot]
        if not np.isfinite(n_slot):
            return np.zeros(len(self.bin_centers)), 0.0
        return hist[row, slot], float(n_slot)

    def _row_curve(self, row: int, state) -> np.ndarray | None:
        """One session's plotted curve (density or counts, optionally cumulative), None if empty."""
        counts, n_peaks = self._row_counts(row, state)
        if n_peaks == 0:
            return None
        curve = counts / (n_peaks * self.bin_width) if state["density"] else counts
        if state["cumulative"]:
            curve = np.cumsum(curve) * (self.bin_width if state["density"] else 1.0)
        return curve

    def refresh_data(self, state):
        """Rebuild every curve on the figure: the example mouse's, the population's, and the slots'."""
        rows = self._rows_by_mouse[state["mouse"]]
        self.mouse_curves = [self._row_curve(row, state) for row in rows]
        self.by_session = by_session_curves(
            [[self._row_curve(row, state) for row in self._rows_by_mouse[mouse]] for mouse in self.mice],
            len(self.bin_centers),
        )

        # Per-slot (mice, sessions) summary: x is "how many sessions of this environment the mouse
        # has had", not the session number in the experiment.
        summarize = np.median if state["summary_stat"] == "median" else np.mean
        self.slot_stacks: dict[int, np.ndarray] = {}
        peak = self._peak_cache[state["filter_by_reliability"]]
        for slot in range(self.num_slots):
            per_mouse = []
            for mouse in self.mice:
                values = []
                for row in self._rows_by_mouse[mouse]:
                    peaks = peak[row, slot]
                    peaks = peaks[np.isfinite(peaks)]
                    if peaks.size:
                        values.append(float(summarize(peaks)))
                per_mouse.append(np.array(values))
            self.slot_stacks[slot] = pad_stack(per_mouse)

        # The optional 4th panel: the example mouse's own per-session summary.
        self.mouse_summary = np.array(
            [summarize(peaks) if peaks.size else np.nan for peaks in (self._row_peaks(row, state) for row in rows)]
        )

    # -------------------------------------------------------------------- drawing --

    def _draw_example_mouse(self, ax, state, fontsize, colors) -> float:
        """ax[0]: one histogram per session of the example mouse, returning their visible max."""
        xlow, xhigh = tuple(state["xrange"])
        in_range = (self.bin_centers >= xlow) & (self.bin_centers <= xhigh)
        ymax = 0.0
        for curve, color in zip(self.mouse_curves, colors):
            if curve is None:
                continue
            if state["style"] == "line":
                ax.plot(self.bin_centers, curve, color=color, linewidth=state["linewidth"], alpha=state["alpha"])
            elif state["style"] == "step":
                ax.stairs(curve, self.bin_edges, color=color, linewidth=state["linewidth"], alpha=state["alpha"])
            elif state["style"] == "fill":
                ax.fill_between(self.bin_centers, curve, color=color, alpha=state["alpha"], linewidth=0)
            else:
                raise ValueError(f"Invalid style: {state['style']!r}")
            visible = np.asarray(curve)[in_range]
            visible = visible[np.isfinite(visible)]
            if visible.size:
                ymax = max(ymax, float(np.max(visible)))

        ylabel = ("Cumulative " if state["cumulative"] else "") + ("density" if state["density"] else "count")
        ax.set_xlabel(r"Peak PF amplitude ($\sigma$)", fontsize=fontsize)
        ax.set_ylabel(ylabel.capitalize(), fontsize=fontsize)
        ax.set_title(state["mouse"], fontsize=fontsize)
        return ymax

    def _draw_slot_panel(self, ax, state, fontsize) -> tuple[list[float] | None, int]:
        """ax[2]: the summary stat against environment experience, one curve per slot."""
        slot_ymin, slot_ymax, slot_xmax = np.inf, -np.inf, 0
        for slot, stack in self.slot_stacks.items():
            length = support_length(stack)
            if length == 0:
                continue
            stack = stack[:, :length]
            render_curve_group(
                ax,
                np.arange(1, length + 1),
                stack,
                env_slot_color(slot),
                state["slot_style"],
                label=f"Env #{slot + 1}",
                hide_error=state["hide_error"],
                linewidth=1.5,
            )
            bounds = curve_group_bounds(stack, state["slot_style"])
            if bounds is not None:
                slot_ymin = min(slot_ymin, bounds[0])
                slot_ymax = max(slot_ymax, bounds[1])
            slot_xmax = max(slot_xmax, length)

        ax.set_xlabel("Env session #", fontsize=fontsize)
        ax.set_ylabel(f"{state['summary_stat'].capitalize()} peak ($\\sigma$)", fontsize=fontsize)
        if state["show_slot_legend"]:
            ax.legend(fontsize=fontsize, frameon=False, handlelength=0.8, handletextpad=0.5)
        return ([slot_ymin, slot_ymax] if np.isfinite(slot_ymin) else None), slot_xmax

    def plot(self, state):
        fontsize = state["fontsize"]
        cmap_name = state["cmap"]
        cmap = mpl.colormaps[cmap_name]
        xlow, xhigh = tuple(state["xrange"])
        in_range = (self.bin_centers >= xlow) & (self.bin_centers <= xhigh)

        width_ratios = [1.2, 1.2, 1.0] + ([0.9] if state["show_summary"] else [])
        fig, ax = self.new_subplots(1, len(width_ratios), figsize=self.figsize, layout="constrained", width_ratios=width_ratios)
        # ax[0] and ax[1] draw the same quantity, so the y-axis is drawn once, on ax[0].
        ax[1].sharey(ax[0])

        # ---- ax[0]: every session of the example mouse, colored by session number ----
        num_sessions = len(self.mouse_curves)
        colors = [cmap(isession / max(num_sessions - 1, 1)) for isession in range(num_sessions)]
        ymax = self._draw_example_mouse(ax[0], state, fontsize, colors)

        if state["show_colorbar"] and num_sessions > 1:
            scalar_map = mpl.cm.ScalarMappable(norm=plt.Normalize(vmin=1, vmax=num_sessions), cmap=cmap)
            cbar = fig.colorbar(scalar_map, ax=ax[0], fraction=0.05, pad=0.02)
            cbar.set_label("Session", fontsize=fontsize)
            cbar.set_ticks([1, num_sessions])
            cbar.ax.tick_params(labelsize=fontsize, length=3)
            cbar.outline.set_visible(False)

        # ---- ax[1]: the population, mouse-averaged overall or grouped by session number ----
        for mean_curve in draw_session_curve_groups(ax[1], self.bin_centers, self.by_session, state, fontsize):
            visible = np.asarray(mean_curve)[in_range]
            visible = visible[np.isfinite(visible)]
            if visible.size:
                ymax = max(ymax, float(np.max(visible)))
        ax[1].set_xlabel(r"Peak PF amplitude ($\sigma$)", fontsize=fontsize)
        # Mice the aggregator knows about but has no results for contribute nothing, so the title
        # counts the ones actually drawn rather than the length of the mouse list.
        ax[1].set_title(f"{int(np.sum(np.isfinite(self.by_session).any(axis=(1, 2))))} mice", fontsize=fontsize)

        for axis in (ax[0], ax[1]):
            axis.set_xlim(xlow, xhigh)
        if state["log_y"]:
            ax[0].set_yscale("log")
            ybounds = None
        else:
            # ymax is 0 when nothing was plotted -- an env slot no session of this mouse ran.
            ax[0].set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
            ybounds = [0, ymax] if ymax > 0 else None

        if state["show_env_label"]:
            ax[0].text(1.0, 1.0, self._env_mode_label(state), transform=ax[0].transAxes, ha="right", va="top", fontsize=fontsize)

        style_axis(ax[0], fontsize=fontsize, xbounds=[xlow, xhigh], ybounds=ybounds)
        # ax[1] borrows ax[0]'s y-axis, so it only gets a bottom spine. Its ticks are hidden with
        # tick_params rather than set_yticks, which would propagate through the shared axis and
        # strip ax[0]'s y ticks too.
        style_axis(ax[1], fontsize=fontsize, xbounds=[xlow, xhigh], spines_visible=["bottom"])
        ax[1].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        # ---- ax[2]: summary stat against experience, one curve per environment slot ----
        slot_ybounds, slot_xmax = self._draw_slot_panel(ax[2], state, fontsize)
        if slot_ybounds is not None and slot_xmax:
            ax[2].set_xlim(1, slot_xmax)
            ax[2].set_ylim(*slot_ybounds)
        style_axis(
            ax[2],
            fontsize=fontsize,
            xbounds=[1, max(slot_xmax, 1)],
            ybounds=slot_ybounds,
            xticks=[1, max(slot_xmax, 1)],
        )

        # ---- optional 4th panel: the same summary for the example mouse alone ----
        if state["show_summary"]:
            summary = self.mouse_summary
            session_numbers = np.arange(1, num_sessions + 1)
            ax[3].plot(session_numbers, summary, color="0.7", linewidth=1, zorder=1)
            ax[3].scatter(session_numbers, summary, c=colors, s=12, zorder=2)
            ax[3].set_xlabel("Session", fontsize=fontsize)
            ax[3].set_ylabel(f"{state['summary_stat'].capitalize()} peak ($\\sigma$)", fontsize=fontsize)
            summary_valid = summary[np.isfinite(summary)]
            style_axis(
                ax[3],
                fontsize=fontsize,
                xbounds=[1, num_sessions],
                ybounds=[float(np.min(summary_valid)), float(np.max(summary_valid))] if summary_valid.size else None,
                xticks=[1, num_sessions],
            )
        return fig
