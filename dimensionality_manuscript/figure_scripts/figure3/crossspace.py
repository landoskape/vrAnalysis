"""Cross-spectrum energy summary: how much of full activity the placefield subspace spans."""

import numpy as np

from vrAnalysis.helpers.plotting import format_spines
from dimensionality_manuscript import average_by_mouse
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
    draw_session_colorbar_inset,
    render_curve_group,
)

from ._selection import ACTIVITY_SELECTION_DEFAULTS
from ._curves import (
    DISTRIBUTION_METRICS,
    SESSION_CMAP,
    SMOOTH_KINDS,
    by_session_groups,
    distribution_metric,
    energy_on_full,
    kink_positions,
    stack_sessions_by_mouse,
    supported_xmax,
    weighted_fraction,
)

# Fixed color lookup for the ax[1] metric-explainer decorations (arrows/patch), kept separate
# from the CONDITION_COLORS/env-slot palettes used elsewhere in the manuscript. Edit here to
# restyle every decoration at once.
CROSS_METRIC_COLORS = {
    "max_captured": "black",
    "kink": "darkviolet",
    "missing_structure": "dimgrey",
}

# ax[1]'s session-number colorbar, in that panel's axes-fraction coordinates.
_COLORBAR_INSET_BOUNDS = [0.05, 0.04, 0.6, 0.075]


class SubspaceCrossspaceViewer(FigureViewer):
    """Cross-spectrum energy figure for aggregated subspace results.

    ``ax[0]`` shows an example placefield-vs-full cross matrix for one session (chosen by
    ``idx_cross`` from sessions sorted by descending mean top-10 diagonal energy). ``ax[1]`` shows
    the mouse-averaged fraction of full-activity variance captured by placefields, optionally
    decorated with the three metrics the remaining panels track. ``ax[2]`` and ``ax[3]`` show those
    metrics -- max captured variance and the missing fraction, then the kink dimension -- as
    per-mouse session trajectories with their supported cross-mouse averages.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``SubspaceConfig`` results providing ``param_axes``, ``sel`` and ``mouse_names``.
    idx_cross : int
        Index (into sessions sorted by descending mean top-10 diagonal energy) of the example
        cross matrix shown in ``ax[0]``.
    plot_energy : bool
        Show squared cross energy (``gray_r``, from 0) when True, otherwise the signed cross
        values (``bwr``, from -1) in ``ax[0]``.
    num_cross_show : int
        Number of cross dimensions to show in ``ax[0]``.
    weighted : bool
        ``ax[1]`` shows the variance-weighted recovery ``Var(X P u_i)/λ_i`` when True
        (down-weights overlap onto low-variance full PCs), otherwise the unweighted subspace
        overlap ``||P u_i||²``.
    curve_mode : {"average", "by_session"}
        ``"average"`` groups all sessions of a mouse into one curve per mouse for ``ax[1]``.
        ``"by_session"`` instead groups curves by within-mouse session number, one color-coded
        group per session number (``coolwarm``, early to late); session numbers with data from
        only one mouse are skipped. Either way, rendering is controlled by ``plot_style``.
    plot_style : {"each", "errorPlot"}
        How every curve group is rendered. ``"each"`` draws each underlying curve thin plus a
        solid mean; ``"errorPlot"`` draws a mean +/- std band instead.
    hide_error : bool
        With ``plot_style="errorPlot"``, suppress the std band in ``ax[1]`` and show only the mean
        curve(s). No effect on ``"each"`` or on the other panels.
    skip_sessions : int
        Only used when ``curve_mode="by_session"``. Thins out which session-number groups are
        drawn in ``ax[1]``: the first and last kept session number are always shown, with the rest
        spaced as evenly as possible to skip roughly ``skip_sessions`` kept session numbers
        between each drawn one. Colors still span the full kept-session range.
    curve_smooth_kind : {"none", "boxcar", "gaussian", "median"}
        Smoothing applied to energy-on-full curves before computing session metrics.
    curve_smooth_width : float
        Smoothing width in full-dimension units.
    kink_threshold : float
        Fraction of maximum energy used to locate the first below-threshold dimension.
    distribution_metric : {"gini", "weighted_missing", "missing_structure"}
        Metric shown with max energy in ``ax[2]``. Missing structure is the mean uncaptured
        fraction over valid full dimensions; weighted missing weights it by each full dimension's
        activity variance; gini is reported here as the Gini coefficient (inequality).
    show_decorations : bool
        Overlay the ``ax[1]`` metric-explainer decorations (colors fixed in
        :data:`CROSS_METRIC_COLORS`): a vertical arrow for max captured, a horizontal arrow for
        kink position, and a patch hugging the curve envelope for missing structure.
    show_marker_labels : bool
        Overlay text labels next to each decoration (only when ``show_decorations`` is also True).
    arrow_linewidth : float
        Line width of both the "max captured" and "kink" arrows.
    arrow_head_size : float
        Arrow head width; head length is fixed at ``2 * arrow_head_size``. Shared by both arrows.
    max_arrow_x : float
        X position (dim units) of the vertical "max captured" arrow.
    max_arrow_y_start, max_arrow_y_end : float
        Start/end y of the "max captured" arrow.
    kink_arrow_x_start, kink_arrow_x_end : float
        Start/end x (dim units) of the horizontal "kink" arrow.
    kink_arrow_y : float
        Y position of the "kink" arrow.
    missing_structure_x_offset : float
        Right offset (dim units, added to the panel's left edge) of where the missing-structure
        patch begins.
    missing_structure_y_offset : float
        Vertical gap added above the curve envelope (max over all sessions at each x) for the
        missing-structure patch's bottom edge.
    missing_structure_alpha : float
        Fill alpha of the missing-structure patch.
    fontsize : float
        Font size applied uniformly across every panel: axis labels, tick labels (major and
        minor, including ``ax[1]``'s log-scale x-axis), the colorbar-inset session-number labels,
        and both legends.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the data-selection widgets (``smooth_width``,
        ``activity_parameters_name``, ``subspace_name``), which come from the aggregator's own
        param axes.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        idx_cross: int = 0,
        plot_energy: bool = True,
        num_cross_show: int = 30,
        weighted: bool = False,
        curve_mode: str = "average",
        plot_style: str = "each",
        hide_error: bool = False,
        skip_sessions: int = 0,
        curve_smooth_kind: str = "none",
        curve_smooth_width: float = 3.0,
        kink_threshold: float = 0.95,
        distribution_metric: str = "weighted_missing",
        show_decorations: bool = True,
        show_marker_labels: bool = True,
        arrow_linewidth: float = 1.5,
        arrow_head_size: float = 0.4,
        max_arrow_x: float = 1.0,
        max_arrow_y_start: float = 0.0,
        max_arrow_y_end: float = 0.9,
        kink_arrow_x_start: float = 1.0,
        kink_arrow_x_end: float = 100.0,
        kink_arrow_y: float = 0.9,
        missing_structure_x_offset: float = 10.0,
        missing_structure_y_offset: float = 0.05,
        missing_structure_alpha: float = 0.15,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (12.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self._selection_cache: tuple | None = None

        self.selection_names = add_data_selection_widgets(
            self, results, defaults={**ACTIVITY_SELECTION_DEFAULTS, **selection_defaults}
        )

        self.add_integer("idx_cross", value=idx_cross, min=0, max=len(results.sessions) - 1)
        self.add_integer("num_cross_show", value=num_cross_show, min=1, max=100)
        self.add_boolean("plot_energy", value=plot_energy)
        self.add_boolean("weighted", value=weighted)
        self.add_selection("curve_mode", value=curve_mode, options=["average", "by_session"])
        self.add_selection("plot_style", value=plot_style, options=["each", "errorPlot"])
        self.add_boolean("hide_error", value=hide_error)
        self.add_integer("skip_sessions", value=skip_sessions, min=0, max=len(results.sessions))
        self.add_selection("curve_smooth_kind", value=curve_smooth_kind, options=SMOOTH_KINDS)
        self.add_float("curve_smooth_width", value=curve_smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_float("kink_threshold", value=kink_threshold, min=0.0, max=1.0, step=0.001)
        self.add_selection("distribution_metric", value=distribution_metric, options=DISTRIBUTION_METRICS)

        # ax[1] metric-explainer decorations (colors fixed in CROSS_METRIC_COLORS).
        self.add_boolean("show_decorations", value=show_decorations)
        self.add_boolean("show_marker_labels", value=show_marker_labels)
        self.add_float("arrow_linewidth", value=arrow_linewidth, min=0.1, max=10.0, step=0.1)
        self.add_float("arrow_head_size", value=arrow_head_size, min=0.05, max=2.0, step=0.05)
        self.add_float("max_arrow_x", value=max_arrow_x, min=1.0, max=1000.0, step=0.1)
        self.add_float("max_arrow_y_start", value=max_arrow_y_start, min=0.0, max=1.5, step=0.01)
        self.add_float("max_arrow_y_end", value=max_arrow_y_end, min=0.0, max=1.5, step=0.01)
        self.add_float("kink_arrow_x_start", value=kink_arrow_x_start, min=1.0, max=1000.0, step=0.1)
        self.add_float("kink_arrow_x_end", value=kink_arrow_x_end, min=1.0, max=1000.0, step=0.1)
        self.add_float("kink_arrow_y", value=kink_arrow_y, min=0.0, max=1.5, step=0.01)
        self.add_float("missing_structure_x_offset", value=missing_structure_x_offset, min=0.0, max=1000.0, step=0.1)
        self.add_float("missing_structure_y_offset", value=missing_structure_y_offset, min=0.0, max=1.0, step=0.01)
        self.add_float("missing_structure_alpha", value=missing_structure_alpha, min=0.0, max=1.0, step=0.01)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0, step=0.5)

        # Everything that changes the arrays, not just their styling. Pure style knobs (the
        # decorations, fontsize, num_cross_show) are left out: syd re-runs plot for those anyway.
        for name in (
            *self.selection_names,
            "weighted",
            "curve_smooth_kind",
            "curve_smooth_width",
            "kink_threshold",
            "distribution_metric",
        ):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _select(self, state) -> dict:
        """``results.sel`` output for the current data selection, memoised on that selection."""
        selection = data_selection(state, self.results, self.selection_names)
        key = tuple(sorted(selection.items(), key=lambda item: item[0]))
        if self._selection_cache is None or self._selection_cache[0] != key:
            self._selection_cache = (key, self.results.sel(**selection))
        return self._selection_cache[1]

    def refresh_data(self, state):
        """Re-select the cross matrices and rebuild every curve and per-session metric."""
        out = self._select(state)
        cross = out["cross"]
        self._cross = cross
        self._energy = cross**2

        curves, valid_full_dims = energy_on_full(cross, state["curve_smooth_kind"], state["curve_smooth_width"])
        self._max_energy = np.nanmax(curves, axis=1)
        self._kink_position = kink_positions(curves, state["kink_threshold"])
        # The metric is measured against the variance of the full dims the cross matrix covers,
        # while the weighted overlap below uses every stored full dim.
        self._distribution = distribution_metric(
            state["distribution_metric"],
            curves,
            valid_full_dims,
            out["variance_activity"][:, : cross.shape[1]],
        )

        if state["weighted"]:
            self._panel_fraction = weighted_fraction(cross, out["variance_activity"])
            self._panel_ylabel = "Variance-Weighted Fraction\nFull Variance Recovered"
        else:
            self._panel_fraction = curves
            self._panel_ylabel = "Fraction Full Variance\nCaptured By Placefields"

        self._panel_fraction_avg = average_by_mouse(self._panel_fraction, self.results.mouse_names)

        mouse_names, unique_mice = self.results.mouse_names, self.results.unique_mice
        self._max_energy_by_session = stack_sessions_by_mouse(self._max_energy, mouse_names, unique_mice)
        self._distribution_by_session = stack_sessions_by_mouse(self._distribution, mouse_names, unique_mice)
        self._kink_by_session = stack_sessions_by_mouse(self._kink_position, mouse_names, unique_mice)

        # Example-session order: most diagonally-aligned session first.
        big_first_10 = np.mean(np.diagonal(self._energy, axis1=1, axis2=2)[:, :10], axis=1)
        self._idx_aligned = np.argsort(-big_first_10)

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        panel_fraction = self._panel_fraction

        fig, ax = self.new_subplots(1, 4, figsize=self.figsize, layout="constrained", width_ratios=[1, 1.2, 0.9, 0.9])

        # ---- ax[0]: example cross matrix ----
        idx_plot = self._idx_aligned[state["idx_cross"]]
        if state["plot_energy"]:
            imshow_data = self._energy[idx_plot][:100, :100]
            cmap = "gray_r"
            vmin = 0
        else:
            imshow_data = self._cross[idx_plot][:100, :100]
            cmap = "bwr"
            vmin = -1

        xbounds0 = [0, state["num_cross_show"]]
        ybounds0 = [state["num_cross_show"], 0]
        ax[0].imshow(imshow_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=1, extent=[0, 100, 100, 0])
        ax[0].set_xlabel("Placefield Dimension", fontsize=fontsize)
        ax[0].set_ylabel("Full Dimension", fontsize=fontsize)
        ax[0].set_xlim([-0.5, state["num_cross_show"] + 0.5])
        ax[0].set_ylim([state["num_cross_show"] + 0.5, -0.5])

        # ---- ax[1]: population overlap curves ----
        xvals = np.arange(panel_fraction.shape[1]) + 1
        xbounds1 = [1, panel_fraction.shape[1] + 1]
        if state["curve_mode"] == "by_session":
            by_session, kept, session_colors, show_idx = by_session_groups(
                panel_fraction, self.results.mouse_names, self.results.unique_mice, state["skip_sessions"]
            )
            finite_chunks = []
            for color_idx in show_idx:
                session_data = by_session[:, kept[color_idx], :]
                render_curve_group(
                    ax[1],
                    xvals,
                    session_data,
                    session_colors[color_idx],
                    state["plot_style"],
                    hide_error=state["hide_error"],
                    linewidth=1.5,
                )
                finite_chunks.append(np.nanmean(session_data, axis=0))
            finite_curve = np.concatenate(finite_chunks) if finite_chunks else np.array([])
            if len(kept):
                draw_session_colorbar_inset(ax[1], kept[0], kept[-1], SESSION_CMAP, fontsize, _COLORBAR_INSET_BOUNDS)
        else:
            render_curve_group(
                ax[1],
                xvals,
                self._panel_fraction_avg,
                "k",
                state["plot_style"],
                hide_error=state["hide_error"],
                linewidth=2.0,
            )
            mean_curve = np.nanmean(self._panel_fraction_avg, axis=0)
            finite_curve = mean_curve[np.isfinite(mean_curve)]
        ax[1].set_xlabel("Full Dimension", fontsize=fontsize)
        ax[1].set_ylabel(self._panel_ylabel, fontsize=fontsize)
        ax[1].set_xscale("log")
        ymax = 1.0
        if state["weighted"] and finite_curve.size:
            ymax = max(1.0, float(finite_curve.max()))
        ybounds1 = [0, ymax]
        ax[1].set_xlim(xbounds1)
        ax[1].set_ylim(ybounds1)

        if state["show_decorations"]:
            self._draw_decorations(ax[1], state, xvals, xbounds1, ybounds1, fontsize)

        # ---- ax[2] / ax[3]: per-mouse session trajectories of the three metrics ----
        session_xvals = np.arange(self._max_energy_by_session.shape[1])

        def plot_session_metric(axis, values, color, label):
            render_curve_group(axis, session_xvals, values, color, state["plot_style"], label=label, min_support=2)
            return supported_xmax(session_xvals, values, min_support=2)

        ax2_xmax = max(
            plot_session_metric(ax[2], self._max_energy_by_session, CROSS_METRIC_COLORS["max_captured"], "max variance"),
            plot_session_metric(ax[2], self._distribution_by_session, CROSS_METRIC_COLORS["missing_structure"], "missing fraction"),
        )
        ax[2].set_xlabel("Session #", fontsize=fontsize)
        ax[2].legend(fontsize=fontsize, loc="lower left", frameon=False, handlelength=0.8, handletextpad=0.5)

        ax3_xmax = plot_session_metric(ax[3], self._kink_by_session, CROSS_METRIC_COLORS["kink"], "kink dimension")
        ax[3].set_xlabel("Session #", fontsize=fontsize)
        ax[3].legend(fontsize=fontsize, loc="upper left", frameon=False, handlelength=0.8, handletextpad=0.5)

        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=xbounds0,
            ybounds=ybounds0,
            tick_fontsize=fontsize,
        )
        format_spines(
            ax[1],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=xbounds1,
            ybounds=ybounds1,
            xticks=[1, 10, 100, 1000],
            yticks=[0, round(ymax, 2)],
            tick_fontsize=fontsize,
        )
        ylim = ax[2].get_ylim()
        ax[2].set_ylim([min(ylim[0], 0.0), max(ylim[1], 1.0)])
        format_spines(
            ax[2],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[0, ax2_xmax],
            ybounds=[0, 1],
            tick_fontsize=fontsize,
        )

        ylim = ax[3].get_ylim()
        format_spines(
            ax[3],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[0, ax3_xmax],
            ybounds=[0, np.floor(ylim[1])],
            tick_fontsize=fontsize,
        )
        # format_spines uses tick_params, which doesn't reach minor ticks on ax[1]'s log x-axis.
        ax[1].tick_params(axis="both", which="both", labelsize=fontsize)
        return fig

    def _draw_decorations(self, ax, state, xvals, xbounds, ybounds, fontsize: float) -> None:
        """Overlay the arrows/patch that name the three metrics tracked by ax[2] and ax[3]."""
        arrow_style = f"-|>,head_width={state['arrow_head_size']},head_length={2 * state['arrow_head_size']}"
        # Max captured: vertical arrow at a fixed x, from y_start up to y_end.
        ax.annotate(
            "",
            xy=(state["max_arrow_x"], state["max_arrow_y_end"]),
            xytext=(state["max_arrow_x"], state["max_arrow_y_start"]),
            arrowprops=dict(arrowstyle=arrow_style, color=CROSS_METRIC_COLORS["max_captured"], lw=state["arrow_linewidth"]),
            annotation_clip=False,
        )
        # Kink: horizontal arrow at a fixed y, from x_start to x_end.
        ax.annotate(
            "",
            xy=(state["kink_arrow_x_end"], state["kink_arrow_y"]),
            xytext=(state["kink_arrow_x_start"], state["kink_arrow_y"]),
            arrowprops=dict(arrowstyle=arrow_style, color=CROSS_METRIC_COLORS["kink"], lw=state["arrow_linewidth"]),
            annotation_clip=False,
        )
        # Missing structure: patch hugging the curve envelope (max over all sessions at each x),
        # offset right and up from it, filling to the panel's top-right corner.
        envelope = np.nanmax(self._panel_fraction, axis=0)
        patch_x_start = xbounds[0] + state["missing_structure_x_offset"]
        patch_bottom = np.minimum(envelope + state["missing_structure_y_offset"], ybounds[1])
        ax.fill_between(
            xvals,
            patch_bottom,
            ybounds[1],
            where=xvals >= patch_x_start,
            color=CROSS_METRIC_COLORS["missing_structure"],
            alpha=state["missing_structure_alpha"],
            linewidth=0,
        )

        if not state["show_marker_labels"]:
            return
        # "max": vertical text to the right of the max-captured arrow, starting at its bottom.
        ax.text(
            state["max_arrow_x"] * 1.15,
            state["max_arrow_y_start"],
            "max",
            ha="left",
            va="bottom",
            rotation=90,
            color=CROSS_METRIC_COLORS["max_captured"],
            fontsize=fontsize,
        )
        # "kink": horizontal text below the kink arrow, right-aligned to its right end.
        ax.text(
            state["kink_arrow_x_end"],
            state["kink_arrow_y"] - 0.05,
            "kink",
            ha="right",
            va="top",
            color=CROSS_METRIC_COLORS["kink"],
            fontsize=fontsize,
        )
        # "missing structure": top-right corner of the panel.
        ax.text(
            0.98,
            0.98,
            "missing\nstructure",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="black",
            fontsize=fontsize,
        )
