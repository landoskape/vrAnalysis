"""Cross-spectrum figure pairing one example mouse with the cross-mouse summary."""

import numpy as np
from matplotlib import pyplot as plt

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
from ._curves import SESSION_CMAP, SMOOTH_KINDS, by_session_groups, energy_on_full, weighted_fraction


class SubspaceCrossspaceExampleViewer(FigureViewer):
    """Cross-spectrum viewer pairing one example mouse with the cross-mouse summary.

    ``ax[0]`` shows an example placefield-vs-full cross matrix from ``mouse`` (chosen by
    ``idx_cross`` from that mouse's sessions sorted by descending mean top-10 diagonal energy),
    transposed so full dimension runs along x like the other panels. ``ax[1]`` shows every session
    of the same mouse -- the variance overlap between each full PC and the placefield span,
    color-coded in session order with ``coolwarm``. ``ax[2]`` shows the same curves for the whole
    population, either mouse-averaged or grouped by session number, and shares ``ax[1]``'s y-axis.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``SubspaceConfig`` results providing ``param_axes``, ``sel``, ``unique_mice``,
        ``mouse_names`` and ``sessions``.
    mouse : str or None
        Example mouse shown in ``ax[0]`` and ``ax[1]``. Defaults to the first mouse in
        ``results.unique_mice``.
    idx_cross : int
        Index (into the chosen mouse's sessions, sorted by descending mean top-10 diagonal energy)
        of the example cross matrix shown in ``ax[0]``.
    plot_energy : bool
        Show squared cross energy (``gray_r``, from 0) when True, otherwise the signed cross
        values (``bwr``, from -1) in ``ax[0]``.
    num_cross_show : int
        Number of cross dimensions to show in ``ax[0]``.
    weighted : bool
        Curve panels show the variance-weighted recovery ``Var(X P u_i)/λ_i`` when True
        (down-weights overlap onto low-variance full PCs), otherwise the unweighted subspace
        overlap ``||P u_i||²``.
    curve_mode : {"average", "by_session"}
        ``ax[2]`` only. ``"average"`` gives one curve per mouse; ``"by_session"`` instead groups
        curves by within-mouse session number, one color-coded group per session number
        (``coolwarm``, early to late), skipping session numbers backed by a single mouse.
    plot_style : {"each", "errorPlot"}
        ``ax[2]`` only. ``"each"`` draws each underlying curve thin plus a solid mean;
        ``"errorPlot"`` draws a mean +/- std band instead.
    hide_error : bool
        With ``plot_style="errorPlot"``, suppress the std band in ``ax[2]``.
    skip_sessions : int
        Only used when ``curve_mode="by_session"``. Thins out which session-number groups are
        drawn in ``ax[2]``; the first and last kept session number are always shown, and the
        colors still span the full kept range.
    curve_smooth_kind : {"none", "boxcar", "gaussian", "median"}
        Smoothing applied to the (unweighted) energy-on-full curves in both curve panels.
    curve_smooth_width : float
        Smoothing width in full-dimension units.
    fontsize : float
        Font size applied uniformly across every panel: axis labels, tick labels (major and
        minor, including the log-scale x-axes) and the colorbar-inset session-number labels.
    inset_x, inset_y, inset_width, inset_height : float
        Placement of ``ax[2]``'s session-number colorbar inset (only drawn when
        ``curve_mode="by_session"``), in that panel's axes-fraction coordinates. Values outside
        ``[0, 1]`` put the bar outside the panel; e.g. a negative ``inset_x`` moves it left of the
        panel's x limits, into the gap between ``ax[1]`` and ``ax[2]``.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the data-selection widgets (``smooth_width``,
        ``activity_parameters_name``, ``subspace_name``).
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        idx_cross: int = 0,
        plot_energy: bool = True,
        num_cross_show: int = 25,
        weighted: bool = False,
        curve_mode: str = "average",
        plot_style: str = "each",
        hide_error: bool = False,
        skip_sessions: int = 0,
        curve_smooth_kind: str = "none",
        curve_smooth_width: float = 3.0,
        fontsize: float = 9.0,
        inset_x: float = 0.05,
        inset_y: float = 0.04,
        inset_width: float = 0.6,
        inset_height: float = 0.075,
        figsize: tuple[float, float] = (9.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self._selection_cache: tuple | None = None

        self.selection_names = add_data_selection_widgets(
            self, results, defaults={**ACTIVITY_SELECTION_DEFAULTS, **selection_defaults}
        )

        mouse = mouse if mouse is not None else results.unique_mice[0]
        self.add_selection("mouse", value=mouse, options=list(results.unique_mice))
        self.add_integer("idx_cross", value=idx_cross, min=0, max=self._max_idx_cross(mouse))
        self.add_integer("num_cross_show", value=num_cross_show, min=1, max=100)
        self.add_boolean("plot_energy", value=plot_energy)
        self.add_boolean("weighted", value=weighted)
        self.add_selection("curve_mode", value=curve_mode, options=["average", "by_session"])
        self.add_selection("plot_style", value=plot_style, options=["each", "errorPlot"])
        self.add_boolean("hide_error", value=hide_error)
        self.add_integer("skip_sessions", value=skip_sessions, min=0, max=len(results.sessions))
        self.add_selection("curve_smooth_kind", value=curve_smooth_kind, options=SMOOTH_KINDS)
        self.add_float("curve_smooth_width", value=curve_smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0, step=0.5)

        # ax[2] session-number colorbar inset placement, in ax[2] axes-fraction coordinates.
        # Negative x moves it left of the panel's x limits (into the gap between ax[1] and ax[2]).
        self.add_float("inset_x", value=inset_x, min=-1.0, max=1.0, step=0.01)
        self.add_float("inset_y", value=inset_y, min=-1.0, max=1.0, step=0.01)
        self.add_float("inset_width", value=inset_width, min=0.01, max=1.5, step=0.01)
        self.add_float("inset_height", value=inset_height, min=0.01, max=1.0, step=0.005)

        self.on_change("mouse", self.update_mouse)
        for name in (*self.selection_names, "mouse", "weighted", "curve_smooth_kind", "curve_smooth_width"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _max_idx_cross(self, mouse: str) -> int:
        """Highest valid ``idx_cross`` for ``mouse`` (0 when it has no sessions)."""
        return max(int(np.sum(self.results.mouse_names == mouse)) - 1, 0)

    def update_mouse(self, state):
        """Clamp ``idx_cross`` to the number of sessions the selected mouse has."""
        self.update_integer("idx_cross", max=self._max_idx_cross(state["mouse"]))

    def _select(self, state) -> dict:
        """``results.sel`` output for the current data selection, memoised on that selection."""
        selection = data_selection(state, self.results, self.selection_names)
        key = tuple(sorted(selection.items(), key=lambda item: item[0]))
        if self._selection_cache is None or self._selection_cache[0] != key:
            self._selection_cache = (key, self.results.sel(**selection))
        return self._selection_cache[1]

    def refresh_data(self, state):
        """Re-select the cross matrices and rebuild the population and example-mouse curves."""
        out = self._select(state)
        cross = out["cross"]
        curves, _ = energy_on_full(cross, state["curve_smooth_kind"], state["curve_smooth_width"])

        if state["weighted"]:
            self._panel_fraction = weighted_fraction(cross, out["variance_activity"])
            self._panel_ylabel = "Weighted Variance Overlap"
        else:
            self._panel_fraction = curves
            self._panel_ylabel = "Variance Overlap"
        self._panel_fraction_avg = average_by_mouse(self._panel_fraction, self.results.mouse_names)

        # Example mouse (ax[0] and ax[1]); the example session is the idx_cross-th most
        # diagonally-aligned session of that mouse.
        mouse_mask = self.results.mouse_names == state["mouse"]
        self._mouse_energy = cross[mouse_mask] ** 2
        self._mouse_cross = cross[mouse_mask]
        self._mouse_fraction = self._panel_fraction[mouse_mask]
        big_first_10 = np.mean(np.diagonal(self._mouse_energy, axis1=1, axis2=2)[:, :10], axis=1)
        self._mouse_order = np.argsort(-big_first_10)

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        panel_fraction = self._panel_fraction
        n_sess_mouse = self._mouse_energy.shape[0]
        idx_plot = self._mouse_order[min(state["idx_cross"], n_sess_mouse - 1)]

        fig, ax = self.new_subplots(1, 3, figsize=self.figsize, layout="constrained", width_ratios=[1, 1.2, 1.2])
        ax[2].sharey(ax[1])

        # ---- ax[0]: example cross matrix ----
        # cross[i, j] = <full PC i, placefield dim j> (rows are full dims), so it is transposed
        # here to put full dimension on x, matching the x-axis of ax[1]/ax[2].
        if state["plot_energy"]:
            imshow_data = self._mouse_energy[idx_plot][:100, :100]
            cmap = "gray_r"
            vmin = 0
        else:
            imshow_data = self._mouse_cross[idx_plot][:100, :100]
            cmap = "bwr"
            vmin = -1

        xbounds0 = [0, state["num_cross_show"]]
        ybounds0 = [state["num_cross_show"], 0]
        ax[0].imshow(imshow_data.T, cmap=cmap, aspect="auto", vmin=vmin, vmax=1, extent=[0, 100, 100, 0])
        ax[0].set_xlabel("Full Dimension", fontsize=fontsize)
        ax[0].set_ylabel("Placefield Dimension", fontsize=fontsize)
        ax[0].set_xlim([-0.5, state["num_cross_show"] + 0.5])
        ax[0].set_ylim([state["num_cross_show"] + 0.5, -0.5])

        xvals = np.arange(panel_fraction.shape[1]) + 1
        xbounds_curves = [1, panel_fraction.shape[1] + 1]
        xticks = [1, 10, 100, 1000]

        # ---- ax[1]: every session of the example mouse, colored by session order ----
        mouse_colors = plt.get_cmap(SESSION_CMAP)(np.linspace(0, 1, max(n_sess_mouse, 1)))
        for i in range(n_sess_mouse):
            ax[1].plot(xvals, self._mouse_fraction[i], color=mouse_colors[i], linewidth=1.0)
        ax[1].set_xlabel("Full Dimension", fontsize=fontsize)
        ax[1].set_ylabel(self._panel_ylabel, fontsize=fontsize)
        ax[1].set_xscale("log")
        ax[1].set_xlim(xbounds_curves)

        # ---- ax[2]: population curves ----
        if state["curve_mode"] == "by_session":
            by_session, kept, session_colors, show_idx = by_session_groups(
                panel_fraction, self.results.mouse_names, self.results.unique_mice, state["skip_sessions"]
            )
            finite_chunks = []
            for color_idx in show_idx:
                session_data = by_session[:, kept[color_idx], :]
                render_curve_group(
                    ax[2],
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
                bounds = [state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]]
                draw_session_colorbar_inset(ax[2], kept[0], kept[-1], SESSION_CMAP, fontsize, bounds)
        else:
            render_curve_group(
                ax[2],
                xvals,
                self._panel_fraction_avg,
                "k",
                state["plot_style"],
                hide_error=state["hide_error"],
                linewidth=2.0,
            )
            mean_curve = np.nanmean(self._panel_fraction_avg, axis=0)
            finite_curve = mean_curve[np.isfinite(mean_curve)]
        ax[2].set_xlabel("Full Dimension", fontsize=fontsize)
        ax[2].set_xscale("log")
        ax[2].set_xlim(xbounds_curves)

        # ax[1]/ax[2] share the y-axis, so one limit covers both panels.
        ymax = 1.0
        if state["weighted"]:
            candidates = [1.0]
            if np.isfinite(self._mouse_fraction).any():
                candidates.append(float(np.nanmax(self._mouse_fraction)))
            if finite_curve.size:
                candidates.append(float(finite_curve.max()))
            ymax = max(candidates)
        ybounds = [0, ymax]
        ax[1].set_ylim(ybounds)

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
            xbounds=xbounds_curves,
            ybounds=ybounds,
            xticks=xticks,
            yticks=[0, round(ymax, 2)],
            tick_fontsize=fontsize,
        )
        # ax[2] borrows ax[1]'s y-axis, so it only gets a bottom spine and no y ticks/labels.
        # Its ticks are hidden with tick_params rather than set_yticks, which would propagate
        # through the shared axis and strip ax[1]'s y ticks too.
        format_spines(
            ax[2],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["bottom"],
            xbounds=xbounds_curves,
            xticks=xticks,
            tick_fontsize=fontsize,
        )
        ax[2].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        # format_spines uses tick_params, which doesn't reach minor ticks on log x-axes.
        for axis in (ax[1], ax[2]):
            axis.tick_params(axis="both", which="both", labelsize=fontsize)
        return fig
