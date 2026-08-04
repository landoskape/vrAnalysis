"""Per-mouse cross-spectrum diagnostic: one mouse's sessions, curve by curve."""

import numpy as np
from matplotlib import pyplot as plt

from vrAnalysis.helpers.plotting import format_spines
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)

from ._selection import ACTIVITY_SELECTION_DEFAULTS
from ._curves import (
    DISTRIBUTION_METRICS,
    SESSION_CMAP,
    SMOOTH_KINDS,
    distribution_metric,
    energy_on_full,
    kink_positions,
)

# Axis label for each distribution metric, in this panel's sign convention.
_DISTRIBUTION_LABELS = {
    "gini": "Gini Equality",
    "weighted_missing": "Weighted Missing",
    "missing_structure": "Missing Structure",
}


class SubspaceCrossPerMouseViewer(FigureViewer):
    """Per-mouse cross-spectrum figure.

    ``ax[0]`` shows the cross energy heat map of one chosen session. ``ax[1]`` shows, for every
    session of the chosen mouse, the fraction of full-activity variance captured by placefields,
    color-coded in session order with ``coolwarm``. ``ax[2]`` tracks that mouse's max energy and
    distribution metric over sessions, with the kink position on a twinned right-hand axis.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``SubspaceConfig`` results providing ``param_axes``, ``sel``, ``unique_mice``,
        ``mouse_names`` and ``session_ids``.
    mouse : str or None
        Mouse to show. Defaults to the first mouse in ``results.unique_mice``.
    session : int
        Index (within the chosen mouse's sessions) of the example cross matrix in ``ax[0]``.
    num_cross_show : int
        Number of cross dimensions to show in ``ax[0]``.
    curve_smooth_kind : {"none", "boxcar", "gaussian", "median"}
        Linear NaN-aware smoothing applied to each session's fraction curve. ``"median"`` is
        edge/kink-preserving; ``"none"`` disables smoothing.
    curve_smooth_width : float
        Boxcar/median full-width in dim units; the Gaussian uses ``sigma = curve_smooth_width / 2``.
    kink_threshold : float
        Threshold (fraction of max) for the kink position metric.
    distribution_metric : {"gini", "weighted_missing", "missing_structure"}
        Metric shown with max energy. Missing structure is the mean uncaptured fraction over valid
        full dimensions; weighted missing is the same fraction weighted by each full dimension's
        activity variance; gini is reported here as the *equality* measure (1 - Gini coefficient),
        the opposite sign convention to :class:`~.crossspace.SubspaceCrossspaceViewer`.
    fontsize : float
        Font size for axis labels, titles and tick labels. The legend is drawn at 0.8x this.
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
        session: int = 0,
        num_cross_show: int = 30,
        curve_smooth_kind: str = "none",
        curve_smooth_width: float = 3.0,
        kink_threshold: float = 0.95,
        distribution_metric: str = "gini",
        fontsize: float = 10.0,
        figsize: tuple[float, float] = (6.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize

        self.selection_names = add_data_selection_widgets(
            self, results, defaults={**ACTIVITY_SELECTION_DEFAULTS, **selection_defaults}
        )

        mouse = mouse if mouse is not None else results.unique_mice[0]
        self.add_selection("mouse", value=mouse, options=list(results.unique_mice))
        self.add_integer("session", value=session, min=0, max=self._n_sessions(mouse))
        self.add_integer("num_cross_show", value=num_cross_show, min=1, max=100)
        self.add_selection("curve_smooth_kind", value=curve_smooth_kind, options=SMOOTH_KINDS)
        self.add_float("curve_smooth_width", value=curve_smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_float("kink_threshold", value=kink_threshold, min=0.0, max=1.0, step=0.01)
        self.add_selection("distribution_metric", value=distribution_metric, options=DISTRIBUTION_METRICS)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0, step=0.5)

        self.on_change("mouse", self.update_mouse)
        for name in (*self.selection_names, "mouse", "curve_smooth_kind", "curve_smooth_width", "kink_threshold", "distribution_metric"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _n_sessions(self, mouse: str) -> int:
        """Highest valid ``session`` index for ``mouse`` (0 when it has none)."""
        return max(int(np.sum(self.results.mouse_names == mouse)) - 1, 0)

    def update_mouse(self, state):
        """Clamp ``session`` to the number of sessions the selected mouse has."""
        self.update_integer("session", max=self._n_sessions(state["mouse"]))

    def refresh_data(self, state):
        """Re-select the chosen mouse's cross matrices and rebuild its curves and metrics."""
        out = self.results.sel(
            mouse=state["mouse"],
            squeeze_ones=False,
            **data_selection(state, self.results, self.selection_names),
        )
        cross = out["cross"]
        self._energy = cross**2
        curves, valid_full_dims = energy_on_full(cross, state["curve_smooth_kind"], state["curve_smooth_width"])
        self._energy_on_full = curves
        self._max_energy = np.nanmax(curves, axis=1)
        self._kink_position = kink_positions(curves, state["kink_threshold"])
        self._distribution = distribution_metric(
            state["distribution_metric"],
            curves,
            valid_full_dims,
            out["variance_activity"][:, : cross.shape[1]],
            gini_equality=True,
        )
        self._distribution_label = _DISTRIBUTION_LABELS[state["distribution_metric"]]
        self._session_ids = [sid for sid, m in zip(self.results.session_ids, self.results.mouse_names) if m == state["mouse"]]

    def plot(self, state):
        fontsize = state["fontsize"]
        energy_on_full_curves = self._energy_on_full
        n_sess = self._energy.shape[0]
        sess = min(state["session"], n_sess - 1)

        fig, ax = self.new_subplots(1, 3, figsize=self.figsize, layout="constrained")

        # ---- ax[0]: cross energy heat map of the chosen session ----
        ax[0].imshow(self._energy[sess][:100, :100].T, cmap="gray_r", aspect="auto", vmin=0, vmax=1, extent=[0, 100, 100, 0])
        ax[0].set_xlabel("Full Dim.", fontsize=fontsize)
        ax[0].set_ylabel("Placefield Dim.", fontsize=fontsize)
        xbounds0 = [0, state["num_cross_show"]]
        ybounds0 = [state["num_cross_show"], 0]
        ax[0].set_xlim([-0.5, state["num_cross_show"] + 0.5])
        ax[0].set_ylim([state["num_cross_show"] + 0.5, -0.5])
        ax[0].set_title(self._session_ids[sess] if sess < len(self._session_ids) else state["mouse"], fontsize=fontsize)

        # ---- ax[1]: every session's fraction curve, colored by session order ----
        colors = plt.get_cmap(SESSION_CMAP)(np.linspace(0, 1, max(n_sess, 1)))
        xvals = np.arange(energy_on_full_curves.shape[1]) + 1
        for i in range(n_sess):
            ax[1].plot(xvals, energy_on_full_curves[i], color=colors[i], linewidth=1.0)
        ax[1].set_xlabel("Full Dim.", fontsize=fontsize)
        ax[1].set_ylabel("Fraction Captured\nBy Placefields", fontsize=fontsize)
        ax[1].set_xscale("log")
        xbounds1 = [1, energy_on_full_curves.shape[1] + 1]
        ybounds1 = [0, 1.0]
        ax[1].set_xlim(xbounds1)
        ax[1].set_ylim(ybounds1)

        # ---- ax[2]: the same session-index metrics, kink on a twinned axis ----
        session_xvals = range(len(self._kink_position))
        ax[2].plot(session_xvals, self._max_energy, color="k", linewidth=1.0, label="Max Energy")
        ax[2].plot(session_xvals, self._distribution, color="b", linewidth=1.0, label=self._distribution_label)
        kink_ax = ax[2].twinx()
        kink_ax.plot(session_xvals, self._kink_position, color="r", linewidth=1.0, label="Kink Position")
        ax[2].set_xlabel("Session Index", fontsize=fontsize)
        ax[2].set_ylabel(f"Max Energy / {self._distribution_label}", fontsize=fontsize)
        kink_ax.set_ylabel("Kink Position", color="r", fontsize=fontsize)
        kink_ax.tick_params(axis="y", colors="r", labelsize=fontsize)
        lines, labels = ax[2].get_legend_handles_labels()
        kink_lines, kink_labels = kink_ax.get_legend_handles_labels()
        ax[2].legend(lines + kink_lines, labels + kink_labels, fontsize=fontsize * 0.8)

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
            yticks=[0, 1],
            tick_fontsize=fontsize,
        )
        format_spines(
            ax[2],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            tick_fontsize=fontsize,
        )
        return fig
