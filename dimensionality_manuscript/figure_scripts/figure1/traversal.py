"""One ROI's place-field traversals in time: activity, prediction, and error, frame by frame."""

import numpy as np

from dimensionality_manuscript.figure_scripts import session_cache
from dimensionality_manuscript.figure_scripts.panels import FigureViewer

from ._shared import draw_vertical_colorscale, style_axis


class TraversalFocus(FigureViewer):
    """One ROI's traversals of its place field, aligned on the peak, in frames rather than position.

    Each row of the top panels is one pass through the place field, ``width`` frames either side
    of the peak; the bottom row is the trial average of each. Where
    :class:`~.placefield.PlaceFieldPredictionFocus` shows the same comparison in position units
    (where every trial's prediction is the same row), this panel keeps the time axis, so the
    error picks up mistimed as well as mis-sized traversals.

    The ROI selector offers only ROIs that clear both filters in the selected environment, so
    changing ``env`` or either threshold can move the selection.

    Parameters
    ----------
    session : B2Session
        Session the maps, reliability, and fraction-active are computed from (cached).
    roi : int
        Initial ROI, in session-filtered ROI coordinates. Ignored if it fails the filters.
    env : int
        Index into ``env_maps.environments``.
    reliability_threshold, fraction_active_threshold : float
        An ROI is offered only if it exceeds both in the selected environment.
    width : int
        Half-width of a traversal, in frames.
    vmax : float
        Upper limit (in sigma) of the gray_r panels; the error panel uses ``+/- vmax``.
    fontsize : float
        Font size of every text element.
    figsize : tuple[float, float]
        Figure size in inches.
    """

    def __init__(
        self,
        session,
        *,
        roi: int = 0,
        env: int = 0,
        reliability_threshold: float = 0.7,
        fraction_active_threshold: float = 0.5,
        width: int = 20,
        vmax: float = 12.0,
        fontsize: float = 12.0,
        figsize: tuple[float, float] = (8.0, 5.0),
    ):
        self.session = session
        self.figsize = figsize
        self.smp = session_cache.get_smp(session)
        self.env_maps = session_cache.get_env_maps(session)
        self.reliability = session_cache.get_reliability(session)
        self.fraction_active = session_cache.get_fraction_active(session)
        self.spks = session_cache.get_spks(session, params=self.smp.params)

        self.add_integer("env", value=env, min=0, max=len(self.env_maps.environments) - 1)
        self.add_float("reliability_threshold", value=reliability_threshold, min=0.0, max=1.0)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0)
        # Options are narrowed to the ROIs passing the filters by update_filters, below.
        self.add_selection("roi", value=roi, options=[roi])
        self.add_integer("width", value=width, min=2, max=100)
        self.add_float("vmax", value=vmax, min=1.0, max=20.0)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)

        self.on_change(["env", "reliability_threshold", "fraction_active_threshold"], self.update_filters)
        self.on_change(["env", "roi", "width"], self.refresh_data)
        self.update_filters(self.state)

    def update_filters(self, state):
        """Offer only ROIs that clear both filters in the selected environment."""
        env = state["env"]
        idx_reliable = self.reliability.values[env] > state["reliability_threshold"]
        idx_active = self.fraction_active[env] > state["fraction_active_threshold"]
        options = [int(roi) for roi in np.where(idx_reliable & idx_active)[0]]
        if not options:
            raise ValueError(f"No ROI of environment {env} passes both thresholds.")
        current = state["roi"] if state["roi"] in options else options[0]
        self.update_selection("roi", value=current, options=options)
        self.refresh_data({**state, "roi": current})

    def refresh_data(self, state):
        """Extract the ROI's traversals and their place-field predictions (the expensive step)."""
        width = state["width"]
        self.traversals, self.pred_travs = self.smp.get_traversals(state["roi"], state["env"], spks=self.spks, width=width)
        self.xvals = np.arange(width * 2 + 1) - width
        self.avg_traversal = np.nanmean(self.traversals, axis=0)
        self.avg_pred_traversal = np.nanmean(self.pred_travs, axis=0)
        self.rms_error = np.sqrt(np.nanmean((self.pred_travs - self.traversals) ** 2, axis=0))

    def plot(self, state):
        width = state["width"]
        vmax = state["vmax"]
        fontsize = state["fontsize"]
        traversals = self.traversals
        num_traversals = traversals.shape[0]
        yavgmax = int(np.ceil(np.max([np.nanmax(self.avg_traversal), np.nanmax(self.avg_pred_traversal), np.nanmax(self.rms_error)]) * 1.05))

        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(2, 5, width_ratios=[5, 5, 5, 1, 1], height_ratios=[6, 1])
        ax_traversals = fig.add_subplot(gs[0, 0])
        ax_pred_travs = fig.add_subplot(gs[0, 1])
        ax_error = fig.add_subplot(gs[0, 2])
        ax_colorbar = fig.add_subplot(gs[0, 3])
        ax_cbar_error = fig.add_subplot(gs[0, 4])
        ax_avg_traversal = fig.add_subplot(gs[1, 0])
        ax_avg_pred_travs = fig.add_subplot(gs[1, 1])
        ax_rms_error = fig.add_subplot(gs[1, 2])

        extent = (-width, width, num_traversals, 0)

        def draw_map(ax, values, cmap, vlow, vhigh, ylabel):
            ax.imshow(values, interpolation="none", aspect="auto", cmap=cmap, vmin=vlow, vmax=vhigh, extent=extent)
            ax.set_xlim(-width, width)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            style_axis(
                ax,
                fontsize=fontsize,
                xbounds=[-width, width],
                ybounds=[0, num_traversals],
                xticks=[],
                yticks=[],
                spines_visible=["left"],
            )

        def draw_average(ax, values, spines_visible):
            ax.plot(self.xvals, values, color="k", linewidth=1.5)
            ax.set_xlim(-width, width)
            ax.set_xlabel("Frames", fontsize=fontsize)
            style_axis(
                ax,
                fontsize=fontsize,
                xbounds=[-width, width],
                ybounds=[0, yavgmax],
                xticks=[-width, width],
                yticks=[0, yavgmax] if "left" in spines_visible else [],
                spines_visible=spines_visible,
            )

        draw_map(ax_traversals, traversals, "gray_r", 0, vmax, "PF Traversals\n(Deconvolved)")
        draw_map(ax_pred_travs, self.pred_travs, "gray_r", 0, vmax, "(PF Pred.)")
        draw_map(ax_error, self.pred_travs - traversals, "bwr", -vmax, vmax, "(Error)")

        draw_average(ax_avg_traversal, self.avg_traversal, ["bottom", "left"])
        ax_avg_traversal.set_ylabel("Avg", fontsize=fontsize)
        draw_average(ax_avg_pred_travs, self.avg_pred_traversal, ["bottom"])
        draw_average(ax_rms_error, self.rms_error, ["bottom"])
        ax_rms_error.text(-width, yavgmax, "RMS\nError", fontsize=fontsize, ha="left", va="top", color="k")

        draw_vertical_colorscale(
            ax_colorbar,
            "gray_r",
            low_label="0",
            high_label=f"{int(vmax)}",
            fontsize=fontsize,
            ylabel=r"Fluorescence ($\sigma$)",
        )
        draw_vertical_colorscale(
            ax_cbar_error,
            "bwr",
            low_label=f"-{int(vmax)}",
            high_label=f"{int(vmax)}",
            low_color="w",
            fontsize=fontsize,
            ylabel=r"Error ($\sigma$)",
        )
        return fig
