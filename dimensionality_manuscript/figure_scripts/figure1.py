from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from syd import Viewer

from vrAnalysis.helpers import sort_by_preferred_environment, vectorRSquared, edge2center
from vrAnalysis.helpers.plotting import format_spines, errorPlot, beeswarm
from vrAnalysis.helpers.vrsupport import _jit_reliability_loo
from vrAnalysis.sessions import B2Session
from vrAnalysis.processors import spkmaps as SMPs
from vrAnalysis.processors.support import median_zscore
from vrAnalysis.processors.placefields import get_frame_behavior, get_placefield, get_placefield_prediction
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.spkmaps import Maps, Reliability

from dimensionality_manuscript.configs.pfpred_quality import PFPredQualityConfig, _kde_r2
from dimensionality_manuscript.configs.behavior_speed_env import ENV_REWARD_MAP, REFERENCE_ENV_LENGTH_CM, WINDOW_FRACTION
from dimensionality_manuscript import ResultsAggregator, average_by_mouse

plt.rcParams["font.size"] = 18

EXAMPLE_MOUSE_NAME = "ATL027"
EXAMPLE_DATE = "2023-07-27"
EXAMPLE_SESSION_ID = "701"
EXAMPLE_SPKS_TYPE = "sigrebase"
EXAMPLE_ENV = 0
EXAMPLE_ROI = 96
_PFPRED_KDE_GRID = PFPredQualityConfig().kde_grid


def _seed_roi_filtered_viewer(
    viewer: Viewer,
    *,
    env: int,
    roi: int,
    reliability_threshold: float,
    fraction_active_threshold: float,
    vmax: float,
) -> None:
    """
    Apply caller kwargs to a ROI-filtered Syd viewer after ``update_filters`` may have reset ROI.
    """
    viewer.update_integer("env", value=env)
    viewer.update_float("reliability_threshold", value=reliability_threshold)
    viewer.update_float("fraction_active_threshold", value=fraction_active_threshold)
    viewer.update_selection("roi", value=roi)
    viewer.update_float("vmax", value=vmax)


def _r2_placefield_arrays(session: B2Session, smp: SMPs.SpkmapProcessor, idx_env: int):
    """
    Compute valid-frame activity, PF predictions, and per-ROI R² for one environment.

    Returns
    -------
    spks_valid, pfpred_valid, r2, reliability
    """
    spks = session.spks[:, session.idx_rois]
    spks = median_zscore(spks, median_subtract=not session.zero_baseline_spks)
    reliability = smp.get_reliability()
    placefield_prediction, extras = smp.get_placefield_prediction()
    idx_best_environment = extras["frame_environment_index"] == idx_env
    idx_keep = extras["idx_valid"] & idx_best_environment
    spks_valid = spks[idx_keep]
    pfpred_valid = placefield_prediction[idx_keep]
    r2 = vectorRSquared(pfpred_valid, spks_valid, axis=0)
    r2[r2 < -1] = np.nan
    return spks_valid, pfpred_valid, r2, reliability


def _add_colorscale_inset(ax, cmap_name, right_label, left_label=None, right_color="w", left_color="k"):
    """Add a horizontal colorscale inset to the bottom-right of an axes.

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
    right_color, left_color : str
        Text colors for the right and left labels.
    """
    colors = mpl.colormaps[cmap_name](np.linspace(0, 1, 255))[None, :, :]  # (1, 255, 4)
    axins = ax.inset_axes([0.72, 0.10, 0.25, 0.15])
    axins.imshow(colors, aspect="auto")
    axins.set_xticks([])
    axins.set_yticks([])
    # for spine in axins.spines.values():
    #     spine.set_visible(False)
    axins.text(0.97, 0.5, right_label, transform=axins.transAxes, ha="right", va="center", color=right_color, fontsize=10)
    if left_label is not None:
        axins.text(0.03, 0.5, left_label, transform=axins.transAxes, ha="left", va="center", color=left_color, fontsize=10)


class StackedRasterFocus(Viewer):
    """Interactive stacked raster: activity, PF prediction, residuals."""

    def __init__(
        self,
        session: B2Session,
        smp: SMPs.SpkmapProcessor,
        spks: np.ndarray,
        placefield_prediction: np.ndarray,
        extras: dict,
        figsize: tuple[float, float],
        xslice_start: int,
        xslice_stop: int,
    ):
        self.session = session
        self.smp = smp
        self.spks = spks
        self.placefield_prediction = placefield_prediction
        self.extras = extras
        self.figsize = figsize
        self.add_selection("prediction_from", value="spkmap", options=["spkmap", "placefield"])
        self.add_boolean("use_reliable", value=True)
        self.add_float("reliability_threshold", value=0.7, min=0, max=1)
        self.add_float("vmax", value=6, min=1, max=20)
        self.add_integer("xslice_start", value=xslice_start, min=0, max=spks.shape[0] - 1)
        self.add_integer("xslice_stop", value=xslice_stop, min=1, max=spks.shape[0])
        self.add_boolean("show_position", value=False)
        self.add_float("position_height", value=0.5, min=0.1, max=3.0)
        self.add_float("env_gap", value=0.2, min=0.0, max=3.0)
        self.add_boolean("show_zero_sigma", value=False)

    def plot(self, state):
        spks_valid = self.spks[self.extras["idx_valid"]]
        prediction_from = state["prediction_from"]
        if prediction_from == "spkmap":
            pred = self.placefield_prediction[self.extras["idx_valid"]]
        elif prediction_from == "placefield":
            frame_behavior = get_frame_behavior(self.session)
            placefield = get_placefield(
                self.spks,
                frame_behavior,
                self.smp.dist_edges,
                self.smp.params.speed_threshold,
                average=True,
                smooth_width=1.0,
            )
            pred = get_placefield_prediction(placefield, frame_behavior)[0]
            pred = pred[self.extras["idx_valid"]]
        else:
            raise ValueError(f"Invalid prediction_from: {prediction_from}")

        if state["use_reliable"]:
            reliability = self.smp.get_reliability()
            idx_reliable = np.where(
                np.any(
                    np.stack([rval > state["reliability_threshold"] for rval in reliability.values], axis=0),
                    axis=0,
                )
            )[0]
        else:
            idx_reliable = np.arange(self.spks.shape[1])

        spks_valid = spks_valid[:, idx_reliable]
        pred_valid = pred[:, idx_reliable]
        idx_sort = sort_by_preferred_environment(self.smp, idx_rois=idx_reliable)

        xslice = slice(state["xslice_start"], state["xslice_stop"])
        num_frames = xslice.stop - xslice.start
        num_rois = spks_valid.shape[1]
        vmax = state["vmax"]

        show_position = state["show_position"]
        show_zero_sigma = state["show_zero_sigma"]
        position_height = state["position_height"]

        plt.close("all")
        fig = plt.figure(figsize=self.figsize, layout="constrained")
        if show_position:
            height_ratios = [1, 1, 1, position_height]
            gs = fig.add_gridspec(4, 1, height_ratios=height_ratios)
        else:
            gs = fig.add_gridspec(3, 1)
        ax = [fig.add_subplot(gs[0, 0])]
        ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0], sharey=ax[0]))
        ax.append(fig.add_subplot(gs[2, 0], sharex=ax[0], sharey=ax[0]))

        ax[0].imshow(spks_valid[xslice].T[idx_sort], aspect="auto", cmap="gray_r", vmin=0, vmax=vmax)
        ax[1].imshow(pred_valid[xslice].T[idx_sort], aspect="auto", cmap="gray_r", vmin=0, vmax=vmax)
        ax[2].imshow(
            spks_valid[xslice].T[idx_sort] - pred_valid[xslice].T[idx_sort],
            aspect="auto",
            cmap="bwr",
            vmin=-vmax,
            vmax=vmax,
        )

        for a in ax:
            a.set_xticks([])
            a.set_xticklabels([])
            a.set_yticks([])
            a.set_yticklabels([])

        panel_titles = (
            "Deconvolved Calcium Activity",
            "Prediction From Place Field",
            "Residuals",
        )
        for a, title in zip(ax, panel_titles):
            a.text(
                1.0,
                1.0,
                title,
                transform=a.transAxes,
                ha="right",
                va="top",
                color="black",
            )
        ax[0].set_ylabel("ROIs")
        ax[1].set_ylabel("ROIs")
        ax[2].set_ylabel("ROIs")

        for spine in ["top", "right", "bottom", "left"]:
            ax[0].spines[spine].set_visible(False)
            ax[1].spines[spine].set_visible(False)
            ax[2].spines[spine].set_visible(False)

        # Colorscale insets: gray_r on the first raster, bwr on the last raster.
        zero_label = r"$0\,\sigma$" if show_zero_sigma else None
        _add_colorscale_inset(ax[0], "gray_r", left_label=zero_label, right_label=rf"${int(vmax)}\,\sigma$", left_color="k", right_color="w")
        _add_colorscale_inset(
            ax[2], "bwr", left_label=rf"$-{int(vmax)}\,\sigma$", right_label=rf"$+{int(vmax)}\,\sigma$", left_color="w", right_color="w"
        )

        # Optional 4th panel: mouse position over the plotted frames.
        if show_position:
            ax_pos = fig.add_subplot(gs[3, 0], sharex=ax[0])
            num_bins = len(self.smp.dist_edges) - 1
            frame_position = self.extras["frame_position_index"][self.extras["idx_valid"]][xslice]
            frame_environment = self.extras["frame_environment_index"][self.extras["idx_valid"]][xslice]
            band = num_bins + state["env_gap"] * num_bins
            xpos_base = np.arange(num_frames, dtype=float)
            # Break the line at lap resets (position wraps back) so vertical jumps disappear.
            lap_resets = np.diff(frame_position) < -num_bins / 2
            breaks = np.where(lap_resets)[0] + 1
            cmap = mpl.colormaps["tab10"]
            # Plot each environment as its own colored line, offset vertically by a gapped band.
            # NaN-masking the other environments' frames breaks the line between environments.
            for env in np.unique(frame_environment[frame_environment >= 0]):
                y = frame_position.astype(float).copy()
                y[frame_environment != env] = np.nan
                y = y + env * band
                xins = np.insert(xpos_base, breaks, np.nan)
                yins = np.insert(y, breaks, np.nan)
                ax_pos.plot(xins, yins, color=cmap(int(env) % 10), linewidth=1)
            ax_pos.set_ylabel("Pos.")
            ax_pos.set_xticks([])
            ax_pos.set_yticks([])
            ax_pos.set_xlabel("Imaging Frames")
            for spine in ["top", "right", "bottom", "left"]:
                ax_pos.spines[spine].set_visible(False)
            ax_pos.set_yinverted(True)
        else:
            ax[2].set_xlabel("Imaging Frames")

        return fig


class TraversalFocus(Viewer):
    """Interactive PF traversal panels for one ROI and environment."""

    def __init__(
        self,
        smp: SMPs.SpkmapProcessor,
        env_maps: Maps,
        reliability: Reliability,
        fraction_active: np.ndarray,
    ):
        self.smp = smp
        self.env_maps = env_maps
        self.reliability = reliability
        self.fraction_active = fraction_active
        self.spks = median_zscore(
            smp.session.spks[:, smp.session.idx_rois],
            median_subtract=not smp.session.zero_baseline_spks,
        )

        self.num_rois = env_maps.spkmap[0].shape[0]
        self.num_envs = len(env_maps.environments)

        self.add_selection("roi", value=0, options=list(range(self.num_rois)))
        self.add_integer("env", value=0, min=0, max=self.num_envs - 1)
        self.add_float("reliability_threshold", value=0.7, min=0, max=1)
        self.add_float("fraction_active_threshold", value=0.5, min=0, max=1)
        self.add_float("vmax", value=12, min=1, max=20)
        self.on_change(["env", "reliability_threshold", "fraction_active_threshold"], self.update_filters)
        self.update_filters(self.state)

    def update_filters(self, state):
        env = state["env"]
        reliability_threshold = state["reliability_threshold"]
        fraction_active_threshold = state["fraction_active_threshold"]
        idx_reliable = self.reliability.values[env] > reliability_threshold
        idx_active = self.fraction_active[env] > fraction_active_threshold
        idx_options = np.where(idx_reliable & idx_active)[0]
        self.update_selection("roi", options=list(idx_options))

    def plot(self, state):
        env = state["env"]
        roi = state["roi"]

        width = 20
        traversals, pred_travs = self.smp.get_traversals(roi, env, spks=self.spks, width=width)
        xvals = np.arange(width * 2 + 1) - width

        avg_traversal = np.nanmean(traversals, axis=0)
        avg_pred_traversal = np.nanmean(pred_travs, axis=0)
        rms_error = np.sqrt(np.nanmean((pred_travs - traversals) ** 2, axis=0))
        yavgmax = int(np.ceil(np.max([np.nanmax(avg_traversal), np.nanmax(avg_pred_traversal), np.nanmax(rms_error)]) * 1.05))

        cmap = mpl.colormaps["gray_r"]
        norm = plt.Normalize(vmin=0, vmax=state["vmax"])
        values = np.linspace(0, state["vmax"], 100)
        rgba = cmap(norm(values))

        cmap_err = mpl.colormaps["bwr"]
        norm_err = plt.Normalize(vmin=-state["vmax"], vmax=state["vmax"])
        values_err = np.linspace(-state["vmax"], state["vmax"], 100)
        rgba_err = cmap_err(norm_err(values_err))

        fig = plt.figure(figsize=(8, 5), layout="constrained")
        gs = fig.add_gridspec(2, 5, width_ratios=[5, 5, 5, 1, 1], height_ratios=[6, 1])
        ax_traversals = fig.add_subplot(gs[0, 0])
        ax_pred_travs = fig.add_subplot(gs[0, 1])
        ax_error = fig.add_subplot(gs[0, 2])
        ax_colorbar = fig.add_subplot(gs[0, 3])
        ax_cbar_error = fig.add_subplot(gs[0, 4])
        ax_avg_traversal = fig.add_subplot(gs[1, 0])
        ax_avg_pred_travs = fig.add_subplot(gs[1, 1])
        ax_rms_error = fig.add_subplot(gs[1, 2])

        extent = (-width, width, traversals.shape[0], 0)
        ax_traversals.imshow(traversals, interpolation="none", aspect="auto", cmap="gray_r", vmin=0, vmax=state["vmax"], extent=extent)
        ax_traversals.set_xlim(-width, width)
        ax_traversals.set_ylabel("PF Traversals\n(Deconvolved)")
        format_spines(
            ax_traversals,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[-width, width],
            ybounds=[0, traversals.shape[0]],
            xticks=[],
            yticks=[],
            tick_length=4,
            spines_visible=["left"],
        )

        ax_pred_travs.imshow(pred_travs, interpolation="none", aspect="auto", cmap="gray_r", vmin=0, vmax=state["vmax"], extent=extent)
        ax_pred_travs.set_xlim(-width, width)
        ax_pred_travs.set_ylabel("(PF Pred.)")
        format_spines(
            ax_pred_travs,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[-width, width],
            ybounds=[0, traversals.shape[0]],
            xticks=[],
            yticks=[],
            tick_length=4,
            spines_visible=["left"],
        )

        ax_error.imshow(
            pred_travs - traversals,
            interpolation="none",
            aspect="auto",
            cmap="bwr",
            vmin=-state["vmax"],
            vmax=state["vmax"],
            extent=extent,
        )
        ax_error.set_xlim(-width, width)
        ax_error.set_ylabel("(Error)")
        format_spines(
            ax_error,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[-width, width],
            ybounds=[0, traversals.shape[0]],
            xticks=[],
            yticks=[],
            tick_length=4,
            spines_visible=["left"],
        )

        ax_avg_traversal.plot(xvals, avg_traversal, color="k", linewidth=1.5)
        ax_avg_traversal.set_xlim(-width, width)
        ax_avg_traversal.set_xlabel("Frames")
        ax_avg_traversal.set_ylabel("Avg")
        format_spines(
            ax_avg_traversal,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[-width, width],
            ybounds=[0, yavgmax],
            xticks=[-width, width],
            yticks=[0, yavgmax],
            tick_length=4,
            spines_visible=["bottom", "left"],
        )

        ax_avg_pred_travs.plot(xvals, avg_pred_traversal, color="k", linewidth=1.5)
        ax_avg_pred_travs.set_xlim(-width, width)
        ax_avg_pred_travs.set_xlabel("Frames")
        format_spines(
            ax_avg_pred_travs,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[-width, width],
            ybounds=[0, yavgmax],
            xticks=[-width, width],
            yticks=[],
            tick_length=4,
            spines_visible=["bottom"],
        )

        ax_rms_error.plot(xvals, rms_error, color="k", linewidth=1.5)
        ax_rms_error.set_xlim(-width, width)
        ax_rms_error.set_xlabel("Frames")
        ax_rms_error.text(-width, yavgmax, "RMS\nError", fontsize=12, ha="left", va="top", color="k")
        format_spines(
            ax_rms_error,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[-width, width],
            ybounds=[0, yavgmax],
            xticks=[-width, width],
            yticks=[],
            tick_length=4,
            spines_visible=["bottom"],
        )

        ax_colorbar.imshow(np.flipud(rgba[:, None, ...]), aspect="auto", extent=(0, 1, 0, 1))
        ax_colorbar.set_xticks([])
        ax_colorbar.set_yticks([])
        ax_colorbar.text(0.5, 0.02, r"0", fontsize=12, ha="center", va="bottom", color="k")
        ax_colorbar.text(0.5, 0.98, f"{int(state['vmax'])}", fontsize=12, ha="center", va="top", color="w")
        ax_colorbar.set_ylabel("Fluorescence ($\sigma$)", fontsize=12)

        ax_cbar_error.imshow(np.flipud(rgba_err[:, None, ...]), aspect="auto", extent=(0, 1, 0, 1))
        ax_cbar_error.set_xticks([])
        ax_cbar_error.set_yticks([])
        ax_cbar_error.text(0.5, 0.02, f"-{int(state['vmax'])}", fontsize=12, ha="center", va="bottom", color="w")
        ax_cbar_error.text(0.5, 0.98, f"{int(state['vmax'])}", fontsize=12, ha="center", va="top", color="w")
        ax_cbar_error.set_ylabel("Error ($\sigma$)", fontsize=12)

        return fig


class R2PlacefieldFocus(Viewer):
    """Two-panel R² vs reliability plot with selectable ROI and environment."""

    def __init__(self, results: ResultsAggregator, session: B2Session, smp: SMPs.SpkmapProcessor, idx_env: int):
        self.results = results
        self.session = session
        self.smp = smp
        self.num_rois = session.spks[:, session.idx_rois].shape[1]
        self.num_envs = len(smp.get_env_maps().environments)
        self.add_integer("env", value=idx_env, min=0, max=self.num_envs - 1)
        self.add_selection("roi", value=0, options=list(range(self.num_rois)))
        self.add_selection("cloud_style", value="hex", options=["hex", "scatter"])
        self.add_selection("hex_count_norm", value="linear", options=["linear", "log"])
        self.add_float("cloud_alpha", value=0.55, min=0.0, max=1.0)
        self.on_change("env", self.recompute_arrays)
        self.recompute_arrays(self.state)

        self.output = self.results.sel(avg_by_mouse=True)

    def recompute_arrays(self, state):
        self.idx_env = state["env"]
        self.spks_valid, self.pfpred_valid, self.r2, self.reliability = _r2_placefield_arrays(self.session, self.smp, self.idx_env)
        kde_result = _kde_r2(self.r2, self.reliability.values[self.idx_env], _PFPRED_KDE_GRID)
        self.kde_grid = kde_result["r2_kde_grid"]
        self.kde_mean = kde_result["r2_kde_mean"]

    def plot(self, state):
        plt.rcParams["font.size"] = 9
        roi = state["roi"]
        idx_env = self.idx_env
        spks_valid = self.spks_valid
        pfpred_valid = self.pfpred_valid
        r2 = self.r2
        reliability = self.reliability

        plt.close("all")
        fig, ax = plt.subplots(1, 3, figsize=(8.0, 2.0), layout="constrained")

        ax0max = np.max([np.nanmax(spks_valid.T[roi]), np.nanmax(pfpred_valid.T[roi])])
        ax[0].plot(
            spks_valid.T[roi],
            pfpred_valid.T[roi],
            markerfacecolor="k",
            markeredgecolor="none",
            marker=".",
            markersize=10,
            linestyle="none",
            alpha=0.1,
        )
        ax[0].set_xlabel("Activity", labelpad=-15)
        ax[0].set_ylabel("PF Pred.", labelpad=-30)
        format_spines(
            ax[0],
            x_pos=-0.002,
            y_pos=-0.08,
            xbounds=[0, ax0max],
            ybounds=[0, ax0max],
            xticks=[0, ax0max],
            yticks=[0, ax0max],
            tick_length=4,
            spines_visible=["left", "bottom"],
        )

        min_r2 = np.nanmin(r2)
        max_r2 = np.nanmax(r2)
        max_tick_r2 = np.round(np.nanmax(r2), 1)
        rel_env = reliability.values[idx_env]
        valid = np.isfinite(r2) & np.isfinite(rel_env)
        cloud_alpha = state["cloud_alpha"]
        if state["cloud_style"] == "hex":
            hex_norm = LogNorm(vmin=1) if state["hex_count_norm"] == "log" else None
            ax[1].hexbin(
                rel_env[valid],
                r2[valid],
                gridsize=30,
                cmap="Greys",
                mincnt=1,
                linewidths=0,
                norm=hex_norm,
                alpha=cloud_alpha,
                zorder=1,
            )
            kde_color = "black"
        elif state["cloud_style"] == "scatter":
            ax[1].plot(
                rel_env[valid],
                r2[valid],
                markerfacecolor="k",
                markeredgecolor="none",
                marker=".",
                markersize=10,
                linestyle="none",
                alpha=cloud_alpha,
                zorder=1,
            )
            kde_color = "blue"
        else:
            raise ValueError(f"Invalid cloud_style: {state['cloud_style']!r}")
        ax[1].plot(
            self.kde_grid,
            self.kde_mean,
            color=kde_color,
            linewidth=1,
            zorder=5,
        )
        ax[1].plot(
            rel_env[roi],
            r2[roi],
            markerfacecolor="r",
            markeredgecolor="none",
            marker=".",
            markersize=15,
            linestyle="none",
            zorder=10,
        )
        ax[1].set_xlim(-1, 1)
        ax[1].set_xlabel("Spatial Reliability")
        ax[1].set_ylabel(r"$R^2$(Activity, PF Pred.)")
        ax1ylim = ax[1].get_ylim()

        linewidth_example = 1
        linewidth_average = 1.5
        alpha_example = 0.3
        alpha_highlight = 0.7
        idx_to_example = self.results.unique_mice.index(self.session.mouse_name)
        kde_grid = self.output["r2_kde_grid"][0]
        kde_mean = self.output["r2_kde_mean"]
        ax[2].plot(kde_grid, kde_mean.T, color=("k", alpha_example), linewidth=linewidth_example)
        ax[2].plot(kde_grid, kde_mean[idx_to_example].T, color=("blue", alpha_highlight), linewidth=linewidth_example)
        ax[2].plot(kde_grid, np.nanmean(kde_mean, axis=0), color="k", linewidth=linewidth_average)
        ax[2].set_xlim(-1, 1)
        ax[2].set_xlabel("Spatial Reliability")
        ax[2].set_ylabel(r"$R^2$(Activity, PF Pred.)")
        legend_elements = [
            Line2D([0], [0], color="k", alpha=alpha_example, linewidth=linewidth_example, label="mouse"),
            Line2D([0], [0], color="blue", alpha=alpha_highlight, linewidth=linewidth_example, label="example"),
            Line2D([0], [0], color="k", linewidth=linewidth_average, label="average"),
        ]
        ax[2].legend(handles=legend_elements)
        ax2ylim = ax[2].get_ylim()

        ax12ylim = (min(ax1ylim[0], ax2ylim[0]), max(ax1ylim[1], ax2ylim[1]))
        ax[1].set_ylim(ax12ylim)
        ax[2].set_ylim(ax12ylim)

        # Format spines once ylims have been set
        format_spines(
            ax[1],
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[-1, 1],
            ybounds=[min_r2, max_r2],
            xticks=[-1, 0, 1],
            yticks=[0, max_tick_r2],
            tick_length=4,
            spines_visible=["left", "bottom"],
        )
        format_spines(
            ax[2],
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[-1, 1],
            ybounds=[min_r2, max_r2],
            xticks=[-1, 0, 1],
            yticks=[0, max_tick_r2],
            tick_length=4,
            spines_visible=["left", "bottom"],
        )
        return fig


# Result keys of BehaviorSpeedEnvConfig used by the speed figure, and the config param axes we
# expose as selections (each dropdown's options come straight from the aggregator's param_axes).
_SPEED_CURVE_KEYS: dict[str, str] = {"all": "speed_curve_all", "first": "speed_curve_first"}
_BEHAVIOR_SPEED_PARAM_AXES: tuple[str, ...] = ("num_bins", "speed_threshold", "regularization")


class MouseSpeedFocus(Viewer):
    """Per-environment mouse speed over VR position, loaded from precomputed results.

    Reads the ``BehaviorSpeedEnvConfig`` aggregator: for the selected config parameters
    (``num_bins``, ``speed_threshold``, ``regularization``) and trial set (all trials vs the
    first trial of each block) it assembles a ``(mice, envs, bins)`` speed array by mapping each
    session's stored ``speed_curve_*`` rows onto the global environment axis via its stored
    ``environments`` key, then averaging across sessions within a mouse.

    ax[0]: for each selected environment, the mouse-average speed curve with a shaded
    ±standard-error band, colored per environment (``tab10``). Both trial sets are drawn:
    all trials (solid) and the first trial of each block (dashed). A dotted line marks each
    environment's reward-zone start (from :data:`ENV_REWARD_MAP`), colored to match.
    """

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (7.0, 5.0)):
        self.results = results
        self.figsize = figsize
        self.cmap = mpl.colormaps["tab10"]

        # Global environment axis (identity-based, independent of the selected parameters).
        env_arr = np.asarray(results.arrays["environments"], dtype=float)
        self.env_list = sorted({int(e) for e in env_arr[np.isfinite(env_arr)] if e >= 0})
        self.reward_position = {env: float(ENV_REWARD_MAP[env]) for env in self.env_list}

        # One dropdown per available config param axis, options straight from the aggregator.
        self._param_axes = [name for name in _BEHAVIOR_SPEED_PARAM_AXES if name in results.param_axes]
        for name in self._param_axes:
            options = list(results.param_axes[name])
            self.add_selection(name, options=options, value=options[0])

        env_options = [str(env) for env in self.env_list]
        self.add_multiple_selection("environments", options=env_options, value=env_options)
        self.add_boolean("show_first", value=True)
        self.add_boolean("show_reward", value=True)
        self.add_boolean("show_legend", value=True)
        self.add_float("alpha_band", value=0.2, min=0.0, max=1.0)
        self.add_float("linewidth_average", value=2.0, min=0.1, max=5.0)

        self.on_change(self._param_axes, self.recompute_arrays)
        self.recompute_arrays(self.state)

    def recompute_arrays(self, state):
        """Assemble ``(mice, envs, bins)`` speed arrays (per trial set) + position axis."""
        params = {name: state[name] for name in self._param_axes}
        curve_keys = list(_SPEED_CURVE_KEYS.values())
        sel = self.results.sel(keys=curve_keys + ["environments", "dist_fraction_centers"], squeeze_ones=False, **params)

        # Pad keys are padded to the grid-wide max num_bins; trim to the selected bin count.
        num_bins = int(state["num_bins"]) if "num_bins" in state else np.asarray(sel[curve_keys[0]]).shape[-1]
        env_sel = np.asarray(sel["environments"], dtype=float)  # (n_sess, max_env)
        frac = np.asarray(sel["dist_fraction_centers"], dtype=float)[:, :num_bins]  # (n_sess, num_bins)

        mouse_per_session = list(self.results.mouse_names)
        # Which (session, row) pairs land on each (mouse, env) slot. Env coverage is the same
        # for every trial set, so this mapping is built once and reused for both curves.
        rows_by_slot: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
        for s in range(env_sel.shape[0]):
            mouse = mouse_per_session[s]
            for r in range(env_sel.shape[1]):
                env = env_sel[s, r]
                if not np.isfinite(env) or env < 0:
                    continue
                rows_by_slot[(mouse, int(env))].append((s, r))

        present = {mouse for mouse, _ in rows_by_slot}
        self.mouse_names = [m for m in dict.fromkeys(mouse_per_session) if m in present]

        # One (mice, envs, bins) array per trial set, averaged across sessions within a mouse.
        self.speed: dict[str, np.ndarray] = {}
        for key in curve_keys:
            curves = np.asarray(sel[key], dtype=float)[..., :num_bins]  # (n_sess, max_env, num_bins)
            arr = np.full((len(self.mouse_names), len(self.env_list), num_bins), np.nan)
            for m, mouse in enumerate(self.mouse_names):
                for e, env in enumerate(self.env_list):
                    rows = rows_by_slot.get((mouse, env))
                    if rows:
                        arr[m, e] = np.nanmean(np.stack([curves[s, r] for s, r in rows], axis=0), axis=0)
            self.speed[key] = arr

        # Position axis (cm): fraction-of-track is identical across matching sessions.
        finite_rows = np.where(np.all(np.isfinite(frac), axis=1))[0]
        frac_centers = frac[finite_rows[0]] if finite_rows.size else frac[0]
        self.dist_centers = frac_centers * REFERENCE_ENV_LENGTH_CM

        # Per-session decoding accuracy for the second panel. All param dims are fixed, so each
        # key is one value per session; n_trials_per_env yields the environment count per session.
        acc = self.results.sel(keys=["acc_test_random", "acc_test_block", "n_trials_per_env"], squeeze_ones=False, avg_by_mouse=False, **params)
        self.acc_random = np.asarray(acc["acc_test_random"], dtype=float)  # (n_sess,)
        self.acc_block = np.asarray(acc["acc_test_block"], dtype=float)  # (n_sess,)
        self.n_envs_per_session = np.sum(np.asarray(acc["n_trials_per_env"], dtype=float) > 0, axis=1)  # (n_sess,)

    def _env_color(self, env: int):
        """tab10 color for an environment, indexed by its position in ``env_list``."""
        return self.cmap(self.env_list.index(env) % 10)

    def plot(self, state):
        selected = [int(env) for env in state["environments"]]
        xvals = self.dist_centers
        env_length = REFERENCE_ENV_LENGTH_CM
        all_key = _SPEED_CURVE_KEYS["all"]
        first_key = _SPEED_CURVE_KEYS["first"]
        lw = state["linewidth_average"]
        fontsize = 9

        plt.close("all")
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=self.figsize, layout="constrained", width_ratios=[1.6, 1])

        # ------------------------------------------------------------------ speed curves --
        drawn_envs = [env for env in self.env_list if env in selected]
        for e, env in enumerate(self.env_list):
            if env not in selected:
                continue
            color = self._env_color(env)
            # All trials: mouse-average with a shaded ±SE band (average over the mouse axis).
            errorPlot(xvals, self.speed[all_key][:, e, :], axis=0, se=True, ax=ax, color=color, alpha=state["alpha_band"], linewidth=lw)
            if state["show_first"]:
                # First trial of each block: mouse-average only (dashed), no band -- keeps the
                # panel readable when both trial sets are shown together.
                ax.plot(xvals, np.nanmean(self.speed[first_key][:, e, :], axis=0), color=color, linewidth=lw, linestyle="--")
        xticks = np.arange(0, env_length + 1, 50)
        ax.set_xlabel("Position (cm)", fontsize=fontsize)
        ax.set_ylabel("Speed (cm/s)", fontsize=fontsize)
        ax.set_xlim(0, env_length)
        ax.set_xticks(xticks)
        _, ymax = ax.get_ylim()
        # Drop the lower limit below 0 to make room for the legend without overlapping curves.
        ax.set_ylim(-10, ymax)

        # Reward lines: draw after ymax is known so they span only the data range [0, ymax],
        # not the negative legend margin.
        if state["show_reward"]:
            for env in drawn_envs:
                ax.vlines(self.reward_position[env], 0, ymax, color=self._env_color(env), linestyle=":", linewidth=1.0)

        # Keep y-ticks at physical speeds (>= 0); the negative margin is legend space only.
        yticks = [t for t in ax.get_yticks() if 0 <= t <= ymax]

        # Bidirectional arrow marking the decoder's fit window (start of track -> earliest
        # reward), annotated at a fixed low speed so it sits under the curves.
        window_end = WINDOW_FRACTION * env_length
        ax.annotate("", xy=(0, 2), xytext=(window_end, 2), arrowprops=dict(arrowstyle="<->", color="k", linewidth=1.0))
        ax.text(window_end / 2, 2.5, "fit window", ha="center", va="bottom", fontsize=8)

        format_spines(
            ax,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[0, env_length],
            ybounds=[0, ymax],
            xticks=xticks,
            yticks=yticks,
            tick_fontsize=fontsize,
            spines_visible=["left", "bottom"],
        )

        # Custom legend: one handle whose segments run blue->orange->green (the env colors),
        # labeled "Envs"; HandlerTuple packs the sub-lines side by side across a single handle.
        if state["show_legend"] and drawn_envs:
            handles = [tuple(Line2D([0], [0], color=self._env_color(env), linewidth=lw) for env in drawn_envs)]
            labels = ["Envs"]
            if state["show_first"]:
                handles.append(Line2D([0], [0], color="0.3", linewidth=lw, linestyle="--"))
                labels.append("1st of block")
            if state["show_reward"]:
                handles.append(Line2D([0], [0], color="0.3", linewidth=1.0, linestyle=":"))
                labels.append("reward zones")
            # pad=0 packs the env segments flush against each other (one continuous swatch).
            ax.legend(
                handles,
                labels,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
                loc="lower right",
                frameon=True,
                fontsize=8,
            )

        # ------------------------------------------------------ decoding accuracy panel --
        # Paired random->first-trial accuracy per session, split by session environment count.
        # x = [0,1] is the 2-env group; x = [2,3] is the 3-env group. The env-count grouping is
        # carried by the hierarchical x-ticks below, so the lines themselves are plain black.
        ypad = 0.05
        n_env_groups = (2, 3)
        for gi, n_env in enumerate(n_env_groups):
            x_rand, x_block = 2 * gi, 2 * gi + 1
            m = self.n_envs_per_session == n_env
            r, b = self.acc_random[m], self.acc_block[m]
            valid = np.isfinite(r) & np.isfinite(b)
            r, b = r[valid], b[valid]
            if r.size == 0:
                continue
            for ri, bi in zip(r, b):
                ax2.plot([x_rand, x_block], [ri, bi], color="k", linewidth=0.5, alpha=0.3)
            ax2.plot([x_rand, x_block], [r.mean(), b.mean()], color="k", linewidth=2.5)

        ax2.set_ylabel("Test accuracy", fontsize=fontsize)
        ax2.set_xlim(-0.4, 3.4)
        ax2.set_ylim(-ypad, 1 + ypad)
        format_spines(
            ax2,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[0, 3],
            ybounds=[0, 1],
            xticks=[0, 1, 2, 3],
            xlabels=["any", "1st", "any", "1st"],
            yticks=[0, 0.5, 1],
            tick_fontsize=fontsize,
            spines_visible=["left", "bottom"],
        )
        # Second tick level: a bracket + label under each split pair naming the env-count group.
        trans = ax2.get_xaxis_transform()  # x in data coords, y in axes fraction
        for (x0, x1), label in zip([(0, 1), (2, 3)], ["2 Envs", "3 Envs"]):
            ax2.plot([x0, x1], [-0.20, -0.20], transform=trans, color="k", linewidth=1.0, clip_on=False)
            ax2.text((x0 + x1) / 2, -0.24, label, transform=trans, ha="center", va="top", fontsize=fontsize)
        return fig


def mouse_speed_by_environment(
    results: ResultsAggregator,
    num_bins: int = 100,
    speed_threshold: float = -np.inf,
    regularization: float = 1.0,
    environments: list[int] | None = None,
    show_first: bool = True,
    show_reward: bool = True,
    show_legend: bool = True,
    alpha_band: float = 0.2,
    linewidth_average: float = 2.5,
    figsize: tuple[float, float] = (7.0, 5.0),
    return_syd_viewer: bool = False,
):
    """
    Mouse speed as a function of VR position, per environment, aggregated across mice.

    Loads the precomputed ``BehaviorSpeedEnvConfig`` results (which already exclude the
    CR_Hippocannula mice and any session whose reward layout does not match ``ENV_REWARD_MAP``).
    For each environment the mouse-average speed curve is drawn with a shaded ±standard-error
    band, colored per environment. Both trial sets are shown: all trials (solid) and the first
    trial of each block (dashed); a dotted vertical line marks the reward-zone start.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``BehaviorSpeedEnvConfig`` results.
    num_bins : int
        Position-bin count (a config param axis).
    speed_threshold : float
        Speed-map sample threshold (a config param axis). ``-np.inf`` keeps all samples.
    regularization : float
        Decoder inverse regularization (a config param axis); does not affect the speed curves
        but selects one stored slice of the result grid.
    environments : list[int] or None
        Environments to draw. If None, all available environments are shown.
    show_first, show_reward, show_legend : bool
        Toggle the first-of-block curve, the dotted reward line, and the legend.
    alpha_band : float
        Opacity of the ±standard-error band.
    linewidth_average : float
        Line width of the mouse-average curves.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer instead of a rendered figure.

    Returns
    -------
    matplotlib.figure.Figure or MouseSpeedFocus
    """
    viewer = MouseSpeedFocus(results, figsize=figsize)

    for name, value in (("num_bins", num_bins), ("speed_threshold", speed_threshold), ("regularization", regularization)):
        if name in viewer._param_axes and value is not None:
            viewer.update_selection(name, value=value)
    if environments is not None:
        viewer.update_multiple_selection("environments", value=[str(env) for env in environments])
    viewer.update_boolean("show_first", value=show_first)
    viewer.update_boolean("show_reward", value=show_reward)
    viewer.update_boolean("show_legend", value=show_legend)
    viewer.update_float("alpha_band", value=alpha_band)
    viewer.update_float("linewidth_average", value=linewidth_average)

    # Seeding via update_* does not fire on_change before deployment, so refresh the arrays for the
    # seeded selections (the on_change hook still drives live updates once the viewer is deployed).
    viewer.recompute_arrays(viewer.state)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


def stacked_raster_plot(
    session: B2Session,
    use_reliable: bool = True,
    reliability_threshold: float = 0.7,
    prediction_from: str = "spkmap",
    xslice: slice = slice(0, 2000),
    vmax: float = 6,
    figsize: tuple[float, float] = (12, 6),
    show_position: bool = False,
    position_height: float = 0.5,
    env_gap: float = 0.2,
    show_zero_sigma: bool = False,
    return_syd_viewer: bool = False,
):
    """
    Plot a stacked raster plot of the deconvolved calcium activity and the prediction from the place field.

    Parameters
    ----------
    session : B2Session
    use_reliable : bool
    reliability_threshold : float
    prediction_from : str
        ``spkmap`` or ``placefield`` — determines how the place-field prediction is built.
    xslice : slice
    vmax : float
    figsize : tuple[float, float]
    show_position : bool
        If True, add a 4th panel showing the mouse position over the plotted frames.
    position_height : float
        Height ratio of the position panel relative to each raster (the ``x`` in ``[1, 1, 1, x]``).
    env_gap : float
        Extra vertical gap between environment bands in the position panel, as a fraction of ``num_bins``.
    show_zero_sigma : bool
        If True, draw a ``0 sigma`` label on the left of the gray_r colorscale inset.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    """
    smp = SMPs.SpkmapProcessor(session, params=SMPs.SpkmapParams())
    spks = session.spks[:, session.idx_rois]
    spks = median_zscore(spks, median_subtract=not session.zero_baseline_spks)
    placefield_prediction, extras = smp.get_placefield_prediction()

    viewer = StackedRasterFocus(
        session,
        smp,
        spks,
        placefield_prediction,
        extras,
        figsize,
        xslice.start if xslice.start is not None else 0,
        xslice.stop if xslice.stop is not None else spks.shape[0],
    )
    viewer.update_selection("prediction_from", value=prediction_from)
    viewer.update_boolean("use_reliable", value=use_reliable)
    viewer.update_float("reliability_threshold", value=reliability_threshold)
    viewer.update_float("vmax", value=vmax)
    viewer.update_integer("xslice_start", value=xslice.start if xslice.start is not None else 0)
    viewer.update_integer("xslice_stop", value=xslice.stop if xslice.stop is not None else spks.shape[0])
    viewer.update_boolean("show_position", value=show_position)
    viewer.update_float("position_height", value=position_height)
    viewer.update_float("env_gap", value=env_gap)
    viewer.update_boolean("show_zero_sigma", value=show_zero_sigma)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


def example_placefield(
    session: B2Session,
    roi: int,
    env: int,
    reliability_threshold: float = 0.7,
    fraction_active_threshold: float = 0.5,
    vmax: float = 5,
    return_syd_viewer: bool = False,
):
    """
    Plot an example placefield for a given session.
    """
    smp = SMPs.SpkmapProcessor(session, params=SMPs.SpkmapParams())

    env_maps = smp.get_env_maps()
    env_maps.distcenters = smp.dist_centers
    env_maps.pop_nan_positions()
    reliability = smp.get_reliability()
    fraction_active = np.stack([FractionActive.compute(spkmap, 2, 1) for spkmap in env_maps.spkmap])

    class PlaceFieldFocus(Viewer):
        def __init__(self, env_maps: Maps, reliability: Reliability, fraction_active: np.ndarray, dist_edges: np.ndarray):
            self.env_maps = env_maps
            self.reliability = reliability
            self.fraction_active = fraction_active
            self.dist_edges = dist_edges

            self.num_rois = env_maps.spkmap[0].shape[0]
            self.num_envs = len(env_maps.environments)

            self.add_selection("roi", value=0, options=list(range(self.num_rois)))
            self.add_integer("env", value=0, min=0, max=self.num_envs - 1)
            self.add_float("reliability_threshold", value=0.7, min=0, max=1)
            self.add_float("fraction_active_threshold", value=0.5, min=0, max=1)
            self.add_float("vmax", value=5, min=1, max=20)
            self.on_change(["env", "reliability_threshold", "fraction_active_threshold"], self.update_filters)

            self.update_filters(self.state)

        def update_filters(self, state):
            env = state["env"]
            reliability_threshold = state["reliability_threshold"]
            fraction_active_threshold = state["fraction_active_threshold"]
            idx_reliable = self.reliability.values[env] > reliability_threshold
            idx_active = self.fraction_active[env] > fraction_active_threshold
            idx_options = np.where(idx_reliable & idx_active)[0]
            self.update_selection("roi", options=list(idx_options))

        def plot(self, state):
            env = state["env"]
            roi = state["roi"]

            spkmap = self.env_maps.spkmap[env][roi]
            placefield = np.nanmean(spkmap, axis=0)

            trial_weights = np.sqrt(np.mean(spkmap**2, axis=1))
            trial_consistency = _jit_reliability_loo(spkmap[None, ...])[0]
            trial_weights = trial_weights / np.max(trial_weights)

            idx_include = trial_weights > 0
            trial_numbers = np.arange(spkmap.shape[0])[idx_include]
            trial_weights = trial_weights[idx_include] / np.max(trial_weights[idx_include])
            trial_consistency = trial_consistency[idx_include]
            half_trial_number = max(trial_numbers) / 2

            xlims = [self.dist_edges[0], self.dist_edges[-1]]
            extent = (0, spkmap.shape[1], spkmap.shape[0], 0)
            ymax_pf = np.nanmax(placefield) * 1.2
            cmap = mpl.colormaps["gray_r"]
            norm = plt.Normalize(vmin=0, vmax=state["vmax"])
            values = np.linspace(0, state["vmax"], 100)
            rgba = cmap(norm(values))

            fig = plt.figure(figsize=(5, 6), layout="constrained")
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[6, 1])
            ax_spkmap = fig.add_subplot(gs[0, 0])
            ax_placefield = fig.add_subplot(gs[1, 0])
            ax_consistency = fig.add_subplot(gs[0, 1])
            ax_reliability = fig.add_subplot(gs[1, 1])
            ax_colorbar = ax_spkmap.inset_axes([0.1, 0.15, 0.075, 0.7])

            ax_spkmap.imshow(spkmap, interpolation="none", aspect="auto", cmap="gray_r", vmin=0, vmax=state["vmax"], extent=extent)
            ax_spkmap.set_ylabel("Trials")
            ax_spkmap.set_xlim(xlims[0], xlims[1])
            ax_spkmap.set_ylim(spkmap.shape[0] + 0.5, -0.5)
            format_spines(ax_spkmap, x_pos=-0.02, y_pos=-0.02, xbounds=xlims, xticks=[], yticks=[], tick_length=4, spines_visible=["left"])

            ax_placefield.plot(placefield, color="k", linewidth=1.5)
            ax_placefield.set_facecolor(("black", 0.04))
            ax_placefield.set_xlabel("VR Position", labelpad=-10)
            ax_placefield.set_xlim(xlims[0], xlims[1])
            ax_placefield.set_ylim(-0.05, ymax_pf)
            ax_placefield.text(xlims[0], ymax_pf, "Place Field", ha="left", va="top", color="k")
            format_spines(
                ax_placefield,
                x_pos=-0.02,
                y_pos=-0.15,
                xbounds=xlims,
                xticks=xlims,
                yticks=[],
                tick_length=4,
                spines_visible=["bottom"],
            )

            ax_consistency.scatter(trial_consistency, trial_numbers, color="k", s=5, alpha=trial_weights)
            ax_consistency.set_facecolor(("black", 0.04))
            ax_consistency.set_xlim(-1.05, 1.05)
            ax_consistency.set_ylim(spkmap.shape[0] + 0.5, -0.5)
            ax_consistency.set_xlabel(r"$\sigma$")
            ax_consistency.text(
                -0.5,
                half_trial_number,
                r"$\sigma = \mathrm{corr}(\langle\mathrm{other\ trials}\rangle)$",
                ha="center",
                va="center",
                rotation=90,
            )

            format_spines(
                ax_consistency,
                x_pos=-0.02,
                y_pos=-0.02,
                xbounds=(-1, 1),
                xticks=[-1, 0, 1],
                yticks=[],
                tick_length=4,
                spines_visible=["bottom"],
            )

            reliability = np.sum(trial_weights * trial_consistency) / np.sum(trial_weights)
            ax_reliability.plot([-1, 1], [0, 0], color="black", linewidth=1.5)
            ax_reliability.plot([reliability], [0], color="black", marker="o", markersize=8)
            ax_reliability.set_xlim(-1, 1)
            ax_reliability.set_ylim(-0.05, 0.05)
            ax_reliability.set_xticks([-1, 0, 1])
            ax_reliability.set_yticks([])
            ax_reliability.set_xlabel("Reliability")
            format_spines(
                ax_reliability,
                x_pos=-0.02,
                y_pos=-0.02,
                xbounds=(-1, 1),
                xticks=[-1, 0, 1],
                yticks=[],
                tick_length=4,
                spines_visible=["bottom"],
            )

            ax_colorbar.imshow(np.flipud(rgba[:, None, ...]), aspect="auto", extent=(0, 1, 0, 1))
            ax_colorbar.set_xticks([])
            ax_colorbar.set_yticks([])
            ax_colorbar.text(0.5, 0.02, r"0", fontsize=12, ha="center", va="bottom", color="k")
            ax_colorbar.text(0.5, 0.98, f"{int(state['vmax'])}", fontsize=12, ha="center", va="top", color="w")
            ax_colorbar.set_ylabel("Fluorescence ($\sigma$)", fontsize=12)

            return fig

    viewer = PlaceFieldFocus(env_maps, reliability, fraction_active, smp.dist_edges)
    _seed_roi_filtered_viewer(
        viewer,
        env=env,
        roi=roi,
        reliability_threshold=reliability_threshold,
        fraction_active_threshold=fraction_active_threshold,
        vmax=vmax,
    )

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


def example_traversal(
    session: B2Session,
    roi: int,
    env: int,
    vmax: float = 12,
    reliability_threshold: float = 0.7,
    fraction_active_threshold: float = 0.5,
    return_syd_viewer: bool = False,
):
    """
    Plot PF traversal panels for one ROI and environment.
    """
    smp = SMPs.SpkmapProcessor(session, params=SMPs.SpkmapParams())
    env_maps = smp.get_env_maps()
    env_maps.pop_nan_positions()
    reliability = smp.get_reliability()
    fraction_active = np.stack([FractionActive.compute(spkmap, 2, 1) for spkmap in env_maps.spkmap])

    viewer = TraversalFocus(smp, env_maps, reliability, fraction_active)
    _seed_roi_filtered_viewer(
        viewer,
        env=env,
        roi=roi,
        reliability_threshold=reliability_threshold,
        fraction_active_threshold=fraction_active_threshold,
        vmax=vmax,
    )

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


def example_r2_placefield(
    results: ResultsAggregator,
    session: B2Session,
    roi: int = EXAMPLE_ROI,
    idx_env: int = 0,
    cloud_style: str = "hex",
    cloud_alpha: float | None = None,
    hex_count_norm: str = "linear",
    return_syd_viewer: bool = False,
):
    """
    Two-panel plot of activity vs PF prediction and R² vs spatial reliability.

    Parameters
    ----------
    cloud_style : {"hex", "scatter"}
        How to draw all ROIs on the R² vs reliability panel.
    cloud_alpha : float or None
        Opacity for the hexbin or scatter cloud. Defaults to 0.55 for hex and
        0.1 for scatter when None.
    hex_count_norm : {"linear", "log"}
        Color mapping for hexbin counts (ignored when ``cloud_style="scatter"``).
        ``log`` uses ``matplotlib.colors.LogNorm`` so sparse regions are visible
        when a few bins dominate the count range.
    """
    if cloud_style not in ("hex", "scatter"):
        raise ValueError(f"cloud_style must be 'hex' or 'scatter', got {cloud_style!r}")
    if hex_count_norm not in ("linear", "log"):
        raise ValueError(f"hex_count_norm must be 'linear' or 'log', got {hex_count_norm!r}")
    if cloud_alpha is None:
        cloud_alpha = 0.55 if cloud_style == "hex" else 0.1

    smp = SMPs.SpkmapProcessor(session, params=SMPs.SpkmapParams())
    smp.get_env_maps().pop_nan_positions()

    viewer = R2PlacefieldFocus(results, session, smp, idx_env)
    viewer.update_integer("env", value=idx_env)
    viewer.update_selection("roi", value=roi)
    viewer.update_selection("cloud_style", value=cloud_style)
    viewer.update_selection("hex_count_norm", value=hex_count_norm)
    viewer.update_float("cloud_alpha", value=cloud_alpha)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


class ReliabilityHistogramViewer(Viewer):
    """Mouse-average spatial-reliability histograms + fraction-of-place-cells beeswarm.

    Reads the ``PFPredQualityConfig`` aggregator, whose per-session ``reliability`` key holds one
    spatial-reliability value per ROI (best environment), the same measure used in
    :func:`example_r2_placefield`.

    - ax[0]: for each session a reliability histogram over ``[-1, 1]`` (normalized to a fraction of
      cells), averaged across sessions within a mouse, then drawn as one thin black alpha line per
      mouse plus a thicker black across-mouse average. A dotted line marks the place-cell threshold.
    - ax[1]: fraction of place cells (reliability above the threshold) as a beeswarm. ``swarm_mode``
      picks either a single pooled swarm of per-mouse averages (``"pooled"``) or one swarm per mouse
      of per-session fractions, sorted by mouse average (``"by_mouse"``).
    """

    def __init__(self, results: ResultsAggregator, fontsize: float = 9.0, figsize: tuple[float, float] = (6.0, 3.0)):
        self.results = results
        self.fontsize = fontsize
        self.figsize = figsize

        # Per-session per-ROI spatial reliability, NaN-padded to the max ROI count: (n_sess, max_rois).
        self.reliability = np.asarray(results.sel(keys=["reliability"], squeeze_ones=False)["reliability"], dtype=float)
        self.mouse_names = results.mouse_names

        self.add_integer("n_bins", value=40, min=5, max=100)
        self.add_float("place_cell_threshold", value=0.3, min=-1.0, max=1.0, step=0.05)
        self.add_selection("swarm_mode", options=["pooled", "by_mouse"], value="pooled")
        self.add_float("beewidth", value=0.2, min=0.0, max=1.0, step=0.01)
        self.add_float("hist_alpha", value=0.3, min=0.0, max=1.0, step=0.05)

    def _session_histograms(self, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-session reliability histograms over ``[-1, 1]``, each normalized to a fraction of cells."""
        edges = np.linspace(-1, 1, n_bins + 1)
        centers = edge2center(edges)
        hist = np.full((self.reliability.shape[0], n_bins), np.nan)
        for i, rel in enumerate(self.reliability):
            rel = rel[np.isfinite(rel)]
            if rel.size == 0:
                continue
            counts, _ = np.histogram(rel, bins=edges)
            hist[i] = counts / counts.sum()
        return centers, hist

    def _fraction_place_cells(self, threshold: float) -> np.ndarray:
        """Per-session fraction of ROIs whose reliability exceeds ``threshold``."""
        finite = np.isfinite(self.reliability)
        n_finite = finite.sum(axis=1)
        n_place = np.sum(finite & (self.reliability > threshold), axis=1)
        return np.where(n_finite > 0, n_place / np.maximum(n_finite, 1), np.nan)

    def plot(self, state):
        plt.rcParams["font.size"] = self.fontsize
        n_bins = int(state["n_bins"])
        threshold = state["place_cell_threshold"]
        swarm_mode = state["swarm_mode"]
        beewidth = state["beewidth"]

        centers, hist = self._session_histograms(n_bins)
        mouse_hist = average_by_mouse(hist, self.mouse_names)
        frac = self._fraction_place_cells(threshold)

        width_ratios = [1, 0.5] if swarm_mode == "pooled" else [1, 1]
        plt.close("all")
        fig, ax = plt.subplots(1, 2, figsize=self.figsize, layout="constrained", width_ratios=width_ratios)

        # --- ax[0]: per-mouse reliability histograms (thin black) + across-mouse average (thick) ---
        hist_max = np.nanmax(mouse_hist)
        ax[0].plot(centers, mouse_hist.T, color=("k", state["hist_alpha"]), linewidth=1.0)
        ax[0].plot(centers, np.nanmean(mouse_hist, axis=0), color="k", linewidth=2.0)
        ax[0].axvline(threshold, color="0.6", linestyle=":", linewidth=1.0)
        ax[0].set_xlim(-1, 1)
        ax[0].set_xlabel("Spatial Reliability")
        ax[0].set_ylabel("Fraction of Cells")
        legend_elements = [
            Line2D([0], [0], color="k", alpha=state["hist_alpha"], linewidth=1.0, label="each mouse"),
            Line2D([0], [0], color="k", linewidth=2.0, label="average"),
        ]
        ax[0].legend(handles=legend_elements, fontsize=self.fontsize, frameon=False)
        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[-1, 1],
            ybounds=[0, hist_max],
            xticks=[-1, 0, 1],
            spines_visible=["left", "bottom"],
        )

        # --- ax[1]: fraction of place cells, pooled per-mouse swarm or per-mouse-of-sessions swarm ---
        if swarm_mode == "pooled":
            vals = average_by_mouse(frac, self.mouse_names)
            finite = np.isfinite(vals)
            offsets = np.zeros_like(vals)
            if finite.any():
                offsets[finite] = beeswarm(vals[finite])
            ax[1].plot(beewidth * offsets, vals, linestyle="none", color="k", marker="o", markersize=4, alpha=0.8)
            ax[1].plot([-0.25, 0.25], [np.nanmean(vals)] * 2, color="k", linewidth=2.0)
            ax[1].set_xlim(-0.5, 0.5)
            xbounds = (0, 0)
            xticks = []
        else:
            mice = list(dict.fromkeys(self.mouse_names))
            mice.sort(key=lambda m: np.nanmean(frac[self.mouse_names == m]), reverse=True)
            for xi, mouse in enumerate(mice):
                vals = frac[self.mouse_names == mouse]
                finite = np.isfinite(vals)
                offsets = np.zeros_like(vals)
                if finite.any():
                    offsets[finite] = beeswarm(vals[finite])
                ax[1].plot(xi + beewidth * offsets, vals, linestyle="none", color="k", marker=".", markersize=5, alpha=0.3)
                ax[1].plot(xi + np.array([-0.4, 0.4]), [np.nanmean(vals)] * 2, color="k", linewidth=1.2)
            ax[1].set_xlim(-1.0, len(mice))
            ax[1].set_xlabel("Mice")
            xbounds = (0, len(mice) - 1)
            xticks = range(len(mice))

        ax[1].set_ylim(0, 1)
        ax[1].set_yticks([0, 0.5, 1])
        ax[1].set_ylabel("Fraction Place Cells")
        format_spines(
            ax[1],
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=xbounds,
            ybounds=(0, 1),
            yticks=[0, 0.5, 1],
            spines_visible=["left", "bottom"],
        )
        ax[1].set_xticks(xticks, labels=[])
        return fig


def placefield_reliability(
    results: ResultsAggregator,
    place_cell_threshold: float = 0.3,
    n_bins: int = 40,
    swarm_mode: str = "pooled",
    beewidth: float = 0.2,
    hist_alpha: float = 0.3,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (6.0, 3.0),
    return_syd_viewer: bool = False,
):
    """
    Mouse-average spatial-reliability histograms and fraction of place cells.

    ax[0] draws one thin black alpha reliability histogram per mouse (session-averaged, normalized
    to a fraction of cells over ``[-1, 1]``) plus a thicker black across-mouse average. ax[1] is a
    fraction-of-place-cells beeswarm using ``place_cell_threshold``: a single pooled swarm of
    per-mouse averages (``swarm_mode="pooled"``, width ratios ``[1, 0.5]``) or one swarm of
    per-session fractions per mouse, sorted by mouse average (``swarm_mode="by_mouse"``, width
    ratios ``[1, 1]``).

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``PFPredQualityConfig`` results (source of the per-ROI ``reliability`` key).
    place_cell_threshold : float
        Reliability cutoff defining a place cell for ax[1] (and the dotted marker on ax[0]).
    n_bins : int
        Number of histogram bins over ``[-1, 1]``.
    swarm_mode : {"pooled", "by_mouse"}
        Beeswarm layout for ax[1].
    beewidth : float
        Horizontal spread of the beeswarm points.
    hist_alpha : float
        Opacity of the per-mouse histogram lines.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.

    Returns
    -------
    matplotlib.figure.Figure or ReliabilityHistogramViewer
    """
    viewer = ReliabilityHistogramViewer(results, fontsize=fontsize, figsize=figsize)
    viewer.update_integer("n_bins", value=n_bins)
    viewer.update_float("place_cell_threshold", value=place_cell_threshold)
    viewer.update_selection("swarm_mode", value=swarm_mode)
    viewer.update_float("beewidth", value=beewidth)
    viewer.update_float("hist_alpha", value=hist_alpha)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig
