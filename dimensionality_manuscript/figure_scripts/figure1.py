from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle
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
from dimensionality_manuscript.blender import RIG_HFOV_DEG, RenderParams, load_vr_room_images
from dimensionality_manuscript.env_order import ENV_SLOT_COLORS

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


# Axes-fraction rectangle (x0, y0, width, height) of the colorscale insets. The scalebar is
# mirrored from it: same vertical center, same inset from the axes edge (left instead of right).
_INSET_RECT = (0.72, 0.10, 0.25, 0.15)


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
    axins = ax.inset_axes(list(_INSET_RECT))
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
        self.extras = extras
        self.figsize = figsize

        # Valid frames define the plotted axis; everything below lives in valid-frame coordinates.
        idx_valid = extras["idx_valid"]
        self.spks_valid = spks[idx_valid]
        self.num_valid = self.spks_valid.shape[0]
        self.frame_position = extras["frame_position_index"][idx_valid]
        self.frame_environment = extras["frame_environment_index"][idx_valid]
        self.frame_period = float(np.median(np.diff(session.timestamps)))

        # Caches for the expensive pieces so widget changes only redo what they invalidate.
        self._prediction_cache = {"spkmap": placefield_prediction[idx_valid]}
        self._reliability = smp.get_reliability()
        self._sort_cache = {}
        self._activity = None
        self._prediction = None

        xslice_start = int(np.clip(xslice_start, 0, self.num_valid - 1))
        xslice_stop = int(np.clip(xslice_stop, xslice_start + 1, self.num_valid))

        self.add_selection("prediction_from", value="spkmap", options=["spkmap", "placefield"])
        self.add_boolean("use_reliable", value=True)
        self.add_float("reliability_threshold", value=0.7, min=0, max=1)
        self.add_float("vmax", value=6, min=1, max=20)
        self.add_integer("xslice_start", value=xslice_start, min=0, max=self.num_valid - 1)
        self.add_integer("xslice_stop", value=xslice_stop, min=xslice_start + 1, max=self.num_valid)
        self.add_boolean("show_position", value=False)
        self.add_float("position_height", value=0.5, min=0.1, max=3.0)
        self.add_float("env_gap", value=0.2, min=0.0, max=3.0)
        self.add_boolean("show_zero_sigma", value=False)
        self.add_boolean("show_scalebar", value=False)
        self.add_float("scalebar_seconds", value=60.0, min=1.0, max=600.0)

        self.on_change(["prediction_from", "use_reliable", "reliability_threshold"], self.recompute_arrays)
        self.on_change("xslice_start", self.update_xslice_bounds)
        self.recompute_arrays(self.state)

    def update_xslice_bounds(self, state):
        """Keep the slice stop strictly after the slice start."""
        self.update_integer("xslice_stop", min=state["xslice_start"] + 1)

    def _get_prediction(self, prediction_from: str) -> np.ndarray:
        """Valid-frame place-field prediction, computed once per ``prediction_from`` mode."""
        if prediction_from not in self._prediction_cache:
            if prediction_from != "placefield":
                raise ValueError(f"Invalid prediction_from: {prediction_from}")
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
            self._prediction_cache[prediction_from] = pred[self.extras["idx_valid"]]
        return self._prediction_cache[prediction_from]

    def recompute_arrays(self, state):
        """Rebuild the ROI-filtered, ROI-sorted (rois, frames) rasters. Slicing them is then a view."""
        pred = self._get_prediction(state["prediction_from"])

        if state["use_reliable"]:
            threshold = state["reliability_threshold"]
            idx_reliable = np.where(np.any(np.stack([rval > threshold for rval in self._reliability.values], axis=0), axis=0))[0]
        else:
            idx_reliable = np.arange(self.spks.shape[1])

        # Sorting is deterministic given the ROI subset, so key the cache on the subset.
        sort_key = idx_reliable.tobytes()
        if sort_key not in self._sort_cache:
            self._sort_cache[sort_key] = sort_by_preferred_environment(self.smp, idx_rois=idx_reliable)
        idx_sort = self._sort_cache[sort_key]

        self._activity = self.spks_valid[:, idx_reliable].T[idx_sort]
        self._prediction = pred[:, idx_reliable].T[idx_sort]

    def plot(self, state):
        xslice = slice(state["xslice_start"], min(state["xslice_stop"], self.num_valid))
        activity = self._activity[:, xslice]
        prediction = self._prediction[:, xslice]
        num_frames = activity.shape[1]
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

        ax[0].imshow(activity, aspect="auto", cmap="gray_r", vmin=0, vmax=vmax)
        ax[1].imshow(prediction, aspect="auto", cmap="gray_r", vmin=0, vmax=vmax)
        ax[2].imshow(activity - prediction, aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)

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
            frame_position = self.frame_position[xslice]
            frame_environment = self.frame_environment[xslice]
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
            for spine in ["top", "right", "bottom", "left"]:
                ax_pos.spines[spine].set_visible(False)
            ax_pos.set_yinverted(True)

        # Optional time scalebar on the bottom raster, mirroring the colorscale inset placement.
        # Frames are non-contiguous (invalid frames are dropped), so this measures plotted imaging
        # time, not wall-clock time spanned by the slice.
        if state["show_scalebar"]:
            seconds = state["scalebar_seconds"]
            bar_width = (seconds / self.frame_period) / num_frames  # axes fraction
            x0 = 1.0 - (_INSET_RECT[0] + _INSET_RECT[2])  # same inset from the edge, mirrored to the left
            ycenter = _INSET_RECT[1] + _INSET_RECT[3] / 2
            label = f"{seconds / 60:g} min" if seconds >= 60 else f"{seconds:g} s"
            ax[2].plot([x0, x0 + bar_width], [ycenter, ycenter], transform=ax[2].transAxes, color="k", linewidth=2.5)
            ax[2].text(
                x0 + bar_width / 2,
                ycenter + 0.05,
                label,
                transform=ax[2].transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
            )

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


class PlaceFieldPredictionFocus(Viewer):
    """Place field, trial consistency, PF prediction, and prediction error for one ROI/environment.

    Up to four columns, all sharing the VR-position axis (columns 1, 3, 4) and the trial axis (row 1):

    1. Trial-by-position spike map with the place field (trial average) below.
    2. Per-trial consistency with the other trials, with the weighted average (reliability) below.
       Dropped when ``show_consistency`` is False.
    3. The place-field prediction of each trial, with its trial average below. Dropped when
       ``show_prediction`` is False.
    4. Prediction error (prediction - activity), with the per-position RMS error below.

    Turning both toggles off leaves the two-column version: the data and the error.

    The place-field prediction of a trial is the place field itself: ``get_placefield_prediction``
    predicts each frame from the trial-averaged map, so in position units every trial's prediction
    is the same row. Column 3 is therefore the place field tiled across trials (masked where the
    trial has no data), which is what makes column 4 a picture of trial-to-trial variability.
    """

    def __init__(
        self,
        env_maps: Maps,
        reliability: Reliability,
        fraction_active: np.ndarray,
        figsize: tuple[float, float] = (12.0, 6.0),
    ):
        self.env_maps = env_maps
        self.reliability = reliability
        self.fraction_active = fraction_active
        self.distcenters = env_maps.distcenters
        self.figsize = figsize

        self.num_rois = env_maps.spkmap[0].shape[0]
        self.num_envs = len(env_maps.environments)

        self.add_selection("roi", value=0, options=list(range(self.num_rois)))
        self.add_integer("env", value=0, min=0, max=self.num_envs - 1)
        self.add_float("reliability_threshold", value=0.7, min=0, max=1)
        self.add_float("fraction_active_threshold", value=0.5, min=0, max=1)
        self.add_float("vmax", value=5, min=1, max=20)
        self.add_float("vmax_error", value=5, min=1, max=20)
        self.add_float("fontsize", value=12, min=4, max=30)
        self.add_boolean("show_consistency", value=True)
        self.add_boolean("show_prediction", value=True)
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
        vmax = state["vmax"]
        vmax_error = state["vmax_error"]
        fontsize = state["fontsize"]
        show_consistency = state["show_consistency"]
        show_prediction = state["show_prediction"]

        # Set before any artist is created so axis labels, tick labels, and legends all pick it up.
        plt.rcParams["font.size"] = fontsize

        spkmap = self.env_maps.spkmap[env][roi]
        placefield = np.nanmean(spkmap, axis=0)

        # Every trial is predicted by the same place field; mask where the trial has no data.
        pred_spkmap = np.broadcast_to(placefield, spkmap.shape).copy()
        pred_spkmap[np.isnan(spkmap)] = np.nan
        error = pred_spkmap - spkmap
        avg_prediction = np.nanmean(pred_spkmap, axis=0)
        rms_error = np.sqrt(np.nanmean(error**2, axis=0))

        xlims = [self.distcenters[0], self.distcenters[-1]]
        ylims = [spkmap.shape[0] + 0.5, -0.5]
        extent = (xlims[0], xlims[1], spkmap.shape[0], 0)
        xlims_clean = (np.round(xlims[0] / 50) * 50, np.round(xlims[1] / 50) * 50)
        xlabels = [f"{int(round(x))}" for x in xlims_clean]
        # One y range for the three line panels so place field, prediction, and error are comparable.
        ymax_pf = np.nanmax([np.nanmax(placefield), np.nanmax(avg_prediction), np.nanmax(rms_error)]) * 1.2

        cmap = mpl.colormaps["gray_r"]
        norm = plt.Normalize(vmin=0, vmax=vmax)
        rgba = cmap(norm(np.linspace(0, vmax, 100)))

        cmap_err = mpl.colormaps["bwr"]
        norm_err = plt.Normalize(vmin=-vmax_error, vmax=vmax_error)
        rgba_err = cmap_err(norm_err(np.linspace(-vmax_error, vmax_error, 100)))

        # Columns are added left to right, so the optional ones only shift what follows them.
        width_ratios = [3]
        icol_consistency = None
        icol_prediction = None
        if show_consistency:
            icol_consistency = len(width_ratios)
            width_ratios.append(1)
        if show_prediction:
            icol_prediction = len(width_ratios)
            width_ratios.append(3)
        icol_error = len(width_ratios)
        width_ratios.append(3)

        fig = plt.figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(2, len(width_ratios), width_ratios=width_ratios, height_ratios=[6, 1])
        ax_spkmap = fig.add_subplot(gs[0, 0])
        ax_placefield = fig.add_subplot(gs[1, 0])
        ax_error = fig.add_subplot(gs[0, icol_error])
        ax_rms_error = fig.add_subplot(gs[1, icol_error])
        ax_colorbar = ax_spkmap.inset_axes([0.225, 0.15, 0.125, 0.7])
        ax_cbar_error = ax_error.inset_axes([0.225, 0.15, 0.125, 0.7])

        # ------------------------------------------------------------- col 1: activity --
        ax_spkmap.imshow(spkmap, interpolation="none", aspect="auto", cmap="gray_r", vmin=0, vmax=vmax, extent=extent)
        ax_spkmap.set_ylabel("Trials")
        ax_spkmap.set_xlim(xlims_clean)
        ax_spkmap.set_ylim(ylims[0], ylims[1])
        format_spines(
            ax_spkmap,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=xlims_clean,
            xticks=[],
            yticks=[],
            tick_length=4,
            tick_fontsize=fontsize,
            spines_visible=["left"],
        )

        ax_placefield.plot(self.distcenters, placefield, color="k", linewidth=1.5)
        ax_placefield.set_facecolor(("black", 0.04))
        ax_placefield.set_xlabel("VR Position", labelpad=-10)
        ax_placefield.set_xlim(xlims_clean)
        ax_placefield.set_ylim(-0.05, ymax_pf)
        ax_placefield.text(xlims[0], ymax_pf, "Place Field", ha="left", va="top", color="k", fontsize=fontsize)
        format_spines(
            ax_placefield,
            x_pos=-0.02,
            y_pos=-0.15,
            xbounds=xlims_clean,
            xticks=xlims_clean,
            xlabels=xlabels,
            yticks=[],
            tick_length=4,
            tick_fontsize=fontsize,
            spines_visible=["bottom"],
        )

        # ---------------------------------------------------------- col 2: consistency --
        if show_consistency:
            ax_consistency = fig.add_subplot(gs[0, icol_consistency])
            ax_reliability = fig.add_subplot(gs[1, icol_consistency])

            trial_weights = np.sqrt(np.mean(spkmap**2, axis=1))
            trial_consistency = _jit_reliability_loo(spkmap[None, ...])[0]
            trial_weights = trial_weights / np.max(trial_weights)

            idx_include = trial_weights > 0
            trial_numbers = np.arange(spkmap.shape[0])[idx_include]
            trial_weights = trial_weights[idx_include] / np.max(trial_weights[idx_include])
            trial_consistency = trial_consistency[idx_include]
            half_trial_number = max(trial_numbers) / 2

            ax_consistency.scatter(trial_consistency, trial_numbers, color="k", s=5, alpha=trial_weights)
            ax_consistency.set_facecolor(("black", 0.04))
            ax_consistency.set_xlim(-1.05, 1.05)
            ax_consistency.set_ylim(ylims[0], ylims[1])
            ax_consistency.set_xlabel(r"$\sigma$")
            ax_consistency.text(
                -0.5,
                half_trial_number,
                r"$\sigma = \mathrm{corr}(\langle\mathrm{other\ trials}\rangle)$",
                ha="center",
                va="center",
                rotation=90,
                fontsize=fontsize,
            )
            format_spines(
                ax_consistency,
                x_pos=-0.02,
                y_pos=-0.02,
                xbounds=(-1, 1),
                xticks=[-1, 0, 1],
                yticks=[],
                tick_length=4,
                tick_fontsize=fontsize,
                spines_visible=["bottom"],
            )

            reliability = np.sum(trial_weights * trial_consistency) / np.sum(trial_weights)
            ax_reliability.plot([-1, 1], [0, 0], color="black", linewidth=1.5)
            ax_reliability.plot([reliability], [0], color="black", marker="o", markersize=8)
            ax_reliability.set_xlim(-1, 1)
            ax_reliability.set_ylim(-0.05, 0.05)
            ax_reliability.set_xlabel("Reliability")
            format_spines(
                ax_reliability,
                x_pos=-0.02,
                y_pos=-0.02,
                xbounds=(-1, 1),
                xticks=[-1, 0, 1],
                yticks=[],
                tick_length=4,
                tick_fontsize=fontsize,
                spines_visible=["bottom"],
            )

        # ----------------------------------------------------------- col 3: prediction --
        if show_prediction:
            ax_prediction = fig.add_subplot(gs[0, icol_prediction])
            ax_avg_prediction = fig.add_subplot(gs[1, icol_prediction])

            ax_prediction.imshow(pred_spkmap, interpolation="none", aspect="auto", cmap="gray_r", vmin=0, vmax=vmax, extent=extent)
            ax_prediction.set_xlim(xlims_clean)
            ax_prediction.set_ylim(ylims[0], ylims[1])
            format_spines(
                ax_prediction,
                x_pos=-0.02,
                y_pos=-0.02,
                xbounds=xlims_clean,
                xticks=[],
                yticks=[],
                tick_length=4,
                tick_fontsize=fontsize,
                spines_visible=[],
            )

            ax_avg_prediction.plot(self.distcenters, avg_prediction, color="k", linewidth=1.5)
            ax_avg_prediction.set_facecolor(("black", 0.04))
            ax_avg_prediction.set_xlabel("VR Position", labelpad=-10)
            ax_avg_prediction.set_xlim(xlims_clean)
            ax_avg_prediction.set_ylim(-0.05, ymax_pf)
            ax_avg_prediction.text(xlims[0], ymax_pf, "PF Prediction", ha="left", va="top", color="k", fontsize=fontsize)
            format_spines(
                ax_avg_prediction,
                x_pos=-0.02,
                y_pos=-0.15,
                xbounds=xlims_clean,
                xticks=xlims_clean,
                xlabels=xlabels,
                yticks=[],
                tick_length=4,
                tick_fontsize=fontsize,
                spines_visible=["bottom"],
            )

        # ---------------------------------------------------------------- col 4: error --
        ax_error.imshow(error, interpolation="none", aspect="auto", cmap="bwr", vmin=-vmax_error, vmax=vmax_error, extent=extent)
        ax_error.set_xlim(xlims_clean)
        ax_error.set_ylim(ylims[0], ylims[1])
        format_spines(
            ax_error,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=xlims_clean,
            xticks=[],
            yticks=[],
            tick_length=4,
            tick_fontsize=fontsize,
            spines_visible=[],
        )

        ax_rms_error.plot(self.distcenters, rms_error, color="k", linewidth=1.5)
        ax_rms_error.set_facecolor(("black", 0.04))
        ax_rms_error.set_xlabel("VR Position", labelpad=-10)
        ax_rms_error.set_xlim(xlims_clean)
        ax_rms_error.set_ylim(-0.05, ymax_pf)
        ax_rms_error.text(xlims[0], ymax_pf, "RMS Error", ha="left", va="top", color="k", fontsize=fontsize)
        format_spines(
            ax_rms_error,
            x_pos=-0.02,
            y_pos=-0.15,
            xbounds=xlims_clean,
            xticks=xlims_clean,
            xlabels=xlabels,
            yticks=[],
            tick_length=4,
            tick_fontsize=fontsize,
            spines_visible=["bottom"],
        )

        # --------------------------------------------------------- inset colorscales --
        ax_colorbar.imshow(np.flipud(rgba[:, None, ...]), aspect="auto", extent=(0, 1, 0, 1))
        ax_colorbar.set_xticks([])
        ax_colorbar.set_yticks([])
        ax_colorbar.text(0.5, 0.02, r"0", fontsize=fontsize, ha="center", va="bottom", color="k")
        ax_colorbar.text(0.5, 0.98, f"{int(vmax)}", fontsize=fontsize, ha="center", va="top", color="w")
        ax_colorbar.set_ylabel("Fluorescence ($\sigma$)", fontsize=fontsize)

        ax_cbar_error.imshow(np.flipud(rgba_err[:, None, ...]), aspect="auto", extent=(0, 1, 0, 1))
        ax_cbar_error.set_xticks([])
        ax_cbar_error.set_yticks([])
        ax_cbar_error.text(0.5, 0.02, f"-{int(vmax_error)}", fontsize=fontsize, ha="center", va="bottom", color="w")
        ax_cbar_error.text(0.5, 0.98, f"{int(vmax_error)}", fontsize=fontsize, ha="center", va="top", color="w")
        ax_cbar_error.set_ylabel("Error ($\sigma$)", fontsize=fontsize)

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
        self.add_float_range("r2_ylim", value=(-1.0, 1.0), min=-1.0, max=1.0, step=0.05)
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

        # ax[1] and ax[2] share one user-set y range; spine bounds and ticks are clipped to it.
        ylim = tuple(state["r2_ylim"])
        ax[1].set_ylim(ylim)
        ax[2].set_ylim(ylim)
        ybounds = [max(min_r2, ylim[0]), min(max_r2, ylim[1])]
        # Tick the visible extremes (rounded inward so ticks never exceed the spine bounds), plus 0.
        ytick_low = float(np.ceil(ybounds[0] * 100) / 100)
        ytick_high = float(np.floor(ybounds[1] * 100) / 100)
        yticks = [ytick_low, ytick_high] if ytick_low >= 0 else [ytick_low, 0.0, ytick_high]

        # Format spines once ylims have been set
        for a in (ax[1], ax[2]):
            format_spines(
                a,
                x_pos=-0.02,
                y_pos=-0.02,
                xbounds=[-1, 1],
                ybounds=ybounds,
                xticks=[-1, 0, 1],
                yticks=yticks,
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
    show_scalebar: bool = False,
    scalebar_seconds: float = 60.0,
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
    show_scalebar : bool
        If True, draw a time scalebar on the bottom-left of the residual raster.
    scalebar_seconds : float
        Duration of the scalebar, in seconds of imaging time.
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
    viewer.update_boolean("show_position", value=show_position)
    viewer.update_float("position_height", value=position_height)
    viewer.update_float("env_gap", value=env_gap)
    viewer.update_boolean("show_zero_sigma", value=show_zero_sigma)
    viewer.update_boolean("show_scalebar", value=show_scalebar)
    viewer.update_float("scalebar_seconds", value=scalebar_seconds)

    # Seeding via update_* does not fire on_change before deployment, so refresh the cached rasters.
    viewer.recompute_arrays(viewer.state)

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


def example_placefield_prediction(
    session: B2Session,
    roi: int,
    env: int,
    reliability_threshold: float = 0.7,
    fraction_active_threshold: float = 0.5,
    vmax: float = 5,
    vmax_error: float = 5,
    fontsize: float = 12.0,
    show_consistency: bool = True,
    show_prediction: bool = True,
    figsize: tuple[float, float] = (12.0, 6.0),
    return_syd_viewer: bool = False,
):
    """
    Plot an example place field alongside its prediction and prediction error, in position units.

    Combines :func:`example_placefield` (columns 1-2: the trial-by-position spike map with the
    place field below, and the per-trial consistency with the reliability below) with the
    prediction panels of :func:`example_traversal` recast onto the VR-position axis (column 3: the
    place-field prediction of each trial with its trial average below; column 4: the prediction
    error with the per-position RMS error below). The gray_r and bwr colorscales are drawn as
    insets on the activity and error maps, in the style of :func:`example_placefield`.

    Columns 2 and 3 are optional: ``show_consistency=False, show_prediction=False`` leaves the
    two-column version, the data and the error.

    Parameters
    ----------
    session : B2Session
        Session to load the environment maps and reliability from.
    roi : int
        ROI to plot. Must pass the reliability / fraction-active filters.
    env : int
        Index into ``env_maps.environments``.
    reliability_threshold, fraction_active_threshold : float
        Filters defining which ROIs are selectable in the viewer.
    vmax : float
        Upper limit (in sigma) of the gray_r colorscale for the activity and prediction maps.
    vmax_error : float
        Saturation (in sigma) of the symmetric bwr colorscale for the error map.
    fontsize : float
        Single font size for the whole panel: axis labels, tick labels, in-axes annotations,
        colorscale labels, and legends.
    show_consistency : bool
        Draw the trial-consistency / reliability column (column 2).
    show_prediction : bool
        Draw the place-field prediction column (column 3).
    figsize : tuple[float, float]
        Figure size in inches. Not rescaled when columns are dropped, so the remaining columns
        get wider; pass a narrower width for the two-column (activity + error) version.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.

    Returns
    -------
    matplotlib.figure.Figure or PlaceFieldPredictionFocus
    """
    smp = SMPs.SpkmapProcessor(session, params=SMPs.SpkmapParams())

    env_maps = smp.get_env_maps()
    env_maps.distcenters = smp.dist_centers
    env_maps.pop_nan_positions()
    reliability = smp.get_reliability()
    fraction_active = np.stack([FractionActive.compute(spkmap, 2, 1) for spkmap in env_maps.spkmap])

    viewer = PlaceFieldPredictionFocus(env_maps, reliability, fraction_active, figsize=figsize)
    _seed_roi_filtered_viewer(
        viewer,
        env=env,
        roi=roi,
        reliability_threshold=reliability_threshold,
        fraction_active_threshold=fraction_active_threshold,
        vmax=vmax,
    )
    viewer.update_float("vmax_error", value=vmax_error)
    viewer.update_float("fontsize", value=fontsize)
    viewer.update_boolean("show_consistency", value=show_consistency)
    viewer.update_boolean("show_prediction", value=show_prediction)

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
    r2_ylim: tuple[float, float] = (-1.0, 1.0),
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
    r2_ylim : tuple[float, float]
        Shared y limits for the two R² panels, within ``(-1, 1)``.
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
    viewer.update_float_range("r2_ylim", value=tuple(r2_ylim))

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


# ---------------------------------------------------------------------------------------
# VR environment schematic
# ---------------------------------------------------------------------------------------

# Environments shown, top row first. These are the ATL cohort's three environments, ordered
# so the reward zone walks leftward down the figure (150 -> 100 -> 50 cm).
VR_SCHEMATIC_ENVS: tuple[int, ...] = (1, 3, 4)

# Reward-zone geometry is stored per session, not as a colony-wide constant: ENV_REWARD_MAP
# gives the zone *start* in cm, and the drawn width is a presentation choice.
VR_REWARD_ZONE_WIDTH_CM: float = 20.0
VR_REWARD_LEGEND_LABEL: str = "reward zones (90% of trials)"

# RenderParams fields, in the order the viewer registers them. Changing any of these makes
# the viewer shell out to Blender; every other parameter is pure matplotlib.
_VR_RENDER_PARAMS: tuple[str, ...] = (
    "entrance_offset_cm",
    "hfov_deg",
    "panel_aspect",
    "panel_width_px",
    "camera_height_cm",
    "yaw_deg",
    "use_dof",
    "light_scale",
    "exposure",
    "samples",
)


class VREnvironmentSchematic(Viewer):
    """The three VR environments as rendered room stills, stacked with reward-zone tracks.

    One row per environment: four panels showing what the mouse sees standing at the
    entrance of each room, and below them a track arrow with the reward zone marked. Arrow
    and zone take the environment's color from :data:`ENV_SLOT_COLORS`, the same palette the
    ``by_env`` panels of figure 3 use. The whole thing is drawn into a single axes in units
    of one panel height, with the figure sized so that unit is exactly ``panel_height_in``
    inches -- so every gap, arrow, and swatch keeps its proportion under any scaling.

    Parameters split into two groups. The render parameters (see
    :class:`~dimensionality_manuscript.blender.RenderParams`) control the camera and are
    resolved by driving Blender headlessly; results are cached on disk, so revisiting a
    setting is instant but a new one costs a couple of seconds per environment. The layout
    parameters are plain matplotlib and redraw immediately.
    """

    def __init__(self, envs: tuple[int, ...] = VR_SCHEMATIC_ENVS):
        self.envs = tuple(envs)

        # --- render parameters (each change re-renders in Blender) ---
        defaults = RenderParams()
        self.add_float("entrance_offset_cm", value=defaults.entrance_offset_cm, min=-10.0, max=45.0)
        self.add_float("hfov_deg", value=defaults.hfov_deg, min=30.0, max=RIG_HFOV_DEG)
        self.add_float("panel_aspect", value=defaults.panel_aspect, min=0.6, max=4.0)
        self.add_integer("panel_width_px", value=defaults.panel_width_px, min=160, max=1600)
        self.add_float("camera_height_cm", value=defaults.camera_height_cm, min=0.5, max=14.5)
        self.add_float("yaw_deg", value=defaults.yaw_deg, min=-90.0, max=90.0)
        self.add_boolean("use_dof", value=defaults.use_dof)
        self.add_float("light_scale", value=defaults.light_scale, min=0.1, max=5.0)
        self.add_float("exposure", value=defaults.exposure, min=-3.0, max=3.0)
        self.add_integer("samples", value=defaults.samples, min=4, max=128)

        # --- layout parameters (immediate redraw) ---
        # Finer step than the 0.01 default: this one is multiplied by ~6.7 layout units to
        # get the figure width, so 0.01 increments are ~0.07 in jumps in the saved figure.
        self.add_float("panel_height_in", value=0.85, min=0.3, max=3.0, step=0.001)
        self.add_float("room_gap", value=0.05, min=0.0, max=0.6)
        self.add_float("env_gap", value=0.34, min=0.0, max=1.5)
        self.add_float("arrow_gap", value=0.14, min=0.0, max=1.0)
        self.add_float("track_height", value=0.16, min=0.04, max=0.5)
        self.add_float("margin", value=0.06, min=0.0, max=0.5)
        self.add_float("panel_border", value=0.0, min=0.0, max=3.0)
        self.add_boolean("show_reward_zones", value=True)
        self.add_float("reward_zone_width_cm", value=VR_REWARD_ZONE_WIDTH_CM, min=2.0, max=60.0)
        self.add_float("reward_zone_alpha", value=0.4, min=0.0, max=1.0)
        self.add_boolean("show_legend", value=True)
        self.add_float("legend_yoffset", value=0.0, min=-0.5, max=1.5)
        self.add_boolean("show_scalebar", value=True)
        self.add_float("fontsize", value=9.0, min=4.0, max=24.0)

        self.on_change(list(_VR_RENDER_PARAMS), self.reload_images)
        self.reload_images(self.state)

    def reload_images(self, state):
        """Render (or fetch from cache) the room stills for every environment."""
        params = RenderParams(**{name: state[name] for name in _VR_RENDER_PARAMS})
        self.images = {env: load_vr_room_images(env, params) for env in self.envs}

        room_counts = {env: len(images) for env, images in self.images.items()}
        if len(set(room_counts.values())) != 1:
            raise RuntimeError(f"Environments returned different room counts: {room_counts}. Every environment should have four rooms.")
        self.num_rooms = next(iter(room_counts.values()))

    def _env_color(self, env: int):
        """Shared environment-slot color, matching the ``by_env`` panels in figure 3."""
        return ENV_SLOT_COLORS[self.envs.index(env) % len(ENV_SLOT_COLORS)]

    def plot(self, state):
        panel_w = state["panel_aspect"]
        room_gap = state["room_gap"]
        arrow_gap = state["arrow_gap"]
        track_height = state["track_height"]
        env_gap = state["env_gap"]
        margin = state["margin"]
        fontsize = state["fontsize"]
        show_legend = state["show_legend"]
        reward_zone_alpha = state["reward_zone_alpha"]

        # Everything below is in units of one panel height; the figure is then sized so that
        # unit is panel_height_in inches, which is what keeps the layout rigid under scaling.
        track_w = self.num_rooms * panel_w + (self.num_rooms - 1) * room_gap
        row_pitch = 1.0 + arrow_gap + track_height + env_gap
        panel_tops = [-row * row_pitch for row in range(len(self.envs))]
        arrow_ys = [top - 1.0 - arrow_gap - track_height / 2 for top in panel_tops]

        # The legend sits one environment-gap below the last track, in its own band, nudged
        # by legend_yoffset (positive pushes it further down).
        legend_y = arrow_ys[-1] - track_height / 2 - env_gap - track_height / 2 - state["legend_yoffset"]
        # min() rather than legend_y alone: a negative offset can lift the legend above the
        # last track, and the bottom edge has to follow whichever band ends up lowest.
        y_bottom = (min(legend_y, arrow_ys[-1]) if show_legend else arrow_ys[-1]) - track_height / 2

        width = track_w + 2 * margin
        height = (0.0 - y_bottom) + 2 * margin
        scale = state["panel_height_in"]

        plt.close("all")
        fig = plt.figure(figsize=(width * scale, height * scale))
        # A single axes filling the whole figure, no ticks, no spines. Because the axes box
        # aspect matches the data aspect exactly, set_aspect("equal") introduces no padding
        # and one data unit lands on exactly `scale` inches.
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(-margin, track_w + margin)
        ax.set_ylim(y_bottom - margin, margin)
        ax.set_axis_off()

        for env, panel_top, arrow_y in zip(self.envs, panel_tops, arrow_ys):
            color = self._env_color(env)

            for room, image in enumerate(self.images[env]):
                x0 = room * (panel_w + room_gap)
                ax.imshow(image, extent=(x0, x0 + panel_w, panel_top - 1.0, panel_top), aspect="auto", zorder=2)
                if state["panel_border"] > 0:
                    ax.add_patch(
                        Rectangle(
                            (x0, panel_top - 1.0),
                            panel_w,
                            1.0,
                            facecolor="none",
                            edgecolor="black",
                            linewidth=state["panel_border"],
                            zorder=3,
                        )
                    )

            # Track arrow: the mouse runs left to right over the full environment length.
            ax.annotate(
                "",
                xy=(track_w, arrow_y),
                xytext=(0.0, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.2, shrinkA=0, shrinkB=0, mutation_scale=fontsize * 1.4),
                zorder=4,
            )

            if state["show_reward_zones"]:
                # ENV_REWARD_MAP holds the zone start in cm on the 200 cm reference track.
                start = track_w * ENV_REWARD_MAP[env] / REFERENCE_ENV_LENGTH_CM
                zone_width = track_w * state["reward_zone_width_cm"] / REFERENCE_ENV_LENGTH_CM
                ax.add_patch(
                    Rectangle(
                        (start, arrow_y - track_height / 2),
                        zone_width,
                        track_height,
                        facecolor=color,
                        alpha=reward_zone_alpha,
                        edgecolor="none",
                        zorder=5,
                    )
                )

        if show_legend:
            # The zones are environment-colored, so the swatch is split into one segment per
            # environment rather than drawn in a neutral grey that matches nothing on the plot.
            swatch_w = 0.55 * panel_w / 1.6  # scales with panel width so the row stays balanced
            segment_w = swatch_w / len(self.envs)
            for segment, env in enumerate(self.envs):
                ax.add_patch(
                    Rectangle(
                        (segment * segment_w, legend_y - track_height / 2),
                        segment_w,
                        track_height,
                        facecolor=self._env_color(env),
                        alpha=reward_zone_alpha,
                        edgecolor="none",
                        zorder=5,
                    )
                )
            ax.text(swatch_w + 0.12, legend_y, VR_REWARD_LEGEND_LABEL, ha="left", va="center", fontsize=fontsize, zorder=5)

        if state["show_scalebar"]:
            ax.text(
                track_w,
                legend_y if show_legend else y_bottom,
                f"{REFERENCE_ENV_LENGTH_CM:g}cm",
                ha="right",
                va="center",
                fontsize=fontsize,
                zorder=5,
            )

        # Set last: imshow(aspect="auto") resets the axes aspect on every call, so an earlier
        # set_aspect would be silently undone. The figure was sized from the same layout, so
        # this adds no padding -- it just makes the units isotropic if figsize is overridden.
        ax.set_aspect("equal")
        return fig


def vr_environment_schematic(
    envs: tuple[int, ...] = VR_SCHEMATIC_ENVS,
    entrance_offset_cm: float = 2.0,
    hfov_deg: float = 90.0,
    panel_aspect: float = 1.6,
    panel_width_px: int = 480,
    camera_height_cm: float = 7.5,
    yaw_deg: float = 0.0,
    use_dof: bool = False,
    light_scale: float = 1.0,
    exposure: float = 0.0,
    samples: int = 32,
    panel_height_in: float = 0.85,
    room_gap: float = 0.05,
    env_gap: float = 0.34,
    arrow_gap: float = 0.14,
    track_height: float = 0.16,
    margin: float = 0.06,
    panel_border: float = 0.0,
    show_reward_zones: bool = True,
    reward_zone_width_cm: float = VR_REWARD_ZONE_WIDTH_CM,
    reward_zone_alpha: float = 0.4,
    show_legend: bool = True,
    legend_yoffset: float = 0.0,
    show_scalebar: bool = True,
    fontsize: float = 9.0,
    return_syd_viewer: bool = False,
):
    """
    The three VR environments, each as four room stills over a reward-zone track.

    One row per environment. Each row shows what the mouse sees standing at the entrance of
    each of the four rooms -- rendered from the ``vrEnvironment_*.blend`` files by driving
    Blender headlessly -- above an arrow spanning the 200 cm track with the reward zone
    drawn as a grey box at its true position. Everything lands in a single bare axes laid
    out in units of one panel height, and the figure is sized so that unit is exactly
    ``panel_height_in`` inches.

    Renders are cached on disk under ``RegistryPaths.cache_path / "vr_renders"``, keyed by
    the render parameters and the .blend modification times. A parameter set seen before
    loads instantly; a new one costs roughly two seconds per environment.

    Parameters
    ----------
    envs : tuple of int
        Environments to draw, top row first.
    entrance_offset_cm : float
        Camera position in cm past each room's doorway plane. 0 sits in the doorway itself;
        negative values look into the room from the previous one.
    hfov_deg : float
        Horizontal field of view. The rig's real optics are ~152 deg (``RIG_HFOV_DEG``),
        which is heavily fisheyed; 90 deg reads better at panel size.
    panel_aspect : float
        Panel width / height. Also sets the rendered aspect, so the vertical field of view
        follows from it and ``hfov_deg``.
    panel_width_px : int
        Rendered panel width in pixels.
    camera_height_cm : float
        Eye height above the floor; the corridor walls are 15 cm tall.
    yaw_deg : float
        Camera rotation off the track axis. 0 looks straight down the corridor.
    use_dof : bool
        Enable the camera's depth of field, which softens the far end of the corridor.
    light_scale : float
        Multiplier on every light's energy -- changes shading contrast.
    exposure : float
        Color-management exposure in stops -- brightness only, shading untouched.
    samples : int
        EEVEE render samples.
    panel_height_in : float
        Inches per layout unit, i.e. the height of one rendered panel.
    room_gap, env_gap, arrow_gap : float
        Gaps between panels in a row, between environment rows, and between a row's panels
        and its track arrow. In panel-height units.
    track_height : float
        Height of the arrow band and the reward-zone box, in panel-height units.
    margin : float
        Padding around the whole layout, in panel-height units.
    panel_border : float
        Line width of a black border around each panel; 0 draws none.
    show_reward_zones : bool
        Draw the reward-zone box on each track.
    reward_zone_width_cm : float
        Drawn width of the reward zone. Reward geometry is stored per session rather than as
        a colony constant, so only the zone start comes from the data; this is presentation.
    reward_zone_alpha : float
        Opacity of the reward-zone boxes, which take their environment's color.
    show_legend : bool
        Draw the reward-zone swatch and label below the last environment.
    legend_yoffset : float
        Extra vertical shift of the legend row, in panel-height units; positive pushes it
        further below the last track. The figure grows or shrinks to follow it.
    show_scalebar : bool
        Draw the track-length label at the right of the legend row.
    fontsize : float
        Font size in points for the legend and scale labels.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.

    Returns
    -------
    matplotlib.figure.Figure or VREnvironmentSchematic

    Notes
    -----
    **There is no figsize argument** -- unlike the other figure factories in this module,
    the figure size is *derived* from the layout. The layout is built in abstract units
    where 1 unit is one panel height, and ``panel_height_in`` is the only knob in absolute
    units. Writing ``n_rooms`` for the rooms per environment (4) and ``n_envs`` for the
    number of rows::

        track_w   = n_rooms * panel_aspect + (n_rooms - 1) * room_gap
        row_pitch = 1 + arrow_gap + track_height + env_gap

        width_units  = track_w + 2 * margin
        height_units = n_envs * row_pitch - env_gap + 2 * margin
                       + (env_gap + track_height + legend_yoffset if show_legend else 0)

        figsize = (width_units * panel_height_in, height_units * panel_height_in)

    So width responds to ``panel_aspect``, ``room_gap`` and ``margin``; height responds to
    ``arrow_gap``, ``track_height``, ``env_gap``, ``margin``, ``show_legend`` and
    ``legend_yoffset``; and ``panel_height_in`` scales both together. Because the axes box
    aspect is then exactly the data aspect, ``set_aspect("equal")`` adds no padding and one
    data unit lands on exactly ``panel_height_in`` inches in *both* directions.

    Two things do not scale, because they are specified in points rather than layout units:
    ``fontsize`` and ``panel_border``. Doubling ``panel_height_in`` leaves the legend text
    at the same physical size, so it reads as relatively smaller.

    To target a figure width, back-solve ``panel_height_in`` from ``width_units`` rather than
    guessing. For a 7-inch column at default gaps::

        width_units = 4 * 1.6 + 3 * 0.05 + 2 * 0.06                        # 6.67
        fig = vr_environment_schematic(panel_height_in=7.0 / width_units)  # 1.0495 -> 1.049

    I'm targeting figure width of 2.25 and use room_gap=0, margin=0.05, panel_aspect=1.6, so:
        width_units = 4 * 1.6 + 3 * 0.0 + 2 * 0.05 = 6.67
        fig_width = width_units * panel_height_in = 2.25 = 6.67 * x
        panel_height_in = 2.25 / 6.67 = 0.346

    Syd rounds every float parameter to its slider step, so the width lands within one step
    of the target rather than exactly on it -- here ``panel_height_in`` has ``step=0.001``,
    giving 6.997 in instead of 7.000. That is well under a printer's tolerance; if a figure
    must be exact to the pixel, scale it at the LaTeX or Illustrator stage instead of
    fighting the slider. Height then follows from the vertical parameters; adjust
    ``env_gap`` or ``panel_aspect`` if the result is too tall for the space.
    """
    viewer = VREnvironmentSchematic(envs=envs)

    viewer.update_float("entrance_offset_cm", value=entrance_offset_cm)
    viewer.update_float("hfov_deg", value=hfov_deg)
    viewer.update_float("panel_aspect", value=panel_aspect)
    viewer.update_integer("panel_width_px", value=panel_width_px)
    viewer.update_float("camera_height_cm", value=camera_height_cm)
    viewer.update_float("yaw_deg", value=yaw_deg)
    viewer.update_boolean("use_dof", value=use_dof)
    viewer.update_float("light_scale", value=light_scale)
    viewer.update_float("exposure", value=exposure)
    viewer.update_integer("samples", value=samples)

    viewer.update_float("panel_height_in", value=panel_height_in)
    viewer.update_float("room_gap", value=room_gap)
    viewer.update_float("env_gap", value=env_gap)
    viewer.update_float("arrow_gap", value=arrow_gap)
    viewer.update_float("track_height", value=track_height)
    viewer.update_float("margin", value=margin)
    viewer.update_float("panel_border", value=panel_border)
    viewer.update_boolean("show_reward_zones", value=show_reward_zones)
    viewer.update_float("reward_zone_width_cm", value=reward_zone_width_cm)
    viewer.update_float("reward_zone_alpha", value=reward_zone_alpha)
    viewer.update_boolean("show_legend", value=show_legend)
    viewer.update_float("legend_yoffset", value=legend_yoffset)
    viewer.update_boolean("show_scalebar", value=show_scalebar)
    viewer.update_float("fontsize", value=fontsize)

    # Seeding via update_* does not fire on_change before deployment, so pull the renders for
    # the seeded parameters explicitly.
    viewer.reload_images(viewer.state)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig
