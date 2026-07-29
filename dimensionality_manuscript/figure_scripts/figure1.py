from collections import defaultdict
from itertools import combinations

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle
from rastermap import Rastermap
from scipy.stats import chi2, norm, rankdata
from statsmodels.regression.mixed_linear_model import MixedLM
from syd import Viewer

from vrAnalysis.helpers import vectorRSquared, edge2center
from vrAnalysis.helpers.plotting import format_spines, errorPlot, beeswarm
from vrAnalysis.helpers.typing import PrettyDatetime
from vrAnalysis.helpers.vrsupport import _jit_reliability_loo
from vrAnalysis.sessions import B2Session
from vrAnalysis.processors import spkmaps as SMPs
from vrAnalysis.processors.spkmaps import Maps, Reliability

from dimensionality_manuscript.figure_scripts import session_cache
from dimensionality_manuscript.configs.pfpred_quality import PFPredQualityConfig, _kde_r2
from dimensionality_manuscript.configs.placefield_structure import CrossValidatedPlacefieldsConfig, PlacefieldPredictionConfig
from dimensionality_manuscript.configs.behavior_speed_env import ENV_REWARD_MAP, REFERENCE_ENV_LENGTH_CM
from dimensionality_manuscript import ResultsAggregator, average_by_mouse
from dimensionality_manuscript.blender import RIG_HFOV_DEG, RenderParams, load_vr_room_images
from dimensionality_manuscript.env_order import ENV_SLOT_COLORS, ENV_NUM_COLORS, MAX_ENV_SLOTS, _session_sort_key

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
    spks = session_cache.get_spks(session, params=smp.params)
    reliability = session_cache.get_reliability(session, params=smp.params)
    placefield_prediction = session_cache.get_prediction(session, "spkmap", params=smp.params)
    extras = session_cache.get_prediction_extras(session, params=smp.params)
    idx_best_environment = extras["frame_environment_index"] == idx_env
    idx_keep = extras["idx_valid"] & idx_best_environment
    spks_valid = spks[idx_keep]
    pfpred_valid = placefield_prediction[idx_keep]
    r2 = vectorRSquared(pfpred_valid, spks_valid, axis=0)
    r2[r2 < -1] = np.nan
    return spks_valid, pfpred_valid, r2, reliability


# Axes-fraction rectangle (x0, y0, width, height) of the colorscale insets. The scalebar is
# mirrored from it: same vertical center, same inset from the axes edge (left instead of right).
_INSET_RECT = (0.72, 0.10, 0.25, 0.2)


def _add_colorscale_inset(
    ax,
    cmap_name,
    right_label,
    left_label=None,
    right_color="w",
    left_color="k",
    fontsize=8.0,
    text_y=0.5,
    inset_rect=_INSET_RECT,
):
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
    # for spine in axins.spines.values():
    #     spine.set_visible(False)
    axins.text(
        0.97,
        text_y,
        right_label,
        transform=axins.transAxes,
        ha="right",
        va="center",
        color=right_color,
        fontsize=fontsize,
    )
    if left_label is not None:
        axins.text(
            0.03,
            text_y,
            left_label,
            transform=axins.transAxes,
            ha="left",
            va="center",
            color=left_color,
            fontsize=fontsize,
        )


class StackedRasterFocus(Viewer):
    """Interactive stacked raster: activity, PF prediction, residuals."""

    def __init__(
        self,
        results: ResultsAggregator,
        mouse: str | None,
        session: str | None,
        figsize: tuple[float, float],
        xslice_start: int,
        xslice_stop: int,
    ):
        self.results = results
        self.figsize = figsize
        self._config = PlacefieldPredictionConfig()
        store_rows = results.store.summary_table(
            analysis_type=self._config.display_name,
            session_ids=list(results.session_ids),
            schema_version=self._config.schema_version,
        )
        self._store_rows = {row["session_id"]: row for row in store_rows if row["analysis_key"] == self._config.key()}
        mouse_names = np.asarray(results.mouse_names)
        rows_by_mouse = {
            str(mouse_name): [
                int(row)
                for row in sorted(np.flatnonzero(mouse_names == mouse_name), key=lambda row: _session_sort_key(results.sessions[row]))
                if results.sessions[row].session_uid in self._store_rows
            ]
            for mouse_name in sorted(set(mouse_names))
        }
        self._rows_by_mouse = {name: rows for name, rows in rows_by_mouse.items() if rows}
        if not self._rows_by_mouse:
            raise ValueError("No PlacefieldPredictionConfig schema-v1 results are stored for these sessions.")
        self._session_rows = {
            name: {results.sessions[row].session_print(): row for row in rows} for name, rows in self._rows_by_mouse.items()
        }
        initial_mouse = mouse if mouse in self._rows_by_mouse else next(iter(self._rows_by_mouse))
        session_options = list(self._session_rows[initial_mouse])
        initial_session = session if session in session_options else session_options[0]

        self._result_cache: dict[int, dict] = {}
        self._sort_cache: dict[tuple, np.ndarray] = {}
        self.session = None
        self.smp = None
        self.spks = None
        self.spks_valid = None
        self.frame_position = None
        self.frame_environment = None
        self._reliability = None
        self._activity = None
        self._prediction = None

        self.add_selection("mouse", value=initial_mouse, options=list(self._rows_by_mouse))
        self.add_selection("session", value=initial_session, options=session_options)
        self.add_selection("sort_method", value="environment", options=list(session_cache.SORT_METHODS))
        self.add_boolean("use_reliable", value=True)
        self.add_float("reliability_threshold", value=0.7, min=0, max=1)
        self.add_float("vmax", value=6, min=1, max=20)
        self.add_integer("xslice_start", value=max(0, xslice_start), min=0, max=max(0, xslice_start))
        self.add_integer("xslice_stop", value=max(xslice_start + 1, xslice_stop), min=1, max=max(xslice_start + 1, xslice_stop))
        self.add_boolean("show_position", value=False)
        self.add_float("position_height", value=0.5, min=0.1, max=3.0)
        self.add_float("env_gap", value=0.2, min=0.0, max=3.0)
        self.add_boolean("show_zero_sigma", value=False)
        self.add_boolean("show_scalebar", value=False)
        self.add_float("scalebar_seconds", value=60.0, min=1.0, max=600.0)
        self.add_float("fontsize", value=8.0, min=1.0, max=30.0)
        self.add_float("colorscale_text_y", value=0.5, min=0.0, max=1.0, step=0.001)
        self.add_float("colorscale_inset_x", value=_INSET_RECT[0], min=0.0, max=1.0, step=0.001)
        self.add_float("colorscale_inset_y", value=_INSET_RECT[1], min=0.0, max=1.0, step=0.001)
        self.add_float("colorscale_inset_width", value=_INSET_RECT[2], min=0.001, max=1.0, step=0.001)
        self.add_float("colorscale_inset_height", value=_INSET_RECT[3], min=0.001, max=1.0, step=0.001)

        self.on_change("mouse", self.update_session)
        self.on_change("session", self.load_session)
        self.on_change(["sort_method", "use_reliable", "reliability_threshold"], self.recompute_arrays)
        self.on_change("xslice_start", self.update_xslice_bounds)
        self.load_session(self.state)

    def update_session(self, state):
        """Offer only sessions of the selected mouse with a stored PF prediction."""
        options = list(self._session_rows[state["mouse"]])
        current = state["session"] if state["session"] in options else options[0]
        self.update_selection("session", value=current, options=options)
        self.load_session({**state, "session": current})

    def load_session(self, state):
        """Load activity and the stored PlacefieldPredictionConfig result for one session."""
        row = self._session_rows[state["mouse"]][state["session"]]
        self.session = self.results.sessions[row]
        if row not in self._result_cache:
            result = self._config.get_result(self.results.store, self._store_rows[self.session.session_uid])
            if result is None or "placefield_prediction" not in result or "spks_std" not in result:
                raise KeyError(
                    f"{self.session.session_print()} lacks placefield_prediction or spks_std "
                    "(PlacefieldPredictionConfig schema v1 required)."
                )
            self._result_cache[row] = result
        result = self._result_cache[row]

        previous_spks_type = self.session.params.spks_type
        self.session.params.spks_type = self._config.spks_type
        try:
            raw_roi_indices = np.flatnonzero(self.session.idx_rois)
            raw_spks = np.asarray(self.session.spks)[:, raw_roi_indices]
            scale = np.asarray(result["spks_std"])[raw_roi_indices]
            self.spks = np.divide(
                raw_spks,
                scale[None, :],
                out=np.full(raw_spks.shape, np.nan, dtype=float),
                where=np.isfinite(scale[None, :]) & (scale[None, :] > 0),
            )
            prediction = np.asarray(result["placefield_prediction"])[:, raw_roi_indices]
            num_frames = min(self.spks.shape[0], prediction.shape[0])
            self.spks = self.spks[:num_frames]
            prediction = prediction[:num_frames]
            prediction = np.divide(
                prediction,
                scale[None, :],
                out=np.full(prediction.shape, np.nan, dtype=float),
                where=np.isfinite(scale[None, :]) & (scale[None, :] > 0),
            )
            self.smp = session_cache.get_smp(self.session)
            self._reliability = session_cache.get_reliability(self.session)
            frame_position, _, frame_environment, _ = self.smp.get_frame_behavior()
            frame_position = frame_position[:num_frames]
            frame_environment = frame_environment[:num_frames]
        finally:
            self.session.params.spks_type = previous_spks_type

        idx_valid = np.any(np.isfinite(prediction), axis=1)
        self.spks_valid = self.spks[idx_valid]
        self._stored_prediction_valid = prediction[idx_valid]
        self.num_valid = int(np.sum(idx_valid))
        num_bins = len(self.smp.dist_edges) - 1
        env_length = float(self.session.env_length[0])
        self.frame_position = frame_position[idx_valid] / env_length * num_bins
        env_values = sorted(np.unique(frame_environment[idx_valid][np.isfinite(frame_environment[idx_valid])]))
        env_to_slot = {env: slot for slot, env in enumerate(env_values)}
        self.frame_environment = np.array([env_to_slot.get(env, -1) for env in frame_environment[idx_valid]])
        self.frame_period = float(np.median(np.diff(self.session.timestamps)))

        start = int(np.clip(state["xslice_start"], 0, max(self.num_valid - 1, 0)))
        stop = int(np.clip(state["xslice_stop"], start + 1, max(self.num_valid, start + 1)))
        self.update_integer("xslice_start", value=start, min=0, max=max(self.num_valid - 1, 0))
        self.update_integer("xslice_stop", value=stop, min=start + 1, max=max(self.num_valid, start + 1))
        self.recompute_arrays({**state, "xslice_start": start, "xslice_stop": stop})

    def update_xslice_bounds(self, state):
        """Keep the slice stop strictly after the slice start."""
        self.update_integer("xslice_stop", min=state["xslice_start"] + 1)

    def _env_color(self, env_index: int):
        """Shared environment-slot color, matching the VR schematic and the ``by_env`` panels.

        ``frame_environment_index`` is the position of an environment in the session's
        (ascending) environment list, which is the experience-order slot :data:`ENV_SLOT_COLORS`
        is indexed by.
        """
        return ENV_SLOT_COLORS[int(env_index) % len(ENV_SLOT_COLORS)]

    def recompute_arrays(self, state):
        """Rebuild the ROI-filtered, ROI-sorted (rois, frames) rasters. Slicing them is then a view."""
        if state["use_reliable"]:
            threshold = state["reliability_threshold"]
            idx_reliable = np.where(np.any(np.stack([rval > threshold for rval in self._reliability.values], axis=0), axis=0))[0]
        else:
            idx_reliable = np.arange(self.spks.shape[1])

        sort_key = (self.session.session_uid, state["sort_method"], idx_reliable.tobytes())
        if sort_key not in self._sort_cache:
            if state["sort_method"] == "environment":
                previous_spks_type = self.session.params.spks_type
                self.session.params.spks_type = self._config.spks_type
                try:
                    self._sort_cache[sort_key] = session_cache.get_roi_sort(self.session, "environment", idx_reliable)
                finally:
                    self.session.params.spks_type = previous_spks_type
            else:
                self._sort_cache[sort_key] = Rastermap().fit(self.spks_valid[:, idx_reliable].T).isort
        idx_sort = self._sort_cache[sort_key]

        self._activity = self.spks_valid[:, idx_reliable].T[idx_sort]
        self._prediction = self._stored_prediction_valid[:, idx_reliable].T[idx_sort]

    def plot(self, state):
        xslice = slice(state["xslice_start"], min(state["xslice_stop"], self.num_valid))
        activity = self._activity[:, xslice]
        prediction = self._prediction[:, xslice]
        num_frames = activity.shape[1]
        vmax = state["vmax"]

        show_position = state["show_position"]
        show_zero_sigma = state["show_zero_sigma"]
        position_height = state["position_height"]
        fontsize = state["fontsize"]

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
            "Deconvolved Calcium Activity" if self._config.spks_type == "oasis" else "Fluorescence",
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
                fontsize=fontsize,
            )
        ax[0].set_ylabel("ROIs", fontsize=fontsize)
        ax[1].set_ylabel("ROIs", fontsize=fontsize)
        ax[2].set_ylabel("ROIs", fontsize=fontsize)

        for spine in ["top", "right", "bottom", "left"]:
            ax[0].spines[spine].set_visible(False)
            ax[1].spines[spine].set_visible(False)
            ax[2].spines[spine].set_visible(False)

        # Colorscale insets: gray_r on the first raster, bwr on the last raster.
        inset_rect = (
            state["colorscale_inset_x"],
            state["colorscale_inset_y"],
            state["colorscale_inset_width"],
            state["colorscale_inset_height"],
        )
        zero_label = r"$0\,\sigma$" if show_zero_sigma else None
        _add_colorscale_inset(
            ax[0],
            "gray_r",
            left_label=zero_label,
            right_label=rf"${int(vmax)}\,\sigma$",
            left_color="k",
            right_color="w",
            fontsize=fontsize,
            text_y=state["colorscale_text_y"],
            inset_rect=inset_rect,
        )
        _add_colorscale_inset(
            ax[2],
            "bwr",
            left_label=rf"$-{int(vmax)}\,\sigma$",
            right_label=rf"$+{int(vmax)}\,\sigma$",
            left_color="w",
            right_color="w",
            fontsize=fontsize,
            text_y=state["colorscale_text_y"],
            inset_rect=inset_rect,
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
            # Plot each environment as its own colored line, offset vertically by a gapped band.
            # NaN-masking the other environments' frames breaks the line between environments.
            for env in np.unique(frame_environment[frame_environment >= 0]):
                y = frame_position.astype(float).copy()
                y[frame_environment != env] = np.nan
                y = y + env * band
                xins = np.insert(xpos_base, breaks, np.nan)
                yins = np.insert(y, breaks, np.nan)
                ax_pos.plot(xins, yins, color=self._env_color(env), linewidth=1)
            ax_pos.set_ylabel("Pos.", fontsize=fontsize)
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
                fontsize=fontsize,
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
        self.spks = session_cache.get_spks(smp.session, params=smp.params)

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
        results: ResultsAggregator,
        mouse: str | None = None,
        example_session: str | None = None,
        figsize: tuple[float, float] = (12.0, 6.0),
    ):
        self.results = results
        self.figsize = figsize
        self._example_config = CrossValidatedPlacefieldsConfig()
        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(mouse_name) for mouse_name in mouse_names})
        self._sessions_by_mouse = {
            mouse_name: {
                results.sessions[row].session_print(): results.sessions[row]
                for row in sorted(
                    np.flatnonzero(mouse_names == mouse_name),
                    key=lambda row: _session_sort_key(results.sessions[row]),
                )
            }
            for mouse_name in self.mice
        }
        initial_mouse = mouse if mouse is not None else self.mice[0]
        initial_sessions = list(self._sessions_by_mouse[initial_mouse])
        initial_session = example_session if example_session is not None else initial_sessions[0]

        self.add_selection("mouse", value=initial_mouse, options=self.mice)
        self.add_selection("example_session", value=initial_session, options=initial_sessions)
        self.add_integer("roi", value=0, min=0, max=0)
        self.add_integer("env", value=0, min=0, max=0)
        self.add_float("vmax", value=5, min=1, max=20)
        self.add_float("vmax_error", value=5, min=1, max=20)
        self.add_float("fontsize", value=12, min=4, max=30)
        self.add_boolean("show_consistency", value=True)
        self.add_boolean("show_prediction", value=True)
        self.on_change("mouse", self.update_example_session)
        self.on_change("example_session", self.update_example_bounds)
        self.update_example_session(self.state)

    def update_example_session(self, state):
        """Offer the selected mouse's sessions, in chronological order."""
        labels = list(self._sessions_by_mouse[state["mouse"]])
        current = state["example_session"] if state["example_session"] in labels else labels[0]
        self.update_selection("example_session", value=current, options=labels)
        self.update_example_bounds({**state, "example_session": current})

    def update_example_bounds(self, state):
        """Set ROI/environment bounds for the selected session."""
        env_maps = self._get_env_maps(state)
        self.update_integer("roi", max=max(env_maps.spkmap[0].shape[0] - 1, 0))
        self.update_integer("env", max=max(len(env_maps.environments) - 1, 0))

    def _get_env_maps(self, state) -> Maps:
        """Maps in the same session-filtered ROI coordinates used by ``r2_familiarity``."""
        session = self._sessions_by_mouse[state["mouse"]][state["example_session"]]
        previous_spks_type = session.params.spks_type
        session.params.spks_type = self._example_config.spks_type
        try:
            return session_cache.get_env_maps(session)
        finally:
            session.params.spks_type = previous_spks_type

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

        env_maps = self._get_env_maps(state)
        distcenters = env_maps.distcenters
        spkmap = env_maps.spkmap[env][roi]
        placefield = np.nanmean(spkmap, axis=0)

        # Every trial is predicted by the same place field; mask where the trial has no data.
        pred_spkmap = np.broadcast_to(placefield, spkmap.shape).copy()
        pred_spkmap[np.isnan(spkmap)] = np.nan
        error = pred_spkmap - spkmap
        avg_prediction = np.nanmean(pred_spkmap, axis=0)
        rms_error = np.sqrt(np.nanmean(error**2, axis=0))

        xlims = [distcenters[0], distcenters[-1]]
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

        ax_placefield.plot(distcenters, placefield, color="k", linewidth=1.5)
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

            ax_avg_prediction.plot(distcenters, avg_prediction, color="k", linewidth=1.5)
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

        ax_rms_error.plot(distcenters, rms_error, color="k", linewidth=1.5)
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


def _round_up_labels(ticks) -> list[str]:
    """Tick labels rounded up to a single decimal place (``-0.0`` is normalized to ``0.0``)."""
    labels = []
    for tick in ticks:
        value = float(np.ceil(float(tick) * 10) / 10)
        labels.append(f"{value if value != 0 else 0.0:.1f}")
    return labels


def _decimal_yticks(low: float, high: float, step: float = 0.1) -> list[float]:
    """Multiples of ``step`` inside ``[low, high]``.

    Used instead of matplotlib's locator so several panels of the same quantity tick at exactly
    the same values, and so 0 is always ticked whenever it is on the axis.
    """
    # Nudged tolerances: the bounds are usually themselves arithmetic on multiples of the step,
    # so an exact endpoint can land a hair outside it in floating point.
    first = int(np.ceil(low / step - 1e-9))
    last = int(np.floor(high / step + 1e-9))
    return [round(index * step, 10) for index in range(first, last + 1)]


def _example_roi_traces(session: B2Session, roi: int, idx_env: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    One ROI's activity and place-field prediction on the predictable frames.

    Parameters
    ----------
    session : B2Session
    roi : int
        ROI index in session-filtered coordinates.
    idx_env : int or None
        Keep only the frames the mouse spent in this environment. None keeps every predictable
        frame, whichever environment it came from.

    Returns
    -------
    data, prediction : np.ndarray
    """
    smp = session_cache.get_smp(session)
    spks = session_cache.get_spks(session, params=smp.params)
    prediction = session_cache.get_prediction(session, "spkmap", params=smp.params)
    extras = session_cache.get_prediction_extras(session, params=smp.params)
    idx_keep = extras["idx_valid"]
    if idx_env is not None:
        idx_keep = idx_keep & (extras["frame_environment_index"] == idx_env)
    return spks[idx_keep, roi], prediction[idx_keep, roi]


def _fit_square_panels(fig, axs, tolerance: float = 0.005, max_iter: int = 4) -> None:
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


def _draw_example_roi_panel(
    ax,
    data: np.ndarray,
    prediction: np.ndarray,
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
    ax.set_xlabel("Data", fontsize=fontsize, labelpad=-8)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=-8)
    format_spines(
        ax,
        x_pos=-0.05,
        y_pos=-0.05,
        xbounds=[0, axmax],
        ybounds=[0, axmax],
        xticks=[0, axmax],
        yticks=[0, axmax],
        xlabels=["0", "1"],
        ylabels=["0", "1"],
        tick_length=4,
        tick_fontsize=fontsize,
        spines_visible=["left", "bottom"],
    )


class R2PlacefieldFocus(Viewer):
    """Three-panel R² vs reliability plot with selectable ROI and environment.

    The example ROI's data-vs-prediction panel is either its own leftmost panel or, when
    ``example_as_inset`` is set, an inset placed on the session R² vs reliability panel (which
    then leaves a two-panel figure). It is drawn identically either way apart from font size,
    which drops to ``inset_fontsize`` when it is an inset.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        session: B2Session,
        smp: SMPs.SpkmapProcessor,
        idx_env: int,
        fontsize: float = 9.0,
        inset_fontsize: float = 7.0,
        figsize: tuple[float, float] = (8.0, 2.0),
    ):
        self.results = results
        self.session = session
        self.smp = smp
        self.fontsize = fontsize
        self.inset_fontsize = inset_fontsize
        self.figsize = figsize
        self.num_rois = session_cache.get_spks(session, params=smp.params).shape[1]
        self.num_envs = len(session_cache.get_env_maps(session, params=smp.params).environments)
        self.add_integer("env", value=idx_env, min=0, max=self.num_envs - 1)
        self.add_selection("roi", value=0, options=list(range(self.num_rois)))
        self.add_selection("example_style", value="points", options=["points", "density"])
        self.add_selection("cloud_style", value="hex", options=["hex", "scatter"])
        self.add_selection("hex_count_norm", value="linear", options=["linear", "log"])
        self.add_float("cloud_alpha", value=0.55, min=0.0, max=1.0)
        self.add_float_range("r2_ylim", value=(-1.0, 1.0), min=-1.0, max=1.0, step=0.05)
        self.add_boolean("example_as_inset", value=False)
        self.add_float("inset_x", value=0.06, min=0.0, max=1.0, step=0.01)
        self.add_float("inset_y", value=0.58, min=0.0, max=1.0, step=0.01)
        self.add_float("inset_width", value=0.32, min=0.05, max=1.0, step=0.01)
        self.add_float("inset_height", value=0.38, min=0.05, max=1.0, step=0.01)
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
        plt.rcParams["font.size"] = self.fontsize
        roi = state["roi"]
        idx_env = self.idx_env
        spks_valid = self.spks_valid
        pfpred_valid = self.pfpred_valid
        r2 = self.r2
        reliability = self.reliability

        plt.close("all")
        # The example ROI panel is either the leftmost axes or an inset on the session R² panel.
        if state["example_as_inset"]:
            fig, axs = plt.subplots(1, 2, figsize=self.figsize, layout="constrained")
            ax_session, ax_summary = axs
            inset_rect = [state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]]
            ax_example = ax_session.inset_axes(inset_rect)
        else:
            fig, axs = plt.subplots(1, 3, figsize=self.figsize, layout="constrained")
            ax_example, ax_session, ax_summary = axs
        ax = [ax_example, ax_session, ax_summary]

        # The example panel is drawn the same way whether it is ax[0] or an inset; only its font
        # size changes, since an inset has much less room for labels.
        example_fontsize = self.inset_fontsize if state["example_as_inset"] else self.fontsize
        _draw_example_roi_panel(
            ax[0],
            spks_valid.T[roi],
            pfpred_valid.T[roi],
            style=state["example_style"],
            fontsize=example_fontsize,
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
                markersize=2.5,
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
            markersize=7.5,
            linestyle="none",
            zorder=10,
        )
        ax[1].set_xlim(-1, 1)
        ax[1].set_xlabel("Spatial Reliability", fontsize=self.fontsize)
        ax[1].set_ylabel(r"$R^2$", fontsize=self.fontsize)

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
        ax[2].set_xlabel("Spatial Reliability", fontsize=self.fontsize)
        ax[2].set_ylabel(r"$R^2$", fontsize=self.fontsize)
        legend_elements = [
            Line2D([0], [0], color="k", alpha=alpha_example, linewidth=linewidth_example, label="mouse"),
            Line2D([0], [0], color="blue", alpha=alpha_highlight, linewidth=linewidth_example, label="example"),
            Line2D([0], [0], color="k", linewidth=linewidth_average, label="average"),
        ]
        ax[2].legend(handles=legend_elements, fontsize=self.fontsize)

        # ax[1] and ax[2] share one user-set y range; spine bounds and ticks are clipped to it.
        ylim = tuple(state["r2_ylim"])
        ax[1].set_ylim(ylim)
        ax[2].set_ylim(ylim)
        ybounds = [max(min_r2, ylim[0]), min(max_r2, ylim[1])]
        # Tick the visible extremes (rounded inward so ticks never exceed the spine bounds), plus 0.
        ytick_low = float(np.ceil(ybounds[0] * 100) / 100)
        ytick_high = float(np.floor(ybounds[1] * 100) / 100)
        yticks = [0, ytick_high] if ytick_low >= 0 else [0.0, ytick_high]

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
                xlabels=_round_up_labels([-1, 0, 1]),
                ylabels=_round_up_labels(yticks),
                tick_length=4,
                tick_fontsize=self.fontsize,
                spines_visible=["left", "bottom"],
            )
        return fig


# Result keys of BehaviorSpeedEnvConfig used by the speed figure, and the config param axes we
# expose as selections (each dropdown's options come straight from the aggregator's param_axes).
_SPEED_CURVE_KEYS: dict[str, str] = {"all": "speed_curve_all", "first": "speed_curve_first"}
_BEHAVIOR_SPEED_PARAM_AXES: tuple[str, ...] = ("num_bins", "speed_threshold", "regularization")
# Human-readable names for the two trial sets, used in the printed statistics report.
_SPEED_TRIAL_SET_LABELS: dict[str, str] = {"all": "all trials", "first": "1st trial of block"}


def _significance_label(pvalue: float) -> str:
    """Star notation for a p-value: ``***`` < 0.001, ``**`` < 0.01, ``*`` < 0.05, else ``n.s.``."""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment of a family of p-values."""
    order = np.argsort(pvalues)
    m = pvalues.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _env_speed_mixedlm(window_speed: np.ndarray, env_list: list[int]) -> dict | None:
    """
    Mixed-effects test of an environment effect on per-mouse windowed speed.

    Fits ``speed ~ env + (1 | mouse)`` (treatment-coded environments, ML rather than REML so
    the likelihoods are comparable) and compares it against the intercept-only null
    ``speed ~ 1 + (1 | mouse)`` with a likelihood-ratio test. Mice contribute only the
    environments they actually ran, which the random intercept handles as unbalanced data.

    Parameters
    ----------
    window_speed : np.ndarray
        ``(n_mice, n_envs)`` window-averaged speed, NaN where a mouse lacks an environment.
    env_list : list[int]
        Environment identity of each column of ``window_speed``.

    Returns
    -------
    dict or None
        Omnibus and Holm-adjusted pairwise results, or None if fewer than two environments
        have data.
    """
    mouse_idx, col_idx = np.nonzero(np.isfinite(window_speed))
    y = window_speed[mouse_idx, col_idx]
    columns = sorted(set(col_idx.tolist()))
    if len(columns) < 2:
        return None

    # Treatment coding: intercept + one dummy per environment beyond the first (the reference).
    k = len(columns)
    exog = np.zeros((y.size, k))
    exog[:, 0] = 1.0
    for j, col in enumerate(columns[1:], start=1):
        exog[:, j] = col_idx == col

    full = MixedLM(y, exog, groups=mouse_idx).fit(reml=False)
    null = MixedLM(y, exog[:, :1], groups=mouse_idx).fit(reml=False)
    lr_stat = 2.0 * (full.llf - null.llf)
    lr_df = k - 1

    # Wald contrasts on the fixed effects; the reference level has an implicit coefficient of 0.
    # MixedLM orders the fixed effects first in the parameter vector, so the leading k x k block
    # of the covariance matrix is their covariance.
    fe = np.asarray(full.fe_params)
    cov_fe = np.asarray(full.cov_params())[:k, :k]
    pairs, estimates, standard_errors = [], [], []
    for a, b in combinations(range(k), 2):
        contrast = np.zeros(k)
        if a > 0:
            contrast[a] = -1.0
        if b > 0:
            contrast[b] = 1.0
        pairs.append((env_list[columns[a]], env_list[columns[b]]))
        estimates.append(float(contrast @ fe))
        standard_errors.append(float(np.sqrt(contrast @ cov_fe @ contrast)))

    estimates = np.asarray(estimates)
    standard_errors = np.asarray(standard_errors)
    z = estimates / standard_errors
    p_raw = 2.0 * norm.sf(np.abs(z))

    return dict(
        envs=[env_list[col] for col in columns],
        n_mice_per_env=[int(np.sum(col_idx == col)) for col in columns],
        n_mice=int(np.unique(mouse_idx).size),
        n_obs=int(y.size),
        env_means=[float(np.mean(y[col_idx == col])) for col in columns],
        lr_stat=float(lr_stat),
        lr_df=int(lr_df),
        lr_pvalue=float(chi2.sf(lr_stat, lr_df)),
        pairs=pairs,
        differences=estimates,
        standard_errors=standard_errors,
        z_values=z,
        p_values=p_raw,
        p_values_holm=_holm_adjust(p_raw),
    )


def _format_speed_stats(stats: dict, window: tuple[float, float]) -> str:
    """Render the output of :func:`_env_speed_mixedlm` (per trial set) as a text report."""
    lines = [f"speed ~ env + (1 | mouse), window [{window[0]:g}, {window[1]:g}) cm"]
    for key, result in stats.items():
        label = _SPEED_TRIAL_SET_LABELS[key]
        if result is None:
            lines.append(f"  {label}: fewer than two environments with data -- skipped")
            continue
        counts = ", ".join(f"env {env}: {n}" for env, n in zip(result["envs"], result["n_mice_per_env"]))
        means = ", ".join(f"env {env}: {mu:.2f}" for env, mu in zip(result["envs"], result["env_means"]))
        lines.append(f"  {label}: {result['n_mice']} mice, {result['n_obs']} mouse-environment observations ({counts})")
        lines.append(f"    mean speed ({means}) cm/s")
        lines.append(f"    omnibus LRT vs env-free null: chi2({result['lr_df']}) = {result['lr_stat']:.3f}, p = {result['lr_pvalue']:.3g}")
        for (a, b), diff, se, z, p, p_holm in zip(
            result["pairs"],
            result["differences"],
            result["standard_errors"],
            result["z_values"],
            result["p_values"],
            result["p_values_holm"],
        ):
            lines.append(f"    env {b} - env {a} = {diff:+.3f} cm/s (SE {se:.3f}), z = {z:+.2f}, p = {p:.3g}, Holm p = {p_holm:.3g}")
    return "\n".join(lines)


class MouseSpeedFocus(Viewer):
    """Per-environment mouse speed over VR position, loaded from precomputed results.

    Reads the ``BehaviorSpeedEnvConfig`` aggregator: for the selected config parameters
    (``num_bins``, ``speed_threshold``, ``regularization``) and trial set (all trials vs the
    first trial of each block) it assembles a ``(mice, envs, bins)`` speed array by mapping each
    session's stored ``speed_curve_*`` rows onto the global environment axis via its stored
    ``environments`` key, then averaging across sessions within a mouse.

    For each selected environment the mouse-average speed curve is drawn with a shaded
    ±standard-error band, colored per environment (``tab10``). Both trial sets are drawn:
    all trials (solid) and the first trial of each block (dashed). Each environment's
    reward zone (from :data:`ENV_REWARD_MAP`) is marked either by a dotted line at its start
    or by a shaded patch spanning the zone, colored to match.

    Windowed statistics (:meth:`compute_stats`) fit ``speed ~ env + (1 | mouse)`` to the
    per-mouse mean speed over a fixed position window, separately for each trial set.
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
        self.add_selection("reward_style", options=["patch", "vlines"], value="patch")
        self.add_float("reward_width_cm", value=20.0, min=1.0, max=100.0)
        self.add_float("reward_alpha", value=0.2, min=0.0, max=1.0)
        self.add_boolean("show_legend", value=True)
        self.add_boolean("show_env_legend", value=False)
        self.add_boolean("extra_y_space", value=False)
        self.add_float("alpha_band", value=0.2, min=0.0, max=1.0)
        self.add_float("linewidth_average", value=2.0, min=0.1, max=5.0)
        self.add_float("fontsize", value=9.0, min=4.0, max=24.0)
        self.add_float("stat_window_start", value=40.0, min=0.0, max=REFERENCE_ENV_LENGTH_CM)
        self.add_float("stat_window_end", value=50.0, min=0.0, max=REFERENCE_ENV_LENGTH_CM)
        self.add_boolean("print_stats", value=False)
        self.add_boolean("show_significance", value=True)
        self.add_selection("significance_trial_set", options=list(_SPEED_CURVE_KEYS), value="all")

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

    def _env_color(self, env: int):
        """tab10 color for an environment, indexed by its position in ``env_list``."""
        return ENV_NUM_COLORS[env]

    def compute_stats(self, state) -> dict[str, dict | None]:
        """
        Mixed-effects test of the environment effect on windowed speed, per trial set.

        Averages each mouse's speed curve over the position window
        ``[stat_window_start, stat_window_end)`` and fits ``speed ~ env + (1 | mouse)``
        against an env-free null (see :func:`_env_speed_mixedlm`). Only the selected
        environments are included; a mouse missing an environment simply contributes fewer
        rows.

        Returns
        -------
        dict
            ``{"all": result, "first": result}``, each the dict returned by
            :func:`_env_speed_mixedlm` (or None when fewer than two environments have data).
        """
        window = (float(state["stat_window_start"]), float(state["stat_window_end"]))
        selected = [int(env) for env in state["environments"]]
        env_cols = [e for e, env in enumerate(self.env_list) if env in selected]
        envs = [self.env_list[e] for e in env_cols]
        # Half-open window: a bin center at exactly stat_window_end belongs to the next window.
        in_window = (self.dist_centers >= window[0]) & (self.dist_centers < window[1])

        stats = {}
        for name, key in _SPEED_CURVE_KEYS.items():
            block = self.speed[key][:, env_cols, :][..., in_window]  # (mice, selected envs, window bins)
            # Mean over the window, NaN where a mouse never ran that environment (nanmean would
            # warn on those all-NaN slices).
            valid = np.isfinite(block)
            counts = valid.sum(axis=-1)
            totals = np.where(valid, block, 0.0).sum(axis=-1)
            window_speed = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
            stats[name] = _env_speed_mixedlm(window_speed, envs)
        return stats

    def draw(self, state, ax):
        """Draw the speed curves onto ``ax``.

        Split out of :meth:`plot` so the combined schematic+speed figure
        (:func:`vr_schematic_and_speed`) can draw onto an externally placed axes instead of a
        standalone figure.
        """
        selected = [int(env) for env in state["environments"]]
        xvals = self.dist_centers
        env_length = REFERENCE_ENV_LENGTH_CM
        all_key = _SPEED_CURVE_KEYS["all"]
        first_key = _SPEED_CURVE_KEYS["first"]
        lw = state["linewidth_average"]
        fontsize = state["fontsize"]

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

        # Significance of the omnibus environment effect over the statistics window, at the window
        # center, top-aligned with the reward-zone patches (which span 0 to ymax).
        if state["show_significance"] and len(drawn_envs) > 1:
            stats = self.compute_stats(state)[state["significance_trial_set"]]
            window_center = 0.5 * (float(state["stat_window_start"]) + float(state["stat_window_end"]))
            ax.text(window_center, ymax, _significance_label(stats["lr_pvalue"]), ha="center", va="top", fontsize=fontsize)

        # Optionally drop the lower limit below 0, making room for a legend under the curves.
        ax.set_ylim(-10 if state["extra_y_space"] else 0, ymax)

        # Reward markers: drawn after ymax is known so they span only the data range [0, ymax],
        # not the negative legend margin. ``patch`` shades the whole reward zone, ``vlines``
        # marks only its start.
        if state["show_reward"]:
            for env in drawn_envs:
                start = self.reward_position[env]
                if state["reward_style"] == "patch":
                    width = min(state["reward_width_cm"], env_length - start)
                    # zorder=0 keeps the patch behind the speed curves and their error bands.
                    ax.add_patch(
                        Rectangle(
                            (start, 0),
                            width,
                            ymax,
                            facecolor=self._env_color(env),
                            edgecolor="none",
                            alpha=state["reward_alpha"],
                            zorder=0,
                        )
                    )
                else:
                    ax.vlines(start, 0, ymax, color=self._env_color(env), linestyle=":", linewidth=1.0)

        # Keep y-ticks at physical speeds (>= 0); the negative margin is legend space only.
        yticks = [t for t in ax.get_yticks() if 0 <= t <= ymax]

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

        # Custom legend. The optional "Envs" entry is one handle whose segments run
        # blue->orange->green (the env colors); HandlerTuple packs the sub-lines side by side
        # across that single handle.
        handles, labels = [], []
        if state["show_env_legend"] and drawn_envs:
            handles.append(tuple(Line2D([0], [0], color=self._env_color(env), linewidth=lw) for env in drawn_envs))
            labels.append("Envs")
        if state["show_first"]:
            handles.append(Line2D([0], [0], color="0.3", linewidth=lw, linestyle="--"))
            labels.append("1st of block")
        # Patches are self-explanatory in place (colored by environment, at each reward zone), so
        # only the dotted-line style earns a legend entry.
        if state["show_reward"] and state["reward_style"] == "vlines":
            handles.append(Line2D([0], [0], color="0.3", linewidth=1.0, linestyle=":"))
            labels.append("reward zones")
        if state["show_legend"] and handles:
            # pad=0 packs the env segments flush against each other (one continuous swatch).
            ax.legend(
                handles,
                labels,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
                loc="lower right",
                frameon=True,
                fontsize=fontsize,
            )

        if state["print_stats"]:
            window = (float(state["stat_window_start"]), float(state["stat_window_end"]))
            print(_format_speed_stats(self.compute_stats(state), window))

    def plot(self, state):
        plt.close("all")
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, layout="constrained")
        self.draw(state, ax)
        return fig


def mouse_speed_by_environment(
    results: ResultsAggregator,
    num_bins: int = 100,
    speed_threshold: float = -np.inf,
    regularization: float = 1.0,
    environments: list[int] | None = None,
    show_first: bool = True,
    show_reward: bool = True,
    reward_style: str = "patch",
    reward_width_cm: float = 20.0,
    reward_alpha: float = 0.2,
    show_legend: bool = True,
    show_env_legend: bool = False,
    extra_y_space: bool = False,
    alpha_band: float = 0.2,
    linewidth_average: float = 2.5,
    fontsize: float = 9.0,
    stat_window: tuple[float, float] = (40.0, 50.0),
    print_stats: bool = False,
    show_significance: bool = True,
    significance_trial_set: str = "all",
    figsize: tuple[float, float] = (7.0, 5.0),
    return_syd_viewer: bool = False,
):
    """
    Mouse speed as a function of VR position, per environment, aggregated across mice.

    Loads the precomputed ``BehaviorSpeedEnvConfig`` results (which already exclude the
    CR_Hippocannula mice and any session whose reward layout does not match ``ENV_REWARD_MAP``).
    For each environment the mouse-average speed curve is drawn with a shaded ±standard-error
    band, colored per environment. Both trial sets are shown: all trials (solid) and the first
    trial of each block (dashed); each environment's reward zone is marked by a shaded patch
    (``reward_style="patch"``) or a dotted line at the zone start (``reward_style="vlines"``).

    With ``print_stats=True`` the environment effect on speed is tested over a fixed position
    window: each mouse's speed is averaged within the window and fit with
    ``speed ~ env + (1 | mouse)``, compared against ``speed ~ 1 + (1 | mouse)`` by a
    likelihood-ratio test, plus Holm-corrected pairwise environment contrasts. Both trial sets
    are tested separately. See :meth:`MouseSpeedFocus.compute_stats`.

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
        Environments to draw (and to include in the statistics). If None, all available
        environments are used.
    show_first, show_reward, show_legend : bool
        Toggle the first-of-block curve, the reward-zone markers, and the legend.
    reward_style : str
        ``"patch"`` shades each reward zone, ``"vlines"`` draws a dotted line at its start.
    reward_width_cm : float
        Width of the reward patch, in position units, starting at the reward-zone start.
    reward_alpha : float
        Opacity of the reward patch.
    show_env_legend : bool
        Include the multi-colored "Envs" entry in the legend.
    extra_y_space : bool
        Extend the y-axis below 0, leaving blank space under the curves for the legend.
    alpha_band : float
        Opacity of the ±standard-error band.
    linewidth_average : float
        Line width of the mouse-average curves.
    fontsize : float
        Font size of every text element: axis labels, tick labels, legend, significance marker.
    stat_window : tuple[float, float]
        Position window (cm) the speed statistics average over, half-open: ``[start, stop)``.
    print_stats : bool
        Print the mixed-effects statistics report when plotting.
    show_significance : bool
        Mark the omnibus environment effect above the curves at the window center, as
        ``***`` (p < 0.001), ``**`` (p < 0.01), ``*`` (p < 0.05) or ``n.s.``.
    significance_trial_set : str
        Which model the marked p-value comes from: ``"all"`` or ``"first"``.
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
    viewer.update_selection("reward_style", value=reward_style)
    viewer.update_float("reward_width_cm", value=reward_width_cm)
    viewer.update_float("reward_alpha", value=reward_alpha)
    viewer.update_boolean("show_legend", value=show_legend)
    viewer.update_boolean("show_env_legend", value=show_env_legend)
    viewer.update_boolean("extra_y_space", value=extra_y_space)
    viewer.update_float("alpha_band", value=alpha_band)
    viewer.update_float("linewidth_average", value=linewidth_average)
    viewer.update_float("fontsize", value=fontsize)
    viewer.update_float("stat_window_start", value=stat_window[0])
    viewer.update_float("stat_window_end", value=stat_window[1])
    viewer.update_boolean("print_stats", value=print_stats)
    viewer.update_boolean("show_significance", value=show_significance)
    viewer.update_selection("significance_trial_set", value=significance_trial_set)

    # Seeding via update_* does not fire on_change before deployment, so refresh the arrays for the
    # seeded selections (the on_change hook still drives live updates once the viewer is deployed).
    viewer.recompute_arrays(viewer.state)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


def stacked_raster_plot(
    results: ResultsAggregator,
    mouse: str | None = None,
    session: str | None = None,
    use_reliable: bool = True,
    reliability_threshold: float = 0.7,
    sort_method: str = "environment",
    xslice: slice = slice(0, 2000),
    vmax: float = 6,
    figsize: tuple[float, float] = (12, 6),
    show_position: bool = False,
    position_height: float = 0.5,
    env_gap: float = 0.2,
    show_zero_sigma: bool = False,
    show_scalebar: bool = False,
    scalebar_seconds: float = 60.0,
    fontsize: float = 8.0,
    colorscale_text_y: float = 0.5,
    colorscale_inset_position: tuple[float, float, float, float] = _INSET_RECT,
    return_syd_viewer: bool = False,
):
    """
    Plot a stacked raster plot of the deconvolved calcium activity and the prediction from the place field.

    Parameters
    ----------
    results : ResultsAggregator
        Results collection whose store contains ``PlacefieldPredictionConfig`` schema-v1 results.
    mouse : str or None
        Initial mouse selection. None uses the first mouse with a stored prediction.
    session : str or None
        Initial printable session label (``mouse/date/id``). None uses the selected mouse's first
        session with a stored prediction.
    use_reliable : bool
    reliability_threshold : float
    sort_method : {"environment", "rastermap"}
        ROI ordering of every raster. ``environment`` sorts by preferred environment then
        place-field position; ``rastermap`` sorts by a Rastermap embedding of the activity,
        which is position-agnostic and takes tens of seconds the first time a given ROI
        subset is fit. Either way the sort is computed on the activity.
    xslice : slice
    vmax : float
    figsize : tuple[float, float]
    show_position : bool
        If True, add a 4th panel showing the mouse position over the plotted frames, one
        colored trace per environment (:data:`ENV_SLOT_COLORS`, shared with the VR schematic).
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
    fontsize : float
        Font size of every text element: panel titles, axis labels, colorscale end labels,
        and the scalebar label.
    colorscale_text_y : float
        Vertical position of the colorscale labels in inset-axes coordinates. Values below
        ``0.5`` shift the labels down. The Syd control uses a step of ``0.001``.
    colorscale_inset_position : tuple of float
        Shared ``(x, y, width, height)`` of the colorscale insets on ``ax[0]`` and ``ax[2]``,
        in parent-axes coordinates. Each Syd control uses a step of ``0.001``.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    """
    viewer = StackedRasterFocus(
        results,
        mouse,
        session,
        figsize,
        xslice.start if xslice.start is not None else 0,
        xslice.stop if xslice.stop is not None else 2000,
    )
    viewer.update_selection("sort_method", value=sort_method)
    viewer.update_boolean("use_reliable", value=use_reliable)
    viewer.update_float("reliability_threshold", value=reliability_threshold)
    viewer.update_float("vmax", value=vmax)
    viewer.update_boolean("show_position", value=show_position)
    viewer.update_float("position_height", value=position_height)
    viewer.update_float("env_gap", value=env_gap)
    viewer.update_boolean("show_zero_sigma", value=show_zero_sigma)
    viewer.update_boolean("show_scalebar", value=show_scalebar)
    viewer.update_float("scalebar_seconds", value=scalebar_seconds)
    viewer.update_float("fontsize", value=fontsize)
    viewer.update_float("colorscale_text_y", value=colorscale_text_y)
    viewer.update_float("colorscale_inset_x", value=colorscale_inset_position[0])
    viewer.update_float("colorscale_inset_y", value=colorscale_inset_position[1])
    viewer.update_float("colorscale_inset_width", value=colorscale_inset_position[2])
    viewer.update_float("colorscale_inset_height", value=colorscale_inset_position[3])

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
    smp = session_cache.get_smp(session)
    env_maps = session_cache.get_env_maps(session)
    reliability = session_cache.get_reliability(session)
    fraction_active = session_cache.get_fraction_active(session)

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
    smp = session_cache.get_smp(session)
    env_maps = session_cache.get_env_maps(session)
    reliability = session_cache.get_reliability(session)
    fraction_active = session_cache.get_fraction_active(session)

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
    results: ResultsAggregator,
    mouse: str | None = None,
    example_session: str | None = None,
    roi: int = 0,
    env: int = 0,
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
    results : ResultsAggregator
        Results whose sessions are available to the mouse/session selectors.
    mouse : str or None
        Example mouse. None takes the first alphabetically.
    example_session : str or None
        Which of that mouse's sessions to use, as ``session_print()`` prints it
        (``"mouse/date/id"``). None takes the mouse's first session.
    roi : int
        ROI index within ``example_session``, in session-filtered ROI coordinates. This is
        exactly the same coordinate system as ``r2_familiarity(..., roi=...)``.
    env : int
        Index into ``env_maps.environments``.
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
    viewer = PlaceFieldPredictionFocus(
        results,
        mouse=mouse,
        example_session=example_session,
        figsize=figsize,
    )
    viewer.update_integer("roi", value=roi)
    viewer.update_integer("env", value=env)
    viewer.update_float("vmax", value=vmax)
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
    example_style: str = "points",
    cloud_style: str = "hex",
    cloud_alpha: float | None = None,
    hex_count_norm: str = "linear",
    r2_ylim: tuple[float, float] = (-1.0, 1.0),
    example_as_inset: bool = False,
    inset_position: tuple[float, float, float, float] = (0.06, 0.58, 0.32, 0.38),
    fontsize: float = 9.0,
    inset_fontsize: float = 7.0,
    figsize: tuple[float, float] = (8.0, 2.0),
    return_syd_viewer: bool = False,
):
    """
    Example ROI data vs PF prediction, session R² vs spatial reliability, and the across-mouse
    R² summary.

    Parameters
    ----------
    example_style : {"points", "density"}
        How to draw the example ROI's data vs PF prediction: a translucent point cloud or a
        filled seaborn 2D KDE of the same points. A unity line is drawn either way.
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
    example_as_inset : bool
        Draw the example ROI panel as an inset on the session R² panel instead of as its own
        axes, leaving a 1x2 figure instead of a 1x3 one.
    inset_position : tuple[float, float, float, float]
        ``(x0, y0, width, height)`` of the inset in axes fractions of the session R² panel.
        Only used when ``example_as_inset`` is True.
    fontsize : float
        Single font size for the whole figure: axis labels, tick labels, and the legend.
    inset_fontsize : float
        Font size of the example panel's labels and tick labels, used only when
        ``example_as_inset`` is True.
    figsize : tuple[float, float]
        Figure size in inches. Not rescaled by ``example_as_inset``, so pass a narrower width
        for the two-panel version.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.

    Returns
    -------
    matplotlib.figure.Figure or R2PlacefieldFocus
    """
    if example_style not in ("points", "density"):
        raise ValueError(f"example_style must be 'points' or 'density', got {example_style!r}")
    if cloud_style not in ("hex", "scatter"):
        raise ValueError(f"cloud_style must be 'hex' or 'scatter', got {cloud_style!r}")
    if hex_count_norm not in ("linear", "log"):
        raise ValueError(f"hex_count_norm must be 'linear' or 'log', got {hex_count_norm!r}")
    if cloud_alpha is None:
        cloud_alpha = 0.55 if cloud_style == "hex" else 0.1

    smp = session_cache.get_smp(session)

    viewer = R2PlacefieldFocus(results, session, smp, idx_env, fontsize=fontsize, inset_fontsize=inset_fontsize, figsize=figsize)
    viewer.update_integer("env", value=idx_env)
    viewer.update_selection("roi", value=roi)
    viewer.update_selection("example_style", value=example_style)
    viewer.update_selection("cloud_style", value=cloud_style)
    viewer.update_selection("hex_count_norm", value=hex_count_norm)
    viewer.update_float("cloud_alpha", value=cloud_alpha)
    viewer.update_float_range("r2_ylim", value=tuple(r2_ylim))
    viewer.update_boolean("example_as_inset", value=example_as_inset)
    viewer.update_float("inset_x", value=inset_position[0])
    viewer.update_float("inset_y", value=inset_position[1])
    viewer.update_float("inset_width", value=inset_position[2])
    viewer.update_float("inset_height", value=inset_position[3])

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

        width_ratios = [1, 0.33] if swarm_mode == "pooled" else [1, 1]
        plt.close("all")
        fig, ax = plt.subplots(1, 2, figsize=self.figsize, layout="constrained", width_ratios=width_ratios)

        # --- ax[0]: per-mouse reliability histograms (thin black) + across-mouse average (thick) ---
        hist_max = np.nanmax(mouse_hist)
        ax[0].plot(centers, mouse_hist.T, color=("k", state["hist_alpha"]), linewidth=1.0)
        ax[0].plot(centers, np.nanmean(mouse_hist, axis=0), color="k", linewidth=2.0)
        ax[0].axvline(threshold, color="0.6", linestyle=":", linewidth=1.0)
        ax[0].set_xlim(-1, 1)
        ylim = ax[0].get_ylim()
        ax[0].text(threshold + 0.05, ylim[1] * 0.9, f"{threshold:.1f}", color="0.6", fontsize=self.fontsize - 1, ha="left", va="top")
        ax[0].set_xlabel("Spatial Reliability")
        ax[0].set_ylabel("Fraction of Cells")
        legend_elements = [
            Line2D([0], [0], color="k", alpha=state["hist_alpha"], linewidth=1.0, label="each"),
            Line2D([0], [0], color="k", linewidth=2.0, label="avg"),
        ]
        ax[0].legend(handles=legend_elements, fontsize=self.fontsize, frameon=False, handlelength=1.25)
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
            ax[1].plot(beewidth * offsets, vals, linestyle="none", color="k", marker="o", markersize=2.5, alpha=0.8)
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

    def layout_metrics(self, state) -> dict[str, float]:
        """Layout geometry in panel-height units, ahead of any ``panel_height_in`` scaling.

        Reused by :meth:`plot` (to size its own standalone figure) and by
        :func:`vr_schematic_and_speed` (to size the schematic's row within a shared figure)
        without duplicating the row/legend placement math.

        Returns
        -------
        dict
            ``track_w`` (the 0-to-track_w span the arrow/reward-zones occupy), ``width`` and
            ``height`` (the full axes extent including ``margin``), ``arrow_ys`` (per-environment
            arrow y-positions), ``panel_tops``, ``legend_y``, and ``y_bottom``.
        """
        panel_w = state["panel_aspect"]
        room_gap = state["room_gap"]
        arrow_gap = state["arrow_gap"]
        track_height = state["track_height"]
        env_gap = state["env_gap"]
        margin = state["margin"]
        show_legend = state["show_legend"]

        # Everything here is in units of one panel height; the figure is then sized so that
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

        return {
            "track_w": track_w,
            "width": track_w + 2 * margin,
            "height": (0.0 - y_bottom) + 2 * margin,
            "panel_tops": panel_tops,
            "arrow_ys": arrow_ys,
            "legend_y": legend_y,
            "y_bottom": y_bottom,
        }

    def draw(self, state, ax):
        """Draw the schematic onto ``ax``, whose data limits are set to the layout's own units.

        Split out of :meth:`plot` so the combined schematic+speed figure
        (:func:`vr_schematic_and_speed`) can draw onto an externally placed axes instead of a
        standalone figure.
        """
        panel_w = state["panel_aspect"]
        room_gap = state["room_gap"]
        track_height = state["track_height"]
        margin = state["margin"]
        fontsize = state["fontsize"]
        show_legend = state["show_legend"]
        reward_zone_alpha = state["reward_zone_alpha"]

        metrics = self.layout_metrics(state)
        track_w = metrics["track_w"]
        panel_tops = metrics["panel_tops"]
        arrow_ys = metrics["arrow_ys"]
        legend_y = metrics["legend_y"]
        y_bottom = metrics["y_bottom"]

        # No ticks, no spines -- just the rendered content, filling whatever box ``ax`` occupies.
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
        # set_aspect would be silently undone. The box this axes occupies was sized from the
        # same layout (see plot / vr_schematic_and_speed), so this adds no padding -- it just
        # makes the units isotropic if that box's aspect were ever overridden.
        ax.set_aspect("equal")

    def plot(self, state):
        metrics = self.layout_metrics(state)
        scale = state["panel_height_in"]

        plt.close("all")
        fig = plt.figure(figsize=(metrics["width"] * scale, metrics["height"] * scale))
        # A single axes filling the whole figure, no ticks, no spines.
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.draw(state, ax)
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
    print(fig.get_size_inches())
    plt.show()
    return fig


def vr_schematic_and_speed(
    results: ResultsAggregator,
    fig_width: float = 3.5,
    fig_height: float = 6.0,
    left_margin: float = 0.16,
    right_margin: float = 0.98,
    top_margin: float = 0.02,
    bottom_margin: float = 0.14,
    panel_gap: float = 0.03,
    schematic_kwargs: dict | None = None,
    speed_kwargs: dict | None = None,
):
    """
    Vertically stacked figure: the VR environment schematic on top, mouse speed below it,
    sharing a common physical x-axis so the reward zones in the schematic line up with the
    reward zones drawn on the speed curves.

    Both panels are built the same way their standalone factories build them --
    :func:`vr_environment_schematic` and :func:`mouse_speed_by_environment` -- by seeding a
    viewer via ``return_syd_viewer=True`` and then drawing its state onto an axes placed by
    this function instead of the viewer's own ``plot``.

    Sizing is driven entirely by ``fig_width`` and ``fig_height``. The schematic has a fixed
    aspect ratio (:meth:`VREnvironmentSchematic.layout_metrics`), so given the shared axes
    width (``fig_width * (right_margin - left_margin)``) its height is fully determined; that
    height becomes the top row's share of ``fig_height`` via ``height_ratios``, and the speed
    panel takes what's left. Both axes share ``left_margin``/``right_margin`` so their plotted
    x-range spans identical figure-fraction width -- necessary for x-alignment, since the
    schematic has no y-axis gutter of its own but the speed panel needs one for its ticks and
    label.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``BehaviorSpeedEnvConfig`` results, passed through to
        :func:`mouse_speed_by_environment`.
    fig_width, fig_height : float
        Figure size in inches. Both panels are forced to fit inside it.
    left_margin, right_margin : float
        Shared horizontal extent of both axes, as a fraction of ``fig_width``. Sized to leave
        room for the speed panel's y-axis label and ticks; the schematic donates the same
        margin on either side even though it doesn't need it, so the two axes' x=0 and
        x=``REFERENCE_ENV_LENGTH_CM`` line up.
    top_margin, bottom_margin : float
        Figure-fraction space left above the schematic and below the speed panel's x-axis
        label and ticks.
    panel_gap : float
        Figure-fraction vertical gap between the two panels.
    schematic_kwargs : dict or None
        Keyword arguments forwarded to :func:`vr_environment_schematic` (e.g. ``envs``,
        ``panel_aspect``, ``room_gap``, ``margin``, ``show_legend``). ``panel_height_in`` and
        ``return_syd_viewer`` are set internally and should not be passed.
    speed_kwargs : dict or None
        Keyword arguments forwarded to :func:`mouse_speed_by_environment` (e.g.
        ``environments``, ``show_first``, ``reward_style``, ``fontsize``). ``figsize`` and
        ``return_syd_viewer`` are set internally and should not be passed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    schematic_kwargs = dict(schematic_kwargs or {})
    speed_kwargs = dict(speed_kwargs or {})
    schematic_kwargs.pop("return_syd_viewer", None)
    speed_kwargs.pop("return_syd_viewer", None)
    schematic_kwargs.pop("panel_height_in", None)
    speed_kwargs.pop("figsize", None)

    schem_viewer = vr_environment_schematic(**schematic_kwargs, return_syd_viewer=True)
    speed_viewer = mouse_speed_by_environment(results, **speed_kwargs, return_syd_viewer=True)
    schem_state = schem_viewer.state
    speed_state = speed_viewer.state

    metrics = schem_viewer.layout_metrics(schem_state)
    axes_width_frac = right_margin - left_margin
    # panel_height_in that would make the schematic's own width equal to the shared axes width.
    panel_height_in = (axes_width_frac * fig_width) / metrics["width"]
    schem_height_in = metrics["height"] * panel_height_in
    schem_height_frac = schem_height_in / fig_height

    avail_frac = 1.0 - top_margin - bottom_margin - panel_gap
    if schem_height_frac >= avail_frac:
        raise ValueError(
            f"Schematic needs {schem_height_in:.2f} in of height ({schem_height_frac:.1%} of "
            f"fig_height={fig_height}), leaving no room for the speed panel ({avail_frac:.1%} "
            "available). Increase fig_height or fig_width, or shrink the schematic (e.g. a "
            "smaller panel_aspect or margin)."
        )
    speed_height_frac = avail_frac - schem_height_frac

    plt.close("all")
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_schem = fig.add_axes([left_margin, 1.0 - top_margin - schem_height_frac, axes_width_frac, schem_height_frac])
    ax_speed = fig.add_axes([left_margin, bottom_margin, axes_width_frac, speed_height_frac])

    schem_viewer.draw(schem_state, ax_schem)
    speed_viewer.draw(speed_state, ax_speed)

    # The schematic's track (arrow + reward zones) spans data units [0, track_w] for a physical
    # length of REFERENCE_ENV_LENGTH_CM; converting its margin to the same cm scale and applying
    # it to the speed panel's xlim keeps the two axes' x=0 and x=REFERENCE_ENV_LENGTH_CM aligned
    # regardless of margin, since both axes occupy the same figure-fraction width.
    cm_per_unit = REFERENCE_ENV_LENGTH_CM / metrics["track_w"]
    margin_cm = schem_state["margin"] * cm_per_unit
    ax_speed.set_xlim(-margin_cm, REFERENCE_ENV_LENGTH_CM + margin_cm)

    return fig


# ---------------------------------------------------------------------- experiment timeline --


def _boolean_runs(present: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs of True in a 1-D boolean array."""
    padded = np.concatenate([[False], present, [False]]).astype(int)
    change = np.diff(padded)
    return list(zip(np.where(change == 1)[0], np.where(change == -1)[0]))


class ExperimentTimeline(Viewer):
    """One mouse's experiment as a session timeline: when each environment entered the protocol.

    A phase bar on top splits the timeline into training (not in the dataset -- its length is
    the ``num_training`` knob) and imaging (one slot per session in the data). Below it, one bar
    per environment spanning the sessions in which that environment was actually run, ordered and
    colored by experience order (:data:`ENV_SLOT_COLORS`, the same palette as
    :class:`VREnvironmentSchematic` and the ``by_env`` panels of figure 3). An arrow at the bottom
    carries the session-axis label.

    The layout is fully relative: the vertical stack is built in arbitrary units and normalized to
    fill the figure height, and the horizontal axis is the session count normalized to [0, 1]. So
    the panel keeps its proportions at any ``fig_width``/``fig_height``, and every gap knob is a
    fraction of the whole rather than an absolute size. Font sizes are the exception -- they are in
    points, so scaling the figure up makes the text read relatively smaller.
    """

    def __init__(self, sessions: list[B2Session]):
        by_mouse: dict[str, list[B2Session]] = defaultdict(list)
        for session in sessions:
            by_mouse[session.mouse_name].append(session)
        # Chronological within a mouse; the incoming list order is not trusted.
        self.sessions_by_mouse = {mouse: sorted(ms, key=_session_sort_key) for mouse, ms in by_mouse.items()}
        self.mice = sorted(self.sessions_by_mouse)

        # --- data ---
        self.add_selection("mouse", value=self.mice[0], options=self.mice)
        self.add_integer("num_training", value=6, min=0, max=60)
        self.add_boolean("merge_gaps", value=True)
        self.add_boolean("train_in_first_env", value=True)
        self.add_boolean("label_by_env_id", value=False)

        # --- figure size in inches; everything else below is relative to it ---
        self.add_float("fig_width", value=3.6, min=0.5, max=20.0, step=0.01)
        self.add_float("fig_height", value=2.0, min=0.5, max=20.0, step=0.01)

        # --- vertical layout, in arbitrary units normalized to fill the figure height ---
        self.add_float("phase_label_height", value=0.55, min=0.0, max=3.0)
        self.add_float("phase_bar_height", value=0.3, min=0.02, max=3.0)
        self.add_float("phase_gap", value=0.22, min=0.0, max=3.0)
        self.add_float("env_bar_height", value=1.0, min=0.05, max=3.0)
        self.add_float("env_bar_gap", value=0.06, min=0.0, max=2.0)
        self.add_float("axis_gap", value=0.3, min=0.0, max=3.0)
        self.add_float("axis_label_height", value=0.6, min=0.05, max=3.0)
        self.add_float("margin_x", value=0.02, min=0.0, max=0.5)
        self.add_float("margin_y", value=0.02, min=0.0, max=0.5)

        # --- horizontal layout, in session units (1.0 = the width of one session) ---
        self.add_float("phase_split_gap", value=0.4, min=0.0, max=5.0)
        self.add_float("env_label_pad", value=0.25, min=0.0, max=5.0)

        # --- text ---
        self.add_float("fontsize", value=9.0, min=3.0, max=30.0)
        self.add_float("phase_label_scale", value=1.0, min=0.2, max=3.0)
        self.add_float("env_label_scale", value=1.0, min=0.2, max=3.0)
        self.add_float("axis_label_scale", value=1.0, min=0.2, max=3.0)

        # --- style ---
        self.add_text("training_color", value="0.65")
        self.add_text("imaging_color", value="0.25")
        self.add_text("env_label_color", value="w")
        self.add_boolean("show_phase_bar", value=True)
        self.add_boolean("show_phase_labels", value=True)
        self.add_boolean("show_env_labels", value=True)
        self.add_boolean("show_training_divider", value=False)
        self.add_boolean("show_session_ticks", value=False)
        self.add_text("axis_arrowstyle", value="-|>")

        self.on_change("mouse", self.recompute_timeline)
        self.recompute_timeline(self.state)

    def recompute_timeline(self, state):
        """Session count, environment experience order, and per-session presence for one mouse."""
        mouse_sessions = self.sessions_by_mouse[state["mouse"]]
        self.num_imaging = len(mouse_sessions)

        # Experience order: first appearance walking the sessions chronologically. Same rule as
        # ``env_order.build_env_order``, but kept per-session here because the bars need to know
        # *which* sessions used each environment, not just the order.
        self.env_order: list[int] = []
        for session in mouse_sessions:
            for env in session.environments:
                env = int(env)
                if env < 0:  # negative environmentIndex is the invalid/unlabeled sentinel
                    continue
                if env not in self.env_order:
                    self.env_order.append(env)

        self.env_present = np.zeros((len(self.env_order), self.num_imaging), dtype=bool)
        for isession, session in enumerate(mouse_sessions):
            for env in session.environments:
                env = int(env)
                if env >= 0:
                    self.env_present[self.env_order.index(env), isession] = True

        self.imaging_days = len(mouse_sessions)

    def _env_color(self, slot: int):
        """Experience-slot color, shared with the VR schematic and the ``by_env`` panels."""
        return ENV_SLOT_COLORS[slot % len(ENV_SLOT_COLORS)]

    def plot(self, state):
        fontsize = state["fontsize"]
        # Set before any artist is created so every default-sized text picks it up.
        plt.rcParams["font.size"] = fontsize

        num_training = state["num_training"]
        num_envs = len(self.env_order)
        num_sessions = num_training + self.num_imaging
        show_phase = state["show_phase_bar"] or state["show_phase_labels"]

        def x(session_units: float) -> float:
            """Session units -> [0, 1] across the panel width."""
            return session_units / num_sessions

        # -- vertical stack: arbitrary units, top to bottom, normalized to fill the height --
        bands: list[tuple[str, float]] = []
        if state["show_phase_labels"]:
            bands.append(("phase_label", state["phase_label_height"]))
        if state["show_phase_bar"]:
            bands.append(("phase_bar", state["phase_bar_height"]))
        if show_phase:
            bands.append(("phase_gap", state["phase_gap"]))
        for slot in range(num_envs):
            bands.append((f"env{slot}", state["env_bar_height"]))
            if slot < num_envs - 1:
                bands.append((f"env_gap{slot}", state["env_bar_gap"]))
        bands.append(("axis_gap", state["axis_gap"]))
        bands.append(("axis_label", state["axis_label_height"]))

        total_units = sum(height for _, height in bands)
        edges: dict[str, tuple[float, float]] = {}
        cumulative = 0.0
        for name, height in bands:
            edges[name] = (1.0 - cumulative / total_units, 1.0 - (cumulative + height) / total_units)
            cumulative += height

        plt.close("all")
        fig = plt.figure(figsize=(state["fig_width"], state["fig_height"]))
        # One bare axes filling the figure: data coordinates are the relative layout itself.
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(-state["margin_x"], 1.0 + state["margin_x"])
        ax.set_ylim(edges["axis_label"][1] - state["margin_y"], 1.0 + state["margin_y"])
        ax.set_axis_off()

        # ------------------------------------------------------------------- phase bar --
        # Training and imaging segments, separated by a gap so the split reads without a border.
        half_gap = state["phase_split_gap"] / 2
        segments = []
        if num_training > 0:
            segments.append(("training", 0.0, num_training - half_gap, state["training_color"]))
            segments.append(("imaging", num_training + half_gap, num_sessions, state["imaging_color"]))
        else:
            segments.append(("imaging", 0.0, num_sessions, state["imaging_color"]))

        if state["show_phase_bar"]:
            top, bottom = edges["phase_bar"]
            for _, start, stop, color in segments:
                ax.add_patch(Rectangle((x(start), bottom), x(stop) - x(start), top - bottom, facecolor=color, edgecolor="none", zorder=3))

        if state["show_phase_labels"]:
            top, bottom = edges["phase_label"]
            for label, start, stop, _ in segments:
                ax.text(
                    x((start + stop) / 2),
                    (top + bottom) / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fontsize * state["phase_label_scale"],
                    zorder=4,
                )

        # ------------------------------------------------------------------- env bars --
        for slot, env in enumerate(self.env_order):
            top, bottom = edges[f"env{slot}"]
            color = self._env_color(slot)
            present = self.env_present[slot]
            if state["merge_gaps"]:
                # First appearance -> last appearance, ignoring any dropout in between; the
                # un-merged form draws one rectangle per contiguous run instead.
                runs = [(int(present.argmax()), int(self.num_imaging - present[::-1].argmax()))]
            else:
                runs = _boolean_runs(present)

            for start, stop in runs:
                left = num_training + start
                # The mouse is trained in the environment it sees first, so slot 0's bar covers
                # the training block too.
                if slot == 0 and start == 0 and state["train_in_first_env"]:
                    left = 0.0
                right = num_training + stop
                ax.add_patch(Rectangle((x(left), bottom), x(right) - x(left), top - bottom, facecolor=color, edgecolor="none", zorder=3))

            if state["show_env_labels"]:
                label = f"Env {env}" if state["label_by_env_id"] else f"{_ordinal(slot + 1)} Env"
                ax.text(
                    x(num_sessions - state["env_label_pad"]),
                    (top + bottom) / 2,
                    label,
                    ha="right",
                    va="center",
                    color=state["env_label_color"],
                    fontsize=fontsize * state["env_label_scale"],
                    zorder=5,
                )

        if state["show_training_divider"] and num_training > 0:
            ax.plot(
                [x(num_training)] * 2,
                [edges["env0"][0], edges[f"env{num_envs - 1}"][1]],
                color="w",
                linestyle=":",
                linewidth=1.0,
                zorder=6,
            )

        # ----------------------------------------------------------------- session axis --
        top, bottom = edges["axis_label"]
        ax.annotate(
            "",
            xy=(1.0, top),
            xytext=(0.0, top),
            arrowprops=dict(
                arrowstyle=state["axis_arrowstyle"],
                color="k",
                linewidth=1.0,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=fontsize * 1.2,
            ),
            zorder=4,
        )

        if state["show_session_ticks"]:
            # One tick per imaging session, at the center of its slot.
            tick_height = (top - bottom) * 0.25
            for isession in range(self.num_imaging):
                xt = x(num_training + isession + 0.5)
                ax.plot([xt, xt], [top, top + tick_height], color="k", linewidth=0.8, zorder=4)

        xlabel = f"{self.imaging_days} imaging sessions"
        ax.text(0.5, (top + bottom) / 2, xlabel, ha="center", va="center", fontsize=fontsize * state["axis_label_scale"], zorder=4)

        return fig


def experiment_timeline(
    sessions: list[B2Session],
    mouse: str | None = None,
    num_training: int = 6,
    merge_gaps: bool = True,
    train_in_first_env: bool = True,
    label_by_env_id: bool = False,
    figsize: tuple[float, float] = (3.6, 2.0),
    phase_label_height: float = 0.55,
    phase_bar_height: float = 0.3,
    phase_gap: float = 0.22,
    env_bar_height: float = 1.0,
    env_bar_gap: float = 0.06,
    axis_gap: float = 0.3,
    axis_label_height: float = 0.6,
    margin_x: float = 0.02,
    margin_y: float = 0.02,
    phase_split_gap: float = 0.4,
    env_label_pad: float = 0.25,
    fontsize: float = 9.0,
    phase_label_scale: float = 1.0,
    env_label_scale: float = 1.0,
    axis_label_scale: float = 1.0,
    training_color: str = "0.65",
    imaging_color: str = "0.25",
    env_label_color: str = "w",
    show_phase_bar: bool = True,
    show_phase_labels: bool = True,
    show_env_labels: bool = True,
    show_training_divider: bool = False,
    show_session_ticks: bool = False,
    axis_arrowstyle: str = "-|>",
    return_syd_viewer: bool = False,
):
    """
    One mouse's experiment as a session timeline, built from the real session list.

    A training/imaging phase bar sits on top, one bar per environment below it, and a session-axis
    arrow at the bottom. Environments are ordered and colored by *experience order* -- the order in
    which that mouse first saw them -- using :data:`ENV_SLOT_COLORS`, so slot 1 is black, slot 2
    blue, slot 3 green, matching the VR schematic and figure 3. Environment identities do not carry
    meaning across cohorts (the CR mice run environments 1 and 2, the ATL mice 1, 3 and 4), which is
    why the labels use experience order (``"1st Env"``, ``"2nd Env"``, ...) unless
    ``label_by_env_id`` is set.

    Everything except the training block comes from the data: the number of imaging sessions, which
    environments each session ran, and the imaging date span quoted in the axis label. Training
    sessions predate the imaging dataset, so their count is a knob.

    Parameters
    ----------
    sessions : list of B2Session
        The full session list; grouped by mouse and sorted chronologically internally.
    mouse : str or None
        Mouse to draw. None takes the first alphabetically.
    num_training : int
        Number of training sessions to draw ahead of the imaging sessions. These are not in the
        dataset, so this only sets how much of the timeline the training block occupies.
    merge_gaps : bool
        Draw each environment's bar from its first to its last session, ignoring sessions in
        between where it was not run. False draws one rectangle per contiguous run instead, which
        exposes the dropouts (e.g. CR_Hippocannula6 skips environment 1 mid-way).
    train_in_first_env : bool
        Extend the first environment's bar back through the training block -- the mouse is trained
        in the environment it later sees first.
    label_by_env_id : bool
        Label bars with the raw environment index instead of ordinal experience labels such as
        ``"1st Env"`` and ``"2nd Env"``.
    figsize : tuple of float
        Figure size in inches. The layout is relative, so the panel keeps its proportions at any
        size; only the fonts (in points) do not scale with it.
    phase_label_height, phase_bar_height, phase_gap : float
        Heights of the phase-label row, the phase bar, and the gap below them.
    env_bar_height, env_bar_gap : float
        Height of one environment bar and the gap between consecutive bars.
    axis_gap, axis_label_height : float
        Gap above the session-axis arrow, and the height of the band holding the arrow and its
        label. All of the above are in arbitrary units, normalized so the stack fills the figure --
        only their ratios matter.
    margin_x, margin_y : float
        Padding around the layout as a fraction of the panel width/height.
    phase_split_gap : float
        Gap between the training and imaging segments of the phase bar, in session units.
    env_label_pad : float
        Inset of the environment label from the right edge, in session units.
    fontsize : float
        Base font size in points.
    phase_label_scale, env_label_scale, axis_label_scale : float
        Multipliers on ``fontsize`` for the three text groups.
    xlabel : str
        Session-axis label.
    show_session_span : bool
        Append the mouse's real imaging date span to the axis label.
    training_color, imaging_color : str
        Colors of the two phase-bar segments. Grey by default so they do not compete with the
        black/blue/green environment palette.
    env_label_color : str
        Color of the labels drawn inside the environment bars.
    show_phase_bar, show_phase_labels, show_env_labels : bool
        Toggles for the phase bar, its "training"/"imaging" labels, and the in-bar env labels.
    show_training_divider : bool
        Draw a dotted line at the training/imaging boundary across the environment bars.
    show_session_ticks : bool
        Mark each imaging session with a tick above the axis arrow.
    axis_arrowstyle : str
        Matplotlib arrowstyle for the session axis; ``"<|-|>"`` heads both ends.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.

    Returns
    -------
    matplotlib.figure.Figure or ExperimentTimeline
    """
    viewer = ExperimentTimeline(sessions)

    if mouse is not None:
        viewer.update_selection("mouse", value=mouse)
        # Seeding via update_* does not fire on_change before deployment.
        viewer.recompute_timeline(viewer.state)

    viewer.update_integer("num_training", value=num_training)
    viewer.update_boolean("merge_gaps", value=merge_gaps)
    viewer.update_boolean("train_in_first_env", value=train_in_first_env)
    viewer.update_boolean("label_by_env_id", value=label_by_env_id)

    viewer.update_float("fig_width", value=figsize[0])
    viewer.update_float("fig_height", value=figsize[1])

    viewer.update_float("phase_label_height", value=phase_label_height)
    viewer.update_float("phase_bar_height", value=phase_bar_height)
    viewer.update_float("phase_gap", value=phase_gap)
    viewer.update_float("env_bar_height", value=env_bar_height)
    viewer.update_float("env_bar_gap", value=env_bar_gap)
    viewer.update_float("axis_gap", value=axis_gap)
    viewer.update_float("axis_label_height", value=axis_label_height)
    viewer.update_float("margin_x", value=margin_x)
    viewer.update_float("margin_y", value=margin_y)

    viewer.update_float("phase_split_gap", value=phase_split_gap)
    viewer.update_float("env_label_pad", value=env_label_pad)

    viewer.update_float("fontsize", value=fontsize)
    viewer.update_float("phase_label_scale", value=phase_label_scale)
    viewer.update_float("env_label_scale", value=env_label_scale)
    viewer.update_float("axis_label_scale", value=axis_label_scale)

    viewer.update_text("training_color", value=training_color)
    viewer.update_text("imaging_color", value=imaging_color)
    viewer.update_text("env_label_color", value=env_label_color)
    viewer.update_boolean("show_phase_bar", value=show_phase_bar)
    viewer.update_boolean("show_phase_labels", value=show_phase_labels)
    viewer.update_boolean("show_env_labels", value=show_env_labels)
    viewer.update_boolean("show_training_divider", value=show_training_divider)
    viewer.update_boolean("show_session_ticks", value=show_session_ticks)
    viewer.update_text("axis_arrowstyle", value=axis_arrowstyle)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# -------------------------------- cross-validated place fields across sessions --


class CrossValidatedPlacefields(Viewer):
    """Plot precomputed odd-trial place fields, sorted independently on even trials."""

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (12.0, 3.0)):
        self.results = results
        self.figsize = figsize
        keys = ["even_placefield", "odd_placefield", "reliability", "fraction_active", "env_slot_ids", "spks_std"]
        out = results.sel(keys=keys, squeeze_ones=False)
        missing = [key for key in keys if key not in out]
        if missing:
            raise KeyError(f"Aggregator is missing {missing} -- run CrossValidatedPlacefieldsConfig first.")
        self.even_placefield = np.asarray(out["even_placefield"], dtype=float)
        self.odd_placefield = np.asarray(out["odd_placefield"], dtype=float)
        self.reliability = np.asarray(out["reliability"], dtype=float)
        self.fraction_active = np.asarray(out["fraction_active"], dtype=float)
        self.env_slot_ids = np.asarray(out["env_slot_ids"], dtype=float)
        self.spks_std = np.asarray(out["spks_std"], dtype=float)

        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(mouse) for mouse in mouse_names if np.any(np.isfinite(self.even_placefield[mouse_names == mouse]))})
        if not self.mice:
            raise ValueError("The aggregator contains no computed cross-validated place fields.")
        self._rows_by_mouse = {
            mouse: np.array(sorted(np.flatnonzero(mouse_names == mouse), key=lambda row: _session_sort_key(results.sessions[row])))
            for mouse in self.mice
        }

        self.add_selection("mouse", value=self.mice[0], options=self.mice)
        self.add_selection("environment", value=0, options=[0])
        self.add_integer_range("session_range", value=(1, 1), min=1, max=1)
        self.add_integer("skip_session", value=0, min=0, max=20)
        self.add_float("reliability_threshold", value=0.7, min=-1.0, max=1.0)
        self.add_float("fraction_active_threshold", value=0.2, min=0.0, max=1.0)
        self.add_float("vmax", value=5.0, min=0.1, max=20.0)
        self.add_float("fontsize", value=9.0, min=1.0, max=30.0)
        self.add_boolean("verbose", value=True)
        self.add_float("scalebar_x", value=0.05, min=0.0, max=1.0)
        self.add_float("scalebar_y", value=0.08, min=0.0, max=1.0)
        self.add_float("scalebar_cm", value=50.0, min=0.1, max=REFERENCE_ENV_LENGTH_CM)
        self.add_float("scalebar_fontsize", value=9.0, min=1.0, max=30.0)
        self.add_float("colorbar_width_ratio", value=0.12, min=0.01, max=1.0)

        self.on_change("mouse", self.update_mouse)
        self.on_change("environment", self.update_session_range)
        self.update_mouse(self.state)

    def _slot_for_row(self, row: int, environment: int) -> int | None:
        slots = np.flatnonzero(self.env_slot_ids[row] == environment)
        return int(slots[0]) if slots.size else None

    def _row_has_environment(self, row: int, environment: int) -> bool:
        slot = self._slot_for_row(row, environment)
        return slot is not None and np.any(np.isfinite(self.even_placefield[row, slot]))

    def update_mouse(self, state):
        """Populate the environment selector, then constrain the session-number range."""
        mouse = state["mouse"]
        rows = self._rows_by_mouse[mouse]
        candidate_environments = {int(env) for env in self.env_slot_ids[rows].ravel() if np.isfinite(env) and env >= 0}
        environments = sorted(env for env in candidate_environments if any(self._row_has_environment(int(row), env) for row in rows))
        if not environments:
            raise ValueError(f"No valid environments found for {mouse}")
        value = state["environment"] if state["environment"] in environments else environments[0]
        self.update_selection("environment", value=value, options=environments)
        self.update_session_range(self.state)

    def update_session_range(self, state):
        """Move the integer-range bounds to the first/last session containing the environment."""
        rows = self._rows_by_mouse[state["mouse"]]
        session_numbers = [i + 1 for i, row in enumerate(rows) if self._row_has_environment(int(row), state["environment"])]
        if not session_numbers:
            return
        bounds = (session_numbers[0], session_numbers[-1])
        self.update_integer_range("session_range", value=bounds, min=bounds[0], max=bounds[1])
        self.update_integer("skip_session", max=max(0, len(session_numbers) - 1))

    def plot(self, state):
        start, stop = state["session_range"]
        env = state["environment"]
        rows = self._rows_by_mouse[state["mouse"]]
        available = [
            (session_number, int(rows[session_number - 1]))
            for session_number in range(start, stop + 1)
            if self._row_has_environment(int(rows[session_number - 1]), env)
        ]
        shown = available[:: state["skip_session"] + 1]
        if not shown:
            raise ValueError("The selected range contains no sessions with this environment")

        plt.close("all")
        width_ratios = [state["colorbar_width_ratio"]] + [1.0] * len(shown)
        fig, axes = plt.subplots(
            1,
            len(width_ratios),
            figsize=self.figsize,
            layout="constrained",
            squeeze=False,
            width_ratios=width_ratios,
        )
        color_ax = axes[0, 0]
        placefield_axes = axes[0, 1:]

        color_data = mpl.colormaps["gray_r"](np.linspace(0, 1, 255))[:, None, :]
        color_ax.imshow(color_data, aspect="auto", origin="lower")
        color_ax.set_xticks([])
        color_ax.set_yticks([])
        color_ax.text(0.5, 0.02, "0", transform=color_ax.transAxes, ha="center", va="bottom", fontsize=state["fontsize"])
        color_ax.text(
            0.5,
            0.98,
            f"{state['vmax']:g}",
            transform=color_ax.transAxes,
            ha="center",
            va="top",
            color="white",
            fontsize=state["fontsize"],
        )
        color_ax.text(
            0.5,
            0.5,
            r"Activity ($\sigma$)",
            transform=color_ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=state["fontsize"],
        )
        for spine in color_ax.spines.values():
            spine.set_visible(False)

        for ax, (session_number, row) in zip(placefield_axes, shown):
            slot = self._slot_for_row(row, env)
            even = self.even_placefield[row, slot]
            odd = self.odd_placefield[row, slot]
            scale = self.spks_std[row]
            valid_scale = np.isfinite(scale) & (scale > 0)
            reliability = self.reliability[row, slot]
            fraction_active = self.fraction_active[row, slot]
            idx_keep = (
                np.isfinite(reliability)
                & np.isfinite(fraction_active)
                & valid_scale
                & (reliability > state["reliability_threshold"])
                & (fraction_active > state["fraction_active_threshold"])
                & np.any(np.isfinite(even), axis=1)
                & np.any(np.isfinite(odd), axis=1)
            )
            even = even[idx_keep] / scale[idx_keep, None]
            odd = odd[idx_keep] / scale[idx_keep, None]
            if even.shape[0]:
                order = np.argsort(np.nanargmax(even, axis=1), kind="stable")
                ax.imshow(odd[order], aspect="auto", interpolation="none", cmap="gray_r", vmin=0, vmax=state["vmax"])
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No ROIs pass\nthresholds",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=state["fontsize"],
                )
            if state["verbose"]:
                session = self.results.sessions[row]
                ax.set_title(f"Session {session_number}\n{session.date}", fontsize=state["fontsize"])
            ax.set_xlabel(f"{session_number}", fontsize=state["fontsize"])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis="both", labelsize=state["fontsize"])
            for spine in ax.spines.values():
                spine.set_visible(False)
        placefield_axes[0].set_ylabel(
            "ROIs (even-trial PF order)" if state["verbose"] else "ROIs",
            fontsize=state["fontsize"],
        )

        # The map spans the reference 200-cm track, so convert the requested distance to
        # axes-fraction units. Coordinates are axes fractions to keep placement stable as the
        # number of displayed sessions changes.
        first_ax = placefield_axes[0]
        x0 = state["scalebar_x"]
        y0 = state["scalebar_y"]
        bar_width = state["scalebar_cm"] / REFERENCE_ENV_LENGTH_CM
        first_ax.plot(
            [x0, x0 + bar_width],
            [y0, y0],
            transform=first_ax.transAxes,
            color="black",
            linewidth=1.5,
            solid_capstyle="butt",
            clip_on=False,
        )
        first_ax.text(
            x0 + bar_width / 2,
            y0,
            f"{state['scalebar_cm']:g}cm",
            transform=first_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=state["scalebar_fontsize"],
        )
        fig.supxlabel("Session #", fontsize=state["fontsize"])
        return fig


def cross_validated_placefields(
    results: ResultsAggregator,
    mouse: str | None = None,
    environment: int | None = None,
    session_range: tuple[int, int] | None = None,
    skip_session: int = 0,
    reliability_threshold: float = 0.7,
    fraction_active_threshold: float = 0.2,
    vmax: float = 5.0,
    fontsize: float = 9.0,
    verbose: bool = True,
    scalebar_xy: tuple[float, float] = (0.05, 0.08),
    scalebar_cm: float = 50.0,
    scalebar_fontsize: float = 9.0,
    colorbar_width_ratio: float = 0.12,
    figsize: tuple[float, float] = (12.0, 3.0),
    return_syd_viewer: bool = False,
):
    """Plot precomputed odd-trial place fields, filtered and sorted without rebuilding maps.

    ``verbose=False`` removes the per-session titles and shortens the first raster's y label to
    ``"ROIs"``. ``scalebar_xy`` is the scale bar's lower-left position in axes fractions;
    ``scalebar_cm`` is converted to a fraction of the reference track length. The vertical
    activity color strip occupies its own leading axes, sized by ``colorbar_width_ratio``.
    """
    viewer = CrossValidatedPlacefields(results, figsize=figsize)
    if mouse is not None:
        viewer.update_selection("mouse", value=mouse)
        viewer.update_mouse(viewer.state)
    if environment is not None:
        viewer.update_selection("environment", value=environment)
        viewer.update_session_range(viewer.state)
    if session_range is not None:
        viewer.update_integer_range("session_range", value=tuple(session_range))
    viewer.update_integer("skip_session", value=skip_session)
    viewer.update_float("reliability_threshold", value=reliability_threshold)
    viewer.update_float("fraction_active_threshold", value=fraction_active_threshold)
    viewer.update_float("vmax", value=vmax)
    viewer.update_float("fontsize", value=fontsize)
    viewer.update_boolean("verbose", value=verbose)
    viewer.update_float("scalebar_x", value=scalebar_xy[0])
    viewer.update_float("scalebar_y", value=scalebar_xy[1])
    viewer.update_float("scalebar_cm", value=scalebar_cm)
    viewer.update_float("scalebar_fontsize", value=scalebar_fontsize)
    viewer.update_float("colorbar_width_ratio", value=colorbar_width_ratio)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ------------------------------------------ place field peak amplitude across sessions --


def _render_curve_group(
    ax,
    xvals: np.ndarray,
    data: np.ndarray,
    color,
    plot_style: str,
    label: str | None = None,
    hide_error: bool = False,
    min_support: int = 1,
    line_alpha: float = 0.3,
    linewidth: float = 2.0,
) -> None:
    """Render one group of ``(n_series, n_x)`` curves as individual+mean lines or an errorPlot band.

    Mirrors the helper of the same name in ``figure3``, so the ``each`` / ``errorPlot`` styles mean
    the same thing in both figures. ``"each"`` draws every row thin plus a solid mean;
    ``"errorPlot"`` draws mean +/- std as a band, or just the mean when ``hide_error``. Columns
    with fewer than ``min_support`` finite rows are dropped from the mean/band (not from the thin
    lines).
    """
    valid_count = np.sum(np.isfinite(data), axis=0)
    value_sum = np.nansum(data, axis=0)
    mean_curve = np.divide(value_sum, valid_count, out=np.full_like(value_sum, np.nan, dtype=float), where=valid_count > 0)
    mean_curve[valid_count < min_support] = np.nan

    if plot_style == "each":
        ax.plot(xvals, data.T, color=(color, line_alpha), linewidth=0.5)
        ax.plot(xvals, mean_curve, color=color, linewidth=linewidth, label=label)
    elif plot_style == "errorPlot":
        if hide_error:
            ax.plot(xvals, mean_curve, color=color, linewidth=linewidth, label=label)
        else:
            masked = np.where(valid_count[None, :] >= min_support, data, np.nan)
            errorPlot(xvals, masked, axis=0, ax=ax, color=color, linewidth=linewidth, alpha=0.2, label=label)
    else:
        raise ValueError(f"Unknown plot_style {plot_style!r}. Options: ['each', 'errorPlot']")


def _session_colorbar_inset(axis, first_label, last_label, cmap_name: str, fontsize: float, bounds: list[float]) -> None:
    """Draw the small session-number colorbar inset on ``axis``.

    Mirrors the helper of the same name in ``figure3``, with the colormap left to the caller so it
    tracks the figure's ``cmap`` knob. ``bounds`` is ``[x, y, width, height]`` in axes-fraction
    coordinates; values outside ``[0, 1]`` place the bar outside the axes.
    """
    cmap_data = mpl.colormaps[cmap_name](np.linspace(0, 1, 255))[np.newaxis, :, :]
    inset = axis.inset_axes(bounds)
    inset.imshow(cmap_data, aspect="auto")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.text(0.02, 0.5, f"{first_label}", transform=inset.transAxes, ha="left", va="center", color="white", fontsize=fontsize)
    inset.text(0.98, 0.5, f"{last_label}", transform=inset.transAxes, ha="right", va="center", color="white", fontsize=fontsize)
    inset.text(0.5, 0.5, "session #", transform=inset.transAxes, ha="center", va="center", color="black", fontsize=fontsize)


def _pad_stack(curves: list[np.ndarray]) -> np.ndarray:
    """Stack ragged per-mouse 1-D curves into a NaN-padded ``(n_mice, max_len)`` array."""
    max_len = max((len(values) for values in curves), default=0)
    stack = np.full((len(curves), max_len), np.nan)
    for i, values in enumerate(curves):
        stack[i, : len(values)] = values
    return stack


def _support_length(pad_stack: np.ndarray, min_support: int = 1) -> int:
    """Number of leading columns where more than ``min_support`` mice have finite data."""
    support = np.sum(np.isfinite(pad_stack), axis=0)
    valid = np.where(support > min_support)[0]
    return int(valid[-1] + 1) if valid.size else 0


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


def _ordinal(value: float) -> str:
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


#: Grid resolution of an ECDF curve. The ECDF is a step function, so this only has to be fine
#: enough that the steps are not visible at figure scale.
_ECDF_POINTS = 201

#: How close two of :class:`R2Familiarity`'s y ranges have to be, as a fraction of the larger, for
#: the panels to be put on one shared axis instead of two nearly-identical ones.
_SHARE_Y_TOLERANCE = 0.15


class R2Familiarity(Viewer):
    """R² as a function of familiarity: one ROI, one mouse, the population, the slots.

    ``panel_mode`` picks what ``ax[1]``/``ax[2]`` draw, and the four options answer different
    questions. ``"r2_rel"`` and ``"r2_rel_prct"`` show R² *conditioned* on spatial reliability, so
    they ask whether a cell of given quality is better predicted with experience; ``"histogram"``
    and ``"ecdf"`` show the marginal distribution of R², which folds in any drift of the
    reliability distribution itself.

    - ``"r2_rel"``: the running average E[R² | reliability] -- the same kernel regression drawn on
      ``ax[1]`` of :func:`example_r2_placefield`, but resolved per session rather than pooled, read
      straight from the aggregator's ``r2_kde_slot`` / ``r2_kde_pooled``.
    - ``"r2_rel_prct"``: the same regression against reliability *rank* within the session. The
      reliability distribution is heavily skewed, so uniform bins of reliability are wildly
      unequally populated and the interesting range is squeezed into a sliver of the axis; ranks
      put the same number of ROIs behind every x.
    - ``"histogram"``: the fraction of ROIs per R² bin, with out-of-range R² clipped into the edge
      bins rather than dropped. Fractions, not counts, since sessions differ in ROI count.
    - ``"ecdf"``: the same distribution as a cumulative fraction -- no bin-edge choice, and curves
      of many sessions overlay far more legibly than histograms do.

    ``ax[1]`` onwards is read from the ``PFPredQualityConfig`` aggregator (``r2_kde_slot``,
    ``r2_kde_pooled``, ``r2_slot``, ``reliability_slot``), so nothing there is measured here.

    - ``ax[0]``: one ROI's activity against its place-field prediction -- the example panel of
      :func:`example_r2_placefield`, drawn for a session of whichever mouse the rest of the figure
      is about. Unlike everything to its right this panel is not split by environment: it uses
      every predictable frame of the session. The prediction and per-neuron standard deviation
      are loaded from ``PlacefieldPredictionConfig`` rather than rebuilt here.
    - ``ax[1]``: every session of that mouse, one curve per session colored by session number.
    - ``ax[2]``: the same curves for the whole population -- mouse-averaged overall
      (``curve_mode="average"``) or grouped by session number (``"by_session"``). Shares ``ax[1]``'s
      y-axis, and renders with the ``each`` / ``errorPlot`` styles of ``figure3``.
    - ``ax[3]``: R² collapsed to one number per session (``summary_stat`` over ROIs), against how
      many sessions of that environment the mouse has had, one curve per experience slot. It joins
      the shared y-axis too when its range comes out close enough to the curve panels'.

    Environments are indexed by experience-order slot, so slot 0 is the first environment that
    mouse ever saw; ``env_slot_ids`` carries the underlying environment index, which differs
    between the two cohorts.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        mouse: str | None = None,
        fontsize: float = 8.0,
        figsize: tuple[float, float] = (8.0, 2.0),
    ):
        self.results = results
        self.fontsize = fontsize
        self.figsize = figsize

        keys = ["r2_kde_grid", "r2_kde_slot", "r2_kde_pooled", "r2_slot", "reliability_slot", "best_env_slot", "env_slot_ids"]
        out = results.sel(keys=keys, squeeze_ones=False)
        # The per-slot R² keys arrived in schema v5; against an older store they are simply absent,
        # which would otherwise surface as a bare KeyError on an arbitrary one of them.
        missing = [key for key in keys if key not in out]
        if missing:
            raise KeyError(f"Aggregator is missing {missing} -- rerun the pfpred_quality sweep (these keys need schema v5).")
        num_sessions_total = len(results.sessions)
        self.r2_kde_slot = np.asarray(out["r2_kde_slot"], dtype=float)  # (sessions, slots, grid)
        self.r2_kde_pooled = np.asarray(out["r2_kde_pooled"], dtype=float)  # (sessions, grid)
        self.r2_slot = np.asarray(out["r2_slot"], dtype=float)  # (sessions, slots, max_rois)
        self.reliability_slot = np.asarray(out["reliability_slot"], dtype=float)  # (sessions, slots, max_rois)
        self.env_slot_ids = np.asarray(out["env_slot_ids"], dtype=float)  # (sessions, slots)

        # ax[0] needs one large frame x ROI prediction at a time. Index the store metadata now,
        # but defer unpickling until that session is actually selected; a companion aggregator
        # would pad and materialize this array over every session.
        self._example_config = PlacefieldPredictionConfig()
        example_rows = results.store.summary_table(
            analysis_type=self._example_config.display_name,
            session_ids=list(results.session_ids),
            schema_version=self._example_config.schema_version,
        )
        self._example_store_rows = {row["session_id"]: row for row in example_rows if row["analysis_key"] == self._example_config.key()}
        self._example_result_cache: dict[int, dict] = {}
        self._example_activity_cache: dict[int, np.ndarray] = {}
        self._example_roi_indices_cache: dict[int, np.ndarray] = {}
        # Stored as one number per session; how the aggregator pads a scalar key is not worth
        # depending on, so it is flattened back to one column.
        self.best_env_slot = np.asarray(out["best_env_slot"], dtype=float).reshape(num_sessions_total, -1)[:, 0]
        # Every session stores the same grid (it comes from the config), so take the first session
        # that actually has it.
        grid = np.asarray(out["r2_kde_grid"], dtype=float)
        idx_grid = np.flatnonzero(np.all(np.isfinite(grid), axis=1))
        if idx_grid.size == 0:
            raise ValueError("No session in the aggregator has r2_kde_grid -- rerun the pfpred_quality sweep.")
        self.kde_grid = grid[idx_grid[0]]
        self.num_slots = self.r2_kde_slot.shape[1]

        # Rows of every array above are aggregator sessions; a mouse's rows are ordered
        # chronologically here so "session number" means what it says on the color axis.
        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(mouse_name) for mouse_name in mouse_names})
        self._rows_by_mouse = {
            mouse_name: np.array(sorted(np.flatnonzero(mouse_names == mouse_name), key=lambda row: _session_sort_key(results.sessions[row])))
            for mouse_name in self.mice
        }

        # ax[0]'s session picker works in printable labels rather than aggregator rows, so the
        # sessions themselves are indexed the same way.
        self._sessions_by_mouse = {
            mouse_name: {
                results.sessions[row].session_print(): results.sessions[row]
                for row in rows
                if results.sessions[row].session_uid in self._example_store_rows
            }
            for mouse_name, rows in self._rows_by_mouse.items()
        }
        self._example_rows_by_mouse = {
            mouse_name: {
                results.sessions[row].session_print(): int(row) for row in rows if results.sessions[row].session_uid in self._example_store_rows
            }
            for mouse_name, rows in self._rows_by_mouse.items()
        }

        # --- what goes into a curve ---
        self.add_selection("mouse", value=mouse if mouse is not None else self.mice[0], options=self.mice)
        self.add_selection("env_mode", value="best", options=["pooled", "best", "slot"])
        self.add_integer("env_slot", value=0, min=0, max=MAX_ENV_SLOTS - 1)
        self.add_selection("panel_mode", value="r2_rel", options=["r2_rel", "r2_rel_prct", "histogram", "ecdf"])
        # Each panel mode has its own x quantity, so each carries its own range: reliability for
        # r2_rel, rank for r2_rel_prct (always the full 0-100), R² for the distribution modes.
        self.add_float_range("xrange", value=(-1.0, 1.0), min=-1.0, max=1.0, step=0.05)
        self.add_float_range("r2_range", value=(-1.0, 1.0), min=-2.0, max=2.0, step=0.05)
        self.add_integer("r2_bins", value=20, min=4, max=100)
        self.add_integer("prct_points", value=101, min=11, max=201)
        # Kernel width in percentile units: 5 means E[R² | rank] is smoothed over a 5-point-wide
        # slice of the session's ROIs.
        self.add_float("prct_bandwidth", value=5.0, min=0.5, max=25.0, step=0.5)
        self.add_boolean("auto_ylim", value=True)
        self.add_float_range("r2_ylim", value=(-0.05, 0.4), min=-1.0, max=1.0, step=0.05)

        # --- ax[0]: one ROI of one session of the selected mouse, data vs PF prediction ---
        # Options are filled in by update_example_session, which the mouse selector drives.
        self.add_selection("example_session", value=None, options=[None])
        self.add_integer("roi", value=0, min=0, max=0)
        self.add_selection("example_style", value="points", options=["points", "density"])
        self.add_float("example_alpha", value=0.1, min=0.01, max=1.0, step=0.01)

        # --- ax[2]: population curves (same knobs as figure3's cross-spectrum summary panel) ---
        self.add_selection("curve_mode", value="by_session", options=["average", "by_session"])
        self.add_selection("plot_style", value="each", options=["each", "errorPlot"])
        self.add_boolean("hide_error", value=False)
        self.add_integer("skip_sessions", value=0, min=0, max=max(num_sessions_total, 1))
        # ax[2] session-number colorbar inset placement, in ax[2] axes-fraction coordinates.
        self.add_float("inset_x", value=0.35, min=-1.0, max=1.0, step=0.01)
        self.add_float("inset_y", value=0.85, min=-1.0, max=1.0, step=0.01)
        self.add_float("inset_width", value=0.6, min=0.01, max=1.5, step=0.01)
        self.add_float("inset_height", value=0.075, min=0.01, max=1.0, step=0.005)

        # --- ax[3]: summary stat against experience, one curve per environment slot ---
        self.add_selection("summary_stat", value="mean", options=["mean", "median", "percentile"])
        # Half-percentile steps so the conventional tail levels (2.5, 97.5) are reachable.
        self.add_float("summary_percentile", value=90.0, min=0.0, max=100.0, step=0.5)
        self.add_selection("slot_style", value="errorPlot", options=["each", "errorPlot"])
        self.add_boolean("show_slot_legend", value=True)

        # --- style ---
        self.add_text("cmap", value="coolwarm")
        self.add_float("linewidth", value=1.0, min=0.25, max=4.0)
        self.add_float("alpha", value=0.9, min=0.05, max=1.0)
        # Square panels are all one width, so they cost the panel-specific width ratios; the
        # figure height stops being an input and becomes whatever squareness demands.
        self.add_boolean("square_panels", value=False)

        self.on_change("mouse", self.update_slot_bounds)
        self.on_change("mouse", self.update_example_session)
        self.on_change("example_session", self.update_roi_bounds)
        self.update_slot_bounds(self.state)
        self.update_example_session(self.state)

    def update_slot_bounds(self, state):
        """Limit the slot selector to the environments this mouse actually ran."""
        rows = self._rows_by_mouse[state["mouse"]]
        num_slots = int(np.sum(np.any(np.isfinite(self.r2_kde_slot[rows]), axis=(0, 2))))
        self.update_integer("env_slot", max=max(num_slots - 1, 0))

    def update_example_session(self, state):
        """Offer ax[0] the sessions of the selected mouse, keeping the current one if it survives."""
        labels = list(self._sessions_by_mouse[state["mouse"]])
        if not labels:
            raise ValueError(f"No PlacefieldPredictionConfig schema-v1 result is stored for {state['mouse']}.")
        current = state["example_session"] if state["example_session"] in labels else labels[0]
        self.update_selection("example_session", value=current, options=labels)
        # Seeding a selection does not always fire its callbacks, and the ROI count is a property
        # of the session that was just chosen, so the bound is refreshed here directly.
        self.update_roi_bounds({**state, "example_session": current})

    def update_roi_bounds(self, state):
        """Limit the ROI selector to the ROIs the example session actually has.

        The selector retains the old session-filtered ROI coordinates even though the stored
        prediction contains every ROI.
        """
        row = self._example_rows_by_mouse[state["mouse"]][state["example_session"]]
        num_rois = len(self._example_roi_indices(row))
        self.update_integer("roi", max=max(num_rois - 1, 0))

    def _example_roi_indices(self, row: int) -> np.ndarray:
        """Raw ROI indices corresponding to the example panel's filtered ROI coordinates."""
        if row not in self._example_roi_indices_cache:
            session = self.results.sessions[row]
            previous_spks_type = session.params.spks_type
            session.params.spks_type = self._example_config.spks_type
            try:
                self._example_roi_indices_cache[row] = np.flatnonzero(session.idx_rois)
            finally:
                session.params.spks_type = previous_spks_type
        return self._example_roi_indices_cache[row]

    def _example_roi_traces(self, row: int, roi: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw activity and stored all-trial PF prediction, normalized by the saved std."""
        session = self.results.sessions[row]
        raw_roi = self._example_roi_indices(row)[roi]
        if row not in self._example_result_cache:
            store_row = self._example_store_rows[session.session_uid]
            result = self._example_config.get_result(self.results.store, store_row)
            if result is None or "placefield_prediction" not in result or "spks_std" not in result:
                raise KeyError(
                    f"Stored cross_validated_placefields result for {session.session_print()} "
                    "does not contain placefield_prediction and spks_std (PlacefieldPredictionConfig schema v1 required)."
                )
            self._example_result_cache[row] = result
        if row not in self._example_activity_cache:
            previous_spks_type = session.params.spks_type
            session.params.spks_type = self._example_config.spks_type
            try:
                self._example_activity_cache[row] = np.asarray(session.spks)
            finally:
                session.params.spks_type = previous_spks_type

        result = self._example_result_cache[row]
        data = self._example_activity_cache[row][:, raw_roi]
        prediction = np.asarray(result["placefield_prediction"])[: data.shape[0], raw_roi]
        scale = np.asarray(result["spks_std"])[raw_roi]
        idx_keep = np.isfinite(data) & np.isfinite(prediction)
        if np.isfinite(scale) and scale > 0:
            data = data / scale
            prediction = prediction / scale
        return data[idx_keep], prediction[idx_keep]

    def _grid(self, state) -> np.ndarray:
        """The x values the current panel mode's curves are evaluated on."""
        panel_mode = state["panel_mode"]
        if panel_mode == "r2_rel":
            return self.kde_grid
        if panel_mode == "r2_rel_prct":
            return np.linspace(0.0, 100.0, state["prct_points"])
        low, high = tuple(state["r2_range"])
        if panel_mode == "histogram":
            return edge2center(np.linspace(low, high, state["r2_bins"] + 1))
        if panel_mode == "ecdf":
            return np.linspace(low, high, _ECDF_POINTS)
        raise ValueError(f"Invalid panel_mode: {panel_mode!r}")

    def _xlimits(self, state) -> tuple[float, float]:
        """Visible x range of ``ax[0]``/``ax[1]``, in the units the current panel mode plots."""
        panel_mode = state["panel_mode"]
        if panel_mode == "r2_rel":
            return tuple(state["xrange"])
        if panel_mode == "r2_rel_prct":
            return (0.0, 100.0)
        return tuple(state["r2_range"])

    def _xlabel(self, state) -> str:
        panel_mode = state["panel_mode"]
        if panel_mode == "r2_rel":
            return "Spatial reliability"
        if panel_mode == "r2_rel_prct":
            return "Reliability percentile"
        return r"$R^2$"

    def _ylabel(self, state) -> str:
        panel_mode = state["panel_mode"]
        if panel_mode == "histogram":
            return "Fraction of ROIs"
        if panel_mode == "ecdf":
            return "Cumulative fraction"
        return r"$R^2$"

    def _row_values(self, row: int, state) -> tuple[np.ndarray, np.ndarray] | None:
        """One session's per-ROI ``(R², reliability)`` under the current env mode, None if empty.

        ROIs missing either quantity are dropped from both, so every panel mode describes the same
        set of ROIs regardless of whether it uses reliability.
        """
        env_mode = state["env_mode"]
        if env_mode == "pooled":
            # Every (ROI, environment) pair, the same population r2_kde_pooled is built from.
            r2 = self.r2_slot[row].reshape(-1)
            reliability = self.reliability_slot[row].reshape(-1)
        elif env_mode == "best":
            slot = self.best_env_slot[row]
            if not np.isfinite(slot):
                return None
            r2 = self.r2_slot[row, int(slot)]
            reliability = self.reliability_slot[row, int(slot)]
        elif env_mode == "slot":
            r2 = self.r2_slot[row, state["env_slot"]]
            reliability = self.reliability_slot[row, state["env_slot"]]
        else:
            raise ValueError(f"Invalid env_mode: {env_mode!r}")
        valid = np.isfinite(r2) & np.isfinite(reliability)
        return (r2[valid], reliability[valid]) if np.any(valid) else None

    def _row_kde_curve(self, row: int, state) -> np.ndarray | None:
        """One session's stored E[R² | reliability] under the current env mode, None if it has none."""
        env_mode = state["env_mode"]
        if env_mode == "pooled":
            curve = self.r2_kde_pooled[row]
        elif env_mode == "best":
            slot = self.best_env_slot[row]
            # A session whose best environment is outside the mouse's slot list has no curve.
            if not np.isfinite(slot):
                return None
            curve = self.r2_kde_slot[row, int(slot)]
        elif env_mode == "slot":
            curve = self.r2_kde_slot[row, state["env_slot"]]
        else:
            raise ValueError(f"Invalid env_mode: {env_mode!r}")
        return curve if np.any(np.isfinite(curve)) else None

    def _row_curve(self, row: int, state) -> np.ndarray | None:
        """One session's curve on ``_grid(state)`` under the current panel mode, None if it has none."""
        panel_mode = state["panel_mode"]
        if panel_mode == "r2_rel":
            return self._row_kde_curve(row, state)

        values = self._row_values(row, state)
        if values is None:
            return None
        r2, reliability = values
        grid = self._grid(state)

        if panel_mode == "r2_rel_prct":
            # Rank, mapped to (0, 100) at bin midpoints, is uniform by construction: every x of the
            # regression is backed by the same number of ROIs, which uniform reliability bins are
            # very far from being.
            percentile = 100.0 * (rankdata(reliability) - 0.5) / reliability.size
            return _kde_r2(r2, percentile, grid, bw=state["prct_bandwidth"])["r2_kde_mean"]

        low, high = tuple(state["r2_range"])
        # Clipped rather than masked: badly predicted ROIs belong in the edge bin, and silently
        # dropping them would renormalize the fractions over a session-dependent population.
        clipped = np.clip(r2, low, high)
        if panel_mode == "histogram":
            counts, _ = np.histogram(clipped, bins=np.linspace(low, high, state["r2_bins"] + 1))
            return counts / r2.size
        if panel_mode == "ecdf":
            return np.searchsorted(np.sort(clipped), grid, side="right") / r2.size
        raise ValueError(f"Invalid panel_mode: {panel_mode!r}")

    def _by_session_curves(self, state) -> np.ndarray:
        """Curves of every mouse arranged by session number: ``(mice, max_sessions, grid)``.

        Mice with shorter histories are NaN-padded, so column j is "everyone's j-th session".
        """
        per_mouse = [[self._row_curve(row, state) for row in self._rows_by_mouse[mouse]] for mouse in self.mice]
        max_sessions = max((len(curves) for curves in per_mouse), default=0)
        by_session = np.full((len(per_mouse), max_sessions, len(self._grid(state))), np.nan)
        for imouse, curves in enumerate(per_mouse):
            for isession, curve in enumerate(curves):
                if curve is not None:
                    by_session[imouse, isession] = curve
        return by_session

    def _summarizer(self, state):
        """The ROI-collapsing statistic ax[2] uses, as a callable on a 1-D array of R² values."""
        summary_stat = state["summary_stat"]
        if summary_stat == "mean":
            return np.mean
        if summary_stat == "median":
            return np.median
        if summary_stat == "percentile":
            percentile = state["summary_percentile"]
            return lambda values: np.percentile(values, percentile)
        raise ValueError(f"Invalid summary_stat: {summary_stat!r}")

    def _summary_label(self, state) -> str:
        """Y-axis label of ax[2], naming the statistic (percentiles carry their level)."""
        if state["summary_stat"] == "percentile":
            return f"{_ordinal(state['summary_percentile'])} pct " + r"$R^2$"
        return f"{state['summary_stat'].capitalize()} " + r"$R^2$"

    def _slot_summary_stacks(self, state) -> dict[int, np.ndarray]:
        """Per-slot ``(mice, sessions)`` summary stat of R², densified to sessions that ran the slot.

        A mouse's curve for slot j is the summary stat over ROIs of every session in which it ran
        that environment, in chronological order -- so x is "how many sessions of this environment
        the mouse has had", not the session number in the experiment.
        """
        summarize = self._summarizer(state)
        stacks = {}
        for slot in range(self.num_slots):
            per_mouse = []
            for mouse in self.mice:
                values = []
                for row in self._rows_by_mouse[mouse]:
                    r2 = self.r2_slot[row, slot]
                    r2 = r2[np.isfinite(r2)]
                    if r2.size:
                        values.append(float(summarize(r2)))
                per_mouse.append(np.array(values))
            stacks[slot] = _pad_stack(per_mouse)
        return stacks

    def plot(self, state):
        fontsize = self.fontsize
        # Set before any artist is created so every default-sized text picks it up.
        plt.rcParams["font.size"] = fontsize

        cmap_name = state["cmap"]
        cmap = mpl.colormaps[cmap_name]
        panel_mode = state["panel_mode"]
        grid = self._grid(state)
        xlow, xhigh = self._xlimits(state)
        in_range = (grid >= xlow) & (grid <= xhigh)

        # Every panel mode's quantity is anchored at zero, so only the top of the shared y-axis is
        # fit to the curves.
        ymax = -np.inf

        def _track(curve) -> None:
            """Grow the shared y limit to cover a curve's max inside the visible x range."""
            nonlocal ymax
            visible = np.asarray(curve)[in_range]
            visible = visible[np.isfinite(visible)]
            if visible.size:
                ymax = max(ymax, float(np.max(visible)))

        plt.close("all")
        width_ratios = [1.0, 1.0, 1.0, 1.0] if state["square_panels"] else [0.8, 1.2, 1.2, 1.0]
        fig, ax = plt.subplots(1, 4, figsize=self.figsize, layout="constrained", width_ratios=width_ratios)
        # ax[1] and ax[2] draw the same quantity, so the y-axis is drawn once, on ax[1].
        ax[2].sharey(ax[1])

        # ---- ax[0]: one ROI of one session, activity vs place-field prediction ----
        # Every predictable frame of the session, not just one environment's: this panel is about
        # how well a place field describes a cell at all, which is not a per-environment question.
        example_row = self._example_rows_by_mouse[state["mouse"]][state["example_session"]]
        example_data, example_prediction = self._example_roi_traces(example_row, state["roi"])
        _draw_example_roi_panel(
            ax[0],
            example_data,
            example_prediction,
            style=state["example_style"],
            alpha=state["example_alpha"],
            fontsize=fontsize,
            ylabel="PF Prediction",
        )

        # ---- ax[1]: every session of the example mouse, colored by session number ----
        rows = self._rows_by_mouse[state["mouse"]]
        num_sessions = len(rows)
        colors = [cmap(isession / max(num_sessions - 1, 1)) for isession in range(num_sessions)]

        for row, color in zip(rows, colors):
            curve = self._row_curve(row, state)
            if curve is None:
                continue
            ax[1].plot(grid, curve, color=color, linewidth=state["linewidth"], alpha=state["alpha"])
            _track(curve)

        ax[1].set_xlabel(self._xlabel(state), fontsize=fontsize)
        ax[1].set_ylabel(self._ylabel(state), fontsize=fontsize)

        # ---- ax[2]: the population, mouse-averaged overall or grouped by session number ----
        by_session = self._by_session_curves(state)
        if state["curve_mode"] == "by_session":
            # Only session numbers reached by more than one mouse are drawn: a group backed by a
            # single mouse is that mouse's curve, not a population average.
            support = np.array([np.sum(np.isfinite(by_session[:, j, :]).any(axis=1)) for j in range(by_session.shape[1])])
            kept_js = np.where(support > 1)[0]
            # Color over the kept session numbers, so the full colormap range is used even when
            # trailing sparse session numbers are excluded.
            session_colors = cmap(np.linspace(0, 1, max(len(kept_js), 1)))

            # Thin out which kept sessions are drawn (always first + last), so dense session
            # counts don't overplot the panel. Colors still index the full kept_js range.
            n_kept = len(kept_js)
            step = state["skip_sessions"] + 1
            if n_kept <= 2 or step <= 1:
                show_idx = np.arange(n_kept)
            else:
                n_points = max(2, int(round((n_kept - 1) / step)) + 1)
                show_idx = np.unique(np.round(np.linspace(0, n_kept - 1, n_points)).astype(int))

            for color_idx in show_idx:
                session_data = by_session[:, kept_js[color_idx], :]
                _render_curve_group(
                    ax[2],
                    grid,
                    session_data,
                    session_colors[color_idx],
                    state["plot_style"],
                    hide_error=state["hide_error"],
                    linewidth=1.5,
                )
                _track(np.nanmean(session_data, axis=0))

            if n_kept:
                inset_bounds = [state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]]
                _session_colorbar_inset(ax[2], kept_js[0] + 1, kept_js[-1] + 1, cmap_name, fontsize, inset_bounds)
        else:
            # One curve per mouse (its own sessions averaged), rendered as a single group.
            mouse_average = np.stack(
                [np.nanmean(curves, axis=0) if np.any(np.isfinite(curves)) else np.full(len(grid), np.nan) for curves in by_session]
            )
            _render_curve_group(
                ax[2],
                grid,
                mouse_average,
                "k",
                state["plot_style"],
                hide_error=state["hide_error"],
                linewidth=2.0,
            )
            _track(np.nanmean(mouse_average, axis=0))
        ax[2].set_xlabel(self._xlabel(state), fontsize=fontsize)

        for axis in (ax[1], ax[2]):
            axis.set_xlim(xlow, xhigh)
        # ymax stays infinite when nothing was plotted -- an env slot no session of this mouse
        # ran, with a curve_mode that also produced nothing.
        if panel_mode == "ecdf":
            # A cumulative fraction spans its own axis; r2_ylim is about R².
            ybounds = [0.0, 1.0]
        elif panel_mode == "histogram" or state["auto_ylim"]:
            ybounds = [0.0, ymax] if np.isfinite(ymax) else None
        else:
            ybounds = list(state["r2_ylim"])

        # ---- ax[3]: summary R² against experience, one curve per environment slot ----
        slot_stacks = self._slot_summary_stacks(state)
        slot_ymax, slot_xmax = -np.inf, 0
        for slot, stack in slot_stacks.items():
            length = _support_length(stack)
            if length == 0:
                continue
            stack = stack[:, :length]
            xvals = np.arange(1, length + 1)
            _render_curve_group(
                ax[3],
                xvals,
                stack,
                ENV_SLOT_COLORS[slot % len(ENV_SLOT_COLORS)],
                state["slot_style"],
                label=_ordinal(slot + 1),
                hide_error=state["hide_error"],
                linewidth=1.5,
            )
            # The band, not the individual mice, sets the y range in errorPlot style.
            if state["slot_style"] == "each":
                visible = stack
            else:
                spread = np.nanstd(stack, axis=0)
                mean_curve = np.nanmean(stack, axis=0)
                visible = np.stack([mean_curve - spread, mean_curve + spread])
            visible = visible[np.isfinite(visible)]
            if visible.size:
                slot_ymax = max(slot_ymax, float(np.max(visible)))
            slot_xmax = max(slot_xmax, length)

        ax[3].set_xlabel("Env session #", fontsize=fontsize)
        ax[3].set_ylabel(self._summary_label(state), fontsize=fontsize)
        if state["show_slot_legend"]:
            legend = ax[3].legend(fontsize=fontsize, frameon=False, handlelength=0.8, handletextpad=0.5, title="Env")
            legend.get_title().set_fontsize(fontsize)
        # Anchored at zero like the curve panels, so the two ranges are comparable at a glance.
        slot_ybounds = [0.0, slot_ymax] if np.isfinite(slot_ymax) else None

        # Two nearly-equal y ranges side by side read as one axis but aren't, which is exactly the
        # comparison the eye gets wrong. Within the tolerance they are made into one for real.
        if ybounds is not None and slot_ybounds is not None:
            higher = max(ybounds[1], slot_ybounds[1])
            if higher > 0 and min(ybounds[1], slot_ybounds[1]) >= (1.0 - _SHARE_Y_TOLERANCE) * higher:
                ybounds = [0.0, higher]
                slot_ybounds = [0.0, higher]
                ax[3].sharey(ax[1])

        if ybounds is not None:
            ax[1].set_ylim(ybounds)
        format_spines(
            ax[1],
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[xlow, xhigh],
            ybounds=ybounds,
            yticks=_decimal_yticks(*ybounds) if ybounds is not None else None,
            tick_length=4,
            tick_fontsize=fontsize,
            spines_visible=["left", "bottom"],
        )
        # ax[2] borrows ax[1]'s y-axis, so it only gets a bottom spine. Its ticks are hidden with
        # tick_params rather than set_yticks, which would propagate through the shared axis and
        # strip ax[1]'s y ticks too.
        format_spines(
            ax[2],
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[xlow, xhigh],
            tick_length=4,
            tick_fontsize=fontsize,
            spines_visible=["bottom"],
        )
        ax[2].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        if slot_xmax:
            ax[3].set_xlim(1, slot_xmax)
        if slot_ybounds is not None:
            ax[3].set_ylim(slot_ybounds)
        format_spines(
            ax[3],
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[1, max(slot_xmax, 1)],
            ybounds=slot_ybounds,
            xticks=[1, max(slot_xmax, 1)],
            yticks=_decimal_yticks(*slot_ybounds) if slot_ybounds is not None else None,
            tick_length=4,
            tick_fontsize=fontsize,
            spines_visible=["left", "bottom"],
        )

        # Last, because it measures the drawn figure: everything that takes up room must be on it.
        if state["square_panels"]:
            _fit_square_panels(fig, ax)

        return fig


def r2_familiarity(
    results: ResultsAggregator,
    mouse: str | None = None,
    example_session: str | None = None,
    roi: int = 0,
    example_style: str = "points",
    example_alpha: float = 0.1,
    env_mode: str = "best",
    env_slot: int = 0,
    panel_mode: str = "r2_rel",
    xrange: tuple[float, float] | None = None,
    r2_range: tuple[float, float] = (-1.0, 1.0),
    r2_bins: int = 20,
    prct_points: int = 101,
    prct_bandwidth: float = 5.0,
    auto_ylim: bool = True,
    r2_ylim: tuple[float, float] = (-0.05, 0.4),
    curve_mode: str = "by_session",
    plot_style: str = "each",
    hide_error: bool = False,
    skip_sessions: int = 0,
    inset_position: tuple[float, float, float, float] = (0.35, 0.85, 0.6, 0.075),
    summary_stat: str = "mean",
    summary_percentile: float = 90.0,
    slot_style: str = "errorPlot",
    show_slot_legend: bool = True,
    cmap: str = "coolwarm",
    linewidth: float = 1.0,
    alpha: float = 0.9,
    fontsize: float = 8.0,
    figsize: tuple[float, float] = (10.0, 2.0),
    square_panels: bool = False,
    return_syd_viewer: bool = False,
):
    """
    R² as a function of familiarity: one ROI, one mouse, the population, the environment slots.

    ``panel_mode`` picks what ``ax[1]``/``ax[2]`` draw. The default ``"r2_rel"`` is the running
    average E[R² | reliability] -- the kernel regression drawn on ``ax[1]`` of
    :func:`example_r2_placefield`, resolved per session instead of pooled, read straight from the
    ``PFPredQualityConfig`` aggregator (``r2_kde_slot`` per environment slot, ``r2_kde_pooled`` over
    every (ROI, environment) pair). The other three are computed here from the per-ROI ``r2_slot``
    and ``reliability_slot``: ``"r2_rel_prct"`` regresses on reliability rank instead of reliability
    (equally-populated x), while ``"histogram"`` and ``"ecdf"`` drop reliability and show the
    marginal distribution of R² itself. Conditioning on reliability asks whether a cell of given
    quality improves; the marginal asks whether the population does, drift of the reliability
    distribution included.

    ``ax[0]`` is the example panel of :func:`example_r2_placefield`: one ROI's activity against its
    place-field prediction, over every predictable frame of ``example_session`` rather than one
    environment's. ``ax[1]`` draws one curve per session of ``mouse``, colored by session number
    with ``cmap``, so the experiment's time axis is the color axis. ``ax[2]`` draws the same curves
    for every mouse, either averaged (``curve_mode="average"``) or grouped by session number
    (``"by_session"``), sharing ``ax[1]``'s y-axis. ``ax[3]`` collapses each session to its
    ``summary_stat`` R² over ROIs and plots it against how many sessions of that environment the
    mouse has had, one curve per experience slot -- slot 0 being the first environment the mouse
    ever saw, which is what makes the panel comparable across the two cohorts. Every R² axis starts
    at 0 and ticks in multiples of 0.1, and ``ax[3]`` joins the shared axis outright when its range
    comes out within 15% of the curve panels'.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``PFPredQualityConfig`` results (schema v5 or later).
    mouse : str or None
        Example mouse for ``ax[0]`` and ``ax[1]``. None takes the first alphabetically.
    example_session : str or None
        Which of that mouse's sessions ``ax[0]``'s ROI comes from, as ``session_print()`` prints it
        (``"mouse/date/id"``). None takes the mouse's first session. Its place-field prediction
        and normalization scale are read from ``PlacefieldPredictionConfig`` schema v1.
    roi : int
        ROI index within ``example_session``, in session-filtered ROI coordinates.
    example_style : str
        How ``ax[0]`` draws the frames: ``"points"`` for a translucent cloud, ``"density"`` for a
        filled 2D KDE of the same points.
    example_alpha : float
        Opacity of ``ax[0]``'s point cloud. Ignored by ``example_style="density"``.
    env_mode : str
        Which environments contribute to ``ax[1]`` and ``ax[2]``. ``"best"`` takes the session's
        best environment (the one the top-level R² keys describe), ``"pooled"`` the curve over
        every (ROI, environment) pair, ``"slot"`` a single experience slot. ``ax[0]`` always uses
        every environment and ``ax[3]`` always splits by slot, so neither is affected.
    env_slot : int
        Experience-order slot used when ``env_mode="slot"``. Sessions that did not run that
        environment contribute nothing.
    panel_mode : str
        What ``ax[1]``/``ax[2]`` draw. ``"r2_rel"`` is E[R² | reliability], ``"r2_rel_prct"`` the
        same against within-session reliability rank, ``"histogram"`` the fraction of ROIs per R²
        bin, ``"ecdf"`` the cumulative fraction. ``ax[0]`` and ``ax[3]`` are unaffected.
    xrange : tuple of float or None
        Reliability limits of ``ax[1]``/``ax[2]`` under ``panel_mode="r2_rel"``. None shows the full
        stored range, ``(-1, 1)``. The rank axis is always the full 0-100.
    r2_range : tuple of float
        R² limits of the ``"histogram"`` and ``"ecdf"`` modes, and the range binning and clipping
        use: ROIs outside it land in the edge bin rather than being dropped.
    r2_bins : int
        Number of equal-width R² bins inside ``r2_range`` for ``panel_mode="histogram"``.
    prct_points : int
        Grid resolution of the rank axis for ``panel_mode="r2_rel_prct"``.
    prct_bandwidth : float
        Kernel width of the rank regression, in percentile units.
    auto_ylim : bool
        Fit the top of the shared R² axis of ``ax[1]``/``ax[2]`` to the curves inside ``xrange``;
        the bottom is always 0. False uses ``r2_ylim``. Ignored by ``"histogram"`` (always fit) and
        ``"ecdf"`` (always 0 to 1).
    r2_ylim : tuple of float
        Explicit R² limits, used when ``auto_ylim`` is False.
    curve_mode : str
        ``ax[2]`` content: ``"average"`` for one curve per mouse (its sessions averaged), or
        ``"by_session"`` for one group per session number, colored by session number. Session
        numbers reached by only one mouse are dropped -- a single mouse is not a population.
    plot_style : str
        How ``ax[2]`` renders a group: ``"each"`` (every mouse thin, plus a solid mean) or
        ``"errorPlot"`` (mean +/- std band). Same meaning as in ``figure3``.
    hide_error : bool
        With ``plot_style``/``slot_style`` ``"errorPlot"``, draw the mean line without the band.
    skip_sessions : int
        Thin out how many of ``ax[2]``'s session-number groups are drawn (first and last are
        always kept). 0 draws all of them.
    inset_position : tuple of float
        ``(x, y, width, height)`` of ``ax[2]``'s session-number colorbar inset, in axes fractions.
    summary_stat : str
        How ``ax[3]`` collapses a session's per-ROI R² to one number: ``"mean"``, ``"median"``, or
        ``"percentile"`` at ``summary_percentile``. A high percentile tracks the best-predicted
        cells rather than the bulk, which the mean is dominated by.
    summary_percentile : float
        Percentile in ``[0, 100]`` used when ``summary_stat="percentile"``, in steps of 0.5.
    slot_style : str
        ``ax[3]``'s rendering, with the same options as ``plot_style``.
    show_slot_legend : bool
        Draw the env-slot legend on ``ax[3]``.
    cmap : str
        Colormap sampled across sessions, first session at 0 and last at 1.
    linewidth, alpha : float
        Line width and opacity of ``ax[1]``'s curves.
    fontsize : float
        Base font size in points.
    figsize : tuple of float
        Figure size in inches. The height is ignored when ``square_panels`` is set.
    square_panels : bool
        Give all four panels a square aspect: the panels are drawn one width (the width ratios
        ``[0.8, 1.2, 1.2, 1.0]`` are dropped, since a row of axes shares one height and only equal
        widths can all be square) and the figure height is solved for from ``figsize[0]`` and
        however much room the tick labels, axis labels and padding turn out to need.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.

    Returns
    -------
    matplotlib.figure.Figure or R2Familiarity
    """
    viewer = R2Familiarity(results, mouse=mouse, fontsize=fontsize, figsize=figsize)
    # Seeding via update_* does not fire on_change before deployment.
    viewer.update_slot_bounds(viewer.state)
    viewer.update_example_session(viewer.state)

    # None keeps the mouse's first session, which update_example_session already selected.
    if example_session is not None:
        viewer.update_selection("example_session", value=example_session)
        viewer.update_roi_bounds(viewer.state)
    viewer.update_integer("roi", value=roi)
    viewer.update_selection("example_style", value=example_style)
    viewer.update_float("example_alpha", value=example_alpha)

    viewer.update_selection("env_mode", value=env_mode)
    viewer.update_integer("env_slot", value=env_slot)
    viewer.update_selection("panel_mode", value=panel_mode)
    if xrange is not None:
        viewer.update_float_range("xrange", value=tuple(xrange))
    viewer.update_float_range("r2_range", value=tuple(r2_range))
    viewer.update_integer("r2_bins", value=r2_bins)
    viewer.update_integer("prct_points", value=prct_points)
    viewer.update_float("prct_bandwidth", value=prct_bandwidth)
    viewer.update_boolean("auto_ylim", value=auto_ylim)
    viewer.update_float_range("r2_ylim", value=tuple(r2_ylim))

    viewer.update_selection("curve_mode", value=curve_mode)
    viewer.update_selection("plot_style", value=plot_style)
    viewer.update_boolean("hide_error", value=hide_error)
    viewer.update_integer("skip_sessions", value=skip_sessions)
    viewer.update_float("inset_x", value=inset_position[0])
    viewer.update_float("inset_y", value=inset_position[1])
    viewer.update_float("inset_width", value=inset_position[2])
    viewer.update_float("inset_height", value=inset_position[3])

    viewer.update_selection("summary_stat", value=summary_stat)
    viewer.update_float("summary_percentile", value=summary_percentile)
    viewer.update_selection("slot_style", value=slot_style)
    viewer.update_boolean("show_slot_legend", value=show_slot_legend)

    viewer.update_text("cmap", value=cmap)
    viewer.update_float("linewidth", value=linewidth)
    viewer.update_float("alpha", value=alpha)
    viewer.update_boolean("square_panels", value=square_panels)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


class PlacefieldPeakAmplitude(Viewer):
    """Place-field peak amplitude: one example mouse, the population, and the environment slots.

    A peak is the max over position of an ROI's trial-averaged place field, in units of that ROI's
    standard deviation in time. Everything is read from the ``PFPredQualityConfig`` aggregator --
    ``pf_peak_hist`` (counts per session and environment slot, precomputed on
    ``pf_peak_hist_edges``), ``pf_peak_n`` (how many ROIs went into each), and ``pf_peak`` (the
    per-ROI peaks themselves) -- so nothing is measured here and the figure is cheap to redraw.

    - ``ax[0]``: every session of one mouse, one histogram per session colored by session number
      (``coolwarm`` by default, early sessions cool and late warm). A drift in how strongly cells
      are driven at their preferred position shows up as an ordered fan of curves.
    - ``ax[1]``: the same curves for the whole population -- mouse-averaged overall
      (``curve_mode="average"``) or grouped by session number (``"by_session"``, one group per
      session number with at least two mice, thinned by ``skip_sessions``). Shares ``ax[0]``'s
      y-axis, and renders with the ``each`` / ``errorPlot`` styles of ``figure3``.
    - ``ax[2]``: the distribution collapsed to one number per session (``summary_stat``), against
      how many sessions of that environment the mouse has had, one curve per experience slot.
    - An optional fourth panel repeats that summary for the example mouse alone.

    Environments are indexed by experience-order slot, so slot 0 is the first environment that
    mouse ever saw; ``env_slot_ids`` carries the underlying environment index, which differs
    between the two cohorts.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        mouse: str | None = None,
        fontsize: float = 8.0,
        figsize: tuple[float, float] = (8.0, 2.0),
    ):
        self.results = results
        self.fontsize = fontsize
        self.figsize = figsize

        keys = ["pf_peak", "pf_peak_hist", "pf_peak_n", "pf_peak_hist_edges", "env_slot_ids"]
        out = results.sel(keys=keys, squeeze_ones=False)
        self.pf_peak = np.asarray(out["pf_peak"], dtype=float)  # (sessions, slots, max_rois)
        self.pf_peak_hist = np.asarray(out["pf_peak_hist"], dtype=float)  # (sessions, slots, bins)
        self.pf_peak_n = np.asarray(out["pf_peak_n"], dtype=float)  # (sessions, slots)
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

        # Rows of every array above are aggregator sessions; a mouse's rows are ordered
        # chronologically here so "session number" means what it says on the color axis.
        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(mouse_name) for mouse_name in mouse_names})
        self._rows_by_mouse = {
            mouse_name: np.array(sorted(np.flatnonzero(mouse_names == mouse_name), key=lambda row: _session_sort_key(results.sessions[row])))
            for mouse_name in self.mice
        }

        # --- what goes into a curve ---
        self.add_selection("mouse", value=mouse if mouse is not None else self.mice[0], options=self.mice)
        self.add_selection("env_mode", value="pooled", options=["pooled", "best", "slot"])
        self.add_integer("env_slot", value=0, min=0, max=MAX_ENV_SLOTS - 1)
        self.add_float_range(
            "xrange",
            value=(float(self.bin_edges[0]), float(self.bin_edges[-1])),
            min=float(self.bin_edges[0]),
            max=float(self.bin_edges[-1]),
            step=self.bin_width,
        )
        self.add_boolean("density", value=True)
        self.add_boolean("cumulative", value=False)
        self.add_boolean("log_y", value=False)

        # --- ax[1]: population curves (same knobs as figure3's cross-spectrum summary panel) ---
        self.add_selection("curve_mode", value="by_session", options=["average", "by_session"])
        self.add_selection("plot_style", value="each", options=["each", "errorPlot"])
        self.add_boolean("hide_error", value=False)
        self.add_integer("skip_sessions", value=0, min=0, max=max(len(results.sessions), 1))
        # ax[1] session-number colorbar inset placement, in ax[1] axes-fraction coordinates.
        self.add_float("inset_x", value=0.35, min=-1.0, max=1.0, step=0.01)
        self.add_float("inset_y", value=0.85, min=-1.0, max=1.0, step=0.01)
        self.add_float("inset_width", value=0.6, min=0.01, max=1.5, step=0.01)
        self.add_float("inset_height", value=0.075, min=0.01, max=1.0, step=0.005)

        # --- ax[2]: summary stat against experience, one curve per environment slot ---
        self.add_selection("summary_stat", value="median", options=["median", "mean"])
        self.add_selection("slot_style", value="errorPlot", options=["each", "errorPlot"])
        self.add_boolean("show_slot_legend", value=True)

        # --- style ---
        self.add_selection("style", value="line", options=["line", "step", "fill"])
        self.add_text("cmap", value="coolwarm")
        self.add_float("linewidth", value=1.0, min=0.25, max=4.0)
        self.add_float("alpha", value=0.9, min=0.05, max=1.0)
        self.add_boolean("show_colorbar", value=True)
        self.add_boolean("show_env_label", value=True)
        self.add_boolean("show_summary", value=False)

        self.on_change("mouse", self.update_slot_bounds)
        self.update_slot_bounds(self.state)

    def update_slot_bounds(self, state):
        """Limit the slot selector to the environments this mouse actually ran."""
        rows = self._rows_by_mouse[state["mouse"]]
        num_slots = int(np.sum(np.any(np.isfinite(self.pf_peak_n[rows]), axis=0)))
        self.update_integer("env_slot", max=max(num_slots - 1, 0))

    def _env_mode_label(self, state) -> str:
        """Corner label naming the environments the curves were built from.

        In slot mode it also quotes the underlying environment index, which is what the label
        means for *this* mouse -- the two cohorts run disjoint environment indices, so the slot
        number is the only thing comparable across them.
        """
        if state["env_mode"] == "pooled":
            return "all envs"
        if state["env_mode"] == "best":
            return "best env"
        slot = state["env_slot"]
        slot_ids = self.env_slot_ids[self._rows_by_mouse[state["mouse"]], slot]
        slot_ids = slot_ids[np.isfinite(slot_ids)]
        if slot_ids.size == 0:
            return f"env #{slot + 1}"
        return f"env #{slot + 1} (id {int(slot_ids[0])})"

    def _row_peaks(self, row: int, state) -> np.ndarray:
        """Per-ROI peaks of one session under the current env mode (empty if it ran nothing)."""
        slot_peaks = self.pf_peak[row]  # (slots, max_rois)
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
        if env_mode == "best":
            # Not derivable from the per-slot histograms -- rebin the stored per-ROI peaks.
            values = self._row_peaks(row, state)
            return np.histogram(values, bins=self.bin_edges)[0].astype(float), float(values.size)
        if env_mode == "pooled":
            valid = np.isfinite(self.pf_peak_n[row])
            if not np.any(valid):
                return np.zeros(len(self.bin_centers)), 0.0
            return np.sum(self.pf_peak_hist[row][valid], axis=0), float(np.sum(self.pf_peak_n[row][valid]))
        slot = state["env_slot"]
        n_slot = self.pf_peak_n[row, slot]
        if not np.isfinite(n_slot):
            return np.zeros(len(self.bin_centers)), 0.0
        return self.pf_peak_hist[row, slot], float(n_slot)

    def _row_curve(self, row: int, state) -> np.ndarray | None:
        """One session's plotted curve (density or counts, optionally cumulative), None if empty."""
        counts, n_peaks = self._row_counts(row, state)
        if n_peaks == 0:
            return None
        curve = counts / (n_peaks * self.bin_width) if state["density"] else counts
        if state["cumulative"]:
            curve = np.cumsum(curve) * (self.bin_width if state["density"] else 1.0)
        return curve

    def _by_session_curves(self, state) -> np.ndarray:
        """Curves of every mouse arranged by session number: ``(mice, max_sessions, bins)``.

        Mice with shorter histories are NaN-padded, so column j is "everyone's j-th session".
        """
        per_mouse = [[self._row_curve(row, state) for row in self._rows_by_mouse[mouse]] for mouse in self.mice]
        max_sessions = max((len(curves) for curves in per_mouse), default=0)
        by_session = np.full((len(per_mouse), max_sessions, len(self.bin_centers)), np.nan)
        for imouse, curves in enumerate(per_mouse):
            for isession, curve in enumerate(curves):
                if curve is not None:
                    by_session[imouse, isession] = curve
        return by_session

    def _slot_summary_stacks(self, state) -> dict[int, np.ndarray]:
        """Per-slot ``(mice, sessions)`` summary stat, densified to the sessions that ran the slot.

        A mouse's curve for slot j is the summary stat of every session in which it ran that
        environment, in chronological order -- so x is "how many sessions of this environment the
        mouse has had", not the session number in the experiment.
        """
        summarize = np.median if state["summary_stat"] == "median" else np.mean
        stacks = {}
        for slot in range(self.num_slots):
            per_mouse = []
            for mouse in self.mice:
                values = []
                for row in self._rows_by_mouse[mouse]:
                    peaks = self.pf_peak[row, slot]
                    peaks = peaks[np.isfinite(peaks)]
                    if peaks.size:
                        values.append(float(summarize(peaks)))
                per_mouse.append(np.array(values))
            stacks[slot] = _pad_stack(per_mouse)
        return stacks

    def plot(self, state):
        fontsize = self.fontsize
        # Set before any artist is created so every default-sized text picks it up.
        plt.rcParams["font.size"] = fontsize

        cmap_name = state["cmap"]
        cmap = mpl.colormaps[cmap_name]
        xlow, xhigh = tuple(state["xrange"])
        in_range = (self.bin_centers >= xlow) & (self.bin_centers <= xhigh)

        def _peak_ymax(curve) -> float:
            """Largest value of a curve inside the visible x range."""
            visible = np.asarray(curve)[in_range]
            visible = visible[np.isfinite(visible)]
            return float(np.max(visible)) if visible.size else 0.0

        plt.close("all")
        width_ratios = [1.2, 1.2, 1.0] + ([0.9] if state["show_summary"] else [])
        fig, ax = plt.subplots(1, len(width_ratios), figsize=self.figsize, layout="constrained", width_ratios=width_ratios)
        # ax[0] and ax[1] draw the same quantity, so the y-axis is drawn once, on ax[0].
        ax[1].sharey(ax[0])

        # ---- ax[0]: every session of the example mouse, colored by session number ----
        rows = self._rows_by_mouse[state["mouse"]]
        num_sessions = len(rows)
        colors = [cmap(isession / max(num_sessions - 1, 1)) for isession in range(num_sessions)]

        ymax = 0.0
        for row, color in zip(rows, colors):
            curve = self._row_curve(row, state)
            if curve is None:
                continue
            if state["style"] == "line":
                ax[0].plot(self.bin_centers, curve, color=color, linewidth=state["linewidth"], alpha=state["alpha"])
            elif state["style"] == "step":
                ax[0].stairs(curve, self.bin_edges, color=color, linewidth=state["linewidth"], alpha=state["alpha"])
            elif state["style"] == "fill":
                ax[0].fill_between(self.bin_centers, curve, color=color, alpha=state["alpha"], linewidth=0)
            else:
                raise ValueError(f"Invalid style: {state['style']!r}")
            ymax = max(ymax, _peak_ymax(curve))

        ylabel = ("Cumulative " if state["cumulative"] else "") + ("density" if state["density"] else "count")
        ax[0].set_xlabel(r"Peak PF amplitude ($\sigma$)", fontsize=fontsize)
        ax[0].set_ylabel(ylabel.capitalize(), fontsize=fontsize)
        ax[0].set_title(state["mouse"], fontsize=fontsize)

        if state["show_colorbar"] and num_sessions > 1:
            scalar_map = mpl.cm.ScalarMappable(norm=plt.Normalize(vmin=1, vmax=num_sessions), cmap=cmap)
            cbar = fig.colorbar(scalar_map, ax=ax[0], fraction=0.05, pad=0.02)
            cbar.set_label("Session", fontsize=fontsize)
            cbar.set_ticks([1, num_sessions])
            cbar.ax.tick_params(labelsize=fontsize, length=3)
            cbar.outline.set_visible(False)

        # ---- ax[1]: the population, mouse-averaged overall or grouped by session number ----
        by_session = self._by_session_curves(state)
        if state["curve_mode"] == "by_session":
            # Only session numbers reached by more than one mouse are drawn: a group backed by a
            # single mouse is that mouse's curve, not a population average.
            support = np.array([np.sum(np.isfinite(by_session[:, j, :]).any(axis=1)) for j in range(by_session.shape[1])])
            kept_js = np.where(support > 1)[0]
            # Color over the kept session numbers, so the full colormap range is used even when
            # trailing sparse session numbers are excluded.
            session_colors = cmap(np.linspace(0, 1, max(len(kept_js), 1)))

            # Thin out which kept sessions are drawn (always first + last), so dense session
            # counts don't overplot the panel. Colors still index the full kept_js range.
            n_kept = len(kept_js)
            step = state["skip_sessions"] + 1
            if n_kept <= 2 or step <= 1:
                show_idx = np.arange(n_kept)
            else:
                n_points = max(2, int(round((n_kept - 1) / step)) + 1)
                show_idx = np.unique(np.round(np.linspace(0, n_kept - 1, n_points)).astype(int))

            for color_idx in show_idx:
                session_data = by_session[:, kept_js[color_idx], :]
                _render_curve_group(
                    ax[1],
                    self.bin_centers,
                    session_data,
                    session_colors[color_idx],
                    state["plot_style"],
                    hide_error=state["hide_error"],
                    linewidth=1.5,
                )
                ymax = max(ymax, _peak_ymax(np.nanmean(session_data, axis=0)))

            if n_kept:
                inset_bounds = [state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]]
                _session_colorbar_inset(ax[1], kept_js[0] + 1, kept_js[-1] + 1, cmap_name, fontsize, inset_bounds)
        else:
            # One curve per mouse (its own sessions averaged), rendered as a single group.
            mouse_average = np.stack(
                [np.nanmean(curves, axis=0) if np.any(np.isfinite(curves)) else np.full(len(self.bin_centers), np.nan) for curves in by_session]
            )
            _render_curve_group(
                ax[1],
                self.bin_centers,
                mouse_average,
                "k",
                state["plot_style"],
                hide_error=state["hide_error"],
                linewidth=2.0,
            )
            ymax = max(ymax, _peak_ymax(np.nanmean(mouse_average, axis=0)))
        ax[1].set_xlabel(r"Peak PF amplitude ($\sigma$)", fontsize=fontsize)
        # Mice the aggregator knows about but has no results for contribute nothing, so the title
        # counts the ones actually drawn rather than the length of the mouse list.
        num_mice_drawn = int(np.sum(np.isfinite(by_session).any(axis=(1, 2))))
        ax[1].set_title(f"{num_mice_drawn} mice", fontsize=fontsize)

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

        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[xlow, xhigh],
            ybounds=ybounds,
            tick_length=4,
            tick_fontsize=fontsize,
            spines_visible=["left", "bottom"],
        )
        # ax[1] borrows ax[0]'s y-axis, so it only gets a bottom spine. Its ticks are hidden with
        # tick_params rather than set_yticks, which would propagate through the shared axis and
        # strip ax[0]'s y ticks too.
        format_spines(
            ax[1],
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[xlow, xhigh],
            tick_length=4,
            tick_fontsize=fontsize,
            spines_visible=["bottom"],
        )
        ax[1].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        # ---- ax[2]: summary stat against experience, one curve per environment slot ----
        slot_stacks = self._slot_summary_stacks(state)
        slot_ymin, slot_ymax, slot_xmax = np.inf, -np.inf, 0
        for slot, stack in slot_stacks.items():
            length = _support_length(stack)
            if length == 0:
                continue
            stack = stack[:, :length]
            xvals = np.arange(1, length + 1)
            _render_curve_group(
                ax[2],
                xvals,
                stack,
                ENV_SLOT_COLORS[slot % len(ENV_SLOT_COLORS)],
                state["slot_style"],
                label=f"Env #{slot + 1}",
                hide_error=state["hide_error"],
                linewidth=1.5,
            )
            # The band, not the individual mice, sets the y range in errorPlot style.
            if state["slot_style"] == "each":
                visible = stack
            else:
                spread = np.nanstd(stack, axis=0)
                mean_curve = np.nanmean(stack, axis=0)
                visible = np.stack([mean_curve - spread, mean_curve + spread])
            visible = visible[np.isfinite(visible)]
            if visible.size:
                slot_ymin = min(slot_ymin, float(np.min(visible)))
                slot_ymax = max(slot_ymax, float(np.max(visible)))
            slot_xmax = max(slot_xmax, length)

        ax[2].set_xlabel("Env session #", fontsize=fontsize)
        ax[2].set_ylabel(f"{state['summary_stat'].capitalize()} peak ($\\sigma$)", fontsize=fontsize)
        if state["show_slot_legend"]:
            ax[2].legend(fontsize=fontsize, frameon=False, handlelength=0.8, handletextpad=0.5)
        if np.isfinite(slot_ymin) and slot_xmax:
            ax[2].set_xlim(1, slot_xmax)
            ax[2].set_ylim(slot_ymin, slot_ymax)
        format_spines(
            ax[2],
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=[1, max(slot_xmax, 1)],
            ybounds=[slot_ymin, slot_ymax] if np.isfinite(slot_ymin) else None,
            xticks=[1, max(slot_xmax, 1)],
            tick_length=4,
            tick_fontsize=fontsize,
            spines_visible=["left", "bottom"],
        )

        # ---- optional 4th panel: the same summary for the example mouse alone ----
        if state["show_summary"]:
            summarize = np.median if state["summary_stat"] == "median" else np.mean
            summary = np.array([summarize(peaks) if peaks.size else np.nan for peaks in (self._row_peaks(row, state) for row in rows)])
            session_numbers = np.arange(1, num_sessions + 1)
            ax[3].plot(session_numbers, summary, color="0.7", linewidth=1, zorder=1)
            ax[3].scatter(session_numbers, summary, c=colors, s=12, zorder=2)
            ax[3].set_xlabel("Session", fontsize=fontsize)
            ax[3].set_ylabel(f"{state['summary_stat'].capitalize()} peak ($\\sigma$)", fontsize=fontsize)
            summary_valid = summary[np.isfinite(summary)]
            format_spines(
                ax[3],
                x_pos=-0.02,
                y_pos=-0.02,
                xbounds=[1, num_sessions],
                ybounds=[float(np.min(summary_valid)), float(np.max(summary_valid))] if summary_valid.size else None,
                xticks=[1, num_sessions],
                tick_length=4,
                tick_fontsize=fontsize,
                spines_visible=["left", "bottom"],
            )

        return fig


def placefield_peak_amplitude(
    results: ResultsAggregator,
    mouse: str | None = None,
    env_mode: str = "pooled",
    env_slot: int = 0,
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
    return_syd_viewer: bool = False,
):
    """
    Place-field peak amplitude: one example mouse, the population, and the environment slots.

    Read straight from the ``PFPredQualityConfig`` aggregator: ``pf_peak_hist`` holds the histogram
    of every session and environment slot, precomputed on ``pf_peak_hist_edges``, and ``pf_peak``
    the per-ROI peaks behind it. A peak is the max over position of an ROI's trial-averaged place
    field, in units of that ROI's standard deviation in time.

    ``ax[0]`` draws one curve per session of ``mouse``, colored by session number with ``cmap``, so
    the experiment's time axis is the color axis. ``ax[1]`` draws the same curves for every mouse,
    either averaged (``curve_mode="average"``) or grouped by session number
    (``"by_session"``), sharing ``ax[0]``'s y-axis. ``ax[2]`` collapses each session to its
    ``summary_stat`` and plots it against how many sessions of that environment the mouse has had,
    one curve per experience slot -- slot 0 being the first environment the mouse ever saw, which
    is what makes the panel comparable across the two cohorts.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``PFPredQualityConfig`` results.
    mouse : str or None
        Example mouse for ``ax[0]`` (and the optional 4th panel). None takes the first
        alphabetically.
    env_mode : str
        Which environments contribute to ``ax[0]`` and ``ax[1]``. ``"pooled"`` sums the per-slot
        histograms (every (ROI, environment) pair is a sample), ``"best"`` rebins each ROI's
        largest peak across slots, ``"slot"`` takes a single experience slot. ``ax[2]`` always
        splits by slot, so this does not affect it.
    env_slot : int
        Experience-order slot used when ``env_mode="slot"``. Sessions that did not run that
        environment contribute nothing.
    xrange : tuple of float or None
        X limits of ``ax[0]``/``ax[1]``, in peak-amplitude units. None shows the full stored range.
        This only crops the view -- the bins are fixed by the config that wrote the results.
    density : bool
        Normalize each session's histogram by its ROI count and bin width. False plots raw counts,
        which then also reflect how many ROIs the session had.
    cumulative : bool
        Plot cumulative distributions instead of histograms.
    log_y : bool
        Log-scale the (shared) y axis of ``ax[0]`` and ``ax[1]``.
    curve_mode : str
        ``ax[1]`` content: ``"average"`` for one curve per mouse (its sessions averaged), or
        ``"by_session"`` for one group per session number, colored by session number. Session
        numbers reached by only one mouse are dropped -- a single mouse is not a population.
    plot_style : str
        How ``ax[1]`` renders a group: ``"each"`` (every mouse thin, plus a solid mean) or
        ``"errorPlot"`` (mean +/- std band). Same meaning as in ``figure3``.
    hide_error : bool
        With ``plot_style``/``slot_style`` ``"errorPlot"``, draw the mean line without the band.
    skip_sessions : int
        Thin out how many of ``ax[1]``'s session-number groups are drawn (first and last are
        always kept). 0 draws all of them.
    inset_position : tuple of float
        ``(x, y, width, height)`` of ``ax[1]``'s session-number colorbar inset, in axes fractions.
    summary_stat : str
        ``"median"`` or ``"mean"``: how ``ax[2]`` (and the optional 4th panel) collapses a
        session's peak distribution to one number.
    slot_style : str
        ``ax[2]``'s rendering, with the same options as ``plot_style``.
    show_slot_legend : bool
        Draw the env-slot legend on ``ax[2]``.
    style : str
        ``ax[0]``'s histogram rendering: ``"line"`` (bin centers), ``"step"`` (bin edges) or
        ``"fill"``.
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
    figsize : tuple of float
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldPeakAmplitude
    """
    viewer = PlacefieldPeakAmplitude(results, mouse=mouse, fontsize=fontsize, figsize=figsize)
    # Seeding via update_* does not fire on_change before deployment.
    viewer.update_slot_bounds(viewer.state)

    viewer.update_selection("env_mode", value=env_mode)
    viewer.update_integer("env_slot", value=env_slot)
    if xrange is not None:
        viewer.update_float_range("xrange", value=tuple(xrange))
    viewer.update_boolean("density", value=density)
    viewer.update_boolean("cumulative", value=cumulative)
    viewer.update_boolean("log_y", value=log_y)

    viewer.update_selection("curve_mode", value=curve_mode)
    viewer.update_selection("plot_style", value=plot_style)
    viewer.update_boolean("hide_error", value=hide_error)
    viewer.update_integer("skip_sessions", value=skip_sessions)
    viewer.update_float("inset_x", value=inset_position[0])
    viewer.update_float("inset_y", value=inset_position[1])
    viewer.update_float("inset_width", value=inset_position[2])
    viewer.update_float("inset_height", value=inset_position[3])

    viewer.update_selection("summary_stat", value=summary_stat)
    viewer.update_selection("slot_style", value=slot_style)
    viewer.update_boolean("show_slot_legend", value=show_slot_legend)

    viewer.update_selection("style", value=style)
    viewer.update_text("cmap", value=cmap)
    viewer.update_float("linewidth", value=linewidth)
    viewer.update_float("alpha", value=alpha)
    viewer.update_boolean("show_colorbar", value=show_colorbar)
    viewer.update_boolean("show_env_label", value=show_env_label)
    viewer.update_boolean("show_summary", value=show_summary)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig
