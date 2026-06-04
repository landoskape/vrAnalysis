import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from syd import Viewer

from vrAnalysis.helpers import sort_by_preferred_environment, vectorRSquared
from vrAnalysis.helpers.plotting import format_spines
from vrAnalysis.helpers.vrsupport import _jit_reliability_loo
from vrAnalysis.sessions import B2Session
from vrAnalysis.processors import spkmaps as SMPs
from vrAnalysis.processors.support import median_zscore
from vrAnalysis.processors.placefields import get_frame_behavior, get_placefield, get_placefield_prediction
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.spkmaps import Maps, Reliability

from dimensionality_manuscript.configs.pfpred_quality import PFPredQualityConfig, _kde_r2

plt.rcParams["font.size"] = 12

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

        plt.close("all")
        fig, ax = plt.subplots(3, 1, figsize=self.figsize, layout="constrained", sharex=True, sharey=True)
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
            a.set_xticks([0, num_frames - 1])
            a.set_xticklabels([])
            a.set_yticks([0, num_rois - 1])
            a.set_yticklabels([])

        ax[0].set_title("Deconvolved Calcium Activity")
        ax[1].set_title("Prediction From Place Field")
        ax[2].set_title("Residuals")
        ax[2].set_xlabel(f"{num_frames} Imaging Frames")
        ax[0].set_ylabel(f"{num_rois} ROIs")
        ax[1].set_ylabel(f"{num_rois} ROIs")
        ax[2].set_ylabel(f"{num_rois} ROIs")

        for spine in ["top", "right", "bottom", "left"]:
            ax[0].spines[spine].set_visible(False)
            ax[1].spines[spine].set_visible(False)
            ax[2].spines[spine].set_visible(False)

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

    def __init__(self, session: B2Session, smp: SMPs.SpkmapProcessor, idx_env: int):
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

    def recompute_arrays(self, state):
        self.idx_env = state["env"]
        self.spks_valid, self.pfpred_valid, self.r2, self.reliability = _r2_placefield_arrays(self.session, self.smp, self.idx_env)
        kde_result = _kde_r2(self.r2, self.reliability.values[self.idx_env], _PFPRED_KDE_GRID)
        self.kde_grid = kde_result["r2_kde_grid"]
        self.kde_mean = kde_result["r2_kde_mean"]

    def plot(self, state):
        roi = state["roi"]
        idx_env = self.idx_env
        spks_valid = self.spks_valid
        pfpred_valid = self.pfpred_valid
        r2 = self.r2
        reliability = self.reliability

        plt.close("all")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

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

        return fig


def stacked_raster_plot(
    session: B2Session,
    use_reliable: bool = True,
    reliability_threshold: float = 0.7,
    prediction_from: str = "spkmap",
    xslice: slice = slice(0, 2000),
    vmax: float = 6,
    figsize: tuple[float, float] = (12, 6),
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
            ax_consistency.text(-0.5, half_trial_number, r"$\sigma = \mathrm{corr}(<other\ trials>)$", ha="center", va="center", rotation=90)
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

    viewer = R2PlacefieldFocus(session, smp, idx_env)
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
