from pathlib import Path
from copy import copy
from tqdm import tqdm
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from syd import Viewer
from ..tracking import Tracker
from ..metrics import FractionActive
from ..multisession import MultiSessionSpkmaps
from ..files import analysis_path, repo_path
from ..helpers import (
    beeswarm,
    format_spines,
    save_figure,
    color_violins,
    errorPlot,
    vectorCorrelation,
    blinded_study_legend,
    get_mouse_colors,
    short_mouse_names,
)
from ..database import get_database
from .reliability_continuity import ReliabilityStabilitySummary
from ..analysis.tracked_plasticity.utils import all_combos


def figure_dir(folder: str) -> Path:
    return repo_path() / "figures" / "before_the_reveal" / folder


def make_structural_image(
    stat: list[dict],
    ops: list[list[dict]],
    plane: list[list[int]],
    image_size: float,
    red_scale: float,
    crop_roi: bool,
    vmax: float,
    flip_max: bool = False,
):
    """Build a structural image for a tracked ROI across sessions.

    The structural image is a concatenation of the ROI mask across sessions (top row) and
    the red / green composite of the structural image (bottom row). There are options for
    how to crop and scale the features of the image.
    """
    xpix = [s["xpix"] for s in stat]
    ypix = [s["ypix"] for s in stat]
    lam = [s["lam"] for s in stat]
    ref_image = [ops[pnum]["meanImg"] for pnum, ops in zip(plane, ops)]
    imch2 = [ops[pnum]["meanImg_chan2"] for pnum, ops in zip(plane, ops)]
    xcenter = [int(np.mean(xp)) for xp in xpix]
    ycenter = [int(np.mean(yp)) for yp in ypix]
    xrange = [np.max(xp) - np.min(xp) for xp in xpix]
    yrange = [np.max(yp) - np.min(yp) for yp in ypix]
    image_size = int(np.ceil(max(np.max(xrange), np.max(yrange)) * image_size))
    mask_images = [np.zeros_like(ref_image[0]) for _ in range(len(xpix))]
    red_images = [None for _ in range(len(xpix))]
    for i, (xp, yp, lp) in enumerate(zip(xpix, ypix, lam)):
        mask_images[i][yp, xp] = lp
        red_images[i] = (1 - red_scale) * np.array(np.tile(ref_image[i][:, :, None], (1, 1, 3)), dtype=float) / np.max(ref_image[i])
        red_images[i][:, :, 0] += red_scale * np.array(imch2[i], dtype=float) / np.max(imch2[i])

    if crop_roi:
        for i, (xc, yc) in enumerate(zip(xcenter, ycenter)):
            x_slice = slice(xc - image_size // 2, xc + image_size // 2)
            y_slice = slice(yc - image_size // 2, yc + image_size // 2)
            mask_images[i] = mask_images[i][y_slice][:, x_slice]
            red_images[i] = red_images[i][y_slice][:, x_slice]

    mask_images = np.concatenate(mask_images, axis=1)
    mask_images = mask_images / np.max(mask_images)
    if flip_max:
        mask_images = 1 - mask_images
    red_images = np.concatenate(red_images, axis=1)
    red_images = red_images / np.max(red_images) / vmax  # scale the structural image to the max value

    structural = np.concatenate([np.tile(mask_images[:, :, None], (1, 1, 3)), red_images], axis=0)
    structural = np.clip(structural, 0, 1)
    return structural


class TrackedSpkmapViewer(Viewer):
    def __init__(self, tracked_mice: list[str]):
        self.tracked_mice = list(tracked_mice)
        self.multisessions = {mouse: None for mouse in self.tracked_mice}

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_selection("environment", value=1, options=[1])
        self.add_multiple_selection("sessions", value=[0], options=[0])
        self.add_selection("reference_session", value=0, options=[0])
        self.add_float_range("reliability_range", value=(0.6, 1.0), min=-1.0, max=1.0)
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "mse", "correlation"])
        self.add_selection("smooth_width", value=5, options=[1, 5])
        self.add_float_range("fraction_range", value=(0.0, 1.0), min=0.0, max=1.0)
        self.add_selection("activity_method", value="rms", options=FractionActive.activity_methods)
        self.add_selection("fraction_method", value="participation", options=FractionActive.fraction_methods)
        self.add_boolean("red_only", value=False)
        self.add_selection("spks_type", value="significant", options=["significant", "oasis"])
        self.add_boolean("use_session_filters", value=True)
        self.add_integer("roi_idx", value=0, min=0, max=100)
        self.add_float("vmax_spkmap", value=1.0, min=0.1, max=15.0)
        self.add_float("vmax_reference", value=1.0, min=0.01, max=2.0)
        self.add_float("image_size", value=1.5, min=1.0, max=5.0)
        self.add_float("red_scale", value=0.5, min=0.0, max=1.0)
        self.add_boolean("crop_roi", value=True)

        # Set up callbacks
        self.on_change("mouse", self.reset_mouse)
        self.on_change("environment", self.reset_environment)
        self.on_change("sessions", self.update_sessions)
        self.on_change(
            [
                "sessions",
                "reference_session",
                "reliability_range",
                "reliability_method",
                "smooth_width",
                "fraction_range",
                "activity_method",
                "fraction_method",
                "red_only",
                "spks_type",
                "use_session_filters",
            ],
            self.reset_roi_options,
        )
        self.reset_mouse(self.state)

    def reset_mouse(self, state):
        msm = self.get_multisession(state["mouse"])
        environments = self.get_environments(state["mouse"])
        start_envnum = msm.env_selector(envmethod="first")
        self.update_selection("environment", value=start_envnum, options=list(environments))
        self.reset_environment(self.state)

    def reset_environment(self, state):
        msm = self.get_multisession(state["mouse"])
        idx_ses = msm.idx_ses_with_env(state["environment"])
        num_sessions = len(idx_ses)
        self.update_multiple_selection("sessions", value=idx_ses[: min(3, num_sessions)], options=list(idx_ses))
        self.update_sessions(self.state)
        self.reset_roi_options(self.state)

    def update_sessions(self, state):
        idx_ses = state["sessions"]
        self.update_selection("reference_session", options=list(idx_ses))

    def reset_roi_options(self, state):
        msm = self.get_multisession(state["mouse"])
        spkmaps, extras = msm.get_spkmaps(
            state["environment"],
            average=False,
            reliability_method=state["reliability_method"],
            smooth=float(state["smooth_width"]),
            spks_type=state["spks_type"],
            idx_ses=state["sessions"],
            tracked=True,
            use_session_filters=state["use_session_filters"],
            pop_nan=False,
        )
        fraction_active = [
            FractionActive.compute(
                spkmap,
                activity_axis=2,
                fraction_axis=1,
                activity_method=state["activity_method"],
                fraction_method=state["fraction_method"],
            )
            for spkmap in spkmaps
        ]
        idx_to_reference = state["sessions"].index(state["reference_session"])
        reference_fraction_active = fraction_active[idx_to_reference]
        reference_reliability = extras["reliability"][idx_to_reference]
        reference_red_idx = extras["idx_red"][idx_to_reference]
        idx_reliable_keeps = (reference_reliability >= state["reliability_range"][0]) & (reference_reliability <= state["reliability_range"][1])
        idx_active_keeps = (reference_fraction_active >= state["fraction_range"][0]) & (reference_fraction_active <= state["fraction_range"][1])
        idx_red_keeps = reference_red_idx if state["red_only"] else ~reference_red_idx
        idx_keeps = np.where(idx_reliable_keeps & idx_active_keeps & idx_red_keeps)[0]
        self.update_integer("roi_idx", max=len(idx_keeps) - 1)

        idx_original_red = [msm.processors[isession].session.loadone("mpciROIs.redCellIdx") for isession in state["sessions"]]
        idx_original_red = [ired[it] for ired, it in zip(idx_original_red, extras["idx_tracked"])]

        idx_red_assignment = [msm.processors[isession].session.loadone("mpciROIs.redCellManualAssignments") for isession in state["sessions"]]
        idx_red_man_label = [ired[0] for ired in idx_red_assignment]
        idx_red_man_active = [ired[1] for ired in idx_red_assignment]
        idx_red_man_label = [ired[it] for ired, it in zip(idx_red_man_label, extras["idx_tracked"])]
        idx_red_man_active = [ired[it] for ired, it in zip(idx_red_man_active, extras["idx_tracked"])]

        self._spkmaps = spkmaps
        self._reliability = extras["reliability"]
        self._idx_tracked = extras["idx_tracked"]
        self._idx_red = extras["idx_red"]
        self._idx_original_red = idx_original_red
        self._idx_red_man_label = idx_red_man_label
        self._idx_red_man_active = idx_red_man_active
        self._fraction_active = fraction_active
        self._idx_keeps = idx_keeps
        self._idx_to_reference = idx_to_reference
        self._sample_silhouettes = extras["sample_silhouettes"]
        self._cluster_silhouettes = extras["cluster_silhouettes"]
        self._cluster_ids = extras["cluster_ids"]

        # Get roimask info
        self._stat = [
            msm.processors[isession].session.load_s2p("stat")[itracked] for isession, itracked in zip(state["sessions"], extras["idx_tracked"])
        ]
        self._ops = [msm.processors[isession].session.load_s2p("ops") for isession in state["sessions"]]
        self._planes = [
            msm.processors[isession].session.get_plane_idx()[itracked] for isession, itracked in zip(state["sessions"], extras["idx_tracked"])
        ]

    def get_multisession(self, mouse: str) -> MultiSessionSpkmaps:
        if self.multisessions[mouse] is None:
            tracker = Tracker(mouse)
            self.multisessions[mouse] = MultiSessionSpkmaps(tracker)
        return self.multisessions[mouse]

    def get_environments(self, mouse: str) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        msm = self.get_multisession(mouse)
        environments = np.unique(np.concatenate([session.environments for session in msm.tracker.sessions]))
        return environments

    def _fraction_active_name(self, activity_method: str, fraction_method: str) -> str:
        return "_".join([activity_method, fraction_method])

    def _make_roi_trajectory(self, spkmaps, roi_idx):
        dead_trials = 1
        roi_activity = [s[roi_idx] for s in spkmaps]
        dead_space = [np.full((dead_trials, roi_activity[0].shape[1]), np.nan) for _ in range(len(roi_activity) - 1)]
        dead_space.append(None)
        interleaved = [item for pair in zip(roi_activity, dead_space) for item in pair if item is not None]

        trial_env = [ises * np.ones(r.shape[0]) for ises, r in enumerate(roi_activity)]
        dead_trial_env = [np.nan * np.ones(dead_trials) for _ in range(len(roi_activity) - 1)]
        dead_trial_env.append(None)
        env_trialnum = [item for pair in zip(trial_env, dead_trial_env) for item in pair if item is not None]
        return np.concatenate(interleaved, axis=0), np.concatenate(env_trialnum)

    def plot(self, state):
        msm = self.get_multisession(state["mouse"])
        spkmaps = self._spkmaps
        reliability = self._reliability
        idx_tracked = self._idx_tracked
        idx_red = self._idx_red
        idx_original_red = self._idx_original_red
        idx_red_man_label = self._idx_red_man_label
        idx_red_man_active = self._idx_red_man_active
        fraction_active = self._fraction_active
        idx_keeps = self._idx_keeps
        idx_to_reference = self._idx_to_reference
        sample_silhouettes = self._sample_silhouettes
        cluster_silhouettes = self._cluster_silhouettes
        roi_idx = state["roi_idx"]

        roi_idx_tracked = idx_tracked[:, idx_keeps[roi_idx]]
        roi_spkmaps, snums = self._make_roi_trajectory(spkmaps, idx_keeps[roi_idx])
        roi_session_highlight = np.full((roi_spkmaps.shape[0], 20), 0.0)
        roi_session_highlight[snums == idx_to_reference] = np.nan
        roi_spkmaps = np.concatenate([roi_session_highlight, roi_spkmaps, roi_session_highlight], axis=1)
        roi_reliability = [r[idx_keeps[roi_idx]] for r in reliability]
        roi_idx_red = [ired[idx_keeps[roi_idx]] for ired in idx_red]
        roi_idx_original_red = [ired[idx_keeps[roi_idx]] for ired in idx_original_red]
        roi_idx_red_man_label = [ired[idx_keeps[roi_idx]] for ired in idx_red_man_label]
        roi_idx_red_man_active = [ired[idx_keeps[roi_idx]] for ired in idx_red_man_active]
        roi_fraction_active = [fa[idx_keeps[roi_idx]] for fa in fraction_active]
        roi_sample_silhouettes = sample_silhouettes[:, idx_keeps[roi_idx]]
        roi_cluster_silhouette = cluster_silhouettes[idx_keeps[roi_idx]]

        # Get roimask info
        """
        meanImg(512, 512)
        meanImg_chan2(512, 512)
        refImg(512, 512)
        meanImgE(512, 512)
        Vcorr(478, 442)
        meanImg_chan2_corrected(512, 512)
        """
        xpix = [stat[idx_keeps[roi_idx]]["xpix"] for stat in self._stat]
        ypix = [stat[idx_keeps[roi_idx]]["ypix"] for stat in self._stat]
        lam = [stat[idx_keeps[roi_idx]]["lam"] for stat in self._stat]
        plane = [planes[idx_keeps[roi_idx]] for planes in self._planes]
        ref_image = [ops[pnum]["meanImg"] for pnum, ops in zip(plane, self._ops)]
        imch2 = [ops[pnum]["meanImg_chan2"] for pnum, ops in zip(plane, self._ops)]
        xcenter = [int(np.mean(xp)) for xp in xpix]
        ycenter = [int(np.mean(yp)) for yp in ypix]
        xrange = [np.max(xp) - np.min(xp) for xp in xpix]
        yrange = [np.max(yp) - np.min(yp) for yp in ypix]
        image_size = int(np.ceil(max(np.max(xrange), np.max(yrange)) * state["image_size"]))
        mask_images = [np.zeros_like(ref_image[0]) for _ in range(len(xpix))]
        red_images = [None for _ in range(len(xpix))]
        for i, (xp, yp, lp) in enumerate(zip(xpix, ypix, lam)):
            mask_images[i][yp, xp] = lp
            red_images[i] = (1 - state["red_scale"]) * np.array(np.tile(ref_image[i][:, :, None], (1, 1, 3)), dtype=float) / np.max(ref_image[i])
            red_images[i][:, :, 0] += state["red_scale"] * np.array(imch2[i], dtype=float) / np.max(imch2[i])
        if state["crop_roi"]:
            for i, (xc, yc) in enumerate(zip(xcenter, ycenter)):
                x_slice = slice(xc - image_size // 2, xc + image_size // 2)
                y_slice = slice(yc - image_size // 2, yc + image_size // 2)
                mask_images[i] = mask_images[i][y_slice][:, x_slice]
                red_images[i] = red_images[i][y_slice][:, x_slice]

        mask_images = np.concatenate(mask_images, axis=1)
        mask_images = mask_images / np.max(mask_images)
        red_images = np.concatenate(red_images, axis=1)

        structural = np.concatenate([np.tile(mask_images[:, :, None], (1, 1, 3)), red_images], axis=0)

        fig = plt.figure(figsize=(9, 7), layout="constrained")
        gs = fig.add_gridspec(4, 3)
        ax_spkmaps = fig.add_subplot(gs[:, 0])
        ax_stats = fig.add_subplot(gs[0:2, 1])
        ax_roi_stats = fig.add_subplot(gs[0:2, 2])
        ax_roi_structural = fig.add_subplot(gs[2:, 1:])

        spkmap_cmap = mpl.colormaps["gray_r"]
        spkmap_cmap.set_bad(("orange", 0.3))
        ax_spkmaps.imshow(roi_spkmaps, aspect="auto", cmap=spkmap_cmap, interpolation="none", vmin=0, vmax=state["vmax_spkmap"])
        ax_spkmaps.set_title("Place Field Activity")

        ax_stats.scatter(reliability[idx_to_reference], fraction_active[idx_to_reference], s=8, color="black", alpha=0.1)
        ax_stats.scatter(reliability[idx_to_reference][idx_keeps], fraction_active[idx_to_reference][idx_keeps], s=8, color="red", alpha=0.3)
        ax_stats.scatter(roi_reliability[idx_to_reference], roi_fraction_active[idx_to_reference], s=8, color="blue", alpha=1.0)
        ax_stats.set_xlabel("Reliability")
        ax_stats.set_ylabel("Fraction Active")

        ax_roi_stats.plot(state["sessions"], roi_reliability, color="k", label="Reliability")
        ax_roi_stats.plot(state["sessions"], roi_fraction_active, color="b", label="Fraction Active")
        ax_roi_stats.set_xlabel("Session")
        ax_roi_stats.set_ylabel("Rel / FracAct")
        ax_roi_stats.legend(loc="best", fontsize=8)

        title = "ROI Masks\nReference+Red Channel\n"
        title += f"Cluster Silhouette: {roi_cluster_silhouette:.2f}\n"
        title += "Original Red Assignment:" + "".join(["R" if ired else "X" for ired in roi_idx_original_red]) + "\n"
        title += (
            "Manual Red Assignment:"
            + "".join(["?" if not iactive else "R" if ired else "X" for ired, iactive in zip(roi_idx_red_man_label, roi_idx_red_man_active)])
            + "\n"
        )
        title += "Sample Silhouettes:\n" + ", ".join([f"{s:.2f}" for s in roi_sample_silhouettes]) + "\n"
        structural = np.clip(structural / state["vmax_reference"], 0, 1)
        ax_roi_structural.imshow(structural, aspect="equal", cmap="gray_r", interpolation="none")
        ax_roi_structural.set_title(title)
        ax_roi_structural.set_axis_off()

        return fig


class TrackingStatsFigureMaker(Viewer):
    def __init__(self, tracked_mice: list[str], try_cache: bool = True, save_cache: bool = False):
        self.tracked_mice = list(tracked_mice)
        self.multisessions = {mouse: None for mouse in self.tracked_mice}
        self.mousedb = get_database("vrMice")
        self.ko = dict(zip(self.mousedb.get_table()["mouseName"], self.mousedb.get_table()["KO"]))

        # These are the overall tracking stats and never change by the GUI parameters
        self.gather_tracking_stats(try_cache=try_cache, save_cache=save_cache)

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_selection("environment", value=1, options=[1])
        self.add_integer_range("sessions", value=(0, 1), min=0, max=1)
        self.add_float_range("reliability_range", value=(0.6, 1.0), min=-1.0, max=1.0)
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "mse", "correlation"])
        self.add_selection("smooth_width", value=5, options=[1, 5])
        self.add_float_range("fraction_range", value=(0.1, 1.0), min=0.0, max=1.0)
        self.add_selection("activity_method", value="rms", options=FractionActive.activity_methods)
        self.add_selection("fraction_method", value="participation", options=FractionActive.fraction_methods)
        self.add_selection("spks_type", value="significant", options=["significant", "oasis"])
        self.add_boolean("use_session_filters", value=True)
        self.add_integer("ctl_roi_idx", value=0, min=0, max=100)
        self.add_integer("red_roi_idx", value=0, min=0, max=100)
        self.add_float("vmax_reference", value=1.0, min=0.01, max=2.0)
        self.add_float("image_size", value=1.5, min=1.0, max=5.0)
        self.add_float("red_scale", value=0.5, min=0.0, max=1.0)
        self.add_boolean("crop_roi", value=True)
        self.add_boolean("flip_max", value=False)
        self.add_boolean("add_mouse_ticks", value=False)
        self.add_boolean("blinded", value=True)
        self.add_button("save_example", label="Save Example", callback=self.save_example)

        # Set up callbacks
        self.on_change("mouse", self.reset_mouse)
        self.on_change("environment", self.reset_environment)
        self.on_change(
            [
                "sessions",
                "reliability_range",
                "reliability_method",
                "smooth_width",
                "fraction_range",
                "activity_method",
                "fraction_method",
                "spks_type",
                "use_session_filters",
            ],
            self.reset_roi_options,
        )
        self.reset_mouse(self.state)

    def save_example(self, state):
        fig = self.plot(state)
        fig_dir = figure_dir("tracking_example_and_stats")
        mouse = state["mouse"]
        environment = state["environment"]
        sessions = state["sessions"]
        idx_tracked = self._idx_tracked
        idx_ctl_keeps = self._idx_ctl_keeps
        idx_red_keeps = self._idx_red_keeps
        ctl_roi_idx = state["ctl_roi_idx"]
        red_roi_idx = state["red_roi_idx"]
        true_ctl_idx = idx_tracked[0, idx_ctl_keeps[ctl_roi_idx]]
        true_red_idx = idx_tracked[0, idx_red_keeps[red_roi_idx]]
        fig_name = f"{mouse}_env{environment}_ses{sessions[0]}_ses{sessions[1]}_ctl{true_ctl_idx}_red{true_red_idx}"
        if not state["blinded"]:
            fig_name += "_unblinded"
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_dir / fig_name)

    def gather_tracking_stats(self, try_cache: bool = True, save_cache: bool = False):
        cache_path = analysis_path() / "before_the_reveal_temp_data" / "tracking_stats.joblib"
        if try_cache and cache_path.exists():
            with open(cache_path, "rb") as f:
                tracking_stats = joblib.load(f)
                self.num_clusters_per_session = tracking_stats["num_clusters_per_session"]
                self.num_tracked_red_per_session = tracking_stats["num_tracked_red_per_session"]
            return

        num_clusters_per_session = {}
        num_tracked_red_per_session = {}
        for mouse in tqdm(self.tracked_mice):
            # Get mouse tracker object
            tracker = self.get_multisession(mouse).tracker

            # Go through clusters and count how many sessions each cluster is present in
            mouse_num_sessions = len(tracker.sessions)
            c_num_clusters_per_session = np.zeros(mouse_num_sessions)
            c_num_clusters = tracker.cluster_silhouettes.shape[0]
            cluster_counts = np.zeros(c_num_clusters)
            for labels in tracker.labels:
                c_session_clusters = labels[labels >= 0]
                cluster_counts[c_session_clusters] += 1
            for ccount in cluster_counts:
                if ccount > 0:
                    c_num_clusters_per_session[int(ccount) - 1] += 1
            num_clusters_per_session[mouse] = c_num_clusters_per_session

            # Go through sessions and count how many ROIs are tracked and red
            c_num_tracked_red_per_session = np.zeros(mouse_num_sessions)
            for isession, (session, labels) in enumerate(zip(tracker.sessions, tracker.labels)):
                idx_rois = session.idx_rois  # idx to good ROIs
                idx_red = session.get_red_idx()
                idx_tracked = np.zeros(len(idx_rois), dtype=bool)
                idx_tracked[labels >= 0] = True
                good_tracked_red = np.sum(idx_rois & idx_tracked & idx_red)
                c_num_tracked_red_per_session[isession] = good_tracked_red
            num_tracked_red_per_session[mouse] = c_num_tracked_red_per_session

        self.num_clusters_per_session = num_clusters_per_session
        self.num_tracked_red_per_session = num_tracked_red_per_session

        if save_cache:
            with open(cache_path, "wb") as f:
                tracking_stats = dict(
                    num_clusters_per_session=self.num_clusters_per_session, num_tracked_red_per_session=self.num_tracked_red_per_session
                )
                joblib.dump(tracking_stats, f)

    def reset_mouse(self, state):
        msm = self.get_multisession(state["mouse"])
        environments = self.get_environments(state["mouse"])
        start_envnum = msm.env_selector(envmethod="first")
        self.update_selection("environment", value=start_envnum, options=list(environments))
        self.reset_environment(self.state)

    def reset_environment(self, state):
        msm = self.get_multisession(state["mouse"])
        idx_ses = msm.idx_ses_with_env(state["environment"])
        num_sessions = len(idx_ses)
        self.update_integer_range("sessions", value=(0, min(6, num_sessions)), min=0, max=num_sessions)
        self.reset_roi_options(self.state)

    def reset_roi_options(self, state):
        msm = self.get_multisession(state["mouse"])
        idx_ses = msm.idx_ses_with_env(state["environment"])[state["sessions"][0] : state["sessions"][1]]
        spkmaps, extras = msm.get_spkmaps(
            state["environment"],
            average=False,
            reliability_method=state["reliability_method"],
            smooth=float(state["smooth_width"]),
            spks_type=state["spks_type"],
            idx_ses=idx_ses,
            tracked=True,
            use_session_filters=state["use_session_filters"],
            pop_nan=False,
        )
        fraction_active = [
            FractionActive.compute(
                spkmap,
                activity_axis=2,
                fraction_axis=1,
                activity_method=state["activity_method"],
                fraction_method=state["fraction_method"],
            )
            for spkmap in spkmaps
        ]
        idx_reliable_keeps = np.all(
            np.stack([(rel >= state["reliability_range"][0]) & (rel <= state["reliability_range"][1]) for rel in extras["reliability"]], axis=0),
            axis=0,
        )
        idx_active_keeps = np.all(
            np.stack([(fa >= state["fraction_range"][0]) & (fa <= state["fraction_range"][1]) for fa in fraction_active], axis=0),
            axis=0,
        )
        idx_red = np.any(np.stack(extras["idx_red"], axis=0), axis=0)
        idx_ctl_keeps = np.where(idx_reliable_keeps & idx_active_keeps & ~idx_red)[0]
        idx_red_keeps = np.where(idx_reliable_keeps & idx_active_keeps & idx_red)[0]
        self.update_integer("ctl_roi_idx", max=len(idx_ctl_keeps) - 1)
        self.update_integer("red_roi_idx", max=len(idx_red_keeps) - 1)

        self._spkmaps = spkmaps
        self._reliability = extras["reliability"]
        self._idx_tracked = extras["idx_tracked"]
        self._idx_red = extras["idx_red"]
        self._fraction_active = fraction_active
        self._idx_ctl_keeps = idx_ctl_keeps
        self._idx_red_keeps = idx_red_keeps

        # Get roimask info
        self._stat = [msm.processors[isession].session.load_s2p("stat")[itracked] for isession, itracked in zip(idx_ses, extras["idx_tracked"])]
        self._ops = [msm.processors[isession].session.load_s2p("ops") for isession in idx_ses]
        self._planes = [msm.processors[isession].session.get_plane_idx()[itracked] for isession, itracked in zip(idx_ses, extras["idx_tracked"])]

    def get_multisession(self, mouse: str) -> MultiSessionSpkmaps:
        if self.multisessions[mouse] is None:
            tracker = Tracker(mouse)
            self.multisessions[mouse] = MultiSessionSpkmaps(tracker)
        return self.multisessions[mouse]

    def get_environments(self, mouse: str) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        msm = self.get_multisession(mouse)
        environments = np.unique(np.concatenate([session.environments for session in msm.tracker.sessions]))
        return environments

    def _fraction_active_name(self, activity_method: str, fraction_method: str) -> str:
        return "_".join([activity_method, fraction_method])

    def _make_roi_trajectory(self, spkmaps, roi_idx):
        dead_trials = 1
        roi_activity = [s[roi_idx] for s in spkmaps]
        dead_space = [np.full((dead_trials, roi_activity[0].shape[1]), np.nan) for _ in range(len(roi_activity) - 1)]
        dead_space.append(None)
        interleaved = [item for pair in zip(roi_activity, dead_space) for item in pair if item is not None]

        trial_env = [ises * np.ones(r.shape[0]) for ises, r in enumerate(roi_activity)]
        dead_trial_env = [np.nan * np.ones(dead_trials) for _ in range(len(roi_activity) - 1)]
        dead_trial_env.append(None)
        env_trialnum = [item for pair in zip(trial_env, dead_trial_env) for item in pair if item is not None]
        return np.concatenate(interleaved, axis=0), np.concatenate(env_trialnum)

    def plot(self, state):
        use_mice = [mouse for mouse in self.tracked_mice if mouse not in ["ATL045"]]
        # Always group blinded first
        use_mice = [mouse for mouse in use_mice if self.ko[mouse]] + [mouse for mouse in use_mice if not self.ko[mouse]]
        use_tracked_counts = [self.num_clusters_per_session[mouse] for mouse in use_mice]
        use_tracked_red_counts = [self.num_tracked_red_per_session[mouse] for mouse in use_mice]

        spkmaps = self._spkmaps
        idx_ctl_keeps = self._idx_ctl_keeps
        idx_red_keeps = self._idx_red_keeps
        ctl_roi_idx = state["ctl_roi_idx"]
        red_roi_idx = state["red_roi_idx"]

        ctl_placefields = np.stack([np.nanmean(s[idx_ctl_keeps[ctl_roi_idx]], axis=0) for s in spkmaps])
        red_placefields = np.stack([np.nanmean(s[idx_red_keeps[red_roi_idx]], axis=0) for s in spkmaps])

        # Get roimask info
        ctl_structural = make_structural_image(
            [stat[idx_ctl_keeps[ctl_roi_idx]] for stat in self._stat],
            self._ops,
            [planes[idx_ctl_keeps[ctl_roi_idx]] for planes in self._planes],
            state["image_size"],
            state["red_scale"],
            state["crop_roi"],
            state["vmax_reference"],
            state["flip_max"],
        )
        red_structural = make_structural_image(
            [stat[idx_red_keeps[red_roi_idx]] for stat in self._stat],
            self._ops,
            [planes[idx_red_keeps[red_roi_idx]] for planes in self._planes],
            state["image_size"],
            state["red_scale"],
            state["crop_roi"],
            state["vmax_reference"],
            state["flip_max"],
        )

        fig = plt.figure(figsize=(8, 6), layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
        gs_examples = gs[0].subgridspec(2, 2, width_ratios=[1, 2])
        gs_tracking = gs[1].subgridspec(1, 2)
        ax_ctl_pf = fig.add_subplot(gs_examples[0, 0])
        ax_red_pf = fig.add_subplot(gs_examples[1, 0])
        ax_ctl_structural = fig.add_subplot(gs_examples[0, 1])
        ax_red_structural = fig.add_subplot(gs_examples[1, 1])
        ax_tracking_stats = fig.add_subplot(gs_tracking[0])
        ax_red_counts = fig.add_subplot(gs_tracking[1])

        ax_ctl_pf.plot(range(ctl_placefields.shape[1]), ctl_placefields.T, color="k")
        ax_red_pf.plot(range(red_placefields.shape[1]), red_placefields.T, color="r")
        ax_ctl_structural.imshow(ctl_structural, aspect="equal", interpolation="none")
        ax_red_structural.imshow(red_structural, aspect="equal", interpolation="none")
        ax_ctl_structural.set_xticks([])
        ax_red_structural.set_xticks([])
        ax_ctl_structural.set_yticks([])
        ax_red_structural.set_yticks([])
        for spine in ["left", "right", "top", "bottom"]:
            ax_ctl_structural.spines[spine].set_visible(False)
            ax_red_structural.spines[spine].set_visible(False)

        ax_ctl_pf.set_xlabel("Position (cm)")
        ax_red_pf.set_xlabel("Position (cm)")
        ax_ctl_pf.set_ylabel("Activity")
        ax_red_pf.set_ylabel("Activity")

        ax_ctl_structural.set_ylabel("Structural | Masks")
        ax_red_structural.set_ylabel("Structural | Masks")

        format_spines(
            ax_ctl_pf,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, ctl_placefields.shape[1]),
            ybounds=ax_ctl_pf.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        format_spines(
            ax_red_pf,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, red_placefields.shape[1]),
            ybounds=ax_red_pf.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        colors, linewidths, zorder = get_mouse_colors(use_mice, blinded=state["blinded"], asdict=True, mousedb=self.mousedb)

        ax_blinded_red_counts = ax_red_counts.inset_axes([0.5, 0.5, 0.4, 0.4])
        max_blinded = 0
        for i, mouse in enumerate(use_mice):
            c_num_clusters = use_tracked_counts[i]
            c_num_tracked_red = use_tracked_red_counts[i]
            num_sessions = len(c_num_clusters)
            ax_tracking_stats.plot(
                range(2, num_sessions + 1),
                c_num_clusters[1:],
                color=colors[mouse],
                linewidth=linewidths[mouse],
                zorder=zorder[mouse],
            )
            ax_red_counts.scatter(
                i + 0.25 * beeswarm(c_num_tracked_red, nbins=int(np.ceil(len(c_num_tracked_red) / 3))),
                c_num_tracked_red,
                color=colors[mouse],
                s=10,
                alpha=0.75,
            )
            ax_blinded_red_counts.scatter(
                i + 0.25 * beeswarm(c_num_tracked_red, nbins=int(np.ceil(len(c_num_tracked_red) / 3))),
                c_num_tracked_red,
                color=colors[mouse],
                s=10,
                alpha=0.75,
            )
            if "CR_" not in mouse:
                max_blinded = max(max_blinded, np.max(c_num_tracked_red))

        ax_blinded_red_counts.set_xlim(1.5, len(use_mice) - 0.5)
        ax_blinded_red_counts.set_ylim(0, max_blinded * 1.1)
        ax_blinded_red_counts.set_title("Blinded Mice", fontsize=9)

        xlim = ax_tracking_stats.get_xlim()
        ylim = ax_tracking_stats.get_ylim()
        ax_tracking_stats.set_xlim(1, xlim[1])
        ax_tracking_stats.set_ylim(0, ylim[1])
        ax_tracking_stats.set_xlabel("# Sessions Tracked")
        ax_tracking_stats.set_ylabel("# Clusters")
        blinded_study_legend(
            ax_tracking_stats,
            xpos=xlim[1],
            ypos=ylim[1] * 0.9,
            pilot_colors=[colors[mouse] for mouse in ["CR_Hippocannula6", "CR_Hippocannula7"]],
            blinded_colors=[colors[mouse] for mouse in use_mice if mouse not in ["CR_Hippocannula6", "CR_Hippocannula7"]],
            blinded=state["blinded"],
            origin="upper_right",
        )
        format_spines(
            ax_tracking_stats,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(1, xlim[1]),
            ybounds=(0, ylim[1]),
            spines_visible=["left", "bottom"],
            tick_length=4,
        )

        ax_red_counts.set_xlim(-0.5, len(use_mice) - 0.5)
        ax_red_counts.set_xlabel("Mouse")
        ax_red_counts.set_ylabel("# Red ROIs Tracked")
        ax_red_counts.set_xticks(range(len(use_mice)))

        format_spines(
            ax_red_counts,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(-0.5, len(use_mice) - 0.5),
            ybounds=ax_red_counts.get_ylim(),
            spines_visible=["left", "bottom"],
            tick_length=4,
        )
        format_spines(
            ax_blinded_red_counts,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(1.5, len(use_mice) - 0.5),
            ybounds=(0, max_blinded * 1.1),
            xticks=[],
            spines_visible=["left"],
            tick_length=4,
        )

        if state["add_mouse_ticks"]:
            ax_red_counts.set_xticks(range(len(use_mice)), labels=short_mouse_names(use_mice), rotation=45, ha="right")

        return fig


class ConsistentReliabilityFigureMaker(Viewer):
    def __init__(self, tracked_mice: list[str], try_cache: bool = True, save_cache: bool = False):
        self.tracked_mice = list(tracked_mice)
        self.multisessions = {mouse: None for mouse in self.tracked_mice}

        self.mousedb = get_database("vrMice")
        self.ko = dict(zip(self.mousedb.get_table()["mouseName"], self.mousedb.get_table()["KO"]))

        # This is a different viewer with data handling capabilities for reliability summary data
        self.summary_viewer = ReliabilityStabilitySummary(self.tracked_mice)

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_integer("environment", value=1, min=0, max=3)
        self.add_integer_range("sessions", value=(0, 1), min=0, max=1)
        self.add_integer("reference_session", value=1, min=0, max=1)
        self.add_selection("reliability_threshold", value=0.5, options=[0.3, 0.5, 0.7, 0.9])
        self.add_float_range("reliability_range", value=(0.5, 1.0), min=-1.0, max=1.0)
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out"])
        self.add_selection("smooth_width", value=5, options=[5])
        self.add_float_range("fraction_range", value=(0.1, 1.0), min=0.0, max=1.0)
        self.add_selection("activity_method", value="rms", options=FractionActive.activity_methods)
        self.add_selection("fraction_method", value="participation", options=FractionActive.fraction_methods)
        self.add_selection("spks_type", value="significant", options=["significant", "oasis"])
        self.add_boolean("use_session_filters", value=True)
        self.add_integer("ctl_roi_idx", value=5, min=0, max=100)
        self.add_integer("red_roi_idx", value=7, min=0, max=100)
        self.add_float("vmax_spkmap", value=7.0, min=0.1, max=15.0)
        self.add_boolean("continuous", value=False)
        self.add_selection("forward_backward", value="both", options=["forward", "backward", "both"])
        self.add_boolean("group_novel", value=True)
        self.add_boolean("blinded", value=True)
        self.add_boolean("indicate_example", value=False)
        self.add_button("save_example", label="Save Example", callback=self.save_example)

        # Set up callbacks
        self.on_change("mouse", self.reset_mouse)
        self.on_change("environment", self.reset_environment)
        self.on_change(
            [
                "sessions",
                "reference_session",
                "reliability_range",
                "reliability_method",
                "smooth_width",
                "fraction_range",
                "activity_method",
                "fraction_method",
                "spks_type",
                "use_session_filters",
            ],
            self.reset_roi_options,
        )
        self.on_change("sessions", self.reset_sessions)
        self.reset_mouse(self.state)

    def save_example(self, state):
        fig = self.plot(state)
        fig_dir = figure_dir("consistent_reliability")
        mouse = state["mouse"]
        environment = state["environment"]
        sessions = state["sessions"]
        idx_tracked = self._idx_tracked
        idx_ctl_keeps = self._idx_ctl_keeps
        idx_red_keeps = self._idx_red_keeps
        ctl_roi_idx = state["ctl_roi_idx"]
        red_roi_idx = state["red_roi_idx"]
        true_ctl_idx = idx_tracked[0, idx_ctl_keeps[ctl_roi_idx]]
        true_red_idx = idx_tracked[0, idx_red_keeps[red_roi_idx]]
        fig_name = f"{mouse}_env{environment}_ses{sessions[0]}_ses{sessions[1]}_ctl{true_ctl_idx}_red{true_red_idx}"
        if not state["blinded"]:
            fig_name += "_unblinded"
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_dir / fig_name)
        plt.close(fig)

    def reset_mouse(self, state):
        environments = self.get_environments(state["mouse"])
        num_envs = np.sum(environments != -1)
        self.update_integer("environment", max=num_envs - 1)
        self.reset_environment(self.state)

    def reset_environment(self, state):
        msm = self.get_multisession(state["mouse"])
        envstats = msm.env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        env = env_in_order[state["environment"]]
        idx_ses = msm.idx_ses_with_env(env)
        num_sessions = len(idx_ses)
        self.update_integer_range("sessions", value=(0, min(6, num_sessions)), max=num_sessions)
        self.update_integer("reference_session", max=num_sessions - 1)
        self.reset_roi_options(self.state)

    def reset_sessions(self, state):
        sessions = state["sessions"]
        self.update_integer("reference_session", max=sessions[1] - sessions[0])

    def reset_roi_options(self, state):
        msm = self.get_multisession(state["mouse"])
        envstats = msm.env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        env = env_in_order[state["environment"]]
        idx_ses = msm.idx_ses_with_env(env)[state["sessions"][0] : state["sessions"][1]]
        spkmaps, extras = msm.get_spkmaps(
            env,
            average=False,
            reliability_method=state["reliability_method"],
            smooth=float(state["smooth_width"]),
            spks_type=state["spks_type"],
            idx_ses=idx_ses,
            tracked=True,
            use_session_filters=state["use_session_filters"],
            pop_nan=False,
        )
        fraction_active = [
            FractionActive.compute(
                spkmap,
                activity_axis=2,
                fraction_axis=1,
                activity_method=state["activity_method"],
                fraction_method=state["fraction_method"],
            )
            for spkmap in spkmaps
        ]
        reference_fraction_active = fraction_active[state["reference_session"]]
        reference_reliability = extras["reliability"][state["reference_session"]]
        idx_reliable_keeps = (reference_reliability >= state["reliability_range"][0]) & (reference_reliability <= state["reliability_range"][1])
        idx_active_keeps = (reference_fraction_active >= state["fraction_range"][0]) & (reference_fraction_active <= state["fraction_range"][1])
        idx_red = np.any(np.stack(extras["idx_red"]), axis=0)
        idx_ctl_keeps = np.where(idx_reliable_keeps & idx_active_keeps & ~idx_red)[0]
        idx_red_keeps = np.where(idx_reliable_keeps & idx_active_keeps & idx_red)[0]
        self.update_integer("ctl_roi_idx", max=len(idx_ctl_keeps) - 1)
        self.update_integer("red_roi_idx", max=len(idx_red_keeps) - 1)

        self._spkmaps = spkmaps
        self._reliability = extras["reliability"]
        self._idx_tracked = extras["idx_tracked"]
        self._idx_red = np.any(np.stack(extras["idx_red"]), axis=0)
        self._fraction_active = fraction_active
        self._idx_ctl_keeps = idx_ctl_keeps
        self._idx_red_keeps = idx_red_keeps

    def get_multisession(self, mouse: str) -> MultiSessionSpkmaps:
        if self.multisessions[mouse] is None:
            tracker = Tracker(mouse)
            self.multisessions[mouse] = MultiSessionSpkmaps(tracker)
        return self.multisessions[mouse]

    def get_environments(self, mouse: str) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        msm = self.get_multisession(mouse)
        environments = np.unique(np.concatenate([session.environments for session in msm.tracker.sessions]))
        return environments

    def _fraction_active_name(self, activity_method: str, fraction_method: str) -> str:
        return "_".join([activity_method, fraction_method])

    def _make_roi_trajectory(self, spkmaps, roi_idx, dead_trials: int = 2):
        roi_activity = [s[roi_idx] for s in spkmaps]
        dead_space = [np.full((dead_trials, roi_activity[0].shape[1]), np.nan) for _ in range(len(roi_activity) - 1)]
        dead_space.append(None)
        interleaved = [item for pair in zip(roi_activity, dead_space) for item in pair if item is not None]

        trial_env = [ises * np.ones(r.shape[0]) for ises, r in enumerate(roi_activity)]
        dead_trial_env = [np.nan * np.ones(dead_trials) for _ in range(len(roi_activity) - 1)]
        dead_trial_env.append(None)
        env_trialnum = [item for pair in zip(trial_env, dead_trial_env) for item in pair if item is not None]
        return np.concatenate(interleaved, axis=0), np.concatenate(env_trialnum)

    def _process_summary_results(self, results: dict, state: dict, max_session_diff: int = 6):
        output_keys = [
            "num_stable_ctl",
            "num_stable_red",
            "fraction_stable_ctl",
            "fraction_stable_red",
            "stable_reliability_ctl",
            "stable_reliability_red",
            "all_reliability_ctl",
            "all_reliability_red",
            "pfloc_changes_ctl",
            "pfloc_changes_red",
            "spkmap_correlations_ctl",
            "spkmap_correlations_red",
        ]

        # Prepare summary array
        max_environments = 3
        num_diffs = (2 if state["forward_backward"] == "both" else 1) * max_session_diff
        forward_start = 0 if state["forward_backward"] == "forward" else max_session_diff
        data = {key: np.full((len(results), max_environments, num_diffs), np.nan) for key in output_keys}
        for imouse, mouse in enumerate(results):
            envstats = self.get_multisession(mouse).env_stats()
            env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
            env_in_order = [env for env in env_in_order if env != -1]
            for ienv, env in enumerate(env_in_order):
                if ienv >= max_environments:
                    continue
                if results[mouse][env] is None:
                    continue
                for key in output_keys:
                    if state["forward_backward"] == "forward" or state["forward_backward"] == "both":
                        forward_data = np.nanmean(results[mouse][env]["forward"][key], axis=0)
                        data[key][imouse, ienv, forward_start:] = forward_data
                    if state["forward_backward"] == "backward" or state["forward_backward"] == "both":
                        backward_data = np.nanmean(results[mouse][env]["backward"][key], axis=0)[::-1]
                        data[key][imouse, ienv, :max_session_diff] = backward_data
        return data

    def consolidate_raw_data(self, combos, control_raw_data, red_raw_data):
        control_raw_data = [cd[-1] for cd in control_raw_data]
        red_raw_data = [rd[-1] for rd in red_raw_data]
        all_sessions = sorted(list(set(combos[0]).union(*[set(combo) for combo in combos[1:]])))
        num_diffs = len(all_sessions) - 1
        ctl_data = [[] for _ in range(num_diffs)]
        red_data = [[] for _ in range(num_diffs)]
        used = [False] * num_diffs
        for icombo in range(len(combos)):
            index_reference = all_sessions.index(combos[icombo][0])
            index_target = all_sessions.index(combos[icombo][-1])
            c_delta = abs(index_target - index_reference) - 1
            ctl_data[c_delta].append(control_raw_data[icombo])
            red_data[c_delta].append(red_raw_data[icombo])
            used[c_delta] = True
        ctl_data = [np.concatenate(data) for data, used in zip(ctl_data, used) if used]
        red_data = [np.concatenate(data) for data, used in zip(red_data, used) if used]
        deltas = [c_delta for c_delta, used in zip(range(1, num_diffs + 1), used) if used]
        return ctl_data, red_data, deltas

    def plot(self, state):
        use_mice = [mouse for mouse in self.tracked_mice if mouse not in ["ATL045"]]

        # Always group blinded first
        use_mice = [mouse for mouse in use_mice if self.ko[mouse]] + [mouse for mouse in use_mice if not self.ko[mouse]]

        max_session_diff = 6

        results = {}
        for mouse in use_mice:
            results_state = self.summary_viewer.define_state(
                mouse_name=mouse,
                envnum=1,
                reliability_threshold=state["reliability_threshold"],
                reliability_method=state["reliability_method"],
                smooth_width=int(state["smooth_width"]),
                use_session_filters=state["use_session_filters"],
                continuous=state["continuous"],
                max_session_diff=max_session_diff,
            )
            results[mouse] = self.summary_viewer.gather_data(results_state, try_cache=True)

        data = self._process_summary_results(results, state, max_session_diff=max_session_diff)

        imouse = use_mice.index(state["mouse"])
        envstats = self.get_multisession(state["mouse"]).env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        familiar_envnum = env_in_order[0]
        novel_envnums = env_in_order[1:]

        spkmaps = self._spkmaps
        reliability = self._reliability
        idx_ctl_keeps = self._idx_ctl_keeps
        idx_red_keeps = self._idx_red_keeps
        ctl_roi_idx = state["ctl_roi_idx"]
        red_roi_idx = state["red_roi_idx"]

        ctl_roi_spkmap, snums = self._make_roi_trajectory(spkmaps, idx_ctl_keeps[ctl_roi_idx], dead_trials=2)
        red_roi_spkmap, _ = self._make_roi_trajectory(spkmaps, idx_red_keeps[red_roi_idx], dead_trials=2)
        spkmap_yticks = []
        for ises in range(len(spkmaps)):
            trials_in_ses = np.where(snums == ises)[0]
            center_trial = np.mean(trials_in_ses)
            spkmap_yticks.append(center_trial)

        reliability = np.stack(reliability)
        ctl_reliability = reliability[:, idx_ctl_keeps]
        red_reliability = reliability[:, idx_red_keeps]

        ctl_example_reliability = ctl_reliability[:, ctl_roi_idx]
        red_example_reliability = red_reliability[:, red_roi_idx]

        # Get reliability data for selected summary
        ctl_reliability_summary = ctl_reliability
        red_reliability_summary = red_reliability

        # Get all combo data (all ROIs for each combo of sessions)
        combo_ctl_name = "ctl_reliable"
        combo_red_name = "red_reliable"
        if state["forward_backward"] == "forward" or state["forward_backward"] == "both":
            combos_familiar_forward = copy(results[state["mouse"]][familiar_envnum]["forward_combos"])
            combos_novel_forward = copy(results[state["mouse"]][novel_envnums[0]]["forward_combos"])
            raw_data_control_familiar_forward = copy(results[state["mouse"]][familiar_envnum]["forward_raw"][combo_ctl_name])
            raw_data_control_novel_forward = copy(results[state["mouse"]][novel_envnums[0]]["forward_raw"][combo_ctl_name])
            raw_data_red_familiar_forward = copy(results[state["mouse"]][familiar_envnum]["forward_raw"][combo_red_name])
            raw_data_red_novel_forward = copy(results[state["mouse"]][novel_envnums[0]]["forward_raw"][combo_red_name])

            ctl_data_familiar_forward, red_data_familiar_forward, deltas_familiar_forward = self.consolidate_raw_data(
                combos_familiar_forward,
                raw_data_control_familiar_forward,
                raw_data_red_familiar_forward,
            )
            ctl_data_novel_forward, red_data_novel_forward, deltas_novel_forward = self.consolidate_raw_data(
                combos_novel_forward,
                raw_data_control_novel_forward,
                raw_data_red_novel_forward,
            )

        if state["forward_backward"] == "backward" or state["forward_backward"] == "both":
            combos_familiar_backward = copy(results[state["mouse"]][familiar_envnum]["backward_combos"])
            combos_novel_backward = copy(results[state["mouse"]][novel_envnums[0]]["backward_combos"])
            raw_data_control_familiar_backward = copy(results[state["mouse"]][familiar_envnum]["backward_raw"][combo_ctl_name])
            raw_data_control_novel_backward = copy(results[state["mouse"]][novel_envnums[0]]["backward_raw"][combo_ctl_name])
            raw_data_red_familiar_backward = copy(results[state["mouse"]][familiar_envnum]["backward_raw"][combo_red_name])
            raw_data_red_novel_backward = copy(results[state["mouse"]][novel_envnums[0]]["backward_raw"][combo_red_name])

            ctl_data_familiar_backward, red_data_familiar_backward, deltas_familiar_backward = self.consolidate_raw_data(
                combos_familiar_backward,
                raw_data_control_familiar_backward,
                raw_data_red_familiar_backward,
            )
            ctl_data_novel_backward, red_data_novel_backward, deltas_novel_backward = self.consolidate_raw_data(
                combos_novel_backward,
                raw_data_control_novel_backward,
                raw_data_red_novel_backward,
            )

        # Get final summary difference data
        differences = data["all_reliability_red"] - data["all_reliability_ctl"]
        difference_xpos = (
            -1 * (np.arange(max_session_diff)[::-1] + 1)
            if state["forward_backward"] == "backward" or state["forward_backward"] == "both"
            else np.array([])
        )
        if state["forward_backward"] == "forward" or state["forward_backward"] == "both":
            difference_xpos = np.concatenate([difference_xpos, np.arange(1, max_session_diff + 1)])
        nan_spots = np.isnan(differences)[:, :2]
        more_than_one_or_none = (np.sum(~nan_spots, axis=2) > 1) | (np.sum(~nan_spots, axis=2) == 0)
        good_to_show = np.all(more_than_one_or_none, axis=1)  # & ~np.any(np.diff(1 * nan_spots, axis=2) == -1, axis=(1, 2))
        differences_familiar = differences[:, 0]
        if state["group_novel"]:
            differences_novel = np.nanmean(differences[:, 1:], axis=1)
        else:
            differences_novel = differences[:, 1]

        if not state["blinded"]:
            ko_mice = np.array([self.ko[mouse] for mouse in use_mice])
            diff_familiar_ko = np.nanmean(differences_familiar[ko_mice], axis=0)
            diff_novel_ko = np.nanmean(differences_novel[ko_mice], axis=0)
            diff_familiar_wt = np.nanmean(differences_familiar[~ko_mice], axis=0)
            diff_novel_wt = np.nanmean(differences_novel[~ko_mice], axis=0)

        # Get consistent range for all plots (extended so there's room for a legend)
        min_diff = min(np.nanmin(differences_familiar[good_to_show]), np.nanmin(differences_novel[good_to_show]))
        max_diff = max(np.nanmax(differences_familiar[good_to_show]), np.nanmax(differences_novel[good_to_show]))
        diff_range = max_diff - min_diff
        min_diff = min_diff - 0.05 * diff_range
        max_diff = max_diff + 0.15 * diff_range

        colors, linewidths, zorder = get_mouse_colors(use_mice, blinded=state["blinded"], asdict=True, mousedb=self.mousedb)

        # Make the plots!!!
        fig = plt.figure(figsize=(12, 5.5), layout="constrained")
        main_gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        gs_top = main_gs[0].subgridspec(1, 3, width_ratios=[1, 1, 2])
        gs_right = gs_top[2].subgridspec(2, 1, height_ratios=[1, 1])
        gs_bottom = main_gs[1].subgridspec(2, 2)
        ax_ctl_spkmap = fig.add_subplot(gs_top[0])
        ax_red_spkmap = fig.add_subplot(gs_top[1])
        ax_reliability_stats = fig.add_subplot(gs_right[0])
        ax_reliability_summary = fig.add_subplot(gs_right[1])
        ax_mouse_familiar = fig.add_subplot(gs_bottom[0, 0])
        ax_mouse_novel = fig.add_subplot(gs_bottom[0, 1])
        ax_summary_familiar = fig.add_subplot(gs_bottom[1, 0])
        ax_summary_novel = fig.add_subplot(gs_bottom[1, 1])

        spkmap_ctl_cmap = mpl.colormaps["Greys"]
        spkmap_ctl_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "greys_clipped",
            spkmap_ctl_cmap(np.linspace(0.08, 1.0, 256)),
        )
        spkmap_ctl_cmap.set_bad("white")  # ("orange", 0.3))
        spkmap_red_cmap = mpl.colormaps["Reds"]
        spkmap_red_cmap.set_bad("white")  # ("orange", 0.3))
        ax_ctl_spkmap.imshow(ctl_roi_spkmap, aspect="auto", cmap=spkmap_ctl_cmap, interpolation="none", vmin=0, vmax=state["vmax_spkmap"])
        ax_red_spkmap.imshow(red_roi_spkmap, aspect="auto", cmap=spkmap_red_cmap, interpolation="none", vmin=0, vmax=state["vmax_spkmap"])

        format_spines(
            ax_ctl_spkmap,
            x_pos=-0.05,
            y_pos=-0.02,
            xbounds=(0, ctl_roi_spkmap.shape[1]),
            ybounds=ax_ctl_spkmap.get_ylim(),
            yticks=spkmap_yticks,
            ylabels=range(len(spkmaps)),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        format_spines(
            ax_red_spkmap,
            x_pos=-0.05,
            y_pos=-0.02,
            xbounds=(0, red_roi_spkmap.shape[1]),
            ybounds=ax_red_spkmap.get_ylim(),
            yticks=spkmap_yticks,
            ylabels=range(len(spkmaps)),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        ax_ctl_spkmap.set_xlabel("Position (cm)")
        ax_ctl_spkmap.set_ylabel("Session #")
        ax_red_spkmap.set_xlabel("Position (cm)")
        # ax_red_spkmap.set_ylabel("Session #")

        ctl_violins = ax_reliability_stats.violinplot(
            list(ctl_reliability),
            positions=range(reliability.shape[0]),
            side="low",
            showextrema=False,
            widths=0.9,
        )
        red_violins = ax_reliability_stats.violinplot(
            list(red_reliability),
            positions=range(reliability.shape[0]),
            side="high",
            showextrema=False,
            widths=0.9,
        )
        color_violins(ctl_violins, facecolor="k", linecolor="k")
        color_violins(red_violins, facecolor="r", linecolor="r")
        ax_reliability_stats.plot(
            np.arange(reliability.shape[0]) - 0.15,
            ctl_example_reliability,
            color="k",
            marker=".",
            markersize=3,
            linestyle="none",
        )
        ax_reliability_stats.plot(
            np.arange(reliability.shape[0]) + 0.15,
            red_example_reliability,
            color="r",
            marker=".",
            markersize=3,
            linestyle="none",
        )
        format_spines(
            ax_reliability_stats,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, reliability.shape[0] - 1),
            ybounds=ax_reliability_stats.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        ax_reliability_stats.set_xlabel("Session #")
        ax_reliability_stats.set_ylabel("Reliability\nDistribution")

        errorPlot(
            range(reliability.shape[0]),
            ctl_reliability_summary,
            axis=1,
            se=True,
            ax=ax_reliability_summary,
            color="k",
            alpha=0.3,
        )
        errorPlot(
            range(reliability.shape[0]),
            red_reliability_summary,
            axis=1,
            se=True,
            ax=ax_reliability_summary,
            color="r",
            alpha=0.3,
        )
        format_spines(
            ax_reliability_summary,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, reliability.shape[0] - 1),
            ybounds=ax_reliability_summary.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        ax_reliability_summary.set_xlabel("Session #")
        ax_reliability_summary.set_ylabel("Reliability\n(mean +/- SE)")

        if state["forward_backward"] == "forward" or state["forward_backward"] == "both":
            ctl_violins = ax_mouse_familiar.violinplot(
                ctl_data_familiar_forward,
                positions=deltas_familiar_forward,
                showextrema=False,
                showmeans=True,
                side="low",
                widths=0.5,
            )
            red_violins = ax_mouse_familiar.violinplot(
                red_data_familiar_forward,
                positions=deltas_familiar_forward,
                showextrema=False,
                showmeans=True,
                side="high",
                widths=0.5,
            )
            color_violins(ctl_violins, facecolor="k", linecolor="k")
            color_violins(red_violins, facecolor="r", linecolor="r")

        if state["forward_backward"] == "backward" or state["forward_backward"] == "both":
            ctl_violins = ax_mouse_familiar.violinplot(
                ctl_data_familiar_backward,
                positions=[-d for d in deltas_familiar_backward],
                showextrema=False,
                showmeans=True,
                side="low",
                widths=0.5,
            )
            red_violins = ax_mouse_familiar.violinplot(
                red_data_familiar_backward,
                positions=[-d for d in deltas_familiar_backward],
                showextrema=False,
                showmeans=True,
                side="high",
                widths=0.5,
            )
            color_violins(ctl_violins, facecolor="k", linecolor="k")
            color_violins(red_violins, facecolor="r", linecolor="r")

        if state["forward_backward"] == "both":
            ax_mouse_familiar.axvline(0, color="k", linestyle="--", linewidth=1)

        xmin = min(deltas_familiar_forward) if state["forward_backward"] == "forward" else -max(deltas_familiar_backward)
        xmax = -min(deltas_familiar_backward) if state["forward_backward"] == "backward" else max(deltas_familiar_forward)
        xticks = [xmin, 0, xmax] if state["forward_backward"] == "both" else [xmin, xmax]
        ax_mouse_familiar.set_xlabel("$\Delta$Session")
        ax_mouse_familiar.set_ylabel("Reliability")
        ax_mouse_familiar.set_title(f"Familiar Environment")
        if state["indicate_example"]:
            y_legend_position = ax_mouse_familiar.get_ylim()[0] + 0.05 * np.diff(ax_mouse_familiar.get_ylim())
            ax_mouse_familiar.plot([xmin, xmin + 0.8], [y_legend_position, y_legend_position], color=colors[state["mouse"]], marker=".", markersize=7)
            ax_mouse_familiar.text(xmin + 0.95, y_legend_position, state["mouse"], color=colors[state["mouse"]], ha="left", va="center")
        format_spines(
            ax_mouse_familiar,
            x_pos=-0.01,
            y_pos=-0.05,
            xbounds=(xmin, xmax),
            ybounds=ax_mouse_familiar.get_ylim(),
            xticks=xticks,
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        if state["forward_backward"] == "forward" or state["forward_backward"] == "both":
            ctl_violins = ax_mouse_novel.violinplot(
                ctl_data_novel_forward,
                positions=deltas_novel_forward,
                showextrema=False,
                showmeans=True,
                side="low",
                widths=0.5,
            )
            red_violins = ax_mouse_novel.violinplot(
                red_data_novel_forward,
                positions=deltas_novel_forward,
                showextrema=False,
                showmeans=True,
                side="high",
                widths=0.5,
            )
            color_violins(ctl_violins, facecolor="k", linecolor="k")
            color_violins(red_violins, facecolor="r", linecolor="r")

        if state["forward_backward"] == "backward" or state["forward_backward"] == "both":
            ctl_violins = ax_mouse_novel.violinplot(
                ctl_data_novel_backward,
                positions=[-d for d in deltas_novel_backward],
                showextrema=False,
                showmeans=True,
                side="low",
                widths=0.5,
            )
            red_violins = ax_mouse_novel.violinplot(
                red_data_novel_backward,
                positions=[-d for d in deltas_novel_backward],
                showextrema=False,
                showmeans=True,
                side="high",
                widths=0.5,
            )
            color_violins(ctl_violins, facecolor="k", linecolor="k")
            color_violins(red_violins, facecolor="r", linecolor="r")

        if state["forward_backward"] == "both":
            ax_mouse_novel.axvline(0, color="k", linestyle="--", linewidth=1)

        xmin = min(deltas_novel_forward) if state["forward_backward"] == "forward" else -max(deltas_novel_backward)
        xmax = -min(deltas_novel_backward) if state["forward_backward"] == "backward" else max(deltas_novel_forward)
        xticks = [xmin, 0, xmax] if state["forward_backward"] == "both" else [xmin, xmax]
        ax_mouse_novel.set_xlabel("$\Delta$Session")
        # ax_mouse_novel.set_ylabel("Reliability")
        ax_mouse_novel.set_title(f"Novel Environment")
        if state["indicate_example"]:
            y_legend_position = ax_mouse_novel.get_ylim()[0] + 0.05 * np.diff(ax_mouse_novel.get_ylim())
            ax_mouse_novel.plot([xmin, xmin + 0.8], [y_legend_position, y_legend_position], color=colors[state["mouse"]], marker=".", markersize=7)
            ax_mouse_novel.text(xmin + 0.95, y_legend_position, state["mouse"], color=colors[state["mouse"]], ha="left", va="center")
        format_spines(
            ax_mouse_novel,
            x_pos=-0.01,
            y_pos=-0.05,
            xbounds=(xmin, xmax),
            ybounds=ax_mouse_novel.get_ylim(),
            xticks=xticks,
            ylabels=[],
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        alpha = 1.0 if state["blinded"] else 0.3
        for imouse, mouse in enumerate(use_mice):
            if not good_to_show[imouse]:
                continue
            ax_summary_familiar.plot(
                difference_xpos,
                differences_familiar[imouse],
                color=(colors[mouse], alpha),
                linewidth=linewidths[mouse],
                zorder=zorder[mouse],
                marker=".",
                markersize=7,
            )
        if not state["blinded"]:
            ax_summary_familiar.plot(
                difference_xpos,
                diff_familiar_ko,
                color="purple",
                linewidth=2,
                zorder=3,
                marker=".",
                markersize=8,
            )
            ax_summary_familiar.plot(
                difference_xpos,
                diff_familiar_wt,
                color="gray",
                linewidth=2,
                zorder=2.5,
                marker=".",
                markersize=8,
            )
        ax_summary_familiar.set_ylim(min_diff, max_diff)
        format_spines(
            ax_summary_familiar,
            x_pos=-0.01,
            y_pos=-0.05,
            xbounds=(min(difference_xpos), max(difference_xpos)),
            ybounds=ax_summary_familiar.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        ax_summary_familiar.set_xlabel("$\Delta$Session")
        ax_summary_familiar.set_ylabel(f"Reliability (red - ctl)")

        for imouse, mouse in enumerate(use_mice):
            if not good_to_show[imouse]:
                continue
            ax_summary_novel.plot(
                difference_xpos,
                differences_novel[imouse],
                color=(colors[mouse], alpha),
                linewidth=linewidths[mouse],
                zorder=zorder[mouse],
                marker=".",
                markersize=7,
            )
        if not state["blinded"]:
            ax_summary_novel.plot(
                difference_xpos,
                diff_novel_ko,
                color="purple",
                linewidth=2,
                zorder=3,
                marker=".",
                markersize=8,
            )
            ax_summary_novel.plot(
                difference_xpos,
                diff_novel_wt,
                color="gray",
                linewidth=2,
                zorder=2.5,
                marker=".",
                markersize=8,
            )
        ax_summary_novel.set_ylim(min_diff, max_diff)
        format_spines(
            ax_summary_novel,
            x_pos=-0.01,
            y_pos=-0.05,
            xbounds=(min(difference_xpos), max(difference_xpos)),
            ybounds=ax_summary_novel.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
            ylabels=[],
        )
        ax_summary_novel.set_xlabel("$\Delta$Session")
        # ax_summary_novel.set_ylabel(f"Reliability (red - ctl)")

        ax_summary_familiar.axvline(0, color="k", linestyle="--", linewidth=1)
        ax_summary_novel.axvline(0, color="k", linestyle="--", linewidth=1)

        blinded_study_legend(
            ax_summary_familiar,
            xpos=min(difference_xpos),
            ypos=max_diff,
            pilot_colors=[colors[mouse] for mouse in ["CR_Hippocannula6", "CR_Hippocannula7"]],
            blinded_colors=[colors[mouse] for mouse in use_mice if mouse not in ["CR_Hippocannula6", "CR_Hippocannula7"]],
            blinded=state["blinded"],
            origin="upper_left",
        )

        return fig


class ChangingPlaceFieldFigureMaker(Viewer):
    def __init__(self, tracked_mice: list[str], try_cache: bool = True, save_cache: bool = False):
        self.tracked_mice = list(tracked_mice)
        self.multisessions = {mouse: None for mouse in self.tracked_mice}

        self.mousedb = get_database("vrMice")
        self.ko = dict(zip(self.mousedb.get_table()["mouseName"], self.mousedb.get_table()["KO"]))

        # This is a different viewer with data handling capabilities for reliability summary data
        self.summary_viewer = ReliabilityStabilitySummary(self.tracked_mice)

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_integer("environment", value=1, min=0, max=3)
        self.add_integer_range("sessions", value=(0, 1), min=0, max=1)
        self.add_integer("reference_session", value=0, min=0, max=1)
        self.add_selection("reliability_threshold", value=0.5, options=[0.3, 0.5, 0.7, 0.9])
        self.add_float_range("reliability_range", value=(0.5, 1.0), min=-1.0, max=1.0)
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out"])
        self.add_selection("smooth_width", value=5, options=[5])
        self.add_float_range("fraction_range", value=(0.05, 1.0), min=0.0, max=1.0)
        self.add_selection("activity_method", value="rms", options=FractionActive.activity_methods)
        self.add_selection("fraction_method", value="participation", options=FractionActive.fraction_methods)
        self.add_selection("spks_type", value="significant", options=["significant", "oasis"])
        self.add_boolean("use_session_filters", value=True)
        self.add_integer("ctl_roi_idx", value=15, min=0, max=100)
        self.add_integer("red_roi_idx", value=1, min=0, max=100)
        self.add_boolean("continuous", value=True)
        self.add_selection("forward_backward", value="both", options=["forward", "backward", "both"])
        self.add_selection("summary_type", value="pfcorr", options=["pfdelta", "pfcorr"])
        self.add_boolean("group_novel", value=True)
        self.add_boolean("blinded", value=True)
        self.add_boolean("indicate_example", value=False)
        self.add_button("save_example", label="Save Example", callback=self.save_example)

        # Set up callbacks
        self.on_change("mouse", self.reset_mouse)
        self.on_change("environment", self.reset_environment)
        self.on_change(
            [
                "sessions",
                "reference_session",
                "reliability_range",
                "reliability_method",
                "smooth_width",
                "fraction_range",
                "activity_method",
                "fraction_method",
                "spks_type",
                "use_session_filters",
            ],
            self.reset_roi_options,
        )
        self.on_change("sessions", self.reset_sessions)
        self.reset_mouse(self.state)

    def save_example(self, state):
        fig = self.plot(state)
        fig_dir = figure_dir("changing_placefields")
        mouse = state["mouse"]
        environment = state["environment"]
        sessions = state["sessions"]
        idx_tracked = self._idx_tracked
        idx_ctl_keeps = self._idx_ctl_keeps
        idx_red_keeps = self._idx_red_keeps
        ctl_roi_idx = state["ctl_roi_idx"]
        red_roi_idx = state["red_roi_idx"]
        true_ctl_idx = idx_tracked[0, idx_ctl_keeps[ctl_roi_idx]]
        true_red_idx = idx_tracked[0, idx_red_keeps[red_roi_idx]]
        fig_name = f"{state['summary_type']}_{mouse}_env{environment}_ses{sessions[0]}_ses{sessions[1]}_ctl{true_ctl_idx}_red{true_red_idx}"
        if state["continuous"]:
            fig_name += "_continuous"
        if not state["blinded"]:
            fig_name += "_unblinded"
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_dir / fig_name)
        plt.close(fig)

    def reset_mouse(self, state):
        environments = self.get_environments(state["mouse"])
        num_envs = np.sum(environments != -1)
        self.update_integer("environment", max=num_envs - 1)
        self.reset_environment(self.state)

    def reset_environment(self, state):
        msm = self.get_multisession(state["mouse"])
        envstats = msm.env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        env = env_in_order[state["environment"]]
        idx_ses = msm.idx_ses_with_env(env)
        num_sessions = len(idx_ses)
        self.update_integer_range("sessions", value=(0, min(6, num_sessions)), max=num_sessions)
        self.update_integer("reference_session", max=num_sessions - 1)
        self.reset_roi_options(self.state)

    def reset_sessions(self, state):
        sessions = state["sessions"]
        self.update_integer("reference_session", max=sessions[1] - sessions[0] - 1)

    def reset_roi_options(self, state):
        msm = self.get_multisession(state["mouse"])
        envstats = msm.env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        env = env_in_order[state["environment"]]
        idx_ses = msm.idx_ses_with_env(env)[state["sessions"][0] : state["sessions"][1]]
        spkmaps, extras = msm.get_spkmaps(
            env,
            average=False,
            reliability_method=state["reliability_method"],
            smooth=float(state["smooth_width"]),
            spks_type=state["spks_type"],
            idx_ses=idx_ses,
            tracked=True,
            use_session_filters=state["use_session_filters"],
            pop_nan=False,
        )
        fraction_active = [
            FractionActive.compute(
                spkmap,
                activity_axis=2,
                fraction_axis=1,
                activity_method=state["activity_method"],
                fraction_method=state["fraction_method"],
            )
            for spkmap in spkmaps
        ]
        reference_fraction_active = fraction_active[state["reference_session"]]
        idx_reliable_keeps = np.all(
            np.stack([(rel >= state["reliability_range"][0]) & (rel <= state["reliability_range"][1]) for rel in extras["reliability"]]),
            axis=0,
        )
        idx_active_keeps = (reference_fraction_active >= state["fraction_range"][0]) & (reference_fraction_active <= state["fraction_range"][1])
        idx_red = np.any(np.stack(extras["idx_red"]), axis=0)
        idx_ctl_keeps = np.where(idx_reliable_keeps & idx_active_keeps & ~idx_red)[0]
        idx_red_keeps = np.where(idx_reliable_keeps & idx_active_keeps & idx_red)[0]
        self.update_integer("ctl_roi_idx", max=len(idx_ctl_keeps) - 1)
        self.update_integer("red_roi_idx", max=len(idx_red_keeps) - 1)

        self._spkmaps = spkmaps
        self._pfloc = extras["pfloc"]
        self._reliability = extras["reliability"]
        self._idx_tracked = extras["idx_tracked"]
        self._idx_red = np.any(np.stack(extras["idx_red"]), axis=0)
        self._fraction_active = fraction_active
        self._idx_ctl_keeps = idx_ctl_keeps
        self._idx_red_keeps = idx_red_keeps

    def get_multisession(self, mouse: str) -> MultiSessionSpkmaps:
        if self.multisessions[mouse] is None:
            tracker = Tracker(mouse)
            self.multisessions[mouse] = MultiSessionSpkmaps(tracker)
        return self.multisessions[mouse]

    def get_environments(self, mouse: str) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        msm = self.get_multisession(mouse)
        environments = np.unique(np.concatenate([session.environments for session in msm.tracker.sessions]))
        return environments

    def _fraction_active_name(self, activity_method: str, fraction_method: str) -> str:
        return "_".join([activity_method, fraction_method])

    def _make_roi_trajectory(self, spkmaps, roi_idx, dead_trials: int = 1):
        roi_activity = [s[roi_idx] for s in spkmaps]
        dead_space = [np.full((dead_trials, roi_activity[0].shape[1]), np.nan) for _ in range(len(roi_activity) - 1)]
        dead_space.append(None)
        interleaved = [item for pair in zip(roi_activity, dead_space) for item in pair if item is not None]

        trial_env = [ises * np.ones(r.shape[0]) for ises, r in enumerate(roi_activity)]
        dead_trial_env = [np.nan * np.ones(dead_trials) for _ in range(len(roi_activity) - 1)]
        dead_trial_env.append(None)
        env_trialnum = [item for pair in zip(trial_env, dead_trial_env) for item in pair if item is not None]
        return np.concatenate(interleaved, axis=0), np.concatenate(env_trialnum)

    def _process_summary_results(self, results: dict, state: dict, max_session_diff: int = 6):
        output_keys = [
            "num_stable_ctl",
            "num_stable_red",
            "fraction_stable_ctl",
            "fraction_stable_red",
            "stable_reliability_ctl",
            "stable_reliability_red",
            "pfloc_changes_ctl",
            "pfloc_changes_red",
            "spkmap_correlations_ctl",
            "spkmap_correlations_red",
        ]
        max_environments = 3

        data = {key: np.full((len(results), max_environments, max_session_diff), np.nan) for key in output_keys}

        for imouse, mouse in enumerate(results):
            envstats = self.get_multisession(mouse).env_stats()
            env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
            env_in_order = [env for env in env_in_order if env != -1]
            for ienv, env in enumerate(env_in_order):
                if ienv >= max_environments:
                    continue
                if results[mouse][env] is None:
                    continue
                for key in output_keys:
                    if state["forward_backward"] == "forward":
                        data[key][imouse, ienv] = np.nanmean(results[mouse][env]["forward"][key], axis=0)
                    elif state["forward_backward"] == "backward":
                        data[key][imouse, ienv] = np.nanmean(results[mouse][env]["backward"][key], axis=0)
                    elif state["forward_backward"] == "both":
                        forward_data = np.nanmean(results[mouse][env]["forward"][key], axis=0)
                        backward_data = np.nanmean(results[mouse][env]["backward"][key], axis=0)
                        data[key][imouse, ienv] = np.nanmean(np.stack([forward_data, backward_data]), axis=0)
        return data

    def _make_combo_array(self, results, state, envnum, combo_ctl_name, combo_red_name):
        forward_data_ctl = results[state["mouse"]][envnum]["forward"][combo_ctl_name]
        forward_data_red = results[state["mouse"]][envnum]["forward"][combo_red_name]
        backward_data_ctl = results[state["mouse"]][envnum]["backward"][combo_ctl_name]
        backward_data_red = results[state["mouse"]][envnum]["backward"][combo_red_name]
        forward_combos = results[state["mouse"]][envnum]["forward_combos"]
        backward_combos = results[state["mouse"]][envnum]["backward_combos"]
        all_sessions = sorted(list(set(forward_combos[0]).union(*[set(combo) for combo in forward_combos[1:] + backward_combos])))
        num_sessions = len(all_sessions)
        combo_array = np.zeros((num_sessions, num_sessions, 2))

        for icombo in range(len(forward_combos)):
            index_reference = all_sessions.index(forward_combos[icombo][0])
            index_target = all_sessions.index(forward_combos[icombo][-1])
            combo_array[index_reference, index_target, 0] = forward_data_ctl[icombo, index_target - index_reference - 1]
            combo_array[index_reference, index_target, 1] = forward_data_red[icombo, index_target - index_reference - 1]
        for icombo in range(len(backward_combos)):
            index_reference = all_sessions.index(backward_combos[icombo][0])
            index_target = all_sessions.index(backward_combos[icombo][-1])
            combo_array[index_reference, index_target, 0] = backward_data_ctl[icombo, index_reference - index_target - 1]
            combo_array[index_reference, index_target, 1] = backward_data_red[icombo, index_reference - index_target - 1]

        return combo_array

    def plot(self, state):
        use_mice = [mouse for mouse in self.tracked_mice if mouse not in ["ATL045"]]
        # Always group blinded first
        use_mice = [mouse for mouse in use_mice if self.ko[mouse]] + [mouse for mouse in use_mice if not self.ko[mouse]]

        results = {}
        for mouse in use_mice:
            results_state = self.summary_viewer.define_state(
                mouse_name=mouse,
                envnum=1,
                reliability_threshold=state["reliability_threshold"],
                reliability_method=state["reliability_method"],
                smooth_width=int(state["smooth_width"]),
                use_session_filters=state["use_session_filters"],
                continuous=state["continuous"],
                max_session_diff=6,
            )
            results[mouse] = self.summary_viewer.gather_data(results_state, try_cache=True)

        data = self._process_summary_results(results, state, max_session_diff=6)

        msm = self.get_multisession(state["mouse"])
        envstats = msm.env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]

        familiar_envnum = env_in_order[0]
        novel_envnums = env_in_order[1:]

        imouse = use_mice.index(state["mouse"])
        spkmaps = self._spkmaps
        pfloc = np.stack(self._pfloc)
        idx_ctl_keeps = self._idx_ctl_keeps
        idx_red_keeps = self._idx_red_keeps
        ctl_roi_idx = state["ctl_roi_idx"]
        red_roi_idx = state["red_roi_idx"]

        ctl_spkmaps = [s[idx_ctl_keeps] for s in spkmaps]
        red_spkmaps = [s[idx_red_keeps] for s in spkmaps]
        ctl_placefields = np.stack([np.nanmean(s, axis=1) for s in ctl_spkmaps])
        red_placefields = np.stack([np.nanmean(s, axis=1) for s in red_spkmaps])
        ctl_roi_placefields = ctl_placefields[:, ctl_roi_idx]
        red_roi_placefields = red_placefields[:, red_roi_idx]

        if state["summary_type"] == "pfdelta":
            ctl_pfloc = pfloc[:, idx_ctl_keeps]
            red_pfloc = pfloc[:, idx_red_keeps]
            ctl_example_summary = np.abs(ctl_pfloc - ctl_pfloc[state["reference_session"]])
            red_example_summary = np.abs(red_pfloc - red_pfloc[state["reference_session"]])
            combo_ctl_name = "pfloc_changes_ctl"
            combo_red_name = "pfloc_changes_red"
            summary_ylabel = "|$\Delta$PF Location| (cm)"
        elif state["summary_type"] == "pfcorr":
            # Measure correlation coefficients between placefields
            ctl_example_summary = np.full((ctl_placefields.shape[0], ctl_placefields.shape[1]), np.nan)
            red_example_summary = np.full((red_placefields.shape[0], red_placefields.shape[1]), np.nan)
            # Compute correlation coefficients between placefields
            for isession in range(ctl_placefields.shape[0]):
                ctl_example_summary[isession] = vectorCorrelation(ctl_placefields[state["reference_session"]], ctl_placefields[isession])
                red_example_summary[isession] = vectorCorrelation(red_placefields[state["reference_session"]], red_placefields[isession])
            combo_ctl_name = "spkmap_correlations_ctl"
            combo_red_name = "spkmap_correlations_red"
            summary_ylabel = "PF Correlation"
        else:
            raise ValueError(f"Invalid summary type: {state['summary_type']}")

        if state["forward_backward"] == "forward":
            combos_familiar = copy(results[state["mouse"]][familiar_envnum]["forward_combos"])
            combos_novel = copy(results[state["mouse"]][novel_envnums[0]]["forward_combos"])
            raw_data_control_familiar = copy(results[state["mouse"]][familiar_envnum]["forward_raw"][combo_ctl_name])
            raw_data_control_novel = copy(results[state["mouse"]][novel_envnums[0]]["forward_raw"][combo_ctl_name])
            raw_data_red_familiar = copy(results[state["mouse"]][familiar_envnum]["forward_raw"][combo_red_name])
            raw_data_red_novel = copy(results[state["mouse"]][novel_envnums[0]]["forward_raw"][combo_red_name])
            raw_stability_ctl_familiar = copy(results[state["mouse"]][familiar_envnum]["forward_raw"]["ctl_stability"])
            raw_stability_ctl_novel = copy(results[state["mouse"]][novel_envnums[0]]["forward_raw"]["ctl_stability"])
            raw_stability_red_familiar = copy(results[state["mouse"]][familiar_envnum]["forward_raw"]["red_stability"])
            raw_stability_red_novel = copy(results[state["mouse"]][novel_envnums[0]]["forward_raw"]["red_stability"])
        elif state["forward_backward"] == "backward":
            combos_familiar = copy(results[state["mouse"]][familiar_envnum]["backward_combos"])
            combos_novel = copy(results[state["mouse"]][novel_envnums[0]]["backward_combos"])
            raw_data_control_familiar = copy(results[state["mouse"]][familiar_envnum]["backward_raw"][combo_ctl_name])
            raw_data_control_novel = copy(results[state["mouse"]][novel_envnums[0]]["backward_raw"][combo_ctl_name])
            raw_data_red_familiar = copy(results[state["mouse"]][familiar_envnum]["backward_raw"][combo_red_name])
            raw_data_red_novel = copy(results[state["mouse"]][novel_envnums[0]]["backward_raw"][combo_red_name])
            raw_stability_ctl_familiar = copy(results[state["mouse"]][familiar_envnum]["backward_raw"]["ctl_stability"])
            raw_stability_ctl_novel = copy(results[state["mouse"]][novel_envnums[0]]["backward_raw"]["ctl_stability"])
            raw_stability_red_familiar = copy(results[state["mouse"]][familiar_envnum]["backward_raw"]["red_stability"])
            raw_stability_red_novel = copy(results[state["mouse"]][novel_envnums[0]]["backward_raw"]["red_stability"])
        elif state["forward_backward"] == "both":
            combos_familiar = copy(results[state["mouse"]][familiar_envnum]["forward_combos"])
            combos_novel = copy(results[state["mouse"]][novel_envnums[0]]["forward_combos"])
            combos_familiar += results[state["mouse"]][familiar_envnum]["backward_combos"]
            combos_novel += results[state["mouse"]][novel_envnums[0]]["backward_combos"]
            raw_data_control_familiar = copy(results[state["mouse"]][familiar_envnum]["forward_raw"][combo_ctl_name])
            raw_data_control_novel = copy(results[state["mouse"]][novel_envnums[0]]["forward_raw"][combo_ctl_name])
            raw_data_red_familiar = copy(results[state["mouse"]][familiar_envnum]["forward_raw"][combo_red_name])
            raw_data_red_novel = copy(results[state["mouse"]][novel_envnums[0]]["forward_raw"][combo_red_name])
            raw_data_control_familiar += results[state["mouse"]][familiar_envnum]["backward_raw"][combo_ctl_name]
            raw_data_control_novel += results[state["mouse"]][novel_envnums[0]]["backward_raw"][combo_ctl_name]
            raw_data_red_familiar += results[state["mouse"]][familiar_envnum]["backward_raw"][combo_red_name]
            raw_data_red_novel += results[state["mouse"]][novel_envnums[0]]["backward_raw"][combo_red_name]
            raw_stability_ctl_familiar = copy(results[state["mouse"]][familiar_envnum]["forward_raw"]["ctl_stability"])
            raw_stability_ctl_novel = copy(results[state["mouse"]][novel_envnums[0]]["forward_raw"]["ctl_stability"])
            raw_stability_red_familiar = copy(results[state["mouse"]][familiar_envnum]["forward_raw"]["red_stability"])
            raw_stability_red_novel = copy(results[state["mouse"]][novel_envnums[0]]["forward_raw"]["red_stability"])
            raw_stability_ctl_familiar += results[state["mouse"]][familiar_envnum]["backward_raw"]["ctl_stability"]
            raw_stability_ctl_novel += results[state["mouse"]][novel_envnums[0]]["backward_raw"]["ctl_stability"]
            raw_stability_red_familiar += results[state["mouse"]][familiar_envnum]["backward_raw"]["red_stability"]
            raw_stability_red_novel += results[state["mouse"]][novel_envnums[0]]["backward_raw"]["red_stability"]
        else:
            raise ValueError(f"Invalid forward_backward: {state['forward_backward']}")

        # Only get target data for each combo
        # We filter by rst ("stability") because these are the ones that have a reliable place field in both sessions
        raw_data_control_familiar = [rdc[-1][rst[-1]] for rdc, rst in zip(raw_data_control_familiar, raw_stability_ctl_familiar)]
        raw_data_control_novel = [rdc[-1][rst[-1]] for rdc, rst in zip(raw_data_control_novel, raw_stability_ctl_novel)]
        raw_data_red_familiar = [rdc[-1][rst[-1]] for rdc, rst in zip(raw_data_red_familiar, raw_stability_red_familiar)]
        raw_data_red_novel = [rdc[-1][rst[-1]] for rdc, rst in zip(raw_data_red_novel, raw_stability_red_novel)]
        all_sessions_familiar = sorted(list(set(combos_familiar[0]).union(*[set(combo) for combo in combos_familiar[1:]])))
        all_sessions_novel = sorted(list(set(combos_novel[0]).union(*[set(combo) for combo in combos_novel[1:]])))
        num_diffs_familiar = len(all_sessions_familiar) - 1
        num_diffs_novel = len(all_sessions_novel) - 1
        ctl_data_familiar = [[] for _ in range(num_diffs_familiar)]
        ctl_data_novel = [[] for _ in range(num_diffs_novel)]
        red_data_familiar = [[] for _ in range(num_diffs_familiar)]
        red_data_novel = [[] for _ in range(num_diffs_novel)]
        used_familiar = [False] * num_diffs_familiar
        used_novel = [False] * num_diffs_novel
        for icombo in range(len(combos_familiar)):
            index_reference = all_sessions_familiar.index(combos_familiar[icombo][0])
            index_target = all_sessions_familiar.index(combos_familiar[icombo][-1])
            c_delta = abs(index_target - index_reference) - 1
            ctl_data_familiar[c_delta].append(raw_data_control_familiar[icombo])
            red_data_familiar[c_delta].append(raw_data_red_familiar[icombo])
            used_familiar[c_delta] = True
        for icombo in range(len(combos_novel)):
            index_reference = all_sessions_novel.index(combos_novel[icombo][0])
            index_target = all_sessions_novel.index(combos_novel[icombo][-1])
            c_delta = abs(index_target - index_reference) - 1
            ctl_data_novel[c_delta].append(raw_data_control_novel[icombo])
            red_data_novel[c_delta].append(raw_data_red_novel[icombo])
            used_novel[c_delta] = True

        ctl_data_familiar = [data for data, used in zip(ctl_data_familiar, used_familiar) if used]
        red_data_familiar = [data for data, used in zip(red_data_familiar, used_familiar) if used]
        ctl_data_novel = [data for data, used in zip(ctl_data_novel, used_novel) if used]
        red_data_novel = [data for data, used in zip(red_data_novel, used_novel) if used]
        ctl_data_familiar = [np.concatenate(data) for data in ctl_data_familiar]
        red_data_familiar = [np.concatenate(data) for data in red_data_familiar]
        ctl_data_novel = [np.concatenate(data) for data in ctl_data_novel]
        red_data_novel = [np.concatenate(data) for data in red_data_novel]

        # Get the selected ROIs summary data
        ctl_roi_summary = ctl_example_summary[:, ctl_roi_idx]
        red_roi_summary = red_example_summary[:, red_roi_idx]

        # Grouping differences across environments (only show first novel environment which we have for all mice!!!)
        differences = data[combo_red_name] - data[combo_ctl_name]
        nan_spots = np.isnan(differences)[:, : 3 if state["group_novel"] else 2]
        # no_valid_entries = np.sum(~nan_spots, axis=2) == 0
        # good_to_show = np.any(~no_valid_entries, axis=1)
        more_than_one_or_none = (np.sum(~nan_spots, axis=2) > 1) | (np.sum(~nan_spots, axis=2) == 0)
        good_to_show = ~np.any(np.diff(1 * nan_spots, axis=2) == -1, axis=(1, 2)) & np.all(more_than_one_or_none, axis=1)
        differences_familiar = differences[:, 0]
        if state["group_novel"]:
            differences_novel = np.nanmean(differences[:, 1:], axis=1)
        else:
            differences_novel = differences[:, 1]

        min_diff = min(np.nanmin(differences_familiar[good_to_show]), np.nanmin(differences_novel[good_to_show]))
        max_diff = max(np.nanmax(differences_familiar[good_to_show]), np.nanmax(differences_novel[good_to_show]))
        diff_range = max_diff - min_diff
        min_diff = min_diff - 0.05 * diff_range
        max_diff = max_diff + 0.15 * diff_range

        if not state["blinded"]:
            ko = np.array([self.ko[mouse] for mouse in use_mice])
            diff_familiar_ko = np.nanmean(differences_familiar[ko], axis=0)
            diff_novel_ko = np.nanmean(differences_novel[ko], axis=0)
            diff_familiar_wt = np.nanmean(differences_familiar[~ko], axis=0)
            diff_novel_wt = np.nanmean(differences_novel[~ko], axis=0)

        # Prepare some figure aesthetics
        colors, linewidths, zorder = get_mouse_colors(use_mice, blinded=state["blinded"], asdict=True, mousedb=self.mousedb)

        fig = plt.figure(figsize=(12, 5.5), layout="constrained")
        main_gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])
        gs_example = main_gs[0].subgridspec(3, 1)
        ax_ctl_placefield = fig.add_subplot(gs_example[0])
        ax_red_placefield = fig.add_subplot(gs_example[1])
        ax_roi_summary = fig.add_subplot(gs_example[2])

        gs_summary = main_gs[1].subgridspec(2, 2)
        ax_example_familiar = fig.add_subplot(gs_summary[0, 0])
        ax_example_novel = fig.add_subplot(gs_summary[0, 1])
        ax_summary_familiar = fig.add_subplot(gs_summary[1, 0])
        ax_summary_novel = fig.add_subplot(gs_summary[1, 1])

        placefield_ctl_cmap = mpl.colormaps["Greys"]
        placefield_red_cmap = mpl.colormaps["Reds"]
        pf_ctl_colors = placefield_ctl_cmap(np.linspace(1.0, 0.4, ctl_roi_placefields.shape[0]))
        pf_red_colors = placefield_red_cmap(np.linspace(1.0, 0.4, red_roi_placefields.shape[0]))
        for isession in range(ctl_placefields.shape[0]):
            ax_ctl_placefield.plot(
                range(ctl_roi_placefields.shape[1]),
                ctl_roi_placefields[isession],
                color=pf_ctl_colors[isession],
                linewidth=1.0,
            )
            ax_red_placefield.plot(
                range(red_roi_placefields.shape[1]),
                red_roi_placefields[isession],
                color=pf_red_colors[isession],
                linewidth=1.0,
            )
        format_spines(
            ax_ctl_placefield,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, ctl_roi_placefields.shape[1]),
            ybounds=ax_ctl_placefield.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        format_spines(
            ax_red_placefield,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, red_roi_placefields.shape[1]),
            ybounds=ax_red_placefield.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        ax_ctl_placefield.set_xlabel("Position (cm)")
        ax_ctl_placefield.set_ylabel("Activity ($\sigma$)")
        ax_red_placefield.set_xlabel("Position (cm)")
        ax_red_placefield.set_ylabel("Activity ($\sigma$)")

        ax_roi_summary.plot(range(len(ctl_roi_summary)), ctl_roi_summary, color="k", linewidth=1.0)
        ax_roi_summary.plot(range(len(red_roi_summary)), red_roi_summary, color="r", linewidth=1.0)
        for isession in range(ctl_roi_summary.shape[0]):
            ax_roi_summary.plot(isession, ctl_roi_summary[isession], color=pf_ctl_colors[isession], linestyle="none", marker=".", markersize=10)
            ax_roi_summary.plot(isession, red_roi_summary[isession], color=pf_red_colors[isession], linestyle="none", marker=".", markersize=10)
        ax_roi_summary.set_xlabel("Session #")
        ax_roi_summary.set_ylabel(summary_ylabel)
        format_spines(
            ax_roi_summary,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, ctl_roi_summary.shape[0] - 1),
            ybounds=ax_roi_summary.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        example_ylim = (-0.25, 1)
        ctl_violins = ax_example_familiar.violinplot(
            ctl_data_familiar,
            positions=1 + np.arange(len(ctl_data_familiar)),
            showextrema=False,
            showmeans=True,
            side="low",
            widths=0.5,
        )
        red_violins = ax_example_familiar.violinplot(
            red_data_familiar,
            positions=1 + np.arange(len(red_data_familiar)),
            showextrema=False,
            showmeans=True,
            side="high",
            widths=0.5,
        )
        color_violins(ctl_violins, facecolor="k", linecolor="k")
        color_violins(red_violins, facecolor="r", linecolor="r")
        ax_example_familiar.set_xlabel("$\Delta$Session")
        ax_example_familiar.set_ylabel(summary_ylabel)
        ax_example_familiar.set_title("Familiar Environment")
        ax_example_familiar.set_ylim(example_ylim)
        ylim = ax_example_familiar.get_ylim()
        if state["indicate_example"]:
            if state["summary_type"] == "pfdelta":
                y_legend_position = ylim[1] - 0.15 * (ylim[1] - ylim[0])
            else:
                y_legend_position = ylim[0] + 0.15 * (ylim[1] - ylim[0])
            ax_example_familiar.plot([1, 1.4], [y_legend_position, y_legend_position], color=colors[state["mouse"]], marker=".", markersize=7)
            ax_example_familiar.text(1.5, y_legend_position, state["mouse"], color=colors[state["mouse"]], ha="left", va="center")
        format_spines(
            ax_example_familiar,
            x_pos=-0.01,
            y_pos=-0.05,
            xbounds=(1, len(ctl_data_familiar)),
            ybounds=ax_example_familiar.get_ylim(),
            xticks=np.arange(1, len(ctl_data_familiar) + 1),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        ctl_violins = ax_example_novel.violinplot(
            ctl_data_novel,
            positions=1 + np.arange(len(ctl_data_novel)),
            showextrema=False,
            showmeans=True,
            side="low",
            widths=0.5,
        )
        red_violins = ax_example_novel.violinplot(
            red_data_novel,
            positions=1 + np.arange(len(red_data_novel)),
            showextrema=False,
            showmeans=True,
            side="high",
            widths=0.5,
        )
        color_violins(ctl_violins, facecolor="k", linecolor="k")
        color_violins(red_violins, facecolor="r", linecolor="r")
        ax_example_novel.set_xlabel("$\Delta$Session")
        # ax_example_novel.set_ylabel(summary_ylabel)
        ax_example_novel.set_title("Novel Environment")
        ax_example_novel.set_ylim(example_ylim)
        ylim = ax_example_novel.get_ylim()
        if state["indicate_example"]:
            if state["summary_type"] == "pfdelta":
                y_legend_position = ylim[1] - 0.15 * (ylim[1] - ylim[0])
            else:
                y_legend_position = ylim[0] + 0.15 * (ylim[1] - ylim[0])
            ax_example_novel.plot([1, 1.4], [y_legend_position, y_legend_position], color=colors[state["mouse"]], marker=".", markersize=7)
            ax_example_novel.text(1.5, y_legend_position, state["mouse"], color=colors[state["mouse"]], ha="left", va="center")
        format_spines(
            ax_example_novel,
            x_pos=-0.01,
            y_pos=-0.05,
            xbounds=(1, len(ctl_data_novel)),
            ybounds=ax_example_novel.get_ylim(),
            xticks=np.arange(1, len(ctl_data_novel) + 1),
            ylabels=[],
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        alpha = 1.0 if state["blinded"] else 0.3
        for imouse, mouse in enumerate(use_mice):
            if not good_to_show[imouse]:
                continue
            ax_summary_familiar.plot(
                range(1, differences_familiar.shape[1] + 1),
                differences_familiar[imouse],
                color=(colors[mouse], alpha),
                linewidth=linewidths[mouse],
                zorder=zorder[mouse],
                marker=".",
                markersize=7,
            )
        if not state["blinded"]:
            ax_summary_familiar.plot(
                range(1, differences_familiar.shape[1] + 1),
                diff_familiar_ko,
                color="purple",
                linewidth=2,
                zorder=3,
                marker=".",
                markersize=8,
            )
            ax_summary_familiar.plot(
                range(1, differences_familiar.shape[1] + 1),
                diff_familiar_wt,
                color="gray",
                linewidth=2,
                zorder=2.5,
                marker=".",
                markersize=8,
            )
        ax_summary_familiar.set_ylim(min_diff, max_diff)
        format_spines(
            ax_summary_familiar,
            x_pos=-0.01,
            y_pos=-0.05,
            xbounds=(1, differences_familiar.shape[1]),
            ybounds=ax_summary_familiar.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        ax_summary_familiar.set_xlabel("$\Delta$Session")
        ax_summary_familiar.set_ylabel(f"{summary_ylabel} (red - ctl)")

        for imouse, mouse in enumerate(use_mice):
            if not good_to_show[imouse]:
                continue
            ax_summary_novel.plot(
                range(1, differences_novel.shape[1] + 1),
                differences_novel[imouse],
                color=(colors[mouse], alpha),
                linewidth=linewidths[mouse],
                zorder=zorder[mouse],
                marker=".",
                markersize=7,
            )
        if not state["blinded"]:
            ax_summary_novel.plot(
                range(1, differences_novel.shape[1] + 1),
                diff_novel_ko,
                color="purple",
                linewidth=2,
                zorder=3,
                marker=".",
                markersize=8,
            )
            ax_summary_novel.plot(
                range(1, differences_novel.shape[1] + 1),
                diff_novel_wt,
                color="gray",
                linewidth=2,
                zorder=2.5,
                marker=".",
                markersize=8,
            )
        ax_summary_novel.set_ylim(min_diff, max_diff)
        format_spines(
            ax_summary_novel,
            x_pos=-0.01,
            y_pos=-0.05,
            ylabels=[],
            xbounds=(1, differences_novel.shape[1]),
            ybounds=ax_summary_novel.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        ax_summary_novel.set_xlabel("$\Delta$Session")
        # ax_summary_novel.set_ylabel(f"{summary_ylabel} (red - ctl)")

        blinded_study_legend(
            ax_summary_familiar,
            xpos=1,
            ypos=max_diff,
            pilot_colors=[colors[mouse] for mouse in ["CR_Hippocannula6", "CR_Hippocannula7"]],
            blinded_colors=[colors[mouse] for mouse in use_mice if mouse not in ["CR_Hippocannula6", "CR_Hippocannula7"]],
            blinded=state["blinded"],
            origin="upper_left",
        )

        return fig
