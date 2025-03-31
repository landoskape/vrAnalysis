import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from syd import Viewer
from ..tracking import Tracker
from ..metrics import FractionActive, KernelDensityEstimator, plot_contours
from ..multisession import MultiSessionSpkmaps


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
        self.add_selection("activity_method", value="max", options=FractionActive.activity_methods)
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
