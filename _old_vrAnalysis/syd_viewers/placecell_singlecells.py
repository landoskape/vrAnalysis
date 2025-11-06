from .. import analysis, tracking
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from syd import InteractiveViewer


class PlaceFieldViewer(InteractiveViewer):
    def __init__(self, mouse_name, num_sessions=4, keep_planes=[1, 2, 3, 4]):
        self.mouse_name = mouse_name
        self.env_selection = {}
        self.idx_ses_selection = {}
        self.spkmaps = {}
        self.extras = {}

        self.num_sessions = num_sessions
        self.keep_planes = keep_planes

        self._prepare_mouse_data(mouse_name)

        # Add interactive parameters
        self.add_selection("envmethod", options=["first", "second"], value="second")
        self.add_selection("idx_target_ses", options=self.idx_ses_selection["second"], value=self.idx_ses_selection["second"][0])
        self.add_float_range("percentile_range", value=(90, 100), min_value=0, max_value=100)
        self.add_boolean("red_cells", value=False)
        self.add_selection("roi_idx", options=[0], value=0)

        # Use callback to set roi_idx options consistent with current values
        self.set_roi_options(self.get_state())

        self.on_change(["envmethod", "idx_target_ses", "percentile_range", "red_cells"], self.set_roi_options)

    def set_roi_options(self, state):
        envmethod = state["envmethod"]
        idx_target_ses = state["idx_target_ses"]
        min_percentile = state["percentile_range"][0]
        max_percentile = state["percentile_range"][1]
        red_cells = state["red_cells"]

        idx_roi_options = self._gather_idxs(envmethod, idx_target_ses, min_percentile, max_percentile, red_cells)
        self.update_selection("roi_idx", options=list(idx_roi_options))

    def _gather_idxs(self, envmethod, idx_target_ses, min_percentile=90, max_percentile=100, red_cells=False):
        all_values = np.concatenate(self.extras[envmethod]["relloo"])
        reliability_values = self.extras[envmethod]["relloo"][idx_target_ses]

        min_threshold = np.percentile(all_values, min_percentile)
        max_threshold = np.percentile(all_values, max_percentile)
        idx_in_percentile = (reliability_values > min_threshold) & (reliability_values < max_threshold)

        idx_red = np.any(np.stack([ired for ired in self.extras[envmethod]["idx_red"]]), axis=0)
        if not red_cells:
            idx_red = ~idx_red

        idx_keepers = np.where(np.logical_and(idx_in_percentile, idx_red))[0]
        return idx_keepers

    def _prepare_mouse_data(self, mouse_name):
        keep_planes = self.keep_planes
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=keep_planes)

        for envmethod in ["first", "second"]:
            envnum = pcm.env_selector(envmethod=envmethod)
            idx_ses = pcm.idx_ses_selector(envnum=envnum, sesmethod=self.num_sessions)
            spkmaps, extras = pcm.get_spkmaps(envnum=envnum, idx_ses=idx_ses, trials="full", average=False, tracked=True)
            self.env_selection[envmethod] = (envnum, idx_ses)
            self.idx_ses_selection[envmethod] = idx_ses
            self.spkmaps[envmethod] = spkmaps
            self.extras[envmethod] = extras

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

    def _gather_idxs(self, envmethod, idx_target_ses, min_percentile=90, max_percentile=100, red_cells=False):
        all_values = np.concatenate(self.extras[envmethod]["relloo"])
        reliability_values = self.extras[envmethod]["relloo"][idx_target_ses]

        min_threshold = np.percentile(all_values, min_percentile)
        max_threshold = np.percentile(all_values, max_percentile)
        idx_in_percentile = (reliability_values > min_threshold) & (reliability_values < max_threshold)

        idx_red = np.any(np.stack([ired for ired in self.extras[envmethod]["idx_red"]]), axis=0)
        if not red_cells:
            idx_red = ~idx_red

        idx_keepers = np.where(np.logical_and(idx_in_percentile, idx_red))[0]
        return idx_keepers

    def _com(self, data, axis=-1):
        x = np.arange(data.shape[axis])
        com = np.sum(data * x, axis=axis) / (np.sum(data, axis=axis) + 1e-10)
        com[np.any(data < 0, axis=axis)] = np.nan
        return com

    def plot(self, state):
        envmethod = state["envmethod"]
        idx_target_ses = state["idx_target_ses"]
        min_percentile = state["percentile_range"][0]
        max_percentile = state["percentile_range"][1]
        red_cells = state["red_cells"]
        roi_idx = state["roi_idx"]

        spkmaps = self.spkmaps[envmethod]
        idxs = self._gather_idxs(envmethod, idx_target_ses, min_percentile, max_percentile, red_cells)

        if len(idxs) == 0:
            # Create an empty figure with a message
            fig = plt.figure(figsize=(12, 8))
            plt.text(
                0.5,
                0.5,
                f"No ROIs found between {min_percentile}th and {max_percentile}th percentiles",
                horizontalalignment="center",
                verticalalignment="center",
                transform=fig.transFigure,
                fontsize=14,
            )
            return fig

        # Ensure roi_idx doesn't exceed available ROIs
        roi_idx = roi_idx % len(idxs)  # Wrap around if too large
        idx_roi_to_plot = idxs[roi_idx]

        relcor = ",".join([f"{ses[idx_roi_to_plot]:.2f}" for ses in self.extras[envmethod]["relcor"]])
        relloo = ",".join([f"{ses[idx_roi_to_plot]:.2f}" for ses in self.extras[envmethod]["relloo"]])

        roi_trajectory, env_trialnum = self._make_roi_trajectory(spkmaps, idx_roi_to_plot)

        idx_not_nan = ~np.any(np.isnan(roi_trajectory), axis=1)
        pfmax = np.where(idx_not_nan, np.max(roi_trajectory, axis=1), np.nan)
        pfcom = np.where(idx_not_nan, self._com(roi_trajectory, axis=1), np.nan)
        pfloc = np.where(idx_not_nan, np.argmax(roi_trajectory, axis=1), np.nan)

        cmap = mpl.colormaps["gray_r"]
        cmap.set_bad((1, 0.8, 0.8))  # Light red color
        ses_col = plt.cm.Set1(np.linspace(0, 1, len(self.idx_ses_selection[envmethod])))

        fig = plt.figure(figsize=(10, 8))
        fig.clf()

        ax = fig.add_subplot(131)
        ax.cla()
        ax.imshow(roi_trajectory, aspect="auto", interpolation="none", cmap=cmap, vmin=0, vmax=10)

        # Add vertical bar at x==0 to show which environment is target
        idx_trials_target = np.where(env_trialnum == idx_target_ses)[0]
        if np.any(idx_trials_target):  # Check if there are any target trials
            min_y = np.nanmin(idx_trials_target)
            max_y = np.nanmax(idx_trials_target)
            ax.plot([1, 1], [min_y, max_y], color=ses_col[idx_target_ses], linestyle="-", linewidth=5)
            ax.plot(
                [roi_trajectory.shape[1] - 1, roi_trajectory.shape[1] - 1],
                [min_y, max_y],
                color=ses_col[idx_target_ses],
                linestyle="-",
                linewidth=5,
            )
        ax.text(0, (min_y + max_y) / 2, f"Target Session", color=ses_col[idx_target_ses], ha="right", va="center", rotation=90)
        ax.set_xlim(0, roi_trajectory.shape[1])
        ax.set_ylim(roi_trajectory.shape[0], 0)
        ax.set_ylabel("Trial")
        ax.set_yticks([])
        ax.set_xlabel("Virtual Position")
        ax.set_title("PF Activity")

        alpha_values = np.where(~np.isnan(pfmax), pfmax / np.nanmax(pfmax), 0)
        ax = fig.add_subplot(132)
        ax.cla()
        ax.scatter(pfcom, range(len(pfcom)), s=10, color="k", alpha=alpha_values, linewidth=2)
        ax.scatter(pfloc, range(len(pfloc)), s=10, color="r", alpha=alpha_values, linewidth=2)
        ax.scatter([-10], [-10], color="k", s=10, alpha=1.0, linewidth=2, label="CoM")
        ax.scatter([-10], [-10], color="r", s=10, alpha=1.0, linewidth=2, label="MaxLoc")
        ax.set_xlim(0, roi_trajectory.shape[1])
        ax.set_ylim(roi_trajectory.shape[0], 0)
        ax.legend(loc="upper center")
        ax.set_yticks([])
        ax.set_title("PF Location")
        ax.set_xlabel("Virtual Position")

        ax = fig.add_subplot(133)
        ax.cla()
        ax.scatter(pfmax, range(len(pfmax)), color="k", s=10, alpha=alpha_values)
        ax.set_xlim(0, np.nanmax(pfmax))
        ax.set_ylim(roi_trajectory.shape[0], 0)
        ax.set_title("PF Amplitude")
        ax.set_yticks([])
        ax.set_xlabel("Activity (sigma)")

        fig.suptitle(
            f"Mouse: {self.mouse_name}, Env: {envmethod}, ROI: {idx_roi_to_plot}, Target Session: {idx_target_ses}\nRelCor: {relcor}\nRelLoo: {relloo}"
        )

        return fig
