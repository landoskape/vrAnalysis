import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from vrAnalysis import analysis, database, tracking


class PlaceFieldViewer:
    def __init__(self, fast_mode=False):
        mousedb = database.vrDatabase("vrMice")
        df = mousedb.getTable(trackerExists=True)
        self.mouse_names = df["mouseName"].unique()
        if fast_mode:
            self.mouse_names = self.mouse_names[:2]
        print(self.mouse_names)
        self.env_selection = {}
        self.idx_ses_selection = {}
        self.spkmaps = {}
        self.extras = {}

        for mouse_name in tqdm(self.mouse_names, desc="Preparing mouse data", leave=True):
            self._prepare_mouse_data(mouse_name, fast_mode)

    def _prepare_mouse_data(self, mouse_name, fast_mode):
        keep_planes = [1] if fast_mode else [1, 2, 3, 4]
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=keep_planes)
        envnum, idx_ses = pcm.env_idx_ses_selector(envmethod="second", sesmethod=4)
        spkmaps, extras = pcm.get_spkmaps(envnum=envnum, idx_ses=idx_ses, trials="full", average=False, tracked=True)

        self.env_selection[mouse_name] = (envnum, idx_ses)
        self.idx_ses_selection[mouse_name] = idx_ses
        self.spkmaps[mouse_name] = spkmaps
        self.extras[mouse_name] = extras

    def _make_roi_trajectory(self, spkmaps, roi_idx, dead_trials=None):
        if dead_trials is None:
            dead_trials = 5
        roi_activity = [s[roi_idx] for s in spkmaps]
        dead_space = [np.full((dead_trials, roi_activity[0].shape[1]), np.nan) for _ in range(len(roi_activity) - 1)]
        dead_space.append(None)
        interleaved = [item for pair in zip(roi_activity, dead_space) for item in pair if item is not None]

        trial_env = [ises * np.ones(r.shape[0]) for ises, r in enumerate(roi_activity)]
        dead_trial_env = [np.nan * np.ones(dead_trials) for _ in range(len(roi_activity) - 1)]
        dead_trial_env.append(None)
        env_trialnum = [item for pair in zip(trial_env, dead_trial_env) for item in pair if item is not None]
        return np.concatenate(interleaved, axis=0), np.concatenate(env_trialnum)

    def _gather_idxs(self, mouse_name, idx_target_ses, min_percentile=90, max_percentile=100):
        all_values = np.concatenate(self.extras[mouse_name]["relcor"])
        reliability_values = self.extras[mouse_name]["relcor"][idx_target_ses]
        min_threshold = np.percentile(all_values, min_percentile)
        max_threshold = np.percentile(all_values, max_percentile)
        return np.where((reliability_values > min_threshold) & (reliability_values < max_threshold))[0]

    def _com(self, data, axis=-1):
        x = np.arange(data.shape[axis])
        com = np.sum(data * x, axis=axis) / (np.sum(data, axis=axis) + 1e-10)
        com[np.any(data < 0, axis=axis)] = np.nan
        return com

    def get_plot(self, mouse_name, roi_idx, min_percentile=90, max_percentile=100, idx_target_ses=0, dead_trials=5):
        spkmaps = self.spkmaps[mouse_name]
        idxs = self._gather_idxs(mouse_name, idx_target_ses, min_percentile, max_percentile)

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

        roi_trajectory, env_trialnum = self._make_roi_trajectory(spkmaps, idx_roi_to_plot, dead_trials=dead_trials)

        idx_not_nan = ~np.any(np.isnan(roi_trajectory), axis=1)
        pfmax = np.where(idx_not_nan, np.max(roi_trajectory, axis=1), np.nan)
        pfcom = np.where(idx_not_nan, self._com(roi_trajectory, axis=1), np.nan)
        pfloc = np.where(idx_not_nan, np.argmax(roi_trajectory, axis=1), np.nan)

        cmap = mpl.colormaps["gray_r"]
        cmap.set_bad((1, 0.8, 0.8))  # Light red color
        ses_col = plt.cm.Set1(np.linspace(0, 1, len(self.idx_ses_selection[mouse_name])))

        fig = plt.figure(1, figsize=(10, 8))
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

        fig.suptitle(f"Mouse: {mouse_name}, ROI: {idx_roi_to_plot}, Target Session: {idx_target_ses}")

        return fig

    def __call__(self, mouse_name, roi_idx, min_percentile=90, max_percentile=100, idx_target_ses=0, dead_trials=5):
        return self.get_plot(mouse_name, roi_idx, min_percentile, max_percentile, idx_target_ses, dead_trials)
