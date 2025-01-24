from .. import analysis, tracking, helpers
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from syd import InteractiveViewer


def gini(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = x.shape[axis]
    x = np.sort(x, axis=axis)  # Sort values
    cum_vals = np.sum(x * (n - np.arange(n)), axis=axis)
    gini_coefficient = (n + 1 - 2 * cum_vals / np.sum(x, axis=axis)) / n
    return 1 - gini_coefficient


def entropy(x: np.ndarray, axis: int = -1, relative: bool = True) -> np.ndarray:
    activity = x / np.sum(x, axis=axis, keepdims=True)
    # Handle zeros in activity by setting 0 * log(0) = 0
    log_activity = np.zeros_like(activity)
    nonzero = activity > 0
    log_activity[nonzero] = np.log(activity[nonzero])
    output = -np.sum(activity * log_activity, axis=axis)
    if relative:
        output = output / np.log(activity.shape[axis])
    return output


def fraction_active(spkmap, activity_method: str = "max", fraction_method: str = "gini"):
    if activity_method == "max":
        activity = np.nanmax(spkmap, axis=2)
    elif activity_method == "mean":
        activity = np.nanmean(spkmap, axis=2)
    elif activity_method == "rms":
        activity = np.sqrt(np.nanmean(spkmap**2, axis=2))
    else:
        raise ValueError(f"Invalid activity method: {activity_method}")

    if fraction_method == "gini":
        fraction_active = gini(activity, axis=1)
    elif fraction_method == "entropy":
        fraction_active = entropy(activity, axis=1, relative=False)
    elif fraction_method == "relative_entropy":
        fraction_active = entropy(activity, axis=1, relative=True)
    else:
        raise ValueError(f"Invalid fraction method: {fraction_method}")

    return fraction_active


activity_methods = ["max", "mean", "rms"]
fraction_methods = ["gini", "entropy", "relative_entropy"]
fraction_active_combinations = list(itertools.product(activity_methods, fraction_methods))


class PlaceFieldLoader:
    def __init__(self, mouse_name, sesmethod="all", keep_planes=[1, 2, 3, 4]):
        self.mouse_name = mouse_name
        self.env_selection = {}
        self.idx_ses_selection = {}
        self.spkmaps = {}
        self.extras = {}
        self.all_trial_reliability = {}
        self.session_average_reliability = {}
        self.fraction_active = {}

        self.sesmethod = sesmethod
        self.keep_planes = keep_planes

        self._prepare_mouse_data(mouse_name)

    def _fraction_active_name(self, activity_method: str, fraction_method: str) -> str:
        return "_".join([activity_method, fraction_method])

    def _prepare_mouse_data(self, mouse_name):
        keep_planes = self.keep_planes
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=keep_planes)

        for envoption in ["first", "second"]:
            envnum = pcm.env_selector(envmethod=envoption)
            idx_ses = pcm.idx_ses_selector(envnum=envnum, sesmethod=self.sesmethod)
            spkmaps, extras = pcm.get_spkmaps(envnum=envnum, idx_ses=idx_ses, trials="full", average=False, tracked=True)
            self.env_selection[envoption] = (envoption, idx_ses)
            self.idx_ses_selection[envoption] = idx_ses
            self.spkmaps[envoption] = spkmaps
            self.extras[envoption] = extras
            self.fraction_active[envoption] = {}
            for amethod, fmethod in fraction_active_combinations:
                cdata = []
                for spkmap in spkmaps:
                    cdata.append(fraction_active(spkmap, activity_method=amethod, fraction_method=fmethod))
                cdata = np.stack(cdata, axis=1)
                self.fraction_active[envoption][self._fraction_active_name(amethod, fmethod)] = cdata

            all_trial_reliability = helpers.reliability_loo(np.concatenate(spkmaps, axis=1), weighted=True)
            session_average_reliability = helpers.reliability_loo(np.stack([np.nanmean(spkmap, axis=1) for spkmap in spkmaps], axis=1), weighted=True)
            self.all_trial_reliability[envoption] = all_trial_reliability
            self.session_average_reliability[envoption] = session_average_reliability

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

    def _gather_idxs(self, envoption, min_percentile=90, max_percentile=100, red_cells=False, reliability_type="all_trials"):
        if reliability_type == "all_trials":
            reliability_values = self.all_trial_reliability[envoption]
        elif reliability_type == "session_average":
            reliability_values = self.session_average_reliability[envoption]
        else:
            raise ValueError("reliability_type must be 'all_trials' or 'session_average'")

        min_threshold = np.percentile(reliability_values, min_percentile)
        max_threshold = np.percentile(reliability_values, max_percentile)
        idx_in_percentile = (reliability_values > min_threshold) & (reliability_values < max_threshold)

        idx_red = np.any(np.stack([ired for ired in self.extras[envoption]["idx_red"]]), axis=0)
        if not red_cells:
            idx_red = ~idx_red

        idx_keepers = np.where(np.logical_and(idx_in_percentile, idx_red))[0]

        idx_keepers_reliability = reliability_values[idx_keepers]
        idx_to_ordered_reliability = np.argsort(idx_keepers_reliability)
        idx_keepers = idx_keepers[idx_to_ordered_reliability]

        return idx_keepers

    def _com(self, data, axis=-1):
        x = np.arange(data.shape[axis])
        com = np.sum(data * x, axis=axis) / (np.sum(data, axis=axis) + 1e-10)
        com[np.any(data < 0, axis=axis)] = np.nan
        return com

    def plot(self, viewer, state):
        envoption = state["envoption"]
        min_percentile = state["percentile_range"][0]
        max_percentile = state["percentile_range"][1]
        red_cells = state["red_cells"]
        roi_idx = state["roi_idx"]
        reliability_type = state["reliability_type"]
        spkmaps = self.spkmaps[envoption]
        idxs = self._gather_idxs(envoption, min_percentile, max_percentile, red_cells, reliability_type)

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

        relcor = ",".join([f"{ses[idx_roi_to_plot]:.2f}" for ses in self.extras[envoption]["relcor"]])
        relloo = ",".join([f"{ses[idx_roi_to_plot]:.2f}" for ses in self.extras[envoption]["relloo"]])

        roi_trajectory = self._make_roi_trajectory(spkmaps, idx_roi_to_plot)[0]

        idx_not_nan = ~np.any(np.isnan(roi_trajectory), axis=1)
        pfmax = np.where(idx_not_nan, np.max(roi_trajectory, axis=1), np.nan)
        pfcom = np.where(idx_not_nan, self._com(roi_trajectory, axis=1), np.nan)
        pfloc = np.where(idx_not_nan, np.argmax(roi_trajectory, axis=1), np.nan)

        cmap = mpl.colormaps["gray_r"]
        cmap.set_bad((1, 0.8, 0.8))  # Light red color

        fig = plt.figure(figsize=(7.5, 7.5))
        fig.set_constrained_layout(True)
        gs = fig.add_gridspec(4, 3)  # , left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.15, hspace=0.1)

        ax = fig.add_subplot(gs[:3, 0])
        ax.imshow(roi_trajectory, aspect="auto", interpolation="none", cmap=cmap, vmin=0, vmax=10)

        ax.set_xlim(0, roi_trajectory.shape[1])
        ax.set_ylim(roi_trajectory.shape[0], 0)
        ax.set_ylabel("Trial")
        ax.set_yticks([])
        ax.set_xlabel("Virtual Position")
        ax.set_title("PF Activity")

        alpha_values = np.where(~np.isnan(pfmax), pfmax / np.nanmax(pfmax), 0)
        ax = fig.add_subplot(gs[:3, 1])
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

        ax = fig.add_subplot(gs[:3, 2])
        ax.scatter(pfmax, range(len(pfmax)), color="k", s=10, alpha=alpha_values)
        ax.set_xlim(0, np.nanmax(pfmax))
        ax.set_ylim(roi_trajectory.shape[0], 0)
        ax.set_title("PF Amplitude")
        ax.set_yticks([])
        ax.set_xlabel("Activity (sigma)")

        ax = fig.add_subplot(gs[3:, :-1])
        ax.scatter(
            self.all_trial_reliability[envoption],
            self.session_average_reliability[envoption],
            s=10,
            color="lightgray",
            alpha=0.3,
            label="All Tracked",
        )
        ax.scatter(
            self.all_trial_reliability[envoption][idxs],
            self.session_average_reliability[envoption][idxs],
            s=10,
            color="k",
            alpha=0.5,
            label="Selected",
        )
        ax.scatter(
            self.all_trial_reliability[envoption][idx_roi_to_plot],
            self.session_average_reliability[envoption][idx_roi_to_plot],
            s=40,
            color="r",
            alpha=1,
            label="Selected ROI",
        )
        ax.set_title("Across-Session\nReliability Metrics")
        ax.set_xlabel("On All Trials")
        ax.set_ylabel("On Session Average")
        # ax.legend(loc="upper left")

        fraction_active_name = self._fraction_active_name(state["fraction_active_method"], state["fraction_active_type"])
        fraction_active_data = self.fraction_active[envoption][fraction_active_name]
        compare_with_reliability = state["compare_with_reliability"]

        ax = fig.add_subplot(gs[3:, -1])
        if compare_with_reliability:

            colors = plt.cm.coolwarm(np.linspace(0, 1, fraction_active_data.shape[1]))
            colors = colors[:, :3]

            reliability_data = np.stack(self.extras[envoption]["relloo"], axis=1)
            # ax.scatter(reliability_data, fraction_active_data, s=10, color="lightgray", alpha=0.3)
            # ax.scatter(reliability_data[idxs], fraction_active_data[idxs], s=10, color="k", alpha=0.3)
            # ax.scatter(reliability_data[idx_roi_to_plot], fraction_active_data[idx_roi_to_plot], s=40, color="r", alpha=1)
            # ax.scatter(reliability_data[idxs, 0], fraction_active_data[idxs, 0], s=10, color="k", alpha=1, label="First Session")

            # ax.plot(reliability_data[idxs].T, fraction_active_data[idxs].T, s=10, color="k", alpha=0.3)
            if state["show_all_comparison"]:
                for i in range(fraction_active_data.shape[1]):
                    ax.plot(
                        reliability_data[idxs, i].T,
                        fraction_active_data[idxs, i].T,
                        color=colors[i],
                        marker=".",
                        markersize=5,
                        alpha=0.3,
                    )
            ax.plot(reliability_data[idx_roi_to_plot], fraction_active_data[idx_roi_to_plot], linewidth=1.5, color="k", zorder=1000)
            for i in range(fraction_active_data.shape[1]):
                ax.plot(
                    reliability_data[idx_roi_to_plot, i].T,
                    fraction_active_data[idx_roi_to_plot, i].T,
                    color=colors[i],
                    marker=".",
                    markersize=10,
                    alpha=1,
                    zorder=2000,
                )
            ax.plot(
                reliability_data[idx_roi_to_plot, 0].T,
                fraction_active_data[idx_roi_to_plot, 0].T,
                color="k",
                marker=".",
                markersize=10,
                alpha=1,
                label="First Session",
                zorder=3000,
            )
            # ax.legend(loc="best")
            ax.set_xlabel("Reliability Each Session")
            ax.set_ylabel("Fraction Active Each Session")

        else:
            # ax.plot(range(fraction_active_data.shape[1]), fraction_active_data.T, color="lightgray", alpha=0.3, marker="o", markersize=2)
            ax.plot(range(fraction_active_data.shape[1]), fraction_active_data[idxs].T, color="k", alpha=0.5, marker="o", markersize=2)
            ax.plot(range(fraction_active_data.shape[1]), fraction_active_data[idx_roi_to_plot].T, color="r", alpha=1, marker="o", markersize=5)
            ax.set_xlabel("Session")
            ax.set_ylabel("Fraction Active")
            ax.set_xlim(-0.5, fraction_active_data.shape[1] - 0.5)

        ax.set_title("Fraction Trials Active")

        fig.suptitle(f"Mouse: {self.mouse_name}, Env: {envoption}, ROI: {idx_roi_to_plot}\nRelCor: {relcor}\nRelLoo: {relloo}")
        return fig


class PlaceFieldViewer(InteractiveViewer):
    def __init__(self, placefield_loader: PlaceFieldLoader):
        self.placefield_loader = placefield_loader

        # Add interactive parameters
        self.add_selection("envoption", options=["first", "second"], value="second")
        self.add_float_range("percentile_range", value=(90, 100), min_value=0, max_value=100)
        self.add_boolean("red_cells", value=False)
        self.add_selection("reliability_type", options=["all_trials", "session_average"], value="all_trials")
        self.add_selection("fraction_active_method", options=["max", "mean", "rms"], value="rms")
        self.add_selection("fraction_active_type", options=["gini", "entropy", "relative_entropy"], value="gini")
        self.add_boolean("compare_with_reliability", value=False)
        self.add_boolean("show_all_comparison", value=False)

        self.set_roi_options(self.get_state())
        self.on_change(["envoption", "percentile_range", "red_cells"], self.set_roi_options)

    def set_roi_options(self, state):
        envoption = state["envoption"]
        min_percentile = state["percentile_range"][0]
        max_percentile = state["percentile_range"][1]
        red_cells = state["red_cells"]
        reliability_type = state["reliability_type"]

        idx_roi_options = self.placefield_loader._gather_idxs(envoption, min_percentile, max_percentile, red_cells, reliability_type)
        if "roi_idx" in self.parameters:
            self.update_selection("roi_idx", options=list(idx_roi_options))
        else:
            self.add_selection("roi_idx", options=list(idx_roi_options), value=idx_roi_options[0])


def get_viewer(placefield_loader: PlaceFieldLoader):
    viewer = PlaceFieldViewer(placefield_loader)
    viewer.set_plot(placefield_loader.plot)
    return viewer
