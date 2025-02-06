from .. import analysis, tracking, helpers
from ..metrics import FractionActive, KernelDensityEstimator, plot_contours
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from syd import Viewer


class PlaceFieldLoader:
    def __init__(self, mouse_name, sesmethod="all", keep_planes=[1, 2, 3, 4], onefile=None, summary_only=False):
        self.mouse_name = mouse_name
        self.env_selection = {}
        self.idx_ses_selection = {}
        self.spkmaps = {}
        self.extras = {}
        self.all_trial_reliability = {}
        self.session_average_reliability = {}
        self.fraction_active = {}
        self.kde_ctl = {}
        self.kde_red = {}
        self.onefile = onefile
        self.summary_only = summary_only

        self.sesmethod = sesmethod
        self.keep_planes = keep_planes

        self.reliability_range = (-1.0, 1.0)
        self.fraction_range = (0.0, 1.0)
        self.nbins = 101

        self._prepare_mouse_data(mouse_name, onefile=onefile)

    def _fraction_active_name(self, activity_method: str, fraction_method: str) -> str:
        return "_".join([activity_method, fraction_method])

    def _prepare_mouse_data(self, mouse_name, onefile=None):
        tracked = False if self.summary_only else True

        keep_planes = self.keep_planes
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=keep_planes, onefile=onefile)

        for envoption in ["first", "second"]:
            envnum = pcm.env_selector(envmethod=envoption)
            idx_ses = pcm.idx_ses_selector(envnum=envnum, sesmethod=self.sesmethod)
            spkmaps, extras = pcm.get_spkmaps(envnum=envnum, idx_ses=idx_ses, trials="full", average=False, tracked=tracked, onefile=onefile)
            self.env_selection[envoption] = (envoption, idx_ses)
            self.idx_ses_selection[envoption] = idx_ses
            self.spkmaps[envoption] = spkmaps
            self.extras[envoption] = extras
            self.fraction_active[envoption] = {}
            self.kde_ctl[envoption] = {}
            self.kde_red[envoption] = {}
            for amethod, fmethod in FractionActive.get_combinations():

                # Get the fraction active data for each spkmap
                cdata = []
                for spkmap in spkmaps:
                    cdata.append(FractionActive.compute(spkmap, activity_axis=2, fraction_axis=1, activity_method=amethod, fraction_method=fmethod))

                # Stack it
                if tracked:
                    cdata = np.stack(cdata, axis=0)
                self.fraction_active[envoption][self._fraction_active_name(amethod, fmethod)] = cdata

                fraction_active_data = cdata
                reliability_data = extras["relloo"]
                if tracked:
                    reliability_data = np.stack(reliability_data, axis=0)

                if tracked:
                    idx_red = np.any(np.stack([ired for ired in extras["idx_red"]]), axis=0)
                    idx_red = np.tile(np.expand_dims(idx_red, axis=1), (fraction_active_data.shape[1]), 1)
                else:
                    idx_red = [ired for ired in extras["idx_red"]]

                kde_ctl = []
                kde_red = []
                for i in range(len(spkmaps)):
                    kde_ctl.append(
                        KernelDensityEstimator(
                            reliability_data[i][~idx_red[i]],
                            fraction_active_data[i][~idx_red[i]],
                            xrange=self.reliability_range,
                            yrange=self.fraction_range,
                            nbins=self.nbins,
                        ).fit()
                    )
                    kde_red.append(
                        KernelDensityEstimator(
                            reliability_data[i][idx_red[i]],
                            fraction_active_data[i][idx_red[i]],
                            xrange=self.reliability_range,
                            yrange=self.fraction_range,
                            nbins=self.nbins,
                        ).fit()
                    )
                self.kde_ctl[envoption][self._fraction_active_name(amethod, fmethod)] = kde_ctl
                self.kde_red[envoption][self._fraction_active_name(amethod, fmethod)] = kde_red

            if not self.summary_only:
                all_trial_reliability = helpers.reliability_loo(np.concatenate(spkmaps, axis=1), weighted=True)
                session_average_reliability = helpers.reliability_loo(
                    np.stack([np.nanmean(spkmap, axis=1) for spkmap in spkmaps], axis=1), weighted=True
                )
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
        idx_to_ordered_reliability = np.argsort(-idx_keepers_reliability)
        idx_keepers = idx_keepers[idx_to_ordered_reliability]

        return idx_keepers

    def _com(self, data, axis=-1):
        x = np.arange(data.shape[axis])
        com = np.sum(data * x, axis=axis) / (np.sum(data, axis=axis) + 1e-10)
        com[np.any(data < 0, axis=axis)] = np.nan
        return com

    def plot_cell_activity(self, state):
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

        relcor = ",".join([f"{ses[roi_idx]:.2f}" for ses in self.extras[envoption]["relcor"]])
        relloo = ",".join([f"{ses[roi_idx]:.2f}" for ses in self.extras[envoption]["relloo"]])

        roi_trajectory = self._make_roi_trajectory(spkmaps, roi_idx)[0]

        idx_not_nan = ~np.any(np.isnan(roi_trajectory), axis=1)
        pfmax = np.where(idx_not_nan, np.max(roi_trajectory, axis=1), np.nan)
        pfloc = np.where(idx_not_nan, np.argmax(roi_trajectory, axis=1), np.nan)

        cmap = mpl.colormaps["gray_r"]
        cmap.set_bad((1, 0.8, 0.8))  # Light red color

        fig = plt.figure(figsize=(7.5, 7.5), layout="constrained")
        gs = fig.add_gridspec(4, 3)

        ax = fig.add_subplot(gs[:3, 0])
        ax.imshow(roi_trajectory, aspect="auto", interpolation="none", cmap=cmap, vmin=0, vmax=state["vmax"])

        ax.set_xlim(0, roi_trajectory.shape[1])
        ax.set_ylim(roi_trajectory.shape[0], 0)
        ax.set_ylabel("Trial")
        ax.set_yticks([])
        ax.set_xlabel("Virtual Position")
        ax.set_title("PF Activity")

        alpha_values = np.where(~np.isnan(pfmax), pfmax / np.nanmax(pfmax), 0)
        ax = fig.add_subplot(gs[:3, 1])
        ax.scatter(pfloc, range(len(pfloc)), s=10, color="k", alpha=alpha_values, linewidth=2)
        ax.set_xlim(0, roi_trajectory.shape[1])
        ax.set_ylim(roi_trajectory.shape[0], 0)
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
            self.all_trial_reliability[envoption][roi_idx],
            self.session_average_reliability[envoption][roi_idx],
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
        fraction_active_data = self.fraction_active[envoption][fraction_active_name].T
        compare_with_reliability = state["compare_with_reliability"]

        ax = fig.add_subplot(gs[3:, -1])
        if compare_with_reliability:

            colors = plt.cm.coolwarm(np.linspace(0, 1, fraction_active_data.shape[1]))
            colors = colors[:, :3]

            reliability_data = np.stack(self.extras[envoption]["relloo"], axis=1)
            # ax.scatter(reliability_data, fraction_active_data, s=10, color="lightgray", alpha=0.3)
            # ax.scatter(reliability_data[idxs], fraction_active_data[idxs], s=10, color="k", alpha=0.3)
            # ax.scatter(reliability_data[roi_idx], fraction_active_data[roi_idx], s=40, color="r", alpha=1)
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
                        linestyle="none",
                    )
            ax.plot(reliability_data[roi_idx], fraction_active_data[roi_idx], linewidth=1.5, color="k", zorder=1000)
            for i in range(fraction_active_data.shape[1]):
                ax.plot(
                    reliability_data[roi_idx, i].T,
                    fraction_active_data[roi_idx, i].T,
                    color=colors[i],
                    marker=".",
                    markersize=10,
                    alpha=1,
                    zorder=2000,
                )
            ax.plot(
                reliability_data[roi_idx, 0].T,
                fraction_active_data[roi_idx, 0].T,
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
            ax.set_xlim(-1.0, 1.0)  # Relloo, so it's an average correlation coefficient
            ax.set_ylim(0, 1.0)  # Fractional measures so always between 0 and 1

        else:
            # ax.plot(range(fraction_active_data.shape[1]), fraction_active_data.T, color="lightgray", alpha=0.3, marker="o", markersize=2)
            ax.plot(range(fraction_active_data.shape[1]), fraction_active_data[idxs].T, color="k", alpha=0.5, marker="o", markersize=2)
            ax.plot(range(fraction_active_data.shape[1]), fraction_active_data[roi_idx].T, color="r", alpha=1, marker="o", markersize=5)
            ax.set_xlabel("Session")
            ax.set_ylabel("Fraction Active")
            ax.set_xlim(-0.5, fraction_active_data.shape[1] - 0.5)

        ax.set_title("Fraction Trials Active")

        fig.suptitle(f"Mouse: {self.mouse_name}, Env: {envoption}, ROI: {roi_idx}\nRelCor: {relcor}\nRelLoo: {relloo}")
        return fig

    def plot_summary(self, state):
        envoption = state["envoption"]
        fraction_active_name = self._fraction_active_name(state["fraction_active_method"], state["fraction_active_type"])
        idx_ses = state["idx_ses"]
        min_level, max_level = state["level_range"]
        num_levels = state["num_levels"]

        levels = np.linspace(min_level, max_level, num_levels) if num_levels > 1 else [(max_level + min_level) / 2]
        kde_ctl = self.kde_ctl[envoption][fraction_active_name][idx_ses]
        kde_red = self.kde_red[envoption][fraction_active_name][idx_ses]
        contours_ctl = kde_ctl.contours(levels)
        contours_red = kde_red.contours(levels)

        contour_cmap = plt.cm.coolwarm
        colors = contour_cmap(np.linspace(0, 1, num_levels))

        fig, ax = plt.subplots(1, 3, figsize=(9, 3.5), layout="constrained")

        # Plot the kde estimates
        ctl_plot_data = kde_ctl.plot_data
        red_plot_data = kde_red.plot_data
        max_pdf = np.max([np.max(ctl_plot_data), np.max(red_plot_data)])
        ax[0].imshow(ctl_plot_data, cmap="gray_r", extent=kde_ctl.extent, aspect="auto", vmin=0, vmax=max_pdf)
        ax[1].imshow(red_plot_data, cmap="gray_r", extent=kde_red.extent, aspect="auto", vmin=0, vmax=max_pdf)

        # Plot the contours
        for ctl, red, color in zip(contours_ctl, contours_red, colors):
            plot_contours(ctl, ax=ax[0], color=color, linewidth=1.0, alpha=1.0)
            plot_contours(red, ax=ax[1], color=color, linewidth=1.0, alpha=1.0)

        # Plot the difference in the distributions
        difference = kde_red.plot_data - kde_ctl.plot_data
        max_diff = np.max(np.abs(difference))
        ax[2].imshow(difference, cmap="bwr", extent=kde_red.extent, aspect="auto", vmin=-max_diff, vmax=max_diff)

        for a in ax:
            a.set_xlabel("Reliability")
            a.set_ylabel("Fraction Active")
            a.set_xlim(self.reliability_range)
            a.set_ylim(self.fraction_range)

        ax[0].set_title(f"Control Cells N={self.kde_ctl[envoption][fraction_active_name][idx_ses].x.size}")
        ax[1].set_title(f"Red Cells N={self.kde_red[envoption][fraction_active_name][idx_ses].x.size}")
        ax[2].set_title("Difference (Red - Control)")

        # Boundaries for the colorbar
        inset = ax[0].inset_axes([0.05, 0.57, 0.1, 0.4])
        step_size = levels[1] - levels[0]
        color_lims = [min_level - step_size / 2, max_level + step_size / 2]
        yticks = levels
        max_ticks = 5
        if num_levels > max_ticks:
            from math import ceil

            step = ceil((num_levels - 1) / (max_ticks - 1))  # -1s ensure first/last included
            indices = list(range(0, num_levels - 1, step)) + [num_levels - 1]
            yticks = yticks[indices]
        inset.imshow(np.flipud(np.reshape(colors, (num_levels, 1, 4))), aspect="auto", extent=[0, 1, color_lims[0], color_lims[1]])
        inset.set_xticks([])
        inset.set_yticks(yticks)
        inset.set_yticklabels([f"{level:.2f}" for level in yticks])
        inset.yaxis.tick_right()

        fig.suptitle(
            f"Mouse: {self.mouse_name}, Env: {envoption}, ActivityMethod: {state['fraction_active_method']}, FractionMethod: {state['fraction_active_type']}"
        )
        return fig


class PlaceFieldViewer(Viewer):
    def __init__(self, placefield_loader: PlaceFieldLoader):
        self.placefield_loader = placefield_loader

        # Add interactive parameters
        self.add_selection("envoption", options=["first", "second"], value="second")
        self.add_float_range("percentile_range", value=(90, 100), min_value=0, max_value=100)
        self.add_boolean("red_cells", value=False)
        self.add_selection("reliability_type", options=["all_trials", "session_average"], value="all_trials")
        self.add_selection("fraction_active_method", options=FractionActive.activity_methods, value="rms")
        self.add_selection("fraction_active_type", options=FractionActive.fraction_methods, value="participation")
        self.add_boolean("compare_with_reliability", value=False)
        self.add_boolean("show_all_comparison", value=False)
        self.add_float("vmax", value=5.0, min_value=1.0, max_value=20.0)

        self.set_roi_options(self.state)
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


class SummaryViewer(Viewer):
    def __init__(self, placefield_loader: PlaceFieldLoader):
        self.placefield_loader = placefield_loader

        # Add interactive parameters
        self.add_selection("envoption", options=["first", "second"], value="second")
        self.add_integer("idx_ses", value=0, min_value=0, max_value=1)
        self.add_float_range("level_range", value=(0.5, 0.9), min_value=0.0, max_value=1.0, step=0.01)
        self.add_integer("num_levels", value=10, min_value=1, max_value=100)
        self.add_selection("fraction_active_method", options=FractionActive.activity_methods, value="rms")
        self.add_selection("fraction_active_type", options=FractionActive.fraction_methods, value="participation")

        self.set_idx_ses(self.state)
        self.on_change("envoption", self.set_idx_ses)

    def set_idx_ses(self, state):
        envoption = state["envoption"]
        num_ses = len(self.placefield_loader.spkmaps[envoption])
        self.update_integer("idx_ses", max_value=num_ses - 1)


def get_cell_viewer(placefield_loader: PlaceFieldLoader):
    viewer = PlaceFieldViewer(placefield_loader)
    viewer.set_plot(placefield_loader.plot_cell_activity)
    return viewer


def get_summary_viewer(placefield_loader: PlaceFieldLoader):
    viewer = SummaryViewer(placefield_loader)
    viewer.set_plot(placefield_loader.plot_summary)
    return viewer
