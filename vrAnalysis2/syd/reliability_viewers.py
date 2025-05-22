from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import joblib

from syd import Viewer
from ..tracking import Tracker
from ..processors.spkmaps import SpkmapProcessor
from ..helpers import (
    color_violins,
    fractional_histogram,
    edge2center,
    vectorCorrelation,
    format_spines,
    save_figure,
    blinded_study_legend,
    get_mouse_colors,
)
from ..helpers.vrsupport import _jit_reliability_loo
from ..multisession import MultiSessionSpkmaps
from ..metrics import FractionActive
from ..database import get_database

from ..files import repo_path, analysis_path
import sys

sys.path.append(str(repo_path()))
from scripts.before_the_reveal import get_reliability


def figure_dir(folder: str) -> Path:
    return repo_path() / "figures" / "before_the_reveal" / folder


class ReliabilityTrajectory(Viewer):
    def __init__(self, tracked_mice: list[str], try_cache: bool = True):
        self.tracked_mice = list(tracked_mice)
        self.trackers = {mouse: None for mouse in self.tracked_mice}

        reliability_methods = ["leave_one_out", "correlation"]
        if try_cache:
            cache_file = self.cache_path()
            if cache_file.exists():
                self.reliability_data = joblib.load(cache_file)
        else:
            self.reliability_data = {}
            for method in reliability_methods:
                self.reliability_data[method] = {}
                for mouse in self.tracked_mice:
                    self.reliability_data[method][mouse] = get_reliability(
                        self.trackers[mouse],
                        reliability_method=method,
                        exclude_environments=[-1],
                        clear_one=True,
                    )

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "correlation"])
        self.add_selection("smooth_width", value=5, options=[5])
        self.add_boolean("use_session_filters", value=False)
        self.add_boolean("subsample_ctl", value=False)
        self.add_integer("num_subsamples", value=20, min=1, max=100)
        self.add_boolean("show_distribution", value=True)
        self.add_float("widths", value=0.5, min=0.1, max=3.0)

    def cache_path(self) -> Path:
        return analysis_path() / "before_the_reveal_temp_data" / "reliability_quantile_summary.joblib"

    def get_tracker(self, mouse: str) -> Tracker:
        if self.trackers[mouse] is None:
            self.trackers[mouse] = Tracker(mouse)
        return self.trackers[mouse]

    def get_environments(self, track: Tracker) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        environments = np.unique(np.concatenate([session.environments for session in track.sessions]))
        return environments

    def plot(self, state):
        # Gather data to plot
        use_session_filters = state["use_session_filters"]
        reliability = self.reliability_data[state["reliability_method"]][state["mouse"]]
        environments = reliability["environments"]
        sessions = reliability["sessions"]
        if use_session_filters:
            reliability_ctl = reliability["reliability_ctl"]
            reliability_red = reliability["reliability_red"]
        else:
            reliability_ctl = reliability["reliability_ctl_all"]
            reliability_red = reliability["reliability_red_all"]

        all_sessions = sorted(list(set(np.concatenate(list(sessions.values())))))
        num_cells_ctl = [None] * len(all_sessions)
        num_cells_red = [None] * len(all_sessions)
        for ises, ses in enumerate(all_sessions):
            for env in environments:
                if num_cells_ctl[ises] is None or num_cells_red[ises] is None:
                    if ses in sessions[env]:
                        idx_to_ses = sessions[env].index(ses)
                        num_cells_ctl[ises] = len(reliability_ctl[env][idx_to_ses])
                        num_cells_red[ises] = len(reliability_red[env][idx_to_ses])

        # Create subsampled control cells
        # First select random sample of control cells to match number of red cells
        # Then average their reliability
        # We'll plot the average average of subsampled cells
        if state["subsample_ctl"]:
            num_subsamples = state["num_subsamples"]
            subsampled_ctl = {env: None for env in environments}
            for env in environments:
                subsampled_ctl[env] = [0.0 for _ in range(len(reliability_ctl[env]))]
                for ises, ctl_data in enumerate(reliability_ctl[env]):
                    num_red_this_session = len(reliability_red[env][ises])
                    for _ in range(num_subsamples):
                        c_subsample = np.random.choice(ctl_data, size=num_red_this_session, replace=False)
                        subsampled_ctl[env][ises] += np.mean(c_subsample) / num_subsamples

        figwidth = 3
        figheight = 3
        fig, ax = plt.subplots(
            2, len(environments) + 1, figsize=((len(environments) + 1) * figwidth, 2 * figheight), layout="constrained", sharex=True
        )
        for ienv, env in enumerate(environments):
            if state["show_distribution"]:
                for ises, (ctl, red) in enumerate(zip(reliability_ctl[env], reliability_red[env])):
                    ses_number = sessions[env][ises]
                    parts = ax[0, ienv].violinplot(ctl, positions=[ses_number], widths=state["widths"], showextrema=False, side="low")
                    color_violins(parts, facecolor=("k", 0.1), linecolor="k")
                    parts = ax[0, ienv].violinplot(red, positions=[ses_number], widths=state["widths"], showextrema=False, side="high")
                    color_violins(parts, facecolor=("r", 0.1), linecolor="r")
            ctl_mean = np.array([np.mean(r) for r in reliability_ctl[env]])
            red_mean = np.array([np.mean(r) for r in reliability_red[env]])
            if state["subsample_ctl"]:
                ax[0, ienv].plot(sessions[env], subsampled_ctl[env], color="blue", label="CTL-Subsample", marker="o")
            ax[0, ienv].plot(sessions[env], ctl_mean, color="black", label="CTL", marker="o")
            ax[0, ienv].plot(sessions[env], red_mean, color="red", label="RED", marker="o")
            ax[0, ienv].set_title(f"Environment {env}")

            ax[1, ienv].plot(sessions[env], red_mean - ctl_mean, color="blue", label="DIFF", marker="o")
            ax[1, ienv].axhline(0, color="k", linestyle="--")
            ylim = ax[1, ienv].get_ylim()
            yscale = max(abs(ylim[0]), abs(ylim[1]))
            ax[1, ienv].set_ylim(-yscale, yscale)
            ax[1, ienv].set_title("Difference")
            ax[1, ienv].set_xlabel("Session")

        ax[0, 0].set_ylabel("Reliability")
        ax[1, 0].set_ylabel("RED - CTL")

        all_sessions = sorted(list(set(np.concatenate(list(sessions.values())))))
        ax[0, len(environments)].plot(all_sessions, num_cells_ctl, color="black", label="CTL", marker="o")
        ax[1, len(environments)].plot(all_sessions, num_cells_red, color="red", label="RED", marker="o")
        ax[0, len(environments)].set_xlabel("Session")
        ax[0, len(environments)].set_ylabel("Number of cells")
        ax[1, len(environments)].set_xlabel("Session")
        ax[1, len(environments)].set_ylabel("Number of cells")
        ax[0, len(environments)].set_title("CTL")
        ax[1, len(environments)].set_title("RED")

        fig.suptitle(f"Mouse {state['mouse']} - Smooth width {state['smooth_width']} - {state['reliability_method']}")
        return fig


class FractionActiveTrajectory(Viewer):
    def __init__(self, tracked_mice: list[str], try_cache: bool = True):
        self.tracked_mice = list(tracked_mice)
        self.trackers = {mouse: None for mouse in self.tracked_mice}

        reliability_methods = ["leave_one_out", "correlation"]

        if try_cache:
            cache_file = self.cache_path()
            if cache_file.exists():
                self.reliability_data = joblib.load(cache_file)
        else:
            self.reliability_data = {}
            for method in reliability_methods:
                self.reliability_data[method] = {}
                for mouse in self.tracked_mice:
                    self.reliability_data[method][mouse] = get_reliability(
                        self.trackers[mouse],
                        reliability_method=method,
                        exclude_environments=[-1],
                        clear_one=True,
                    )

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "correlation"])
        self.add_selection("smooth_width", value=5, options=[5])
        self.add_boolean("use_session_filters", value=False)
        self.add_boolean("subsample_ctl", value=False)
        self.add_integer("num_subsamples", value=20, min=1, max=100)
        self.add_boolean("show_distribution", value=True)
        self.add_float("widths", value=0.5, min=0.1, max=3.0)

    def cache_path(self) -> Path:
        return analysis_path() / "before_the_reveal_temp_data" / "reliability_quantile_summary.joblib"

    def get_tracker(self, mouse: str) -> Tracker:
        if self.trackers[mouse] is None:
            self.trackers[mouse] = Tracker(mouse)
        return self.trackers[mouse]

    def get_environments(self, track: Tracker) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        environments = np.unique(np.concatenate([session.environments for session in track.sessions]))
        return environments

    def plot(self, state):
        # Gather data to plot
        use_session_filters = state["use_session_filters"]
        reliability = self.reliability_data[state["reliability_method"]][state["mouse"]]
        environments = reliability["environments"]
        sessions = reliability["sessions"]
        if use_session_filters:
            fraction_active_ctl = reliability["fraction_active_ctl"]
            fraction_active_red = reliability["fraction_active_red"]
        else:
            fraction_active_ctl = reliability["fraction_active_ctl_all"]
            fraction_active_red = reliability["fraction_active_red_all"]

            # Gather data to plot
        use_session_filters = state["use_session_filters"]
        reliability = self.reliability_data[state["reliability_method"]][state["mouse"]]
        environments = reliability["environments"]
        sessions = reliability["sessions"]
        if use_session_filters:
            reliability_ctl = reliability["reliability_ctl"]
            reliability_red = reliability["reliability_red"]
        else:
            reliability_ctl = reliability["reliability_ctl_all"]
            reliability_red = reliability["reliability_red_all"]

        all_sessions = sorted(list(set(np.concatenate(list(sessions.values())))))
        num_cells_ctl = [None] * len(all_sessions)
        num_cells_red = [None] * len(all_sessions)
        for ises, ses in enumerate(all_sessions):
            for env in environments:
                if num_cells_ctl[ises] is None or num_cells_red[ises] is None:
                    if ses in sessions[env]:
                        idx_to_ses = sessions[env].index(ses)
                        num_cells_ctl[ises] = len(reliability_ctl[env][idx_to_ses])
                        num_cells_red[ises] = len(reliability_red[env][idx_to_ses])

        # Create subsampled control cells
        # First select random sample of control cells to match number of red cells
        # Then average their reliability
        # We'll plot the average average of subsampled cells
        if state["subsample_ctl"]:
            num_subsamples = state["num_subsamples"]
            subsampled_ctl = {env: None for env in environments}
            for env in environments:
                subsampled_ctl[env] = [0.0 for _ in range(len(fraction_active_ctl[env]))]
                for ises, ctl_data in enumerate(fraction_active_ctl[env]):
                    num_red_this_session = len(fraction_active_red[env][ises])
                    for _ in range(num_subsamples):
                        c_subsample = np.random.choice(ctl_data, size=num_red_this_session, replace=False)
                        subsampled_ctl[env][ises] += np.mean(c_subsample) / num_subsamples

        figwidth = 3
        figheight = 3
        fig, ax = plt.subplots(
            2, len(environments) + 1, figsize=((len(environments) + 1) * figwidth, 2 * figheight), layout="constrained", sharex=True
        )
        for ienv, env in enumerate(environments):
            if state["show_distribution"]:
                for ises, (ctl, red) in enumerate(zip(fraction_active_ctl[env], fraction_active_red[env])):
                    ses_number = sessions[env][ises]
                    parts = ax[0, ienv].violinplot(ctl, positions=[ses_number], widths=state["widths"], showextrema=False, side="low")
                    color_violins(parts, facecolor=("k", 0.1), linecolor="k")
                    parts = ax[0, ienv].violinplot(red, positions=[ses_number], widths=state["widths"], showextrema=False, side="high")
                    color_violins(parts, facecolor=("r", 0.1), linecolor="r")
            ctl_mean = np.array([np.mean(r) for r in fraction_active_ctl[env]])
            red_mean = np.array([np.mean(r) for r in fraction_active_red[env]])
            if state["subsample_ctl"]:
                ax[0, ienv].plot(sessions[env], subsampled_ctl[env], color="blue", label="CTL-Subsample", marker="o")
            ax[0, ienv].plot(sessions[env], ctl_mean, color="black", label="CTL", marker="o")
            ax[0, ienv].plot(sessions[env], red_mean, color="red", label="RED", marker="o")
            ax[0, ienv].set_title(f"Environment {env}")

            ax[1, ienv].plot(sessions[env], red_mean - ctl_mean, color="blue", label="DIFF", marker="o")
            ax[1, ienv].axhline(0, color="k", linestyle="--")
            ylim = ax[1, ienv].get_ylim()
            yscale = max(abs(ylim[0]), abs(ylim[1]))
            ax[1, ienv].set_ylim(-yscale, yscale)
            ax[1, ienv].set_title("Difference")
            ax[1, ienv].set_xlabel("Session")

        ax[0, 0].set_ylabel("Fraction Active")
        ax[1, 0].set_ylabel("RED - CTL")

        all_sessions = sorted(list(set(np.concatenate(list(sessions.values())))))
        ax[0, len(environments)].plot(all_sessions, num_cells_ctl, color="black", label="CTL", marker="o")
        ax[1, len(environments)].plot(all_sessions, num_cells_red, color="red", label="RED", marker="o")
        ax[0, len(environments)].set_xlabel("Session")
        ax[0, len(environments)].set_ylabel("Number of cells")
        ax[1, len(environments)].set_xlabel("Session")
        ax[1, len(environments)].set_ylabel("Number of cells")
        ax[0, len(environments)].set_title("CTL")
        ax[1, len(environments)].set_title("RED")

        fig.suptitle(f"Mouse {state['mouse']} - {state['reliability_method']}")
        return fig


class ReliabilitySingleSession(Viewer):
    def __init__(self, tracked_mice: list[str]):
        self.tracked_mice = list(tracked_mice)

        self.tracked_mice = list(tracked_mice)
        self._reliability = {
            method: {smooth_width: {mouse: None for mouse in self.tracked_mice} for smooth_width in [1, 5]}
            for method in ["leave_one_out", "mse", "correlation"]
        }
        self.trackers = {mouse: None for mouse in self.tracked_mice}

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_integer("session", value=0, min=0, max=1)
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "mse", "correlation"])
        self.add_selection("smooth_width", value=1, options=[1, 5])
        self.add_boolean("use_session_filters", value=True)
        self.add_boolean("subsample_ctl", value=False)
        self.add_integer("num_subsamples", value=20, min=1, max=100)

        self.on_change("mouse", self.reset_mouse)

        self.reset_mouse(self.state)

    def reset_mouse(self, state):
        track = self.get_tracker(state["mouse"])
        self.update_integer("session", value=0, max=len(track.sessions) - 1)

    def get_tracker(self, mouse: str) -> Tracker:
        if self.trackers[mouse] is None:
            self.trackers[mouse] = Tracker(mouse)
        return self.trackers[mouse]

    def get_environments(self, mouse: str) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        track = self.get_tracker(mouse)
        environments = np.unique(np.concatenate([session.environments for session in track.sessions]))
        return environments

    def get_reliability(self, mouse: str, method: str, smooth_width: float, exclude_environments: Optional[list[int] | int] = [-1]) -> dict:
        """Get reliability data for all tracked sessions"""
        if self._reliability[method][smooth_width][mouse] is not None:
            return self._reliability[method][smooth_width][mouse]

        track = self.get_tracker(mouse)
        environments = list(self.get_environments(mouse))
        reliability_ctl = {env: [] for env in environments}
        reliability_red = {env: [] for env in environments}
        reliability_ctl_all = {env: [] for env in environments}
        reliability_red_all = {env: [] for env in environments}
        sessions = {env: [] for env in environments}
        num_cells_ctl_all = []
        num_cells_red_all = []
        num_cells_ctl = []
        num_cells_red = []

        for isession, session in enumerate(tqdm(track.sessions)):
            envs = session.environments
            idx_rois = session.idx_rois
            idx_red_all = session.get_red_idx()
            idx_red = idx_red_all[idx_rois]

            smp = SpkmapProcessor(session)
            reliability_all = smp.get_reliability(use_session_filters=False, params=dict(reliability_method=method, smooth=float(smooth_width)))
            reliability_selected = reliability_all.filter_rois(idx_rois)

            num_cells_ctl_all.append(np.sum(~idx_red_all))
            num_cells_red_all.append(np.sum(idx_red_all))
            num_cells_ctl.append(np.sum(~idx_red))
            num_cells_red.append(np.sum(idx_red))

            for ienv, env in enumerate(envs):
                reliability_red_all[env].append(reliability_all.values[ienv, idx_red_all])
                reliability_ctl_all[env].append(reliability_all.values[ienv, ~idx_red_all])
                reliability_red[env].append(reliability_selected.values[ienv, idx_red])
                reliability_ctl[env].append(reliability_selected.values[ienv, ~idx_red])
                sessions[env].append(isession)

        results = dict(
            environments=environments,
            reliability_ctl=reliability_ctl,
            reliability_red=reliability_red,
            reliability_ctl_all=reliability_ctl_all,
            reliability_red_all=reliability_red_all,
            sessions=sessions,
            num_cells_ctl_all=num_cells_ctl_all,
            num_cells_red_all=num_cells_red_all,
            num_cells_ctl=num_cells_ctl,
            num_cells_red=num_cells_red,
        )

        if exclude_environments:
            if not isinstance(exclude_environments, list):
                exclude_environments = [exclude_environments]
            for env in exclude_environments:
                for key in results:
                    if isinstance(results[key], dict):
                        results[key] = {k: v for k, v in results[key].items() if k != env}
                    else:
                        results[key] = [r for r in results[key] if r != env]

        self._reliability[method][smooth_width][mouse] = results
        return results

    def plot(self, state):
        # Gather data to plot
        use_session_filters = state["use_session_filters"]
        reliability = self.get_reliability(state["mouse"], state["reliability_method"], state["smooth_width"])
        sessions = reliability["sessions"]
        sesnum = state["session"]
        environments = [env for env, ises in sessions.items() if sesnum in ises]
        if use_session_filters:
            reliability_ctl = reliability["reliability_ctl"]
            reliability_red = reliability["reliability_red"]
            num_cells_ctl = reliability["num_cells_ctl"]
            num_cells_red = reliability["num_cells_red"]
        else:
            reliability_ctl = reliability["reliability_ctl_all"]
            reliability_red = reliability["reliability_red_all"]
            num_cells_ctl = reliability["num_cells_ctl_all"]
            num_cells_red = reliability["num_cells_red_all"]

        figwidth = 3
        figheight = 3
        num_envs = len(environments)
        fig, ax = plt.subplots(1, num_envs, figsize=(num_envs * figwidth, figheight), layout="constrained")
        if num_envs == 1:
            ax = [ax]

        for ienv, env in enumerate(environments):
            ises = sessions[env].index(sesnum)
            ctl_counts, bins = fractional_histogram(reliability_ctl[env][ises], bins=21)
            red_counts = fractional_histogram(reliability_red[env][ises], bins=bins)[0]
            centers = edge2center(bins)

            ax[ienv].plot(centers, ctl_counts, color="k", label="CTL")
            ax[ienv].plot(centers, red_counts, color="r", label="RED")
            ax[ienv].set_title(f"Environment {env}")
            ax[ienv].set_xlabel("Reliability")
            ax[ienv].set_ylabel("Number of cells")

        suptitle = f"{state['mouse']} - sig:{state['smooth_width']} - {state['reliability_method']}"
        suptitle += f"\nSession {sesnum} - #Ctl:{num_cells_ctl[sesnum]} - #Red:{num_cells_red[sesnum]}"
        fig.suptitle(suptitle)
        return fig


class ReliabilityPlasticity(Viewer):
    def __init__(self, tracked_mice: list[str]):
        self.tracked_mice = list(tracked_mice)
        self.multisessions = {mouse: None for mouse in self.tracked_mice}

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_selection("environment", value=1, options=[1])
        self.add_selection("idx_session_x", value=0, options=[0])
        self.add_selection("idx_session_y", value=1, options=[1])
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "mse", "correlation"])
        self.add_selection("smooth_width", value=5, options=[1, 5])
        self.add_integer("num_bins", value=9, min=3, max=21)
        self.add_boolean("use_session_filters", value=True)
        self.add_boolean("normalize_by_column", value=False)

        self.on_change("mouse", self.reset_mouse)
        self.on_change("environment", self.reset_environment)
        self.reset_mouse(self.state)

    def reset_mouse(self, state):
        msm = self.get_multisession(state["mouse"])
        environments = self.get_environments(state["mouse"])
        start_envnum = msm.env_selector(envmethod="first")
        idx_ses = msm.idx_ses_with_env(start_envnum)
        self.update_selection("environment", value=start_envnum, options=list(environments))
        self.update_selection("idx_session_x", value=idx_ses[0], options=idx_ses)
        self.update_selection("idx_session_y", value=idx_ses[0], options=idx_ses)

    def reset_environment(self, state):
        msm = self.get_multisession(state["mouse"])
        idx_ses = msm.idx_ses_with_env(state["environment"])
        self.update_selection("idx_session_x", value=idx_ses[0], options=idx_ses)
        self.update_selection("idx_session_y", value=idx_ses[0], options=idx_ses)

    def get_multisession(self, mouse: str) -> MultiSessionSpkmaps:
        if self.multisessions[mouse] is None:
            track = Tracker(mouse)
            self.multisessions[mouse] = MultiSessionSpkmaps(track)
        return self.multisessions[mouse]

    def get_environments(self, mouse: str) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        msm = self.get_multisession(mouse)
        environments = np.unique(np.concatenate([session.environments for session in msm.tracker.sessions]))
        return environments

    def plot(self, state):
        # Gather data to plot
        use_session_filters = state["use_session_filters"]
        msm = self.get_multisession(state["mouse"])
        idx_session_x = state["idx_session_x"]
        idx_session_y = state["idx_session_y"]
        idx_ses = [idx_session_x, idx_session_y]
        reliability, extras = msm.get_reliability(
            state["environment"],
            idx_ses=idx_ses,
            use_session_filters=use_session_filters,
            reliability_method=state["reliability_method"],
            smooth=float(state["smooth_width"]),
        )
        idx_red = np.any(np.stack(extras["idx_red"]), axis=0)

        num_ctl = np.sum(~idx_red)
        num_red = np.sum(idx_red)

        # Create figure with three subplots
        figwidth = 3
        figheight = 3

        fig = plt.figure(figsize=(figwidth * 3 + 2 * 0.1, figheight), layout="constrained")
        gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.1, 1, 0.1])
        ax_ctl = fig.add_subplot(gs[0, 0])
        ax_red = fig.add_subplot(gs[0, 1])
        ax_diff = fig.add_subplot(gs[0, 3])
        ax_cbar = fig.add_subplot(gs[0, 2])
        ax_cbar_diff = fig.add_subplot(gs[0, 4])

        # Define common histogram parameters
        bins = state["num_bins"]
        range_x = (-1, 1)
        range_y = (-1, 1)

        # Plot control cells (2D histogram)
        h_ctl, xedges, yedges = np.histogram2d(reliability[0][~idx_red], reliability[1][~idx_red], bins=bins, range=[range_x, range_y])
        h_red, _, _ = np.histogram2d(reliability[0][idx_red], reliability[1][idx_red], bins=bins, range=[range_x, range_y])

        h_ctl = h_ctl.T
        h_red = h_red.T

        # Normalize by total counts to get density difference
        if state["normalize_by_column"]:
            ctl_col_num = np.where(np.sum(h_ctl, axis=0) > 0, np.sum(h_ctl, axis=0), np.nan)
            red_col_num = np.where(np.sum(h_red, axis=0) > 0, np.sum(h_red, axis=0), np.nan)
            h_ctl = h_ctl / ctl_col_num
            h_red = h_red / red_col_num
            h_diff = h_red - h_ctl
        else:
            h_ctl = h_ctl / np.sum(h_ctl) if np.sum(h_ctl) > 0 else h_ctl
            h_red = h_red / np.sum(h_red) if np.sum(h_red) > 0 else h_red
            h_diff = h_red - h_ctl

        max_density = max(np.max(h_ctl), np.max(h_red))
        max_diff = np.max(np.abs(h_diff))

        get_kwargs = lambda cmap, min, max: dict(
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            vmin=min,
            vmax=max,
        )

        cmap = mpl.colormaps["gray_r"]
        cmap_diff = mpl.colormaps["bwr"]
        cmap.set_bad("orange", alpha=0.3)
        cmap_diff.set_bad("orange", alpha=0.3)

        im = ax_ctl.imshow(h_ctl, **get_kwargs(cmap, 0, max_density))
        ax_red.imshow(h_red, **get_kwargs(cmap, 0, max_density))
        im_diff = ax_diff.imshow(h_diff, **get_kwargs(cmap_diff, -max_diff, max_diff))

        ax_diff.set_title("Difference (Red - Control)")

        # Add colorbar for difference plot
        plt.colorbar(im, cax=ax_cbar, label="Fraction of ROIs")
        plt.colorbar(im_diff, cax=ax_cbar_diff, label="Difference (RED - CTL)")

        # Set common labels and formatting
        for ax in [ax_ctl, ax_red, ax_diff]:
            ax.set_xlabel(f"Reliability {idx_session_x}")
            ax.set_ylabel(f"Reliability {idx_session_y}")
            ax.set_xlim(range_x)
            ax.set_ylim(range_y)

        ax_ctl.set_title(f"Control - #ROIs: {num_ctl}")
        ax_red.set_title(f"Red - #ROIs: {num_red}")
        ax_diff.set_title("Difference")

        # Add overall title
        suptitle = f"{state['mouse']} - sig:{state['smooth_width']} - {state['reliability_method']}"
        suptitle += f"\nEnvironment {state['environment']}"
        fig.suptitle(suptitle)

        return fig


class ReliabilityStability(Viewer):
    def __init__(self, tracked_mice: list[str]):
        self.tracked_mice = list(tracked_mice)
        self.multisessions = {mouse: None for mouse in self.tracked_mice}

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_selection("environment", value=1, options=[1])
        self.add_float("reliability_threshold", value=0.5, min=0.1, max=1)
        self.add_selection("reference_session", value=0, options=[0])
        self.add_integer("num_sessions", value=100, min=1, max=100)
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "mse", "correlation"])
        self.add_selection("smooth_width", value=5, options=[1, 5])
        self.add_boolean("use_session_filters", value=True)
        self.add_boolean("continuous", value=True)
        self.add_selection("last_plot", value="centroid-plasticity", options=["centroid-plasticity", "spkmap-correlation", "number-stable"])
        self.add_selection("last_type", value="mean", options=["mean", "distribution"])
        self.add_boolean("use_fraction", value=False)
        self.add_float("fraction_active_threshold", value=0.2, min=0, max=1)

        # Set up callbacks
        self.on_change("mouse", self.reset_mouse)
        self.on_change("environment", self.reset_environment)
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
        self.update_selection("reference_session", value=idx_ses[0], options=list(idx_ses))
        self.update_integer("num_sessions", max=len(idx_ses))

    def get_multisession(self, mouse: str) -> MultiSessionSpkmaps:
        if self.multisessions[mouse] is None:
            track = Tracker(mouse)
            self.multisessions[mouse] = MultiSessionSpkmaps(track)
        return self.multisessions[mouse]

    def get_environments(self, mouse: str) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        msm = self.get_multisession(mouse)
        environments = np.unique(np.concatenate([session.environments for session in msm.tracker.sessions]))
        return environments

    def gather_data(self, state: dict, idx_ses: list[int], forward: bool = True):
        envnum = state["environment"]
        use_session_filters = state["use_session_filters"]
        msm = self.get_multisession(state["mouse"])
        idx_ref = idx_ses.index(state["reference_session"])

        if len(idx_ses) == 1:
            return None

        average = False if state["use_fraction"] else True
        spkmaps, extras = msm.get_spkmaps(
            envnum,
            idx_ses=idx_ses,
            use_session_filters=use_session_filters,
            tracked=True,
            average=average,
            pop_nan=True,
            reliability_method=state["reliability_method"],
            smooth=float(state["smooth_width"]),
        )
        reliability = np.stack(extras["reliability"])
        pflocs = np.stack(extras["pfloc"])

        threshold = state["reliability_threshold"]
        target_rois = reliability[idx_ref] > threshold

        if state["use_fraction"]:
            fraction_active = FractionActive.compute(spkmaps[idx_ref], activity_axis=2, fraction_axis=1)
            target_rois = target_rois & (fraction_active > state["fraction_active_threshold"])
            spkmaps = [np.nanmean(spkmap, axis=1) for spkmap in spkmaps]

        spkmaps = np.stack(spkmaps)
        idx_red = np.any(np.stack(extras["idx_red"]), axis=0)

        # For labeling and fractional analyses
        num_ctl = np.sum(~idx_red & target_rois)
        num_red = np.sum(idx_red & target_rois)

        ctl_reliable = reliability[:, target_rois & ~idx_red]
        red_reliable = reliability[:, target_rois & idx_red]
        if state["last_plot"] == "centroid-plasticity":
            ctl_pflocs = pflocs[:, target_rois & ~idx_red]
            red_pflocs = pflocs[:, target_rois & idx_red]
        elif state["last_plot"] == "spkmap-correlation":
            ctl_spkmaps = spkmaps[:, target_rois & ~idx_red]
            red_spkmaps = spkmaps[:, target_rois & idx_red]

        ctl_stability = np.full(ctl_reliable.shape, False, dtype=bool)
        red_stability = np.full(red_reliable.shape, False, dtype=bool)

        ctl_stability[idx_ref] = True
        red_stability[idx_ref] = True
        if forward:
            for idx in range(1, len(idx_ses)):
                if state["continuous"]:
                    ctl_stability[idx] = (ctl_reliable[idx] > threshold) & ctl_stability[idx - 1]
                    red_stability[idx] = (red_reliable[idx] > threshold) & red_stability[idx - 1]
                else:
                    ctl_stability[idx] = ctl_reliable[idx] > threshold
                    red_stability[idx] = red_reliable[idx] > threshold
        else:
            for idx in range(len(idx_ses) - 2, -1, -1):
                if state["continuous"]:
                    ctl_stability[idx] = (ctl_reliable[idx] > threshold) & ctl_stability[idx + 1]
                    red_stability[idx] = (red_reliable[idx] > threshold) & red_stability[idx + 1]
                else:
                    ctl_stability[idx] = ctl_reliable[idx] > threshold
                    red_stability[idx] = red_reliable[idx] > threshold

        num_stable_ctl = np.sum(ctl_stability, axis=1)
        num_stable_red = np.sum(red_stability, axis=1)
        fraction_stable_ctl = num_stable_ctl / num_ctl
        fraction_stable_red = num_stable_red / num_red
        stable_reliability_ctl = np.sum(ctl_stability * ctl_reliable, axis=1) / np.sum(ctl_stability, axis=1)
        stable_reliability_red = np.sum(red_stability * red_reliable, axis=1) / np.sum(red_stability, axis=1)

        results = dict(
            num_ctl=num_ctl,
            num_red=num_red,
            num_stable_ctl=num_stable_ctl,
            num_stable_red=num_stable_red,
            fraction_stable_ctl=fraction_stable_ctl,
            fraction_stable_red=fraction_stable_red,
            stable_reliability_ctl=stable_reliability_ctl,
            stable_reliability_red=stable_reliability_red,
        )

        if state["last_plot"] == "centroid-plasticity":
            ctl_pfloc_changes = np.abs(ctl_pflocs - ctl_pflocs[idx_ref])
            red_pfloc_changes = np.abs(red_pflocs - red_pflocs[idx_ref])
            if state["last_type"] == "mean":
                ctl_pfloc_changes = np.sum(ctl_pfloc_changes * ctl_stability, axis=1) / np.sum(ctl_stability, axis=1)
                red_pfloc_changes = np.sum(red_pfloc_changes * red_stability, axis=1) / np.sum(red_stability, axis=1)
            elif state["last_type"] == "distribution":
                ctl_pfloc_changes = [cpc[cs] for cpc, cs in zip(ctl_pfloc_changes, ctl_stability)]
                red_pfloc_changes = [rpc[rs] for rpc, rs in zip(red_pfloc_changes, red_stability)]
            results["ctl_pfloc_changes"] = ctl_pfloc_changes
            results["red_pfloc_changes"] = red_pfloc_changes
        elif state["last_plot"] == "spkmap-correlation":
            ctl_spkmap_correlations = np.full(ctl_spkmaps.shape[:2], np.nan)
            red_spkmap_correlations = np.full(red_spkmaps.shape[:2], np.nan)
            for ii in range(len(idx_ses)):
                ctl_spkmap_correlations[ii] = vectorCorrelation(ctl_spkmaps[ii], ctl_spkmaps[idx_ref], axis=1)
                red_spkmap_correlations[ii] = vectorCorrelation(red_spkmaps[ii], red_spkmaps[idx_ref], axis=1)
            if state["last_type"] == "mean":
                ctl_spkmap_correlations = np.sum(ctl_spkmap_correlations * ctl_stability, axis=1) / np.sum(ctl_stability, axis=1)
                red_spkmap_correlations = np.sum(red_spkmap_correlations * red_stability, axis=1) / np.sum(red_stability, axis=1)
            elif state["last_type"] == "distribution":
                ctl_spkmap_correlations = [csc[cs] for csc, cs in zip(ctl_spkmap_correlations, ctl_stability)]
                red_spkmap_correlations = [rsc[rs] for rsc, rs in zip(red_spkmap_correlations, red_stability)]
            results["ctl_spkmap_correlations"] = ctl_spkmap_correlations
            results["red_spkmap_correlations"] = red_spkmap_correlations

        return results

    def plot(self, state):
        # Gather data to plot
        msm = self.get_multisession(state["mouse"])
        envnum = state["environment"]
        ref_session = state["reference_session"]
        num_sessions = state["num_sessions"]
        idx_ses = msm.idx_ses_with_env(envnum)
        idx_ref = idx_ses.index(ref_session)
        first_backward = max(0, idx_ref - num_sessions)
        last_forward = min(len(idx_ses), idx_ref + num_sessions + 1)
        idx_ses_backward = idx_ses[first_backward : idx_ref + 1]
        idx_ses_forward = idx_ses[idx_ref:last_forward]
        num_backward = len(idx_ses_backward)
        num_forward = len(idx_ses_forward)

        xticks = []
        xticklabels = []
        fraction_stable_ctl = []
        fraction_stable_red = []
        stable_reliability_ctl = []
        stable_reliability_red = []
        ctl_seed_cells_message = []
        red_seed_cells_message = []

        last_plot_ctl = []
        last_plot_red = []
        if num_backward > 1:
            results_backward = self.gather_data(state, idx_ses_backward, forward=False)
            xticks.append(list(np.arange(num_backward) - num_backward + 1))
            xticklabels.append([f"{idx - ref_session}" if idx != ref_session else "ref" for idx in idx_ses_backward])
            fraction_stable_ctl.append(list(results_backward["fraction_stable_ctl"]))
            fraction_stable_red.append(list(results_backward["fraction_stable_red"]))
            stable_reliability_ctl.append(list(results_backward["stable_reliability_ctl"]))
            stable_reliability_red.append(list(results_backward["stable_reliability_red"]))
            ctl_seed_cells_message.append(f"#Backward:{results_backward['num_ctl']}")
            red_seed_cells_message.append(f"#Backward:{results_backward['num_red']}")

            if state["last_plot"] == "centroid-plasticity":
                last_plot_ctl.append(list(results_backward["ctl_pfloc_changes"]))
                last_plot_red.append(list(results_backward["red_pfloc_changes"]))
            elif state["last_plot"] == "spkmap-correlation":
                last_plot_ctl.append(list(results_backward["ctl_spkmap_correlations"]))
                last_plot_red.append(list(results_backward["red_spkmap_correlations"]))
            elif state["last_plot"] == "number-stable":
                last_plot_ctl.append(list(results_backward["num_stable_ctl"]))
                last_plot_red.append(list(results_backward["num_stable_red"]))
            else:
                raise ValueError(f"Didn't recognize plot type ({state['last_plot']})")

        if num_forward > 1:
            results_forward = self.gather_data(state, idx_ses_forward, forward=True)
            xticks.append(list(np.arange(num_forward)))
            xticklabels.append([f"{idx - ref_session}" if idx != ref_session else "ref" for idx in idx_ses_forward])
            fraction_stable_ctl.append(list(results_forward["fraction_stable_ctl"]))
            fraction_stable_red.append(list(results_forward["fraction_stable_red"]))
            stable_reliability_ctl.append(list(results_forward["stable_reliability_ctl"]))
            stable_reliability_red.append(list(results_forward["stable_reliability_red"]))
            ctl_seed_cells_message.append(f"#Forward:{results_forward['num_ctl']}")
            red_seed_cells_message.append(f"#Forward:{results_forward['num_red']}")

            if state["last_plot"] == "centroid-plasticity":
                last_plot_ctl.append(list(results_forward["ctl_pfloc_changes"]))
                last_plot_red.append(list(results_forward["red_pfloc_changes"]))
            elif state["last_plot"] == "spkmap-correlation":
                last_plot_ctl.append(list(results_forward["ctl_spkmap_correlations"]))
                last_plot_red.append(list(results_forward["red_spkmap_correlations"]))
            elif state["last_plot"] == "number-stable":
                last_plot_ctl.append(list(results_forward["num_stable_ctl"]))
                last_plot_red.append(list(results_forward["num_stable_red"]))
            else:
                raise ValueError(f"Didn't recognize plot type ({state['last_plot']})")

        # Create figure with three subplots
        figwidth = 3
        figheight = 3.5

        fig = plt.figure(figsize=(3 * figwidth, figheight), layout="constrained")
        gs = fig.add_gridspec(2, 3)
        ax_frac = fig.add_subplot(gs[:, 0])
        ax_rel = fig.add_subplot(gs[:, 1])
        if state["last_plot"] == "number-stable":
            ax_num_ctl = fig.add_subplot(gs[0, 2])
            ax_num_red = fig.add_subplot(gs[1, 2])
        else:
            ax_last_plot = fig.add_subplot(gs[:, 2])

        xticks_all = np.concatenate(xticks)
        xticklabels_all = np.concatenate(xticklabels)
        idx_to_ref_label = np.where(xticklabels_all == "ref")[0]
        if len(idx_to_ref_label) > 1:
            xticklabels_all = np.delete(xticklabels_all, idx_to_ref_label[1])
            xticks_all = np.delete(xticks_all, idx_to_ref_label[1])

        for i in range(len(xticks)):
            label = "CTL" if i == 0 else None
            label = "RED" if i == 0 else None
            ax_frac.plot(xticks[i], fraction_stable_ctl[i], color="k", linewidth=1.5, label=label, marker=".", markersize=8)
            ax_frac.plot(xticks[i], fraction_stable_red[i], color="r", linewidth=1.5, label=label, marker=".", markersize=8)
        ax_frac.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax_frac.set_ylim(-0.05, 1.05)
        ax_frac.set_xticks(xticks_all, labels=xticklabels_all)
        ax_frac.set_xlabel("Session")
        ax_frac.set_ylabel("Fraction")
        ax_frac.set_title("Fraction Stable Reliability")
        ax_frac.legend(fontsize=9, loc="best")

        for i in range(len(xticks)):
            ax_rel.plot(xticks[i], stable_reliability_ctl[i], color="k", linewidth=1.5, marker=".", markersize=8)
            ax_rel.plot(xticks[i], stable_reliability_red[i], color="r", linewidth=1.5, marker=".", markersize=8)
        ax_rel.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax_rel.set_xticks(xticks_all, labels=xticklabels_all)
        ax_rel.set_xlabel("Session")
        ax_rel.set_ylabel("Reliability (mean)")
        ax_rel.set_title("Stable Cells Reliability")

        if state["last_plot"] == "number-stable":
            for i in range(len(xticks)):
                ax_num_ctl.plot(xticks[i], last_plot_ctl[i], color="k", linewidth=1.5, marker=".", markersize=8)
            ax_num_ctl.axvline(0, color="k", linewidth=0.5, linestyle="--")
            ax_num_ctl.set_ylim(0)
            ax_num_ctl.set_xticks(xticks_all, labels=xticklabels_all)
            ax_num_ctl.set_ylabel("#ROIs")
            ax_num_ctl.set_title(",".join(ctl_seed_cells_message), fontsize=10)

            for i in range(len(xticks)):
                ax_num_red.plot(xticks[i], last_plot_red[i], color="r", linewidth=1.5, marker=".", markersize=8)
            ax_num_red.axvline(0, color="k", linewidth=0.5, linestyle="--")
            ax_num_red.set_ylim(0)
            ax_num_red.set_xticks(xticks_all, labels=xticklabels_all)
            ax_num_red.set_xlabel("Session")
            ax_num_red.set_ylabel("#ROIs")
            ax_num_red.set_title(",".join(red_seed_cells_message), fontsize=10)
        else:
            if state["last_type"] == "mean":
                for i in range(len(xticks)):
                    ax_last_plot.plot(xticks[i], last_plot_ctl[i], color="k", linewidth=1.5, marker=".", markersize=8)
                    ax_last_plot.plot(xticks[i], last_plot_red[i], color="r", linewidth=1.5, marker=".", markersize=8)
            elif state["last_type"] == "distribution":
                for i in range(len(xticks)):
                    for ii in range(len(xticks[i])):
                        if len(last_plot_ctl[i][ii]) > 0:
                            iiictl = ax_last_plot.violinplot(
                                last_plot_ctl[i][ii],
                                positions=[xticks[i][ii]],
                                widths=[0.5],
                                showmeans=True,
                                showextrema=False,
                                side="low",
                            )
                            color_violins(iiictl, facecolor=("k", 0.3), linecolor="k")
                        if len(last_plot_red[i][ii]) > 0:
                            iiired = ax_last_plot.violinplot(
                                last_plot_red[i][ii],
                                positions=[xticks[i][ii]],
                                widths=[0.5],
                                showmeans=True,
                                showextrema=False,
                                side="high",
                            )
                            color_violins(iiired, facecolor=("r", 0.3), linecolor="r")

            ax_last_plot.set_xticks(xticks_all, labels=xticklabels_all)
            ax_last_plot.set_xlabel("Session")
            ax_last_plot.axvline(0, color="k", linewidth=0.5, linestyle="--")
            if state["last_plot"] == "centroid-plasticity":
                ax_last_plot.set_ylabel("$\Delta$PF Centroid")
                ax_last_plot.set_title("Stable - PF Centroid Change")
                ax_last_plot.set_ylim(0)
            elif state["last_plot"] == "spkmap-correlation":
                ax_last_plot.set_ylabel("Spkmap Correlation")
                ax_last_plot.axhline(0, color="k", linewidth=0.5, linestyle="--")
                ax_last_plot.set_title("Stable - Spkmap Correlation")
            else:
                raise ValueError(f"Didn't recognize plot type ({state['last_plot']})")

        # Add overall title
        suptitle = f"{state['mouse']} - sig:{state['smooth_width']} - {state['reliability_method']}"
        suptitle += f"\nEnvironment {state['environment']} - Threshold: {state['reliability_threshold']} - Continuous: {state['continuous']}"
        suptitle += f"\nCTL Counts:{ctl_seed_cells_message} - RED Counts:{red_seed_cells_message}"
        fig.suptitle(suptitle)

        return fig


class ReliabilityToSpkmap(Viewer):
    def __init__(self, tracked_mice: list[str]):
        self.tracked_mice = list(tracked_mice)
        self.multisessions = {mouse: None for mouse in self.tracked_mice}

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_selection("session", value=0, options=[0])
        self.add_selection("environment", value=1, options=[1])
        self.add_selection("roi", value=0, options=[0])
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "mse", "correlation"])
        self.add_float_range("reliability_range", value=(0.5, 1.0), min=-1.0, max=1.0)
        self.add_float_range("fraction_active_range", value=(0.0, 1.0), min=0, max=1)
        self.add_selection("smooth_width", value=5, options=[1, 5])
        self.add_boolean("use_session_filters", value=True)
        self.add_float("vmax", value=2, min=1.0, max=10.0)

        # Set up callbacks
        self.on_change("mouse", self.reset_mouse)
        self.on_change("session", self.reset_session)
        self.on_change(
            ["environment", "fraction_active_range", "reliability_range", "reliability_method", "smooth_width", "use_session_filters"],
            self.reset_rois,
        )

        # Implement callback to update selections
        self.reset_mouse(self.state)

    def reset_mouse(self, state):
        msm = self.get_multisession(state["mouse"])
        self.update_selection("session", options=list(range(len(msm.processors))))
        self.reset_session(self.state)

    def reset_session(self, state):
        msm = self.get_multisession(state["mouse"])
        idx_ses = state["session"]
        environments = msm.processors[idx_ses].session.environments
        self.update_selection("environment", options=list(environments))
        self.reset_rois(self.state)

    def reset_rois(self, state):
        msm = self.get_multisession(state["mouse"])
        idx_ses = int(state["session"])
        envnum = state["environment"]
        env_maps = msm.processors[idx_ses].get_env_maps(
            use_session_filters=state["use_session_filters"],
            params=dict(smooth_width=float(state["smooth_width"]), reliability_method=state["reliability_method"]),
        )
        env_maps.filter_environments([envnum])
        fraction_active = FractionActive.compute(env_maps.spkmap[0], activity_axis=2, fraction_axis=1)
        reliability = (
            msm.processors[idx_ses]
            .get_reliability(
                use_session_filters=state["use_session_filters"],
                params=dict(smooth_width=float(state["smooth_width"]), reliability_method=state["reliability_method"]),
            )
            .filter_by_environment([envnum])
            .values[0]
        )
        reliability_range = state["reliability_range"]
        potential_reliable = np.logical_and(reliability >= reliability_range[0], reliability <= reliability_range[1])
        potential_fraction = np.logical_and(
            fraction_active >= state["fraction_active_range"][0], fraction_active <= state["fraction_active_range"][1]
        )
        idx_potential_rois = np.where(np.logical_and(potential_reliable, potential_fraction))[0]
        self.update_selection("roi", options=list(idx_potential_rois))

    def get_multisession(self, mouse: str) -> MultiSessionSpkmaps:
        if self.multisessions[mouse] is None:
            track = Tracker(mouse)
            self.multisessions[mouse] = MultiSessionSpkmaps(track)
        return self.multisessions[mouse]

    def plot(self, state):
        # Gather data to plot
        msm = self.get_multisession(state["mouse"])
        idx_ses = int(state["session"])
        envnum = state["environment"]
        idx_roi = int(state["roi"])
        env_maps = msm.processors[idx_ses].get_env_maps(
            state["use_session_filters"],
            params=dict(smooth_width=float(state["smooth_width"]), reliability_method=state["reliability_method"]),
        )
        env_maps.filter_environments([envnum])
        reliability = (
            msm.processors[idx_ses]
            .get_reliability(
                use_session_filters=state["use_session_filters"],
                params=dict(smooth_width=float(state["smooth_width"]), reliability_method=state["reliability_method"]),
            )
            .filter_by_environment([envnum])
            .values[0]
        )
        fraction_active = FractionActive.compute(env_maps.spkmap[0], activity_axis=2, fraction_axis=1)
        spkmap = env_maps.spkmap[0][idx_roi]

        fig, ax = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
        ax[0].imshow(spkmap, aspect="auto", interpolation="none", vmin=0, vmax=state["vmax"], cmap="gray_r")
        ax[0].set_xlabel("Position")
        ax[0].set_ylabel("Trials")
        ax[0].set_title(f"ROI {idx_roi}")

        ax[1].hist(reliability, bins=11, facecolor="k", edgecolor="k")
        ax[1].axvline(reliability[idx_roi], color="b", linewidth=1.5)
        ax[1].set_xlabel("Reliability")
        ax[1].set_ylabel("Counts")
        ax[1].set_title(f"Reliability: {reliability[idx_roi]:.2f}")

        ax[2].hist(fraction_active, bins=11, facecolor="k", edgecolor="k")
        ax[2].axvline(fraction_active[idx_roi], color="b", linewidth=1.5)
        ax[2].set_xlabel("Fraction Active")
        ax[2].set_ylabel("Counts")
        ax[2].set_title(f"Fraction Active: {fraction_active[idx_roi]:.2f}")

        # Add overall title
        suptitle = f"{state['mouse']} - sig:{state['smooth_width']} - {state['reliability_method']}"
        suptitle += f"\nEnvironment {state['environment']}"
        fig.suptitle(suptitle)

        return fig


class SingleCellReliabilityFigureMaker(ReliabilityToSpkmap):
    def __init__(self, tracked_mice: list[str]):
        super().__init__(tracked_mice)
        self.update_selection("reliability_method", options=["leave_one_out"])
        self.add_float_range("spks_xlim", value=(0.0, 1.0), min=0.0, max=1.0)
        self.add_selection("summary_type", value="reliability", options=["reliability", "fraction_active"])
        self.add_button("save_example", label="Save Example", callback=self.save_example)

    def save_example(self, state):
        fig = self.plot(state)
        fig_dir = figure_dir("reliability_single_cell_examples")
        msm = self.get_multisession(state["mouse"])
        idx_ses = int(state["session"])
        roi = state["roi"]
        idx_roi = np.where(msm.processors[idx_ses].session.idx_rois)[0][roi]
        fig_name = f"{state['mouse']}_{state['session']}_{state['environment']}_{idx_roi}_{state['summary_type']}"
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_dir / fig_name)

    def plot(self, state):
        # Gather data to plot
        msm = self.get_multisession(state["mouse"])
        idx_ses = int(state["session"])
        envnum = state["environment"]
        idx_roi = int(state["roi"])
        env_maps = msm.processors[idx_ses].get_env_maps(
            state["use_session_filters"],
            params=dict(smooth_width=float(state["smooth_width"]), reliability_method=state["reliability_method"]),
        )
        env_maps.filter_environments([envnum])
        reliability = (
            msm.processors[idx_ses]
            .get_reliability(
                use_session_filters=state["use_session_filters"],
                params=dict(smooth_width=float(state["smooth_width"]), reliability_method=state["reliability_method"]),
            )
            .filter_by_environment([envnum])
            .values[0]
        )
        fraction_active = FractionActive.compute(env_maps.spkmap[0], activity_axis=2, fraction_axis=1)
        spkmap = env_maps.spkmap[0][idx_roi]

        # Get raw spks data
        idx_rois = msm.processors[idx_ses].session.idx_rois
        timestamps = msm.processors[idx_ses].session.timestamps
        spks = msm.processors[idx_ses].session.spks[:, idx_rois][:, idx_roi]
        vr_times, vr_position, _, _ = msm.processors[idx_ses].session.positions

        min_spks = np.min(spks)
        max_spks = np.max(spks)
        fraction_for_position = 0.25
        units_for_position = (max_spks - min_spks) * fraction_for_position
        ymin = min_spks - units_for_position
        ymax = max_spks * 1.025
        vr_position = (vr_position - np.min(vr_position)) / (np.max(vr_position) - np.min(vr_position)) * (
            0.95 * units_for_position
        ) - units_for_position

        # Break down reliability metrics
        idx_nan_positions = np.any(np.isnan(spkmap), axis=0)
        spkmap = spkmap[:, ~idx_nan_positions]
        trial_consistency = _jit_reliability_loo(spkmap[None, ...])[0]
        trial_weights = np.sqrt(np.mean(spkmap**2, axis=1))

        fig = plt.figure(figsize=(6, 9), layout="constrained")
        gs = fig.add_gridspec(3, 3, width_ratios=[4, 1, 1], height_ratios=[1, 3, 1])
        ax_spks = fig.add_subplot(gs[0, :])
        ax_spkmap = fig.add_subplot(gs[1, 0])
        ax_amplitude = fig.add_subplot(gs[1, 1])
        ax_correlation = fig.add_subplot(gs[1, 2])
        ax_summary = fig.add_subplot(gs[2, :])

        ax_spks.plot(timestamps, spks, color="k", linewidth=0.5)
        ax_spks.plot(vr_times, vr_position, color="b", linewidth=0.5)
        og_xlim = list(ax_spks.get_xlim())
        og_xlim[0] = max(0, og_xlim[0])
        og_xlim[1] = min(max(timestamps), og_xlim[1])
        og_range = og_xlim[1] - og_xlim[0]
        xmin = og_xlim[0] + og_range * state["spks_xlim"][0]
        xmax = og_xlim[0] + og_range * state["spks_xlim"][1]
        ax_spks.set_yticks([])
        ax_spks.set_xlim(xmin, xmax)
        ax_spks.set_xlabel("Time", labelpad=-5)
        ax_spks.set_ylim(ymin, ymax)
        ax_spks.text(xmin, ymax, "Fluorescence", ha="left", va="top", fontsize=12, color="k")
        ax_spks.text(xmax, ymax, "VR Position", ha="right", va="top", fontsize=12, color="b")
        format_spines(ax_spks, x_pos=-0.05, y_pos=-0.05, xbounds=(xmin, xmax), xticks=[xmin, xmax], tick_length=4, spines_visible=["bottom"])

        extent = (0, spkmap.shape[1], spkmap.shape[0], 0)
        ax_spkmap.imshow(spkmap, aspect="auto", origin="upper", interpolation="none", vmin=0, vmax=state["vmax"], cmap="gray_r", extent=extent)
        ax_spkmap.set_xlabel("VR Position")
        ax_spkmap.set_ylabel("Trials")
        ax_spkmap.set_ylim(spkmap.shape[0] + 1, -1)
        format_spines(ax_spkmap, x_pos=-0.05, y_pos=-0.05, tick_length=4, spines_visible=["bottom", "left"])

        idx_include = trial_weights > 0
        trial_numbers = np.arange(spkmap.shape[0])[idx_include]
        trial_weights = trial_weights[idx_include] / np.max(trial_weights[idx_include])
        trial_consistency = trial_consistency[idx_include]

        ax_amplitude.scatter(trial_weights, trial_numbers, color="k", s=5, alpha=trial_weights)
        ax_amplitude.set_facecolor(("black", 0.05))
        ax_amplitude.set_xlim(-0.05, 1.05)
        ax_amplitude.set_ylim(spkmap.shape[0] + 1, -1)
        ax_amplitude.set_xlabel("RMS Amplitude")
        format_spines(ax_amplitude, x_pos=-0.05, y_pos=-0.05, xbounds=(0, 1), xticks=[0, 1], yticks=[], tick_length=4, spines_visible=["bottom"])

        ax_correlation.scatter(trial_consistency, trial_numbers, color="k", s=5, alpha=trial_weights)
        ax_correlation.set_facecolor(("black", 0.05))
        ax_correlation.set_xlim(-1.05, 1.05)
        ax_correlation.set_ylim(spkmap.shape[0] + 1, -1)
        ax_correlation.set_xlabel("Corrcoef")
        format_spines(
            ax_correlation, x_pos=-0.05, y_pos=-0.05, xbounds=(-1, 1), xticks=[-1, 0, 1], yticks=[], tick_length=4, spines_visible=["bottom"]
        )

        if state["summary_type"] == "reliability":
            ax_summary.hist(reliability, bins=11, facecolor="k", edgecolor="k")
            ax_summary.axvline(reliability[idx_roi], color="b", linewidth=1.5, label=f"Reliability: {reliability[idx_roi]:.2f}")
            ax_summary.set_xlabel("Reliability")
            ax_summary.legend(loc="best")
            xbounds = (-1, 1)

        elif state["summary_type"] == "fraction_active":
            ax_summary.hist(fraction_active, bins=11, facecolor="k", edgecolor="k")
            ax_summary.axvline(fraction_active[idx_roi], color="b", linewidth=1.5, label=f"Fraction Active: {fraction_active[idx_roi]:.2f}")
            ax_summary.set_xlabel("Fraction Active")
            ax_summary.legend(loc="best")
            xbounds = (0, 1)

        else:
            raise ValueError(f"Didn't recognize summary type ({state['summary_type']})")

        format_spines(ax_summary, x_pos=-0.05, y_pos=-0.05, xbounds=xbounds, yticks=[], tick_length=4, spines_visible=["bottom"])

        # Add overall title
        suptitle = f"{state['mouse']} - sig:{state['smooth_width']} - {state['summary_type']}"
        suptitle += f"\nEnvironment {state['environment']}"
        fig.suptitle(suptitle)

        return fig


class ReliabilityQuantileSummary(Viewer):
    def __init__(self, tracked_mice: list[str], try_cache: bool = True, save_cache: bool = True):
        self.tracked_mice = list(tracked_mice)
        self.trackers = {mouse: Tracker(mouse) for mouse in self.tracked_mice}
        self.msms = {mouse: MultiSessionSpkmaps(self.trackers[mouse]) for mouse in self.tracked_mice}

        # Set the maximum number of environments and sessions to plot
        self.max_environments = 3
        self.max_sessions = 10

        # Get the reliability data for all mice...
        reliability_methods = ["leave_one_out", "correlation"]

        if try_cache:
            cache_file = self.cache_path()
            if cache_file.exists():
                self.reliability_data = joblib.load(cache_file)
        else:
            self.reliability_data = {}
            for method in reliability_methods:
                self.reliability_data[method] = {}
                for mouse in self.tracked_mice:
                    self.reliability_data[method][mouse] = get_reliability(
                        self.trackers[mouse],
                        reliability_method=method,
                        exclude_environments=[-1],
                        clear_one=True,
                    )
            if save_cache:
                joblib.dump(self.reliability_data, self.cache_path())

        self.add_boolean("selected", value=True)
        self.add_selection("reliability_method", value=reliability_methods[0], options=reliability_methods)
        self.add_integer("num_bins", value=5, min=3, max=11)
        self.add_integer("quantile_focus", value=3, min=0, max=3)

        self.on_change("num_bins", self.update_quantiles)
        self.update_quantiles(self.state)

    def cache_path(self) -> Path:
        return analysis_path() / "before_the_reveal_temp_data" / "reliability_quantile_summary.joblib"

    def update_quantiles(self, state):
        num_bins = state["num_bins"]
        quantile_focus_limits = num_bins - 2
        self.update_integer("quantile_focus", value=quantile_focus_limits, min=0, max=quantile_focus_limits)

    def plot(self, state):
        num_bins = state["num_bins"]
        quantile_focus = state["quantile_focus"]
        bins = np.linspace(0, 1, num_bins)
        centers = edge2center(bins)

        reliability = self.reliability_data[state["reliability_method"]]

        ctl_key = "reliability_ctl" if state["selected"] else "reliability_ctl_all"
        red_key = "reliability_red" if state["selected"] else "reliability_red_all"
        ctl_reliability = {mouse: rel_data[ctl_key] for mouse, rel_data in reliability.items()}
        red_reliability = {mouse: rel_data[red_key] for mouse, rel_data in reliability.items()}

        num_mice = len(reliability)

        ctl_reliability = [np.full((num_mice, self.max_sessions, len(centers)), np.nan) for _ in range(self.max_environments)]
        red_reliability = [np.full((num_mice, self.max_sessions, len(centers)), np.nan) for _ in range(self.max_environments)]
        red_deviation = [np.full((num_mice, self.max_sessions, len(centers)), np.nan) for _ in range(self.max_environments)]

        for imouse, mouse in enumerate(reliability):
            envstats = self.msms[mouse].env_stats()
            env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
            env_in_order = [env for env in env_in_order if env != -1]
            for ienv, env in enumerate(env_in_order):
                for ises, (ctlval, redval) in enumerate(zip(reliability[mouse][ctl_key][env], reliability[mouse][red_key][env])):
                    if ises >= self.max_sessions:
                        continue
                    # Get the session number for this list element
                    sesnum = reliability[mouse]["sessions"][env][ises]
                    rel_sesnum = envstats[env].index(sesnum)

                    # Measure the reliability of the control and red cells (in true reliability value bins)
                    ctl_reliability[ienv][imouse, rel_sesnum] = fractional_histogram(ctlval, bins=bins)[0]
                    red_reliability[ienv][imouse, rel_sesnum] = fractional_histogram(redval, bins=bins)[0]

                    # Measure the quantiles of the control reliabilty data and use it to measure the deviation of the red cells relative to the quantiles
                    c_quantiles = np.quantile(ctlval, bins)
                    red_deviation[ienv][imouse, rel_sesnum] = fractional_histogram(redval, bins=c_quantiles)[0] - (1 / len(centers))

        cmap_mice = mpl.colormaps["tab10"]
        colors_mice = cmap_mice(np.linspace(0, 1, num_mice))
        colors_mice = {mouse: colors_mice[imouse] for imouse, mouse in enumerate(reliability)}
        colors_mice["CR_Hippocannula6"] = "black"
        colors_mice["CR_Hippocannula7"] = "dimgrey"
        linewidth = [1.5 if mouse in ["CR_Hippocannula6", "CR_Hippocannula7"] else 0.75 for mouse in reliability]
        zorder = [1 if mouse in ["CR_Hippocannula6", "CR_Hippocannula7"] else 0 for mouse in reliability]

        figheight = 3
        figwidth = 3
        fig, ax = plt.subplots(1, self.max_environments, figsize=(figwidth * self.max_environments, figheight), layout="constrained")
        for ienv in range(self.max_environments):
            for imouse, mouse in enumerate(reliability):
                ax[ienv].plot(
                    range(self.max_sessions),
                    red_deviation[ienv][imouse][:, quantile_focus],
                    color=colors_mice[mouse],
                    linewidth=linewidth[imouse],
                    zorder=zorder[imouse],
                )
            title = f"Environment #{ienv+1}\n"
            ax[ienv].axhline(y=0, color="k", linewidth=0.5, zorder=-1)
            ax[ienv].set_title(title)
            ax[ienv].set_xlabel("Session #")
            ax[ienv].set_ylabel("Red Deviation")

        return fig


class ReliabilityQuantileFigureMaker(ReliabilityQuantileSummary):
    def __init__(self, tracked_mice: list[str], try_cache: bool = True, save_cache: bool = True):
        super().__init__(tracked_mice, try_cache, save_cache)
        self.add_selection("example_mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_integer("example_environment", value=0, min=0, max=self.max_environments - 1)
        self.add_integer("example_session", value=0, min=0, max=self.max_sessions - 1)
        self.add_float_range("fraction_active_threshold", value=(0, 1), min=0, max=1)
        self.add_boolean("blinded", value=True)
        self.add_boolean("group_novel", value=True)
        self.add_boolean("hide_quantiles", value=False)
        self.add_boolean("hide_red", value=False)
        self.add_button("save_example", label="Save Example", callback=self.save_example)
        self.on_change("example_mouse", self.update_example_mouse)
        self.on_change("example_environment", self.update_example_environment)
        self.update_example_mouse(self.state)

        self.mousedb = get_database("vrMice")
        self.ko = dict(zip(self.mousedb.get_table()["mouseName"], self.mousedb.get_table()["KO"]))

    def update_example_mouse(self, state):
        mouse_reliability = self.reliability_data[state["reliability_method"]][state["example_mouse"]]
        self.update_integer("example_environment", max=len(mouse_reliability["environments"]) - 1)
        self.update_example_environment(self.state)

    def update_example_environment(self, state):
        envstats = self.msms[state["example_mouse"]].env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        mouse_reliability = self.reliability_data[state["reliability_method"]][state["example_mouse"]]
        example_environment = env_in_order[state["example_environment"]]
        env_sessions = mouse_reliability["sessions"][example_environment]
        self.update_integer("example_session", max=len(env_sessions) - 1)

    def save_example(self, state):
        fig = self.plot(state)
        fig_dir = figure_dir("reliability_quantile_examples")
        name_elements = [
            state["example_mouse"],
            f"Env{state['example_environment']}",
            f"Session{state['example_session']}",
            state["reliability_method"],
            f"Quantile{state['quantile_focus']+1}-{state['num_bins']-1}",
            f"FractionActiveBW{state['fraction_active_threshold'][0]}-{state['fraction_active_threshold'][1]}",
        ]
        fig_name = "_".join(name_elements)
        if state["hide_red"]:
            fig_name += "_hide_red"
        if state["hide_quantiles"]:
            fig_name += "_hide_quantiles"
        if not state["blinded"]:
            fig_name += "_unblinded"
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_dir / fig_name)

    def plot(self, state):
        num_bins = state["num_bins"]
        quantile_focus = state["quantile_focus"]
        bins = np.linspace(0, 1, num_bins)
        centers = edge2center(bins)

        relbins = np.linspace(-1, 1, 21)
        relcenters = edge2center(relbins)

        reliability = self.reliability_data[state["reliability_method"]]

        ctl_key = "reliability_ctl" if state["selected"] else "reliability_ctl_all"
        red_key = "reliability_red" if state["selected"] else "reliability_red_all"
        fa_ctl_key = "fraction_active_ctl" if state["selected"] else "fraction_active_ctl_all"
        fa_red_key = "fraction_active_red" if state["selected"] else "fraction_active_red_all"
        ctl_reliability = {mouse: rel_data[ctl_key] for mouse, rel_data in reliability.items()}
        red_reliability = {mouse: rel_data[red_key] for mouse, rel_data in reliability.items()}

        num_mice = len(reliability)

        ctl_reliability = [np.full((num_mice, self.max_sessions, len(relcenters)), np.nan) for _ in range(self.max_environments)]
        red_reliability = [np.full((num_mice, self.max_sessions, len(relcenters)), np.nan) for _ in range(self.max_environments)]
        red_deviation = [np.full((num_mice, self.max_sessions, len(centers)), np.nan) for _ in range(self.max_environments)]

        for imouse, mouse in enumerate(reliability):
            envstats = self.msms[mouse].env_stats()
            env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
            env_in_order = [env for env in env_in_order if env != -1]

            if mouse == state["example_mouse"]:
                example_env_in_order = env_in_order

            for ienv, env in enumerate(env_in_order):
                zipped = zip(
                    reliability[mouse][ctl_key][env],
                    reliability[mouse][red_key][env],
                    reliability[mouse][fa_ctl_key][env],
                    reliability[mouse][fa_red_key][env],
                )
                for ises, (ctlval, redval, factlval, faredval) in enumerate(zipped):
                    if ises >= self.max_sessions:
                        continue

                    idx_control = (factlval >= state["fraction_active_threshold"][0]) & (factlval <= state["fraction_active_threshold"][1])
                    idx_red = (faredval >= state["fraction_active_threshold"][0]) & (faredval <= state["fraction_active_threshold"][1])
                    ctlval = ctlval[idx_control]
                    redval = redval[idx_red]

                    # Get the session number for this list element
                    sesnum = reliability[mouse]["sessions"][env][ises]
                    rel_sesnum = envstats[env].index(sesnum)

                    # Measure the reliability of the control and red cells (in true reliability value bins)
                    ctl_reliability[ienv][imouse, rel_sesnum] = fractional_histogram(ctlval, bins=relbins)[0]
                    red_reliability[ienv][imouse, rel_sesnum] = fractional_histogram(redval, bins=relbins)[0]

                    # Measure the quantiles of the control reliabilty data and use it to measure the deviation of the red cells relative to the quantiles
                    c_quantiles = np.quantile(ctlval, bins)
                    red_deviation[ienv][imouse, rel_sesnum] = fractional_histogram(redval, bins=c_quantiles)[0] - (1 / len(centers))

        # Stack across environments to combine novel potentially
        ctl_reliability = np.stack(ctl_reliability, axis=0)
        red_reliability = np.stack(red_reliability, axis=0)
        red_deviation = np.stack(red_deviation, axis=0)

        # Separate familiar and novel environments
        red_deviation_familiar = red_deviation[0]
        if state["group_novel"]:
            red_deviation_novel = np.nanmean(red_deviation[1:], axis=0)
        else:
            red_deviation_novel = red_deviation[1]

        # Prepare example data
        imouse_example = list(reliability.keys()).index(state["example_mouse"])
        ctl_data = reliability[state["example_mouse"]][ctl_key][example_env_in_order[state["example_environment"]]][state["example_session"]]
        ctl_fa_data = reliability[state["example_mouse"]][fa_ctl_key][example_env_in_order[state["example_environment"]]][state["example_session"]]
        ctl_data = ctl_data[(ctl_fa_data >= state["fraction_active_threshold"][0]) & (ctl_fa_data <= state["fraction_active_threshold"][1])]
        ctl_quantiles = np.quantile(ctl_data, bins)

        # Get colorscheme for mouse data
        colors_mice, linewidth, zorder = get_mouse_colors(reliability, blinded=state["blinded"], asdict=True, mousedb=self.mousedb)

        fontsize = 12

        fig = plt.figure(figsize=(9, 5), layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=(1, 1.5))
        gs_example = gs[0].subgridspec(2, 1)
        gs_summary = gs[1].subgridspec(2, 1)
        ax_ctl_example = fig.add_subplot(gs_example[0])
        ax_red_example = fig.add_subplot(gs_example[1])
        ax_summary_familiar = fig.add_subplot(gs_summary[0])
        ax_summary_novel = fig.add_subplot(gs_summary[1])

        ax_ctl_example.plot(
            relcenters,
            ctl_reliability[state["example_environment"]][imouse_example, state["example_session"]],
            color="k",
            linewidth=1.5,
        )
        if not state["hide_quantiles"]:
            for cq in ctl_quantiles:
                ax_ctl_example.axvline(cq, color="k", linewidth=0.5, zorder=-1)
        if not state["hide_red"]:
            ax_ctl_example.plot(
                relcenters,
                red_reliability[state["example_environment"]][imouse_example, state["example_session"]],
                color="r",
                linewidth=1.5,
            )
            ax_ctl_example.fill_between(
                relcenters,
                ctl_reliability[state["example_environment"]][imouse_example, state["example_session"]],
                red_reliability[state["example_environment"]][imouse_example, state["example_session"]],
                color="r",
                alpha=0.2,
            )
        max_yval = max(
            np.max(ctl_reliability[state["example_environment"]][imouse_example, state["example_session"]]),
            np.max(red_reliability[state["example_environment"]][imouse_example, state["example_session"]]),
        )

        ylim = (0, max_yval * 1.1)
        ax_ctl_example.set_ylim(ylim)
        format_spines(
            ax_ctl_example,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(-1, 1),
            xticks=[-1, 0, 1],
            ybounds=ylim,
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        ax_ctl_example.set_xlabel("Reliability Values", fontsize=fontsize)
        ax_ctl_example.set_ylabel("Relative Counts", fontsize=fontsize)

        ax_red_example.bar(
            range(1, len(centers) + 1),
            red_deviation[state["example_environment"]][imouse_example, state["example_session"]],
            facecolor=("r", 0.2),
            edgecolor="r",
            width=1,
            linewidth=1.5,
        )
        ax_red_example.axhline(y=0, color="k", linewidth=2.5, zorder=100)
        ax_red_example.set_xlim(0.3, len(centers) + 0.7)
        ylim = ax_red_example.get_ylim()
        format_spines(
            ax_red_example,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(1, len(centers)),
            xticks=range(1, len(centers) + 1),
            ybounds=(ylim[0], ylim[1]),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        ax_red_example.set_xlabel("Ctl Reliability Quantiles", fontsize=fontsize)
        ax_red_example.set_ylabel("$\Delta$Red Reliability", fontsize=fontsize)

        alpha = 1.0 if state["blinded"] else 0.3
        if not state["blinded"]:
            ko_array = np.array([self.ko[mouse] for mouse in reliability])

        ylim_min = 1
        ylim_max = -1
        for imouse, mouse in enumerate(reliability):
            ax_summary_familiar.plot(
                range(self.max_sessions),
                red_deviation_familiar[imouse][:, quantile_focus],
                color=(colors_mice[mouse], alpha),
                linewidth=linewidth[mouse],
                zorder=zorder[mouse],
            )
        if not state["blinded"]:
            ax_summary_familiar.plot(
                range(self.max_sessions),
                np.nanmean(red_deviation_familiar[ko_array, :, quantile_focus], axis=0),
                color="purple",
                linewidth=2,
                zorder=3,
            )
            ax_summary_familiar.plot(
                range(self.max_sessions),
                np.nanmean(red_deviation_familiar[~ko_array, :, quantile_focus], axis=0),
                color="gray",
                linewidth=2,
                zorder=2.5,
            )
        ax_summary_familiar.axhline(y=0, color="k", linewidth=0.5, zorder=-1)
        ax_summary_familiar.set_xlim(-0.5, self.max_sessions - 0.5)
        ylim = ax_summary_familiar.get_ylim()
        ylim_min = min(ylim_min, ylim[0])
        ylim_max = max(ylim_max, ylim[1])
        ax_summary_familiar.set_ylabel(f"$\Delta$Red Reliability\nQuantile {quantile_focus+1}/{num_bins-1}", fontsize=fontsize)

        for imouse, mouse in enumerate(reliability):
            ax_summary_novel.plot(
                range(self.max_sessions),
                red_deviation_novel[imouse][:, quantile_focus],
                color=(colors_mice[mouse], alpha),
                linewidth=linewidth[mouse],
                zorder=zorder[mouse],
            )
        if not state["blinded"]:
            ax_summary_novel.plot(
                range(self.max_sessions),
                np.nanmean(red_deviation_novel[ko_array, :, quantile_focus], axis=0),
                color="purple",
                linewidth=2,
                zorder=3,
            )
            ax_summary_novel.plot(
                range(self.max_sessions),
                np.nanmean(red_deviation_novel[~ko_array, :, quantile_focus], axis=0),
                color="gray",
                linewidth=2,
                zorder=2,
            )
        ax_summary_novel.axhline(y=0, color="k", linewidth=0.5, zorder=-1)
        ax_summary_novel.set_xlim(-0.5, self.max_sessions - 0.5)
        ylim = ax_summary_novel.get_ylim()
        ylim_min = min(ylim_min, ylim[0])
        ylim_max = max(ylim_max, ylim[1])
        ax_summary_novel.set_ylabel(f"$\Delta$Red Reliability\nQuantile {quantile_focus+1}/{num_bins-1}", fontsize=fontsize)

        ylim_min = np.ceil(ylim_min * 100) / 100
        ylim_max = np.floor(ylim_max * 100) / 100

        names = ["Familiar Environment", "Novel Environment"]
        for ienv, ax in enumerate([ax_summary_familiar, ax_summary_novel]):
            ax.set_ylim(ylim_min, ylim_max)
            ax.text(self.max_sessions - 1, ylim_max, names[ienv], ha="right", va="top", fontsize=fontsize, color="k")

        format_spine_kwargs = dict(x_pos=-0.05, y_pos=0, ybounds=(ylim_min, ylim_max), yticks=[ylim_min, 0, ylim_max], tick_length=4)
        format_spines(ax_summary_familiar, xticks=[], spines_visible=["left"], **format_spine_kwargs)
        format_spines(ax_summary_novel, xbounds=(0, self.max_sessions - 1), spines_visible=["left", "bottom"], **format_spine_kwargs)
        ax_summary_novel.set_xlabel("Session #", fontsize=fontsize)

        pilot_colors = [colors_mice[mouse] for mouse in ["CR_Hippocannula6", "CR_Hippocannula7"]]
        blinded_colors = [colors_mice[mouse] for mouse in reliability if "CR_" not in mouse]
        blinded_study_legend(
            ax_summary_novel,
            xpos=(self.max_sessions - 1) * 1.0,
            ypos=ylim_max * 0.55,
            pilot_colors=pilot_colors,
            blinded_colors=blinded_colors,
            blinded=state["blinded"],
            origin="lower_right",
            fontsize=fontsize,
        )

        return fig


class FractionActiveQuantileFigureMaker(ReliabilityQuantileSummary):
    def __init__(self, tracked_mice: list[str], try_cache: bool = True, save_cache: bool = True):
        super().__init__(tracked_mice, try_cache, save_cache)
        self.add_selection("example_mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_integer("example_environment", value=0, min=0, max=self.max_environments - 1)
        self.add_integer("example_session", value=0, min=0, max=self.max_sessions - 1)
        self.add_float_range("reliability_threshold", value=(-1.0, 1.0), min=-1.0, max=1.0)
        self.add_button("save_example", label="Save Example", callback=self.save_example)
        self.on_change("example_mouse", self.update_example_mouse)
        self.on_change("example_environment", self.update_example_environment)
        self.update_example_mouse(self.state)

    def update_example_mouse(self, state):
        mouse_reliability = self.reliability_data[state["reliability_method"]][state["example_mouse"]]
        self.update_integer("example_environment", max=len(mouse_reliability["environments"]) - 1)
        self.update_example_environment(self.state)

    def update_example_environment(self, state):
        envstats = self.msms[state["example_mouse"]].env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        mouse_reliability = self.reliability_data[state["reliability_method"]][state["example_mouse"]]
        example_environment = env_in_order[state["example_environment"]]
        env_sessions = mouse_reliability["sessions"][example_environment]
        self.update_integer("example_session", max=len(env_sessions) - 1)

    def save_example(self, state):
        fig = self.plot(state)
        fig_dir = figure_dir("fraction_active_quantile_examples")
        name_elements = [
            state["example_mouse"],
            f"Env{state['example_environment']}",
            f"Session{state['example_session']}",
            state["reliability_method"],
            f"Quantile{state['quantile_focus']+1}-{state['num_bins']-1}",
            f"ReliabilityBW{state['reliability_threshold'][0]}-{state['reliability_threshold'][1]}",
        ]
        fig_name = "_".join(name_elements)
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_dir / fig_name)

    def plot(self, state):
        num_bins = state["num_bins"]
        quantile_focus = state["quantile_focus"]
        bins = np.linspace(0, 1, num_bins)
        centers = edge2center(bins)

        relbins = np.linspace(0, 1, 21)
        relcenters = edge2center(relbins)

        reliability = self.reliability_data[state["reliability_method"]]

        ctl_key = "reliability_ctl" if state["selected"] else "reliability_ctl_all"
        red_key = "reliability_red" if state["selected"] else "reliability_red_all"
        fa_ctl_key = "fraction_active_ctl" if state["selected"] else "fraction_active_ctl_all"
        fa_red_key = "fraction_active_red" if state["selected"] else "fraction_active_red_all"
        num_mice = len(reliability)

        ctl_fraction_active = [np.full((num_mice, self.max_sessions, len(relcenters)), np.nan) for _ in range(self.max_environments)]
        red_fraction_active = [np.full((num_mice, self.max_sessions, len(relcenters)), np.nan) for _ in range(self.max_environments)]
        red_deviation = [np.full((num_mice, self.max_sessions, len(centers)), np.nan) for _ in range(self.max_environments)]

        for imouse, mouse in enumerate(reliability):
            envstats = self.msms[mouse].env_stats()
            env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
            env_in_order = [env for env in env_in_order if env != -1]

            if mouse == state["example_mouse"]:
                example_env_in_order = env_in_order

            for ienv, env in enumerate(env_in_order):
                zipped = zip(
                    reliability[mouse][ctl_key][env],
                    reliability[mouse][red_key][env],
                    reliability[mouse][fa_ctl_key][env],
                    reliability[mouse][fa_red_key][env],
                )
                for ises, (ctlval, redval, factlval, faredval) in enumerate(zipped):
                    if ises >= self.max_sessions:
                        continue

                    idx_control = (ctlval >= state["reliability_threshold"][0]) & (ctlval <= state["reliability_threshold"][1])
                    idx_red = (redval >= state["reliability_threshold"][0]) & (redval <= state["reliability_threshold"][1])
                    factlval = factlval[idx_control]
                    faredval = faredval[idx_red]

                    # Get the session number for this list element
                    sesnum = reliability[mouse]["sessions"][env][ises]
                    rel_sesnum = envstats[env].index(sesnum)

                    # Measure the reliability of the control and red cells (in true reliability value bins)
                    ctl_fraction_active[ienv][imouse, rel_sesnum] = fractional_histogram(factlval, bins=relbins)[0]
                    red_fraction_active[ienv][imouse, rel_sesnum] = fractional_histogram(faredval, bins=relbins)[0]

                    # Measure the quantiles of the control reliabilty data and use it to measure the deviation of the red cells relative to the quantiles
                    c_quantiles = np.quantile(factlval, bins)
                    red_deviation[ienv][imouse, rel_sesnum] = fractional_histogram(faredval, bins=c_quantiles)[0] - (1 / len(centers))

        # Prepare example data
        imouse_example = list(reliability.keys()).index(state["example_mouse"])
        ctl_data = reliability[state["example_mouse"]][fa_ctl_key][example_env_in_order[state["example_environment"]]][state["example_session"]]
        ctl_quantiles = np.quantile(ctl_data, bins)

        colors_mice, linewidth, zorder = get_mouse_colors(reliability, blinded=True, asdict=True)

        fig = plt.figure(figsize=(8, 5), layout="constrained")
        gs = fig.add_gridspec(1, 2)
        gs_example = gs[0].subgridspec(2, 1)
        gs_summary = gs[1].subgridspec(3, 1)
        ax_ctl_example = fig.add_subplot(gs_example[0])
        ax_red_example = fig.add_subplot(gs_example[1])
        ax_env0_summary = fig.add_subplot(gs_summary[0])
        ax_env1_summary = fig.add_subplot(gs_summary[1])
        ax_env2_summary = fig.add_subplot(gs_summary[2])

        ax_ctl_example.plot(
            relcenters,
            ctl_fraction_active[state["example_environment"]][imouse_example, state["example_session"]],
            color="k",
            linewidth=1.5,
        )
        for cq in ctl_quantiles:
            ax_ctl_example.axvline(cq, color="k", linewidth=0.5, zorder=-1)
        ax_ctl_example.plot(
            relcenters,
            red_fraction_active[state["example_environment"]][imouse_example, state["example_session"]],
            color="r",
            linewidth=1.5,
        )
        ax_ctl_example.fill_between(
            relcenters,
            ctl_fraction_active[state["example_environment"]][imouse_example, state["example_session"]],
            red_fraction_active[state["example_environment"]][imouse_example, state["example_session"]],
            color="r",
            alpha=0.2,
        )

        ylim = ax_ctl_example.get_ylim()
        ax_ctl_example.set_ylim(0, ylim[1])
        format_spines(
            ax_ctl_example,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, 1),
            xticks=[0, 1],
            ybounds=(0, ylim[1]),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        ax_ctl_example.set_xlabel("Fraction Active Values")
        ax_ctl_example.set_ylabel("Fractional Counts")

        ax_red_example.plot(
            centers,
            red_deviation[state["example_environment"]][imouse_example, state["example_session"]],
            color="r",
            linewidth=1.5,
        )
        ax_red_example.fill_between(
            centers,
            0,
            red_deviation[state["example_environment"]][imouse_example, state["example_session"]],
            color="r",
            alpha=0.2,
        )
        ax_red_example.axhline(y=0, color="k", linewidth=0.5, zorder=-1)

        ylim = ax_red_example.get_ylim()
        format_spines(
            ax_red_example,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, 1),
            xticks=bins,
            xlabels=["0", *["" for _ in range(len(bins) - 2)], "1"],
            ybounds=(ylim[0], ylim[1]),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        ax_red_example.set_xlabel("Fraction Active Quantiles")
        ax_red_example.set_ylabel("$\Delta$Red Fraction Active")

        ylim_min = 1
        ylim_max = -1
        for ienv, ax in enumerate([ax_env0_summary, ax_env1_summary, ax_env2_summary]):
            for imouse, mouse in enumerate(reliability):
                ax.plot(
                    range(self.max_sessions),
                    red_deviation[ienv][imouse][:, quantile_focus],
                    color=colors_mice[mouse],
                    linewidth=linewidth[mouse],
                    zorder=zorder[mouse],
                )
            ax.axhline(y=0, color="k", linewidth=0.5, zorder=-1)
            ax.set_xlim(-0.5, self.max_sessions - 0.5)
            ylim = ax.get_ylim()
            ylim_min = min(ylim_min, ylim[0])
            ylim_max = max(ylim_max, ylim[1])
            ax.set_ylabel(f"$\Delta$Red Fraction Active\nQuantile {quantile_focus+1}/{num_bins-1}")

        ylim_min = np.ceil(ylim_min * 100) / 100
        ylim_max = np.floor(ylim_max * 100) / 100
        # ylim_min = -0.1
        # ylim_max = 0.225
        for ienv, ax in enumerate([ax_env0_summary, ax_env1_summary, ax_env2_summary]):
            ax.set_ylim(ylim_min, ylim_max)
            env_name = f"Environment #{ienv+1}\n"
            ax.text(self.max_sessions - 1, ylim_max, env_name, ha="right", va="top", fontsize=12, color="k")

        format_spine_kwargs = dict(x_pos=-0.05, y_pos=0, ybounds=(ylim_min, ylim_max), yticks=[ylim_min, 0, ylim_max], tick_length=4)
        format_spines(ax_env0_summary, xticks=[], spines_visible=["left"], **format_spine_kwargs)
        format_spines(ax_env1_summary, xticks=[], spines_visible=["left"], **format_spine_kwargs)
        format_spines(ax_env2_summary, xbounds=(0, self.max_sessions - 1), spines_visible=["left", "bottom"], **format_spine_kwargs)
        ax_env2_summary.set_xlabel("Session #")

        pilot_colors = [colors_mice[mouse] for mouse in ["CR_Hippocannula6", "CR_Hippocannula7"]]
        blinded_colors = [colors_mice[mouse] for mouse in reliability if "CR_" not in mouse]
        blinded_study_legend(
            ax_env2_summary,
            xpos=(self.max_sessions - 1) * 1.0,
            ypos=ylim_max * 0.2,
            pilot_colors=pilot_colors,
            blinded_colors=blinded_colors,
            origin="lower_right",
        )

        return fig
