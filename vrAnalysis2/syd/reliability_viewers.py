from typing import Optional
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from syd import Viewer
from vrAnalysis2.tracking import Tracker
from vrAnalysis2.processors.spkmaps import SpkmapProcessor
from vrAnalysis2.helpers import color_violins, fractional_histogram, edge2center


class ReliabilityTrajectory(Viewer):
    def __init__(self, tracked_mice: list[str]):
        self.tracked_mice = list(tracked_mice)
        self._reliability = {
            method: {smooth_width: {mouse: None for mouse in self.tracked_mice} for smooth_width in [1, 5]}
            for method in ["leave_one_out", "mse", "correlation"]
        }
        self.trackers = {mouse: None for mouse in self.tracked_mice}

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "mse", "correlation"])
        self.add_selection("smooth_width", value=1, options=[1, 5])
        self.add_boolean("use_session_filters", value=False)
        self.add_boolean("subsample_ctl", value=False)
        self.add_integer("num_subsamples", value=20, min=1, max=100)
        self.add_boolean("show_distribution", value=True)
        self.add_float("widths", value=0.5, min=0.1, max=3.0)

    def get_tracker(self, mouse: str) -> Tracker:
        if self.trackers[mouse] is None:
            self.trackers[mouse] = Tracker(mouse)
        return self.trackers[mouse]

    def get_environments(self, track: Tracker) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        environments = np.unique(np.concatenate([session.environments for session in track.sessions]))
        return environments

    def get_reliability(self, mouse: str, method: str, smooth_width: float, exclude_environments: Optional[list[int] | int] = [-1]) -> dict:
        """Get reliability data for all tracked sessions"""
        if self._reliability[method][smooth_width][mouse] is not None:
            return self._reliability[method][smooth_width][mouse]

        track = self.get_tracker(mouse)
        environments = list(self.get_environments(track))
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
            idx_red_all = session.loadone("mpciROIs.redCellIdx")
            idx_red = idx_red_all[idx_rois]

            smp = SpkmapProcessor(session)
            reliability_all = smp.get_reliability(use_session_filters=False, params=dict(smooth_width=float(smooth_width)))
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
        environments = reliability["environments"]
        sessions = reliability["sessions"]
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
        self.add_boolean("use_session_filters", value=False)
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
            idx_red_all = session.loadone("mpciROIs.redCellIdx")
            idx_red = idx_red_all[idx_rois]

            smp = SpkmapProcessor(session)
            reliability_all = smp.get_reliability(use_session_filters=False, params=dict(smooth_width=float(smooth_width)))
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
            ctl_counts, bins = fractional_histogram(reliability_ctl[env][ises], bins=50)
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
