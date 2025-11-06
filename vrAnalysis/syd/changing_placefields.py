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
    edge2center,
)
from ..database import get_database
from .reliability_continuity import ReliabilityStabilitySummary
from ..analysis.tracked_plasticity.utils import all_combos


def figure_dir(folder: str) -> Path:
    return repo_path() / "figures" / "before_the_reveal" / folder


temp_data_dir = analysis_path() / "before_the_reveal_temp_data_new251024"


def gather_combo_data(mice: list[str], state: dict):
    rss = ReliabilityStabilitySummary(mice)
    results = {}
    for mouse in mice:
        msm = MultiSessionSpkmaps(Tracker(mouse))
        env_stats = msm.env_stats()
        env_in_order = sorted(env_stats, key=lambda x: env_stats[x][0])
        env_in_order = [int(env) for env in env_in_order if env != -1]
        state["mouse"] = mouse
        results_path = temp_data_dir / rss.get_results_name(state)
        if results_path.exists():
            results[mouse] = joblib.load(results_path)
            results[mouse]["env_in_order"] = env_in_order
            results[mouse]["env_stats"] = env_stats
        else:
            print(f"No results found for {mouse} in {results_path}")
    return results


# I want to see what it looks like when place fields change
# Data:
# -- In principle I can get all the summary data from the forward/backward cached data.....
# Plots:
# -- Example: Spkmap over several sessions (maybe with a particular correlation coefficient from first to last?)
# --          - Either full trials or just the average place field....
# Components:
# -- Mouse selector
# -- Environment selector (in order of sessions, e.g. familiar, novel, novel-2)
# -- Session limiters (limit the sessions that can be chosen from for the combo, this is a way to keep it similar for all mice!)
# -- Combo selector (for sessions, it's a selection object where the combos are predefined)
# -- Reliability range
# -- Fraction active range
# -- Correlation coefficient range (for first to last session in combo)


class ChangingPlaceFieldFocusedViewer(Viewer):
    def __init__(self, tracked_mice: list[str]):
        self.tracked_mice = list(tracked_mice)
        self.multisessions = {mouse: None for mouse in self.tracked_mice}

        self.mousedb = get_database("vrMice")
        self.ko = dict(zip(self.mousedb.get_table()["mouseName"], self.mousedb.get_table()["KO"]))

        # This is a different viewer with data handling capabilities for reliability summary data
        self.summary_viewer = ReliabilityStabilitySummary(self.tracked_mice)

        # Set up syd parameters
        self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
        self.add_integer("environment", value=0, min=0, max=3)
        self.add_integer("session_limit", value=10, min=1, max=100)
        self.add_integer("max_session_diff", value=5, min=1, max=12)
        self.add_selection("session_combo", value=(0, 1), options=[(0, 1)])
        self.add_float_range("reliability_range", value=(0.5, 1.0), min=-1.0, max=1.0)
        self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "correlation"])
        self.add_selection("smooth_width", value=5, options=[1, 5])
        self.add_float_range("fraction_range", value=(0.00, 1.0), min=0.0, max=1.0)
        self.add_selection("activity_method", value="rms", options=FractionActive.activity_methods)
        self.add_selection("fraction_method", value="participation", options=FractionActive.fraction_methods)
        self.add_selection("spks_type", value="significant", options=["significant", "oasis"])
        self.add_boolean("use_session_filters", value=True)
        self.add_float_range("corr_coef_range", value=(-1.0, 1.0), min=-1.0, max=1.0)
        self.add_integer("ctl_roi_idx", value=0, min=0, max=100)
        self.add_integer("red_roi_idx", value=0, min=0, max=100)
        self.add_float("vmax_spkmap", value=5.0, min=0.1, max=15.0)
        self.add_float("correlation_xmin", value=-1.0, min=-1.0, max=1.0)
        self.add_button("save_example", label="Save example", callback=self.save_example)

        # Set up callbacks
        self.on_change("mouse", self.reset_mouse)
        self.on_change("environment", self.reset_environment)
        self.on_change(["session_limit", "max_session_diff"], self.reset_combos)
        self.on_change(
            [
                "session_combo",
                "reliability_method",
                "smooth_width",
                "activity_method",
                "fraction_method",
                "spks_type",
                "use_session_filters",
            ],
            self.gather_example_data,
        )
        self.on_change(
            [
                "reliability_range",
                "fraction_range",
                "corr_coef_range",
            ],
            self.set_roi_options,
        )
        self.reset_mouse(self.state)

    def save_example(self, state):
        fig = self.plot(state)
        fig_dir = figure_dir("changing_placefields_focus")
        mouse = state["mouse"]
        environment = state["environment"]
        sessions = ",".join([str(s) for s in state["session_combo"]])
        idx_tracked = self._idx_tracked
        idx_ctl_keeps = self._idx_ctl_keeps
        idx_red_keeps = self._idx_red_keeps
        ctl_roi_idx = state["ctl_roi_idx"]
        red_roi_idx = state["red_roi_idx"]
        true_ctl_idx = idx_tracked[0, idx_ctl_keeps[ctl_roi_idx]]
        true_red_idx = idx_tracked[0, idx_red_keeps[red_roi_idx]]
        fig_name = f"{mouse}_env{environment}_ses{sessions}_ctl{true_ctl_idx}_red{true_red_idx}"
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_dir / fig_name)
        plt.close(fig)

    def get_multisession(self, mouse: str) -> MultiSessionSpkmaps:
        if self.multisessions[mouse] is None:
            tracker = Tracker(mouse)
            self.multisessions[mouse] = MultiSessionSpkmaps(tracker)
        return self.multisessions[mouse]

    def reset_mouse(self, state):
        msm = self.get_multisession(state["mouse"])
        environments = list(msm.env_stats().keys())
        num_envs = np.sum(np.array(environments) != -1)
        self.update_integer("environment", max=num_envs - 1)
        self.reset_environment(self.state)

    def reset_environment(self, state):
        msm = self.get_multisession(state["mouse"])
        envstats = msm.env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        env = env_in_order[state["environment"]]
        idx_ses = msm.idx_ses_with_env(env)
        self.update_integer("session_limit", value=len(idx_ses) - 1, max=len(idx_ses) - 1)
        self.reset_combos(self.state)

    def reset_combos(self, state):
        msm = self.get_multisession(state["mouse"])
        envstats = msm.env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        env = env_in_order[state["environment"]]
        idx_ses = msm.idx_ses_with_env(env)[: state["session_limit"]]
        combos = all_combos(
            idx_ses,
            state["max_session_diff"],
            continuous=True,
        )
        self.update_selection("session_combo", options=combos)
        self.gather_example_data(self.state)

    def gather_example_data(self, state):
        msm = self.get_multisession(state["mouse"])
        envstats = msm.env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        env = env_in_order[state["environment"]]
        idx_ses = state["session_combo"]
        spkmaps, extras = msm.get_spkmaps(
            env,
            average=False,
            reliability_method=state["reliability_method"],
            smooth=float(state["smooth_width"]),
            spks_type=state["spks_type"],
            idx_ses=idx_ses,
            tracked=True,
            use_session_filters=state["use_session_filters"],
            pop_nan=True,
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
        avg_spkmaps = [np.nanmean(spkmap, axis=1) for spkmap in spkmaps]
        correlation = vectorCorrelation(avg_spkmaps[0], avg_spkmaps[-1])

        self._spkmaps = spkmaps
        self._avg_spkmaps = avg_spkmaps
        self._correlation = correlation
        self._pfloc = extras["pfloc"]
        self._reliability = extras["reliability"]
        self._fraction_active = fraction_active
        self._idx_tracked = extras["idx_tracked"]
        self._idx_red = np.any(np.stack(extras["idx_red"]), axis=0)

        self.set_roi_options(self.state)

    def set_roi_options(self, state):
        idx_reliable_keeps = np.all(
            np.stack([(rel >= state["reliability_range"][0]) & (rel <= state["reliability_range"][1]) for rel in self._reliability]),
            axis=0,
        )
        idx_active_keeps = np.all(
            np.stack([(fa >= state["fraction_range"][0]) & (fa <= state["fraction_range"][1]) for fa in self._fraction_active]),
            axis=0,
        )
        idx_corr_keeps = (self._correlation >= state["corr_coef_range"][0]) & (self._correlation <= state["corr_coef_range"][1])
        idx_ctl_keeps = np.where(idx_reliable_keeps & idx_active_keeps & ~self._idx_red & idx_corr_keeps)[0]
        idx_red_keeps = np.where(idx_reliable_keeps & idx_active_keeps & self._idx_red & idx_corr_keeps)[0]
        self._idx_reliable_keeps = idx_reliable_keeps
        self._idx_active_keeps = idx_active_keeps
        self._idx_corr_keeps = idx_corr_keeps
        self._idx_ctl_keeps = idx_ctl_keeps
        self._idx_red_keeps = idx_red_keeps
        self.update_integer("ctl_roi_idx", max=len(idx_ctl_keeps) - 1)
        self.update_integer("red_roi_idx", max=len(idx_red_keeps) - 1)

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

    def plot(self, state):
        idx_tracked = self._idx_tracked
        spkmaps = self._spkmaps
        avg_spkmaps = self._avg_spkmaps
        pfloc = np.stack(self._pfloc)
        reliability = np.stack(self._reliability)
        fraction_active = np.stack(self._fraction_active)
        correlation = self._correlation
        idx_ctl_keeps = self._idx_ctl_keeps
        idx_red_keeps = self._idx_red_keeps
        show_red = len(idx_red_keeps) > 0
        ctl_roi_idx = state["ctl_roi_idx"]
        red_roi_idx = state["red_roi_idx"]
        true_ctl_idx = ", ".join([str(i) for i in idx_tracked[[0, -1], idx_ctl_keeps[ctl_roi_idx]]])
        if show_red:
            true_red_idx = ", ".join([str(i) for i in idx_tracked[[0, -1], idx_red_keeps[red_roi_idx]]])
        else:
            true_red_idx = "None"

        # This is the set of ROIs we select from (then we subselect by ctl/red and correlation coefficient)
        idx_select_from = self._idx_reliable_keeps & self._idx_active_keeps

        # Example data for selected ROIs
        ctl_spkmaps = [s[idx_ctl_keeps] for s in spkmaps]
        ctl_placefields = np.stack([np.nanmean(s, axis=1) for s in ctl_spkmaps])
        ctl_roi_placefields = ctl_placefields[:, ctl_roi_idx]
        if show_red:
            red_spkmaps = [s[idx_red_keeps] for s in spkmaps]
            red_placefields = np.stack([np.nanmean(s, axis=1) for s in red_spkmaps])
            red_roi_placefields = red_placefields[:, red_roi_idx]

        ctl_roi_spkmap, snums = self._make_roi_trajectory(spkmaps, idx_ctl_keeps[ctl_roi_idx], dead_trials=2)
        if show_red:
            red_roi_spkmap, _ = self._make_roi_trajectory(spkmaps, idx_red_keeps[red_roi_idx], dead_trials=2)
        spkmap_yticks = []
        for ises in range(len(spkmaps)):
            trials_in_ses = np.where(snums == ises)[0]
            center_trial = np.mean(trials_in_ses)
            spkmap_yticks.append(center_trial)

        # Get summary stats
        bins = np.linspace(-1.0, 1.0, 21)
        centers = edge2center(bins)
        corr_counts = np.histogram(correlation[idx_select_from], bins=bins)[0]
        corr_counts_ctl = np.histogram(correlation[idx_ctl_keeps], bins=bins)[0]
        corr_counts = corr_counts / np.sum(corr_counts)
        corr_counts_ctl = corr_counts_ctl / np.sum(corr_counts_ctl)
        corr_val_ctl_roi = correlation[idx_ctl_keeps[ctl_roi_idx]]
        if show_red:
            corr_counts_red = np.histogram(correlation[idx_red_keeps], bins=bins)[0]
            corr_counts_red = corr_counts_red / np.sum(corr_counts_red)
            corr_val_red_roi = correlation[idx_red_keeps[red_roi_idx]]

        fig = plt.figure(figsize=(7, 7), layout="constrained")
        main_gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        spkmap_gs = main_gs[0].subgridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 2])
        summary_gs = main_gs[1].subgridspec(3, 1)
        ax_ctl_placefield = fig.add_subplot(spkmap_gs[0, 0])
        ax_red_placefield = fig.add_subplot(spkmap_gs[0, 1])
        ax_ctl_spkmaps = fig.add_subplot(spkmap_gs[1, 0])
        ax_red_spkmaps = fig.add_subplot(spkmap_gs[1, 1])
        ax_correlation = fig.add_subplot(summary_gs[0])
        ax_reliability = fig.add_subplot(summary_gs[1])
        ax_fraction_active = fig.add_subplot(summary_gs[2])

        placefield_ctl_cmap = mpl.colormaps["Greys"]
        placefield_ctl_cmap = mpl.colormaps["Greys"]
        placefield_red_cmap = mpl.colormaps["Reds"]
        pf_ctl_colors = placefield_ctl_cmap(np.linspace(1.0, 0.4, ctl_roi_placefields.shape[0]))
        pf_red_colors = placefield_red_cmap(np.linspace(1.0, 0.4, ctl_roi_placefields.shape[0]))
        for isession in range(ctl_placefields.shape[0]):
            ax_ctl_placefield.plot(
                range(ctl_roi_placefields.shape[1]),
                ctl_roi_placefields[isession],
                color=pf_ctl_colors[isession],
                linewidth=1.0,
            )
            if show_red:
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
        if show_red:
            format_spines(
                ax_red_placefield,
                x_pos=-0.05,
                y_pos=-0.05,
                xbounds=(0, ctl_roi_placefields.shape[1]),
                ybounds=ax_red_placefield.get_ylim(),
                spines_visible=["bottom", "left"],
                tick_length=4,
            )
        else:
            ax_red_placefield.axis("off")
        ax_ctl_placefield.set_xlabel("Position (cm)")
        ax_ctl_placefield.set_ylabel("Activity ($\sigma$)")
        ax_ctl_placefield.set_title(f"Ctl ROI: {true_ctl_idx}")
        ax_red_placefield.set_xlabel("Position (cm)")
        if show_red:
            ax_red_placefield.set_title(f"Red ROI: {true_red_idx}")
        else:
            ax_red_placefield.set_title("No red ROIs found")

        spkmap_ctl_cmap = mpl.colormaps["Greys"]
        spkmap_ctl_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "greys_clipped",
            spkmap_ctl_cmap(np.linspace(0.08, 1.0, 256)),
        )
        spkmap_ctl_cmap.set_bad("white")  # ("orange", 0.3))
        spkmap_red_cmap = mpl.colormaps["Reds"]
        spkmap_red_cmap.set_bad("white")  # ("orange", 0.3))
        ax_ctl_spkmaps.imshow(ctl_roi_spkmap, aspect="auto", cmap=spkmap_ctl_cmap, interpolation="none", vmin=0, vmax=state["vmax_spkmap"])
        if show_red:
            ax_red_spkmaps.imshow(red_roi_spkmap, aspect="auto", cmap=spkmap_red_cmap, interpolation="none", vmin=0, vmax=state["vmax_spkmap"])

        format_spines(
            ax_ctl_spkmaps,
            x_pos=-0.05,
            y_pos=-0.02,
            xbounds=(0, ctl_roi_spkmap.shape[1]),
            ybounds=ax_ctl_spkmaps.get_ylim(),
            yticks=spkmap_yticks,
            ylabels=range(len(spkmaps)),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )
        if show_red:
            format_spines(
                ax_red_spkmaps,
                x_pos=-0.05,
                y_pos=-0.02,
                xbounds=(0, ctl_roi_spkmap.shape[1]),
                ybounds=ax_ctl_spkmaps.get_ylim(),
                yticks=spkmap_yticks,
                ylabels=[],
                spines_visible=["bottom", "left"],
                tick_length=4,
            )
        else:
            ax_red_spkmaps.set_visible(False)

        ax_ctl_spkmaps.set_xlabel("Position (cm)")
        ax_ctl_spkmaps.set_ylabel("Session #")
        ax_red_spkmaps.set_xlabel("Position (cm)")

        width = bins[1] - bins[0]
        ax_correlation.bar(centers, corr_counts, width=width, color="k", alpha=0.4)
        ax_correlation.plot(centers, corr_counts_ctl, color="k", linewidth=1.5)
        ax_correlation.axvline(corr_val_ctl_roi, color="k", linestyle="-", linewidth=2.0)
        if show_red:
            ax_correlation.plot(centers, corr_counts_red, color="r", linewidth=1.5)
            ax_correlation.axvline(corr_val_red_roi, color="r", linestyle="-", linewidth=2.0)
        ax_correlation.set_xlabel("Correlation coefficient")
        ax_correlation.set_ylabel("Relative counts")
        ax_correlation.set_xlim(state["correlation_xmin"], 1.0)
        ax_correlation.set_title(f"#Ctl:{len(idx_ctl_keeps)}, #Red:{len(idx_red_keeps)}")
        format_spines(
            ax_correlation,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(state["correlation_xmin"], 1.0),
            ybounds=ax_correlation.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        ctl_violins = ax_reliability.violinplot(
            list(reliability[:, idx_ctl_keeps]),
            positions=range(reliability.shape[0]),
            widths=0.5,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            side="low",
        )
        color_violins(ctl_violins, facecolor="k", linecolor="k")
        ax_reliability.plot(
            range(reliability.shape[0]),
            reliability[:, idx_ctl_keeps[ctl_roi_idx]],
            color="k",
            linewidth=1.5,
            marker=".",
            markersize=7,
        )
        if show_red:
            red_violins = ax_reliability.violinplot(
                list(reliability[:, idx_red_keeps]),
                positions=range(reliability.shape[0]),
                widths=0.5,
                showmeans=False,
                showextrema=False,
                showmedians=False,
                side="high",
            )
            color_violins(red_violins, facecolor="r", linecolor="r")
            ax_reliability.plot(
                range(reliability.shape[0]),
                reliability[:, idx_red_keeps[red_roi_idx]],
                color="r",
                linewidth=1.5,
                marker=".",
                markersize=7,
            )
        ax_reliability.set_xlabel("Session")
        ax_reliability.set_ylabel("Reliability")
        format_spines(
            ax_reliability,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, reliability.shape[0] - 1),
            ybounds=ax_reliability.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        ctl_violins = ax_fraction_active.violinplot(
            list(fraction_active[:, idx_ctl_keeps]),
            positions=range(reliability.shape[0]),
            widths=0.5,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            side="low",
        )
        color_violins(ctl_violins, facecolor="k", linecolor="k")
        ax_fraction_active.plot(
            range(reliability.shape[0]),
            fraction_active[:, idx_ctl_keeps[ctl_roi_idx]],
            color="k",
            linewidth=1.5,
            marker=".",
            markersize=7,
        )
        if show_red:
            red_violins = ax_fraction_active.violinplot(
                list(fraction_active[:, idx_red_keeps]),
                positions=range(reliability.shape[0]),
                widths=0.5,
                showmeans=False,
                showextrema=False,
                showmedians=False,
                side="high",
            )

            color_violins(red_violins, facecolor="r", linecolor="r")
            ax_fraction_active.plot(
                range(reliability.shape[0]),
                fraction_active[:, idx_red_keeps[red_roi_idx]],
                color="r",
                linewidth=1.5,
                marker=".",
                markersize=7,
            )
        ax_fraction_active.set_xlabel("Session")
        ax_fraction_active.set_ylabel("Fraction active")
        format_spines(
            ax_fraction_active,
            x_pos=-0.05,
            y_pos=-0.05,
            xbounds=(0, reliability.shape[0] - 1),
            ybounds=ax_fraction_active.get_ylim(),
            spines_visible=["bottom", "left"],
            tick_length=4,
        )

        fig.suptitle(f"Mouse: {state['mouse']}, Env: {state['environment']}, Combo: {state['session_combo']}")

        return fig


class DistributionViewer(Viewer):
    def __init__(self, tracked_mice: list[str], max_session_difference: int = 6):
        self.tracked_mice = tracked_mice
        self.mousedb = get_database("vrMice")
        self.ko = dict(zip(self.mousedb.get_table()["mouseName"], self.mousedb.get_table()["KO"]))
        self.max_session_difference = max_session_difference

        # Define syd parameters
        self.add_selection("reliability_threshold", value=0.5, options=[0.3, 0.5, 0.7, 0.9])
        self.add_boolean("continuous", value=True)
        self.add_integer("max_session_included", value=20, min=1, max=20)
        self.add_selection("forward_backward", value="both", options=["forward", "backward", "both"])
        self.add_integer("min_rois", value=2, min=1, max=10)
        self.add_integer("num_bins", value=7, min=3, max=15)
        self.add_integer("environment", value=0, min=0, max=2)
        self.add_selection("spks_type", value="significant", options=["significant", "oasis"])
        self.add_integer("session_difference", value=1, min=1, max=max_session_difference)

        self.on_change("reliability_threshold", self.update_combo_data)
        self.update_combo_data(self.state)

        self.on_change(["max_session_included", "forward_backward", "num_bins", "environment", "min_rois", "spks_type"], self.process_combo_data)
        self.process_combo_data(self.state)

    def define_state(self, state: dict):
        state = dict(
            reliability_method="leave_one_out",
            reliability_threshold=state["reliability_threshold"],
            smooth_width=5,
            continuous=state["continuous"],
            use_session_filters=True,
            spks_type=state["spks_type"],
        )
        return state

    def update_combo_data(self, state: dict):
        self.combo_data = gather_combo_data(self.tracked_mice, self.define_state(state))
        self.process_combo_data(state)

    def process_combo_data(self, state):
        max_session_included = state["max_session_included"]
        self.bins = np.linspace(-1.0, 1.0, state["num_bins"])
        directions = []
        if state["forward_backward"] == "forward" or state["forward_backward"] == "both":
            directions.append("forward")
        if state["forward_backward"] == "backward" or state["forward_backward"] == "both":
            directions.append("backward")
        mouse_ids = [[] for _ in range(self.max_session_difference)]
        mouse_kos = [[] for _ in range(self.max_session_difference)]
        ctl_distributions = [[] for _ in range(self.max_session_difference)]
        red_distributions = [[] for _ in range(self.max_session_difference)]
        ctl_averages = [[] for _ in range(self.max_session_difference)]
        red_averages = [[] for _ in range(self.max_session_difference)]
        ctl_num_cells = [[] for _ in range(self.max_session_difference)]
        red_num_cells = [[] for _ in range(self.max_session_difference)]

        mouse_lookup = dict(enumerate(self.tracked_mice))
        for imouse, mouse in enumerate(self.combo_data):
            env_in_order = self.combo_data[mouse]["env_in_order"]
            if state["environment"] >= len(env_in_order):
                continue
            environment = env_in_order[state["environment"]]
            c_combo_data = self.combo_data[mouse][environment]
            c_env_sessions = self.combo_data[mouse]["env_stats"][environment]
            for direction in directions:
                if c_combo_data is None:
                    continue
                for icombo, combo in enumerate(c_combo_data[f"{direction}_combos"]):
                    idx_reference = c_env_sessions.index(combo[0])
                    idx_target = c_env_sessions.index(combo[-1])
                    idx_diff = abs(idx_target - idx_reference)
                    if idx_diff > self.max_session_difference:
                        continue
                    if (idx_reference > max_session_included) or (idx_target > max_session_included):
                        continue

                    ctl_stability = c_combo_data[f"{direction}_raw"]["ctl_stability"][icombo][-1]
                    red_stability = c_combo_data[f"{direction}_raw"]["red_stability"][icombo][-1]
                    ctl_data = c_combo_data[f"{direction}_raw"]["spkmap_correlations_ctl"][icombo][-1][ctl_stability]
                    red_data = c_combo_data[f"{direction}_raw"]["spkmap_correlations_red"][icombo][-1][red_stability]
                    if len(red_data) < state["min_rois"]:
                        continue
                    ctl_counts = np.histogram(ctl_data, bins=self.bins)[0]
                    red_counts = np.histogram(red_data, bins=self.bins)[0]

                    mouse_ids[idx_diff - 1].append(imouse)
                    mouse_kos[idx_diff - 1].append(self.ko[mouse])
                    ctl_averages[idx_diff - 1].append(np.nanmean(ctl_data))
                    red_averages[idx_diff - 1].append(np.nanmean(red_data))
                    ctl_distributions[idx_diff - 1].append(ctl_counts)
                    red_distributions[idx_diff - 1].append(red_counts)
                    ctl_num_cells[idx_diff - 1].append(len(ctl_data))
                    red_num_cells[idx_diff - 1].append(len(red_data))

        valid_difference = [i for i, mid in enumerate(mouse_ids) if len(mid) > 0]
        mouse_ids = [np.stack(mid) for i, mid in enumerate(mouse_ids) if i in valid_difference]
        mouse_kos = [np.stack(kd) for i, kd in enumerate(mouse_kos) if i in valid_difference]
        ctl_distributions = [np.stack(cd) for i, cd in enumerate(ctl_distributions) if i in valid_difference]
        red_distributions = [np.stack(rd) for i, rd in enumerate(red_distributions) if i in valid_difference]
        ctl_averages = [np.stack(ca) for i, ca in enumerate(ctl_averages) if i in valid_difference]
        red_averages = [np.stack(ra) for i, ra in enumerate(red_averages) if i in valid_difference]
        ctl_num_cells = [np.stack(nc) for i, nc in enumerate(ctl_num_cells) if i in valid_difference]
        red_num_cells = [np.stack(nc) for i, nc in enumerate(red_num_cells) if i in valid_difference]

        self._processed_data = dict(
            mouse_lookup=mouse_lookup,
            mouse_ids=mouse_ids,
            mouse_kos=mouse_kos,
            ctl_distributions=ctl_distributions,
            red_distributions=red_distributions,
            ctl_averages=ctl_averages,
            red_averages=red_averages,
            ctl_num_cells=ctl_num_cells,
            red_num_cells=red_num_cells,
            valid_difference=valid_difference,
        )

    def plot(self, state):
        idx_session_difference = state["session_difference"] - 1
        if idx_session_difference + 1 not in self._processed_data["valid_difference"]:
            self.update_integer("session_difference", value=1)
            return self.plot(self.state)

        mouse_lookup = self._processed_data["mouse_lookup"]
        mouse_ids = self._processed_data["mouse_ids"][idx_session_difference]
        mouse_kos = self._processed_data["mouse_kos"][idx_session_difference]
        ctl_data = self._processed_data["ctl_distributions"][idx_session_difference]
        red_data = self._processed_data["red_distributions"][idx_session_difference]
        ctl_avg = self._processed_data["ctl_averages"][idx_session_difference]
        red_avg = self._processed_data["red_averages"][idx_session_difference]
        ctl_num_cells = self._processed_data["ctl_num_cells"][idx_session_difference]
        red_num_cells = self._processed_data["red_num_cells"][idx_session_difference]

        ctl_data = ctl_data / np.sum(ctl_data, axis=1, keepdims=True)
        red_data = red_data / np.sum(red_data, axis=1, keepdims=True)

        mouse_ids_ko = mouse_ids[mouse_kos]
        mouse_ids_wt = mouse_ids[~mouse_kos]

        ctl_data_ko = ctl_data[mouse_kos]
        red_data_ko = red_data[mouse_kos]
        ctl_data_wt = ctl_data[~mouse_kos]
        red_data_wt = red_data[~mouse_kos]
        ctl_avg_ko = ctl_avg[mouse_kos]
        red_avg_ko = red_avg[mouse_kos]
        ctl_avg_wt = ctl_avg[~mouse_kos]
        red_avg_wt = red_avg[~mouse_kos]
        ctl_num_cells_ko = ctl_num_cells[mouse_kos]
        red_num_cells_ko = red_num_cells[mouse_kos]
        ctl_num_cells_wt = ctl_num_cells[~mouse_kos]
        red_num_cells_wt = red_num_cells[~mouse_kos]

        diff_data_ko = red_data_ko - ctl_data_ko
        diff_data_wt = red_data_wt - ctl_data_wt

        vlimits = max(np.max(np.abs(diff_data_ko)), np.max(np.abs(diff_data_wt)))

        max_mouse_id = np.max(mouse_ids)
        cmap = plt.get_cmap("jet")
        norm = plt.Normalize(vmin=0, vmax=max_mouse_id)
        colors_ko = cmap(norm(mouse_ids_ko))
        colors_wt = cmap(norm(mouse_ids_wt))
        extent_ko = [self.bins[0], self.bins[-1], 0, ctl_data_ko.shape[0]]
        extent_wt = [self.bins[0], self.bins[-1], 0, ctl_data_wt.shape[0]]

        idx_ko = np.lexsort((ctl_avg_ko, mouse_ids_ko))
        idx_wt = np.lexsort((ctl_avg_wt, mouse_ids_wt))

        fig, ax = plt.subplots(2, 7, width_ratios=[0.1, 1, 1, 1, 0.3, 0.3, 0.3], figsize=(7, 5), layout="constrained")
        ax[0, 0].imshow(colors_ko[idx_ko, None], aspect="auto", extent=extent_ko, interpolation="none", origin="lower")
        ax[0, 1].imshow(ctl_data_ko[idx_ko], aspect="auto", cmap="gray_r", interpolation="none", extent=extent_ko, origin="lower")
        ax[0, 2].imshow(red_data_ko[idx_ko], aspect="auto", cmap="gray_r", interpolation="none", extent=extent_ko, origin="lower")
        ax[0, 3].imshow(
            diff_data_ko[idx_ko], aspect="auto", cmap="bwr", interpolation="none", extent=extent_ko, origin="lower", vmin=-vlimits, vmax=vlimits
        )
        ax[0, 4].plot(ctl_avg_ko[idx_ko], 0.5 + np.arange(len(ctl_avg_ko)), color="k", marker=".")
        ax[0, 4].plot(red_avg_ko[idx_ko], 0.5 + np.arange(len(red_avg_ko)), color="r", marker=".")

        ax[1, 0].imshow(colors_wt[idx_wt, None], aspect="auto", extent=extent_wt, interpolation="none", origin="lower")
        ax[1, 1].imshow(ctl_data_wt[idx_wt], aspect="auto", cmap="gray_r", interpolation="none", extent=extent_wt, origin="lower")
        ax[1, 2].imshow(red_data_wt[idx_wt], aspect="auto", cmap="gray_r", interpolation="none", extent=extent_wt, origin="lower")
        ax[1, 3].imshow(
            diff_data_wt[idx_wt], aspect="auto", cmap="bwr", interpolation="none", extent=extent_wt, origin="lower", vmin=-vlimits, vmax=vlimits
        )
        ax[1, 4].plot(ctl_avg_wt[idx_wt], 0.5 + np.arange(len(ctl_avg_wt)), color="k", marker=".")
        ax[1, 4].plot(red_avg_wt[idx_wt], 0.5 + np.arange(len(red_avg_wt)), color="r", marker=".")

        ax[0, 5].plot(ctl_num_cells_ko[idx_ko], 0.5 + np.arange(len(ctl_num_cells_ko)), color="k", marker=".")
        ax[0, 6].plot(red_num_cells_ko[idx_ko], 0.5 + np.arange(len(red_num_cells_ko)), color="r", marker=".")
        ax[1, 5].plot(ctl_num_cells_wt[idx_wt], 0.5 + np.arange(len(ctl_num_cells_wt)), color="k", marker=".")
        ax[1, 6].plot(red_num_cells_wt[idx_wt], 0.5 + np.arange(len(red_num_cells_wt)), color="r", marker=".")

        for a in ax.flatten():
            a.set_yticks([])
        ax[0, 0].set_xticks([])
        ax[0, 0].set_ylabel("Knockout Mice")
        ax[1, 0].set_xticks([])
        ax[1, 0].set_ylabel("Wildtype Mice")
        for a in ax[0, 1:4]:
            a.set_xticks([self.bins[0], self.bins[-1]])
        for a in ax[1, 1:4]:
            a.set_xticks([self.bins[0], self.bins[-1]])
        for a in ax[0, 5:]:
            xlim = a.get_xlim()
            a.set_xlim(0, xlim[1])
            a.set_xticks(a.get_xlim())
            a.set_ylim(0, ctl_data_ko.shape[0])
        for a in ax[1, 5:]:
            xlim = a.get_xlim()
            a.set_xlim(0, xlim[1])
            a.set_xticks(a.get_xlim())
            a.set_ylim(0, ctl_data_wt.shape[0])

        return fig


class DistributionFigureMaker(DistributionViewer):
    def __init__(self, tracked_mice: list[str], max_session_difference: int = 6):
        super().__init__(tracked_mice, max_session_difference)
        self.add_selection("highlight_mouse", value="none", options=["none"] + list(self.tracked_mice))
        self.add_button("save_figure", label="Save Figure", callback=self.save_figure)

    def save_figure(self, state):
        name_components = [
            f"session_difference_{state['session_difference']}",
            f"forward_backward_{state['forward_backward']}",
            f"num_bins_{state['num_bins']}",
            f"environment_{state['environment']}",
            f"reliability_threshold_{state['reliability_threshold']}",
            f"continuous_{state['continuous']}",
            f"highlight_mouse_{state['highlight_mouse']}",
        ]
        fig_name = "_".join(name_components)
        fig = self.plot(state)
        fig_dir = figure_dir("changing_placefields_detail")
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_dir / fig_name)
        plt.close(fig)

    def plot(self, state):
        idx_session_difference = state["session_difference"] - 1
        if idx_session_difference + 1 not in self._processed_data["valid_difference"]:
            self.update_integer("session_difference", value=1)
            return self.plot(self.state)

        mouse_lookup = self._processed_data["mouse_lookup"]
        mouse_ids = self._processed_data["mouse_ids"][idx_session_difference]
        mouse_kos = self._processed_data["mouse_kos"][idx_session_difference]
        ctl_data = self._processed_data["ctl_distributions"][idx_session_difference]
        red_data = self._processed_data["red_distributions"][idx_session_difference]
        ctl_avg = self._processed_data["ctl_averages"][idx_session_difference]
        red_avg = self._processed_data["red_averages"][idx_session_difference]
        ctl_num_cells = self._processed_data["ctl_num_cells"][idx_session_difference]
        red_num_cells = self._processed_data["red_num_cells"][idx_session_difference]

        ctl_data = ctl_data / np.sum(ctl_data, axis=1, keepdims=True)
        red_data = red_data / np.sum(red_data, axis=1, keepdims=True)

        mouse_ids_ko = mouse_ids[mouse_kos]
        mouse_ids_wt = mouse_ids[~mouse_kos]

        ctl_data_ko = ctl_data[mouse_kos]
        red_data_ko = red_data[mouse_kos]
        ctl_data_wt = ctl_data[~mouse_kos]
        red_data_wt = red_data[~mouse_kos]
        ctl_avg_ko = ctl_avg[mouse_kos]
        red_avg_ko = red_avg[mouse_kos]
        ctl_avg_wt = ctl_avg[~mouse_kos]
        red_avg_wt = red_avg[~mouse_kos]
        ctl_num_cells_ko = ctl_num_cells[mouse_kos]
        red_num_cells_ko = red_num_cells[mouse_kos]
        ctl_num_cells_wt = ctl_num_cells[~mouse_kos]
        red_num_cells_wt = red_num_cells[~mouse_kos]
        diff_data_ko = red_data_ko - ctl_data_ko
        diff_data_wt = red_data_wt - ctl_data_wt

        # This is to keep the plots ordered by KO then WT
        imice_ko = np.unique(mouse_ids_ko)
        imice_wt = np.unique(mouse_ids_wt)

        fig = plt.figure(figsize=(4.5, 5.5), layout="constrained")
        gs = fig.add_gridspec(2, 1)
        ax_combo_means = fig.add_subplot(gs[0])
        ax_combo_diffs = fig.add_subplot(gs[1])
        for ii, imouse in enumerate(np.concatenate((imice_ko, imice_wt))):
            c_ko = self.ko[self.tracked_mice[imouse]]
            idx_to_mouse = np.where(mouse_ids == imouse)[0]
            color = "purple" if c_ko else "gray"
            ctl_data = ctl_avg[idx_to_mouse]
            red_data = red_avg[idx_to_mouse]
            offset = 0.15
            offlims = np.array([-offset, offset])
            ax_combo_means.scatter(-offset + ii * np.ones(len(ctl_data)), ctl_data, color="k", s=9, alpha=0.5)
            ax_combo_means.scatter(offset + ii * np.ones(len(red_data)), red_data, color="r", s=9, alpha=0.5)
            ax_combo_means.plot(ii - offset + offlims, np.nanmean(ctl_data) * np.ones(2), color="k", linewidth=1)
            ax_combo_means.plot(ii + offset + offlims, np.nanmean(red_data) * np.ones(2), color="r", linewidth=1)
            ax_combo_diffs.scatter(ii * np.ones(len(red_data - ctl_data)), red_data - ctl_data, color=color, s=9, alpha=0.5)
            ax_combo_diffs.plot(ii + offlims, np.nanmean(red_data - ctl_data) * np.ones(2), color=color, linewidth=1)

        ax_combo_diffs.axhline(0, color="k", linewidth=0.5, linestyle="--", zorder=-10)

        ax_combo_means.set_xlim(-0.5, len(imice_ko) + len(imice_wt) - 0.5)
        ax_combo_diffs.set_xlim(-0.5, len(imice_ko) + len(imice_wt) - 0.5)
        ylim_means = ax_combo_means.get_ylim()
        ax_combo_means.set_ylim(0, ylim_means[1])
        ylim_means = ax_combo_means.get_ylim()
        ylim_diffs = ax_combo_diffs.get_ylim()
        for ii, imouse in enumerate(np.concatenate((imice_ko, imice_wt))):
            c_ko = self.ko[self.tracked_mice[imouse]]
            color = "purple" if c_ko else "gray"
            rect = mpl.patches.Rectangle((ii - 0.49, ylim_means[0]), 0.98, ylim_means[1] - ylim_means[0], facecolor=color, alpha=0.2, zorder=-10)
            ax_combo_means.add_patch(rect)

            if state["highlight_mouse"] != "none" and mouse_lookup[imouse] == state["highlight_mouse"]:
                highlight_color = "b"
                rect = mpl.patches.Rectangle(
                    (ii - 0.49, ylim_means[0]), 0.98, 0.1 * (ylim_means[1] - ylim_means[0]), facecolor=highlight_color, alpha=0.2, zorder=-9
                )
                ax_combo_means.add_patch(rect)

        format_spines(
            ax_combo_means,
            x_pos=-0.01,
            y_pos=-0.05,
            xticks=[],
            xbounds=(0, len(imice_ko) + len(imice_wt) - 1),
            ybounds=(0, ylim_means[1]),
            spines_visible=["left"],
        )

        format_spines(
            ax_combo_diffs,
            x_pos=-0.01,
            y_pos=-0.05,
            xbounds=(0, len(imice_ko) + len(imice_wt) - 1),
            ybounds=ylim_diffs,
        )

        num_mice_ko = len(imice_ko)
        num_mice_wt = len(imice_wt)
        ax_combo_means.text(0, ylim_means[1] * 0.1, "knockout mice", ha="left", va="top")
        ax_combo_means.text(num_mice_ko, ylim_means[1] * 0.1, "control mice", ha="left", va="top")

        ax_combo_diffs.set_xlabel("mouse")
        ax_combo_means.set_ylabel("Correlation Coefficient")
        ax_combo_diffs.set_ylabel(r"$\Delta$ CC (red - ctl)")

        return fig


class NumROIsInCombosViewer(Viewer):
    def __init__(self, tracked_mice: list[str]):
        self.tracked_mice = tracked_mice
        self.mousedb = get_database("vrMice")
        self.ko = dict(zip(self.mousedb.get_table()["mouseName"], self.mousedb.get_table()["KO"]))
        self.max_session_difference = 6

        # Define syd parameters
        self.add_selection("reliability_threshold", value=0.5, options=[0.3, 0.5, 0.7, 0.9])
        self.add_boolean("continuous", value=True)
        self.add_selection("forward_backward", value="both", options=["forward", "backward", "both"])
        self.add_integer("environment", value=0, min=0, max=2)
        self.add_selection("ctl_or_red", value="red", options=["ctl", "red"])
        self.add_boolean("include_unique_clusters", value=False)
        self.add_selection("spks_type", value="significant", options=["significant", "oasis"])
        self.add_integer("session_difference", value=1, min=1, max=self.max_session_difference)
        self.add_selection("highlight_mouse", value="none", options=["none"] + list(self.tracked_mice))

        self.on_change("reliability_threshold", self.update_combo_data)
        self.on_change(["forward_backward", "environment"], self.process_combo_data)
        self.update_combo_data(self.state)

    def define_state(self, state: dict):
        state = dict(
            reliability_method="leave_one_out",
            reliability_threshold=state["reliability_threshold"],
            smooth_width=5,
            continuous=state["continuous"],
            use_session_filters=True,
            spks_type=state["spks_type"],
        )
        return state

    def update_combo_data(self, state: dict):
        self.combo_data = gather_combo_data(self.tracked_mice, self.define_state(state))
        self.process_combo_data(state)

    def process_combo_data(self, state):
        directions = []
        if state["forward_backward"] == "forward" or state["forward_backward"] == "both":
            directions.append("forward")
        if state["forward_backward"] == "backward" or state["forward_backward"] == "both":
            directions.append("backward")

        num_ctl = {mouse: [] for mouse in self.combo_data}
        num_red = {mouse: [] for mouse in self.combo_data}
        num_tracked_ctl = {mouse: [] for mouse in self.combo_data}
        num_tracked_red = {mouse: [] for mouse in self.combo_data}
        num_uniq_tracked_ctl = {mouse: [] for mouse in self.combo_data}
        num_uniq_tracked_red = {mouse: [] for mouse in self.combo_data}
        num_good_uniq_tracked_ctl = {mouse: [] for mouse in self.combo_data}
        num_good_uniq_tracked_red = {mouse: [] for mouse in self.combo_data}
        num_rel_good_uniq_tracked_ctl = {mouse: [] for mouse in self.combo_data}
        num_rel_good_uniq_tracked_red = {mouse: [] for mouse in self.combo_data}
        unique_clusters_ctl = {mouse: [] for mouse in self.combo_data}
        unique_clusters_red = {mouse: [] for mouse in self.combo_data}
        num_unique_clusters_ctl = {mouse: [] for mouse in self.combo_data}
        num_unique_clusters_red = {mouse: [] for mouse in self.combo_data}
        session_difference = {mouse: [] for mouse in self.combo_data}
        for mouse in tqdm(self.combo_data, desc="Processing mice"):
            tracker = Tracker(mouse)
            env_in_order = self.combo_data[mouse]["env_in_order"]
            if state["environment"] >= len(env_in_order):
                continue
            environment = env_in_order[state["environment"]]
            c_combo_data = self.combo_data[mouse][environment]
            c_env_sessions = self.combo_data[mouse]["env_stats"][environment]
            if c_combo_data is None:
                continue
            for direction in directions:
                for icombo, combo in enumerate(c_combo_data[f"{direction}_combos"]):
                    idx_reference = c_env_sessions.index(combo[0])
                    idx_target = c_env_sessions.index(combo[-1])
                    idx_diff = abs(idx_target - idx_reference)
                    if idx_diff > self.max_session_difference:
                        continue

                    session_difference[mouse].append(idx_diff)

                    c_red_idx = tracker.sessions[combo[-1]].get_red_idx()
                    num_ctl[mouse].append(np.sum(~c_red_idx))
                    num_red[mouse].append(np.sum(c_red_idx))

                    c_tracking_idx = tracker.get_tracked_idx(idx_ses=combo)[0]
                    c_red_tracked_idx = c_red_idx[c_tracking_idx[-1]]
                    num_tracked_ctl[mouse].append(np.sum(~c_red_tracked_idx))
                    num_tracked_red[mouse].append(np.sum(c_red_tracked_idx))

                    c_redundant_idx = tracker.sessions[combo[-1]].valid_redundancy_idx()
                    c_redundant_tracked_idx = c_redundant_idx[c_tracking_idx[-1]]
                    num_uniq_tracked_ctl[mouse].append(np.sum(~c_red_tracked_idx & c_redundant_tracked_idx))
                    num_uniq_tracked_red[mouse].append(np.sum(c_red_tracked_idx & c_redundant_tracked_idx))

                    c_good_mask_idx = np.all(np.stack(tracker.sessions[combo[-1]].valid_mask_idx()), axis=0)
                    c_good_mask_tracked_idx = c_good_mask_idx[c_tracking_idx[-1]]
                    num_good_uniq_tracked_ctl[mouse].append(np.sum(~c_red_tracked_idx & c_redundant_tracked_idx & c_good_mask_tracked_idx))
                    num_good_uniq_tracked_red[mouse].append(np.sum(c_red_tracked_idx & c_redundant_tracked_idx & c_good_mask_tracked_idx))

                    ctl_stability = c_combo_data[f"{direction}_raw"]["ctl_stability"][icombo][-1]
                    red_stability = c_combo_data[f"{direction}_raw"]["red_stability"][icombo][-1]
                    c_num_rel_ctl = np.sum(ctl_stability)
                    c_num_rel_red = np.sum(red_stability)
                    num_rel_good_uniq_tracked_ctl[mouse].append(c_num_rel_ctl)
                    num_rel_good_uniq_tracked_red[mouse].append(c_num_rel_red)

                    c_ctl_cluster_ids = c_combo_data[f"{direction}_raw"]["ctl_cluster_ids"][icombo][ctl_stability]
                    c_red_cluster_ids = c_combo_data[f"{direction}_raw"]["red_cluster_ids"][icombo][red_stability]
                    unique_clusters_ctl[mouse].append(c_ctl_cluster_ids)
                    unique_clusters_red[mouse].append(c_red_cluster_ids)

        all_unique_clusters_ctl = {mouse: np.concatenate(unique_clusters_ctl[mouse]) for mouse in self.combo_data}
        all_unique_clusters_red = {mouse: np.concatenate(unique_clusters_red[mouse]) for mouse in self.combo_data}
        num_unique_clusters_ctl = {mouse: len(np.unique(all_unique_clusters_ctl[mouse])) for mouse in self.combo_data}
        num_unique_clusters_red = {mouse: len(np.unique(all_unique_clusters_red[mouse])) for mouse in self.combo_data}

        self._processed_data = dict(
            num_ctl=num_ctl,
            num_red=num_red,
            num_tracked_ctl=num_tracked_ctl,
            num_tracked_red=num_tracked_red,
            num_uniq_tracked_ctl=num_uniq_tracked_ctl,
            num_uniq_tracked_red=num_uniq_tracked_red,
            num_good_uniq_tracked_ctl=num_good_uniq_tracked_ctl,
            num_good_uniq_tracked_red=num_good_uniq_tracked_red,
            num_rel_good_uniq_tracked_ctl=num_rel_good_uniq_tracked_ctl,
            num_rel_good_uniq_tracked_red=num_rel_good_uniq_tracked_red,
            num_unique_clusters_ctl=num_unique_clusters_ctl,
            num_unique_clusters_red=num_unique_clusters_red,
            session_difference=session_difference,
        )

    def plot(self, state):
        data = self._processed_data

        order_of_plot = ["", "tracked", "uniq_tracked", "good_uniq_tracked", "rel_good_uniq_tracked"]
        xlabels = ["all", "tracked", "unique", "good", "reliable"]

        if state["include_unique_clusters"]:
            order_of_plot.append("unique_clusters")
            xlabels.append("unique(clusters)")

        session_differences = data["session_difference"]
        mouse_data = {}
        for mouse in self.combo_data:
            mouse_data[mouse] = []
            for order in order_of_plot:
                name = f"num_{order}{'_' if order else ''}{state['ctl_or_red']}"
                c_data = data[name][mouse]
                c_diffs = session_differences[mouse]
                if not isinstance(c_data, list):
                    c_data = [c_data] * len(mouse_data[mouse][-1])
                else:
                    c_data = [cd for cd, sd in zip(c_data, c_diffs) if sd == state["session_difference"]]
                mouse_data[mouse].append(c_data)

        fig, ax = plt.subplots(2, 2, figsize=(7, 5.5), width_ratios=[1, 1], layout="constrained")
        idx_ko = 0
        idx_wt = 0
        for imouse, mouse in enumerate(mouse_data):
            ko = self.ko[mouse]
            axx = ax[0, 0] if ko else ax[1, 0]
            color = "purple" if ko else "gray"
            if state["highlight_mouse"] != "none" and mouse == state["highlight_mouse"]:
                color = "b"
            data = np.mean(np.stack(mouse_data[mouse]), axis=1)
            axx.plot(range(len(data)), data, color=color, marker=".", alpha=0.5)

            idx = idx_ko if ko else idx_wt
            axx = ax[0, 1] if ko else ax[1, 1]
            data = mouse_data[mouse][-1]
            xvals = beeswarm(data, nbins=max(1, len(data) // 2))
            axx.plot(idx + xvals / 3, data, color=color, marker=".", alpha=0.25, linestyle="none")
            if ko:
                idx_ko += 1
            else:
                idx_wt += 1

        for color, name, axx in zip(["purple", "gray"], ["knockout mice", "control mice"], [ax[0, 0], ax[1, 0]]):
            c_ylim = axx.get_ylim()
            axx.set_ylim(-2, c_ylim[1])
            c_xlim = axx.get_xlim()
            axx.text(c_xlim[1] * 0.9, c_ylim[1] * 0.9, name, ha="right", va="top", color=color)
        for axx in [ax[0, 1], ax[1, 1]]:
            c_ylim = axx.get_ylim()
            axx.set_ylim(-0.5, c_ylim[1])

        ax[0, 0].set_ylabel(f"Number of ROIs ({state['ctl_or_red']})")
        ax[1, 0].set_ylabel(f"Number of ROIs ({state['ctl_or_red']})")
        ax[0, 1].set_ylabel(f"Number of ROIs ({state['ctl_or_red']})")
        ax[1, 1].set_ylabel(f"Number of ROIs ({state['ctl_or_red']})")
        format_spines(
            ax[0, 0],
            x_pos=-0.01,
            y_pos=-0.05,
            xticks=[],
            xbounds=(0, len(order_of_plot) - 1),
            ybounds=(0, ax[0, 0].get_ylim()[1]),
            spines_visible=["left"],
        )
        format_spines(
            ax[0, 1],
            x_pos=-0.01,
            y_pos=-0.05,
            xticks=range(idx_ko),
            ybounds=(0, ax[0, 1].get_ylim()[1]),
            spines_visible=["left", "bottom"],
        )
        format_spines(
            ax[1, 0],
            x_pos=-0.01,
            y_pos=-0.05,
            xbounds=(0, len(order_of_plot) - 1),
            ybounds=(0, ax[1, 0].get_ylim()[1]),
            spines_visible=["left", "bottom"],
        )
        format_spines(
            ax[1, 1],
            x_pos=-0.01,
            y_pos=-0.05,
            xticks=range(idx_wt),
            ybounds=(0, ax[1, 1].get_ylim()[1]),
            spines_visible=["left", "bottom"],
        )
        ax[1, 0].set_xticks(range(len(order_of_plot)), labels=xlabels, rotation=0, ha="center")
        ax[1, 1].set_xlabel("Mouse")
        return fig
