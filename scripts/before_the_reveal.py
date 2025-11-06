from typing import Optional, Literal
from pathlib import Path
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import joblib

from vrAnalysis.files import analysis_path
from vrAnalysis.database import get_database
from vrAnalysis.helpers import Timer, save_figure, edge2center, fractional_histogram
from vrAnalysis.processors.spkmaps import SpkmapProcessor
from vrAnalysis.tracking import Tracker
from vrAnalysis.metrics import FractionActive
from vrAnalysis.multisession import MultiSessionSpkmaps
from vrAnalysis.syd.reliability_continuity import ReliabilityStabilitySummary

sessiondb = get_database("vrSessions")
mousedb = get_database("vrMice")
tracked_mice = mousedb.get_table(tracked=True)["mouseName"].unique()


def figure_path(figure_type: str, name: str) -> Path:
    """Get the path to a figure"""
    directory = analysis_path() / "before_the_reveal" / figure_type
    if not directory.exists():
        directory.mkdir(parents=True)
    return directory / name


def get_environments(track: Tracker) -> np.ndarray:
    """Get all environments represented in tracked sessions"""
    environments = np.unique(np.concatenate([session.environments for session in track.sessions]))
    return environments


def get_reliability(
    track: Tracker,
    exclude_environments: Optional[list[int] | int] = None,
    reliability_method: str = "leave_one_out",
    clear_one: bool = False,
) -> dict:
    """Get reliability data for all tracked sessions"""
    environments = list(get_environments(track))
    reliability_ctl = {env: [] for env in environments}
    reliability_red = {env: [] for env in environments}
    reliability_ctl_all = {env: [] for env in environments}
    reliability_red_all = {env: [] for env in environments}
    fraction_active_ctl = {env: [] for env in environments}
    fraction_active_red = {env: [] for env in environments}
    fraction_active_ctl_all = {env: [] for env in environments}
    fraction_active_red_all = {env: [] for env in environments}
    sessions = {env: [] for env in environments}

    for isession, session in enumerate(tqdm(track.sessions)):
        envs = session.environments
        idx_rois = session.idx_rois
        idx_red_all = session.get_red_idx()
        idx_red = idx_red_all[idx_rois]

        smp = SpkmapProcessor(session)
        env_maps = smp.get_env_maps(use_session_filters=False)
        reliability_all = smp.get_reliability(use_session_filters=False, params=dict(smooth_width=5.0, reliability_method=reliability_method))
        reliability_selected = reliability_all.filter_rois(idx_rois)

        for ienv, env in enumerate(envs):
            c_fraction_active = FractionActive.compute(
                env_maps.spkmap[ienv],
                activity_axis=2,
                fraction_axis=1,
                activity_method="rms",
                fraction_method="participation",
            )
            c_fraction_active_selected = c_fraction_active[idx_rois]
            reliability_red_all[env].append(reliability_all.values[ienv, idx_red_all])
            reliability_ctl_all[env].append(reliability_all.values[ienv, ~idx_red_all])
            reliability_red[env].append(reliability_selected.values[ienv, idx_red])
            reliability_ctl[env].append(reliability_selected.values[ienv, ~idx_red])
            fraction_active_red_all[env].append(c_fraction_active[idx_red_all])
            fraction_active_ctl_all[env].append(c_fraction_active[~idx_red_all])
            fraction_active_red[env].append(c_fraction_active_selected[idx_red])
            fraction_active_ctl[env].append(c_fraction_active_selected[~idx_red])
            sessions[env].append(isession)

        if clear_one:
            session.clear_cache()

    results = dict(
        environments=environments,
        reliability_ctl=reliability_ctl,
        reliability_red=reliability_red,
        reliability_ctl_all=reliability_ctl_all,
        reliability_red_all=reliability_red_all,
        fraction_active_ctl=fraction_active_ctl,
        fraction_active_red=fraction_active_red,
        fraction_active_ctl_all=fraction_active_ctl_all,
        fraction_active_red_all=fraction_active_red_all,
        sessions=sessions,
    )

    if exclude_environments:
        if not isinstance(exclude_environments, list):
            exclude_environments = [exclude_environments]
        for env in exclude_environments:
            for key in results:
                if isinstance(results[key], dict):
                    results[key].pop(env, None)
                else:
                    results[key] = [r for r in results[key] if r != env]

    return results


def plot_reliability_distribution(
    track: Tracker,
    reliability: dict,
    selected: bool = True,
    reduction: str = "mean",
    show: bool = False,
    save: bool = False,
) -> None:
    """Plot reliability data for a mouse"""
    num_sessions = len(track.sessions)
    num_environments = len(reliability["environments"])

    def color_violins(parts, facecolor=None, linecolor=None):
        """Helper to color parts manually."""
        if facecolor is not None:
            for pc in parts["bodies"]:
                pc.set_facecolor(facecolor)
        if linecolor is not None:
            for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
                if partname in parts:
                    lc = parts[partname]
                    lc.set_edgecolor(linecolor)

    ctl_key = "reliability_ctl" if selected else "reliability_ctl_all"
    red_key = "reliability_red" if selected else "reliability_red_all"

    reduce_func = np.nanmedian if reduction == "median" else np.nanmean

    plt.close("all")
    figheight = 5
    figwidth = 4
    fig, ax = plt.subplots(1, num_environments, figsize=(figwidth * num_environments, figheight), layout="constrained")
    for ienv, env in enumerate(reliability["environments"]):
        env_sessions = reliability["sessions"][env]
        for ii, idx_ses in enumerate(env_sessions):
            if len(reliability[ctl_key][env][ii]) > 0:
                parts = ax[ienv].violinplot(reliability[ctl_key][env][ii], positions=[idx_ses], showextrema=False, side="low")
                color_violins(parts, facecolor=("k", 0.1))

            if len(reliability[red_key][env][ii]) > 0:
                parts = ax[ienv].violinplot(reliability[red_key][env][ii], positions=[idx_ses], showextrema=False, side="high")
                color_violins(parts, facecolor=("r", 0.1))

        reduced_ctl = [reduce_func(rel_values) for rel_values in reliability[ctl_key][env]]
        reduced_red = [reduce_func(rel_values) for rel_values in reliability[red_key][env]]

        ax[ienv].plot(env_sessions, reduced_ctl, color="k", label=f"Control {reduction}")
        ax[ienv].plot(env_sessions, reduced_red, color="r", label=f"Red {reduction}")
        ax[ienv].set_xticks(np.arange(num_sessions))
        ax[ienv].set_xlabel("Session #")
        ax[ienv].set_ylabel("Reliability")
        ax[ienv].legend(loc="best")
        ax[ienv].set_title(f"Environment: {env}")
        ax[ienv].set_xlim(-0.5, num_sessions - 0.5)

    title = f"{track.mouse_name}-({reduction})-{'Good' if selected else 'All'}_ROIs"
    fig.suptitle(title)

    if show:
        plt.show(block=True)

    if save:
        fig_path = figure_path("reliability-distribution", title)
        save_figure(fig, fig_path)


def plot_reliability_extreme(
    track: Tracker,
    reliability: dict,
    threshold: float,
    selected: bool = True,
    show: bool = False,
    save: bool = False,
) -> None:
    """Plot reliability data for a mouse"""
    num_sessions = len(track.sessions)
    num_environments = len(reliability["environments"])

    ctl_key = "reliability_ctl" if selected else "reliability_ctl_all"
    red_key = "reliability_red" if selected else "reliability_red_all"

    plt.close("all")
    figheight = 5
    figwidth = 4
    fig, ax = plt.subplots(
        3, num_environments, figsize=(figwidth * num_environments, 3 * figheight), layout="constrained", height_ratios=[1, 0.5, 0.5]
    )
    for ienv, env in enumerate(reliability["environments"]):
        env_sessions = reliability["sessions"][env]

        num_above_ctl = np.array([np.sum(rel_values > threshold) for rel_values in reliability[ctl_key][env]])
        num_above_red = np.array([np.sum(rel_values > threshold) for rel_values in reliability[red_key][env]])
        num_total_ctl = np.array([len(rel_values) for rel_values in reliability[ctl_key][env]])
        num_total_red = np.array([len(rel_values) for rel_values in reliability[red_key][env]])

        ax[0, ienv].plot(env_sessions, num_above_ctl / num_total_ctl, color="k", label=f"Control", marker="o")
        ax[0, ienv].plot(env_sessions, num_above_red / num_total_red, color="r", label=f"Red", marker="o")
        ax[0, ienv].set_xticks(np.arange(num_sessions))
        ax[0, ienv].set_xlabel("Session #")
        ax[0, ienv].set_ylabel("Fraction Extremely Reliable")
        ax[0, ienv].legend(loc="best")
        ax[0, ienv].set_title(f"Environment:{env}")
        ax[0, ienv].set_xlim(-0.5, num_sessions - 0.5)

        ax[1, ienv].plot(env_sessions, num_above_ctl, color="k", label=f"Control", marker="o")
        ax[1, ienv].set_xticks(np.arange(num_sessions))
        ax[1, ienv].set_xlabel("Session #")
        ax[1, ienv].set_ylabel("Number Extremely Reliable")
        ax[1, ienv].set_xlim(-0.5, num_sessions - 0.5)

        ax[2, ienv].plot(env_sessions, num_above_red, color="r", label=f"Red", marker="o")
        ax[2, ienv].set_xticks(np.arange(num_sessions))
        ax[2, ienv].set_xlabel("Session #")
        ax[2, ienv].set_ylabel("Number Extremely Reliable")
        ax[2, ienv].set_xlim(-0.5, num_sessions - 0.5)

    title = f"{track.mouse_name}-(rel_gt_{threshold})-{'Good' if selected else 'All'}_ROIs"
    fig.suptitle(title)

    if show:
        plt.show(block=True)

    if save:
        fig_path = figure_path(f"reliability-extremefraction-{threshold}", title)
        save_figure(fig, fig_path)

    plt.close(fig)


def plot_number_of_rois(
    track: Tracker,
    reliability: dict,
    selected: bool = True,
    show: bool = False,
    save: bool = False,
) -> None:
    """Plot the number of ROIs in each environment"""
    num_sessions = len(track.sessions)
    num_environments = len(reliability["environments"])

    ctl_key = "reliability_ctl" if selected else "reliability_ctl_all"
    red_key = "reliability_red" if selected else "reliability_red_all"

    plt.close("all")
    figheight = 5
    figwidth = 4
    fig, ax = plt.subplots(1, num_environments, figsize=(figwidth * num_environments, figheight), layout="constrained")
    for ienv, env in enumerate(reliability["environments"]):
        env_sessions = reliability["sessions"][env]

        num_rois_ctl = [len(rel_values) for rel_values in reliability[ctl_key][env]]
        num_rois_red = [len(rel_values) for rel_values in reliability[red_key][env]]

        ax[ienv].plot(env_sessions, num_rois_ctl, color="k", label=f"Control", marker="o")
        ax[ienv].plot(env_sessions, num_rois_red, color="r", label=f"Red", marker="o")
        ax[ienv].set_xticks(np.arange(num_sessions))
        ax[ienv].set_xlabel("Session #")
        ax[ienv].set_ylabel("Number of ROIs")
        ax[ienv].legend(loc="best")
        ax[ienv].set_title(f"Environment:{env}")
        ax[ienv].set_xlim(-0.5, num_sessions - 0.5)

    title = f"{track.mouse_name}-{'Good' if selected else 'All'}_ROIs"
    fig.suptitle(title)

    if show:
        plt.show(block=True)

    if save:
        fig_path = figure_path(f"number-of-rois", title)
        save_figure(fig, fig_path)

    plt.close(fig)


def plot_reliability_histogram(
    track: Tracker,
    reliability: dict,
    selected: bool = True,
    num_bins: int = 9,
    show: bool = False,
    save: bool = False,
) -> None:
    """Plot the number of ROIs in each environment"""
    num_sessions = len(track.sessions)
    num_environments = len(reliability["environments"])
    bins = np.linspace(-1, 1, num_bins)
    centers = edge2center(bins)

    ctl_key = "reliability_ctl" if selected else "reliability_ctl_all"
    red_key = "reliability_red" if selected else "reliability_red_all"
    ctl_reliability = reliability[ctl_key]
    red_reliability = reliability[red_key]

    # Construct the histograms
    ctl_hist = {env: np.full((num_sessions, len(centers)), np.nan) for env in reliability["environments"]}
    red_hist = {env: np.full((num_sessions, len(centers)), np.nan) for env in reliability["environments"]}

    max_fraction = 0
    max_difference = 0
    for ienv, env in enumerate(reliability["environments"]):
        for ises, (ctlval, redval) in enumerate(zip(ctl_reliability[env], red_reliability[env])):
            sesnum = reliability["sessions"][env][ises]
            ctl_hist[env][sesnum] = fractional_histogram(ctlval, bins=bins)[0]
            red_hist[env][sesnum] = fractional_histogram(redval, bins=bins)[0]
            max_fraction = max(max_fraction, max(np.max(red_hist[env][sesnum]), np.max(ctl_hist[env][sesnum])))
            max_difference = max(max_difference, np.max(np.abs(red_hist[env][sesnum] - ctl_hist[env][sesnum])))

    cmap = mpl.colormaps["gray_r"]
    cmap.set_bad(color=("orange", 0.25))
    vmin = 0
    vmax = max_fraction
    cmap_diff = mpl.colormaps["bwr"]
    cmap_diff.set_bad(color=("orange", 0.25))
    vmin_diff = -max_difference
    vmax_diff = max_difference

    def get_kwargs(difference):
        if difference:
            return dict(cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff, origin="upper", extent=[bins[0], bins[-1], num_sessions, 0], aspect="auto")
        else:
            return dict(cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", extent=[bins[0], bins[-1], num_sessions, 0], aspect="auto")

    plt.close("all")
    figheight = 3
    figwidth = 3

    # Use GridSpec for more flexibility with layout
    fig = plt.figure(figsize=(figwidth * num_environments + 0.5, 3 * figheight), layout="constrained")
    gs = fig.add_gridspec(3, num_environments + 1, width_ratios=[1] * num_environments + [0.05])

    ax = np.empty((3, num_environments), dtype=object)
    for i in range(3):
        for j in range(num_environments):
            ax[i, j] = fig.add_subplot(gs[i, j])

    # Create colorbar axes
    cax_top = fig.add_subplot(gs[0, -1])
    cax_middle = fig.add_subplot(gs[1, -1])
    cax_bottom = fig.add_subplot(gs[2, -1])

    for ienv, env in enumerate(reliability["environments"]):
        # Create main heatmaps
        im_top = ax[0, ienv].imshow(ctl_hist[env], **get_kwargs(False))
        im_middle = ax[1, ienv].imshow(red_hist[env], **get_kwargs(False))
        im_bottom = ax[2, ienv].imshow(red_hist[env] - ctl_hist[env], **get_kwargs(True))

        ax[0, ienv].set_title(f"Env {env}\nControl")
        ax[1, ienv].set_title(f"Red")
        ax[2, ienv].set_title(f"Red - Control")
        ax[0, ienv].set_ylabel("Session #")
        ax[1, ienv].set_ylabel("Session #")
        ax[2, ienv].set_ylabel("Session #")

        # Add x-axis label to bottom plots
        ax[2, ienv].set_xlabel("Reliability")

    # Add colorbars
    plt.colorbar(im_top, cax=cax_top, label="Fraction of Cells")
    plt.colorbar(im_middle, cax=cax_middle, label="Fraction of Cells")
    plt.colorbar(im_bottom, cax=cax_bottom, label="Difference in Fraction")

    title = f"{track.mouse_name}-{'Good' if selected else 'All'}_ROIs-ReliabilityHistogram"
    fig.suptitle(title)

    if show:
        plt.show(block=True)

    if save:
        fig_path = figure_path(f"reliability-histogram", title)
        save_figure(fig, fig_path)

    plt.close(fig)


def plot_quantile_histogram(
    track: Tracker,
    reliability: dict,
    selected: bool = True,
    num_bins: int = 9,
    show: bool = False,
    save: bool = False,
) -> None:
    """Plot the number of ROIs in each environment"""
    num_sessions = len(track.sessions)
    num_environments = len(reliability["environments"])
    bins = np.linspace(0, 1, num_bins)
    centers = edge2center(bins)

    ctl_key = "reliability_ctl" if selected else "reliability_ctl_all"
    red_key = "reliability_red" if selected else "reliability_red_all"
    ctl_reliability = reliability[ctl_key]
    red_reliability = reliability[red_key]

    # Construct the histograms
    red_deviation = {env: np.full((num_sessions, len(centers)), np.nan) for env in reliability["environments"]}

    max_deviation = 0
    for ienv, env in enumerate(reliability["environments"]):
        for ises, (ctlval, redval) in enumerate(zip(ctl_reliability[env], red_reliability[env])):
            sesnum = reliability["sessions"][env][ises]
            c_quantiles = np.quantile(ctlval, bins)
            red_deviation[env][sesnum] = fractional_histogram(redval, bins=c_quantiles)[0] - (1 / len(centers))
            max_deviation = max(max_deviation, np.max(np.abs(red_deviation[env][sesnum])))

    cmap_dev = mpl.colormaps["bwr"]
    cmap_dev.set_bad(color=("orange", 0.25))
    vmin_dev = -max_deviation
    vmax_dev = max_deviation

    kwargs = dict(cmap=cmap_dev, vmin=vmin_dev, vmax=vmax_dev, origin="upper", extent=[bins[0], bins[-1], num_sessions, 0], aspect="auto")

    plt.close("all")
    figheight = 3
    figwidth = 3

    # Use GridSpec for more flexibility with layout
    fig = plt.figure(figsize=(figwidth * num_environments + 0.5, figheight), layout="constrained")
    gs = fig.add_gridspec(1, num_environments + 1, width_ratios=[1] * num_environments + [0.05])

    ax = np.empty((1, num_environments), dtype=object)
    for i in range(1):
        for j in range(num_environments):
            ax[i, j] = fig.add_subplot(gs[i, j])

    # Create colorbar axes
    cax = fig.add_subplot(gs[0, -1])

    for ienv, env in enumerate(reliability["environments"]):
        # Create main heatmaps
        im = ax[0, ienv].imshow(red_deviation[env], **kwargs)

        ax[0, ienv].set_title(f"Env {env}")
        ax[0, ienv].set_ylabel("Session #")

        # Add x-axis label to bottom plots
        ax[0, ienv].set_xlabel("Reliability Quantile")

    # Add colorbars
    plt.colorbar(im, cax=cax, label="Red Deviation")

    title = f"{track.mouse_name}-{'Good' if selected else 'All'}_ROIs-QuantileDeviation"
    fig.suptitle(title)

    if show:
        plt.show(block=True)

    if save:
        fig_path = figure_path(f"reliability-quantile-deviation", title)
        save_figure(fig, fig_path)

    plt.close(fig)


def plot_tracking_summary(tracked_mice: list[str], show: bool = False, save: bool = False, save_results: bool = False):
    # Get this for it's data producing code, not to use the viewer!
    max_session_diff = 6
    summary_viewer = ReliabilityStabilitySummary(tracked_mice, use_cache=False)
    for mouse in tqdm(tracked_mice):
        print(f"Working on {mouse}")
        for reliability_threshold in [0.3, 0.5, 0.7, 0.9]:
            for reliability_method in ["leave_one_out"]:
                for smooth_width in [5]:
                    for use_session_filters in [True]:
                        for continuous in [True, False]:
                            for spks_type in ["oasis", "significant"]:
                                state = summary_viewer.define_state(
                                    mouse_name=mouse,
                                    envnum=1,
                                    reliability_threshold=reliability_threshold,
                                    reliability_method=reliability_method,
                                    smooth_width=smooth_width,
                                    use_session_filters=use_session_filters,
                                    continuous=continuous,
                                    spks_type=spks_type,
                                    max_session_diff=max_session_diff,
                                )
                                figure_path_name = f"reliability-summary-{reliability_method}-Threshold{reliability_threshold}-SmoothWidth{smooth_width}-Continuous{continuous}-GoodROIOnly{use_session_filters}"
                                results = summary_viewer.gather_data(state, try_cache=False)
                                if save_results:
                                    results_name = ReliabilityStabilitySummary.get_results_name(state)
                                    results_path = analysis_path() / "before_the_reveal_temp_data_new251024" / results_name
                                    if not results_path.parent.exists():
                                        results_path.parent.mkdir(parents=True, exist_ok=True)
                                    joblib.dump(results, results_path)

                                if not show and not save:
                                    continue

                                environments = [envnum for envnum in results if results[envnum] is not None]
                                for envnum in environments:
                                    state["environment"] = envnum
                                    fig = summary_viewer.plot(state, results=results)
                                    title = f"{mouse}-{envnum}-{reliability_threshold}-{reliability_method}-{smooth_width}-{use_session_filters}-{continuous}"
                                    fig.suptitle(title)

                                    if show:
                                        plt.show(block=True)

                                    if save:
                                        fig_path = figure_path(figure_path_name, title)
                                        save_figure(fig, fig_path)
                                        print(f"Saved {fig_path}")


def plot_tracking_full_summary(
    tracked_mice: list[str],
    reliability_threshold: float,
    continuous: bool = True,
    split_environments: bool = True,
    forward_backward: Literal["forward", "backward", "both"] = "both",
    show: bool = False,
    save: bool = False,
):
    if forward_backward not in ["forward", "backward", "both"]:
        raise ValueError(f"Invalid forward_backward: {forward_backward}")

    reliability_method = "leave_one_out"
    smooth_width = float(5.0)
    use_session_filters = True
    max_session_diff = 6
    permitted_thresholds = [0.3, 0.5, 0.7, 0.9]
    permitted_continuous = [True, False]
    if reliability_threshold not in permitted_thresholds:
        raise ValueError(f"Invalid reliability threshold: {reliability_threshold}")
    if continuous not in permitted_continuous:
        raise ValueError(f"Invalid continuous: {continuous}")

    summary_viewer = ReliabilityStabilitySummary(tracked_mice, use_cache=True)
    results = {}
    for mouse in tracked_mice:
        state = summary_viewer.define_state(
            mouse_name=mouse,
            envnum=1,
            reliability_threshold=reliability_threshold,
            reliability_method=reliability_method,
            smooth_width=int(smooth_width),
            use_session_filters=use_session_filters,
            continuous=continuous,
            max_session_diff=max_session_diff,
        )
        results[mouse] = summary_viewer.gather_data(state, try_cache=True)

    msms = {mouse: MultiSessionSpkmaps(Tracker(mouse)) for mouse in results}

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
    output_names = ["fraction_stable", "pfloc_changes", "spkmap_correlations"]

    max_environments = 3

    data = {key: np.full((len(results), max_environments, max_session_diff), np.nan) for key in output_keys}

    for imouse, mouse in enumerate(results):
        envstats = msms[mouse].env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        for ienv, env in enumerate(env_in_order):
            if ienv >= max_environments:
                continue
            if results[mouse][env] is None:
                continue
            for key in output_keys:
                if forward_backward == "forward":
                    data[key][imouse, ienv] = np.nanmean(results[mouse][env]["forward"][key], axis=0)
                elif forward_backward == "backward":
                    data[key][imouse, ienv] = np.nanmean(results[mouse][env]["backward"][key], axis=0)
                elif forward_backward == "both":
                    forward_data = np.nanmean(results[mouse][env]["forward"][key], axis=0)
                    backward_data = np.nanmean(results[mouse][env]["backward"][key], axis=0)
                    data[key][imouse, ienv] = np.nanmean(np.stack([forward_data, backward_data]), axis=0)

    num_cols = len(output_names)
    num_rows = max_environments if split_environments else 1

    cmap_mice = mpl.colormaps["tab10"]
    colors_mice = cmap_mice(np.linspace(0, 1, len(results)))
    colors_mice = {mouse: colors_mice[imouse] for imouse, mouse in enumerate(results)}
    colors_mice["CR_Hippocannula6"] = "black"
    colors_mice["CR_Hippocannula7"] = "dimgrey"
    linewidth = [1.5 if mouse in ["CR_Hippocannula6", "CR_Hippocannula7"] else 0.75 for mouse in results]
    zorder = [1 if mouse in ["CR_Hippocannula6", "CR_Hippocannula7"] else 0 for mouse in results]

    figwidth = 3
    figheight = 3
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(figwidth * num_cols, figheight * num_rows), layout="constrained")
    if num_rows == 1:
        ax = np.reshape(ax, (1, num_cols))
    for icol, col_name in enumerate(output_names):
        ctl_data = data[col_name + "_ctl"]
        red_data = data[col_name + "_red"]
        if not split_environments:
            ctl_data = np.nanmean(ctl_data, axis=1, keepdims=True)
            red_data = np.nanmean(red_data, axis=1, keepdims=True)
        diff_data = red_data - ctl_data

        for irow in range(num_rows):
            for imouse, mouse in enumerate(results):
                ax[irow, icol].plot(
                    range(1, max_session_diff + 1),
                    diff_data[imouse, irow],
                    color=colors_mice[mouse],
                    linewidth=linewidth[imouse],
                    zorder=zorder[imouse],
                )
            ax[irow, icol].set_ylabel(f"$\Delta$ {col_name} (red-ctl)")

        ax[-1, icol].set_xlabel("$\Delta$ Session")
        ax[0, icol].set_title(col_name)

        ax_titles = [(f"{col_name}" if irow == 0 else "") for irow in range(num_rows)]
        ax_titles = [title + ("\n" if irow == 0 else "") + f"Env#{irow+1}" for irow, title in enumerate(ax_titles)]
        for irow, title in enumerate(ax_titles):
            ax[irow, icol].set_title(title)

    suptitle = f"Full Summary: Threshold={reliability_threshold}, Continuous={continuous}, ForwardBackward={forward_backward}"
    fig.suptitle(suptitle)

    if show:
        plt.show(block=True)

    if save:
        title = (
            f"summary-threshold{reliability_threshold}-continuous{continuous}-forwardbackward{forward_backward}-splitenvironments{split_environments}"
        )
        fig_path = figure_path(f"summary-tracked_changes", title)
        save_figure(fig, fig_path)

    plt.close(fig)


def reliability_quantile_summary(
    reliability: dict,
    selected: bool = True,
    num_bins: int = 9,
    quantile_focus: list[int] = [-2, -1],
    max_sessions: int = 10,
    show: bool = False,
    save: bool = False,
):
    max_environments = 3
    bins = np.linspace(0, 1, num_bins)
    centers = edge2center(bins)

    ctl_key = "reliability_ctl" if selected else "reliability_ctl_all"
    red_key = "reliability_red" if selected else "reliability_red_all"
    ctl_reliability = {mouse: rel_data[ctl_key] for mouse, rel_data in reliability.items()}
    red_reliability = {mouse: rel_data[red_key] for mouse, rel_data in reliability.items()}

    msms = {mouse: MultiSessionSpkmaps(Tracker(mouse)) for mouse in reliability}
    num_mice = len(reliability)

    ctl_reliability = [np.full((num_mice, max_sessions, len(centers)), np.nan) for _ in range(max_environments)]
    red_reliability = [np.full((num_mice, max_sessions, len(centers)), np.nan) for _ in range(max_environments)]
    red_deviation = [np.full((num_mice, max_sessions, len(centers)), np.nan) for _ in range(max_environments)]

    for imouse, mouse in enumerate(reliability):
        envstats = msms[mouse].env_stats()
        env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
        env_in_order = [env for env in env_in_order if env != -1]
        for ienv, env in enumerate(env_in_order):
            for ises, (ctlval, redval) in enumerate(zip(reliability[mouse][ctl_key][env], reliability[mouse][red_key][env])):
                if ises >= max_sessions:
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

    num_quantiles = len(quantile_focus)
    figheight = 3
    figwidth = 3
    fig, ax = plt.subplots(num_quantiles, max_environments, figsize=(figwidth * max_environments, figheight * num_quantiles), layout="constrained")
    if num_quantiles == 1:
        ax = np.reshape(ax, (1, max_environments))
    for iquantile in range(num_quantiles):
        for ienv in range(max_environments):
            for imouse, mouse in enumerate(reliability):
                ax[iquantile, ienv].plot(
                    range(max_sessions),
                    red_deviation[ienv][imouse][:, quantile_focus[iquantile]],
                    color=colors_mice[mouse],
                    linewidth=linewidth[imouse],
                    zorder=zorder[imouse],
                )
            title = f"Environment #{ienv+1}\n" if iquantile == 0 else ""
            title += f"Quantile Focus: {np.arange(num_bins-1)[quantile_focus[iquantile]]+1}/{num_bins-1}"
            ax[iquantile, ienv].axhline(y=0, color="k", linewidth=0.5, zorder=-1)
            ax[iquantile, ienv].set_title(title)
            ax[iquantile, ienv].set_xlabel("Session #")
            ax[iquantile, ienv].set_ylabel("Red Deviation")

    if show:
        plt.show(block=True)

    if save:
        quantile_focus_number = [np.arange(num_bins - 1)[qf] for qf in quantile_focus]
        quantile_focus_str = ",".join([str(quantile_focus_number[i] + 1) for i in range(len(quantile_focus_number))])
        selection_string = "GoodROIs" if selected else "AllROIs"
        title = f"summary-{selection_string}-QF({quantile_focus_str})({num_bins-1}bins)"
        fig_path = figure_path(f"summary-quantile-deviation", title)
        save_figure(fig, fig_path)

    plt.close(fig)


start_at_mouse = None
if __name__ == "__main__":
    """
    The following plotting methods depend heavily on cached results in the analysis/before_the_reveal_temp_data directory!

    If any changes are made to processing methods, these should be recomputed. To do so, set do_tracking_summary=True and set save_results=True.
    Within the plot_tracking_summary method, make sure to set try_cache=False. In addition, we need use ReliabilityQuantileSummary with
    try_cache=False, save_cache=True!

    There's also a cache saving in placecell_reliability.py
    """
    do_reliability = False
    do_tracking_summary = True
    do_tracking_full_summary = False
    show = False
    save = False

    if start_at_mouse:
        start_index = np.where(tracked_mice == start_at_mouse)[0][0]
        tracked_mice = tracked_mice[start_index:]
        print("Starting at mouse:", start_at_mouse, "using these mice -->", tracked_mice[0])

    if do_tracking_summary:
        # Save intermediate results to cache
        save_results = True
        plot_tracking_summary(tracked_mice=tracked_mice, show=show, save=save, save_results=save_results)

    # if do_tracking_full_summary:
    #     for reliability_threshold in [0.3, 0.5, 0.7, 0.9]:
    #         for continuous in [True, False]:
    #             for forward_backward in ["forward", "backward", "both"]:
    #                 for split_environments in [True, False]:
    #                     plot_tracking_full_summary(
    #                         tracked_mice=tracked_mice,
    #                         reliability_threshold=reliability_threshold,
    #                         continuous=continuous,
    #                         forward_backward=forward_backward,
    #                         split_environments=split_environments,
    #                         show=show,
    #                         save=save,
    #                     )

    # if do_reliability:
    #     reliability_data = {}
    #     for mouse in tracked_mice:
    #         track = Tracker(mouse)
    #         print(f"Working on {track}")
    #         reliability = get_reliability(track)
    #         reliability_data[mouse] = reliability
    #         for selected in [True, False]:
    #             plot_reliability_distribution(track, reliability, selected=selected, show=show, save=save)
    #             plot_number_of_rois(track, reliability, selected=selected, show=show, save=save)
    #             for threshold in [0.5, 0.7, 0.9, 0.95]:
    #                 plot_reliability_extreme(track, reliability, threshold=threshold, selected=selected, show=show, save=save)
    #             plot_quantile_histogram(track, reliability, selected=selected, show=show, save=save)
    #             plot_reliability_histogram(track, reliability, selected=selected, show=show, save=save)

    #     for selected in [True, False]:
    #         num_bins = 7
    #         quantile_focus = [-1, 0]
    #         reliability_quantile_summary(reliability_data, selected=selected, num_bins=num_bins, quantile_focus=quantile_focus, show=show, save=save)
