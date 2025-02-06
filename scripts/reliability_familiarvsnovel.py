from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from vrAnalysis.helpers import argbool
from vrAnalysis import helpers
from vrAnalysis import database
from vrAnalysis.tracking import tracker
from vrAnalysis.analysis import placeCellMultiSession
from vrAnalysis.metrics import FractionActive, KernelDensityEstimator, plot_contours

keep_planes = [1, 2, 3, 4]
onefiles = ["mpci.roiActivityDeconvolvedOasis", "mpci.roiSignificantFluorescence"]


def get_mice(keep_mice=None, ignore_mice=None):
    mousedb = database.vrDatabase("vrMice")
    tracked_mice = list(mousedb.getTable(tracked=True)["mouseName"])
    if keep_mice is not None:
        tracked_mice = [mouse for mouse in tracked_mice if mouse in keep_mice]
    if ignore_mice is not None:
        tracked_mice = [mouse for mouse in tracked_mice if mouse not in ignore_mice]
    return tracked_mice


def temp_file_name_reliability_values(mouse_name):
    return f"{mouse_name}_reliability_values.pkl"


def get_args():
    parser = ArgumentParser(description="Calculate reliability of place cells between familiar and novel environments")
    parser.add_argument("--process-mouse-data", default=False, type=argbool, help="Process mouse data and save to temp files")
    parser.add_argument(
        "--mouse-summary-data",
        default=False,
        type=argbool,
        help="Load mouse data from temp files and make summary plots for each one.",
    )
    parser.add_argument(
        "--grand-comparison-reliability-only",
        default=False,
        type=argbool,
        help="Make a grand comparison plot of reliability between familiar and novel environments for all mice.",
    )
    parser.add_argument(
        "--mouse-summary-fraction-active-vs-reliability",
        default=False,
        type=argbool,
        help="Make a summary plot of fraction active vs reliability for each mouse.",
    )
    parser.add_argument("--keep_mice", type=str, nargs="+", help="Mice to keep", default=None)
    parser.add_argument("--ignore_mice", type=str, nargs="+", help="Mice to ignore", default=None)
    return parser.parse_args()


def pick_sessions(pcm: placeCellMultiSession) -> tuple[int, int, list[int]]:
    env_stats = pcm.env_stats()
    env_first = pcm.env_selector(envmethod="first")
    env_second = pcm.env_selector(envmethod="second")
    idx_ses = pcm.idx_ses_selector(envnum=env_second, sesmethod="all")
    idx_ses = [idx for idx in idx_ses if idx in env_stats[env_first]]
    return env_first, env_second, idx_ses


def process_mouse_data(mouse):
    mouse_reliability_data = {}
    for onefile in onefiles:
        progress_bar.set_description(f"Processing {mouse}/{onefile}")

        mouse_reliability_data[onefile] = {}

        # Get analysis object for this mouse (do it for each onefile type independently for clean dataloading)
        track = tracker(mouse)
        pcm = placeCellMultiSession(track, autoload=False, keep_planes=keep_planes, onefile=onefile)

        # Select sessions to compare first to second environment as many times as we can
        env_first, env_second, idx_ses = pick_sessions(pcm)

        # Then go get the reliability values for each environment / session combo
        rel_first = []
        rel_second = []
        fraction_active_first = []
        fraction_active_second = []
        idx_red = []
        for idx in tqdm(idx_ses, desc=f"Processing sessions...", leave=False):
            spkmaps = pcm.pcss[idx].get_spkmap(envnum=[env_first, env_second], average=False, smooth=5, trials="full")

            if onefile == onefiles[0]:
                idx_red.append(pcm.pcss[idx].vrexp.getRedIdx(keep_planes=keep_planes))

            relloo = [helpers.reliability_loo(smap) for smap in spkmaps]
            rel_first.append(relloo[0])
            rel_second.append(relloo[1])

            fraction_active = [
                FractionActive.compute(
                    spkmap,
                    activity_axis=2,
                    fraction_axis=1,
                    activity_method="rms",
                    fraction_method="participation",
                )
                for spkmap in spkmaps
            ]
            fraction_active_first.append(fraction_active[0])
            fraction_active_second.append(fraction_active[1])

        mouse_reliability_data[onefile]["rel_first"] = rel_first
        mouse_reliability_data[onefile]["rel_second"] = rel_second
        mouse_reliability_data[onefile]["fraction_active_first"] = fraction_active_first
        mouse_reliability_data[onefile]["fraction_active_second"] = fraction_active_second

        if onefile == onefiles[0]:
            mouse_reliability_data["env_first"] = env_first
            mouse_reliability_data["env_second"] = env_second
            mouse_reliability_data["idx_ses"] = idx_ses
            mouse_reliability_data["idx_red"] = idx_red

    # Save data
    pcm.save_temp_file(mouse_reliability_data, temp_file_name_reliability_values(mouse))


def mouse_summary_reliability_only(mouse, reduction="mean"):
    track = tracker(mouse)
    pcm = placeCellMultiSession(track, autoload=False)
    mouse_reliability_data = pcm.load_temp_file(temp_file_name_reliability_values(mouse))

    env_first, env_second, idx_ses = pick_sessions(pcm)

    num_sessions = len(idx_ses)

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

    if reduction == "mean":
        reduce_func = np.mean
    elif reduction == "median":
        reduce_func = np.median
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

    plt.close("all")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    for ione, onefile in enumerate(onefiles):

        for ii, ises in enumerate(idx_ses):
            parts = ax[ione].violinplot(mouse_reliability_data[onefile]["rel_first"][ii], positions=[ii], showextrema=False, side="low")
            color_violins(parts, facecolor=("k", 0.1))

            parts = ax[ione].violinplot(mouse_reliability_data[onefile]["rel_second"][ii], positions=[ii], showextrema=False, side="high")
            color_violins(parts, facecolor=("b", 0.1))

        ax[ione].plot(
            range(num_sessions),
            [reduce_func(mouse_reliability_data[onefile]["rel_first"][ii]) for ii in range(num_sessions)],
            "k-",
            label=f"Familiar {reduction}",
        )
        ax[ione].plot(
            range(num_sessions),
            [reduce_func(mouse_reliability_data[onefile]["rel_second"][ii]) for ii in range(num_sessions)],
            "b-",
            label=f"Novel {reduction}",
        )
        ax[ione].set_xticks(np.arange(num_sessions))
        ax[ione].set_xticklabels(idx_ses)
        ax[ione].legend(loc="lower right")
        ax[ione].set_xlabel("Session #")
        ax[ione].set_ylabel("Reliability")
        ax[ione].set_title(f"{onefile}")

    fig.suptitle(f"{mouse}")

    pcm.saveFigure(fig, "reliability_familiar_vs_novel", f"{mouse}_summary_{reduction}")


def grand_comparison_reliability_only(mice, reduction="mean"):
    if reduction == "mean":
        reduce_func = np.mean
    elif reduction == "median":
        reduce_func = np.median
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

    reliability_first = {}
    reliability_second = {}
    for mouse in mice:
        reliability_first[mouse] = {}
        reliability_second[mouse] = {}

        track = tracker(mouse)
        pcm = placeCellMultiSession(track, autoload=False)
        mouse_reliability_data = pcm.load_temp_file(temp_file_name_reliability_values(mouse))

        env_first, env_second, idx_ses = pick_sessions(pcm)
        num_sessions = len(idx_ses)

        for onefile in onefiles:
            reliability_first[mouse][onefile] = [reduce_func(mouse_reliability_data[onefile]["rel_first"][ii]) for ii in range(num_sessions)]
            reliability_second[mouse][onefile] = [reduce_func(mouse_reliability_data[onefile]["rel_second"][ii]) for ii in range(num_sessions)]

    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(len(mice))]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    for ione, onefile in enumerate(onefiles):
        for imouse, mouse in enumerate(mice):
            difference = np.array(reliability_second[mouse][onefile]) - np.array(reliability_first[mouse][onefile])
            ax[ione].plot(range(len(difference)), difference, color=colors[imouse], label=f"{mouse}")

        ax[ione].axhline(0, color="k", linestyle="--")
        ax[ione].set_xlabel("Session #")
        ax[ione].set_ylabel("Reliability")
        ax[ione].set_title(f"{onefile}")

    inset = ax[0].inset_axes([0.85, 0.57, 0.1, 0.4])
    inset.imshow(np.flipud(np.reshape(colors, (len(mice), 1, 4))), aspect="auto", extent=[0, 1, -0.5, len(mice) - 0.5])
    inset.set_xticks([])
    inset.set_yticks(np.arange(len(mice)))
    inset.set_yticklabels(mice)
    # inset.yaxis.tick_right()

    fig.suptitle(f"{reduction} Reliability Comparison")

    pcm.saveFigure(fig, "reliability_familiar_vs_novel", f"comparison_summary_{reduction}")


def mouse_summary_reliability_vs_fraction_active(mouse, onefile=onefiles[-1]):
    track = tracker(mouse)
    pcm = placeCellMultiSession(track, autoload=False)
    mouse_reliability_data = pcm.load_temp_file(temp_file_name_reliability_values(mouse))

    rel_first = mouse_reliability_data[onefile]["rel_first"]
    rel_second = mouse_reliability_data[onefile]["rel_second"]
    fraction_active_first = mouse_reliability_data[onefile]["fraction_active_first"]
    fraction_active_second = mouse_reliability_data[onefile]["fraction_active_second"]
    idx_red = mouse_reliability_data["idx_red"]

    env_first, env_second, idx_ses = pick_sessions(pcm)
    num_sessions = len(idx_ses)

    reliability_range = (-1.0, 1.0)
    fraction_active_range = (0.0, 1.0)
    params = dict(xrange=reliability_range, yrange=fraction_active_range, nbins=100)
    plt.close("all")
    fig, ax = plt.subplots(2, num_sessions, figsize=(12, 4), layout="constrained", sharex=True, sharey=True)
    if num_sessions == 1:
        ax = np.reshape(ax, (2, 1))

    for ii, idx in enumerate(idx_ses):
        if np.sum(idx_red[ii]) == 0:
            continue

        kde_ctl_first = KernelDensityEstimator(rel_first[ii][~idx_red[ii]], fraction_active_first[ii][~idx_red[ii]], **params).fit()
        kde_ctl_second = KernelDensityEstimator(rel_second[ii][~idx_red[ii]], fraction_active_second[ii][~idx_red[ii]], **params).fit()
        kde_red_first = KernelDensityEstimator(rel_first[ii][idx_red[ii]], fraction_active_first[ii][idx_red[ii]], **params).fit()
        kde_red_second = KernelDensityEstimator(rel_second[ii][idx_red[ii]], fraction_active_second[ii][idx_red[ii]], **params).fit()

        ctl_plot_data_first = kde_ctl_first.plot_data
        red_plot_data_first = kde_red_first.plot_data
        difference_first = red_plot_data_first - ctl_plot_data_first
        max_diff_first = np.max(np.abs(difference_first))

        ctl_plot_data_second = kde_ctl_second.plot_data
        red_plot_data_second = kde_red_second.plot_data
        difference_second = red_plot_data_second - ctl_plot_data_second
        max_diff_second = np.max(np.abs(difference_second))

        max_diff = np.max([max_diff_first, max_diff_second])

        ax[0, ii].imshow(difference_first, extent=kde_ctl_first.extent, aspect="auto", cmap="bwr", vmin=-max_diff, vmax=max_diff)
        ax[1, ii].imshow(difference_second, extent=kde_ctl_second.extent, aspect="auto", cmap="bwr", vmin=-max_diff, vmax=max_diff)

        ax[0, ii].set_xlim(reliability_range)
        ax[1, ii].set_xlim(reliability_range)
        ax[0, ii].set_ylim(fraction_active_range)
        ax[1, ii].set_ylim(fraction_active_range)

        ax[1, ii].set_xlabel("Reliability")
        if ii == 0:
            ax[0, ii].set_ylabel("Fraction Active")
            ax[1, ii].set_ylabel("Fraction Active")

        ax[0, ii].set_title(f"Familiar {idx}")
        ax[1, ii].set_title(f"Novel {idx}")

    fig.suptitle(f"{mouse}")
    pcm.saveFigure(fig, "reliability_familiar_vs_novel", f"{mouse}_fraction_active_vs_reliability_ctlred")


if __name__ == "__main__":

    args = get_args()
    mice = get_mice(keep_mice=args.keep_mice, ignore_mice=args.ignore_mice)

    if args.process_mouse_data:
        progress_bar = tqdm(mice, desc="Processing mice")
        for mouse in progress_bar:
            process_mouse_data(mouse)

    if args.mouse_summary_data:
        with helpers.batch_plot_context():
            for mouse in tqdm(mice, desc="Reliability only plots for each mouse..."):
                mouse_summary_reliability_only(mouse, reduction="mean")
                mouse_summary_reliability_only(mouse, reduction="median")

    if args.grand_comparison_reliability_only:
        with helpers.batch_plot_context():
            grand_comparison_reliability_only(mice, reduction="mean")
            grand_comparison_reliability_only(mice, reduction="median")

    if args.mouse_summary_fraction_active_vs_reliability:
        with helpers.batch_plot_context():
            for mouse in tqdm(mice, desc="Reliability vs Fraction Active plots for each mouse..."):
                mouse_summary_reliability_vs_fraction_active(mouse)
