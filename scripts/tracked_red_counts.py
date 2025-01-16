from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import itertools
from argparse import ArgumentParser

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from vrAnalysis import analysis
from vrAnalysis import helpers
from vrAnalysis import database
from vrAnalysis import tracking
from vrAnalysis import fileManagement as fm

sessiondb = database.vrDatabase("vrSessions")
mousedb = database.vrDatabase("vrMice")


def temp_file_name(mouse_name, consecutive_only=False):
    if consecutive_only:
        return f"{mouse_name}_tracked_red_counts_consecutive.pkl"
    else:
        return f"{mouse_name}_tracked_red_counts.pkl"


def handle_args():
    parser = ArgumentParser(description="Measure tracking statistics for red cells")
    parser.add_argument("--load_red_counts", type=helpers.argbool, default=False, help="Load red counts from temp files")
    parser.add_argument("--consecutive_only", type=helpers.argbool, default=False, help="Only consider consecutive sessions")
    parser.add_argument("--plot_individual_stats", type=helpers.argbool, default=False, help="Plot individual stats")
    parser.add_argument("--plot_summary_stats", type=helpers.argbool, default=False, help="Plot summary stats")
    parser.add_argument("--max_combinations", type=int, default=200, help="Maximum number of combinations to use for each group of sessions")
    parser.add_argument("--max_display_comb", type=int, default=24, help="Maximum number of combinations to display")
    return parser.parse_args()


def get_mice():
    mousedb = database.vrDatabase("vrMice")
    tracked_mice = list(mousedb.getTable(tracked=True)["mouseName"])
    ignore_mice = []
    use_mice = [mouse for mouse in tracked_mice if mouse not in ignore_mice]
    return use_mice


def get_combinations(idx, max_combinations=200, consecutive_only=False):
    num_per_group = range(2, len(idx) + 1)
    combinations_per_group = {}

    for num in num_per_group:
        if consecutive_only:
            # For consecutive combinations, we only need to consider windows
            consecutive_combos = []
            for i in range(len(idx) - num + 1):
                consecutive_combos.append(tuple(idx[i : i + num]))
            combinations_per_group[num] = consecutive_combos
        else:
            combinations_per_group[num] = list(itertools.combinations(idx, num))

        # if the number of combinations is more than max_combinations, pick max_combinations random ones
        if len(combinations_per_group[num]) > max_combinations:
            idx_random_selection = sorted(np.random.choice(len(combinations_per_group[num]), max_combinations, replace=False))
            combinations_per_group[num] = [combinations_per_group[num][i] for i in idx_random_selection]

    combinations = itertools.chain(*[combinations_per_group[num] for num in num_per_group])
    return list(combinations)


def get_red_cells(pcm, idx_ses, include_reliable=True, keep_planes=None, relcor_cutoff=0.6):
    # Get tracked cells
    idx_tracked = pcm.track.get_tracked_idx(idx_ses=idx_ses, keep_planes=keep_planes or pcm.keep_planes)
    num_tracked = idx_tracked.shape[1]

    # Get tracked red cells
    idx_red = [pcm.pcss[i].vrexp.getRedIdx(keep_planes=keep_planes or pcm.keep_planes) for i in idx_ses]
    idx_red = np.stack([idx_red[ii][idx_tracked[ii]] for ii in range(len(idx_ses))])
    red = np.any(idx_red, axis=0)

    results = dict(
        num_tracked=num_tracked,
        num_tracked_red=np.sum(red),
        fraction_marked_red=np.mean(idx_red[:, red]),
    )

    # Get tracked reliable cells
    if include_reliable:
        relmse, relcor, relloo = helpers.named_transpose([pcm.pcss[i].get_reliability_values() for i in idx_ses])
        for ii in range(len(idx_ses)):
            for jj in range(len(relmse[ii])):
                relmse[ii][jj] = relmse[ii][jj][idx_tracked[ii]]
                relcor[ii][jj] = relcor[ii][jj][idx_tracked[ii]]
                relloo[ii][jj] = relloo[ii][jj][idx_tracked[ii]]
        # all_relmse = np.concatenate([np.stack(rel) for rel in relmse], axis=0)
        all_relcor = np.concatenate([np.stack(rel) for rel in relcor], axis=0)
        # all_relloo = np.concatenate([np.stack(rel) for rel in relloo], axis=0)
        idx_reliable = all_relcor > relcor_cutoff
        reliable = np.any(idx_reliable, axis=0)

        # Update results to include reliable stuff
        update_dict = dict(
            num_tracked_reliable=np.sum(reliable),
            num_tracked_reliable_red=np.sum(reliable & red),
        )
        results.update(update_dict)

    return results


def plot_session_tracking(mouse_name, summary, key="num_tracked_red", max_display_comb=24):
    num_sessions = max(max(comb) for comb in summary["idx_sessions_combs"]) + 1
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get color map for sessions
    colors = plt.cm.tab20(np.linspace(0, 1, num_sessions))

    # Calculate the number of columns needed (max combination size)
    max_combo_size = max(len(combo) for combo in summary["idx_sessions_combs"])

    # Parameters for visualization
    box_width = 0.8
    box_height = 3
    x_spacing = 1.5  # Increased spacing between columns
    total_height = num_sessions * box_height

    # Group combinations by size
    combos_by_size = {}
    for idx, (combo, result) in enumerate(zip(summary["idx_sessions_combs"], summary[key])):
        size = len(combo)
        if size not in combos_by_size:
            combos_by_size[size] = []
        combos_by_size[size].append((combo, result))

    for size in combos_by_size:
        num_per_size = len(combos_by_size[size])
        if num_per_size > max_display_comb:
            idx_random_selection = sorted(np.random.choice(num_per_size, max_display_comb, replace=False))
            combos_by_size[size] = [combos_by_size[size][i] for i in idx_random_selection]

    # Draw color-coded session boxes on the left
    for i in range(num_sessions):
        y_pos = i * box_height
        rect = plt.Rectangle((0, y_pos), box_width, box_height, facecolor=colors[i], edgecolor="black")
        ax.add_patch(rect)
        ax.text(box_width / 2, y_pos + box_height / 2, f"S{i}", ha="center", va="center", color="white" if np.mean(colors[i][:3]) < 0.5 else "black")

    # Calculate and draw bubbles with session arcs
    for size in range(2, max_combo_size + 1):
        if size not in combos_by_size:
            continue

        combos = combos_by_size[size]
        x_pos = (size - 1) * x_spacing
        ax.axvline(x_pos - x_spacing / 2, color="black", linewidth=1)

        # Calculate spacing for this column
        num_combos = len(combos)
        if num_combos > 1:
            y_spacing = total_height / num_combos
        else:
            y_spacing = 0

        # Calculate bubble radius
        max_radius = min(y_spacing / 2 if num_combos > 1 else box_height / 2, x_spacing / 2)

        for idx, (combo, num_tracked) in enumerate(combos):
            if num_combos > 1:
                y_pos = idx * y_spacing + y_spacing / 2
            else:
                y_pos = total_height / 2

            # Draw base bubble
            circle = plt.Circle((x_pos, y_pos), max_radius, facecolor="white", edgecolor="black", alpha=0.2)
            ax.add_patch(circle)

            # Draw colored arcs for each session in the combination
            arc_length = 360 / len(combo)  # Divide circle evenly
            for i, session_idx in enumerate(sorted(combo)):
                start_angle = i * arc_length
                arc = Arc(
                    (x_pos, y_pos),
                    max_radius * 2,
                    max_radius * 2,  # width and height
                    theta1=start_angle,
                    theta2=start_angle + arc_length,
                    color=colors[session_idx],
                    linewidth=2,
                )
                ax.add_patch(arc)

            # Add tracked number
            ax.text(x_pos, y_pos, str(num_tracked), ha="center", va="center")

    # Set plot limits and remove axes
    ax.set_xlim(-0.5, (max_combo_size + 1) * x_spacing)
    ax.set_ylim(-0.5, total_height + box_height)
    ax.axis("off")
    ax.set_aspect("equal")

    # Add column labels
    for i in range(2, max_combo_size + 1):
        ax.text((i - 1) * x_spacing, -0.5, f"{i}", ha="center", va="top")

    ax.text(max_combo_size * x_spacing / 2, -1.0, "Number of Sessions Included", ha="center", va="top")
    ax.text(max_combo_size * x_spacing / 2, total_height + 1.0, key, ha="center", va="top")
    ax.text(max_combo_size * x_spacing / 2, total_height + 2.0, mouse_name, ha="center", va="top")

    plt.tight_layout()
    return fig, ax


def condense_results(results, idx_sessions_combs):
    summary = {}
    summary["idx_sessions_combs"] = idx_sessions_combs
    averages = {}
    for res, comb in zip(results, idx_sessions_combs):
        num_sessions = len(comb)
        for key in res:
            if key not in summary:
                summary[key] = []
            if key not in averages:
                averages[key] = {}
            if num_sessions not in averages[key]:
                averages[key][num_sessions] = []
            summary[key].append(res[key])
            averages[key][num_sessions].append(res[key])
    for key in averages:
        for num_sessions in averages[key]:
            averages[key][num_sessions] = np.mean(averages[key][num_sessions])
    summary["averages"] = averages
    return summary


def get_pcm(mouse_name, autoload=False):
    track = tracking.tracker(mouse_name)
    pcm = analysis.placeCellMultiSession(track, autoload=autoload, keep_planes=[1, 2, 3, 4])
    return pcm


def get_mouse_data(pcm, max_combinations=200, consecutive_only=False):
    idx_sessions = [i for i in range(len(pcm.pcss))]
    idx_sessions_combs = get_combinations(idx_sessions, max_combinations=max_combinations, consecutive_only=consecutive_only)

    results = [get_red_cells(pcm, idx_ses, include_reliable=True) for idx_ses in tqdm(idx_sessions_combs)]

    return results, idx_sessions_combs


if __name__ == "__main__":
    args = handle_args()
    mouse_names = get_mice()
    short_names = helpers.short_mouse_names(mouse_names)

    dirname = "red_tracking_stats"
    if args.consecutive_only:
        dirname += "_consecutive"

    if args.load_red_counts:
        print("Loading red statistics again... this will take a while!")
        for mouse_name in tqdm(mouse_names, desc="Loading red statistics"):
            pcm = get_pcm(mouse_name, autoload=True)
            results, idx_sessions_combs = get_mouse_data(pcm, max_combinations=args.max_combinations, consecutive_only=args.consecutive_only)
            summary = condense_results(results, idx_sessions_combs)
            pcm.save_temp_file(summary, temp_file_name(mouse_name, consecutive_only=args.consecutive_only))

    if args.plot_individual_stats:
        print("Plotting individual stats from saved data...")
        keys = ["num_tracked_red", "num_tracked_reliable_red"]
        for mouse_name in tqdm(mouse_names, desc="Plotting individual stats"):
            pcm = get_pcm(mouse_name, autoload=False)
            summary = pcm.load_temp_file(temp_file_name(mouse_name, consecutive_only=args.consecutive_only))
            for key in keys:
                fig, ax = plot_session_tracking(mouse_name, summary, key=key, max_display_comb=args.max_display_comb)
                fig_path = (pcm.saveDirectory(dirname) / f"{mouse_name}_{key}.png").with_suffix(".png")
                helpers.save_figure(fig, fig_path)
                plt.close("all")

    if args.plot_summary_stats:
        print("Plotting summary stats from saved data...")
        line_plot_keys = ["num_tracked", "num_tracked_red", "num_tracked_reliable_red"]
        extra_plot_keys = ["fraction_marked_red"]

        line_plot_data = {key: {} for key in line_plot_keys}
        extra_plot_data = {key: {} for key in extra_plot_keys}
        extra_plot_data["fraction_tracked_red"] = {}
        summary_stats = {}
        for mouse_name, short_name in zip(mouse_names, short_names):
            pcm = get_pcm(mouse_name, autoload=False)
            summary_stats[short_name] = pcm.load_temp_file(temp_file_name(mouse_name, consecutive_only=args.consecutive_only))

        for short_name in summary_stats:
            for key in line_plot_keys:
                xvals = list(summary_stats[short_name]["averages"][key].keys())
                yvals = list(summary_stats[short_name]["averages"][key].values())
                line_plot_data[key][short_name] = (xvals, yvals)
            for key in extra_plot_keys:
                xvals = list(summary_stats[short_name]["averages"][key].keys())
                yvals = list(summary_stats[short_name]["averages"][key].values())
                extra_plot_data[key][short_name] = (xvals, yvals)

            # Also get fraction of tracked cells that are red
            xvals = list(summary_stats[short_name]["averages"]["num_tracked"].keys())
            num_tracked = list(summary_stats[short_name]["averages"]["num_tracked"].values())
            num_tracked_red = list(summary_stats[short_name]["averages"]["num_tracked_red"].values())
            yvals = [num_tracked_red[i] / num_tracked[i] if num_tracked[i] > 0 else np.nan for i in range(len(num_tracked))]
            extra_plot_data["fraction_tracked_red"][short_name] = (xvals, yvals)

        plt.rcParams["font.size"] = 14
        figdim = 4.2
        fig, ax = plt.subplots(len(line_plot_keys), 1, figsize=(figdim * 2, figdim * len(line_plot_keys)), layout="constrained", sharex=True)
        for i, key in enumerate(line_plot_keys):
            for short_name in line_plot_data[key]:
                ax[i].plot(line_plot_data[key][short_name][0], line_plot_data[key][short_name][1], label=short_name)
            ax[i].set_ylabel(key)
            if i == 0:
                ax[i].legend()
            ax[i].set_yscale("log")
        ax[-1].set_xlabel("Number of Sessions Tracked")

        fig_path = (pcm.saveDirectory(dirname) / f"numerical_red_stats").with_suffix(".png")
        helpers.save_figure(fig, fig_path)

        figdim = 5
        fig, ax = plt.subplots(1, len(extra_plot_data), figsize=(figdim * len(extra_plot_data), figdim), layout="constrained", sharex=True)
        for i, key in enumerate(extra_plot_data):
            for short_name in extra_plot_data[key]:
                ax[i].plot(extra_plot_data[key][short_name][0], extra_plot_data[key][short_name][1], label=short_name)
            ax[i].set_ylabel(key)
            if i == len(extra_plot_data) - 1:
                ax[i].legend(loc="best")
            ax[i].set_xlabel("Number of Sessions Tracked")
            ax[i].set_ylim(0, 1)

        fig_path = (pcm.saveDirectory(dirname) / f"fractional_red_stats").with_suffix(".png")
        helpers.save_figure(fig, fig_path)
