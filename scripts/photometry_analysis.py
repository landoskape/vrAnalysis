from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from scipy.stats import ttest_rel
from matplotlib import pyplot as plt
import matplotlib as mpl

from vrAnalysis import fileManagement as files
from vrAnalysis.helpers import errorPlot, argbool, save_figure, batch_plot_context, beeswarm
from photometry.loaders import get_files, process_data_parallel


mice_list = ["ATL061", "ATL063", "ATL064", "ATL065", "ATL066", "ATL068", "ATL069", "ATL070", "ATL071", "ATL072", "ATL073"]
colors = ["k", "k", "k", "r", "r", "g", "g", "g", "g", "g", "g"]
preperiod = 0.2
postperiod = 2.0


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--process", type=argbool, default=False, help="Process the data")
    parser.add_argument("--save", type=argbool, default=False, help="Save the data")
    parser.add_argument("--analyze_single_sessions", type=argbool, default=False, help="Analyze single sessions")
    parser.add_argument("--analyze_single_mice", type=argbool, default=False, help="Analyze single mice")
    parser.add_argument("--compare_mice", type=argbool, default=False, help="Compare mice")
    return parser.parse_args()


def photometry_dir():
    return files.analysisPath() / "photometry"


def get_save_path(mouse_name):
    save_path = photometry_dir() / "saved_data" / f"{mouse_name}_results.npz"
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    return save_path


def get_figure_path(mouse_name, figure_name):
    figure_path = photometry_dir() / f"{mouse_name}" / f"{figure_name}"
    if not figure_path.parent.exists():
        figure_path.parent.mkdir(parents=True)
    return figure_path


def process_mouse_data(mouse_name, preperiod, postperiod):
    dirs, findex, data = get_files(mouse_name)
    print(f"Found {len(data)} files for {mouse_name}.")

    # Get the results from all sessions of the data files
    results, opto_responses = process_data_parallel(data, preperiod=preperiod, postperiod=postperiod, parallel=False)

    return results, opto_responses


def save_mouse_data(mouse_name, results, opto_responses):
    file_path = get_save_path(mouse_name)
    np.savez_compressed(file_path, results=results, opto_responses=opto_responses)


def analyze_single_session(session_data):
    relative_data = session_data["in2_opto"] - session_data["in1_opto"]
    trial_time = session_data["time_opto"]

    baseline_window = (-0.15, -0.05)
    peak_window = (0.05, 0.15)
    baseline_idx = np.where((trial_time >= baseline_window[0]) & (trial_time <= baseline_window[1]))[0]
    peak_idx = np.where((trial_time >= peak_window[0]) & (trial_time <= peak_window[1]))[0]

    baseline_mean = np.mean(relative_data[:, baseline_idx], axis=1)
    peak_mean = np.mean(relative_data[:, peak_idx], axis=1)

    data_compare = np.stack([baseline_mean, peak_mean], axis=0)
    ttest_result = ttest_rel(peak_mean, baseline_mean)

    # Also measure effect size
    difference = np.mean(peak_mean - baseline_mean)
    average_std = np.mean(np.std(data_compare, axis=1))
    effect_size = difference / average_std
    return data_compare, ttest_result, effect_size


def get_single_mouse_delta_trajectory(session_analysis):
    data_compare = [c_session["data_compare"] for c_session in session_analysis]
    deltas = [dc[1] - dc[0] for dc in data_compare]
    max_trials = max([len(d) for d in deltas])
    deltas_nanpad = [np.pad(d, (0, max_trials - len(d)), mode="constant", constant_values=np.nan) for d in deltas]
    deltas_nanpad = np.stack(deltas_nanpad, axis=0)
    return deltas_nanpad


def plot_single_mouse_data(mouse_name, session_analysis):
    # data_compare = [c_session["data_compare"] for c_session in session_analysis]
    # ttest_result = [c_session["ttest_result"] for c_session in session_analysis]
    effect_size = [c_session["effect_size"] for c_session in session_analysis]
    deltas = get_single_mouse_delta_trajectory(session_analysis)

    cmap = plt.get_cmap("cividis")
    colors = np.array([cmap(i / len(session_analysis)) for i in range(len(session_analysis))])

    fig = plt.figure(figsize=(10, 4), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.2])

    ax = fig.add_subplot(gs[0])
    errorPlot(np.arange(len(session_analysis)), deltas, axis=1, ax=ax, color="k", alpha=0.2, se=True)
    for i in range(len(session_analysis)):
        ax.plot(i, np.nanmean(deltas[i]), color=colors[i], marker="o", markersize=5)
    ax.set_xlim(-0.5, len(session_analysis) - 0.5)
    ax.set_ylim(-0.002, 0.012)
    ax.set_ylabel(r"$\Delta$ Fluorescence (a.u.)")
    ax.set_title("Baseline vs Peak")

    ax = fig.add_subplot(gs[1])
    ax.plot(effect_size, color="black")
    for i in range(len(session_analysis)):
        ax.plot(i, effect_size[i], color=colors[i], marker="o", markersize=8)
    ax.set_xlabel("Session #")
    ax.set_ylabel("Effect Size")
    ax.set_title("Effect Size")
    ax.set_ylim(0, 3)

    ax = fig.add_subplot(gs[2])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(session_analysis) - 1))
    cbar = plt.colorbar(sm, cax=ax, label="Session #")
    cbar.set_ticks([0, len(session_analysis) - 1])  # Only show first and last session numbers
    cbar.set_ticklabels(["1", str(len(session_analysis))])  # Label them as 1 and max

    fig.suptitle(mouse_name)
    return fig


def plot_single_session_data(mouse_name, session_analysis, session_idx):

    session_data = session_analysis[session_idx]["session_data"]
    data_compare = session_analysis[session_idx]["data_compare"]

    fig = plt.figure(figsize=(10, 4), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.2])
    ax = fig.add_subplot(gs[0])
    ax.plot([0, 1], data_compare, color="black", alpha=0.2, linewidth=0.5)
    ax.plot([0, 1], np.mean(data_compare, axis=1), color="black", alpha=1, linewidth=2.5)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.01, 0.02)
    ax.set_xticks([0, 1], ["Baseline", "Peak"])
    ax.set_ylabel("Fluorescence (a.u.)")

    cmap = plt.get_cmap("bwr")
    relative_data = session_data["in2_opto"] - session_data["in1_opto"]
    max_val = 0.025  # np.max(np.abs(relative_data))
    ax = fig.add_subplot(gs[1])
    imdata = ax.imshow(
        relative_data,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        extent=[-preperiod, postperiod, 0, relative_data.shape[0]],
        vmin=-max_val,
        vmax=max_val,
    )
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlim(-preperiod, postperiod)
    ax.set_ylim(0, session_data["in2_opto"].shape[0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trial #")
    ax.set_title("Session Trials")

    ax = fig.add_subplot(gs[2])
    plt.colorbar(imdata, cax=ax, label="Fluorescence (a.u.)")

    fig.suptitle(f"{mouse_name} - Session {session_idx}")
    return fig


def compare_mice_deltas(mice_list, session_analysis, deltas, colors=None):
    cmap = plt.get_cmap("tab20")
    colors = colors or np.array([cmap(i / len(mice_list)) for i in range(len(mice_list))])
    ttest_star_offset = 0.008
    ttest_start_delta = 0.001

    fig = plt.figure(figsize=(8, 4), layout="constrained")
    ax = fig.add_subplot(1, 2, 1)
    for i, mouse_name in enumerate(mice_list):
        errorPlot(np.arange(deltas[i].shape[0]), deltas[i], axis=1, ax=ax, color=colors[i], alpha=0.2, se=True, label=mouse_name)

        for isession, session in enumerate(session_analysis[mouse_name]):
            ttest_result = session["ttest_result"].pvalue
            if ttest_result < 0.0001:
                ax.annotate("***", (isession, ttest_star_offset + i * ttest_start_delta), fontsize=8, fontweight="bold", color=colors[i])
            elif ttest_result < 0.001:
                ax.annotate("**", (isession, ttest_star_offset + i * ttest_start_delta), fontsize=8, fontweight="bold", color=colors[i])
            elif ttest_result < 0.01:
                ax.annotate("*", (isession, ttest_star_offset + i * ttest_start_delta), fontsize=8, fontweight="bold", color=colors[i])
    ax.set_xlim(-0.5, max([len(d) for d in deltas]) + 1.5)
    ax.set_ylim(-0.005, 0.015)
    ax.legend(loc="lower right")

    ax.annotate("***=0.0001", (8, -0.004), fontsize=12, color="black")
    ax.annotate("**=0.001", (4, -0.004), fontsize=12, color="black")
    ax.annotate("*=0.01", (0, -0.004), fontsize=12, color="black")

    ax.set_xlabel("Session #")
    ax.set_ylabel(r"$\Delta$ Fluorescence (a.u.)")
    ax.set_title("Delta Trajectory")

    ax = fig.add_subplot(1, 2, 2)
    for i, mouse_name in enumerate(mice_list):
        c_means = np.nanmean(deltas[i], axis=1)
        if len(c_means) <= 1:
            continue
        c_xx = beeswarm(c_means)
        ax.scatter(i + c_xx / 2, c_means, color=colors[i], s=10, alpha=0.5)

    ax.set_xlabel("Mouse #")
    ax.set_ylabel(r"$\Delta$ Fluorescence (a.u.)")

    return fig


def analyze_mouse_sessions(mouse_name, mouse_results):
    session_analysis = []
    for isession, session_data in enumerate(mouse_results):
        data_compare, ttest_result, effect_size = analyze_single_session(session_data)
        c_session = dict(
            mouse_name=mouse_name,
            session_idx=isession,
            session_data=session_data,
            data_compare=data_compare,
            ttest_result=ttest_result,
            effect_size=effect_size,
        )
        session_analysis.append(c_session)
    return session_analysis


if __name__ == "__main__":
    args = parse_args()

    results = []
    opto_responses = []
    if args.process:
        for mouse_name in tqdm(mice_list, desc="Processing raw data for each mouse."):
            dirs, findex, data = get_files(mouse_name)
            print(f"Found {len(data)} files for {mouse_name}.")

            # Get the results from all sessions of the data files
            output = process_data_parallel(data, preperiod=preperiod, postperiod=postperiod, parallel=False)
            results.append(output[0])
            opto_responses.append(output[1])

    else:
        for mouse_name in tqdm(mice_list, desc="Loading saved data for each mouse."):
            file_path = get_save_path(mouse_name)
            saved_data = np.load(file_path, allow_pickle=True)
            results.append(saved_data["results"])
            opto_responses.append(saved_data["opto_responses"])

    if args.process and args.save:
        for i, mouse_name in enumerate(tqdm(mice_list, desc="Saving raw data for each mouse.")):
            save_mouse_data(mouse_name, results[i], opto_responses[i])

    if args.analyze_single_sessions or args.analyze_single_mice or args.compare_mice:
        session_analysis = {}
        for i, mouse_name in enumerate(tqdm(mice_list, desc="Analyzing Single Sessions")):
            session_analysis[mouse_name] = analyze_mouse_sessions(mouse_name, results[i])

    if args.analyze_single_sessions:
        with batch_plot_context():
            for i, mouse_name in enumerate(tqdm(mice_list, desc="Plotting single session data")):
                for isession, session in enumerate(session_analysis[mouse_name]):
                    fig = plot_single_session_data(mouse_name, session_analysis[mouse_name], isession)
                    save_figure(fig, get_figure_path(mouse_name, f"single_session_data_{isession}"))
                    plt.close("all")

    if args.analyze_single_mice:
        with batch_plot_context():
            for i, mouse_name in enumerate(tqdm(mice_list, desc="Plotting single mouse data")):
                fig = plot_single_mouse_data(mouse_name, session_analysis[mouse_name])
                save_figure(fig, get_figure_path(mouse_name, "single_session_analysis"))
                plt.close("all")

    if args.compare_mice:
        with batch_plot_context():
            deltas = []
            for i, mouse_name in enumerate(tqdm(mice_list, desc="Gathering summary data (deltas) from each mouse")):
                deltas.append(get_single_mouse_delta_trajectory(session_analysis[mouse_name]))
            fig = compare_mice_deltas(mice_list, session_analysis, deltas, colors=colors)
            save_figure(fig, photometry_dir() / "compare_mice_deltas")
            plt.close("all")
