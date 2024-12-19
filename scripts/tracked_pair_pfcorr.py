from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt


import os, sys

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from vrAnalysis import analysis
from vrAnalysis import helpers
from vrAnalysis import database
from vrAnalysis import tracking


def handle_args():
    parser = ArgumentParser(description="Compute correlations between sessions for tracked cells")
    parser.add_argument("--max_diff", type=int, default=4, help="Maximum difference between session indices to consider (default=4)")
    parser.add_argument("--relcor_cutoff", type=float, default=0.5, help="Minimum reliability correlation to consider (default=0.5)")
    parser.add_argument("--smooth", type=int, default=2, help="Smoothing window for spkmaps (default=2)")
    parser.add_argument("--use_saved", type=helpers.argbool, default=True, help="Use saved data instead of recomputing (default=True)")
    parser.add_argument("--save_results", type=helpers.argbool, default=True, help="Save results to temp file (default=True)")
    parser.add_argument("--analyze_mice", type=helpers.argbool, default=False, help="Analyze all tracked mice (default=True)")
    parser.add_argument("--make_summary", type=helpers.argbool, default=False, help="Make summary plot (default=False)")
    return parser.parse_args()


def temp_file_name(mouse_name):
    return f"{mouse_name}_tracked_pair_pfcorr.pkl"


def get_mice():
    mousedb = database.vrDatabase("vrMice")
    tracked_mice = list(mousedb.getTable(tracked=True)["mouseName"])
    ignore_mice = []
    use_mice = [mouse for mouse in tracked_mice if mouse not in ignore_mice]
    return use_mice


def make_corrs(pcm, envnum, idx_ses, smooth=None, relcor_cutoff=-np.inf, max_diff=None):
    pcm.load_pcss_data(idx_ses=idx_ses)
    idx_pairs = helpers.all_pairs(idx_ses)
    if max_diff is not None:
        idx_pairs = idx_pairs[np.abs(idx_pairs[:, 1] - idx_pairs[:, 0]) <= max_diff]
    ses_diffs = idx_pairs[:, 1] - idx_pairs[:, 0]
    ctlcorrs = []
    redcorrs = []
    ctlcorrs_reliable = []
    redcorrs_reliable = []
    ctl_latent_corrs = []
    red_latent_corrs = []
    ctl_latent_corrs_reliable = []
    red_latent_corrs_reliable = []
    for idx in tqdm(idx_pairs, desc="computing correlations", leave=False):
        spkmaps, extras = pcm.get_spkmaps(
            envnum,
            trials="full",
            pop_nan=True,
            smooth=smooth,
            tracked=True,
            average=True,
            idx_ses=idx,
            include_latents=True,
        )
        relcor = extras["relcor"]
        idx_red = extras["idx_red"]
        latents = extras["latents"]

        any_rel = np.any(np.stack(relcor) > relcor_cutoff, axis=0)
        any_red = np.any(np.stack(idx_red), axis=0)

        # For reliable ROIs
        ctlmaps = [s[~any_red & any_rel] for s in spkmaps]
        redmaps = [s[any_red & any_rel] for s in spkmaps]
        ctlcorrs_reliable.append(helpers.vectorCorrelation(ctlmaps[0], ctlmaps[1], axis=1))
        redcorrs_reliable.append(helpers.vectorCorrelation(redmaps[0], redmaps[1], axis=1))

        # For all ROIs
        ctlmaps = [s[~any_red] for s in spkmaps]
        redmaps = [s[any_red] for s in spkmaps]
        ctlcorrs.append(helpers.vectorCorrelation(ctlmaps[0], ctlmaps[1], axis=1))
        redcorrs.append(helpers.vectorCorrelation(redmaps[0], redmaps[1], axis=1))

        # For latents of reliable ROIs
        ctllats = [s[~any_red & any_rel] for s in latents]
        redlats = [s[any_red & any_rel] for s in latents]
        ctl_latent_corrs_reliable.append(helpers.vectorCorrelation(ctllats[0], ctllats[1], axis=1))
        red_latent_corrs_reliable.append(helpers.vectorCorrelation(redlats[0], redlats[1], axis=1))

        # For latents of all ROIs
        ctllats = [s[~any_red] for s in latents]
        redlats = [s[any_red] for s in latents]
        ctl_latent_corrs.append(helpers.vectorCorrelation(ctllats[0], ctllats[1], axis=1))
        red_latent_corrs.append(helpers.vectorCorrelation(redlats[0], redlats[1], axis=1))

    output = dict(
        ctlcorrs=ctlcorrs,
        redcorrs=redcorrs,
        ctlcorrs_reliable=ctlcorrs_reliable,
        redcorrs_reliable=redcorrs_reliable,
        ctl_latent_corrs=ctl_latent_corrs,
        red_latent_corrs=red_latent_corrs,
        ctl_latent_corrs_reliable=ctl_latent_corrs_reliable,
        red_latent_corrs_reliable=red_latent_corrs_reliable,
        ses_diffs=ses_diffs,
    )
    return output


def summarize_mouse(results):
    """Make summary histograms and averages for results dictionary from a single mouse."""
    bins = np.linspace(-1, 1, 11)
    centers = helpers.edge2center(bins)

    num_diffs = len(np.unique(results["ses_diffs"]))
    ctl_counts = np.zeros((num_diffs, len(centers)))
    red_counts = np.zeros((num_diffs, len(centers)))
    ctl_means = np.zeros(num_diffs)
    red_means = np.zeros(num_diffs)
    ctl_counts_rel = np.zeros((num_diffs, len(centers)))
    red_counts_rel = np.zeros((num_diffs, len(centers)))
    ctl_means_rel = np.zeros(num_diffs)
    red_means_rel = np.zeros(num_diffs)

    do_latents = "ctl_latent_corrs" in results
    if do_latents:
        ctl_counts_latents = np.zeros((num_diffs, len(centers)))
        red_counts_latents = np.zeros((num_diffs, len(centers)))
        ctl_means_latents = np.zeros(num_diffs)
        red_means_latents = np.zeros(num_diffs)
        ctl_counts_latents_rel = np.zeros((num_diffs, len(centers)))
        red_counts_latents_rel = np.zeros((num_diffs, len(centers)))
        ctl_means_latents_rel = np.zeros(num_diffs)
        red_means_latents_rel = np.zeros(num_diffs)

    for idiff in range(num_diffs):
        # Idx to pairs
        idx = results["ses_diffs"] == (idiff + 1)

        # Make histogram of correlations for spkmaps
        c_ctlcorrs = np.concatenate([c for i, c in enumerate(results["ctlcorrs"]) if idx[i]])
        c_redcorrs = np.concatenate([c for i, c in enumerate(results["redcorrs"]) if idx[i]])
        ctl_counts[idiff] = helpers.fractional_histogram(c_ctlcorrs, bins=bins)[0]
        red_counts[idiff] = helpers.fractional_histogram(c_redcorrs, bins=bins)[0]
        ctl_means[idiff] = np.mean(c_ctlcorrs)
        red_means[idiff] = np.mean(c_redcorrs)

        c_ctlcorrs = np.concatenate([c for i, c in enumerate(results["ctlcorrs_reliable"]) if idx[i]])
        c_redcorrs = np.concatenate([c for i, c in enumerate(results["redcorrs_reliable"]) if idx[i]])
        ctl_counts_rel[idiff] = helpers.fractional_histogram(c_ctlcorrs, bins=bins)[0]
        red_counts_rel[idiff] = helpers.fractional_histogram(c_redcorrs, bins=bins)[0]
        ctl_means_rel[idiff] = np.mean(c_ctlcorrs)
        red_means_rel[idiff] = np.mean(c_redcorrs)

        # Now do same thing for latents
        if do_latents:
            c_ctlcorr_latents = np.concatenate([c for i, c in enumerate(results["ctl_latent_corrs"]) if idx[i]])
            c_redcorr_latents = np.concatenate([c for i, c in enumerate(results["red_latent_corrs"]) if idx[i]])
            ctl_counts_latents[idiff] = helpers.fractional_histogram(c_ctlcorr_latents, bins=bins)[0]
            red_counts_latents[idiff] = helpers.fractional_histogram(c_redcorr_latents, bins=bins)[0]
            ctl_means_latents[idiff] = np.mean(c_ctlcorr_latents)
            red_means_latents[idiff] = np.mean(c_redcorr_latents)

            c_ctlcorr_latents = np.concatenate([c for i, c in enumerate(results["ctl_latent_corrs_reliable"]) if idx[i]])
            c_redcorr_latents = np.concatenate([c for i, c in enumerate(results["red_latent_corrs_reliable"]) if idx[i]])
            ctl_counts_latents_rel[idiff] = helpers.fractional_histogram(c_ctlcorr_latents, bins=bins)[0]
            red_counts_latents_rel[idiff] = helpers.fractional_histogram(c_redcorr_latents, bins=bins)[0]
            ctl_means_latents_rel[idiff] = np.mean(c_ctlcorr_latents)
            red_means_latents_rel[idiff] = np.mean(c_redcorr_latents)

    if do_latents:
        latent_dict = dict(
            ctl_counts_latents=ctl_counts_latents,
            red_counts_latents=red_counts_latents,
            ctl_means_latents=ctl_means_latents,
            red_means_latents=red_means_latents,
            ctl_counts_latents_rel=ctl_counts_latents_rel,
            red_counts_latents_rel=red_counts_latents_rel,
            ctl_means_latents_rel=ctl_means_latents_rel,
            red_means_latents_rel=red_means_latents_rel,
        )

    summary = dict(
        ctl_counts=ctl_counts,
        red_counts=red_counts,
        ctl_means=ctl_means,
        red_means=red_means,
        ctl_counts_rel=ctl_counts_rel,
        red_counts_rel=red_counts_rel,
        ctl_means_rel=ctl_means_rel,
        red_means_rel=red_means_rel,
        centers=centers,
        num_diffs=num_diffs,
    ) | (latent_dict if do_latents else {})
    return summary


def analyze_mouse(mouse_name, args):
    """Analyze data for each mouse and make a plot."""
    max_diff = args.max_diff
    relcor_cutoff = args.relcor_cutoff
    smooth = args.smooth
    track = tracking.tracker(mouse_name)
    pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=[1, 2, 3, 4], speedThreshold=1)

    process_data = True  # Assume we need to...
    if args.use_saved:
        # Check if exists
        if not pcm.check_temp_file(temp_file_name(mouse_name)):
            process_data = True
        else:
            process_data = False
            tracked_pair_pfcorr_results = pcm.load_temp_file(temp_file_name(mouse_name))
            if (
                tracked_pair_pfcorr_results["max_diff"] != max_diff
                or tracked_pair_pfcorr_results["relcor_cutoff"] != relcor_cutoff
                or tracked_pair_pfcorr_results["smooth"] != smooth
            ):
                print(f"Parameters changed for {mouse_name}, recomputing...")
                process_data = True

    if process_data:
        env_stats = pcm.env_stats()
        envs = list(env_stats.keys())
        first_session = [env_stats[env][0] for env in envs]
        idx_first_session = np.argsort(first_session)

        # use environment that was introduced second
        use_environment = envs[idx_first_session[1]]
        idx_ses = env_stats[use_environment][: min(8, len(env_stats[use_environment]))]

        if len(idx_ses) < 2:
            # Attempt to use first environment if not enough sessions in second
            use_environment = envs[idx_first_session[0]]
            idx_ses = env_stats[use_environment][: min(8, len(env_stats[use_environment]))]

        if len(idx_ses) < 2:
            print(f"Skipping {mouse_name} due to not enough sessions!")
            return

        output = make_corrs(pcm, use_environment, idx_ses, smooth=smooth, relcor_cutoff=relcor_cutoff, max_diff=max_diff)

        tracked_pair_pfcorr_results = (
            dict(
                mouse_name=mouse_name,
                max_diff=max_diff,
                relcor_cutoff=relcor_cutoff,
                smooth=smooth,
                use_environment=use_environment,
                idx_ses=idx_ses,
            )
            | output  # Include everything from output
        )

    if args.save_results:
        pcm.save_temp_file(tracked_pair_pfcorr_results, temp_file_name(mouse_name))

    summary = summarize_mouse(tracked_pair_pfcorr_results)

    fig, ax = plt.subplots(3, summary["num_diffs"], figsize=(12, 6), layout="constrained")
    for i in range(summary["num_diffs"]):
        # Plot histograms for all ROIs
        ax[0, i].plot(summary["centers"], summary["ctl_counts"][i], color="k", lw=1)
        ax[0, i].plot(summary["centers"], summary["red_counts"][i], color="r", lw=1)
        ax[0, i].axvline(summary["ctl_means"][i], color="k", lw=1, label="CTL Mean")
        ax[0, i].axvline(summary["red_means"][i], color="r", lw=1, label="RED Mean")
        ylim = ax[0, i].get_ylim()
        ax[0, i].text(summary["centers"][0], ylim[0] + 0.95 * (ylim[1] - ylim[0]), "All ROIs", ha="left", va="top")

        # Plot histograms for reliable ROIs
        ax[1, i].plot(summary["centers"], summary["ctl_counts_rel"][i], color="k", lw=1)
        ax[1, i].plot(summary["centers"], summary["red_counts_rel"][i], color="r", lw=1)
        ax[1, i].axvline(summary["ctl_means_rel"][i], color="k", lw=1, label="CTL Mean")
        ax[1, i].axvline(summary["red_means_rel"][i], color="r", lw=1, label="RED Mean")
        ylim = ax[1, i].get_ylim()
        ax[1, i].text(summary["centers"][0], ylim[0] + 0.95 * (ylim[1] - ylim[0]), "Reliable ROIs", ha="left", va="top")

        # Plot difference histograms
        ax[2, i].plot(summary["centers"], summary["red_counts"][i] - summary["ctl_counts"][i], color="k", lw=2, label="All")
        ax[2, i].plot(summary["centers"], summary["red_counts_rel"][i] - summary["ctl_counts_rel"][i], color="b", lw=2, label="Reliable")
        ax[2, i].legend(loc="lower left")

        ax[0, i].set_title(f"$\Delta$ Session: {i + 1}")
        ax[2, i].set_xlabel("Correlation")
        ax[0, i].set_ylabel("Fractional Counts")
        ax[1, i].set_ylabel("Fractional Counts")
        ax[2, i].set_ylabel("(Red - Ctl) Counts")
    fig.suptitle(f"{mouse_name} - Env:{use_environment}")
    pcm.saveFigure(fig, "tracked_pair_pfcorr", f"{mouse_name}_env{use_environment}")
    plt.close(fig)

    fig, ax = plt.subplots(3, summary["num_diffs"], figsize=(12, 6), layout="constrained")
    for i in range(summary["num_diffs"]):
        # Plot histograms for all ROIs
        ax[0, i].plot(summary["centers"], summary["ctl_counts_latents"][i], color="k", lw=1)
        ax[0, i].plot(summary["centers"], summary["red_counts_latents"][i], color="r", lw=1)
        ax[0, i].axvline(summary["ctl_means_latents"][i], color="k", lw=1, label="CTL Mean")
        ax[0, i].axvline(summary["red_means_latents"][i], color="r", lw=1, label="RED Mean")
        ylim = ax[0, i].get_ylim()
        ax[0, i].text(summary["centers"][0], ylim[0] + 0.95 * (ylim[1] - ylim[0]), "All ROIs", ha="left", va="top")

        # Plot histograms for reliable ROIs
        ax[1, i].plot(summary["centers"], summary["ctl_counts_latents_rel"][i], color="k", lw=1)
        ax[1, i].plot(summary["centers"], summary["red_counts_latents_rel"][i], color="r", lw=1)
        ax[1, i].axvline(summary["ctl_means_latents_rel"][i], color="k", lw=1, label="CTL Mean")
        ax[1, i].axvline(summary["red_means_latents_rel"][i], color="r", lw=1, label="RED Mean")
        ylim = ax[1, i].get_ylim()
        ax[1, i].text(summary["centers"][0], ylim[0] + 0.95 * (ylim[1] - ylim[0]), "Reliable ROIs", ha="left", va="top")

        # Plot difference histograms
        ax[2, i].plot(
            summary["centers"],
            summary["red_counts_latents"][i] - summary["ctl_counts_latents"][i],
            color="k",
            lw=2,
            label="All",
        )
        ax[2, i].plot(
            summary["centers"],
            summary["red_counts_latents_rel"][i] - summary["ctl_counts_latents_rel"][i],
            color="b",
            lw=2,
            label="Reliable",
        )
        ax[2, i].legend(loc="lower left")

        ax[0, i].set_title(f"$\Delta$ Session: {i + 1}")
        ax[2, i].set_xlabel("Correlation")
        ax[0, i].set_ylabel("Fractional Counts")
        ax[1, i].set_ylabel("Fractional Counts")
        ax[2, i].set_ylabel("(Red - Ctl) Counts")
    fig.suptitle(f"{mouse_name} - Env:{use_environment}")
    pcm.saveFigure(fig, "tracked_pair_latents", f"{mouse_name}_env{use_environment}")
    plt.close(fig)


def make_summary(use_mice, args):
    """Make summary plot across mice."""
    # Start by loading results file
    results = []
    for mouse_name in use_mice:
        track = tracking.tracker(mouse_name)
        pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=[1, 2, 3, 4], speedThreshold=1)
        if pcm.check_temp_file(temp_file_name(mouse_name)):
            tracked_pair_pfcorr_results = pcm.load_temp_file(temp_file_name(mouse_name))
            summary = summarize_mouse(tracked_pair_pfcorr_results)
            results.append(summary)
        else:
            print(f"Skipping {mouse_name} due to missing data!")

    # Compile all results together
    max_num_diffs = np.max([r["num_diffs"] for r in results])
    centers = results[0]["centers"]
    num_centers = len(centers)
    num_mice = len(results)
    ctl_counts = np.full((num_mice, max_num_diffs, num_centers), np.nan)
    red_counts = np.full((num_mice, max_num_diffs, num_centers), np.nan)
    ctl_means = np.full((num_mice, max_num_diffs), np.nan)
    red_means = np.full((num_mice, max_num_diffs), np.nan)
    ctl_counts_rel = np.full((num_mice, max_num_diffs, num_centers), np.nan)
    red_counts_rel = np.full((num_mice, max_num_diffs, num_centers), np.nan)
    ctl_means_rel = np.full((num_mice, max_num_diffs), np.nan)
    red_means_rel = np.full((num_mice, max_num_diffs), np.nan)
    for imouse, res in enumerate(results):
        for idiff in range(res["num_diffs"]):
            ctl_counts[imouse, idiff] = res["ctl_counts"][idiff]
            red_counts[imouse, idiff] = res["red_counts"][idiff]
            ctl_means[imouse, idiff] = res["ctl_means"][idiff]
            red_means[imouse, idiff] = res["red_means"][idiff]
            ctl_counts_rel[imouse, idiff] = res["ctl_counts_rel"][idiff]
            red_counts_rel[imouse, idiff] = res["red_counts_rel"][idiff]
            ctl_means_rel[imouse, idiff] = res["ctl_means_rel"][idiff]
            red_means_rel[imouse, idiff] = res["red_means_rel"][idiff]

    fig, ax = plt.subplots(2, max_num_diffs + 2, figsize=(16, 6), layout="constrained")
    for idiff in range(max_num_diffs):
        helpers.errorPlot(centers, ctl_counts[:, idiff], axis=0, ax=ax[0, idiff], color="k", lw=1, alpha=0.1)
        helpers.errorPlot(centers, red_counts[:, idiff], axis=0, ax=ax[0, idiff], color="r", lw=1, alpha=0.1)
        ax[0, idiff].axvline(np.nanmean(ctl_means[:, idiff]), color="k", lw=1, label="CTL Mean")
        ax[0, idiff].axvline(np.nanmean(red_means[:, idiff]), color="r", lw=1, label="RED Mean")
        ylim = ax[0, idiff].get_ylim()
        ax[0, idiff].text(centers[0], ylim[0] + 0.95 * (ylim[1] - ylim[0]), "All ROIs", ha="left", va="top")

        helpers.errorPlot(centers, ctl_counts_rel[:, idiff], axis=0, ax=ax[1, idiff], color="k", lw=1, alpha=0.1)
        helpers.errorPlot(centers, red_counts_rel[:, idiff], axis=0, ax=ax[1, idiff], color="r", lw=1, alpha=0.1)
        ax[1, idiff].axvline(np.nanmean(ctl_means_rel[:, idiff]), color="k", lw=1, label="CTL Mean")
        ax[1, idiff].axvline(np.nanmean(red_means_rel[:, idiff]), color="r", lw=1, label="RED Mean")
        ylim = ax[1, idiff].get_ylim()
        ax[1, idiff].text(centers[0], ylim[0] + 0.95 * (ylim[1] - ylim[0]), "Reliable ROIs", ha="left", va="top")

        ax[1, idiff].set_xlabel("Correlation")
        ax[0, idiff].set_ylabel("Fractional Counts")
        ax[1, idiff].set_ylabel("Fractional Counts")
        ax[0, idiff].set_title(f"$\Delta$ Session: {idiff + 1}")

    ax[0, -2].plot(range(1, max_num_diffs + 1), ctl_means.T, color="k", lw=1)
    ax[0, -2].plot(range(1, max_num_diffs + 1), red_means.T, color="r", lw=1)
    ax[1, -2].plot(range(1, max_num_diffs + 1), ctl_means_rel.T, color="k", lw=1)
    ax[1, -2].plot(range(1, max_num_diffs + 1), red_means_rel.T, color="r", lw=1)
    ax[0, -2].set_xticks(range(1, max_num_diffs + 1))
    ax[1, -2].set_xticks(range(1, max_num_diffs + 1))
    ax[0, -2].set_xlabel("$\Delta$ Session")
    ax[1, -2].set_xlabel("$\Delta$ Session")
    ax[0, -2].set_ylabel("Mean Correlation")
    ax[1, -2].set_ylabel("Mean Correlation")
    ax[0, -2].legend(loc="lower left")
    ax[1, -2].legend(loc="lower left")
    ax[0, -2].set_ylim(0, None)
    ax[1, -2].set_ylim(0, None)
    ax[0, -2].set_title("Average across mice")

    ax[0, -1].plot(range(1, max_num_diffs + 1), (ctl_means - red_means).T, color="k", lw=1)
    ax[1, -1].plot(range(1, max_num_diffs + 1), (ctl_means_rel - red_means_rel).T, color="k", lw=1)
    ax[0, -1].set_xticks(range(1, max_num_diffs + 1))
    ax[1, -1].set_xticks(range(1, max_num_diffs + 1))
    ax[0, -1].set_xlabel("$\Delta$ Session")
    ax[1, -1].set_xlabel("$\Delta$ Session")
    ax[0, -1].set_ylabel("$\Delta$ Correlation")
    ax[1, -1].set_ylabel("$\Delta$  Correlation")
    ax[0, -1].set_title("Difference all mice")

    track = tracking.tracker(use_mice[0])
    pcm = analysis.placeCellMultiSession(track, autoload=False)
    pcm.saveFigure(fig, "tracked_pair_pfcorr", "summary_across_mice")

    plt.show()


if __name__ == "__main__":
    args = handle_args()
    use_mice = get_mice()

    if args.analyze_mice:
        for mouse_name in use_mice:
            analyze_mouse(mouse_name, args)

    if args.make_summary:
        make_summary(use_mice, args)
