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
    for idx in tqdm(idx_pairs, desc="computing correlations", leave=False):
        spkmaps, _, relcor, _, _, idx_red, _ = pcm.get_spkmaps(
            envnum,
            trials="full",
            pop_nan=True,
            smooth=smooth,
            tracked=True,
            average=True,
            idx_ses=idx,
        )
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
    return ctlcorrs, redcorrs, ctlcorrs_reliable, redcorrs, ses_diffs


if __name__ == "__main__":
    args = handle_args()
    use_mice = get_mice()
    max_diff = args.max_diff
    relcor_cutoff = args.relcor_cutoff
    smooth = args.smooth
    for imouse, mouse_name in enumerate(use_mice):
        track = tracking.tracker(mouse_name)
        pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=[1, 2, 3, 4], speedThreshold=1)

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
                else:
                    # Retrieve variables for plotting
                    mouse_name = tracked_pair_pfcorr_results["mouse_name"]
                    ctlcorrs = tracked_pair_pfcorr_results["ctlcorrs"]
                    redcorrs = tracked_pair_pfcorr_results["redcorrs"]
                    ctlcorrs_reliable = tracked_pair_pfcorr_results["ctlcorrs_reliable"]
                    redcorrs_reliable = tracked_pair_pfcorr_results["redcorrs_reliable"]
                    ses_diffs = tracked_pair_pfcorr_results["ses_diffs"]
                    max_diff = tracked_pair_pfcorr_results["max_diff"]
                    relcor_cutoff = tracked_pair_pfcorr_results["relcor_cutoff"]
                    smooth = tracked_pair_pfcorr_results["smooth"]
                    use_environment = tracked_pair_pfcorr_results["use_environment"]
                    idx_ses = tracked_pair_pfcorr_results["idx_ses"]

        if process_data:
            tracked_pair_pfcorr_results = pcm.load_temp_file(f"{mouse_name}_tracked_pair_pfcorr.pkl")
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
                continue

            ctlcorrs, redcorrs, ctlcorrs_reliable, redcorrs_reliable, ses_diffs = make_corrs(
                pcm,
                use_environment,
                idx_ses,
                smooth=smooth,
                relcor_cutoff=relcor_cutoff,
                max_diff=max_diff,
            )

            tracked_pair_pfcorr_results = dict(
                mouse_name=mouse_name,
                ctlcorrs=ctlcorrs,
                redcorrs=redcorrs,
                ctlcorrs_reliable=ctlcorrs_reliable,
                redcorrs_reliable=redcorrs_reliable,
                ses_diffs=ses_diffs,
                max_diff=max_diff,
                relcor_cutoff=relcor_cutoff,
                smooth=smooth,
                use_environment=use_environment,
                idx_ses=idx_ses,
            )

        if args.save_results:
            pcm.save_temp_file(tracked_pair_pfcorr_results, temp_file_name(mouse_name))

        bins = np.linspace(-1, 1, 11)
        centers = helpers.edge2center(bins)

        num_diffs = len(np.unique(ses_diffs))
        ctl_counts = np.zeros((num_diffs, len(centers)))
        red_counts = np.zeros((num_diffs, len(centers)))
        ctl_means = np.zeros(num_diffs)
        red_means = np.zeros(num_diffs)
        ctl_counts_rel = np.zeros((num_diffs, len(centers)))
        red_counts_rel = np.zeros((num_diffs, len(centers)))
        ctl_means_rel = np.zeros(num_diffs)
        red_means_rel = np.zeros(num_diffs)
        for idiff in range(num_diffs):
            idx = ses_diffs == (idiff + 1)
            c_ctlcorrs = np.concatenate([c for i, c in enumerate(ctlcorrs) if idx[i]])
            c_redcorrs = np.concatenate([c for i, c in enumerate(redcorrs) if idx[i]])
            ctl_counts[idiff] = helpers.fractional_histogram(c_ctlcorrs, bins=bins)[0]
            red_counts[idiff] = helpers.fractional_histogram(c_redcorrs, bins=bins)[0]
            ctl_means[idiff] = np.mean(c_ctlcorrs)
            red_means[idiff] = np.mean(c_redcorrs)

            c_ctlcorrs = np.concatenate([c for i, c in enumerate(ctlcorrs_reliable) if idx[i]])
            c_redcorrs = np.concatenate([c for i, c in enumerate(redcorrs_reliable) if idx[i]])
            ctl_counts_rel[idiff] = helpers.fractional_histogram(c_ctlcorrs, bins=bins)[0]
            red_counts_rel[idiff] = helpers.fractional_histogram(c_redcorrs, bins=bins)[0]
            ctl_means_rel[idiff] = np.mean(c_ctlcorrs)
            red_means_rel[idiff] = np.mean(c_redcorrs)

        fig, ax = plt.subplots(3, num_diffs, figsize=(12, 6), layout="constrained")
        for i in range(num_diffs):
            # Plot histograms for all ROIs
            ax[0, i].plot(centers, ctl_counts[i], color="k", lw=1)
            ax[0, i].plot(centers, red_counts[i], color="r", lw=1)
            ax[0, i].axvline(ctl_means[i], color="k", lw=1, label="CTL Mean")
            ax[0, i].axvline(red_means[i], color="r", lw=1, label="RED Mean")
            ylim = ax[0, i].get_ylim()
            ax[0, i].text(centers[0], ylim[0] + 0.95 * (ylim[1] - ylim[0]), "All ROIs", ha="left", va="top")

            # Plot histograms for reliable ROIs
            ax[1, i].plot(centers, ctl_counts_rel[i], color="k", lw=1)
            ax[1, i].plot(centers, red_counts_rel[i], color="r", lw=1)
            ax[1, i].axvline(ctl_means_rel[i], color="k", lw=1, label="CTL Mean")
            ax[1, i].axvline(red_means_rel[i], color="r", lw=1, label="RED Mean")
            ylim = ax[1, i].get_ylim()
            ax[1, i].text(centers[0], ylim[0] + 0.95 * (ylim[1] - ylim[0]), "Reliable ROIs", ha="left", va="top")

            # Plot difference histograms
            ax[2, i].plot(centers, red_counts[i] - ctl_counts[i], color="k", lw=2, label="All")
            ax[2, i].plot(centers, red_counts_rel[i] - ctl_counts_rel[i], color="b", lw=2, label="Reliable")
            ax[2, i].legend(loc="lower left")

            ax[0, i].set_title(f"$\Delta$ Session: {i + 1}")
            ax[2, i].set_xlabel("Correlation")
            ax[0, i].set_ylabel("Fractional Counts")
            ax[1, i].set_ylabel("Fractional Counts")
            ax[2, i].set_ylabel("(Red - Ctl) Counts")
        fig.suptitle(f"{mouse_name} - Env:{use_environment}")
        pcm.saveFigure(fig, "tracked_pair_pfcorr", f"{mouse_name}_env{use_environment}")
