from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import os, sys

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from vrAnalysis import analysis
from vrAnalysis import helpers
from vrAnalysis import database
from vrAnalysis import tracking


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
    for idx in tqdm(idx_pairs, desc="computing correlations", leave=False):
        spkmaps, _, relcor, _, _, idx_red, _ = pcm.get_spkmaps(envnum, trials="full", pop_nan=True, smooth=smooth, average=True, idx_ses=idx)
        any_rel = np.any(np.stack(relcor) > relcor_cutoff, axis=0)
        any_red = np.any(np.stack(idx_red), axis=0)
        ctlmaps = [s[~any_red & any_rel] for s in spkmaps]
        redmaps = [s[any_red & any_rel] for s in spkmaps]
        ctlcorrs.append(helpers.vectorCorrelation(ctlmaps[0], ctlmaps[1], axis=1))
        redcorrs.append(helpers.vectorCorrelation(redmaps[0], redmaps[1], axis=1))
    return ctlcorrs, redcorrs, ses_diffs


if __name__ == "__main__":
    use_mice = get_mice()
    max_diff = 4
    relcor_cutoff = 0.5
    smooth = 2
    for imouse, mouse_name in enumerate(use_mice):
        track = tracking.tracker(mouse_name)
        pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=[1, 2, 3, 4], speedThreshold=1)
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

        ctlcorrs, redcorrs, ses_diffs = make_corrs(pcm, use_environment, idx_ses, smooth=smooth, relcor_cutoff=relcor_cutoff, max_diff=max_diff)

        bins = np.linspace(-1, 1, 11)
        centers = helpers.edge2center(bins)

        num_diffs = len(np.unique(ses_diffs))
        ctl_counts = np.zeros((num_diffs, len(centers)))
        red_counts = np.zeros((num_diffs, len(centers)))
        for idiff in range(num_diffs):
            idx = ses_diffs == (idiff + 1)
            c_ctlcorrs = np.concatenate([c for i, c in enumerate(ctlcorrs) if idx[i]])
            c_redcorrs = np.concatenate([c for i, c in enumerate(redcorrs) if idx[i]])
            ctl_counts[idiff] = helpers.fractional_histogram(c_ctlcorrs, bins=bins)[0]
            red_counts[idiff] = helpers.fractional_histogram(c_redcorrs, bins=bins)[0]

        fig, ax = plt.subplots(2, num_diffs, figsize=(12, 4), layout="constrained")
        for i in range(num_diffs):
            ax[0, i].plot(centers, ctl_counts[i], color="k", lw=1)
            ax[0, i].plot(centers, red_counts[i], color="r", lw=1)
            ax[0, i].set_title(f"$\Delta$ Session: {i + 1}")
            ax[1, i].plot(centers, red_counts[i] - ctl_counts[i], color="k", lw=2)
            ax[1, i].set_xlabel("Correlation")
            ax[0, i].set_ylabel("Fractional Counts")
            ax[1, i].set_ylabel("(Red - Ctl) Counts")
        fig.suptitle(f"{mouse_name} - Env:{use_environment}")
        pcm.saveFigure(fig, "tracked_pair_pfcorr", f"{mouse_name}_env{use_environment}")
