import numpy as np
import torch
import matplotlib.pyplot as plt

from _old_vrAnalysis import analysis
from _old_vrAnalysis import database
from _old_vrAnalysis import tracking

sessiondb = database.vrDatabase("vrSessions")
mousedb = database.vrDatabase("vrMice")


def get_idx_quantile(arr, num_quantiles=6):
    # Get index of quantile for each element in arr along the 0th axis
    quantiles = np.linspace(0, 1, num_quantiles + 1)
    idx_quantile = np.digitize(arr, quantiles, right=True)
    idx_quantile = np.maximum(idx_quantile - 1, 0)
    return idx_quantile


def get_avg_quantile(arr, idx_quantile, num_quantiles=6):
    # Get the average of the array by quantile
    avg_quantile = [np.sum(arr * (idx_quantile == iq), axis=0) / np.sum(idx_quantile == iq, axis=0) for iq in range(num_quantiles)]
    return np.stack(avg_quantile)


if __name__ == "__main__":

    mouse_name = "ATL022"
    track = tracking.tracker(mouse_name)
    pcm = analysis.placeCellMultiSession(track, autoload=False)
    ises = 16
    pcss = analysis.placeCellSingleSession(pcm.pcss[ises].vrexp, keep_planes=[1, 2, 3, 4], autoload=False)
    split_params = dict(total_folds=2, train_folds=1)
    pcss.define_train_test_split(**split_params)
    pcss.load_data(new_split=False)

    train_spkmaps = pcss.get_spkmap(average=True, smooth=0.1, trials="train")
    test_spkmaps = pcss.get_spkmap(average=False, smooth=0.1, trials="test")
    spks = pcss.prepare_spks()

    idx_nan = np.any(
        np.stack([np.any(np.isnan(t), axis=0) for t in train_spkmaps] + [np.any(np.isnan(t), axis=(0, 1)) for t in test_spkmaps]), axis=0
    )
    train_spkmaps = [t[:, ~idx_nan] for t in train_spkmaps]
    test_spkmaps = [t[:, :, ~idx_nan] for t in test_spkmaps]
    max_rate = np.max(
        np.concatenate([np.max(t, axis=1, keepdims=True) for t in train_spkmaps] + [np.max(spks, axis=0).reshape(-1, 1)], axis=1), axis=1
    )

    norm_train_spkmaps = [t / max_rate.reshape(-1, 1) for t in train_spkmaps]
    norm_test_spkmaps = [t / max_rate.reshape(-1, 1, 1) for t in test_spkmaps]
    idx_include = max_rate > 15

    norm_train_spkmaps = [t[idx_include] for t in norm_train_spkmaps]
    norm_test_spkmaps = [t[idx_include] for t in norm_test_spkmaps]

    print([t.shape for t in norm_train_spkmaps], [t.shape for t in norm_test_spkmaps], max_rate.shape)

    # For each trial, position, calculate quantiles of expected (from train place field)
    # Calculate the average expected by quantile
    # Calculate the average observed by quantile
    num_quantiles = 11
    idx_quantile = [get_idx_quantile(t, num_quantiles=num_quantiles) for t in norm_train_spkmaps]
    avg_quantile = [get_avg_quantile(t, iq, num_quantiles=num_quantiles) for t, iq in zip(norm_train_spkmaps, idx_quantile)]
    avg_observed = [get_avg_quantile(t, iq[:, None, :], num_quantiles=num_quantiles) for t, iq in zip(norm_test_spkmaps, idx_quantile)]
    avg_quantile_bc = [np.tile(aq[:, None, :], (1, t.shape[1], 1)) for t, aq in zip(norm_test_spkmaps, avg_quantile)]

    fig, ax = plt.subplots(1, len(avg_quantile), figsize=(5 * len(avg_quantile), 5), layout="constrained")
    for i, (aq, ao) in enumerate(zip(avg_quantile_bc, avg_observed)):
        num_trials = ao.shape[1]
        num_positions = ao.shape[2]
        xvals = aq.reshape(num_quantiles, num_trials * num_positions)
        yvals = ao.reshape(num_quantiles, num_trials * num_positions)
        cvals = np.arange(num_trials)[None, :, None]
        cvals = np.tile(cvals, [num_quantiles, 1, num_positions]).reshape(num_quantiles, num_trials * num_positions)
        ax[i].scatter(xvals, yvals, 10, cvals)
    plt.show()
