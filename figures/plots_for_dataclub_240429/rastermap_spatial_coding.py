import os, sys
import numpy as np
import torch
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from rastermap import Rastermap

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from _old_vrAnalysis import analysis
from _old_vrAnalysis import tracking
from _old_vrAnalysis import helpers
from dimilibi import RidgeRegression, ReducedRankRegression, Population, measure_r2, measure_rms

from scripts.dimilibi.rrr_state_optimization import rrr_state_tempfile_name

if __name__ == "__main__":
    mouse_name = "ATL027"
    population_name = "fast"
    track = tracking.tracker(mouse_name)  # get tracker object for mouse
    pcm = analysis.placeCellMultiSession(track, autoload=False)  # open up place cell multi session analysis object (don't autoload!!!)

    ises = 10  # 10 ---- note I'm usually using session 10
    rrr_file_name = rrr_state_tempfile_name(pcm.pcss[ises].vrexp, population_name=population_name)
    results = pcm.pcss[ises].load_temp_file(rrr_file_name)

    vss = analysis.VarianceStructure(pcm.pcss[ises].vrexp, autoload=False)
    spks = vss.prepare_spks()

    pop = Population(spks.T)
    source_idx, target_idx = pop.cell_split_indices

    train_source, train_target = pop.get_split_data(0)

    spkmaps = vss.get_spkmap(average=True, smooth=0.1)
    tspkmaps = vss.get_spkmap(average=False, smooth=0.1)
    frame_position, idx_valid, frame_speed = vss.get_frame_behavior(use_average=True, return_speed=True)
    spks_prediction = vss.generate_spks_prediction(spks, spkmaps, frame_position, idx_valid, background_value=np.nan)
    speed_threshold = 1

    # Build a transfer function that is 1 above the threshold and smoothly transitions to 0 below the threshold
    all_speed = np.zeros(frame_speed.shape[1])
    for i in range(frame_speed.shape[0]):
        idx_not_nan = ~np.isnan(frame_speed[i])
        all_speed[idx_not_nan] = frame_speed[i, idx_not_nan]
    transfer_function = 1 / (1 + np.exp(-10 * (all_speed - speed_threshold)))

    source_spks = torch.tensor(spks[:, source_idx])
    target_spks = spks[:, target_idx]
    target_spkmaps = [spkmap[target_idx] for spkmap in spkmaps]
    target_prediction = spks_prediction[:, target_idx]

    # Now build direct RRR model to compare
    best_params = results["params_direct"]
    model = ReducedRankRegression(alpha=best_params["alpha"], fit_intercept=True).fit(train_source.T, train_target.T)
    rrr_prediction = model.predict(source_spks, nonnegative=True)

    spks_prediction = spks_prediction * transfer_function.reshape(-1, 1)
    idx_reliable = vss.get_reliable(cutoffs=(0.4, 0.7))
    idx_unreliable = vss.get_reliable(cutoffs=None, maxcutoffs=(0.1, 0.4))
    target_idx_reliable = [ir[target_idx] for ir in idx_reliable]
    target_idx_unreliable = [ir[target_idx] for ir in idx_unreliable]

    relcor = vss.get_reliability_values()[1]
    target_relcor = [rc[target_idx] for rc in relcor]
    target_any_reliable = np.any(np.stack(target_idx_reliable), axis=0)
    target_unreliable = ~target_any_reliable  # np.all(np.stack(target_idx_unreliable), axis=0)
    target_best_environment = np.argmax(np.stack(target_relcor), axis=0)
    distedges = vss.distedges
    distcenters = vss.distcenters

    num_reliable = np.sum(target_any_reliable)
    num_unreliable = np.sum(target_unreliable)
    num_to_use = int(min(num_reliable, num_unreliable) * 1)
    idx_random_reliable = np.sort(np.random.choice(np.sum(target_any_reliable), num_to_use, replace=False))
    idx_random_unreliable = np.random.choice(np.sum(target_unreliable), num_to_use, replace=False)

    spks_reliable = target_spks[:, target_any_reliable][:, idx_random_reliable]
    spks_pred_reliable = target_prediction[:, target_any_reliable][:, idx_random_reliable]
    rrr_pred_reliable = rrr_prediction[:, target_any_reliable][:, idx_random_reliable]
    spks_unreliable = target_spks[:, target_unreliable][:, idx_random_unreliable]
    spks_pred_unreliable = target_prediction[:, target_unreliable][:, idx_random_unreliable]
    rrr_pred_unreliable = rrr_prediction[:, target_unreliable][:, idx_random_unreliable]

    spkmaps_reliable = [spkmap[target_any_reliable][idx_random_reliable] for spkmap in target_spkmaps]
    spkmaps_unreliable = [spkmap[target_unreliable][idx_random_unreliable] for spkmap in target_spkmaps]

    environments = np.unique(target_best_environment)
    best_environment_reliable = target_best_environment[target_any_reliable][idx_random_reliable]
    sorted_reliable = []
    sorted_reliable_prediction = []
    sorted_reliable_rrrpred = []
    for ienv in environments:
        idx = best_environment_reliable == ienv
        pfidx = vss.get_place_field(target_spkmaps[ienv][target_any_reliable][idx_random_reliable][idx])[1]
        sorted_reliable.append(spks_reliable[:, idx][:, pfidx])
        sorted_reliable_prediction.append(spks_pred_reliable[:, idx][:, pfidx])
        sorted_reliable_rrrpred.append(rrr_pred_reliable[:, idx][:, pfidx])

    sorted_reliable = np.concatenate(sorted_reliable, axis=1)
    sorted_reliable_prediction = np.concatenate(sorted_reliable_prediction, axis=1)
    sorted_reliable_rrrpred = np.concatenate(sorted_reliable_rrrpred, axis=1)

    # Z scores
    zsorted_reliable = sp.stats.zscore(sorted_reliable, axis=0)
    zsorted_reliable_prediction = sp.stats.zscore(sorted_reliable_prediction, axis=0)
    zsorted_reliable_rrrpred = sp.stats.zscore(sorted_reliable_rrrpred, axis=0)
    zspks_unreliable = sp.stats.zscore(spks_unreliable, axis=0)
    zspks_pred_unreliable = sp.stats.zscore(spks_pred_unreliable, axis=0)
    zrrr_pred_unreliable = sp.stats.zscore(rrr_pred_unreliable, axis=0)

    # fit rastermap to unreliable ROIs
    rmap = Rastermap(n_PCs=200, n_clusters=10, locality=0, time_lag_window=0).fit(zspks_unreliable.T)
    isort = rmap.isort

    spks_to_plot = np.concatenate((sorted_reliable, spks_unreliable[:, isort]), axis=1)
    pred_to_plot = np.concatenate((sorted_reliable_prediction, spks_pred_unreliable[:, isort]), axis=1)
    rrr_to_plot = np.concatenate((sorted_reliable_rrrpred, rrr_pred_unreliable[:, isort]), axis=1)

    spks_to_plot = np.fliplr(spks_to_plot)
    pred_to_plot = np.fliplr(pred_to_plot)
    rrr_to_plot = np.fliplr(rrr_to_plot)

    vmin = 0
    vmax = 4
    interpolation = "sinc"

    plt.rcParams.update({"font.size": 24})

    include_rrr_prediction = False
    wide_view = True
    num_rows = 4 if include_rrr_prediction else 3
    width_scale = 3 if wide_view else 2.5

    fs = 1 / 6
    xlim = (2249.5, 3680.5) if wide_view else (3074.5, 3568.5)
    xlim = fs * np.array(xlim)
    extent = [0, spks_to_plot.shape[0] * fs, 0, spks_to_plot.shape[1]]

    # plot
    figdim = 6.5
    fig, ax = plt.subplots(num_rows, 1, figsize=(figdim, figdim), layout="constrained", sharex=True)

    im = ax[0].imshow(spks_to_plot.T, extent=extent, vmin=vmin, vmax=vmax, cmap="gray_r", aspect="auto", interpolation=interpolation)
    ax[0].set_yticks([])
    ax[0].set_ylabel("ROIs")
    ax[0].text(xlim[0] + 0.02 * np.diff(xlim), 0.93 * extent[3], "True Activity", ha="left", va="top")

    ax[1].imshow(pred_to_plot.T, extent=extent, vmin=vmin, vmax=vmax, cmap="gray_r", aspect="auto", interpolation=interpolation)
    ax[1].set_yticks([])
    ax[1].set_ylabel("ROIs")
    ax[1].text(xlim[0] + 0.02 * np.diff(xlim), 0.93 * extent[3], "Place Field\nPrediction", ha="left", va="top")

    if include_rrr_prediction:
        ax[2].imshow(rrr_to_plot.T, extent=extent, vmin=vmin, vmax=vmax, cmap="gray_r", aspect="auto", interpolation=interpolation)
        ax[2].set_ylabel("ROIs")

    inset_position = [0.725, 0.85, 0.2, 0.075]
    inset = ax[1].inset_axes(inset_position)
    inset.xaxis.set_ticks_position("bottom")
    cb0 = fig.colorbar(im, cax=inset, orientation="horizontal", ticks=[])
    ax[1].text(
        inset_position[0] + inset_position[2] / 2,
        inset_position[1] - 0.07,
        "Activity ($\sigma$)",
        ha="center",
        va="top",
        transform=ax[1].transAxes,
    )
    # cb0.set_label("Activity ($\sigma$)", loc="center", y=-125)
    tick_xoffset = inset_position[2] * 0.05
    ax[1].text(
        inset_position[0] - tick_xoffset,
        inset_position[1] + inset_position[3] / 4,
        vmin,
        ha="right",
        va="center",
        transform=ax[1].transAxes,
    )
    ax[1].text(
        inset_position[0] + inset_position[2] + tick_xoffset,
        inset_position[1] + inset_position[3] / 4,
        vmax,
        ha="left",
        va="center",
        transform=ax[1].transAxes,
    )

    use_offsets = True
    if use_offsets:
        offset = 200
    cols = "krb"
    for ii, fp in enumerate(frame_position):
        ax[-1].plot(fs * np.arange(len(fp)), fp + ii * offset, c=cols[ii], label=ii)
    ax[-1].set_xlabel("Time (seconds)", labelpad=-10)
    ax[-1].set_ylabel("Pos (cm)")
    ax[-1].set_xlim(xlim)
    ax[-1].set_ylim(0, 3 * offset)
    xticks = np.array([np.ceil(xlim[0]), np.floor(xlim[1])])
    ax[-1].set_xticks(xticks, labels=[f"{xt:.0f}" for xt in xticks - xticks[0]])
    ax[-1].set_yticks([])
    ax[-1].plot((xlim[0] + 0.08 * np.diff(xlim)) * np.array([1, 1]), [350, 450], c="k", lw=2, linestyle="-")
    ax[-1].text(xlim[0] + 0.07 * np.diff(xlim), 400, "100 cm", ha="right", va="center", rotation=90, fontsize=18)
    # ax[-1].legend(loc="upper right")
    legend_x = xlim[0] + np.diff(xlim) * 0.985
    legend_y = 3 * offset - 0.175 * offset
    legend_offset = 0.5 * offset
    for ii in range(len(frame_position)):
        ax[-1].text(legend_x, legend_y - ii * legend_offset, f"Env {ii}", ha="right", va="top", color=cols[ii])

    if with_show:
        plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = pcm.pcss[ises].saveDirectory("example_plots")
        save_name = f"{pcm.pcss[ises].vrexp.sessionPrint('_')}_rastermap_spatial_coding_pfprediction"
        save_name += "_rrr" if include_rrr_prediction else ""
        save_name += "_wide" if wide_view else ""
        save_path = save_directory / save_name
        helpers.save_figure(fig, save_path)

    # ------------------------------------------------
    # ------------------------------------------------
    # ----- comparing RRR predictions........... -----
    # ------------------------------------------------
    # ------------------------------------------------

    include_rrr_prediction = True
    wide_view = True
    num_rows = 4 if include_rrr_prediction else 3
    width_scale = 3 if wide_view else 2.5

    fs = 1 / 6
    xlim = (2249.5, 3680.5) if wide_view else (3074.5, 3568.5)
    xlim = fs * np.array(xlim)
    extent = [0, spks_to_plot.shape[0] * fs, 0, spks_to_plot.shape[1]]

    vmin = 0
    vmax = 1

    # plot
    figdim = 3
    fig, ax = plt.subplots(num_rows, 1, figsize=(num_rows * figdim, 3 * figdim), layout="constrained", sharex=True)

    im = ax[0].imshow(spks_to_plot.T, extent=extent, vmin=vmin, vmax=vmax, cmap="gray_r", aspect="auto", interpolation=interpolation)
    ax[0].set_yticks([])
    ax[0].set_ylabel("ROIs\nRel -- Unrel")
    ax[0].text(xlim[0] + 0.02 * np.diff(xlim), 0.85 * extent[3], "True Activity", ha="left", va="center")

    ax[1].imshow(pred_to_plot.T, extent=extent, vmin=vmin, vmax=vmax, cmap="gray_r", aspect="auto", interpolation=interpolation)
    ax[1].set_yticks([])
    ax[1].set_ylabel("ROIs\nRel -- Unrel")
    ax[1].text(xlim[0] + 0.02 * np.diff(xlim), 0.85 * extent[3], "Prediction from Place Field", ha="left", va="center")

    if include_rrr_prediction:
        ax[2].imshow(rrr_to_plot.T, extent=extent, vmin=vmin, vmax=vmax, cmap="gray_r", aspect="auto", interpolation=interpolation)
        ax[2].set_ylabel("ROIs\nRel -- Unrel")

    inset = ax[1].inset_axes([0.75, 0.85, 0.2, 0.075])
    inset.xaxis.set_ticks_position("bottom")
    cb0 = fig.colorbar(im, cax=inset, orientation="horizontal", ticks=[vmin, vmax])
    cb0.set_label("Activity ($\sigma$)", loc="center", y=10)

    cols = "krb"
    for ii, fp in enumerate(frame_position):
        ax[-1].plot(fs * np.arange(len(fp)), fp, c=cols[ii], label=ii)
    ax[-1].set_xlabel("Time (seconds)")
    ax[-1].set_ylabel("Position (cm)")
    ax[-1].legend(loc="lower right")
    ax[-1].set_xlim(xlim)
    xticks = np.array([np.ceil(xlim[0]), np.floor(xlim[1])])
    ax[-1].set_xticks(xticks, labels=[f"{xt:.0f}" for xt in xticks - xticks[0]])

    # plt.show()

    # ------------------------------------------------
    # ------------------------------------------------
    # ----- example ROI rastermap spatial coding -----
    # ------------------------------------------------
    # ------------------------------------------------

    # test r2 for individual neurons
    target_prediction_nonnan = target_prediction.copy().astype(float)
    target_prediction_nonnan[np.isnan(target_prediction_nonnan)] = 0.0
    idx_compare = np.any(idx_valid, axis=0)
    r2_pred = measure_r2(target_prediction_nonnan[idx_compare], target_spks[idx_compare], reduce=None)
    r2_rrr = measure_r2(rrr_prediction[idx_compare], target_spks[idx_compare], reduce=None)
    rms_pred = measure_rms(target_prediction[idx_compare], target_spks[idx_compare], reduce=None)
    rms_rrr = measure_rms(rrr_prediction[idx_compare], target_spks[idx_compare], reduce=None)
    target_all_unreliable = np.all(np.stack(target_idx_unreliable), axis=0)
    r2_ratio = r2_rrr / r2_pred

    # predictable and reliable
    idx_very_reliable = np.any(np.stack(vss.get_reliable(cutoffs=(0.7, 0.9))), axis=0)[target_idx]
    idx_pred_reliable = target_any_reliable & np.array(r2_rrr > np.percentile(r2_rrr, 60))
    idx_bad_unreliable = target_all_unreliable & np.array(r2_rrr < np.percentile(r2_rrr, 20))
    idx_ratio_r2 = np.array(r2_ratio > np.percentile(r2_ratio, 90))
    idx_high_rrr = np.array(r2_rrr > np.percentile(r2_rrr, 90))
    idx_low_rms = np.array(rms_rrr < np.percentile(rms_rrr, 10))

    # set rng for consistency
    # np.random.seed(0)
    # good choices: [3625]
    cellidx = np.random.choice(np.where(idx_low_rms & idx_very_reliable)[0])
    original_idx = target_idx[cellidx].item()
    envidx = target_best_environment[cellidx]
    print("Chosen ROI:", cellidx, "Original index:", original_idx)

    # good choices: [4620]
    unrel_cell_idx = np.random.choice(np.where(idx_very_reliable)[0])  # idx_bad_unreliable)[0])
    unrel_original_idx = target_idx[unrel_cell_idx].item()
    unrel_envidx = target_best_environment[unrel_cell_idx]
    print("Chosen Unreliable ROI:", unrel_cell_idx, "Original index:", unrel_original_idx)

    width = 10
    travs, pred_travs = vss.get_traversals(
        target_spks,
        target_prediction,
        target_spkmaps,
        frame_position,
        envidx,
        cellidx,
        width=width,
        # fill_nan=True,
    )
    _, rrr_travs = vss.get_traversals(
        target_spks,
        rrr_prediction,
        target_spkmaps,
        frame_position,
        envidx,
        cellidx,
        width=width,
        # fill_nan=True,
    )

    unrel_travs, unrel_pred_travs = vss.get_traversals(
        target_spks,
        target_prediction,
        target_spkmaps,
        frame_position,
        unrel_envidx,
        unrel_cell_idx,
        width=width,
        # fill_nan=True,
    )
    _, unrel_rrr_travs = vss.get_traversals(
        target_spks,
        rrr_prediction,
        target_spkmaps,
        frame_position,
        unrel_envidx,
        unrel_cell_idx,
        width=width,
        # fill_nan=True,
    )

    error_pred = np.abs(travs - pred_travs)
    error_rrr = np.abs(travs - rrr_travs)
    rms_error_pred = np.sqrt(np.mean(error_pred**2, axis=0))
    rms_error_rrr = np.sqrt(np.mean(error_rrr**2, axis=0))

    unrel_error_pred = np.abs(unrel_travs - unrel_pred_travs)
    unrel_error_rrr = np.abs(unrel_travs - unrel_rrr_travs)
    unrel_rms_error_pred = np.sqrt(np.mean(unrel_error_pred**2, axis=0))
    unrel_rms_error_rrr = np.sqrt(np.mean(unrel_error_rrr**2, axis=0))

    c_r2_pred = measure_r2(pred_travs, travs, reduce=None)
    c_r2_rrr = measure_r2(rrr_travs, travs, reduce=None)
    c_r2_pred_unrel = measure_r2(unrel_pred_travs, unrel_travs, reduce=None)
    c_r2_rrr_unrel = measure_r2(unrel_rrr_travs, unrel_travs, reduce=None)

    vmin = 0
    rel_vmax = travs.max()
    unrel_vmax = unrel_travs.max()
    vmax = max(rel_vmax, unrel_vmax) * 0.3

    interpolation = "none"
    extent = [-width, width, 0, travs.shape[0]]

    show_r2_error = False
    scatter_trav_only = False

    fig, ax = plt.subplots(5, 2, figsize=(8, 8), layout="constrained", sharex=False)
    ax[0, 0].imshow(travs, extent=extent, aspect="auto", cmap="gray_r", interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax[0, 0].set_yticks([])

    ax[1, 0].imshow(pred_travs, extent=extent, aspect="auto", cmap="gray_r", interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax[1, 0].set_yticks([])

    ax[2, 0].imshow(rrr_travs, extent=extent, aspect="auto", cmap="gray_r", interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax[2, 0].set_yticks([])

    if show_r2_error:
        ax[3, 0].plot(np.linspace(-width, width, 2 * width + 1), c_r2_pred, label="PF R^2")
        ax[3, 0].plot(np.linspace(-width, width, 2 * width + 1), c_r2_rrr, label="RRR R^2")
    else:
        ax[3, 0].plot(np.linspace(-width, width, 2 * width + 1), rms_error_pred, label="PF RMS Error")
        ax[3, 0].plot(np.linspace(-width, width, 2 * width + 1), rms_error_rrr, label="RRR RMS Error")

    ax[3, 0].legend(fontsize=8)

    if scatter_trav_only:
        ax[4, 0].scatter(travs.flatten(), pred_travs.flatten(), s=1, c="r", alpha=0.2)
        ax[4, 0].scatter(travs.flatten(), rrr_travs.flatten(), s=1, c="b", alpha=0.2)
    else:
        ax[4, 0].scatter(target_spks[:, cellidx].flatten(), target_prediction[:, cellidx].flatten(), s=1, c="r", alpha=0.2)
        ax[4, 0].scatter(target_spks[:, cellidx].flatten(), rrr_prediction[:, cellidx].flatten(), s=1, c="b", alpha=0.2)

    scatter_xlim = ax[4, 0].get_xlim()
    scatter_ylim = ax[4, 0].get_ylim()
    scatter_min = min(scatter_xlim[0], scatter_ylim[0])
    scatter_max = max(scatter_xlim[1], scatter_ylim[1])
    ax[4, 0].set_xlim(0)  # , scatter_max)
    ax[4, 0].set_ylim(0)  # , scatter_max)
    helpers.refline(1, 0, ax[4, 0], color="k", linestyle="--")

    ax[0, 1].imshow(unrel_travs, extent=extent, aspect="auto", cmap="gray_r", interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax[0, 1].set_yticks([])

    ax[1, 1].imshow(unrel_pred_travs, extent=extent, aspect="auto", cmap="gray_r", interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax[1, 1].set_yticks([])

    ax[2, 1].imshow(unrel_rrr_travs, extent=extent, aspect="auto", cmap="gray_r", interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax[2, 1].set_yticks([])

    if show_r2_error:
        ax[3, 1].plot(np.linspace(-width, width, 2 * width + 1), c_r2_pred_unrel, label="PF R^2")
        ax[3, 1].plot(np.linspace(-width, width, 2 * width + 1), c_r2_rrr_unrel, label="RRR R^2")
    else:
        ax[3, 1].plot(np.linspace(-width, width, 2 * width + 1), unrel_rms_error_pred, label="PF RMS Error")
        ax[3, 1].plot(np.linspace(-width, width, 2 * width + 1), unrel_rms_error_rrr, label="RRR RMS Error")

    ax[3, 1].legend(fontsize=8)

    if scatter_trav_only:
        ax[4, 1].scatter(unrel_travs.flatten(), unrel_pred_travs.flatten(), s=1, c="r", alpha=0.2)
        ax[4, 1].scatter(unrel_travs.flatten(), unrel_rrr_travs.flatten(), s=1, c="b", alpha=0.2)
    else:
        ax[4, 1].scatter(target_spks[:, unrel_cell_idx].flatten(), target_prediction[:, unrel_cell_idx].flatten(), s=1, c="r", alpha=0.2)
        ax[4, 1].scatter(target_spks[:, unrel_cell_idx].flatten(), rrr_prediction[:, unrel_cell_idx].flatten(), s=1, c="b", alpha=0.2)

    scatter_xlim = ax[4, 1].get_xlim()
    scatter_ylim = ax[4, 1].get_ylim()
    scatter_min = min(scatter_xlim[0], scatter_ylim[0])
    scatter_max = max(scatter_xlim[1], scatter_ylim[1])
    ax[4, 1].set_xlim(0)  # , scatter_max)
    ax[4, 1].set_ylim(0)  # , scatter_max)
    helpers.refline(1, 0, ax[4, 1], color="k", linestyle="--")

    for i in range(4):
        for j in range(2):
            ax[i, j].set_xlim(-width, width)
            # ax[i, j].set_xticks([])
            # link x to first axes
            if i > 0:
                # link x axis to ax[0, j]
                ax[i, j].sharex(ax[0, j])
    plt.show()

    # ------------------------------------------------
    # ------------------------------------------------
    # ----- example basic results (speed && PFs) -----
    # ------------------------------------------------
    # ------------------------------------------------

    mouse_name = "ATL012"
    track = tracking.tracker(mouse_name)  # get tracker object for mouse
    pcm = analysis.placeCellMultiSession(track, autoload=False)  # open up place cell multi session analysis object (don't autoload!!!)

    ises = 18
    pcss = pcm.pcss[ises]
    pcss.load_behavioral_data(distStep=5)
    speedmap = pcss.speedmap
    envindex = pcss.vrexp.loadone("trials.environmentIndex")
    environments = pcss.environments
    num_env = len(environments)

    distcenters = pcss.distcenters
    distedges = pcss.distedges

    rewzones = pcss.vrexp.loadone("trials.rewardPosition")
    rewhw = np.unique(pcss.vrexp.loadone("trials.rewardZoneHalfWidth"))
    assert len(rewhw) == 1, "Multiple reward half widths found"
    rewzones = np.array([rewzones[np.where(envindex == envnum)[0][0]] for envnum in environments])
    rewstarts = rewzones - rewhw
    rewends = rewzones + rewhw

    use_se = False
    avg_speed = np.zeros((num_env, speedmap.shape[1]))
    dev_speed = np.zeros((num_env, speedmap.shape[1]))
    for ienv in range(num_env):
        idx = envindex == environments[ienv]
        avg_speed[ienv] = np.nanmean(speedmap[idx], axis=0)
        dev_speed[ienv] = np.nanstd(speedmap[idx], axis=0)
        if use_se:
            dev_speed[ienv] /= np.sqrt(np.sum(idx))

    plt.rcParams.update({"font.size": 24})

    cols = "krb"
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), layout="constrained")
    for ienv in range(num_env):
        ax.plot(distcenters, avg_speed[ienv], c=cols[ienv], label=environments[ienv])
        ax.fill_between(
            distcenters,
            avg_speed[ienv] - dev_speed[ienv],
            avg_speed[ienv] + dev_speed[ienv],
            color=cols[ienv],
            alpha=0.2,
        )
    ylims = ax.get_ylim()
    ax.set_ylim(0, ylims[1])
    for ienv, (start, end) in enumerate(zip(rewstarts, rewends)):
        ax.fill_between([start, end], 0, ylims[1], color=cols[ienv], alpha=0.1, edgecolor=None)

    ax.set_xlim(distedges[0], distedges[-1])
    ax.set_ylim(0)
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Speed (cm/s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

    from copy import copy

    def fancyplot_remap_snakes(
        self,
        reliable=True,
        cutoffs=(0.4, 0.7),
        method="max",
        normalize=0,
        rewzone=True,
        interpolation="none",
        force_single_env=False,
        withLabels=True,
        labelSize=24,
        withShow=True,
        withSave=False,
    ):
        """method for plotting cross-validated snake plot"""
        # plotting remap snakes always uses all environments
        envnum = helpers.check_iterable(copy(self.environments))  # always use all environments (as an iterable)
        numEnv = len(envnum)

        if numEnv == 1 and not (force_single_env):
            print(f"Session {self.vrexp.sessionPrint()} only uses 1 environment, not plotting remap snakes")
            return None

        # make snakes
        snake_remap = self.make_remap_data(reliable=reliable, cutoffs=cutoffs, method=method)

        # prepare plotting data
        extent = lambda ii, jj: [
            self.distedges[0],
            self.distedges[-1],
            0,
            snake_remap[ii][jj].shape[0],
        ]
        if normalize > 0:
            vmin, vmax = 0, np.abs(normalize)
        elif normalize < 0:
            maxrois = np.concatenate([np.concatenate([np.nanmax(np.abs(srp), axis=1) for srp in s_remap]) for s_remap in snake_remap])
            vmin, vmax = 0, np.percentile(maxrois, -normalize)
        else:
            magnitude = np.nanmax(np.abs(np.vstack([np.concatenate(srp) for srp in snake_remap])))
            vmin, vmax = 0, magnitude

        cb_ticks = [vmin, vmax]
        cb_unit = r"$\sigma$" if self.standardizeSpks else "au"
        cb_label = f"Activity ({cb_unit})"
        cols = "krb"

        # load reward zone information
        if rewzone:
            # get reward zone start and stop, and filter to requested environments
            rewPos, rewHalfwidth = helpers.environmentRewardZone(self.vrexp)
            rewPos = [rewPos[np.where(self.environments == ev)[0][0]] for ev in envnum]
            rewHalfwidth = [rewHalfwidth[np.where(self.environments == ev)[0][0]] for ev in envnum]
            rect = lambda ii, jj: mpl.patches.Rectangle(
                (rewPos[jj] - rewHalfwidth[jj], 0),
                rewHalfwidth[jj] * 2,
                snake_remap[ii][jj].shape[0],
                edgecolor="none",
                facecolor=cols[jj],
                alpha=0.2,
            )

        plt.close("all")
        cmap = mpl.colormaps["gray_r"]

        fig_dim = 1.8
        width_ratios = [*[fig_dim for _ in range(numEnv)], fig_dim / 10]

        if not (withLabels):
            width_ratios = width_ratios[:-1]
        fig, ax = plt.subplots(
            numEnv,
            numEnv + 1 * withLabels,
            width_ratios=width_ratios,
            figsize=(sum(width_ratios) * 1.7, fig_dim * numEnv),
            layout="constrained",
        )

        if numEnv == 1:
            ax = np.reshape(ax, (1, -1))

        for ii in range(numEnv):
            for jj in range(numEnv):
                # make image
                aim = ax[ii, jj].imshow(
                    snake_remap[ii][jj],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent(ii, jj),
                    aspect="auto",
                    interpolation=interpolation,
                )

                # label images
                if ii == numEnv - 1:
                    if jj == 1:
                        ax[ii, jj].set_xlabel("Position (cm)", fontsize=labelSize)

                if jj == 0:
                    ax[ii, jj].set_ylabel(f"ROIs {ii}", fontsize=labelSize)

                if ii == 0:
                    ax[ii, jj].set_title(f"Env {jj}", fontsize=labelSize)

                if rewzone:
                    ax[ii, jj].add_patch(rect(ii, jj))

        if withLabels:
            for ii in range(numEnv):
                if ii == 1:
                    fig.colorbar(aim, ticks=cb_ticks, orientation="vertical", cax=ax[ii, -1])
                    ax[ii, -1].set_ylabel(cb_label, fontsize=labelSize)
                else:
                    ax[ii, -1].axis("off")

        for ii in range(numEnv):
            for jj in range(numEnv):
                ax[ii, jj].set_xticks([])
                ax[ii, jj].set_yticks([])
                ax[ii, jj].xaxis.set_tick_params(labelbottom=False)
                ax[ii, jj].yaxis.set_tick_params(labelleft=False)

        if withSave:
            name = f"remap_snake_plot"
            if not (withLabels):
                name = name + "_nolabel"
            self.saveFigure(fig.number, name)

        # Show figure if requested
        plt.show() if withShow else plt.close()

        return fig, ax

    plt.rcParams.update({"font.size": 24})

    fig, ax = fancyplot_remap_snakes(pcss, normalize=4)

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = pcm.pcss[ises].saveDirectory("example_plots")
        save_name = f"{pcm.pcss[ises].vrexp.sessionPrint('_')}_example_snakes"
        save_path = save_directory / save_name
        helpers.save_figure(fig, save_path)
