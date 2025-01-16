import numpy as np
import matplotlib.pyplot as plt
from vrAnalysis import analysis
from vrAnalysis import tracking
from vrAnalysis import helpers

if __name__ == "__main__":
    mouse_name = "ATL027"
    track = tracking.tracker(mouse_name)  # get tracker object for mouse
    pcm = analysis.placeCellMultiSession(track, autoload=False)  # open up place cell multi session analysis object (don't autoload!!!)

    ises = 10
    vss = analysis.VarianceStructure(pcm.pcss[ises].vrexp, autoload=False)
    spks = vss.prepare_spks()
    spkmaps = vss.get_spkmap(average=True, smooth=0.1)
    tspkmaps = vss.get_spkmap(average=False, smooth=0.1)
    frame_position, idx_valid = vss.get_frame_behavior(use_average=True)
    frame_position_xx, idx_valid_xx = vss.get_frame_behavior(use_average=False)
    spks_prediction = vss.generate_spks_prediction(spks, spkmaps, frame_position, idx_valid, background_value=np.nan)
    spks_prediction_xx = vss.generate_spks_prediction(spks, spkmaps, frame_position_xx, idx_valid_xx, background_value=np.nan)
    idx_reliable = vss.get_reliable(cutoffs=(0.4, 0.7))
    relmse, relcor, relloo = vss.get_reliability_values()
    any_reliable = np.any(np.stack(idx_reliable), axis=0)
    distedges = vss.distedges
    distcenters = vss.distcenters

    spks_reliable = spks[:, any_reliable]
    spks_pred_reliable = spks_prediction[:, any_reliable]

    ienv = 0
    icmp = 1

    cutoffs = (0.8, 0.95)
    icell = 1989  # np.random.choice(np.where((relmse[ienv]>cutoffs[0]) & (relcor[ienv]>cutoffs[1]))[0], 1)[0]

    tcell_map = [tspkmaps[ienv][icell], tspkmaps[icmp][icell]]
    cell_map = np.stack([spkmaps[ienv][icell], spkmaps[icmp][icell]])

    use_trial_peak = True
    if use_trial_peak:
        pfmin = np.nanmin(cell_map)
        pfmax = np.nanmax(cell_map)
        vmin = np.mean([np.nanmin([np.nanmin(cm) for cm in tcell_map]), pfmin])
        vmax = np.mean([np.nanmax([np.nanmax(cm) for cm in tcell_map]), pfmax])
    else:
        vmin = np.nanmin(cell_map)
        vmax = np.nanmax(cell_map)

    distcenters = pcm.pcss[ises].distcenters
    distedges = pcm.pcss[ises].distedges
    extent = [[distedges[0], distedges[-1], 0, cm.shape[0]] for cm in tcell_map]

    figdim = 3
    num_per_heatmap = 2
    fig = plt.figure(figsize=((num_per_heatmap + 1) * figdim, 2 * figdim), layout="constrained")
    fig.suptitle(f"Session: {pcm.pcss[ises].vrexp.sessionPrint()} - Cell: {icell} Envs: {ienv}/{icmp}")

    gs = fig.add_gridspec(nrows=num_per_heatmap + 1, ncols=2, left=0.05, right=0.95, hspace=0.1, wspace=0.05)
    ax00 = fig.add_subplot(gs[:num_per_heatmap, 0])
    ax01 = fig.add_subplot(gs[:num_per_heatmap, 1], sharex=ax00)
    ax10 = fig.add_subplot(gs[num_per_heatmap, 0], sharex=ax00)
    ax11 = fig.add_subplot(gs[num_per_heatmap, 1], sharex=ax00)

    ax00.imshow(tcell_map[0], extent=extent[0], vmin=vmin, vmax=vmax, cmap="gray_r", aspect="auto", interpolation="none")
    im = ax01.imshow(tcell_map[1], extent=extent[1], vmin=vmin, vmax=vmax, cmap="gray_r", aspect="auto", interpolation="none")
    inset = ax01.inset_axes([0.25, 0.25, 0.5, 0.05])
    inset.xaxis.set_ticks_position("bottom")
    cb0 = fig.colorbar(im, cax=inset, orientation="horizontal")
    # set label with greek letter sigma
    cb0.set_label("ROI Activity ($\sigma$)", loc="center", y=10, fontsize=12)

    ax10.plot(distcenters, cell_map[0], c="k", linewidth=1.5)
    ax11.plot(distcenters, cell_map[1], c="k", linewidth=1.5)

    for ii, ax in enumerate([ax00, ax01]):
        ax.set_xlabel("Virtual Position (cm)")
        ax.set_ylabel("Trials")
        ax.invert_yaxis()
        ax.set_title(f"Activity Env {ii}")

    for ax in [ax10, ax11]:
        ax.set_xlabel("Virtual Position (cm)")
        ax.set_ylabel("Activity ($\sigma$)")
        ax.set_ylim(0, pfmax)
        ax.set_title("Average Place Field")

    plt.show()

    icell_relvals = [relcor[ienv][icell], relcor[icmp][icell]]

    trainidx, testidx = helpers.named_transpose([helpers.cvFoldSplit(tc.shape[0], 2) for tc in tcell_map])

    train_trials = [tc[tidx] for tc, tidx in zip(tcell_map, trainidx)]
    test_trials = [tc[tidx] for tc, tidx in zip(tcell_map, testidx)]
    train_avg = [np.mean(tc[tidx], axis=0) for tc, tidx in zip(tcell_map, trainidx)]
    test_avg = [np.mean(tc[tidx], axis=0) for tc, tidx in zip(tcell_map, testidx)]

    use_trial_peak = True
    if use_trial_peak:
        pfmin = [np.nanmin([np.nanmin(train_avg[ii]), np.nanmin(test_avg[ii])]) for ii in range(2)]
        pfmax = [np.nanmax([np.nanmax(train_avg[ii]), np.nanmax(test_avg[ii])]) for ii in range(2)]
        vmin = [np.mean([np.nanmin([np.nanmin(train_trials[ii]), np.nanmin(test_trials[ii])]), pfmin[ii]]) for ii in range(2)]
        vmax = [np.mean([np.nanmax([np.nanmax(train_trials[ii]), np.nanmax(test_trials[ii])]), pfmax[ii]]) for ii in range(2)]
    else:
        vmin = [np.nanmin([np.nanmin(train_avg[ii]), np.nanmin(test_avg[ii])]) for ii in range(2)]
        vmax = [np.nanmax([np.nanmax(train_avg[ii]), np.nanmax(test_avg[ii])]) for ii in range(2)]

    distcenters = pcm.pcss[ises].distcenters
    distedges = pcm.pcss[ises].distedges
    train_extent = [[distedges[0], distedges[-1], 0, cm.shape[0]] for cm in train_trials]
    test_extent = [[distedges[0], distedges[-1], 0, cm.shape[0]] for cm in test_trials]

    figdim = 3
    num_per_heatmap = 2
    fig = plt.figure(figsize=((num_per_heatmap + 1) * figdim, 2 * figdim), layout="constrained")
    fig.suptitle(f"Session: {pcm.pcss[ises].vrexp.sessionPrint()} - Cell: {icell} Envs: {ienv}/{icmp}")

    gs = fig.add_gridspec(nrows=num_per_heatmap * 2 + 1, ncols=2, left=0.05, right=0.95, hspace=0.1, wspace=0.05)
    ax00 = fig.add_subplot(gs[:num_per_heatmap, 0])
    ax01 = fig.add_subplot(gs[:num_per_heatmap, 1], sharex=ax00)
    ax10 = fig.add_subplot(gs[num_per_heatmap : 2 * num_per_heatmap, 0], sharex=ax00)
    ax11 = fig.add_subplot(gs[num_per_heatmap : 2 * num_per_heatmap, 1], sharex=ax00)
    ax20 = fig.add_subplot(gs[2 * num_per_heatmap, 0], sharex=ax00)
    ax21 = fig.add_subplot(gs[2 * num_per_heatmap, 1], sharex=ax00)

    im0 = ax00.imshow(train_trials[0], extent=train_extent[0], vmin=vmin[0], vmax=vmax[0], cmap="gray_r", aspect="auto", interpolation="none")
    im1 = ax01.imshow(train_trials[1], extent=train_extent[1], vmin=vmin[1], vmax=vmax[1], cmap="gray_r", aspect="auto", interpolation="none")
    ax10.imshow(test_trials[0], extent=test_extent[0], vmin=vmin[0], vmax=vmax[0], cmap="Reds", aspect="auto", interpolation="none")
    ax11.imshow(test_trials[1], extent=test_extent[1], vmin=vmin[1], vmax=vmax[1], cmap="Reds", aspect="auto", interpolation="none")

    ax20.plot(distcenters, train_avg[0], c="k", linewidth=1.5)
    ax20.plot(distcenters, test_avg[0], c="r", linewidth=1.5)
    ax21.plot(distcenters, train_avg[1], c="k", linewidth=1.5)
    ax21.plot(distcenters, test_avg[1], c="r", linewidth=1.5)

    # add text to upper right of ax20 with relval for each plot
    for ii, ax in enumerate([ax20, ax21]):
        ax.text(0.99, 0.9, f"RelVal: {icell_relvals[ii]:.2f}", ha="right", va="top", fontsize=8, transform=ax.transAxes)

    inset = ax10.inset_axes([0.45, 0.35, 0.5, 0.05])
    inset.xaxis.set_ticks_position("bottom")
    cb0 = fig.colorbar(im0, cax=inset, orientation="horizontal")
    # set label with greek letter sigma
    cb0.set_label("ROI Activity ($\sigma$)", loc="center", fontsize=8)
    cb0.ax.tick_params(labelsize=6)  # Adjust fontsize of tick labels

    inset = ax11.inset_axes([0.45, 0.35, 0.5, 0.05])
    inset.xaxis.set_ticks_position("bottom")
    cb1 = fig.colorbar(im1, cax=inset, orientation="horizontal")
    # set label with greek letter sigma
    cb1.set_label("ROI Activity ($\sigma$)", loc="center", fontsize=8)
    cb1.ax.tick_params(labelsize=6)  # Adjust fontsize of tick labels

    for ii, ax in enumerate([ax00, ax01]):
        ax.set_xlabel("Virtual Position (cm)")
        ax.set_ylabel("Train Trials")
        ax.invert_yaxis()
        ax.set_title(f"Activity Env {ii}")

    for ax in [ax10, ax11]:
        ax.set_xlabel("Virtual Position (cm)")
        ax.set_ylabel("Test Trials")
        ax.invert_yaxis()

    for ii, ax in enumerate([ax20, ax21]):
        ax.set_xlabel("Virtual Position (cm)")
        ax.set_ylabel("Activity ($\sigma$)")
        ax.set_ylim(pfmin[ii], pfmax[ii])
        ax.set_title("Average Place Field")

    plt.show()
