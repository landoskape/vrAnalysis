import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from rastermap import Rastermap
from vrAnalysis import analysis
from vrAnalysis import tracking


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
    relmse, relcor = vss.get_reliability_values()
    any_reliable = np.any(np.stack(idx_reliable), axis=0)
    distedges = vss.distedges
    distcenters = vss.distcenters

    spks_reliable = spks[:, any_reliable]
    spks_pred_reliable = spks_prediction[:, any_reliable]

    # fit rastermap
    zspks_reliable = sp.stats.zscore(spks_reliable, axis=0)
    model = Rastermap(n_PCs=200, n_clusters=40, locality=0.75, time_lag_window=5).fit(zspks_reliable.T)
    isort = model.isort

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), layout="constrained", sharex=True)
    im = ax[0].imshow(spks_reliable.T[isort], vmin=0, vmax=2, cmap="gray_r", aspect="auto")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("ROI")
    inset = ax[0].inset_axes([0.75, 0.25, 0.2, 0.075])
    inset.xaxis.set_ticks_position("bottom")
    cb0 = fig.colorbar(im, cax=inset, orientation="horizontal")
    # set label with greek letter sigma
    cb0.set_label("ROI Activity ($\sigma$)", loc="center", y=10)

    for ii, fp in enumerate(frame_position):
        ax[1].plot(fp, label=ii)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Virtual Position / Env")
    ax[1].legend()

    plt.show()
