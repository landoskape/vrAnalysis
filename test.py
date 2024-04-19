import os
from copy import copy

from vrAnalysis import analysis
from vrAnalysis import tracking
from vrAnalysis import helpers
from vrAnalysis import database
from tqdm import tqdm
import numpy as np
import faststats as fs
import pickle

mousedb = database.vrDatabase("vrMice")
df = mousedb.getTable(trackerExists=True)
mouse_names = df["mouseName"].unique()

# use this one for testing on ATL028 because it has fewer sessions
# mouse_names = ["ATL028", "ATL012", "ATL020", "ATL022", "ATL027", "CR_Hippocannula6", "CR_Hippocannula7", "ATL045"]


def add_to_spectra_data(pcm, args):
    """skeleton for adding something without reloading everything"""
    with open(pcm.saveDirectory("temp") / f"{args.mouse_name}_spectra_data.pkl", "rb") as f:
        temp_files = pickle.load(f)

    vss = []
    for p in pcm.pcss:
        vss.append(analysis.VarianceStructure(p.vrexp, distStep=args.dist_step, autoload=False))

    # first load session data (this can take a while)
    for v in tqdm(vss, leave=True, desc="loading session data"):
        v.load_data()

    # get spkmaps of all cells / just reliable cells
    kernels = []
    cv_kernels = []
    for v in tqdm(vss, leave=False, desc="preparing spkmaps"):
        c_spkmaps = v.prepare_spkmaps(envnum=None, smooth=args.smooth, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs, reliable=False)
        train_idx, test_idx = helpers.named_transpose([helpers.cvFoldSplit(np.arange(spkmap.shape[1]), 2) for spkmap in c_spkmaps])

        # full place fields
        c_placefields_all = [np.nanmean(spkmap, axis=1) for spkmap in c_spkmaps]
        c_pf_train = [np.nanmean(spkmap[:, tidx], axis=1) for spkmap, tidx in zip(c_spkmaps, train_idx)]
        c_pf_test = [np.nanmean(spkmap[:, tidx], axis=1) for spkmap, tidx in zip(c_spkmaps, test_idx)]
        c_pf_train_centered = [pf - np.nanmean(pf, axis=0) for pf in c_pf_train]
        c_pf_test_centered = [pf - np.nanmean(pf, axis=0) for pf in c_pf_test]

        kernels.append([np.cov(pf.T) for pf in c_placefields_all])
        cv_kernels.append([cpftrain.T @ cpftest / (cpftrain.shape[0] - 1) for cpftrain, cpftest in zip(c_pf_train_centered, c_pf_test_centered)])

    update_dict = {"kernels": kernels, "cv_kernels": cv_kernels}

    temp_files.update(update_dict)

    pcm.save_temp_file(temp_files, f"{args.mouse_name}_spectra_data.pkl")


if __name__ == "__main__":
    for mouse_name in mouse_names:
        print(f"Analyzing {mouse_name}")

        # load spectra data for target mouse
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False)  # open up place cell multi session analysis object (don't autoload!!!)

        # load spectra data (use temp if it matches)
        args = helpers.AttributeDict(
            dict(
                mouse_name=mouse_name,
                dist_step=1,
                smooth=0.1,
                cutoffs=(0.4, 0.7),
                maxcutoffs=None,
                reload_spectra_data=False,
            )
        )

        add_to_spectra_data(pcm, args)

        # os.system(f"python scripts/mouse_summary.py --mouse-name {mouse_name} --do-spectra")
