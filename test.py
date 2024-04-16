import os

mouse_names = ["ATL012", "ATL028", "ATL020", "ATL022", "ATL027", "CR_Hippocannula6"]

from vrAnalysis import analysis
from vrAnalysis import tracking
from vrAnalysis import helpers
from tqdm import tqdm
import numpy as np
import faststats as fs
import pickle


def add_to_spectra_data(pcm, args):
    """skeleton for adding something without reloading everything"""
    with open(pcm.saveDirectory("temp") / f"{args.mouse_name}_spectra_data.pkl", "rb") as f:
        temp_files = pickle.load(f)

    vss = []
    for p in pcm.pcss:
        vss.append(analysis.VarianceStructure(p.vrexp, distStep=args.dist_step, autoload=False))

    # get spks of all cells (in time, not space)
    idx_rois = []
    for v in vss:
        v.get_plane_idx(keep_planes=[1, 2, 3, 4])
        idx_rois.append(v.idxUseROI)
    ospks = [v.vrexp.loadone("mpci.roiActivityDeconvolvedOasis")[:, idx] for v, idx in zip(vss, idx_rois)]

    # get spkmaps of all cells / just reliable cells
    svca_shared = []
    svca_total = []
    for ospk in tqdm(ospks, leave=False, desc="doing SVCA"):
        c_shared_var, c_tot_cov_space_var = helpers.split_and_svca(ospk.T, verbose=False)[:2]
        svca_shared.append(c_shared_var)
        svca_total.append(c_tot_cov_space_var)

    temp_files["svca_shared"] = svca_shared
    temp_files["svca_total"] = svca_total

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
