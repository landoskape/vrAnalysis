import os

mouse_names = ["ATL012", "ATL020", "ATL022", "ATL027", "ATL028", "CR_Hippocannula6", "CR_Hippocannula7", "ATL045"]

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

    # first load session data (this can take a while)
    for v in tqdm(vss, leave=True, desc="loading session data"):
        v.load_data()

    # get spkmaps of all cells / just reliable cells
    allcell_maps = []
    relcell_maps = []
    rel_mse = []
    rel_cor = []
    all_pf_mean = []
    all_pf_var = []
    all_pf_cv = []
    all_pf_tcv = []
    rel_pf_mean = []
    rel_pf_var = []
    rel_pf_cv = []
    rel_pf_tcv = []
    for v in tqdm(vss, leave=False, desc="preparing spkmaps"):
        # get reliable cells (for each environment) and spkmaps for each environment (with all cells)
        c_idx_reliable = v.get_reliable(envnum=None, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs)
        c_spkmaps = v.prepare_spkmaps(envnum=None, smooth=args.smooth, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs, reliable=False)
        c_rel_spkmaps = [spkmap[cir] for spkmap, cir in zip(c_spkmaps, c_idx_reliable)]

        # get reliable values for each environment
        c_mse, c_cor = v.get_reliability_values(envnum=None, with_test=False)

        # get place field for each cell
        c_placefields_all = [np.nanmean(spkmap, axis=1) for spkmap in c_spkmaps]
        c_placefields_rel = [np.nanmean(spkmap[cir], axis=1) for spkmap, cir in zip(c_spkmaps, c_idx_reliable)]
        # make place field a unit vector
        c_all_unitpf = [placefield / np.linalg.norm(placefield, axis=1, keepdims=True) for placefield in c_placefields_all]
        c_rel_unitpf = [placefield / np.linalg.norm(placefield, axis=1, keepdims=True) for placefield in c_placefields_rel]

        # add each to list
        allcell_maps.append(c_spkmaps)
        relcell_maps.append(c_rel_spkmaps)
        rel_mse.append(c_mse)
        rel_cor.append(c_cor)
        all_pf_var.append([np.nanvar(placefield, axis=1) for placefield in c_placefields_all])
        rel_pf_var.append([np.nanvar(placefield, axis=1) for placefield in c_placefields_rel])

        # get other place field statistics
        c_all_pf_mean = [fs.nanmean(placefield, axis=1) for placefield in c_placefields_all]
        c_all_pf_cv = [fs.nanstd(placefield, axis=1) / fs.nanmean(placefield, axis=1) for placefield in c_placefields_all]
        c_all_pf_amplitude = [fs.nansum(np.expand_dims(placefield, 1) * spkmap, axis=2) for placefield, spkmap in zip(c_all_unitpf, c_spkmaps)]
        c_all_pf_tcv = [fs.nanstd(amplitude, axis=1) / fs.nanmean(amplitude, axis=1) for amplitude in c_all_pf_amplitude]

        all_pf_mean.append(c_all_pf_mean)
        all_pf_cv.append(c_all_pf_cv)
        all_pf_tcv.append(c_all_pf_tcv)

        c_rel_pf_mean = [fs.nanmean(placefield, axis=1) for placefield in c_placefields_rel]
        c_rel_pf_cv = [fs.nanstd(placefield, axis=1) / fs.nanmean(placefield, axis=1) for placefield in c_placefields_rel]
        c_rel_pf_amplitude = [fs.nansum(np.expand_dims(placefield, 1) * spkmap, axis=2) for placefield, spkmap in zip(c_rel_unitpf, c_rel_spkmaps)]
        c_rel_pf_tcv = [fs.nanstd(amplitude, axis=1) / fs.nanmean(amplitude, axis=1) for amplitude in c_rel_pf_amplitude]

        rel_pf_mean.append(c_rel_pf_mean)
        rel_pf_cv.append(c_rel_pf_cv)
        rel_pf_tcv.append(c_rel_pf_tcv)

    update_dict = {
        "rel_mse": rel_mse,
        "rel_cor": rel_cor,
        "all_pf_mean": all_pf_mean,
        "all_pf_var": all_pf_var,
        "all_pf_cv": all_pf_cv,
        "all_pf_tcv": all_pf_tcv,
        "rel_pf_mean": rel_pf_mean,
        "rel_pf_var": rel_pf_var,
        "rel_pf_cv": rel_pf_cv,
        "rel_pf_tcv": rel_pf_tcv,
    }

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
