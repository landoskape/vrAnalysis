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

from dimilibi import Population, SVCA

mousedb = database.vrDatabase("vrMice")
df = mousedb.getTable(trackerExists=True)
mouse_names = df["mouseName"].unique()


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

    # measure the "dimensionality" of the place field lookup by the dimensionality of the spkmaps...
    svca_shared_prediction = []
    svca_total_prediction = []
    for ii, v in enumerate(tqdm(vss, leave=True, desc="creating place field lookup")):
        c_spks = v.prepare_spks()
        c_spkmaps_train = v.get_spkmap(average=True, smooth=args.smooth, trials="train")
        c_spkmaps_test = v.get_spkmap(average=True, smooth=args.smooth, trials="test")
        c_frame_position, c_idx_valid = v.get_frame_behavior(use_average=True)
        c_pf_pred_train = v.generate_spks_prediction(c_spks, c_spkmaps_train, c_frame_position, c_idx_valid, background_value=0.0)
        c_pf_pred_test = v.generate_spks_prediction(c_spks, c_spkmaps_test, c_frame_position, c_idx_valid, background_value=0.0)
        idx_nan = np.isnan(c_pf_pred_train).any(axis=1) | np.isnan(c_pf_pred_test).any(axis=1)
        c_pf_pred_train[idx_nan] = 0.0
        c_pf_pred_test[idx_nan] = 0.0

        time_split_prms = dict(
            num_groups=2,
            chunks_per_group=-2,
            num_buffer=2,
        )
        npop = Population(c_pf_pred_train.T, time_split_prms=time_split_prms)
        train_source_pred = npop.apply_split(c_pf_pred_train.T, 0, center=True, scale=False)[npop.cell_split_indices[0]]
        train_target_pred = npop.apply_split(c_pf_pred_train.T, 0, center=True, scale=False)[npop.cell_split_indices[1]]
        test_source_pred = npop.apply_split(c_pf_pred_test.T, 1, center=True, scale=False)[npop.cell_split_indices[0]]
        test_target_pred = npop.apply_split(c_pf_pred_test.T, 1, center=True, scale=False)[npop.cell_split_indices[1]]

        svca = SVCA(num_components=temp_files["rank_pf_prediction"][ii], truncated=True).fit(train_source_pred, train_target_pred)
        pf_shared_var, pf_total_var = svca.score(test_source_pred, test_target_pred, normalize=True)
        svca_shared_prediction.append(pf_shared_var)
        svca_total_prediction.append(pf_total_var)

    update_dict = dict(
        svca_shared_prediction_cv=svca_shared_prediction,
        svca_total_prediction_cv=svca_total_prediction,
    )

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
