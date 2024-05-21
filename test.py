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
    rel_mse = []
    for v in tqdm(vss, leave=False, desc="preparing spkmaps"):
        c_mse = v.get_reliability_values(envnum=None, with_test=False)[0]
        rel_mse.append(c_mse)

    update_dict = {
        "rel_mse": rel_mse,
    }

    temp_files.update(update_dict)

    pcm.save_temp_file(temp_files, f"{args.mouse_name}_spectra_data.pkl")


if __name__ == "__main__":
    for mouse_name in mouse_names:
        print(f"Analyzing {mouse_name}")

        # # load spectra data for target mouse
        # track = tracking.tracker(mouse_name)  # get tracker object for mouse
        # pcm = analysis.placeCellMultiSession(track, autoload=False)  # open up place cell multi session analysis object (don't autoload!!!)

        # # load spectra data (use temp if it matches)
        # args = helpers.AttributeDict(
        #     dict(
        #         mouse_name=mouse_name,
        #         dist_step=1,
        #         smooth=0.1,
        #         cutoffs=(0.4, 0.7),
        #         maxcutoffs=None,
        #         reload_spectra_data=False,
        #     )
        # )

        # add_to_spectra_data(pcm, args)

        os.system(f"python scripts/mouse_summary.py --mouse-name {mouse_name} --do-spectra")
