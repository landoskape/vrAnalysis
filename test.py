import os

mouse_names = ["ATL012", "ATL028", "ATL020", "ATL022", "ATL027", "CR_Hippocannula6"]

from vrAnalysis import analysis
from vrAnalysis import tracking
from vrAnalysis import helpers
from tqdm import tqdm
import numpy as np
import faststats as fs
import pickle

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

        # os.system(f"python scripts/mouse_summary.py --mouse-name {mouse_name} --do-spectra")
    pass
