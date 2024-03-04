import time
import random
from copy import copy
from tqdm import tqdm
from pathlib import Path
import numpy as np

import os
import sys

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from vrAnalysis import session
from vrAnalysis import functions
from vrAnalysis import analysis
from vrAnalysis import helpers
from vrAnalysis import fileManagement as fm
from vrAnalysis import database

# shared data folder name
data_name = "forAgnes_231018"


def create_folder():
    """target folder for this shared data dump"""
    folder = fm.sharedDataPath() / data_name
    if not folder.exists():
        folder.mkdir()


def generate_filepath(name):
    """specific file path for particular file in this shared data dump"""
    path_name = fm.sharedDataPath() / data_name / name
    return path_name


def generate_dictionary():
    """method for getting data as requested by Agnes"""
    keep_planes = [1, 2, 3, 4]
    filename = []
    data = []

    vrdb = database.vrDatabase()
    ises = vrdb.iterSessions(imaging=True, vrRegistration=True)
    for ses in tqdm(ises):
        filename.append(ses.__str__())

        cdata = {}
        cdata["session_name"] = ses.sessionPrint()

        mpciTime = ses.loadone("mpci.times")
        wheelTime = ses.loadone("wheelPosition.times")
        wheelPosition = ses.loadone("wheelPosition.position")
        frame_to_tl = helpers.nearestpoint(mpciTime, wheelTime)[0]

        # get frame index
        stackPosition = ses.loadone("mpciROIs.stackPosition")
        roiPlaneIdx = stackPosition[:, 2].astype(np.int32)  # plane index

        # figure out which ROIs are in the target planes
        idxUseROI = np.any(np.stack([roiPlaneIdx == pidx for pidx in keep_planes]), axis=0)

        # add data to dictionary
        cdata["timestamps"] = mpciTime
        cdata["roi_activity_deconvolved"] = ses.loadone("mpci.roiActivityDeconvolved")[
            :, idxUseROI
        ]
        cdata["wheel_position_cm"] = wheelPosition[frame_to_tl]

        # add data dictionary to list
        data.append(np.array(cdata))

    return filename, data


if __name__ == "__main__":
    create_folder()

    print("Generate data dictionary...")
    filename, data = generate_dictionary()

    print("Saving data...")
    for fname, d in zip(filename, data):
        fpath = generate_filepath(fname)
        print(f"Saving: {fpath}")
        np.save(fpath, d)
