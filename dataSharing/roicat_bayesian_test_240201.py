from argparse import ArgumentParser
import numpy as np

import os
import sys

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from vrAnalysis import tracking
from vrAnalysis import analysis
from vrAnalysis import fileManagement as fm
from vrAnalysis import helpers

# shared data folder name
data_name = "roicat_bayesian_test"


def handle_inputs():
    parser = ArgumentParser(description="generate ROICaT and place field data for testing a bayesian analysis")
    parser.add_argument("--mouse-name", type=str, default="ATL027", help="the mouse name to copy sharing data from")
    parser.add_argument("--cutoffs", default=[0.4, 0.7], nargs=2, type=float, help="reliability cutoffs")
    return parser.parse_args()


def create_folder():
    """target folder for this shared data dump"""
    folder = fm.sharedDataPath() / data_name
    if not folder.exists():
        folder.mkdir()


def generate_filepath(mouse_name, plane_name, file_name):
    """specific file path for particular file in this shared data dump"""
    path_name = fm.sharedDataPath() / data_name / mouse_name / plane_name / file_name
    if not path_name.parent.exists():
        path_name.parent.mkdir(parents=True)
    return path_name


def generate_dictionary(mouse_name, cutoffs):
    """method for getting data to share for ROICaT testing"""
    # get tracking object for accessing ROICaT information
    track = tracking.tracker(mouse_name)

    # create ROICaT analysis object
    roistat = analysis.RoicatStats(track)

    # Select best VR environment and sessions
    env, idx_ses = roistat.env_idx_ses_selector(envmethod="most", sesmethod=-1)
    spkmaps, extras = roistat.get_spkmaps(
        env,
        trials="full",
        average=True,
        tracked=False,
        pf_method="max",
        by_plane=True,
        idx_ses=idx_ses,
    )
    relmse = extras["relmse"]
    relcor = extras["relcor"]
    idx_reliable = [[(mse > cutoffs[0]) & (cor > cutoffs[1]) for mse, cor in zip(rmse, rcor)] for rmse, rcor in zip(relmse, relcor)]
    spkmap_centers = roistat.get_from_pcss("distcenters", idx_ses[0])

    # get ROICaT data from relevant planes
    rundata = [track.rundata[p] for p in roistat.keep_planes]
    results = [track.results[p] for p in roistat.keep_planes]

    # create neural data dictionary with transposed lists so outer list has len==len(keep_planes) and inner list corresponds to each session
    spkmaps = helpers.transpose_list(spkmaps)
    relmse = helpers.transpose_list(relmse)
    relcor = helpers.transpose_list(relcor)
    idx_reliable = helpers.transpose_list(idx_reliable)

    neural_data = []
    for ii, plane_idx in enumerate(roistat.keep_planes):
        cdict = dict(
            plane_idx=ii,
            plane_name=f"plane{plane_idx}",
            spkmaps=spkmaps[ii],
            spkmap_centers=spkmap_centers,
            relmse=relmse[ii],
            relcor=relcor[ii],
            idx_reliable=idx_reliable[ii],
            spkmap_desc="(ROI x num_spatial_bins): Average Deconvolved Firing Rate",
            spkmap_centers_desc="(num_spatial_bins, ): virtual location (in cm) of each spatial bin",
            relmse_desc="(ROI, ): spatial reliability computed with mean-squared error (sensitive to changes in firing rate)",
            relcor_desc="(ROI, ): spatial reliability computed with correlation (only sensitive to shape, not scale, of spatial tuning curve)",
            idx_reliable_desc="(ROI, ): boolean vector indicating which ROIs have a place field using my standard cutoffs (0.4 for relmse, 0.7 for relcor)",
        )
        neural_data.append(cdict)

    # plane_names
    plane_names = [f"plane{p}" for p in roistat.keep_planes]

    # return all data
    return results, rundata, neural_data, plane_names


if __name__ == "__main__":
    args = handle_inputs()

    create_folder()

    print("Generate data dictionary...")
    results, rundata, neural_data, plane_names = generate_dictionary(args.mouse_name, args.cutoffs)

    print("Saving data...")
    for pname, res, run, ndata in zip(plane_names, results, rundata, neural_data):
        print(f"Saving data for {pname}...")
        np.save(generate_filepath(args.mouse_name, pname, "results"), res)
        np.save(generate_filepath(args.mouse_name, pname, "rundata"), run)
        np.save(generate_filepath(args.mouse_name, pname, "neural_data"), ndata)
