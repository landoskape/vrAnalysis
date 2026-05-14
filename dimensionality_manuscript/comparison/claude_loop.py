import numpy as np
from vrAnalysis.files import literature_data_path
from vrAnalysis.external import pixease

DATA_PATH = literature_data_path() / "CortexLab_ZebraNoise"

EXPERIMENT_TARGETS = ["natural_images", "resting_state", "video", "full_field_drifting_grating"]


def gather_by_experiment_type(fpath, targets=EXPERIMENT_TARGETS):
    files_by_target = {target: list(fpath.glob(f"expcache_{target}_*.npz")) for target in targets}
    return files_by_target


def print_data_structure(data):
    for key in data.files:
        if key.endswith("___TYPE"):
            continue
        val = data[key]
        type_key = key + "___TYPE"
        type_info = str(data[type_key]) if type_key in data.files else ""
        print(f"{key}: shape={val.shape}, dtype={val.dtype}  {type_info}")


def print_metadata(data):
    # Inspect any scalar/string fields (dtype=object, shape=())

    # Note one of these is literally the full code and other metadata to run - might need some filtering before printing
    # and putting everything in context!!!
    for key in data.files:
        if key.endswith("___TYPE"):
            continue
        val = data[key]
        if val.ndim == 0 or val.dtype == object:
            print(f"{key}: {val.item() if val.ndim == 0 else val}")


def load(fpath) -> pixease.Experiment:
    """path_format = 'D:/literatureData/CortexLab_ZebraNoise/expcache_natural_images_BZ014_2025-04-16_2.npz'"""
    mouse, date, expnum = fpath.stem.split("_")[-3:]
    exp = pixease.load(mouse, date, expnum, from_dir=DATA_PATH)
    return exp


"""
Example usage: 
for a loaded experiment:
exp.summary()

help(exp.timeseries)  # to see what timeseries are available and their shapes (not all available, since this is going to be 2P data only, not ephys)
exp.stimulus_timings() # returns a dataframe with stimulus timing and metadata info, used for filtering timeseries, or timeseries by interval, etc.

# Get intervals in a list of (start, stop) tuples and then retrieve averaged neural responses on that interval
interval = list(zip(*(exp.stimulus_timings()["time_start"], exp.stimulus_timings()["time_stop"])))
exp.interval_mean(interval, "dspikes", cells=exp.cellinfo.iscell).shape

# Get timeseries of deconvolved spikes for all cells, for a specific time window (e.g. 10-30s)
exp.timeseries((10, 30), "dspikes", cells=exp.cellinfo.iscell).shape

"""
