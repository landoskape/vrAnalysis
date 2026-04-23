"""Pixease

-------
A script for preprocessing and analysing data from cortexlab 2p and npix/ephys experiments.

2022-2024 Max Shinn <m.shinn@ucl.ac.uk>

Analysing data with Pixease
---------------------------

1. Install numpy, scipy, pandas, scikit-image, and imageio.

2. Add "import pixease" to your code.

3. Load an experiment with "exp = pixease.load(mouse, date, expnum)".

Preprocessing with Pixease
--------------------------

1. You can run all preprocessing steps and then generate a cache file with:

$ python pixease.py run MOUSE DATE EXPNUM

This will autodetect the version of gcamp (from mouseinfo.json) and the rig
used (from explog.json), as well as the pipeline to use (from explog.json).

Due to the difficulty of getting suite2p, cellpose, and deeplabcut all
installed in the same python environment, sometimes you many need to run these
commands separately.  Not all experiments will need all of these to be run.
They can be run with:

$ python pixease.py s2p MOUSE DATE EXPNU
$ python pixease.py eye MOUSE DATE EXPNUM
$ python pixease.py cellpose MOUSE DATE EXPNUM
$ python pixease.py volume MOUSE DATE EXPNUM

Once you have run all of the prerequisites, run

$ python pixease.py cache MOUSE DATE EXPNUM

This will generate a cache file, which you can analyse with Pixease.

There are additional command line arguments you can use.  For more information,
run:

$ python pixease.py --help

Terminology
-----------

Experiment:
    One experiment within a session.  Should have a unique ID given by
    Mouse-Date-Experiment.  An experiment should also have a single type, e.g.,
    a retinotopy or a drifting grating.

Session:
    All experiments from a single mouse on a single day.  We assume these are
    all on the same rig, even though in theory they might not be.

Module:
    One modality of data collected from the rig.  For instance, it may be
    photodiode information, wheel information, etc.  Not all modules may be
    needed for each experiment, e.g., we don't need running speed for a
    retinotopy.  Sometimes, modules may be the output of another script, e.g.,
    eye tracking information is the output from deeplabcut.  Some modules are
    also output from some other script, e.g., lossy compressed images of the
    FOV.

Microscope:
    Used synonymously with "rig".

FOV:
    Field of view.  It usually refers here to the size rather than the
    contents.

Mixin:
    Some piece of information which can be calculated, but depends on multiple
    modules.  This allows multiple experiment types which use similar modules
    to perform the same computation.  For example, the timing of cell spikes
    depends on both the suite2p output as well as information about the FOV.
    Likewise, the activity which occurred while a given stimulus was being
    displayed depends on suite2p output and the stimlus timing.

Pipeline:
    An independent function of Pixease which can be run on its own.  Generally
    either generating the cache file or something which will end up in the
    cache file.

Cache file:
    All of the information needed to load an experiment.  Once a cache file is
    generated, it can be copied to another computer and analysed without the
    original data on the server.

Setting up paths
----------------

Pixease relies on several paths, which are hard-coded.  These paths are only
necessary for the initial data processing, and are not needed when data is
loaded or analysed from a cache file.  After the large introductory comment,
scroll down and find the paths.  They have the following meaning, listed from
most important to know about to least important:

- PROCESSED_DATA_PATH: Should be a path where processed data can be saved.
  Often it is a good idea for this to be local, but it could be on a server
  too.  For example, suite2p output, deeplabcut output, etc.

- DATA_PATHS: A list of paths to the "Subjects" directory where mouse data is
  located.  Should be a list, with each element on the list a path to a
  different server where data might be located.  Data can be split across
  servers for a single mouse.

- XFILES_PATH: The path to zserver/Data/xfiles for working with mpep data.

- DEEPLABCUT_EYE_CONFIG_PATH: The path to Sylvia Schroeder's DeepLabCut
  configuration.

- EXPLOG_PATH: The file name to the explogng2 output json file for managing the
  session, relative to the directory of the session.  Normally should not need
  to be changed.

- MOUSEINFO_PATH: A json file which contains information about the current
  mouse.  This file should be created manually for each new mouse.  This config
  option usually does not need to be changed.

- NOTEBOOK_FILE: The name of the cache file.  This should normally not be
  changed.  If it is changed, however, this will impact analysis with pixease
  as well as initial data processing.

- CHANNELWISE_MEAN_IMAGE_FILENAME: Filename for mean images.  Should not need
  to be changed.

Additionally, when rigs are changed, the following config options are useful:

- CHANNELMAP: For each rig, the PMT number for green and red channels.

- ZOOM_TO_UM_MEASUREMENTS: For each rig, a tuple containing three lists.  The
  first is a list of zoom levels.  The other two lists are the y and x sizes of
  the FOV for that given zoom level, respectively, measured by hand.  A linear
  regression model will be used to compute the actual zoom level from these
  measurements.

- zoom_to_pixel_size: A function which currently assumes a 512x512 pixel FOV.
  If this is different, the function needs to be changed.

Reproducibility
---------------

Pixease includes a copy of itself in every cache file.  Therefore, you will
always know what version of Pixease was used to generate a given cache file.
Additionally, if you can't find a copy of Pixease to analyse your data, you can
extract it from the cache file.

Additionally, Pixease includes a copy of any "patches" used to fix your data
(see "Fixing data")

Fixing data
-----------

Sometimes, experiments fail in subtle ways that make them not work with Pixease,
even though the data are still recoverable.  To help fix this, Pixease allows
you to preprocess any piece of data using a "patch" file.  Patch files must be
named "pixease_patch.py" and placed in the same directory as the experiment.
They should have the function "loader()", which loads and modifies any files
that need to be changed, and passes silently over those that don't.  See the
documentation of "load_with_patch()" for more details.

Tips and tricks
---------------

- The "--help" command line argument gives a description of each option.  Using
  "--help" in along with a pipeline (e.g. "python3 pixease.py cache --help")
  will give additional information about that pipeline.

- If there are specific modules you don't need, you can exclude them using the
  "--skip" command line argument.  This can save a substantial amount of
  space, especially by excluding AudioHQ, and the ΔF/F measurements.  Essential
  modules cannot be excluded and these exclusions will be ignored.

Design notes
------------

File format: I spent some time exploring alternative file formats before
settling on the current format.  Plain npz appears to be the best for our
purposes.

    - HDF5 seems to produce larger files, so there is not much point to using a
      more complex format to get worse storage.  I'm not using memory map
      anyway so the advantages are lost.
    - If you create non-compressed npz files and then compress them using
      bzip2, the compression ratio is a lot better than the default DEFLATE.
      This generates files which are, in general, about 20% smaller.  However,
      it adds memory requirements, since these files would have to be loaded
      into memory and then decompressed with bzip2 and then loaded by numpy.
      This also adds complexity and fragility to the code.  It also means you
      can't open them like a normal zip file and access the underlying npy
      files, you would first have to bzip2 decompress them.  Lastly, it takes
      about 3x longer to decompress.
    - lzma (xz) is worse than bzip2 in this regard.  When I tried lzma, it
      consistently made larger files than bzip2 and took an order of magnitude
      longer to compress.
    - There are some Python libraries that look promising but they would be
      difficult to open in matlab and I don't trust that these libraries will
      still be around in 10 years.

Todo
----

The following upgrades are planned:

- Transition some of the special photodiode classes to use raw photodiode
- Revamp stack image, lossy stack image, lossy image, etc.
- Compress the temporary files
- Make things have more logical names, e.g., "spike_intervals"
- Reduce library dependence to accelerate load times
- Use red channel in suite2p if available
- Merge NeuralFrameTimings with FunctionalF, FunctionalSpikes, etc. and turn *Intervals into a Module
- Automatic choice of smoothing based on dt
- Add experiment start time
- Never overwrite suite2p files that are on the server
- Save other intermediate files to the server instead of locally
- Group modules into "providers" for imaging/npix ("brain provider" or something), camera analysis ("camera provider"), lossy/non-lossy images ("image provider"), etc.  Then for each provider have default modules to include.  That way experiments don't have to list every module, and the same one can be used for ephys/2p/etc.  (Some modules may only support some providers.)

The following are legacy and should be phased out:

- Many of the options in the registration module
- Multiple versions of the *Intervals modules
- Multiple z stack modules
- Change the name to be descriptive of preprocessing, e.g., "kilosort" module instead of "ephys_spikes", etc.
"""

import os
import glob
import argparse
import json
import io
import re
import itertools
import collections
from pathlib import Path
import hashlib
import packaging.version
from datetime import datetime
import gzip
import time
import functools
import textwrap
import filecmp

import numpy as np
import imageio
import pandas
import scipy.signal
import scipy.stats
import scipy.interpolate
import scipy.ndimage
import skimage.registration


if __name__ == "__main__":
    import tifffile
    import skvideo.io
    import shutil
    import itertools
    import scipy.io

# Config options you will probably want to change

DATA_PATHS = [  # With trailing slash
    "/home/max/servers/zubjects/Subjects/{mouse}/{date}/{expnum}/",
    "/home/max/servers/zortex/Subjects/{mouse}/{date}/{expnum}/",
    "/home/max/servers/zaru/Subjects/{mouse}/{date}/{expnum}/",
    "/home/max/servers/zinu/Subjects/{mouse}/{date}/{expnum}/",
    "/home/max/servers/znas/Subjects/{mouse}/{date}/{expnum}/",
]
EXPLOG_PATH = "explog*.json"
MOUSEINFO_PATH = "mouseinfo.json"
PROCESSED_DATA_PATH = "/home/max/Research_data/Subjects/{mouse}/{date}/{expnum}/"  # With trailing slash
XFILES_PATH = "/home/max/servers/znas/Code/xfiles/"  # With trailing slash
NOTEBOOK_FILE = "expcache_{exptype}_{mouse}_{date}_{expnum}.npz"  # Should end in .npz
DEEPLABCUT_EYE_CONFIG_PATH = "/home/max/Downloads/PupilDetection_DLC/DLC data/PupilDetector-cortex_lab-2021-04-28/config.yaml"
CHANNELWISE_MEAN_IMAGE_FILENAME = "mean_by_channel_plane.npy"
CHANNELWISE_MAX_IMAGE_FILENAME = "max_by_channel_plane.npy"
ZSTACK_YX_REGISTRATION_SHIFT_FILENAME = "zstack_yx_shifts.npy"
ZSTACK_Z_REGISTRATION_SHIFT_FILENAME = "zstack_z_shifts.npy"
VIDEOS_PATH = "/home/max/servers/znas/Code/Rigging/ExpDefinitions/Max/videos/"
EPHYS_PATH = "/home/max/servers/rw_zortex/Subjects/{mouse}/{date}/ephys"
LOCAL_KS_CACHE = "/media/data/scratch/"
OUTPUT_DIR = "/home/max/"

# Config options which should only be needed if you upgrade a rig

# Convert the FOV to units of microns.  This requires physically measuring the
# FOV in the microscope.  This changes sometimes, and is current as of June
# 2022.  The measurements are a tuple containing a list of zoom and a
# corresponding list of x and y widths (in microns) for that zoom.  For all
# unmeasured zooms, interpolate.
CHANNELMAP = {"bscope": {"green": 1, "red": 2, "blue": -1}, "b2": {"green": 2, "red": 3, "blue": 1, "farred": 4}}
ZOOM_TO_UM_MEASUREMENTS = {
    "bscope": {
        "1970-01-01": (
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1],
            [914, 853, 776, 711, 674, 624, 588, 555, 522, 492, 474, 449, 306, 231, 188, 155, 135, 117, 105, 95, 87, 79, 73, 68, 64],
            [914, 853, 776, 711, 674, 624, 588, 555, 522, 492, 474, 449, 306, 231, 188, 155, 135, 117, 105, 95, 87, 79, 73, 68, 64],
        )
    },
    # "b2": ([1.6, 1.9, 2, 2.2],
    #        [772, 653, 619.5, 565.5],
    #        [755, 640, 605.5, 547]),
    "b2": {
        "1970-01-01": ([1.6, 1.8, 2.0, 2.6], [705, 652, 599, 439], [705, 652, 599, 439]),
        "1970-05-11": (
            [
                1.3,
                1.2,
                1.1,
                1,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2,
                1,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
                7,
                2,
                1.5,
                1,
                1.2,
                1.3,
                1.1,
                10,
            ],  # Actually 2023-05-11, previous measurement wasn't taken for zooms below 1.6 due to incorrect extrapolation assumptions.
            [
                873.5,
                952,
                1093.5,
                1307.5,
                829,
                756,
                710,
                673,
                630.5,
                598.5,
                568.5,
                1271,
                467,
                382.5,
                329.5,
                289,
                258,
                232.5,
                168,
                571.5,
                767,
                1275,
                962.5,
                876.5,
                1081.5,
                118,
            ],
            [
                873.5,
                952,
                1093.5,
                1307.5,
                829,
                756,
                710,
                673,
                630.5,
                598.5,
                568.5,
                1271,
                467,
                382.5,
                329.5,
                289,
                258,
                232.5,
                168,
                571.5,
                767,
                1275,
                962.5,
                876.5,
                1081.5,
                118,
            ],
        ),
        "2024-02-19": (
            [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [985, 914, 853, 800, 753, 711, 674, 640, 582, 400, 298, 237, 197, 171, 149, 132, 119, 108, 99],
            [985, 914, 853, 800, 753, 711, 674, 640, 582, 400, 298, 237, 197, 171, 149, 132, 119, 108, 99],
        ),
    },
}


def zoom_to_pixel_size(rig, zoom, date):
    """Given a zoom, determine the size of each pixel for a given rig.

    Use the most recent calibration data.
    """
    meas_by_date = ZOOM_TO_UM_MEASUREMENTS[rig]
    dates = list(meas_by_date.keys())
    pdate = lambda x: datetime.strptime(x, "%Y-%m-%d")
    date_diffs = [(pdate(date) - pdate(d)).days if (pdate(date) - pdate(d)).days > 0 else np.inf for d in meas_by_date.keys()]
    meas = meas_by_date[dates[np.argmin(date_diffs)]]
    zooms = np.sort(np.unique(meas[0]))
    readings_x = [np.mean([meas[1][i] for i in range(0, len(meas[1])) if meas[0][i] == z]) for z in zooms]
    readings_y = [np.mean([meas[2][i] for i in range(0, len(meas[2])) if meas[0][i] == z]) for z in zooms]
    interp_x_inv = scipy.interpolate.interp1d(zooms, 1 / np.asarray(readings_x))
    interp_y_inv = scipy.interpolate.interp1d(zooms, 1 / np.asarray(readings_y))
    return (1 / interp_y_inv(zoom) / 512, 1 / interp_x_inv(zoom) / 512)


# Wheel/treadmill conversion information
# (date of change, diameter of the wheel in cm, number of ticks per revolution, description)
RUNNING_MEASUREMENTS = {
    "b2": [
        ("2023-07-07", 17.0, 4096, "Changed to mesh wheel"),
        ("2022-08-11", 20.0, 4096, "Changed to thin styrofoam wheel"),
        ("1971-01-01", 15.0, 4096, "Don't know if this is correct, got it from Lauren's old code"),
    ],
    "bscope": [("1971-01-01", 18.0, 2988, "Ball")],
}


def get_speed_scale_to_cm(rig, date):
    """Return distance traveled in cm per tick on the rotary encoder.

    This is necessary because the size of the wheel can change in the rig.
    """
    if rig not in RUNNING_MEASUREMENTS.keys():
        raise IndexError(f"Could not find measurements for the rig {rig}")
    d = datetime.strptime(date, "%Y-%m-%d")
    wheel_change_dates = [datetime.strptime(v[0], "%Y-%m-%d") for v in RUNNING_MEASUREMENTS[rig]]
    diffs = [(dc - d).days if (dc - d).days >= 0 else np.inf for dc in wheel_change_dates]
    if len(diffs) == 0:
        raise IndexError(f"No valid speed measurements on rig {rig} for date {date}")
    m = RUNNING_MEASUREMENTS[rig][np.argmin(diffs)]
    return np.pi * m[1] / m[2]


# Visual angle of the edges of the full screen, y then x
SCREEN_VISUAL_ANGLE = {"b2": [(-38, 38), (-135, 135)]}

_CACHE_TIMESTAMP = round(time.time())

# Workaround: we need to use an internal scipy function for smoothing, but this
# function changed in scipy 1.10.  First choose the correct parameterization,
# and then run a test to make sure the output is correct for future scipy
# versions.  If not, you'll need to add another condition here to make this
# compatible with your version of scipy.
if packaging.version.parse("1.10.0") > packaging.version.parse(scipy.__version__):
    gke = scipy.stats._stats.gaussian_kernel_estimate["double"]
else:
    gke = lambda points, values, xi, precision, dtype: scipy.stats._stats.gaussian_kernel_estimate["double"](
        points, values, xi, scipy.linalg.cholesky(scipy.linalg.inv(precision)), dtype
    )
_res = gke(
    np.asarray([0.5], dtype="double")[:, None],
    np.asarray([2.0], dtype="double")[:, None],
    np.asarray([0.1, 0.4, 0.5, 1, 2], dtype="double")[:, None],
    np.asarray([[1 / 0.3**2 / 2]], dtype="double"),
    "double",
).flatten()
assert np.all(
    np.isclose(_res, [1.20582432, 1.82911105, 1.88063195, 0.93909693, 0.00363047])
), "Internal scipy function changed, pixease cannot be used"


#################### SECTION: Utility functions ####################


def _multiglob(paths, expr, one=False, recursive=False):
    """Perform a separate glob for `expr` for each path and return all matches.

    If `one` is True, then ensure there is only one match and return that.
    """
    res = [x for y in [glob.glob(p + expr, recursive=recursive) for p in paths] for x in y]
    if one:
        assert len(res) != 0, f"No matching files found for {paths} and {expr}"
        assert len(res) < 2 or all(filecmp.cmp(res[0], r) for r in res[1:]), f"Multiple matching files found for {paths} and {expr}:\n\n{res}"
        return res[0]
    return res


def _get_explog(mouse, date, mode=""):
    """Get the parsed json from the explog and the path to the json file"""
    json_paths = _multiglob([DP.format(mouse=mouse, date=date, expnum=".") for DP in DATA_PATHS], EXPLOG_PATH)
    if len(json_paths) == 0:
        raise RuntimeError(f"Experiment log not found for mouse {mouse} on date {date}")
    json_path = json_paths[0] if len(json_paths) == 1 else next(p for p in json_paths if mode in os.path.split(p)[1])
    if not os.path.isfile(json_path):
        raise RuntimeError("Could not find explog json file")
    with open(json_path, "r") as f:
        s = json.load(f)
    # Fix bug: if "experiments" is not a list, then make it a list
    if not isinstance(s["experiments"], list):
        s["experiments"] = [s["experiments"]]
    return s, json_path


def _auto_detect_rig(args):
    """Detect the rig using the metadata from explog"""
    s, _ = _get_explog(args.mouse, args.date, args.mode)
    if "rig" in s.keys() and s["rig"].lower() in CHANNELMAP.keys():
        return s["rig"].lower()
    raise RuntimeError(f"Invalid rig specified in explog json file: {s['rig'] if 'rig' in s else 'none'}")


def _auto_detect_exptype(args, expnum):
    """Detect the experiment type using the metadata from explog"""
    s, _ = _get_explog(args.mouse, args.date, args.mode)
    for exp in s["experiments"]:
        try:
            if int(exp["number"]) != int(expnum):
                continue
        except ValueError:  # If expnum or int(expnum) is wrong
            print(f"Skipping {exp['number']} and {expnum}")
            continue
        if "pipeline" not in exp.keys():
            raise RuntimeError(f"Pipeline was not specified for experiment {expnum} in explog.json")
        pipeline = exp["pipeline"].lower()
        break
    else:
        raise RuntimeError(f"Experiment number {expnum} not found in explog.json")
    for e in EXPERIMENT_TYPES.values():
        if pipeline.lower() in [e.NAME] + e.ALTERNATIVE_NAMES:
            return e.NAME
    raise RuntimeError(f"Invalid experiment type '{pipeline}' specified in explog.json file for experiment {expnum}.")


def _auto_detect_gcamp(args):
    """Detect the type of gcamp used from the mouseinfo.json file"""
    json_path = _multiglob([DP.format(mouse=args.mouse, date=".", expnum=".") for DP in DATA_PATHS], MOUSEINFO_PATH, one=True)
    if not os.path.isfile(json_path):
        raise RuntimeError("Could not autodetect gcamp type, mouseinfo.json file not found")
    with open(json_path, "r") as f:
        s = json.load(f)
    genotype = s["genotype"] if isinstance(s["genotype"], list) else [s["genotype"]]
    for x in genotype + s["viruses"]:
        m = re.match(".*gcamp([45678][smf]).*", x.lower())
        if m is not None:
            print("Detected", m[1])
            return m[1]
    raise RuntimeError("gcamp type not found in mouseinfo.json file")


def _auto_detect_mode(mouse, date):
    """Infer acquisition mode (2p or ephys) by checking for two-photon TIFF files.

    Args:
        mouse: Mouse identifier used in data paths and cache names.
        date: Session date string (`YYYY-MM-DD`).
    """
    paths = list(sorted(_multiglob([DP.format(mouse=mouse, date=date, expnum="*") for DP in DATA_PATHS], "*2P*.tif")))
    if len(paths) == 0:
        return "ephys"
    else:
        return "2p"


def _auto_detect_expnums(mouse, date, mode):
    """Detect the sequence of experiment numbers.

    This compiles a list of the valid experiment numbers from the experiment.
    It separates the subsequent experiments by a '-' if they have the same fov
    and can be merged when detecting cells, or ',' if they cannot be merged.
    This is the same input format expected by pixease for preprocessing.

    Args:
        mouse: Mouse identifier used in data paths and cache names.
        date: Session date string (`YYYY-MM-DD`).
        mode: Acquisition mode (`2p`, `ephys`, or `None` to infer ephys behavior).
    """
    # Get the experiment log and the correct directory
    explog, jsonpath = _get_explog(mouse, date, mode)
    p = Path(jsonpath).parent
    # This is easy if the mode is ephys: just find all of the experiments which
    # don't have a failure reason listed.
    if mode == "ephys" or mode is None:
        return "-".join([str(int(e["number"])) for e in explog["experiments"] if not e["success"]])

    # Get all of the experiment ids, and for each, the path to the first tiff
    exps = list(sorted([x for x in p.glob("*") if re.match("[0-9]+", x.name)], key=lambda x: int(x.name)))
    first_tiff = [(int(exp.name), list(sorted(exp.glob("*.tif")))[0]) for exp in exps if len(list(exp.glob("*.tif"))) > 0]

    compare = lambda x, y, key: x[key] == y[key]
    exp_string = ""
    prev = None
    prev_expnum = None
    end_string = ""
    # Loop through and see if we can merge.  Construct the input to pixease as exp_string.
    for expnum, tfile in first_tiff:
        # If it is not in explog, skip
        try:
            explog_curr = next(e for e in explog["experiments"] if e["number"] == expnum)
        except StopIteration:
            continue
        # If this failed, skip it.  Note that the key named "success" actually
        # describes the failure reason.
        if explog_curr["success"]:
            continue
        # Load the tiff and extract the metadata
        t = tifffile.TiffFile(tfile)
        curr = t.scanimage_metadata["FrameData"]
        if prev is None:  # First one in sequence
            prev = curr
            prev_expnum = expnum
            explog_prev = explog_curr
            exp_string += str(expnum)
            continue
        # Experiment types which are always on their own and added at the end.
        # Just pretend they don't exist for all other aspects of this algorithm.
        disjoined_types = ["functional_zstack", "structural_zstack", "redcell"]
        if explog_curr["pipeline"] in disjoined_types:
            end_string += "," + str(expnum)
            continue
        # First check basic properties.
        if (
            compare(prev, curr, "SI.hStackManager.numSlices")
            and compare(prev, curr, "SI.hRoiManager.linesPerFrame")
            and compare(prev, curr, "SI.hRoiManager.pixelsPerLine")
            and compare(prev, curr, "SI.hRoiManager.scanZoomFactor")
        ):
            # We have to do a big setup to see if the planes are in the same
            # position.  Sometimes scanimage does not save the axes position.
            try:
                pos_curr = curr["SI.hMotors.axesPosition"]
                pos_prev = prev["SI.hMotors.axesPosition"]
            except KeyError:
                pos_curr = np.asarray([0, 0, 0])
                pos_prev = np.asarray([0, 0, 0])
            if np.all(np.asarray(pos_curr) == 0) or np.all(np.asarray(pos_prev) == 0):
                try:
                    pos_curr = [float(explog_prev["x"]), float(explog_prev["y"]), float(explog_prev["z"])]
                    pos_prev = [float(explog_curr["x"]), float(explog_curr["y"]), float(explog_curr["z"])]
                except ValueError:  # Entered incorrectly in explog, so force a separate experiment
                    pos_curr = np.random.randn(3) * 1000
                    pos_prev = np.random.randn(3) * 1000
            # Now compare the plane positions
            if np.sum(np.abs(np.asarray(pos_curr) - pos_prev)) < 5:
                # Conclude we can merge these experiments
                prev = curr
                prev_expnum = expnum
                explog_prev = explog_curr
                exp_string += "-" + str(expnum)
                continue
        # Conclude we can't merge the experiments
        prev = curr
        prev_expnum = expnum
        explog_prev = explog_curr
        exp_string += "," + str(expnum)
    print(f"Auto-detected experiment number string {exp_string+end_string}")
    return exp_string + end_string


def _smooth(times, spikes, target_times, width):
    """Generic spike smoothing for a timeseries using convolution with a gaussian.

    Works for univariate and multivariate spikes, though the timing of those
    spikes should be synchronised, as in calcium imaging or similar.  Input can
    be irregularly spaced, output can be irregularly spaced and different from
    input.

    `times` - Array of spike times (shape T)
    `spikes` - Array of spike heights (1s for APs) (shape NxT)
    `target_times` - Array of timepoints for smoothed data (
    `width` - Standard deviation of Gaussian smoothing

    This will work with most data, but is best for large matrices of
    deconvolved calcium spikes.  Use _smooth_univariate for ephys spikes or for
    a single timeseries, _smooth_fast for regularly spaced inputs and outputs
    (e.g. eye tracking), _smooth_interp for ΔF/F or raw calcium traces (which
    performs interpolation instead of smoothing), and
    _smooth_univariate_nochunk for small univariate timeseries.  I don't know
    why anyone would want to use _smooth_multivariate_nochunk.
    """
    # This takes a long time for long lists of times, so we split it into
    # chunks with some overlap based on the width of the standard deviation.
    # But in the case that there aren't very many spikes or target times, we
    # run it without chunking.
    times = np.asarray(times).astype("double")
    spikes = np.asarray(spikes).astype("double")
    target_times = np.asarray(target_times).astype("double")
    if spikes.ndim == 1:
        # Both functions work for univariate smoothing but _smooth_univariate
        # is faster
        if len(times) < 100:
            return _smooth_univariate_nochunk(times, spikes, target_times, width)
        else:
            return _smooth_univariate(times, spikes, target_times, width)
        # spikes = spikes[:,None]
    elif spikes.ndim == 2:
        spikes = spikes.T
    else:
        raise ValueError("Invalid spikes dimensions")
    if times.ndim != 1:
        raise ValueError("Times must be the same for all spikes")
    # if len(times)<100:
    #    print(times.shape, spikes.shape, target_times.shape)
    #    return _smooth_multivariate_nochunk(times, spikes, target_times, width)
    # Filter out zeros and sort by time
    o = np.argsort(times)
    spikes = spikes[o]
    times = times[o]
    ind = np.any((spikes != 0) & (~np.isnan(spikes)), axis=1)
    spikes = spikes[ind]
    times = times[ind]
    # Split into chunks based on smoothing width size
    chunk_width = width * 14  # Determined empirically through benchmarks
    chunk_overlap = width * 7
    _chunk_spacing = chunk_width - chunk_overlap
    chunk_borders = (
        target_times[0]
        - chunk_overlap / 2
        + _chunk_spacing * np.arange(0, (target_times[-1] - target_times[0] + chunk_overlap / 2) / _chunk_spacing - 1)
    )
    if len(chunk_borders) == 0:
        chunk_borders = np.asarray([target_times[0] - chunk_overlap / 2])
    split_points_chunks = []
    for b in chunk_borders:
        _locs_start = np.where(times > b)
        _locs_stop = np.where(times > b + chunk_width)
        locs_start = _locs_start[0][0] if len(_locs_start[0]) > 0 else split_points_chunks[-1][1] if len(split_points_chunks) > 0 else 0
        locs_stop = _locs_stop[0][0] if len(_locs_stop[0]) > 0 else len(times) - 1  # if len(times) > 0 else 0
        split_points_chunks.append((locs_start, locs_stop))
    # split_points_chunks = [(np.where(times>b)[0][0],(np.where(times>b+chunk_width)[0][0] if times[-1]>b+chunk_width else len(times)-1)) for b in chunk_borders]
    split_points_chunks[-1] = (split_points_chunks[-1][0], len(times))
    chunks_times = [times[ss:se] for ss, se in split_points_chunks]
    chunks_spikes = [spikes[ss:se] for ss, se in split_points_chunks]
    split_points_tt = [
        (np.where(target_times >= b)[0][0], np.where(np.concatenate([target_times, [np.inf]]) > b + chunk_width)[0][0]) for b in chunk_borders
    ]
    split_points_tt[-1] = (split_points_tt[-1][0], len(target_times))
    chunks_target_times = [target_times[(target_times >= b) & (target_times < b + chunk_width)] for b in chunk_borders]
    chunks_target_times[-1] = target_times[(target_times >= chunk_borders[-1])]
    chunk_tss = []
    for cspikes, ctimes, ctarget_times in zip(chunks_spikes, chunks_times, chunks_target_times):
        # Note: This is slower with numba
        ret = gke(ctimes[:, None], cspikes, ctarget_times[:, None], np.asarray([[1 / (width**2 / 2)]], dtype="double"), "double") / np.sqrt(2)
        chunk_tss.append(ret)
    mean_tt_split_point = np.mean(split_points_tt, axis=1)
    # closest_tt = [np.argmin(abs(t-(chunk_borders+chunk_width/2))) for t in target_times]
    closest_tt = np.argmin(abs(np.asarray(target_times)[:, None] - (chunk_borders + chunk_width / 2)), axis=1)
    subinds = [i - split_points_tt[closest][0] for i, closest in enumerate(closest_tt)]
    tss = np.asarray([chunk_tss[c][i] for c, i in zip(closest_tt, subinds)]).T
    return tss


def _smooth_multivariate_nochunk(times, spikes, target_times, width):
    """Smoothing a multivariate timeseries with a gaussian.

    This directly uses the _scipy method like _smooth, but doesn't use
    chunking.  In practice, you probably don't want to use this function unless
    you are Max.
    """
    if spikes.ndim == 2:
        ind = np.any((spikes != 0) & (~np.isnan(spikes)), axis=0)
    else:
        ind = (spikes != 0) & (~np.isnan(spikes))
    times = np.asarray(times).astype("double")[ind]
    assert times.ndim == 1, "Times dims is not 1"
    if spikes.ndim == 1:
        spikes = np.asarray(spikes).astype("double")[ind][:, None]
    else:
        spikes = np.asarray(spikes).astype("double")[:, ind].T
    target_times = np.asarray(target_times).astype("double")
    assert target_times.ndim == 1, "Target times dim is not 1"
    ret = gke(times[:, None], spikes, target_times[:, None], np.asarray([[1 / (width**2 / 2)]], dtype="double"), "double") / np.sqrt(2)
    return ret.T.squeeze()


def _smooth_univariate_old(times, spikes, target_times, width, wmult=15):
    """Generic smoothing for a univariate timeseries with Gaussian convolution.

    This function is similar to _smooth except it tends to be faster for
    univariate timeseries.
    """
    # If smoothing is off, just use a histogram.  Note that the
    # timeseries length will be one less than the target_times, since target_times will be interpreted as bin positions.
    if width == 0:
        return np.histogram(a=times, bins=target_times, density=False, weights=spikes)[0]
    # This takes a long time for long lists of times, so we split it into
    # chunks with some overlap based on the width of the standard deviation.
    # But in the case that there aren't very many spikes or target times, we
    # run it without chunking.
    if len(times) * len(target_times) < 500000:
        # If we don't have many spikes or target times, skip chunking
        return _smooth_univariate_nochunk(times, spikes, target_times, width)
    o = np.argsort(times)
    chunk_width = width * wmult
    chunk_overlap = width * 7
    _chunk_spacing = chunk_width - chunk_overlap
    chunk_borders = (
        target_times[0]
        - chunk_overlap / 2
        + _chunk_spacing * np.arange(0, (target_times[-1] - target_times[0] + chunk_overlap / 2) / _chunk_spacing - 1)
    )
    split_points_chunks = [
        (
            np.where(times[o] > b)[0][0] if times[o][-1] > b else len(times[o]) - 1,
            (np.where(times[o] > b + chunk_width)[0][0] if times[o][-1] > b + chunk_width else len(times[o]) - 1),
        )
        for b in chunk_borders
    ]
    split_points_chunks[-1] = (split_points_chunks[-1][0], len(times[o]))
    chunks_times = [times[o][ss:se] for ss, se in split_points_chunks]
    chunks_spikes = [spikes[o][ss:se] for ss, se in split_points_chunks]
    split_points_tt = [
        (np.where(target_times >= b)[0][0], np.where(np.concatenate([target_times, [np.inf]]) > b + chunk_width)[0][0]) for b in chunk_borders
    ]
    split_points_tt[-1] = (split_points_tt[-1][0], len(target_times))
    chunks_target_times = [target_times[(target_times >= b) & (target_times < b + chunk_width)] for b in chunk_borders]
    chunks_target_times[-1] = target_times[(target_times >= chunk_borders[-1])]
    chunk_tss = []
    ONE_OVER_WIDTH_TIMES_SQRT_2_PI = 1 / (width * np.sqrt(2 * 3.1415926535))
    for cspikes, ctimes, ctarget_times in zip(chunks_spikes, chunks_times, chunks_target_times):
        # Note: This is slower with numba
        ind = (cspikes != 0) & (~np.isnan(cspikes))
        s = cspikes[ind].reshape(1, -1) * ONE_OVER_WIDTH_TIMES_SQRT_2_PI
        t = ctimes[ind].reshape(1, -1)
        tt = ctarget_times.reshape(-1, 1)
        m = s * np.exp(-(((tt - t) / width) ** 2))
        chunk_tss.append(np.sum(m, axis=1))
    mean_tt_split_point = np.mean(split_points_tt, axis=1)
    # closest_tt = [np.argmin(abs(t-(chunk_borders+chunk_width/2))) for t in target_times]
    closest_tt = np.argmin(abs(np.asarray(target_times)[:, None] - (chunk_borders + chunk_width / 2)), axis=1)
    subinds = [i - split_points_tt[closest][0] for i, closest in enumerate(closest_tt)]
    tss = [chunk_tss[c][i] for c, i in zip(closest_tt, subinds)]
    return tss


def _smooth_univariate(times, spikes, target_times, width):
    """Generic smoothing for a univariate timeseries with Gaussian convolution.

    This function is similar to _smooth except it tends to be faster for
    univariate timeseries.  Assume all times, and target_times are sorted.
    """
    # Algorithm works by finding "pivot points".  Pivot points are the first
    # timepoint that a chunk is assigned to, and the last timepoint that the
    # previous chunk was assigned to.  Chunks always occur at the timing of
    # spikes, except possible the first and last chunk boundaries.  Chunks have
    # a minimum size.  In order to properly represent the timepoints,
    # computations need to consider spikes from up to 7 standard deviations of
    # the smoothing width before and after the chunk's boundaries (its two pivot
    # points).
    times = np.asarray(times)
    spikes = np.asarray(spikes)
    target_times = np.asarray(target_times)
    wmult = 15  # Determined empirically to be the fastest
    chunk_min_width = width * wmult
    chunk_min_overlap = width * 7
    if np.any(np.isnan(spikes)) or np.any(spikes == 0):
        inds = (spikes != 0) & (~np.isnan(spikes))  # Only work with non-zero spikes
        spikes = spikes[inds]
        times = times[inds]
    pivots = [target_times[0]]
    while True:
        pind = np.searchsorted(times, pivots[-1] + chunk_min_width)
        if pind == len(times) or times[pind] >= target_times[-1]:
            pivots.append(target_times[-1])
            break
        pivots.append(times[pind])
    # Pre-allocate the output tiemeseries, and then fill it up by computing it
    # step by step.
    output_timeseries = np.zeros_like(target_times)
    ONE_OVER_WIDTH_TIMES_SQRT_2_PI = 1 / (width * np.sqrt(2 * 3.1415926535))
    for i in range(1, len(pivots)):
        chunk_times_ind_start = np.searchsorted(times, pivots[i - 1] - chunk_min_overlap)
        chunk_times_ind_end = np.searchsorted(times, pivots[i] + chunk_min_overlap)
        chunk_spikes = spikes[chunk_times_ind_start:chunk_times_ind_end].reshape(1, -1)
        chunk_times = times[chunk_times_ind_start:chunk_times_ind_end].reshape(1, -1)
        chunk_target_times_ind_start = np.searchsorted(target_times, pivots[i - 1])
        chunk_target_times_ind_end = np.searchsorted(target_times, pivots[i]) + 1
        chunk_target_times = target_times[chunk_target_times_ind_start:chunk_target_times_ind_end].reshape(-1, 1)
        ts = ONE_OVER_WIDTH_TIMES_SQRT_2_PI * chunk_spikes * np.exp(-(((chunk_target_times - chunk_times) / width) ** 2))
        output_timeseries[chunk_target_times_ind_start:chunk_target_times_ind_end] = np.sum(ts, axis=1)
    return output_timeseries


def _smooth_univariate_nochunk(times, spikes, target_times, width):
    """Generic smoothing for a univariate timeseries with Gaussian convolution.

    This function is similar to _smooth_univariate except it does not break
    long timeseries into smaller chunks, so it will be very slow for long
    timeseries.
    """
    # Note: This is slower with numba
    ind = (spikes != 0) & (~np.isnan(spikes))
    s = spikes[ind].reshape(1, -1)
    t = times[ind].reshape(1, -1)
    tt = target_times.reshape(-1, 1)
    m = s * np.exp(-(((tt - t) / width) ** 2)) / (width * np.sqrt(2 * 3.1415926535))
    return np.sum(m, axis=1)


def _smooth_fast(x, y, target_times, width=0.3, replacenan=False):
    """Fast smoothing method, alternative to _smooth.

    This requires that the domain x is uniformly spaced.  Performs numerical
    instead of analytical convolution.

    WARNING: Not sure if this actually works?
    """
    width = width / np.sqrt(2)
    if np.any(np.isnan(y)) and replacenan:
        y = y.copy()
        y[np.isnan(y)] = 0
    dt = np.median(np.abs(np.diff(x)))
    assert np.std(np.abs(np.diff(x))) < 0.001, "Domain not uniformly spaced"
    domain = np.arange(-width * 4, width * 4 + 1e-7, dt)
    gauss = scipy.stats.norm(0, width).pdf(domain)
    correction_len = len(gauss) // 2
    y_corrected = np.concatenate([[y[0]] * correction_len, y, [y[-1]] * correction_len])
    smoothed = scipy.signal.oaconvolve(y_corrected, gauss, mode="same")[correction_len:-correction_len] / np.sum(gauss)
    return smoothed[np.minimum(np.searchsorted(x, target_times), len(smoothed) - 1)]


def _smooth_fast_nans(times, vals, target_times, width=0.3, replacenan=False):
    """Fast smoothing method, alternative to _smooth.

    Basic algorithm: Get nans, linearly interpolate nans, convolve with a
    gaussian, linearly interpolate to get target_times, and replace all times
    which fall in a "nan" region with nan.

    This requires that the domain x is uniformly spaced.  Performs numerical
    instead of analytical convolution.

    The "replacenan" argument currently does nothing.
    """
    assert len(times) == len(vals)
    vals = vals.copy()
    # Handle the case where the first or last value is nan.  Set it to non-nan,
    # and then save it in the "extranans" list so we can set it back to nan at
    # the end.
    extranans = []
    if np.isnan(vals[0]):
        vals[0] = vals[~np.isnan(vals)][0]
        extranans.append(0)
    if np.isnan(vals[-1]):
        vals[-1] = vals[~np.isnan(vals)][-1]
        extranans.append(-1)
    # First linearly interpolate the nan values
    nans = np.isnan(vals)
    nanblocks_start = np.where(np.diff(nans.astype(int)) == 1)[0]
    nanblocks_end = np.where(np.diff(nans.astype(int)) == -1)[0] + 1
    assert len(nanblocks_start) == len(nanblocks_end), "Nans in eye data and start and end not yet supported"
    inds = np.arange(0, len(times))
    vals[nans] = scipy.interpolate.interp1d(inds[~nans], vals[~nans])(inds[nans])
    # Convolve with a gaussian
    width = width / np.sqrt(2)
    dt = np.median(np.abs(np.diff(times)))
    assert np.std(np.abs(np.diff(times))) < 0.001, "Domain not uniformly spaced"
    domain = np.arange(-width * 4, width * 4 + 1e-7, dt)
    gauss = scipy.stats.norm(0, width).pdf(domain)
    correction_len = len(gauss) // 2 + 1
    y_corrected = np.concatenate([[vals[0]] * correction_len, vals, [vals[-1]] * correction_len])
    smoothed = scipy.signal.oaconvolve(y_corrected, gauss, mode="same")[correction_len:-correction_len] / np.sum(gauss)
    # Linearly interpolate to get desired timepoints
    ts = scipy.interpolate.interp1d(times, smoothed, fill_value=np.nan, bounds_error=False)(target_times)
    # Make sure interpolated values don't end up in the returned timeseries
    for bs, be in zip(nanblocks_start, nanblocks_end):
        if be - bs <= 2:
            continue  # Interpolate if it is a short run of nans
        ts[(target_times > times[bs]) & (target_times < times[be])] = np.nan
    for en in extranans:
        ts[en] = np.nan
    return ts


def _smooth_interp(times, spikes, target_times, width):
    """Interpolation without smoothing.

    Just like _smooth except performs interpolation and does not smooth.
    Extremely fast.  Good for densely sampled data with high SNR, or data which
    has already been smoothed.

    Args:
        times: Observation timestamps corresponding to `spikes`.
        spikes: One-dimensional trace or two-dimensional cell-by-time matrix.
        target_times: Timepoints at which values should be interpolated.
        width: Smoothing width in seconds (ignored by this implementation).
    """
    if width != 0:
        print("Warning, `width` parameter ignored, not smoothing")
    spikes = np.asarray(spikes)
    if spikes.ndim == 1:
        return np.interp(target_times, times, spikes).squeeze()
    elif spikes.ndim == 2:
        return np.asarray([np.interp(target_times, times, spikes[i]).squeeze() for i in range(0, spikes.shape[0])])


def _bin_spikes(times, spikes, target_times):
    """Get firing rate through binning, not smoothing.

    `times` should be a vector of the times of each spike in seconds, `spikes`
    should be a list of spike heights of the same length as `times`, and
    `target_times` should be a vector of times at which the binning should be
    performed.  Returns the firing rate, in spikes per second, during each bin.

    Note that `target_times` does not need to be uniformly spaced.  This means
    bin size will not be homogeneous.  Bins are constructed such that any data
    point in between two times in `target_times` will go to the closest one.
    (Think one-dimensional Voronoi.)  The first and last bins in target_times
    will be symmetric around the centre point, i.e., incorporating spikes from
    just outside the given range.

    This assumes `target_times` is sorted.
    """
    first_bin_start = target_times[0] - (target_times[1] - target_times[0]) / 2
    last_bin_end = target_times[-1] + (target_times[-1] - target_times[-2]) / 2
    _bin_boundaries = target_times[:-1] + (target_times[1:] - target_times[:-1]) / 2
    bin_boundaries = np.concatenate([[first_bin_start], _bin_boundaries, [last_bin_end]])
    hist, _ = np.histogram(times, bins=bin_boundaries, weights=spikes)
    bin_widths = bin_boundaries[1:] - bin_boundaries[:-1]
    return hist / bin_widths


def _find_subclasses(cls):
    """Look for all of the subtypes of a class.

    There may be subclasses of the subclass, though, so try to recurse down one
    more level as well.  Anything with a name counts.  Names must be unique.

    Args:
        cls: Base class whose descendants should be enumerated.
    """
    TYPES = {}
    _queue = cls.__subclasses__()
    while len(_queue) > 0:
        c = _queue.pop()
        if hasattr(c, "NAME"):
            assert c.NAME not in TYPES.keys()
            TYPES[c.NAME] = c
        _queue.extend(c.__subclasses__())
    return TYPES


def _get_source_file():
    """Return the full `pixease.py` source text as a string."""
    with open(__file__, "r", encoding="utf8") as f:
        text = f.read()
    return text


def _confirm_overwrite(msg=""):
    """Prevent the --force option from being used accidentally"""
    print("WARNING: Using the --force tag")
    if msg:
        print("Previous output from", msg, "will be overwritten.")
    print("==== To abort, press Ctrl+C within the next 10 seconds. ===")
    time.sleep(10)


def _string_hash(s):
    """Create a short deterministic hash used for naming temporary cache directories.

    Args:
        s: Input string to hash.
    """
    return hashlib.md5(s.encode("utf8")).hexdigest()[0:10]


def _load_npy_compressed(path, *args, **kwargs):
    """Load a NumPy array from either plain `.npy` or gzip-compressed `.npy.gz`.

    Any .npy file can be compressed and saved as a .npy.gz file and this
    function will open it transparently.

    Right now it doesn't support zstd because it isn't standard on most
    computers and there isn't a standard library python package for it.  Sadly,
    because it sure would be nice.

    Args:
        path: Base path to the `.npy` file, with or without `.gz`.
        *args: Positional arguments forwarded to `numpy.load`.
        **kwargs: Keyword arguments forwarded to `numpy.load`.
    """
    if os.path.isfile(path):
        return np.load(path, *args, **kwargs)
    if os.path.isfile(path + ".gz"):
        return np.load(gzip.GzipFile(path + ".gz"), *args, **kwargs)
    raise FileNotFoundError(f"Could not find {path} or {path}.gz")


def _save_npy_compressed(path, *args, **kwargs):
    """Save a NumPy array in gzip-compressed `.npy.gz` form.

    Args:
        path: Filesystem path to read from or write to.
    """
    f = gzip.open(path + ".gz", "wb")
    np.save(f, *args, **kwargs)


def load_with_patch(path, *args, **kwargs):
    """Load a .mat or .npy file while allowing a patch to load it in a different way.

    The point of patches is that, if data is corrupted or irregular, you may
    want to load data differently before passing it to pixease.  If no patch is
    present, this function will load the file using np.load() or
    scipy.io.loadmat().  It passes all of its arguments to these.  If the patch
    is present, it tries to load the file with the patch's "loader()" function
    instead.  If this loader() function in the patch returns None, it will fall
    back to np.load() or scipy.io.loadmat().  If the function loader() returns
    anything else, it will return this.  (Functions in Python which do not
    encounter a "return" statement return None by default.)

    There are two places it looks for patches: the same directory as the file it
    is trying to load, and the directory above that.  It will first try the same
    directory before trying the directory above.  They should be named
    "pixease_patch.py".

    The idea is that, when you want to use a patch, the loader() function should
    return None for all files except the one you want to modify.  For that, it
    should call np.load() or scipy.io.loadmat() with all the passed args and
    kwargs, make modifications, and return it.

    E.g., here is a simple patch to double the amplitude of the photodiode.

    def loader(path, *args, **kwargs):
        if "photoDiode.raw.npy" in path:
            diode = np.load(path, *args, **kwargs)
            diode *= 2
            return diode
    """
    p = Path(path)
    if p.parent.parent.joinpath("pixease_patch.py").is_file():
        with open(p.parent.parent.joinpath("pixease_patch.py"), "r") as f:
            _patch = f.read()
        namespace = {}
        exec(_patch, namespace)
        assert "loader" in namespace.keys(), "Must define the 'loader' function in pixease_patch.py"
        print("WARNING: This experiment has been patched at the session level to correct an error during data acquisition.")
        loader_parent = namespace["loader"]
    else:
        loader_parent = lambda *args, **kwargs: None
    if p.parent.joinpath("pixease_patch.py").is_file():
        with open(p.parent.joinpath("pixease_patch.py"), "r") as f:
            _patch = f.read()
        namespace = {}
        exec(_patch, namespace)
        assert "loader" in namespace.keys(), "Must define the 'loader' function in pixease_patch.py"
        print("WARNING: This experiment has been patched at the experiment level to correct an error during data acquisition.")
        loader = namespace["loader"]
    else:
        loader = lambda *args, **kwargs: None
    loaded = loader(path, *args, **kwargs)
    if loaded is None:
        loaded = loader_parent(path, *args, **kwargs)
    if loaded is None:
        if p.suffix == ".npy":
            loaded = np.load(path, *args, **kwargs)
        elif p.suffix == ".mat":
            loaded = scipy.io.loadmat(path, *args, **kwargs)
    return loaded


def phase_cross_correlation_regularized(reference_image, moving_image, regularization=1):
    # Lower regularization number is weaker regularization
    # images must be the same shape
    """Estimate whole-pixel translation between two images using phase correlation with spatial regularization.

    Args:
        reference_image: Fixed reference image for phase-correlation alignment.
        moving_image: Image to shift into alignment with `reference_image`.
        regularization: Strength of spatial regularization in phase-correlation scoring.
    """
    if reference_image.shape != moving_image.shape:
        raise ValueError("images must be same shape")
    src_freq = scipy.fft.fftn(reference_image)
    target_freq = scipy.fft.fftn(moving_image)
    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    # Phase normalisation
    eps = np.finfo(image_product.real.dtype).eps
    image_product /= np.maximum(np.abs(image_product), 100 * eps)
    # Compute cross correlation
    cross_correlation = scipy.fft.ifftn(image_product)
    # TODO: This isn't the best form of the regularization matrix in the world
    # but hopefully it will be good enough.
    regularization_matrix = 1 / (
        1
        + np.sqrt(
            np.minimum(np.arange(0, shape[0]), shape[0] - np.arange(0, shape[0]))[:, None] ** 2
            + np.minimum(np.arange(0, shape[1]), shape[1] - np.arange(0, shape[1]))[None, :] ** 2
        )
        * regularization
        * 100
    )

    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation) * regularization_matrix), cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])
    float_dtype = image_product.real.dtype
    shifts = np.stack(maxima).astype(float_dtype, copy=False)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0
    return shifts


def fib_flipper(n_frames):
    """Create a repeatable flipper sequence.

    This sequence follows the fibonacci numbers.  Create a massive list of
    subsequent fibonacci numbers, and concatenate the string representations.
    Insert a number of off (i.e. false or 0) values equal to the first number
    in the string.  Then insert the number of on (i.e. true or 1) values for
    the next number.  And so on.  Skip any zeros.  So, it starts:

        1123581321

    And corresponds to:

        010011100000111111110111001

    and so on.  The idea is that this should create a flipper-like signal, and
    it should be trivial to reproduce this sequence in any programming language.

    Note that, since zeros are skipped, they do not cause a flip.  So, there
    should be no runs longer than 9.  This is done to reduce the
    autocorrelation.  So if the number "303" showed up, it would correspond to
    "111000" or "000111" rather than "111111" or "000000".

    Args:
        n_frames: Number of frames to generate in the output sequence.

    Returns a boolean numpy array of length `n_frames`.
    """

    @functools.lru_cache(None)
    def fib(n):
        """Compute the Fibonacci number used to synthesize the deterministic flipper pattern.

        Args:
            n: Integer index for sequence generation.
        """
        if n < 2:
            return 1
        if n % 2 == 1:
            return fib((n + 1) // 2 - 1) * (2 * fib((n + 1) // 2) - fib((n + 1) // 2 - 1))
        a, b = fib(n // 2 - 1), fib(n // 2)
        return a**2 + b**2

    frames = [False]
    i = 1
    while len(frames) < n_frames:
        digs = list(str(fib(i)))
        for d in digs:
            frames.extend([not frames[-1]] * (int(d) + 0))
        i += 1
    return np.asarray(frames[0:n_frames])


def boolean_index(inds, length):
    """Convert a scalar/list of indices into a boolean mask of a fixed length.

    Args:
        inds: Indices to activate in the returned boolean mask.
        length: Length of the returned boolean mask.
    """
    if isinstance(inds, (float, int)):
        inds = [inds]
    arr = np.zeros(length).astype(bool)
    arr[inds] = True
    return arr


def subindex(cells1, cells2):
    """Combine indices of different shapes.

    If `cells1` is a list of booleans of length N with M values equal to True,
    and `cells2` is a list of booleans of length M with K values equal to True,
    then subindex will return a boolean index of length N where the True values
    in `cells1` are replaced by the values of `cells2`, such that there are K
    values equal to true.

    Args:
        cells1: Boolean mask in the original index space.
        cells2: Boolean mask in the subspace defined by `cells1 == True`.
    """
    assert cells1.dtype == cells2.dtype == np.bool
    assert len(cells1) > len(cells2)
    assert sum(cells1) == len(cells2)
    inds1 = np.where(cells1)[0]
    newinds = inds1[cells2]
    newcells = np.zeros(len(cells1)).astype(bool)
    newcells[newinds] = True
    return newcells


def _interpret_cells_argument(cells, max_num):
    """Allow subsets of cells to be specified in different ways.

    Many functions have an argument called "cells" that allows them to operate
    only on a subset of cells.  This is better than subindexing the output
    because sometimes these operations are slow, and especially if you only need
    one cell, this can save a considerable amount of time.

    The "cells" argument can be specified as:

    - A list of cell ids
    - A boolean index
    - None

    This function will validate the argument and return a boolean index vector
    of length "max_num".
    """
    if isinstance(cells, np.ndarray) and cells.dtype == bool:
        assert cells.shape == (max_num,), "Invalid shape for cell specification"
        return cells
    if isinstance(cells, np.ndarray) or isinstance(cells, list):
        assert np.asarray(cells).ndim == 1
        assert np.max(cells) < max_num and np.min(cells) >= 0, 'A cell id in "cells" argument is out of range'
        assert np.all(cells == np.sort(cells)), "List of cells must be explicitly sorted"
        return boolean_index(cells, max_num)
    if cells is None:
        return np.ones(max_num).astype(bool)
    raise ValueError('Invalid form for "cells" argument')


def _blit(source, target, loc):
    """Copy the n-dimensional image "source" onto "target" at position "loc"."""
    source_size = np.asarray(source.shape)
    target_size = np.asarray(target.shape)
    # If we had infinite boundaries, where would we put it?
    target_loc_tl = loc
    target_loc_br = target_loc_tl + source_size
    # Compute the index for the source
    source_loc_tl = -np.minimum(0, target_loc_tl)
    source_loc_br = source_size - np.maximum(0, target_loc_br - target_size)
    # Recompute the index for the target
    target_loc_br = np.minimum(target_size, target_loc_tl + source_size)
    target_loc_tl = np.maximum(0, target_loc_tl)
    # Compute slices from positions
    target_slices = [slice(s1, s2) for s1, s2 in zip(target_loc_tl, target_loc_br)]
    source_slices = [slice(s1, s2) for s1, s2 in zip(source_loc_tl, source_loc_br)]
    # Perform the blit
    target[tuple(target_slices)] = source[tuple(source_slices)]


def load(mouse, date, expnum, from_dir="."):
    """Load an experiment using the mouse's name, date, and experiment number.

    By default, this will look in the current directory for the cache file.  The
    optional parameter `from_dir` will search in a different directory.
    """
    # Use this functionality to automatically select the appropriate subclass
    # if only one cache file exists.
    matches = glob.glob(os.path.join(from_dir, NOTEBOOK_FILE.format(mouse=mouse, date=date, expnum=expnum, exptype="*")))
    if len(matches) == 1:
        m = re.match(
            ".*" + re.escape(NOTEBOOK_FILE.format(mouse=mouse, date=date, expnum=expnum, exptype="XXXXXXXX")).replace("XXXXXXXX", "(.*)"), matches[0]
        )
        assert m is not None, "Could not match cache file filename format"
        for et_name, et in EXPERIMENT_TYPES.items():
            if m.group(1) == et_name or m.group(1) in et.ALTERNATIVE_NAMES:
                return et(mouse=mouse, date=date, expnum=expnum, from_dir=from_dir)
    if len(matches) > 1:
        raise NotImplementedError(
            "Multiple matching cache files, please call the appropriate experiment type directly:\n    " + "\n    ".join(matches)
        )
    raise IOError(f"Could not find a cache file with mouse={mouse}, date={date}, expnum={expnum}")


#################### SECTION: Save and load ####################

# These strings are chosen for backward compatibility with the old notebook format.
_SAVE_SECTION_SEP = "_-_"  # Separate section name from value name
_SAVE_TYPEMETA = "___TYPE"  # Filename suffix for type information
_SAVE_SKIP = ["TIME_CREATED", "PAGEINFO", "___TIME"]  # For backward compatibility


# Since numpy converts all types to arrays before saving in an npz, we need to
# keep track of what type things were before they were saved, saved that type
# alongside the value in the npz, and then cast it back into that type upon
# loading.
def _get_save_type(v):
    """Map a Python value to the lightweight type tag stored in cache files.

    Args:
        v: Value to inspect/convert for cache serialization.
    """
    types = [("list", list), ("string", (str, np.str_)), ("int", (int, np.int_)), ("number", (float, np.float_)), ("ndarray", np.ndarray)]
    for t in types:
        if isinstance(v, t[1]):
            return t[0]
    raise ValueError(f"Could not find type of value {v} type {type(v)}")


def _cast_save_type(v, typ):
    """Reconstruct a Python value from its cached array representation and saved type tag.

    Args:
        v: Value to inspect/convert for cache serialization.
        typ: Serialized type tag used during cache deserialization.
    """
    types = [
        ("list", lambda x: list(x)),
        ("string", lambda x: str(x[()])),
        ("int", lambda x: int(x[()])),
        ("number", lambda x: float(x[()])),
        ("ndarray", lambda x: x),
    ]
    for t in types:
        if typ == t[0]:
            return t[1](v)
    raise ValueError(f"Could not decode type {typ} of variable {v}")


def save_cachefile(filename, data):
    """Serialize a nested cache dictionary to `.npz` while preserving original value types.

    Args:
        filename: Output filename or cache filename, depending on context.
    """
    sections = [k for k in data.keys()]
    towrite = {}
    for section in sections:
        for varname, val in data[section].items():
            towrite[section + _SAVE_SECTION_SEP + varname] = val
            towrite[section + _SAVE_SECTION_SEP + varname + _SAVE_TYPEMETA] = _get_save_type(val)
    np.savez_compressed(filename, **towrite)


def load_cachefile(filename):
    """Load a cache file written by `save_cachefile` and restore its original nested structure.

    Args:
        filename: Output filename or cache filename, depending on context.
    """
    f = np.load(filename)
    data = {}
    realkeys = set([k.replace(_SAVE_TYPEMETA, "") for k in f.keys() if all(s not in k for s in _SAVE_SKIP)])
    for k in realkeys:
        section, key = k.split(_SAVE_SECTION_SEP)
        if section not in data.keys():
            data[section] = {}
        data[section][key] = _cast_save_type(f[k], str(f[k + _SAVE_TYPEMETA][()]))
    return data


#################### SECTION: Basic classes ####################


class Module:
    """Base class for Module, should not be used directly.

    Modules are sets of data from the experiment which come from a single
    place.  They have a few jobs:

    - Extract the necessary information from the original data files
    - Perform basic preprocessing on the files
    - Compress the data before saving
    - Extract the compressed data
    - Provide useful functions for working with the data

    The first three of these are handled by the ``prepare`` static method, and
    the last two are by the ``__init__`` function.

    Attributes:
    name : str
        The name of the Module.  The Module will always be referred to by this
        name.

    Methods:
    prepare(timeline, im_meta, rig, data_paths, processed_data_path, expnum)
        This static method is the main method for extracting, preprocessing,
        and compressing data.  All arguments are optional and can be excluded
        with a **kwargs at the end.  They will be passed by keyword so ordering
        is optional.  This method should extract the relevant information and
        do the relevant preprocessing, and then return a dictionary containing
        the data which will be saved directly in the cache file.  All keys of
        the returned dictionary and their values will then be passed to the
        constructor.
    __init__(self, general_info, ...)
        The constructor should take arguments corresponding to the names of the
        values returned by ``prepare``.  It also receives the ``general_info``
        argument, which contains useful information such as the rig, the
        mouse name, etc.  This should uncompress the data (if necessary) and
        save it to the object itself.  It may also perform computations which
        are needed by othre features.
    """

    MODE = ["2p", "ephys"]
    EXPORT = []


class Mixin:
    """Base class for Mixin, should not be used directly.

    Mixins are useful computations which are needed across a wide variety of
    stimuli.  They may depend on multiple Modules, which must be included
    within the experiment for the mixin to be valid.

    Attributes:
    REQUIRED_MODULES : list of strs
        A list of the names of the modules which are essential for the Mixin to
        function correctly
    OPTIONAL_MODULES : list of strs
        A list of the names of the modules which may be used by the Mixin
    EXPORT : list of strs
        A list of the names of methods which should be top-level within the
        Experiment.  For example, if the experiment X had the Mixin Y where
        Y.EXPORT == ['Z'], then the method Z can be called as X.Z instead of
        X.Y.Z.

    Notes:
    As an alternative to EXPORT, the Mixin can simply define the __call__
    function, which will allow the Mixin to be called directly.  In general,
    use __call__ if the Mixin only has one piece of functionality, or EXPORT if
    it has more than one.
    """

    MODE = ["2p", "ephys"]
    EXPORT = []
    REQUIRED_MODULES = {"all": []}
    OPTIONAL_MODULES = []
    AVAILABLE_MODULES = []
    REQUIRED_MIXINS = []
    OPTIONAL_MIXINS = []

    def prepare(self):
        """Hook for mixins to precompute state after required modules/mixins are attached."""
        pass

    def __init__(self, modules, mode, mixins):
        """Attach required/optional modules and mixins, then run `prepare()` to finalize mixin state.

        Args:
            modules: Already-instantiated module objects keyed by module name.
            mode: Acquisition mode (`2p` or `ephys`).
            mixins: Already-instantiated mixin objects keyed by mixin name.
        """
        self.mode = mode
        if isinstance(self.REQUIRED_MODULES, list):
            mods = self.REQUIRED_MODULES
        elif isinstance(self.REQUIRED_MODULES, dict):
            mods = self.REQUIRED_MODULES[mode] if mode in self.REQUIRED_MODULES else []
            mods += self.REQUIRED_MODULES["all"] if "all" in self.REQUIRED_MODULES else []
        for m in mods:
            if m.NAME not in modules.keys():
                raise RuntimeError(f"Module {m.NAME} is not in the experiment's module list, needed for mixin {self.NAME}")
            setattr(self, m.NAME, modules[m.NAME])
        for m in self.OPTIONAL_MODULES + self.AVAILABLE_MODULES:
            if m.NAME in modules.keys():
                setattr(self, m.NAME, modules[m.NAME])
        # TODO This isn't as flexible as modules, because it only works if the
        # mixin has been added already.  There isn't logic to order it in terms
        # of mixin dependencies.  Also, you can't specify required or optional
        # mixins, they all currently just get thrown into the same pot.  Also,
        # this is very messy.
        for m in mixins.keys():
            if mixins[m].NAME in [_m.NAME for _m in self.REQUIRED_MIXINS + self.OPTIONAL_MIXINS]:
                setattr(self, mixins[m].NAME, mixins[m])
        self.prepare()


class Experiment:
    """Base class that assembles experiment modules, mixins, and cached outputs.

    Should not be used directly.
    """

    NAME = ""
    MODULES = {"all": []}
    OPTIONAL_MODULES = []
    AVAILABLE_MODULES = []
    MIXINS = {"all": []}
    OPTIONAL_MIXINS = []
    ALTERNATIVE_NAMES = []

    @classmethod
    def generate_notebook(cls, mouse, date, expnum, expgroup, rig, mode, force=False, regroup=False, skip=[], include=[]):
        """Build or load the cache notebook for one experiment instance.

        This resolves data paths, chooses pipelines/modules/mixins for the
        requested mode, runs missing preprocessing steps, and then serializes a
        cache file that can be loaded quickly later.

        Args:
            mouse: Mouse identifier used in data paths and cache names.
            date: Session date string (`YYYY-MM-DD`).
            expnum: Experiment number within the session.
            expgroup: Grouped experiment-id string used for joint preprocessing.
            rig: Rig identifier used for channel maps and calibration.
            mode: Acquisition mode (`2p`, `ephys`, or `auto`).
            force: If true, overwrite any existing notebook cache.
            regroup: If true, do not reuse existing grouped Suite2p outputs.
            skip: Pipeline names to skip explicitly.
            include: Pipeline names to run in addition to defaults.
        """
        # Cache output file
        notebook_file = os.path.join(OUTPUT_DIR, NOTEBOOK_FILE.format(mouse=mouse, date=date, expnum=expnum, exptype=cls.NAME))
        if os.path.exists(notebook_file) and force:
            os.unlink(notebook_file)
        if os.path.exists(notebook_file):
            return load_cachefile(notebook_file)
        # Get paths to the data
        data_paths = [DP.format(mouse=mouse, date=date, expnum=expnum) for DP in DATA_PATHS]
        mouseinfo_paths = [DP.format(mouse=mouse, date=".", expnum=".") for DP in DATA_PATHS]
        explog_paths = [DP.format(mouse=mouse, date=date, expnum=".") for DP in DATA_PATHS]
        processed_data_path = PROCESSED_DATA_PATH.format(mouse=mouse, date=date, expnum=expnum)
        # Detect ephys or 2p
        mode = _auto_detect_mode(mouse, date) if mode == "auto" else mode
        # If we are not regrouping, we can reuse other processed data.  Choose
        # the one with the largest number of sessions processed at once.
        # TODO This doesn't work and needs to be redone TODO TODO TODO TODO
        potential_s2p_paths = [dp.format(mouse=mouse, date=date, expnum=f"pixease/{expgroup}") for dp in DATA_PATHS] + [
            PROCESSED_DATA_PATH.format(mouse=mouse, date=date, expnum=expgroup)
        ]
        processed_s2p_path = next(p for p in potential_s2p_paths if Path(p).exists())
        if mode == "2p" and processed_s2p_path and not Path(processed_s2p_path).joinpath("suite2p").exists() and not regroup and "-" not in expgroup:
            potential_processed_s2p_paths = glob.glob(PROCESSED_DATA_PATH.format(mouse=mouse, date=date, expnum="*/suite2p"))
            valid_processed_s2p_paths = [p for p in potential_processed_s2p_paths if expgroup in Path(p).parent.name.split("-")]
            if len(valid_processed_s2p_paths) > 1:
                print(f"Reusing data from grouping {valid_processed_s2p_paths[-1]} instead of {valid_processed_s2p_paths[:-1]}")
            if len(valid_processed_s2p_paths) == 1:
                sorted_valid_paths = sorted(valid_processed_s2p_paths, key=lambda x: len(Path(x).parent.name.split("-")))
                processed_s2p_path = str(Path(sorted_valid_paths[-1]).parent.joinpath("")) + os.path.sep
                expgroup = Path(sorted_valid_paths[-1]).parent.name
                print("Using group", expgroup)
        # Support for a mode (ephys or 2p) is determined by whether there is a
        # key for it in the PIPELINES dict.  If "all" is a key, every mode is
        # supported.
        if mode not in cls.PIPELINES and "all" not in cls.PIPELINES:
            raise RuntimeError(f"The experiment {cls.NAME} does not support {mode}.")
        modules = list(cls.MODULES[mode] if mode in cls.MODULES else []) + list(cls.MODULES["all"] if "all" in cls.MODULES else [])
        mixins = list(cls.MIXINS[mode] if mode in cls.MIXINS else []) + list(cls.MIXINS["all"] if "all" in cls.MIXINS else [])
        pipelines = list(cls.PIPELINES[mode] if mode in cls.PIPELINES else []) + list(cls.PIPELINES["all"] if "all" in cls.PIPELINES else [])
        # Now make sure all modules/mixins/pipelines are compatible with the mode.
        for m in modules:
            if mode not in m.MODE and "all" not in m.MODE and mode != m.MODE and m.MODE != "all":
                raise RuntimeError(f"This is a {mode} experiment but the module {m.NAME} requires {m.MODE}.")
        for m in mixins:
            if mode not in m.MODE and "all" not in m.MODE and mode != m.MODE and m.MODE != "all":
                raise RuntimeError(f"This is a {mode} experiment but the mixin {m.NAME} requires {m.MODE}.")
        for p in pipelines:
            if mode not in PIPELINES[p][3] and "all" not in PIPELINES[p][3] and mode != PIPELINES[p][3] and PIPELINES[p][3] != "all":
                raise RuntimeError(f"This is a {mode} experiment but the pipeline {p} requires {m.MODE}.")

        # Check to see if the source data is here.  If not, we are probably on
        # the wrong computer or specified an invalid experiment.
        if not any(os.path.isdir(DP.format(mouse=".", date=".", expnum=".")) for DP in DATA_PATHS):
            raise RuntimeError("Subjects directory not found: are you running this on the correct computer?")
        if not any(os.path.isdir(d) for d in data_paths):
            raise RuntimeError("Invalid mouse, date, or experiment number")

        explog_paths = _multiglob(explog_paths, EXPLOG_PATH)
        explog_path = explog_paths[0] if len(explog_paths) == 1 else next(p for p in explog_paths if mode in p)
        mouseinfo_path = _multiglob(mouseinfo_paths, MOUSEINFO_PATH, one=True)
        # Load data for alignment
        tl_path = _multiglob(data_paths, "*_Timeline.mat", one=True)
        timeline = load_with_patch(tl_path, squeeze_me=True, struct_as_record=False)["Timeline"]
        system = "signals" if len(_multiglob(data_paths, "*expDef.m")) > 0 else "mpep"
        # All arguments that a module could need
        args = {
            "timeline": timeline,
            "rig": rig,
            "data_paths": data_paths,
            "explog_path": explog_path,
            "mouseinfo_path": mouseinfo_path,
            "processed_data_path": processed_data_path,
            "mouse": mouse,
            "date": date,
            "expnum": expnum,
            "expgroup": expgroup,
            "expid": f"{mouse}_{date}_{expnum}",
            "mode": mode,
            "system": system,
        }
        if mode == "2p":
            im_paths = _multiglob(data_paths, "*.tif")
            im_meta = tifffile.TiffReader(im_paths[0]).scanimage_metadata["FrameData"]
            args["im_meta"] = im_meta
            args["processed_s2p_path"] = processed_s2p_path
        elif mode == "ephys":
            params_file_paths = _multiglob(data_paths, "../**/params.py", recursive=True)
            ephys_paths = [os.path.dirname(params_file_path) for params_file_path in params_file_paths]
            args["ephys_paths"] = ephys_paths

        # List off all patches
        patches_paths = _multiglob(data_paths, "pixease_patch.py") + _multiglob(data_paths, "../pixease_patch.py")
        patches = ""
        for p in patches_paths:
            with open(p, "r") as f:
                patches += f"# {p}\n"
                patches += f.read()
                patches += "\n\n\n"
        # Build the notebook from the sub-modules
        pages = {}
        for m in modules:
            pages[m.NAME] = m.prepare(**args)
        # Create a general info page which all modules will have access to.
        _general_info = {
            "mouse": args["mouse"],
            "date": args["date"],
            "expnum": args["expnum"],
            "expid": args["expid"],
            "expgroup": args["expgroup"],
            "rig": args["rig"],
            "patches": patches,
            "cache_timestamp": _CACHE_TIMESTAMP,
            "pixease_source": _get_source_file(),  # Old name
            "mode": mode,
            "system": system,
        }
        if mode == "2p":
            # TODO: Confirm x and y are not swapped
            _general_info["fov"] = [
                args["im_meta"]["SI.hRoiManager.pixelsPerLine"],
                args["im_meta"]["SI.hRoiManager.linesPerFrame"],
                args["im_meta"]["SI.hStackManager.numSlices"],
            ]
            _general_info["zoom"] = args["im_meta"]["SI.hRoiManager.scanZoomFactor"]
            try:
                _general_info["position"] = im_meta["SI.hMotors.axesPosition"]  # I think this is only reliable in B2.  Also it is backwards.
            except KeyError:
                pass
        elif mode == "ephys":
            pass  # TODO anything here?
        pages["general_info"] = _general_info
        # Optional modules.  Good to have but not necessary.
        for m in cls.OPTIONAL_MODULES:
            if m.NAME in skip or (mode != m.MODE and mode not in m.MODE):
                continue
            try:
                pages[m.NAME] = m.prepare(**args)
            except Exception as e:
                print(f"Skipping {m.NAME} because of exception:\n    {e}")
        for m in cls.AVAILABLE_MODULES:
            if m.NAME in include:
                pages[m.NAME] = m.prepare(**args)
        save_cachefile(notebook_file, pages)

    # Create a cache file.  This should only be called by subclasses.
    def __init__(self, mouse, date, expnum, from_dir="."):
        """Load a cache file, instantiate modules/mixins, and expose their exported methods on the experiment object.

        Args:
            mouse: Mouse identifier used in data paths and cache names.
            date: Session date string (`YYYY-MM-DD`).
            expnum: Experiment number within the session.
            from_dir: Directory where cache files are searched.
        """
        if self.NAME == "" or len(self.MODULES) == 0:
            raise NotImplementedError("You must subclass Experiment")
        # Test if we already have a notebook
        notebook_files = [
            os.path.join(from_dir, NOTEBOOK_FILE.format(mouse=mouse, date=date, expnum=expnum, exptype=name))
            for name in [self.NAME] + self.ALTERNATIVE_NAMES
        ]
        notebook_file = next((f for f in notebook_files if os.path.isfile(f)), None)
        assert notebook_file is not None, f"Cache file '{notebook_files[0]}' or alternatives {notebook_files[1:]} not found"
        nb = load_cachefile(notebook_file)
        # Create objects to simplify access to processed data
        self.modules = {}
        general_info = nb["general_info"]
        mode = general_info["mode"] if "mode" in general_info.keys() else "2p"
        if mode == "2p":
            general_info["fov"] = [
                general_info["fov"][2],
                general_info["fov"][0],
                general_info["fov"][1],
            ]  # Correct a bug where fov was saved x-y-z instead of z-x-y
            if "position" in general_info.keys():
                general_info["position"] = general_info["position"][::-1]  # Correct a bug where position was saved as (x,y,z) instead of (z,y,x)
        modules = list(self.MODULES[mode] if mode in self.MODULES else []) + list(self.MODULES["all"] if "all" in self.MODULES else [])
        mixins = list(self.MIXINS[mode] if mode in self.MIXINS else []) + list(self.MIXINS["all"] if "all" in self.MIXINS else [])
        for m in modules + [GeneralInfo]:
            mod = m(**nb[m.NAME], general_info=general_info)
            self.modules[m.NAME] = mod
            setattr(self, m.NAME, mod)
            for e in mod.EXPORT:
                setattr(self, e, getattr(mod, e))
        for m in self.OPTIONAL_MODULES + self.AVAILABLE_MODULES:
            if m.NAME in nb.keys():
                try:
                    mod = m(**nb[m.NAME], general_info=general_info)
                    self.modules[m.NAME] = mod
                    setattr(self, m.NAME, mod)
                    for e in mod.EXPORT:
                        setattr(self, e, getattr(mod, e))
                except Exception as e:
                    print(f"Skipping {m.NAME} because of exception:\n    {e}")

            else:
                if m in self.OPTIONAL_MODULES:
                    print(f"Skipping {m.NAME} because it is not in the cache file")
        # Now create mixins based on the modules
        self.mixins = {}
        for m in mixins:
            mixin = m(self.modules, mode, self.mixins)
            self.mixins[m.NAME] = mixin
            setattr(self, m.NAME, mixin)
            for e in mixin.EXPORT:
                setattr(self, e, getattr(mixin, e))
        for m in self.OPTIONAL_MIXINS:
            if mode not in m.MODE and mode != m.MODE:
                print(mode, "not supported by", m.NAME)
                continue
            try:
                mixin = m(self.modules, mode, self.mixins)
            except Exception as e:
                print(f"Skipping mixin {m.NAME} because of exception:\n    {e}")
            else:
                self.mixins[m.NAME] = mixin
                setattr(self, m.NAME, mixin)
                for e in mixin.EXPORT:
                    setattr(self, e, getattr(mixin, e))

    def summary(self):
        """Print a concise human-readable summary of the experiment metadata."""
        e = self.explog.exp
        g = self.general_info
        assert "explog" in self.modules.keys(), "Need explog to have a summary"
        s = f"""
        === {e['expname']} ===
        Type: {self.NAME} ({self.mode}, pipeline "{e['pipeline']}")
        Mouse: {g.mouse}
        Session: {g.date} exp {g.expnum} ({g.expid})
        Duration: {e['duration']}
        Rig: {g.rig} ({g.zoom} zoom)
        FOV: {g.fov[0]} x {g.fov[1]} x {g.fov[2]} ({e['spacing']} μm spacing)
        Laser: {e['laserwavelength']}nm ({e['laser_percentage']}% power)
        Position: {g.position if "position" in self.general_info.__dict__ else (e['x'], e['y'], e['z'], e['angle']+" degrees")}
        Notes:
            {e['notes'].replace(os.linesep, os.linesep+"            ")}
        """
        print(textwrap.dedent(s[1:]))


#################### SECTION: Modules ####################


class GeneralInfo(Module):
    """General information, included by default for all experiments"""

    NAME = "general_info"
    EXPORT = ["mouse", "date", "expid", "expnum", "mode"]

    # Note: There is no prepare method here because this is a special module
    # and is handled separately from the others.
    def __init__(self, general_info, **kwargs):
        """Populate basic session metadata fields from the `general_info` cache section.

        Args:
            general_info: General metadata dict from the cache file.
        """
        for k, v in general_info.items():
            setattr(self, k, v)
        if "expid" not in general_info.keys():
            self.expid = f"{general_info['mouse']}_{general_info['date']}_{general_info['expnum']}"


class Diode(Module):
    """The photodiode for synchronising visual stimuli.

    Currently supports information about a binary status for the photodiode,
    either on or off.
    """

    NAME = "diode"

    @staticmethod
    def prepare(timeline, **_):
        """Detect photodiode on/off transitions from timeline data and return flip times with transition direction.

        Args:
            timeline: Timeline structure containing DAQ channels and timestamps.
        """
        timeline_channels = {i.name: i.arrayColumn - 1 for i in timeline.hw.inputs}
        d = timeline.rawDAQData[:, timeline_channels["photoDiode"]]
        # High pass filter/make the timeseries stationary-ish by diffing, and
        # then convolve since it may sometimes take longer to hit a peak for
        # some frames than others
        dflat = np.convolve(np.diff(scipy.signal.medfilt(d, 5)), [1] * 20, mode="same")
        # Remove small fluctuations as noise
        dflat[np.abs(dflat) < np.max(np.abs(dflat)) * 0.3] = (
            0  # This was changed from 0.3 to 0.7 on 2023-11-01 due to 2023-10-19_6_MS016 # Changed to np.max(np.abs(dflat))/2 on 2024-08-20 due to BZ007_2024-05-02_12 # CHanged from .5 to .3 on 2024-08-24 due to BZ008_2024-06-27_12
        )
        # To find both diode-on and diode-off events, take the absolute value.
        # The run the scipy algorithm for finding peaks.  Distance=4
        # corresponds to 8 ms.  This was tested on b2 only.
        dpeaks = scipy.signal.find_peaks(np.abs(dflat), width=5, prominence=0.4, distance=4)[
            0
        ]  # Width changed from 10 to 5 on 2024-08-20 due to BZ007_2024-05-02_12
        dpeaks_direction = (dflat[dpeaks] > 0) * 2 - 1
        # Map the indices detected here back onto the actual timestamps from the DAQ
        peaktimes = timeline.rawDAQTimestamps[dpeaks]
        return {"diode_flip_times": peaktimes, "diode_flips": dpeaks_direction}
        # Old way of doing this:
        # timeline_channels = {i.name: i.arrayColumn-1  for i in timeline.hw.inputs}
        # diode_ts = timeline.rawDAQData[:,timeline_channels['photoDiode']]
        # diode_thresh = 0.5*(np.max(diode_ts)+np.min(diode_ts))
        # _diode_flips = np.concatenate([[0], np.diff((diode_ts>diode_thresh).astype(int))])
        # diode_flips = _diode_flips[_diode_flips!=0]
        # diode_flip_times = timeline.rawDAQTimestamps[np.abs(_diode_flips).astype(bool)]
        # return {"diode_flips": diode_flips, "diode_flip_times": diode_flip_times}

    def __init__(self, diode_flips, diode_flip_times, general_info):
        """Store photodiode transition directions and timestamps.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.diode_flips = diode_flips
        self.diode_flip_times = diode_flip_times

    def flip_times(self):
        """Return timestamps for all detected photodiode transitions."""
        return self.diode_flip_times

    def on_times(self):
        """Return timestamps where the diode transitions into the ON state."""
        return self.diode_flip_times[self.diode_flips == 1]

    def off_times(self):
        """Return timestamps where the diode transitions into the OFF state."""
        return self.diode_flip_times[self.diode_flips == -1]

    def on_durations(self):
        """Pair each ON transition with its following OFF transition."""
        assert np.sum(self.diode_flips == 1) == np.sum(
            self.diode_flips == -1
        ), "Invalid flip sequence, did diode never turn off at the end of the experiment?  Or did it start on?"
        return list(zip(self.on_times(), self.off_times()))

    def match_onoff_sequence(self, N):
        """Return ON/OFF intervals and assert the expected number of intervals `N`.

        Args:
            N: Expected number of intervals/events used for consistency checks.
        """
        durations = self.on_durations()
        assert len(durations) == N
        return durations

    def match_flip_sequency(self, N):
        """Expand alternating ON/OFF transitions into contiguous stimulus-state intervals and validate length `N`.

        Args:
            N: Expected number of intervals/events used for consistency checks.
        """
        times = []
        on = self.on_times()
        off = self.off_times()
        for i in range(0, len(on) - 1):
            times.append((on[i], off[i]))
            times.append((off[i], on[i + 1]))
        if len(on) == len(off):
            times.append((on[i + 1], off[i + 1]))
        elif len(on) == len(off) + 1:
            pass
        else:
            raise ValueError("Invalid flips")
        assert len(times) == N, f"Invalid N count, got {N} but should be {len(times)}"
        return times

    def find_groups(self, gaptime=0.4):
        # Find long periods of silence, and call that a new stimulus set
        """Split photodiode transitions into groups separated by long temporal gaps.

        Args:
            gaptime: Minimum temporal gap (seconds) used to separate groups.
        """
        groupings = (
            [0] + list(1 + np.where(np.diff(self.diode_flips) > gaptime)[0]) + [len(self.diode_flips)]
        )  # >400 ms between frames indicates new stimulus
        groups = [(self.diode_flips[groupings[i]], self.diode_flisps[groupings[i + 1] - 1]) for i in range(0, len(groupings) - 1)]
        # The line above used to be the following.  I'm not sure why the adjacency condition was here - I know it was important
        # groups = [(self.diode_flips[groupings[i]], self.diode_flips[groupings[i+1]-1]) for i in range(0, len(groupings)-1) if groupings[i]!=groupings[i+1]-1]
        return groups

    def group_frame_times(self, gaptime=0.4):
        # Find long periods of silence, and call that a new stimulus set
        """Return per-group photodiode transition arrays separated by long inactivity gaps.

        Args:
            gaptime: Minimum temporal gap (seconds) used to separate groups.
        """
        groupings = (
            [0] + list(1 + np.where(np.diff(self.diode_flips) > gaptime)[0]) + [len(self.diode_flips)]
        )  # >400 ms between frames indicates new stimulus
        groups = [(self.diode_flips[groupings[i] : groupings[i + 1]]) for i in range(0, len(groupings) - 1) if groupings[i] != groupings[i + 1] - 1]
        # groups = [(self.diode_flips[groupings[i]:groupings[i+1]]) for i in range(0, len(groupings)-1) if groupings[i]!=groupings[i+1]-1]
        return groups

    def fib_flipper_sync(self):
        """Return the times of the frames if the stimulus used the fib_flipper() function for the sync square/photodiode.

        Maybe have up to 10 extra frames.

        Currently has about 5ms error.
        """
        # TODO This algorithm could be improved because it doesn't currently
        # account for the possibility of dropped frames from the photodiode.  It
        # also could get MUCH better alignment by finding the corresponding
        # flips and then interpolating between them instead of fitting a
        # regression to everything.
        flip_times = self.flip_times()
        flipper_signal = fib_flipper(len(flip_times) * 10)  # Maximum possible flipper signal length
        _flipper_times = np.where(np.diff(flipper_signal))[0]
        flipper_times = _flipper_times[0 : len(flip_times)]  # Keep correct number of flips
        # Realign in case we are off by a couple
        starting_shift_corrs = [np.corrcoef(np.diff(flip_times)[4 + i : i - 4], np.diff(flipper_times)[4:-4])[0, 1] for i in range(-3, 4)]
        assert np.max(starting_shift_corrs) > 0.99, "A flip from the sync square may have been missed by the photodiode"
        shift = np.argmax(starting_shift_corrs) - 3
        print(len(flipper_times), len(flip_times), len(flipper_times[4:-4]), len(flipper_times[shift + 4 : shift - 4]), shift)
        coefs = np.polyfit(flipper_times[4:-4], flip_times[shift + 4 : shift - 4], 1)
        frame_times_timeline = np.arange(0, np.max(flipper_times) + 10) * coefs[0] + coefs[1]
        # Error is about 1/3 of a 60fps frame (5 ms) at most:
        # plt.hist(flipper_times[4:-4]*coefs[0]+coefs[1]-flip_times[shift+4:shift-4]); plt.show()
        return frame_times_timeline


class DiodeVideo(Module):
    """Use the diode for tracking frames in a video.

    This is deprecated, and all the functionality contained here is contained within Diode.
    """

    NAME = "diode_video"

    @staticmethod
    def prepare(timeline, **_):
        """Extract photodiode edge times for frame-level video synchronization.

        Args:
            timeline: Timeline structure containing DAQ channels and timestamps.
        """
        timeline_channels = {i.name: i.arrayColumn - 1 for i in timeline.hw.inputs}
        d = timeline.rawDAQData[:, timeline_channels["photoDiode"]]
        # High pass filter/make the timeseries stationary-ish by diffing, and
        # then convolve since it may sometimes take longer to hit a peak for
        # some frames than others
        dflat = np.convolve(np.diff(scipy.signal.medfilt(d, 5)), [1] * 20, mode="same")
        # Remove small fluctuations as noise
        dflat[np.abs(dflat) < np.max(np.abs(dflat)) * 0.3] = (
            0  # This was changed from 0.3 to 0.7 on 2023-11-01 due to 2023-10-19_6_MS016 # Changed to np.max(np.abs(dflat))/2 on 2024-08-20 due to BZ007_2024-05-02_12 # CHanged from .5 to .3 on 2024-08-24 due to BZ008_2024-06-27_12
        )
        # To find both diode-on and diode-off events, take the absolute value.
        # The run the scipy algorithm for finding peaks.  Distance=4
        # corresponds to 8 ms.  This was tested on b2 only.
        dpeaks = scipy.signal.find_peaks(np.abs(dflat), width=5, prominence=0.4, distance=4)[
            0
        ]  # Width changed from 10 to 5 on 2024-08-20 due to BZ007_2024-05-02_12
        dpeaks_direction = dflat[dpeaks] > 0
        # Map the indices detected here back onto the actual timestamps from the DAQ
        peaktimes = timeline.rawDAQTimestamps[dpeaks]
        return {"peaks": peaktimes, "peak_directions": dpeaks_direction}

    def __init__(self, peaks, peak_directions, general_info):
        """Store photodiode edge times and edge polarity for video frame grouping.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.peaks = peaks
        self.peak_directions = peak_directions

    def find_groups(self, gaptime=0.4):
        # Find long periods of silence, and call that a new stimulus set
        """Find contiguous photodiode segments corresponding to distinct stimulus blocks.

        Args:
            gaptime: Minimum temporal gap (seconds) used to separate groups.
        """
        groupings = [0] + list(1 + np.where(np.diff(self.peaks) > gaptime)[0]) + [len(self.peaks)]  # >400 ms between frames indicates new stimulus
        groups = [
            (self.peaks[groupings[i]], self.peaks[groupings[i + 1] - 1]) for i in range(0, len(groupings) - 1) if groupings[i] != groupings[i + 1] - 1
        ]
        groups = [g for g in groups if g[1] - g[0] > 1]  # At least 1 sec
        # The line above used to be the following.  I'm not sure why the adjacency condition was here - I know it was important
        # groups = [(self.peaks[groupings[i]], self.peaks[groupings[i+1]-1]) for i in range(0, len(groupings)-1) if groupings[i]!=groupings[i+1]-1]
        return groups

    def group_frame_times(self, gaptime=0.4):
        # Find long periods of silence, and call that a new stimulus set
        """Return edge-time arrays for each contiguous video stimulus block.

        Args:
            gaptime: Minimum temporal gap (seconds) used to separate groups.
        """
        groupings = [0] + list(1 + np.where(np.diff(self.peaks) > gaptime)[0]) + [len(self.peaks)]  # >400 ms between frames indicates new stimulus
        groups = [(self.peaks[groupings[i] : groupings[i + 1]]) for i in range(0, len(groupings) - 1) if groupings[i] != groupings[i + 1] - 1]
        groups = [g for g in groups if g[-1] - g[0] > 1]  # At least 1 sec
        # groups = [(self.peaks[groupings[i]:groupings[i+1]]) for i in range(0, len(groupings)-1) if groupings[i]!=groupings[i+1]-1]
        return groups


# TODO merge with DiodeRaw
class DiodeLevels(Module):
    """Module for quantized multi-level photodiode state transitions."""

    NAME = "diode_levels"

    @staticmethod
    def prepare(timeline, **_):
        """Quantize a multilevel photodiode trace into stable luminance states and transition times.

        Args:
            timeline: Timeline structure containing DAQ channels and timestamps.
        """
        timeline_channels = {i.name: i.arrayColumn - 1 for i in timeline.hw.inputs}
        d = timeline.rawDAQData[:, timeline_channels["photoDiode"]].squeeze()
        # Use a median filter to get rid of artifacts from the screen not being
        # full brightness.  It can't hurt, so I'll put it here instead of in a
        # patch file.
        d = scipy.signal.medfilt(d, 101)
        # Downsample to about 100,000 points points to generate the kde for performance reasons
        spacing = max(len(d) // 100000, 1)
        kde = scipy.stats.gaussian_kde(d[::spacing])
        domain = np.linspace(0, np.ceil(np.max(d)), 501)
        smoothed = kde(domain)
        # Find the center point between the peaks of the kde
        peaks = scipy.signal.find_peaks(smoothed, height=0.001)[0]
        peakvals = domain[peaks]
        bins = [0.5 * (peakvals[i] + peakvals[i + 1]) for i in range(0, len(peakvals) - 1)]
        val = np.digitize(d, bins)
        # Get the times it changed and the value it changed to.  Need to make
        # sure it lasts at least 100 ms.
        ts = timeline.rawDAQTimestamps
        change_vals = [g[0] for g in np.split(val, np.where(np.diff(val) != 0)[0] + 1) if len(g) > 100]
        change_times = [g[0] for g in np.split(ts, np.where(np.diff(val) != 0)[0] + 1) if len(g) > 100]
        return {"value": change_vals, "time": change_times}

    def __init__(self, value, time, general_info):
        """Store quantized diode state sequence and transition timestamps.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.value = value
        self.time = time


# class DiodeGreyTerminated(Module):
#     NAME = "diode_grey"
#     @staticmethod
#     def prepare(timeline, **_):
#         # Look for places where the photodiode is sustained on grey, and then
#         # call regions where it is over 4 sec long new trials, and regions
#         # where it is over 500 ms long "regions".  Align to the start of the
#         # non-grey after the long grey period for the trials, and the beginning
#         # of the grey for the regions.
#         timeline_channels = {i.name: i.arrayColumn-1  for i in timeline.hw.inputs}
#         d = timeline.rawDAQData[:,timeline_channels['photoDiode']].squeeze()
#         d = scipy.signal.medfilt(d, 5)
#         i_grey = np.where(d[:-100]-d[100:] < -1)[0][0]+130
#         grey_val = np.mean(d[i_grey:(i_grey+500)])
#         grey_tol = .2
#         grey_regions = np.logical_and(d > grey_val-grey_tol, d < grey_val + grey_tol)
#         grey_chop_dur = 4000 # 4 sec to cut to a new trial
#         grey_mark_dur = 500 # .5 sec to mark as a spot of interest
#         gs = [(key, len(list(group))) for key,group in itertools.groupby(grey_regions)]
#         gs_ind = np.cumsum([0]+[g[1] for g in gs])
#         grey_blocks = [(i,i+l) for i,(k,l) in zip(gs_ind,gs) if k == True]
#         trial_starts = [b[1] for b in grey_blocks if b[1]-b[0] > grey_chop_dur]
#         region_marks = [b[0] for b in grey_blocks if b[1]-b[0] > grey_mark_dur and b[1] not in trial_starts]
#         return {"trials": trial_starts[:-1], "regions": region_marks}
#     def __init__(self, trials, regions, general_info):
#         self.trials = trials
#         self.regions = regions


class DiodeRaw(Module):
    """Module exposing a denoised but otherwise raw photodiode trace."""

    NAME = "diode_raw"

    @staticmethod
    def prepare(timeline, **_):
        """Load and denoise the raw photodiode signal and timeline timestamps.

        Args:
            timeline: Timeline structure containing DAQ channels and timestamps.
        """
        timeline_channels = {i.name: i.arrayColumn - 1 for i in timeline.hw.inputs}
        d = timeline.rawDAQData[:, timeline_channels["photoDiode"]].squeeze()
        d = scipy.signal.medfilt(d, 5)
        return {"times": timeline.rawDAQTimestamps, "value": d}

    def __init__(self, times, value, general_info):
        """Store raw photodiode timestamps and signal values.

        Args:
            times: Time points (seconds) at which values are requested.
            general_info: General metadata dict from the cache file.
        """
        self.times = times
        self.value = value


# TODO merge with DiodeRaw
class DiodeTriangles(Module):
    """Module for experiments where the photodiode encodes triangular pulse amplitudes."""

    NAME = "diode_triangles"

    @staticmethod
    def prepare(timeline, **_):
        # First, find the peaks of all of the triangles in the photodiode
        """Detect triangular photodiode pulses and estimate precise onset/peak timing for each pulse.

        Args:
            timeline: Timeline structure containing DAQ channels and timestamps.
        """
        timeline_channels = {i.name: i.arrayColumn - 1 for i in timeline.hw.inputs}
        d = timeline.rawDAQData[:, timeline_channels["photoDiode"]].squeeze()
        times = timeline.rawDAQTimestamps
        d = scipy.signal.medfilt(d, 5)
        vals, bins = np.histogram(d, bins=np.linspace(0, 5, 501))
        grey = 0.5 * (bins[np.argmax(vals)] + bins[np.argmax(vals) + 1])
        peak_height = (np.max(d) - np.min(d)) / 3
        peaks = scipy.signal.find_peaks(np.abs(d - grey), height=peak_height, width=1000, distance=1000)[0]
        # Find the midpoint of the period of grey in between the peaks
        grey_centres = []
        for i in range(0, len(peaks) - 1):
            d_range = d[peaks[i] : peaks[i + 1]]
            d_i_grey = np.where(np.abs(d_range - grey) < 0.03)[0]
            grey_centres.append(peaks[i] + d_i_grey[len(d_i_grey) // 2])
        # Use the distances from the first and last centres to the peaks to
        # make sure each peak is surrounded by a nearby grey region
        grey_centres.insert(0, peaks[0] - (grey_centres[0] - peaks[0]))
        grey_centres.append(peaks[-1] + (peaks[-1] - grey_centres[-1]))
        # Fit a relu-triangle-like-thing to each interval surrounded by grey
        # regions.  These are guaranteed to be symmetric.
        widths = []
        precise_peaks = []
        for i in range(0, len(grey_centres) - 1):
            domain = d[grey_centres[i] : grey_centres[i + 1]]
            peakval = d[peaks[i]] - grey

            def triangle_relu(width, pos_adj):
                # Width is width of triangle/2, pos_adj is adjustment for the peak
                """Generate a symmetric triangle waveform used during pulse-shape fitting."""
                ramp = np.linspace(0, 1, int(width) + 1)
                pad_left = (peaks[i] + int(pos_adj)) - int(width) - grey_centres[i]
                pad_right = grey_centres[i + 1] - ((peaks[i] + int(pos_adj)) + int(width)) - 1
                return grey + peakval * np.concatenate([np.zeros(pad_left), ramp[:-1], ramp[::-1], np.zeros(pad_right)])

            bounds = ((1, np.minimum(peaks[i] - grey_centres[i], grey_centres[i + 1] - peaks[i]) - 100), (-100, 100))
            res = scipy.optimize.differential_evolution(lambda x: np.sum(np.square(domain - triangle_relu(x[0], x[1]))), bounds)
            widths.append(int(res.x[0]))
            precise_peaks.append(int(peaks[i] + res.x[1]))
        return {"starts": times[np.asarray(precise_peaks) - widths], "peaks": times[precise_peaks], "sign": np.sign(d[precise_peaks] - grey)}

    def __init__(self, starts, peaks, sign, general_info):
        """Store fitted triangular stimulus onset/peak times and polarity.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.starts = starts
        self.peaks = peaks
        self.sign = sign

    def stimulus(self, min_dur=0.5):
        """Convert triangle widths and polarity into signed stimulus identity values.

        Args:
            min_dur: Base duration step for quantizing triangle stimuli.
        """
        return np.round((self.peaks - self.starts) / min_dur) * min_dur * 2 * self.sign

    def intervals_stimulus(self):
        """Interval during the duration of the stimulus.  All will be different lengths."""
        return [(start, start + 2 * (peak - start)) for start, peak in zip(self.starts, self.peaks)]

    def intervals_from_peak(self, padding=2):
        """Intervals aligned to the peak, padding surrounding the peak in seconds"""
        return [(peak - padding, peak + padding) for peak in self.peaks]


class ExpDef(Module):
    """Wrapper around experiment-definition source text and metadata.

    This works for both mpep and Rigbox.
    """

    NAME = "expdef"

    @staticmethod
    def prepare(data_paths, **_):
        """Load experiment-definition (matlab) source files and optional video metadata for Rigbox or MPEP files.

        Args:
            data_paths: Candidate raw-data directories for the experiment.
        """
        rigbox_expdef = _multiglob(data_paths, "*_expDef.m")
        if len(rigbox_expdef) >= 1:
            with open(rigbox_expdef[0], "r") as f:
                expdef = f.read()
            params_json = _multiglob(data_paths, "*_parameters.json", one=True)
            with open(params_json, "r") as f:
                params = json.load(f)
            # If it is a video, see if there is an associated video information file.
            vids = set([x for st in expdef.split("\n") for x in re.findall(r'^[^\%]*[\'"]([^\'"]+\.mp4)[\'"]', st, flags=re.MULTILINE)])
            video_info = {}
            for v in vids:
                # If this is actually a video file
                if os.path.isfile(VIDEOS_PATH + v):
                    _video_meta = skvideo.io.ffprobe(VIDEOS_PATH + v)["video"]
                    video_meta = repr({k: v for k, v in _video_meta.items() if isinstance(v, str)})
                    # Look for a .txt file
                    txt_files = glob.glob(VIDEOS_PATH + v + "*.txt")
                    txt_file_content = {}
                    for txt_file in txt_files:
                        with open(txt_file, "r") as f:
                            txt_file_content[re.split("[/\\\\]", txt_file)[-1]] = f.read()
                    video_info[v] = (video_meta, repr(txt_file_content))
            return {"expdef": expdef, "filename": params["defFunction"], "system": "rigbox", "video_info": str(video_info)}
        elif len(rigbox_expdef) == 0:
            # Get the name of the x file from the first line of the p file.
            p_file = _multiglob(data_paths, "*.p", one=True)
            with open(p_file, "r") as f:
                x_file = f.readline().strip()
            assert x_file[-2:] == ".x"
            # Get the name of the m file from the x file
            m_file = x_file[:-1] + "m"
            # Get the contents of the m file
            with open(XFILES_PATH + m_file, "r") as f:
                m_file_content = f.read()
            return {"expdef": m_file_content, "filename": m_file, "system": "mpep"}
        raise ValueError("Couldn't find rigbox or mpep experiment")

    def __init__(self, expdef, filename, system, general_info, video_info="{}"):
        """Store experiment-definition text and parse serialized video metadata structures.

        Args:
            filename: Output filename or cache filename, depending on context.
            general_info: General metadata dict from the cache file.
        """
        self.expdef = expdef
        self.filename = filename
        self.system = system
        _eval = lambda x: eval(x, {"inf": np.inf, "nan": np.nan}, globals())
        self.video_meta = {k: (_eval(v[0]), {k1: _eval(v1) for k1, v1 in _eval(v[1]).items()}) for k, v in _eval(video_info).items()}


# TODO merge this with FunctionalF, FunctionalSpikes, etc
class NeuralFrameTimings(Module):
    """Timing information for 2p images.

    This gets the DAQ timing of each 2p image from each plane.
    """

    NAME = "neural_timings"
    MODE = "2p"

    @staticmethod
    def prepare(timeline, im_meta, **_):
        # Extract the signal from timeline
        """Recover per-frame acquisition times from timeline neural frame counters.

        Args:
            timeline: Timeline structure containing DAQ channels and timestamps.
            im_meta: ScanImage frame metadata for the experiment.
        """
        n_planes = im_meta["SI.hStackManager.numSlices"]
        timeline_channels = {i.name: i.arrayColumn - 1 for i in timeline.hw.inputs}
        _neuralframes = timeline.rawDAQData[:, timeline_channels["neuralFrames"]].astype(int)
        assert _neuralframes[0] == 0, "Timeline started after scanimage started recording"
        # Since timeline exports a step function, look at where all of the
        # changes occur to get the timing.  This gives the indices
        neuralframes_start = np.concatenate([[0], np.diff(_neuralframes)]).astype(bool)
        # Now this gets the frame number
        neuralframes = _neuralframes[neuralframes_start]
        # Confirm we have all of the frames.  This ensures we don't need to
        # save "neuralframes", since it can be easily reconstructed.
        assert np.all(neuralframes == list(range(1, len(neuralframes) + 1)))
        # Get the timing of each of the frames
        neuralframes_times = timeline.rawDAQTimestamps[neuralframes_start]
        # Basic sanity check for preprocessing or recording errors, make sure
        # frames are evenly spaced
        frame_spacing = np.std(np.diff(neuralframes_times))
        assert frame_spacing < 0.001, "Frames irregularly spaced: spacing stdev = {frame_spacing}"
        return {"neuralframes_times": neuralframes_times, "n_planes": n_planes}

    def __init__(self, neuralframes_times, n_planes, general_info):
        """Store frame timestamps and derived acquisition cadence information.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.neuralframes_times = neuralframes_times
        self.frame_dur = np.mean(np.diff(neuralframes_times))
        self.n_planes = n_planes

    def frame_times(self, plane=0):
        """Get the times of the observations for a glane.

        Given the plane ID (using zero-based indexing), return an array of the
        times for that plane.

        There will be more frame times than there are tiffs because scanimage
        stops recording before the scanner turns off.
        """
        assert plane >= 0 and plane < self.n_planes, "Invalid plane"
        return self.neuralframes_times[plane :: self.n_planes]

    def n_frames(self):
        """The total number of frames across all planes.

        This will return a larger number than there are tiffs because scanimage
        stops recording before the scanner turns off.
        """
        return len(self.neuralframes_times)


class Audio(Module):
    """Low-bandwidth audio captured from the stimulus monitor output.

    This is low quality audio.
    """

    NAME = "audio"

    @staticmethod
    def prepare(data_paths, **_):
        """Load low-bandwidth monitor audio captured during the experiment.

        Args:
            data_paths: Candidate raw-data directories for the experiment.
        """
        aud = load_with_patch(_multiglob(data_paths, "/audioMonitor.raw.npy", one=True)).flatten()
        return {"audio": aud}

    def __init__(self, audio, general_info):
        """Store monitor-audio samples.

        Args:
            audio: One-dimensional monitor-audio trace from acquisition hardware.
            general_info: General metadata dict from the cache file.
        """
        self.audio = audio


class AudioHQ(Module):
    """High-fidelity microphone audio captured during the experiment.

    This is very high quality audio and will substantially increase the file
    size.
    """

    NAME = "audio_hq"

    @staticmethod
    # TODO Finish this.  Also not sure if I should be resampling since mice can hear higher frequencies...
    def prepare(data_paths, **_):
        """Load high-fidelity microphone recordings captured during the experiment.

        Args:
            data_paths: Candidate raw-data directories for the experiment.
        """
        fn = _multiglob(data_paths, "/*_mic.mat", one=True)
        m = load_with_patch(fn)
        # I think this is better than the default scipy fourier method
        # resampy.resample(x=m['micData'].flatten(), sr_orig=m['Fs'][0][0], sr_new=44100, filter="kaiser_best")
        return {"fs": int(m["Fs"][0][0]), "audiostream": m["micData"].astype(np.int16).flatten()}

    def __init__(self, fs, audiostream, general_info):
        """Store high-fidelity audio samples and sample rate.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.fs = fs
        self.audiostream = audiostream


class MpepEvents(Module):
    """Module wrapping protocol-level MPEP/MC experiment events as structured intervals."""

    NAME = "mpep_events"
    # TODO This was originally written with events not as a numpy array, could
    # be rewritten to make it more efficient
    EVENTS_ENUM = {"ExpStart": 0, "ExpEnd": 1, "BlockStart": 2, "BlockEnd": 3, "StimStart": 4, "StimEnd": 5, "ExpInterrupt": 6}

    @classmethod
    def prepare(cls, timeline, mouse, date, expnum, data_paths, **_):
        """Build a normalized event table for MPEP/MC experiment structure events.

        Args:
            timeline: Timeline structure containing DAQ channels and timestamps.
            mouse: Mouse identifier used in data paths and cache names.
            date: Session date string (`YYYY-MM-DD`).
            expnum: Experiment number within the session.
            data_paths: Candidate raw-data directories for the experiment.
        """
        block_files = _multiglob(data_paths, "/*Block.mat")
        if len(block_files) == 1:  # MC
            block = load_with_patch(block_files[0], struct_as_record=False, squeeze_me=True)["block"]
            actual_mpep_events = [s.split(" ") for s in timeline.mpepUDPEvents if isinstance(s, str) and s != "hello"]
            assert len(actual_mpep_events) == 4, "Mpep events not as expected, pixease is broken"
            trial_start_times = block.events.newTrialTimes
            trial_end_times = block.events.endTrialTimes
            if len(trial_end_times) + 1 == len(trial_start_times):
                trial_end_times = np.concatenate([trial_end_times, [block.events.expStopTimes]])
            trial_durations = trial_end_times - trial_start_times
            fake_mpep_events = []
            fake_mpep_events.append([block.events.expStartTimes, cls.EVENTS_ENUM["ExpStart"], -1, -1, -1])
            fake_mpep_events.append([block.events.expStartTimes, cls.EVENTS_ENUM["BlockStart"], 1, -1, -1])
            for i in range(0, len(trial_start_times)):  # THESE TIMES ARE NOT TIMELINE TIMES!!!  I can't seem to figure out how to convert.
                fake_mpep_events.append([trial_start_times[i], cls.EVENTS_ENUM["StimStart"], -1, i + 1, trial_durations[i]])
                fake_mpep_events.append([trial_end_times[i], cls.EVENTS_ENUM["StimEnd"], -1, i + 1, -1])
            fake_mpep_events.append([block.events.expStopTimes, cls.EVENTS_ENUM["BlockEnd"], 1, -1, -1])
            fake_mpep_events.append([block.events.expStopTimes, cls.EVENTS_ENUM["ExpEnd"], -1, -1, -1])
            return {"mpep_events_matrix": np.asarray(fake_mpep_events)}
        elif len(block_files) == 0:  # Mpep
            # event structure: ["event_type", "animal", "date", "exp_num", "irepeat", "istimulus", "duration"]
            mpep_events = [s.split(" ") for s in timeline.mpepUDPEvents if isinstance(s, str) and s != "hello"]
            assert len(set(r[1] for r in mpep_events)), "Animal not always the same"
            assert len(set(r[2] for r in mpep_events)), "Date not always the same"
            assert len(set(r[3] for r in mpep_events)), "Experiment number not always the same"
            events = []
            for i in range(0, len(mpep_events)):
                ev_text = cls.EVENTS_ENUM[mpep_events[i][0]]
                irepeat = int(mpep_events[i][4]) if len(mpep_events[i]) >= 5 else -1
                istimulus = int(mpep_events[i][5]) if len(mpep_events[i]) >= 6 else -1
                duration = int(mpep_events[i][6]) if len(mpep_events[i]) >= 7 else -1
                events.append([timeline.mpepUDPTimes[i], ev_text, irepeat, istimulus, duration])
            return {"mpep_events_matrix": np.asarray(events)}

    def __init__(self, mpep_events_matrix, general_info):
        """Normalize event streams (including interrupted runs) and cache trial/block identifiers.

        Args:
            general_info: General metadata dict from the cache file.
        """
        events = list(mpep_events_matrix)
        # Handle the case of an interrupted experiment
        if events[-1][1] == self.EVENTS_ENUM["ExpInterrupt"]:
            # If we have a trial open, or a block open with no trials, delete it
            while events[-2][1] in [self.EVENTS_ENUM["StimStart"], self.EVENTS_ENUM["BlockStart"]]:
                events.pop(-2)
            # If we have a trial as the last element, insert a BlockEnd at the time of the interrupt
            if events[-2][1] == self.EVENTS_ENUM["StimEnd"]:
                prev_block = next(e for e in events[::-1] if e[1] == self.EVENTS_ENUM["BlockStart"])
                events.insert(-1, [events[-1][0], self.EVENTS_ENUM["BlockEnd"], prev_block[2], -1, -1])
            # Insert experiment end at the time of the interrupt
            events.insert(-1, [events[-1][0], self.EVENTS_ENUM["ExpEnd"], -1, -1, -1])
        self.events = np.asarray(events)
        self.stimuli = np.asarray(list(sorted(set(self.events[:, 3]) - set([-1]))), dtype=int)
        self.blocks = np.asarray(list(sorted(set(self.events[:, 2]) - set([-1]))), dtype=int)

    def experiment_interval(self):
        """Return the overall experiment start/end interval from event markers."""
        start = next(r[0] for r in self.events if r[1] == self.EVENTS_ENUM["ExpStart"])
        end = next(r[0] for r in self.events if r[1] == self.EVENTS_ENUM["ExpEnd"])
        return (start, end)

    def interrupted(self):
        """Report whether the underlying event stream ended with an interruption marker."""
        return self.events[-1][1] == self.EVENTS_ENUM["ExpInterrupt"]

    def block_intervals(self):
        """Return ordered block intervals as `(block_id, start_time, end_time)` tuples."""
        all_blocks_start = [int(e[2]) for e in self.events if e[1] == self.EVENTS_ENUM["BlockStart"]]
        all_blocks_end = [int(e[2]) for e in self.events if e[1] == self.EVENTS_ENUM["BlockEnd"]]
        # Confirm not only are all the blocks present, but they are in the correct order
        order = list(range(1, 1 + len(all_blocks_start)))
        assert np.all(all_blocks_start == np.asarray(order)), "Missing blocks starting"
        assert np.all(all_blocks_end == np.asarray(order)), "Missing blocks ending"
        # Since we can now assume they are in the right order, just zip them together
        blocks_start_time = [e[0] for e in self.events if e[1] == self.EVENTS_ENUM["BlockStart"]]
        blocks_end_time = [e[0] for e in self.events if e[1] == self.EVENTS_ENUM["BlockEnd"]]
        return list(zip(order, blocks_start_time, blocks_end_time))

    def trials(self):
        """Return all trial intervals as `(trial_id, start_time, end_time)` rows."""
        block_starts_order = [int(e[3]) for e in self.events if e[1] == self.EVENTS_ENUM["StimStart"]]
        block_ends_order = [int(e[3]) for e in self.events if e[1] == self.EVENTS_ENUM["StimEnd"]]
        # The order of trials is matched
        assert np.all(block_starts_order == np.asarray(block_ends_order)), "Trials are out of order"
        block_start_times = [e[0] for e in self.events if e[1] == self.EVENTS_ENUM["StimStart"]]
        block_end_times = [e[0] for e in self.events if e[1] == self.EVENTS_ENUM["StimEnd"]]
        return np.asarray(list(zip(block_starts_order, block_start_times, block_end_times)))

    def trials_in_block(self, block):
        """Return trial intervals for one block as `(trial_id, start_time, end_time)` tuples.

        Args:
            block: Block identifier/index used for filtering events.
        """
        block_starts_order = [int(e[3]) for e in self.events if e[1] == self.EVENTS_ENUM["StimStart"] and int(e[2]) == block]
        block_ends_order = [int(e[3]) for e in self.events if e[1] == self.EVENTS_ENUM["StimEnd"] and int(e[2]) == block]
        # The order of trials is matched
        assert np.all(block_starts_order == np.asarray(block_ends_order)), "Block has trials out of order"
        block_start_times = [e[0] for e in self.events if e[1] == self.EVENTS_ENUM["StimStart"] and int(e[2]) == block]
        block_end_times = [e[0] for e in self.events if e[1] == self.EVENTS_ENUM["StimEnd"] and int(e[2]) == block]
        return list(zip(block_starts_order, block_start_times, block_end_times))

    def trial_from_all_blocks(self, trial):
        """Return occurrences of a trial index across blocks in chronological block order.

        Args:
            trial: Trial identifier/index used for filtering events.
        """
        trial_starts_block_order = [int(e[2]) for e in self.events if e[1] == self.EVENTS_ENUM["StimStart"] and int(e[3]) == trial]
        trial_ends_block_order = [int(e[2]) for e in self.events if e[1] == self.EVENTS_ENUM["StimEnd"] and int(e[3]) == trial]
        # The order of trials is matched
        order = list(range(1, 1 + len(trial_starts_block_order)))
        assert np.all(trial_starts_block_order == np.asarray(order)), "Missing blocks starting"
        assert np.all(trial_ends_block_order == np.asarray(order)), "Missing blocks ending"
        trial_start_times = [e[0] for e in self.events if e[1] == self.EVENTS_ENUM["StimStart"] and int(e[3]) == trial]
        trial_end_times = [e[0] for e in self.events if e[1] == self.EVENTS_ENUM["StimEnd"] and int(e[3]) == trial]
        return list(zip(order, trial_start_times, trial_end_times))


class Pixel2Micron(Module):
    """Convert measurements in pixels to measurements in microns"""

    NAME = "pixel2micron"
    MODE = "2p"

    @classmethod
    def prepare(cls, im_meta, timeline, rig, **_):
        # Basic info
        """Estimate z-position profile and pixel-to-micron mapping for each imaging plane.

        Args:
            im_meta: ScanImage frame metadata for the experiment.
            timeline: Timeline structure containing DAQ channels and timestamps.
            rig: Rig identifier used for channel maps and calibration.
        """
        n_planes = im_meta["SI.hStackManager.numSlices"]
        horiz_size = im_meta["SI.hRoiManager.pixelsPerLine"]  # TODO: Confirm this and next line are not swapped
        vert_size = im_meta["SI.hRoiManager.linesPerFrame"]
        timeline_channels = {i.name: i.arrayColumn - 1 for i in timeline.hw.inputs}
        # For each plane, find the position of the piezo each time that plane
        # is scanned.  The piezo should always be in approximately the same
        # position.  Likewise, collect the times of the DAQ observations so we
        # can later find the "position over time since the start of the frame"
        # curve.
        _neuralframes = timeline.rawDAQData[:, timeline_channels["neuralFrames"]].astype(int)
        plane_all_poss = {i: [] for i in range(0, n_planes)}
        plane_all_times = {i: [] for i in range(0, n_planes)}
        # It is slow and unnecessary to compute this for all the planes, so we
        # select a random subset of 500*#frames
        _i_frames = np.arange(1, np.max(_neuralframes))
        if len(_i_frames) <= 500 * n_planes:
            i_frames = _i_frames
        else:
            i_frames = np.random.choice(_i_frames, 500 * n_planes, replace=False)
        for i in i_frames:
            piezo_pos = timeline.rawDAQData[:, timeline_channels["piezoPosition"]][_neuralframes == i]
            times = timeline.rawDAQTimestamps[_neuralframes == i]
            assert (
                len(times) > 0
            ), "This neural frame has no timepoints associated with it, something went wrong with the neural frame counter in this experiment."
            times -= np.min(times)
            plane_all_poss[i % n_planes].extend(list(piezo_pos))
            plane_all_times[i % n_planes].extend(list(times))
        # Not sure where this came from but it is in other scripts and seems to work
        z_voltage_to_microns = 40
        # For each plane, find the average position across time.  Use lowess
        # regression.  Use the number of pixels as the number of timebins,
        # since we shouldn't need more than this.
        plane_poss = np.zeros((vert_size, n_planes))
        for i in range(0, n_planes):
            pos_range = np.linspace(0, np.max(plane_all_times[i]), horiz_size)
            lowess = sm.nonparametric.lowess(plane_all_poss[i], plane_all_times[i], frac=0.3, xvals=pos_range)
            plane_poss[:, i] = lowess * z_voltage_to_microns
        return {"plane_poss": plane_poss}

    def __init__(self, plane_poss, general_info, **kwargs):
        # In the version of plane_poss passed here, plane 7 is stored as plane
        # 0 (from the modulo) so we need to shift it.
        """Store voxel scaling factors and derived field-of-view dimensions in microns.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.plane_poss = np.roll(plane_poss, -1, 1)
        self.horiz_scale, self.vert_scale = zoom_to_pixel_size(
            general_info["rig" if "rig" in general_info.keys() else "microscope"], general_info["zoom"], general_info["date"]
        )
        z_size = (lambda x: np.max(x) - np.min(x))(np.median(self.plane_poss, axis=0))
        self.fov_in_microns = [z_size, general_info["fov"][1] * self.vert_scale, general_info["fov"][2] * self.horiz_scale]
        _diffs = np.diff(self.plane_poss, axis=1)
        self.median_z_step = np.median(_diffs[_diffs > 0])
        self.voxel_size = np.asarray([self.median_z_step, self.vert_scale, self.horiz_scale])

    def get_position_2d(self, yx):
        """Convert 2D pixel coordinates `(y, x)` into micron coordinates.

        Args:
            yx: 2D coordinates in `(y, x)` pixel order.
        """
        yx = np.asarray(yx)
        if yx.ndim == 1:
            yx = yx[None]
        assert yx.shape[1] == 2, "Invalid shape, should be points as rows and coords as columns"
        return yx * np.asarray([[self.vert_scale, self.horiz_scale]])

    def get_position(self, zyx):
        """Convert 3D pixel coordinates `(z, y, x)` into micron coordinates using plane-specific z calibration.

        Args:
            zyx: 3D coordinates in `(z, y, x)` pixel order.
        """
        zyx = np.asarray(zyx)
        if zyx.ndim == 1:
            zyx = zyx[None]
        assert zyx.shape[1] == 3, "Invalid shape, should be points as rows and coords as columns"
        rounded_zyx = np.round(zyx).astype(int)
        assert np.max(rounded_zyx[:, 1]) < self.plane_poss.shape[0], "Y coordinate too high"
        assert np.max(rounded_zyx[:, 0]) < self.plane_poss.shape[1], "Z coordinate too high"
        assert np.min(rounded_zyx[:, [0, 2]]) >= 0, "Y and Z coordinates must be non-negative"
        new_zyx = np.zeros(zyx.shape)
        new_zyx[:, 1:3] = self.get_position_2d(zyx[:, 1:3])
        # TODO Check that this is right.  I'm assuming it moves quickly across x and slower across y, and that the direction is right too.
        new_zyx[:, 0] = -self.plane_poss[tuple(rounded_zyx[:, 1]), tuple(rounded_zyx[:, 0])]
        return new_zyx


class CellInfo(Module):
    """Module containing Suite2p cell masks, labels, and geometry-derived summaries."""

    NAME = "cellinfo"
    MODE = "2p"

    def prepare(processed_s2p_path, im_meta, **_):
        """Load Suite2p cell masks and flatten them into a cache-friendly sparse coordinate representation.

        Args:
            processed_s2p_path: Directory containing Suite2p outputs.
            im_meta: ScanImage frame metadata for the experiment.
        """
        n_planes = im_meta["SI.hStackManager.numSlices"]
        stats = {}
        iscells = {}
        ops = {}
        for i in range(0, n_planes):
            path_stat = f"{processed_s2p_path}suite2p/plane{i}/stat.npy"
            stats[i] = _load_npy_compressed(path_stat, allow_pickle=True)
            path_iscell = f"{processed_s2p_path}suite2p/plane{i}/iscell.npy"
            iscells[i] = _load_npy_compressed(path_iscell, allow_pickle=True)
            path_ops = f"{processed_s2p_path}suite2p/plane{i}/ops.npy"
            ops[i] = _load_npy_compressed(path_ops, allow_pickle=True)
        # We want to be able to save the pixels occupied by each cell in a
        # numpy array, but since it is a different number of pixels for each
        # cell, we also need to save information about how many pixels each
        # cell has so we can reconstruct it.  We can't save lists of lists of
        # variable length.
        npix = []
        coords = []
        iscell = []
        for i in range(0, n_planes):
            for j in range(0, len(stats[i])):
                npix.append(len(stats[i][j]["xpix"]))
                coords.extend(list(zip(stats[i][j]["xpix"], stats[i][j]["ypix"], itertools.repeat(i))))
                # TODO Check this - after manual curation, do we need to save the first column too?
                iscell.append(iscells[i][j])
        npix = np.asarray(npix)
        coords = np.asarray(coords)
        iscell = np.asarray(iscell)
        return {"number_of_pixels": npix, "cell_coordinates": coords, "iscell": iscell}

    def __init__(self, general_info, number_of_pixels, cell_coordinates, iscell):
        """Reconstruct per-cell voxel coordinates and a labeled segmentation volume from cached mask data.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.number_of_pixels = number_of_pixels
        self.iscell = iscell.reshape(-1, 2)[:, 0].astype(bool)
        self.iscell_prob = iscell.reshape(-1, 2)[:, 1]
        self._cell_coordinates = cell_coordinates
        self._number_of_pixels = number_of_pixels
        self.cell_coordinates = []
        self.fov = general_info["fov"]
        curpx = 0
        repeated = np.repeat(np.arange(1, len(self._number_of_pixels) + 1), self._number_of_pixels)
        vol = np.zeros(self.fov).astype(np.int32)
        # Since cellpose allows overlapping cells, choose the first cell in the
        # list which satisfies each coordinate
        _, inds_unique = np.unique(self._cell_coordinates, axis=0, return_index=True)
        coords_unique = self._cell_coordinates[inds_unique]
        repeated_unique = repeated[inds_unique]
        for i in range(0, self.fov[0]):
            inds = coords_unique[:, 2] == i
            # Changed below line XYZ
            vol[i] = (
                scipy.sparse.coo_matrix((repeated_unique[inds], (coords_unique[inds, 0], coords_unique[inds, 1])), shape=self.fov[1:3]).todense().T
            )
        self.segmented_volume = vol
        for i in range(0, len(self.number_of_pixels)):
            self.cell_coordinates.append(cell_coordinates[curpx : (curpx + self.number_of_pixels[i])])
            curpx += self.number_of_pixels[i]
        # Reorder to z,y,x
        self.cell_coordinates = [cc[:, [2, 1, 0]] for cc in self.cell_coordinates]
        self.n_cells = len(self.iscell)

    def cell_positions(self):
        """Return per-cell median voxel coordinate as a robust cell centroid estimate."""
        return np.asarray(list(map(lambda x: np.median(x, axis=0), self.cell_coordinates)))

    def volume(self, cells=None, return_shadow=False):
        """Return cells as a volume.

        This only returns the cells passed in "cells".  The point here is to
        minimise overlapping cells.

        If "return_shadow=True", a second volume the same size as the first will
        be returned, but this will have parts of cells that were replaced
        because they overlapped with other cells.  There are usually very few
        triple-overlapping pixels (about 0.04% of pixels) so this captures
        almost the whole cell for all overlapping cells.

        Args:
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
            return_shadow: If true, also return displaced labels from overlaps.
        """
        cells = _interpret_cells_argument(cells, len(self.cell_coordinates))
        vol = np.zeros_like(self.segmented_volume)
        if return_shadow:
            shadow = np.zeros_like(self.segmented_volume)
        for c in np.where(cells)[0]:
            if return_shadow:
                shadow[tuple(self.cell_coordinates[c].T)] = vol[tuple(self.cell_coordinates[c].T)]
            vol[tuple(self.cell_coordinates[c].T)] = c + 1
        if return_shadow:
            return vol, shadow
        return vol

    def volume_renumbered(self, cells):
        """Return a labeled volume where selected cells are reindexed densely and unselected labels are cleared.

        Args:
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        if cells is None:
            return self.segmented_volume
        cells = np.append(False, cells)
        original_cells = np.sort(np.unique(self.segmented_volume))
        new_cells = original_cells.copy()
        new_cells[cells] = np.arange(1, sum(cells) + 1)
        new_cells[~cells] = 0
        sort_inds = np.argsort(original_cells)
        inds = np.searchsorted(original_cells, self.segmented_volume, sorter=sort_inds)
        return new_cells[sort_inds][inds]


class FunctionalBase(Module):
    """Base module for Suite2p-derived neural time-series matrices."""

    MODE = "2p"

    @classmethod
    def prepare(cls, processed_s2p_path, im_meta, expgroup, expnum, **_):
        """Load Suite2p time series data, slice by experiment group, and concatenate planes into a cell-major matrix.

        Args:
            processed_s2p_path: Directory containing Suite2p outputs.
            im_meta: ScanImage frame metadata for the experiment.
            expgroup: Grouped experiment-id string used for joint preprocessing.
            expnum: Experiment number within the session.
        """
        n_planes = im_meta["SI.hStackManager.numSlices"]
        plane_data = {}
        for i in range(0, n_planes):
            # For jointly processed experiments, load the ops file and only
            # extract the portion which came from the desired experiment.
            path_ops = f"{processed_s2p_path}suite2p/plane{i}/ops.npy"
            ops = _load_npy_compressed(path_ops, allow_pickle=True)
            assert len(expgroup.split("-")) == len(ops[()]["frames_per_folder"]), "Exp group length mismatch with s2p output"
            # If the experiment was processed on its own, then expgroup should
            # be a list with one element equal to expnum.  Here we sort
            # alphabetically, consistent with what is done in the suite2p
            # processing.
            i_group = list(sorted(expgroup.split("-"))).index(expnum)
            _frame_inds = np.cumsum(np.concatenate([[0], ops[()]["frames_per_folder"]]))
            start, end = _frame_inds[i_group : i_group + 2]
            path = f"{processed_s2p_path}suite2p/plane{i}/{cls.DATAFILE}"
            plane_data[i] = _load_npy_compressed(path)[:, start:end]

        # For ease of storage, I am going to cut off the final frame if there
        # isn't an equal number of frames for all planes.
        tslen = min(map(lambda x: x.shape[1], plane_data.values()))
        ncells = sum(map(lambda x: x.shape[0], plane_data.values()))
        data = np.zeros((ncells, tslen))
        plane = np.zeros(ncells)
        curr_cell = 0
        for i in range(0, n_planes):
            data[curr_cell : (curr_cell + plane_data[i].shape[0])] = plane_data[i][:, 0:tslen]
            plane[curr_cell : (curr_cell + plane_data[i].shape[0])] = i
            curr_cell += plane_data[i].shape[0]
        assert curr_cell == ncells
        return {"functional_timeseries": data.astype(cls.DTYPE), "functional_plane": plane}

    def __init__(self, functional_timeseries, functional_plane, general_info):
        """Store functional time-series matrix and per-cell plane assignments.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.timeseries = functional_timeseries.astype("float")
        self.plane = functional_plane.astype(int)
        assert self.timeseries.shape[0] == len(self.plane)
        self.n_cells = self.timeseries.shape[0]
        self.n_timepoints = self.timeseries.shape[1]


class FunctionalSpikes(FunctionalBase):
    """Suite2p deconvolved spike estimates for each detected cell."""

    NAME = "spikes"
    DATAFILE = "spks.npy"
    DTYPE = "float32"


class FunctionalF(FunctionalBase):
    """Suite2p raw fluorescence traces (`F`) for each detected cell."""

    NAME = "f"
    DATAFILE = "F.npy"
    DTYPE = "float16"


class FunctionalNeuropil(FunctionalBase):
    """Suite2p neuropil fluorescence traces (`Fneu`) for each detected cell."""

    NAME = "neuropil"
    DATAFILE = "Fneu.npy"
    DTYPE = "float16"


class EphysSpikes(Module):
    """Module containing timeline-aligned ephys spike trains and cluster metadata."""

    NAME = "ephys_spikes"
    MODE = "ephys"

    @classmethod
    def prepare(cls, ephys_paths, timeline, processed_data_path, rig, **_):
        # TODO this doesn't support multiple probes
        """Load Kilosort output, align spike times to timeline time, and collect waveform/cluster metadata.

        Args:
            ephys_paths: Directories containing Kilosort/Neuropixels outputs.
            timeline: Timeline structure containing DAQ channels and timestamps.
            processed_data_path: Directory containing processed intermediate outputs.
            rig: Rig identifier used for channel maps and calibration.
        """
        print(ephys_paths)
        for rec in range(0, len(ephys_paths)):
            ephys_path = ephys_paths[rec]
            spike_times = np.load(ephys_path + "/spike_times.npy")
            spike_clusters = np.load(ephys_path + "/spike_clusters.npy")
            cluster_group = pandas.read_csv(ephys_path + "/cluster_group.tsv", sep="\t")
            groupname = "group" if "group" in cluster_group.columns else "KSLabel"  # Changed in kilosort4
            cluster_group["rating_id"] = cluster_group[groupname].map(lambda x: {"mua": 1, "good": 2, "noise": -1}[x])
            clusters_mat = np.asarray(cluster_group[["cluster_id", "rating_id"]])
            # Cluster centroid
            templates = np.load(ephys_path + "/templates.npy")  # n_clusters x n_timepoints x n_channels
            channel_positions = np.load(ephys_path + "/channel_positions.npy")  # n_channels x 2
            channel_weight = np.sum(np.abs(templates), axis=1)  # n_clusters x n_channels
            cluster_centroid = channel_weight @ channel_positions / np.sum(channel_weight, axis=1, keepdims=True)
            # Spike waveforms
            # How to determine the best channel?  Phy uses the amplitude (max - min) of the whitened signal.  We don't have that here so we have to use the unwhitened.  But, we also want the example waveforms to match the template waveforms.  So, if example waveforms are present, use the first channel.  If not, use the amplitude from the available (unwhitened) templates.
            # Do the same best_channel calculations
            unwhite = np.load(ephys_path + "/whitening_mat_inv.npy")
            best_channel = np.argmax(np.max(templates @ unwhite, axis=1) - np.min(templates @ unwhite, axis=1), axis=1)
            template_waveforms = np.asarray([templates[i][:, best_channel[i]] for i in range(0, templates.shape[0])])
            try:
                _example_waveforms = np.load(ephys_path + "/example_waveforms.npz")
                all_spike_waveforms = _example_waveforms["example_waveforms"]
                spike_waveform_times = _example_waveforms["example_waveform_times"]
                waveform_channels = _example_waveforms["channels"]
                # These two lines are just to make sure phylib gives sorted output.  Commetned out now for backward compatibility with a couple recordings I don't want to rerun.
                # spike_waveform_channels = _example_waveforms['channels']
                # assert all([np.where(spike_waveform_channels[i]==best_channel[i])[0][0]==0 for i in range(0, len(best_channel))]), "Error calculating best waveforms"
                spike_waveforms = all_spike_waveforms[:, :, :, 0]
                mean_waveforms = np.mean(all_spike_waveforms, axis=1)
            except FileNotFoundError:
                spike_waveforms = np.asarray([])
                spike_waveform_times = np.asarray([])
                mean_waveforms = np.asarray([])
                waveform_channels = np.asarray([])
            # Find timepoints we care about with timeline and the flipper signal
            flipper = np.load(Path(processed_data_path).parent.joinpath(f"flipper{rec}.ap.npz"))["arr_0"]
            tl_flip_id = next(i for i in range(0, len(timeline.hw.inputs)) if timeline.hw.inputs[i].name == "flipper")
            tlflipper = timeline.rawDAQData[:, tl_flip_id]
            # Sometimes there are spurious high-voltage samples in the flipper
            # signal.  Set these to the value of their nearest lower neighbour
            # which is not spurious.
            spurious = flipper > np.quantile(flipper, 0.9999) + 10
            for i_spur in np.where(spurious)[0]:
                i_dec = next(i for i in range(1, 10) if i_spur - i not in spurious)
                flipper[i_spur] = flipper[i_spur - i_dec]
            # Binarize flipper signal and find actual flips.
            flipperbin = flipper > 1.5
            tlflipperbin = tlflipper > 1.5
            flips = np.where(np.diff(flipperbin))[0]
            tlflips = timeline.rawDAQTimestamps[1 + np.where(np.diff(tlflipperbin))[0]]
            # Find corresponding flips between the two flipper signals
            flips_spacing = np.diff(flips).astype(float)
            tlflips_spacing = np.diff(tlflips).astype(float)
            corrs = [
                np.corrcoef(tlflips_spacing[:-1], flips_spacing[i : (i + len(tlflips_spacing) - 1)])[0, 1]
                for i in range(0, len(flips_spacing) - len(tlflips_spacing) + 2)
            ]
            if len(corrs) == 0:
                print(f"Length of flipper is not long enough, continuing")
                continue
            shift = np.argmax(corrs)
            if not corrs[shift] > 0.99:  # Use not here so that it will catch nans too
                print(f"Invalid shift {shift} has correlation {corrs[shift]} from recording {rec}, trying again")
                continue
            # Regression to find npix -> timeline transform.  We use regression
            # because the frequencies are not exactly 1000 hz and 30000 hz and
            # there is a shift of about 50 ms in timing over the course of the
            # experiment if you don't adjust for this.
            coefs = np.polyfit(flips[shift : (shift + len(tlflips))], tlflips, 1)
            # transform_npix_time_to_timeline_time = lambda x : coefs[0]*x + coefs[1]
            # Get timeline time and filter to those inside the experimental window
            spike_times_timeline = spike_times * coefs[0] + coefs[1]
            # Only save the spikes that were a part of this experiment.
            valid = np.logical_and(spike_times_timeline >= 0, spike_times_timeline < timeline.lastTimestamp)
            print("Success")
            break
        if "valid" not in vars():
            raise ValueError("Error: could not find flipper signal in any recordings")
        return {
            "spike_times": spike_times_timeline[valid],
            "spike_clusters": spike_clusters[valid],
            "cluster_group": clusters_mat,
            "cluster_centroid": cluster_centroid,
            "timeline_transform": coefs,
            "spike_waveforms": spike_waveforms,
            "spike_waveform_times": spike_waveform_times,
            "template_waveforms": template_waveforms,
            "mean_waveforms": mean_waveforms,
            "channel_positions": channel_positions,
            "channel_weight": channel_weight,
            "waveform_channels": waveform_channels,
        }

    def __init__(
        self,
        spike_times,
        spike_clusters,
        cluster_group,
        timeline_transform,
        cluster_centroid,
        general_info,
        spike_waveform_times=None,
        spike_waveforms=None,
        template_waveforms=None,
        mean_waveforms=None,
        channel_positions=None,
        channel_weight=None,
        waveform_channels=None,
    ):
        """Store aligned spike trains, cluster annotations, and optional waveform metadata.

        Args:
            general_info: General metadata dict from the cache file.
            channel_positions: Probe channel coordinates.
        """
        self.spike_times = spike_times
        self.spike_clusters = spike_clusters
        self.cluster_group = cluster_group
        self.cluster_centroid = cluster_centroid
        self._timeline_transform = timeline_transform
        self.template_waveforms = template_waveforms
        self.mean_waveforms = mean_waveforms
        self.spike_waveforms = spike_waveforms
        self.spike_waveform_times = spike_waveform_times
        self.channel_positions = channel_positions
        self.channel_weight = channel_weight
        self.waveform_channels = waveform_channels
        # Use the "cells" terminology instead of "clusters" for similarity
        # with 2p
        self.n_cells = cluster_group.shape[0]
        self.good_cells = cluster_group[:, 1] == 2
        self.mua_cells = cluster_group[:, 1] == 1
        self.noise_cells = cluster_group[:, 1] == -1
        self.all_cells = cluster_group[:, 1] > -np.inf

    def spikes_interval(self, interval, cells=None):
        # Interval should be (start, end) tuple
        """Return per-cell spike times relative to an interval start.

        Args:
            interval: Time interval as `(start, stop)` in timeline seconds.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        cells = _interpret_cells_argument(cells, self.all_cells.shape[0])
        valid = np.logical_and(self.spike_times > interval[0], self.spike_times < interval[1])
        spikes = self.spike_times[valid]
        clusters = self.spike_clusters[valid]
        spiketimes = np.zeros(sum(cells), dtype=object)
        for i, cellid in enumerate(self.cluster_group[:, 0][cells]):
            spiketimes[i] = spikes[clusters == cellid] - interval[0]
        return spiketimes

    def spikes_intervals(self, intervals, cells=None):
        """Return per-cell spike-time arrays for each interval in a batch.

        Args:
            intervals: List/array of time intervals.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        cells = _interpret_cells_argument(cells, self.all_cells.shape[0])
        spiketimes = np.zeros((len(intervals), sum(cells)), dtype=object)
        for i, interval in enumerate(intervals):
            spiketimes[i] = self.spikes_interval(interval, cells=cells)
        return spiketimes

    def spikes_histogram(self, interval, dt=0.1, cells=None):
        """Bin spike times into fixed-width bins over an interval for selected cells.

        Args:
            interval: Time interval as `(start, stop)` in timeline seconds.
            dt: Sampling/bin width in seconds.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        cells = _interpret_cells_argument(cells, self.all_cells.shape[0])
        bins = np.arange(interval[0], interval[1], dt)
        hists = np.asarray([np.histogram(self.spike_times[self.spike_clusters == i], bins)[0] for i in np.where(cells)[0]])
        return hists

    def spikes_histogram_at(self, bins, cells=None):
        """Bin spike times using user-supplied bin edges for selected cells.

        Args:
            bins: Histogram bin edges.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        cells = _interpret_cells_argument(cells, self.all_cells.shape[0])
        hists = np.asarray([np.histogram(self.spike_times[self.spike_clusters == i], bins)[0] for i in np.where(cells)[0]])
        return hists


class SpikeSortingInfo(Module):
    """Module containing extra spike-sorting diagnostics and waveform metadata."""

    NAME = "spike_sorting_info"
    MODE = "ephys"

    @classmethod
    def prepare(cls, ephys_paths, timeline, processed_data_path, rig, **_):
        # TODO this doesn't support multiple probes
        """Load additional spike-sorting diagnostics such as templates, contamination, and waveform snippets.

        Args:
            ephys_paths: Directories containing Kilosort/Neuropixels outputs.
            timeline: Timeline structure containing DAQ channels and timestamps.
            processed_data_path: Directory containing processed intermediate outputs.
            rig: Rig identifier used for channel maps and calibration.
        """
        print(ephys_paths)
        for rec in range(0, len(ephys_paths)):
            ephys_path = ephys_paths[rec]
            templates = np.load(ephys_path + "/templates.npy")  # n_clusters x n_timepoints x n_channels
            _example_waveforms = np.load(ephys_path + "/example_waveforms.npz")
            all_spike_waveforms = _example_waveforms["example_waveforms"]
            spike_waveform_times = _example_waveforms["example_waveform_times"]
            waveform_channels = _example_waveforms["channels"]
            whitening_matrix = np.load(ephys_path + "/whitening_mat.npy")
            channel_positions = np.load(ephys_path + "/channel_positions.npy")  # n_channels x 2
            similar_templates = np.load(ephys_path + "/similar_templates.npy")
            cluster_amplitude = np.asarray(pandas.read_csv(ephys_path + "/cluster_Amplitude.tsv", sep="\t")["Amplitude"])
            cluster_contampct = np.asarray(pandas.read_csv(ephys_path + "/cluster_ContamPct.tsv", sep="\t")["ContamPct"])
            break
        return {
            "templates": templates,
            "example_waveforms": all_spike_waveforms,
            "example_waveform_times": spike_waveform_times,
            "example_waveform_channels": waveform_channels,
            "whitening_matrix": whitening_matrix,
            "channel_positions": channel_positions,
            "similar_templates": similar_templates,
            "cluster_amplitude": cluster_amplitude,
            "cluster_contampct": cluster_contampct,
        }

    def __init__(
        self,
        templates,
        example_waveforms,
        example_waveform_times,
        example_waveform_channels,
        whitening_matrix,
        channel_positions,
        similar_templates,
        cluster_amplitude,
        cluster_contampct,
        **kwargs,
    ):
        """Store spike-sorting diagnostics and compute waveform-derived unit locations.

        Args:
            channel_positions: Probe channel coordinates.
        """
        self.templates = templates
        self.example_waveforms = example_waveforms
        self.example_waveform_times = example_waveform_times
        self.example_waveform_channels = example_waveform_channels
        self.whitening_matrix = whitening_matrix
        self.channel_positions = channel_positions
        self.similar_templates = similar_templates
        self.cluster_amplitude = cluster_amplitude
        self.cluster_contampct = cluster_contampct
        self.mean_waveform_amplitudes = np.max(np.abs(np.mean(self.example_waveforms, axis=1)), axis=1)
        _mean_waveform_positions = self.channel_positions[self.example_waveform_channels]
        self.mean_waveform_positions = np.sum(self.mean_waveform_amplitudes[:, :, None] * _mean_waveform_positions, axis=1) / np.sum(
            self.mean_waveform_amplitudes, axis=1, keepdims=True
        )


class LFP(Module):
    """Local-field-potential channels aligned from Neuropixels data into timeline time."""

    NAME = "lfp"
    MODE = "ephys"

    @classmethod
    def prepare(cls, processed_data_path, data_paths, timeline, rig, **_):
        """Load LFP channels, align them to timeline time via flipper synchronization, and extract the experiment interval.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            data_paths: Candidate raw-data directories for the experiment.
            timeline: Timeline structure containing DAQ channels and timestamps.
            rig: Rig identifier used for channel maps and calibration.
        """
        import pykilosort
        import spikeglx

        data_paths = _multiglob([EPHYS_PATH.format(mouse=args.mouse, date=args.date)], "/**/*.lf.bin", recursive=True)
        if len(data_paths) > 1:
            raise NotImplementedError("Only one lfp file")
        lfp_file = data_paths[0]
        rec = 0
        r = spikeglx.Reader(lfp_file)  # TODO support multiple recordings/probes
        n_channels = r.nc
        xy = np.asarray([r.geometry["x"], r.geometry["y"]])
        # This signal is already binary, but for some reason this function
        # returns a matrix of 16 channels, 15 of which are empty.
        # flipper = np.max(r.read_sync_digital(slice(0, None)), axis=1)
        flipper = np.load(Path(processed_data_path).parent.joinpath(f"flipper{rec}.lf.npz"))["arr_0"]
        flipper_times = np.arange(0, len(flipper), dtype="int32")

        # This is all copied from the EphysSpikes module above.
        tl_flip_id = next(i for i in range(0, len(timeline.hw.inputs)) if timeline.hw.inputs[i].name == "flipper")
        tlflipperbin = timeline.rawDAQData[:, tl_flip_id] > 1.5
        flips = np.where(np.diff(flipper))[0]
        tlflips = timeline.rawDAQTimestamps[1 + np.where(np.diff(tlflipperbin))[0]]
        # Binarize flipper signal and find actual flips
        # Find corresponding flips between the two flipper signals
        flips_spacing = np.diff(flips).astype(float)
        tlflips_spacing = np.diff(tlflips).astype(float)
        corrs = [
            np.corrcoef(tlflips_spacing[:-1], flips_spacing[i : (i + len(tlflips_spacing) - 1)])[0, 1]
            for i in range(0, len(flips_spacing) - len(tlflips_spacing) + 2)
        ]
        shift = np.argmax(corrs)
        assert corrs[shift] > 0.99, f"Invalid shift {shift} has correlation {corrs[shift]}"
        # Regression to find npix -> timeline transform.  We use regression
        # because the frequencies are not exactly 1000 hz and 30000 hz and
        # there is a shift of about 50 ms in timing over the course of the
        # experiment if you don't adjust for this.
        coefs = np.polyfit(flips[shift : (shift + len(tlflips))], tlflips, 1)
        # Get timeline time for LFPs
        lfp_times_timeline = flipper_times * coefs[0] + coefs[1]

        # Now this stuff isn't copied from EphysSpikes, all original!
        lfp_ind_start = np.where(lfp_times_timeline > 0)[0][0]
        lfp_ind_end = np.where(lfp_times_timeline > timeline.lastTimestamp)[0][0]
        lfp_len = lfp_ind_end - lfp_ind_start
        print(lfp_ind_end, lfp_ind_start)
        # lfp_chunk = r._raw[lfp_ind_start:lfp_ind_end,0:(r.nc-r.nsync)])
        lfp_chunk = np.fromfile(lfp_file, dtype="int16", count=n_channels * lfp_len, offset=2 * lfp_ind_start * n_channels).reshape(-1, n_channels).T
        lfp_times_chunk = lfp_times_timeline[lfp_ind_start:lfp_ind_end]

        return {"lfp": lfp_chunk, "lfp_times_start": lfp_times_chunk[0], "lfp_times_end": lfp_times_chunk[-1], "xy": xy}

    def __init__(self, lfp, lfp_times_start, lfp_times_end, xy, **kwargs):
        """Store aligned LFP traces and reconstruct an evenly sampled timeline vector."""
        self.lfp = lfp[0:-1]
        self.sync = lfp[-1]
        self.lfp_times = np.linspace(lfp_times_start, lfp_times_end, len(lfp[0]))
        self.xy = xy

    def interval_timeseries(self, start, stop):
        """Extract LFP samples spanning a single time interval.

        Args:
            start: Interval start time in timeline seconds.
            stop: Interval stop time in timeline seconds.
        """
        start_ind = np.argmin(np.abs(self.lfp_times - start))
        stop_ind = np.argmin(np.abs(self.lfp_times - stop))
        return self.lfp[:, start_ind:stop_ind]

    def intervals_timeseries(self, intervals):
        """Extract LFP segments for multiple intervals and trim them to a shared length.

        Args:
            intervals: List/array of time intervals.
        """
        timeseries = []
        for intvl in intervals:
            timeseries.append(self.interval_timeseries(*intvl))
        minlen = min(map(lambda x: x.shape[1], timeseries))
        timeseries = [ts[:, 0:minlen] for ts in timeseries]
        return np.asarray(timeseries)


class _EyeTracking(Module):  # Do not use directly
    """Generic interface for eye-tracking modules.  Should not be used directly.

    This makes it easier to define eye tracking modules.  They must define one
    function:

    Methods:
    _get_eye_data(processed_data_path, data_path)
        Given the two data paths (directly passed from ``prepare``, return a
        3-tuple (pos,size,lid) of numpy arrays: a Nx2 of the x and y coordinate
        of the pupil's position, a length-N array of the pupil size, and the
        length-N array of the lid distance.  NaN indicates an error in
        detecting at that position.
    """

    # Need to define the _get_eye_data() function
    @classmethod
    def prepare(cls, processed_data_path, data_paths, timeline, rig, **_):
        # Align eye camera to timeline.  B2 scope has extra synchronisation
        # information by flickering the light off and on, but bscope doesn't.
        """Align eye-camera frames to timeline time and package pupil/lid/motion-energy measurements.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            data_paths: Candidate raw-data directories for the experiment.
            timeline: Timeline structure containing DAQ channels and timestamps.
            rig: Rig identifier used for channel maps and calibration.
        """
        frame_times = None
        fn = _multiglob(data_paths, "/*eye.mj2", one=True)
        m = cv2.VideoCapture(fn)
        # Compute two things here: the mean trace (which is used to
        # determine lights on/off) and the mean motion energy (a generally
        # useful property that is best to compute here since we already
        # have the alignment information).
        meantrace = []
        mean_motion_energy = [0]
        prev_frame = None
        # TODO this loop is very slow, it could be moved to the preprocessing
        # as a pipeline
        while True:
            ret, frame = m.read()
            if not ret:
                break
            frame = frame[:, :, 0]  # [:,:,0] to remove colour
            if prev_frame is not None:
                diff = frame - prev_frame
                mean_motion_energy.append(np.mean(np.abs(diff)))
            meantrace.append(np.mean(frame))
            prev_frame = frame
        if rig == "b2":
            # Simple threshold
            camera_offtimes = np.where(np.diff(np.asarray(meantrace) < 20).astype(int))[0]
            tl_offtimes = np.where(np.diff(load_with_patch(_multiglob(data_paths, "camSync.raw.npy", one=True)).flatten() < 0.5))[0]
            if len(camera_offtimes) == 4 and len(tl_offtimes) == 4:
                # Align based on the start times
                reg_m1 = (tl_offtimes[2] - tl_offtimes[0]) / (camera_offtimes[2] - camera_offtimes[0])  # Should always be 33.333
                reg_b1 = tl_offtimes[0] - reg_m1 * camera_offtimes[0]
                # Now based on end times
                reg_m2 = (tl_offtimes[3] - tl_offtimes[1]) / (camera_offtimes[3] - camera_offtimes[1])
                reg_b2 = tl_offtimes[1] - reg_m2 * camera_offtimes[1]
                # Take the mean to get the actual coefficients.  The m's should be
                # the same but the b's may have an offset.
                reg_m = np.mean([reg_m1, reg_m2])
                reg_b = np.mean([reg_b1, reg_b2])
                # Convert using this
                frame_times = reg_m * np.arange(0, len(meantrace)) + reg_b
                alignment = "double_flash"
            elif len(camera_offtimes) == 2 and len(tl_offtimes) == 4 and np.all(camera_offtimes < len(meantrace) / 2):
                print("Only one flash was detected, but it was the early one so we can try to align with it")
                reg_b1 = tl_offtimes[0] - 33.3331 * camera_offtimes[0]
                reg_b2 = tl_offtimes[1] - 33.3331 * camera_offtimes[1]
                frame_times = 33.3331 * np.arange(0, len(meantrace)) + np.mean([reg_b1, reg_b2])
                alignment = "single_flash"
            else:
                print("Aligning based on the flash failed, falling back to udp events")
        if frame_times is None:  # either we are on bscope, or else the b2 light flash wasn't detected
            eye_fn = _multiglob(data_paths, "*_eye.mat", one=True)
            eye = load_with_patch(eye_fn, squeeze_me=True, struct_as_record=False)["eyeLog"]
            # The eye computer receives mpep events too, so align based on identical events.
            tl_events = [e for e in timeline.mpepUDPEvents if e][1:-1]  # 1:-1 to drop the start and stop, which are unreliable in mpep
            tl_event_times = [t for e, t in zip(timeline.mpepUDPEvents, timeline.mpepUDPTimes) if e][1:-1]
            eye_events = [e.split("'")[1] for e in eye.udpEvents][1:-1]
            eye_event_times = [e[3] * 60 * 60 + e[4] * 60 + e[5] for e in eye.udpEventTimes][1:-1]
            assert tl_events == eye_events, "Timeline and eye tracking events don't match"
            eye_time_offset = np.median(eye_event_times - np.asarray(tl_event_times))
            frame_numbers, frame_times = zip(
                *[(f.FrameNumber, f.AbsTime[3] * 60 * 60 + f.AbsTime[4] * 60 + f.AbsTime[5] - eye_time_offset) for f in eye.TriggerData]
            )
            assert np.all(frame_numbers == np.asarray(range(0, len(frame_numbers))) + 1), "Missing eye frame"
            alignment = "udp"
        # Now get the actual measurements of the eye
        pupil_pos, pupil_size, lid = cls._get_eye_data(processed_data_path, data_paths)
        assert len(pupil_pos) == len(pupil_size)
        assert len(pupil_pos) == len(lid)
        assert len(pupil_pos) == len(frame_times)
        return {
            "eye_times": np.asarray(frame_times),
            "pupil_pos": np.asarray(pupil_pos),
            "pupil_size": np.asarray(pupil_size),
            "lid": lid,
            "alignment": alignment,
            "mean_motion_energy": np.asarray(mean_motion_energy),
        }

    def __init__(self, eye_times, pupil_pos, pupil_size, lid, alignment, general_info, mean_motion_energy=None):
        """Store aligned eye-tracking measurements and normalize time units when needed.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.eye_times = eye_times
        # Sometimes eye_times is in sec and sometimes it is in msec???
        if self.eye_times[-1] > 60 * 60 * 4:  # Cannot have a recording over 4 h and likely won't have one under 14.4 sec
            self.eye_times = self.eye_times / 1000
        self.pupil_pos = pupil_pos
        self.pupil_size = pupil_size
        self.lid = lid
        self.alignment = alignment
        if mean_motion_energy is not None:
            self.mean_motion_energy = mean_motion_energy

    def _outlier_filter_nan(self, x):
        """Suppress extreme outliers in an eye trace by replacing them with `NaN`.

        Args:
            x: Input vector to filter.
        """
        x = x.copy()
        nonan = ~np.isnan(x)
        median = np.median(x[nonan])
        try:
            median_to_99quant = np.quantile(x[nonan], 0.99) - median
            median_to_01quant = median - np.quantile(x[nonan], 0.01)
        except IndexError:
            raise ValueError("Eye tracking not available for this experiment")
        _outliers = (x > (median_to_99quant * 2 + median)) | (x < (median - median_to_01quant * 2))
        x[_outliers] = np.nan
        # outliers = np.where(_outliers)[0]
        # nonoutliers = np.where(~_outliers)[0]
        ## Make each outlier the median of the surrounding 4 non-outlier points.
        # for o in outliers:
        #    try:
        #        surround = list(nonoutliers[nonoutliers<o][-2:]) + list(nonoutliers[nonoutliers>o][:2])
        #        x[o] = np.median(surround)
        #    except IndexError:
        #        x[o] = np.nan
        #
        # x[~np.isnan(x)] = scipy.ndimage.median_filter(x[~np.isnan(x)], size=size)
        return x

    def interval_timeseries_at(self, target_times, smooth=0.3, replacenan=False, prefilter=True, measurement=None):
        """Interpolate and smooth eye measurements at specific target times.

        Args:
            target_times: Explicit timestamps at which values should be sampled.
            smooth: Gaussian smoothing width in seconds.
            replacenan: Whether to fill NaNs during interpolation/smoothing.
            prefilter: Whether to apply outlier filtering before interpolation.
            measurement: Measurement key selecting which signal to extract.
        """
        if prefilter != 0:
            medfilt = lambda x: self._outlier_filter_nan(x)
        else:
            medfilt = lambda x: x
        interp_x = _smooth_fast_nans(self.eye_times, medfilt(self.pupil_pos[:, 0]), target_times, width=smooth, replacenan=replacenan)
        interp_y = _smooth_fast_nans(self.eye_times, medfilt(self.pupil_pos[:, 1]), target_times, width=smooth, replacenan=replacenan)
        interp_pupil = _smooth_fast_nans(self.eye_times, medfilt(self.pupil_size), target_times, width=smooth, replacenan=replacenan)
        interp_lid = _smooth_fast_nans(self.eye_times, medfilt(self.lid), target_times, width=smooth, replacenan=replacenan)
        if hasattr(self, "mean_motion_energy"):
            interp_motion_energy = _smooth_fast_nans(self.eye_times, self.mean_motion_energy, target_times, width=smooth, replacenan=replacenan)
        else:
            interp_motion_energy = None
        if measurement is None:
            return {"x": interp_x, "y": interp_y, "pupil_size": interp_pupil, "lid_dist": interp_lid, "motion_energy": interp_motion_energy}
        elif measurement == "x":
            return interp_x
        elif measurement == "y":
            return interp_y
        elif measurement == "pupil":
            return interp_pupil
        elif measurement == "lid":
            return interp_lid
        elif measurement == "motion_energy":
            return interp_motion_energy

    def interval_timeseries(self, start, end, dt=0.1, smooth=0.3, replacenan=False, measurement=None, prefilter=True):
        """Sample eye measurements on an evenly spaced grid over an interval.

        Args:
            start: Interval start time in timeline seconds.
            end: Interval end time in timeline seconds.
            dt: Sampling/bin width in seconds.
            smooth: Gaussian smoothing width in seconds.
            replacenan: Whether to fill NaNs during interpolation/smoothing.
            measurement: Measurement key selecting which signal to extract.
            prefilter: Whether to apply outlier filtering before interpolation.
        """
        times = np.arange(start, end, dt)
        return self.interval_timeseries_at(times, smooth=smooth, replacenan=replacenan, measurement=measurement, prefilter=prefilter)

    def interval_mean(self, start, end, measurement=None):
        """Compute mean eye measurements over an interval.

        Args:
            start: Interval start time in timeline seconds.
            end: Interval end time in timeline seconds.
            measurement: Measurement key selecting which signal to extract.
        """
        times = (self.eye_times >= start) & (self.eye_times < end)
        _measurements = {"x": self.pupil_pos[:, 0], "y": self.pupil_pos[:, 1], "pupil_size": self.pupil_size, "lid_dist": self.lid}
        if hasattr(self, "mean_motion_energy"):
            _measurements["motion_energy"] = self.mean_motion_energy
        measurements = {mkey: np.mean(mval[times]) for mkey, mval in _measurements.items()}
        if measurement is None:
            return measurements
        else:
            return measurements[measurement]


# To run facemap: python -m facemap --movie '/home/carsen/movie.avi' --savedir '/media/carsen/SSD/'
class EyeTrackingFacemap(_EyeTracking):
    """Eye-tracking backend that loads Facemap-derived pupil measurements."""

    NAME = "eye_facemap"

    @staticmethod
    def _get_eye_data(processed_data_path, data_paths):
        """Load pupil center/area traces generated by Facemap processing.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            data_paths: Candidate raw-data directories for the experiment.
        """
        fm_fn = glob.glob(processed_data_path + "*_eye_proc.npy")
        fm = np.load(fm_fn, allow_pickle=True)[()]
        pupil_pos = fm["pupil"][0]["com_smooth"]
        pupil_size = fm["pupil"][0]["area_smooth"]
        lid = pupil_size * 0  # No support for lid position here
        return pupil_pos, pupil_size, lid


class EyeTrackingDLC(_EyeTracking):
    """Eye-tracking backend that loads DeepLabCut landmark-derived measurements."""

    NAME = "eye_dlc"

    # This is set up for Sylvia Schroeder's script: https://github.com/sylviaschroeder/PupilDetection_DLC
    def _get_eye_data(processed_data_path, data_paths):
        """Load DeepLabCut pupil/lid landmarks and derive pupil center, area, and lid distance.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            data_paths: Candidate raw-data directories for the experiment.
        """
        fn = glob.glob(processed_data_path + "*.csv")
        assert len(fn) == 1, f"Only one csv file should be in processed data, not {len(fn)}"
        df = pandas.read_csv(fn[0], skiprows=1, header=[0, 1])
        pupil_names = ["pupil_left", "pupil_right", "pupil_top", "pupil_bot"]
        likes = df.loc[:, (pupil_names, "likelihood")]
        blinks = np.any(likes < 0.8, axis=1)
        df.loc[blinks, (pupil_names, "x")] = np.nan
        df.loc[blinks, (pupil_names, "y")] = np.nan
        xs = df.loc[:, (pupil_names, "x")]
        ys = df.loc[:, (pupil_names, "y")]
        loc = np.asarray([xs.mean(axis=1), ys.mean(axis=1)]).T
        pupil_size_y = ys[("pupil_bot", "y")] - ys[("pupil_top", "y")]
        pupil_size_x = xs[("pupil_right", "x")] - xs[("pupil_left", "x")]
        pupil_size = np.mean([pupil_size_y, pupil_size_x], axis=0) ** 2 * np.pi
        invalid_lid = np.any(df.loc[:, (["lid_bot", "lid_top"])] < 0.8, axis=1)
        lid = np.asarray(df.loc[:, ("lid_bot", "y")] - df.loc[:, ("lid_top", "y")])
        lid[invalid_lid] = np.nan
        return loc, pupil_size, lid


class Ball(Module):
    """Module for ball-motion telemetry converted to physical running-speed units."""

    NAME = "ball"

    @staticmethod
    def prepare(timeline, **_):
        """Decode UDP ball packets and convert them into synchronized motion increments.

        Args:
            timeline: Timeline structure containing DAQ channels and timestamps.
        """
        events = [
            (float(e.split(" ")[0]), int(e.split(" ")[1]), int(e.split(" ")[2]), int(e.split(" ")[3]), int(e.split(" ")[4]), t)
            for e, t in zip(timeline.ballUDPEvents, timeline.ballUDPTimes)
            if e
        ]
        time_diff = np.median([e[0] / 1000 - e[5] for e in events])
        times = np.asarray([e[0] / 1000 for e in events]) - time_diff
        # In format (sideways, rotation1, foward, rotation2)
        ball_deltas = np.asarray([e[1:5] for e in events])
        ball_deltas_int = ball_deltas.astype(np.int8)
        assert np.all(ball_deltas == ball_deltas_int)
        return {"ball_times": times, "ball_deltas": ball_deltas}

    def __init__(self, ball_times, ball_deltas, general_info):
        """Store ball motion signals in physical units and estimate sampling interval.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.ball_times = np.asarray(ball_times)
        self.ball_deltas = np.asarray(ball_deltas) * get_speed_scale_to_cm(general_info["rig"], general_info["date"])
        # Setting a single dt, used for scaling in speed_interval_timeseries_at,
        # assumes that times are uniformly spaced.  If not, speed may be
        # slightly off.
        self.dt = np.median(self.ball_times[1:] - self.ball_times[0:-1])

    def get_forward_speed(self):
        """Return timeline timestamps and forward-speed component from ball measurements."""
        return (self.ball_times, self.ball_deltas[:, 2] * self.PX_PER_CM)

    def speed_interval_timeseries_at(self, target_times, smooth=0.3):
        """Return smoothed forward-speed samples at requested target times.

        Args:
            target_times: Explicit timestamps at which values should be sampled.
            smooth: Gaussian smoothing width in seconds.
        """
        padded_start = target_times[0] - smooth * 3
        padded_stop = target_times[-1] + smooth * 3
        inds = (self.ball_times >= padded_start) & (self.ball_times < padded_stop)
        smoothed = _smooth(
            0.5 * (self.ball_times[inds] + self.ball_times[inds]), self.get_forward_speed()[1][inds] / self.dt, target_times, width=smooth
        )
        return smoothed

    def speed_interval_timeseries(self, start, end, dt=0.1, smooth=0.3):
        """Return smoothed forward-speed series on an evenly spaced interval grid.

        Args:
            start: Interval start time in timeline seconds.
            end: Interval end time in timeline seconds.
            dt: Sampling/bin width in seconds.
            smooth: Gaussian smoothing width in seconds.
        """
        times = np.arange(start, end, dt)
        return self.speed_interval_timeseries_at(times, smooth=smooth)

    def speed_interval_mean(self, start, end):
        """Compute mean forward-running speed over an interval.

        Args:
            start: Interval start time in timeline seconds.
            end: Interval end time in timeline seconds.
        """
        times, speed = self.get_forward_speed()
        inds = (times >= start) & (times < end)
        return np.mean(self.speed[inds])


class Treadmill(Module):
    """Module for rotary-encoder treadmill signals converted to speed traces."""

    NAME = "treadmill"

    @staticmethod
    def prepare(data_paths, **_):
        """Load rotary-encoder counts and timeline alignment metadata.

        Args:
            data_paths: Candidate raw-data directories for the experiment.
        """
        dat = load_with_patch(_multiglob(data_paths, "rotaryEncoder.raw.npy", one=True)).flatten().astype("uint32")
        times = load_with_patch(_multiglob(data_paths, "rotaryEncoder.timestamps_Timeline.npy", one=True))
        assert len(dat) == int(times[1, 0]) + 1 and int(times[0, 0]) == 0, f"Invalid times for treadmill: {len(dat)}, {times}"
        return {"treadmill": dat, "times": times[:, 1]}

    def __init__(self, treadmill, times=None, general_info=None):
        # It wraps around in the uint type, so we adjust it so it doesn't wrap
        # around anymore
        """Unwrap rotary-encoder counts, convert to distance, and derive instantaneous speed.

        Args:
            times: Time points (seconds) at which values are requested.
            general_info: General metadata dict from the cache file.
        """
        self.treadmill = ((treadmill - 2**31).astype("int64") - 2**31) * get_speed_scale_to_cm(general_info["rig"], general_info["date"])
        if times is not None:
            self.times = np.linspace(times[0], times[1], len(self.treadmill))
            self.dt = np.median(self.times[1:] - self.times[0:-1])
            self.treadmill_speed = np.diff(self.treadmill) / self.dt

    def speed_interval_timeseries_at(self, target_times, smooth=0.3):
        """Return smoothed treadmill-speed samples at requested target times.

        Args:
            target_times: Explicit timestamps at which values should be sampled.
            smooth: Gaussian smoothing width in seconds.
        """
        time_start = np.searchsorted(self.times[:-1], target_times[0] - smooth * 6)
        time_stop = np.searchsorted(self.times[:-1], target_times[-1] + smooth * 6)
        smoothed = _smooth(
            0.5 * (self.times[time_start:time_stop][:-1] + self.times[time_start:time_stop][1:]),
            self.treadmill_speed[time_start : (time_stop - 1)],
            target_times,
            width=smooth,
        )
        return -np.asarray(smoothed)

    def speed_interval_timeseries(self, start, end, dt=0.1, smooth=0.3):
        """Return smoothed treadmill speed on an evenly spaced interval grid.

        Args:
            start: Interval start time in timeline seconds.
            end: Interval end time in timeline seconds.
            dt: Sampling/bin width in seconds.
            smooth: Gaussian smoothing width in seconds.
        """
        times = np.arange(start, end, dt)
        return self.speed_interval_timeseries_at(times, smooth=smooth)

    def speed_interval_mean(self, start, end):
        """Compute mean treadmill speed over an interval.

        Args:
            start: Interval start time in timeline seconds.
            end: Interval end time in timeline seconds.
        """
        times = (self.times >= start) & (self.times < end)
        return np.mean(self.treadmill_speed[times[:-1]])


class EyeCameraExampleImage(Module):
    """Module storing a representative frame from the eye camera video."""

    NAME = "eye_camera_example_image"

    @staticmethod
    def prepare(data_paths, **_):
        """Load a representative middle-frame grayscale image from the eye video.

        Args:
            data_paths: Candidate raw-data directories for the experiment.
        """
        fn = _multiglob(data_paths, "/*eye.mj2", one=True)
        f = cv2.VideoCapture(fn)
        framecount = int(f.get(cv2.CAP_PROP_FRAME_COUNT))
        framenum = framecount // 2
        f.set(cv2.CAP_PROP_POS_FRAMES, framenum)
        frame = cv2.cvtColor(f.read()[1], cv2.COLOR_BGR2GRAY)
        return {"image": frame, "framenum": framenum}

    def __init__(self, image, framenum, general_info):
        """Store the extracted example eye frame and its frame index.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.image = image
        self.framenum = framenum


class Explog(Module):
    """Module exposing per-session and per-experiment metadata from explog JSON."""

    NAME = "explog"

    @staticmethod
    def prepare(explog_path, expnum, mode, **_):
        """Load the raw experiment log text for the selected experiment and mode.

        Args:
            explog_path: Path to experiment-log JSON file.
            expnum: Experiment number within the session.
            mode: Acquisition mode (`2p` or `ephys`).
        """
        assert mode == "2p" or mode in explog_path, "Wrong mode for explog"
        with open(explog_path, "r") as f:
            contents = f.read()
        return {"rawlog": contents, "expnum": expnum}

    def __init__(self, rawlog, expnum, general_info):
        """Parse session-level and experiment-level metadata from raw explog JSON text.

        Args:
            expnum: Experiment number within the session.
            general_info: General metadata dict from the cache file.
        """
        self.rawlog = rawlog
        log = json.loads(self.rawlog)
        self.session = {k: v for k, v in log.items() if k != "experiments"}
        if not isinstance(log["experiments"], list):
            log["experiments"] = [log["experiments"]]  # If there is only one experiment
        matching_exps = [e for e in log["experiments"] if int(e["number"]) == int(expnum)]
        assert len(matching_exps) == 1, "Exp not found or multiple found with the same number"
        self.exp = matching_exps[0]


class MouseInfo(Module):
    """Module exposing mouse-level metadata fields from `mouseinfo.json`."""

    NAME = "mouse_info"

    @staticmethod
    def prepare(mouseinfo_path, **_):
        """Load raw `mouseinfo.json` text for later attribute-style access.

        Args:
            mouseinfo_path: Path to `mouseinfo.json`.
        """
        with open(mouseinfo_path, "r") as f:
            contents = f.read()
        return {"rawlog": contents}

    def __init__(self, rawlog, general_info):
        """Expose mouse metadata keys as attributes on the module instance.

        Args:
            general_info: General metadata dict from the cache file.
        """
        self.rawlog = rawlog
        keys = json.loads(self.rawlog)
        for k, v in keys.items():
            setattr(self, k, v)


class Protocol(Module):
    """Module for trial parameter tables and condition parsing across MPEP/Rigbox."""

    NAME = "protocol"

    # This is called "protocol" because that is what mpep calls it, and I wrote
    # this for mpep before for mc
    @staticmethod
    def prepare(data_paths, timeline, **_):
        """Parse trial-parameter protocol tables from MPEP `Protocol.mat` or Rigbox block files.

        Args:
            data_paths: Candidate raw-data directories for the experiment.
            timeline: Timeline structure containing DAQ channels and timestamps.
        """
        prot_files_mpep = _multiglob(data_paths, "/Protocol.mat")
        if len(prot_files_mpep) == 1:  # mpep
            prot = load_with_patch(prot_files_mpep[0], struct_as_record=False)["Protocol"][0, 0]
            # If the experiment was terminated early, we need to truncate the
            # protocol to the number of trials that were actually shown.
            mpep_events = [s.split(" ") for s in timeline.mpepUDPEvents if isinstance(s, str)]
            n_trials = sum(e[0] == "StimEnd" for e in mpep_events)
            # Check to be 100% sure
            if not any(e[0] == "ExpInterrupt" for e in mpep_events):
                assert n_trials == np.multiply(*prot.seqnums.shape)
            return {
                "protocol_order": prot.seqnums,
                "xfile": prot.xfile[0],
                "n_trials": n_trials,
                "protocol_param_desc": [str(e[0]).strip() for e in prot.pardefs.squeeze()],
                "protocol_param_names": [str(e[0]).strip() for e in prot.parnames.squeeze()],
                "protocol_params": prot.pars,
            }
        elif len(prot_files_mpep) == 0:  # MC/signals/rigbox
            block_file = _multiglob(data_paths, "/*Block.mat", one=True)
            block = load_with_patch(block_file, struct_as_record=False, squeeze_me=True)["block"]
            xfile = block.expDef
            field_names = np.asarray(
                [f for f in list(block.paramsValues[0]._fieldnames) if f != "services"]
            )  # Services is a string and causes an object array
            aslist = lambda x: x if "__iter__" in x.__dir__() and not isinstance(x, str) else [x]
            field_names_expanded = np.concatenate([[fn] * len(aslist(getattr(block.paramsValues[0], fn))) for fn in field_names])
            _pp = np.asarray([np.concatenate([aslist(getattr(pv, n)) for n in field_names]) for pv in block.paramsValues])
            # Also add the events variables from mc if the number of events is the same as the number of trials
            field_names_events = [
                k[:-6]
                for k in block.events._fieldnames
                if "Values" in k and isinstance(getattr(block.events, k), np.ndarray) and getattr(block.events, k).shape[0] == _pp.shape[0]
            ]
            pp_events = np.asarray([getattr(block.events, k + "Values") for k in field_names_events])
            pp = np.concatenate([_pp, pp_events.T], axis=1)
            if pp.dtype == "O":
                raise ValueError("MC parameters must be numbers, not strings")
            return {
                "protocol_order": np.arange(0, len(block.paramsTimes)),
                "xfile": block.expDef,  # Not actually an xfile for mc, but the closest we have
                "n_trials": len(block.paramsTimes),
                "protocol_param_names": np.concatenate([field_names_expanded, field_names_events]),
                "protocol_param_desc": np.concatenate([field_names_expanded, field_names_events]),
                "protocol_params": pp,
            }
        else:
            raise SystemError("Could not autodetect mc or mpep")

    def __init__(self, protocol_order, xfile, protocol_param_desc, protocol_param_names, protocol_params, n_trials, general_info):
        """Normalize protocol tables into trial-aligned parameter arrays and trial-type indices.

        Args:
            protocol_order: Protocol trial-order matrix or vector.
            protocol_param_desc: Per-parameter human-readable descriptions.
            protocol_param_names: Parameter names from protocol metadata.
            protocol_params: Parameter values by trial type.
            n_trials: Number of valid trials to keep after truncation.
            general_info: General metadata dict from the cache file.
        """
        self.xfile = xfile
        self._param_desc = [d.strip() for d in protocol_param_desc]
        self.param_names = protocol_param_names.tolist()
        # This was built for mpep, so we have to do some weird stuff to get mc
        # in the right format.  Adjust format if coming from mc instead of mpep
        if protocol_order.ndim == 1:
            protocol_order = protocol_order[:, None] + 1
            protocol_params = protocol_params.T
        self._protocol_order = protocol_order
        self._protocol_params = protocol_params
        n_trial_types = protocol_order.shape[0]
        n_repeats = protocol_order.shape[1]
        n_param_types = len(protocol_param_names)
        self.trials = np.zeros((n_param_types, n_trial_types * n_repeats), dtype=object)
        self.trial_index = np.zeros(n_trial_types * n_repeats, dtype=int)
        for i in range(0, protocol_order.shape[0]):
            for j in range(0, protocol_order.shape[1]):
                try:
                    self.trials[:, protocol_order[i, j] - 1] = protocol_params[:, i]
                except:  # If it is a string
                    self.trials[:, protocol_order[i, j] - 1] = 0
                self.trial_index[protocol_order[i, j] - 1] = i + 1
        self.trials = self.trials[:, 0:n_trials]
        self.trial_index = self.trial_index[0:n_trials]

    def param(self, param_name):
        """Return one or more protocol parameter vectors in numeric form when possible.

        Args:
            param_name: Protocol parameter name (or list of names).
        """
        if not isinstance(param_name, list):
            param_name = [param_name]
        for pn in param_name:
            assert pn in self.param_names, "Invalid parameter name"
        i = [self.param_names.index(pn) for pn in param_name]
        trials = self.trials[i, :].squeeze()
        try:
            if np.all(trials.astype(float).round() == trials.astype(float)):
                return trials.astype(int)
            else:
                return trials.astype(float)
        except ValueError:
            return trials

    def param_desc(self, param_name):
        """Return human-readable description text for a protocol parameter.

        Args:
            param_name: Protocol parameter name (or list of names).
        """
        assert param_name in self.param_names, "Invalid parameter name"
        i = self.param_names.index(param_name)
        return self._param_desc[i]

    def expr_to_condition(self, expr):
        """Evaluate a boolean expression over protocol parameters to produce a trial-selection mask.

        Args:
            expr: Boolean expression evaluated against protocol parameter vectors.
        """
        locs = {n: self.param(n) for n in self.param_names}
        inds = eval(expr, globals(), locs)
        assert isinstance(inds, np.ndarray)
        return inds


class _Segmented(Module):
    """Internal base module for sparse cached 3D segmentation labels."""

    MODE = "2p"

    @classmethod
    def prepare(cls, processed_data_path, rig, **_):
        """Convert dense segmentation volumes into sparse `(z, y, x, label)` rows for compact caching.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            rig: Rig identifier used for channel maps and calibration.
        """
        cells = cls._load(processed_data_path, rig)
        inds = np.where(cells != 0)
        vals = cells[inds]
        position_values = np.vstack([*inds, vals]).T
        assert np.max(position_values) < np.iinfo(np.int16).max
        assert np.min(position_values) > np.iinfo(np.int16).min
        return {"zstack": np.int16(position_values)}

    def __init__(self, zstack, general_info):
        """Reconstruct dense segmentation volume and initialize cached per-cell metrics.

        Args:
            zstack: Sparse segmentation representation (`z, y, x, label`).
            general_info: General metadata dict from the cache file.
        """
        self._zstack = zstack
        volume = np.zeros(general_info["fov"])
        volume[zstack[:, 0], zstack[:, 1], zstack[:, 2]] = zstack[:, 3]
        self.zstack = volume
        self._planespercell = None
        self._voxelspercell = None

    def cell_positions(self):
        """Return median `(z, y, x)` position for each segmented label."""
        cells = list(sorted(set(self._zstack[:, 3])))
        medians = []
        for c in cells:
            cell_poss = self._zstack[self._zstack[:, 3] == c]
            medians.append(np.median(cell_poss[:, :3], axis=0))
        return np.asarray(medians)

    def n_planes_per_cell(self):
        """Return number of z-planes occupied by each segmented cell label."""
        if self._planespercell is not None:
            return self._planespercell
        _cells_per_plane = [np.unique(plane) for plane in self.zstack]
        _unique = np.unique(np.concatenate(_cells_per_plane), return_counts=True)
        assert np.all(_unique[0] == np.arange(0, len(_unique[0]), dtype=int)), "Order not followed"
        n_planes = _unique[1][1:]  # [1:] excludes zero
        self._planespercell = n_planes
        return n_planes

    def n_voxels_per_cell(self):
        """Return voxel count for each segmented cell label."""
        if self._voxelspercell is not None:
            return self._voxelspercell
        # Count the size of each cell in the mask
        _unique = np.unique(self.zstack, return_counts=True)
        assert np.all(_unique[0] == np.arange(0, len(_unique[0]), dtype=int)), "Order not followed"
        n_voxels = _unique[1][1:]  # [1:] excludes zero
        self._voxelspercell = n_voxels
        return n_voxels


class SegmentedGCamp(_Segmented):
    """Segmented GCaMP structural mask volume from Cellpose/volume preprocessing."""

    NAME = "segmented_gcamp"

    @staticmethod
    def _load(processed_data_path, rig):
        """Load GCaMP segmentation masks from Cellpose outputs.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            rig: Rig identifier used for channel maps and calibration.
        """
        try:
            fn = f"{processed_data_path}segmentation_channel{CHANNELMAP[rig]['green']}_seg.npy"
            f = _load_npy_compressed(fn, allow_pickle=True)[()]
            return f["masks"]
        except FileNotFoundError:
            fn = f"{processed_data_path}segmentation_channel{CHANNELMAP[rig]['green']}_mask.npz"
            return np.asarray(np.load(fn)["mask"])


class SegmentedMCherry(_Segmented):
    """Segmented mCherry structural mask volume from Cellpose/volume preprocessing."""

    NAME = "segmented_mcherry"

    @staticmethod
    def _load(processed_data_path, rig):
        """Load mCherry segmentation masks from Cellpose outputs.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            rig: Rig identifier used for channel maps and calibration.
        """
        try:
            fn = f"{processed_data_path}segmentation_channel{CHANNELMAP[rig]['red']}_seg.npy"
            f = _load_npy_compressed(fn, allow_pickle=True)[()]
            return f["masks"]
        except FileNotFoundError:
            fn = f"{processed_data_path}segmentation_channel{CHANNELMAP[rig]['red']}_mask.npz"
            return np.asarray(np.load(fn)["mask"])


class SegmentedThreshMCherry(_Segmented):
    """Threshold-derived mCherry segmentation volume."""

    NAME = "segmented_thresh_mcherry"

    @staticmethod
    def _load(processed_data_path, rig):
        """Load threshold-based mCherry segmentation labels.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            rig: Rig identifier used for channel maps and calibration.
        """
        fn = f"{processed_data_path}segmentation_threshold_chan{CHANNELMAP[rig]['red']}.npz"
        f = _load_npy_compressed(fn, allow_pickle=True)
        return f["labels"]


class _StackImage(Module):
    """Internal base module for dense channel-specific stack images."""

    MODE = "2p"
    channel = None  # Set to "green" or "red" when inheriting

    @classmethod
    def prepare(cls, processed_data_path, rig, im_meta, **_):
        """Load precomputed mean/max stack images for a selected imaging channel.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            rig: Rig identifier used for channel maps and calibration.
            im_meta: ScanImage frame metadata for the experiment.
        """
        _chansav = im_meta["SI.hChannels.channelSave"]
        channels = [_chansav] if isinstance(_chansav, (float, int)) else np.asarray(_chansav).flatten().tolist()
        channel_for_colour = CHANNELMAP[rig][cls.channel]
        if channel_for_colour not in channels:
            raise RuntimeError(f"Channel {cls.channel} number {channel_for_colour} not found: instead, found channels {channels}")
        fn = f"{processed_data_path}{CHANNELWISE_MEAN_IMAGE_FILENAME}"
        img = _load_npy_compressed(fn)[channels.index(channel_for_colour)]
        fn = f"{processed_data_path}{CHANNELWISE_MAX_IMAGE_FILENAME}"
        img_max_proj = _load_npy_compressed(fn)[channels.index(channel_for_colour)]
        return {"images": img.astype("float16"), "images_max_projection": img_max_proj.astype("float16")}

    def __init__(self, images, general_info, images_max_projection=None):
        """Store stack images and optional max-projection images.

        Args:
            images: Image stack array stored in cache.
            general_info: General metadata dict from the cache file.
            images_max_projection: Precomputed max-projection stack image.
        """
        self.images = images
        self.images_max_projection = images_max_projection


class StackImageGCamp(_StackImage):
    """Dense GCaMP stack image volume (non-lossy per-plane arrays)."""

    NAME = "stack_image_gcamp"
    channel = "green"


class StackImageMCherry(_StackImage):
    """Dense mCherry stack image volume (non-lossy per-plane arrays)."""

    NAME = "stack_image_mcherry"
    channel = "red"


class StackImageFarRed(_StackImage):
    """Dense far-red stack image volume (non-lossy per-plane arrays)."""

    NAME = "stack_image_farred"
    channel = "farred"


class StackImageDiB(_StackImage):
    """Dense blue-channel stack image volume (non-lossy per-plane arrays)."""

    NAME = "stack_image_blue"
    channel = "blue"


class _LossyImages(Module):
    """Lossy JPEG-compressed summary images of the 2p field of view.

    These images are made for viewing rather than detailed analysis, since they
    JPEG-compress data first.  This gives a substantial (100s of mb to ~1mb)
    space savings, but makes them inappropriate for detailed quantitative analysis.
    """

    MODE = "2p"

    @classmethod
    def prepare(cls, processed_s2p_path, rig, im_meta, **_):
        """JPEG-compress stack images into a compact cache representation for visualization workflows.

        Args:
            processed_s2p_path: Directory containing Suite2p outputs.
            rig: Rig identifier used for channel maps and calibration.
            im_meta: ScanImage frame metadata for the experiment.
        """
        files = []
        prevmins = []
        prevmaxes = []
        kinds = []
        transforms = []
        for kind in ["mean", "max", "meanred", "maxred"]:
            try:
                f = cls._load(processed_s2p_path, rig, im_meta, kind)
            except FileNotFoundError:
                continue
            except KeyError:
                continue
            if f is None:
                continue
            transform = "none"
            # Cut off small values to avoid outliers messing with the
            # normalisation.
            q = np.quantile(f.flatten(), 0.02)
            f[f < q] = q
            # Transform if it will help equalise the histogram
            if scipy.stats.skew(f.flatten()) > 25:
                transform = "log100truncate"
                f = np.log(100 + np.maximum(0, f))
            # Rescale uniformly for all images, otherwise it is automatic
            prevmin, prevmax = np.min(f), np.max(f)
            prevmins.append(prevmin)
            prevmaxes.append(prevmax)
            fnorm = ((f - prevmin) / (prevmax - prevmin) * 255).astype(np.uint8)
            for i in range(0, f.shape[0]):
                pseudofile = io.BytesIO()
                imageio.imsave(pseudofile, fnorm[i], format="jpeg", quality=95)
                files.append(np.frombuffer(pseudofile.getvalue(), dtype=np.uint8))
                kinds.append(kind)
                transforms.append(transform)
        lens = list(map(len, files))
        return {"imagedata": np.concatenate(files), "lengths": lens, "imrange": [prevmins, prevmaxes], "kinds": kinds, "transform": transforms}

    def __init__(self, imagedata, lengths, general_info, imrange=(0, 255), kinds=None, transform=None):
        """Decode cached lossy images and group them by image kind.

        Args:
            imagedata: Concatenated JPEG byte stream for lossy images.
            lengths: Per-image byte lengths used to decode `imagedata`.
            general_info: General metadata dict from the cache file.
            imrange: Per-kind min/max range used when de-normalizing lossy images.
            kinds: Per-image kind labels aligned to `lengths`.
            transform: Per-image transform labels used before compression.
        """
        if kinds is None:  # Old data format compatibility
            imrange = [[imrange[0]], [imrange[1]]]
            kinds = ["mean"] * len(lengths)
        if isinstance(transform, str) or transform is None:
            transform = [transform] * len(lengths)
        # Decode images from all
        kinds_order = np.asarray(kinds)[np.sort(np.unique(kinds, return_index=True)[1])].tolist()
        images_by_kind = {str(k): [] for k in np.unique(kinds)}
        ibase = 0
        for i, l in enumerate(lengths):
            pseudofile = io.BytesIO(imagedata[ibase : (ibase + l)].tobytes())
            ki = kinds_order.index(kinds[i])
            im = np.asarray(imageio.imread(pseudofile, format="jpeg")) / 255 * (imrange[1][ki] - imrange[0][ki]) + imrange[0][ki]
            if transform[i] == "log100truncate":
                im = np.exp(im) - 100
            images_by_kind[kinds[i]].append(im)
            ibase += l
        self.images_by_kind = {k: np.asarray(v) for k, v in images_by_kind.items()}
        self.images = self.images_by_kind["mean"]
        self.imrange = imrange
        self.transforms = transform

    def __call__(self, kind=None):
        """Return default mean images or images for a specific stored kind.

        Args:
            kind: Image kind key (for example `mean`, `max`, `meanred`).
        """
        if kind is None:
            return self.images
        return self.images_by_kind[kind]


class LossyImages(_LossyImages):
    """Lossy JPEG-compressed Suite2p summary images across imaging planes."""

    NAME = "lossy_images"

    @classmethod
    def _load(cls, processed_s2p_path, rig, im_meta, kind):
        """Load Suite2p summary images for the requested image kind across planes.

        Args:
            processed_s2p_path: Directory containing Suite2p outputs.
            rig: Rig identifier used for channel maps and calibration.
            im_meta: ScanImage frame metadata for the experiment.
            kind: Image kind key (for example `mean`, `max`, `meanred`).
        """
        namemap = {"mean": "meanImg", "max": "max_proj", "meanred": "meanImg_chan2", "maxred": "max_proj_chan2"}
        if kind not in namemap.keys():
            return
        n_planes = im_meta["SI.hStackManager.numSlices"]
        ops = {}
        meanimg = []
        for i in range(0, n_planes):
            path_ops = f"{processed_s2p_path}suite2p/plane{i}/ops.npy"
            f = _load_npy_compressed(path_ops, allow_pickle=True)[()]
            img = f[namemap[kind]]
            # Max projection is a different size in s2p, this is a way to get around that
            if img.shape != (f["Ly"], f["Lx"]):
                img2 = np.zeros((f["Ly"], f["Lx"]), dtype=img.dtype)
                img2[f["yrange"][0] : (f["yrange"][0] + img.shape[0]), f["xrange"][0] : (f["xrange"][0] + img.shape[1])] = img
                img = img2
            meanimg.append(img)
        return np.asarray(meanimg)


class BrainViewer(Module):
    """Get a lossy video of the fov for a single plane"""

    MODE = "2p"
    NAME = "brain_viewer"
    EXPORT = ["save_video"]

    @classmethod
    def prepare(cls, data_paths, processed_data_path, im_meta, **_):
        """Build an 8-bit visualization movie from raw 2p frames and record frame rate.

        Args:
            data_paths: Candidate raw-data directories for the experiment.
            processed_data_path: Directory containing processed intermediate outputs.
            im_meta: ScanImage frame metadata for the experiment.
        """
        paths = list(sorted(_multiglob(data_paths, "*2P*.tif")))
        tiffs = [tifffile.TiffFile(p) for p in paths]
        fps = im_meta["SI.hRoiManager.scanFrameRate"]
        alltiffs = np.concatenate([tiff.asarray() for tiff in tiffs])
        alltiffs[alltiffs < 0] = 0
        upper_bound = np.quantile(np.sqrt(alltiffs), 0.99)
        alltiffs = np.digitize(np.sqrt(alltiffs), np.linspace(1, upper_bound, 255)).astype("uint8")
        return {"video": alltiffs, "fps": fps}

    def __init__(self, video, fps, general_info):
        """Store visualization movie data with frame-rate metadata.

        Args:
            video: Video frames array stored in cache.
            fps: Template sampling rate in frames/second.
            general_info: General metadata dict from the cache file.
        """
        self.video = video
        self.fps = fps
        self.fov = general_info["fov"]

    def save_video(self, filename, n_frames_smoothing=1):
        """Write the cached movie to an MP4 file with optional temporal smoothing.

        Args:
            filename: Output filename or cache filename, depending on context.
            n_frames_smoothing: Temporal averaging window (in frames) before video export.
        """
        import cv2

        pseudofile = io.BytesIO()
        if n_frames_smoothing == 1:
            v = self.video
        else:
            v = scipy.ndimage.convolve(self.video.astype(float), np.ones((n_frames_smoothing, 1, 1)) / n_frames_smoothing).astype("uint8")
        vid = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.fov[1], self.fov[2]), False)
        for i in range(0, v.shape[0]):
            vid.write(v[i])
        vid.release()


class MeanPixels(Module):
    """Per-plane mean-intensity traces derived from raw two-photon frames."""

    MODE = "2p"
    NAME = "meanpixels"

    @classmethod
    def prepare(cls, data_paths, processed_data_path, im_meta, **_):
        """Compute per-frame mean intensity and reshape it into plane-major traces.

        Args:
            data_paths: Candidate raw-data directories for the experiment.
            processed_data_path: Directory containing processed intermediate outputs.
            im_meta: ScanImage frame metadata for the experiment.
        """
        N_planes = im_meta["SI.hStackManager.numSlices"]
        paths = list(sorted(_multiglob(data_paths, "*2P*.tif")))
        tiffs = [tifffile.TiffFile(p) for p in paths]
        mean = np.concatenate([tiff.asarray().mean(axis=(1, 2)) for tiff in tiffs])
        meancrop = mean[0 : (N_planes * (len(mean) // N_planes))]
        rsmean = meancrop.reshape(-1, N_planes).T
        return {"mean_pixels": rsmean}

    def __init__(self, mean_pixels, general_info):
        # We want to mimic the format of actual readouts of neural activity for
        # use with the intervals module
        """Expose mean-pixel traces using the same interface as neural time-series modules.

        Args:
            mean_pixels: Plane-major per-frame mean-intensity traces.
            general_info: General metadata dict from the cache file.
        """
        self.timeseries = mean_pixels
        self.n_cells = mean_pixels.shape[0]
        self.n_timepoints = mean_pixels.shape[1]
        self.plane = np.arange(0, self.n_cells).astype(int)


class _LossyStackImage(_LossyImages):
    """Lossy JPEG-compressed stack images produced by the structural volume pipeline."""

    channel = None  # Set to "green" or "red" when inheriting

    @classmethod
    def _load(cls, processed_data_path, rig, im_meta, kind):
        """Load channel-specific stack mean/max images produced by the volume pipeline.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            rig: Rig identifier used for channel maps and calibration.
            im_meta: ScanImage frame metadata for the experiment.
            kind: Image kind key (for example `mean`, `max`, `meanred`).
        """
        if kind not in ["mean", "max"]:
            return
        _chansav = im_meta["SI.hChannels.channelSave"]
        channels = [_chansav] if isinstance(_chansav, (float, int)) else np.asarray(_chansav).flatten().tolist()
        channel_for_colour = CHANNELMAP[rig][cls.channel]
        if channel_for_colour not in channels:
            raise RuntimeError(f"Channel {cls.channel} number {channel_for_colour} not found: instead, found channels {channels}")
        kindstr = CHANNELWISE_MEAN_IMAGE_FILENAME if kind == "mean" else CHANNELWISE_MAX_IMAGE_FILENAME
        fn = f"{processed_data_path}{kindstr}"
        return _load_npy_compressed(fn)[channels.index(channel_for_colour)]


class LossyStackImageGCamp(_LossyStackImage):
    """GCaMP lossy stack images."""

    NAME = "lossy_stack_image_gcamp"
    channel = "green"


class LossyStackImageMCherry(_LossyStackImage):
    """mCherry lossy stack images."""

    NAME = "lossy_stack_image_mcherry"
    channel = "red"


class LossyStackImageDiB(_LossyStackImage):
    """Blue-channel lossy stack images."""

    NAME = "lossy_stack_image_blue"
    channel = "blue"


class LossyStackImageFarRed(_LossyStackImage):
    """Far-red lossy stack images."""

    NAME = "lossy_stack_image_farred"
    channel = "farred"


class RetinotopicMap(Module):
    """The output of the SVD-based retinotopy mapping script.

    This requires the mapping script quickPixelMapRet.m to be run first (the
    version I modified to save its output).  This is just a simple wrapper
    which saves the full retinotopy mapping output and loads it again, with
    some helper functions to display relevant plots.
    """

    MODE = "2p"
    NAME = "retinotopic_map"

    def prepare(processed_data_path, data_paths, **_):
        """Load retinotopy map output and associated stimulus frames/timestamps from preprocessing files.

        Args:
            processed_data_path: Directory containing processed intermediate outputs.
            data_paths: Candidate raw-data directories for the experiment.
        """
        fn_mpep = _multiglob([processed_data_path] + data_paths, "retinotopy.mat")
        # fn_mpep = f"{processed_data_path}retinotopy.mat"
        block_files = _multiglob(data_paths, "/*Block.mat")
        if len(fn_mpep) > 0:  # mpep
            fn_mpep = fn_mpep[0]
            mat = scipy.io.loadmat(fn_mpep)
            m = mat["planeRFs"]
            m = m.reshape(*m.shape[0:2], int(np.sqrt(m.shape[2])), int(np.sqrt(m.shape[2]))).astype("float16")
            azi = mat["aziDeg"]
            elv = mat["elvDeg"]
            stim = scipy.io.loadmat(_multiglob([processed_data_path] + data_paths, "stimImg.mat")[0])["stimImg"].astype(np.int8)
            stimtimes = scipy.io.loadmat(_multiglob([processed_data_path] + data_paths, "stimTimes.mat")[0])["stimTimes"].flatten()
            return {"retinotopy": m, "azimouth_angle": azi, "elevation_angle": elv, "stim": stim, "stimtimes": stimtimes, "mode": "mpep"}
        elif len(block_files) == 1:  # mc
            block = load_with_patch(block_files[0], struct_as_record=False, squeeze_me=True)["block"]
            ny = block.events.stimuliOnValues.shape[0]
            nx = block.events.stimuliOnValues.shape[1] // block.events.stimuliOnTimes.shape[0]
            stim = block.events.stimuliOnValues[:, nx:].reshape(ny, nx, -1)
            stimtimes_before_diode_alignment = block.events.stimuliOnTimes[1:]
            return {
                "retinotopy": np.asarray([0]),
                "azimouth_angle": 0,
                "elevation_angle": 0,
                "stim": stim,
                "stimtimes": stimtimes_before_diode_alignment,
                "mode": "mc_AP",
            }
        else:
            raise ValueError("Retinotopy not found!  If this was run via mpep, you need to run the processing script first to generate the stimulus.")

    def __init__(self, retinotopy, azimouth_angle, elevation_angle, general_info, stim=None, stimtimes=None, mode="mpep"):
        """Store retinotopy maps, stimulus metadata, and source-mode indicator.

        Args:
            retinotopy: Retinotopy map tensor from preprocessing output.
            azimouth_angle: Azimuth axis values for retinotopy maps.
            elevation_angle: Elevation axis values for retinotopy maps.
            general_info: General metadata dict from the cache file.
            stim: Stimulus tensor associated with retinotopy mapping.
            stimtimes: Stimulus-frame timestamps for retinotopy mapping.
            mode: Acquisition mode (`2p` or `ephys`).
        """
        self.retinotopy = retinotopy.astype("float32")
        self.azimouth_angle = azimouth_angle
        self.elevation_angle = elevation_angle
        self.stim = stim
        self.stimtimes = stimtimes
        self.mode = mode

    def selectivity_map(self):
        """For each point on the 2p image, which areas are selective?"""
        return np.mean(self.abs(self.retinotopy), axis=(0, 1))

    def visual_space_selectivity(self):
        """For each point on the screen, which areas have cells selective for them?"""
        return np.mean(self.abs(self.retinotopy), axis=(2, 3))


MODULE_TYPES = _find_subclasses(Module)


#################### SECTION: Mixins ####################


class CellPositionInMicrons(Mixin):
    """Mixin converting Suite2p cell centroids from voxel coordinates to microns."""

    MODE = "2p"
    REQUIRED_MODULES = [CellInfo, Pixel2Micron]
    NAME = "cell_position_in_microns"

    def __call__(self):
        """Return Suite2p cell centroids converted to micron coordinates."""
        return self.pixel2micron.get_position(self.cellinfo.cell_positions())


class CellPositionInMicronsMCherry(Mixin):
    """Cell positions of mCherry cells from manual segmentation"""

    MODE = "2p"
    REQUIRED_MODULES = [Pixel2Micron, SegmentedMCherry]
    NAME = "cell_position_in_microns_mcherry"

    def __call__(self):
        """Return mCherry segmented cell centroids converted to micron coordinates."""
        return self.pixel2micron.get_position(self.segmented_mcherry.cell_positions())


class CellPositionInMicronsGCamp(Mixin):
    """Cell positions of gcamp cells from static gcamp or a functional zstack"""

    MODE = "2p"
    REQUIRED_MODULES = [Pixel2Micron, SegmentedGCamp]
    NAME = "cell_position_in_microns_gcamp"

    def __call__(self):
        """Return GCaMP segmented cell centroids converted to micron coordinates."""
        return self.pixel2micron.get_position(self.segmented_gcamp.cell_positions())


# TODO support ephys
class _Intervals(Mixin):
    """Internal mixin for extracting interval-aligned neural time series."""

    MODE = ["2p", "ephys"]
    REQUIRED_MODULES = {"ephys": [EphysSpikes], "2p": [NeuralFrameTimings]}
    OPTIONAL_MODULES = [FunctionalF, FunctionalSpikes, FunctionalNeuropil, MeanPixels]
    default_measurement = None
    _measurements = {
        "f": lambda self: (self.f, _smooth_interp),
        "spikes": lambda self: (self.spikes, _smooth),
        "neuropil": lambda self: (self.neuropil, _smooth_interp),
        "mean_pixels": lambda self: (self.meanpixels, _smooth_interp),
    }

    # TODO This is mostly repeated code from interval_timeseries
    def interval_mean(self, start, end, cells=None, measurement=None):
        """Find the mean spike rate in the interval between start and end"""
        if self.mode == "ephys":
            if measurement is not None and measurement != "lfp" and measurement != "spikes":
                print(f'Warning: Measurement "{measurement}" ignored in ephys mode')
            if measurement == "lfp":
                if cells is not None:
                    print('Warning: "cells" argument ignored in lfp mode')
                return np.asarray(list(map(mean, self.lfp.interval_timeseries(start, end)))) / (end - start)
            return np.asarray(list(map(len, self.ephys_spikes.spikes_interval((start, end), cells=cells)))) / (end - start)
        if measurement is None:
            measurement = self.default_measurement
        data = self._measurements[measurement](self)[0]
        cells = _interpret_cells_argument(cells, data.n_cells)
        n_timepoints = data.timeseries.shape[1]
        # Just in case the length of the frame times from timeline is shorter
        # than the number of tiffs due to technical issues
        min_frame_time_len = min([len(self.neural_timings.frame_times(i)) for i in range(0, self.neural_timings.n_planes)])
        n_timepoints = min(n_timepoints, min_frame_time_len)
        plane_times = np.asarray([self.neural_timings.frame_times(i)[0:n_timepoints] for i in range(0, self.neural_timings.n_planes)])
        timeseries_cells = data.timeseries[cells][:, 0:n_timepoints]
        plane_cells = data.plane[cells]
        plane_inds = np.asarray([(plane_times[plane] >= start) & (plane_times[plane] < end) for plane in range(0, self.neural_timings.n_planes)])
        plane_timepoints = [plane_times[i][plane_inds[i]] for i in range(0, self.neural_timings.n_planes)]
        cell_inds = np.asarray([plane_inds[plane_cells[cell]] for cell in range(0, len(plane_cells))]).astype(bool)
        means = []
        for cell in range(0, cell_inds.shape[0]):
            means.append(np.mean(timeseries_cells[cell][cell_inds[cell]]))
        return np.asarray(means)

    def binned_timeseries(self, start, end, cells=None, dt=0.1, measurement=None, continuity_correction=True, normalise_by="bin"):
        """Compute binned activity over an interval using evenly spaced bins.

        Args:
            start: Interval start time in timeline seconds.
            end: Interval end time in timeline seconds.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
            dt: Sampling/bin width in seconds.
            measurement: Measurement key selecting which signal to extract.
            continuity_correction: If true, expands interval edges by half a bin to reduce edge bias.
            normalise_by: Binning normalization mode (`bin`, `cell`, or `none`).
        """
        if continuity_correction:
            start = start - dt / 2
            end = end + dt / 2
        bins = np.arange(start, end + 1e-6, dt)
        return self.binned_timeseries_at(bins=bins, cells=cells, measurement=measurement, normalise_by=normalise_by)

    def binned_timeseries_at(self, bins, cells=None, measurement=None, normalise_by="bin"):
        # normalise_by can be "bin" (divide each bin count by the number of
        # observations from that bin), "cell" (divide by the total number of
        # bins from the cell), or "none".
        """Compute binned activity using explicit bin edges.

        Args:
            bins: Histogram bin edges.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
            measurement: Measurement key selecting which signal to extract.
            normalise_by: Binning normalization mode (`bin`, `cell`, or `none`).
        """
        assert self.mode == "2p", "Cannot be used for ephys"
        if measurement is None:
            measurement = self.default_measurement
        data = self._measurements[measurement](self)[0]
        cells = _interpret_cells_argument(cells, data.n_cells)
        n_timepoints = data.timeseries.shape[1]
        # Just in case the length of the frame times from timeline is shorter
        # than the number of tiffs due to technical issues
        plane_times = [self.neural_timings.frame_times(i)[0:n_timepoints] for i in range(0, self.neural_timings.n_planes)]
        timeseries_cells = data.timeseries[cells][:, 0:n_timepoints]
        bin_counts_by_plane = np.asarray([np.histogram(plane_times[i], bins=bins)[0] for i in range(0, self.neural_timings.n_planes)])
        if np.any(bin_counts_by_plane == 0):
            print("Warning: Some bins in this timeseries have zero neural frames")
        hists = []
        for i in range(0, len(cells)):
            h = np.histogram(plane_times[data.plane[i]], weights=timeseries_cells[i], bins=bins)[0]
            if normalise_by == "bin":
                h = h / bin_counts_by_plane[data.plane[i]]
            elif normalise_by == "cell":
                h = h / np.sum(bin_counts_by_plane[data.plane[i]], axis=1)[data.plane[i]]
            hists.append(h)
        return np.asarray(hists)

    def interval_timeseries(self, start, end, cells=None, smooth=0.3, dt=0.1, measurement=None):
        """Find the smoothed spike rate at times [start, start+dt, ..., end]"""
        if self.mode == "ephys" and measurement == "lfp":
            return self.lfp.interval_timeseries(start, end)
        target_times = np.arange(start, end, dt)
        return self.interval_timeseries_at(target_times, cells=cells, smooth=smooth, measurement=measurement)

    def interval_timeseries_at(self, target_times, cells=None, smooth=0.3, measurement=None):
        """Find the smoothed spike rate at each time point in target_times"""
        padded_start = target_times[0] - smooth * 3
        padded_end = target_times[-1] + smooth * 3
        if self.mode == "ephys":
            if measurement is not None and measurement != "lfp" and measurement != "spikes":
                print(f'Warning: Measurement "{measurement}" ignored in ephys mode')
            cells = _interpret_cells_argument(cells, self.ephys_spikes.cluster_group.shape[0])
            tss = []
            spike_times_filt = self.ephys_spikes.spike_times[
                (self.ephys_spikes.spike_times > padded_start) & (self.ephys_spikes.spike_times < padded_end)
            ]
            spike_clusters_filt = self.ephys_spikes.spike_clusters[
                (self.ephys_spikes.spike_times > padded_start) & (self.ephys_spikes.spike_times < padded_end)
            ]
            for i, cellid in enumerate(self.ephys_spikes.cluster_group[:, 0][cells]):
                spike_times = spike_times_filt[spike_clusters_filt == cellid]
                tss.append(_smooth_univariate(spike_times, spike_times * 0 + 1, target_times, smooth))
            return np.asarray(tss)
        if measurement is None:
            measurement = self.default_measurement
        data, smoothfunc = self._measurements[measurement](self)
        n_timepoints = data.timeseries.shape[1]
        cells = _interpret_cells_argument(cells, data.n_cells)
        # Just in case the length of the frame times from timeline is shorter
        # than the number of tiffs due to technical issues
        min_frame_time_len = min([len(self.neural_timings.frame_times(i)) for i in range(0, self.neural_timings.n_planes)])
        n_timepoints = min(n_timepoints, min_frame_time_len)
        plane_times = np.asarray([self.neural_timings.frame_times(i)[0:n_timepoints] for i in range(0, self.neural_timings.n_planes)])
        timeseries_cells = data.timeseries[cells][:, 0:n_timepoints]
        plane_cells = data.plane[cells]
        plane_inds = np.asarray(
            [(plane_times[plane] >= padded_start) & (plane_times[plane] < padded_end) for plane in range(0, self.neural_timings.n_planes)]
        )
        plane_timepoints = [plane_times[i][plane_inds[i]] for i in range(0, self.neural_timings.n_planes)]
        cell_inds = np.asarray([plane_inds[plane_cells[cell]] for cell in range(0, len(plane_cells))]).astype(bool)
        tss = []
        # for cell in range(0, cell_inds.shape[0]):
        #     # TODO do this by plane
        #     tss.append(smoothfunc(plane_timepoints[plane_cells[cell]], timeseries_cells[cell][cell_inds[cell]], target_times, smooth))
        # return np.asarray(tss)
        tss = np.zeros((cell_inds.shape[0], len(target_times))) * np.nan
        for plane in range(0, self.neural_timings.n_planes):
            if not np.any(plane_cells == plane):
                continue
            _tss = smoothfunc(plane_timepoints[plane], timeseries_cells[plane_cells == plane][:, plane_inds[plane]], target_times, smooth)
            tss[plane_cells == plane] = _tss
        return np.asarray(tss)


class SpikeIntervals(_Intervals):
    """Interval-analysis mixin defaulting to deconvolved spike measurements."""

    NAME = "spike_intervals"
    default_measurement = "spikes"


class FIntervals(_Intervals):
    """Interval-analysis mixin defaulting to raw fluorescence measurements."""

    NAME = "f_intervals"
    default_measurement = "f"


class NeuropilIntervals(_Intervals):
    """Interval-analysis mixin defaulting to neuropil fluorescence measurements."""

    NAME = "neuropil_intervals"
    default_measurement = "neuropil"


class EasyTimeseries(Mixin):
    """Provide a single API for timeseries data across 2p, ephys, eye, and running signals."""

    MODE = ["2p", "ephys"]
    NAME = "easy_timeseries"
    REQUIRED_MODULES = [GeneralInfo]
    OPTIONAL_MODULES = [
        FunctionalSpikes,
        FunctionalF,
        FunctionalNeuropil,
        EphysSpikes,
        EyeTrackingDLC,
        MeanPixels,
        Treadmill,
        Ball,
        LFP,
        MpepEvents,
        NeuralFrameTimings,
    ]
    OPTIONAL_MIXINS = [SpikeIntervals]
    EXPORT = ["timeseries", "timeseries_at", "interval_mean", "start_end_time"]

    # TODO
    def timeseries(self, intervals, measurement=None, dt=0.1, smooth=0.3, replacenan=False, prefilter=True, cells=None):
        """Return a timeseries from this experiment.  Timeseries may be multidimensional.

        `intervals` can be either a tuple in the form of (start, end), or a list
        of tuples in the form (start, end).  If it is a list of tuples, this
        function returns a list of timeseries.

        `measurement` is the type of timeseries.  Valid values are:
        - "dspikes": Deconvolved fluorescence for all cells (2p only)
        - "f": Raw fluorescence for all cells (2p only)
        - "neuropil": Neuropil fluorescence for all cells (2p only)
        - "mean_pixels": Mean fluorescence of the frame (2p only)
        - "dff": Δf/f for all cells (2p only)
        - "pupil_size": Pupil size
        - "eye_x": Pupil x position
        - "eye_y": Pupil y position
        - "motion_energy": Motion energy from the eye camera video
        - "running": Forward running speed
        - "lfp": Local field potential (ephys only)
        - "spikes": action potentials for all cells (ephys only)
        - "times": The timepoints corresponding to the interval(s).  Ignores all parameteres besides `intervals` and `dt`.

        `dt` is the spacing of the timeseries

        `smooth` can be one of the following:
        - A number, specifying the smoothing Gaussian width
        - The string "bin", for the sum of spikes (ephys or 2p) in the window
        - The string "meanbin", for the sum of spikes divided by the number of samples in the bin (recommended for 2p)

        `replacenan` and `prefilter` apply only to eye data

        `cells` limits the cells returned by the function, for ephys and 2p timeseries only.
        """
        if hasattr(intervals[0], "__iter__"):
            return [
                self.timeseries(intvl, measurement=measurement, dt=dt, smooth=smooth, replacenan=replacenan, prefilter=prefilter, cells=cells)
                for intvl in intervals
            ]
        # TODO df/f
        # TODO LFP
        # TODO get replacenans to work in the smoothing function
        # TODO timeseries_at
        if measurement == "dspikes":
            if smooth in ["bin", "meanbin"]:
                return self.spike_intervals.binned_timeseries(
                    intervals[0],
                    intervals[1],
                    measurement="spikes",
                    dt=dt,
                    cells=cells,
                    continuity_correction=True,
                    normalisation=("none" if smooth == "bin" else "bin"),
                )
            else:
                return self.spike_intervals.interval_timeseries(intervals[0], intervals[1], measurement="spikes", dt=dt, smooth=smooth, cells=cells)
        elif measurement == "f":
            if smooth in ["bin", "meanbin"]:
                return self.spike_intervals.binned_timeseries(
                    intervals[0],
                    intervals[1],
                    measurement="f",
                    dt=dt,
                    cells=cells,
                    continuity_correction=True,
                    normalisation=("none" if smooth == "bin" else "bin"),
                )
            else:
                return self.spike_intervals.interval_timeseries(intervals[0], intervals[1], measurement="f", dt=dt, smooth=smooth, cells=cells)
        elif measurement == "neuropil":
            if smooth in ["bin", "meanbin"]:
                return self.spike_intervals.binned_timeseries(
                    intervals[0],
                    intervals[1],
                    measurement="neuropil",
                    dt=dt,
                    cells=cells,
                    continuity_correction=True,
                    normalisation=("none" if smooth == "bin" else "bin"),
                )
            else:
                return self.spike_intervals.interval_timeseries(intervals[0], intervals[1], measurement="neuropil", dt=dt, smooth=smooth, cells=cells)
        elif measurement == "mean_pixels":
            return self.spike_intervals.interval_timeseries(intervals[0], intervals[1], measurement="mean_pixels", dt=dt, smooth=smooth)
        elif measurement == "dff":
            raise NotImplementedError("dF/F is not yet implemented")
        elif measurement == "spikes":
            if smooth == "bin":
                return self.ephys_spikes.spikes_histogram(intervals, dt=dt, cells=cells)
            else:
                return self.spike_intervals.interval_timeseries(intervals[0], intervals[1], measurement="spikes", dt=dt, smooth=smooth, cells=cells)
        elif measurement == "spiketimes":
            return self.ephys_spikes.spikes_interval(intervals, cells=cells)
        elif measurement == "lfp":
            raise NotImplementedError("lfp is not yet implemented")
        elif measurement in ["pupil_size", "eye_x", "eye_y", "motion_energy"] and not hasattr(self, "eye_dlc"):
            return np.full(len(np.arange(intervals[0], intervals[1], dt)), np.nan)
        elif measurement == "pupil_size":
            return self.eye_dlc.interval_timeseries(
                intervals[0], intervals[1], measurement="pupil", dt=dt, smooth=smooth, replacenan=replacenan, prefilter=prefilter
            )
        elif measurement == "eye_x":
            return self.eye_dlc.interval_timeseries(
                intervals[0], intervals[1], measurement="x", dt=dt, smooth=smooth, replacenan=replacenan, prefilter=prefilter
            )
        elif measurement == "eye_y":
            return self.eye_dlc.interval_timeseries(
                intervals[0], intervals[1], measurement="y", dt=dt, smooth=smooth, replacenan=replacenan, prefilter=prefilter
            )
        elif measurement == "motion_energy":
            return self.eye_dlc.interval_timeseries(
                intervals[0], intervals[1], measurement="motion_energy", dt=dt, smooth=smooth, replacenan=replacenan, prefilter=prefilter
            )
        elif measurement == "running":
            try:
                return self.treadmill.speed_interval_timeseries(intervals[0], intervals[1], dt=dt, smooth=smooth)
            except AttributeError:
                return self.ball.speed_interval_timeseries(intervals[0], intervals[1], dt=dt, smooth=smooth)
        elif measurement == "times":
            return np.arange(intervals[0], intervals[1], dt)

    def timeseries_at(self, times, measurement=None, smooth=0.3, replacenan=True, prefilter=False, cells=None):
        # Smooth cannot be "bin" or "meanbin"
        """Sample a requested measurement at explicit timestamps with optional smoothing.

        Args:
            times: Time points (seconds) at which values are requested.
            measurement: Measurement key selecting which signal to extract.
            smooth: Gaussian smoothing width in seconds.
            replacenan: Whether to fill NaNs during interpolation/smoothing.
            prefilter: Whether to apply outlier filtering before interpolation.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        if measurement == "dspikes":
            if smooth == "bin":
                return self.spike_intervals.binned_timeseries_at(times, measurement="spikes", cells=cells)
            else:
                return self.spike_intervals.interval_timeseries_at(times, measurement="spikes", smooth=smooth, cells=cells)
        elif measurement in ["f", "neuropil", "mean_pixels"]:
            return self.spike_intervals.interval_timeseries_at(times, measurement=measurement, smooth=smooth, cells=cells)
        elif measurement == "dff":
            raise NotImplementedError("dF/F is not yet implemented")
        elif measurement == "spikes":
            if smooth == "bin":
                return self.ephys_spikes.spikes_histogram_at(times, cells=cells)
            else:
                return self.spike_intervals.interval_timeseries_at(times, measurement="spikes", smooth=smooth, cells=cells)
        elif measurement == "lfp":
            raise NotImplementedError("lfp is not yet implemented")
        elif measurement == "pupil_size":
            return self.eye_dlc.interval_timeseries_at(times, measurement="pupil", smooth=smooth, replacenan=replacenan, prefilter=prefilter)
        elif measurement == "eye_x":
            return self.eye_dlc.interval_timeseries_at(times, measurement="x", smooth=smooth, replacenan=replacenan, prefilter=prefilter)
        elif measurement == "eye_y":
            return self.eye_dlc.interval_timeseries_at(times, measurement="y", smooth=smooth, replacenan=replacenan, prefilter=prefilter)
        elif measurement == "motion_energy":
            return self.eye_dlc.interval_timeseries_at(times, measurement="motion_energy", smooth=smooth, replacenan=replacenan, prefilter=prefilter)
        elif measurement == "running":
            try:
                return self.treadmill.speed_interval_timeseries_at(times, smooth=smooth)
            except AttributeError:
                return self.ball.speed_interval_timeseries_at(times, smooth=smooth)

    def interval_mean(self, intervals, measurement=None, cells=None):
        """Compute interval means for a requested measurement, optionally per cell.

        Args:
            intervals: List/array of time intervals.
            measurement: Measurement key selecting which signal to extract.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        if hasattr(intervals[0], "__iter__"):
            return np.asarray([self.interval_mean(intvl, measurement=measurement, cells=cells) for intvl in intervals])
        # TODO df/f
        if measurement == "dspikes":
            return self.spike_intervals.interval_mean(intervals[0], intervals[1], measurement="spikes", cells=cells)
        elif measurement in ["f", "neuropil", "mean_pixels"]:
            return self.spike_intervals.interval_mean(intervals[0], intervals[1], measurement=measurement, cells=cells)
        elif measurement == "dff":
            raise NotImplementedError("dF/F is not yet implemented")
        elif measurement == "spikes":
            return [len(x) / (intervals[1] - intervals[0]) for x in self.ephys_spikes.spikes_interval(intervals, cells=cells)]
        elif measurement == "lfp":
            raise NotImplementedError("lfp is not yet implemented")
        elif measurement == "pupil_size":
            return self.eye_dlc.interval_mean(intervals[0], intervals[1], measurement="pupil")
        elif measurement == "eye_x":
            return self.eye_dlc.interval_mean(intervals[0], intervals[1], measurement="x")
        elif measurement == "eye_y":
            return self.eye_dlc.interval_mean(intervals[0], intervals[1], measurement="y")
        elif measurement == "motion_energy":
            return self.eye_dlc.interval_mean(intervals[0], intervals[1], measurement="motion_energy")
        elif measurement == "running":
            try:
                return self.treadmill.speed_interval_mean(intervals[0], intervals[1])
            except AttributeError:
                return self.ball.speed_interval_mean(intervals[0], intervals[1])
        raise ValueError(f"Invalid measurement {measurement}")

    def start_end_time(self):
        """Estimate the common valid time window across available modalities."""
        start_times = []
        end_times = []
        if hasattr(self, "neural_timings"):
            start_times.append(self.neural_timings.neuralframes_times[0])
            end_times.append(self.neural_timings.neuralframes_times[-1])
        if hasattr(self, "mpep_events"):
            start_times.append(self.mpep_events.experiment_interval()[0])
            end_times.append(self.mpep_events.experiment_interval()[1])
        if hasattr(self, "eye_dlc"):
            start_times.append(self.eye_dlc.eye_times[0])
            end_times.append(self.eye_dlc.eye_times[-1])
        assert len(start_times) > 0 and len(end_times) > 0, "Could not determine experiment starting and ending time"
        return (float(np.max(start_times)), float(np.min(end_times)))


class Registration(Mixin):
    """Mixin for cross-experiment alignment and cell matching within a session."""

    MODE = "2p"
    NAME = "registration"
    REQUIRED_MODULES = [GeneralInfo, LossyImages, Pixel2Micron, CellInfo, Explog]

    def align_exp_to_stack(self, stack):
        """Find the position of a fov within a z stack.

        This function does not perform rotation, only translation.

        For each plane in the present experiment, find the z, y, and x shift
        necessary to align the images.  A list (of length equal to the number
        of planes in the present experiment) of tuples is returned.  The first
        element in the tuple is the plane index of the z stack which best
        matches, and the second and third elements are the y and x shift,
        respectively, of the best match with that plane.

        Args:
            stack: Structural z-stack experiment used as the registration target.
        """
        if self.general_info.mouse != stack.general_info.mouse or self.general_info.date != stack.general_info.date:
            raise ValueError("These experiments were on different days and cannot be aligned")
        if self.general_info.zoom != stack.general_info.zoom:
            raise ValueError("Experiments must have the same zoom")
        if hasattr(self.general_info, "position"):
            expected_offset = np.asarray(self.general_info.position) - stack.general_info.position
            if np.any(np.abs(expected_offset) > stack.pixel2micron.fov_in_microns):
                raise ValueError("These fields of view do not intersect")
        exp_img = self.lossy_images().astype(float)
        stack_img = stack.lossy_stack_image_gcamp().astype(float)
        filtered_exp_img = np.asarray([img - scipy.ndimage.gaussian_filter(img, 30 / self.pixel2micron.vert_scale) for img in exp_img])[:, 3:-3, 3:-3]
        filtered_stack_img = np.asarray([img - scipy.ndimage.gaussian_filter(img, 30 / stack.pixel2micron.vert_scale) for img in stack_img])[
            :, 3:-3, 3:-3
        ]
        ress = []
        for img in filtered_exp_img:
            res = [(i,) + skimage.registration.phase_cross_correlation(img, simg) for i, simg in enumerate(filtered_stack_img)]
            ress.append(min(res, key=lambda x: x[2]))
        # z pos in stack, x shift, y shift
        return np.asarray([(res[0], res[1][0], res[1][1]) for res in ress], dtype="int")

    def match_cells_to_experiment(self, exp, cells=None, cells_target=None):
        """Match two experiments to each other when recorded in the same session with the same fov

        This function aligns each plane in the present experiment to the
        corresponding plane in `exp`.

        Returns two equally sized lists, one of cell indices from the present
        experiments, and the second of cell indices from `exp`.  If `cells`
        (for the present experiment) or `cells_target` (for exp) is passed, the
        indices will refer to the cells after subsetting by these cells.

        This function is tested and working, but may be no longer necessary
        since sessions can be sorted together in suite2p.
        """
        vol = self.cellinfo.volume_renumbered(cells)
        vol_other = exp.cellinfo.volume_renumbered(cells_target)
        # First, find the shift for each plane
        shifts = []
        for l in range(0, vol.shape[0]):
            thresh_base = vol[l] > 0
            thresh_other = vol_other[l] > 0
            shift, error, _ = skimage.registration.phase_cross_correlation(thresh_base, thresh_other)
            shifts.append(shift)
        # Shift the other image by the median of the plane shifts
        shift = np.median(shifts, axis=0)
        print("Shift amount: ", shift)
        shifted = scipy.ndimage.shift(vol_other, np.append(0, shift))
        return self._match_cells_in_volumes(vol, shifted)

    @staticmethod
    def _match_cells_in_volumes(base, other, pixel_overlap=10):
        """If two volumes/planes are already aligned to each other, find cells which overlap"""
        # For each image, find cells in the other image which overlap the most.
        # Make sure the overlap is at least ten pixels.
        labels_base = np.sort(np.unique(base))
        labels_other = np.sort(np.unique(other))
        modefunc = lambda x: scipy.stats.mode(x[x != 0])[0][0] if len(x[x != 0]) > 0 and scipy.stats.mode(x[x != 0])[1][0] > pixel_overlap else -1
        a = scipy.ndimage.labeled_comprehension(base, other, index=labels_other, out_dtype=int, default=-1, func=modefunc)
        b = scipy.ndimage.labeled_comprehension(other, base, index=labels_base, out_dtype=int, default=-1, func=modefunc)
        # Only take the bijective pairs, i.e. where the two were mapped to each other
        pairs_a = list(zip(a[a != -1] - 1, labels_other[a != -1] - 1))
        pairs_b = list(zip(labels_base[b != -1] - 1, b[b != -1] - 1))
        pairs = set(pairs_a).intersection(set(pairs_b))
        return tuple(map(list, zip(*sorted(pairs))))

    def match_red_cells(self, exp_red, min_overlap=0.5, cells=None):
        """Match a structural red cell scan to an experiment

        `exp_red` should be a structural image, in the same FOV, as the present
        experiment.  This can be limited to the cells `cells`, as in other
        functions.  `min_overlap` is the fraction by which a red cell must
        overlap a green cell to declare a match.

        Note that when you limit the cells, you may get different matches than
        when you don't.  This is because each red cell is only allowed to be
        associated with a single green cell, and if two cells overlap, then the
        one with the most overlap is the chosen cell.  If you limit the cells,
        there will be fewer overlaps, so you might get different matches.

        This function is tested and working.
        """
        cells = _interpret_cells_argument(cells, len(self.cellinfo.iscell))
        assert self.general_info.fov == exp_red.general_info.fov, "Experiments have different sizes"
        if hasattr(self.general_info, "position"):
            assert np.max(np.asarray(self.general_info.position) - exp_red.general_info.position) < 5, "Experiment coordinates don't match"
        else:
            for coord in ["x", "y", "z", "angle"]:
                assert self.explog.exp[coord] == exp_red.explog.exp[coord], "Recorded experiment coordinates in explog don't match"
        N_planes = self.general_info.fov[0]
        shifts = np.asarray(
            [
                skimage.registration.phase_cross_correlation(self.lossy_images()[i], exp_red.lossy_stack_image_gcamp()[i])[0]
                for i in range(0, N_planes)
            ]
        )
        red_voxel_pos = exp_red.segmented_thresh_mcherry._zstack
        redcells = list(sorted(set(red_voxel_pos[:, 3])))
        greencellid = np.repeat(np.arange(1, len(self.cellinfo._number_of_pixels) + 1), self.cellinfo._number_of_pixels)
        coordsmask = np.isin(greencellid, np.arange(1, len(cells) + 1)[cells])
        coordstuple = list(map(tuple, self.cellinfo._cell_coordinates[coordsmask]))
        green_cells_with_red = []
        for c in redcells:
            if c % 100 == 0:
                print(round(c / len(redcells) * 100), "%")
            pos = red_voxel_pos[red_voxel_pos[:, 3] == c].copy()
            assert len(set(pos[:, 0])) == 1, f"Cell spans multiple planes: {pos[:,0]}"
            plane = pos[0, 0]
            pos = pos[:, 0:3]
            pos[:, 1:3] += shifts[plane].astype(int)[None, :]
            postuple = set(map(tuple, pos[:, ::-1]))
            greencells = greencellid[coordsmask][[row in postuple for row in coordstuple]]
            N_green = len(greencells)
            greencells = greencells[greencells != 0]
            if len(greencells) == 0:
                continue
            most_common, n_most_common = collections.Counter(greencells).most_common()[0]
            if n_most_common / N_green > min_overlap:
                green_cells_with_red.append(most_common)
        # if collections.Counter(green_cells_with_red).most_common()[0][1] > 1:
        #    print("Warning: green cell is best for two red cells.  Consider increasing min_overlap.")
        green_and_red_cells = np.asarray(list(sorted(set(green_cells_with_red)))) - 1
        all_cells = np.zeros(len(self.cellinfo.iscell)).astype(bool)
        all_cells[green_and_red_cells] = True
        return all_cells[cells]

    def match_cells_multiple(self, experiments, cells, cells_experiments):
        """For multiple experiments in the same fov, find cells present in all of them."""
        matches = []
        for exp, other_cells in zip(experiments, cells_experiments):
            matches.append(np.asarray(self.match_cells_to_experiment(exp, cells, other_cells)))
        # Index using the cell from self
        valid_cells = list(sorted(set.intersection(*[set(match[0]) for match in matches])))
        valid = [m[:, np.isin(m[0], valid_cells)].tolist() for m in matches]
        assert all(np.all(valid[0][0] == v[0]) for v in valid)
        return [valid[0][0]] + [v[1] for v in valid]

    def match_to_redcell_via_green_zstack_old(self, exp_greenstack, exp_redstack, cells=None):
        """Legacy red-cell matching workflow via an intermediate green structural stack alignment.

        Args:
            exp_greenstack: Green structural stack used as an alignment bridge.
            exp_redstack: Red structural stack used as a red-cell label source.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        red_cells_in_green_stack = exp_greenstack.registration_stack.mcherry_labeled_cells(exp_redstack)
        planes = self.registration.align_exp_to_stack(exp_greenstack)
        vol = self.cellinfo.volume_renumbered(cells)
        matches_exp = []
        matches_stack = []
        for i in range(0, planes.shape[0]):
            p = planes[i]
            stack_match = scipy.ndimage.shift(exp_greenstack.segmented_gcamp.zstack[p[0]], [p[1], p[2]]).round().astype(int)
            match = exp.registration._match_cells_in_volumes(vol[i], stack_match)
            matches_exp.extend(match[0])
            matches_stack.extend(match[1])
        matches_exp = np.asarray(matches_exp)  # TODO finish
        matches_stack = np.asarray(matches_stack)
        isred = np.isin(matches_stack, np.intersect1d(red_cells_in_green_stack, matches_stack))
        red_matches_exp = matches_exp[isred]
        nonred_matches_exp = matches_exp[~isred]
        allcells = np.sort(np.unique(vol))
        return np.isin(allcells, red_matches_exp), np.isin(allcells, nonred_matches_exp), ~np.isin(allcells, matches_exp)

    def match_to_redcell_via_green_zstack(self, exp_greenstack, exp_redstack, cells=None):
        """Experimental red-cell matching workflow that composes stack and channel alignment transforms.

        Args:
            exp_greenstack: Green structural stack used as an alignment bridge.
            exp_redstack: Red structural stack used as a red-cell label source.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        stack_shift = self.align_exp_to_stack(exp_greenstack)
        red_shift = exp_greenstack.registration_stack.mcherry_stack_shift(exp_redstack)
        total_shift = (stack_shift + red_shift).astype(int)
        print(total_shift)
        vol = self.cellinfo.volume_renumbered(cells)
        # TODO
        cellsets = []
        for i in range(0, total_shift.shape[0]):
            shifted = scipy.ndimage.shift(exp_redstack.segmented_thresh_mcherry.zstack[total_shift[i, 0]], total_shift[i][1:3]).round().astype(int)
            cellset = exp_greenstack.registration_stack._mcherry_overlap_volumes(vol[i], shifted)
            cellsets.append(cellset)
            print(cellset)
        return cellsets


class RegistrationStack(Mixin):
    """Mixin for aligning and merging structural z-stacks across recordings."""

    MODE = "2p"
    NAME = "registration_stack"
    REQUIRED_MODULES = [GeneralInfo, Pixel2Micron]
    OPTIONAL_MODULES = [LossyStackImageGCamp, SegmentedGCamp, SegmentedMCherry, LossyStackImageMCherry, Explog]

    @staticmethod
    def merge_stacks(exps, channel="green", remove_n_flyback=3, filter=True, align_channel=None):
        """Given several z stacks at approximately the same x-y position, merge them into a single 3D volume.

        This is tested and working.  Do not remove.
        """
        if align_channel is None:
            align_channel = channel
        # Assumes the exps are ordered from superficial to deep
        shifts = []
        for i in range(0, len(exps) - 1):
            shifts.append(exps[i].registration_stack.align_stack_to_stack(exps[i + 1], channel=align_channel))
        cumshifts = np.cumsum(shifts, axis=0)
        fov = exps[-1].general_info.fov
        stack_height = cumshifts[-1][0] + fov[0]
        outimg = np.zeros([stack_height, *fov[1:3]])
        # Shift relative to the first one, which is assumed to be the top.
        ims_by_kind = lambda exp: exp.lossy_stack_image_gcamp.images_by_kind if channel == "green" else exp.lossy_stack_image_mcherry.images_by_kind
        kind = "max" if "max" in ims_by_kind(exps[0]) else "mean"
        for i, s in enumerate(reversed(cumshifts)):
            _blit(ims_by_kind(exps[-1 - i])[kind], outimg, s)
            # outimg[s[0]:(s[0]+fov[0])] = im(exps[-1-i])
        outimg[0 : fov[0]] = ims_by_kind(exps[0])[kind]
        outimg /= np.mean(outimg, axis=(1, 2), keepdims=True)
        outimg = outimg[remove_n_flyback:]  # Chop off flyback planes
        return outimg

    def bounding_box(self):
        # Seems like this finds the bounding box in microns?  Not sure why this is useful.  Delete?
        """Compute FOV corner positions in micron space across depth to estimate stack bounds."""
        bb = []
        for y in [0, self.general_info.fov[1] - 1]:
            for x in [0, self.general_info.fov[2] - 1]:
                all_zs = [self.pixel2micron.get_position(np.asarray([[z, y, x]]))[0][0] for z in range(0, self.general_info.fov[0] - 1)]
                minz = np.argmin(all_zs)
                maxz = np.argmax(all_zs)
                bb.append(self.pixel2micron.get_position(np.asarray([[maxz, y, x]])))
                bb.append(self.pixel2micron.get_position(np.asarray([[minz, y, x]])))
        return np.concatenate(bb)

    # TODO this works for vertical shifts but not horizontal shifts
    def align_stack_to_stack(self, exp, expected_offset=None, channel="green"):
        """Aligns the present stack to another stack.  Important function."""
        # if self.general_info.mouse != exp.general_info.mouse or self.general_info.date != exp.general_info.date:
        #    raise ValueError("These experiments were on different days and cannot be aligned")
        if self.general_info.zoom != exp.general_info.zoom:
            raise ValueError("Experiments must have the same zoom")
        if abs(self.pixel2micron.median_z_step - exp.pixel2micron.median_z_step) > 0.1:
            raise ValueError("Experiments must have the same z step")
        if expected_offset is None:
            expected_offset = np.asarray(self.general_info.position) - exp.general_info.position
            if np.all(expected_offset == [0, 0, 0]):
                try:
                    expected_offset = [float(self.explog.exp[i]) - float(exp.explog.exp[i]) for i in ["z", "y", "x"]]
                except:
                    expected_offset = [0, 0, 0]
                expected_offset = np.asarray(expected_offset)
        if np.any(np.abs(expected_offset) > exp.pixel2micron.fov_in_microns):
            raise ValueError("These fields of view do not intersect")
        if channel == "green":
            img1 = self.lossy_stack_image_gcamp().astype(float)
            img2 = exp.lossy_stack_image_gcamp().astype(float)
        elif channel == "red":
            img1 = self.lossy_stack_image_mcherry().astype(float)
            img2 = exp.lossy_stack_image_mcherry().astype(float)
        filtered_img1 = np.asarray([img - scipy.ndimage.gaussian_filter(img, 20) for img in img1])
        filtered_img2 = np.asarray([img - scipy.ndimage.gaussian_filter(img, 20) for img in img2])
        expected_offset_ind = int(expected_offset[0] / exp.pixel2micron.median_z_step)
        print("expected offset", expected_offset, expected_offset_ind)

        def patched_phase_cross_correlation(img1, img2):
            """Works around a bug in skimage phase_cross_correlation.

            Old versions didn't accept the "normalization" keyword.  New
            versions give incorrect results if you exclude it.
            """
            try:
                val = skimage.registration.phase_cross_correlation(img1, img2, normalization=None)
            except TypeError:
                val = skimage.registration.phase_cross_correlation(img1, img2)
            return val

        def median_cc(img1, img2, offset):
            """Summarize per-plane phase-correlation fit at a candidate z-offset using median score and shift."""
            vals = []
            N = img1.shape[0]
            if offset > 0:
                for i in range(0, N - offset):
                    vals.append(patched_phase_cross_correlation(img1[offset + i], img2[i]))
            else:
                for i in range(0, N + offset):
                    vals.append(patched_phase_cross_correlation(img1[i], img2[-offset + i]))
            fit = np.median([v[1] for v in vals])
            pos = np.median([v[0] for v in vals], axis=0)
            return fit, pos

        # Iterate through possible z-shifts using an intelligent algorithm,
        # because this is really slow if you use brute force.  First, estimate
        # the z offset using the reading from the manipulator on the
        # microscope.  (FYI, this is only valid for B2.)  Add this value and
        # its neighbours (3 on each side) to a list of z offsets to check.
        # Then, for each one, find the optimal shift of the two images using
        # the phase correlation, and save the median loss value.  If the loss
        # value is lower, then save that and add its three neighbours to the
        # list of offsets to check.  Once we have tried all offsets in the
        # queue, the procedure terminates.
        offsets_tried = set()
        add_offsets = lambda i: set([i + j for j in range(-3, 4)]) - offsets_tried
        offsets_to_try = add_offsets(expected_offset_ind)
        best_score = np.inf
        best_offset = np.inf
        best_score_shift = None
        while len(offsets_to_try):
            offset = offsets_to_try.pop()
            offsets_tried.add(offset)
            score, shift = median_cc(filtered_img1, filtered_img2, offset)
            print(offsets_to_try, offset, score, shift)
            if score < best_score:
                offsets_to_try.update(add_offsets(offset))
                best_score = score
                best_offset = offset
                best_score_shift = shift
        return [best_offset, int(best_score_shift[0]), int(best_score_shift[1])]

    # Finding mcherry labelled cells in a volume.
    def mcherry_stack_shift(self, exp_mcherry):
        """Estimate 3D translation aligning GCaMP and mCherry segmented stack volumes.

        Part of the suite to label mcherry cells in a volume.

        Args:
            exp_mcherry: Structural stack experiment containing mCherry segmentation.
        """
        if self.general_info.mouse != exp_mcherry.general_info.mouse or self.general_info.date != exp_mcherry.general_info.date:
            raise ValueError("These experiments were on different days and cannot be aligned")
        if self.general_info.zoom != exp_mcherry.general_info.zoom:
            raise ValueError("Experiments must have the same zoom")
        vol = self.segmented_gcamp.zstack
        shift, _, _ = skimage.registration.phase_cross_correlation(vol != 0, exp_mcherry.segmented_mcherry.zstack != 0)
        return shift

    @staticmethod
    def _mcherry_overlap_volumes(green, red):
        """Find green labels overlapped by aligned red labels in segmented volumes/planes.

        Part of the suite to label mcherry cells in a volume.

        Args:
            green: Labeled green-channel segmentation volume or plane.
            red: Labeled red-channel segmentation volume or plane, already aligned to `green`.
        """
        vol = green
        shifted = red
        labels_base = np.sort(np.unique(vol))
        labels_other = np.sort(np.unique(shifted))

        modefunc = lambda x: scipy.stats.mode(x[x != 0])[0][0] if len(x[x != 0]) > 0 and scipy.stats.mode(x[x != 0])[1][0] >= 0.9 * len(x) else -1
        a = scipy.ndimage.labeled_comprehension(vol, shifted, index=labels_other, out_dtype=int, default=-1, func=modefunc)
        # Sometimes a cell will have more than one mcherry label is the mcherry
        # planes from a single cell weren't segmented together.  So don't
        # bother looking at counts.
        ret = np.unique(a)
        ret.sort()
        return ret[1:]

    # TODO not sure if this function works
    def mcherry_labeled_cells(self, exp_mcherry):
        """Return green-cell labels with mCherry overlap after cross-stack alignment.

        Part of the suite to label mcherry cells in a volume.

        Args:
            exp_mcherry: Structural stack experiment containing mCherry segmentation.
        """
        shift = self.mcherry_stack_shift(exp_mcherry)
        shifted = np.round(scipy.ndimage.shift(exp_mcherry.segmented_mcherry.zstack, shift)).astype(int)
        return _mcherry_overlap_volumes(self.segmented_gcamp.zstack, shifted)


MIXIN_TYPES = _find_subclasses(Mixin)

#################### SECTION: Experiment types ####################


class CheckerboardCSD(Experiment):
    """Experiment type for checkerboard-evoked LFP/CSD analysis."""

    NAME = "checkerboard_csd"
    ALTERNATIVE_NAMES = ["csd"]
    PIPELINES = {"all": ["flipper"]}
    MODULES = {"ephys": [LFP, ExpDef, DiodeRaw, DiodeVideo]}
    OPTIONAL_MODULES = [Explog, MouseInfo, Protocol]  # Protocol only needed if mpep
    MIXINS = [EasyTimeseries]

    def stimulus_times(self):
        """Return checkerboard inversion times inferred from photodiode timing and protocol metadata."""
        if self.general_info.system == "mpep":
            # Divided up into multiple blocks
            block_times = self.diode_video.group_frame_times(0.1)
            # Find the number of frames it takes to invert the checkerboard
            _nfr = self.protocol.param("nfr")
            assert len(set(_nfr)) == 1, "Invalid experiment, all should have the same checkerboard speed (# frames per switch)"
            nfr = _nfr[0]
            # Find the time of each of these
            times = [t for bt in block_times for t in bt[::nfr]]
            return np.asarray(times)
        else:
            flip_times = self.diode_video.group_frame_times(0.5)[0][2:-2]
            return np.asarray(flip_times)

    def lfp_aligned(self, before=0.1, after=0.3):
        """Return trial-aligned, denoised LFP averages around checkerboard inversions.

        Args:
            before: Seconds before each event used for alignment windows.
            after: Seconds after each event used for alignment windows.
        """
        if hasattr(self, "_lfpcache"):
            if (before, after) in self._lfpcache.keys():
                return self._lfpcache[(before, after)]
        else:
            self._lfpcache = {}
        sts = self.stimulus_times()
        tss = self.lfp.intervals_timeseries(np.asarray([sts - before, sts + after]).T)
        _channel_means = scipy.ndimage.gaussian_filter1d(np.mean(tss, axis=0), 1, axis=1)
        channel_means = _channel_means - np.mean(_channel_means, axis=1, keepdims=True)
        lfp = scipy.ndimage.gaussian_filter1d(scipy.signal.medfilt(channel_means, kernel_size=(3, 1)), 1, axis=0)
        self._lfpcache[(before, after)] = lfp
        return lfp

    def csd_aligned(self, before=0.1, after=0.3):
        """Compute current-source density estimates from aligned LFP traces.

        Args:
            before: Seconds before each event used for alignment windows.
            after: Seconds after each event used for alignment windows.
        """
        lfp = self.lfp_aligned(before=before, after=after)
        # Compute CSD separately for each column of contacts and then interleave them
        csds = []
        for i in range(0, 4):
            l = lfp[i::4]
            _csd = -0.23 * l[4:] - 0.08 * l[3:-1] + 0.62 * l[2:-2] - 0.08 * l[1:-3] - 0.23 * l[:-4]
            csd = np.concatenate([[_csd[0]] * 2, _csd, [_csd[-1]] * 2])
            csds.append(csd)
        csds_interleaved = np.reshape(np.asarray(csds).transpose(1, 0, 2), (len(csds) * csds[0].shape[0], -1))
        return csds_interleaved

    def show_lfp(self, before=0.1, after=0.3):
        """Plot aligned LFP traces as a depth-by-time image.

        Args:
            before: Seconds before each event used for alignment windows.
            after: Seconds after each event used for alignment windows.
        """
        import matplotlib.pyplot as plt

        plt.imshow(
            self.lfp_aligned(before=before, after=after), aspect="auto", interpolation="none", origin="lower", extent=[-before, after, 3.84, 0]
        )
        plt.axvline(0, c="k")
        plt.xlabel("Time (ms)")
        plt.ylabel("Depth (um)")
        plt.show()

    def show_csd(self, before=0.1, after=0.3):
        """Plot aligned CSD traces as a depth-by-time image.

        Args:
            before: Seconds before each event used for alignment windows.
            after: Seconds after each event used for alignment windows.
        """
        import matplotlib.pyplot as plt

        plt.imshow(
            self.csd_aligned(before=before, after=after), aspect="auto", interpolation="none", origin="lower", extent=[-before, after, 3.84, 0]
        )
        plt.axvline(0, c="k")
        plt.xlabel("Time (ms)")
        plt.ylabel("Depth (um)")
        plt.show()

    def detect_surface_channel(self):
        """Estimate cortical surface channel from depth profile of LFP variability."""
        lfp = self.lfp_aligned()
        csdvar = np.var(self.lfp_aligned(), axis=1)
        low_snr = csdvar / np.max(csdvar) > 0.1
        np.max(low_snr)
        return np.where(low_snr)[0][-1]

    def detect_probe_depth(self):
        # Stats for npix phase3b
        """Estimate probe insertion depth from the detected surface channel."""
        tip_length = 175  # um
        site_spacing = 10  # um, actually 20 but there are two sites per row
        return tip_length + self.detect_surface_channel() * site_spacing


class Video(Experiment):
    """A video stimulus where the photodiode oscillates between 0 and 1 on subsequent frames."""

    NAME = "video"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {"all": [DiodeVideo, ExpDef], "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages], "ephys": [EphysSpikes]}
    OPTIONAL_MODULES = [EyeTrackingDLC, EyeCameraExampleImage, Explog, MouseInfo, Ball, Treadmill, MeanPixels, SpikeSortingInfo]
    AVAILABLE_MODULES = [Audio, AudioHQ, FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}
    OPTIONAL_MIXINS = [FIntervals, NeuropilIntervals]

    def frame_times(self):
        """Return flattened frame timestamps for all detected video stimulus groups."""
        return np.concatenate(self.diode_video.group_frame_times())

    def frame_times_by_group(self):
        """Return frame timestamps grouped by contiguous stimulus presentation blocks."""
        return self.diode_video.group_frame_times()


# TODO This doesn't work anymore since I switched it away from DiodeTriangles
class Triangles(Experiment):
    """Stimulus where the entire display is a single shade of grey, switching abruptly throughout the trial to discrete brightness levels."""

    NAME = "triangles"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {"all": [DiodeRaw, ExpDef], "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages], "ephys": [EphysSpikes]}
    OPTIONAL_MODULES = [EyeTrackingDLC, EyeCameraExampleImage, Explog, MouseInfo, Ball, Treadmill, MeanPixels]
    AVAILABLE_MODULES = [Audio, AudioHQ, FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}
    OPTIONAL_MIXINS = [FIntervals, NeuropilIntervals]

    @staticmethod
    def _make_stimulus(dur, pre, post, dt):
        """Generate an idealized triangular stimulus waveform segment with configurable pre/post context.

        Args:
            dur: Stimulus duration/sign parameter used for synthetic waveform generation.
            pre: Seconds before event used in synthetic waveform generation.
            post: Seconds after event used in synthetic waveform generation.
            dt: Sampling/bin width in seconds.
        """
        pre_steps = round(pre / dt)
        post_steps = round(post / dt)
        stim = np.zeros(pre_steps + post_steps)
        ramp_steps = int(abs(dur) / 2 / dt)
        ramp = np.linspace(0, 1, ramp_steps)
        if ramp_steps > pre_steps:
            stim[0 : pre_steps - 1] = ramp[-pre_steps:-1]
        else:
            stim[(pre_steps - ramp_steps + 1) : pre_steps] = ramp[:-1]
        if ramp_steps > post_steps:
            stim[pre_steps - 1 :] = ramp[-post_steps - 1 :][::-1]
        else:
            stim[pre_steps : (pre_steps + ramp_steps)] = ramp[::-1]
        return stim * np.sign(dur)

    def session_stimulus(self, start, stop, dt=0.1):
        """Render the reconstructed triangular stimulus over a session time grid.

        Args:
            start: Interval start time in timeline seconds.
            stop: Interval stop time in timeline seconds.
            dt: Sampling/bin width in seconds.
        """
        ident = self.diode_triangles.stimulus()
        starts = self.diode_triangles.starts
        signs = self.diode_triangles.sign
        assert len(ident) == len(starts) == len(signs)
        base_times = np.arange(start, stop, dt)
        base = np.zeros(len(base_times))
        nearest = lambda x: np.argmin(np.abs(base_times - x))
        for i in range(0, len(ident)):
            i_near = nearest(starts[i])
            stim = self._make_stimulus(ident[i], np.abs(ident[i]) / 2, np.abs(ident[i]) / 2, dt)
            if len(base) > i_near + len(stim):
                base[i_near : (i_near + len(stim))] = stim
        return base

    def stimulus_timeseries(self, padding=4, cells=None, dt=0.1, smooth=0.3):
        """Return stimulus-locked neural time series grouped by stimulus identity.

        Args:
            padding: Time padding (seconds) around an alignment event.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
            dt: Sampling/bin width in seconds.
            smooth: Gaussian smoothing width in seconds.
        """
        ident = self.diode_triangles.stimulus()
        intervals = self.diode_triangles.intervals_from_peak(padding + 0.001)
        spikes = np.asarray(
            [
                self.spike_intervals.interval_timeseries(intervals[i][0], intervals[i][1], dt=dt, cells=cells, smooth=smooth)
                for i in range(0, len(intervals))
            ]
        )
        idents = list(sorted(set(ident)))
        matrix = [spikes[ident == i] for i in idents]
        return idents, [self._make_stimulus(i, padding + 0.001, padding + 0.001, dt) for i in idents], matrix


class Flashes(Experiment):
    """Stimulus where the entire display is a single shade of grey, switching abruptly throughout the trial to discrete brightness levels."""

    NAME = "flashes"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {
        "all": [DiodeLevels, ExpDef],
        "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages],
        "ephys": [EphysSpikes],
    }
    OPTIONAL_MODULES = [EyeTrackingDLC, EyeCameraExampleImage, Explog, MouseInfo, Ball, Treadmill, MeanPixels, DiodeRaw]
    AVAILABLE_MODULES = [Audio, AudioHQ, FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}
    OPTIONAL_MIXINS = [FIntervals, NeuropilIntervals]

    def frame_sequence(self):
        """Recover per-frame flash identity with onset/offset times from diode and metadata signals."""
        if hasattr(self, "diode_raw"):  # Use the most precise way of getting the stimulus identity
            d = scipy.signal.medfilt(self.diode_raw.value, 101)
            times = self.diode_raw.times
            dfilt = scipy.ndimage.gaussian_filter1d(d, 30)
            transition_signal = np.abs(np.diff(dfilt))
            change_times = times[scipy.signal.find_peaks(transition_signal, height=0.0005, distance=300)[0]]
            # Find the start and end of the stimuli
            long_times = np.where(np.diff(change_times) > 5)[0]
            i_start = max(long_times[long_times < 7]) + 1
            i_stop = min(long_times[long_times > 7])
            change_times_trimmed = change_times[i_start : (i_stop + 1)]
            # Get the sequence from metadata
            key = next(k for k in self.expdef.video_meta.keys() if "flashes" in k)
            key2 = next(k for k in self.expdef.video_meta[key][1].keys() if "sequence" in k)
            sequence = self.expdef.video_meta[key][1][key2]
            # Hande the cases of repeated luminance levels
            diffs = np.diff(change_times_trimmed)
            durations = np.round(diffs / np.median(diffs)).astype(int)
            n_frames = durations // round(np.median(durations))
            _start_times = np.concatenate(
                [
                    [change_times_trimmed[i] + j * (change_times_trimmed[i + 1] - change_times_trimmed[i]) / l for j in range(0, l)]
                    for i, l in enumerate(durations)
                ]
                + [[change_times_trimmed[len(durations)]]]
            )
            start_times = _start_times[:-1]
            end_times = _start_times[1:]
            return (sequence, start_times, end_times)
        else:  # Use a less precise method
            print("Running flashes in compatibility mode, regenerate cache file if there are problems")
            # Filter diode levels
            dlv = self.diode_levels.value
            # for i in range(1, len(dlv)-1):
            #     if dlv[i] not in [dlv[i-1], dlv[i+1]]:
            #         dlv[i] = dlv[i-1]
            # Start at 2 to get rid of initial grey screen
            long_times = np.where(np.diff(self.diode_levels.time) > 5)[0]
            i_start = max(long_times[long_times < 7]) + 1
            i_stop = min(long_times[long_times > 7])
            diffs = np.diff(self.diode_levels.time[i_start : (i_stop + 1)])
            durations = np.round(diffs * 2 / np.median(diffs)).astype(int)
            framelen = round(np.median(durations))
            n_frames = durations // framelen
            frame_identity = np.repeat(dlv[i_start:i_stop], n_frames)
            times_on = []
            times_off = []
            for i in range(0, len(durations)):
                total_duration = np.diff(self.diode_levels.time[i_start:])[i]
                each_frame_duration = total_duration / n_frames[i]
                if i == len(durations) - 1 and n_frames[i] > 2:  # Sometimes there is a blip at the end
                    frame_identity = frame_identity[: -n_frames[i]]
                    continue
                times_on.extend([self.diode_levels.time[i_start:][i] + total_duration * j / n_frames[i] for j in range(0, n_frames[i])])
                times_off.extend([self.diode_levels.time[i_start:][i] + total_duration * (j + 1) / n_frames[i] for j in range(0, n_frames[i])])
            if min(frame_identity) > 0:
                frame_identity = frame_identity - min(frame_identity)
            return frame_identity, np.asarray(times_on), np.asarray(times_off)

    def session_stimulus(self, start, stop, dt=0.1):
        """Render reconstructed flash luminance over a session time grid.

        Args:
            start: Interval start time in timeline seconds.
            stop: Interval stop time in timeline seconds.
            dt: Sampling/bin width in seconds.
        """
        ident, on, off = self.frame_sequence()
        ident = ident / np.max(ident)
        base_times = np.arange(start, stop, dt)
        base = np.zeros(len(base_times)) + 0.5
        nearest = lambda x: np.argmin(np.abs(base_times - x))
        for i in range(0, len(ident)):
            i_near_on = nearest(on[i])
            i_near_off = nearest(off[i])
            base[i_near_on:i_near_off] = ident[i]
        return base

    def matrix_timeseries(self, pre=1, post=1, cells=None):
        """Build transition-conditioned response matrices for flash-level changes.

        Args:
            pre: Seconds before event used in synthetic waveform generation.
            post: Seconds after event used in synthetic waveform generation.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        ident, on, off = self.frame_sequence()
        spikes = np.asarray([self.spike_intervals.interval_timeseries(on[i] - pre, on[i] + post + 1e-4, cells=cells) for i in range(0, len(on))])
        n_levels = max(ident) + 1
        levs = {
            i: {j: [k for k in range(1, len(ident)) if ident[k] == j and ident[k - 1] == i] for j in range(0, n_levels)} for i in range(0, n_levels)
        }
        matrix = [[spikes[levs[i][j]] for i in range(0, n_levels)] for j in range(0, n_levels)]
        return matrix


class Chirps(Experiment):
    """Experiment type for chirp visual stimuli with template-based trial alignment."""

    NAME = "chirps"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {"all": [DiodeRaw, ExpDef], "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages], "ephys": [EphysSpikes]}
    OPTIONAL_MODULES = [EyeTrackingDLC, EyeCameraExampleImage, Treadmill, Ball, Explog, MouseInfo, MeanPixels]
    AVAILABLE_MODULES = [Audio, AudioHQ, FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}
    OPTIONAL_MIXINS = [FIntervals, NeuropilIntervals]

    @staticmethod
    def _chirp_template(fps=1000):
        """Generate the canonical chirp luminance waveform template and key event indices.

        Args:
            fps: Template sampling rate in frames/second.
        """
        sequence = [
            ("grey", 2),
            ("black", 2),
            ("white", 2),
            ("black", 2),
            ("grey", 2),
            ("sin_squared", 8, 0, 1, 4, 1),  # Not sure why this is squared, but it's in their code
            ("grey", 2),
            ("sin", 8, 2, 0, 2, 1),
            ("grey", 2),
        ]
        _colours = {"white": 1, "grey": 0.5, "black": 0}
        frames = []
        events = []
        for s in sequence:
            events.append(len(frames))
            if s[0] in _colours.keys():
                frames.extend([_colours[s[0]]] * int(s[1] * fps))
            if s[0] == "sin":
                t = np.linspace(0, s[1], s[1] * fps)
                wave = np.sin(t * 2 * 3.141592 * np.linspace(s[2], s[4], len(t)))
                scaled_wave = wave / 2 * np.linspace(s[3], s[5], len(t)) + 0.5
                frames.extend(scaled_wave)
            if s[0] == "sin_squared":
                t = np.linspace(0, s[1], s[1] * fps) ** 2 / s[1]
                wave = np.sin(t * 2 * 3.141592 * np.linspace(s[2], s[4], len(t)))
                scaled_wave = wave / 2 * np.linspace(s[3], s[5], len(t)) + 0.5
                frames.extend(scaled_wave)
        return (np.asarray(frames), events)

    def session_stimulus(self, start, stop, dt=0.1):
        """Render repeated chirp templates over a session time grid.

        Args:
            start: Interval start time in timeline seconds.
            stop: Interval stop time in timeline seconds.
            dt: Sampling/bin width in seconds.
        """
        tt = self.get_trial_times()
        base_times = np.arange(start, stop, dt)
        base = base_times * 0
        template = self._chirp_template(round(1 / dt))[0]
        for t in tt:
            loc = np.searchsorted(base_times, t[0])
            if len(base) >= loc + len(template):
                base[loc : (loc + len(template))] = template
            else:
                base[loc:] = template[: (base.shape[0] - loc)]
        return base

    def get_trial_times(self):
        """Detect chirp trial boundaries by correlating the diode trace with the chirp template."""
        if hasattr(self, "_trial_times"):
            return self._trial_times
        # Each row is a trial, and each column is a checkpoint in the trial
        chirp_frames, chirp_events = self._chirp_template()
        c = np.correlate(self.diode_raw.value - np.median(self.diode_raw.value), chirp_frames - np.median(chirp_frames), mode="valid")
        h = (lambda x: x.max() - x.min())((c[int(len(c) * 0.25) : int(len(c) * 0.75)]))
        peaks = scipy.signal.find_peaks(c, height=h / 2, prominence=h / 2, distance=10000)[0]
        trials = np.asarray([[self.diode_raw.times[p + ce] for ce in chirp_events] for p in peaks])
        self._trial_times = trials
        return trials


class Sines(Experiment):
    """Experiment type for sinusoidal luminance modulation stimuli."""

    NAME = "sines"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {"all": [DiodeRaw, ExpDef], "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages], "ephys": [EphysSpikes]}
    OPTIONAL_MODULES = [EyeTrackingDLC, EyeCameraExampleImage, Treadmill, Ball, Explog, MouseInfo, MeanPixels]
    AVAILABLE_MODULES = [Audio, AudioHQ, FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals], "2p": [CellPositionInMicrons, Registration]}

    def trials(self):  # TODO this doesn't work yet
        # It is hard to find the trials here because we need to get the precise
        # place where the sine wave starts.  We can't detect grey because that
        # won't give precise timings since the sine waves start on grey. We
        # can't correlate the ground truth sequence since that might not be
        # good enough alignment.
        """Infer sine stimulus trial intervals by matching photodiode transitions to known stimulus sequences."""
        d = scipy.signal.medfilt(self.diode_raw.value, 101)
        times = self.diode_raw.times
        hist = np.histogram(d, bins=np.arange(0, 5, 0.001))
        grey = hist[1][np.argmax(hist[0]) + 1]
        isgrey = np.abs(d - grey) < 0.01
        switches = np.where(np.diff(isgrey) != 0)[0]
        long_switches_start = switches[:-1][np.diff(switches) > 1000 * 0.9]
        long_switches_end = switches[1:][np.diff(switches) > 1000 * 0.9]
        long_switches_to_grey = [
            (long_switches_start[i], long_switches_end[i]) for i in range(0, len(long_switches_start)) if not isgrey[long_switches_start[i]]
        ]
        delims = [sum(ls) // 2 for ls in long_switches_to_grey]
        ground_truth = self.expdef.video_meta["sines1-b2.mp4"][1]["sines1-b2.mp4.sequence.txt"]
        # stims = self.expdef.video_meta['sines1-b2.mp4'][1]['sines1-b2.mp4.stimuli.txt']
        stims = eval(open("sines2-b2.stimuli.txt", "r").read())
        assert len(delims) == len(stims) + 1
        stim_info = []
        for i in range(0, len(delims) - 1):
            # Fit the true sine wave stimulus to each segment
            d_segment = d[delims[i] : delims[i + 1]]
            ground_truth_segment = ground_truth[stims[i][1] : stims[i][2]]
            ground_truth_peaks = scipy.signal.find_peaks(np.concatenate([ground_truth_segment, [ground_truth_segment[0]] * 10]))[0]
            ground_truth_minima = scipy.signal.find_peaks(-np.concatenate([ground_truth_segment, [ground_truth_segment[0]] * 10]))[0]
            d_peaks = scipy.signal.find_peaks(d_segment, prominence=0.01, distance=150)[0]
            d_minima = scipy.signal.find_peaks(-d_segment, prominence=0.01, distance=150)[0]
            assert len(d_peaks) == len(ground_truth_peaks), f"Peaks not the same for {i}"
            assert len(d_minima) == len(ground_truth_minima), f"Minima not the same for {i}"
            coefs = np.polyfit(np.concatenate([ground_truth_peaks, ground_truth_minima]), np.concatenate([d_peaks, d_minima]), 1)
            stim_info.append((stims[i][0], times[int(coefs[1])], times[int(coefs[1] + coefs[0] * len(ground_truth_segment))]))
        return stim_info


class TemporalContrast(Experiment):
    """Experiment type for temporal-contrast block stimuli."""

    NAME = "temporal_contrast"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {"all": [DiodeRaw, ExpDef], "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages], "ephys": [EphysSpikes]}
    OPTIONAL_MODULES = [FunctionalF, FunctionalNeuropil, EyeTrackingDLC, EyeCameraExampleImage, Treadmill, Ball, Explog, MouseInfo, MeanPixels]
    AVAILABLE_MODULES = [Audio, AudioHQ]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}

    def trials(self):
        """Segment temporal-contrast blocks from long gray inter-block intervals in photodiode data."""
        d = scipy.signal.medfilt(self.diode_raw.value, 11)
        times = self.diode_raw.times
        hist = np.histogram(d, bins=np.arange(0, 5, 0.001))
        grey = hist[1][np.argmax(hist[0]) + 1]
        d = np.concatenate([d, [10, 10, 10, grey, grey, grey]])
        isgrey = np.abs(d - grey) < 0.05
        switches = np.where(np.diff(isgrey) != 0)[0]
        long_switches_start = switches[:-1][np.diff(switches) > 1000 * 4.5]
        long_switches_end = switches[1:][np.diff(switches) > 1000 * 4.5]
        long_switches_to_grey = [
            (long_switches_start[i], long_switches_end[i]) for i in range(0, len(long_switches_start)) if not isgrey[long_switches_start[i]]
        ]
        blocks = [(times[long_switches_to_grey[i][1]], times[long_switches_to_grey[i + 1][0]]) for i in range(0, len(long_switches_to_grey) - 1)]
        return blocks


class Telegraph(Experiment):
    """Experiment type for binary telegraph-like luminance switching stimuli."""

    NAME = "telegraph"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {"all": [DiodeRaw, ExpDef], "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages], "ephys": [EphysSpikes]}
    OPTIONAL_MODULES = [FunctionalF, FunctionalNeuropil, EyeTrackingDLC, EyeCameraExampleImage, Treadmill, Ball, Explog, MouseInfo, MeanPixels]
    AVAILABLE_MODULES = [Audio, AudioHQ]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}

    def stimulus_timings(self):
        """Returns a tuple of two lists: the times when the stimulus switched from off to on, and from on to off"""
        # Mostly copied from DiodeVideo
        d = self.diode_raw.value
        dflat = np.convolve(np.diff(scipy.signal.medfilt(d, 5)), [1] * 20, mode="same")
        dflat[np.abs(dflat) < 0.3] = 0
        # To find both diode-on and diode-off events, take the absolute value.
        # The run the scipy algorithm for finding peaks.  Distance=4 corresponds
        # to 8 ms.  This was tested on b2 only.  Discard the first and last,
        # since they're just the screens turning on and off.
        dpeaks = scipy.signal.find_peaks(np.abs(dflat), width=10, prominence=0.4, distance=4)[0][1:-1]
        dpeaks_direction = dflat[dpeaks] > 0
        peaktimes = self.diode_raw.times[dpeaks]
        # Return first the times it switched to on, then the times it switchd to off
        return (peaktimes[dpeaks_direction], peaktimes[~dpeaks_direction])


class FullFieldDriftingGrating(Experiment):
    """Drifting grating which is displayed on the entire screen, from the mpep experiment"""

    NAME = "full_field_drifting_grating"
    ALTERNATIVE_NAMES = ["full_field_drifting_gratings"]
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {
        "all": [Diode, MpepEvents, Protocol],
        "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages],
        "ephys": [EphysSpikes],
    }
    OPTIONAL_MODULES = [EyeTrackingDLC, EyeCameraExampleImage, Treadmill, Ball, Explog, DiodeVideo, ExpDef]
    AVAILABLE_MODULES = [Audio, AudioHQ, FunctionalF, FunctionalNeuropil, LFP]
    MIXINS = {"2p": [CellPositionInMicrons], "all": [SpikeIntervals, EasyTimeseries]}
    OPTIONAL_MIXINS = [FIntervals, NeuropilIntervals, Registration]

    def stimulus_timings(self):
        # mpep
        """Assemble per-trial grating parameter table with aligned start/stop times."""
        trial_intervals = np.asarray([x for x in self.diode_video.find_groups(0.1) if x[1] - x[0] > 0.1])
        lim = slice(None)  # The case where the experiment terminates early, we can still use the incomplete stimuli
        if "lb" in self.protocol.param_names:  # Max's mpep script for full-field drifting gratings
            assert len(trial_intervals) == len(self.protocol.param("sf")), "Invalid photodiode signal"
            column_names = [
                "duration",
                "temporal_frequency",
                "spatial_frequency",
                "temporal_phase",
                "spatial_phase",
                "orientation",
                "contrast",
                "luminance",
            ]
            columns = ["dur", "tf", "sf", "tph", "sph", "ori", "cb", "lb"]  # mpep
        elif "lu" in self.protocol.param_names:  # Tinya's mpep script for spatially localised drifting gratings
            assert len(trial_intervals) == len(self.protocol.param("sf")), "Invalid photodiode signal"
            column_names = [
                "duration",
                "temporal_frequency",
                "spatial_frequency",
                "temporal_phase",
                "spatial_phase",
                "orientation",
                "angle",
                "x",
                "y",
                "contrast",
                "luminance",
            ]
            columns = ["dur", "tf", "sf", "tph", "sph", "ori", "angle", "xc", "yc", "co", "lu"]  # mpep
        elif "gratingTF" in self.protocol.param_names:  # mc
            if len(trial_intervals) < len(self.protocol.param("gratingOrient")):
                lim = slice(0, len(trial_intervals))
            assert len(trial_intervals) <= len(self.protocol.param("gratingOrient")), "Invalid photodiode signal"
            column_names = ["duration", "temporal_frequency", "spatial_frequency", "orientation", "contrast"]
            columns = ["stimulusDuration", "gratingTF", "gratingSF", "gratingOrient", "contrast"]  # mpep
        else:
            raise NameError("Unknown stimulus presentation system")
        df = pandas.DataFrame({cn: self.protocol.param(c)[lim] for c, cn in zip(columns, column_names)})
        df["time_start"] = trial_intervals[:, 0]
        df["time_stop"] = trial_intervals[:, 1]
        return df


class NaturalImages(Experiment):
    """Sequence of natural images using the MWS .p files in mpep, i.e., where each image is a separate trial"""

    NAME = "natural_images"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {
        "all": [Diode, MpepEvents, ExpDef],
        "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages],
        "ephys": [EphysSpikes],
    }
    OPTIONAL_MODULES = [EyeTrackingDLC, EyeCameraExampleImage, Explog, MouseInfo, Ball, Treadmill, Protocol]  # Protocol for mc natural images
    AVAILABLE_MODULES = [Audio, AudioHQ, FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}
    OPTIONAL_MIXINS = [FIntervals, NeuropilIntervals]

    def stimulus_timings(self):
        """Return natural-image onset/offset timing with image identity per presentation."""
        flips = self.diode.flip_times()
        valid_flips = np.where((np.diff(flips) > 0.1) & (np.diff(flips) < 0.4))[0]
        valid_flips = np.concatenate([valid_flips, [valid_flips[-1] + 1]])
        valid_flip_times = flips[valid_flips]
        # The last one sticks on the screen so we can exclude it
        return pandas.DataFrame({"onset_time": valid_flip_times[:-1], "offset_time": valid_flip_times[1:], "image": self.protocol.param("num")[:-1]})


# class NaturalImagesMpep(Experiment):
#     """Sequence of natural images using the MWS .p files in mpep, i.e., where each image is a separate trial"""
#     NAME = "natural_images_mpep"
#     PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
#     MODULES = {"all": [Diode, MpepEvents, ExpDef], "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages], "ephys": [EphysSpikes]}
#     OPTIONAL_MODULES = [EyeTrackingDLC, EyeCameraExampleImage, Explog, MouseInfo, Ball, Treadmill, Protocol]
#     AVAILABLE_MODULES = [Audio, AudioHQ, FunctionalF, FunctionalNeuropil]
#     MIXINS = [CellPositionInMicrons, Registration, EasyTimeseries]
#     OPTIONAL_MIXINS = [FIntervals, NeuropilIntervals]
#     def image_repeat_spikes(self, cells=None):
#         params, _spikes = self.spikes_by_param('img', cells=cells)
#         spikes = awkward.Array(_spikes)
#         mean_per_presentation = np.mean(spikes, axis=3)
#         # Remove the zero position, which is the blank
#         return np.asarray(mean_per_presentation[1:])

# # TODO this doesn't work anymore
# class NaturalImagesLegacy(Experiment):
#     """Sequence of natural images using the non-MWS .p files in mpep, i.e., each trial is from an unnamed pseudorandom sequence"""
#     NAME = "natural_images_legacy"
#     PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
#     MODULES = [Diode, MpepEvents, Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, FunctionalF, FunctionalNeuropil, Ball, Protocol, Explog, MouseInfo, LossyImages]
#     OPTIONAL_MODULES = [EyeTrackingDLC, EyeCameraExampleImage]
#     AVAILABLE_MODULES = [Audio, AudioHQ, FunctionalF, FunctionalNeuropil]
#     MIXINS = [CellPositionInMicrons, SpikeConditions, Registration]
#     OPTIONAL_MIXINS = [FIntervals, NeuropilIntervals]
#     def image_repeat_spikes(self, n_stimuli=112, cells=None):
#         n_trials = len(self.mpep_events.blocks)
#         assert n_trials % n_stimuli == 0, "Invalid n stimuli from experiment"
#         n_repeats = n_trials // n_stimuli
#         spikes_stimuli = np.zeros((n_stimuli, self.spikes.timeseries[cells].shape[0], n_repeats))
#         for b in range(0, n_stimuli):
#             blocks = (self.mpep_events.blocks - 1) % n_stimuli == b
#             spikes_stimuli[b] = np.mean(awkward.Array(self.spike_conditions.spikes_by_condition(blocks, cells=cells)), axis=2)
#         return spikes_stimuli

# class FunctionalZStack(Experiment):
#     NAME = "functional_zstack"
#     MODULES = [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalF, Explog]
#     MIXINS = [CellPositionInMicrons]
#     # TODO finish this function
#     def process_zstack(self):
#         pos = self.cell_position_in_microns()[self.cellinfo.iscell]
#         F = self.f.timeseries[self.cellinfo.iscell]
#         F = F[:,2:-2] # Cut off the first and last timepoints which sometimes have outliers
#         tree = scipy.spatial.KDTree(pos)
#         matches = []
#         for p in range(0, 1000):
#             #_,closest = tree.query(pos[p], 10)
#             closest = tree.query_ball_point(pos[p], 15)
#             for c in closest:
#                 if c == p: continue
#                 if np.corrcoef(F[c], F[p])[0,1] > .4:
#                     matches.append((c, p))


# class Volume(Experiment):
#     """Any 3D image"""
#     NAME = "volume"
#     ALTERNATIVE_NAMES = ["image"]
#     PIPELINES = {"2p": ["volume"]}
#     MODULES = {"2p": [Pixel2Micron]}
#     OPTIONAL_MODULES = [LossyStackImageMCherry, LossyStackImageGCamp, LossyStackImageDiB, LossyStackImageFarRed, Explog, MouseInfo]
#     AVAILABLE_MODULES = [StackImageMCherry, StackImageGCamp, StackImageDiB]


# TODO
class BrainExplorer(Experiment):
    """Experiment type exposing a browsable movie of a single-plane 2p field of view."""

    NAME = "brain_explorer"
    ALTERNATIVE_NAMES = ["brainview"]
    PIPELINES = {"2p": []}
    MODULES = {"2p": [Pixel2Micron, BrainViewer]}
    OPTIONAL_MODULES = [Explog, MouseInfo, EyeCameraExampleImage]


# class RedCell(Experiment):
#     """Red cell z-stack, or else just separate planes of red cells.  Optionally includes the green channel."""
#     NAME = "redcell"
#     ALTERNATIVE_NAMES = ["mcherry", "zstack_mcherry", "redcell_zstack"]
#     PIPELINES = {"2p": ["cellpose"]}
#     MODULES = {"2p": [Pixel2Micron, SegmentedMCherry, SegmentedThreshMCherry, LossyStackImageMCherry]}
#     OPTIONAL_MODULES = [SegmentedGCamp, LossyStackImageGCamp, LossyStackImageDiB, Explog, MouseInfo]
#     MIXINS = {"2p": [CellPositionInMicronsMCherry, RegistrationStack]}
#     OPTIONAL_MIXINS = [CellPositionInMicronsGCamp]

# class FunctionalZStackSegmented(Experiment):
#     """Functional z-stack with cells identified using cellpose"""
#     NAME = "functional_zstack_segmented"
#     ALTERNATIVE_NAMES = ["functional_zstack"]
#     PIPELINES = {"2p": ["cellpose"]}
#     MODULES = {"2p": [Pixel2Micron, LossyStackImageGCamp]}
#     OPTIONAL_MODULES = [Explog, MouseInfo, CellInfo, LossyStackImageMCherry, LossyStackImageDiB]
#     AVAILABLE_MODULES = [FunctionalF, SegmentedGCamp, StackImageGCamp, StackImageMCherry, StackImageDiB]
#     MIXINS = {"2p": [CellPositionInMicronsGCamp, RegistrationStack]}

# class StaticGcampZStack(FunctionalZStackSegmented):
#     """Z-stack of gcamp, identified with cellpose, but non-dynamic version for calcium-independent signal"""
#     NAME = "static_gcamp_zstack"
#     ALTERNATIVE_NAMES = ["static_gcamp", "structural_gcamp"]
#     OPTIONAL_MODULES = [Explog, MouseInfo]
#     MIXINS = {"2p": [RegistrationStack, CellPositionInMicronsGCamp]}


class StructuralZStack(Experiment):
    """All channels of a z stack but without dynamics"""

    NAME = "structural_zstack"
    ALTERNATIVE_NAMES = ["functional_zstack", "functional_zstack_segmented", "static_gcamp", "structural_gcamp", "redcell", "volume", "image"]
    PIPELINES = {"2p": ["volume"]}
    MODULES = {"2p": [Pixel2Micron]}
    OPTIONAL_MODULES = [Explog, MouseInfo, CellInfo, LossyStackImageGCamp, LossyStackImageMCherry, LossyStackImageDiB, LossyStackImageFarRed]
    AVAILABLE_MODULES = [StackImageGCamp, StackImageMCherry, StackImageDiB, StackImageFarRed, SegmentedGCamp, SegmentedMCherry]
    MIXINS = {"2p": [RegistrationStack]}
    OPTIONAL_MIXINS = [CellPositionInMicronsGCamp]


class RestingState(Experiment):
    """Experiment type for blank-screen resting-state recordings."""

    NAME = "resting_state"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {"2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages], "ephys": [EphysSpikes]}
    OPTIONAL_MODULES = [EyeTrackingDLC, Explog, MouseInfo, Treadmill, Ball, EyeCameraExampleImage, Protocol, ExpDef, MpepEvents, SpikeSortingInfo]
    AVAILABLE_MODULES = [Audio, AudioHQ, FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}
    OPTIONAL_MIXINS = [FIntervals, NeuropilIntervals]


class AuditoryStimuli(Experiment):
    """Experiment type for sound-driven protocols with synchronized neural/behavioral signals."""

    NAME = "audio"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {
        "2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages],
        "ephys": [EphysSpikes],
        "all": [Audio, DiodeRaw, Diode],
    }
    OPTIONAL_MODULES = [EyeTrackingDLC, Explog, MouseInfo, Treadmill, Ball, Protocol, ExpDef, EyeCameraExampleImage]
    AVAILABLE_MODULES = [AudioHQ, FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}


class DriftingPink(Experiment):
    """Experiment type for drifting-pink-image stimuli with trial metadata extraction."""

    NAME = "drifting_pink"
    ALTERNATIVE_NAMES = ["drifting_image"]
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {"2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages], "ephys": [EphysSpikes], "all": [DiodeVideo]}
    OPTIONAL_MODULES = [EyeTrackingDLC, Explog, MouseInfo, Treadmill, Ball, Protocol, ExpDef, EyeCameraExampleImage]
    AVAILABLE_MODULES = [AudioHQ, FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}

    def stimulus_timings(self):
        """Gives the trial type and the start and end time of the trial.

        Blank trials are those where a gray screen is shown instead of a stimulus.

        Static trials are those where the image is shown but it is not drifting.
        """
        trial_types = np.asarray(self.expdef.video_meta["drifting_pink.mp4"][1]["drifting_pink.mp4.angle_speeds.txt"])
        # times = max(exp.diode_video.group_frame_times(3), key=len)
        trials = np.asarray([(g[0], g[-1]) for g in self.diode_video.group_frame_times(0.8) if len(g) < 200])
        assert len(trials) == len(trial_types)
        blank = np.any(np.isinf(trial_types), axis=1)
        static = trial_types[:, 1] == 0
        speed = trial_types[:, 1].copy()
        angle = trial_types[:, 0].copy()
        angle[static | blank] = np.nan
        speed[blank] = np.nan
        return pandas.DataFrame(
            {"blank": blank, "static": static, "speed": speed, "angle": angle, "time_start": trials[:, 0], "time_stop": trials[:, 1]}
        )


class ShiftChirp(Experiment):
    """Like the chirps stimulus, but with horizontal shifts on a static image"""

    NAME = "shiftchirp"
    PIPELINES = {"all": ["eye"], "2p": ["s2p"], "ephys": ["kilosort", "flipper"]}
    MODULES = {"2p": [Pixel2Micron, NeuralFrameTimings, CellInfo, FunctionalSpikes, LossyImages], "ephys": [EphysSpikes], "all": [Diode]}
    OPTIONAL_MODULES = [EyeTrackingDLC, Explog, MouseInfo, Treadmill, Ball, Protocol, ExpDef, EyeCameraExampleImage]
    AVAILABLE_MODULES = [AudioHQ, FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals, EasyTimeseries], "2p": [CellPositionInMicrons, Registration]}

    def frame_times(self):
        """Align shiftchirp frame indices to timeline timestamps using Fibonacci flipper synchronization."""
        shifts = self.expdef.video_meta["shiftchirp.mp4"][1]["shiftchirp.mp4.shifts.txt"]
        frames = self.diode.fib_flipper_sync()
        # TODO why is the shifts array so much shorter than the frames array?
        frame_times = frames[0 : len(shifts)]
        return frame_times

    def frame_times_by_trial(self):
        """Split shiftchirp frame times into per-trial start/stop intervals."""
        shifts = self.expdef.video_meta["shiftchirp.mp4"][1]["shiftchirp.mp4.shifts.txt"]
        frame_times = self.frame_times()
        trial_structure = self.expdef.video_meta["shiftchirp.mp4"][1]["shiftchirp.mp4.sequence.txt"]
        fps = int(self.expdef.video_meta["shiftchirp.mp4"][0]["@r_frame_rate"].split("/")[0])
        trial_length = fps * sum([e[1] for e in trial_structure])
        n_trials = len(shifts) // trial_length
        assert len(shifts) == trial_length * n_trials, "Invalid trial structure"
        return [(frame_times[i * trial_length], frame_times[(i + 1) * trial_length - 1]) for i in range(0, n_trials)]

    def stimulus(self):
        """Return one trial of shift values from shiftchirp stimulus metadata."""
        shifts = self.expdef.video_meta["shiftchirp.mp4"][1]["shiftchirp.mp4.shifts.txt"]
        trial_structure = self.expdef.video_meta["shiftchirp.mp4"][1]["shiftchirp.mp4.sequence.txt"]
        fps = int(self.expdef.video_meta["shiftchirp.mp4"][0]["@r_frame_rate"].split("/")[0])
        trial_length = fps * sum([e[1] for e in trial_structure])
        return shifts[0:trial_length]


class Retinotopy(Experiment):
    """Retinotopic mapping without cell detection.  This requires running the matlab processing script first."""

    NAME = "retinotopy"
    MODULES = {"2p": [Pixel2Micron, MeanPixels], "ephys": [EphysSpikes]}
    OPTIONAL_MODULES = [RetinotopicMap, Explog, MouseInfo, LossyImages, LossyStackImageGCamp, CellInfo]
    PIPELINES = {"2p": ["s2p"], "ephys": ["kilosort", "flipper", "eye"]}
    MIXINS = {"ephys": [SpikeIntervals]}


class RetinotopyMultiplane(Experiment):
    """Retinotopic mapping with cell detection.  This requires running the matlab processing script first."""

    NAME = "retinotopy_multiplane"
    MODULES = {"2p": [Pixel2Micron, MeanPixels]}
    OPTIONAL_MODULES = [
        RetinotopicMap,
        Explog,
        MouseInfo,
        LossyImages,
        CellInfo,
        FunctionalSpikes,
        NeuralFrameTimings,
        EyeTrackingDLC,
        Diode,
        DiodeRaw,
        DiodeVideo,
    ]
    AVAILABLE_MODULES = [FunctionalF, FunctionalNeuropil]
    MIXINS = {"all": [SpikeIntervals], "2p": [CellPositionInMicrons]}
    OPTIONAL_MIXINS = [SpikeIntervals, FIntervals, NeuropilIntervals, Registration]
    PIPELINES = {"2p": ["s2p", "eye"]}

    def single_cell_retinotopy_simple(self, spatial_smooth_input=False, cells=None):
        """Compute a simple receptive-field estimate by weighting stimulus maps with post-stimulus activity.

        Args:
            spatial_smooth_input: Whether to spatially smooth/broaden retinotopy stimulus maps.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        if spatial_smooth_input:
            stim = scipy.ndimage.convolve(exp.retinotopic_map.stim, np.asarray([[0.5, 1, 0.5], [1, 2, 1], [0.5, 1, 0.5]])[:, :, None])
        else:
            stim = exp.retinotopic_map.stim

        stimtimes = exp.retinotopic_map.stimtimes
        means_after = np.asarray([exp.spike_intervals.interval_mean(t, t + 2, cells=cells) for t in stimtimes]).flatten()
        rfs = np.asarray(
            [
                np.mean(means_after[i][None, None, :] * np.abs(stim), axis=2)
                / np.quantile(means_after[i].flatten(), 0.95)
                / np.mean(np.abs(stim), axis=2)
                for i in range(0, means_after.shape[0])
            ]
        )
        return rfs

    def single_cell_retinotopy_by_correlation(self, spatial_smooth_input=True, resamples=50, cells=None):
        # NOTE: I haven't tested this inside pixease, only outside

        # First, smooth the input stimulus a bit
        """Estimate receptive-field reliability maps via resampled correlation with a canonical transient response profile.

        Args:
            spatial_smooth_input: Whether to spatially smooth/broaden retinotopy stimulus maps.
            resamples: Number of bootstrap/resample iterations.
            cells: Cell selection (boolean mask, ids, or `None` for all cells).
        """
        stimtimes = self.retinotopic_map.stimtimes
        if spatial_smooth_input:
            stim = scipy.ndimage.binary_dilation(self.retinotopic_map.stim != 0, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])[:, :, None])
        else:
            stim = self.retinotopic_map.stim
        L = 31  # Length of timeseries to do correlation, units of "interval between stim frames" / 2
        Lprev = 10  # Length of time before stimulus start to include in correlation, same units as L
        # Lag the timeseries by the number of points, assuming even spacing to get pre- and post-stimulus timepoints
        stimtimesplus = np.concatenate(
            [stimtimes[-(Lprev // 2 + 1) : -1] - stimtimes[-1] + stimtimes[0], stimtimes, stimtimes[1 : (L + 1)] - stimtimes[0] + stimtimes[-1]]
        )
        # Two samples per stimulus, not one
        interp = np.zeros(len(stimtimesplus) * 2 - 1)
        interp[::2] = stimtimesplus
        interp[1::2] = (stimtimesplus[:-1] + stimtimesplus[1:]) / 2
        fulltss = exp.spike_intervals.interval_timeseries_at(interp, cells=cells)
        # Get neural data from all timepoints
        itssf = np.zeros((len(stimtimes), fulltss.shape[0], L))
        for i in range(0, len(stimtimes)):
            itssf[i] = fulltss[:, (i * 2) : (i * 2 + L)]

        # Faster version of Pearson correlation
        pearson = (
            lambda x, y: (x - np.mean(x, axis=1, keepdims=True))
            @ (y - np.mean(y, axis=1, keepdims=True)).T
            / x.shape[1]
            / np.sqrt(np.var(x, axis=1)[:, None])
            / np.sqrt(np.var(y, axis=1)[None, :])
        )
        # The algorithm works as follows: resample half of the timepoints.
        # Then, for each cell, find the psth using just those subsampled
        # timepoints.  Then, find the Pearson correlation between the psth and
        # the first bump of a sine wave.  Finally, repeat this many times, and
        # take the mean divided by the standard deviation.  The justification
        # is that, when it correlates well with the first bump of a sine wave
        # on all trials, the mean correlation will be high and the std will be
        # low, so the ratio will be high.  >10 is what you can expect for
        # strong rfs.  The first bump of a sine wave is just a proxy for a
        # signal which goes up and then goes down.
        _scrambles = [np.random.RandomState(i).rand(stim.shape[2]) > 0.5 for i in range(0, resamples)]
        up_and_down = np.sin(np.linspace(0, 3.141592, itssf.shape[2]))
        # This is the slow step
        reliabilities = np.asarray(
            [
                [
                    [pearson(up_and_down[None, :], np.mean(itssf[(stim[y, x] != 0) & (_scrambles[i])], axis=0)) for i in range(0, resamples)]
                    for x in range(0, stim.shape[1])
                ]
                for y in range(0, stim.shape[0])
            ]
        )
        reliabilities = reliabilities.squeeze()
        m = np.mean(reliabilities, axis=2) / np.std(reliabilities, axis=2)
        # m = scipy.ndimage.median_filter(m, footprint=np.ones((3,3))[:,:,None]) # Median filter
        return m


EXPERIMENT_TYPES = _find_subclasses(Experiment)

#################### SECTION: Pipelines ####################


def pipeline_s2p(args):
    """Run Suite2p preprocessing for one or more experiment groups and mirror results back to server storage."""
    import suite2p

    gcamp = _auto_detect_gcamp(args) if args.gcamp == "auto" else args.gcamp
    if args.expnums == "auto":
        args.expnums = _auto_detect_expnums(args.mouse, args.date, "2p")
    tau = {"6s": 1.4, "6m": 1.0, "6f": 0.7, "8m": 0.7}[gcamp]  # From s2p docs
    for expgroup in args.expnums.split(","):
        print("Processing", args.expnums)
        data_paths = [DP.format(mouse=args.mouse, date=args.date, expnum=expnum) for DP in DATA_PATHS for expnum in expgroup.split("-")]
        if (
            any(
                Path(p.format(mouse=args.mouse, date=args.date, expnum="pixease")).joinpath(expgroup).joinpath("suite2p").exists() for p in DATA_PATHS
            )
            and not args.force
        ):
            print(f"Experiment {expgroup} already exists on the server, continuing")
            continue
        # Assume the correct place on the server to store the processed data is
        # the root directory of the first valid data path
        remote_root = next(Path(dp).parent for dp in data_paths if os.path.isdir(dp)).joinpath("pixease", expgroup, "suite2p")
        processed_data_path = PROCESSED_DATA_PATH.format(mouse=args.mouse, date=args.date, expnum=expgroup)
        os.makedirs(processed_data_path, exist_ok=True)
        if os.path.isdir(processed_data_path + "suite2p"):
            if args.force:
                _confirm_overwrite("suite2p")
                shutil.rmtree(processed_data_path + "suite2p")
            else:
                print(f"Already processed experiment {expgroup}, continuing")
                continue
        elif args.regroup and "-" not in expgroup:
            matching_groups = _multiglob(PROCESSED_DATA_PATH.format(mouse=args.mouse, date=args.date, expnum="."), f"*-{expgroup}-*")
            if len(matching_groups) > 0:
                continue
        print(data_paths)
        tiffs = _multiglob(data_paths, "*.tif")
        im = tifffile.TiffReader(tiffs[0])
        nslices = im.scanimage_metadata["FrameData"]["SI.hStackManager.numSlices"]
        fs = im.scanimage_metadata["FrameData"]["SI.hRoiManager.scanVolumeRate"]

        ops = suite2p.default_ops()
        ops["nplanes"] = nslices
        # In case you accidentally (or intentionally?) turned on an extra
        # channel while recording.  Assume the correct channels is the first
        # one.
        _channels = im.scanimage_metadata["FrameData"]["SI.hChannels.channelSave"]
        ops["nchannels"] = 1 if isinstance(_channels, (float, int)) else len(np.asarray(_channels).flatten())
        ops["tau"] = tau
        ops["fs"] = fs
        ops["delete_bin"] = True
        ops["batch_size"] = 1000
        ops["combined"] = False
        ops["fast_disk"] = LOCAL_KS_CACHE

        # NOTE: this sorts in alphabetical, not numerical order, e.g., 1, 10,
        # 11, 2, 3 instead of 1, 2, 3, 10, 11
        db = {
            "data_path": list(sorted(set([os.path.dirname(t) for t in tiffs]))),
            "save_path0": processed_data_path,
        }
        opsEnd = suite2p.run_s2p(ops=ops, db=db)
        # TODO I'm not sure if this will work with the "force" argument - we
        # need to explicitly delete it from the remote server.
        print("Copying suite2p output back to server")
        try:
            shutil.copytree(Path(processed_data_path).joinpath("suite2p"), remote_root)
        except OSError as e:
            print("Failed to copy s2p output back to server, continuing anyway.  Exception was:\n\n" + str(e))


def pipeline_kilosort(args):
    """Run Kilosort preprocessing, export representative waveforms, and copy results back to server storage."""
    import pykilosort
    import phylib.io.model

    # Only works with a single probe right now
    data_paths = _multiglob([EPHYS_PATH.format(mouse=args.mouse, date=args.date)], "/**/*.ap.*bin", recursive=True)
    for p in data_paths:
        base_path = Path(p).parent

        data_path_local = Path(LOCAL_KS_CACHE).joinpath(_string_hash(str(base_path)))
        if os.path.exists(base_path.joinpath("kilosort").joinpath("output").joinpath("params.py")) and os.path.exists(
            base_path.joinpath("kilosort").joinpath("output").joinpath("example_waveforms.npz")
        ):
            print("Kilosort output already on server, continuing")
            continue
        if not os.path.exists(data_path_local):
            print(f"Copying {base_path} from server")
            shutil.copytree(base_path, data_path_local)
        else:
            print("Local copy of data exists, skipping")
        output_path = data_path_local.joinpath("kilosort")
        if os.path.exists(output_path.joinpath("output").joinpath("params.py")):
            print("Already ran pykilosort for", p, "in", data_path_local)
        else:
            pykilosort.add_default_handler(level="INFO")
            print(data_path_local, p)
            probe = pykilosort.neuropixel_probe_from_metafile(p)
            print("Running pykilosort for", p)
            pykilosort.run(
                glob.glob(str(data_path_local.joinpath("*.ap.*bin")))[0],
                dir_path=output_path,
                probe=probe,
                low_memory=False,
                template_snapshots=[0.2, 0.5, 0.8],
            )  # No idea what template_snapshots does
            print("Finished pykilosort for", p)
            output_path.joinpath("output").joinpath("kilsort_version.txt").write_text(pykilosort.__version__)
            print("Copying back to server")
            shutil.copytree(output_path, base_path.joinpath("kilosort"))
        if not output_path.joinpath("output").joinpath("example_waveforms.npz").is_file():
            print("Starting example waveforms")
            m = phylib.io.model.load_model(output_path.joinpath("output").joinpath("params.py"))
            N_EXAMPLES = 100
            example_waveforms = []
            example_waveform_times = []
            all_channel_ids = []
            # TODO this dosen't work with splits and merges
            for template_id in range(0, m.n_templates):
                spike_ids = m.get_template_spikes(template_id)
                if len(spike_ids) >= N_EXAMPLES:
                    spike_ids_subset = spike_ids[np.random.RandomState(template_id).choice(spike_ids.shape[0], N_EXAMPLES, replace=False)]
                else:
                    spike_ids_subset = np.concatenate([spike_ids, [spike_ids[0]] * (N_EXAMPLES - len(spike_ids))])
                channel_ids = m.get_template_channels(template_id)
                all_channel_ids.append(channel_ids)
                spikes = m.get_waveforms(spike_ids_subset, channel_ids)
                if spikes is None:
                    raise ValueError("No waveforms found - does params.py have the correct path?")
                example_waveforms.append(spikes)
                example_waveform_times.append(m.spike_times[spike_ids_subset])
            np.savez_compressed(
                output_path.joinpath("output").joinpath("example_waveforms.npz"),
                example_waveforms=example_waveforms,
                example_waveform_times=example_waveform_times,
                channels=np.asarray(all_channel_ids),
            )
            print("Saved example waveforms")
            shutil.copyfile(
                output_path.joinpath("output").joinpath("example_waveforms.npz"),
                base_path.joinpath("kilosort").joinpath("output").joinpath("example_waveforms.npz"),
            )
        print("Deleting local copy")
        shutil.rmtree(data_path_local)


# For converting to iblsorter
# def pipeline_kilosort(args):
#     import iblsorter
#     import iblsorter.ibl
#     import phylib.io.model
#     # Only works with a single probe right now
#     data_paths = _multiglob([EPHYS_PATH.format(mouse=args.mouse, date=args.date)], "/**/*.ap.*bin", recursive=True)
#     for p in data_paths:
#         base_path = Path(p).parent

#         data_path_local = Path(LOCAL_KS_CACHE).joinpath(_string_hash(str(base_path)))
#         if os.path.exists(base_path.joinpath("kilosort").joinpath("output").joinpath("cluster_KSLabel.tsv")) and \
#            os.path.exists(base_path.joinpath("kilosort").joinpath("output").joinpath("example_waveforms.npz")):
#             print("Kilosort output already on server, continuing")
#             continue
#         if not os.path.exists(data_path_local):
#             print(f"Copying {base_path} from server")
#             shutil.copytree(base_path, data_path_local)
#         else:
#             print("Local copy of data exists, skipping")
#         output_path = data_path_local.joinpath("kilosort")
#         if os.path.exists(output_path.joinpath("output").joinpath("cluster_KSLabel.tsv")):
#             print("Already ran ibl pykilosort for", p, "in", data_path_local)
#         else:
#             #pykilosort.add_default_handler(level='INFO')
#             print(data_path_local, p)
#             probe = iblsorter.ibl.probe_geometry(p)
#             print("Running ibl pykilosort for", p)
#             iblsorter.run(glob.glob(str(data_path_local.joinpath("*.ap.*bin")))[0], dir_path=output_path, probe=probe, low_memory=False, ntbuff=16, template_snapshots=[0.2, 0.5, 0.8]) # No idea what template_snapshots does
#             print("Finished ibl pykilosort for", p)
#             output_path.joinpath("output").joinpath("kilsort_version.txt").write_text(iblsorter.__version__)
#             print("Copying back to server")
#             shutil.copytree(output_path, base_path.joinpath("kilosort"))
#         if not output_path.joinpath("output").joinpath("example_waveforms.npz").is_file():
#             print("Starting example waveforms")
#             m = phylib.io.model.load_model(output_path.joinpath("output").joinpath("params.py"))
#             N_EXAMPLES = 100
#             example_waveforms = []
#             example_waveform_times = []
#             all_channel_ids = []
#             # TODO this dosen't work with splits and merges
#             for template_id in range(0, m.n_templates):
#                 spike_ids = m.get_template_spikes(template_id)
#                 if len(spike_ids) >= N_EXAMPLES:
#                     spike_ids_subset = spike_ids[np.random.RandomState(template_id).choice(spike_ids.shape[0], N_EXAMPLES, replace=False)]
#                 else:
#                     spike_ids_subset = np.concatenate([spike_ids, [spike_ids[0]]*(N_EXAMPLES-len(spike_ids))])
#                 channel_ids = m.get_template_channels(template_id)
#                 all_channel_ids.append(channel_ids)
#                 spikes = m.get_waveforms(spike_ids_subset, channel_ids)
#                 if spikes is None:
#                     raise ValueError("No waveforms found - does params.py have the correct path?")
#                 example_waveforms.append(spikes)
#                 example_waveform_times.append(m.spike_times[spike_ids_subset])
#             np.savez_compressed(output_path.joinpath("output").joinpath("example_waveforms.npz"), example_waveforms=example_waveforms, example_waveform_times=example_waveform_times, channels=np.asarray(all_channel_ids))
#             print("Saved example waveforms")
#             shutil.copyfile(output_path.joinpath("output").joinpath("example_waveforms.npz"), base_path.joinpath("kilosort").joinpath("output").joinpath("example_waveforms.npz"))
#         print("Deleting local copy")
#         shutil.rmtree(data_path_local)


def pipeline_extract_flipper(args):
    """For neuropixels experiments, extract the flipper signal"""
    paths = [DP.format(mouse=args.mouse, date=args.date, expnum=".") for DP in DATA_PATHS]
    n_probes = 1  # Currently only one simultaneous probe supported
    n_recordings = len(_multiglob(paths, f"**/*imec0*/", recursive=True))
    for rec in range(0, n_recordings):
        for bin_type in ["ap", "lf"]:
            flipper_path = PROCESSED_DATA_PATH.format(mouse=args.mouse, date=args.date, expnum=".") + f"flipper{rec}.{bin_type}.npz"
            if os.path.exists(flipper_path) and not args.force:
                print(f"{bin_type} flipper {rec} already extracted, continuing")
                continue
            n_channels = 385  # TODO get this from the imec meta file
            # Load aps
            sync_file = _multiglob(paths, f"**/*_g{rec}_*/*.{bin_type}_sync.dat", recursive=True)
            print("Sync files", sync_file)
            if len(sync_file) == 1:
                print(f"Using sync file for flipper {rec} {bin_type}")
                flipper_signal = np.fromfile(sync_file[0], dtype="int16")
            else:
                print(f"No sync file for flipper {rec} {bin_type}, extracting from bin file")
                ap_file = _multiglob(paths, f"**/*_g{rec}_*/*.{bin_type}.bin", one=True, recursive=True)
                n_samples = os.path.getsize(ap_file) // (2 * n_channels)
                flipper_signal = np.zeros(n_samples, dtype="int16")  # Will fill this up in the loop
                i = 0
                batch_size = 100000000 // n_channels
                while True:  # Cheap substitute for a do-while loop
                    aps = (
                        np.fromfile(ap_file, dtype="int16", count=n_channels * batch_size, offset=2 * i * n_channels * batch_size)
                        .reshape(-1, n_channels)
                        .T
                    )
                    if aps.shape[1] == 0:  # Not at the end
                        break
                    flipper_signal[(i * batch_size) : (i * batch_size + aps.shape[1])] = aps[n_channels - 1]
                    i += 1
            processed_data_path = PROCESSED_DATA_PATH.format(mouse=args.mouse, date=args.date, expnum=".")
            os.makedirs(processed_data_path, exist_ok=True)
            np.savez_compressed(flipper_path, flipper_signal)  # Compresses from 1gb to 1mb!


def pipeline_eye(args):
    """Run eye-tracking inference with DeepLabCut for selected experiments."""
    if args.expnums == "auto":
        args.expnums = _auto_detect_expnums(args.mouse, args.date, None)
    expnums = args.expnums.replace("-", ",").split(",")
    for expnum in expnums:
        print("Processing", expnum)
        data_paths = [DP.format(mouse=args.mouse, date=args.date, expnum=expnum) for DP in DATA_PATHS]
        processed_data_path = PROCESSED_DATA_PATH.format(mouse=args.mouse, date=args.date, expnum=expnum)
        os.makedirs(processed_data_path, exist_ok=True)
        videos = _multiglob(data_paths, "*eye.mj2")
        if len(videos) != 1:
            if len(videos) == 0:
                print("mj2 file not found, skipping")
            else:
                print("multiple mj2 files in directory, skipping")
            continue
        if len(glob.glob(processed_data_path + "*eyeDLC*.csv")) == 0:
            import deeplabcut

            deeplabcut.analyze_videos(DEEPLABCUT_EYE_CONFIG_PATH, videos, save_as_csv=True, destfolder=processed_data_path)


def pipeline_cellpose(args):
    """Run volume preprocessing and additionally request Cellpose segmentation outputs."""
    args.run_cellpose = True
    pipeline_volume(args)


def pipeline_volume(args):
    # I spent a few days trying to get the suite2p registration pipeline to work
    # with functional z stacks.  It really just doesn't work well, it creates
    # these big shifts in random frames, I think partially because they are a z
    # stack with low temporal resolution relative to the calcium indicator, but
    # moreso because there aren't very many total frames.  Anyway, the below
    # uses a form of regularised phase correlation with strong regularisation,
    # which seems to work pretty well.
    """Register z-stack frames, build channel/plane summary images, and optionally run Cellpose segmentation."""
    import skimage.registration

    # Need a version of tifffile 2022 or newer!
    from skimage.feature import peak_local_max

    if args.rig == "auto":
        args.rig = _auto_detect_rig(args)
    expnums = args.expnums.replace("-", ",").split(",")
    for expnum in expnums:
        print("Processing", expnum)
        data_paths = [DP.format(mouse=args.mouse, date=args.date, expnum=expnum) for DP in DATA_PATHS]
        processed_data_path = PROCESSED_DATA_PATH.format(mouse=args.mouse, date=args.date, expnum=expnum)
        os.makedirs(processed_data_path, exist_ok=True)

        # Do this here instead of in the if statement because then we can use N_planes, N_channels, and is_zstack
        paths = list(sorted(_multiglob(data_paths, "*2P*.tif")))
        tiffs = [tifffile.TiffFile(p) for p in paths]
        N_planes = tiffs[0].scanimage_metadata["FrameData"]["SI.hStackManager.numSlices"]
        plane_spacing = tiffs[0].scanimage_metadata["FrameData"]["SI.hStackManager.stackZStepSize"]
        zoom = tiffs[0].scanimage_metadata["FrameData"]["SI.hRoiManager.scanZoomFactor"]
        is_zstack = plane_spacing <= 5
        _chansav = tiffs[0].scanimage_metadata["FrameData"]["SI.hChannels.channelSave"]
        N_channels = 1 if isinstance(_chansav, (float, int)) else len(np.asarray(_chansav).flatten())
        channels = [_chansav] if isinstance(_chansav, (float, int)) else np.asarray(_chansav).flatten().tolist()
        try:
            if args.force:
                raise FileNotFoundError("Forcing rerun")
            print("Loading")
            mean_by_channel_plane = _load_npy_compressed(processed_data_path + CHANNELWISE_MEAN_IMAGE_FILENAME)
        except FileNotFoundError:
            print("Running registration")
            frames = np.concatenate([t.asarray() for t in tiffs])
            mean_by_channel_plane = []
            max_by_channel_plane = []
            yxshifts_by_channel_plane = []
            # TODO These shouldn't be different for different channels, but I don't think it matters too much
            for channel in range(0, N_channels):
                mean_by_plane = []
                max_by_plane = []
                yxshifts_by_plane = []
                for plane in range(0, N_planes):
                    pframes = frames[plane::N_planes]
                    if len(pframes.shape) == 4:
                        pframes = pframes[:, channel]
                    mean_img = np.mean(pframes, axis=0)
                    shifted_frames = []
                    yxshifts = []
                    for i in range(1, pframes.shape[0]):
                        shifts = phase_cross_correlation_regularized(mean_img, pframes[i])
                        yxshifts.append(shifts)
                        shifted = scipy.ndimage.shift(pframes[i], shifts)
                        shifted_frames.append(shifted)
                    mean_by_plane.append(np.mean(shifted_frames, axis=0))
                    max_by_plane.append(np.max(shifted_frames, axis=0))
                    yxshifts_by_plane.append(yxshifts)
                mean_by_channel_plane.append(mean_by_plane)
                max_by_channel_plane.append(max_by_plane)
                yxshifts_by_channel_plane.append(yxshifts_by_plane)
            mean_by_channel_plane = np.asarray(mean_by_channel_plane)
            max_by_channel_plane = np.asarray(max_by_channel_plane)
            yxshifts_by_channel_plane = np.asarray(
                [[s + [[0, 0]] * (len(a[0]) - len(s)) for s in a] for a in yxshifts_by_channel_plane]
            )  # Make the array rectangular
            plane_shift_amount = np.zeros((mean_by_channel_plane.shape[1], 2))
            if is_zstack:
                # Don't align to the first frame because it is flyback, and don't align to the last one because it often doesn't have many cells.  Choose the middle frame.
                align_to = N_planes // 2
                # First sweep forward through subsequent frames, then backward through earlier ones
                for channel in range(0, N_channels):
                    for i in range(align_to + 1, N_planes):
                        shifts = phase_cross_correlation_regularized(mean_by_channel_plane[channel][i - 1], mean_by_channel_plane[channel][i])
                        plane_shift_amount[i] = shifts
                        shifted = scipy.ndimage.interpolation.shift(mean_by_channel_plane[channel][i], shifts)
                        mean_by_channel_plane[channel][i] = shifted
                        shifted = scipy.ndimage.interpolation.shift(max_by_channel_plane[channel][i], shifts)
                        max_by_channel_plane[channel][i] = shifted
                    for i in range(align_to - 1, -1, -1):
                        shifts = phase_cross_correlation_regularized(mean_by_channel_plane[channel][i + 1], mean_by_channel_plane[channel][i])
                        plane_shift_amount[i] = shifts
                        shifted = scipy.ndimage.interpolation.shift(mean_by_channel_plane[channel][i], shifts)
                        mean_by_channel_plane[channel][i] = shifted
                        shifted = scipy.ndimage.interpolation.shift(max_by_channel_plane[channel][i], shifts)
                        max_by_channel_plane[channel][i] = shifted
            # We don't need to align channels to each other since they are obtained
            # simultaneously, and both aligned to the mean of the same frame.  So these
            # means should be aligned, and hence, everything here should be too.
            _save_npy_compressed(processed_data_path + CHANNELWISE_MEAN_IMAGE_FILENAME, mean_by_channel_plane)
            _save_npy_compressed(processed_data_path + CHANNELWISE_MAX_IMAGE_FILENAME, max_by_channel_plane)
            _save_npy_compressed(processed_data_path + ZSTACK_Z_REGISTRATION_SHIFT_FILENAME, plane_shift_amount)
            _save_npy_compressed(processed_data_path + ZSTACK_YX_REGISTRATION_SHIFT_FILENAME, yxshifts_by_channel_plane)

        if not hasattr(args, "run_cellpose"):
            continue
        # I don't recommend running cellpose here because you'll need to train
        # it on your own data.  It also takes forever.  Really, don't run
        # cellpose through pixease.
        _N_channels, _N_planes, _, _ = mean_by_channel_plane.shape
        assert _N_channels == N_channels
        assert _N_planes == N_planes
        for cid in range(0, N_channels):
            path_base = f"{processed_data_path}segmentation_channel{channels[cid]}_seg.npy"
            other_path = f"{processed_data_path}segmentation_channel{channels[cid]}_mask.npz"
            if (os.path.isfile(path_base) or os.path.isfile(path_base + ".gz") or os.path.isfile(other_path)) and not args.force:
                print("Continuing")
                continue
            # Import here because these imports are slow and we don't want to
            # slow everything down if we don't need to run cellpose at all
            import cellpose.models
            import cellpose.io

            # In addition to cellpose for the red channel, also do a basic threshold
            if CHANNELMAP[args.rig]["red"] == channels[cid]:
                image = mean_by_channel_plane[cid].astype(float)
                image -= np.min(image)
                image /= np.max(image)
                microns_per_px_x, microns_per_px_y = zoom_to_pixel_size(args.rig, zoom, args.date)
                # Z isn't a consistent spacing so this is just an estimate
                microns_per_px_z = plane_spacing if is_zstack else 10000
                psf_ratio = 3  # Point spread function in z vs x/y
                # Filter and threshold
                sigma_in_microns = 1.3
                sigmas = sigma_in_microns / np.asarray([microns_per_px_z * psf_ratio, microns_per_px_x, microns_per_px_y])
                imf = scipy.ndimage.gaussian_filter(image.astype(float), sigmas) - scipy.ndimage.gaussian_filter(image.astype(float), sigmas * 5)
                thresh = 0.02
                imft = imf
                imft[imft < thresh] = 0
                # Find the distance (in microns) to the border of each blob and take only the ones with a given distance
                labels, nlabels = scipy.ndimage.label(
                    imft, structure=(None if is_zstack else np.asarray([np.zeros((3, 3)), [[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.zeros((3, 3))]))
                )
                label_names = list(range(0, nlabels + 1))
                dist = scipy.ndimage.distance_transform_edt(imft, sampling=(microns_per_px_z, microns_per_px_x, microns_per_px_y))
                # dist = np.asarray([scipy.ndimage.distance_transform_edt(imft[i], sampling=(microns_per_px_x, microns_per_px_y)) for i in range(0, imft.shape[0])])
                isbig = scipy.ndimage.maximum(dist, labels, label_names) >= 2
                newlabel = np.cumsum(isbig)
                relabel = {i: newlabel[i] if isbig[i] else 0 for i in label_names}
                relabel[0] = 0
                labels = np.vectorize(relabel.get)(labels)
                np.savez_compressed(f"{processed_data_path}segmentation_threshold_chan{channels[cid]}", labels=labels)
                print("Saved")
            # Set up cellpose
            model = cellpose.models.Cellpose(gpu=True, model_type=("nuclei" if CHANNELMAP[args.rig]["red"] == channels[cid] else "cyto"))
            # Choose a plane in the middle to estimate cell size
            estimated_diam = model.eval(mean_by_channel_plane[cid][N_planes // 2], diameter=None, flow_threshold=None, channels=[0, 0])[3]
            masks, flows, styles, diams = model.eval(
                mean_by_channel_plane[cid],
                diameter=estimated_diam,
                flow_threshold=None,
                channels=[0, 0],
                do_3D=False,
                stitch_threshold=(0.3 if is_zstack else 1),
            )
            # Fake file name which cell pose modifies to create its output filename instead of allowing the user to specify an output file name directly (???)
            # cellpose.io.masks_flows_to_seg(image=mean_by_channel_plane[cid], masks=masks, flows=flows, diams=diams, image_name=f"{processed_data_path}segmentation_channel{channels[cid]}.tiff", channels=[0,0])
            # Do this ourselves instead of using the cellpose function, because cellpose changes its api in a way that breaks backward compatibility every couple releases
            np.savez_compressed(f"{processed_data_path}segmentation_channel{channels[cid]}_mask.npz", mask=masks)


def pipeline_cache(args):
    # Sub-functions need access to these.  We don't want to import them at the
    # top of the file because not all subcommands need these libraries.  If we
    # import them here, they go out of scope outside the function.  Thus, we
    # need to add them to globals() for them to be accessible to other
    # functions.
    """Build cache files by loading configured modules for each requested experiment and writing notebook outputs."""
    import statsmodels.api as sm
    import cv2

    # import resampy
    globals()["sm"] = sm
    globals()["cv2"] = cv2
    # globals()['resampy'] = resampy

    # We can automatically detect the rig from the explog json file, if
    # it is present.
    if args.mode == "auto":
        args.mode = _auto_detect_mode(args.mouse, args.date)
    if args.rig == "auto":
        args.rig = _auto_detect_rig(args)
    if args.expnums == "auto":
        args.expnums = _auto_detect_expnums(args.mouse, args.date, args.mode)
    else:
        print(f"'{args.expnums}'")
    # The command line arguments allow generating multiple cache files from a
    # single command.  To do this, separate the experiment number with a comma
    # and the experiment types with a comma as well.  If there is no comma in
    # the experiment types, they are all assumed to come from the same type.
    # Otherwise, there should be an equal number of elements in both.  So,
    # iterate through the experiment numbers and match them with the
    # appropriate experiment type.
    expnums = args.expnums.split(",")
    exptypes = args.exptype.split(",")
    assert len(exptypes) == 1 or len(args.expnums.replace("-", ",").split(",")) == len(exptypes), "Invalid number of experiment types"
    i = 0
    for expgroup in args.expnums.split(","):
        for expnum in expgroup.split("-"):
            print("Processing", expnum, expgroup, args.expnums, args.exptype)
            exptype = exptypes[0 if len(exptypes) == 1 else i]
            if exptype == "auto":
                exptype = _auto_detect_exptype(args, expnum)
            Exp = EXPERIMENT_TYPES[exptype]
            Exp.generate_notebook(
                mouse=args.mouse,
                date=args.date,
                expnum=expnum,
                rig=args.rig,
                force=args.force,
                skip=args.skip,
                expgroup=expgroup,
                regroup=args.regroup,
                include=args.include,
                mode=args.mode,
            )
            i += 1


def pipeline_all(args):
    """Execute all required preprocessing pipelines for each experiment type, then build final cache files."""
    mode = _auto_detect_mode(args.mouse, args.date) if args.mode == "auto" else args.mode
    if args.expnums == "auto":
        args.expnums = _auto_detect_expnums(args.mouse, args.date, mode)
    # Create lists of all experiment numbers and types, where experiment type
    # may just have a single element.
    flat_exptypes = args.exptype.replace("-", ",").split(",")
    flat_expnums = args.expnums.replace("-", ",").split(",")
    # Created a nested list of groups of experiments
    expnums = list(map(lambda x: x.split("-"), args.expnums.split(",")))
    # If there are multiple experiment numbers but only one experiment type,
    # assume the same experiment type for all experiment numbers
    if len(flat_exptypes) == 1:
        flat_exptypes *= len(flat_expnums)
    assert len(flat_exptypes) == 1 or len(flat_expnums) == len(flat_exptypes), "Invalid number of experiment types"
    # Where we specified an "auto" experiment type, perform the auto-detection
    flat_exptypes = [e if e != "auto" else _auto_detect_exptype(args, n) for e, n in zip(flat_exptypes, flat_expnums)]
    # Create a nested structure of experiment types in the same nested form as
    # expnums above
    exptypes = [[flat_exptypes[flat_expnums.index(expnums[i][j])] for j in range(0, len(expnums[i]))] for i in range(0, len(expnums))]
    # Run everything which is needed by the experiment type except for "cache"
    # (which shouldn't be in the pipelines variable for any experiment type)
    for p in PIPELINES.keys():
        print("Running", p)
        # For a given pipeline, filter out experiments from the nested
        # structure for which the pipeline does not need to run
        pipes = lambda i, j, m: EXPERIMENT_TYPES[exptypes[i][j]].PIPELINES[m] if m in EXPERIMENT_TYPES[exptypes[i][j]].PIPELINES.keys() else []
        pipe_expnums = [
            [expnums[i][j] for j in range(0, len(expnums[i])) if p in pipes(i, j, mode) + pipes(i, j, "all")] for i in range(0, len(expnums))
        ]
        pipe_exptypes = [
            [exptypes[i][j] for j in range(0, len(exptypes[i])) if p in EXPERIMENT_TYPES[exptypes[i][j]].PIPELINES] for i in range(0, len(exptypes))
        ]
        if all(len(g) == 0 for g in pipe_expnums):
            print("Skipping", p)
            continue
        args.expnums = ",".join(map("-".join, [v for v in pipe_expnums if len(v) > 0]))
        args.exptype = ",".join(map(",".join, [v for v in pipe_exptypes if len(v) > 0]))
        PIPELINES[p][1](args)
    # Now run the cache
    args.expnums = ",".join(map("-".join, [v for v in expnums if len(v) > 0]))
    args.exptype = ",".join(map(",".join, [v for v in exptypes if len(v) > 0]))
    PIPELINES["cache"][1](args)


# TODO add a brain_explorer pipeline

# A list of supported pipelines.  Some pipelines must be run before the others;
# notably, the "cache" pipeline must be run last.  The keys of this dictionary
# should be the name of the pipeline, which the user can call on the command
# line.  The value is a tuple.  The first element is a description of what the
# pipeline does, to be shown in the help text.  The second is the pipeline
# function, which accepts an object containing the command line arguments as
# its only argument.  The final element is a list of the command line arguments
# that are supported by this pipeline.  These are identified by name and
# defined below in the ARGS dictionary.
PIPELINES = {
    "cellpose": (
        "Preprocess a 2p z-stack with Cellpose",
        pipeline_cellpose,
        ["mouse", "date", "expnums", "rig", "force"],
        ["2p"],
    ),
    "volume": (
        "Preprocess a 2p z-stack as a volumetric image",
        pipeline_volume,
        ["mouse", "date", "expnums", "rig", "force", "mode"],
        ["2p"],
    ),
    "s2p": (
        "Preprocess 2p data with suite2p",
        pipeline_s2p,
        ["mouse", "date", "expnums", "gcamp", "force", "regroup"],
        ["2p"],
    ),
    "flipper": (
        "Extract ephys flipper signal from .ap.bin and .lf.bin",
        pipeline_extract_flipper,
        ["mouse", "date", "force"],
        ["ephys"],
    ),
    "kilosort": (
        "Preprocess ephys data with kilosort",
        pipeline_kilosort,
        ["mouse", "date", "expnums", "force"],
        ["ephys"],
    ),
    "eye": (
        "Preprocess eyetracking data with DeepLabCut",
        pipeline_eye,
        ["mouse", "date", "expnums"],
        ["2p", "ephys"],
    ),
    "cache": (
        "Build the cache of a mouse on a pipeline",
        pipeline_cache,
        ["mouse", "date", "expnums", "mode", "rig", "exptype", "include", "skip", "force", "regroup"],
        ["2p", "ephys"],
    ),
    "run": (
        "Run all required preprocessing steps and generate cache",
        pipeline_all,
        ["mouse", "date", "expnums", "mode", "rig", "gcamp", "exptype", "include", "skip", "force", "regroup"],
        ["2p", "ephys"],
    ),
}

# Arguments for pipelines when passed on the command line.  The key is the name
# of the argument, as it is referred to elsewhere in the script.  The value is
# a tuple.  The first element of the tuple is the command line argument itself.
# If it starts with "--" then it is an optional named argument; otherwise, it
# is a positional argument.  The second element is a dictionary of options
# which are passed directly to ArgumentParser.add_argument.
ARGS = {
    "mouse": ("mouse", dict(type=str, help="Name of the mouse")),
    "date": ("date", dict(type=str, help="Date of the experiment, YYYY-MM-DD")),
    "expnums": ("--expnums", dict(type=str, help="Experiment number", default="auto")),
    "mode": ("--mode", dict(choices=["2p", "ephys", "auto"], default="auto", help="What kind of experiment?")),
    "rig": ("--rig", dict(metavar="scope", choices=["auto", "bscope", "b2", "fusi"], help="On which rig were these data obtained?", default="auto")),
    "gcamp": ("--gcamp", dict(choices=["6s", "6m", "6f", "8m", "auto"], default="auto", help="GCaMP version")),
    "exptype": (
        "--exptype",
        dict(
            type=str, metavar="exptype", default="auto", help="Experiment type, can be: " + str(", ".join(["auto"] + list(EXPERIMENT_TYPES.keys())))
        ),
    ),
    "force": ("--force", dict(action="store_true", help="Overwrite previous output")),
    "regroup": (
        "--regroup",
        dict(
            action="store_true",
            help="If a session in its own group (from expnums) has already been processed in a group, reprocess it by itself instead of using the existing group.  Only applies to two-photon.",
        ),
    ),
    "skip": (
        "--skip",
        dict(
            type=str, nargs="+", default=[], metavar="module", help="List of optional modules to skip, can be: " + str(", ".join(MODULE_TYPES.keys()))
        ),
    ),
    "include": (
        "--include",
        dict(
            type=str,
            nargs="+",
            default=[],
            metavar="module",
            help="List of available modules to include, can be: " + str(", ".join(MODULE_TYPES.keys())),
        ),
    ),
}


#################### SECTION: Main loop ####################

if __name__ == "__main__":
    # We use the parser-subparser structure here for arguments.  This means it
    # operates somewhat like "git" on the command line: the main command
    # (pixease) has several sub-commands to perform different tasks.  These are
    # separated because it may not be necessary to rerun everything all of the
    # time.  Additionally, some of these tools like to be installed in their
    # own conda environment and don't allow other packages to be installed
    # there very easily.  Thus, the different subcommands can be run by
    # different versions of Python in different conda environments.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # This generates the valid command line arguments for this script based on
    # the PIPELINES and ARGS mappings above.  It also ensures it is printed
    # nicely when the user uses the "-?" command or provides an invalid
    # command.
    for pname, pinfo in PIPELINES.items():
        p = subparsers.add_parser(pname, help=pinfo[0])
        for argname in pinfo[2]:
            arg = ARGS[argname]
            p.add_argument(arg[0], **arg[1])
        p.set_defaults(func=pinfo[1])
    parser.set_defaults(func=lambda x: parser.print_help())
    args = parser.parse_args()
    args.func(args)
