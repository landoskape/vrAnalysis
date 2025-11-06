from argparse import ArgumentParser
import shutil

import os
import sys

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from _old_vrAnalysis import tracking
from _old_vrAnalysis import fileManagement as fm

# shared data folder name
data_name = "roicatExample"


def handle_inputs():
    parser = ArgumentParser(description="copy ROICaT input data for sharing")
    parser.add_argument("--mouse-name", type=str, required=True, help="the mouse name to copy sharing data from")
    return parser.parse_args()


def create_folder():
    """target folder for this shared data dump"""
    folder = fm.sharedDataPath() / data_name
    if not folder.exists():
        folder.mkdir()


def generate_filepath(mouse_name, plane_name):
    """specific file path for particular file in this shared data dump"""
    path_name = fm.sharedDataPath() / data_name / mouse_name / plane_name
    return path_name


def generate_dictionary(mouse_name):
    """method for getting data to share for ROICaT testing"""
    track = tracking.tracker(mouse_name)

    # dictionary to store list of files and session ids
    files = {}

    # contains an iterable of tuples of mouse_name / date / session ID
    for plane in range(track.num_planes):
        c_dates = []
        c_session_id = []
        c_stat_files = []
        c_ops_files = []
        for _, date, sesid in track.session_names:
            cpath = fm.localDataPath() / mouse_name / date / sesid / "suite2p" / f"plane{plane}"
            c_dates.append(date)
            c_session_id.append(sesid)
            c_stat_files.append(cpath / "stat.npy")
            c_ops_files.append(cpath / "ops.npy")

        files[f"plane{plane}"] = dict(
            date=c_dates,
            session_id=c_session_id,
            stat=c_stat_files,
            ops=c_ops_files,
        )

    return files


def copy_files(mouse_name, files):
    for plane in files:
        for date, stat, ops in zip(files[plane]["date"], files[plane]["stat"], files[plane]["ops"]):
            c_path = generate_filepath(mouse_name, plane) / date
            _copy_file(stat, c_path / "stat.npy")
            _copy_file(ops, c_path / "ops.npy")


def _copy_file(src, dest):
    if not dest.parent.exists():
        dest.parent.mkdir(parents=True)
    shutil.copy(src, dest)


if __name__ == "__main__":
    args = handle_inputs()

    create_folder()

    print("Generate data dictionary...")
    files = generate_dictionary(args.mouse_name)

    print("Copying files...")
    copy_files(args.mouse_name, files)
