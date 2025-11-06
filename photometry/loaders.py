import h5py
import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import detrend
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from _old_vrAnalysis import fileManagement as files

from .process import analyze_data


data_path = files.localDataPath()

h5py_file_tree = dict(
    in_time="DataAcquisition/FPConsole/Signals/Series0001/AnalogIn/Time",
    in_data="DataAcquisition/FPConsole/Signals/Series0001/AnalogIn/AIN03",
    out_time="DataAcquisition/FPConsole/Signals/Series0001/AnalogOut/Time",
    out1="DataAcquisition/FPConsole/Signals/Series0001/AnalogOut/AOUT01",
    out2="DataAcquisition/FPConsole/Signals/Series0001/AnalogOut/AOUT02",
    out3="DataAcquisition/FPConsole/Signals/Series0001/AnalogOut/AOUT03",
)

csv_file_tree = dict(
    in_time="Time(s)",
    in_data="AIn-1",
    out_time="Time(s)",
    out1="AOut-1",
    out2="AOut-2",
    out3="AOut-3",
)


def create_file_dict(file, tree, clear_nan: bool = False):
    data = {}
    for key, value in tree.items():
        data[key] = np.array(file[value])
    if clear_nan:
        idx_nan = np.any(np.stack([np.isnan(data[key]) for key in data]), axis=0)
        for key in data:
            data[key] = data[key][~idx_nan]
    return data


def get_files(mouse_name):
    mouse_number = int(re.match(r".*?(\d+)", mouse_name).group(1))
    if mouse_number < 60:
        return get_csv_files(mouse_name)
    else:
        return get_doric_files(mouse_name)


def get_doric_files(mouse_name):
    """Get all doric files and there dates from the data path"""
    directory = []
    file_index = []
    data = []
    mouse_directory = data_path / mouse_name
    date_directories = [x for x in mouse_directory.iterdir() if x.is_dir()]
    for date_directory in date_directories:
        for file in date_directory.glob("*.doric"):
            file_index_match = re.match(r".*_(\d+).doric", file.name)
            if file_index_match:
                c_file_index = int(file_index_match.group(1))
            else:
                print(f"Could not parse file index from {file.parent}/{file.name}")
                continue
            with h5py.File(file, "r") as f:
                # clear_nan=False because I don't think it needs to for the doric files
                file_data = create_file_dict(f, h5py_file_tree, clear_nan=False)
            file_index.append(c_file_index)
            directory.append(date_directory.name)
            file_data["index"] = file_index
            data.append(file_data)
    return directory, file_index, data


def get_csv_files(mouse_name):
    """Get all csv files and there dates from the data path"""
    directory = []
    file_index = []
    data = []
    mouse_directory = data_path / mouse_name
    date_directories = [x for x in mouse_directory.iterdir() if x.is_dir()]
    for date_directory in date_directories:
        for file in date_directory.glob("*.csv"):
            file_index_match = re.match(r".*_(\d+).csv", file.name)
            if file_index_match:
                c_file_index = int(file_index_match.group(1))
            else:
                print(f"Could not parse file index from {file.parent}/{file.name}")
                continue
            df = pd.read_csv(file, header=1)
            file_data = create_file_dict(df, csv_file_tree, clear_nan=True)
            file_index.append(c_file_index)
            directory.append(date_directory.name)
            file_data["index"] = file_index
            data.append(file_data)
    return directory, file_index, data


def check_doric_filetree(mouse_name):
    """Find a doric file and print the filetree to inspect contents"""
    mouse_directory = data_path / mouse_name
    date_directories = [x for x in mouse_directory.iterdir() if x.is_dir()]
    for date_directory in date_directories:
        for file in date_directory.glob("*.doric"):
            file_index_match = re.match(r".*_(\d+).doric", file.name)
            if file_index_match:
                c_file_index = int(file_index_match.group(1))
            else:
                raise ValueError(f"Could not parse file index from {file.name}")
            with h5py.File(file, "r") as f:
                # Print the full filetree
                f.visit(print)
            return None


def process_single_file(file, preperiod, postperiod, samples=None):
    """Process a single data file and return interpolated data."""
    results = analyze_data(file, preperiod=preperiod + 0.01, postperiod=postperiod + 0.01)

    if samples is None:
        samples = np.linspace(-preperiod, postperiod, int((postperiod + preperiod) * 1000 + 1))

    # Extract and process data timelocked to each opto pulse
    c_idx = results["time_opto"] < postperiod + preperiod
    c_time = results["time_opto"][c_idx]
    c_data = np.mean(results["in2_opto"][:, c_idx] - results["in1_opto"][:, c_idx], axis=0)

    # Interpolate and detrend
    c_interp = interp1d(c_time, c_data, kind="cubic", bounds_error=False, fill_value="extrapolate")(samples)
    c_interp = detrend(c_interp)

    results["time_opto_response"] = samples
    results["opto_response"] = c_interp
    return results


def process_data_parallel(data, preperiod=0.2, postperiod=1.0, n_processes=4, parallel: bool = True):
    """
    Process data files in parallel using multiprocessing.

    Parameters:
    -----------
    data : list
        List of data files to process
    preperiod : float
        Pre-period time in seconds
    postperiod : float
        Post-period time in seconds
    n_processes : int, optional
        Number of processes to use. If None, uses all available CPU cores
    parallel : bool, optional
        If True, process data in parallel

    Returns:
    --------
    numpy.ndarray
        Stacked array of processed data
    """
    # Generate samples once
    samples = np.linspace(-preperiod, postperiod, int((postperiod - preperiod) * 1000))

    # Create partial function with fixed parameters
    process_func = partial(process_single_file, samples=samples, preperiod=preperiod, postperiod=postperiod)

    if parallel:
        # Process files in parallel
        with Pool(processes=n_processes) as pool:
            results = list(pool.imap(process_func, tqdm(data, desc="Processing files")))
    else:
        results = []
        for d in tqdm(data, desc="Processing files"):
            try:
                results.append(process_func(d))
            except Exception as e:
                print(f"Error processing file {d['index']}: {e}")

    opto_responses = [x["opto_response"] for x in results]
    average_opto_response = np.stack(opto_responses)

    # Stack results
    return results, average_opto_response
