from tqdm import tqdm
import numpy as np

import os
import sys

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from _old_vrAnalysis import helpers
from _old_vrAnalysis import fileManagement as fm
from _old_vrAnalysis import database
from _old_vrAnalysis import analysis
from _old_vrAnalysis import tracking

# shared data folder name
data_name = "forAli_250106"


def create_folder():
    """target folder for this shared data dump"""
    folder = fm.sharedDataPath() / data_name
    if not folder.exists():
        folder.mkdir(parents=True)


def generate_filepath(name):
    """specific file path for particular file in this shared data dump"""
    path_name = fm.sharedDataPath() / data_name / name
    return path_name


def generate_data_stores():
    """generate data stores for shared data dump"""
    keep_planes = [1, 2, 3]
    filename = []
    data = []

    mousedb = database.vrDatabase("vrMice")
    df = mousedb.getTable(trackerExists=True)
    mouse_names = df["mouseName"].unique()

    for mouse_name in tqdm(mouse_names, desc="Gathering data for each mouse.", leave=True):
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=keep_planes)

        environments = pcm.environments
        idx_ses = {ises: [] for ises in range(len(pcm.pcss))}
        for env in environments:
            c_idx_ses = pcm.idx_ses_selector(env, sesmethod=-0.5)
            if c_idx_ses[0] >= 4:
                for c in c_idx_ses:
                    idx_ses[c].append(env)

        # Filter by sessions that have at least one environment
        idx_ses = {ises: ienvs for ises, ienvs in idx_ses.items() if len(ienvs) > 0}

        # Include no more than 3 sessions
        idx_ses = {ises: ienvs for ii, (ises, ienvs) in enumerate(idx_ses.items()) if ii < 3}

        # For each session, get the data from the desired environments and add it to the list of data to save
        for ises, ienvs in tqdm(idx_ses.items(), desc="Getting data for each session.", leave=False):
            pcss = pcm.pcss[ises]
            spkmaps = pcss.get_spkmap(envnum=ienvs, average=False)

            # To concatenate across environments we need to equalize the number of trials
            min_trials = np.min([s.shape[1] for s in spkmaps])
            spkmaps = [s[:, sorted(np.random.choice(s.shape[1], min_trials, replace=False))] for s in spkmaps]

            # Concatenate and save which position corresponds to which environment
            allmaps = np.concatenate(spkmaps, axis=2)
            environment = np.repeat(np.arange(len(spkmaps)), spkmaps[0].shape[2])

            # Extend with a dictionary for each session
            data.append(np.array({"maps": allmaps, "environment": environment}))
            env_string = ",".join([str(i) for i in ienvs])
            filename.append(f"{mouse_name}_Session{ises}_Envs{env_string}")

    return filename, data


def load_data(data_path):
    """
    Load all numpy files from the specified data folder.

    Parameters:
    -----------
    data_path : str
        Path to the folder containing all the files.

    Returns:
    --------
    data : list
        List of numpy arrays containing the maps data
    environments : list
        List of numpy arrays containing the environment labels
    """
    from pathlib import Path
    import numpy as np

    # Get the base path
    path = Path(data_path)

    # Get all .npy files in the folder
    files = sorted(path.glob("*.npy"))

    # Load each file and extract data
    data = []
    environments = []

    for file in files:
        loaded = np.load(file, allow_pickle=True).item()
        data.append(loaded["maps"])
        environments.append(loaded["environment"])

    return data, environments


if __name__ == "__main__":
    create_folder()

    print("Generate data dictionary...")
    filename, data = generate_data_stores()

    print("Saving data...")
    for fname, d in zip(filename, data):
        fpath = generate_filepath(fname)
        print(f"Saving: {fpath}")
        np.save(fpath, d)
