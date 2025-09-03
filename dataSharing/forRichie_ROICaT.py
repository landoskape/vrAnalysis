import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import joblib

from vrAnalysis2.database import get_database
from vrAnalysis2.tracking import Tracker
from vrAnalysis2.multisession import MultiSessionSpkmaps


def get_save_path() -> Path:
    """Get path to saved data for current machine.

    Returns
    -------
    save_path : Path
        Path to saved data

    Notes
    -----
    This function returns the path to the saved data for the current machine.
    The path is stored in the path_registry dictionary, which is a dictionary of
    hostname to path. Run ``os.getenv("COMPUTERNAME")`` to get the hostname and
    then add it to the `path_registry` dictionary.
    """

    hostname = os.getenv("COMPUTERNAME")
    path_registry = {
        "ZANDAUZAND": Path(r"D:\localData\sharedData\forRichie_ROICaT_Development"),
    }
    save_path = path_registry[hostname]
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def split_by_plane(data: list[np.ndarray], plane_idx: list[int]) -> list[np.ndarray]:
    """Split data by plane.

    Data and plane_idx should be a list of numpy arrays, where the list dimension
    corresponds to the number of sessions and the numpy array length corresponds to
    the number of ROIs in the session.

    Returns
    -------
    data_by_plane : list[list[np.ndarray]]
        A list of lists of numpy arrays, where the outer list corresponds to the number of sessions
        and the inner list corresponds to the number of planes in the session.
        Each numpy array in the inner list has shape of the number of ROIs in the session for that plane.
    """
    data_by_plane = []
    for session_data, session_plane_idx in zip(data, plane_idx):
        if len(session_data) != len(session_plane_idx):
            raise ValueError("Length of data and plane_idx must be the same!")
        num_planes = np.max(session_plane_idx) + 1
        c_data = [session_data[plane == session_plane_idx] for plane in range(num_planes)]
        data_by_plane.append(c_data)
    return data_by_plane


def summarize_mouse(mouse: str) -> dict[str, any]:
    """Summarize mouse data for ROICaT analysis.

    Collects all relevant data for ROICaT analysis, including suite2p outputs,
    place field analysis data, and quality metrics.

    Includes
    --------
    mouse_name : str
        The name of the mouse.
    suite2p_stats : list[np.ndarray[dict]]
        A list of numpy arrays of suite2p stats for each session.
    suite2p_ops : list[np.ndarray[dict]]
        A list of numpy arrays of suite2p ops for each session.
    vrenv_stats : dict[int, list[int]]
        A dictionary containing the environment statistics.
    roicat_labels : list[np.ndarray[int]]
        A list of roicat labels for each session.
        Outer list has length equal to the number of sessions.
        Inner lists have length equal to the number of ROIs in the session.
    sample_silhouettes : list[np.ndarray[float]]
        A list of numpy arrays of sample silhouettes for each session.
        Outer list has length equal to the number of sessions.
        Inner lists have length equal to the number of ROIs in the session.
    cluster_silhouettes : np.ndarray[float]
        A numpy array of cluster silhouettes.
        Has length equal to the total number of clusters across all sessions.
    placefields : dict[int, list[np.ndarray[float]]]
        A dictionary containing the spike maps for each environment.
        Keys are environment numbers, values are lists of numpy arrays of spike maps.
        Outer lists have length equal to the number of sessions in the environment.
        Inner lists have shape equal to the (number of ROIs, number of spatial bins).
    reliability : dict[int, list[np.ndarray[float]]]
        A dictionary containing the reliability for each environment.
        Keys are environment numbers, values are lists of numpy arrays of reliability.
        Outer lists have length equal to the number of sessions in the environment.
        Inner lists have length equal to the number of ROIs in the session.
    pfloc : dict[int, list[float]]
        A dictionary containing the place field locations for each environment.
        Keys are environment numbers, values are lists of numpy arrays of place field locations.
        Outer lists have length equal to the number of sessions in the environment.
        Inner lists have length equal to the number of ROIs in the session.
    pfidx : dict[int, list[int]]
        A dictionary containing the place field indices for each environment.
        Keys are environment numbers, values are lists of numpy arrays of place field indices.
        Outer lists have length equal to the number of sessions in the environment.
        Inner lists have length equal to the number of ROIs in the session.
    positions : dict[int, list[float]]
        A dictionary containing the positions for each environment (e.g. the virtual locations
        of the spatial bins).
    idx_quality_filters : list[np.ndarray[bool]]
        A list of numpy arrays of session filter indices for each session. These are filters
        used to select for "good" ROIs in each session, including classification filters based on
        an ROICaT morphology classifier, activity filters, and redundancy filters.
        Outer list has length equal to the number of sessions.
        Inner numpy array has shape equal to the number of ROIs in the session.

    Parameters
    ----------
    mouse : str
        The name of the mouse to summarize.

    Returns
    -------
    mouse_data : dict
        A dictionary containing the summarized mouse data.
    """
    tracker = Tracker(mouse)
    msm = MultiSessionSpkmaps(tracker)
    env_stats = msm.env_stats()

    if len(tracker.sessions) != len(msm.processors):
        return {}

    labels = tracker.labels
    sample_silhouettes = tracker.sample_silhouettes
    cluster_silhouettes = tracker.cluster_silhouettes

    # Load suite2p data
    plane_idx = [session.get_plane_idx() for session in tracker.sessions]
    concatenated_suite2p_stat = [session.load_s2p("stat") for session in tracker.sessions]
    suite2p_stat = split_by_plane(concatenated_suite2p_stat, plane_idx)
    suite2p_ops = [session.load_s2p("ops") for session in tracker.sessions]

    # Get session filter indices for each session
    idx_quality_filters = [session.idx_rois for session in tracker.sessions]

    placefields = {}
    reliability = {}
    pfloc = {}
    pfidx = {}
    positions = {}
    for envnum in tqdm(env_stats, desc="... processing each environment", leave=False):
        placefields[envnum], extras = msm.get_spkmaps(
            envnum=envnum,
            idx_ses=env_stats[envnum],
            average=True,
            tracked=False,
            use_session_filters=False,
        )
        reliability[envnum] = extras["reliability"]
        pfloc[envnum] = extras["pfloc"]
        pfidx[envnum] = extras["pfidx"]
        positions[envnum] = extras["positions"]

    mouse_data = {
        "mouse_name": mouse,
        "suite2p_stat": suite2p_stat,
        "suite2p_ops": suite2p_ops,
        "vrenv_stats": env_stats,
        "roicat_labels": labels,
        "sample_silhouettes": sample_silhouettes,
        "cluster_silhouettes": cluster_silhouettes,
        "placefields": placefields,
        "reliability": reliability,
        "pfloc": pfloc,
        "pfidx": pfidx,
        "vrenv_positions": positions,
        "idx_quality_filters": idx_quality_filters,
    }

    return mouse_data


def save_data(mouse_data: dict[str, any]) -> None:
    """Save mouse data to disk.

    mouse_data is a dictionary (see above for structure).
    It includes suite2p data & place field imaging and ROICaT tracking data.

    Upon saving, we recreate the structure of suite2p folders by session and plane.
    In the root folder, we save the other data by key using joblib dumps.

    Parameters
    ----------
    mouse_data : dict[str, any]
        A dictionary containing the mouse data to save.

    Returns
    -------
    None
    """
    mouse_path = get_save_path() / f"{mouse_data['mouse_name']}"
    suite2p_path = mouse_path / "suite2p"
    suite2p_path.mkdir(parents=True, exist_ok=True)
    for key, value in mouse_data.items():
        if key.startswith("suite2p_"):
            suite2p_datatype = key.split("_")[-1]
            for session_idx, session_data in enumerate(value):
                for plane_idx, plane_data in enumerate(session_data):
                    session_plane_path = suite2p_path / f"session{session_idx}" / f"plane{plane_idx}"
                    session_plane_path.mkdir(parents=True, exist_ok=True)
                    np.save(session_plane_path / f"{suite2p_datatype}.npy", plane_data, allow_pickle=True)
        else:
            joblib.dump(value, mouse_path / f"{key}.joblib")


def process_and_save_data() -> tuple[list[str], list[str]]:
    """Process and save data for each mouse.

    Returns
    -------
    successful_mice : list[str]
        A list of mice that were successfully processed and saved.
    failed_mice : list[str]
        A list of mice that were not successfully processed and saved.
    """
    mousedb = get_database("vrMice")
    tracked_mice = mousedb.get_table(tracked=True)["mouseName"].unique()
    successful_mice = []
    failed_mice = []
    for mouse in tqdm(tracked_mice, desc="Processing each mouse", leave=True):
        mouse_data = summarize_mouse(mouse)
        if mouse_data is {}:
            failed_mice.append(mouse + ": processing failed")
            continue

        try:
            save_data(mouse_data)
            successful_mice.append(mouse)
        except:
            failed_mice.append(mouse + ": saving failed")
            continue

    return successful_mice, failed_mice


def load_mouse(mouse: str) -> dict[str, any]:
    """Load mouse data from disk.

    Ignores all suite2p data.

    Parameters
    ----------
    mouse : str
        The name of the mouse to load.

    Returns
    -------
    mouse_data : dict[str, any]
        A dictionary containing the mouse data.
    """
    mouse_path = get_save_path() / f"{mouse}"
    mouse_data = {}
    for file in mouse_path.glob("*.joblib"):
        mouse_data[file.stem] = joblib.load(file)
    return mouse_data


def get_env_stats(mouse_data):
    """
    Helper function to get environment stats from mouse_data.

    Returns a dictionary where the keys represent the environments (by index)
    and the values represent the list of session indices in which the environment is
    present.

    Parameters
    ----------
    mouse_data : dict
        Dictionary containing all mouse data (from load_mouse or summarize_mouse)

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping environment numbers to session indices
    """
    return mouse_data["vrenv_stats"]


def get_placefields(
    mouse_data: dict[str, any],
    envnum: int,
    *,
    idx_ses: list[int] | None = None,
    tracked: bool = True,
    use_session_filters: bool = True,
    pop_nan: bool = True,
) -> tuple[list[np.ndarray], dict]:
    """
    Standalone function to get placefields from mouse_data dictionary.

    This replicates the functionality of MultiSessionSpkmaps.get_spkmaps but works
    with the saved mouse_data dictionary structure.

    Parameters
    ----------
    mouse_data : dict
        Dictionary containing all mouse data (from load_mouse or summarize_mouse)
    envnum : int
        Environment number
    idx_ses : list[int], default=None
        Indices of sessions to process
    tracked : bool, default=True
        Whether to include only tracked cells
    use_session_filters : bool, default=True
        Whether to use session filters to select "good" cells (based on B2Session.idx_rois)
    pop_nan : bool, default=True
        Whether to remove positions that have any NaN values in the placefields (which sometimes
        occur at the beginning or end of the linear track due to sampling).

    Returns
    -------
    placefields : list[np.ndarray]
        List of placefields, one for each selected session (already averaged)
        Shape is (number of ROIs, number of spatial bins)
    extras : dict
        Dictionary containing additional information about the placefields and tracking.
        Always includes:
        - 'idx_tracked' : np.ndarray or None
            Array of tracking indices if tracked=True, else None.
            Shape is (number of sessions, number of tracked ROIs).
        - 'reliability' : list[np.ndarray]
            List of spatial reliability values for each session.
            Each array has shape (number of ROIs,).
        - 'pfloc' : list[np.ndarray]
            List of place field locations for each session.
            Each array has shape (number of ROIs,).
        - 'pfidx' : list[np.ndarray]
            List of place field indices for each session.
            Each array has shape (number of ROIs,).
        - 'positions' : np.ndarray
            Spatial positions/bins for the environment.
            Shape is (number of spatial bins,).
        When tracked=True, also includes:
        - 'cluster_ids' : np.ndarray
            Cluster IDs for tracked ROIs. Shape is (number of tracked ROIs,).
        - 'sample_silhouettes' : np.ndarray
            Sample silhouettes for tracked ROIs across sessions.
            Shape is (number of sessions, number of tracked ROIs).
        - 'cluster_silhouettes' : np.ndarray
            Cluster silhouettes for tracked ROIs. Shape is (number of tracked ROIs,).

    Notes
    -----
    Session filters applied include:
    - Plane filtering (keep_planes)
    - ROICaT classifier filtering (good_labels, fraction_filled_threshold, footprint_size_threshold)
    - Activity filtering (exclude_silent_rois)
    - Redundancy filtering (exclude_redundant_rois)
    """

    # Check that requested environment exists in the data
    if envnum not in mouse_data["placefields"]:
        raise ValueError(f"Environment {envnum} not found in mouse_data")

    # Check session indices or use all if not requested
    sessions_in_environment = mouse_data["vrenv_stats"][envnum]
    if idx_ses is None:
        idx_ses = sessions_in_environment
    else:
        if not all(i in sessions_in_environment for i in idx_ses):
            raise ValueError(f"Requested session indices {idx_ses} not found in environment {envnum}")

    # Relative index to session for current environment
    rel_idx_ses = [sessions_in_environment.index(i) for i in idx_ses]

    # Get vrenv and calcium imaging data for requested sessions
    placefields = [mouse_data["placefields"][envnum][i] for i in rel_idx_ses]
    reliability = [mouse_data["reliability"][envnum][i] for i in rel_idx_ses]
    positions = mouse_data["vrenv_positions"][envnum]

    # Initialize tracking variables
    idx_tracked = None
    tracking_extras = None

    # Handle tracked cells if requested
    if tracked:
        if len(idx_ses) < 2:
            raise ValueError("Can't track for a single session! Need at least 2 sessions.")

        labels = [mouse_data["roicat_labels"][i] for i in idx_ses]
        sample_silhouettes = [mouse_data["sample_silhouettes"][i] for i in idx_ses]
        cluster_silhouettes = mouse_data["cluster_silhouettes"]

        # Get array of labels shared by all requested sessions
        shared_labels = np.intersect1d(labels[0], labels[1])
        for i in range(2, len(labels)):
            shared_labels = np.intersect1d(shared_labels, labels[i])

        # Get rid of -1 placeholder for no cluster found
        shared_labels = shared_labels[shared_labels >= 0]

        # For each session, get index to shared cluster label for each ROI
        # This replicates the functionality of helpers.index_in_target
        def index_in_target(value, target):
            """returns boolean array for whether each value is in target and location array such that target[loc_target] = value"""
            target_to_index = {val: idx for idx, val in enumerate(target)}
            in_target = np.array([val in target_to_index for val in value], dtype=bool)
            loc_target = np.array([target_to_index[val] if in_t else -1 for in_t, val in zip(in_target, value)], dtype=int)
            return in_target, loc_target

        idx_tracked = np.stack([index_in_target(shared_labels, session_labels)[1] for session_labels in labels])

        # Get cluster_ids and quality metrics for tracked ROIs over these sessions
        cluster_ids = labels[0][idx_tracked[0]]
        sample_silhouettes_tracked = np.stack([ss[idx] for idx, ss in zip(idx_tracked, sample_silhouettes)])

        # Apply session filters to tracked cells if requested
        if use_session_filters:
            # Get session filter indices for the tracked sessions
            idx_rois_filters = [mouse_data["idx_quality_filters"][i] for i in idx_ses]
            # Apply session filters to the tracked indices - keep only cells that are both tracked AND pass session filters
            tracked_idx_rois = np.stack([iroi[it] for iroi, it in zip(idx_rois_filters, idx_tracked)], axis=0)
            # Keep only ROIs that pass session filters (use 'any' method like in original tracking.py)
            roi_valid_in_sessions = np.any(tracked_idx_rois, axis=0)
            idx_tracked = idx_tracked[:, roi_valid_in_sessions]
            cluster_ids = cluster_ids[roi_valid_in_sessions]
            sample_silhouettes_tracked = sample_silhouettes_tracked[:, roi_valid_in_sessions]

        cluster_silhouettes_tracked = cluster_silhouettes[cluster_ids]

        tracking_extras = dict(
            cluster_ids=cluster_ids,
            sample_silhouettes=sample_silhouettes_tracked,
            cluster_silhouettes=cluster_silhouettes_tracked,
        )

        # Apply tracking filter to data
        placefields = [pf[it] for pf, it in zip(placefields, idx_tracked)]
        reliability = [rel[it] for rel, it in zip(reliability, idx_tracked)]

    # Handle session filters if requested and not already applied via tracking
    if use_session_filters and not tracked:
        # Apply session filters using the saved idx_rois data
        idx_rois_filters = [mouse_data["idx_quality_filters"][i] for i in idx_ses]
        placefields = [pf[iroi] for pf, iroi in zip(placefields, idx_rois_filters)]
        reliability = [rel[iroi] for rel, iroi in zip(reliability, idx_rois_filters)]

    # Handle NaN removal if requested
    if pop_nan:
        # For averaged data, check for NaN positions
        idx_nan_positions = np.any(np.stack([np.any(np.isnan(s), axis=0) for s in placefields]), axis=0)
        placefields = [pf[..., ~idx_nan_positions] for pf in placefields]
        positions = positions[~idx_nan_positions]

    # Get place field locations and indices using the saved data
    pfloc = [mouse_data["pfloc"][envnum][i] for i in rel_idx_ses]
    pfidx = [mouse_data["pfidx"][envnum][i] for i in rel_idx_ses]

    # Apply tracking filter to place field data if needed
    if tracked and idx_tracked is not None:
        pfloc = [pf[it] for pf, it in zip(pfloc, idx_tracked)]
        pfidx = [pf[it] for pf, it in zip(pfidx, idx_tracked)]
    elif use_session_filters and not tracked:
        # Apply session filters to place field data
        idx_rois_filters = [mouse_data["idx_quality_filters"][i] for i in idx_ses]
        pfloc = [pf[iroi] for pf, iroi in zip(pfloc, idx_rois_filters)]
        pfidx = [pf[iroi] for pf, iroi in zip(pfidx, idx_rois_filters)]

    extras = dict(
        idx_tracked=idx_tracked,
        reliability=reliability,
        pfloc=list(pfloc),
        pfidx=list(pfidx),
        positions=positions,
    )
    if tracking_extras is not None:
        extras.update(tracking_extras)

    return placefields, extras


if __name__ == "__main__":
    # successful_mice, failed_mice = process_and_save_data()
    # print(f"Successful mice: {successful_mice}")
    # print(f"Failed mice: {failed_mice}")

    mouse_data = load_mouse("CR_Hippocannula6")
    placefields, extras = get_placefields(mouse_data, 2, idx_ses=None, tracked=True, use_session_filters=True, pop_nan=True)
    print("\n".join(extras.keys()))
    print("hi')")
