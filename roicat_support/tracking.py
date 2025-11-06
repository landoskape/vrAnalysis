import re
import joblib
import numpy as np
from vrAnalysis import files

PLANE_NAMES = ["plane0", "plane1", "plane2", "plane3", "plane4"]


def roicat_tracking_directory(mouse_name: str):
    return files.local_data_path() / mouse_name / "ROICaT2025"


def get_roicat_tracking_paths(mouse_name: str):
    """Get the paths to the ROICaT tracking results for a given mouse.

    Parameters
    ----------
    mouse_name : str
        The name of the mouse to get the ROICaT tracking results for.

    Returns
    -------
    paths_save : dict
        A lambda function to a dictionary of paths to the ROICaT tracking
        results for the given mouse that accepts the plane name as input.
    """
    dir_save = roicat_tracking_directory(mouse_name)
    paths_save = lambda plane_name: {
        "results_clusters": str(dir_save / f"{mouse_name}.{plane_name}.tracking.results_clusters.joblib"),
        "params_used": str(dir_save / f"{mouse_name}.{plane_name}.tracking.params_used.joblib"),
        "results_all": str(dir_save / f"{mouse_name}.{plane_name}.tracking.results_all.joblib"),
        "run_data": str(dir_save / f"{mouse_name}.{plane_name}.tracking.run_data.joblib"),
    }
    return paths_save


def split_by_session(labels, rois_per_session):
    indices = np.cumsum(rois_per_session)[:-1]
    return np.split(labels, indices)


def get_label_padding_for_planes(labels):
    max_label = [np.max(l) for l in labels]
    unique_labels = [len(np.unique(l)) for l in labels]
    for ul, ml in zip(unique_labels, max_label):
        if ul != ml + 2:
            raise ValueError(f"#Unique labels != (max label + 2)... {ul} != {ml + 2}")
    num_unique_labels = [ul - 1 for ul in unique_labels]  # exclude -1
    padding = np.cumsum([0, *num_unique_labels])[:-1]  # first plane isn't padded
    return padding


def group_planes_with_padding(list_across_planes, padding):
    if len(list_across_planes) != len(padding):
        raise ValueError("number of planes in list and padding not equal!")
    # First identify -1s because these should stay as -1s!
    not_in_cluster = np.concatenate([np.array(list_values) == -1 for list_values in list_across_planes])
    # Pad each list
    padded_across_planes = np.concatenate([list_values + pad_value for list_values, pad_value in zip(list_across_planes, padding)])
    # Restore not in clusters
    padded_across_planes[not_in_cluster] = -1
    return padded_across_planes


def identify_session_info(path):
    pattern = r".*\\([^\\]+)\\(\d{4}-\d{2}-\d{2})\\(\d+)\\.*"
    match = re.match(pattern, path)

    if match:
        mouse_name = match.group(1)
        date = match.group(2)
        session = match.group(3)

        return mouse_name, date, session

    else:
        raise ValueError(f"Path {path} does not match expected format")


def identify_tracking_file_session(path):
    pattern = r".*\\([^\\]+)_(\d{4}-\d{2}-\d{2})_(\d+)*"
    match = re.match(pattern, path)

    if match:
        mouse_name = match.group(1)
        date = match.group(2)
        session = match.group(3)

        return mouse_name, date, session

    else:
        raise ValueError(f"Path {path} does not match expected format")


def consolidate_labels(mouse_name: str, plane_names: list[str] = PLANE_NAMES, save: bool = True):
    paths_save = get_roicat_tracking_paths(mouse_name)

    results = [joblib.load(paths_save(plane_name)["results_all"]) for plane_name in plane_names]
    paths = [result["input_data"]["paths_stat"] for result in results]
    mnames, date, session = map(list, zip(*[identify_session_info(p) for p in paths[0]]))

    # Handle clusters
    labels = [rc["clusters"]["labels"] for rc in results]
    labels_by_session = [rc["clusters"]["labels_bySession"] for rc in results]
    roi_per_session = [[len(l) for l in lbs] for lbs in labels_by_session]
    sample_silhouette = [split_by_session(rc["clusters"]["quality_metrics"]["sample_silhouette"], rps) for rc, rps in zip(results, roi_per_session)]
    cluster_silhouettes = np.concatenate([rc["clusters"]["quality_metrics"]["cluster_silhouette"][1:] for rc in results])

    # Group across planes
    padding = get_label_padding_for_planes(labels)
    lbs_transpose = list(map(list, zip(*labels_by_session)))
    ss_transpose = list(map(list, zip(*sample_silhouette)))
    full_labels_by_session = [group_planes_with_padding(lbs, padding) for lbs in lbs_transpose]
    full_sample_silhouettes_by_session = [group_planes_with_padding(ss, 0 * padding) for ss in ss_transpose]

    if save:
        dir_save = roicat_tracking_directory(mouse_name)
        for isession in range(len(full_labels_by_session)):
            path_string = f"{mnames[isession]}_{date[isession]}_{session[isession]}"
            np.save(dir_save / (path_string + ".labels.npy"), full_labels_by_session[isession])
            np.save(dir_save / (path_string + ".sample_silhouettes.npy"), full_sample_silhouettes_by_session[isession])
        np.save(dir_save / "cluster_silhouettes.npy", cluster_silhouettes)

    return full_labels_by_session, full_sample_silhouettes_by_session, cluster_silhouettes


def make_red_cell_labeling_coherent(tracker, save: bool = True):
    # Load red assignment data for each session
    idx_red = [ses.loadone("mpciROIs.redCellIdx") for ses in tracker.sessions]
    manual_red = [ses.loadone("mpciROIs.redCellManualAssignments") for ses in tracker.sessions]
    manual_label = [mr[0] for mr in manual_red]
    manual_active = [mr[1] for mr in manual_red]

    # Create a copy of the red labels we'll update for tracked ROIs
    coherent_red_label = [np.copy(ired) for ired in idx_red]

    # For each cluster for the mouse, figure out which sessions it's in
    # and which ROI corresponds to that cluster for each session...
    num_clusters = tracker.cluster_silhouettes.shape[0]
    for icluster in range(num_clusters):
        sessions_with_cluster = []
        idx_to_cluster = []

        for idx_session, session_labels in enumerate(tracker.labels):
            itc = np.where(session_labels == icluster)[0]
            if len(itc) > 0:
                sessions_with_cluster.append(idx_session)
                idx_to_cluster.append(itc)

        # Get the red data for this particular cluster
        red_value = [idx_red[isession][itc] for isession, itc in zip(sessions_with_cluster, idx_to_cluster)]
        red_manual_label = [manual_label[isession][itc] for isession, itc in zip(sessions_with_cluster, idx_to_cluster)]
        red_manual_active = [manual_active[isession][itc] for isession, itc in zip(sessions_with_cluster, idx_to_cluster)]

        # Generate a coherent red label based on a simple algorithm
        if np.any(red_manual_active):
            # If I manually labeled the ROI at any point,
            # check if the label(s) are coherent
            # and if they are, use that as the "coherent" label for the tracked ROI
            roi_coherent_label = np.unique([rml for rml, rma in zip(red_manual_label, red_manual_active) if rma])
            if len(roi_coherent_label) > 1:
                raise ValueError(f"Manual labels inconsistent")

        else:
            # Otherwise, we call it red if at least 33% of the tracked ROIs were labeled red
            roi_coherent_label = np.sum(red_value) / len(red_value) > 0.33

        # Relabel the roi across sessions with a coherent label
        for isession, itc in zip(sessions_with_cluster, idx_to_cluster):
            coherent_red_label[isession][itc] = roi_coherent_label

    # Save the coherent red labels
    if save:
        for isession, red_label in enumerate(coherent_red_label):
            tracker.sessions[isession].saveone(red_label, "mpciROIs.redCellIdxCoherent")

    return coherent_red_label
