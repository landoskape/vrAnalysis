from typing import List
import joblib
import numpy as np
import speedystats as ss
from vrAnalysis import database
from vrAnalysis.session import vrExperiment
from vrAnalysis2.sessions import create_b2session, B2Session
from vrAnalysis2.analysis.same_cell_candidates import SameCellProcessor, SameCellParams, SameCellClusterParameters, get_connected_groups


# File path for storing clustering data
def get_results_path(session: B2Session):
    cluster_data_path = session.data_path / "clusters"
    if not cluster_data_path.exists():
        cluster_data_path.mkdir(parents=True, exist_ok=True)
    return cluster_data_path


# Master parameters for same cell cluster analysis
cluster_params = SameCellClusterParameters()


def convert_to_b2session(session: vrExperiment, spks_type: str):
    return create_b2session(session.mouseName, session.dateString, session.sessionid, dict(spks_type=cluster_params.spks_type))


def iterate_through_sessions():
    sessiondb = database.vrDatabase("vrSessions")
    for session in sessiondb.iterSessions(imaging=True):
        yield convert_to_b2session(session, cluster_params.spks_type)


def identify_redundant_rois(session: B2Session):
    processor_params = SameCellParams(
        spks_type=cluster_params.spks_type,
        keep_planes=cluster_params.keep_planes,
        npix_cutoff=cluster_params.npix_cutoff,
    )
    scp = SameCellProcessor(session, processor_params).load_data()

    extra_filter = np.ones(scp.num_pairs, dtype=bool)
    if cluster_params.min_distance is not None and cluster_params.min_distance > 0.0:
        extra_filter &= scp.pairwise_distances >= cluster_params.min_distance
    if cluster_params.max_correlation is not None and cluster_params.max_correlation < 1.0:
        extra_filter &= scp.pairwise_correlations <= cluster_params.max_correlation

    pair_filter = scp.get_pair_filter(
        corr_cutoff=cluster_params.corr_cutoff,
        distance_cutoff=cluster_params.distance_cutoff,
        keep_planes=cluster_params.keep_planes,
        npix_cutoff=cluster_params.npix_cutoff,
        extra_filter=extra_filter,
    )

    n_rois = scp.num_rois
    adj_matrix = np.zeros((n_rois, n_rois), dtype=bool)
    filtered_data = scp.filter_pairs(pair_filter)
    for idx1, idx2 in zip(filtered_data["idx_roi1"], filtered_data["idx_roi2"]):
        adj_matrix[int(idx1), int(idx2)] = True
        adj_matrix[int(idx2), int(idx1)] = True

    clusters = get_connected_groups(adj_matrix)
    redundant_rois = np.zeros(session.get_value("numROIs"), dtype=bool)
    for cluster in clusters:
        idx_best_roi = get_best_roi(scp, cluster, method=cluster_params.best_in_cluster_method)
        for iroi, roi in enumerate(cluster):
            if iroi != idx_best_roi:
                redundant_rois[scp.idx_rois[roi]] = True

    results = dict(
        clusters=clusters,
        redundant_rois=redundant_rois,
    )
    return results


def get_best_roi(scp: SameCellProcessor, cluster: List[int], method: str):
    if method == "max_sum_significant":
        activity = scp.session.get_spks("significant")[:, scp.idx_rois[cluster]]
        return np.argmax(ss.sum(activity, axis=0))
    else:
        raise ValueError(f"Invalid best in cluster method: {method}")


if __name__ == "__main__":
    for session in iterate_through_sessions():
        print("Clustering session", session)
        cluster_path = get_results_path(session) / "clusters.joblib"
        results = identify_redundant_rois(session)

        joblib.dump(results, cluster_path)
        session.saveone(results["redundant_rois"], "mpciROIs.redundant")
