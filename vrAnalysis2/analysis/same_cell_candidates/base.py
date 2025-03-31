from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from scipy.spatial.distance import squareform
import speedystats as ss
from ...sessions.b2session import B2Session
from .support import pair_val_from_vec, dist_between_points, torch_corrcoef
from ...tracking import Tracker


@dataclass
class SameCellParams:
    """Parameters for same-cell candidate analysis.

    Attributes
    ----------
    spks_type : str
        Source of activity data (default: "corrected")
    keep_planes : List[int] | None
        List of plane indices to analyze. When None, will use the keep_planes
        defined in the session parameters (default: None)
    good_labels : List[str] | None
        List of good labels to include. When None, will use all labels.
    npix_cutoff : Optional[int]
        Minimum number of pixels for ROI masks (default: None)
    pix_to_um : float
        Conversion factor from pixels to micrometers (default: 1.3)
    neuropil_coefficient : float | None
        Neuropil coefficient for activity data (default: 1.0)
        If None, will use the neuropil coefficient from the session.ops (which
        is inherited from suite2p)
    exclude_redundant_rois: bool
        Whether to exclude redundant ROIs from the analysis
        (if this is set to True, will use the "mpciROIs.redundant" mask from the session and therefore
        won't see most ROIs that are considered part of a cluster)
    """

    spks_type: str = "corrected"
    keep_planes: List[int] | None = None
    good_labels: List[str] | None = None
    npix_cutoff: Optional[int] = None
    pix_to_um: float = 1.3
    neuropil_coefficient: float | None = 1.0
    exclude_redundant_rois: bool = False

    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> "SameCellParams":
        """Create a SameCellParams instance from a dictionary."""
        return cls(**params_dict)


@dataclass
class SameCellClusterParameters:
    """Parameters for identifying same-cell clusters.

    Attributes
    ----------
    corr_cutoff : float
        Minimum correlation threshold for a pair to be considered a same-cell candidate
    distance_cutoff : float
        Maximum distance between ROI pairs in μm
    keep_planes : List[int]
        List of plane indices to include default to all planes [0, 1, 2, 3, 4] to make sure
        that the session objects won't filter out planes before clustering analysis!
    good_labels : List[str] | None
        List of good labels to include
    npix_cutoff : float
        Minimum number of pixels for ROI masks
    min_distance : float | None
        Minimum distance between ROI pairs in μm
    max_correlation : float | None
        Maximum correlation between ROI pairs
    best_in_cluster_method : str
        Method for choosing the best ROI in a cluster
    """

    spks_type: str = "corrected"
    corr_cutoff: float = 0.4
    distance_cutoff: float = 20.0
    keep_planes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    good_labels: List[str] | None = None
    npix_cutoff: float = 0.0
    min_distance: float | None = None
    max_correlation: float | None = None
    best_in_cluster_method: str = "max_sum_significant"


@dataclass
class SameCellProcessor:
    """Processes session data to extract ROI correlations, distances, and metadata.

    This class handles the data preparation phase of same-cell candidate analysis.
    It extracts ROI activity, positions, and calculates pairwise correlations and distances.

    Attributes
    ----------
    session : B2Session
        Session data object containing imaging data and ROI information
    params : SameCellParams
        Parameters for the analysis
    """

    session: B2Session
    params: SameCellParams = field(default_factory=SameCellParams)
    _data_loaded: bool = field(default=False, init=False, repr=False)

    # Data containers
    idx_rois: np.ndarray = field(default=None, init=False, repr=False)
    num_rois: int = field(default=0, init=False, repr=False)

    # ROI metadata
    roi_plane_idx: np.ndarray = field(default=None, init=False, repr=False)
    roi_in_target_plane: np.ndarray = field(default=None, init=False, repr=False)
    roi_npix: np.ndarray = field(default=None, init=False, repr=False)
    roi_xy_pos: np.ndarray = field(default=None, init=False, repr=False)

    # Pairwise data
    pairwise_correlations: np.ndarray = field(default=None, init=False, repr=False)
    pairwise_distances: np.ndarray = field(default=None, init=False, repr=False)

    # Pair indices
    idx_roi1: np.ndarray = field(default=None, init=False, repr=False)
    idx_roi2: np.ndarray = field(default=None, init=False, repr=False)
    plane_pair1: np.ndarray = field(default=None, init=False, repr=False)
    plane_pair2: np.ndarray = field(default=None, init=False, repr=False)
    npix_pair1: np.ndarray = field(default=None, init=False, repr=False)
    npix_pair2: np.ndarray = field(default=None, init=False, repr=False)
    xpos_pair1: np.ndarray = field(default=None, init=False, repr=False)
    xpos_pair2: np.ndarray = field(default=None, init=False, repr=False)
    ypos_pair1: np.ndarray = field(default=None, init=False, repr=False)
    ypos_pair2: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize the processor after creation."""
        # Convert params dict to SameCellParams if needed
        if isinstance(self.params, dict):
            self.params = SameCellParams.from_dict(self.params)

        # Make sure the session uses the parameters requested by the processor
        updates = dict(
            keep_planes=self.params.keep_planes,
            good_labels=self.params.good_labels,
            neuropil_coefficient=self.params.neuropil_coefficient,
            spks_type=self.params.spks_type,
            exclude_redundant_rois=self.params.exclude_redundant_rois,
        )
        self.session.update_params(**updates)

        # Load the tracker for choosing best ROIs if this session was tracked (if not will be None)
        self.tracker = Tracker.from_session(self.session, verbose=True)

    @property
    def keep_planes(self) -> List[int]:
        """Get the keep planes from the session parameters."""
        if self.params.keep_planes is None:
            session_keep_planes = self.session.params.keep_planes
            if session_keep_planes is None:
                num_planes = len(self.session.get_value("planeNames"))
                session_keep_planes = np.arange(num_planes)
            return session_keep_planes
        else:
            return self.params.keep_planes

    def load_data(self) -> "SameCellProcessor":
        """Load and process data from the session.

        This method:
        1. Loads ROI metadata (plane indices, positions, sizes)
        2. Loads activity data
        3. Calculates pairwise correlations and distances
        4. Prepares pair indices for filtering
        """
        # Get rois from session (and index to ROIs we should use)
        self.idx_rois = np.where(self.session.idx_rois)[0]
        self.num_rois = len(self.idx_rois)
        spks = self.session.spks[:, self.idx_rois]

        # Get ROI metadata
        stack_position = self.session.loadone("mpciROIs.stackPosition")
        self.roi_plane_idx = stack_position[:, 2].astype(np.int32)  # plane index

        # Load ROI stats and positions
        stat = self.session.load_s2p("stat")
        self.roi_npix = np.array([s["npix"] for s in stat[self.idx_rois]]).astype(np.int32)
        self.roi_xy_pos = stack_position[self.idx_rois, 0:2] * self.params.pix_to_um
        self.roi_plane_idx = self.roi_plane_idx[self.idx_rois].astype(np.int32)

        # Calculate pairwise correlations and distances
        self.pairwise_correlations = squareform(torch_corrcoef(spks.T), checks=False)
        self.pairwise_distances = dist_between_points(self.roi_xy_pos[:, 0], self.roi_xy_pos[:, 1])

        # Create pair indices
        self.idx_roi1, self.idx_roi2 = pair_val_from_vec(np.arange(self.num_rois))
        self.plane_pair1, self.plane_pair2 = pair_val_from_vec(self.roi_plane_idx)
        self.npix_pair1, self.npix_pair2 = pair_val_from_vec(self.roi_npix)
        self.xpos_pair1, self.xpos_pair2 = pair_val_from_vec(self.roi_xy_pos[:, 0])
        self.ypos_pair1, self.ypos_pair2 = pair_val_from_vec(self.roi_xy_pos[:, 1])

        # Calculate number of pairs
        self.num_pairs = len(self.pairwise_correlations)
        assert self.num_pairs == self.num_rois * (self.num_rois - 1) / 2, "Pair calculation error"

        self._data_loaded = True

        return self

    def get_pair_filter(
        self,
        *,
        npix_cutoff: Optional[int] = None,
        keep_planes: Optional[List[int]] = None,
        corr_cutoff: Optional[float] = None,
        distance_cutoff: Optional[float] = None,
        extra_filter: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate boolean filter for ROI pairs based on multiple criteria.

        Parameters
        ----------
        npix_cutoff : int, optional
            Minimum number of pixels for ROI masks
        keep_planes : list, optional
            List of plane indices to include
        corr_cutoff : float, optional
            Minimum correlation threshold
        distance_cutoff : float, optional
            Maximum distance between ROI pairs in μm
        extra_filter : np.ndarray, optional
            Additional boolean filter to apply

        Returns
        -------
        np.ndarray
            Boolean array indicating which pairs pass all filters
        """
        if not self._data_loaded:
            self.load_data()

        if keep_planes is not None:
            assert set(keep_planes) <= set(self.keep_planes), f"Requested planes not in loaded data. Available planes: {self.keep_planes}"

        # Start with all pairs
        pair_idx = np.full(self.num_pairs, True)

        # Apply filters
        if npix_cutoff is not None:
            pair_idx &= (self.npix_pair1 > npix_cutoff) & (self.npix_pair2 > npix_cutoff)

        if keep_planes is not None:
            pair_idx &= np.any(np.stack([self.plane_pair1 == pidx for pidx in keep_planes]), axis=0)
            pair_idx &= np.any(np.stack([self.plane_pair2 == pidx for pidx in keep_planes]), axis=0)

        if corr_cutoff is not None:
            pair_idx &= self.pairwise_correlations > corr_cutoff

        if distance_cutoff is not None:
            pair_idx &= self.pairwise_distances < distance_cutoff

        if extra_filter is not None:
            pair_idx &= extra_filter

        return pair_idx

    def filter_pairs(self, pair_idx: np.ndarray) -> Dict[str, np.ndarray]:
        """Filter ROI pair data based on boolean index.

        Parameters
        ----------
        pair_idx : np.ndarray
            Boolean array indicating which pairs to keep

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing filtered versions of all pair measurements
        """
        if not self._data_loaded:
            self.load_data()

        return {
            "pairwise_correlations": self.pairwise_correlations[pair_idx],
            "pairwise_distances": self.pairwise_distances[pair_idx],
            "plane_pair1": self.plane_pair1[pair_idx],
            "plane_pair2": self.plane_pair2[pair_idx],
            "npix_pair1": self.npix_pair1[pair_idx],
            "npix_pair2": self.npix_pair2[pair_idx],
            "xpos_pair1": self.xpos_pair1[pair_idx],
            "xpos_pair2": self.xpos_pair2[pair_idx],
            "ypos_pair1": self.ypos_pair1[pair_idx],
            "ypos_pair2": self.ypos_pair2[pair_idx],
            "idx_roi1": self.idx_roi1[pair_idx],
            "idx_roi2": self.idx_roi2[pair_idx],
        }


def get_connected_groups(adjacency_matrix: np.ndarray, filter_islands: bool = True) -> List[List[int]]:
    """Find connected components in an undirected graph.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Square adjacency matrix where non-zero values indicate connections

    Returns
    -------
    List[List[int]]
        List of connected components, where each component is a list of node indices
    """
    # Validate input
    assert adjacency_matrix.ndim == 2 and adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Input must be a square matrix"
    assert np.all(adjacency_matrix == adjacency_matrix.T), "Input must be symmetric (undirected graph)"

    n = adjacency_matrix.shape[0]

    # Convert to set representation for efficient operations
    graph = []
    for i in range(n):
        connected = np.where(adjacency_matrix[i] > 0)[0]
        graph.append(set(connected))

    # Find all connected components
    visited = set()
    components = []

    for node in range(n):
        if node not in visited:
            # Start a new component
            component = set([node])
            frontier = set(graph[node])

            # Expand component until no new nodes are added
            while frontier:
                component.update(frontier)
                new_frontier = set()
                for f_node in frontier:
                    new_frontier.update(graph[f_node])
                frontier = new_frontier - component

            components.append(sorted(list(component)))
            visited.update(component)

    if filter_islands:
        components = [c for c in components if len(c) > 1]

    return components


def get_best_roi(scp: SameCellProcessor, cluster: List[int]):
    """This picks which ROI to keep in a cluster. It uses a funny algorithm explained here:

    We want to pick an ROI that has:
    1. Good SNR (defined by sum of significant activity which is a good proxy for SNR)
    2. Is in the preferred mask type (cells or dendrites, not long dendrites or bad masks)
    3. Is tracked in many sessions
    4. Has a high tracking cluster silhouette score (indicating a good quality cluster)

    This is challenging to implement because they might not agree!

    The algorithm uses an attritional and thresholding strategy:
    0. First, if there's a tracker available and one of the ROIs was already chosen as best in
       another session, then we pick that ROI!
    0.1. If there are multiple ROIs that were chosen as best in another session, then we use the
         chosen_as_best array to be our first candidate filter (this is a weird scenario but I
         guess it's possible so the code accounts for it).
    1. Filter candidates that are ROIs in the good mask classes (cells or dendrites)
    2. Filter candidates with top X% SNR (defined relative to max value of sum(significant))
     -- the following two criteria sometimes only subselect candidates based on the above criteria --
    3. Filter candidates with top X% tracked sessions (defined relative to max number of sessions tracked)
    4. Filter candidates with highest silhouette score (defined relative to max value of silhouette score)
    5. If any ROIs pass the above criteria, then we pick the one with the highest SNR.

    If this filtration process doesn't find any ROIs, then we start lowering the thresholds to make it
    easier to find an ROI.
    This annealing algorithm works as follows:
    1. First we reduce the silhouette score threshold until it reaches a minimum (this is the least important criterion).
    2. Then we reduce the SNR threshold until it reaches a minimum.
    3. Then we reduce the number of sessions tracked threshold until it reaches a first minimum (we want to keep this high!).
    4. Next, we start subselecting based on SNR / Label clusters to look for the relatively best candidate instead the absolute
       best in terms of number of sessions tracked and silhouette score.
    5. Then we simply ignore the silhouette score.
    6. Next we reduce the number of sessions tracked threshold until it reaches a second minimum.
    7. Then, we ignore SNR entirely.
    8. Next, we ignore the mask labels entirely.
    9. Finally, we ignore the tracked sessions entirely so the SNR is the only remaining criterion.
    """
    # Define valid mask classes
    # {0: "C", 1: "L", 2: "B", 3: "D"}
    good_classes = [0, 3]  # Cells and dendrites

    # Define SNR threshold
    snr_threshold = 0.9  # relative to max value of sum(significant)
    snr_threshold_dropoff = 0.1  # relative to max value of sum(significant)
    snr_threshold_minimum = 0.4  # after this point, we extend the ignore threshold to include all ROIs

    # Define silhouette score threshold
    silhouette_threshold = 0.9
    silhouette_threshold_dropoff = 0.1
    silhouette_threshold_minimum = 0.3

    # Define sessions_tracked_thresholds
    num_sessions_threshold = 0.9
    num_sessions_threshold_dropoff = 0.1
    num_sessions_threshold_first_minimum = 0.6
    num_sessions_threshold_second_minimum = 0.3

    # Define which criteria to use
    roi_found = False
    using_labels = True  # consider the mask class
    using_snr = True  # consider the SNR
    using_tracked = True  # consider the number of sessions tracked
    using_silhouette = True  # consider the silhouette score
    subselect_tracking = False  # subselect ROIs based on mask label and SNR (if false, will include potential excluded candidates)

    # Get mask classes of ROI type
    mask_classes = scp.session.roicat_classifier["class_predictions"][scp.idx_rois[cluster]]

    # Identify which ROIs are tracked and how much etc
    if scp.tracker is not None:
        index_to_tracked = scp.tracker.sessions.index(scp.session)
        tracked_clusters = scp.tracker.labels[index_to_tracked][scp.idx_rois[cluster]]
        cluster_silhouettes = scp.tracker.cluster_silhouettes[tracked_clusters]
        cluster_silhouettes[tracked_clusters == -1] = -np.inf
        index_to_clusters = [scp.tracker.get_cluster_idx(cluster_id) if cluster_id != -1 else [] for cluster_id in tracked_clusters]
        num_sessions_tracked = np.array([np.sum(itc != -1) for itc in index_to_clusters])

        # Check if any ROIs have already been identified as a best ROI in another session
        chosen_as_best = np.zeros(len(cluster), dtype=bool)
        for icluster, itc in enumerate(index_to_clusters):
            if len(itc) > 0:
                c_chosen = []
                for isession, idx in enumerate(itc):
                    if "mpciROIs.bestInCluster" in scp.tracker.sessions[isession].get_saved_one():
                        c_chosen.append(scp.tracker.sessions[isession].loadone("mpciROIs.bestInCluster")[idx])
                chosen_as_best[icluster] = np.any(c_chosen)

        # If any ROIs have already been identified as a best ROI in another session, then pick that one!
        if np.any(chosen_as_best):
            if np.sum(chosen_as_best) == 1:
                idx_choice = np.where(chosen_as_best)[0][0]
                return idx_choice
            # Else:
            # This means multiple were chosen as best, so we need to pick one of them -- we use
            # the chosen_as_best array to be our first candidate filter
        else:
            # We use chosen as best as the first candidate filter, so when none were chosen
            # just pretend they all were to pick from any of them
            chosen_as_best = np.ones(len(cluster), dtype=bool)

        # If we have a tracker, use an average of the SNR of the ROIs in the tracked clusters
        roi_snr = []
        for itc in index_to_clusters:
            if len(itc) > 0:
                c_sum_sig = []
                for isession, idx in enumerate(itc):
                    c_sum_sig.append(ss.sum(scp.tracker.sessions[isession].get_spks("significant")[:, idx], axis=0))
                roi_snr.append(np.mean(c_sum_sig))
            else:
                roi_snr.append(-np.inf)

        roi_snr = np.array(roi_snr)

    else:
        # Don't use these criteria if there is no tracker
        using_tracked = False
        using_silhouette = False

        # Get sum of significant activity for each ROI (good measure of SNR)
        # We only consider SNR in this session only because it's not tracked
        activity = scp.session.get_spks("significant")[:, scp.idx_rois[cluster]]
        roi_snr = ss.sum(activity, axis=0)

        # We need this as a first filter for when there is no tracker, see comment above for explanation
        chosen_as_best = np.ones(len(cluster), dtype=bool)

    while not roi_found:
        idx_candidates = np.ones(len(cluster), dtype=bool) & chosen_as_best
        if using_labels:
            idx_candidates &= (mask_classes == good_classes[0]) | (mask_classes == good_classes[1])
        if using_snr and np.any(idx_candidates):
            idx_candidates &= roi_snr > (snr_threshold * np.max(roi_snr))
        if using_tracked and np.any(idx_candidates):
            # Get max number of sessions tracked (potentially of current candidates)
            max_tracked = np.max(num_sessions_tracked[idx_candidates]) if subselect_tracking else np.max(num_sessions_tracked)

            # Consider only ROIs that are tracked in most sessions
            idx_candidates &= num_sessions_tracked >= (max_tracked * num_sessions_threshold)

            if using_silhouette and np.any(idx_candidates):
                max_silhouette = np.max(cluster_silhouettes[idx_candidates]) if subselect_tracking else np.max(cluster_silhouettes)
                idx_candidates &= cluster_silhouettes >= max_silhouette

        if np.any(idx_candidates):
            # Get the best ROI
            best_of_candidates = np.argmax(roi_snr[idx_candidates])
            idx_choice = np.where(idx_candidates)[0][best_of_candidates]
            roi_found = True
        else:
            if using_silhouette and silhouette_threshold > silhouette_threshold_minimum:
                silhouette_threshold -= silhouette_threshold_dropoff
            elif using_snr and snr_threshold > snr_threshold_minimum:
                snr_threshold -= snr_threshold_dropoff
            elif using_tracked and num_sessions_threshold > num_sessions_threshold_first_minimum:
                num_sessions_threshold -= num_sessions_threshold_dropoff
            elif not subselect_tracking:
                subselect_tracking = True
            elif using_silhouette:
                using_silhouette = False
            elif using_tracked and num_sessions_threshold > num_sessions_threshold_second_minimum:
                num_sessions_threshold -= num_sessions_threshold_dropoff
            elif using_snr:
                using_snr = False
            elif using_labels:
                using_labels = False
            elif using_tracked:
                using_tracked = False
            else:
                raise ValueError("No criteria to use, this means that the code logic is broken!")

    return idx_choice
