from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from scipy.spatial.distance import squareform
from ...sessions.b2session import B2Session
from .support import pair_val_from_vec, dist_between_points, torch_corrcoef


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
        List of plane indices to include
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
    keep_planes: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
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
            neuropil_coefficient=self.params.neuropil_coefficient,
            spks_type=self.params.spks_type,
            exclude_redundant_rois=self.params.exclude_redundant_rois,
        )
        self.session.update_params(**updates)

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
