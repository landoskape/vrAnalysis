from typing import Union, Tuple, Iterable, List, Optional, Dict, Any
from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field, asdict, fields
from functools import wraps
from pathlib import Path
from copy import copy
import json
import hashlib
import numpy as np
import numba as nb
import speedystats as ss
from .. import helpers
from ..sessions.base import SessionData
from ..sessions.b2session import B2Session
from .support import median_zscore, get_gauss_kernel, convolve_toeplitz
from .support import get_summation_map, correct_map, replace_missing_data
from .support import placefield_prediction_numba


@dataclass
class Maps:
    """Container for occupancy, speed, and spike maps.

    This class holds spatial maps representing neural activity, behavioral occupancy,
    and speed across position bins. Maps can be organized either as single arrays
    (all trials combined) or as lists of arrays (separated by environment).

    Attributes
    ----------
    occmap : np.ndarray or list of np.ndarray
        Occupancy map(s) representing time spent in each position bin.
        Shape: (trials, positions) for single array, or list of (trials, positions)
        arrays when by_environment=True.
    speedmap : np.ndarray or list of np.ndarray
        Speed map(s) representing average speed in each position bin.
        Shape: (trials, positions) for single array, or list of (trials, positions)
        arrays when by_environment=True.
    spkmap : np.ndarray or list of np.ndarray
        Spike map(s) representing neural activity in each position bin.
        Shape depends on rois_first:
        - If rois_first=True: (rois, trials, positions) or list of (rois, trials, positions)
        - If rois_first=False: (trials, positions, rois) or list of (trials, positions, rois)
    by_environment : bool
        Whether maps are separated by environment (True) or combined (False).
    rois_first : bool
        Whether ROI dimension is first (True) or last (False) in spkmap arrays.
    environments : list of int, optional
        List of environment numbers when by_environment=True. Default is None.
    distcenters : np.ndarray, optional
        Center positions of distance bins. Default is None.
    _averaged : bool
        Internal flag indicating whether trials have been averaged. Default is False.

    Notes
    -----
    The Maps class supports two organizational modes:
    1. Single maps: All trials combined in single arrays (by_environment=False)
    2. Environment-separated maps: Maps split by environment (by_environment=True)

    The spkmap can have ROIs as the first or last dimension depending on rois_first.
    This allows flexibility in how data is organized for different processing steps.
    """

    occmap: np.ndarray | list[np.ndarray]
    speedmap: np.ndarray | list[np.ndarray]
    spkmap: np.ndarray | list[np.ndarray]
    by_environment: bool
    rois_first: bool
    environments: list[int] | None = None
    distcenters: np.ndarray | None = None
    _averaged: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.occmap is None or self.speedmap is None or self.spkmap is None:
            raise ValueError("occmap, speedmap, and spkmap must be provided")

        if self.by_environment:
            if self.environments is None:
                raise ValueError("environments must be provided if by_environment is True")
            if not isinstance(self.occmap, list) or not isinstance(self.speedmap, list) or not isinstance(self.spkmap, list):
                raise ValueError("occmap, speedmap, and spkmap must be lists if by_environment is True")
        else:
            if isinstance(self.occmap, list) or isinstance(self.speedmap, list) or isinstance(self.spkmap, list):
                raise ValueError("occmap, speedmap, and spkmap must be single arrays if by_environment is False")

        if not self.by_environment:
            spkmap_shape = self.spkmap.shape[1:] if self.rois_first else self.spkmap.shape[:2]
            if not (self.occmap.shape == self.speedmap.shape == spkmap_shape):
                raise ValueError("occmap, speedmap, and spkmap must have the same shape")
        else:
            if not (len(self.occmap) == len(self.speedmap) == len(self.spkmap) == len(self.environments)):
                raise ValueError("occmap, speedmap, and spkmap must have the same number of environments")
            for i in range(len(self.environments)):
                spkmap_shape = self.spkmap[i].shape[1:] if self.rois_first else self.spkmap[i].shape[:2]
                if not (self.occmap[i].shape == self.speedmap[i].shape == spkmap_shape):
                    raise ValueError("occmap, speedmap, and spkmap must have the same shape for each environment")
            roi_axis = 0 if self.rois_first else -1
            rois_per_env = [spkmap.shape[roi_axis] for spkmap in self.spkmap]
            if not all([rpe == rois_per_env[0] for rpe in rois_per_env]):
                raise ValueError("All environments must have the same number of ROIs")

    def __repr__(self) -> str:
        # Get number of positions
        if self.by_environment:
            num_positions = self.occmap[0].shape[-1]
        else:
            num_positions = self.occmap.shape[-1]
        # Get number of trials
        if self._averaged:
            num_trials = "averaged"
        else:
            if self.by_environment:
                num_trials = [occmap.shape[0] for occmap in self.occmap]
                num_trials = "{" + ", ".join([str(nt) for nt in num_trials]) + "}"
            else:
                num_trials = self.occmap.shape[0]
        # Get number of ROIs
        if self.by_environment:
            num_rois = self.spkmap[0].shape[0] if self.rois_first else self.spkmap[0].shape[1]
        else:
            num_rois = self.spkmap.shape[0] if self.rois_first else self.spkmap.shape[1]
        environments = f", environments={{{', '.join([str(env) for env in self.environments])}}}" if self.by_environment else ""
        return f"Maps(num_trials={num_trials}, num_positions={num_positions}, num_rois={num_rois}{environments}, rois_first={self.rois_first})"

    @classmethod
    def create_raw_maps(cls, occmap: np.ndarray, speedmap: np.ndarray, spkmap: np.ndarray, distcenters: np.ndarray = None) -> "Maps":
        """Create a Maps instance from raw (unprocessed) map data.

        Parameters
        ----------
        occmap : np.ndarray
            Occupancy map with shape (trials, positions).
        speedmap : np.ndarray
            Speed map with shape (trials, positions).
        spkmap : np.ndarray
            Spike map with shape (trials, positions, rois).
        distcenters : np.ndarray, optional
            Center positions of distance bins. Default is None.

        Returns
        -------
        Maps
            Maps instance with by_environment=False and rois_first=False.
        """
        return cls(occmap=occmap, speedmap=speedmap, spkmap=spkmap, distcenters=distcenters, by_environment=False, rois_first=False)

    @classmethod
    def create_processed_maps(cls, occmap: np.ndarray, speedmap: np.ndarray, spkmap: np.ndarray, distcenters: np.ndarray = None) -> "Maps":
        """Create a Maps instance from processed map data.

        Parameters
        ----------
        occmap : np.ndarray
            Occupancy map with shape (trials, positions).
        speedmap : np.ndarray
            Speed map with shape (trials, positions).
        spkmap : np.ndarray
            Spike map with shape (rois, trials, positions).
        distcenters : np.ndarray, optional
            Center positions of distance bins. Default is None.

        Returns
        -------
        Maps
            Maps instance with by_environment=False and rois_first=True.
        """
        return cls(occmap=occmap, speedmap=speedmap, spkmap=spkmap, distcenters=distcenters, by_environment=False, rois_first=True)

    @classmethod
    def create_environment_maps(
        cls,
        occmap: list[np.ndarray],
        speedmap: list[np.ndarray],
        spkmap: list[np.ndarray],
        environments: list[int],
        distcenters: np.ndarray = None,
    ) -> "Maps":
        """Create a Maps instance with maps separated by environment.

        Parameters
        ----------
        occmap : list of np.ndarray
            List of occupancy maps, one per environment. Each with shape (trials, positions).
        speedmap : list of np.ndarray
            List of speed maps, one per environment. Each with shape (trials, positions).
        spkmap : list of np.ndarray
            List of spike maps, one per environment. Each with shape (rois, trials, positions).
        environments : list of int
            List of environment numbers corresponding to each map in the lists.
        distcenters : np.ndarray, optional
            Center positions of distance bins. Default is None.

        Returns
        -------
        Maps
            Maps instance with by_environment=True and rois_first=True.
        """
        return cls(
            occmap=occmap,
            speedmap=speedmap,
            spkmap=spkmap,
            distcenters=distcenters,
            environments=environments,
            by_environment=True,
            rois_first=True,
        )

    @classmethod
    def map_types(cls) -> List[str]:
        """Get the list of map type names.

        Returns
        -------
        list of str
            List containing ["occmap", "speedmap", "spkmap"].
        """
        return ["occmap", "speedmap", "spkmap"]

    def __getitem__(self, key: str) -> np.ndarray | list[np.ndarray]:
        """Get a map by name using dictionary-like access.

        Parameters
        ----------
        key : str
            Name of the map to retrieve ("occmap", "speedmap", or "spkmap").

        Returns
        -------
        np.ndarray or list of np.ndarray
            The requested map array(s).
        """
        return getattr(self, key)

    def __setitem__(self, key: str, value: np.ndarray | list[np.ndarray]) -> None:
        """Set a map by name using dictionary-like access.

        Parameters
        ----------
        key : str
            Name of the map to set ("occmap", "speedmap", or "spkmap").
        value : np.ndarray or list of np.ndarray
            The map array(s) to assign.
        """
        setattr(self, key, value)

    def _get_position_axis(self, mapname: str) -> int:
        """Get the axis index for the position dimension.

        Parameters
        ----------
        mapname : str
            Name of the map ("occmap", "speedmap", or "spkmap").

        Returns
        -------
        int
            Axis index for the position dimension. Typically -1 (last axis),
            except for spkmap when rois_first=False, where it's -2.

        Notes
        -----
        The only time the position axis isn't the last one is for spkmap when
        rois_first is False, where the shape is (trials, positions, rois).
        """
        average_offset = -1 if self._averaged else 0
        if mapname == "spkmap" and not self.rois_first:
            return -2 + average_offset
        else:
            return -1

    def filter_positions(self, idx_positions: np.ndarray) -> None:
        """Filter maps to keep only specified position bins.

        Parameters
        ----------
        idx_positions : np.ndarray
            Indices of position bins to keep. Must be a 1D array of integers.

        Notes
        -----
        This method modifies the maps in-place, keeping only the position bins
        specified by idx_positions. Also updates distcenters if present.
        """
        if self.distcenters is not None:
            self.distcenters = self.distcenters[idx_positions]
        for mapname in self.map_types():
            axis = self._get_position_axis(mapname)
            if self.by_environment:
                self[mapname] = [np.take(x, idx_positions, axis=axis) for x in self[mapname]]
            else:
                self[mapname] = np.take(self[mapname], idx_positions, axis=axis)

    def filter_rois(self, idx_rois: np.ndarray) -> None:
        """Filter spike maps to keep only specified ROIs.

        Parameters
        ----------
        idx_rois : np.ndarray
            Indices of ROIs to keep. Must be a 1D array of integers.

        Notes
        -----
        This method modifies the spkmap in-place, keeping only the ROIs
        specified by idx_rois. Only affects spkmap; occmap and speedmap
        are unchanged.
        """
        axis = 0 if self.rois_first else -1
        if self.by_environment:
            self.spkmap = [np.take(x, idx_rois, axis=axis) for x in self.spkmap]
        else:
            self.spkmap = np.take(self.spkmap, idx_rois, axis=axis)

    def filter_environments(self, environments: list[int]) -> None:
        """Filter maps to keep only specified environments.

        Parameters
        ----------
        environments : list of int
            List of environment numbers to keep.

        Raises
        ------
        ValueError
            If by_environment is False, since environments cannot be filtered
            when maps are not separated by environment.

        Notes
        -----
        This method modifies the maps in-place, keeping only the environments
        specified. Only works when by_environment=True.
        """
        if self.by_environment:
            idx_to_requested_env = [i for i, env in enumerate(self.environments) if env in environments]
            self.occmap = [self.occmap[i] for i in idx_to_requested_env]
            self.speedmap = [self.speedmap[i] for i in idx_to_requested_env]
            self.spkmap = [self.spkmap[i] for i in idx_to_requested_env]
            self.environments = [self.environments[i] for i in idx_to_requested_env]
        else:
            raise ValueError("Cannot filter environments when maps aren't separated by environment!")

    def pop_nan_positions(self) -> None:
        """Remove position bins that contain NaN values in any map.

        Notes
        -----
        This method identifies position bins that have NaN values in any of the
        maps (occmap, speedmap, or spkmap) and removes them from all maps.
        Useful for cleaning data before analysis.
        """
        if self.by_environment:
            idx_valid_positions = np.where(~np.any(np.stack([np.any(np.isnan(occmap), axis=0) for occmap in self.occmap], axis=0), axis=0))[0]
        else:
            idx_valid_positions = np.where(~np.any(np.isnan(self.occmap), axis=0))[0]
        self.filter_positions(idx_valid_positions)

    def smooth_maps(self, positions: np.ndarray, kernel_width: float) -> None:
        """Smooth the maps using a Gaussian kernel.

        Parameters
        ----------
        positions : np.ndarray
            Position values corresponding to the position bins. Used to compute
            the Gaussian kernel.
        kernel_width : float
            Width of the Gaussian smoothing kernel in spatial units.

        Notes
        -----
        This method applies Gaussian smoothing to all maps (occmap, speedmap, spkmap).
        NaN values are temporarily replaced with 0 during smoothing, then restored
        afterward. The smoothing is applied along the position dimension.
        """
        kernel = get_gauss_kernel(positions, kernel_width)

        # Replace nans with 0s
        if self.by_environment:
            idxnan = [np.isnan(occmap) for occmap in self.occmap]
        else:
            idxnan = np.isnan(self.occmap)

        if self.rois_first:
            # Move the rois axis to the last axis
            if self.by_environment:
                self.spkmap = [np.moveaxis(map, 0, -1) for map in self.spkmap]
            else:
                self.spkmap = np.moveaxis(self.spkmap, 0, -1)

        for mapname in self.map_types():
            if self.by_environment:
                for ienv, inanenv in enumerate(idxnan):
                    self[mapname][ienv][inanenv] = 0
            else:
                self[mapname][idxnan] = 0

        for mapname in self.map_types():
            # Since we moved ROIs to the last axis position will be axis=1 for all map types
            if self.by_environment:
                self[mapname] = [convolve_toeplitz(map, kernel, axis=1) for map in self[mapname]]
            else:
                self[mapname] = convolve_toeplitz(self[mapname], kernel, axis=1)

        # Put nans back in place
        for mapname in self.map_types():
            if self.by_environment:
                for ienv, inanenv in enumerate(idxnan):
                    self[mapname][ienv][inanenv] = np.nan
            else:
                self[mapname][idxnan] = np.nan

        # Move the rois axis back to the first axis
        if self.rois_first:
            if self.by_environment:
                self.spkmap = [np.moveaxis(map, -1, 0) for map in self.spkmap]
            else:
                self.spkmap = np.moveaxis(self.spkmap, -1, 0)

    def average_trials(self, keepdims: bool = False) -> None:
        """Average the trials within each environment.

        Parameters
        ----------
        keepdims : bool, optional
            Whether to keep the trial dimension with size 1 after averaging.
            Default is False.

        Notes
        -----
        This method computes the mean across trials for each map. After averaging,
        the _averaged flag is set to True to prevent redundant averaging.
        The trial dimension is removed unless keepdims=True.
        """
        if self._averaged:
            return
        for mapname in self.map_types():
            axis = 1 if mapname == "spkmap" and self.rois_first else 0
            if self.by_environment:
                self[mapname] = [ss.mean(map, axis=axis, keepdims=keepdims) for map in self[mapname]]
            else:
                self[mapname] = ss.mean(self[mapname], axis=axis, keepdims=keepdims)
        self._averaged = True

    def nbytes(self) -> int:
        """Calculate the total memory size of all maps in bytes.

        Returns
        -------
        int
            Total number of bytes used by all map arrays.
        """
        num_bytes = 0
        for name in self.map_types():
            if self.by_environment:
                num_bytes += sum(x.nbytes for x in getattr(self, name))
            else:
                num_bytes += getattr(self, name).nbytes
        return num_bytes

    def raw_to_processed(self, positions: np.ndarray, smooth_width: float | None = None) -> "Maps":
        """Convert raw maps to processed maps.

        Processing steps:
        1. Optionally smooth maps with a Gaussian kernel
        2. Divide speedmap and spkmap by occmap (correct_map)
        3. Reorganize spkmap to have ROIs as the first dimension

        Parameters
        ----------
        positions : np.ndarray
            Position values corresponding to the position bins.
        smooth_width : float, optional
            Width of the Gaussian smoothing kernel. If None, no smoothing is applied.
            Default is None.

        Returns
        -------
        Maps
            Self, with maps now in processed format (rois_first=True).

        Notes
        -----
        This method modifies the maps in-place. After processing, spkmap will
        have shape (rois, trials, positions) instead of (trials, positions, rois).
        """
        if smooth_width is not None:
            self.smooth_maps(positions, smooth_width)

        self.speedmap = correct_map(self.occmap, self.speedmap)
        self.spkmap = correct_map(self.occmap, self.spkmap)

        # Change spkmap to be ROIs first
        self.spkmap = np.moveaxis(self.spkmap, -1, 0)
        self.rois_first = True

        return self


@dataclass
class Reliability:
    """Container for reliability values.

    Attributes
    ----------
    values : np.ndarray
        Reliability values for each neuron
    environments : np.ndarray
        Environments for which the reliability was computed
    method : str
        Method used to compute the reliability
    """

    values: np.ndarray
    environments: np.ndarray
    method: str

    def __post_init__(self):
        if self.values.shape[0] != len(self.environments):
            raise ValueError("values and environments must have the same number of environments")

    def __repr__(self) -> str:
        return f"Reliability(num_rois={self.values.shape[1]}, environments={self.environments}, method={self.method})"

    def filter_rois(self, idx_rois: np.ndarray) -> "Reliability":
        """Filter reliability values to keep only specified ROIs.

        Parameters
        ----------
        idx_rois : np.ndarray
            Indices of ROIs to keep. Must be a 1D array of integers.

        Returns
        -------
        Reliability
            New Reliability instance with filtered ROI values.
        """
        return Reliability(self.values[:, idx_rois], self.environments, self.method)

    def filter_environments(self, idx_environments: np.ndarray) -> "Reliability":
        """Filter reliability values to keep only specified environments by index.

        Parameters
        ----------
        idx_environments : np.ndarray
            Indices of environments to keep. Must be a 1D array of integers.

        Returns
        -------
        Reliability
            New Reliability instance with filtered environment values.
        """
        return Reliability(self.values[idx_environments], self.environments[idx_environments], self.method)

    def filter_by_environment(self, environments: list[int]) -> "Reliability":
        """Filter reliability values to keep only specified environments by environment number.

        Parameters
        ----------
        environments : list of int
            List of environment numbers to keep.

        Returns
        -------
        Reliability
            New Reliability instance with filtered environment values.
        """
        idx_to_requested_env = [i for i, env in enumerate(self.environments) if env in environments]
        return Reliability(self.values[idx_to_requested_env], self.environments[idx_to_requested_env], self.method)


@runtime_checkable
class SessionToSpkmapProtocol(Protocol):
    """Protocol defining the required interface for sessions that can be processed into spike maps.

    This protocol specifies the required properties that must be implemented by any
    session class that will be used for spike map processing. Each property provides
    essential data about the recording session, including neural activity, behavioral
    measurements, and session metadata.

    Required Properties
    ------------------
    spks : np.ndarray
        Spike data for all neurons across all timepoints
    spks_type : str
        Type of spike data to load (e.g. "raw", "deconvolved", "corrected", "neuropil", "significant")
    idx_rois : np.ndarray
        An array of shape (num_rois,) containing a boolean mask of which ROIs to load
    timestamps : np.ndarray
        Timestamps for each imaging frame
    env_length : np.ndarray
        Length of the environment for each trial
    positions : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Position-related data including timestamps, positions, trial numbers, and frame indices
    trial_environment : np.ndarray
        Environment for each trial
    num_trials : int
        Total number of trials in the session
    zero_baseline_spks : bool
        Whether spike data is already zero-baselined
    """

    @property
    def spks(self) -> np.ndarray:
        """Spike data for all neurons across all timepoints.

        Returns
        -------
        np.ndarray
            Array of shape (timepoints, neurons) containing spike counts or activity
        """
        ...

    @property
    def spks_type(self) -> str:
        """Type of spike data to load (e.g. "raw", "deconvolved", "corrected", "neuropil", "significant")"""
        ...

    @property
    def idx_rois(self) -> np.ndarray:
        """Indices of the ROIs to load."""
        ...

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps for each imaging frame.

        Returns
        -------
        np.ndarray
            1D array of timestamps in seconds
        """
        ...

    @property
    def env_length(self) -> np.ndarray:
        """Length of the environment for each trial.

        Returns
        -------
        np.ndarray
            Array containing environment length(s) in spatial units
        """
        ...

    @property
    def positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Position-related data arrays.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing:
            - timestamps: Time of each position sample
            - positions: Position values
            - trial_numbers: Trial number for each sample
            - idx_behave_to_frame: Indices mapping behavioral samples to imaging frames
        """
        ...

    @property
    def trial_environment(self) -> np.ndarray:
        """Environment for each trial.

        Returns
        -------
        np.ndarray
            Array of shape (trials,) containing environment number for each trial
        """
        ...

    @property
    def environments(self) -> List[int]:
        """List of environments used in the session"""
        ...

    @property
    def num_trials(self) -> int:
        """Total number of trials in the session.

        Returns
        -------
        int
            Number of trials
        """
        ...

    @property
    def zero_baseline_spks(self) -> bool:
        """Whether spike data is already zero-baselined.

        Returns
        -------
        bool
            True if spike data is zero-baselined, False otherwise
        """
        ...


@dataclass
class SpkmapParams:
    """Parameters for spike map processing.

    Contains configuration settings that control how spike maps are processed,
    including distance steps, speed thresholds, and standardization options.

    Parameters
    ----------
    dist_step : float, default=1
        Step size for distance calculations in spatial units
    speed_threshold : float, default=1.0
        Minimum speed threshold for valid movement periods
    speed_max_allowed : float, default=np.inf
        Maximum speed allowed for valid movement periods (default is no maximum,
        can be useful when behavioral computer allows jumps in position which
        are usually due to hardware issues
    full_trial_flexibility : float | None, default=None
        Flexibility parameter for trial alignment. If None, no flexibility
    standardize_spks : bool, default=True
        Whether to standardize spike counts by dividing by the standard deviation
    smooth_width : float | None, default=1
        Width of the Gaussian smoothing kernel to apply to the maps (width in spatial units)
    reliability_method : str, default="leave_one_out"
        Method to use for calculating reliability
    autosave : bool, default=True
        Whether to save the cache automatically
    """

    dist_step: float = 1.0
    speed_threshold: float = 1.0
    speed_max_allowed: float = np.inf
    full_trial_flexibility: Union[float, None] = 3.0
    standardize_spks: bool = True
    smooth_width: Union[float, None] = 1.0
    reliability_method: str = "leave_one_out"
    autosave: bool = False

    def __repr__(self) -> str:
        class_fields = fields(self)
        lines = []
        for field in class_fields:
            field_name = field.name
            field_value = getattr(self, field_name)
            lines.append(f"{field_name}={repr(field_value)}")

        class_name = self.__class__.__name__
        joined_lines = ",\n    ".join(lines)
        return f"{class_name}(\n    {joined_lines}\n)"

    @classmethod
    def from_dict(cls, params_dict: dict) -> "SpkmapParams":
        """Create a SpkmapParams instance from a dictionary.

        Parameters
        ----------
        params_dict : dict
            Dictionary of parameter names and values. Missing parameters will
            use default values from SpkmapParams.

        Returns
        -------
        SpkmapParams
            New SpkmapParams instance with values from the dictionary.
        """
        return cls(**{k: params_dict[k] for k in params_dict})

    @classmethod
    def from_path(cls, path: Path) -> "SpkmapParams":
        """Create a SpkmapParams instance from a JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON file containing parameter values.

        Returns
        -------
        SpkmapParams
            New SpkmapParams instance loaded from the JSON file.
        """
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def compare(self, other: "SpkmapParams", filter_keys: Optional[List[str]] = None) -> bool:
        """Compare two SpkmapParams instances.

        Parameters
        ----------
        other : SpkmapParams
            Another SpkmapParams instance to compare against.
        filter_keys : list of str, optional
            If provided, only compare the specified parameter keys.
            If None, compare all parameters. Default is None.

        Returns
        -------
        bool
            True if the parameters match (or specified keys match), False otherwise.
        """
        if filter_keys is None:
            return self == other
        else:
            return all(getattr(self, key) == getattr(other, key) for key in filter_keys)

    def save(self, path: Path) -> None:
        """Save the parameters to a JSON file.

        Parameters
        ----------
        path : Path
            Path where the JSON file will be saved.
        """
        with open(path, "w") as f:
            json.dump(asdict(self), f, sort_keys=True)

    def __post_init__(self):
        if self.dist_step <= 0:
            raise ValueError("dist_step must be positive")
        if self.speed_threshold <= 0:
            raise ValueError("speed_threshold must be positive")
        if self.full_trial_flexibility is not None and self.full_trial_flexibility < 0:
            raise ValueError("If used, full_trial_flexibility must be nonnegative (can also be None)")
        if self.smooth_width is not None and self.smooth_width <= 0:
            raise ValueError("smooth_width must be positive (can also be None)")
        # Convert floats to floats when not None
        self.dist_step = float(self.dist_step)
        self.speed_threshold = float(self.speed_threshold)
        self.speed_max_allowed = float(self.speed_max_allowed)
        self.full_trial_flexibility = float(self.full_trial_flexibility) if self.full_trial_flexibility is not None else None
        self.smooth_width = float(self.smooth_width) if self.smooth_width is not None else None


def get_spkmap_params(param_type: str, updates: Optional[dict] = None) -> SpkmapParams:
    """Get the parameters for the SpkmapProcessor

    Parameters
    ----------
    param_type : str
        The type of parameters to get
    updates : dict, optional
        A dictionary of parameters to update

    Returns
    -------
    params : SpkmapParams
        The parameters for the SpkmapProcessor
    """
    if param_type == "default":
        params = SpkmapParams(
            dist_step=1.0,
            speed_threshold=1.0,
            speed_max_allowed=np.inf,
            full_trial_flexibility=3.0,
            standardize_spks=True,
            smooth_width=1.0,
            reliability_method="leave_one_out",
            autosave=False,
        )
    elif param_type == "smoothed":
        params = SpkmapParams(
            dist_step=1.0,
            speed_threshold=1.0,
            speed_max_allowed=np.inf,
            full_trial_flexibility=3.0,
            standardize_spks=True,
            smooth_width=5.0,
            reliability_method="leave_one_out",
            autosave=False,
        )
    else:
        raise ValueError(f"Invalid param_type: {param_type}")

    if updates is not None and isinstance(updates, dict):
        invalid_updates = set(updates.keys()) - set(params.__dict__.keys())
        if invalid_updates:
            raise ValueError(f"Invalid updates: {invalid_updates}")
        for k, v in updates.items():
            setattr(params, k, v)

    return params


def manage_one_cache(func):
    """Decorator to manage the onefile cache for the SpkmapProcessor

    Used on bound methods of SpkmapProcessor. When clear_one_cache is True, this
    decorator will detect which onefiles have been cached by the decorated
    method and clear them from the cache after running. This is useful for
    large batch processing where memory might be a concern.
    """

    @wraps(func)
    def wrapper(self: "SpkmapProcessor", *args, **kwargs):
        clear_one_cache = kwargs.get("clear_one_cache", False)
        if clear_one_cache:
            previous_cache = set(self.session.one_cache)
        output = func(self, *args, **kwargs)
        if clear_one_cache:
            current_cache = set(self.session.one_cache)
            new_cache = current_cache - previous_cache
            self.session.clear_cache(list(new_cache))
        return output

    return wrapper


def with_temp_params(func):
    """Decorator that temporarily modifies SpkmapProcessor parameters during method execution.

    If a params argument is provided to the decorated method, it will:
    1. Store the original parameters
    2. Update parameters with the provided ones (allowing partial updates)
    3. Execute the method
    4. Restore the original parameters
    """

    @wraps(func)
    def wrapper(self: "SpkmapProcessor", *args, **kwargs):
        # Extract params from kwargs if present
        temp_params = kwargs.get("params", None)
        if temp_params is None:
            # Use original params if no temp params are provided
            return func(self, *args, **kwargs)
        else:
            # If temp params are provided, first make a copy of the original params
            original_params = copy(self.params)
            if isinstance(temp_params, dict):
                # If temp params are a dictionary, update the original params with the temp params (allowing partial updates)
                use_params = copy(original_params)
                for k, v in temp_params.items():
                    setattr(use_params, k, v)
            elif isinstance(temp_params, SpkmapParams):
                # If temp params are a SpkmapParams instance, use the temp params directly
                use_params = temp_params
            else:
                raise ValueError(f"params must be a SpkmapParams instance or a dictionary, not {type(temp_params)}")

            try:
                # Use the temporary params
                self.params = use_params
                result = func(self, *args, **kwargs)
            finally:
                # Restore the original params
                self.params = original_params
            return result

    return wrapper


def cached_processor(data_type: str, disable: bool = False):
    """
    Decorator for methods that can be cached based on parameters.

    Parameters
    ----------
    data_type : str
        Type of data being processed (e.g., 'raw_maps', 'processed_maps')
    disable : bool, optional
        If True, the cache will not be used. Default is False.
    """

    def maps_loader(self: "SpkmapProcessor", process_method: callable, *args, **kwargs):
        # Attempt to load from cache
        if not disable and not kwargs.get("force_recompute", False):
            cached_data, valid_cache = self.load_from_cache(data_type)
            if valid_cache:
                return cached_data

        # If forcing a recompute or not valid, process data with the pipeline method
        result = process_method(self, *args, **kwargs)

        # And cache if requested (only if all maps are loaded)
        if not disable and self.params.autosave:
            self.save_cache(data_type, result)

        return result

    def env_maps_loader(self: "SpkmapProcessor", process_method: callable, *args, **kwargs):
        # Attempt to load from cache
        if not disable and not kwargs.get("force_recompute", False):
            cached_data, valid_cache = self.load_from_cache(data_type)
            if valid_cache:
                if kwargs.get("use_session_filters", True):
                    # If we're using session filters, we need to filter ROIs
                    cached_data.filter_rois(np.where(self.session.idx_rois)[0])
                return cached_data

        # If forcing a recompute or not valid, process data with the pipeline method
        result = process_method(self, *args, **kwargs)

        # And cache if requested
        if not disable and self.params.autosave:
            # Don't save the cache if we're using session filters!!!
            if not kwargs.get("use_session_filters", True):
                self.save_cache(data_type, result)

        return result

    def reliability_loader(self: "SpkmapProcessor", process_method: callable, *args, **kwargs):
        if not disable and not kwargs.get("force_recompute", False):
            cached_data, valid_cache = self.load_from_cache(data_type)
            if valid_cache:
                if kwargs.get("use_session_filters", True):
                    # If we're using session filters, we need to filter ROIs
                    cached_data.values = cached_data.values[:, self.session.idx_rois]
                return cached_data

        # If forcing a recompute or not valid, process data with the pipeline method
        result = process_method(self, *args, **kwargs)

        # And cache if requested
        if not disable and self.params.autosave:
            if not kwargs.get("use_session_filters", True):
                self.save_cache("reliability", result)

        return result

    def decorator(process_method):
        """
        Reminder to myself: The process_method is the method that is decorated!
        It is opaque like this because we need to pass arguments to the decorator (data_type & disable).
        So process_method will be "get_raw_maps" or whatever else below.
        """

        @wraps(process_method)
        def wrapper(self: "SpkmapProcessor", *args, **kwargs):
            if data_type == "raw_maps" or data_type == "processed_maps":
                return maps_loader(self, process_method, *args, **kwargs)
            elif data_type == "env_maps":
                return env_maps_loader(self, process_method, *args, **kwargs)
            elif data_type == "reliability":
                return reliability_loader(self, process_method, *args, **kwargs)
            else:
                raise ValueError(f"Invalid data type: {data_type}, doesn't have a cache loader yet!")

        return wrapper

    return decorator


@dataclass
class SpkmapProcessor:
    """Class for processing and caching spike maps from session data

    NOTES ON ENGINEERING:
    I want the variables required for processing spkmaps to be properties (@property)
    that have hidden attributes for caching. Therefore, we can use the property method
    to get the attribute and each property method can do whatever processing is needed
    for that attribute. (Uh, duh). Time to get modern. lol.

    Right now I've almost got the register_spkmaps method working again (not tested yet)
    but now is when the dataclass refactoring comes in.
    1. Make it possible to separate the occmap from the spkmap loading.
       - do so by making the preliminary variables properties with caching
    2. Consider how to implement smoothing then correctMap functionality -- it should
       be possible to do this in a way that allows me to iteratively try different
       parameterizations without having to go through the whole pipeline again.
    3. Consider how / when to implement reliability measures. In PCSS, they're done all
       right there with get_spkmaps. But it's probably not always necessary and can
       actually take a bit of time? It would also be nice to save reliability scores for
       the neurons... but then we'd also need an independent params saving system for them.
    4. Re: the point above, I wonder if the one.data loading system is ideal or if I should
       use a more explicit and dedicated SpkmapProcessor saving / loading system.
    """

    session: Union[SessionData, B2Session, SessionToSpkmapProtocol]
    params: SpkmapParams = field(default_factory=SpkmapParams, repr=False)
    data_cache: dict = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        # Check if the session provided is compatible with SpkmapProcessing
        if not isinstance(self.session, SessionData):
            raise ValueError(f"session must be a SessionData instance, not {type(self.session)}")
        # (Don't check if it's a SessionToSpkmapProtocol because hasattr() will call properties which loads data...)

        # We need to handle the case where params is a dictionary of partial updates to the default params
        self.params = helpers.resolve_dataclass(self.params, SpkmapParams)

    def cached_dependencies(self, data_type: str) -> List[str]:
        """Get the parameter dependencies for a given data type.

        Parameters
        ----------
        data_type : str
            Type of cached data ("raw_maps", "processed_maps", "env_maps", or "reliability").

        Returns
        -------
        list of str
            List of parameter names that affect the cache validity for this data type.
        """
        if data_type == "raw_maps":
            return ["dist_step", "speed_threshold", "speed_max_allowed", "standardize_spks"]
        elif data_type == "processed_maps":
            return ["dist_step", "speed_threshold", "speed_max_allowed", "standardize_spks", "smooth_width"]
        elif data_type == "env_maps":
            return ["dist_step", "speed_threshold", "speed_max_allowed", "standardize_spks", "smooth_width", "full_trial_flexibility"]
        elif data_type == "reliability":
            return [
                "dist_step",
                "speed_threshold",
                "speed_max_allowed",
                "standardize_spks",
                "smooth_width",
                "full_trial_flexibility",
                "reliability_method",
            ]
        # Otherwise just return all params
        return list(self.params.__dict__.keys())

    def show_cache(self, data_type: Optional[str] = None) -> None:
        """Helper function that scrapes the cache directory and shows cached files

        Parameters
        ----------
        data_type: Optional[str] = None
            Indicate a data type to filter which parts of the cache to show

        Notes
        -----
        Prints a formatted table showing cache information including data_type, size,
        parameters, and modification date. If no cache directory exists, prints a message.
        """
        import os
        from datetime import datetime

        # Get the base cache directory
        base_cache_dir = self.cache_directory()

        if not base_cache_dir.exists():
            print(f"No cache directory found at: {base_cache_dir}")
            return

        # Collect information about all cache files
        cache_info = []

        # Define the data types to check
        if data_type is not None:
            data_types_to_check = [data_type]
        else:
            data_types_to_check = ["raw_maps", "processed_maps", "env_maps", "reliability"]

        for dt in data_types_to_check:
            cache_dir = self.cache_directory(dt)
            if not cache_dir.exists():
                continue

            # Find all parameter files (they define what caches exist)
            param_files = list(cache_dir.glob("params_*.npz"))

            for param_file in param_files:
                # Extract the hash from the filename
                params_hash = param_file.stem.replace("params_", "")

                # Load the parameters
                try:
                    cached_params = dict(np.load(param_file))
                    param_str = ", ".join([f"{k}={v}" for k, v in cached_params.items()])
                except Exception as e:
                    param_str = f"Error loading params: {e}"

                # Get file modification time
                mod_time = datetime.fromtimestamp(param_file.stat().st_mtime)
                date_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")

                # Calculate total size of all related cache files
                total_size = param_file.stat().st_size

                if dt in ["raw_maps", "processed_maps"]:
                    # For maps, look for data files for each map type
                    for mapname in ["occmap", "speedmap", "spkmap"]:
                        data_file = cache_dir / f"data_{mapname}_{params_hash}.npy"
                        if data_file.exists():
                            total_size += data_file.stat().st_size

                elif dt == "env_maps":
                    # For env_maps, look for environment file and individual environment data files
                    env_file = cache_dir / f"data_environments_{params_hash}.npy"
                    if env_file.exists():
                        total_size += env_file.stat().st_size
                        # Load environments to find all data files
                        try:
                            environments = np.load(env_file)
                            for env in environments:
                                for mapname in ["occmap", "speedmap", "spkmap"]:
                                    data_file = cache_dir / f"data_{mapname}_{env}_{params_hash}.npy"
                                    if data_file.exists():
                                        total_size += data_file.stat().st_size
                        except Exception:
                            pass  # Continue even if we can't load environments

                elif dt == "reliability":
                    # For reliability, look for environments and reliability data files
                    env_file = cache_dir / f"data_environments_{params_hash}.npy"
                    rel_file = cache_dir / f"data_reliability_{params_hash}.npy"
                    if env_file.exists():
                        total_size += env_file.stat().st_size
                    if rel_file.exists():
                        total_size += rel_file.stat().st_size

                # Convert size to human readable format
                size_str = self._format_file_size(total_size)

                cache_info.append(
                    {
                        "data_type": dt,
                        "size": size_str,
                        "parameters": param_str,
                        "date": date_str,
                        "hash": params_hash[:8],  # Show first 8 chars of hash
                    }
                )

        if not cache_info:
            print("No cache files found.")
            return

        # Format the output as a table
        output_lines = []
        output_lines.append("Cache Files Summary")
        output_lines.append("=" * 80)
        output_lines.append(f"{'Data Type':<15} {'Size':<10} {'Date':<20} {'Hash':<10} {'Parameters'}")
        output_lines.append("-" * 80)

        for info in cache_info:
            output_lines.append(f"{info['data_type']:<15} {info['size']:<10} {info['date']:<20} " f"{info['hash']:<10} {info['parameters']}")

        output_lines.append("-" * 80)
        output_lines.append(f"Total cache entries: {len(cache_info)}")

        result = "\n".join(output_lines)
        print(result)

    def _format_file_size(self, size_bytes: int) -> str:
        """Convert bytes to human-readable format.

        Parameters
        ----------
        size_bytes : int
            Size in bytes.

        Returns
        -------
        str
            Human-readable size string (e.g., "1.5 MB").
        """
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math

        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def cache_directory(self, data_type: Optional[str] = None) -> Path:
        """Get the cache directory path for a given data type.

        Parameters
        ----------
        data_type : str, optional
            Type of cached data. If None, returns the base cache directory.
            Default is None.

        Returns
        -------
        Path
            Path to the cache directory for the specified data type.
        """
        if data_type is None:
            return self.session.data_path / "spkmaps"
        else:
            folder_name = f"{data_type}_{self.session.spks_type}"
            return self.session.data_path / "spkmaps" / folder_name

    def dependent_params(self, data_type: str) -> dict:
        """Get the dependent parameters for a given data type as a dictionary.

        Parameters
        ----------
        data_type : str
            Type of cached data.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their values for the given data type.
        """
        return {k: getattr(self.params, k) for k in self.cached_dependencies(data_type)}

    def _params_hash(self, data_type: str) -> str:
        """Get the hash of the dependent parameters for a given data type.

        Parameters
        ----------
        data_type : str
            Type of cached data.

        Returns
        -------
        str
            SHA256 hash of the dependent parameters (as hexadecimal string).
        """
        return hashlib.sha256(json.dumps(self.dependent_params(data_type), sort_keys=True).encode()).hexdigest()

    def save_cache(self, data_type: str, data: Union[Maps, Reliability]) -> None:
        """Save the cached parameters and data for a given data type.

        Parameters
        ----------
        data_type : str
            Type of data being cached ("raw_maps", "processed_maps", "env_maps", or "reliability").
        data : Maps or Reliability
            The data object to cache.

        Notes
        -----
        Creates the cache directory if it doesn't exist. Saves parameters as an NPZ file
        and data as NPY files, using a hash of the parameters in the filenames.
        """
        cache_dir = self.cache_directory(data_type)
        params_hash = self._params_hash(data_type)
        cache_param_path = cache_dir / f"params_{params_hash}.npz"
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_param_path, **self.dependent_params(data_type))
        if data_type == "raw_maps" or data_type == "processed_maps":
            for mapname in Maps.map_types():
                cache_data_path = cache_dir / f"data_{mapname}_{params_hash}.npy"
                np.save(cache_data_path, getattr(data, mapname))
        elif data_type == "env_maps":
            environments = data.environments
            np.save(cache_dir / f"data_environments_{params_hash}.npy", environments)
            for ienv, env in enumerate(environments):
                for mapname in Maps.map_types():
                    cache_data_path = cache_dir / f"data_{mapname}_{env}_{params_hash}.npy"
                    np.save(cache_data_path, getattr(data, mapname)[ienv])
        elif data_type == "reliability":
            values = data.values
            environments = data.environments
            # don't need data.method because it's in params...
            np.save(cache_dir / f"data_environments_{params_hash}.npy", environments)
            np.save(cache_dir / f"data_reliability_{params_hash}.npy", values)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def load_from_cache(self, data_type: str) -> Tuple[Union[Maps, Reliability, None], bool]:
        """Load cached parameters and data for a given data type.

        Parameters
        ----------
        data_type : str
            Type of cached data to load.

        Returns
        -------
        tuple
            A tuple containing:
            - The cached data (Maps or Reliability), or None if not found
            - A boolean indicating whether valid cache was found
        """
        cache_dir = self.cache_directory(data_type)
        if cache_dir.exists():
            # If the directory exists, check if there are any cached params that match the expected hash
            params_hash = self._params_hash(data_type)
            cached_params_path = cache_dir / f"params_{params_hash}.npz"
            if cached_params_path.exists():
                cached_params = dict(np.load(cached_params_path))
                # Check if the cached params match the dependent params
                if self.check_params_match(cached_params):
                    return self._load_from_cache(data_type, params_hash, params=cached_params), True
        return None, False

    def check_params_match(self, cached_params: dict) -> bool:
        """Check if the cached params and the current params are the same.

        Parameters
        ----------
        cached_params : dict
            The cached params to check against the current params

        Returns
        -------
        bool
            True if the cached params are nonempty and match the current params, False otherwise
        """
        return cached_params and all(cached_params[k] == getattr(self.params, k) for k in cached_params)

    def _load_from_cache(self, data_type: str, params_hash: str, params: Optional[Dict[str, Any]] | None = None) -> Union[Maps, Reliability]:
        """Load cached data from disk using a parameter hash.

        Parameters
        ----------
        data_type : str
            Type of cached data to load.
        params_hash : str
            Hash string identifying the cached parameters.
        params : dict, optional
            Dictionary of cached parameters. Used for reliability method.
            Default is None.

        Returns
        -------
        Maps or Reliability
            The loaded cached data object.

        Raises
        ------
        ValueError
            If data_type is not recognized.
        """
        cache_dir = self.cache_directory(data_type)
        if data_type == "raw_maps" or data_type == "processed_maps":
            cached_data = {}
            for name in Maps.map_types():
                cached_data[name] = np.load(cache_dir / f"data_{name}_{params_hash}.npy", mmap_mode="r")
            if data_type == "raw_maps":
                return Maps.create_raw_maps(**cached_data)
            elif data_type == "processed_maps":
                return Maps.create_processed_maps(**cached_data)
        elif data_type == "env_maps":
            environments = np.load(cache_dir / f"data_environments_{params_hash}.npy")
            cached_data = dict(environments=environments)
            for name in Maps.map_types():
                cached_data[name] = []
                for env in environments:
                    cached_data[name].append(np.load(cache_dir / f"data_{name}_{env}_{params_hash}.npy", mmap_mode="r"))
            return Maps.create_environment_maps(**cached_data)
        elif data_type == "reliability":
            environments = np.load(cache_dir / f"data_environments_{params_hash}.npy")
            values = np.load(cache_dir / f"data_reliability_{params_hash}.npy")
            method = params["reliability_method"]
            return Reliability(values, environments, method)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    @manage_one_cache
    def _filter_environments(
        self,
        envnum: Union[int, Iterable[int], None] = None,
        clear_one_cache: bool = True,
    ) -> np.ndarray:
        """Filter the session data to only include trials from certain environments.

        Parameters
        ----------
        envnum : int, iterable of int, or None, optional
            Environment number(s) to filter. If None, returns all trials.
            Default is None.
        clear_one_cache : bool, optional
            Whether to clear the onefile cache after filtering. Default is True.

        Returns
        -------
        np.ndarray
            Boolean array indicating which trials belong to the specified environment(s).

        Notes
        -----
        This assumes that the trials are in order. We might want to use the third
        output of session.positions to get the "real" trial numbers which aren't
        always contiguous and 0 indexed.
        """
        if envnum is None:
            envnum = self.session.environments
        envnum = helpers.check_iterable(envnum)
        return np.isin(self.session.trial_environment, envnum)

    @property
    def dist_edges(self) -> np.ndarray:
        """Distance edges for the position bins.

        Returns
        -------
        np.ndarray
            1D array of position bin edges. Shape is (num_positions + 1,).

        Raises
        ------
        ValueError
            If not all trials have the same environment length.

        Notes
        -----
        The number of position bins is determined by dividing the environment
        length by dist_step. This property caches the environment length
        internally after first access.
        """
        if not hasattr(self, "_env_length"):
            env_length = self.session.env_length
            if hasattr(env_length, "__len__"):
                if np.unique(env_length).size != 1:
                    msg = "SpkmapProcessor (currently) requires all trials to have the same env length!"
                    raise ValueError(msg)
                env_length = env_length[0]
            self._env_length = env_length

        num_positions = int(self._env_length / self.params.dist_step)
        return np.linspace(0, self._env_length, num_positions + 1)

    @property
    def dist_centers(self) -> np.ndarray:
        """Distance centers for the position bins.

        Returns
        -------
        np.ndarray
            1D array of position bin centers. Shape is (num_positions,).
        """
        return helpers.edge2center(self.dist_edges)

    @manage_one_cache
    def _idx_required_position_bins(self, clear_one_cache: bool = True) -> np.ndarray:
        """Get the indices of the position bins that are required for a full trial

        Parameters
        ----------
        clear_one_cache : bool, default=False
            Whether to clear the onefile cache after getting the indices

        Returns
        -------
        np.ndarray
            The indices of the position bins that are required for a trial to be considered full
        """
        num_position_bins = len(self.dist_centers)
        if self.params.full_trial_flexibility is None:
            idx_to_required_bins = np.arange(num_position_bins)
        else:
            start_idx = np.where(self.dist_edges >= self.params.full_trial_flexibility)[0][0]
            end_idx = np.where(self.dist_edges <= self.dist_edges[-1] - self.params.full_trial_flexibility)[0][-1]
            idx_to_required_bins = np.arange(start_idx, end_idx)
        return idx_to_required_bins

    @with_temp_params
    @manage_one_cache
    @cached_processor("raw_maps", disable=False)
    def get_raw_maps(
        self,
        force_recompute: bool = False,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ) -> Maps:
        """Get raw maps (occupancy, speed, spkmap) from session data.

        This method processes session data to create spatial maps representing
        occupancy, speed, and neural activity across position bins. The maps
        are in raw format (not smoothed or normalized by occupancy).

        Parameters
        ----------
        force_recompute : bool, optional
            Whether to force recomputation even if cached data exists. Default is False.
        clear_one_cache : bool, optional
            Whether to clear the onefile cache after processing. Default is True.
        params : SpkmapParams, dict, or None, optional
            Parameters for processing. If None, uses instance parameters.
            If a dict, updates instance parameters temporarily.
            Parameters are restored after method execution. Default is None.

        Returns
        -------
        Maps
            Maps instance containing raw occupancy, speed, and spike maps.
            Shape: (trials, positions) for occmap/speedmap,
            (trials, positions, rois) for spkmap.

        Notes
        -----
        The method:
        1. Bins positions according to dist_step
        2. Filters by speed threshold
        3. Computes occupancy, speed, and spike maps
        4. Sets unvisited position bins to NaN
        5. Optionally standardizes spike data

        Results are cached based on parameter hash for efficient reuse.
        """
        dist_edges = self.dist_edges
        dist_centers = self.dist_centers
        num_positions = len(dist_centers)

        # Get behavioral timestamps and positions
        timestamps, positions, trial_numbers, idx_behave_to_frame = self.session.positions

        # compute behavioral speed on each sample
        within_trial_sample = np.append(np.diff(trial_numbers) == 0, True)
        sample_duration = np.append(np.diff(timestamps), 0)
        speeds = np.append(np.diff(positions) / sample_duration[:-1], 0)
        # do this after division so no /0 errors
        sample_duration = sample_duration * within_trial_sample
        # speed 0 in last sample for each trial (it's undefined)
        speeds = speeds * within_trial_sample
        # Convert positions to position bins
        position_bin = np.digitize(positions, dist_edges) - 1

        # get imaging information
        frame_time_stamps = self.session.timestamps
        sampling_period = np.median(np.diff(frame_time_stamps))
        dist_cutoff = sampling_period / 2
        delay_position_to_imaging = frame_time_stamps[idx_behave_to_frame] - timestamps

        # get spiking information
        spks = self.session.spks
        num_rois = self.session.get_value("numROIs")

        # Do standardization
        if self.params.standardize_spks:
            spks = median_zscore(spks, median_subtract=not self.session.zero_baseline_spks)

        # Get high resolution occupancy and speed maps
        dtype = np.float32
        occmap = np.zeros((self.session.num_trials, num_positions), dtype=dtype)
        counts = np.zeros((self.session.num_trials, num_positions), dtype=dtype)
        speedmap = np.zeros((self.session.num_trials, num_positions), dtype=dtype)
        spkmap = np.zeros((self.session.num_trials, num_positions, num_rois), dtype=dtype)
        extra_counts = np.zeros((self.session.num_trials, num_positions), dtype=dtype)

        # Get maps -- doing this independently for each map allows for more
        # flexibility in which data to load (basically the occmap & speedmap
        # are instantaneous, but the spkmap is a bit slower)
        get_summation_map(
            sample_duration,
            trial_numbers,
            position_bin,
            occmap,
            counts,
            speeds,
            self.params.speed_threshold,
            self.params.speed_max_allowed,
            delay_position_to_imaging,
            dist_cutoff,
            sample_duration,
            scale_by_sample_duration=False,
            use_sample_to_value_idx=False,
            sample_to_value_idx=idx_behave_to_frame,
        )
        get_summation_map(
            speeds,
            trial_numbers,
            position_bin,
            speedmap,
            counts,
            speeds,
            self.params.speed_threshold,
            self.params.speed_max_allowed,
            delay_position_to_imaging,
            dist_cutoff,
            sample_duration,
            scale_by_sample_duration=True,
            use_sample_to_value_idx=False,
            sample_to_value_idx=idx_behave_to_frame,
        )
        get_summation_map(
            spks,
            trial_numbers,
            position_bin,
            spkmap,
            extra_counts,
            speeds,
            self.params.speed_threshold,
            self.params.speed_max_allowed,
            delay_position_to_imaging,
            dist_cutoff,
            sample_duration,
            scale_by_sample_duration=True,
            use_sample_to_value_idx=True,
            sample_to_value_idx=idx_behave_to_frame,
        )

        # Figure out the valid range (outside of this range, set the maps to nan, because their values are not meaningful)
        position_bin_per_trial = [position_bin[trial_numbers == tnum] for tnum in range(self.session.num_trials)]

        # offsetting by 1 because there is a bug in the vrControl software where the first sample is always set
        # to the minimum position (which is 0), but if there is a built-up buffer in the rotary encoder, the position
        # will jump at the second sample. In general this will always work unless the mice have a truly ridiculous
        # speed at the beginning of the trial...
        first_valid_bin = [np.min(bpb[1:] if len(bpb) > 1 else bpb) for bpb in position_bin_per_trial]
        last_valid_bin = [np.max(bpb) for bpb in position_bin_per_trial]

        # set bins to nan when mouse didn't visit them
        occmap = replace_missing_data(occmap, first_valid_bin, last_valid_bin)
        speedmap = replace_missing_data(speedmap, first_valid_bin, last_valid_bin)
        spkmap = replace_missing_data(spkmap, first_valid_bin, last_valid_bin)

        return Maps.create_raw_maps(occmap, speedmap, spkmap)

    @with_temp_params
    @manage_one_cache
    @cached_processor("processed_maps", disable=False)
    def get_processed_maps(
        self,
        force_recompute: bool = False,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ) -> Maps:
        """Get processed maps (smoothed and normalized by occupancy).

        This method creates processed maps by:
        1. Getting raw maps
        2. Optionally smoothing with a Gaussian kernel
        3. Normalizing speedmap and spkmap by occupancy
        4. Reorganizing spkmap to have ROIs as the first dimension

        Parameters
        ----------
        force_recompute : bool, optional
            Whether to force recomputation even if cached data exists. Default is False.
        clear_one_cache : bool, optional
            Whether to clear the onefile cache after processing. Default is True.
        params : SpkmapParams, dict, or None, optional
            Parameters for processing. If None, uses instance parameters.
            If a dict, updates instance parameters temporarily.
            Parameters are restored after method execution. Default is None.

        Returns
        -------
        Maps
            Maps instance containing processed occupancy, speed, and spike maps.
            Shape: (trials, positions) for occmap/speedmap,
            (rois, trials, positions) for spkmap.
        """
        # Get the raw maps first (don't need to specify params because they're already set by this method)
        maps = self.get_raw_maps(
            force_recompute=force_recompute,
            clear_one_cache=clear_one_cache,
        )

        # Process the maps (smooth, divide by occupancy, and change to ROIs first)
        return maps.raw_to_processed(self.dist_centers, self.params.smooth_width)

    @with_temp_params
    @manage_one_cache
    @cached_processor("env_maps", disable=False)
    def get_env_maps(
        self,
        use_session_filters: bool = True,
        force_recompute: bool = False,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ) -> Maps:
        """Get processed maps separated by environment.

        This method creates environment-separated maps by:
        1. Getting processed maps
        2. Filtering to include only full trials (based on full_trial_flexibility)
        3. Filtering ROIs if use_session_filters=True
        4. Grouping maps by environment

        Parameters
        ----------
        use_session_filters : bool, optional
            Whether to filter ROIs using session.idx_rois. Default is True.
        force_recompute : bool, optional
            Whether to force recomputation even if cached data exists. Default is False.
        clear_one_cache : bool, optional
            Whether to clear the onefile cache after processing. Default is True.
        params : SpkmapParams, dict, or None, optional
            Parameters for processing. If None, uses instance parameters.
            If a dict, updates instance parameters temporarily.
            Parameters are restored after method execution. Default is None.

        Returns
        -------
        Maps
            Maps instance with by_environment=True, containing lists of maps
            for each environment. Shape per environment:
            (trials_in_env, positions) for occmap/speedmap,
            (rois, trials_in_env, positions) for spkmap.
        """
        # Make sure it's an iterable -- the output will always be a list
        envnum = helpers.check_iterable(self.session.environments)

        # Get the indices of the trials to each environment
        idx_each_environment = [self._filter_environments(env) for env in envnum]

        # Then get the indices of the position bins that are required for a full trial
        idx_required_position_bins = self._idx_required_position_bins(clear_one_cache)

        # Get the processed maps (don't need to specify params because they're already set by the decorator)
        maps = self.get_processed_maps(
            force_recompute=force_recompute,
            clear_one_cache=clear_one_cache,
        )

        # Add the list of environments to the maps
        maps.environments = envnum

        # Make a list of the maps we are processing
        maps_to_process = Maps.map_types()

        # Filter the maps to only include the ROIs we want
        if use_session_filters:
            idx_rois = np.where(self.session.idx_rois)[0]
        else:
            idx_rois = np.arange(self.session.get_value("numROIs"), dtype=int)

        # Filter the maps to only include the full trials
        full_trials = np.where(np.all(~np.isnan(maps.occmap[:, idx_required_position_bins]), axis=1))[0]

        # Implement trial & ROI filtering here
        for mapname in maps_to_process:
            if mapname == "spkmap":
                maps[mapname] = np.take(np.take(maps[mapname], idx_rois, axis=0), full_trials, axis=1)
            else:
                maps[mapname] = np.take(maps[mapname], full_trials, axis=0)

        # Filter the trial indices to only include full trials
        idx_each_environment = [np.where(np.take(idx, full_trials, axis=0))[0] for idx in idx_each_environment]

        # Then group each one by environment
        # -> this is now (trials_in_env, position_bins, ...(roi if spkmap)...)
        maps.by_environment = True
        for mapname in maps_to_process:
            if mapname == "spkmap":
                maps[mapname] = [np.take(maps[mapname], idx, axis=1) for idx in idx_each_environment]
            else:
                maps[mapname] = [np.take(maps[mapname], idx, axis=0) for idx in idx_each_environment]

        return maps

    @with_temp_params
    @manage_one_cache
    @cached_processor("reliability", disable=False)
    def get_reliability(
        self,
        use_session_filters: bool = True,
        force_recompute: bool = False,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ) -> Reliability:
        """Calculate reliability of spike maps across trials.

        Reliability measures how consistent neural activity is across trials
        within each environment. Multiple methods are supported.

        Parameters
        ----------
        use_session_filters : bool, optional
            Whether to filter ROIs using session.idx_rois. Default is True.
        force_recompute : bool, optional
            Whether to force recomputation even if cached data exists. Default is False.
        clear_one_cache : bool, optional
            Whether to clear the onefile cache after processing. Default is True.
        params : SpkmapParams, dict, or None, optional
            Parameters for processing. If None, uses instance parameters.
            If a dict, updates instance parameters temporarily.
            Parameters are restored after method execution. Default is None.

        Returns
        -------
        Reliability
            Reliability instance containing reliability values for each ROI
            in each environment. Shape: (num_environments, num_rois).

        Notes
        -----
        Supported reliability methods:
        - "leave_one_out": Leave-one-out cross-validation
        - "correlation": Correlation between trial pairs
        - "mse": Mean squared error between trial pairs

        All reliability measures require maps with no NaN positions.
        """
        envnum = helpers.check_iterable(self.session.environments)

        # A list of the requested environments (all if not specified)
        maps = self.get_env_maps(
            use_session_filters=use_session_filters,
            force_recompute=force_recompute,
            clear_one_cache=clear_one_cache,
            params={"autosave": False},  # Prevent saving in the case of a recompute
        )

        # All reliability measures require no NaNs
        maps.pop_nan_positions()

        if self.params.reliability_method == "leave_one_out":
            rel_values = [helpers.reliability_loo(spkmap) for spkmap in maps.spkmap]
        elif self.params.reliability_method == "correlation" or self.params.reliability_method == "mse":
            rel_mse, rel_cor = helpers.named_transpose([helpers.measureReliability(spkmap) for spkmap in maps.spkmap])
            rel_values = rel_mse if self.params.reliability_method == "mse" else rel_cor
        else:
            raise ValueError(f"Method {self.params.reliability_method} not supported")

        return Reliability(
            np.stack(rel_values),
            environments=envnum,
            method=self.params.reliability_method,
        )

    # ------------------- convert between imaging and behavioral time -------------------
    @with_temp_params
    @manage_one_cache
    def get_frame_behavior(
        self,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get position and environment data for each imaging frame.

        This method aligns behavioral data (position, speed, environment, trial)
        to imaging frame timestamps. Returns NaN for frames where no position
        data is available (e.g., if the closest behavioral sample is further
        away in time than half the sampling period).

        Parameters
        ----------
        clear_one_cache : bool, optional
            Whether to clear the onefile cache after processing. Default is True.
        params : SpkmapParams, dict, or None, optional
            Parameters for processing. If None, uses instance parameters.
            If a dict, updates instance parameters temporarily.
            Parameters are restored after method execution. Default is None.

        Returns
        -------
        tuple
            A tuple containing four arrays (all with shape (num_frames,)):
            - frame_position: Position for each frame (NaN if unavailable)
            - frame_speed: Speed for each frame (NaN if unavailable)
            - frame_environment: Environment number for each frame (NaN if unavailable)
            - frame_trial: Trial number for each frame (NaN if unavailable)
        """
        timestamps = self.session.loadone("positionTracking.times")
        position = self.session.loadone("positionTracking.position")
        idx_behave_to_frame = self.session.loadone("positionTracking.mpci")
        trial_start_index = self.session.loadone("trials.positionTracking")
        num_samples = len(position)
        trial_numbers = np.arange(len(trial_start_index))
        trial_lengths = np.append(np.diff(trial_start_index), num_samples - trial_start_index[-1])
        trial_numbers = np.repeat(trial_numbers, trial_lengths)
        trial_environment = self.session.loadone("trials.environmentIndex")
        trial_environment = np.repeat(trial_environment, trial_lengths)

        within_trial = np.append(np.diff(trial_numbers) == 0, True)
        sample_duration = np.append(np.diff(timestamps), 0)
        speed = np.append(np.diff(position) / sample_duration[:-1], 0)
        sample_duration = sample_duration * within_trial
        speed = speed * within_trial

        frame_timestamps = self.session.loadone("mpci.times")
        difference_timestamps = np.abs(timestamps - frame_timestamps[idx_behave_to_frame])
        sampling_period = np.median(np.diff(frame_timestamps))
        dist_cutoff = sampling_period / 2

        frame_position = np.zeros_like(frame_timestamps)
        count = np.zeros_like(frame_timestamps)
        helpers.get_average_frame_position(position, idx_behave_to_frame, difference_timestamps, dist_cutoff, frame_position, count)
        frame_position[count > 0] /= count[count > 0]
        frame_position[count == 0] = np.nan
        frame_speed = np.diff(frame_position) / np.diff(frame_timestamps)
        frame_speed = np.append(frame_speed, 0)

        # Get a map from frame to behavior time for quick lookup
        idx_frame_to_behave, dist_frame_to_behave = helpers.nearestpoint(frame_timestamps, timestamps)
        idx_get_position = dist_frame_to_behave < dist_cutoff

        frame_environment = np.full(len(frame_timestamps), np.nan)
        frame_environment[idx_get_position] = trial_environment[idx_frame_to_behave[idx_get_position]]
        frame_environment[count == 0] = np.nan

        frame_trial = np.full(len(frame_timestamps), np.nan)
        frame_trial[idx_get_position] = trial_numbers[idx_frame_to_behave[idx_get_position]]
        frame_trial[count == 0] = np.nan

        return frame_position, frame_speed, frame_environment, frame_trial

    @with_temp_params
    @manage_one_cache
    def get_placefield_prediction(
        self,
        use_session_filters: bool = True,
        spks_type: Union[str, None] = None,
        use_speed_threshold: bool = True,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Predict neural activity from place field maps.

        This method uses averaged environment maps to predict neural activity
        at each imaging frame based on the animal's position and environment.

        Parameters
        ----------
        use_session_filters : bool, optional
            Whether to filter ROIs using session.idx_rois. Default is True.
        spks_type : str or None, optional
            Type of spike data to use. If None, uses session's current spks_type.
            Temporarily changes session.spks_type if provided. Default is None.
        use_speed_threshold : bool, optional
            Whether to only predict for frames where speed exceeds threshold.
            Default is True.
        clear_one_cache : bool, optional
            Whether to clear the onefile cache after processing. Default is True.
        params : SpkmapParams, dict, or None, optional
            Parameters for processing. If None, uses instance parameters.
            If a dict, updates instance parameters temporarily.
            Parameters are restored after method execution. Default is None.

        Returns
        -------
        tuple
            A tuple containing:
            - placefield_prediction: Predicted activity array with shape (frames, rois).
              NaN for frames where prediction is not possible.
            - extras: Dictionary with additional information:
              - frame_position_index: Position bin index for each frame
              - frame_environment_index: Environment index for each frame
              - idx_valid: Boolean array indicating valid predictions

        Notes
        -----
        Predictions are based on averaged trial maps. Frames where the animal
        is not moving (if use_speed_threshold=True) or where position/environment
        data is unavailable will have NaN predictions.
        """
        if spks_type is not None:
            _spks_type = self.session.spks_type
            self.session.params.spks_type = spks_type

        frame_position, frame_speed, frame_environment, _ = self.get_frame_behavior(clear_one_cache, params)
        idx_valid = ~np.isnan(frame_position)
        if use_speed_threshold:
            idx_valid = idx_valid & (frame_speed > self.params.speed_threshold)

        # Convert frame position to bins indices
        frame_position_index = np.searchsorted(self.dist_edges, frame_position, side="right") - 1

        # Get the place field for each neuron
        env_maps = self.get_env_maps(use_session_filters=use_session_filters)
        env_maps.average_trials()

        # Convert frame environment to indices
        env_to_idx = {env: i for i, env in enumerate(env_maps.environments)}
        frame_environment_index = np.array([env_to_idx[env] if not np.isnan(env) else -1000 for env in frame_environment], dtype=int)

        # Get the original spks data
        spks = self.session.spks
        if use_session_filters:
            spks = spks[:, self.session.idx_rois]

        # Use a numba speed up to get the placefield prediction (single pass simple algorithm)
        placefield_prediction = np.full(spks.shape, np.nan)
        spkmaps = np.stack(list(map(lambda x: x.T, env_maps.spkmap)))
        placefield_prediction = placefield_prediction_numba(
            placefield_prediction,
            spkmaps,
            frame_environment_index,
            frame_position_index,
            idx_valid,
        )

        # This will add samples for which a place field was not estimable (at the edges of the environment)
        idx_valid = np.all(~np.isnan(placefield_prediction), axis=1)

        # Reset spks_type
        if spks_type is not None:
            self.session.params.spks_type = _spks_type

        # Include extra details in a dictionary for forward compatibility
        extras = dict(
            frame_position_index=frame_position_index,
            frame_environment_index=frame_environment_index,
            idx_valid=idx_valid,
        )

        return placefield_prediction, extras

    def get_traversals(
        self,
        idx_roi: int,
        idx_env: int,
        width: int = 10,
        placefield_threshold: float = 5.0,
        fill_nan: bool = False,
        spks: np.ndarray = None,
        spks_prediction: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract neural activity around place field peak during traversals.

        This method identifies trials where the animal passes through a neuron's
        place field peak and extracts activity windows around those moments.

        Parameters
        ----------
        idx_roi : int
            Index of the ROI (neuron) to analyze.
        idx_env : int
            Index of the environment to analyze (index into env_maps.environments).
        width : int, optional
            Number of frames on each side of the peak to include. Total window
            size is 2*width + 1. Default is 10.
        placefield_threshold : float, optional
            Maximum distance from place field peak to include a trial (in spatial units).
            Default is 5.0.
        fill_nan : bool, optional
            Whether to fill NaN values with 0. Default is False.
        spks : np.ndarray, optional
            Spike data array. If None, loads from session. Default is None.
        spks_prediction : np.ndarray, optional
            Place field prediction array. If None, computes it. Default is None.

        Returns
        -------
        tuple
            A tuple containing:
            - traversals: Array of shape (num_traversals, 2*width+1) containing
              actual neural activity around each traversal.
            - pred_travs: Array of shape (num_traversals, 2*width+1) containing
              predicted activity around each traversal.

        Notes
        -----
        Only includes trials where the animal passes within placefield_threshold
        of the place field peak. The peak is determined from the averaged spike
        map for the specified ROI and environment.
        """
        frame_position, _, frame_environment, frame_trial = self.get_frame_behavior()
        if spks_prediction is None:
            spks_prediction = self.get_placefield_prediction(use_session_filters=True)[0]
        if spks is None:
            spks = self.session.spks[:, self.session.idx_rois]

        if spks.shape != spks_prediction.shape:
            raise ValueError("spks and spks_prediction must have the same shape")

        env_maps = self.get_env_maps()
        pos_peak = self.dist_centers[np.nanargmax(np.nanmean(env_maps.spkmap[idx_env][idx_roi], axis=0))]
        envnum = env_maps.environments[idx_env]

        env_trials = np.unique(frame_trial[frame_environment == envnum])

        num_trials = len(env_trials)
        idx_traversal = -1 * np.ones(num_trials, dtype=int)
        for itrial, trialnum in enumerate(env_trials):
            idx_trial = frame_trial == trialnum
            idx_closest_pos = np.nanargmin(np.abs(frame_position - pos_peak) + 10000 * ~idx_trial)

            # Only include the trial if the closest position is within placefield threshold of the peak
            if np.abs(frame_position[idx_closest_pos] - pos_peak) < placefield_threshold:
                idx_traversal[itrial] = idx_closest_pos

        # Filter out trials that don't have a traversal
        idx_traversal = idx_traversal[idx_traversal != -1]

        # Get traversals through place field in requested environment
        traversals = np.zeros((len(idx_traversal), width * 2 + 1))
        pred_travs = np.zeros((len(idx_traversal), width * 2 + 1))
        for ii, it in enumerate(idx_traversal):
            istart = it - width
            iend = it + width + 1
            istartoffset = max(0, -istart)
            iendoffset = max(0, iend - spks.shape[0])
            traversals[ii, istartoffset : width * 2 + 1 - iendoffset] = spks[istart + istartoffset : iend - iendoffset, idx_roi]
            pred_travs[ii, istartoffset : width * 2 + 1 - iendoffset] = spks_prediction[istart + istartoffset : iend - iendoffset, idx_roi]

        if fill_nan:
            traversals[np.isnan(traversals)] = 0.0
            pred_travs[np.isnan(pred_travs)] = 0.0

        return traversals, pred_travs
