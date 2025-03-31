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
from .support import median_zscore, get_gauss_kernel, convolve_toeplitz, get_summation_map, correct_map, replace_missing_data


@dataclass
class Maps:
    """Base class for occupancy, speed, and spike maps."""

    occmap: np.ndarray | list[np.ndarray]
    speedmap: np.ndarray | list[np.ndarray]
    spkmap: np.ndarray | list[np.ndarray]
    by_environment: bool
    rois_first: bool
    environments: list[int] | None = None
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
        if self.by_environment:
            num_trials = [occmap.shape[0] for occmap in self.occmap]
            num_trials = ":".join([str(nt) for nt in num_trials])
            num_positions = self.occmap[0].shape[1]
        else:
            num_trials = self.occmap.shape[0]
            num_positions = self.occmap.shape[1]
        if self.by_environment:
            num_rois = self.spkmap[0].shape[0] if self.rois_first else self.spkmap[0].shape[1]
        else:
            num_rois = self.spkmap.shape[0] if self.rois_first else self.spkmap.shape[1]
        environments = f", environments={self.environments}" if self.by_environment else ""
        return f"Maps(num_trials={num_trials}, num_positions={num_positions}, num_rois={num_rois}{environments}, rois_first={self.rois_first})"

    @classmethod
    def create_raw_maps(cls, occmap: np.ndarray, speedmap: np.ndarray, spkmap: np.ndarray) -> "Maps":
        return cls(occmap=occmap, speedmap=speedmap, spkmap=spkmap, by_environment=False, rois_first=False)

    @classmethod
    def create_processed_maps(cls, occmap: np.ndarray, speedmap: np.ndarray, spkmap: np.ndarray) -> "Maps":
        return cls(occmap=occmap, speedmap=speedmap, spkmap=spkmap, by_environment=False, rois_first=True)

    @classmethod
    def create_environment_maps(
        cls,
        occmap: list[np.ndarray],
        speedmap: list[np.ndarray],
        spkmap: list[np.ndarray],
        environments: list[int],
    ) -> "Maps":
        return cls(occmap=occmap, speedmap=speedmap, spkmap=spkmap, environments=environments, by_environment=True, rois_first=True)

    @classmethod
    def map_types(self) -> List[str]:
        return ["occmap", "speedmap", "spkmap"]

    def __getitem__(self, key: str) -> np.ndarray:
        return getattr(self, key)

    def __setitem__(self, key: str, value: np.ndarray):
        setattr(self, key, value)

    def _get_position_axis(self, mapname: str) -> int:
        """The only time the position axis isn't the last one is for spkmap when rois_first is False"""
        average_offset = -1 if self._averaged else 0
        if mapname == "spkmap" and not self.rois_first:
            return -2 + average_offset
        else:
            return -1

    def filter_positions(self, idx_positions: np.ndarray) -> None:
        for mapname in self.map_types():
            axis = self._get_position_axis(mapname)
            if self.by_environment:
                self[mapname] = [np.take(x, idx_positions, axis=axis) for x in self[mapname]]
            else:
                self[mapname] = np.take(self[mapname], idx_positions, axis=axis)

    def filter_rois(self, idx_rois: np.ndarray) -> None:
        axis = 0 if self.rois_first else -1
        if self.by_environment:
            self.spkmap = [np.take(x, idx_rois, axis=axis) for x in self.spkmap]
        else:
            self.spkmap = np.take(self.spkmap, idx_rois, axis=0)

    def filter_environments(self, environments: list[int]) -> None:
        if self.by_environment:
            idx_to_requested_env = [i for i, env in enumerate(self.environments) if env in environments]
            self.occmap = [self.occmap[i] for i in idx_to_requested_env]
            self.speedmap = [self.speedmap[i] for i in idx_to_requested_env]
            self.spkmap = [self.spkmap[i] for i in idx_to_requested_env]
            self.environments = [self.environments[i] for i in idx_to_requested_env]
        else:
            raise ValueError("Cannot filter environments when maps aren't separated by environment!")

    def pop_nan_positions(self) -> None:
        """Remove positions with nans from the maps"""
        if self.by_environment:
            idx_valid_positions = np.where(~np.any(np.stack([np.any(np.isnan(occmap), axis=0) for occmap in self.occmap], axis=0), axis=0))[0]
        else:
            idx_valid_positions = np.where(~np.any(np.isnan(self.occmap), axis=0))[0]
        self.filter_positions(idx_valid_positions)

    def smooth_maps(self, positions: np.ndarray, kernel_width: float) -> None:
        """Smooth the maps using a Gaussian kernel"""
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
                self[mapname] = [map[idx] for map, idx in zip(self[mapname], idxnan)]
            else:
                self[mapname][idxnan] = 0

        for mapname in self.map_types():
            # Since we moved ROIs to the last axis position will be axis=1 for all map types
            self[mapname] = convolve_toeplitz(self[mapname], kernel, axis=1)

        # Put nans back in place
        for mapname in self.map_types():
            self[mapname][idxnan] = np.nan

        # Move the rois axis back to the first axis
        if self.rois_first:
            if self.by_environment:
                self.spkmap = [np.moveaxis(map, -1, 0) for map in self.spkmap]
            else:
                self.spkmap = np.moveaxis(self.spkmap, -1, 0)

    def average_trials(self, keepdims: bool = False) -> None:
        """Average the trials within each environment"""
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
        num_bytes = 0
        for name in self.map_types():
            if self.by_environment:
                num_bytes += sum(x.nbytes for x in getattr(self, name))
            else:
                num_bytes += getattr(self, name).nbytes
        return num_bytes

    def raw_to_processed(self, positions: np.ndarray, smooth_width: float | None = None) -> "Maps":
        """Convert raw maps to processed maps"""
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
        return Reliability(self.values[:, idx_rois], self.environments, self.method)

    def filter_environments(self, idx_environments: np.ndarray) -> "Reliability":
        return Reliability(self.values[idx_environments], self.environments[idx_environments], self.method)

    def filter_by_environment(self, environments: list[int]) -> "Reliability":
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
        """Create a SpkmapParams instance from a dictionary, using defaults for missing values"""
        return cls(**{k: params_dict[k] for k in params_dict})

    @classmethod
    def from_path(cls, path: Path) -> "SpkmapParams":
        """Create a SpkmapParams instance from a json file"""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def compare(self, other: "SpkmapParams", filter_keys: Optional[List[str]] = None) -> bool:
        """Compare two SpkmapParams instances"""
        if filter_keys is None:
            return self == other
        else:
            return all(getattr(self, key) == getattr(other, key) for key in filter_keys)

    def save(self, path: Path) -> None:
        """Save the parameters to a json file"""
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

    Used on bound methods of SpkmapProcessor. When no_cache is True, this
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

    session: Union[SessionData, SessionToSpkmapProtocol]
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
        """Get the dependencies for a given data type"""
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

    def cache_directory(self, data_type: Optional[str] = None) -> Path:
        """Get the cache directory for a given data type and spks_type"""
        if data_type is None:
            return self.session.data_path / "spkmaps"
        else:
            folder_name = f"{data_type}_{self.session.spks_type}"
            return self.session.data_path / "spkmaps" / folder_name

    def dependent_params(self, data_type: str) -> dict:
        """Get the dependent parameters for a given data type"""
        return {k: getattr(self.params, k) for k in self.cached_dependencies(data_type)}

    def _params_hash(self, data_type: str) -> str:
        """Get the hash of the dependent parameters for a given data type"""
        return hashlib.sha256(json.dumps(self.dependent_params(data_type), sort_keys=True).encode()).hexdigest()

    def save_cache(self, data_type: str, data: Union[Maps, Reliability]):
        """Save the cached params and data for a given data type"""
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

    def load_from_cache(self, data_type: str) -> Tuple[Union[Maps, Reliability], bool]:
        """Get the cached params and data for a given data type"""
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
        """Load the cached data for a given data type"""
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
    ) -> np.ndarray[bool]:
        """Filter the session data to only include trials from certain environments

        NOTE:
        This assumes that the trials are in order. We might want to use the third output of session.positions to
        get the "real" trial numbers which aren't always contiguous and 0 indexed.

        If envnum is not provided, will return all trials.
        """
        if envnum is None:
            envnum = self.session.environments
        envnum = helpers.check_iterable(envnum)
        return np.isin(self.session.trial_environment, envnum)

    @property
    def dist_edges(self) -> np.ndarray[float]:
        """Distance edges for the position bins"""
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
    def dist_centers(self) -> np.ndarray[float]:
        """Distance centers for the position bins"""
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
    @cached_processor("raw_maps", disable=True)  # It's probably better to only cache processed maps
    def get_raw_maps(
        self,
        force_recompute: bool = False,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ) -> Maps:
        """Get maps (occupancy, speed, spkmap) from session data by processing with provided parameters.

        Parameters
        ----------
        force_recompute : bool, default=False
            Whether to force the recomputation of the maps even if they exist in the cache.
        clear_one_cache : bool, default=False
            Whether to clear the onefile cache after getting the maps (only clears the onecache for this method)
        params : SpkmapParams, dict, or None, default=None
            Parameters for the maps. If None, the parameters will be taken from the SpkmapProcessor instance.
            If a dictionary, it will be used to update the parameters.
            Will always be temporary -- so the original parameters will be restored after the method is finished.
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
    @cached_processor("processed_maps")
    def get_processed_maps(
        self,
        force_recompute: bool = False,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ) -> Maps:
        """Process the maps"""
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
        """Get the map for a given environment number"""
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
    ):
        """Get the reliability of the maps"""
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


@nb.njit(parallel=True, cache=True)
def getAverageFramePosition(behavePosition, behaveSpeed, speedThreshold, idxBehaveToFrame, distBehaveToFrame, distCutoff, frame_position, count):
    """
    get the position of each frame by averaging across positions within a sample
    """
    for sample in nb.prange(len(behavePosition)):
        if (distBehaveToFrame[sample] < distCutoff) and (behaveSpeed[sample] > speedThreshold):
            frame_position[idxBehaveToFrame[sample]] += behavePosition[sample]
            count[idxBehaveToFrame[sample]] += 1


@nb.njit(parallel=True, cache=True)
def getAverageFrameSpeed(behaveSpeed, speedThreshold, idxBehaveToFrame, distBehaveToFrame, distCutoff, frame_speed, count):
    """
    get the speed of each frame by averaging across speeds within a sample
    """
    for sample in nb.prange(len(behaveSpeed)):
        if (distBehaveToFrame[sample] < distCutoff) and (behaveSpeed[sample] > speedThreshold):
            frame_speed[idxBehaveToFrame[sample]] += behaveSpeed[sample]
            count[idxBehaveToFrame[sample]] += 1
