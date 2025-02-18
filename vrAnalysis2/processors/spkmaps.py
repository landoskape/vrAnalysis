from typing import Union, Tuple, NamedTuple, Iterable, List, Optional, Dict, Any
from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field, asdict
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
    """Container for occupancy, speed, and spike maps.

    Attributes
    ----------
    occmap : np.ndarray
        Occupancy map showing time spent at each position
    speedmap : np.ndarray | None
        Speed map showing average speed at each position
    spkmap : np.ndarray | None
        Spike map showing neural activity at each position
    """

    occmap: np.ndarray
    speedmap: Union[np.ndarray, None] = None
    spkmap: Union[np.ndarray, None] = None

    @classmethod
    def map_types(self) -> List[str]:
        return ["occmap", "speedmap", "spkmap"]

    def full_maps(self) -> bool:
        return all([getattr(self, maptype) is not None for maptype in self.map_types])

    def __repr__(self) -> str:
        shapes = []
        for name in self.map_types():
            if getattr(self, name) is None:
                shape = None
            elif isinstance(getattr(self, name), np.ndarray):
                shape = getattr(self, name).shape
            elif isinstance(getattr(self, name), list) and all(isinstance(x, np.ndarray) for x in getattr(self, name)):
                shape = [x.shape for x in getattr(self, name)]
            else:
                raise ValueError(f"Unknown map type: {type(getattr(self, name))}")
            shapes.append(f"{name}: {shape}")
        return f"Maps({', '.join(shapes)})"

    def __getitem__(self, key: str) -> np.ndarray:
        return getattr(self, key)

    def __setitem__(self, key: str, value: np.ndarray):
        setattr(self, key, value)

    def nbytes(self) -> int:
        num_bytes = 0
        for name in self.map_types():
            if getattr(self, name) is not None:
                if isinstance(getattr(self, name), np.ndarray):
                    num_bytes += getattr(self, name).nbytes
                elif isinstance(getattr(self, name), list) and all(isinstance(x, np.ndarray) for x in getattr(self, name)):
                    num_bytes += sum(x.nbytes for x in getattr(self, name))
                else:
                    raise ValueError(f"Unknown map type: {type(getattr(self, name))}")
        return num_bytes


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
    autosave : bool, default=True
        Whether to save the cache automatically
    """

    dist_step: float = 1.0
    speed_threshold: float = 1.0
    speed_max_allowed: float = np.inf
    full_trial_flexibility: Union[float, None] = 3.0
    standardize_spks: bool = True
    smooth_width: Union[float, None] = 1.0
    autosave: bool = True

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

    def decorator(process_method):
        @wraps(process_method)
        def wrapper(self: "SpkmapProcessor", *args, **kwargs):
            # Attempt to load from cache
            maps_to_load = ["occmap"]
            if kwargs.get("get_speedmap", True):
                maps_to_load.append("speedmap")
            if kwargs.get("get_spkmap", True):
                maps_to_load.append("spkmap")

            if not disable and not kwargs.get("force_recompute", False):
                cached_data, valid_cache = self.load_from_cache(data_type, maps_to_load)
                if valid_cache:
                    return cached_data

            # If forcing a recompute or not valid, process data with the pipeline method
            result = process_method(self, *args, **kwargs)

            # And cache if requested (only if all maps are loaded)
            if not disable and self.params.autosave and maps_to_load == Maps.map_types():
                self.save_cache(data_type, result)
            return result

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

    NOTES:
    It would be nice to define a decorator around bound methods that would automatically
    detect which onefiles are loaded and clear them when the method is called if the user
    requests...
    """

    session: Union[SessionData, SessionToSpkmapProtocol]
    params: SpkmapParams = field(default_factory=SpkmapParams, repr=False)
    data_cache: dict = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        # Check if the session provided is compatible with SpkmapProcessing
        if not isinstance(self.session, SessionData):
            raise ValueError(f"session must be a SessionData instance, not {type(self.session)}")
        # TODO: this loads one data because they are properties... consider removing the property decorator
        # or alternatively just removing this check and duck typing (using the Protocol for instructions only)
        # if not isinstance(self.session, SessionToSpkmapProtocol):
        #     raise ValueError("session must meet the criteria of a SessionToSpkmapProtocol!")

        # We need to handle the case where params is a dictionary of partial updates to the default params
        if isinstance(self.params, dict):
            self.params = SpkmapParams.from_dict(self.params)
        else:
            if not isinstance(self.params, SpkmapParams):
                raise ValueError(f"params must be a SpkmapParams instance or a dictionary, not {type(self.params)}")

    def cached_dependencies(self, data_type: str) -> List[str]:
        """Get the dependencies for a given data type"""
        if data_type == "raw_maps":
            return ["dist_step", "speed_threshold", "speed_max_allowed", "standardize_spks"]
        elif data_type == "processed_maps":
            return ["dist_step", "speed_threshold", "speed_max_allowed", "standardize_spks", "smooth_width"]
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

    def save_cache(self, data_type: str, data: Maps):
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
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def load_from_cache(self, data_type: str, maps_to_load: Optional[List[str]] = None) -> Tuple[Union[Maps, Dict[str, np.ndarray]], bool]:
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
                    return self._load_from_cache(data_type, params_hash, maps_to_load), True
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

    def _load_from_cache(self, data_type: str, params_hash: str, maps_to_load: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Load the cached data for a given data type"""
        cache_dir = self.cache_directory(data_type)
        if data_type == "raw_maps" or data_type == "processed_maps":
            cached_data = {}
            for name in Maps.map_types():
                if maps_to_load is None or name in maps_to_load:
                    cached_data[name] = np.load(cache_dir / f"data_{name}_{params_hash}.npy")
            return Maps(**cached_data)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    @manage_one_cache
    def _filter_environments(
        self,
        envnum: Union[int, Iterable[int], None] = None,
        clear_one_cache: bool = False,
    ) -> np.ndarray:
        """Filter the session data to only include trials from certain environments

        NOTE:
        This assumes that the trials are in order. We might want to use the third output of session.positions to
        get the "real" trial numbers which aren't always contiguous and 0 indexed.

        If envnum is not provided, will return all trials.
        """
        if envnum is None:
            envnum = self.session.environments
        envnum = helpers.check_iterable(envnum)
        return np.where(np.isin(self.session.trial_environment, envnum))[0]

    def maps_to_process(self, get_speedmap: bool = True, get_spkmap: bool = True) -> List[str]:
        maps_to_process = ["occmap"]
        if get_speedmap:
            maps_to_process.append("speedmap")
        if get_spkmap:
            maps_to_process.append("spkmap")
        return maps_to_process

    @manage_one_cache
    def _idx_required_position_bins(self, clear_one_cache: bool = False) -> np.ndarray:
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
    def raw_maps(
        self,
        get_speedmap: bool = True,
        get_spkmap: bool = True,
        force_recompute: bool = False,
        clear_one_cache: bool = False,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ) -> Maps:
        """Get maps (occupancy, speed, spkmap) from session data by processing with provided parameters.

        Parameters
        ----------
        get_speedmap : bool, default=True
            Whether to get the speed map
        get_spkmap : bool, default=True
            Whether to get the spkmap
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

        if get_spkmap:
            # get spiking information
            spks = self.session.spks

            # Do standardization
            if self.params.standardize_spks:
                spks = median_zscore(spks, median_subtract=not self.session.zero_baseline_spks)

        # Get high resolution occupancy and speed maps
        dtype = np.float32
        occmap = np.zeros((self.session.num_trials, num_positions), dtype=dtype)
        counts = np.zeros((self.session.num_trials, num_positions), dtype=dtype)
        if get_speedmap:
            speedmap = np.zeros((self.session.num_trials, num_positions), dtype=dtype)
        else:
            speedmap = None
        if get_spkmap:
            spkmap = np.zeros((self.session.num_trials, num_positions, spks.shape[1]), dtype=dtype)
        else:
            spkmap = None
        if get_speedmap or get_spkmap:
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
        )
        if get_speedmap:
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
            )
        if get_spkmap:
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
        if get_speedmap:
            speedmap = replace_missing_data(speedmap, first_valid_bin, last_valid_bin)
        if get_spkmap:
            spkmap = replace_missing_data(spkmap, first_valid_bin, last_valid_bin)

        return Maps(occmap, speedmap, spkmap)

    @with_temp_params
    @manage_one_cache
    @cached_processor("processed_maps")
    def maps(
        self,
        get_speedmap: bool = True,
        get_spkmap: bool = True,
        force_recompute: bool = False,
        clear_one_cache: bool = False,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ):
        """Process the maps"""
        maps_to_process = self.maps_to_process(get_speedmap, get_spkmap)

        # Get the raw maps first (don't need to specify params because they're already set by this method)
        maps = self.raw_maps(
            get_speedmap=get_speedmap,
            get_spkmap=get_spkmap,
            force_recompute=force_recompute,
            clear_one_cache=clear_one_cache,
        )

        if self.params.smooth_width is not None:
            # TODO: Consider stacking maps to do this convolution all at once?
            kernel = get_gauss_kernel(self.dist_centers, self.params.smooth_width)

            # Replace nans with 0s
            idxnan = np.isnan(maps.occmap)
            for mapname in maps_to_process:
                maps[mapname][idxnan] = 0

            for mapname in maps_to_process:
                maps[mapname] = convolve_toeplitz(maps[mapname], kernel, axis=1)

            # Put nans back in place
            for mapname in maps_to_process:
                maps[mapname][idxnan] = np.nan

        if get_speedmap:
            maps.speedmap = correct_map(maps.occmap, maps.speedmap)
        if get_spkmap:
            maps.spkmap = correct_map(maps.occmap, maps.spkmap)

        return maps

    @with_temp_params
    @manage_one_cache
    def get_env_maps(
        self,
        envnum: Union[int, List[int], None] = None,
        average: bool = False,
        popnan: bool = True,
        get_speedmap: bool = True,
        get_spkmap: bool = True,
        force_recompute: bool = False,
        clear_one_cache: bool = False,
        params: Union[SpkmapParams, Dict[str, Any], None] = None,
    ):
        """Get the map for a given environment number"""
        # First get environments to process (all if not specified)
        if envnum is None:
            envnum = self.session.environments

        # Make sure it's an iterable -- the output will always be a list
        envnum = helpers.check_iterable(envnum)

        # Then get the indices of the position bins that are required for a full trial
        idx_required_position_bins = self._idx_required_position_bins(clear_one_cache)

        # Get the processed maps (don't need to specify params because they're already set by the decorator)
        maps = self.maps(
            get_speedmap=get_speedmap,
            get_spkmap=get_spkmap,
            force_recompute=force_recompute,
            clear_one_cache=clear_one_cache,
        )

        # Make a list of the maps we are processing
        maps_to_process = self.maps_to_process(get_speedmap, get_spkmap)

        # Filter the maps to only include full trials
        full_trials = np.all(~np.isnan(maps.occmap[:, idx_required_position_bins]), axis=1)
        for mapname in maps_to_process:
            maps[mapname] = maps[mapname][full_trials]

        # If popping nan positions, figure out which positions have a nan in any trial
        # (it'll be the same for occmap and other maps) and remove them
        if popnan:
            idx_nan_positions = np.any(np.isnan(maps.occmap), axis=0)
            for mapname in maps_to_process:
                maps[mapname] = maps[mapname][:, ~idx_nan_positions]

        # Then group each one by environment
        # -> this is now (trials_in_env, position_bins, ...(roi if spkmap)...)
        idx_each_environment = [self._filter_environments(env)[0] for env in envnum]
        for mapname in maps_to_process:
            maps[mapname] = [maps[mapname][idx] for idx in idx_each_environment]

        # Then average within environment if requested
        if average:
            for mapname in maps_to_process:
                maps[mapname] = [ss.mean(envmap, axis=0) for envmap in maps[mapname]]

        # Change spkmap to be ROIs first
        if get_spkmap:
            maps.spkmap = [np.moveaxis(spkmap, -1, 0) for spkmap in maps.spkmap]

        return maps

    @property
    def dist_edges(self):
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
    def dist_centers(self):
        """Distance centers for the position bins"""
        return helpers.edge2center(self.dist_edges)

    def reliability_measures(self):
        """Empty for planning and notetaking purposes


        Reliability used to be computed by default in spkmap loading methods.
        Now it should be computed independently, only when requested.
        I used to include numcv in the spkmap loading methods for this purpose!!
        (Might only use reliability loo---)
        """


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
