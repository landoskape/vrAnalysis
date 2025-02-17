from dataclasses import dataclass, field, asdict
import numpy as np
import numba as nb
import json
from typing import Dict, Any, Union, Tuple, NamedTuple, Iterable
from typing import Protocol, runtime_checkable
import speedystats as ss
from .. import helpers
from ..sessions.base import SessionData
from .support import median_zscore


class Maps(NamedTuple):
    occmap: np.ndarray
    speedmap: Union[np.ndarray, None]
    spkmap: Union[np.ndarray, None]


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
    dist_step : int, default=1
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
    clear_buffer : bool, default=False
        Whether to clear the one_buffer after getting the maps. Useful for large batch
        processing where memory might be a concern.
    """

    dist_step: int = 1
    speed_threshold: float = 1.0
    speed_max_allowed: float = np.inf
    full_trial_flexibility: Union[float, None] = 3
    standardize_spks: bool = True
    clear_buffer: bool = False

    @classmethod
    def from_dict(cls, params_dict: dict) -> "SpkmapParams":
        """Create a SpkmapParams instance from a dictionary, using defaults for missing values"""
        return cls(**{k: params_dict[k] for k in params_dict})

    def __post_init__(self):
        if self.dist_step <= 0:
            raise ValueError("dist_step must be positive")
        if self.speed_threshold <= 0:
            raise ValueError("speed_threshold must be positive")
        if self.full_trial_flexibility is not None and self.full_trial_flexibility < 0:
            raise ValueError("If used, full_trial_flexibility must be nonnegative (can also be None)")

    def json(self) -> str:
        """Create a json string of the parameters"""
        # Convert the dataclass to a dictionary
        params_dict = asdict(self)
        # Sort the dictionary by key
        sorted_params = sorted(params_dict.items())
        # Convert the sorted dictionary to a json string
        return json.dumps(sorted_params, sort_keys=True)


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
        if not isinstance(self.session, SessionToSpkmapProtocol):
            raise ValueError("session must meet the criteria of the SessionToSpkmapProtocol!")

        # We need to handle the case where params is a dictionary of partial updates to the default params
        if isinstance(self.params, dict):
            self.params = SpkmapParams.from_dict(self.params)
        else:
            if not isinstance(self.params, SpkmapParams):
                raise ValueError(f"params must be a SpkmapParams instance or a dictionary, not {type(self.params)}")

    def _filter_environments(self, envnum: Union[int, Iterable[int], None] = None) -> np.ndarray:
        """Filter the session data to only include trials from certain environments

        If envnum is not provided, will return all trials.
        """
        if envnum is None:
            return np.arange(self.session.num_trials)
        else:
            envnum = helpers.check_iterable(envnum)
            return np.where(np.isin(self.session.trial_environment, envnum))[0]

    def get_maps(
        self,
        envnum: Union[int, Iterable[int], None] = None,
        get_speedmap: bool = True,
        get_spkmap: bool = True,
        clear_buffer: bool = False,
    ) -> Maps:
        """Get maps (occupancy, speed, spkmap) from session data by processing with provided parameters.

        Notes on engineering:
        >>>>>>> These notes are not needed here but I'm keeping them to remember changes that are required elsewhere!
        - I used to include "onefile" here -- but we don't need that anymore because the sessionData object will be used to define how to get the spks to use
        - I used to include an idxROIs (based on get planes) -- but that's also not needed because the sessionData object will determine which spks to load...
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
        occmap = np.zeros((self.session.num_trials, num_positions))
        counts = np.zeros((self.session.num_trials, num_positions))
        if get_speedmap:
            speedmap = np.zeros((self.session.num_trials, num_positions))
        else:
            speedmap = None
        if get_spkmap:
            spkmap = np.zeros((self.session.num_trials, num_positions, spks.shape[1]))
        else:
            spkmap = None
        if get_speedmap or get_spkmap:
            extra_counts = np.zeros((self.session.num_trials, num_positions))

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

        print("WARNING: Smoothing of speedmap not implemented yet!")
        # # correct speedmap immediately
        # if speedSmoothing is not None:
        #     kk = helpers.getGaussKernel(distcenter, speedSmoothing)
        #     speedmap = helpers.convolveToeplitz(speedmap, kk, axis=1)
        #     smoothocc = helpers.convolveToeplitz(occmap, kk, axis=1)
        #     speedmap[smoothocc != 0] /= smoothocc[smoothocc != 0]
        # else:
        #     speedmap[occmap != 0] /= occmap[occmap != 0]

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

        if clear_buffer:
            # TODO:
            # Keep a list of all the onefiles loaded for this method, only clear those!
            self.session.clear_cache()

        # NOTE:
        # The "correctMap" function isn't here by design:
        # It's important to do smoothing before correcting when smoothing is requested.
        # (That's why it isn't here....)

        return Maps(occmap, speedmap, spkmap)

    @property
    def dist_edges(self):
        """Distance edges for the position bins"""
        if not hasattr(self, "_dist_edges"):
            self._dist_edges = self._get_bin_edges()
        return self._dist_edges

    @property
    def dist_centers(self):
        """Distance centers for the position bins"""
        return helpers.edge2center(self.dist_edges)

    def _get_bin_edges(self):
        """Get the bin edges for the position bins"""
        env_length = self.session.env_length
        if hasattr(env_length, "__len__"):
            if np.unique(env_length).size != 1:
                msg = "SpkmapProcessor (currently) requires all trials to have the same env length!"
                raise ValueError(msg)
            env_length = env_length[0]

        num_positions = int(env_length / self.params.dist_step)
        return np.linspace(0, env_length, num_positions + 1)

    def check_params_match(self, new_params: Dict[str, Any]) -> bool:
        """Check if new parameters match those used in saved data.

        Will need to use the json method of the SpkmapParams class to compare the parameters.
        """

    def reliability_measures(self):
        """Empty for planning and notetaking purposes


        Reliability used to be computed by default in spkmap loading methods.
        Now it should be computed independently, only when requested.
        I used to include numcv in the spkmap loading methods for this purpose!!
        (Might only use reliability loo---)
        """

    def register_trials(self):
        """Empty for planning and notetaking purposes

        The spkmap loading method used to measure which trials were "full" and in each trial.
        This can be done independently of the spkmap registration!!!
        I used to include full_trial_flexibility in the spkmap loading methods. Should be here.
        """
        # The following code used to be in placeCellSingleSession.load_data
        self.distcenters = helpers.edge2center(self.distedges)
        self.numTrials = self.occmap.shape[0]
        self.boolFullTrials, self.idxFullTrials, self.idxFullTrialEachEnv = self._return_trial_indices(
            self.occmap, self.distedges, self.full_trial_flexibility
        )


@nb.njit(parallel=True, cache=True)
def get_summation_map(
    value_to_sum,
    trial_idx,
    position_bin,
    map_data,
    counts,
    speed,
    speed_threshold,
    speed_max_threshold,
    dist_behave_to_frame,
    dist_cutoff,
    sample_duration,
    scale_by_sample_duration: bool = False,
):
    """
    this is the fastest way to get a single summation map
    -- accepts 1d arrays value, trialidx, positionbin of the same size --
    -- shape determines the number of trials and position bins (they might not all be represented in trialidx or positionbin, or we could just do np.max()) --
    -- each value represents some number to be summed as a function of which trial it was in and which positionbin it was in --

    NOTE:
    This is now refactored to be a a single target function rather than the getAllMaps in vrAnalysis1.
    It's really worth it to get occmaps by themselves when necessary, and I think it won't be that much
    longer to do both occmap / spkmap independently (spkmap can use this too-- the third dimension of ROIs is permitted with this indexing!)

    go through each behavioral frame
    if speed faster than threshold, keep, otherwise continue
    if distance to frame lower than threshold, keep, otherwise continue
    for current trial and position, add sample duration to occupancy map
    for current trial and position, add speed to speed map
    for current trial and position, add full list of spikes to spkmap
    every single time, add 1 to count for that position
    """
    for sample in nb.prange(len(value_to_sum)):
        if (speed[sample] > speed_threshold) and (speed[sample] < speed_max_threshold) and (dist_behave_to_frame[sample] < dist_cutoff):
            if scale_by_sample_duration:
                map_data[trial_idx[sample]][position_bin[sample]] += value_to_sum[sample] * sample_duration[sample]
            else:
                map_data[trial_idx[sample]][position_bin[sample]] += value_to_sum[sample]
            counts[trial_idx[sample]][position_bin[sample]] += 1


# ---------------------------------- numba code for speedups ---------------------------------
def replace_missing_data(data, firstValidBin, lastValidBin, replaceWith=np.nan):
    """switch to nan for any bins that the mouse didn't visit (excluding those in between visited bins)"""
    for trial, (fvb, lvb) in enumerate(zip(firstValidBin, lastValidBin)):
        data[trial, :fvb] = replaceWith
        data[trial, lvb + 1 :] = replaceWith
    return data


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


@nb.njit(parallel=True, cache=True)
def getMap(valueToSum, trialidx, positionbin, smap):
    """
    this is the fastest way to get a single summation map
    -- accepts 1d arrays value, trialidx, positionbin of the same size --
    -- shape determines the number of trials and position bins (they might not all be represented in trialidx or positionbin, or we could just do np.max()) --
    -- each value represents some number to be summed as a function of which trial it was in and which positionbin it was in --
    """
    for sample in nb.prange(len(valueToSum)):
        smap[trialidx[sample]][positionbin[sample]] += valueToSum[sample]


def correctMap(smap, amap, raise_error=False):
    """
    divide amap by smap with broadcasting where smap isn't 0 (with some handling of other cases)

    amap: [N, M, ...] "average map" where the values will be divided by relevant value in smap
    smap: [N, M] "summation map" which is used to divide out values in amap

    Why?
    ----
    amap is usually spiking activity or speed, and smap is usually occupancy. To convert temporal recordings
    to spatial maps, I start by summing up the values of speed/spiking in each position, along with summing
    up the time spent in each position. Then, with this method, I divide the summed speed/spiking by the time
    spent, to get an average (weighted) speed or spiking.

    correct amap by smap (where amap[i, j, r] output is set to amap[i, j, r] / smap[i, j] if smap[i, j]>0)

    if raise_error=True, then:
    - if smap[i, j] is 0 and amap[i, j, r]>0 for any r, will generate an error
    - if smap[i, j] is nan and amap[i, j, r] is not nan for any r, will generate an error
    otherwise,
    - sets amap to 0 wherever smap is 0
    - sets amap to nan wherever smap is nan

    function:
    correct a summation map (amap) by time spent (smap) if they were computed separately and the summation map should be averaged across time
    """
    zero_check = smap == 0
    nan_check = np.isnan(smap)
    if raise_error:
        if np.any(amap[zero_check] > 0):
            raise ValueError("amap was greater than 0 where smap was 0 in at least one location")
        if np.any(~np.isnan(amap[nan_check])):
            raise ValueError("amap was not nan where smap was nan in at least one location")
    else:
        amap[zero_check] = 0
        amap[nan_check] = np.nan

    # correct amap by smap and return corrected amap
    _numba_correctMap(smap, amap)
    return amap


@nb.njit(parallel=True, cache=True)
def _numba_correctMap(smap, amap):
    """
    correct amap by smap (where amap[i, j, r] output is set to amap[i, j, r] / smap[i, j] if smap[i, j]>0)
    """
    for t in nb.prange(smap.shape[0]):
        for p in nb.prange(smap.shape[1]):
            if smap[t, p] > 0:
                amap[t, p] /= smap[t, p]
