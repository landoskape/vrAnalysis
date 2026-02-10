from typing import Union, Tuple, List, Optional
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import json
import numpy as np
import numba as nb
import pandas as pd
from .. import helpers
from ..sessions.b2session import B2Session
from .support import convert_position_to_bins, convolve_toeplitz, get_gauss_kernel, placefield_prediction_numba


@dataclass
class PlacefieldParams:
    """Parameters for place field processing.

    Contains configuration settings that control how place fields are processed,
    including distance steps, speed thresholds, and standardization options.

    Parameters
    ----------
    dist_step : float, default=1
        Step size for distance calculations in spatial units
    speed_threshold : float, default=1.0
        Minimum speed threshold for valid movement periods
    standardize_spks : bool, default=True
        Whether to standardize spike counts by dividing by the standard deviation
    smooth_width : float | None, default=1
        Width of the Gaussian smoothing kernel to apply to the maps (width in spatial units)
    reliability_method : str, default="leave_one_out"
        Method to use for calculating reliability
    """

    dist_step: float = 1.0
    speed_threshold: float = 1.0
    standardize_spks: bool = True
    smooth_width: Union[float, None] = 1.0
    reliability_method: str = "leave_one_out"

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
    def from_dict(cls, params_dict: dict) -> "PlacefieldParams":
        """Create a PlacefieldParams instance from a dictionary.

        Parameters
        ----------
        params_dict : dict
            Dictionary of parameter names and values. Missing parameters will
            use default values from PlacefieldParams.

        Returns
        -------
        PlacefieldParams
            New PlacefieldParams instance with values from the dictionary.
        """
        return cls(**{k: params_dict[k] for k in params_dict})

    @classmethod
    def from_path(cls, path: Path) -> "PlacefieldParams":
        """Create a PlacefieldParams instance from a JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON file containing parameter values.

        Returns
        -------
        PlacefieldParams
            New PlacefieldParams instance loaded from the JSON file.
        """
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def compare(self, other: "PlacefieldParams", filter_keys: Optional[List[str]] = None) -> bool:
        """Compare two PlacefieldParams instances.

        Parameters
        ----------
        other : PlacefieldParams
            Another PlacefieldParams instance to compare against.
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
        if self.smooth_width is not None and self.smooth_width <= 0:
            raise ValueError("smooth_width must be positive (can also be None)")
        # Convert floats to floats when not None
        self.dist_step = float(self.dist_step)
        self.speed_threshold = float(self.speed_threshold)
        self.smooth_width = float(self.smooth_width) if self.smooth_width is not None else None


def get_placefield_params(param_type: str, updates: Optional[dict] = None) -> PlacefieldParams:
    """Get the parameters for the PlacefieldProcessor

    Parameters
    ----------
    param_type : str
        The type of parameters to get
    updates : dict, optional
        A dictionary of parameters to update

    Returns
    -------
    params : PlacefieldParams
        The parameters for the PlacefieldProcessor
    """
    if param_type == "default":
        params = PlacefieldParams(
            dist_step=1.0,
            speed_threshold=1.0,
            standardize_spks=True,
            smooth_width=1.0,
            reliability_method="leave_one_out",
        )
    elif param_type == "smoothed":
        params = PlacefieldParams(
            dist_step=1.0,
            speed_threshold=1.0,
            standardize_spks=True,
            smooth_width=5.0,
            reliability_method="leave_one_out",
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


@dataclass
class FrameBehavior:
    position: np.ndarray
    speed: np.ndarray
    environment: np.ndarray
    trial: np.ndarray
    idx: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize idx to np.arange(num_samples) if not provided."""
        if self.idx is None:
            self.idx = np.arange(len(self.position))

    def valid_frames(self, full_check: bool = False) -> np.ndarray:
        if full_check:
            valid_position = ~np.isnan(self.position)
            valid_speed = ~np.isnan(self.speed)
            valid_environment = ~np.isnan(self.environment)
            valid_trial = ~np.isnan(self.trial)
            return valid_position & valid_speed & valid_environment & valid_trial
        else:
            return ~np.isnan(self.position)

    def filter_valid_frames(self) -> Tuple["FrameBehavior", np.ndarray]:
        idx_valid = self.valid_frames()
        return self.filter(idx_valid), idx_valid

    def filter(self, idx: np.ndarray) -> "FrameBehavior":
        return FrameBehavior(
            position=self.position[idx],
            speed=self.speed[idx],
            environment=self.environment[idx],
            trial=self.trial[idx],
            idx=self.idx[idx],
        )

    def position_by_environment(self) -> np.ndarray:
        """Get the position by environment.

        Returns
        -------
        position_by_environment : np.ndarray
            An array of shape (num_environments, num_frames) with the position for each frame in each environment.
            Na values are used for frames where no position data is available.
        """
        environments = np.unique(self.environment)
        position_by_environment = np.full((len(environments), len(self.position)), np.nan)
        for ienv, env in enumerate(environments):
            idx_env = self.environment == env
            position_by_environment[ienv, idx_env] = self.position[idx_env]
        return position_by_environment

    def __len__(self) -> int:
        return len(self.position)

    def __getitem__(self, idx) -> "FrameBehavior":
        """Allows indexing FrameBehavior like arrays to retrieve a subset of the data."""
        data = {
            "position": self.position[idx],
            "speed": self.speed[idx],
            "environment": self.environment[idx],
            "trial": self.trial[idx],
            "idx": self.idx[idx],
        }
        return pd.DataFrame(data)


@dataclass
class BehaviorImagingMapping:
    """Mapping between behavioral samples and imaging frames.

    Parameters
    ----------
    idx_behave_to_frame : np.ndarray
        Array mapping each behavioral sample to its corresponding imaging frame index.
        Shape: (num_behavioral_samples,)
    frame_timestamps : np.ndarray
        Timestamps for each imaging frame. Shape: (num_frames,)
    difference_timestamps : np.ndarray
        Absolute time difference between each behavioral sample timestamp and its
        corresponding imaging frame timestamp. Shape: (num_behavioral_samples,)
    sampling_period : float
        Median time between consecutive imaging frames.
    dist_cutoff : float
        Time cutoff for associating behavioral samples with imaging frames.
        Typically sampling_period / 2.
    """

    idx_behave_to_frame: np.ndarray
    frame_timestamps: np.ndarray
    difference_timestamps: np.ndarray
    sampling_period: float
    dist_cutoff: float


def get_frame_behavior(session: B2Session, clear_one_cache: bool = True) -> FrameBehavior:
    """Get position and environment data for each imaging frame.

    This method aligns behavioral data (position, speed, environment, trial)
    to imaging frame timestamps. Returns NaN for frames where no position
    data is available (e.g., if the closest behavioral sample is further
    away in time than half the sampling period).

    Parameters
    ----------
    session : B2Session
        The session to process.
    clear_one_cache : bool, optional
        Whether to clear the onefile cache after processing. Default is True.

    Returns
    -------
    frame_behavior : FrameBehavior
        The frame behavior data.
    """
    timestamps = session.loadone("positionTracking.times")
    position = session.loadone("positionTracking.position")
    idx_behave_to_frame = session.loadone("positionTracking.mpci")
    trial_start_index = session.loadone("trials.positionTracking")
    num_samples = len(position)
    trial_numbers = np.arange(len(trial_start_index))
    trial_lengths = np.append(np.diff(trial_start_index), num_samples - trial_start_index[-1])
    trial_numbers = np.repeat(trial_numbers, trial_lengths)
    trial_environment = session.loadone("trials.environmentIndex")
    trial_environment = np.repeat(trial_environment, trial_lengths)

    within_trial = np.append(np.diff(trial_numbers) == 0, True)
    sample_duration = np.append(np.diff(timestamps), 0)
    speed = np.append(np.diff(position) / sample_duration[:-1], 0)
    sample_duration = sample_duration * within_trial
    speed = speed * within_trial

    frame_timestamps = session.loadone("mpci.times")
    difference_timestamps = np.abs(timestamps - frame_timestamps[idx_behave_to_frame])
    sampling_period = np.median(np.diff(frame_timestamps))
    dist_cutoff = sampling_period / 2

    frame_position = np.zeros_like(frame_timestamps)
    count = np.zeros_like(frame_timestamps)
    helpers.get_average_frame_position(position, idx_behave_to_frame, difference_timestamps, dist_cutoff, frame_position, count)
    frame_position[count > 0] /= count[count > 0]
    frame_position[count == 0] = np.nan

    # Get a map from frame to behavior time for quick lookup
    idx_frame_to_behave, dist_frame_to_behave = helpers.nearestpoint(frame_timestamps, timestamps)
    idx_get_position = dist_frame_to_behave < dist_cutoff

    frame_environment = np.full(len(frame_timestamps), np.nan)
    frame_environment[idx_get_position] = trial_environment[idx_frame_to_behave[idx_get_position]]
    frame_environment[count == 0] = np.nan

    frame_trial = np.full(len(frame_timestamps), np.nan)
    frame_trial[idx_get_position] = trial_numbers[idx_frame_to_behave[idx_get_position]]
    frame_trial[count == 0] = np.nan

    # Compute speed only on in trial frames
    idx_valid = np.where(count != 0)[0]
    sub_in_trial = np.diff(frame_trial[idx_valid]) == 0
    sub_speed = np.diff(frame_position[idx_valid]) / np.diff(frame_timestamps[idx_valid])
    sub_speed[~sub_in_trial] = np.nan
    sub_speed = np.append(sub_speed, np.nan)
    frame_speed = np.full_like(frame_timestamps, np.nan)
    frame_speed[idx_valid] = sub_speed

    if clear_one_cache:
        session.clear_cache()

    return FrameBehavior(
        position=frame_position,
        speed=frame_speed,
        environment=frame_environment,
        trial=frame_trial,
    )


def get_session_behavior(session: B2Session, clear_one_cache: bool = True) -> Tuple[FrameBehavior, np.ndarray]:
    """Get position and environment data for each behavioral sample.

    This method returns behavioral data aligned to behavioral samples (not imaging frames).
    All behavioral samples are included with their computed trial numbers, environments,
    speeds, and sample durations.

    Parameters
    ----------
    session : B2Session
        The session to process.
    clear_one_cache : bool, optional
        Whether to clear the onefile cache after processing. Default is True.

    Returns
    -------
    behave_behavior : FrameBehavior
        The behavioral data aligned to behavioral samples. Contains:
        - position: Position for each behavioral sample
        - speed: Speed for each behavioral sample (NaN for last sample of each trial)
        - environment: Environment index for each behavioral sample
        - trial: Trial number for each behavioral sample
        - idx: Behavioral sample indices (0, 1, 2, ...)
    sample_duration : np.ndarray
        Duration of each behavioral sample. Last sample of each trial is set to 0.
        Shape: (num_behavioral_samples,)
    """
    timestamps = session.loadone("positionTracking.times")
    position = session.loadone("positionTracking.position")
    trial_start_index = session.loadone("trials.positionTracking")

    # Compute trial numbers for each behavioral sample
    num_samples = len(position)
    trial_numbers = np.arange(len(trial_start_index))
    trial_lengths = np.append(np.diff(trial_start_index), num_samples - trial_start_index[-1])
    trial_numbers_behave = np.repeat(trial_numbers, trial_lengths)
    trial_environment_behave = session.loadone("trials.environmentIndex")
    trial_environment_behave = np.repeat(trial_environment_behave, trial_lengths)

    # Compute within-trial mask and sample durations
    within_trial = np.append(np.diff(trial_numbers_behave) == 0, True)
    sample_duration = np.append(np.diff(timestamps), 0)
    speed = np.append(np.diff(position) / sample_duration[:-1], 0)
    sample_duration = sample_duration * within_trial  # Zero out last sample without valid duration
    speed = speed * within_trial  # Zero out speed for last sample of each trial

    if clear_one_cache:
        session.clear_cache()

    return (
        FrameBehavior(
            position=position,
            speed=speed,
            environment=trial_environment_behave,
            trial=trial_numbers_behave,
        ),
        sample_duration,
    )


def get_behavior_imaging_mapping(session: B2Session, clear_one_cache: bool = True) -> BehaviorImagingMapping:
    """Get mapping between behavioral samples and imaging frames.

    This function computes the temporal alignment between behavioral samples
    and imaging frames, including timestamps, differences, and cutoff thresholds.

    Parameters
    ----------
    session : B2Session
        The session to process.
    clear_one_cache : bool, optional
        Whether to clear the onefile cache after processing. Default is True.

    Returns
    -------
    mapping : BehaviorImagingMapping
        Dataclass containing:
        - idx_behave_to_frame: Mapping from behavioral samples to imaging frames
        - frame_timestamps: Timestamps for each imaging frame
        - difference_timestamps: Time differences between behavioral and frame timestamps
        - sampling_period: Median time between consecutive imaging frames
        - dist_cutoff: Time cutoff for associating samples with frames (sampling_period / 2)
    """
    timestamps = session.loadone("positionTracking.times")
    idx_behave_to_frame = session.loadone("positionTracking.mpci")
    frame_timestamps = session.loadone("mpci.times")

    difference_timestamps = np.abs(timestamps - frame_timestamps[idx_behave_to_frame])
    sampling_period = np.median(np.diff(frame_timestamps))
    dist_cutoff = sampling_period / 2

    if clear_one_cache:
        session.clear_cache()

    return BehaviorImagingMapping(
        idx_behave_to_frame=idx_behave_to_frame,
        frame_timestamps=frame_timestamps,
        difference_timestamps=difference_timestamps,
        sampling_period=sampling_period,
        dist_cutoff=dist_cutoff,
    )


@dataclass
class Placefield:
    placefield: np.ndarray
    dist_edges: np.ndarray
    environment: np.ndarray
    count: np.ndarray
    trials: np.ndarray | None = None
    idx_positions: np.ndarray | None = None

    def __post_init__(self):
        if self.idx_positions is None:
            self.idx_positions = np.arange(len(self.dist_edges) - 1)

    def __repr__(self) -> str:
        pfshape = self.placefield.shape
        summarize_edges = f"edges:{self.dist_edges[0]:.1f}-{self.dist_edges[-1]:.1f} ({len(self.dist_edges) - 1} bins)"
        return f"Placefield(placefield={pfshape}, {summarize_edges}, environment={np.unique(self.environment)})"

    def __getitem__(self, idx):
        return self.placefield[idx]

    def __len__(self) -> int:
        return len(self.placefield)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.placefield.shape

    def filter_by_environment(self, environment: int) -> "Placefield":
        idx_environment = self.environment == environment
        return Placefield(
            placefield=self.placefield[idx_environment],
            dist_edges=self.dist_edges,
            environment=self.environment[idx_environment],
            count=self.count[idx_environment],
            trials=self.trials[idx_environment] if self.trials is not None else None,
            idx_positions=self.idx_positions,
        )

    def filter_by_coverage(
        self,
        start_bins: int,
        end_bins: int,
        filter_positions: bool = True,
    ) -> "Placefield":
        """Filter by coverage, excluding those with insufficient valid bins.

        Filters out rows (either trials or environments) that have zero counts in any bin between start and end. Useful for keeping the central part of the linear track in which data is likely available.

        Parameters
        ----------
        start_bins : int
            Start bins where counts are not required.
        end_bins : int
            End bins where counts are required (inclusive!).
        filter_positions : bool
            Whether to filter the positions where any counts are 0. If False,
            only trials/environments (aka rows) are filtered. If True, will
            only filter positions where any counts are 0 -- independent of
            start and end bins, so it could include bins outside required range
            after filtering.

        Returns
        -------
        Placefield
            A new Placefield object with filtered rows. Trials with any zero count bins
            between start and end (inclusive) are excluded.
        """
        num_bins = self.count.shape[1]
        required_slice = slice(start_bins, num_bins - end_bins)
        required_counts = self.count[:, required_slice]
        valid_rows = np.all(required_counts > 0, axis=1)

        # Filter all arrays by valid rows
        new_placefield = Placefield(
            placefield=self.placefield[valid_rows],
            dist_edges=self.dist_edges,
            environment=self.environment[valid_rows],
            count=self.count[valid_rows],
            trials=self.trials[valid_rows] if self.trials is not None else None,
        )

        # Filter positions if requested
        if filter_positions:
            valid_columns = np.all(new_placefield.count > 0, axis=0)
            new_placefield.placefield = new_placefield.placefield[:, valid_columns]
            new_placefield.count = new_placefield.count[:, valid_columns]
            new_placefield.idx_positions = new_placefield.idx_positions[valid_columns]

        return new_placefield


def get_placefield(
    spks: np.ndarray,
    frame_behavior: FrameBehavior,
    dist_edges: np.ndarray,
    speed_threshold: Optional[float] = None,
    average: bool = True,
    idx_to_spks: Optional[np.ndarray] = None,
    trial_filter: Optional[Union[list, np.ndarray]] = None,
    smooth_width: Optional[float] = None,
    zero_to_nan: bool = False,
    use_fast_sampling: bool = False,
    by_sample_duration: bool = True,
    session: Optional[B2Session] = None,
) -> Placefield:
    """Get the place field over spks for each frame in frame_behavior.

    Note that the shape of the resulting place field is (num_environments/num_trials, num_bins, num_rois).
    The num_bins and num_rois are given by dist_edges and spks.shape[1], respectively. The first dimension
    is determined based on the structure of frame_behavior: if average=True, then num_environments is the
    number of unique environments in frame_behavior.environment, otherwise it is the number of unique trials
    in frame_behavior.trial. The placefield.environment refers to the environment over each row (1st dim) of
    the placefield.placefield array, whether it's an average across trials or each trial separately.

    Critically, if frame_behavior.environment does not contain all environments in the session, then the placefield
    will not be able to be computed for those environments and the shape might be not what you expect! Fixing shapes
    is the responsibility of the caller where required to keep this function agnostic to the session structure.

    Parameters
    ----------
    spks : np.ndarray
        The spike counts for the given neuron. (Frames x ROIs)
    frame_behavior : FrameBehavior
        The frame behavior data. (Frames x 1)
    dist_edges : np.ndarray
        The edges of the distance bins. (N_bins + 1)
    speed_threshold : Optional[float]
        The speed threshold for the place field. Default is None, indicating that no speed threshold will be used.
    average : bool
        Whether to average the place field over trials. Default is True.
    idx_to_spks : Optional[np.ndarray]
        If the frame_behavior is filtered to only include some frames, this indicates which frames are kept.
        It will be used to efficiently index into the full spks array. Default is None, indicating that the
        frame_behavior is not filtered and has the same length as the spks array.
    trial_filter : Optional[Union[list, np.ndarray]]
        If included, will be used to filter which trials are included in the analysis.
    smooth_width : Optional[float]
        The width of the Gaussian kernel to use for smoothing the place field. If provided, will perform
        symmetric gaussian smoothing of the place field by convolving both the accumulated spikes and
        counts, then dividing the spikes by the counts. Default is None, indicating no smoothing will be done.
    zero_to_nan : bool
        Whether to convert zero count frames to NaNs in the place field. Default is False.
    use_fast_sampling : bool
        If True, iterates over behavioral samples (faster sampling) instead of imaging frames.
        Requires session parameter to load behavioral data. Default is False.
    by_sample_duration : bool
        If True (and use_fast_sampling=True), weights accumulation by sample duration.
        If False, uses equal weighting. Default is True.
    session : Optional[B2Session]
        Required when use_fast_sampling=True. Used to load behavioral data for fast sampling mode.
        Default is None.

    Returns
    -------
    Placefield
        The place field for the given neuron and environment.
    """
    # Validation checks
    if len(spks) == 0:
        raise ValueError("spks array cannot be empty")
    if len(frame_behavior) == 0:
        raise ValueError("frame_behavior cannot be empty")
    if len(dist_edges) < 2:
        raise ValueError("dist_edges must have at least 2 elements (need at least 1 bin)")
    if spks.shape[1] == 0:
        raise ValueError("spks must have at least one ROI (spks.shape[1] > 0)")
    if use_fast_sampling and session is None:
        raise ValueError("session parameter is required when use_fast_sampling=True")

    if idx_to_spks is None:
        idx_to_spks = np.arange(len(spks))

    if len(idx_to_spks) != len(frame_behavior):
        raise ValueError("spks and frame_behavior must have the same length")

    # Filter only valid frames (with position data) and also filter spks
    idx_valid_frames = frame_behavior.valid_frames(full_check=True)
    if not np.any(idx_valid_frames):
        raise ValueError("frame_behavior contains no valid (non-NaN) frames")

    # Filter target trials if provided
    if trial_filter is not None:
        idx_target_trials = np.isin(frame_behavior.trial, trial_filter)
        idx_valid_frames = idx_valid_frames & idx_target_trials

    # Calculate whether the frame is "fast" e.g. if the mouse is moving above the speed threshold
    if speed_threshold is not None:
        fast_frame = frame_behavior.speed >= speed_threshold
    else:
        fast_frame = np.ones(len(frame_behavior), dtype=bool)

    # We need to know the number of ROIs to initialize the placefields
    num_rois = spks.shape[1]

    if use_fast_sampling:
        # Get information related to fast behavior sampling
        fast_behavior, sample_duration = get_session_behavior(session)
        mapping = get_behavior_imaging_mapping(session)

        # Convert behavioral positions to bins
        behave_position_bin = convert_position_to_bins(fast_behavior.position, dist_edges, check_invalid=False)

        # True frame indices of full session might not map to indices in frame_behavior if frame_behavior comes pre-filtered.
        # Therefore, we need to convert it with a lookup.
        bad_value = -1000
        max_frame_idx = max(np.max(mapping.idx_behave_to_frame), np.max(frame_behavior.idx))
        lookup = np.full(max_frame_idx + 1, bad_value, dtype=int)

        # This gives the index to frame_behavior frame like a dictionary
        lookup[frame_behavior.idx] = np.arange(len(frame_behavior.idx))

        # Convert frame indices to frame_behavior indices
        idx_behave_to_frame = lookup[mapping.idx_behave_to_frame]

        # Get valid fast samples
        idx_valid_fast_samples = fast_behavior.valid_frames() & (idx_behave_to_frame != bad_value)
        idx_valid_fast_samples &= idx_valid_frames[idx_behave_to_frame]

        environments, trials, row_indices, num_rows = _prepare_row_indices(average, fast_behavior, idx_valid_fast_samples)

        # Initialize the placefields and counts and compute them
        placefields = np.zeros((num_rows, len(dist_edges) - 1, num_rois), dtype=spks.dtype)
        counts = np.zeros((num_rows, len(dist_edges) - 1), dtype=spks.dtype)
        _get_placefield_fast_sampling(
            placefields,
            counts,
            spks,
            behave_position_bin,
            row_indices,
            idx_behave_to_frame,
            mapping.difference_timestamps,
            mapping.dist_cutoff,
            idx_valid_fast_samples,
            idx_valid_frames,
            fast_frame,
            sample_duration,
            by_sample_duration,
            idx_to_spks,
        )

    else:
        # If not use fast sampling, go with the frame-centric approach

        # Convert positions to bins
        frame_bins = convert_position_to_bins(frame_behavior.position, dist_edges, check_invalid=False)

        environments, trials, row_indices, num_rows = _prepare_row_indices(average, frame_behavior, idx_valid_frames)

        # Initialize the placefields and counts and compute them
        placefields = np.zeros((num_rows, len(dist_edges) - 1, num_rois), dtype=spks.dtype)
        counts = np.zeros((num_rows, len(dist_edges) - 1), dtype=spks.dtype)
        _get_placefield(
            placefields,
            counts,
            spks,
            frame_bins,
            row_indices,
            fast_frame,
            idx_valid_frames,
            idx_to_spks,
        )

    if smooth_width is not None and smooth_width > 0:
        # Get a gaussian kernel to smooth the place fields
        kernel = get_gauss_kernel(helpers.edge2center(dist_edges), smooth_width)

        # Compute spike rate by dividing placefields by counts
        _correct_placefield(placefields, counts)

        # Compute the valid indices for real measurements (zero means no data for that bin)
        valid = (counts > 0).astype(float)

        # Smooth the spike rate and valid indices for normalized measure of smoothed activity
        # (This prevents inflation via sparse denominator when missing data is present)
        placefields = convolve_toeplitz(placefields, kernel, axis=1)
        counts = convolve_toeplitz(valid, kernel, axis=1)

    # Correct the place field by dividing by the counts
    _correct_placefield(placefields, counts)

    if zero_to_nan:
        placefields[counts == 0] = np.nan

    return Placefield(placefield=placefields, dist_edges=dist_edges, environment=environments, count=counts, trials=trials)


def get_placefield_prediction(placefield: Placefield, frame_behavior: FrameBehavior) -> tuple[np.ndarray, dict]:
    """Predict neural activity from place field maps.

    This method uses averaged environment maps to predict neural activity
    at each imaging frame based on the animal's position and environment.

    Parameters
    ----------
    placefield : Placefield
        The place field to use for prediction.
    frame_behavior : FrameBehavior
        The frame behavior to use for prediction.

    Returns
    -------
    np.ndarray
        The predicted activity array with shape (frames, rois). NaN for frames where prediction is not possible.
    """
    idx_valid_frames = frame_behavior.valid_frames()
    num_frames = len(frame_behavior)
    frame_behavior_valid = frame_behavior.filter(idx_valid_frames)
    frame_position_indices = np.full(num_frames, -10000, dtype=int)
    frame_position_indices[idx_valid_frames] = convert_position_to_bins(frame_behavior_valid.position, placefield.dist_edges, check_invalid=True)
    if np.any(~np.isin(frame_behavior_valid.environment, placefield.environment)):
        raise ValueError(f"frame_behavior.environment contains environments that are not in the placefield.environment!!!")
    frame_environment_indices = np.full(num_frames, -10000, dtype=int)
    frame_environment_indices[idx_valid_frames] = np.searchsorted(placefield.environment, frame_behavior_valid.environment)
    num_rois = placefield.shape[2]

    # Use a numba speed up to get the placefield prediction (single pass simple algorithm)
    placefield_prediction = np.full((len(frame_behavior), num_rois), np.nan)
    placefield_prediction = placefield_prediction_numba(
        placefield_prediction,
        placefield.placefield,
        frame_environment_indices,
        frame_position_indices,
        idx_valid_frames,
    )

    idx_valid = np.all(~np.isnan(placefield_prediction), axis=1)

    # Return the extras as a dictionary for forward compatibility
    extras = dict(idx_valid=idx_valid)

    return placefield_prediction, extras


@nb.njit(parallel=True)
def _get_placefield(
    placefield: np.ndarray,
    counts: np.ndarray,
    spks: np.ndarray,
    position_bin: np.ndarray,
    trial_or_environment: np.ndarray,
    fast: np.ndarray,
    idx_valid_frames: np.ndarray,
    idx_to_spks: np.ndarray,
) -> None:
    """Get the average place field for a given neuron and environment.

    Uses numba speed up to get the place field for a given neuron and environment.
    """
    for sample in nb.prange(len(position_bin)):
        if fast[sample] and idx_valid_frames[sample]:
            row_idx = trial_or_environment[sample]
            if row_idx < 0 or row_idx >= placefield.shape[0]:
                continue
            frame_bin = position_bin[sample]
            if frame_bin < 0 or frame_bin >= placefield.shape[1]:
                continue
            placefield[row_idx, frame_bin] += spks[idx_to_spks[sample]]
            counts[row_idx, frame_bin] += 1


@nb.njit(parallel=True)
def _correct_placefield(placefield: np.ndarray, counts: np.ndarray) -> None:
    """Correct the place field by dividing by the counts.

    Uses numba speed up to correct the place field by dividing by the counts.
    """
    for ii in nb.prange(placefield.shape[0]):
        for jj in nb.prange(placefield.shape[1]):
            if counts[ii, jj] > 0:
                placefield[ii, jj] /= counts[ii, jj]


@nb.njit(parallel=True)
def _get_placefield_fast_sampling(
    placefield: np.ndarray,
    counts: np.ndarray,
    spks: np.ndarray,
    behave_position_bin: np.ndarray,
    behave_env_or_trial_indices: np.ndarray,
    idx_behave_to_frame: np.ndarray,
    difference_timestamps: np.ndarray,
    dist_cutoff: float,
    idx_valid_samples: np.ndarray,
    idx_valid_frames: np.ndarray,
    fast_frame: np.ndarray,
    sample_duration: np.ndarray,
    by_sample_duration: bool,
    idx_to_spks: np.ndarray,
) -> None:
    """Get the average place field using fast sampling (iterating over behavioral samples).

    Iterates over behavioral samples, checks if the closest frame is valid according to
    idx_valid_frames and fast_frame, and accumulates spikes if valid.

    Parameters
    ----------
    placefield : np.ndarray
        Output array to accumulate spikes. Shape: (num_env, num_bins, num_rois)
    counts : np.ndarray
        Output array to accumulate counts. Shape: (num_env, num_bins)
    spks : np.ndarray
        Spike data. Shape: (num_frames, num_rois)
    behave_position_bin : np.ndarray
        Position bin index for each behavioral sample.
    behave_env_or_trial_indices : np.ndarray
        Environment index for each behavioral sample.
    idx_behave_to_frame : np.ndarray
        Index into frame_behavior for each behavioral sample (-1 if frame not in frame_behavior).
    difference_timestamps : np.ndarray
        Temporal distance between behavioral sample and associated frame.
    dist_cutoff : float
        Maximum allowed distance for valid association.
    idx_valid_samples : np.ndarray
        Boolean mask indicating which samples are valid (position data available, trial filter passed).
    idx_valid_frames : np.ndarray
        Boolean mask indicating which frames are valid (position data available, trial filter passed).
    fast_frame : np.ndarray
        Boolean mask indicating which frames pass speed threshold.
    sample_duration : np.ndarray
        Duration of each behavioral sample.
    by_sample_duration : bool
        If True, weight accumulation by sample_duration. If False, use equal weighting.
    idx_to_spks : np.ndarray
        Mapping from frame indices to spks array indices.
    """
    for sample in nb.prange(len(behave_position_bin)):
        # Skip if sample is not valid
        if not idx_valid_samples[sample]:
            continue

        # Check if behavioral sample is within cutoff distance of its associated frame
        if difference_timestamps[sample] >= dist_cutoff:
            continue

        # Get the frame index in frame_behavior associated with this behavioral sample
        frame_idx = idx_behave_to_frame[sample]

        # Skip if frame is not in frame_behavior (invalid mapping)
        if frame_idx < 0:
            continue

        # Check if the frame is valid (position data available, trial filter passed, speed threshold passed)
        if not (idx_valid_frames[frame_idx] and fast_frame[frame_idx]):
            continue

        # Get environment/trial and position bin for this sample
        row_idx = behave_env_or_trial_indices[sample]
        if row_idx < 0:  # Skip if environment/trial not in frame_behavior
            continue

        pos_bin = behave_position_bin[sample]
        if pos_bin < 0 or pos_bin >= placefield.shape[1]:  # Skip invalid bins
            continue

        # Get spike data for the associated frame
        spk_idx = idx_to_spks[frame_idx]

        # Accumulate spikes and counts
        if by_sample_duration:
            weight = sample_duration[sample]
            placefield[row_idx, pos_bin] += spks[spk_idx] * weight
            counts[row_idx, pos_bin] += weight
        else:
            placefield[row_idx, pos_bin] += spks[spk_idx]
            counts[row_idx, pos_bin] += 1


def _prepare_row_indices(average: bool, sample_behavior: FrameBehavior, idx_valid_samples: np.ndarray):
    if average:
        # We need the row index to follow the environment index
        # - indexing the sorted environments
        # - not necessarily 0 indexed or consecutive
        _environment_with_nan = np.unique(sample_behavior.environment[idx_valid_samples])
        environments = _environment_with_nan[~np.isnan(_environment_with_nan)]
        row_indices = np.searchsorted(environments, sample_behavior.environment)
        num_rows = len(environments)
        if num_rows == 0:
            raise ValueError("No valid environments found")

        trials = None

    else:
        # Then we don't average trials within environment
        # So row index follows trial number
        # - indexing sorted trials,
        # - not necessarily 0 indexed or consecutive
        _trial_with_nan = np.unique(sample_behavior.trial[idx_valid_samples])
        trials = _trial_with_nan[~np.isnan(_trial_with_nan)].astype(int)
        row_indices = np.searchsorted(trials, sample_behavior.trial)
        num_rows = len(trials)

        if num_rows == 0:
            raise ValueError("No valid trials found")

        environments, confirmed = _get_env_of_each_trial(
            sample_behavior.trial,
            sample_behavior.environment,
            np.ones(len(sample_behavior), dtype=bool),
            trials,
        )

        if not np.all(confirmed):
            raise ValueError("Not all trials got an environment")

    return environments, trials, row_indices, num_rows


def _get_env_of_each_trial(
    frame_trial: np.ndarray,
    frame_environment: np.ndarray,
    idx_frame_valid: np.ndarray,
    trials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    environments = np.full(len(trials), -1, dtype=int)
    confirmed = np.zeros(len(trials), dtype=bool)

    # Compute trial_indices mapping
    trial_indices = np.full(max(trials) + 1, -1, dtype=int)
    trial_indices[trials] = np.arange(len(trials))

    # Iterate through valid frames to find first environment for each trial
    for idx in range(len(idx_frame_valid)):
        if not idx_frame_valid[idx]:
            continue

        _trial_number = frame_trial[idx]
        if np.isnan(_trial_number):
            continue

        if _trial_number < 0 or _trial_number >= len(trial_indices):
            continue

        _trial_index = trial_indices[int(_trial_number)]

        if confirmed[_trial_index]:
            continue

        _environment_number = frame_environment[idx]
        if np.isnan(_environment_number):
            continue

        environments[_trial_index] = int(_environment_number)
        confirmed[_trial_index] = True

        if np.all(confirmed):
            break

    return environments, confirmed
