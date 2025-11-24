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

    def valid_frames(self) -> np.ndarray:
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
        }
        return pd.DataFrame(data)


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


@dataclass
class Placefield:
    placefield: np.ndarray
    dist_edges: np.ndarray
    environment: np.ndarray

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

    Returns
    -------
    Placefield
        The place field for the given neuron and environment.
    """
    if idx_to_spks is None:
        idx_to_spks = np.arange(len(spks))

    if len(idx_to_spks) != len(frame_behavior):
        raise ValueError("spks and frame_behavior must have the same length")

    # Filter only valid frames (with position data) and also filter spks
    idx_valid_frames = frame_behavior.valid_frames()

    # Filter target trials if provided
    if trial_filter is not None:
        idx_target_trials = np.isin(frame_behavior.trial, trial_filter)
        idx_valid_frames = idx_valid_frames & idx_target_trials

    # We need to know the number of ROIs to initialize the placefields
    num_rois = spks.shape[1]

    # Convert positions to bins
    frame_bins = convert_position_to_bins(frame_behavior.position, dist_edges, check_invalid=False)

    # Calculate whether the frame is "fast" e.g. if the mouse is moving above the speed threshold
    if speed_threshold is not None:
        fast_frame = frame_behavior.speed >= speed_threshold
    else:
        fast_frame = np.ones(len(frame_behavior), dtype=bool)

    if average:
        # Get the environment indices for each frame
        environment = np.unique(frame_behavior.environment)
        environment_indices = np.searchsorted(environment, frame_behavior.environment)
        num_env = len(environment)

        # Initialize the placefields and counts and compute them
        placefields = np.zeros((num_env, len(dist_edges) - 1, num_rois), dtype=spks.dtype)
        counts = np.zeros((num_env, len(dist_edges) - 1), dtype=int)
        _get_placefield_average(placefields, counts, spks, frame_bins, environment_indices, fast_frame, idx_valid_frames, idx_to_spks)

    else:
        # We need to get a list of environments included in each trial
        first_frame_in_trial = np.append(0, np.where(np.diff(frame_behavior.trial) == 1.0)[0] + 1)
        environment = frame_behavior.environment[first_frame_in_trial]

        # Then get the trial indices for each frame
        unique_trials = np.unique(frame_behavior.trial)
        trial_indices = np.searchsorted(unique_trials, frame_behavior.trial)
        num_trials = len(unique_trials)

        # Initialize the placefields and counts and compute them
        placefields = np.zeros((num_trials, len(dist_edges) - 1, num_rois), dtype=spks.dtype)
        counts = np.zeros((num_trials, len(dist_edges) - 1), dtype=int)
        _get_placefield_trials(placefields, counts, spks, frame_bins, trial_indices, fast_frame, idx_valid_frames, idx_to_spks)

    if zero_to_nan:
        idx_nan = counts == 0

    if smooth_width is not None:
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
        placefields[idx_nan] = np.nan

    return Placefield(placefield=placefields, dist_edges=dist_edges, environment=environment)


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
    frame_position_indices = convert_position_to_bins(frame_behavior.position, placefield.dist_edges, check_invalid=True)
    if np.any(~np.isin(frame_behavior.environment, placefield.environment)):
        raise ValueError(f"frame_behavior.environment contains environments that are not in the placefield.environment!!!")
    frame_environment_indices = np.searchsorted(placefield.environment, frame_behavior.environment)
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
def _get_placefield_average(
    placefield: np.ndarray,
    counts: np.ndarray,
    spks: np.ndarray,
    position_bin: np.ndarray,
    environment: np.ndarray,
    fast: np.ndarray,
    idx_valid_frames: np.ndarray,
    idx_to_spks: np.ndarray,
) -> None:
    """Get the average place field for a given neuron and environment.

    Uses numba speed up to get the place field for a given neuron and environment.
    """
    for sample in nb.prange(len(position_bin)):
        if fast[sample] and idx_valid_frames[sample]:
            frame_env = environment[sample]
            frame_bin = position_bin[sample]
            placefield[frame_env, frame_bin] += spks[idx_to_spks[sample]]
            counts[frame_env, frame_bin] += 1


@nb.njit(parallel=True)
def _get_placefield_trials(
    placefield: np.ndarray,
    counts: np.ndarray,
    spks: np.ndarray,
    position_bin: np.ndarray,
    trial: np.ndarray,
    fast: np.ndarray,
    idx_valid_frames: np.ndarray,
    idx_to_spks: np.ndarray,
) -> None:
    """Get the place field for a given neuron and trial.

    Uses numba speed up to get the place field for a given neuron and trial.
    """
    for sample in nb.prange(len(position_bin)):
        if fast[sample] and idx_valid_frames[sample]:
            frame_trial = trial[sample]
            frame_bin = position_bin[sample]
            placefield[frame_trial, frame_bin] += spks[idx_to_spks[sample]]
            counts[frame_trial, frame_bin] += 1


@nb.njit(parallel=True)
def _correct_placefield(placefield: np.ndarray, counts: np.ndarray) -> None:
    """Correct the place field by dividing by the counts.

    Uses numba speed up to correct the place field by dividing by the counts.
    """
    for ii in nb.prange(placefield.shape[0]):
        for jj in nb.prange(placefield.shape[1]):
            if counts[ii, jj] > 0:
                placefield[ii, jj] /= counts[ii, jj]
