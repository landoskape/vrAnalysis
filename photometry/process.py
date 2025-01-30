import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy import ndimage
from vrAnalysis.helpers import nearestpoint


def resample_with_antialiasing(data, time, new_sampling_rate, filter_order=8):
    """
    Resample data with anti-aliasing filter to prevent aliasing artifacts.

    Parameters:
    -----------
    data : array-like
        Input signal to be resampled
    time : array-like
        Original time points
    new_sampling_rate : float
        Target sampling rate in Hz
    filter_order : int
        Order of the anti-aliasing filter

    Returns:
    --------
    resampled_data : array
        Resampled signal
    new_time : array
        New time points
    """
    # Calculate original approximate sampling rate
    orig_sampling_rate = 1 / np.median(np.diff(time))

    # Design anti-aliasing filter
    nyquist_freq = new_sampling_rate / 2

    # Only apply filter if we're downsampling
    if new_sampling_rate < orig_sampling_rate:
        # Normalize cutoff frequency to Nyquist frequency
        cutoff_freq = nyquist_freq / (orig_sampling_rate / 2)

        # Apply Butterworth low-pass filter
        b, a = butter(filter_order, cutoff_freq, btype="low")
        filtered_data = filtfilt(b, a, data)
    else:
        filtered_data = data

    # Create new time points
    min_time = np.min(time) + 1e-3
    max_time = np.max(time) - 1e-3
    period = 1 / new_sampling_rate
    num_periods = int((max_time - min_time) / period)
    new_time = np.linspace(min_time, max_time, num_periods)

    # Interpolate filtered data
    resampled_data = interp1d(time, filtered_data, bounds_error=False, fill_value="extrapolate")(new_time)

    return resampled_data, new_time


def _filter_cycle_markers(markers, first, last, keep_first=False, keep_last=False):
    """Filter cycle markers to only include valid cycles."""
    if keep_first:
        markers = markers[markers >= first]
    else:
        markers = markers[markers > first]
    if keep_last:
        markers = markers[markers <= last]
    else:
        markers = markers[markers < last]
    return markers


def get_cycles(data, cycle_period_tolerance=0.1):
    """Get output cycles with interleaved data on out1 and out2.

    Find start stop indices for each cycle.
    Check that the cycles are interleaved correctly.
    Return the start and stop indices for each cycle.
    Return an index to all samples within the target cycles.

    Target cycle definition:
    First cycle is always on out1 (will clip if necessary) - last cycle is out2.
    """
    diff1 = np.diff(data["out1"])
    diff2 = np.diff(data["out2"])
    start1 = np.where(diff1 == 1)[0] + 1
    start2 = np.where(diff2 == 1)[0] + 1
    stop1 = np.where(diff1 == -1)[0] + 1
    stop2 = np.where(diff2 == -1)[0] + 1
    first_valid_idx = start1[0]
    last_valid_idx = stop2[-1]

    start1 = _filter_cycle_markers(start1, first_valid_idx, last_valid_idx, keep_first=True)
    start2 = _filter_cycle_markers(start2, first_valid_idx, last_valid_idx)
    stop1 = _filter_cycle_markers(stop1, first_valid_idx, last_valid_idx)
    stop2 = _filter_cycle_markers(stop2, first_valid_idx, last_valid_idx, keep_last=True)

    start1 = _filter_cycle_markers(start1, first_valid_idx, stop1[-1], keep_first=True)
    stop2 = _filter_cycle_markers(stop2, start2[0], last_valid_idx, keep_last=True)

    if len(start1) != len(start2):
        raise ValueError("Unequal number of start markers")
    if len(stop1) != len(stop2):
        raise ValueError("Unequal number of stop markers")
    if len(start1) != len(stop1):
        raise ValueError("Unequal number of start and stop markers")
    if not np.all(start1 < stop1):
        raise ValueError("Start marker after stop marker for channel 1")
    if not np.all(start2 < stop2):
        raise ValueError("Start marker after stop marker for channel 2")

    period1 = stop1 - start1
    period2 = stop2 - start2
    period1_deviation = period1 / np.mean(period1)
    period2_deviation = period2 / np.mean(period2)
    bad_period1 = np.abs(period1_deviation - 1) > cycle_period_tolerance
    bad_period2 = np.abs(period2_deviation - 1) > cycle_period_tolerance
    if np.sum(np.diff(np.where(bad_period1)[0]) < 2) > 2:
        raise ValueError("Too many consecutive bad periods in channel 1")
    if np.sum(np.diff(np.where(bad_period2)[0]) < 2) > 2:
        raise ValueError("Too many consecutive bad periods in channel 2")

    # Remove bad periods and filter stop / start signals
    valid_period = ~bad_period1 & ~bad_period2
    start1 = start1[valid_period]
    stop1 = stop1[valid_period]
    start2 = start2[valid_period]
    stop2 = stop2[valid_period]

    if not np.all(data["out1"][start1] == 1) or not np.all(data["out2"][start2] == 1):
        raise ValueError("Start indices are not positive for out1 / out2!")
    if not np.all(data["out1"][stop1] == 0) or not np.all(data["out2"][stop2] == 0):
        raise ValueError("Stop indices are not zero for out1 / out2!")

    return start1, stop1, start2, stop2


def get_opto_cycles(data, min_period=1, cycle_period_tolerance=0.01):
    """Get opto cycles (out3) with a minimum period.

    Returns the start times for each cycle and an average cycle signal.
    """
    diff3 = np.diff(data["out3"])
    start3 = np.where(diff3 == 1)[0] + 1
    stop3 = np.where(diff3 == -1)[0] + 1
    first_valid_idx = start3[0]
    last_valid_idx = stop3[-1]
    start3 = _filter_cycle_markers(start3, first_valid_idx, last_valid_idx, keep_first=True)
    start_time = data["out_time"][start3]

    valid_starts = [start3[0]]
    valid_times = [start_time[0]]

    for i in range(1, len(start3)):
        if start_time[i] > (valid_times[-1] + min_period):
            valid_starts.append(start3[i])
            valid_times.append(start_time[i])

    # Convert valid starts to numpy array (reuse start3 for consistent terminology with get_cycles)
    start3 = np.array(valid_starts)

    # Measure period between cycles
    period3 = start3[1:] - start3[:-1]
    period3_deviation = period3 / np.mean(period3)
    if not np.all(period3_deviation >= 1 - cycle_period_tolerance) and np.all(period3_deviation <= 1 + cycle_period_tolerance):
        min_period = np.min(period3)
        max_period = np.max(period3)
        raise ValueError(f"Excess period variation in opto cycles! min={min_period:.2f}, max={max_period:.2f}")

    min_period = np.min(period3)
    stop3 = start3 + min_period

    if stop3[-1] >= len(data["out3"]):
        start3 = start3[:-1]
        stop3 = stop3[:-1]

    cycles = []
    for istart, istop in zip(start3, stop3):
        cycles.append(data["out3"][istart:istop])
    average_cycle = np.mean(np.stack(cycles), axis=0)

    return start3, stop3, average_cycle


def get_cycle_data(signal, start, stop, keep_fraction=0.5, signal_cv_tolerance=0.05):
    """Extract cycle data from a signal."""
    num_samples = len(start)
    assert keep_fraction > 0 and keep_fraction < 1, "Invalid keep_fraction, must be in between 0 and 1"
    assert num_samples == len(stop), "Start and stop indices mismatch"
    cycle_data = []
    invalid_cycle = []
    for i in range(num_samples):
        c_stop = stop[i] - 1
        c_start = start[i] + int(keep_fraction * (c_stop - start[i]))
        cycle_signal = signal[c_start:c_stop]
        cycle_cv = np.std(cycle_signal) / np.mean(cycle_signal)
        invalid_cycle.append(cycle_cv > signal_cv_tolerance)
        cycle_data.append(signal[c_start:c_stop])
    cycle_data = np.array([np.mean(cd) for cd in cycle_data])
    return cycle_data, np.array(invalid_cycle)


def analyze_data(
    data,
    preperiod=0.1,
    postperiod=1.0,
    cycle_period_tolerance=0.5,
    keep_fraction=0.5,
    signal_cv_tolerance=0.05,
    sampling_rate=1000,
    filter_func=None,
):
    """Process a data file, return results and filtered signals."""
    # First check if the data is valid and meets criteria for processing.
    num_samples = len(data["in_data"])
    if not num_samples > 0:
        raise ValueError("No data found! in_data has 0 samples.")
    for key in ["out1", "out2", "out3"]:
        assert num_samples == len(data[key]), f"{key} and in_data length mismatch"
        uvals = np.unique(data[key])
        if not np.array_equal(uvals, np.array([0.0, 1.0])):
            raise ValueError(f"Invalid values in {key}: {uvals}")
    for key in ["in_time", "out_time"]:
        assert num_samples == len(data[key]), f"{key} and in_data length mismatch"

    # Get start and top indices for the interleaved cycles
    time = data["in_time"]
    start1, stop1, start2, stop2 = get_cycles(data, cycle_period_tolerance=cycle_period_tolerance)
    cycle_timestamps = (time[stop2] + time[start1]) / 2  # Midpoint of full cycles
    in1, invalid1 = get_cycle_data(data["in_data"], start1, stop1, keep_fraction=keep_fraction, signal_cv_tolerance=signal_cv_tolerance)
    in2, invalid2 = get_cycle_data(data["in_data"], start2, stop2, keep_fraction=keep_fraction, signal_cv_tolerance=signal_cv_tolerance)

    if np.any(invalid1) or np.any(invalid2):
        print(
            f"Warning: excess co. of var. detected for {np.sum(invalid1)/num_samples*100:.2f}% of cycles are invalid for channel 1 and {np.sum(invalid2)/num_samples*100:.2f}% for channel 2."
        )

    # Resample the data to the new sampling rate
    data_in1, time_data = resample_with_antialiasing(in1, cycle_timestamps, sampling_rate)
    data_in2, in2_time_rs = resample_with_antialiasing(in2, cycle_timestamps, sampling_rate)

    # Do a 50% percentile filter over 1 second to capture the drift in baseline fluorescence
    data_in1 = data_in1 - ndimage.percentile_filter(data_in1, 50, size=1000)
    data_in2 = data_in2 - ndimage.percentile_filter(data_in2, 50, size=1000)

    if filter_func is not None:
        data_in1 = filter_func(data_in1)
        data_in2 = filter_func(data_in2)

    if not np.allclose(time_data, in2_time_rs):
        raise ValueError("Inconsistent time stamps for in1 and in2")

    # Get start indices and times for opto cycles
    start3, stop3, _ = get_opto_cycles(data, min_period=1.0, cycle_period_tolerance=cycle_period_tolerance)
    opto_start_time = data["out_time"][start3]

    # And also get the time of opto start / stops in the new sampling rate
    # returns index of y closest to each point in x and distance between points
    start3, error_start3 = nearestpoint(opto_start_time, time_data)

    # Interpolate opto data to in_data timestamps
    opto_data = interp1d(data["out_time"], data["out3"], bounds_error=False, fill_value="extrapolate")(time_data)

    # Get cycle data for opto cycles
    in1_opto = []
    in2_opto = []
    out3_opto = []
    time_opto = []
    samples_pre = int(preperiod * sampling_rate)
    samples_post = int(postperiod * sampling_rate)
    for istart in start3:
        in1_opto.append(data_in1[istart - samples_pre : istart + samples_post])
        in2_opto.append(data_in2[istart - samples_pre : istart + samples_post])
        out3_opto.append(opto_data[istart - samples_pre : istart + samples_post])

        # Relative time... should always be the same actually
        time_opto.append(time_data[istart - samples_pre : istart + samples_post] - time_data[istart])

    in1_opto = np.stack(in1_opto)
    in2_opto = np.stack(in2_opto)
    out3_opto = np.stack(out3_opto)
    time_opto = np.mean(np.stack(time_opto), axis=0)  # variance across opto cycles should be within sample error

    results = dict(
        in1_opto=in1_opto,
        in2_opto=in2_opto,
        out3_opto=out3_opto,
        time_opto=time_opto,
        opto_start_time=opto_start_time - time_data[0],
        data_in1=data_in1,
        data_in2=data_in2,
        data_opto=opto_data,
        time_data=time_data - time_data[0],
    )

    return results
