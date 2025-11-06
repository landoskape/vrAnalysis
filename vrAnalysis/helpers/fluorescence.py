from tqdm import tqdm
import numpy as np
from numba import njit, prange
from .signals import percentile_filter
from _old_vrAnalysis import faststats


def get_transients(dff_std, threshold):
    if not isinstance(dff_std, np.ndarray) or dff_std.ndim != 1:
        raise ValueError("dff_std must be a 1D numpy array")

    putative_transients = np.diff(1.0 * (dff_std > threshold))
    negative_transients = np.diff(1.0 * (dff_std < -threshold))

    # Identify start and end of each transient
    pos_starts = np.where(putative_transients == 1)[0]
    pos_stops = np.where(putative_transients == -1)[0]
    neg_starts = np.where(negative_transients == 1)[0]
    neg_stops = np.where(negative_transients == -1)[0]

    if dff_std[0] > threshold:
        pos_starts = np.append(0, pos_starts)
    if dff_std[-1] > threshold:
        pos_stops = np.append(pos_stops, len(dff_std) - 1)
    if dff_std[0] < -threshold:
        neg_starts = np.append(0, neg_starts)
    if dff_std[-1] < -threshold:
        neg_stops = np.append(neg_stops, len(dff_std) - 1)

    # Return data
    return pos_starts, pos_stops, neg_starts, neg_stops


@njit(parallel=True, fastmath=True)
def handle_transient_edges(starts, stops, stitch_threshold=1, min_duration=2):
    if len(starts) != len(stops):
        raise ValueError("starts and stops must be the same length")

    if len(starts) == 0:
        return starts, stops

    # Stitch transients that are less than or equal to the stitch threshold apart
    if stitch_threshold is not None:
        keep_starts = np.ones(len(starts), dtype=np.bool_)
        keep_stops = np.ones(len(stops), dtype=np.bool_)
        for itransient in prange(len(starts) - 1):
            if starts[itransient + 1] - stops[itransient] <= stitch_threshold:
                keep_starts[itransient + 1] = False
                keep_stops[itransient] = False
        starts = starts[keep_starts]
        stops = stops[keep_stops]

    # Remove transients that are too short
    if min_duration is not None:
        durations = stops - starts
        valid_durations = durations >= min_duration
        starts = starts[valid_durations]
        stops = stops[valid_durations]

    return starts, stops


def get_standardized_dff(fcorr, fs, percentile=30, window_duration=60):
    # Compute baseline with percentile filter
    window_size = int(window_duration * fs)
    baseline = percentile_filter(fcorr, window_size, percentile)

    # Compute dF/F
    df = fcorr - baseline
    dff = df / baseline

    # Standardize dF/F
    dff_standardized = (dff - faststats.median(dff, axis=0)) / faststats.std(dff, axis=0)

    return dff_standardized


def get_significant_transients(dff, threshold_levels=np.arange(0.8, 4.2, 0.2), fpr_threshold=0.001, verbose=True, return_stats=False):

    # Prepare array to record significant transients
    num_thresholds = len(threshold_levels)
    significant_transients = np.zeros((dff.shape[0], dff.shape[1], num_thresholds), dtype=bool)

    if return_stats:
        fpr_stats = []

    # Compute putative transients
    progress = tqdm(threshold_levels, desc="Measuring transients at each threshold...", leave=True) if verbose else threshold_levels
    for ithreshold, threshold in enumerate(progress):

        if return_stats:
            fpr_stats.append({})

        # For each ROI, compute the FPR for each transient duration
        roi_progress = tqdm(range(dff.shape[1]), desc="Measuring each ROI...", leave=False) if verbose else range(dff.shape[1])
        for iroi in roi_progress:
            starts, stops, neg_starts, neg_stops = get_transients(dff[:, iroi], threshold)
            starts, stops = handle_transient_edges(starts, stops)
            neg_starts, neg_stops = handle_transient_edges(neg_starts, neg_stops)

            transient_significant = np.zeros(len(starts), dtype=bool)

            # Measure durations of the transients
            durations = stops - starts
            neg_durations = neg_stops - neg_starts
            unique_durations = np.unique(durations)

            # Estimate the FPR for each duration (in positive transients only)
            for duration in unique_durations:
                fpr = np.sum(neg_durations >= duration) / np.sum(durations >= duration)
                if fpr < fpr_threshold:
                    transient_significant[durations == duration] = True

                if return_stats:
                    fpr_stats[ithreshold]["positive_durations"] = durations
                    fpr_stats[ithreshold]["negative_durations"] = neg_durations
                    fpr_stats[ithreshold]["fpr"] = fpr

            # Record the significant transients
            for i, is_sig in enumerate(transient_significant):
                if is_sig:
                    significant_transients[starts[i] : stops[i], iroi, ithreshold] = True

    if return_stats:
        return significant_transients, fpr_stats
    else:
        return significant_transients
