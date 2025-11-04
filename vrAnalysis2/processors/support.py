import numpy as np
import numba as nb
import torch
from scipy.linalg import convolution_matrix


@nb.njit(parallel=True, fastmath=True, cache=True)
def median_zscore(spks: np.ndarray, median_subtract: bool = True) -> np.ndarray:
    """Standardize the spks by subtracting the median and dividing by the standard deviation."""
    output = np.zeros_like(spks)
    for iroi in nb.prange(spks.shape[1]):
        if median_subtract:
            median = np.median(spks[:, iroi])
        else:
            median = 0
        std = np.std(spks[:, iroi])
        if std == 0:
            output[:, iroi] = 0
        else:
            output[:, iroi] = (spks[:, iroi] - median) / std
    return output


def get_gauss_kernel(timestamps: np.ndarray, width: float, nonzero: bool = True) -> np.ndarray:
    """
    create gaussian kernel (sum=1) around the "timestamps" array with width in units of timestamps
    if nonzero=True, will remove zeros from the returned values (numerical zeros, obviously)
    """
    kernel_domain = timestamps - np.mean(timestamps)
    kernel = np.exp(-(kernel_domain**2) / (2 * width) ** 2)
    kernel = kernel / np.sum(kernel)
    if nonzero:
        # since sp.linalg.convolution_matrix only needs nonzero values, this is way faster
        kernel = kernel[kernel > 0]
    return kernel


@torch.no_grad()
def convolve_toeplitz(
    data: np.ndarray,
    kernel: np.ndarray,
    axis: int = -1,
    mode: str = "same",
    device: str = "cpu",
    force_torch: bool = False,
) -> np.ndarray:
    # convolve data on requested axis (default:-1) using a toeplitz matrix of kk
    # equivalent to np.convolve(data,kk,mode=mode) for each array on requested axis in data
    # uses torch for possible GPU speed up, default device set after imports
    assert -1 <= axis <= data.ndim, "requested axis does not exist"

    use_torch = isinstance(data, torch.Tensor)
    if use_torch:
        data = data.to(device).float()
        data = torch.moveaxis(data, axis, -1)
        data_shape = data.shape
        conv_mat = torch.tensor(convolution_matrix(kernel, data_shape[-1], mode=mode).T).to(device).float()
        data_reshaped = data.reshape(-1, data_shape[-1]).contiguous()
        output = torch.matmul(data_reshaped, conv_mat)
        new_data_shape = (*data_shape[:-1], conv_mat.shape[1])
        output = output.reshape(new_data_shape)
        if device == "cuda":
            # don't keep data on GPU
            del conv_mat, data_reshaped  # delete variables
            torch.cuda.empty_cache()  # clear memory
        return output.moveaxis(-1, axis)
    elif force_torch:
        data = np.moveaxis(data, axis, -1)  # move target axis
        data_shape = data.shape
        # if there are not many signals to convolve, this is a tiny slower
        # if there are many signals to convolve (order of ROIs in a recording), this is waaaaayyyy faster
        conv_mat = torch.tensor(convolution_matrix(kernel, data_shape[-1], mode=mode).T).to(device).float()
        data_reshaped = torch.tensor(np.reshape(data, (-1, data_shape[-1]))).to(device).float()
        output = torch.matmul(data_reshaped, conv_mat).cpu().numpy()
        new_data_shape = (*data_shape[:-1], conv_mat.shape[1])
        output = np.reshape(output, new_data_shape)
        if device == "cuda":
            # don't keep data on GPU
            del conv_mat, data_reshaped  # delete variables
            torch.cuda.empty_cache()  # clear memory
        return np.moveaxis(output, -1, axis)
    else:
        data = np.moveaxis(data, axis, -1)  # move target axis
        data_shape = data.shape
        conv_mat = convolution_matrix(kernel, data_shape[-1], mode=mode).T.astype(data.dtype)
        data_reshaped = np.reshape(data, (-1, data_shape[-1]))
        output = np.matmul(data_reshaped, conv_mat)
        new_data_shape = (*data_shape[:-1], conv_mat.shape[1])
        output = np.reshape(output, new_data_shape)
        return np.moveaxis(output, -1, axis)


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
    scale_by_sample_duration: bool,
    use_sample_to_value_idx: bool,
    sample_to_value_idx: np.ndarray,
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
    for sample in nb.prange(len(trial_idx)):
        if (speed[sample] > speed_threshold) and (speed[sample] < speed_max_threshold) and (dist_behave_to_frame[sample] < dist_cutoff):
            if use_sample_to_value_idx:
                if scale_by_sample_duration:
                    map_data[trial_idx[sample]][position_bin[sample]] += value_to_sum[sample_to_value_idx[sample]] * sample_duration[sample]
                else:
                    map_data[trial_idx[sample]][position_bin[sample]] += value_to_sum[sample_to_value_idx[sample]]
            else:
                if scale_by_sample_duration:
                    map_data[trial_idx[sample]][position_bin[sample]] += value_to_sum[sample] * sample_duration[sample]
                else:
                    map_data[trial_idx[sample]][position_bin[sample]] += value_to_sum[sample]
            counts[trial_idx[sample]][position_bin[sample]] += 1


@nb.njit(parallel=True, cache=True)
def get_aligned_summation_map(
    value_to_sum,
    trial_idx,
    position_bin,
    map_data,
    counts,
    speed,
    speed_threshold,
    speed_max_threshold,
    idx_valid,
):
    """
    this is the fastest way to get a single aligned summation map
    -- accepts 1d arrays value, trialidx, positionbin of the same size --
    -- shape determines the number of trials and position bins (they might not all be represented in trialidx or positionbin, or we could just do np.max()) --
    -- each value represents some number to be summed as a function of which trial it was in and which positionbin it was in --
    """
    for sample in nb.prange(len(trial_idx)):
        if idx_valid[sample] and (speed[sample] > speed_threshold) and (speed[sample] < speed_max_threshold):
            map_data[trial_idx[sample]][position_bin[sample]] += value_to_sum[sample]
            counts[trial_idx[sample]][position_bin[sample]] += 1


def correct_map(smap, amap, raise_error=False):
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
    _numba_correct_map(smap, amap)
    return amap


@nb.njit(parallel=True, cache=True)
def _numba_correct_map(smap, amap):
    """
    correct amap by smap (where amap[i, j, r] output is set to amap[i, j, r] / smap[i, j] if smap[i, j]>0)
    """
    for t in nb.prange(smap.shape[0]):
        for p in nb.prange(smap.shape[1]):
            if smap[t, p] > 0:
                amap[t, p] /= smap[t, p]


def replace_missing_data(data, firstValidBin, lastValidBin, replaceWith=np.nan):
    """switch to nan for any bins that the mouse didn't visit (excluding those in between visited bins)"""
    for trial, (fvb, lvb) in enumerate(zip(firstValidBin, lastValidBin)):
        data[trial, :fvb] = replaceWith
        data[trial, lvb + 1 :] = replaceWith
    return data
