import numpy as np
import numba as nb


@nb.njit(parallel=True)
def _numba_ptp(data):
    """numba speed up for ptp"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.ptp(data[n])
    return output


@nb.njit(parallel=True)
def _numba_percentile(data, q):
    """numba speed up for percentile"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.percentile(data[n], q)
    return output


@nb.njit(parallel=True)
def _numba_nanpercentile(data, q):
    """numba speed up for nanpercentile"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.nanpercentile(data[n], q)
    return output


@nb.njit(parallel=True)
def _numba_quantile(data, q):
    """numba speed up for quantile"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.quantile(data[n], q)
    return output


@nb.njit(parallel=True)
def _numba_nanquantile(data, q):
    """numba speed up for nanquantile"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.nanquantile(data[n], q)
    return output


@nb.njit(parallel=True)
def _numba_median(data):
    """numba speed up for median"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.median(data[n])
    return output


@nb.njit(parallel=True)
def _numba_average(data):
    """numba speed up for average"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.average(data[n])
    return output


@nb.njit(parallel=True)
def _numba_mean(data):
    """numba speed up for mean"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.mean(data[n])
    return output


@nb.njit(parallel=True)
def _numba_std(data):
    """numba speed up for nanstd"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.std(data[n])
    return output


@nb.njit(parallel=True)
def _numba_var(data):
    """numba speed up for nanvar"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.var(data[n])
    return output


@nb.njit(parallel=True)
def _numba_nanmedian(data):
    """numba speed up for nanmedian"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.nanmedian(data[n])
    return output


@nb.njit(parallel=True)
def _numba_nanmean(data):
    """numba speed up for nanmean"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.nanmean(data[n])
    return output


@nb.njit(parallel=True)
def _numba_nanstd(data):
    """numba speed up for nanstd"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.nanstd(data[n])
    return output


@nb.njit(parallel=True)
def _numba_nanvar(data):
    """numba speed up for nanvar"""
    output = np.zeros(data.shape[0])
    for n in nb.prange(data.shape[0]):
        output[n] = np.nanvar(data[n])
    return output
