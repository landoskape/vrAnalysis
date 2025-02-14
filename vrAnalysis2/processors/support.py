import numpy as np
import numba as nb


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
