import numpy as np

from .utils import (
    _check_axis,
    _get_target_axis,
    _get_result_shape,
    _get_reduce_size,
    _get_numpy_method,
)

from ._numba import (
    numba_sum,
    numba_nansum,
    numba_ptp,
    numba_percentile,
    numba_nanpercentile,
    numba_quantile,
    numba_nanquantile,
    numba_median,
    numba_average,
    numba_mean,
    numba_std,
    numba_var,
    numba_nanmedian,
    numba_nanmean,
    numba_nanstd,
    numba_nanvar,
    numba_zscore,
    numba_median_zscore,
)


# for each string, lookup for which numba method to use
_method_lookup = dict(
    sum=numba_sum,
    nansum=numba_nansum,
    ptp=numba_ptp,
    percentile=numba_percentile,
    nanpercentile=numba_nanpercentile,
    quantile=numba_quantile,
    nanquantile=numba_nanquantile,
    median=numba_median,
    average=numba_average,
    mean=numba_mean,
    std=numba_std,
    var=numba_var,
    nanmedian=numba_nanmedian,
    nanmean=numba_nanmean,
    nanstd=numba_nanstd,
    nanvar=numba_nanvar,
    zscore=numba_zscore,
    median_zscore=numba_median_zscore,
)

# these methods require a "q" argument
_requires_q = ["percentile", "nanpercentile", "quantile", "nanquantile"]

# these methods don't have a final reduction (their output should be same size as input)
_noreduction = ["zscore", "median_zscore"]


def faststat(data, method, axis=-1, keepdims=False, q=None):
    """shapes data and deploys numba speed ups of standard numpy stat methods"""
    if axis is None:
        # no reason to parallelize when reducing across all elements
        _func = _get_numpy_method(method)
        return _func(data, q) if method in _requires_q else _func(data)

    # check if axis is valid
    assert _check_axis(axis, data.ndim), "requested axis is not valid"

    # get numba method
    if method in _method_lookup:
        _func = _method_lookup[method]
    else:
        raise ValueError(f"{method} is not permitted")

    # check if q provided when required
    if method in _requires_q:
        assert q is not None, f"q required for {method}"
        use_q = True
    else:
        use_q = False

    # measure output and reduction shapes
    data_shape = data.shape
    num_reduce = _get_reduce_size(data_shape, axis)

    # move reduction axis(s) to last dims in array
    target = _get_target_axis(axis)
    data = np.moveaxis(data, axis, target)

    # reshape to (num_output_elements, num_elements_to_reduce)
    data = np.reshape(data, (-1, num_reduce))

    # implement numba speed of numpy stats method
    result = _func(data, q) if use_q else _func(data)

    # if no reduction is required, then reorganize dimensions and put back into original shape
    if method in _noreduction:
        return np.reshape(np.moveaxis(result, target, axis), data_shape)

    # otherwise return to desired output shape
    result_shape = _get_result_shape(data_shape, axis, keepdims)
    return np.reshape(result, result_shape)


# ==========================================================================================
# ===================== library of functions for similar use as numpy ======================
# ==========================================================================================
def sum(data, axis=None, keepdims=False):
    return faststat(data, "sum", axis=axis, keepdims=keepdims)


def nansum(data, axis=None, keepdims=False):
    return faststat(data, "nansum", axis=axis, keepdims=keepdims)


def ptp(data, axis=None, keepdims=False):
    return faststat(data, "ptp", axis=axis, keepdims=keepdims)


def percentile(data, q, axis=None, keepdims=False):
    return faststat(data, "percentile", axis=axis, keepdims=keepdims, q=q)


def nanpercentile(data, q, axis=None, keepdims=False):
    return faststat(data, "nanpercentile", axis=axis, keepdims=keepdims, q=q)


def quantile(data, q, axis=None, keepdims=False):
    return faststat(data, "quantile", axis=axis, keepdims=keepdims, q=q)


def nanquantile(data, q, axis=None, keepdims=False):
    return faststat(data, "nanquantile", axis=axis, keepdims=keepdims, q=q)


def median(data, axis=None, keepdims=False):
    return faststat(data, "median", axis=axis, keepdims=keepdims)


def average(data, axis=None, keepdims=False):
    return faststat(data, "average", axis=axis, keepdims=keepdims)


def mean(data, axis=None, keepdims=False):
    return faststat(data, "mean", axis=axis, keepdims=keepdims)


def std(data, axis=None, keepdims=False):
    return faststat(data, "std", axis=axis, keepdims=keepdims)


def var(data, axis=None, keepdims=False):
    return faststat(data, "var", axis=axis, keepdims=keepdims)


def nanmedian(data, axis=None, keepdims=False):
    return faststat(data, "nanmedian", axis=axis, keepdims=keepdims)


def nanmean(data, axis=None, keepdims=False):
    return faststat(data, "nanmean", axis=axis, keepdims=keepdims)


def nanstd(data, axis=None, keepdims=False):
    return faststat(data, "nanstd", axis=axis, keepdims=keepdims)


def nanvar(data, axis=None, keepdims=False):
    return faststat(data, "nanvar", axis=axis, keepdims=keepdims)


def zscore(data, axis=None, keepdims=False):
    return faststat(data, "zscore", axis=axis, keepdims=keepdims)


def median_zscore(data, axis=None, keepdims=False):
    return faststat(data, "median_zscore", axis=axis, keepdims=keepdims)
