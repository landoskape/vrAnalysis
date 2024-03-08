import numpy as np

from .utils import (
    _check_axis,
    _get_target_axis,
    _get_result_shape,
    _get_reduce_size,
    _get_numpy_method,
)

from ._numba import (
    _numba_ptp,
    _numba_percentile,
    _numba_nanpercentile,
    _numba_quantile,
    _numba_nanquantile,
    _numba_median,
    _numba_average,
    _numba_mean,
    _numba_std,
    _numba_var,
    _numba_nanmedian,
    _numba_nanmean,
    _numba_nanstd,
    _numba_nanvar,
    _numba_zscore,
)


# for each string, lookup for which numba method to use
_method_lookup = dict(
    ptp=_numba_ptp,
    percentile=_numba_percentile,
    nanpercentile=_numba_nanpercentile,
    quantile=_numba_quantile,
    nanquantile=_numba_nanquantile,
    median=_numba_median,
    average=_numba_average,
    mean=_numba_mean,
    std=_numba_std,
    var=_numba_var,
    nanmedian=_numba_nanmedian,
    nanmean=_numba_nanmean,
    nanstd=_numba_nanstd,
    nanvar=_numba_nanvar,
    zscore=_numba_zscore,
)

# these methods require a "q" argument
_requires_q = ["percentile", "nanpercentile", "quantile", "nanquantile"]

# these methods don't have a final reduction
_noreduction = ["zscore"]


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
