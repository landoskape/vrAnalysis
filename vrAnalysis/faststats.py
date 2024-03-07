import numpy as np
import numba as nb


def _check_iterable(var):
    """checks if variable is iterable"""
    return hasattr(var, "__iter__")


def _validate_axis(axis, ndim):
    """checks if axis is valid for an array with ndim dimensions"""
    if (axis < -ndim) or (axis >= ndim):
        return False
    return True


def _check_axis(axis, ndim):
    """checks if axis is valid for an array with ndim dimensions"""
    if _check_iterable(axis):
        return all([_validate_axis(a, ndim) for a in axis])
    return _validate_axis(axis, ndim)


def _get_target_axis(axis):
    """convert axis to -len(axis):-1 or just -1"""
    if _check_iterable(axis):
        num_axes = len(axis)
        target = np.arange(-num_axes, 0)
    else:
        target = -1
    return target


def _get_reduce(shape, target):
    """returns the number of samples to reduce over given the shape and target dims"""
    if _check_iterable(target):
        return np.prod([shape[i] for i in target])
    else:
        return shape[target]


def _get_result_shape(data_shape, target):
    """returns expected result shape after reducing over target dimension"""
    if _check_iterable(target):
        return data_shape[: -len(target)]
    return data_shape[:-1]


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
)

# these methods require a "q" argument
_requires_q = ["percentile", "nanpercentile", "quantile", "nanquantile"]


# fallback to numpy if axis==None
def _get_numpy_method(method):
    """simple way to get numpy version of stats method"""
    return getattr(np, method)


def faststat(data, method, axis=-1, q=None):
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

    # move target reduction axis to end
    target = _get_target_axis(axis)
    data = np.moveaxis(data, axis, target)

    # get shape
    data_shape = data.shape

    # measure expected result shape and transform data so reduction dimension is at end
    result_shape = _get_result_shape(data_shape, target)
    num_reduce = _get_reduce(data_shape, target)
    data = np.reshape(data, (-1, num_reduce))

    # implement numba speed of numpy stats method
    result = _func(data, q) if use_q else _func(data)

    # return correctly shaped output
    return np.reshape(result, result_shape)
