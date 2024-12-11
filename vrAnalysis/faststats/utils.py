import numpy as np


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


def _get_reduce_size(data_shape, axis):
    """returns the number of samples to reduce over given the data_shape and requested axis(s)"""
    if _check_iterable(axis):
        return np.prod([data_shape[i] for i in axis])
    else:
        return data_shape[axis]


def _get_result_shape(data_shape, axis, keepdims):
    """returns expected result shape after reducing over axis dimension(s)"""
    if not _check_iterable(axis):
        axis = [axis]
    if keepdims:
        return [1 if idim in axis else dim for idim, dim in enumerate(data_shape)]
    else:
        return [dim for idim, dim in enumerate(data_shape) if idim not in axis]


# fallback to numpy if axis==None
def _get_numpy_method(method):
    """simple way to get numpy version of stats method"""
    return getattr(np, method)
