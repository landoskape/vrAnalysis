from typing import Union
import torch


def _mse(x, y, dim=None):
    return ((x - y) ** 2).mean(dim=dim)


def scaled_mse(y_pred: torch.Tensor, y_true: torch.Tensor, reduce: Union[str, None] = "mean", eps=1e-8):
    """
    Calculate the scaled mean squared error scaled by the error of a constant model.

    Parameters
    ----------
    y_pred : array-like of shape (num_features, num_samples)
        The predicted target values.
    y_true : array-like of shape (num_features, num_samples)
        The true target values.
    reduce : str
        The reduction to apply to the scaled error. If None, the error is returned unscaled. If "mean", the mean of the
        scaled error is returned. If "sum", the sum of the scaled error is returned. Default is mean.

    Returns
    -------
    float
        The scaled mean squared error.
    """
    prediction_error = _mse(y_pred, y_true, dim=0)
    constant_error = _mse(y_true.mean(dim=0, keepdim=True), y_true, dim=0)
    scaled_error = prediction_error / (constant_error + eps)

    if reduce == "mean":
        return scaled_error.mean()

    if reduce == "sum":
        return scaled_error.sum()

    return scaled_error
