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


def measure_r2(y_pred: torch.Tensor, y_true: torch.Tensor):
    """
    Measure r-squared between predicted and true target values.

    Will measure the r-squared for each sample, then take the average across samples.

    In the case where the target has no variance, the R^2 value will be set to 0.0.
    In the case where the prediction is perfect, the R^2 value will be set to 1.0.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted values (num_features, num_samples).
    y_true : torch.Tensor
        The true target values (num_features, num_samples).

    Returns
    -------
    torch.Tensor
        The r-squared value.
    """
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2 = 1 - ss_res / ss_tot
    r2[ss_res == 0] = 1.0
    r2[ss_tot == 0] = 0.0
    return r2.mean()
