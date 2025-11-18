from typing import Union, Literal, Optional, overload
import torch
from .population import _ensure_tensor

# Type aliases
DimType = Optional[Union[int, tuple[int, ...]]]
ReductionMSE = Literal["mean", "sum", None]
ReductionR2 = Literal["mean", "none"]
ReductionRMS = Literal["mean", "sum", None]


def _mse(x: torch.Tensor, y: torch.Tensor, dim: DimType = None) -> torch.Tensor:
    """
    Compute mean squared error between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First tensor.
    y : torch.Tensor
        Second tensor. Must have the same shape as x.
    dim : DimType
        Dimension(s) along which to compute the mean squared error. If None, computes over all dimensions.
        Default is None.

    Returns
    -------
    torch.Tensor
        The mean squared error tensor.
    """
    return ((x - y) ** 2).mean(dim=dim)


@overload
def mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: Literal["mean", "sum"],
    dim: DimType = ...,
) -> float: ...
@overload
def mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: None,
    dim: DimType = ...,
) -> torch.Tensor: ...
def mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: ReductionMSE = "mean",
    dim: DimType = 0,
) -> Union[float, torch.Tensor]:
    """
    Calculate the mean squared error.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted target values. Must have the same shape as y_true.
    y_true : torch.Tensor
        The true target values. Must have the same shape as y_pred.
    reduce : ReductionMSE
        The reduction to apply to the error. If None, the error is returned without additional reduction.
        If "mean", the mean of the error is returned. If "sum", the sum of the error is returned. Default is mean.
    dim : DimType
        Dimension(s) along which to compute the mean squared error. If None, computes over all dimensions.
        Default is 0.

    Returns
    -------
    float or torch.Tensor
        The mean squared error. If reduce is None, returns a tensor. Otherwise returns a scalar.
    """
    y_pred = _ensure_tensor(y_pred)
    y_true = _ensure_tensor(y_true)

    prediction_error = _mse(y_pred, y_true, dim=dim)

    if reduce == "mean":
        return prediction_error.mean().item()

    if reduce == "sum":
        return prediction_error.sum().item()

    return prediction_error


@overload
def scaled_mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: Literal["mean", "sum"],
    eps: float = ...,
    dim: DimType = ...,
) -> float: ...
@overload
def scaled_mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: None,
    eps: float = ...,
    dim: DimType = ...,
) -> torch.Tensor: ...
def scaled_mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: ReductionMSE = "mean",
    eps: float = 1e-8,
    dim: DimType = 0,
) -> Union[float, torch.Tensor]:
    """
    Calculate the scaled mean squared error scaled by the error of a constant model.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted target values. Must have the same shape as y_true.
    y_true : torch.Tensor
        The true target values. Must have the same shape as y_pred.
    reduce : ReductionMSE
        The reduction to apply to the scaled error. If None, the error is returned without additional reduction.
        If "mean", the mean of the scaled error is returned. If "sum", the sum of the scaled error is returned.
        Default is mean.
    eps : float
        Small epsilon value for numerical stability. Default is 1e-8.
    dim : DimType
        Dimension(s) along which to compute the mean squared error. If None, computes over all dimensions.
        Default is 0.

    Returns
    -------
    float or torch.Tensor
        The scaled mean squared error. If reduce is None, returns a tensor. Otherwise returns a scalar.
    """
    y_pred = _ensure_tensor(y_pred)
    y_true = _ensure_tensor(y_true)

    prediction_error = _mse(y_pred, y_true, dim=dim)
    constant_error = _mse(y_true.mean(dim=dim, keepdim=True), y_true, dim=dim)
    scaled_error = prediction_error / (constant_error + eps)

    if reduce == "mean":
        return scaled_error.mean().item()

    if reduce == "sum":
        return scaled_error.sum().item()

    return scaled_error


@overload
def measure_r2(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: Literal["mean"],
    dim: DimType = ...,
) -> float: ...
@overload
def measure_r2(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: Literal["none"],
    dim: DimType = ...,
) -> torch.Tensor: ...
def measure_r2(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: ReductionR2 = "mean",
    dim: DimType = 0,
) -> Union[float, torch.Tensor]:
    """
    Measure r-squared between predicted and true target values.

    Computes R^2 along the specified dimension(s), measuring the proportion of variance explained
    by the predictions relative to the variance of the true values.

    In the case where the target has no variance, the R^2 value will be set to 0.0.
    In the case where the prediction is perfect, the R^2 value will be set to 1.0.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted values. Must have the same shape as y_true.
    y_true : torch.Tensor
        The true target values. Must have the same shape as y_pred.
    reduce : ReductionR2
        The reduction to apply to the r-squared value. If "mean", the mean of the r-squared value
        is returned. If "none", the r-squared value is returned without reduction. Default is mean.
    dim : DimType
        Dimension(s) along which to compute the r-squared. If None, computes over all dimensions.
        Default is 0.

    Returns
    -------
    float or torch.Tensor
        The r-squared value. If reduce is "mean", returns a scalar. If reduce is "none", returns a tensor.
    """
    y_pred = _ensure_tensor(y_pred)
    y_true = _ensure_tensor(y_true)
    ss_res = ((y_true - y_pred) ** 2).sum(dim=dim)
    ss_tot = ((y_true - y_true.mean(dim=dim, keepdim=True)) ** 2).sum(dim=dim)
    r2 = 1 - ss_res / ss_tot
    r2[ss_res == 0] = 1.0
    r2[ss_tot == 0] = 0.0
    if reduce == "mean":
        return r2.mean().item()
    else:
        return r2


@overload
def measure_rms(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: Literal["mean", "sum"],
    dim: DimType = ...,
) -> float: ...
@overload
def measure_rms(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: None,
    dim: DimType = ...,
) -> torch.Tensor: ...
def measure_rms(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce: ReductionRMS = "mean",
    dim: DimType = 0,
) -> Union[float, torch.Tensor]:
    """
    Measure root-mean-square error between predicted and true target values.

    Computes the RMS error along the specified dimension(s).

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted values. Must have the same shape as y_true.
    y_true : torch.Tensor
        The true target values. Must have the same shape as y_pred.
    reduce : ReductionRMS
        The reduction to apply to the RMS value. If "mean", the mean of the RMS value is returned.
        If "sum", the sum of the RMS value is returned. If None, the RMS value is returned
        without reduction. Default is mean.
    dim : DimType
        Dimension(s) along which to compute the root-mean-square error. If None, computes over all dimensions.
        Default is 0.

    Returns
    -------
    float or torch.Tensor
        The RMS value. If reduce is "mean" or "sum", returns a scalar. If reduce is None, returns a tensor.
    """
    y_pred = _ensure_tensor(y_pred)
    y_true = _ensure_tensor(y_true)
    rms = torch.sqrt(((y_true - y_pred) ** 2).mean(dim=dim))
    if reduce == "mean":
        return rms.mean().item()
    elif reduce == "sum":
        return rms.sum().item()
    else:
        return rms
