"""TilburyFitConfig — fit the Tilbury tuning law to each neuron's placefield.

Implements the "AI-discovered tuning law" of Tilbury et al. (bioRxiv
2025.11.12.688086): each tuning curve is modelled as a single *asymmetric
generalized-Gaussian* peak.  For one neuron over positions ``theta``::

    f(theta) = b + A * exp(-(|theta - phi| / sigma(theta))^p)

where ``sigma(theta) = sigma_left`` when ``theta < phi`` and ``sigma_right``
otherwise, giving 6 free parameters (``b, A, phi, sigma_left, sigma_right, p``).

Three models are fit per neuron: this generalized-Gaussian ("tilbury"), a
plain-Gaussian control (``p = 2``, symmetric width), and a
*generalized-shrinkage* variant -- the same generalized-Gaussian, but with a
log-space penalty pulling ``p`` toward 2 and ``sigma_left`` toward
``sigma_right``, i.e. shrinking the fit toward the Gaussian control. Read as a
Bayesian model, the penalty is a Gaussian prior on ``log p`` (centred at
``log 2``) and on ``log(sigma_left / sigma_right)`` (centred at 0), with the
lambdas as prior precisions and the fit as the MAP estimate.

The two prior precisions are chosen *per neuron* by a grid sweep scored on the
validation curve (see :meth:`TilburyFitConfig.process`). Each neuron therefore
gets its own prior strength rather than one population-level value.

Per session the trial-averaged placefield is built for the train, validation
and test splits (``registry.time_split``), exactly as the rest of the
manuscript pipeline does.  Each neuron is fitted *independently* on the
**train** curve (batched Adam gradient descent); the reported quality is R^2
on the held-out **test** curve.

This mirrors the placefield extraction path of :mod:`configs.cvpca` and the
split / activity-normalisation handling of :mod:`configs.placefield_structure`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Optional

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from dimilibi.pca import PCA
from vrAnalysis.helpers import edge2center, reliability_loo
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.placefields import get_placefield
from vrAnalysis.sessions import B2Session, SpksTypes

from ..pipeline.base import AnalysisConfigBase
from ..registry import (
    ACTIVITY_PARAMETERS_NAMES,
    PopulationRegistry,
    get_activity_parameters,
)
from .regression import VALID_ACTIVITY_PARAMETERS

# Splits used for fit / model-selection / reporting.
_SPLITS = ("train", "validation", "test")

# Floor on a neuron's train-curve variance when normalising its data term,
# expressed as a fraction of the whole train matrix's variance. Relative (not a
# population percentile) so it is invariant to ``activity_parameters_name`` and
# does not couple neurons through the population composition; at 1e-6 it binds
# only for neurons ~6 orders of magnitude below the population scale.
_VAR_FLOOR_FRACTION = 1e-6


# ---------------------------------------------------------------------------
# Tilbury generalized-Gaussian math (one neuron at a time)
# ---------------------------------------------------------------------------


def _eval_tilbury(theta: npt.NDArray[np.floating], params: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Evaluate the Tilbury tuning law on ``theta`` for a single neuron.

    Parameters
    ----------
    theta : np.ndarray
        Position bin centres, shape ``(P,)``.
    params : np.ndarray
        Packed parameter vector ``[b, A, phi, sigma_left, sigma_right, p]``.

    Returns
    -------
    np.ndarray
        Fitted curve, shape ``(P,)``.
    """
    b, A, phi, sl, sr, p = _unpack(params)
    diff = theta - phi
    sigma = np.where(diff < 0, sl, sr)
    ratio = np.abs(diff) / np.clip(sigma, 1e-6, None)
    # ratio**p in log space, clamped before exponentiating back out: with p
    # unbounded above (softplus, no ceiling) and ratio > 1 away from the peak,
    # ratio**p can overflow past float range, and exp(-inf) * inf is nan.
    log_pow = np.clip(p * np.log(np.clip(ratio, 1e-12, None)), None, 30.0)
    return b + A * np.exp(-np.exp(log_pow))


def _unpack(params: npt.NDArray[np.floating]) -> tuple:
    """Split a packed parameter vector into ``(b, A, phi, sigma_left, sigma_right, p)``."""
    return tuple(params)


def _param_names() -> list[str]:
    """Parameter names in the same order as ``_unpack`` packs them."""
    return ["b", "A", "phi", "sigma_left", "sigma_right", "p"]


def _r2(pred: npt.NDArray[np.floating], actual: npt.NDArray[np.floating]) -> float:
    """Coefficient of determination of ``pred`` against ``actual`` (one curve)."""
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _r2_batched_torch(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Per-row R^2 of ``pred`` against ``actual``, both shape ``(n_cells, P)``.

    Batched torch equivalent of :func:`_r2`, used during training to score
    every neuron's current-step params against a held-out curve without
    leaving torch (no per-neuron python loop).
    """
    ss_res = ((actual - pred) ** 2).sum(dim=1)
    ss_tot = ((actual - actual.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    return torch.where(ss_tot > 0, 1.0 - ss_res / ss_tot.clamp(min=1e-12), torch.full_like(ss_tot, float("nan")))


def _val_error_and_se(pred: torch.Tensor, actual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-neuron validation error ``1 - R^2`` and its standard error.

    The error for one neuron is a mean over the ``P`` position bins, so it
    carries a standard error directly -- no cross-validation folds needed to
    estimate one. Only ``err`` drives lambda selection; ``se`` is reported so
    diagnostics can weigh a combination's advantage against the noise in the
    estimate (see ``scripts/test_lambda_selection.py``).

    Parameters
    ----------
    pred, actual : torch.Tensor
        Predicted and measured validation curves, shape ``(n_cells, P)``.

    Returns
    -------
    err : torch.Tensor
        ``1 - R^2`` per neuron, shape ``(n_cells,)``; NaN where the validation
        curve is flat.
    se : torch.Tensor
        Standard error of ``err``, same shape.

    Notes
    -----
    The standard error treats position bins as independent. Trial-averaged
    placefields are spatially correlated, so this *under*-estimates the true
    standard error, giving a narrower tolerance band and hence less shrinkage
    than a correlation-aware (e.g. block-bootstrap) estimate would. That errs
    toward the unregularized fit, which is the safe direction.
    """
    sq = (actual - pred) ** 2  # (n_cells, P)
    ss_tot = ((actual - actual.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)  # (n_cells,)
    n_pos = actual.shape[1]
    # err = P * mean(sq) / ss_tot, so se(err) = P * std(sq)/sqrt(P) / ss_tot.
    err = sq.sum(dim=1) / ss_tot
    se = float(np.sqrt(n_pos)) * sq.std(dim=1, unbiased=True) / ss_tot
    nan = torch.full_like(err, float("nan"))
    flat = ss_tot <= 0
    return torch.where(flat, nan, err), torch.where(flat, nan, se)


def _select_lambda_per_neuron(val_err: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
    """Per-neuron selection of the shrinkage lambda: lowest validation error.

    Parameters
    ----------
    val_err : np.ndarray
        Validation error ``1 - R^2``, shape ``(n_combos, n_cells)``. NaN marks a
        combination that failed to fit for that neuron.

    Returns
    -------
    np.ndarray
        Chosen combination index per neuron, ``-1`` where every combination
        failed, shape ``(n_cells,)``.

    Notes
    -----
    A 1-SE variant of this rule -- take the strongest penalty within one
    standard error of the minimum -- was tried and discarded. Scored on the held
    out split across four sessions and two different lambda grids, test R^2 fell
    monotonically as the tolerance band widened (plain > quarter-SE > half-SE >
    1-SE) under both grids, with the 1-SE rule the only variant that lost to not
    shrinking at all. The band assumed the bin-wise standard error was a
    calibrated estimate of selection noise; squared residuals are heavy-tailed
    enough that it instead admitted most of the grid, collapsing "strongest
    admissible" to "grid maximum" for most neurons.
    """
    n_cells = val_err.shape[1]
    idx_selected = np.full(n_cells, -1, dtype=int)
    for n in range(n_cells):
        err = val_err[:, n]
        finite = np.flatnonzero(np.isfinite(err))
        if finite.size:
            idx_selected[n] = int(finite[np.argmin(err[finite])])
    return idx_selected


def _pearson(pred: npt.NDArray[np.floating], actual: npt.NDArray[np.floating]) -> float:
    """Pearson correlation of ``pred`` against ``actual`` (one curve).

    Shape-only quality: unlike :func:`_r2` it is invariant to affine rescaling
    of ``pred``, so amplitude/offset mismatches don't penalise the score.
    """
    p = pred - pred.mean()
    a = actual - actual.mean()
    denom = float(np.sqrt(np.sum(p**2) * np.sum(a**2)))
    if denom <= 0:
        return np.nan
    return float(np.sum(p * a) / denom)


def _curve_spectrum(matrix: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Descending eigenvalue spectrum of the ``(N, P)`` curve matrix's neuron covariance.

    Positions are treated as samples; per-neuron centering is applied by PCA.
    Returns eigenvalues of shape ``(min(N, P),)`` in descending order.
    """
    t = torch.as_tensor(matrix, dtype=torch.float32)
    return PCA(center=True).fit(t).get_eigenvalues().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Per-neuron fitting
# ---------------------------------------------------------------------------


def _contour_sigma_p(
    theta: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    phi: float,
    amplitude: float,
    baseline: float,
    side: str,
    sigma_fallback: float,
    sigma_min: float,
    sigma_max: float,
) -> tuple[float, Optional[float]]:
    """Estimate one side's width from contour radii at 50%/25% peak height.

    Mirrors Tilbury's analytic init: the radius ``d`` where a generalized
    Gaussian ``exp(-(d/sigma)^p)`` crosses a given fraction of the peak height
    determines both ``p`` (from the ratio of two radii) and ``sigma`` (from
    one radius and the solved ``p``). Falls back to ``(sigma_fallback, None)``
    when the curve doesn't have enough resolved structure on this side (flat
    top, too few bins, non-monotonic crossing) to trust the estimate.

    Returns
    -------
    sigma : float
        Width estimate (or ``sigma_fallback``), already clipped to bounds.
    p_candidate : float or None
        Exponent estimate from this side, or ``None`` if untrustworthy.
    """
    mask = theta < phi if side == "left" else theta >= phi
    if not np.any(mask):
        return sigma_fallback, None
    dist = np.abs(theta[mask] - phi)
    height = curve[mask]
    above_half = dist[height >= baseline + 0.5 * amplitude]
    above_qtr = dist[height >= baseline + 0.25 * amplitude]
    if above_half.size == 0 or above_qtr.size == 0:
        return sigma_fallback, None
    d_half, d_qtr = float(above_half.max()), float(above_qtr.max())
    if d_half <= 0 or d_qtr <= d_half:
        return sigma_fallback, None
    p_cand = np.log(2.0) / np.log(d_qtr / d_half)
    if not np.isfinite(p_cand) or p_cand <= 0:
        return sigma_fallback, None
    sigma = float(np.clip(d_half / (2.0 * np.log(2.0)) ** (1.0 / p_cand), sigma_min, sigma_max))
    return sigma, p_cand


def _initial_raw_tilbury(
    theta: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    sigma_min: float,
) -> npt.NDArray[np.floating]:
    """Initial raw (unconstrained) parameter vector for the Tilbury model.

    The peak is seeded at the largest residual above a low-percentile baseline.
    Returns ``raw`` in softplus-space: ``[b0, inv_sp(A0), phi0, inv_sp(sl0-sigma_min),
    inv_sp(sr0-sigma_min), inv_sp(p0)]``.
    """
    theta_lo, theta_hi = float(theta.min()), float(theta.max())
    span = max(theta_hi - theta_lo, 1e-6)
    sigma_max = span
    sigma0 = max(span / 10.0, 2.0 * sigma_min)

    b0 = float(np.percentile(curve, 10.0))
    resid = curve - b0

    i = int(np.argmax(resid))
    phi0 = float(theta[i])
    A0 = max(float(resid[i]), 1e-3)

    sl0, p_l = _contour_sigma_p(theta, curve, phi0, A0, b0, "left", sigma0, sigma_min, sigma_max)
    sr0, p_r = _contour_sigma_p(theta, curve, phi0, A0, b0, "right", sigma0, sigma_min, sigma_max)
    p_candidates = [p for p in (p_l, p_r) if p is not None]
    p0 = float(np.mean(p_candidates)) if p_candidates else 2.0

    return np.array(
        [
            b0,
            float(_inv_softplus(A0)),
            phi0,
            float(_inv_softplus(sl0 - sigma_min)),
            float(_inv_softplus(sr0 - sigma_min)),
            float(_inv_softplus(p0)),
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Control model: plain (single) Gaussian peaks, fixed exponent p=2, symmetric
# width. Fitted alongside Tilbury so the two can be compared on r2_test.
# ---------------------------------------------------------------------------


def _eval_gaussian(theta: npt.NDArray[np.floating], params: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Evaluate the control Gaussian model on ``theta`` for a single neuron.

    Packed parameter vector ``[b, A, phi, sigma]``.
    """
    b, A, phi, sigma = _unpack_gaussian(params)
    diff = theta - phi
    return b + A * np.exp(-0.5 * (diff / np.clip(sigma, 1e-6, None)) ** 2)


def _unpack_gaussian(params: npt.NDArray[np.floating]) -> tuple:
    """Split a packed Gaussian parameter vector into ``(b, A, phi, sigma)``."""
    return tuple(params)


def _param_names_gaussian() -> list[str]:
    """Parameter names in the same order as ``_unpack_gaussian`` packs them."""
    return ["b", "A", "phi", "sigma"]


def _fwhm_sigma(
    theta: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    phi: float,
    amplitude: float,
    baseline: float,
    sigma_fallback: float,
    sigma_min: float,
    sigma_max: float,
) -> float:
    """Estimate a symmetric Gaussian width from the full-width-half-max crossing."""
    dist = np.abs(theta - phi)
    above_half = dist[curve >= baseline + 0.5 * amplitude]
    if above_half.size == 0:
        return sigma_fallback
    d_half = float(above_half.max())
    if d_half <= 0:
        return sigma_fallback
    sigma = d_half / np.sqrt(2.0 * np.log(2.0))
    return float(np.clip(sigma, sigma_min, sigma_max))


def _initial_raw_gaussian(
    theta: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    sigma_min: float,
) -> npt.NDArray[np.floating]:
    """Initial raw (unconstrained) parameter vector for the control Gaussian model.

    Peak seeding mirrors :func:`_initial_raw_tilbury`; width from FWHM.
    Returns ``[b0, inv_sp(A0), phi0, inv_sp(sigma0-sigma_min)]``.
    """
    theta_lo, theta_hi = float(theta.min()), float(theta.max())
    span = max(theta_hi - theta_lo, 1e-6)
    sigma_max = span
    sigma0 = max(span / 10.0, 2.0 * sigma_min)

    b0 = float(np.percentile(curve, 10.0))
    resid = curve - b0

    i = int(np.argmax(resid))
    phi0 = float(theta[i])
    A0 = max(float(resid[i]), 1e-3)

    sigma0_est = _fwhm_sigma(theta, curve, phi0, A0, b0, sigma0, sigma_min, sigma_max)

    return np.array(
        [
            b0,
            float(_inv_softplus(A0)),
            phi0,
            float(_inv_softplus(sigma0_est - sigma_min)),
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Batched torch / gradient-descent fitting
# ---------------------------------------------------------------------------


def _inv_softplus(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Inverse softplus log(exp(x) - 1), numerically stable for large x."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0, x, np.log(np.expm1(np.clip(x, 1e-6, 20.0))))


def _apply_constraints_tilbury(raw: torch.Tensor, sigma_min: float) -> torch.Tensor:
    """Map unconstrained ``raw`` (n_cells, 6) to bounded Tilbury parameters.

    Layout: ``[b, A, phi, sigma_left, sigma_right, p]``.
    ``b`` and ``phi`` are free; the rest use softplus to stay positive.
    """
    F = torch.nn.functional
    b = raw[:, 0:1]
    A = F.softplus(raw[:, 1:2])
    phi = raw[:, 2:3]
    sl = sigma_min + F.softplus(raw[:, 3:4])
    sr = sigma_min + F.softplus(raw[:, 4:5])
    p = F.softplus(raw[:, 5:6])
    return torch.cat([b, A, phi, sl, sr, p], dim=1)


def _apply_constraints_gaussian(raw: torch.Tensor, sigma_min: float) -> torch.Tensor:
    """Map unconstrained ``raw`` (n_cells, 4) to bounded Gaussian parameters.

    Layout: ``[b, A, phi, sigma]``.
    ``b`` and ``phi`` are free; sigma uses softplus.
    """
    F = torch.nn.functional
    b = raw[:, 0:1]
    A = F.softplus(raw[:, 1:2])
    phi = raw[:, 2:3]
    sigma = sigma_min + F.softplus(raw[:, 3:4])
    return torch.cat([b, A, phi, sigma], dim=1)


def _eval_tilbury_torch(theta: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Batched, differentiable equivalent of :func:`_eval_tilbury`.

    Parameters
    ----------
    theta : torch.Tensor
        Position bin centres, shape ``(P,)``.
    params : torch.Tensor
        Packed parameters, shape ``(n_cells, 6)``, same layout as :func:`_unpack`.

    Returns
    -------
    torch.Tensor
        Fitted curves, shape ``(n_cells, P)``.
    """
    b = params[:, 0:1]  # (n_cells, 1)
    A = params[:, 1:2]
    phi = params[:, 2:3]
    sl = params[:, 3:4]
    sr = params[:, 4:5]
    p = params[:, 5:6]

    diff = theta[None, :] - phi  # (n_cells, P)
    sigma = torch.where(diff < 0, sl, sr)
    ratio = diff.abs() / sigma.clamp(min=1e-6)
    # ratio**p in log space, clamped before exponentiating back out: with p
    # unbounded above (softplus, no ceiling) and ratio > 1 away from the peak,
    # ratio**p can overflow to inf, and the inf * exp(-inf)=0 in its gradient is nan.
    log_pow = (p * torch.log(ratio.clamp(min=1e-12))).clamp(max=30.0)
    bumps = A * torch.exp(-torch.exp(log_pow))
    return b + bumps


def _eval_gaussian_torch(theta: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Batched, differentiable equivalent of :func:`_eval_gaussian`.

    ``params`` has shape ``(n_cells, 4)``, same layout as :func:`_unpack_gaussian`.
    """
    b = params[:, 0:1]
    A = params[:, 1:2]
    phi = params[:, 2:3]
    sigma = params[:, 3:4]

    diff = theta[None, :] - phi  # (n_cells, P)
    return b + A * torch.exp(-0.5 * (diff / sigma.clamp(min=1e-6)) ** 2)


def _shrinkage_penalty(params: torch.Tensor) -> torch.Tensor:
    """Gaussian-centered shrinkage penalties for the generalized-Gaussian parameters.

    Pulls the exponent ``p`` toward 2 and the two widths toward each other, both
    in log space so the penalties are scale-free. ``params`` has shape
    ``(n_cells, 6)``.

    Returns
    -------
    torch.Tensor
        The two penalty terms ``[p_penalty, asym_penalty]``, each averaged over
        neurons, weighted separately by ``lam`` in :func:`_fit_all_neurons`.
    """
    p_penalty = ((torch.log(params[:, 5].clamp(min=1e-6)) - np.log(2.0)) ** 2).mean()
    asym_penalty = ((torch.log(params[:, 3].clamp(min=1e-6)) - torch.log(params[:, 4].clamp(min=1e-6))) ** 2).mean()
    return torch.stack([p_penalty, asym_penalty])


@dataclass(frozen=True)
class _ModelSpec:
    """Everything :func:`_fit_all_neurons` needs to fit one tuning-curve model.

    Attributes
    ----------
    name : str
        Short label, used for the tqdm description.
    param_names : list of str
        Names of the packed parameters, in order.
    eval_np : callable
        ``(theta, params) -> curve`` for one neuron (numpy).
    eval_torch : callable
        ``(theta, params) -> curves`` batched over neurons (torch).
    constrain : callable
        ``(raw, sigma_min) -> params`` mapping unconstrained to bounded parameters.
    init_raw : callable
        ``(theta, curve, sigma_min) -> raw`` analytic initial guess for one neuron.
    penalty : callable or None
        ``(params) -> (n_penalties,)`` regularizer terms added to the fit loss,
        each weighted by the matching entry of ``lam``. ``None`` for
        unregularized models.
    """

    name: str
    param_names: list[str]
    eval_np: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]]
    eval_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    constrain: Callable[[torch.Tensor, float], torch.Tensor]
    init_raw: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating], float], npt.NDArray[np.floating]]
    penalty: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


_SPEC_TILBURY = _ModelSpec(
    name="generalized",
    param_names=_param_names(),
    eval_np=_eval_tilbury,
    eval_torch=_eval_tilbury_torch,
    constrain=_apply_constraints_tilbury,
    init_raw=_initial_raw_tilbury,
)

_SPEC_CONTROL = _ModelSpec(
    name="gaussian control",
    param_names=_param_names_gaussian(),
    eval_np=_eval_gaussian,
    eval_torch=_eval_gaussian_torch,
    constrain=_apply_constraints_gaussian,
    init_raw=_initial_raw_gaussian,
)

_SPEC_SHRINKAGE = _ModelSpec(
    name="generalized shrinkage",
    param_names=_param_names(),
    eval_np=_eval_tilbury,
    eval_torch=_eval_tilbury_torch,
    constrain=_apply_constraints_tilbury,
    init_raw=_initial_raw_tilbury,
    penalty=_shrinkage_penalty,
)


@dataclass(frozen=True)
class _FitResult:
    """One model's batched fit over every neuron.

    Attributes
    ----------
    params : np.ndarray
        Packed parameters, shape ``(n_cells, n_params)``, selected per neuron by
        best validation R^2 over the descent. Rows are NaN for neurons that
        never fitted.
    r2_init : np.ndarray
        Train-curve R^2 of the analytic initial guess (pre-training), shape ``(n_cells,)``.
    val_err : np.ndarray
        Validation error ``1 - R^2`` of ``params``, shape ``(n_cells,)``.
    val_se : np.ndarray
        Standard error of ``val_err``, same shape.
    n_var_floored : int
        Neurons whose train-curve variance hit the normalisation floor. Expected
        to be 0; a nonzero count means the data term was rescaled for those
        neurons and their effective prior is stronger than ``lam`` implies.
    """

    params: npt.NDArray[np.floating]
    r2_init: npt.NDArray[np.floating]
    val_err: npt.NDArray[np.floating]
    val_se: npt.NDArray[np.floating]
    n_var_floored: int


def _fit_all_neurons(
    theta: npt.NDArray[np.floating],
    curves_train: npt.NDArray[np.floating],
    curves_val: npt.NDArray[np.floating],
    spec: _ModelSpec,
    sigma_min: float,
    device: str,
    num_steps: int,
    learning_rate: float,
    lam: tuple[float, ...] = (),
    verbose: bool = False,
) -> _FitResult:
    """Batched Adam fit of one model (``spec``) to every neuron jointly.

    Each neuron's data term is its own MSE divided by its own train-curve
    variance, so the loss is ``mean_n(MSE_n / var_n) + sum_k lam_k * penalty_k``.
    The penalty is a dimensionless log-space distance, so this per-neuron
    normalisation is what makes ``lam`` the *same* scale-free data-to-prior
    ratio for every neuron -- a single global scale instead makes low-amplitude
    neurons feel a far stronger prior than high-amplitude ones. It also makes
    ``lam = 0`` reproduce :data:`_SPEC_TILBURY` exactly, since the models then
    differ in nothing.

    The normalisation is applied to every spec, including the unpenalized ones,
    so that the ``lam = 0`` identity holds. It is close to free there: the loss
    is separable across neurons (a neuron's ``raw`` row only ever receives
    gradient from its own terms) and Adam's per-parameter update is invariant to
    a constant rescaling of that row's loss. On synthetic curves spanning four
    orders of magnitude in amplitude the median per-neuron test-R^2 shift is
    ~1e-9. It is not exactly free, though: the invariance breaks down in float32
    for neurons whose variance sits orders of magnitude below the population
    scale, where the unnormalised gradients are small enough for Adam's ``eps``
    and float32 rounding to bite. Those neurons move (mostly improving, since
    normalising conditions their gradients better), so the unregularized fits
    are unchanged for the bulk of the population but not for that tail.

    Per-neuron early stopping on the validation curve: at every step the
    current params are scored against ``curves_val`` (same ``theta`` grid as
    train, so the train-step prediction is reused, no extra forward pass),
    and the best-so-far params are kept per neuron rather than just the final
    step's params. This guards against overfitting the train curve.

    Parameters
    ----------
    theta : np.ndarray
        Position bin centres, shape ``(P,)``.
    curves_train, curves_val : np.ndarray
        Trial-averaged placefields, shape ``(n_cells, P)``.
    spec : _ModelSpec
        Model to fit.
    sigma_min : float
        Lower bound on peak widths.
    device : str
        Torch device.
    num_steps : int
        Adam iterations.
    learning_rate : float
        Adam learning rate.
    lam : tuple of float
        One weight per term returned by ``spec.penalty``, as a dimensionless
        ratio against the variance-normalised data term; ignored when the spec
        has no penalty.
    verbose : bool
        Show a tqdm progress bar.

    Returns
    -------
    _FitResult
    """
    n_cells = curves_train.shape[0]
    raw_init = np.stack([spec.init_raw(theta, curves_train[n], sigma_min) for n in range(n_cells)])

    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    train_t = torch.as_tensor(curves_train, dtype=torch.float32, device=device)
    val_t = torch.as_tensor(curves_val, dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(torch.as_tensor(raw_init, dtype=torch.float32, device=device))

    # Per-neuron data-term normaliser, floored relative to the population scale
    # (see _VAR_FLOOR_FRACTION). A neuron pushed to the floor has a data term
    # too weak to resist the prior, so it falls back toward the Gaussian --
    # the right behaviour for a neuron carrying no signal, but worth counting.
    var_train = train_t.var(dim=1, unbiased=False)
    var_floor = _VAR_FLOOR_FRACTION * float(train_t.var(unbiased=False))
    n_var_floored = int((var_train < var_floor).sum())
    var_train = var_train.clamp(min=var_floor)

    with torch.no_grad():
        init_params = spec.constrain(raw, sigma_min)
        r2_init = _r2_batched_torch(spec.eval_torch(theta_t, init_params), train_t)

    best_val_r2 = torch.full((n_cells,), float("-inf"), device=device)
    best_params = init_params.clone()

    lam_t = torch.as_tensor(lam, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    steps = tqdm(range(num_steps), desc=f"descent fit {spec.name}") if verbose else range(num_steps)
    for step in steps:
        optimizer.zero_grad()
        params = spec.constrain(raw, sigma_min)
        pred = spec.eval_torch(theta_t, params)
        loss = (((pred - train_t) ** 2).mean(dim=1) / var_train).mean()
        if spec.penalty is not None:
            loss = loss + (lam_t * spec.penalty(params)).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_r2 = _r2_batched_torch(pred, val_t)
            improved = val_r2 > best_val_r2
            best_val_r2 = torch.where(improved, val_r2, best_val_r2)
            best_params = torch.where(improved[:, None], params.detach(), best_params)

        if verbose and step % 100 == 0:
            steps.set_postfix(loss=f"{loss.item():.4g}", val_r2=f"{val_r2.mean().item():.3g}")

    # A neuron whose validation curve is flat scores NaN R^2 every step, so
    # ``improved`` is never True and it still holds its analytic initial guess.
    # Mark those rows NaN so every downstream mask drops them uniformly instead
    # of some paths treating an unfitted seed as a fit.
    with torch.no_grad():
        failed = ~torch.isfinite(best_val_r2)
        val_err, val_se = _val_error_and_se(spec.eval_torch(theta_t, best_params), val_t)
        nan_row = torch.full_like(best_params, float("nan"))
        nan_vec = torch.full_like(val_err, float("nan"))
        best_params = torch.where(failed[:, None], nan_row, best_params)
        val_err = torch.where(failed, nan_vec, val_err)
        val_se = torch.where(failed, nan_vec, val_se)

    return _FitResult(
        params=best_params.cpu().numpy().astype(np.float64),
        r2_init=r2_init.cpu().numpy().astype(np.float64),
        val_err=val_err.cpu().numpy().astype(np.float64),
        val_se=val_se.cpu().numpy().astype(np.float64),
        n_var_floored=n_var_floored,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TilburyFitConfig(AnalysisConfigBase):
    """Fit the Tilbury generalized-Gaussian tuning law per neuron.

    Parameters
    ----------
    spks_type : SpksTypes
        Spike type retrieved from the registry.
    activity_parameters_name : str
        Name of the ``ActivityParameters`` preset used to normalise activity
        (train-split statistics applied to every split).
    num_bins : int
        Number of spatial bins.
    smooth_width : float or None
        Gaussian smoothing width (cm) applied to the averaged placefields before
        fitting; ``None`` for raw (unsmoothed) trial averages.
    reliability_fraction_active_thresholds : tuple of (float or None, float or None)
        Minimum reliability and minimum fraction active for neuron inclusion
        (computed on the train split). ``(None, None)`` keeps all neurons.
    sigma_min : ClassVar[float]
        Lower bound on peak widths (cm).
    lambda_grid_p : ClassVar[tuple of float]
        Prior precisions swept for the exponent penalty. Capped at ``1e-1``:
        held-out scoring across sessions found ``lam_p = 1`` to be the worst
        column of the grid every time, so the range is spent at the weak end
        where the useful values sit.
    lambda_grid_asym : ClassVar[tuple of float]
        Prior precisions swept for the asymmetry penalty. Reaches two decades
        higher than :attr:`lambda_grid_p` because the two penalties empirically
        want very different strengths -- the best fixed asymmetry penalty sat at
        the top of a grid ending at ``1``, while the exponent penalty wanted the
        bottom. Both are dimensionless ratios against ``1 - R^2`` (the data term
        is variance-normalised per neuron), so the offset reflects the data, not
        a difference in units.

    Notes
    -----
    The sweep is the outer product of the two grids, evaluated per neuron, so a
    session costs ``len(lambda_grid_p) * len(lambda_grid_asym)`` shrinkage fits
    plus the two unregularized ones.

    Both grids start at exactly ``0.0``, which turns that penalty off. The
    ``(0, 0)`` combination therefore reproduces the unregularized generalized
    fit (``params``) exactly -- a free consistency check, and an explicit "no
    shrinkage wanted" outcome available to every neuron.

    Because ``(0, 0)`` is in the grid, the per-neuron selection guarantees the
    shrinkage model's *validation* score is at least as good as the generalized
    model's for every neuron. That comparison is vacuous; only the ``*_test``
    metrics compare the two models informatively.

    A strong lambda *saturates* rather than scaling linearly, which is why the
    top of each grid does not need to be pushed further. The data term is a
    variance fraction, so the whole budget a penalty can trade against is at
    most ~1; a penalty ``lam * d^2`` (with ``d`` a log-space distance) therefore
    admits at most ``d = sqrt(1 / lam)`` no matter how much the fit would
    improve. For the asymmetry penalty that bounds the width ratio
    ``sigma_left / sigma_right`` at ``exp(sqrt(1 / lam_asym))``: 2.7 at
    ``lam_asym = 1``, and 1.37 at ``lam_asym = 10`` -- with realised
    asymmetries far tighter, since real fits recover nothing like a full unit of
    variance. Neurons clipping at the top of ``lambda_grid_asym`` are thus
    already clamped to near-symmetry, and extending the grid buys little.
    """

    schema_version: str = "v10"
    data_config_name: str = "even"
    spks_type: SpksTypes = "sigrebase"
    activity_parameters_name: str = "raw"
    num_bins: int = 100
    reliability_fraction_active_thresholds: Optional[tuple[float, float]] = (0.3, 0.1)

    sigma_min: ClassVar[float] = 1.0
    max_missing_position_percentage: ClassVar[float] = 5.0
    lambda_grid_p: ClassVar[tuple[float, ...]] = (0.0, 1e-3, 1e-2, 1e-1)
    lambda_grid_asym: ClassVar[tuple[float, ...]] = (0.0, 1e-1, 1.0, 10.0)
    display_name: ClassVar[str] = "tilbury_fit"
    _result_handling: ClassVar[dict[str, str]] = {
        "idx_keep": "ragged",
        "dist_centers": "skip",
        "best_env": "skip",
        "param_names": "skip",
        "param_names_control": "skip",
        "param_names_shrinkage": "skip",
        "lambda_combos": "pad",
        "lambda_scores": "pad",
        "eig_tilbury": "pad",
        "eig_control": "pad",
        "eig_shrinkage": "pad",
        "eig_raw_train": "pad",
        "eig_raw_test": "pad",
    }

    @staticmethod
    def _param_grid() -> dict:
        return {"activity_parameters_name": ["default", "raw"]}

    @property
    def param_names(self) -> list[str]:
        return _param_names()

    @property
    def param_names_control(self) -> list[str]:
        return _param_names_gaussian()

    @property
    def param_names_shrinkage(self) -> list[str]:
        return _param_names()

    def lambda_combos(self) -> npt.NDArray[np.floating]:
        """The ``(lam_p, lam_asym)`` pairs the shrinkage sweep will evaluate.

        Returns
        -------
        np.ndarray
            Outer product of :attr:`lambda_grid_p` and :attr:`lambda_grid_asym`,
            shape ``(n_combos, 2)``.
        """
        return np.array([(lp, la) for lp in self.lambda_grid_p for la in self.lambda_grid_asym], dtype=float)

    def validate(self) -> None:
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(f"Unknown activity_parameters_name {self.activity_parameters_name!r}. Available: {list(ACTIVITY_PARAMETERS_NAMES)}")

    def summary(self) -> str:
        rel, frac = self.reliability_fraction_active_thresholds
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"ap={self.activity_parameters_name}",
            f"bins={self.num_bins}",
            f"rel={rel}",
            f"frac={frac}",
            self.schema_version,
        ]
        return "_".join(parts)

    # -- data path ---------------------------------------------------------

    def _get_split_data(self, session: B2Session, registry: PopulationRegistry) -> tuple[dict[str, np.ndarray], dict]:
        """Load ``(frames, rois)`` activity and aligned behaviour for every split.

        Scaling is delegated to ``population.apply_split`` (same path as
        ``configs.cvpca``) so the ``ActivityParameters`` ``scale`` / ``scale_type``
        / ``presplit`` semantics are honoured: with ``presplit=True`` the scaling
        statistics are computed on the full (pre-split) data, then each split is
        selected, rather than recomputed per split.
        """
        population, frame_behavior = registry.get_population(session, self.spks_type)
        ap = get_activity_parameters(self.activity_parameters_name)
        full = population.data[population.idx_neurons][:, population.idx_samples]  # (N, T)

        spks: dict[str, np.ndarray] = {}
        fb: dict = {}
        for s in _SPLITS:
            split_idx = registry.time_split[s]
            scaled = population.apply_split(
                full,
                time_idx=split_idx,
                prefiltered=True,
                scale=ap.scale,
                scale_type=ap.scale_type,
                pre_split=ap.presplit,
            )
            spks[s] = np.array(scaled).T  # (frames_split, rois)
            idx_orig = np.array(population.get_split_times(split_idx, within_idx_samples=False))
            fb[s] = frame_behavior.filter(idx_orig)
        return spks, fb

    def _select_neurons(self, spks: np.ndarray, frame_behavior, dist_edges, best_env, session) -> np.ndarray:
        """Boolean mask of reliable / active neurons from the train split."""
        rel, frac = self.reliability_fraction_active_thresholds
        if rel is None and frac is None:
            return np.ones(spks.shape[1], dtype=bool)

        pf = get_placefield(spks, frame_behavior, dist_edges, average=False, use_fast_sampling=True, session=session).filter_by_environment(best_env)
        pf_data = np.transpose(pf.placefield, (2, 0, 1))  # (N, trials, bins)
        idx_keep = np.ones(pf_data.shape[0], dtype=bool)
        if rel is not None:
            idx_keep &= reliability_loo(pf_data) > rel
        if frac is not None:
            idx_keep &= (
                FractionActive.compute(pf_data, activity_axis=2, fraction_axis=1, activity_method="rms", fraction_method="participation") > frac
            )
        return idx_keep

    def _avg_placefield(self, spks, frame_behavior, dist_edges, best_env, session):
        """Trial-averaged placefield for ``best_env``: returns ``((N, P), (P,) counts)``."""
        pf = get_placefield(
            spks,
            frame_behavior,
            dist_edges,
            average=True,
            use_fast_sampling=True,
            session=session,
        ).filter_by_environment(best_env)
        return pf.placefield[0].T, pf.count[0]  # (N, P), (P,)

    def prepare_curves(self, session: B2Session, registry: PopulationRegistry) -> tuple[npt.NDArray[np.floating], dict, npt.NDArray[np.bool_], int]:
        """Build the per-split trial-averaged placefields the fits are run on.

        Everything :meth:`process` does before any fitting: pick the
        best-sampled environment, load and scale the per-split activity, select
        reliable/active neurons on the train split, trial-average each split,
        and drop position bins missing from any split so ``theta`` and the
        curves stay aligned.

        Split out from :meth:`process` so diagnostics can drive the fitting
        internals on exactly the same curves the pipeline uses.

        Parameters
        ----------
        session : B2Session
        registry : PopulationRegistry

        Returns
        -------
        theta : np.ndarray
            Retained position bin centres, shape ``(P,)``.
        curves : dict of str to np.ndarray
            Trial-averaged placefields per split, each ``(n_kept, P)``.
        idx_keep : np.ndarray
            Boolean neuron-selection mask over the session's ROIs.
        best_env : int
            Environment the curves were built from.

        Raises
        ------
        ValueError
            If more than :attr:`max_missing_position_percentage` of bins are
            unvisited in some split, or a curve still holds NaNs afterwards.
        """
        num_per_env = {i: int(np.sum(session.trial_environment == i)) for i in session.environments}
        best_env = max(num_per_env, key=num_per_env.get)

        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)
        dist_centers = edge2center(dist_edges)

        # Per-split activity, scaled via population.apply_split (honours presplit).
        spks, fb = self._get_split_data(session, registry)

        # Neuron selection on the train split.
        idx_keep = self._select_neurons(spks["train"], fb["train"], dist_edges, best_env, session)
        for s in _SPLITS:
            spks[s] = spks[s][:, idx_keep]

        # Trial-averaged placefields and counts per split.
        curves, counts = {}, {}
        for s in _SPLITS:
            curves[s], counts[s] = self._avg_placefield(spks[s], fb[s], dist_edges, best_env, session)

        # Drop bins missing in any split so theta / curves stay aligned.
        bad = np.zeros(self.num_bins, dtype=bool)
        for s in _SPLITS:
            bad |= counts[s] == 0
        max_missing = int(self.num_bins * self.max_missing_position_percentage / 100)
        if int(bad.sum()) > max_missing:
            raise ValueError(f"Too many missing positions: {int(bad.sum())} > {max_missing}")
        good = ~bad
        theta = dist_centers[good]
        curves = {s: curves[s][:, good] for s in _SPLITS}

        for s in _SPLITS:
            if np.any(np.isnan(curves[s])):
                raise ValueError(f"NaNs in {s} placefield after dropping empty bins!")

        return theta, curves, idx_keep, best_env

    # -- main --------------------------------------------------------------

    def process(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        verbose: bool = False,
        device: Optional[str] = None,
        gd_num_steps: int = 10000,
        gd_learning_rate: float = 0.1,
    ) -> dict:
        """Fit the three tuning-curve models per neuron with train/val/test cross-validation.

        Params are fit by Adam on the train curve, but selected per neuron by
        best R^2 on the validation curve across the training sweep (guards
        against overfitting) rather than just taking the final step's params.

        The generalized-shrinkage model additionally sweeps every
        ``(lam_p, lam_asym)`` combination from :meth:`lambda_combos` and picks,
        **per neuron**, the one minimising that neuron's validation error
        ``1 - R^2_val`` (:func:`_select_lambda_per_neuron`). Each neuron
        therefore gets its own prior strength; the shrinkage "model" is a
        per-neuron mixture rather than one population-level model. Because the
        selection ranks combinations *within* a neuron, its per-neuron error
        normaliser cancels, so the choice is invariant to whether train- or
        validation-curve variance is used as the denominator.

        Note the validation split does triple duty here (per-neuron early
        stopping, per-neuron lambda selection, and the ``eig_better`` model
        choice), so ``r2_val*`` is optimistically biased and should not be read
        as a performance number. The test split stays untouched, which is what
        the reported ``*_test`` metrics depend on.

        Parameters
        ----------
        session : B2Session
        registry : PopulationRegistry
        verbose : bool
            If True, show a tqdm progress bar with optimizer step + loss.
        device : str or None
            Torch device for the batched Adam fit (e.g. ``"cpu"``, ``"cuda"``).
            ``None`` (the default) auto-detects: ``"cuda"`` if
            ``torch.cuda.is_available()`` else ``"cpu"``.
        gd_num_steps : int
            Adam iterations to sweep for validation-best param selection.
        gd_learning_rate : float
            Adam learning rate.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        theta, curves, idx_keep, best_env = self.prepare_curves(session, registry)

        # Per-neuron fits: generalized (Tilbury), plain-Gaussian control, and
        # generalized-shrinkage (lambda swept on the validation curves).
        n_neurons = curves["train"].shape[0]
        r2 = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        r2c = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        r2s = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        pearson = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        pearson_c = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        pearson_s = {split: np.full(n_neurons, np.nan) for split in _SPLITS}

        fit_kwargs = dict(sigma_min=self.sigma_min, device=device, num_steps=gd_num_steps, learning_rate=gd_learning_rate, verbose=verbose)
        fit = _fit_all_neurons(theta, curves["train"], curves["validation"], _SPEC_TILBURY, **fit_kwargs)
        fit_c = _fit_all_neurons(theta, curves["train"], curves["validation"], _SPEC_CONTROL, **fit_kwargs)
        params, params_c = fit.params, fit_c.params
        r2_init, r2_init_c = fit.r2_init, fit_c.r2_init

        # Shrinkage sweep: fit every (lam_p, lam_asym) combination, keeping each
        # one's per-neuron validation error and its standard error so the choice
        # can be made per neuron afterwards.
        lambda_combos = self.lambda_combos()
        n_params_s = len(_param_names())
        sweep_params = np.full((len(lambda_combos), n_neurons, n_params_s), np.nan)
        sweep_err = np.full((len(lambda_combos), n_neurons), np.nan)
        sweep_se = np.full((len(lambda_combos), n_neurons), np.nan)
        r2_init_s = None
        for i, (lam_p, lam_asym) in enumerate(lambda_combos):
            fit_s = _fit_all_neurons(
                theta, curves["train"], curves["validation"], _SPEC_SHRINKAGE, lam=(float(lam_p), float(lam_asym)), **fit_kwargs
            )
            sweep_params[i], sweep_err[i], sweep_se[i] = fit_s.params, fit_s.val_err, fit_s.val_se
            # The analytic seed ignores the penalty, so r2_init is the same for
            # every combination; keeping the last is keeping all of them.
            r2_init_s = fit_s.r2_init

        # Per-neuron choice. ``lambda_scores`` (the population mean error per
        # combination) is kept purely as a diagnostic of the grid's shape.
        lambda_scores = np.nanmean(sweep_err, axis=1)
        idx_sel = _select_lambda_per_neuron(sweep_err)

        params_s = np.full((n_neurons, n_params_s), np.nan)
        lambda_selected = np.full((n_neurons, 2), np.nan)
        lambda_val_err = np.full(n_neurons, np.nan)
        rows = np.flatnonzero(idx_sel >= 0)
        params_s[rows] = sweep_params[idx_sel[rows], rows]
        lambda_selected[rows] = lambda_combos[idx_sel[rows]]
        lambda_val_err[rows] = sweep_err[idx_sel[rows], rows]

        # Grid-clipping diagnostic: how many neurons want the strongest penalty
        # available. Interpret it against the fact that a strong asymmetry
        # penalty saturates (see the class Notes) -- clipping on lam_asym costs
        # little, clipping on lam_p would mean the exponent range is too narrow.
        frac_lambda_clipped_p = float(np.mean(lambda_selected[rows, 0] == self.lambda_grid_p[-1])) if rows.size else np.nan
        frac_lambda_clipped_asym = float(np.mean(lambda_selected[rows, 1] == self.lambda_grid_asym[-1])) if rows.size else np.nan
        n_var_floored = fit.n_var_floored

        # Score each fit on every split (cheap, done serially).
        for n in range(n_neurons):
            x, xc, xs = params[n], params_c[n], params_s[n]
            for split in _SPLITS:
                c = curves[split][n]
                if not np.any(np.isnan(x)):
                    pred = _eval_tilbury(theta, x)
                    r2[split][n] = _r2(pred, c)
                    pearson[split][n] = _pearson(pred, c)
                if not np.any(np.isnan(xc)):
                    pred_c = _eval_gaussian(theta, xc)
                    r2c[split][n] = _r2(pred_c, c)
                    pearson_c[split][n] = _pearson(pred_c, c)
                if not np.any(np.isnan(xs)):
                    pred_s = _eval_tilbury(theta, xs)
                    r2s[split][n] = _r2(pred_s, c)
                    pearson_s[split][n] = _pearson(pred_s, c)

        # Population dimensionality: eigenvalue spectra of the (N, P) tuning-curve
        # matrices. Every spectrum -- modelled and raw alike -- uses the *same*
        # neuron set, the ones all three models fitted. Comparing spectra built
        # from different neuron sets would confound the model comparison with a
        # change in the population being measured.
        ok_all = ~(np.isnan(params).any(axis=1) | np.isnan(params_c).any(axis=1) | np.isnan(params_s).any(axis=1))
        rows_ok = np.flatnonzero(ok_all)
        if rows_ok.size == 0:
            raise ValueError("No neuron fitted successfully under all three models; cannot compute spectra.")
        mat_tilbury = np.stack([_eval_tilbury(theta, params[n]) for n in rows_ok])
        mat_control = np.stack([_eval_gaussian(theta, params_c[n]) for n in rows_ok])
        mat_shrinkage = np.stack([_eval_tilbury(theta, params_s[n]) for n in rows_ok])
        # Per-neuron winner picked on the *validation* split (lowest validation
        # MSE, which within a neuron is the same ranking as highest validation
        # R^2 -- same denominator), so the test split stays held out. Only the
        # two unregularized models compete here: the shrinkage model contains
        # the generalized one at lam=(0,0), so it can never lose on validation
        # and including it would make the comparison vacuous.
        better = r2["validation"][ok_all] > r2c["validation"][ok_all]
        mat_better = np.where(better[:, None], mat_tilbury, mat_control)
        eig_tilbury = _curve_spectrum(mat_tilbury)
        eig_control = _curve_spectrum(mat_control)
        eig_shrinkage = _curve_spectrum(mat_shrinkage)
        eig_better = _curve_spectrum(mat_better)
        eig_raw_train = _curve_spectrum(curves["train"][ok_all])
        eig_raw_test = _curve_spectrum(curves["test"][ok_all])

        return {
            "params": params,
            "r2_init": r2_init,
            "r2_train": r2["train"],
            "r2_val": r2["validation"],
            "r2_test": r2["test"],
            "pearson_train": pearson["train"],
            "pearson_val": pearson["validation"],
            "pearson_test": pearson["test"],
            "params_control": params_c,
            "r2_init_control": r2_init_c,
            "r2_train_control": r2c["train"],
            "r2_val_control": r2c["validation"],
            "r2_test_control": r2c["test"],
            "pearson_train_control": pearson_c["train"],
            "pearson_val_control": pearson_c["validation"],
            "pearson_test_control": pearson_c["test"],
            "params_shrinkage": params_s,
            "r2_init_shrinkage": r2_init_s,
            "r2_train_shrinkage": r2s["train"],
            "r2_val_shrinkage": r2s["validation"],
            "r2_test_shrinkage": r2s["test"],
            "pearson_train_shrinkage": pearson_s["train"],
            "pearson_val_shrinkage": pearson_s["validation"],
            "pearson_test_shrinkage": pearson_s["test"],
            "lambda_combos": lambda_combos,
            "lambda_scores": lambda_scores,
            "lambda_selected": lambda_selected,
            "lambda_val_err": lambda_val_err,
            "frac_lambda_clipped_p": frac_lambda_clipped_p,
            "frac_lambda_clipped_asym": frac_lambda_clipped_asym,
            "n_var_floored": n_var_floored,
            "idx_keep": idx_keep,
            "idx_fit": ok_all,
            "eig_tilbury": eig_tilbury,
            "eig_control": eig_control,
            "eig_better": eig_better,
            "eig_shrinkage": eig_shrinkage,
            "eig_raw_train": eig_raw_train,
            "eig_raw_test": eig_raw_test,
            "dist_centers": theta,
            "best_env": best_env,
            "param_names": _param_names(),
            "param_names_control": _param_names_gaussian(),
            "param_names_shrinkage": _param_names(),
        }
