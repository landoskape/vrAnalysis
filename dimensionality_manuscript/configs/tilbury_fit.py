"""TilburyFitConfig — fit the Tilbury tuning law to each neuron's placefield.

Implements the "AI-discovered tuning law" of Tilbury et al. (bioRxiv
2025.11.12.688086): each tuning curve is modelled as a single *asymmetric
generalized-Gaussian* peak.  For one neuron over positions ``theta``::

    f(theta) = b + A * exp(-(|theta - phi| / sigma(theta))^p)

where ``sigma(theta) = sigma_left`` when ``theta < phi`` and ``sigma_right``
otherwise, giving 6 free parameters (``b, A, phi, sigma_left, sigma_right, p``).

Alongside this single-peak Tilbury model and its plain-Gaussian control, two
double-peak variants are fit per neuron: a *double generalized-Gaussian* and a
*double Gaussian*, each the sum of two peaks of the same family sharing one
baseline. These let genuinely bimodal tuning curves (two place fields per
neuron/environment) be modelled without splitting the fit between two bumps.

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
from typing import ClassVar, Optional

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


# ---------------------------------------------------------------------------
# Tilbury generalized double-Gaussian math (one neuron at a time)
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
# Double-peak models: two summed peaks (shared baseline) per neuron, for
# genuinely bimodal tuning curves. One variant per family above (generalized /
# plain Gaussian), built by reusing the single-peak eval functions per bump.
# ---------------------------------------------------------------------------


def _local_mask(
    theta: npt.NDArray[np.floating],
    phi: float,
    other_phi: float,
) -> npt.NDArray[np.bool_]:
    """Boolean mask of points nearer to ``phi`` than to ``other_phi`` (Voronoi split).

    Used to restrict per-peak width estimation to each peak's own
    neighborhood when seeding a two-peak initial guess.
    """
    return np.abs(theta - phi) <= np.abs(theta - other_phi)


def _seed_two_peaks(
    theta: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    sigma_min: float,
) -> tuple[float, float, float, float, float]:
    """Seed two peak centers/amplitudes from the two largest residual maxima.

    Finds the largest residual peak the same way the single-peak models do,
    masks out a window around it, then seeds the second peak at the largest
    residual maximum outside that window.

    Returns
    -------
    b0, phi1_0, A1_0, phi2_0, A2_0 : float
        Shared baseline and each peak's seeded center/amplitude.
    """
    b0 = float(np.percentile(curve, 10.0))
    resid = curve - b0

    i1 = int(np.argmax(resid))
    phi1_0 = float(theta[i1])
    A1_0 = max(float(resid[i1]), 1e-3)

    span = max(float(theta.max() - theta.min()), 1e-6)
    sigma0 = max(span / 10.0, 2.0 * sigma_min)
    resid_masked = np.where(np.abs(theta - phi1_0) < 2.0 * sigma0, -np.inf, resid)

    if np.all(~np.isfinite(resid_masked)):
        # Nothing left outside the mask (e.g. very few bins): fall back to
        # the second-largest raw residual value overall.
        order = np.argsort(resid)[::-1]
        i2 = int(order[1]) if order.size > 1 else i1
    else:
        i2 = int(np.argmax(resid_masked))
    phi2_0 = float(theta[i2])
    A2_0 = max(float(resid[i2]), 1e-3)

    return b0, phi1_0, A1_0, phi2_0, A2_0


def _eval_double_generalized(theta: npt.NDArray[np.floating], params: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Evaluate the double generalized-Gaussian model on ``theta`` for a single neuron.

    Packed parameter vector ``[b, A1, phi1, sl1, sr1, p1, A2, phi2, sl2, sr2, p2]``;
    the two peaks share one baseline and are summed. Built from two calls to
    :func:`_eval_tilbury` with each peak's own baseline zeroed.
    """
    b, A1, phi1, sl1, sr1, p1, A2, phi2, sl2, sr2, p2 = params
    peak1 = _eval_tilbury(theta, np.array([0.0, A1, phi1, sl1, sr1, p1]))
    peak2 = _eval_tilbury(theta, np.array([0.0, A2, phi2, sl2, sr2, p2]))
    return b + peak1 + peak2


def _param_names_double_generalized() -> list[str]:
    """Parameter names in the same order as :func:`_eval_double_generalized` unpacks them."""
    return ["b", "A1", "phi1", "sigma_left1", "sigma_right1", "p1", "A2", "phi2", "sigma_left2", "sigma_right2", "p2"]


def _initial_raw_double_generalized(
    theta: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    sigma_min: float,
) -> npt.NDArray[np.floating]:
    """Initial raw (unconstrained) parameter vector for the double generalized-Gaussian model.

    Seeds two peaks via :func:`_seed_two_peaks`, then estimates each peak's own
    widths/exponent via :func:`_contour_sigma_p` restricted to the points
    nearer to that peak than to the other (:func:`_local_mask`).
    Returns ``raw`` in softplus-space, 11 entries matching
    :func:`_param_names_double_generalized`.
    """
    span = max(float(theta.max() - theta.min()), 1e-6)
    sigma_max = span
    sigma0 = max(span / 10.0, 2.0 * sigma_min)

    b0, phi1_0, A1_0, phi2_0, A2_0 = _seed_two_peaks(theta, curve, sigma_min)

    mask1 = _local_mask(theta, phi1_0, phi2_0)
    mask2 = ~mask1
    sl1, pl1 = _contour_sigma_p(theta[mask1], curve[mask1], phi1_0, A1_0, b0, "left", sigma0, sigma_min, sigma_max)
    sr1, pr1 = _contour_sigma_p(theta[mask1], curve[mask1], phi1_0, A1_0, b0, "right", sigma0, sigma_min, sigma_max)
    sl2, pl2 = _contour_sigma_p(theta[mask2], curve[mask2], phi2_0, A2_0, b0, "left", sigma0, sigma_min, sigma_max)
    sr2, pr2 = _contour_sigma_p(theta[mask2], curve[mask2], phi2_0, A2_0, b0, "right", sigma0, sigma_min, sigma_max)
    p1_candidates = [p for p in (pl1, pr1) if p is not None]
    p2_candidates = [p for p in (pl2, pr2) if p is not None]
    p1_0 = float(np.mean(p1_candidates)) if p1_candidates else 2.0
    p2_0 = float(np.mean(p2_candidates)) if p2_candidates else 2.0

    return np.array(
        [
            b0,
            float(_inv_softplus(A1_0)),
            phi1_0,
            float(_inv_softplus(sl1 - sigma_min)),
            float(_inv_softplus(sr1 - sigma_min)),
            float(_inv_softplus(p1_0)),
            float(_inv_softplus(A2_0)),
            phi2_0,
            float(_inv_softplus(sl2 - sigma_min)),
            float(_inv_softplus(sr2 - sigma_min)),
            float(_inv_softplus(p2_0)),
        ],
        dtype=float,
    )


def _eval_double_gaussian(theta: npt.NDArray[np.floating], params: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Evaluate the double Gaussian model on ``theta`` for a single neuron.

    Packed parameter vector ``[b, A1, phi1, sigma1, A2, phi2, sigma2]``; the
    two peaks share one baseline and are summed. Built from two calls to
    :func:`_eval_gaussian` with each peak's own baseline zeroed.
    """
    b, A1, phi1, s1, A2, phi2, s2 = params
    peak1 = _eval_gaussian(theta, np.array([0.0, A1, phi1, s1]))
    peak2 = _eval_gaussian(theta, np.array([0.0, A2, phi2, s2]))
    return b + peak1 + peak2


def _param_names_double_gaussian() -> list[str]:
    """Parameter names in the same order as :func:`_eval_double_gaussian` unpacks them."""
    return ["b", "A1", "phi1", "sigma1", "A2", "phi2", "sigma2"]


def _initial_raw_double_gaussian(
    theta: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    sigma_min: float,
) -> npt.NDArray[np.floating]:
    """Initial raw (unconstrained) parameter vector for the double Gaussian model.

    Seeds two peaks via :func:`_seed_two_peaks`, then estimates each peak's
    width via :func:`_fwhm_sigma` restricted to the points nearer to that peak
    than to the other (:func:`_local_mask`). Returns ``raw`` in softplus-space,
    7 entries matching :func:`_param_names_double_gaussian`.
    """
    span = max(float(theta.max() - theta.min()), 1e-6)
    sigma_max = span
    sigma0 = max(span / 10.0, 2.0 * sigma_min)

    b0, phi1_0, A1_0, phi2_0, A2_0 = _seed_two_peaks(theta, curve, sigma_min)

    mask1 = _local_mask(theta, phi1_0, phi2_0)
    mask2 = ~mask1
    s1_0 = _fwhm_sigma(theta[mask1], curve[mask1], phi1_0, A1_0, b0, sigma0, sigma_min, sigma_max)
    s2_0 = _fwhm_sigma(theta[mask2], curve[mask2], phi2_0, A2_0, b0, sigma0, sigma_min, sigma_max)

    return np.array(
        [
            b0,
            float(_inv_softplus(A1_0)),
            phi1_0,
            float(_inv_softplus(s1_0 - sigma_min)),
            float(_inv_softplus(A2_0)),
            phi2_0,
            float(_inv_softplus(s2_0 - sigma_min)),
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


def _apply_constraints_double_generalized(raw: torch.Tensor, sigma_min: float) -> torch.Tensor:
    """Map unconstrained ``raw`` (n_cells, 11) to bounded double generalized-Gaussian parameters.

    Layout: ``[b, A1, phi1, sigma_left1, sigma_right1, p1, A2, phi2,
    sigma_left2, sigma_right2, p2]``. ``b``, ``phi1``, ``phi2`` are free; the
    rest use softplus to stay positive.
    """
    F = torch.nn.functional
    b = raw[:, 0:1]
    A1 = F.softplus(raw[:, 1:2])
    phi1 = raw[:, 2:3]
    sl1 = sigma_min + F.softplus(raw[:, 3:4])
    sr1 = sigma_min + F.softplus(raw[:, 4:5])
    p1 = F.softplus(raw[:, 5:6])
    A2 = F.softplus(raw[:, 6:7])
    phi2 = raw[:, 7:8]
    sl2 = sigma_min + F.softplus(raw[:, 8:9])
    sr2 = sigma_min + F.softplus(raw[:, 9:10])
    p2 = F.softplus(raw[:, 10:11])
    return torch.cat([b, A1, phi1, sl1, sr1, p1, A2, phi2, sl2, sr2, p2], dim=1)


def _apply_constraints_double_gaussian(raw: torch.Tensor, sigma_min: float) -> torch.Tensor:
    """Map unconstrained ``raw`` (n_cells, 7) to bounded double Gaussian parameters.

    Layout: ``[b, A1, phi1, sigma1, A2, phi2, sigma2]``. ``b``, ``phi1``,
    ``phi2`` are free; the sigmas use softplus.
    """
    F = torch.nn.functional
    b = raw[:, 0:1]
    A1 = F.softplus(raw[:, 1:2])
    phi1 = raw[:, 2:3]
    s1 = sigma_min + F.softplus(raw[:, 3:4])
    A2 = F.softplus(raw[:, 4:5])
    phi2 = raw[:, 5:6]
    s2 = sigma_min + F.softplus(raw[:, 6:7])
    return torch.cat([b, A1, phi1, s1, A2, phi2, s2], dim=1)


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


def _eval_double_generalized_torch(theta: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Batched, differentiable equivalent of :func:`_eval_double_generalized`.

    ``params`` has shape ``(n_cells, 11)``, same layout as
    :func:`_param_names_double_generalized`. Built from two calls to
    :func:`_eval_tilbury_torch` (each peak's own baseline zeroed) summed with
    the shared baseline.
    """
    b = params[:, 0:1]
    zeros = torch.zeros_like(b)
    peak1 = torch.cat([zeros, params[:, 1:6]], dim=1)
    peak2 = torch.cat([zeros, params[:, 6:11]], dim=1)
    return b + _eval_tilbury_torch(theta, peak1) + _eval_tilbury_torch(theta, peak2)


def _eval_double_gaussian_torch(theta: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Batched, differentiable equivalent of :func:`_eval_double_gaussian`.

    ``params`` has shape ``(n_cells, 7)``, same layout as
    :func:`_param_names_double_gaussian`. Built from two calls to
    :func:`_eval_gaussian_torch` (each peak's own baseline zeroed) summed with
    the shared baseline.
    """
    b = params[:, 0:1]
    zeros = torch.zeros_like(b)
    peak1 = torch.cat([zeros, params[:, 1:4]], dim=1)
    peak2 = torch.cat([zeros, params[:, 4:7]], dim=1)
    return b + _eval_gaussian_torch(theta, peak1) + _eval_gaussian_torch(theta, peak2)


def _fit_all_neurons_torch(
    theta: npt.NDArray[np.floating],
    curves_train: npt.NDArray[np.floating],
    curves_val: npt.NDArray[np.floating],
    sigma_min: float,
    device: str,
    num_steps: int,
    learning_rate: float,
    verbose: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Batched Adam fit of the Tilbury model to every neuron jointly.

    Per-neuron early stopping on the validation curve: at every step the
    current params are scored against ``curves_val`` (same ``theta`` grid as
    train, so the train-step prediction is reused, no extra forward pass),
    and the best-so-far params are kept per neuron rather than just the final
    step's params. This guards against overfitting the train curve.

    Returns
    -------
    best_params : np.ndarray
        Packed parameters, shape ``(n_cells, 6)``, best on validation R^2.
    r2_init : np.ndarray
        Train-curve R^2 of the analytic initial guess (pre-training), shape ``(n_cells,)``.
    """
    n_cells = curves_train.shape[0]
    raw_init = np.stack([_initial_raw_tilbury(theta, curves_train[n], sigma_min) for n in range(n_cells)])

    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    train_t = torch.as_tensor(curves_train, dtype=torch.float32, device=device)
    val_t = torch.as_tensor(curves_val, dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(torch.as_tensor(raw_init, dtype=torch.float32, device=device))

    with torch.no_grad():
        init_params = _apply_constraints_tilbury(raw, sigma_min)
        r2_init = _r2_batched_torch(_eval_tilbury_torch(theta_t, init_params), train_t)

    best_val_r2 = torch.full((n_cells,), float("-inf"), device=device)
    best_params = init_params.clone()

    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    steps = tqdm(range(num_steps), desc="descent fit") if verbose else range(num_steps)
    for step in steps:
        optimizer.zero_grad()
        params = _apply_constraints_tilbury(raw, sigma_min)
        pred = _eval_tilbury_torch(theta_t, params)
        loss = torch.mean((pred - train_t) ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_r2 = _r2_batched_torch(pred, val_t)
            improved = val_r2 > best_val_r2
            best_val_r2 = torch.where(improved, val_r2, best_val_r2)
            best_params = torch.where(improved[:, None], params.detach(), best_params)

        if verbose and step % 100 == 0:
            steps.set_postfix(loss=f"{loss.item():.4g}", val_r2=f"{val_r2.mean().item():.3g}")

    return best_params.cpu().numpy().astype(np.float64), r2_init.cpu().numpy().astype(np.float64)


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


def _fit_all_neurons_torch_gaussian(
    theta: npt.NDArray[np.floating],
    curves_train: npt.NDArray[np.floating],
    curves_val: npt.NDArray[np.floating],
    sigma_min: float,
    device: str,
    num_steps: int,
    learning_rate: float,
    verbose: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Batched Adam fit of the control Gaussian model to every neuron jointly.

    Per-neuron early stopping on the validation curve; see :func:`_fit_all_neurons_torch`.

    Returns
    -------
    best_params : np.ndarray
        Packed parameters, shape ``(n_cells, 4)``, best on validation R^2.
    r2_init : np.ndarray
        Train-curve R^2 of the analytic initial guess (pre-training), shape ``(n_cells,)``.
    """
    n_cells = curves_train.shape[0]
    raw_init = np.stack([_initial_raw_gaussian(theta, curves_train[n], sigma_min) for n in range(n_cells)])

    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    train_t = torch.as_tensor(curves_train, dtype=torch.float32, device=device)
    val_t = torch.as_tensor(curves_val, dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(torch.as_tensor(raw_init, dtype=torch.float32, device=device))

    with torch.no_grad():
        init_params = _apply_constraints_gaussian(raw, sigma_min)
        r2_init = _r2_batched_torch(_eval_gaussian_torch(theta_t, init_params), train_t)

    best_val_r2 = torch.full((n_cells,), float("-inf"), device=device)
    best_params = init_params.clone()

    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    steps = tqdm(range(num_steps), desc="descent fit control") if verbose else range(num_steps)
    for step in steps:
        optimizer.zero_grad()
        params = _apply_constraints_gaussian(raw, sigma_min)
        pred = _eval_gaussian_torch(theta_t, params)
        loss = torch.mean((pred - train_t) ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_r2 = _r2_batched_torch(pred, val_t)
            improved = val_r2 > best_val_r2
            best_val_r2 = torch.where(improved, val_r2, best_val_r2)
            best_params = torch.where(improved[:, None], params.detach(), best_params)

        if verbose and step % 100 == 0:
            steps.set_postfix(loss=f"{loss.item():.4g}", val_r2=f"{val_r2.mean().item():.3g}")

    return best_params.cpu().numpy().astype(np.float64), r2_init.cpu().numpy().astype(np.float64)


def _fit_all_neurons_torch_double_generalized(
    theta: npt.NDArray[np.floating],
    curves_train: npt.NDArray[np.floating],
    curves_val: npt.NDArray[np.floating],
    sigma_min: float,
    device: str,
    num_steps: int,
    learning_rate: float,
    verbose: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Batched Adam fit of the double generalized-Gaussian model to every neuron jointly.

    Per-neuron early stopping on the validation curve; see :func:`_fit_all_neurons_torch`.

    Returns
    -------
    best_params : np.ndarray
        Packed parameters, shape ``(n_cells, 11)``, best on validation R^2.
    r2_init : np.ndarray
        Train-curve R^2 of the analytic initial guess (pre-training), shape ``(n_cells,)``.
    """
    n_cells = curves_train.shape[0]
    raw_init = np.stack([_initial_raw_double_generalized(theta, curves_train[n], sigma_min) for n in range(n_cells)])

    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    train_t = torch.as_tensor(curves_train, dtype=torch.float32, device=device)
    val_t = torch.as_tensor(curves_val, dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(torch.as_tensor(raw_init, dtype=torch.float32, device=device))

    with torch.no_grad():
        init_params = _apply_constraints_double_generalized(raw, sigma_min)
        r2_init = _r2_batched_torch(_eval_double_generalized_torch(theta_t, init_params), train_t)

    best_val_r2 = torch.full((n_cells,), float("-inf"), device=device)
    best_params = init_params.clone()

    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    steps = tqdm(range(num_steps), desc="descent fit double generalized") if verbose else range(num_steps)
    for step in steps:
        optimizer.zero_grad()
        params = _apply_constraints_double_generalized(raw, sigma_min)
        pred = _eval_double_generalized_torch(theta_t, params)
        loss = torch.mean((pred - train_t) ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_r2 = _r2_batched_torch(pred, val_t)
            improved = val_r2 > best_val_r2
            best_val_r2 = torch.where(improved, val_r2, best_val_r2)
            best_params = torch.where(improved[:, None], params.detach(), best_params)

        if verbose and step % 100 == 0:
            steps.set_postfix(loss=f"{loss.item():.4g}", val_r2=f"{val_r2.mean().item():.3g}")

    return best_params.cpu().numpy().astype(np.float64), r2_init.cpu().numpy().astype(np.float64)


def _fit_all_neurons_torch_double_gaussian(
    theta: npt.NDArray[np.floating],
    curves_train: npt.NDArray[np.floating],
    curves_val: npt.NDArray[np.floating],
    sigma_min: float,
    device: str,
    num_steps: int,
    learning_rate: float,
    verbose: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Batched Adam fit of the double Gaussian model to every neuron jointly.

    Per-neuron early stopping on the validation curve; see :func:`_fit_all_neurons_torch`.

    Returns
    -------
    best_params : np.ndarray
        Packed parameters, shape ``(n_cells, 7)``, best on validation R^2.
    r2_init : np.ndarray
        Train-curve R^2 of the analytic initial guess (pre-training), shape ``(n_cells,)``.
    """
    n_cells = curves_train.shape[0]
    raw_init = np.stack([_initial_raw_double_gaussian(theta, curves_train[n], sigma_min) for n in range(n_cells)])

    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    train_t = torch.as_tensor(curves_train, dtype=torch.float32, device=device)
    val_t = torch.as_tensor(curves_val, dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(torch.as_tensor(raw_init, dtype=torch.float32, device=device))

    with torch.no_grad():
        init_params = _apply_constraints_double_gaussian(raw, sigma_min)
        r2_init = _r2_batched_torch(_eval_double_gaussian_torch(theta_t, init_params), train_t)

    best_val_r2 = torch.full((n_cells,), float("-inf"), device=device)
    best_params = init_params.clone()

    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    steps = tqdm(range(num_steps), desc="descent fit double gaussian") if verbose else range(num_steps)
    for step in steps:
        optimizer.zero_grad()
        params = _apply_constraints_double_gaussian(raw, sigma_min)
        pred = _eval_double_gaussian_torch(theta_t, params)
        loss = torch.mean((pred - train_t) ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_r2 = _r2_batched_torch(pred, val_t)
            improved = val_r2 > best_val_r2
            best_val_r2 = torch.where(improved, val_r2, best_val_r2)
            best_params = torch.where(improved[:, None], params.detach(), best_params)

        if verbose and step % 100 == 0:
            steps.set_postfix(loss=f"{loss.item():.4g}", val_r2=f"{val_r2.mean().item():.3g}")

    return best_params.cpu().numpy().astype(np.float64), r2_init.cpu().numpy().astype(np.float64)


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
    """

    schema_version: str = "v8"
    data_config_name: str = "even"
    spks_type: SpksTypes = "sigrebase"
    activity_parameters_name: str = "raw"
    num_bins: int = 100
    reliability_fraction_active_thresholds: Optional[tuple[float, float]] = (0.3, 0.1)

    sigma_min: ClassVar[float] = 1.0
    max_missing_position_percentage: ClassVar[float] = 5.0
    display_name: ClassVar[str] = "tilbury_fit"
    _result_handling: ClassVar[dict[str, str]] = {
        "idx_keep": "ragged",
        "dist_centers": "skip",
        "best_env": "skip",
        "param_names": "skip",
        "param_names_control": "skip",
        "param_names_double_tilbury": "skip",
        "param_names_double_control": "skip",
        "eig_tilbury": "pad",
        "eig_control": "pad",
        "eig_double_tilbury": "pad",
        "eig_double_control": "pad",
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
    def param_names_double_tilbury(self) -> list[str]:
        return _param_names_double_generalized()

    @property
    def param_names_double_control(self) -> list[str]:
        return _param_names_double_gaussian()

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
        """Fit the Tilbury law per neuron with train/val/test cross-validation.

        Params are fit by Adam on the train curve, but selected per neuron by
        best R^2 on the validation curve across the training sweep (guards
        against overfitting) rather than just taking the final step's params.

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

        # Per-neuron fits: Tilbury model, plain-Gaussian control, and their
        # double-peak (two summed bumps) counterparts, all side by side.
        n_neurons = curves["train"].shape[0]
        r2 = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        r2c = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        r2dg = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        r2dc = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        pearson = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        pearson_c = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        pearson_dg = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        pearson_dc = {split: np.full(n_neurons, np.nan) for split in _SPLITS}

        params, r2_init = _fit_all_neurons_torch(
            theta, curves["train"], curves["validation"], self.sigma_min, device, gd_num_steps, gd_learning_rate, verbose
        )
        params_c, r2_init_c = _fit_all_neurons_torch_gaussian(
            theta, curves["train"], curves["validation"], self.sigma_min, device, gd_num_steps, gd_learning_rate, verbose
        )
        params_dg, r2_init_dg = _fit_all_neurons_torch_double_generalized(
            theta, curves["train"], curves["validation"], self.sigma_min, device, gd_num_steps, gd_learning_rate, verbose
        )
        params_dc, r2_init_dc = _fit_all_neurons_torch_double_gaussian(
            theta, curves["train"], curves["validation"], self.sigma_min, device, gd_num_steps, gd_learning_rate, verbose
        )

        # Score each fit on every split (cheap, done serially).
        for n in range(n_neurons):
            x, xc = params[n], params_c[n]
            xdg, xdc = params_dg[n], params_dc[n]
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
                if not np.any(np.isnan(xdg)):
                    pred_dg = _eval_double_generalized(theta, xdg)
                    r2dg[split][n] = _r2(pred_dg, c)
                    pearson_dg[split][n] = _pearson(pred_dg, c)
                if not np.any(np.isnan(xdc)):
                    pred_dc = _eval_double_gaussian(theta, xdc)
                    r2dc[split][n] = _r2(pred_dc, c)
                    pearson_dc[split][n] = _pearson(pred_dc, c)

        # Population dimensionality: eigenvalue spectra of the (N, P) tuning-curve
        # matrices. Modeled curves are reconstructed for neurons with finite fits;
        # raw measured placefields (NaN-free by construction) use every selected neuron.
        ok = ~np.isnan(params).any(axis=1)
        okc = ~np.isnan(params_c).any(axis=1)
        ok_both = ok & okc
        mat_tilbury = np.stack([_eval_tilbury(theta, params[n]) for n in np.flatnonzero(ok_both)])
        mat_control = np.stack([_eval_gaussian(theta, params_c[n]) for n in np.flatnonzero(ok_both)])
        better = pearson["test"][ok_both] > pearson_c["test"][ok_both]
        mat_better = np.where(better[:, None], mat_tilbury, mat_control)
        eig_tilbury = _curve_spectrum(mat_tilbury)
        eig_control = _curve_spectrum(mat_control)
        eig_better = _curve_spectrum(mat_better)
        eig_raw_train = _curve_spectrum(curves["train"])
        eig_raw_test = _curve_spectrum(curves["test"])

        ok_double_both = (~np.isnan(params_dg).any(axis=1)) & (~np.isnan(params_dc).any(axis=1))
        mat_double_tilbury = np.stack([_eval_double_generalized(theta, params_dg[n]) for n in np.flatnonzero(ok_double_both)])
        mat_double_control = np.stack([_eval_double_gaussian(theta, params_dc[n]) for n in np.flatnonzero(ok_double_both)])
        better_double = pearson_dg["test"][ok_double_both] > pearson_dc["test"][ok_double_both]
        mat_better_double = np.where(better_double[:, None], mat_double_tilbury, mat_double_control)
        eig_double_tilbury = _curve_spectrum(mat_double_tilbury)
        eig_double_control = _curve_spectrum(mat_double_control)
        eig_better_double = _curve_spectrum(mat_better_double)

        # Overall best-of-four: per neuron, whichever of the single/double
        # generalized/Gaussian fits scores highest on test Pearson, restricted
        # to neurons where all four fits are finite.
        ok_all = ok_both & ok_double_both
        idx_all = np.flatnonzero(ok_all)
        cand_mats = np.stack(
            [
                np.stack([_eval_tilbury(theta, params[n]) for n in idx_all]),
                np.stack([_eval_gaussian(theta, params_c[n]) for n in idx_all]),
                np.stack([_eval_double_generalized(theta, params_dg[n]) for n in idx_all]),
                np.stack([_eval_double_gaussian(theta, params_dc[n]) for n in idx_all]),
            ],
            axis=1,
        )  # (n_kept, 4, P)
        cand_pearson = np.stack(
            [pearson["test"][ok_all], pearson_c["test"][ok_all], pearson_dg["test"][ok_all], pearson_dc["test"][ok_all]],
            axis=1,
        )  # (n_kept, 4)
        best_idx = np.argmax(cand_pearson, axis=1)
        mat_better_overall = cand_mats[np.arange(cand_mats.shape[0]), best_idx]
        eig_better_overall = _curve_spectrum(mat_better_overall)

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
            "params_double_tilbury": params_dg,
            "r2_init_double_tilbury": r2_init_dg,
            "r2_train_double_tilbury": r2dg["train"],
            "r2_val_double_tilbury": r2dg["validation"],
            "r2_test_double_tilbury": r2dg["test"],
            "pearson_train_double_tilbury": pearson_dg["train"],
            "pearson_val_double_tilbury": pearson_dg["validation"],
            "pearson_test_double_tilbury": pearson_dg["test"],
            "params_double_control": params_dc,
            "r2_init_double_control": r2_init_dc,
            "r2_train_double_control": r2dc["train"],
            "r2_val_double_control": r2dc["validation"],
            "r2_test_double_control": r2dc["test"],
            "pearson_train_double_control": pearson_dc["train"],
            "pearson_val_double_control": pearson_dc["validation"],
            "pearson_test_double_control": pearson_dc["test"],
            "idx_keep": idx_keep,
            "eig_tilbury": eig_tilbury,
            "eig_control": eig_control,
            "eig_better": eig_better,
            "eig_double_tilbury": eig_double_tilbury,
            "eig_double_control": eig_double_control,
            "eig_better_double": eig_better_double,
            "eig_better_overall": eig_better_overall,
            "eig_raw_train": eig_raw_train,
            "eig_raw_test": eig_raw_test,
            "dist_centers": theta,
            "best_env": best_env,
            "param_names": _param_names(),
            "param_names_control": _param_names_gaussian(),
            "param_names_double_tilbury": _param_names_double_generalized(),
            "param_names_double_control": _param_names_double_gaussian(),
        }
