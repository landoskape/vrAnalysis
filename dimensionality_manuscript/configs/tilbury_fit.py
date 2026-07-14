"""TilburyFitConfig — fit the Tilbury tuning law to each neuron's placefield.

Implements the "AI-discovered tuning law" of Tilbury et al. (bioRxiv
2025.11.12.688086): each tuning curve is modelled as a single *asymmetric
generalized-Gaussian* peak.  For one neuron over positions ``theta``::

    f(theta) = b + A * exp(-(|theta - phi| / sigma(theta))^p)

where ``sigma(theta) = sigma_left`` when ``theta < phi`` and ``sigma_right``
otherwise, giving 6 free parameters (``b, A, phi, sigma_left, sigma_right, p``).

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
from .regression import VALID_SPKS_TYPES

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


def _fit_all_neurons_torch(
    theta: npt.NDArray[np.floating],
    curves: npt.NDArray[np.floating],
    sigma_min: float,
    device: str,
    num_steps: int,
    learning_rate: float,
    verbose: bool = False,
) -> npt.NDArray[np.floating]:
    """Batched Adam fit of the Tilbury model to every neuron jointly.

    Returns
    -------
    np.ndarray
        Packed parameters, shape ``(n_cells, 6)``.
    """
    n_cells = curves.shape[0]
    raw_init = np.stack([_initial_raw_tilbury(theta, curves[n], sigma_min) for n in range(n_cells)])

    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    curves_t = torch.as_tensor(curves, dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(torch.as_tensor(raw_init, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    steps = tqdm(range(num_steps), desc="descent fit") if verbose else range(num_steps)
    for step in steps:
        optimizer.zero_grad()
        params = _apply_constraints_tilbury(raw, sigma_min)
        pred = _eval_tilbury_torch(theta_t, params)
        loss = torch.mean((pred - curves_t) ** 2)
        loss.backward()
        optimizer.step()
        if verbose and step % 100 == 0:
            steps.set_postfix(loss=f"{loss.item():.4g}")

    with torch.no_grad():
        final = _apply_constraints_tilbury(raw, sigma_min)
    return final.cpu().numpy().astype(np.float64)


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
    curves: npt.NDArray[np.floating],
    sigma_min: float,
    device: str,
    num_steps: int,
    learning_rate: float,
    verbose: bool = False,
) -> npt.NDArray[np.floating]:
    """Batched Adam fit of the control Gaussian model to every neuron jointly.

    Returns
    -------
    np.ndarray
        Packed parameters, shape ``(n_cells, 4)``.
    """
    n_cells = curves.shape[0]
    raw_init = np.stack([_initial_raw_gaussian(theta, curves[n], sigma_min) for n in range(n_cells)])

    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    curves_t = torch.as_tensor(curves, dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(torch.as_tensor(raw_init, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    steps = tqdm(range(num_steps), desc="descent fit control") if verbose else range(num_steps)
    for step in steps:
        optimizer.zero_grad()
        params = _apply_constraints_gaussian(raw, sigma_min)
        pred = _eval_gaussian_torch(theta_t, params)
        loss = torch.mean((pred - curves_t) ** 2)
        loss.backward()
        optimizer.step()
        if verbose and step % 100 == 0:
            steps.set_postfix(loss=f"{loss.item():.4g}")

    with torch.no_grad():
        final = _apply_constraints_gaussian(raw, sigma_min)
    return final.cpu().numpy().astype(np.float64)


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

    schema_version: str = "v6"
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
        "eig_tilbury": "pad",
        "eig_control": "pad",
        "eig_raw_train": "pad",
        "eig_raw_test": "pad",
    }

    @staticmethod
    def _param_grid() -> dict:
        return {}

    @property
    def param_names(self) -> list[str]:
        return _param_names()

    @property
    def param_names_control(self) -> list[str]:
        return _param_names_gaussian()

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
        gd_num_steps: int = 8000,
        gd_learning_rate: float = 0.1,
    ) -> dict:
        """Fit the Tilbury law per neuron with train/val/test cross-validation.

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
            Adam iterations.
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

        # Per-neuron fits: Tilbury model and the plain-Gaussian control, side by side.
        n_neurons = curves["train"].shape[0]
        r2 = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        r2c = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        pearson = {split: np.full(n_neurons, np.nan) for split in _SPLITS}
        pearson_c = {split: np.full(n_neurons, np.nan) for split in _SPLITS}

        params = _fit_all_neurons_torch(theta, curves["train"], self.sigma_min, device, gd_num_steps, gd_learning_rate, verbose)
        params_c = _fit_all_neurons_torch_gaussian(theta, curves["train"], self.sigma_min, device, gd_num_steps, gd_learning_rate, verbose)

        # Score each fit on every split (cheap, done serially).
        for n in range(n_neurons):
            x, xc = params[n], params_c[n]
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

        return {
            "params": params,
            "r2_train": r2["train"],
            "r2_val": r2["validation"],
            "r2_test": r2["test"],
            "pearson_train": pearson["train"],
            "pearson_val": pearson["validation"],
            "pearson_test": pearson["test"],
            "params_control": params_c,
            "r2_train_control": r2c["train"],
            "r2_val_control": r2c["validation"],
            "r2_test_control": r2c["test"],
            "pearson_train_control": pearson_c["train"],
            "pearson_val_control": pearson_c["validation"],
            "pearson_test_control": pearson_c["test"],
            "idx_keep": idx_keep,
            "eig_tilbury": eig_tilbury,
            "eig_control": eig_control,
            "eig_better": eig_better,
            "eig_raw_train": eig_raw_train,
            "eig_raw_test": eig_raw_test,
            "dist_centers": theta,
            "best_env": best_env,
            "param_names": _param_names(),
            "param_names_control": _param_names_gaussian(),
        }
