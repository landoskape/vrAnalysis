"""TilburyFitConfig — fit the Tilbury tuning law to each neuron's placefield.

Implements the "AI-discovered tuning law" of Tilbury et al. (bioRxiv
2025.11.12.688086): each tuning curve is modelled as a sum of up to two
*asymmetric generalized-Gaussian* peaks.  For one neuron over positions
``theta``::

    f(theta) = b + sum_i A_i * exp(-(|theta - phi_i| / sigma_i(theta))^p_i)

where ``sigma_i(theta) = sigma_left_i`` when ``theta < phi_i`` and
``sigma_right_i`` otherwise.  A single peak has 6 free parameters
(``b, A, phi, sigma_left, sigma_right, p``); two peaks have 11.

Unlike the orientation special case in the paper (shared width, fixed
exponent 2, second peak pinned a half-period away), place fields on a linear
track are not periodic, so the two peak centres are fitted independently.

Per session the trial-averaged placefield is built for the train, validation
and test splits (``registry.time_split``), exactly as the rest of the
manuscript pipeline does.  Each neuron is fitted *independently*: a 1-peak and
a 2-peak model are least-squares-fitted on the **train** curve, the number of
peaks is chosen by R^2 on the **validation** curve, and the reported quality is
R^2 on the held-out **test** curve.

This mirrors the placefield extraction path of :mod:`configs.cvpca` and the
split / activity-normalisation handling of :mod:`configs.placefield_structure`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Optional

import contextlib

import joblib
import numpy as np
import numpy.typing as npt
import torch
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from tqdm import tqdm

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


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object: tqdm):
    """Patch ``joblib`` so a tqdm bar advances as each task completes."""

    class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# ---------------------------------------------------------------------------
# Tilbury generalized double-Gaussian math (one neuron at a time)
# ---------------------------------------------------------------------------


def _eval_tilbury(theta: npt.NDArray[np.floating], params: npt.NDArray[np.floating], n_peaks: int) -> npt.NDArray[np.floating]:
    """Evaluate the Tilbury tuning law on ``theta`` for a single neuron.

    Parameters
    ----------
    theta : np.ndarray
        Position bin centres, shape ``(P,)``.
    params : np.ndarray
        Packed parameter vector of length ``1 + 5 * n_peaks`` laid out as
        ``[b, A(k), phi(k), sigma_left(k), sigma_right(k), p(k)]``.
    n_peaks : int
        Number of peaks (1 or 2).

    Returns
    -------
    np.ndarray
        Fitted curve, shape ``(P,)``.
    """
    b, A, phi, sl, sr, p = _unpack(params, n_peaks)
    diff = theta[None, :] - phi[:, None]  # (k, P)
    sigma = np.where(diff < 0, sl[:, None], sr[:, None])  # (k, P)
    bumps = A[:, None] * np.exp(-((np.abs(diff) / np.clip(sigma, 1e-6, None)) ** p[:, None]))
    return b + bumps.sum(axis=0)


def _unpack(params: npt.NDArray[np.floating], n_peaks: int) -> tuple:
    """Split a packed parameter vector into ``(b, A, phi, sigma_left, sigma_right, p)``."""
    k = n_peaks
    b = params[0]
    A = params[1 : 1 + k]
    phi = params[1 + k : 1 + 2 * k]
    sl = params[1 + 2 * k : 1 + 3 * k]
    sr = params[1 + 3 * k : 1 + 4 * k]
    p = params[1 + 4 * k : 1 + 5 * k]
    return b, A, phi, sl, sr, p


def _param_names(n_peaks: int) -> list[str]:
    """Parameter names in the same order as ``_unpack`` packs them."""
    k = n_peaks
    return (
        ["b"]
        + [f"A{i + 1}" for i in range(k)]
        + [f"phi{i + 1}" for i in range(k)]
        + [f"sigma_left{i + 1}" for i in range(k)]
        + [f"sigma_right{i + 1}" for i in range(k)]
        + [f"p{i + 1}" for i in range(k)]
    )


def _sort_peaks_by_amplitude(params: npt.NDArray[np.floating], n_peaks: int) -> npt.NDArray[np.floating]:
    """Reorder peaks so peak 1 always has the larger amplitude ``A``."""
    if n_peaks < 2 or np.any(np.isnan(params)):
        return params
    b, A, phi, sl, sr, p = _unpack(params, n_peaks)
    order = np.argsort(-A)
    return np.concatenate([[b], A[order], phi[order], sl[order], sr[order], p[order]])


def _r2(pred: npt.NDArray[np.floating], actual: npt.NDArray[np.floating]) -> float:
    """Coefficient of determination of ``pred`` against ``actual`` (one curve)."""
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


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
    p_min: float,
    p_max: float,
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
    p_cand = float(np.clip(p_cand, p_min, p_max))
    sigma = float(np.clip(d_half / (2.0 * np.log(2.0)) ** (1.0 / p_cand), sigma_min, sigma_max))
    return sigma, p_cand


def _initial_guess_and_bounds(
    theta: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    n_peaks: int,
    sigma_min: float,
    p_min: float,
    p_max: float,
) -> tuple[npt.NDArray[np.floating], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]:
    """Build an initial parameter vector and (lower, upper) bounds for one neuron.

    Peaks are seeded greedily from the largest residual; the second peak is
    seeded after masking out a window around the first peak.
    """
    k = n_peaks
    theta_lo, theta_hi = float(theta.min()), float(theta.max())
    span = max(theta_hi - theta_lo, 1e-6)
    sigma_max = span
    sigma0 = max(span / 10.0, 2.0 * sigma_min)

    b0 = float(np.percentile(curve, 10.0))
    resid = curve - b0

    phi0, A0 = [], []
    work = resid.copy()
    for _ in range(k):
        i = int(np.argmax(work))
        phi0.append(float(theta[i]))
        A0.append(max(float(work[i]), 1e-3))
        # mask a window around this peak so the next peak is found elsewhere
        work = np.where(np.abs(theta - theta[i]) < sigma0, -np.inf, work)

    # Per-peak, per-side contour-radius estimate of (sigma_left, sigma_right, p);
    # combine the two sides' p estimates, falling back to the flat seed above
    # when a side's contour isn't trustworthy (mirrors Tilbury's analytic init).
    sl0, sr0, p0 = [], [], []
    for phi_i, A_i in zip(phi0, A0):
        sigma_l, p_l = _contour_sigma_p(theta, curve, phi_i, A_i, b0, "left", sigma0, sigma_min, sigma_max, p_min, p_max)
        sigma_r, p_r = _contour_sigma_p(theta, curve, phi_i, A_i, b0, "right", sigma0, sigma_min, sigma_max, p_min, p_max)
        sl0.append(sigma_l)
        sr0.append(sigma_r)
        p_candidates = [p for p in (p_l, p_r) if p is not None]
        p0.append(float(np.mean(p_candidates)) if p_candidates else 2.0)

    x0 = np.concatenate([[b0], A0, phi0, sl0, sr0, p0]).astype(float)

    lower = np.concatenate([[-np.inf], [0.0] * k, [theta_lo] * k, [sigma_min] * k, [sigma_min] * k, [p_min] * k])
    upper = np.concatenate([[np.inf], [np.inf] * k, [theta_hi] * k, [sigma_max] * k, [sigma_max] * k, [p_max] * k])

    # Keep the initial guess strictly inside the box.
    x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)
    return x0, (lower, upper)


def _fit_neuron(
    theta: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    n_peaks: int,
    sigma_min: float,
    p_min: float,
    p_max: float,
    max_nfev: int = 2000,
) -> npt.NDArray[np.floating]:
    """Least-squares fit of an ``n_peaks`` Tilbury model to one curve.

    Returns the packed parameter vector (length ``1 + 5 * n_peaks``), or all-NaN
    if the optimiser fails.
    """
    x0, bounds = _initial_guess_and_bounds(theta, curve, n_peaks, sigma_min, p_min, p_max)

    def residual(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return _eval_tilbury(theta, x, n_peaks) - curve

    try:
        result = least_squares(residual, x0, bounds=bounds, method="trf", max_nfev=max_nfev)
        return result.x
    except (ValueError, np.linalg.LinAlgError):
        return np.full(1 + 5 * n_peaks, np.nan)


def _fit_one_neuron(
    theta: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    sigma_min: float,
    p_min: float,
    p_max: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Fit both the 1-peak and 2-peak Tilbury models to one (train) curve.

    Module-level so it can be dispatched by ``joblib.Parallel``.
    """
    x1 = _fit_neuron(theta, curve, 1, sigma_min, p_min, p_max)
    x2 = _fit_neuron(theta, curve, 2, sigma_min, p_min, p_max)
    x2 = _sort_peaks_by_amplitude(x2, 2)
    return x1, x2


# ---------------------------------------------------------------------------
# Batched torch / gradient-descent fitting (alternative to scipy least_squares)
# ---------------------------------------------------------------------------


def _eval_tilbury_torch(theta: torch.Tensor, params: torch.Tensor, n_peaks: int) -> torch.Tensor:
    """Batched, differentiable equivalent of :func:`_eval_tilbury`.

    Parameters
    ----------
    theta : torch.Tensor
        Position bin centres, shape ``(P,)``.
    params : torch.Tensor
        Packed parameters, shape ``(n_cells, 1 + 5 * n_peaks)``, same layout as
        :func:`_unpack`.
    n_peaks : int

    Returns
    -------
    torch.Tensor
        Fitted curves, shape ``(n_cells, P)``.
    """
    k = n_peaks
    b = params[:, 0:1]  # (n_cells, 1)
    A = params[:, 1 : 1 + k]  # (n_cells, k)
    phi = params[:, 1 + k : 1 + 2 * k]
    sl = params[:, 1 + 2 * k : 1 + 3 * k]
    sr = params[:, 1 + 3 * k : 1 + 4 * k]
    p = params[:, 1 + 4 * k : 1 + 5 * k]

    diff = theta[None, None, :] - phi[:, :, None]  # (n_cells, k, P)
    sigma = torch.where(diff < 0, sl[:, :, None], sr[:, :, None])
    bumps = A[:, :, None] * torch.exp(-((diff.abs() / sigma.clamp(min=1e-6)) ** p[:, :, None]))
    return b + bumps.sum(dim=1)


def _fit_all_neurons_torch(
    theta: npt.NDArray[np.floating],
    curves: npt.NDArray[np.floating],
    n_peaks: int,
    sigma_min: float,
    p_min: float,
    p_max: float,
    device: str,
    num_steps: int,
    learning_rate: float,
    verbose: bool = False,
) -> npt.NDArray[np.floating]:
    """Batched Adam fit of an ``n_peaks`` Tilbury model to every neuron jointly.

    All neurons share one optimizer step (Tilbury-style batched gradient
    descent) instead of scipy's per-neuron ``least_squares``. Bounds are the
    same ``(lower, upper)`` box used by the scipy path, enforced via
    ``torch.clamp`` inside the forward pass.

    Returns
    -------
    np.ndarray
        Packed parameters, shape ``(n_cells, 1 + 5 * n_peaks)``.
    """
    n_cells = curves.shape[0]
    n_params = 1 + 5 * n_peaks

    x0 = np.empty((n_cells, n_params), dtype=np.float64)
    lower = np.empty((n_cells, n_params), dtype=np.float64)
    upper = np.empty((n_cells, n_params), dtype=np.float64)
    for n in range(n_cells):
        x0[n], (lower[n], upper[n]) = _initial_guess_and_bounds(theta, curves[n], n_peaks, sigma_min, p_min, p_max)

    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    curves_t = torch.as_tensor(curves, dtype=torch.float32, device=device)
    lower_t = torch.as_tensor(lower, dtype=torch.float32, device=device)
    upper_t = torch.as_tensor(upper, dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(torch.as_tensor(x0, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    steps = tqdm(range(num_steps), desc=f"descent fit ({n_peaks}-peak)") if verbose else range(num_steps)
    for step in steps:
        optimizer.zero_grad()
        params = raw.clamp(min=lower_t, max=upper_t)
        pred = _eval_tilbury_torch(theta_t, params, n_peaks)
        loss = torch.mean((pred - curves_t) ** 2)
        loss.backward()
        optimizer.step()
        if verbose and step % 100 == 0:
            steps.set_postfix(loss=f"{loss.item():.4g}")

    with torch.no_grad():
        final = raw.clamp(min=lower_t, max=upper_t)
    return final.cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TilburyFitConfig(AnalysisConfigBase):
    """Fit the Tilbury double generalized-Gaussian tuning law per neuron.

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
    peakiness_bounds : ClassVar[tuple[float, float]]
        Lower / upper bounds on the per-peak peakiness exponent ``p``.
    """

    schema_version: str = "v1"
    data_config_name: str = "default"

    spks_type: SpksTypes = "sigrebase"
    activity_parameters_name: str = "raw"
    num_bins: int = 100
    reliability_fraction_active_thresholds: Optional[tuple[float, float]] = (0.3, 0.1)
    method: str = "descent"

    sigma_min: ClassVar[float] = 1.0
    peakiness_bounds: ClassVar[tuple[float, float]] = (0.5, 8.0)
    max_missing_position_percentage: ClassVar[float] = 5.0
    display_name: ClassVar[str] = "tilbury_fit"
    _result_handling: ClassVar[dict[str, str]] = {
        "idx_keep": "skip",
        "dist_centers": "skip",
        "best_env": "skip",
    }

    @staticmethod
    def _param_grid() -> dict:
        return {
            "method": ["least_squares", "descent"],
        }

    @property
    def param_names_1peak(self) -> list[str]:
        return _param_names(1)

    @property
    def param_names_2peak(self) -> list[str]:
        return _param_names(2)

    def validate(self) -> None:
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(f"Unknown activity_parameters_name {self.activity_parameters_name!r}. Available: {list(ACTIVITY_PARAMETERS_NAMES)}")
        lo, hi = self.peakiness_bounds
        if not (lo > 0 and hi > lo):
            raise ValueError(f"peakiness_bounds must satisfy 0 < lo < hi, got {self.peakiness_bounds}")

    def summary(self) -> str:
        rel, frac = self.reliability_fraction_active_thresholds
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"ap={self.activity_parameters_name}",
            f"bins={self.num_bins}",
            f"rel={rel}",
            f"frac={frac}",
            f"method={self.method}",
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
        n_jobs: Optional[int] = None,
        method: Optional[str] = None,
        device: Optional[str] = None,
        gd_num_steps: int = 4000,
        gd_learning_rate: float = 0.005,
    ) -> dict:
        """Fit the Tilbury law per neuron with train/val/test cross-validation.

        Parameters
        ----------
        session : B2Session
        registry : PopulationRegistry
        verbose : bool
            If True, show a tqdm progress bar — per-neuron completion for
            ``method="least_squares"``, optimizer step + loss for ``method="descent"``.
        n_jobs : int or None
            Worker count for the per-neuron fits (joblib). ``None`` (the default,
            used by the queue) resolves to SGE's ``NSLOTS`` when set — so on
            MYRIAD it uses exactly the cores requested via ``-pe smp`` and never
            oversubscribes the shared node — falling back to serial (``1``)
            off-cluster. Pass an explicit int to override (``-1`` = all local
            cores). Parallelism pays off for many / slow fits; for a handful of
            neurons the worker-startup overhead can make ``n_jobs=1`` faster.
            Ignored when ``method="descent"``.
        method : {"least_squares", "descent"} or None
            Per-neuron ``scipy.optimize.least_squares`` (default, CPU-only,
            matches existing cluster behaviour) or a batched ``torch``/Adam
            gradient descent fit over all neurons jointly. ``None`` (the default,
            used by the queue) resolves to ``"descent"`` when ``device="cuda"``
            and ``"least_squares"`` otherwise.
        device : str or None
            Torch device for ``method="descent"`` (e.g. ``"cpu"``, ``"cuda"``).
            ``None`` (the default) auto-detects: ``"cuda"`` if
            ``torch.cuda.is_available()`` else ``"cpu"``. Ignored when
            ``method="least_squares"``.
        gd_num_steps : int
            Adam iterations for ``method="descent"``. Ignored otherwise.
        gd_learning_rate : float
            Adam learning rate for ``method="descent"``. Ignored otherwise.
        """
        method = method or self.method
        if method not in ("least_squares", "descent"):
            raise ValueError(f"Unknown method {method!r}. Expected 'least_squares' or 'descent'.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if n_jobs is None:
            n_jobs = int(os.environ.get("NSLOTS", "0")) or 1

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

        # Per-neuron fits.
        n_neurons = curves["train"].shape[0]
        p_min, p_max = self.peakiness_bounds
        params_1 = np.full((n_neurons, 6), np.nan)
        params_2 = np.full((n_neurons, 11), np.nan)
        r2 = {f"{split}_{k}": np.full(n_neurons, np.nan) for split in _SPLITS for k in (1, 2)}

        if method == "least_squares":
            # Fit every neuron in parallel (each task fits both the 1- and 2-peak model).
            task = (delayed(_fit_one_neuron)(theta, curves["train"][n], self.sigma_min, p_min, p_max) for n in range(n_neurons))
            if verbose:
                with _tqdm_joblib(tqdm(total=n_neurons, desc="least_squares fit")):
                    fits = Parallel(n_jobs=n_jobs)(task)
            else:
                fits = Parallel(n_jobs=n_jobs)(task)
            for n, (x1, x2) in enumerate(fits):
                params_1[n] = x1
                params_2[n] = x2
        else:  # method == "descent"
            params_1 = _fit_all_neurons_torch(
                theta, curves["train"], 1, self.sigma_min, p_min, p_max, device, gd_num_steps, gd_learning_rate, verbose
            )
            params_2 = _fit_all_neurons_torch(
                theta, curves["train"], 2, self.sigma_min, p_min, p_max, device, gd_num_steps, gd_learning_rate, verbose
            )
            params_2 = np.stack([_sort_peaks_by_amplitude(params_2[n], 2) for n in range(n_neurons)])

        # Score each fit on every split (cheap, done serially).
        for n in range(n_neurons):
            x1, x2 = params_1[n], params_2[n]
            for split in _SPLITS:
                c = curves[split][n]
                if not np.any(np.isnan(x1)):
                    r2[f"{split}_1"][n] = _r2(_eval_tilbury(theta, x1, 1), c)
                if not np.any(np.isnan(x2)):
                    r2[f"{split}_2"][n] = _r2(_eval_tilbury(theta, x2, 2), c)

        # Model selection on validation: prefer 2 peaks only if it does better.
        v1, v2 = r2["validation_1"], r2["validation_2"]
        choose_2 = np.nan_to_num(v2, nan=-np.inf) > np.nan_to_num(v1, nan=-np.inf)
        n_peaks = np.where(choose_2, 2, 1).astype(int)
        r2_test = np.where(choose_2, r2["test_2"], r2["test_1"])

        return {
            "params_1peak": params_1,
            "params_2peak": params_2,
            "r2_train_1": r2["train_1"],
            "r2_train_2": r2["train_2"],
            "r2_val_1": r2["validation_1"],
            "r2_val_2": r2["validation_2"],
            "r2_test_1": r2["test_1"],
            "r2_test_2": r2["test_2"],
            "n_peaks": n_peaks,
            "r2_test": r2_test,
            "idx_keep": idx_keep,
            "dist_centers": theta,
            "best_env": best_env,
            "params_1peak_names": _param_names(1),
            "params_2peak_names": _param_names(2),
        }
