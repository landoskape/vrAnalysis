"""GainRegressionConfig — reduced-rank regression across per-trial place-field gain.

:class:`~dimensionality_manuscript.figure_scripts.figure2.placefield_gain.PlacefieldGainViewer`
measures a per-trial multiplicative place-field gain matrix (neurons x trials) and displays its
neuron covariance. That covariance establishes *that* trial-to-trial gain is shared across neurons;
it says nothing about how low-dimensional the sharing is, or whether it generalizes off the trials
it was measured on.

This module answers both. Within each environment the gain matrix is split into a source half and a
target half of neurons, the trials are split into train/validation/test, and a reduced-rank
regression is fit from source gain to target gain. The cross-validated optimal rank is a direct
estimate of the dimensionality of shared gain fluctuation, and the held-out score says how much of
another neuron's gain is predictable from the population at all. Both are reported against a
trial-permuted null that keeps each neuron's own gain statistics and destroys only what the
neurons share.

Deliberately standalone. The ``regression_models`` / ``PopulationRegistry`` stack regresses
frame-by-frame activity through ``Population``/``TimeSplit`` chunk splits; here the sample axis is
*trials*, the data are a derived quantity, and the splits are trial splits. The Gaussian place-field
fit is likewise rebuilt here rather than imported from :class:`TilburyFitConfig` (a batched-Adam
torch fitter over a different data path) or from the figure script (an interactive viewer that is
free to change without invalidating stored results).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import torch
from scipy.optimize import curve_fit

from dimilibi.metrics import measure_r2, mse
from dimilibi.regression import ReducedRankRegression
from vrAnalysis.helpers import cross_validate_trials, reliability_loo, stable_hash
from vrAnalysis.helpers.optimization import golden_section_search
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.spkmaps import SpkmapParams, SpkmapProcessor
from vrAnalysis.sessions import B2Session, SpksTypes

from ..env_order import MAX_ENV_SLOTS, load_env_order
from ..pipeline.base import AnalysisConfigBase
from ..registry import PopulationRegistry
from .regression import VALID_SPKS_TYPES

#: Legal values for ``GainRegressionConfig.placefield_split``.
PLACEFIELD_SPLITS: tuple[str, ...] = ("train", "all")

#: Legal values for ``GainRegressionConfig.gain_transform``.
GAIN_TRANSFORMS: tuple[str, ...] = ("raw", "sqrt")

#: Fold order returned by ``cross_validate_trials`` given ``trial_fractions``.
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

#: Curve keys emitted per slot, NaN-padded to the widest rank range in the session.
CURVE_KEYS: tuple[str, ...] = (
    "rank_values",
    "mse_val_curve",
    "r2_val_curve",
    "mse_test_curve",
    "r2_test_curve",
    "mse_val_curve_null",
    "r2_val_curve_null",
    "mse_test_curve_null",
    "r2_test_curve_null",
)

#: Relative validation-MSE slack within which the *smallest* rank wins. The reported rank is a
#: dimensionality estimate, so a flat tail must not hand the answer to the largest rank evaluated:
#: past the point where extra components stop helping, the source data can be rank-deficient and
#: every higher rank scores identically.
RANK_SCORE_TOLERANCE: float = 1e-3


def _full_trial_indices(occupancy: np.ndarray, required_bins: np.ndarray) -> np.ndarray:
    """Return trials covering every required bin under Spkmap's NaN convention.

    Parameters
    ----------
    occupancy : np.ndarray
        Raw occupancy map, ``(trials, position bins)``.
    required_bins : np.ndarray
        Bin indices a traversal must cover to count as a full trial.

    Returns
    -------
    np.ndarray
        Indices of the full trials.
    """
    occupancy = np.asarray(occupancy)
    required_bins = np.asarray(required_bins, dtype=int)
    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a (trials, position bins) matrix")
    return np.flatnonzero(np.all(~np.isnan(occupancy[:, required_bins]), axis=1))


def _gaussian(position: np.ndarray, baseline: float, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Four-parameter Gaussian place-field model."""
    return baseline + amplitude * np.exp(-0.5 * ((position - center) / sigma) ** 2)


def _fit_gaussian(position: np.ndarray, curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one place field with a bounded four-parameter Gaussian.

    Parameters
    ----------
    position : np.ndarray
        Bin centers, ``(bins,)``.
    curve : np.ndarray
        Trial-averaged place field for one ROI, ``(bins,)``.

    Returns
    -------
    fitted : np.ndarray
        The fitted curve including its baseline, ``(bins,)``. All NaN if the fit failed.
    bump : np.ndarray
        The baseline-free Gaussian bump normalized to a peak of one, ``(bins,)``. Used as the
        spatial weighting for the gain numerator and denominator. All NaN if the fit failed.
    """
    valid = np.isfinite(position) & np.isfinite(curve)
    if np.sum(valid) < 4 or np.ptp(position[valid]) <= 0:
        return np.full(curve.shape, np.nan), np.full(curve.shape, np.nan)
    x, y = position[valid], curve[valid]
    span = float(np.ptp(x))
    bin_width = float(np.median(np.diff(np.unique(x))))
    baseline = float(np.nanpercentile(y, 10))
    amplitude = max(float(np.nanmax(y) - baseline), 1e-6)
    initial = (baseline, amplitude, float(x[np.nanargmax(y)]), max(span / 10, bin_width))
    try:
        params, _ = curve_fit(
            _gaussian,
            x,
            y,
            p0=initial,
            bounds=([-np.inf, 0.0, x.min(), max(bin_width / 2, 1e-6)], [np.inf, np.inf, x.max(), span]),
            maxfev=5000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return np.full(curve.shape, np.nan), np.full(curve.shape, np.nan)
    fitted = _gaussian(position, *params)
    bump = np.exp(-0.5 * ((position - params[2]) / params[3]) ** 2)
    return fitted, bump


def _weighted_row_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean over the bin axis of ``(rois, trials, bins)`` values."""
    valid = np.isfinite(values) & np.isfinite(weights[:, None, :]) & (weights[:, None, :] > 0)
    effective_weights = np.where(valid, weights[:, None, :], 0.0)
    weight_sum = np.sum(effective_weights, axis=2)
    return np.divide(
        np.sum(np.where(valid, values, 0.0) * effective_weights, axis=2),
        weight_sum,
        out=np.full(values.shape[:2], np.nan, dtype=float),
        where=weight_sum > 0,
    )


def _safe_gain(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide a ``(rois, trials)`` numerator by a per-ROI denominator, NaN where degenerate."""
    valid = np.isfinite(denominator) & (denominator > np.finfo(float).eps)
    return np.divide(
        numerator,
        denominator[:, None],
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=valid[:, None],
    )


def gaussian_gain_matrix(
    trial_maps: np.ndarray,
    positions: np.ndarray,
    prediction_trials: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trial multiplicative gain from locally fitted Gaussian place fields.

    The trial-averaged place field is fit with a four-parameter Gaussian. Its baseline-free bump
    supplies spatial weights; the numerator is the weighted trial map and the denominator is the
    identically weighted fitted curve, so a trial that matches the average place field has a gain
    of one.

    Parameters
    ----------
    trial_maps : np.ndarray
        Place-field maps in raster orientation, ``(rois, trials, bins)``.
    positions : np.ndarray
        Bin centers, ``(bins,)``.
    prediction_trials : np.ndarray or None
        Trial indices used to build the trial-averaged place field that is fit and used as the
        denominator. When None, every trial contributes. Gain is always returned for *all* trials,
        so passing the training trials here gives held-out trials an honest denominator.

    Returns
    -------
    gain : np.ndarray
        Multiplicative gain, ``(rois, trials)``. NaN where the fit or denominator is degenerate.
    fitted : np.ndarray
        Fitted place fields, ``(rois, bins)``.
    prediction : np.ndarray
        The trial-averaged place field that was fit, ``(rois, bins)``.
    """
    trial_maps = np.asarray(trial_maps, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if trial_maps.ndim != 3:
        raise ValueError("trial_maps must have shape (rois, trials, bins)")
    if positions.shape != (trial_maps.shape[2],):
        raise ValueError("positions must contain one value per place-field bin")

    subset = trial_maps if prediction_trials is None else trial_maps[:, np.asarray(prediction_trials, dtype=int)]
    if subset.shape[1] == 0:
        raise ValueError("prediction_trials must select at least one trial")
    prediction = np.nanmean(subset, axis=1)

    fitted = np.full_like(prediction, np.nan, dtype=float)
    weights = np.full_like(prediction, np.nan, dtype=float)
    for roi, curve in enumerate(prediction):
        fit, bump = _fit_gaussian(positions, curve)
        fitted[roi] = fit
        weights[roi] = bump
    numerator = _weighted_row_mean(trial_maps, weights)
    denominator = _weighted_row_mean(fitted[:, None, :], weights)[:, 0]
    return _safe_gain(numerator, denominator), fitted, prediction


def apply_gain_transform(gain: np.ndarray, transform: str) -> np.ndarray:
    """Compress the gain distribution before regression.

    Single-trial gain is sparse and heavy tailed: the median is zero and the upper tail runs past
    twenty, so squared error under ``"raw"`` is dominated by a handful of entries. ``"sqrt"`` pulls
    that tail in while leaving silent trials at exactly zero and staying non-negative, so rectified
    predictions remain meaningful. A log transform would do the opposite -- the many near-silent
    trials would dominate instead, at large negative values.

    Parameters
    ----------
    gain : np.ndarray
        Multiplicative gain, ``(neurons, trials)``.
    transform : str
        One of :data:`GAIN_TRANSFORMS`.

    Returns
    -------
    np.ndarray
        The transformed gain. Negative entries become NaN under ``"sqrt"`` and are dropped by the
        caller's finite screen.
    """
    if transform == "raw":
        return gain
    if transform != "sqrt":
        raise ValueError(f"Unknown gain transform {transform!r}. Available: {GAIN_TRANSFORMS}")
    return np.sqrt(gain, out=np.full_like(gain, np.nan), where=gain >= 0)


def _seed(*items) -> int:
    """Deterministic 32-bit seed from arbitrary hashable descriptors."""
    return int(stable_hash(*items), 16)


def split_trials(
    trial_environment: np.ndarray,
    fractions: tuple[float, ...],
    seed: int,
) -> list[np.ndarray]:
    """Environment-stratified trial folds with a reproducible seed.

    ``cross_validate_trials`` splits each environment's trials independently at the normalized
    cumulative fractions, so every fold holds the requested share of *each* environment rather than
    of the session as a whole. It draws from the global numpy random state and takes no seed
    argument, so the state is set and restored here.

    Parameters
    ----------
    trial_environment : np.ndarray
        Environment index per trial, ``(trials,)``.
    fractions : tuple of float
        Relative fold sizes; normalized internally.
    seed : int
        Seed for the global numpy random state during the split.

    Returns
    -------
    list of np.ndarray
        Sorted trial indices per fold, one array per entry in ``fractions``.
    """
    state = np.random.get_state()
    np.random.seed(seed % (2**32))
    try:
        folds = cross_validate_trials(np.asarray(trial_environment), list(fractions))
    finally:
        np.random.set_state(state)
    return [np.sort(np.asarray(fold, dtype=int)) for fold in folds]


@dataclass(frozen=True)
class _GainData:
    """Everything one environment contributes to the regression."""

    gain: np.ndarray  # (neurons, trials)
    roi_indices: np.ndarray  # session ROI index per gain row
    reliability: np.ndarray  # (valid rois,) over the prediction trials
    fraction_active: np.ndarray  # (valid rois,) over the prediction trials
    source_rows: np.ndarray
    target_rows: np.ndarray
    columns: dict[str, np.ndarray]  # split name -> gain column indices
    n_neurons_env: int
    n_neurons_kept: int


@dataclass(frozen=True)
class _RRRFit:
    """A fitted reduced-rank regression, the hyperparameters that won, and its validation curve."""

    model: ReducedRankRegression
    alpha: float
    rank: int
    rank_gss: int
    max_rank: int
    ranks: list[int]
    val_mse: list[float]
    val_r2: list[float]


def _parsimonious_rank(ranks: list[int], scores: list[float]) -> int:
    """Smallest rank whose score is within ``RANK_SCORE_TOLERANCE`` of the best score.

    Parameters
    ----------
    ranks : list of int
        Ranks that were scored.
    scores : list of float
        Validation MSE aligned with ``ranks``; non-finite entries are ignored.

    Returns
    -------
    int
        The most parsimonious rank that is statistically indistinguishable from the best one.
    """
    finite = [(rank, score) for rank, score in zip(ranks, scores) if np.isfinite(score)]
    if not finite:
        raise ValueError("No rank produced a finite validation score")
    best = min(score for _, score in finite)
    return min(rank for rank, score in finite if score <= best * (1 + RANK_SCORE_TOLERANCE))


def optimize_rrr(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    alpha_bounds: tuple[float, float],
    max_iterations: int,
) -> _RRRFit:
    """Optimize ridge alpha then rank by golden-section search on validation MSE.

    Mirrors ``ReducedRankRegressionModel._optimize_golden`` with three differences.

    Alpha is swept at the achievable full rank rather than a hardcoded rank of 200, which would
    exceed ``max_rank`` for the small trial counts here. The rank sweep refits nothing: alpha enters
    the ridge solve at fit time, but rank is applied at prediction time, so one fit serves every
    rank. And the reported rank is refined against the exhaustive validation rank curve.

    That last step matters because rank is the scientific quantity, not just a nuisance
    hyperparameter. Golden-section search samples roughly a dozen ranks, so on a curve that
    improves and then goes flat it reports whichever plateau point it happened to land on rather
    than where the plateau begins. ``score_curve`` scores every rank in a single accumulating pass
    for about the cost of one prediction, so the elbow can simply be read off exactly. Both answers
    are returned: ``rank`` from the curve and ``rank_gss`` from the search alone.

    Parameters
    ----------
    x_train, y_train : torch.Tensor
        Training source and target gain, ``(trials, neurons)``.
    x_val, y_val : torch.Tensor
        Validation source and target gain, ``(trials, neurons)``.
    alpha_bounds : tuple of float
        Lower and upper bound of the log-space alpha search.
    max_iterations : int
        Iteration cap for each golden-section search.

    Returns
    -------
    _RRRFit
        The model fit at the winning alpha, the winning rank, and the validation rank curve.
    """
    # The intercept column counts as a feature, so it raises the achievable rank by one.
    max_rank = int(min(x_train.shape[1] + 1, y_train.shape[1], x_train.shape[0]))

    def _validation_mse(model: ReducedRankRegression, rank: int) -> float:
        score = mse(model.predict(x_val, rank=rank, nonnegative=True), y_val, reduce="mean", dim=None)
        return float("inf") if not np.isfinite(score) else float(score)

    def evaluate_alpha(alpha: float) -> float:
        model = ReducedRankRegression(alpha=float(alpha), fit_intercept=True).fit(x_train, y_train)
        return _validation_mse(model, max_rank)

    best_alpha, _, _ = golden_section_search(
        func=evaluate_alpha,
        a=alpha_bounds[0],
        b=alpha_bounds[1],
        tolerance_param=1e-2,
        tolerance_score=1e-3,
        max_iterations=max_iterations,
        minimize=True,
        logspace=True,
    )

    model = ReducedRankRegression(alpha=float(best_alpha), fit_intercept=True).fit(x_train, y_train)

    searched: list[tuple[int, float]] = []
    if max_rank > 1:

        def evaluate_rank(rank: float) -> float:
            rank = int(np.clip(int(rank), 1, max_rank))
            score = _validation_mse(model, rank)
            searched.append((rank, score))
            return score

        golden_section_search(
            func=evaluate_rank,
            a=1.0,
            b=float(max_rank),
            tolerance_param=1.0,  # one rank unit
            tolerance_score=1e-3,
            max_iterations=max_iterations,
            minimize=True,
            logspace=False,
        )
    else:
        searched.append((1, _validation_mse(model, 1)))

    ranks, scores = model.score_curve(x_val, y_val, ranks=list(range(1, max_rank + 1)), nonnegative=True, dim=None, verbose=False)
    searched_ranks = [rank for rank, _ in searched]
    searched_scores = [score for _, score in searched]
    return _RRRFit(
        model=model,
        alpha=float(best_alpha),
        rank=_parsimonious_rank(ranks, scores["mse"]),
        rank_gss=_parsimonious_rank(searched_ranks, searched_scores),
        max_rank=max_rank,
        ranks=ranks,
        val_mse=[float(value) for value in scores["mse"]],
        val_r2=[float(value) for value in scores["r2"]],
    )


def score_rrr(model: ReducedRankRegression, rank: int, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Pooled MSE and R^2 of a rank-truncated prediction."""
    prediction = model.predict(x, rank=rank, nonnegative=True)
    return float(mse(prediction, y, reduce="mean", dim=None)), float(measure_r2(prediction, y, reduce="mean", dim=None))


def split_tensors(
    gain: np.ndarray,
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    columns: dict[str, np.ndarray],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Slice a gain matrix into per-split ``(trials, neurons)`` source/target tensor pairs."""
    return {
        name: (
            torch.tensor(gain[source_rows][:, cols].T, dtype=torch.float64),
            torch.tensor(gain[target_rows][:, cols].T, dtype=torch.float64),
        )
        for name, cols in columns.items()
    }


def roll_gain_rows(gain: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Circularly shift each neuron's trial series by its own random offset.

    Every neuron keeps its exact gain distribution and its own trial-to-trial autocorrelation; only
    the alignment *between* neurons is destroyed. Offsets are drawn from ``1 .. trials - 1`` so no
    neuron keeps its original alignment.

    Parameters
    ----------
    gain : np.ndarray
        Gain matrix, ``(neurons, trials)``.
    rng : np.random.Generator
        Source of the per-neuron offsets.

    Returns
    -------
    np.ndarray
        The rolled gain matrix, same shape.
    """
    n_neurons, n_trials = gain.shape
    if n_trials < 2:
        return gain.copy()
    offsets = rng.integers(1, n_trials, size=n_neurons)
    return np.take_along_axis(gain, (np.arange(n_trials)[None, :] - offsets[:, None]) % n_trials, axis=1)


def _roll_average(values: list[np.ndarray]) -> np.ndarray:
    """Mean over rolls, ignoring non-finite entries and returning NaN where none are finite."""
    stacked = np.asarray(values, dtype=float)
    stacked = np.where(np.isfinite(stacked), stacked, np.nan)
    counts = np.sum(np.isfinite(stacked), axis=0)
    return np.divide(
        np.nansum(stacked, axis=0),
        counts,
        out=np.full(np.shape(counts), np.nan, dtype=float),
        where=counts > 0,
    )


@dataclass(frozen=True)
class _NullFit:
    """Roll-averaged scores and rank curves for the trial-permuted null."""

    rank: float
    mse_val: float
    r2_val: float
    mse_test: float
    r2_test: float
    mse_val_curve: np.ndarray
    r2_val_curve: np.ndarray
    mse_test_curve: np.ndarray
    r2_test_curve: np.ndarray


def evaluate_null_rolls(
    gain: np.ndarray,
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    columns: dict[str, np.ndarray],
    *,
    alpha: float,
    rank: int,
    ranks: list[int],
    n_rolls: int,
    rng: np.random.Generator,
) -> _NullFit:
    """Refit and rescore the regression on trial-permuted gain, averaged over rolls.

    Each roll shifts every neuron's trial series independently (:func:`roll_gain_rows`) on the
    *full* environment gain matrix, and only then applies the same train/validation/test column
    split as the real fit. Rolling before splitting keeps the null pipeline identical to the real
    one: the null model sees the same number of trials, the same neurons, and the same split
    boundaries, differing only in that no trial structure is shared across neurons.

    Hyperparameters are not re-optimized. ``alpha`` and ``rank`` come from the real fit, so the
    comparison is "the same model, on data with the shared structure removed" rather than "the best
    model that can be fit to noise". The full rank curves are recorded anyway, since they cost one
    accumulating pass, and give a floor to read the real rank curve against at every rank.

    Parameters
    ----------
    gain : np.ndarray
        Gain matrix for one environment, ``(neurons, trials)``.
    source_rows, target_rows : np.ndarray
        Row indices of the two neuron halves.
    columns : dict
        Split name -> gain column indices.
    alpha : float
        Ridge alpha from the real fit.
    rank : int
        Winning rank from the real fit; where the scalar null scores are read.
    ranks : list of int
        Ranks spanned by the real score curves; the null curves use the same grid.
    n_rolls : int
        Number of independent rolls to average. Zero returns all-NaN.
    rng : np.random.Generator
        Source of the roll offsets.

    Returns
    -------
    _NullFit
        Roll-averaged validation and test scores and rank curves.
    """
    empty = np.full(len(ranks), np.nan)
    if n_rolls <= 0:
        return _NullFit(np.nan, np.nan, np.nan, np.nan, np.nan, empty, empty.copy(), empty.copy(), empty.copy())

    points: dict[str, list[float]] = {f"{metric}_{split}": [] for metric in ("mse", "r2") for split in ("val", "test")}
    curves: dict[str, list[np.ndarray]] = {key: [] for key in points}
    for _ in range(n_rolls):
        tensors = split_tensors(roll_gain_rows(gain, rng), source_rows, target_rows, columns)
        model = ReducedRankRegression(alpha=float(alpha), fit_intercept=True).fit(*tensors["train"])
        for split in ("val", "test"):
            mse_point, r2_point = score_rrr(model, rank, *tensors[split])
            points[f"mse_{split}"].append(mse_point)
            points[f"r2_{split}"].append(r2_point)
            _, scores = model.score_curve(*tensors[split], ranks=ranks, nonnegative=True, dim=None, verbose=False)
            curves[f"mse_{split}"].append(np.asarray(scores["mse"], dtype=float))
            curves[f"r2_{split}"].append(np.asarray(scores["r2"], dtype=float))

    averaged_points = {key: float(_roll_average(values)) for key, values in points.items()}
    averaged_curves = {key: _roll_average(values) for key, values in curves.items()}
    mse_val_curve = averaged_curves["mse_val"]
    null_rank = float(_parsimonious_rank(ranks, list(mse_val_curve))) if np.any(np.isfinite(mse_val_curve)) else np.nan
    return _NullFit(
        rank=null_rank,
        mse_val=averaged_points["mse_val"],
        r2_val=averaged_points["r2_val"],
        mse_test=averaged_points["mse_test"],
        r2_test=averaged_points["r2_test"],
        mse_val_curve=mse_val_curve,
        r2_val_curve=averaged_curves["r2_val"],
        mse_test_curve=averaged_curves["mse_test"],
        r2_test_curve=averaged_curves["r2_test"],
    )


def _stack_curves(curves: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """NaN-pad per-slot rank curves onto a shared ``(MAX_ENV_SLOTS, R)`` grid.

    Parameters
    ----------
    curves : dict
        Slot index -> curve name -> values. Slots with no fit are absent.

    Returns
    -------
    dict[str, np.ndarray]
        One ``(MAX_ENV_SLOTS, R)`` array per entry of ``CURVE_KEYS``, where ``R`` is the longest
        curve present (one column of NaN when no slot produced a fit).
    """
    width = max((values["rank_values"].size for values in curves.values()), default=1)
    stacked = {key: np.full((MAX_ENV_SLOTS, width), np.nan) for key in CURVE_KEYS}
    for slot, values in curves.items():
        for key in CURVE_KEYS:
            stacked[key][slot, : values[key].size] = values[key]
    return stacked


@dataclass(frozen=True)
class GainRegressionConfig(AnalysisConfigBase):
    """Reduced-rank regression from source-neuron to target-neuron place-field gain.

    One model is fit per environment. Within an environment the neurons that survive the
    reliability and fraction-active screen are split 50/50 into source and target halves, the full
    trials are split into train/validation/test, and a reduced-rank regression maps source gain to
    target gain. Ridge alpha and rank are optimized on validation, and performance is reported on
    test. Results are indexed by the mouse's environment experience-order slot.

    Gain is not centered and predictions are rectified (``nonnegative=True``) exactly as
    ``ReducedRankRegressionModel`` does; the intercept carries the per-neuron mean gain.
    ``gain_transform`` selects whether the regressed quantity is gain itself or its square root.

    Every fit is paired with a trial-permuted null. Once alpha and rank are chosen on the real data,
    the environment's gain matrix is rolled ``n_null_rolls`` times -- each neuron's trial series
    shifted circularly by its own random offset -- and the same fit-and-score pipeline is rerun on
    each roll at those fixed hyperparameters. The roll-averaged scores are stored beside the real
    ones under the ``*_null`` keys.

    Sessions with too few trials in an environment leave that slot NaN rather than failing. Every
    trial and neuron count is stored so those sessions can be filtered post hoc.

    Parameters
    ----------
    spks_type : SpksTypes
        Spike type used for the place fields.
    num_bins : int
        Number of spatial bins; sets ``dist_step`` as ``env_length / num_bins``.
    speed_threshold : float
        Minimum running speed for a sample to enter a place field.
    smooth_width : float or None
        Gaussian smoothing width for the place fields, or None for no smoothing.
    full_trial_flexibility : float or None
        Distance from each end of the track exempt from the full-traversal requirement.
    reliability_threshold : float
        Minimum leave-one-out reliability for a neuron to enter the regression.
    fraction_active_threshold : float
        Minimum participation-ratio fraction-active for a neuron to enter the regression.
    trial_fractions : tuple of float
        Relative train/validation/test trial fractions, stratified within each environment.
    placefield_split : str
        ``"train"`` builds the trial-averaged place field, its Gaussian fit, and the neuron screen
        from training trials only, then measures every trial's gain against that fit. ``"all"`` uses
        every trial, reproducing ``PlacefieldGainViewer``. ``"train"`` is the default because an
        all-trial denominator lets a held-out trial contribute to its own gain, and injects the same
        all-trial mean into every neuron -- a shared component that inflates cross-neuron
        predictability.
    gain_transform : str
        ``"raw"`` regresses gain as measured. ``"sqrt"`` takes its square root first, which pulls in
        the heavy upper tail without displacing the many zero-activity trials the way a log would.
        See :func:`apply_gain_transform`.
    split_seed : int
        Base seed for the trial and source/target splits; mixed with the session UID so different
        sessions get different splits reproducibly.
    n_null_rolls : int
        Number of trial-permuted null repetitions scored per environment, or 0 to skip the null and
        leave every ``*_null`` key NaN. See :func:`evaluate_null_rolls`.

    Result keys
    -----------
    ``S`` is ``MAX_ENV_SLOTS``, ``N`` the number of ROIs with usable activity, ``R`` the largest
    achievable rank across slots. Every ``*_null`` entry is the roll-average of the same quantity
    measured on trial-permuted gain at the real fit's hyperparameters.

    =======================================  ==========  ============================================
    key                                      shape       description
    =======================================  ==========  ============================================
    env_slot_ids                             (S,)        environment index occupying each slot
    rrr_alpha                                (S,)        winning ridge alpha
    rrr_rank                                 (S,)        winning rank, off the exact validation curve
    rrr_rank_gss                             (S,)        winning rank from the golden-section search
    rrr_rank_null                            (S,)        rank the null validation curve would choose
    max_rank                                 (S,)        largest achievable rank
    mse_val, r2_val                          (S,)        validation score at the winning rank
    mse_test, r2_test                        (S,)        test score at the winning rank
    mse_val_null, r2_val_null                (S,)        null validation score at the winning rank
    mse_test_null, r2_test_null              (S,)        null test score at the winning rank
    rank_values                              (S, R)      ranks spanned by the score curves
    mse_val_curve, r2_val_curve              (S, R)      validation score at every rank
    mse_test_curve, r2_test_curve            (S, R)      test score at every rank
    mse_val_curve_null, r2_val_curve_null    (S, R)      null validation score at every rank
    mse_test_curve_null, r2_test_curve_null  (S, R)      null test score at every rank
    n_trials_env                             (S,)        full trials in the environment
    n_trials_train/_val/_test                (S,)        trials per split
    n_neurons_env                            (S,)        ROIs before the reliability screen
    n_neurons_kept                           (S,)        ROIs passing the screen
    n_neurons_finite                         (S,)        ROIs with a finite gain on every trial
    n_neurons_source/_target                 (S,)        ROIs in each half of the regression
    reliability, fraction_active             (S, N)      screening metrics over prediction trials
    gain_matrices                            per slot    (neurons, trials) gain, object list
    gain_roi_indices                         per slot    session ROI index per gain row, object list
    source_rows, target_rows                 per slot    gain row indices per half, object list
    trial_columns                            per slot    {split: gain column indices}, object list
    roi_lookup                               (N,)        session ROI index per metric-array column
    =======================================  ==========  ============================================

    Notes
    -----
    The ``*_null`` keys are the reference the real numbers should be read against. A rolled matrix
    keeps every neuron's own gain distribution and autocorrelation and destroys only the alignment
    between neurons, so the null carries everything the regression can exploit *except* shared trial
    structure. It is not zero: the intercept still predicts each target neuron's mean gain, and with
    tens of training trials a high-rank fit can chase noise. ``r2_test_null`` is therefore the floor
    against which ``r2_test`` counts, and ``mse_test_curve_null`` is the floor for the whole rank
    curve. Where the two curves separate is the range of ranks that carry real shared structure.

    Two properties of the measured data shape how these results should be read.

    Single-trial gain from deconvolved spikes is sparse and heavy tailed. Across sessions the
    median gain is 0 -- a screened neuron is still silent in its place field on most individual
    trials -- while the upper tail runs past 20 and the maximum past 50. Regressing raw gain
    therefore puts most of the squared error on a small number of large-gain entries, and its
    pooled R^2 (0.02 to 0.17 on held out trials) is a conservative summary of a real but modest
    effect. ``gain_transform="sqrt"`` is the counterweight; both variants run in the grid, and
    ``gain_matrices`` stores whichever quantity was regressed. ``PlacefieldGainViewer`` shows the
    same measurement clipped at a ``gain_vmax`` of 2, which hides the tail entirely.

    ``max_rank`` is almost always the training trial count, since the neuron counts are in the
    hundreds and the trials in the tens. When ``rrr_rank`` lands close to ``max_rank`` the
    validation curve was still improving where it ran out of room, so the rank is reporting the
    trial budget rather than the dimensionality of shared gain. Compare ``rrr_rank`` against
    ``max_rank`` and ``n_trials_train`` before reading it as a dimensionality estimate.
    """

    schema_version: str = "v2"
    data_config_name: str = "default"

    spks_type: SpksTypes = "sigrebase"
    num_bins: int = 100
    speed_threshold: float = 1.0
    smooth_width: Optional[float] = 1.0
    full_trial_flexibility: Optional[float] = 3.0
    reliability_threshold: float = 0.3
    fraction_active_threshold: float = 0.1
    trial_fractions: tuple[float, float, float] = (1.0, 0.25, 0.25)
    placefield_split: str = "train"
    gain_transform: str = "raw"
    split_seed: int = 0
    n_null_rolls: int = 10

    display_name: ClassVar[str] = "gain_regression"

    #: A split with fewer trials than this cannot support a fit or an honest score.
    MIN_TRIALS_PER_SPLIT: ClassVar[int] = 3
    #: Source and target halves each need at least this many neurons.
    MIN_NEURONS_PER_GROUP: ClassVar[int] = 2
    #: Log-space bounds of the ridge alpha search.
    ALPHA_BOUNDS: ClassVar[tuple[float, float]] = (1e-2, 1e6)
    #: Iteration cap for each golden-section search.
    GSS_MAX_ITERATIONS: ClassVar[int] = 25

    _result_handling: ClassVar[dict[str, str]] = {
        "gain_matrices": "skip",
        "gain_roi_indices": "skip",
        "source_rows": "skip",
        "target_rows": "skip",
        "trial_columns": "skip",
        "roi_lookup": "skip",
    }

    @staticmethod
    def _param_grid() -> dict:
        # Both denominator conventions run so the leakage-free default can be compared against the
        # viewer's all-trial place field, and both transforms run so the skew correction can be
        # judged against untransformed gain. Thresholds are deliberately not swept -- they change
        # which neurons exist, and the counts stored here already support post-hoc filtering.
        return {"placefield_split": list(PLACEFIELD_SPLITS), "gain_transform": list(GAIN_TRANSFORMS)}

    def validate(self) -> None:
        if self.spks_type not in VALID_SPKS_TYPES:
            raise ValueError(f"Unknown spks_type {self.spks_type!r}. Available: {VALID_SPKS_TYPES}")
        if self.placefield_split not in PLACEFIELD_SPLITS:
            raise ValueError(f"Unknown placefield_split {self.placefield_split!r}. Available: {PLACEFIELD_SPLITS}")
        if self.gain_transform not in GAIN_TRANSFORMS:
            raise ValueError(f"Unknown gain_transform {self.gain_transform!r}. Available: {GAIN_TRANSFORMS}")
        if len(self.trial_fractions) != len(SPLIT_NAMES):
            raise ValueError(f"trial_fractions must have {len(SPLIT_NAMES)} entries for {SPLIT_NAMES}")
        if any(fraction <= 0 for fraction in self.trial_fractions):
            raise ValueError("every entry of trial_fractions must be positive")
        if self.num_bins < 4:
            raise ValueError("num_bins must be at least 4")
        if self.n_null_rolls < 0:
            raise ValueError("n_null_rolls must be non-negative")

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"bins={self.num_bins}",
            f"smooth={self.smooth_width}",
            f"rel={self.reliability_threshold}",
            f"frac={self.fraction_active_threshold}",
            f"pf={self.placefield_split}",
            f"gain={self.gain_transform}",
            f"seed={self.split_seed}",
            f"null={self.n_null_rolls}",
            self.schema_version,
        ]
        return "_".join(parts)

    def _load_session(self, session: B2Session) -> dict:
        """Load per-environment place-field maps and the full-trial bookkeeping.

        Reproduces ``PlacefieldGainViewer._load`` minus the display-only pieces: each ROI's activity
        is divided by its full-session standard deviation, and only trials that traverse every
        required position bin are kept.

        The standard-deviation division does not affect any result. Gain is a ratio between a
        neuron's own trial activity and its own fitted place field, so any per-neuron scalar
        cancels; reliability and fraction-active are likewise per-neuron scale invariant. It is kept
        because the same computation supplies ``valid_rois`` -- ROIs with a zero or non-finite
        standard deviation carry no signal and must be dropped -- and because it keeps this loader
        identical to the viewer's. This is also why the config exposes no activity-scaling
        parameter: ``std`` and ``max`` scaling would be indistinguishable here.

        Parameters
        ----------
        session : B2Session
            Session to load.

        Returns
        -------
        dict
            ``env_maps``, ``positions``, ``full_trials``, ``trials_by_environment``, ``roi_lookup``.
        """
        previous_spks_type = session.params.spks_type
        session.params.spks_type = self.spks_type
        try:
            idx_rois = np.asarray(session.idx_rois, dtype=bool)
            raw = np.asarray(session.spks)[:, idx_rois]
            scale = np.nanstd(raw, axis=0)
            valid_rois = np.isfinite(scale) & (scale > 0)
            env_length = float(np.asarray(session.env_length).flat[0])
            smp = SpkmapProcessor(
                session,
                params=SpkmapParams(
                    dist_step=env_length / self.num_bins,
                    speed_threshold=self.speed_threshold,
                    standardize_spks=False,
                    smooth_width=self.smooth_width,
                    full_trial_flexibility=self.full_trial_flexibility,
                    autosave=False,
                ),
            )
            # Reproduce get_env_maps' trial bookkeeping before its processing mutates maps: a trial
            # is full when every bin outside the allowed edge flexibility lies inside that trial's
            # observed positional range. With autosave=False this recomputes the raw maps that
            # get_env_maps will build again, which is the price of getting the trial indices out.
            raw_maps = smp.get_raw_maps()
            required_bins = smp._idx_required_position_bins()
            full_trials = _full_trial_indices(raw_maps.occmap, required_bins)
            trial_environment = np.asarray(session.trial_environment)
            trials_by_environment = {int(env): full_trials[trial_environment[full_trials] == env] for env in session.environments}
            env_maps = smp.get_env_maps()
            env_maps.distcenters = smp.dist_centers
            # Remove only bins that remain NaN in at least one retained traversal; reliability_loo
            # is numba-compiled and NaN-intolerant, so this has to happen before any metric.
            env_maps.pop_nan_positions()
            env_maps.spkmap = [maps[valid_rois] / scale[valid_rois, None, None] for maps in env_maps.spkmap]
        finally:
            session.params.spks_type = previous_spks_type

        return {
            "env_maps": env_maps,
            "positions": np.asarray(env_maps.distcenters, dtype=float),
            "full_trials": full_trials,
            "trials_by_environment": trials_by_environment,
            "roi_lookup": np.flatnonzero(idx_rois)[valid_rois],
        }

    def _environment_gain(
        self,
        session: B2Session,
        data: dict,
        env: int,
        folds: list[np.ndarray],
    ) -> Optional[_GainData]:
        """Build one environment's gain matrix and its neuron and trial partitions.

        Returns None when the environment cannot support a regression: too few trials in a split,
        or too few neurons in either half after screening.
        """
        env_maps = data["env_maps"]
        env_index = list(env_maps.environments).index(env)
        trial_maps = np.asarray(env_maps.spkmap[env_index], dtype=float)
        trials_env = data["trials_by_environment"][env]
        if trial_maps.shape[1] < 2 or trial_maps.shape[2] < 4:
            return None
        # get_env_maps groups the same full trials by environment in ascending order, so the trial
        # axis of the map and trials_env line up column for column. Every index mapping relies on it.
        if trial_maps.shape[1] != trials_env.size:
            raise ValueError(f"Environment {env} trial bookkeeping is inconsistent with its place-field maps")

        full_trials = data["full_trials"]
        columns = {}
        for name, fold in zip(SPLIT_NAMES, folds):
            columns[name] = np.flatnonzero(np.isin(trials_env, full_trials[fold]))
        if any(columns[name].size < self.MIN_TRIALS_PER_SPLIT for name in SPLIT_NAMES):
            return None

        prediction_trials = columns["train"] if self.placefield_split == "train" else None
        screen_maps = trial_maps if prediction_trials is None else trial_maps[:, prediction_trials]
        reliability = reliability_loo(screen_maps)
        fraction_active = FractionActive.compute(
            screen_maps,
            activity_axis=2,
            fraction_axis=1,
            activity_method="rms",
            fraction_method="participation",
        )
        idx_keep = np.isfinite(reliability) & np.isfinite(fraction_active)
        idx_keep &= reliability >= self.reliability_threshold
        idx_keep &= fraction_active >= self.fraction_active_threshold
        if np.sum(idx_keep) < 2 * self.MIN_NEURONS_PER_GROUP:
            return None

        gain, _, _ = gaussian_gain_matrix(trial_maps[idx_keep], data["positions"], prediction_trials)
        gain = apply_gain_transform(gain, self.gain_transform)
        # The regression needs complete matrices, so a neuron with any degenerate trial is dropped.
        # Screening after the transform also catches anything the transform sent to NaN.
        idx_finite = np.all(np.isfinite(gain), axis=1)
        gain = gain[idx_finite]
        if gain.shape[0] < 2 * self.MIN_NEURONS_PER_GROUP:
            return None

        rng = np.random.default_rng(_seed(session.session_uid, env, self.split_seed, "neurons"))
        order = rng.permutation(gain.shape[0])
        midpoint = gain.shape[0] // 2
        source_rows, target_rows = np.sort(order[:midpoint]), np.sort(order[midpoint:])
        if min(source_rows.size, target_rows.size) < self.MIN_NEURONS_PER_GROUP:
            return None

        return _GainData(
            gain=gain,
            roi_indices=data["roi_lookup"][np.flatnonzero(idx_keep)[idx_finite]],
            reliability=reliability,
            fraction_active=fraction_active,
            source_rows=source_rows,
            target_rows=target_rows,
            columns=columns,
            n_neurons_env=int(trial_maps.shape[0]),
            n_neurons_kept=int(np.sum(idx_keep)),
        )

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Fit and score one reduced-rank gain regression per environment experience slot.

        ``registry`` is unused: this config reads ``session.spks`` directly under the
        ``session.params.spks_type`` save/restore idiom, like ``CrossValidatedPlacefieldsConfig``.
        """
        data = self._load_session(session)
        n_rois = int(data["roi_lookup"].size)

        # One stratified split for the whole session; each environment then takes its own share of
        # each fold. cross_validate_trials balances the folds within every environment, so this is
        # equivalent to splitting per environment while keeping the folds mutually consistent.
        folds = split_trials(
            np.asarray(session.trial_environment)[data["full_trials"]],
            self.trial_fractions,
            _seed(session.session_uid, self.split_seed, "trials"),
        )

        mouse_order = load_env_order().get(session.mouse_name)
        env_slot_ids = np.full(MAX_ENV_SLOTS, np.nan)
        if mouse_order is not None:
            env_slot_ids[: len(mouse_order)] = mouse_order

        scalar_keys = (
            "rrr_alpha",
            "rrr_rank",
            "rrr_rank_gss",
            "rrr_rank_null",
            "max_rank",
            "mse_val",
            "r2_val",
            "mse_test",
            "r2_test",
            "mse_val_null",
            "r2_val_null",
            "mse_test_null",
            "r2_test_null",
            "n_trials_env",
            "n_trials_train",
            "n_trials_val",
            "n_trials_test",
            "n_neurons_env",
            "n_neurons_kept",
            "n_neurons_finite",
            "n_neurons_source",
            "n_neurons_target",
        )
        scalars = {key: np.full(MAX_ENV_SLOTS, np.nan) for key in scalar_keys}
        reliability = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
        fraction_active = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
        curves: dict[int, dict[str, np.ndarray]] = {}
        gain_matrices: list[Optional[np.ndarray]] = [None] * MAX_ENV_SLOTS
        gain_roi_indices: list[Optional[np.ndarray]] = [None] * MAX_ENV_SLOTS
        source_rows: list[Optional[np.ndarray]] = [None] * MAX_ENV_SLOTS
        target_rows: list[Optional[np.ndarray]] = [None] * MAX_ENV_SLOTS
        trial_columns: list[Optional[dict]] = [None] * MAX_ENV_SLOTS

        for env in sorted(int(e) for e in np.asarray(data["env_maps"].environments) if e >= 0):
            if mouse_order is None or env not in mouse_order:
                continue
            slot = mouse_order.index(env)
            if slot >= MAX_ENV_SLOTS:
                continue

            scalars["n_trials_env"][slot] = data["trials_by_environment"][env].size
            gain_data = self._environment_gain(session, data, env, folds)
            if gain_data is None:
                continue

            reliability[slot] = gain_data.reliability
            fraction_active[slot] = gain_data.fraction_active
            for name in SPLIT_NAMES:
                scalars[f"n_trials_{name}"][slot] = gain_data.columns[name].size
            scalars["n_neurons_env"][slot] = gain_data.n_neurons_env
            scalars["n_neurons_kept"][slot] = gain_data.n_neurons_kept
            scalars["n_neurons_finite"][slot] = gain_data.gain.shape[0]
            scalars["n_neurons_source"][slot] = gain_data.source_rows.size
            scalars["n_neurons_target"][slot] = gain_data.target_rows.size
            gain_matrices[slot] = gain_data.gain
            gain_roi_indices[slot] = gain_data.roi_indices
            source_rows[slot] = gain_data.source_rows
            target_rows[slot] = gain_data.target_rows
            trial_columns[slot] = dict(gain_data.columns)

            tensors = split_tensors(gain_data.gain, gain_data.source_rows, gain_data.target_rows, gain_data.columns)
            fit = optimize_rrr(
                *tensors["train"],
                *tensors["val"],
                alpha_bounds=self.ALPHA_BOUNDS,
                max_iterations=self.GSS_MAX_ITERATIONS,
            )
            scalars["rrr_alpha"][slot] = fit.alpha
            scalars["rrr_rank"][slot] = fit.rank
            scalars["rrr_rank_gss"][slot] = fit.rank_gss
            scalars["max_rank"][slot] = fit.max_rank
            scalars["mse_val"][slot], scalars["r2_val"][slot] = score_rrr(fit.model, fit.rank, *tensors["val"])
            scalars["mse_test"][slot], scalars["r2_test"][slot] = score_rrr(fit.model, fit.rank, *tensors["test"])

            # The validation curve came back from the optimizer; the test curve is one more pass,
            # cheap because the nested low-rank structure accumulates across ranks.
            _, test_scores = fit.model.score_curve(*tensors["test"], ranks=fit.ranks, nonnegative=True, dim=None, verbose=False)

            # The null reuses this fit's alpha and rank, so it has to run after the optimizer.
            null = evaluate_null_rolls(
                gain_data.gain,
                gain_data.source_rows,
                gain_data.target_rows,
                gain_data.columns,
                alpha=fit.alpha,
                rank=fit.rank,
                ranks=fit.ranks,
                n_rolls=self.n_null_rolls,
                rng=np.random.default_rng(_seed(session.session_uid, env, self.split_seed, "null")),
            )
            scalars["rrr_rank_null"][slot] = null.rank
            scalars["mse_val_null"][slot], scalars["r2_val_null"][slot] = null.mse_val, null.r2_val
            scalars["mse_test_null"][slot], scalars["r2_test_null"][slot] = null.mse_test, null.r2_test

            curves[slot] = {
                "rank_values": np.asarray(fit.ranks, dtype=float),
                "mse_val_curve": np.asarray(fit.val_mse, dtype=float),
                "r2_val_curve": np.asarray(fit.val_r2, dtype=float),
                "mse_test_curve": np.asarray(test_scores["mse"], dtype=float),
                "r2_test_curve": np.asarray(test_scores["r2"], dtype=float),
                "mse_val_curve_null": null.mse_val_curve,
                "r2_val_curve_null": null.r2_val_curve,
                "mse_test_curve_null": null.mse_test_curve,
                "r2_test_curve_null": null.r2_test_curve,
            }

        return {
            **scalars,
            **_stack_curves(curves),
            "env_slot_ids": env_slot_ids,
            "reliability": reliability,
            "fraction_active": fraction_active,
            "gain_matrices": gain_matrices,
            "gain_roi_indices": gain_roi_indices,
            "source_rows": source_rows,
            "target_rows": target_rows,
            "trial_columns": trial_columns,
            "roi_lookup": data["roi_lookup"],
        }
