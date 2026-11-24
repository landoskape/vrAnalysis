"""PFPredQualityConfig — R² between neural activity and place-field prediction.

Measures how well the place-field prediction model captures neural activity,
and how that quality relates to spatial reliability. Stores per-ROI stats
and session-level summaries for mouse-average reliability vs R² curves.

Also measures the *peak amplitude* of every ROI's place field — the max of its
trial-averaged map over position, in units of that ROI's standard deviation in
time — for every environment the session ran. Those keys are emitted on the
per-mouse experience-order slot axis (``MAX_ENV_SLOTS``, see ``..env_order``) so
column j means "the j-th environment this mouse ever saw" in every session of
every mouse, and the aggregator can stack them across mice. A histogram over
``peak_bin_edges`` is precomputed per slot so the distribution can be plotted
straight from disk without re-measuring any session.

The prediction-quality keys come in two flavours: the top-level ones (``r2``,
``rms``, ``norm_rms``, ``var_total``, ``var_pred``, ``var_residual``,
``frac_var_pred``, ``reliability``, ``r2_kde_mean``, ``rms_kde_mean``, ...) describe the
session's *best* environment, while the ``*_slot`` keys repeat the measurement
for every environment on that same experience-order slot axis, so prediction
quality can be followed per environment as a mouse gets more familiar with it.

Trial-wise activity amplitude is measured separately within every environment and then reduced
across trials for each cell. The result keys are the Cartesian product
``{trial_rms, trial_mean, trial_pf_dot, trial_gain, trial_gain_least_squares}_{mean, variance, cv}``, returned literally by
:func:`trial_activity_summary_keys`; every array has shape ``(MAX_ENV_SLOTS, rois)``. Finite-only
across-ROI mean and median summaries are also stored for all cells, quality-filtered cells, and
their complement using the established ``{mean|median}_{quality_filtered_|notquality_filtered_}``
naming convention; :func:`trial_activity_population_summary_keys` enumerates them.

Spatial prediction variability is saved as ``pf_{all,f1,f2}`` and
``rms_{all_from_all,f1_from_f1,f2_from_f1,f1_from_f2,f2_from_f2}``, each shaped
``(rois, position_bins)`` for the best environment. Their ``_slot`` counterparts are
``(MAX_ENV_SLOTS, rois, position_bins)``; positions are in ``pf_rms_distcenters``.
F1/f2 alternate chronological trials within each environment. ``from`` identifies
the PF template split, while the preceding split supplies the residual trials.
``pf_rms_r_<comparison>`` stores per-ROI Pearson r across pairwise-finite positions,
with ``_slot`` counterparts for every environment. ``pf_rms_r_cv`` averages the
two held-out directions; ``pf_rms_r_independent_cv`` averages PF f1 versus RMS f2
from f2 and PF f2 versus RMS f1 from f1, keeping all inputs on independent splits.
Both directional r values must be finite for their average to be finite. Constant
maps and fewer than two paired positions give NaN. Existing ``cc`` remains the
temporal activity/prediction correlation.

Schema v13 adds ``trial_residual_rms_{mean,variance,cv}_<fold>`` arrays shaped
``(MAX_ENV_SLOTS, rois)`` and their quality-subset population summaries. Each trial's
RMS is computed across finite position bins after subtracting its least-squares
gain times the selected mean PF template. These are the trial-position measurements
shown by PlacefieldBreakdownFocus, distinct from the temporal variances below.
Folds match PF_VARIANCE_COMPARISONS; cv/within_cv average directional statistics.

Temporal variance decomposition uses the same average PF templates, evaluated on actual
moving imaging frames, not equally weighted trial-position bins. For each trial, project
``f - P`` onto ``P`` without an intercept to obtain ``M = delta_gain * P``; the additive
component is ``A = f - P - M``. ``pf_var_{total,template,gain,additive,covariance}_<comparison>``
stores population variances (ddof=0), where covariance is twice the sum of the three pairwise
covariances. ``pf_frac_var_{template,gain,additive,covariance}_<comparison>`` divides by
``pf_var_total_<comparison>``. The four fractions sum to one; the covariance term is signed
and individual component fractions may exceed one. Zero total variance gives NaN fractions.
``pf_var_n_frames_<comparison>`` records the common finite frame count per ROI.
Schema v14 adds ``pf_frac_unexplained_<comparison>`` and its slot counterparts:
the uncentered additive residual mean square divided by total activity variance,
equivalent to 1 − R² for the combined PF × gain prediction. Values can exceed one.
Schema v15 splits that fraction with complementary soft membership weights from
the unmodulated template (its height divided by its positive peak). The two
``frac_unexplained_*_shared`` terms share the overall SST and sum to the original
fraction. The two ``frac_unexplained_*_weighted`` terms each use their own
weighted activity mean and SST. The complementary ``r2_*_shared`` terms split
explained R² by allocating each region its share of the globally centered SST.
Schema v16 adds ``pf_residual_rms_{within,outside}_<comparison>`` and slot
counterparts. Each is the square root of the weighted mean squared residual
on the evaluation frames, using the same soft membership weights and each
region's own total weight. Regions with zero weight yield NaN.

The v15 Gamma null conditions on the observed all-trial template and the
actual trajectory, with no trial modulation. Independent frame draws use
shape PF(position) and rate 1 in the existing standardized activity units, so
their mean and variance both equal PF(position). A zero shape gives zero.
The same draws are scored against the fixed observed fold templates and are
binned and smoothed for trial gain, amplitude, and residual metrics. Per-ROI
Monte Carlo means and SDs are stored as ``gamma_{mean|std}_<observed_key>``,
as are summaries over the observed quality subsets, using ROIs with a finite
observed value for each metric.
Fold scores are conditional on the all-trial template: the draws do not
re-estimate templates or account for uncertainty in the generating PF.
The legacy PF-only prediction metrics have a separate Gamma reference using
their own (occupancy-weighted) template and matching evaluation frames.

Comparisons are ``all_from_all``, ``f1_from_f1``, ``f2_from_f1``, ``f1_from_f2``, and
``f2_from_f2``. Only the template is held out in cross-fold comparisons: trial gains are
estimated on the evaluation frames. ``cv`` averages the two cross-fold directions and
``within_cv`` averages the two within-fold measurements, requiring both directions to be
finite. Variances and fractions are averaged separately; frame counts are summed. These
averages are fold summaries, not variances of concatenated traces. The independent PF/RMS
correlation pairings do not define additional activity decompositions.

:func:`pf_variance_keys` enumerates the per-ROI keys. Top-level arrays describe the best
environment; ``_slot`` arrays have shape ``(MAX_ENV_SLOTS, rois)``. Finite-only mean/median
population summaries for all / quality / not-quality ROIs are enumerated by
:func:`pf_variance_population_summary_keys` and have shape ``(MAX_ENV_SLOTS,)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import blake2b
from typing import ClassVar

import numpy as np
from scipy.stats import spearmanr, skew, kurtosis

from vrAnalysis.helpers import vectorRSquared
from vrAnalysis.helpers.signals import vectorCorrelation
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.placefields import FrameBehavior, get_frame_behavior, get_placefield
from vrAnalysis.processors.spkmaps import SpkmapProcessor, SpkmapParams
from vrAnalysis.processors.support import median_zscore
from vrAnalysis.sessions import B2Session, SpksTypes
from ..env_order import MAX_ENV_SLOTS, load_env_order
from ..pipeline.base import AnalysisConfigBase
from ..registry import PopulationRegistry
from .gain_regression import (
    _fit_gaussian, _safe_gain, _weighted_row_mean, gaussian_gain_matrix, least_squares_gain_matrix,
)
from .regression import _placefield_membership_weights

VALID_SPKS_TYPES: list[SpksTypes] = ["oasis", "sigrebase"]

# Each trial metric is reduced across trials with each statistic.  The completed result key is
# ``f"{metric}_{statistic}"``; ``trial_activity_summary_keys()`` returns the literal 15-key list.
TRIAL_ACTIVITY_METRICS: tuple[str, ...] = ("trial_rms", "trial_mean", "trial_pf_dot", "trial_gain", "trial_gain_least_squares")
TRIAL_ACTIVITY_STATISTICS: tuple[str, ...] = ("mean", "variance", "cv")
TRIAL_ACTIVITY_ROI_STATISTICS: tuple[str, ...] = ("mean", "median")
TRIAL_ACTIVITY_SUBSETS: tuple[str, ...] = ("", "quality_filtered_", "notquality_filtered_")

# Names read as RMS on <evaluation trials> from <place-field template trials>.
PF_RMS_COMPARISONS = {
    "all_from_all": ("all", "all_from_all"),
    "f1_from_f1": ("f1", "f1_from_f1"),
    "f2_from_f1": ("f1", "f2_from_f1"),
    "f1_from_f2": ("f2", "f1_from_f2"),
    "f2_from_f2": ("f2", "f2_from_f2"),
    "independent_f1": ("f1", "f2_from_f2"),
    "independent_f2": ("f2", "f1_from_f1"),
}

PF_VARIANCE_COMPARISONS: tuple[str, ...] = (
    "all_from_all", "f1_from_f1", "f2_from_f1", "f1_from_f2", "f2_from_f2", "cv", "within_cv",
)
PF_VARIANCE_METRICS: tuple[str, ...] = (
    "var_total", "var_template", "var_gain", "var_additive", "var_covariance",
    "frac_var_template", "frac_var_gain", "frac_var_additive", "frac_var_covariance", "frac_unexplained", "var_n_frames",
    "frac_unexplained_within_shared", "frac_unexplained_outside_shared",
    "frac_unexplained_within_weighted", "frac_unexplained_outside_weighted",
    "r2_within_shared", "r2_outside_shared",
    "residual_rms_within", "residual_rms_outside",
)


def pf_variance_keys() -> list[str]:
    """Per-ROI temporal decomposition keys; per-environment counterparts append ``_slot``."""
    return [f"pf_{metric}_{comparison}" for comparison in PF_VARIANCE_COMPARISONS for metric in PF_VARIANCE_METRICS]


def pf_variance_population_summary_keys() -> list[str]:
    """Per-slot finite-only mean/median for all, quality, and not-quality ROI subsets."""
    return [
        f"{statistic}_{subset}{key}"
        for statistic in TRIAL_ACTIVITY_ROI_STATISTICS
        for subset in TRIAL_ACTIVITY_SUBSETS
        for key in pf_variance_keys()
    ]


def _trial_gain_component(activity: np.ndarray, prediction: np.ndarray, trial: np.ndarray) -> np.ndarray:
    """PF-shaped component of the remaining activity, fitted over each complete trial.

    Inputs are ``(frames, rois)`` activity/template and ``(frames,)`` trial labels.
    Use the same pairwise-finite frame mask as the variance measurement. Zero template
    energy gives zero modulation; undefined samples remain NaN.
    """
    valid = np.isfinite(activity) & np.isfinite(prediction) & np.isfinite(trial[:, None])
    p = np.where(valid, prediction, 0.0)
    remainder = np.where(valid, activity - prediction, 0.0)
    modulation = np.zeros_like(p, dtype=float)
    for label in np.unique(trial[np.isfinite(trial)]):
        rows = np.flatnonzero(trial == label)
        energy = np.sum(p[rows] ** 2, axis=0)
        delta_gain = np.divide(
            np.sum(p[rows] * remainder[rows], axis=0), energy,
            out=np.zeros_like(energy), where=energy > 0,
        )
        modulation[rows] = p[rows] * delta_gain
    return np.where(valid, modulation, np.nan)


def _temporal_variance_components(
    activity: np.ndarray, prediction: np.ndarray, trial: np.ndarray, *, template_peak: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Decompose ``(frames, rois)`` activity using a fixed PF and residual trial gain.

    ``delta_gain = sum(P * (f - P)) / sum(P**2)`` within each trial and ROI, using
    frames finite in both activity and prediction. No intercept, centering, or gain clipping
    is applied in this fit. A zero-energy template contributes zero gain activity, leaving
    the remainder additive. All variances/covariances use the same finite frames and ddof=0.
    Trial labels must be finite. Components need not be uncorrelated.
    """
    activity = np.asarray(activity, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    trial = np.asarray(trial)
    if activity.ndim != 2 or prediction.shape != activity.shape or trial.shape != (len(activity),):
        raise ValueError("activity/prediction must be (frames, rois), with one trial label per frame")
    valid = np.isfinite(activity) & np.isfinite(prediction) & np.isfinite(trial[:, None])
    y = np.where(valid, activity, 0.0)
    p = np.where(valid, prediction, 0.0)
    remainder = y - p
    modulation = _trial_gain_component(y, p, trial)
    additive = remainder - modulation
    count = np.sum(valid, axis=0)

    def mean(values):
        return np.divide(
            np.sum(np.where(valid, values, 0.0), axis=0), count,
            out=np.full(activity.shape[1], np.nan), where=count > 0,
        )

    centered = [np.where(valid, values - mean(values), 0.0) for values in (y, p, modulation, additive)]
    result = {f"var_{name}": mean(values**2) for name, values in zip(
        ("total", "template", "gain", "additive"), centered,
    )}
    _, pc, mc, ac = centered
    result["var_covariance"] = 2 * mean(pc * mc + pc * ac + mc * ac)
    for name in ("template", "gain", "additive", "covariance"):
        result[f"frac_var_{name}"] = np.divide(
            result[f"var_{name}"], result["var_total"],
            out=np.full(activity.shape[1], np.nan), where=result["var_total"] > 0,
        )
    result["frac_unexplained"] = np.divide(
        mean(additive**2), result["var_total"],
        out=np.full(activity.shape[1], np.nan), where=result["var_total"] > 0,
    )
    if template_peak is None:
        template_peak = np.max(np.where(valid, p, -np.inf), axis=0, initial=-np.inf)
    centered_squared = centered[0] ** 2
    sst = np.sum(centered_squared, axis=0)
    shared_ok = (sst > 0) & np.isfinite(template_peak) & (template_peak > 0)
    residual_squared = np.where(valid, additive, 0.0) ** 2
    for region, weight in zip(
        ("within", "outside"), _placefield_membership_weights(p.T, template_peak, valid.T)
    ):
        weight = weight.T
        total_weight = weight.sum(axis=0)
        sse = np.sum(weight * residual_squared, axis=0)
        result[f"residual_rms_{region}"] = np.sqrt(np.divide(
            sse, total_weight, out=np.full_like(sst, np.nan), where=total_weight > 0,
        ))
        mean_y = np.divide(np.sum(weight * y, axis=0), total_weight,
                           out=np.zeros_like(sst), where=total_weight > 0)
        regional_sst = np.sum(weight * (y - mean_y)**2, axis=0)
        result[f"frac_unexplained_{region}_shared"] = np.divide(
            sse, sst, out=np.full_like(sst, np.nan),
            where=shared_ok,
        )
        result[f"r2_{region}_shared"] = np.divide(
            np.sum(weight * centered_squared, axis=0) - sse, sst,
            out=np.full_like(sst, np.nan), where=shared_ok,
        )
        result[f"frac_unexplained_{region}_weighted"] = np.divide(
            sse, regional_sst, out=np.full_like(sst, np.nan), where=regional_sst > 0,
        )
    result["var_n_frames"] = count.astype(float)
    return result


def _placefield_variance_by_fold(
    activity: np.ndarray,
    frame_bins: np.ndarray,
    frame_trials: np.ndarray,
    trials: np.ndarray,
    templates: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Temporal decomposition for chronological trial folds and their average PF templates.

    ``trials`` is the ordered environment-local trial list used to build the PF/RMS maps.
    ``templates`` contains ``all``, ``f1``, and ``f2`` arrays shaped ``(rois, positions)``.
    Excluded behavior frames have NaN trial labels; invalid position indices are excluded.
    ROI chunks bound temporary frame-by-ROI arrays without storing component time series.
    """
    n_rois = activity.shape[1]
    n_bins = templates["all"].shape[1]
    folds = {"all": trials, "f1": trials[::2], "f2": trials[1::2]}
    peaks = {name: np.max(np.where(np.isfinite(pf), pf, -np.inf), axis=1, initial=-np.inf)
             for name, pf in templates.items()}
    rows = {
        fold: np.flatnonzero(np.isin(frame_trials, labels) & (frame_bins >= 0) & (frame_bins < n_bins))
        for fold, labels in folds.items()
    }
    result = {key: np.full(n_rois, np.nan) for key in pf_variance_keys()}
    for comparison in PF_VARIANCE_COMPARISONS[:5]:
        evaluation, template = comparison.split("_from_")
        indices = rows[evaluation]
        for start in range(0, n_rois, 128):
            stop = min(start + 128, n_rois)
            prediction = templates[template][start:stop, frame_bins[indices]].T
            measured = _temporal_variance_components(
                activity[indices, start:stop], prediction, frame_trials[indices],
                template_peak=peaks[template][start:stop],
            )
            for metric, values in measured.items():
                result[f"pf_{metric}_{comparison}"][start:stop] = values
    for comparison, directions in {
        "cv": ("f2_from_f1", "f1_from_f2"),
        "within_cv": ("f1_from_f1", "f2_from_f2"),
    }.items():
        for metric in PF_VARIANCE_METRICS:
            left, right = [result[f"pf_{metric}_{direction}"] for direction in directions]
            result[f"pf_{metric}_{comparison}"] = (left + right) / (1 if metric == "var_n_frames" else 2)
    return result


def pf_rms_map_keys() -> list[str]:
    """Per-ROI, per-position maps; slot counterparts append ``_slot``."""
    return [f"pf_{fold}" for fold in ("all", "f1", "f2")] + [
        f"rms_{name}" for name in list(PF_RMS_COMPARISONS)[:5]
    ]


def pf_rms_correlation_keys() -> list[str]:
    """Per-ROI Pearson r across position, including both two-fold averages."""
    return [f"pf_rms_r_{name}" for name in PF_RMS_COMPARISONS] + ["pf_rms_r_cv", "pf_rms_r_independent_cv"]


def _spatial_pearson_r(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise-finite Pearson r across position; constant/undersampled maps give NaN."""
    valid = np.isfinite(left) & np.isfinite(right)
    count = np.sum(valid, axis=-1, keepdims=True)
    centered = []
    for values in (left, right):
        mean = np.divide(
            np.sum(np.where(valid, values, 0.0), axis=-1, keepdims=True),
            count, out=np.zeros_like(count, dtype=float), where=count > 0,
        )
        centered.append(np.where(valid, values - mean, 0.0))
    x, y = centered
    denominator = np.sqrt(np.sum(x**2, axis=-1) * np.sum(y**2, axis=-1))
    r = np.divide(
        np.sum(x * y, axis=-1), denominator,
        out=np.full(denominator.shape, np.nan), where=(count[..., 0] >= 2) & (denominator > 0),
    )
    return np.clip(r, -1, 1)


def _placefield_rms_maps(
    spkmap: np.ndarray, templates: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """PF/RMS maps and spatial r from ``(rois, chronological trials, positions)``.

    F1 takes alternating environment-local trials starting with the first; f2 takes the
    remainder. RMS squares each trial's residual before averaging trials at each position.
    ``f2_from_f1`` and ``f1_from_f2`` evaluate held-out trials against the training PF.
    ``independent_f1/f2`` correlate a PF with the other fold's within-fold RMS: neither
    the residual activity nor its template shares trials with the correlated PF.
    The two ``*_cv`` r values are arithmetic means of the two directions, requiring both.
    """
    spkmap = np.asarray(spkmap, dtype=float)
    rows = {"all": np.arange(spkmap.shape[1]), "f1": np.arange(0, spkmap.shape[1], 2),
            "f2": np.arange(1, spkmap.shape[1], 2)}

    def trial_mean(values):
        finite = np.isfinite(values)
        count = np.sum(finite, axis=1)
        return np.divide(
            np.sum(np.where(finite, values, 0.0), axis=1), count,
            out=np.full((spkmap.shape[0], spkmap.shape[2]), np.nan), where=count > 0,
        )

    result = {f"pf_{fold}": trial_mean(spkmap[:, indices]) if templates is None else templates[fold]
              for fold, indices in rows.items()}
    for name in list(PF_RMS_COMPARISONS)[:5]:
        evaluation, template = name.split("_from_")
        result[f"rms_{name}"] = np.sqrt(trial_mean(
            (spkmap[:, rows[evaluation]] - result[f"pf_{template}"][:, None, :]) ** 2
        ))
    for name, (fold, rms) in PF_RMS_COMPARISONS.items():
        result[f"pf_rms_r_{name}"] = _spatial_pearson_r(result[f"pf_{fold}"], result[f"rms_{rms}"])
    result["pf_rms_r_cv"] = (result["pf_rms_r_f2_from_f1"] + result["pf_rms_r_f1_from_f2"]) / 2
    result["pf_rms_r_independent_cv"] = (result["pf_rms_r_independent_f1"] + result["pf_rms_r_independent_f2"]) / 2
    return result


def trial_activity_summary_keys() -> list[str]:
    """Every per-slot, per-ROI trial-activity summary key emitted by this config."""
    return [f"{metric}_{statistic}" for metric in TRIAL_ACTIVITY_METRICS for statistic in TRIAL_ACTIVITY_STATISTICS]


def residual_rms_summary_keys() -> list[str]:
    """Per-slot/ROI statistics of gain-corrected trial RMS, for each template fold."""
    return [f"trial_residual_rms_{statistic}_{fold}"
            for fold in PF_VARIANCE_COMPARISONS for statistic in TRIAL_ACTIVITY_STATISTICS]


def residual_rms_population_summary_keys() -> list[str]:
    return [f"{statistic}_{subset}{key}" for statistic in TRIAL_ACTIVITY_ROI_STATISTICS
            for subset in TRIAL_ACTIVITY_SUBSETS for key in residual_rms_summary_keys()]


def _trial_unexplained_fraction(activity: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Per-trial SSE / SST over jointly finite position bins on the last axis."""
    valid = np.isfinite(activity) & np.isfinite(residual)
    count = valid.sum(axis=-1)
    trial_mean = np.divide(
        np.sum(np.where(valid, activity, 0.0), axis=-1), count,
        out=np.full(count.shape, np.nan), where=count > 0,
    )
    centered = np.where(valid, activity - trial_mean[..., None], 0.0)
    sst = np.sum(centered**2, axis=-1)
    sse = np.sum(np.where(valid, residual, 0.0)**2, axis=-1)
    return np.divide(sse, sst, out=np.full(sst.shape, np.nan), where=sst > 0)


def _trial_residual_rms_summaries(
    spkmap: np.ndarray, templates: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Summarize per-trial RMS of activity minus fitted gain × PF over position.

    Trials and position bins have equal weight; missing bins/trials are excluded.
    Gains are least-squares fits on evaluation trials, without an intercept. Only
    the template is held out. Degenerate template energy yields undefined gains.
    CV/within-CV average directional statistics per ROI (both must be finite),
    rather than pooling trials or averaging before computing the statistic.
    """
    spkmap = np.asarray(spkmap, dtype=float)
    rows = {"all": np.arange(spkmap.shape[1]), "f1": np.arange(0, spkmap.shape[1], 2),
            "f2": np.arange(1, spkmap.shape[1], 2)}
    result = {key: np.full(spkmap.shape[0], np.nan) for key in residual_rms_summary_keys()}
    if templates is None:
        templates = {}
        for name, indices in rows.items():
            subset = spkmap[:, indices]
            finite = np.isfinite(subset)
            count = finite.sum(axis=1)
            templates[name] = np.divide(np.sum(np.where(finite, subset, 0.0), axis=1), count,
                                        out=np.full(count.shape, np.nan), where=count > 0)
    for fold in PF_VARIANCE_COMPARISONS[:5]:
        evaluation, template = fold.split("_from_")
        if not rows[evaluation].size or not rows[template].size:
            continue
        # Fit the evaluation trials against an independently chosen trial-mean template.
        pf = templates[template]
        activity = spkmap[:, rows[evaluation]]
        valid = np.isfinite(activity) & np.isfinite(pf[:, None, :])
        y = np.where(valid, activity, 0.0)
        p = np.where(valid, pf[:, None, :], 0.0)
        energy = np.sum(p**2, axis=2)
        gain = np.divide(np.sum(y * p, axis=2), energy,
                         out=np.full(energy.shape, np.nan), where=energy > np.finfo(float).eps)
        residual = activity - gain[:, :, None] * pf[:, None, :]
        finite = np.isfinite(residual)
        count = finite.sum(axis=2)
        rms = np.sqrt(np.divide(np.sum(np.where(finite, residual**2, 0.0), axis=2), count,
                                out=np.full(count.shape, np.nan), where=count > 0))
        summaries = _summarize_trial_activity({"trial_residual_rms": rms})
        result.update({f"{key}_{fold}": value for key, value in summaries.items()})
    for fold, directions in {"cv": ("f2_from_f1", "f1_from_f2"),
                             "within_cv": ("f1_from_f1", "f2_from_f2")}.items():
        for statistic in TRIAL_ACTIVITY_STATISTICS:
            prefix = f"trial_residual_rms_{statistic}"
            result[f"{prefix}_{fold}"] = (result[f"{prefix}_{directions[0]}"] + result[f"{prefix}_{directions[1]}"]) / 2
    return result


def trial_activity_population_summary_keys() -> list[str]:
    """Every finite-only across-ROI summary key emitted by this config."""
    return [
        f"{roi_statistic}_{subset}{key}"
        for roi_statistic in TRIAL_ACTIVITY_ROI_STATISTICS
        for subset in TRIAL_ACTIVITY_SUBSETS
        for key in trial_activity_summary_keys()
    ]


@dataclass(frozen=True)
class PFPredQualityConfig(AnalysisConfigBase):
    """Configuration for place-field prediction quality analysis.

    Parameters
    ----------
    spks_type : SpksTypes
        Spike type to use for the place-field prediction.
    reliability_threshold : float
        Reliability cutoff for the R² histogram and quality-filtered trial summaries.
    fraction_active_threshold : float
        Fraction-active cutoff for quality-filtered trial summaries.
    accuracy_pct : float
        Percentile of true activity used as the accuracy threshold for
        fraction_accurate (fraction of frames where |pred - act| < threshold).
    n_hist_bins : int
        Number of bins in np.linspace(-1, 1) for binned reliability vs R²
        and the R² histogram.
    n_kde_grid : int
        Number of evaluation points for the KDE running-average curve.
    peak_hist_max : float
        Upper edge of the place-field peak-amplitude histogram, in units of an
        ROI's standard deviation in time. Peaks above it are counted in
        ``pf_peak_n`` but fall in no bin.
    n_peak_hist_bins : int
        Number of bins in np.linspace(0, peak_hist_max) for that histogram.
    n_gamma_runs : int
        Number of fixed-template null draws per environment. Zero skips the null.
    """

    # v4: added the per-experience-slot place-field peak amplitude keys (pf_peak,
    # pf_peak_hist, pf_peak_hist_edges, pf_peak_n, env_slot_ids).
    # v5: added the per-experience-slot R² keys (r2_slot, reliability_slot, r2_kde_slot,
    # r2_kde_pooled, best_env_slot), so R² vs reliability can be tracked per environment.
    # v6: added per-ROI RMS error and its binned, KDE, and per-slot counterparts.
    # v7: added RMS normalized by each ROI's activity standard deviation.
    # v8: added total, prediction, and residual variance, plus prediction/total variance.
    # v9: added per-cell summaries of four trial-wise activity-amplitude measurements, per-cell
    # fraction active and quality masks, and finite-only population summaries by quality subset.
    # v10: added least-squares trial gain and its per-cell / population summaries.
    # v11: two-fold spatial PF/RMS maps and their per-ROI Pearson correlations.
    # v12: frame-weighted PF / residual trial gain / additive variance and signed covariance,
    # with fold comparisons, per-slot ROI arrays, and quality-subset population summaries.
    # v13: position-bin-weighted gain-corrected trial residual RMS statistics by fold.
    # v14: temporal 1 − R² of the PF × trial gain prediction by fold.
    # v15: regional PF × gain scores and a fixed-template Gamma reference.
    # v16: regional PF × gain residual RMS in activity units.
    schema_version: str = "v16"
    data_config_name: str = "default"
    spks_type: SpksTypes = "sigrebase"
    reliability_threshold: float = 0.3
    fraction_active_threshold: float = 0.1
    accuracy_pct: float = 5.0
    n_hist_bins: int = 40
    n_kde_grid: int = 200
    peak_hist_max: float = 10.0
    n_peak_hist_bins: int = 100
    n_gamma_runs: int = 20

    display_name: ClassVar[str] = "pfpred_quality"

    @staticmethod
    def _param_grid() -> dict:
        return {}

    def validate(self):
        if self.spks_type not in VALID_SPKS_TYPES:
            raise ValueError(f"Unknown spks_type {self.spks_type!r}. Available: {VALID_SPKS_TYPES}")
        if not -1 <= self.reliability_threshold <= 1:
            raise ValueError("reliability_threshold must be between -1 and 1")
        if not 0 <= self.fraction_active_threshold <= 1:
            raise ValueError("fraction_active_threshold must be between 0 and 1")
        if self.n_gamma_runs < 0:
            raise ValueError("n_gamma_runs must be nonnegative")

    @property
    def bin_edges(self) -> np.ndarray:
        return np.linspace(-1, 1, self.n_hist_bins + 1)

    @property
    def kde_grid(self) -> np.ndarray:
        return np.linspace(-1, 1, self.n_kde_grid)

    @property
    def peak_bin_edges(self) -> np.ndarray:
        return np.linspace(0, self.peak_hist_max, self.n_peak_hist_bins + 1)

    def summary(self) -> str:
        gamma_suffix = "" if self.n_gamma_runs == 20 else f"_gamma={self.n_gamma_runs}"
        return (
            f"{self.display_name}_spks={self.spks_type}_rel={self.reliability_threshold:g}"
            f"_fa={self.fraction_active_threshold:g}{gamma_suffix}_{self.schema_version}"
        )

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        prev_spks_type = session.params.spks_type
        session.params.spks_type = self.spks_type
        try:
            smp = SpkmapProcessor(session, params=SpkmapParams())

            spks = session.spks[:, session.idx_rois]
            spks = median_zscore(spks, median_subtract=not session.zero_baseline_spks)

            reliability = smp.get_reliability()
            env_maps = smp.get_env_maps()
            best_env = int(np.argmax([omap.shape[0] for omap in env_maps.occmap]))

            placefield_prediction, extras = smp.get_placefield_prediction()

            idx_keep = extras["idx_valid"] & (extras["frame_environment_index"] == best_env)
            spks_valid = spks[idx_keep]
            pfpred_valid = placefield_prediction[idx_keep]

            r2 = vectorRSquared(pfpred_valid, spks_valid, axis=0)
            r2[r2 < -1] = np.nan
            cc = vectorCorrelation(pfpred_valid, spks_valid, axis=0)
            rms = _rms_error(pfpred_valid, spks_valid)
            norm_rms = _normalized_rms_error(pfpred_valid, spks_valid)
            variance = _variance_components(pfpred_valid, spks_valid)

            relia = reliability.values[best_env]  # best env, shape (n_rois,)

            result = {"r2": r2, "cc": cc, "rms": rms, "norm_rms": norm_rms, "reliability": relia}
            result.update(variance)
            result.update(_per_roi_stats(spks_valid, pfpred_valid, r2, relia, self.accuracy_pct))

            bin_edges = self.bin_edges
            result.update(_binned_r2(r2, relia, bin_edges))
            result.update(_binned_rms(rms, relia, bin_edges))
            result.update(_kde_r2(r2, relia, self.kde_grid))
            result.update(_kde_rms(rms, relia, self.kde_grid))

            idx_reliable = np.isfinite(r2) & (relia > self.reliability_threshold)
            r2_hist_counts, _ = np.histogram(r2[idx_reliable], bins=bin_edges)
            result["r2_hist_counts"] = r2_hist_counts.astype(float)

            result.update(_placefield_peaks(session, spks, smp, self.peak_bin_edges))
            gamma_seed = (
                int.from_bytes(blake2b(session.session_print().encode(), digest_size=8).digest(), "little")
                if self.n_gamma_runs else 0
            )
            result.update(
                _trial_activity_by_slot(
                    session,
                    spks,
                    smp,
                    reliability,
                    env_maps,
                    best_env,
                    self.reliability_threshold,
                    self.fraction_active_threshold,
                    n_gamma_runs=self.n_gamma_runs,
                    gamma_seed=gamma_seed,
                )
            )
            result.update(_r2_by_slot(
                session, spks, placefield_prediction, extras, reliability, env_maps, best_env, self.kde_grid,
                n_gamma_runs=self.n_gamma_runs, gamma_seed=gamma_seed,
            ))

            return result
        finally:
            session.params.spks_type = prev_spks_type


def _summarize_trial_activity(trial_values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reduce per-ROI, per-trial measurements to mean, variance, and coefficient of variation.

    Every input has shape ``(rois, trials)``. Variance and standard deviation use ``ddof=0``.
    The coefficient of variation is ``std / mean`` and is NaN when the mean is non-finite or
    effectively zero. Result names follow ``{trial_metric}_{statistic}`` and are enumerated by
    :func:`trial_activity_summary_keys`.
    """
    result: dict[str, np.ndarray] = {}
    for metric, measurements in trial_values.items():
        values = np.asarray(measurements, dtype=float)
        finite = np.isfinite(values)
        count = np.sum(finite, axis=1)
        total = np.sum(np.where(finite, values, 0.0), axis=1)
        mean = np.divide(total, count, out=np.full(values.shape[0], np.nan), where=count > 0)
        centered = np.where(finite, values - mean[:, None], 0.0)
        variance = np.divide(
            np.sum(centered**2, axis=1),
            count,
            out=np.full(values.shape[0], np.nan),
            where=count > 0,
        )
        valid_mean = np.isfinite(mean) & (np.abs(mean) > np.finfo(float).eps)
        cv = np.divide(np.sqrt(variance), mean, out=np.full_like(mean, np.nan), where=valid_mean)
        result[f"{metric}_mean"] = mean
        result[f"{metric}_variance"] = variance
        result[f"{metric}_cv"] = cv
    return result


def _summarize_trial_activity_across_rois(
    per_roi: dict[str, np.ndarray],
    quality_filtered_roi_mask: np.ndarray,
) -> dict[str, float]:
    """Finite-only mean and median of every per-ROI trial summary for three ROI subsets.

    Naming matches :class:`RegressionPlacefieldResidualConfig`: no subset prefix means all ROIs,
    ``quality_filtered_`` means cells above both thresholds, and ``notquality_filtered_`` is the
    complement. In particular, an invalid gain does not poison a subset summary; that cell is
    omitted for that gain key while remaining available for other finite metrics.
    """
    return _summarize_metrics_across_rois(per_roi, quality_filtered_roi_mask)


def _summarize_metrics_across_rois(
    per_roi: dict[str, np.ndarray], quality_filtered_roi_mask: np.ndarray,
) -> dict[str, float]:
    """Finite-only mean/median for each supplied metric and the established quality subsets."""
    quality_filtered_roi_mask = np.asarray(quality_filtered_roi_mask, dtype=bool)
    subset_masks = {
        "": np.ones_like(quality_filtered_roi_mask, dtype=bool),
        "quality_filtered_": quality_filtered_roi_mask,
        "notquality_filtered_": ~quality_filtered_roi_mask,
    }
    result: dict[str, float] = {}
    for key, values in per_roi.items():
        values = np.asarray(values, dtype=float)
        for subset, subset_mask in subset_masks.items():
            finite_values = values[subset_mask & np.isfinite(values)]
            result[f"mean_{subset}{key}"] = float(np.mean(finite_values)) if finite_values.size else np.nan
            result[f"median_{subset}{key}"] = float(np.median(finite_values)) if finite_values.size else np.nan
    return result


def _trial_activity_measurements(
    spkmap: np.ndarray, bin_centers: np.ndarray, *,
    template: np.ndarray | None = None,
    gaussian: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Measure five activity amplitudes for every ROI and trial in one environment.

    Parameters
    ----------
    spkmap : np.ndarray
        Trial-resolved activity with shape ``(rois, trials, position_bins)``.
    bin_centers : np.ndarray
        Position-bin centers used by the Gaussian gain estimator.

    Returns
    -------
    dict[str, np.ndarray]
        ``trial_rms`` is RMS across position, ``trial_mean`` is the position mean,
        ``trial_pf_dot`` is the dot product with the unit-L2 place field formed from all trials,
        and ``trial_gain`` is the all-trial Gaussian place-field gain used by
        :class:`~dimensionality_manuscript.configs.gain_regression.GainRegressionConfig`.
        ``trial_gain_least_squares`` uses the nonparametric least-squares projection from
        :func:`~.gain_regression.least_squares_gain_matrix`, with all trials building the template
        (the descriptive equivalent of ``placefield_split="all"``). It fits no intercept or
        Gaussian and is NaN for degenerate template energy. These summaries are not held-out
        gain-regression targets. Each array has shape ``(rois, trials)``.
        No reliability filtering is applied; failed or
        degenerate Gaussian place-field fits yield NaN gain for that cell.
    """
    spkmap = np.asarray(spkmap, dtype=float)
    finite = np.isfinite(spkmap)
    bin_count = np.sum(finite, axis=2)
    trial_sum = np.sum(np.where(finite, spkmap, 0.0), axis=2)
    trial_square_sum = np.sum(np.where(finite, spkmap**2, 0.0), axis=2)
    trial_mean = np.divide(trial_sum, bin_count, out=np.full(spkmap.shape[:2], np.nan), where=bin_count > 0)
    trial_rms = np.sqrt(np.divide(trial_square_sum, bin_count, out=np.full(spkmap.shape[:2], np.nan), where=bin_count > 0))

    mean_pf = template
    if mean_pf is None:
        pf_count = np.sum(finite, axis=1)
        mean_pf = np.divide(
            np.sum(np.where(finite, spkmap, 0.0), axis=1),
            pf_count,
            out=np.full((spkmap.shape[0], spkmap.shape[2]), np.nan),
            where=pf_count > 0,
        )
    pf_norm = np.sqrt(np.nansum(mean_pf**2, axis=1))
    valid_norm = np.isfinite(pf_norm) & (pf_norm > np.finfo(float).eps)
    unit_pf = np.divide(
        mean_pf,
        pf_norm[:, None],
        out=np.full_like(mean_pf, np.nan),
        where=valid_norm[:, None],
    )
    dot_valid = finite & np.isfinite(unit_pf[:, None, :])
    trial_pf_dot = np.sum(np.where(dot_valid, spkmap * unit_pf[:, None, :], 0.0), axis=2)
    trial_pf_dot[~np.any(dot_valid, axis=2)] = np.nan

    if template is None:
        trial_gain, _, _ = gaussian_gain_matrix(spkmap, np.asarray(bin_centers, dtype=float))
        trial_gain_least_squares, _, _ = least_squares_gain_matrix(spkmap)
    else:
        weights, denominator = gaussian
        trial_gain = _safe_gain(_weighted_row_mean(spkmap, weights), denominator)
        p = np.where(dot_valid, template[:, None, :], 0.0)
        energy = np.sum(p**2, axis=2)
        trial_gain_least_squares = np.divide(
            np.sum(np.where(dot_valid, spkmap, 0.0) * p, axis=2), energy,
            out=np.full(spkmap.shape[:2], np.nan), where=energy > np.finfo(float).eps,
        )
    return {
        "trial_rms": trial_rms,
        "trial_mean": trial_mean,
        "trial_pf_dot": trial_pf_dot,
        "trial_gain": trial_gain,
        "trial_gain_least_squares": trial_gain_least_squares,
    }


def _gamma_observed_keys() -> list[str]:
    return (pf_variance_keys() + trial_activity_summary_keys()
            + residual_rms_summary_keys() + pf_rms_correlation_keys())


def _gamma_summary_keys() -> list[str]:
    return [f"gamma_{stat}_{key}" for stat in ("mean", "std") for key in _gamma_observed_keys()]


def _gamma_moments(draws: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Finite-only Monte Carlo moments, without retaining individual draws."""
    moments = {}
    for metrics in draws:
        for key, values in metrics.items():
            if key not in moments:
                moments[key] = np.zeros(values.shape, dtype=int), np.zeros(values.shape), np.zeros(values.shape)
            count, mean, m2 = moments[key]
            finite = np.isfinite(values)
            count += finite
            delta = np.where(finite, values - mean, 0.0)
            mean += np.divide(delta, count, out=np.zeros_like(delta), where=count > 0)
            m2 += delta * np.where(finite, values - mean, 0.0)
    return {
        f"gamma_{stat}_{key}": values
        for key, (count, mean, m2) in moments.items()
        for stat, values in (
            ("mean", np.where(count > 0, mean, np.nan)),
            ("std", np.sqrt(np.divide(m2, count - 1, out=np.full_like(m2, np.nan), where=count > 1))),
        )
    }


def _gamma_reference(
    behavior: FrameBehavior, activity: np.ndarray, frame_bins: np.ndarray, trials: np.ndarray,
    templates: dict[str, np.ndarray],
    dist_edges: np.ndarray, speed_threshold: float, smooth_width: float | None,
    n_runs: int, rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Run the observed frame- and position-based measurements under a fixed-PF null."""
    n_rois = activity.shape[1]
    result = {key: np.full(n_rois, np.nan) for key in _gamma_summary_keys()}
    if not len(activity) or not len(trials):
        return result
    centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])
    frame_trials = np.where(behavior.speed >= speed_threshold, behavior.trial, np.nan)
    moving = np.isfinite(frame_trials) & (frame_bins >= 0) & (frame_bins < len(centers))
    for start in range(0, n_rois, 128):
        roi = slice(start, start + 128)
        fixed = {fold: pf[roi] for fold, pf in templates.items()}
        expectation = fixed["all"][:, frame_bins.clip(0, len(centers) - 1)].T
        valid = np.isfinite(expectation) & np.isfinite(activity[:, roi])
        gamma_shape = np.where(valid & moving[:, None], np.clip(expectation, 0, None), 0.0)
        fitted, weights = map(np.asarray, zip(*(_fit_gaussian(centers, row) for row in fixed["all"])))
        gaussian = weights, _weighted_row_mean(fitted[:, None, :], weights)[:, 0]
        def draws():
            for _ in range(n_runs):
                sim = rng.standard_gamma(gamma_shape)
                sim[~valid] = np.nan
                temporal = _placefield_variance_by_fold(sim, frame_bins, frame_trials, trials, fixed)
                trial_pf = get_placefield(sim, behavior, dist_edges, speed_threshold,
                                          average=False, smooth_width=smooth_width)
                # Match the order of observed environment-local trials, including silent trials.
                simulated_maps = np.full((expectation.shape[1], len(trials), len(centers)), np.nan)
                present = np.isin(trial_pf.trials, trials)
                indices = np.searchsorted(trials, trial_pf.trials[present])
                simulated_maps[:, indices] = trial_pf.placefield[present].transpose(2, 0, 1)
                spatial = _placefield_rms_maps(simulated_maps, fixed)
                yield {
                    **temporal,
                    **_summarize_trial_activity(_trial_activity_measurements(
                        simulated_maps, centers, template=fixed["all"], gaussian=gaussian,
                    )),
                    **_trial_residual_rms_summaries(simulated_maps, fixed),
                    **{key: spatial[key] for key in pf_rms_correlation_keys()},
                }
        for key, values in _gamma_moments(draws()).items():
            result[key][roi] = values
    return result


def _trial_activity_by_slot(
    session: B2Session,
    spks: np.ndarray,
    smp: SpkmapProcessor,
    reliability,
    env_maps,
    best_env: int,
    reliability_threshold: float,
    fraction_active_threshold: float,
    *,
    n_gamma_runs: int = 0,
    gamma_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-cell and quality-subset trial summaries on the environment-slot axis.

    A cell is quality filtered within an environment when both its reliability and fraction
    active are strictly above their thresholds. ``notquality_filtered`` is the complement.
    Across-ROI summaries explicitly select finite values, so failed trial-gain fits are ignored.
    The same trial maps supply the PF/RMS maps and spatial correlations (see module docstring).
    """
    frame_behavior = get_frame_behavior(session)
    trial_placefields = get_placefield(
        spks,
        frame_behavior,
        smp.dist_edges,
        smp.params.speed_threshold,
        average=False,
        smooth_width=smp.params.smooth_width,
    )
    n_rois = spks.shape[1]
    result = {key: np.full((MAX_ENV_SLOTS, n_rois), np.nan) for key in trial_activity_summary_keys()}
    result.update({key: np.full(MAX_ENV_SLOTS, np.nan) for key in trial_activity_population_summary_keys()})
    result.update({key: np.full((MAX_ENV_SLOTS, n_rois), np.nan) for key in residual_rms_summary_keys()})
    result.update({key: np.full(MAX_ENV_SLOTS, np.nan) for key in residual_rms_population_summary_keys()})
    spatial_keys = pf_rms_map_keys() + pf_rms_correlation_keys()
    for key in spatial_keys + pf_variance_keys():
        shape = (n_rois, len(smp.dist_edges) - 1) if key in pf_rms_map_keys() else (n_rois,)
        result[key] = np.full(shape, np.nan)
        result[f"{key}_slot"] = np.full((MAX_ENV_SLOTS, *shape), np.nan)
    result.update({key: np.full(MAX_ENV_SLOTS, np.nan) for key in pf_variance_population_summary_keys()})
    if n_gamma_runs:
        result["gamma_n_runs"] = n_gamma_runs
        for key in _gamma_summary_keys():
            result[key] = np.full(n_rois, np.nan)
            result[f"{key}_slot"] = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
            for statistic in TRIAL_ACTIVITY_ROI_STATISTICS:
                for subset in TRIAL_ACTIVITY_SUBSETS:
                    result[f"{statistic}_{subset}{key}"] = np.full(MAX_ENV_SLOTS, np.nan)
    result["pf_rms_distcenters"] = 0.5 * (np.asarray(smp.dist_edges[:-1]) + np.asarray(smp.dist_edges[1:]))
    fraction_active_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    quality_filtered_roi_mask_slot = np.zeros((MAX_ENV_SLOTS, n_rois), dtype=bool)
    num_rois_slot = np.full(MAX_ENV_SLOTS, np.nan)
    num_quality_filtered_rois_slot = np.full(MAX_ENV_SLOTS, np.nan)
    num_notquality_filtered_rois_slot = np.full(MAX_ENV_SLOTS, np.nan)
    best_fraction_active = np.full(n_rois, np.nan)
    best_quality_filtered_roi_mask = np.zeros(n_rois, dtype=bool)
    mouse_order = load_env_order().get(session.mouse_name)
    if mouse_order is None:
        return {
            **result,
            "fraction_active": best_fraction_active,
            "fraction_active_slot": fraction_active_slot,
            "quality_filtered_roi_mask": best_quality_filtered_roi_mask,
            "quality_filtered_roi_mask_slot": quality_filtered_roi_mask_slot,
            "num_rois_slot": num_rois_slot,
            "num_quality_filtered_rois_slot": num_quality_filtered_rois_slot,
            "num_notquality_filtered_rois_slot": num_notquality_filtered_rois_slot,
        }

    bin_centers = 0.5 * (np.asarray(smp.dist_edges[:-1]) + np.asarray(smp.dist_edges[1:]))
    frame_bins = np.searchsorted(smp.dist_edges, frame_behavior.position, side="right") - 1
    valid_frames = np.isfinite(frame_behavior.position) & np.isfinite(frame_behavior.trial)
    valid_frames &= np.isfinite(frame_behavior.speed) & (frame_behavior.speed >= smp.params.speed_threshold)
    env_to_index = {int(env): idx for idx, env in enumerate(env_maps.environments)}
    for env in sorted(int(value) for value in np.unique(trial_placefields.environment) if value >= 0):
        if env not in mouse_order or env not in env_to_index:
            continue
        slot = mouse_order.index(env)
        if slot >= MAX_ENV_SLOTS:
            continue
        pf_env = trial_placefields.filter_by_environment(env)
        spkmap = np.transpose(pf_env.placefield, (2, 0, 1))
        # Placefield rows are ordered by true trial number; explicitly preserve chronology.
        order = np.argsort(pf_env.trials)
        spkmap = spkmap[:, order]
        spatial = _placefield_rms_maps(spkmap)
        for key, values in spatial.items():
            result[f"{key}_slot"][slot] = values
            if env_to_index[env] == best_env:
                result[key] = values
        trials = np.asarray(pf_env.trials)[order]
        templates = {fold: spatial[f"pf_{fold}"] for fold in ("all", "f1", "f2")}
        decomposition = _placefield_variance_by_fold(
            spks, frame_bins,
            np.where(valid_frames & (frame_behavior.environment == env), frame_behavior.trial, np.nan),
            trials, templates,
        )
        for key, values in decomposition.items():
            result[f"{key}_slot"][slot] = values
            if env_to_index[env] == best_env:
                result[key] = values
        summaries = _summarize_trial_activity(_trial_activity_measurements(spkmap, bin_centers))
        residual_summaries = _trial_residual_rms_summaries(spkmap)
        for key, values in residual_summaries.items():
            result[key][slot] = values
        for key, values in summaries.items():
            result[key][slot] = values

        fraction_active = FractionActive.compute(
            spkmap,
            activity_axis=2,
            fraction_axis=1,
            activity_method="rms",
            fraction_method="participation",
        )
        env_index = env_to_index[env]
        relia = np.asarray(reliability.values[env_index])
        quality_filtered = np.isfinite(relia) & np.isfinite(fraction_active)
        quality_filtered &= relia > reliability_threshold
        quality_filtered &= fraction_active > fraction_active_threshold

        fraction_active_slot[slot] = fraction_active
        quality_filtered_roi_mask_slot[slot] = quality_filtered
        num_rois_slot[slot] = n_rois
        num_quality_filtered_rois_slot[slot] = np.sum(quality_filtered)
        num_notquality_filtered_rois_slot[slot] = np.sum(~quality_filtered)
        for key, value in _summarize_trial_activity_across_rois(summaries, quality_filtered).items():
            result[key][slot] = value
        for key, value in _summarize_metrics_across_rois(decomposition, quality_filtered).items():
            result[key][slot] = value
        for key, value in _summarize_metrics_across_rois(residual_summaries, quality_filtered).items():
            result[key][slot] = value
        if n_gamma_runs:
            env_frames = (frame_behavior.environment == env) & np.isin(frame_behavior.trial, trials)
            gamma = _gamma_reference(
                frame_behavior.filter(env_frames), spks[env_frames], frame_bins[env_frames], trials,
                templates, np.asarray(smp.dist_edges),
                smp.params.speed_threshold, smp.params.smooth_width, n_gamma_runs,
                np.random.default_rng(np.random.SeedSequence([gamma_seed, slot])),
            )
            for key, values in gamma.items():
                result[f"{key}_slot"][slot] = values
                if env_index == best_env:
                    result[key] = values
            observed = {**decomposition, **summaries, **residual_summaries, **spatial}
            paired_gamma = {
                key: np.where(np.isfinite(observed[key.split("_", 2)[2]]), values, np.nan)
                for key, values in gamma.items()
            }
            for key, value in _summarize_metrics_across_rois(paired_gamma, quality_filtered).items():
                result[key][slot] = value
        if env_index == best_env:
            best_fraction_active = fraction_active
            best_quality_filtered_roi_mask = quality_filtered

    return {
        **result,
        "fraction_active": best_fraction_active,
        "fraction_active_slot": fraction_active_slot,
        "quality_filtered_roi_mask": best_quality_filtered_roi_mask,
        "quality_filtered_roi_mask_slot": quality_filtered_roi_mask_slot,
        "num_rois_slot": num_rois_slot,
        "num_quality_filtered_rois_slot": num_quality_filtered_rois_slot,
        "num_notquality_filtered_rois_slot": num_notquality_filtered_rois_slot,
    }


def _placefield_peaks(
    session: B2Session,
    spks: np.ndarray,
    smp: SpkmapProcessor,
    bin_edges: np.ndarray,
) -> dict:
    """Peak place-field amplitude per ROI, on the experience-order environment slot axis.

    Place fields are trial-averaged maps from :func:`get_placefield` — one per environment the
    session ran — built from the already standardized ``spks``, so a peak (the max of a map over
    position) is in units of that ROI's standard deviation in time. Unlike the R² keys, which
    describe the session's best environment only, every environment contributes here.

    Results are written to the per-mouse experience-order slot axis rather than to the session's
    own environment list: slot j is the j-th environment *that mouse* ever saw, so column j means
    the same thing in every session of every mouse and the aggregator can stack across mice.
    Environments a session did not run (and mice missing from the environment-order map) stay NaN.

    Parameters
    ----------
    session : B2Session
        Session being processed, already switched to the config's ``spks_type``.
    spks : np.ndarray
        Standardized activity, shape ``(frames, rois)`` — the same array the R² keys use.
    smp : SpkmapProcessor
        Supplies the position bins and the speed/smoothing parameters of the map.
    bin_edges : np.ndarray
        Edges of the precomputed peak-amplitude histogram, shape ``(n_bins + 1,)``.

    Returns
    -------
    dict
        ``pf_peak`` ``(MAX_ENV_SLOTS, rois)``, ``pf_peak_hist`` ``(MAX_ENV_SLOTS, n_bins)`` counts,
        ``pf_peak_n`` ``(MAX_ENV_SLOTS,)`` finite peaks per slot (including any above the last
        edge, so a density is ``pf_peak_hist / (pf_peak_n * bin_width)``), ``pf_peak_hist_edges``,
        and ``env_slot_ids`` ``(MAX_ENV_SLOTS,)`` mapping each slot to its environment index.
    """
    frame_behavior = get_frame_behavior(session)
    placefield = get_placefield(
        spks,
        frame_behavior,
        smp.dist_edges,
        smp.params.speed_threshold,
        average=True,
        smooth_width=smp.params.smooth_width,
    )
    environments = np.asarray(placefield.environment, dtype=int)
    peaks = np.nanmax(placefield.placefield, axis=1)  # (environments, rois)

    n_rois = spks.shape[1]
    n_bins = len(bin_edges) - 1
    mouse_order = load_env_order().get(session.mouse_name)

    env_slot_ids = np.full(MAX_ENV_SLOTS, np.nan)
    if mouse_order is not None:
        env_slot_ids[: len(mouse_order)] = mouse_order

    pf_peak = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    # NaN rather than 0 for slots the session did not run: a zero count is a real measurement
    # ("no ROI peaked in this bin"), and summing missing slots as zeros would understate them.
    pf_peak_hist = np.full((MAX_ENV_SLOTS, n_bins), np.nan)
    pf_peak_n = np.full(MAX_ENV_SLOTS, np.nan)

    for env, env_peaks in zip(environments, peaks):
        # env < 0 marks invalid trials; a mouse missing from the order map has no slot axis.
        if env < 0 or mouse_order is None or env not in mouse_order:
            continue
        slot = mouse_order.index(env)
        finite_peaks = env_peaks[np.isfinite(env_peaks)]
        pf_peak[slot] = env_peaks
        pf_peak_hist[slot] = np.histogram(finite_peaks, bins=bin_edges)[0]
        pf_peak_n[slot] = finite_peaks.size

    return {
        "pf_peak": pf_peak,
        "pf_peak_hist": pf_peak_hist,
        "pf_peak_hist_edges": bin_edges,
        "pf_peak_n": pf_peak_n,
        "env_slot_ids": env_slot_ids,
    }


def _r2_by_slot(
    session: B2Session,
    spks: np.ndarray,
    placefield_prediction: np.ndarray,
    extras: dict,
    reliability,
    env_maps,
    best_env: int,
    kde_grid: np.ndarray,
    *,
    n_gamma_runs: int = 0,
    gamma_seed: int = 0,
) -> dict:
    """R² vs reliability per environment, on the experience-order environment slot axis.

    The top-level R² keys describe the session's best environment only. These repeat the same
    measurement for *every* environment the session ran, written to the per-mouse experience-order
    slot axis (slot j is the j-th environment that mouse ever saw) so column j means the same
    thing in every session of every mouse. Nothing is recomputed from disk: the prediction and the
    frame masks are the ones already built for the best environment, just restricted to a
    different set of frames.

    ``r2_kde_slot`` and ``rms_kde_slot`` are the running averages of R² and RMS error given
    reliability for one slot. Their ``*_pooled`` counterparts contain the same curves over every
    (ROI, environment) pair at once. They are precomputed here because a kernel regression per
    session is too slow to redo on every redraw.

    Parameters
    ----------
    session : B2Session
        Session being processed, already switched to the config's ``spks_type``.
    spks : np.ndarray
        Standardized activity, shape ``(frames, rois)``.
    placefield_prediction : np.ndarray
        Place-field prediction, shape ``(frames, rois)``.
    extras : dict
        ``get_placefield_prediction`` extras, supplying ``idx_valid`` and
        ``frame_environment_index`` (positional into ``env_maps.environments``).
    reliability : Reliability
        Per-environment spatial reliability, rows ordered like ``env_maps.environments``.
    env_maps : Maps
        Per-environment maps, used only for its environment list.
    best_env : int
        Positional index of the environment the top-level R² keys describe.
    kde_grid : np.ndarray
        Reliability values the running average is evaluated on.

    Returns
    -------
    dict
        ``r2_slot``, ``rms_slot``, ``norm_rms_slot``, ``var_total_slot``,
        ``var_pred_slot``, ``var_residual_slot``, ``frac_var_pred_slot``, and ``reliability_slot``
        ``(MAX_ENV_SLOTS, rois)``;
        ``r2_kde_slot`` and ``rms_kde_slot`` ``(MAX_ENV_SLOTS, n_kde_grid)``; the corresponding
        ``*_kde_pooled`` arrays ``(n_kde_grid,)``; and ``best_env_slot``, the slot of ``best_env``
        (NaN if it has none).
    """
    n_rois = spks.shape[1]
    mouse_order = load_env_order().get(session.mouse_name)

    r2_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    rms_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    norm_rms_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    var_total_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    var_pred_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    var_residual_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    frac_var_pred_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    reliability_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    r2_kde_slot = np.full((MAX_ENV_SLOTS, kde_grid.size), np.nan)
    rms_kde_slot = np.full((MAX_ENV_SLOTS, kde_grid.size), np.nan)
    best_env_slot = np.nan
    gamma = {}
    if n_gamma_runs:
        for stat in ("mean", "std"):
            for metric in ("r2", "rms", "norm_rms", "cc", "var_total", "var_pred", "var_residual", "frac_var_pred"):
                key = f"gamma_{stat}_{metric}"
                gamma[key] = np.full(n_rois, np.nan)
                gamma[f"{key}_slot"] = np.full((MAX_ENV_SLOTS, n_rois), np.nan)

    for idx_env, env in enumerate(env_maps.environments):
        env = int(env)
        # env < 0 marks invalid trials; a mouse missing from the order map has no slot axis.
        if env < 0 or mouse_order is None or env not in mouse_order:
            continue
        slot = mouse_order.index(env)
        idx_keep = extras["idx_valid"] & (extras["frame_environment_index"] == idx_env)
        if not np.any(idx_keep):
            continue
        r2 = vectorRSquared(placefield_prediction[idx_keep], spks[idx_keep], axis=0)
        r2[r2 < -1] = np.nan
        rms = _rms_error(placefield_prediction[idx_keep], spks[idx_keep])
        norm_rms = _normalized_rms_error(placefield_prediction[idx_keep], spks[idx_keep])
        variance = _variance_components(placefield_prediction[idx_keep], spks[idx_keep])
        relia = reliability.values[idx_env]
        r2_slot[slot] = r2
        rms_slot[slot] = rms
        norm_rms_slot[slot] = norm_rms
        var_total_slot[slot] = variance["var_total"]
        var_pred_slot[slot] = variance["var_pred"]
        var_residual_slot[slot] = variance["var_residual"]
        frac_var_pred_slot[slot] = variance["frac_var_pred"]
        reliability_slot[slot] = relia
        r2_kde_slot[slot] = _kde_r2(r2, relia, kde_grid)["r2_kde_mean"]
        rms_kde_slot[slot] = _kde_rms(rms, relia, kde_grid)["rms_kde_mean"]
        if idx_env == best_env:
            best_env_slot = float(slot)
        if n_gamma_runs:
            rng = np.random.default_rng(np.random.SeedSequence([gamma_seed, slot, 1]))
            indices = np.flatnonzero(idx_keep)
            for start in range(0, n_rois, 128):
                roi = slice(start, start + 128)
                prediction = placefield_prediction[indices, roi]
                valid = np.isfinite(prediction) & np.isfinite(spks[indices, roi])
                shape = np.where(valid, np.clip(prediction, 0, None), 0.0)

                def draws():
                    for _ in range(n_gamma_runs):
                        sim = np.where(valid, rng.standard_gamma(shape), np.nan)
                        r2 = vectorRSquared(prediction, sim, axis=0)
                        yield {
                            "r2": np.where(r2 < -1, np.nan, r2),
                            "rms": _rms_error(prediction, sim),
                            "norm_rms": _normalized_rms_error(prediction, sim),
                            "cc": vectorCorrelation(prediction, sim, axis=0) if len(sim) > 1 else np.full(r2.shape, np.nan),
                            **_variance_components(prediction, sim),
                        }
                for key, values in _gamma_moments(draws()).items():
                    gamma[f"{key}_slot"][slot, roi] = values
                    if idx_env == best_env:
                        gamma[key][roi] = values

    # Every (ROI, environment) pair as its own sample -- not the same as any single slot's curve,
    # since environments differ in how many reliable cells they have.
    r2_pooled = _kde_r2(r2_slot.reshape(-1), reliability_slot.reshape(-1), kde_grid)["r2_kde_mean"]
    rms_pooled = _kde_rms(rms_slot.reshape(-1), reliability_slot.reshape(-1), kde_grid)["rms_kde_mean"]

    return {
        **gamma,
        "r2_slot": r2_slot,
        "rms_slot": rms_slot,
        "norm_rms_slot": norm_rms_slot,
        "var_total_slot": var_total_slot,
        "var_pred_slot": var_pred_slot,
        "var_residual_slot": var_residual_slot,
        "frac_var_pred_slot": frac_var_pred_slot,
        "reliability_slot": reliability_slot,
        "r2_kde_slot": r2_kde_slot,
        "rms_kde_slot": rms_kde_slot,
        "r2_kde_pooled": r2_pooled,
        "rms_kde_pooled": rms_pooled,
        "best_env_slot": best_env_slot,
    }


def _per_roi_stats(
    spks_valid: np.ndarray,
    pfpred_valid: np.ndarray,
    r2: np.ndarray,
    relia: np.ndarray,
    accuracy_pct: float,
) -> dict:
    """Compute per-ROI summaries; undefined Spearman correlations remain NaN.

    Constant, non-finite, or fewer-than-two-sample inputs do not define a temporal
    rank correlation. Check them explicitly to avoid SciPy's ConstantInputWarning.
    """
    n_rois = spks_valid.shape[1]

    act_pct_thresh = np.percentile(spks_valid, accuracy_pct, axis=0)

    spearman_r = np.full(n_rois, np.nan)
    frac_accurate = np.full(n_rois, np.nan)
    for i in range(n_rois):
        a = spks_valid[:, i]
        p = pfpred_valid[:, i]
        if (
            a.size >= 2
            and np.all(np.isfinite(a))
            and np.all(np.isfinite(p))
            and np.any(a != a[0])
            and np.any(p != p[0])
        ):
            spearman_r[i] = spearmanr(a, p).statistic
        frac_accurate[i] = np.mean(np.abs(p - a) < act_pct_thresh[i])

    def _stats(x: np.ndarray, prefix: str) -> dict:
        return {
            f"{prefix}_max": np.nanmax(x, axis=0),
            f"{prefix}_median": np.nanmedian(x, axis=0),
            f"{prefix}_std": np.nanstd(x, axis=0),
            f"{prefix}_skew": skew(x, axis=0, nan_policy="omit"),
            f"{prefix}_kurtosis": kurtosis(x, axis=0, nan_policy="omit"),
            f"{prefix}_frac_zeros": np.mean(x == 0, axis=0),
            f"{prefix}_p95": np.nanpercentile(x, 95, axis=0),
        }

    result = {}
    result.update(_stats(spks_valid, "act"))
    result.update(_stats(pfpred_valid, "pred"))
    result["spearman_r"] = spearman_r
    result["frac_accurate"] = frac_accurate
    return result


def _rms_error(prediction: np.ndarray, activity: np.ndarray) -> np.ndarray:
    """Root-mean-square prediction error for each ROI (axis 0)."""
    return np.sqrt(np.mean((prediction - activity) ** 2, axis=0))


def _normalized_rms_error(prediction: np.ndarray, activity: np.ndarray) -> np.ndarray:
    """Per-ROI RMS error divided by activity standard deviation (axis 0)."""
    rms = _rms_error(prediction, activity)
    activity_std = np.std(activity, axis=0)
    valid_scale = np.isfinite(activity_std) & (activity_std > np.finfo(float).eps)
    return np.divide(rms, activity_std, out=np.full_like(rms, np.nan), where=valid_scale)


def _variance_components(prediction: np.ndarray, activity: np.ndarray) -> dict[str, np.ndarray]:
    """Per-ROI activity, prediction, and residual variance over valid frames.

    ``frac_var_pred`` is the prediction variance divided by activity variance. It is
    undefined for ROIs with zero or non-finite activity variance. The three variances
    need not be exactly additive because prediction and residual can have non-zero
    covariance.
    """
    var_total = np.var(activity, axis=0)
    var_pred = np.var(prediction, axis=0)
    var_residual = np.var(activity - prediction, axis=0)
    valid_scale = np.isfinite(var_total) & (var_total > np.finfo(float).eps)
    frac_var_pred = np.divide(
        var_pred,
        var_total,
        out=np.full_like(var_pred, np.nan),
        where=valid_scale,
    )
    return {
        "var_total": var_total,
        "var_pred": var_pred,
        "var_residual": var_residual,
        "frac_var_pred": frac_var_pred,
    }


def _binned_r2(r2: np.ndarray, relia: np.ndarray, bin_edges: np.ndarray) -> dict:
    """Mean and SEM of R² in each reliability bin."""
    return _binned_metric(r2, relia, bin_edges, prefix="r2")


def _binned_rms(rms: np.ndarray, relia: np.ndarray, bin_edges: np.ndarray) -> dict:
    """Mean and SEM of RMS error in each reliability bin."""
    return _binned_metric(rms, relia, bin_edges, prefix="rms")


def _binned_metric(values: np.ndarray, relia: np.ndarray, bin_edges: np.ndarray, prefix: str) -> dict:
    """Mean, SEM, and sample count of a per-ROI metric in reliability bins."""
    n_bins = len(bin_edges) - 1
    bin_mean = np.full(n_bins, np.nan)
    bin_sem = np.full(n_bins, np.nan)
    bin_n = np.zeros(n_bins, dtype=float)

    bin_idx = np.digitize(relia, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for b in range(n_bins):
        mask = (bin_idx == b) & np.isfinite(values) & np.isfinite(relia)
        vals = values[mask]
        if vals.size > 0:
            bin_mean[b] = np.mean(vals)
            bin_sem[b] = np.std(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            bin_n[b] = vals.size

    return {f"{prefix}_bin_mean": bin_mean, f"{prefix}_bin_sem": bin_sem, f"{prefix}_bin_n": bin_n}


def _kde_r2(r2: np.ndarray, relia: np.ndarray, kde_grid: np.ndarray, bw: float | None = None) -> dict:
    """Kernel regression: E[R² | reliability = x] evaluated on a uniform grid."""
    return _kde_metric(r2, relia, kde_grid, prefix="r2", bw=bw)


def _kde_rms(rms: np.ndarray, relia: np.ndarray, kde_grid: np.ndarray, bw: float | None = None) -> dict:
    """Kernel regression: E[RMS error | reliability = x] on a uniform grid."""
    return _kde_metric(rms, relia, kde_grid, prefix="rms", bw=bw)


def _kde_metric(
    values: np.ndarray,
    relia: np.ndarray,
    kde_grid: np.ndarray,
    prefix: str,
    bw: float | None = None,
) -> dict:
    """Kernel regression of a per-ROI metric against reliability."""
    valid = np.isfinite(values) & np.isfinite(relia)
    metric_values = values[valid]
    reliav = relia[valid]

    if metric_values.size == 0:
        return {f"{prefix}_kde_grid": kde_grid, f"{prefix}_kde_mean": np.full(kde_grid.size, np.nan)}

    if bw is None:
        # Scott's rule
        bw = reliav.std() * reliav.size ** (-0.2)
        bw = max(bw, 0.05)

    kde_mean = np.full(kde_grid.size, np.nan)
    for i, x in enumerate(kde_grid):
        w = np.exp(-0.5 * ((reliav - x) / bw) ** 2)
        w_sum = w.sum()
        if w_sum > 0:
            kde_mean[i] = np.dot(w, metric_values) / w_sum

    return {f"{prefix}_kde_grid": kde_grid, f"{prefix}_kde_mean": kde_mean}
