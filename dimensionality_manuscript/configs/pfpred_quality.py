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
``rms``, ``reliability``, ``r2_kde_mean``, ``rms_kde_mean``, ...) describe the
session's *best* environment, while the ``*_slot`` keys repeat the measurement
for every environment on that same experience-order slot axis, so prediction
quality can be followed per environment as a mouse gets more familiar with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.stats import spearmanr, skew, kurtosis

from vrAnalysis.helpers import vectorRSquared
from vrAnalysis.helpers.signals import vectorCorrelation
from vrAnalysis.processors.placefields import get_frame_behavior, get_placefield
from vrAnalysis.processors.spkmaps import SpkmapProcessor, SpkmapParams
from vrAnalysis.processors.support import median_zscore
from vrAnalysis.sessions import B2Session, SpksTypes
from ..env_order import MAX_ENV_SLOTS, load_env_order
from ..pipeline.base import AnalysisConfigBase
from ..registry import PopulationRegistry

VALID_SPKS_TYPES: list[SpksTypes] = ["oasis", "sigrebase"]


@dataclass(frozen=True)
class PFPredQualityConfig(AnalysisConfigBase):
    """Configuration for place-field prediction quality analysis.

    Parameters
    ----------
    spks_type : SpksTypes
        Spike type to use for the place-field prediction.
    reliability_threshold : float
        Reliability cutoff for the R² histogram of reliable ROIs.
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
    """

    # v4: added the per-experience-slot place-field peak amplitude keys (pf_peak,
    # pf_peak_hist, pf_peak_hist_edges, pf_peak_n, env_slot_ids).
    # v5: added the per-experience-slot R² keys (r2_slot, reliability_slot, r2_kde_slot,
    # r2_kde_pooled, best_env_slot), so R² vs reliability can be tracked per environment.
    # v6: added per-ROI RMS error and its binned, KDE, and per-slot counterparts.
    schema_version: str = "v6"
    data_config_name: str = "default"
    spks_type: SpksTypes = "sigrebase"
    reliability_threshold: float = 0.7
    accuracy_pct: float = 5.0
    n_hist_bins: int = 40
    n_kde_grid: int = 200
    peak_hist_max: float = 10.0
    n_peak_hist_bins: int = 100

    display_name: ClassVar[str] = "pfpred_quality"

    @staticmethod
    def _param_grid() -> dict:
        return {}

    def validate(self):
        if self.spks_type not in VALID_SPKS_TYPES:
            raise ValueError(f"Unknown spks_type {self.spks_type!r}. Available: {VALID_SPKS_TYPES}")

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
        return f"{self.display_name}_spks={self.spks_type}_{self.schema_version}"

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

            relia = reliability.values[best_env]  # best env, shape (n_rois,)

            result = {"r2": r2, "cc": cc, "rms": rms, "reliability": relia}
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
            result.update(_r2_by_slot(session, spks, placefield_prediction, extras, reliability, env_maps, best_env, self.kde_grid))

            return result
        finally:
            session.params.spks_type = prev_spks_type


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
        ``r2_slot``, ``rms_slot``, and ``reliability_slot`` ``(MAX_ENV_SLOTS, rois)``;
        ``r2_kde_slot`` and ``rms_kde_slot`` ``(MAX_ENV_SLOTS, n_kde_grid)``; the corresponding
        ``*_kde_pooled`` arrays ``(n_kde_grid,)``; and ``best_env_slot``, the slot of ``best_env``
        (NaN if it has none).
    """
    n_rois = spks.shape[1]
    mouse_order = load_env_order().get(session.mouse_name)

    r2_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    rms_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    reliability_slot = np.full((MAX_ENV_SLOTS, n_rois), np.nan)
    r2_kde_slot = np.full((MAX_ENV_SLOTS, kde_grid.size), np.nan)
    rms_kde_slot = np.full((MAX_ENV_SLOTS, kde_grid.size), np.nan)
    best_env_slot = np.nan

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
        relia = reliability.values[idx_env]
        r2_slot[slot] = r2
        rms_slot[slot] = rms
        reliability_slot[slot] = relia
        r2_kde_slot[slot] = _kde_r2(r2, relia, kde_grid)["r2_kde_mean"]
        rms_kde_slot[slot] = _kde_rms(rms, relia, kde_grid)["rms_kde_mean"]
        if idx_env == best_env:
            best_env_slot = float(slot)

    # Every (ROI, environment) pair as its own sample -- not the same as any single slot's curve,
    # since environments differ in how many reliable cells they have.
    r2_pooled = _kde_r2(r2_slot.reshape(-1), reliability_slot.reshape(-1), kde_grid)["r2_kde_mean"]
    rms_pooled = _kde_rms(rms_slot.reshape(-1), reliability_slot.reshape(-1), kde_grid)["rms_kde_mean"]

    return {
        "r2_slot": r2_slot,
        "rms_slot": rms_slot,
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
    """Compute per-ROI summary statistics."""
    n_rois = spks_valid.shape[1]

    act_pct_thresh = np.percentile(spks_valid, accuracy_pct, axis=0)

    spearman_r = np.full(n_rois, np.nan)
    frac_accurate = np.full(n_rois, np.nan)
    for i in range(n_rois):
        a = spks_valid[:, i]
        p = pfpred_valid[:, i]
        if np.any(np.isfinite(a)) and np.any(np.isfinite(p)):
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
