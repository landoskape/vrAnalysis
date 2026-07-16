"""PFPredQualityConfig — R² between neural activity and place-field prediction.

Measures how well the place-field prediction model captures neural activity,
and how that quality relates to spatial reliability. Stores per-ROI stats
and session-level summaries for mouse-average reliability vs R² curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.stats import spearmanr, skew, kurtosis

from vrAnalysis.helpers import vectorRSquared
from vrAnalysis.processors.spkmaps import SpkmapProcessor, SpkmapParams
from vrAnalysis.processors.support import median_zscore
from vrAnalysis.sessions import B2Session, SpksTypes
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
    """

    schema_version: str = "v2"
    data_config_name: str = "default"
    spks_type: SpksTypes = "sigrebase"
    reliability_threshold: float = 0.7
    accuracy_pct: float = 5.0
    n_hist_bins: int = 40
    n_kde_grid: int = 200

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

            relia = reliability.values[best_env]  # best env, shape (n_rois,)

            result = {"r2": r2, "reliability": relia}
            result.update(_per_roi_stats(spks_valid, pfpred_valid, r2, relia, self.accuracy_pct))

            bin_edges = self.bin_edges
            result.update(_binned_r2(r2, relia, bin_edges))
            result.update(_kde_r2(r2, relia, self.kde_grid))

            idx_reliable = np.isfinite(r2) & (relia > self.reliability_threshold)
            r2_hist_counts, _ = np.histogram(r2[idx_reliable], bins=bin_edges)
            result["r2_hist_counts"] = r2_hist_counts.astype(float)

            return result
        finally:
            session.params.spks_type = prev_spks_type


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


def _binned_r2(r2: np.ndarray, relia: np.ndarray, bin_edges: np.ndarray) -> dict:
    """Mean and SEM of R² in each reliability bin."""
    n_bins = len(bin_edges) - 1
    r2_bin_mean = np.full(n_bins, np.nan)
    r2_bin_sem = np.full(n_bins, np.nan)
    r2_bin_n = np.zeros(n_bins, dtype=float)

    bin_idx = np.digitize(relia, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for b in range(n_bins):
        mask = (bin_idx == b) & np.isfinite(r2)
        vals = r2[mask]
        if vals.size > 0:
            r2_bin_mean[b] = np.mean(vals)
            r2_bin_sem[b] = np.std(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            r2_bin_n[b] = vals.size

    return {"r2_bin_mean": r2_bin_mean, "r2_bin_sem": r2_bin_sem, "r2_bin_n": r2_bin_n}


def _kde_r2(r2: np.ndarray, relia: np.ndarray, kde_grid: np.ndarray, bw: float | None = None) -> dict:
    """Kernel regression: E[R² | reliability = x] evaluated on a uniform grid."""
    valid = np.isfinite(r2) & np.isfinite(relia)
    r2v = r2[valid]
    reliav = relia[valid]

    if bw is None:
        # Scott's rule
        bw = reliav.std() * reliav.size ** (-0.2)
        bw = max(bw, 0.05)

    kde_mean = np.full(kde_grid.size, np.nan)
    for i, x in enumerate(kde_grid):
        w = np.exp(-0.5 * ((reliav - x) / bw) ** 2)
        w_sum = w.sum()
        if w_sum > 0:
            kde_mean[i] = np.dot(w, r2v) / w_sum

    return {"r2_kde_grid": kde_grid, "r2_kde_mean": kde_mean}
