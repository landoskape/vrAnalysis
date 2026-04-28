"""LocPrediction - a config for analyzing estimates of location from activity."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.special import log_softmax
from vrAnalysis.sessions import B2Session
from ..registry import PopulationRegistry
from vrAnalysis.processors.placefields import get_placefield, get_frame_behavior, convert_position_to_bins
from vrAnalysis.processors.placefields import Placefield, FrameBehavior
from vrAnalysis.helpers import cross_validate_trials, reliability_loo
from vrAnalysis.metrics import FractionActive
from ..pipeline.base import AnalysisConfigBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _true_position_bins(frame_behavior: FrameBehavior, placefield: Placefield) -> np.ndarray:
    num_bins = len(placefield.dist_edges) - 1
    env_to_idx = {env: i for i, env in enumerate(placefield.environment)}
    true_env_idx = np.array([env_to_idx[e] for e in frame_behavior.environment])
    true_pos_idx = convert_position_to_bins(frame_behavior.position, placefield.dist_edges, check_invalid=False)
    return true_env_idx * num_bins + true_pos_idx


def _pf_flat(placefield: Placefield) -> np.ndarray:
    """Reshape placefield (num_envs, num_bins, num_rois) → (num_envs*num_bins, num_rois).

    Row order is environment-major: all bins of env0, then env1, etc.
    This must match the flat-bin indexing in _true_position_bins:
        flat_idx = env_idx * num_bins + pos_idx
    """
    pf = placefield.placefield
    assert pf.ndim == 3, f"Expected 3D placefield, got shape {pf.shape}"
    assert pf.shape[0] == len(
        placefield.environment
    ), f"placefield.placefield.shape[0]={pf.shape[0]} != len(environment)={len(placefield.environment)}"
    assert pf.shape[1] == len(placefield.dist_edges) - 1, f"placefield.placefield.shape[1]={pf.shape[1]} != num_bins={len(placefield.dist_edges) - 1}"
    return pf.reshape(-1, pf.shape[2])


def _estimate_residual_variance(
    spks_tr: np.ndarray,
    frame_behavior_tr: FrameBehavior,
    placefield_tr: Placefield,
    min_variance: float = 1e-6,
) -> np.ndarray:
    """Estimate per-ROI variance from training residuals around placefield predictions."""
    train_bins = _true_position_bins(frame_behavior_tr, placefield_tr)
    mu_train = _pf_flat(placefield_tr)[train_bins]  # (frames_tr, rois)
    residuals = spks_tr - mu_train
    variance = np.var(residuals, axis=0, ddof=1)
    return np.maximum(variance, min_variance)


# ---------------------------------------------------------------------------
# Likelihood classes
# ---------------------------------------------------------------------------


@dataclass
class LikelihoodBase:
    name: ClassVar[str]
    eps: float = 1e-9

    def __call__(self, spks: np.ndarray, placefield: Placefield) -> np.ndarray:
        """Returns log-likelihood, shape (frames, num_envs * num_bins)."""
        raise NotImplementedError


@dataclass
class PoissonLikelihood(LikelihoodBase):
    """Poisson/quasi-Poisson decoder for nonnegative activity.

    Score: s_t(x) = Σ_i r_ti·log λ_i(x) - Σ_i λ_i(x)

    The log-factorial term is omitted (independent of x).
    Works as a quasi-likelihood when activity is continuous but nonnegative.
    """

    name: ClassVar[str] = "poisson"

    def __call__(self, spks: np.ndarray, placefield: Placefield) -> np.ndarray:
        lam = np.maximum(_pf_flat(placefield), self.eps)  # (total_bins, rois)
        return spks @ np.log(lam).T - np.sum(lam, axis=1)  # (frames, total_bins)


@dataclass
class GaussianLikelihood(LikelihoodBase):
    """Euclidean template-matching decoder.

    Equivalent to a Gaussian decoder with isotropic noise:
        r_t | x ~ N(μ(x), σ² I)

    Since σ² is constant across neurons and bins, the score is:
        -‖r_t - μ(x)‖²
    """

    name: ClassVar[str] = "gaussian"

    def __call__(self, spks: np.ndarray, placefield: Placefield) -> np.ndarray:
        pf = _pf_flat(placefield)  # (total_bins, rois)
        spks_sq = np.sum(spks**2, axis=1, keepdims=True)  # (frames, 1)
        pf_sq = np.sum(pf**2, axis=1)  # (total_bins,)
        cross = spks @ pf.T  # (frames, total_bins)
        return -(spks_sq + pf_sq - 2 * cross)  # (frames, total_bins)


@dataclass
class DiagonalGaussianLikelihood(LikelihoodBase):
    """Naive Bayes Gaussian decoder with one variance per ROI.

    Treats: r_t | x ~ N(μ(x), diag(σ_i²))

    Score: -0.5 · Σ_i (r_ti - μ_i(x))² / σ_i²

    The log-variance term is omitted (independent of x when variance is per-neuron only).
    Pass variance estimated from training residuals for best results.
    If variance is None, falls back to unit variance (equivalent to GaussianLikelihood up to scale).
    """

    name: ClassVar[str] = "diag_gaussian"
    variance: np.ndarray | None = None
    min_variance: float = 1e-6

    def __call__(self, spks: np.ndarray, placefield: Placefield) -> np.ndarray:
        mu = _pf_flat(placefield)  # (total_bins, rois)

        if self.variance is None:
            inv_var = np.ones(mu.shape[1], dtype=float)
        else:
            inv_var = 1.0 / np.maximum(self.variance, self.min_variance)  # (rois,)

        spks_weighted_sq = np.sum((spks**2) * inv_var[None, :], axis=1, keepdims=True)  # (frames, 1)
        mu_weighted_sq = np.sum((mu**2) * inv_var[None, :], axis=1)  # (total_bins,)
        cross = (spks * inv_var[None, :]) @ mu.T  # (frames, total_bins)
        return -0.5 * (spks_weighted_sq + mu_weighted_sq[None, :] - 2.0 * cross)


@dataclass
class ExponentialMeanLikelihood(LikelihoodBase):
    """Exponential decoder parameterized by mean activity μ.

    For Exp(rate=λ), E[r] = 1/λ, so λ = 1/μ.
    Place fields estimate μ, so we invert to get the rate.

    Score: log p(r|μ) = -log μ - r/μ
    """

    name: ClassVar[str] = "exponential_mean"

    def __call__(self, spks: np.ndarray, placefield: Placefield) -> np.ndarray:
        mu = np.maximum(_pf_flat(placefield), self.eps)  # (total_bins, rois)
        return -np.sum(np.log(mu), axis=1) - spks @ (1.0 / mu).T  # (frames, total_bins)


# ---------------------------------------------------------------------------
# Loss classes
# ---------------------------------------------------------------------------


@dataclass
class LossBase:
    name: ClassVar[str]

    def __call__(
        self,
        log_likelihood: np.ndarray,
        true_bin_idx: np.ndarray,
    ) -> tuple[dict[str, float], np.ndarray]:
        """Returns (scalars dict, per-sample trajectory array)."""
        raise NotImplementedError


@dataclass
class CrossEntropyLoss(LossBase):
    """Softmax cross-entropy treating position bins as classes.

    Optional temperature scaling: higher temperature flattens the distribution
    and can make comparisons fairer across likelihood families with different score scales.
    """

    name: ClassVar[str] = "cross_entropy"
    temperature: float = 1.0

    def __call__(self, log_likelihood: np.ndarray, true_bin_idx: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        log_probs = log_softmax(log_likelihood / self.temperature, axis=1)  # (frames, total_bins)
        traj = -log_probs[np.arange(len(true_bin_idx)), true_bin_idx]  # (frames,)
        return {"loss": float(np.mean(traj))}, traj


@dataclass
class RankOrderingLoss(LossBase):
    """Pairwise surrogate for rank of true bin.

    For each frame, sums g(score_true - score_other) over all non-true bins.

    logistic: g(x) = log(1 + exp(-x))  [numerically stable via logaddexp]
    hinge:    g(x) = max(0, margin - x)

    reduction="mean" divides by (num_bins - 1) so losses are comparable
    across different numbers of environments/bins.
    """

    g: str = "logistic"
    reduction: str = "mean"
    margin: float = 1.0

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"rank_loss_{self.g}_{self.reduction}"

    def __call__(self, log_likelihood: np.ndarray, true_bin_idx: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        n_frames, n_bins = log_likelihood.shape
        true_score = log_likelihood[np.arange(n_frames), true_bin_idx]  # (frames,)
        margin_to_other = true_score[:, None] - log_likelihood  # (frames, total_bins)

        if self.g == "logistic":
            values = np.logaddexp(0.0, -margin_to_other)
        elif self.g == "hinge":
            values = np.maximum(0.0, self.margin - margin_to_other)
        else:
            raise ValueError(f"Unknown g function: {self.g!r}")

        values[np.arange(n_frames), true_bin_idx] = 0.0  # exclude self-comparison

        if self.reduction == "sum":
            traj = np.sum(values, axis=1)
        elif self.reduction == "mean":
            traj = np.sum(values, axis=1) / max(n_bins - 1, 1)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction!r}")

        return {"loss": float(np.mean(traj))}, traj


@dataclass
class RankOrderingMetric(LossBase):
    """Rank of true bin by descending log-likelihood (1 = best).

    Optimistic tie handling: only strictly greater scores beat the true bin.
    Avoids constructing the full rank matrix — O(frames * bins) time and memory.
    """

    name: ClassVar[str] = "rank_metric"

    def __call__(self, log_likelihood: np.ndarray, true_bin_idx: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        n_frames = len(true_bin_idx)
        true_score = log_likelihood[np.arange(n_frames), true_bin_idx]  # (frames,)
        ranks = 1 + np.sum(log_likelihood > true_score[:, None], axis=1)  # (frames,)
        traj = ranks.astype(float)
        scalars = {
            "mean_rank": float(np.mean(traj)),
            "median_rank": float(np.median(traj)),
            "mrr": float(np.mean(1.0 / traj)),
            "top1": float(np.mean(traj <= 1)),
            "top5": float(np.mean(traj <= 5)),
            "top10": float(np.mean(traj <= 10)),
        }
        return scalars, traj


@dataclass
class DistanceErrorLoss(LossBase):
    """Absolute position error (in track units) when predicted env matches true env.

    Frames where environments differ are NaN in the trajectory.
    """

    name: ClassVar[str] = "distance_error"
    num_bins: int = 100
    dist_edges: np.ndarray | None = None

    def __call__(self, log_likelihood: np.ndarray, true_bin_idx: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        if self.dist_edges is None:
            raise ValueError("dist_edges must be provided.")
        pred_bin_idx = np.argmax(log_likelihood, axis=1)  # (frames,)
        pred_env = pred_bin_idx // self.num_bins
        pred_pos = pred_bin_idx % self.num_bins
        true_env = true_bin_idx // self.num_bins
        true_pos = true_bin_idx % self.num_bins
        bin_centers = (self.dist_edges[:-1] + self.dist_edges[1:]) / 2.0
        same_env = pred_env == true_env
        traj = np.where(same_env, np.abs(bin_centers[pred_pos] - bin_centers[true_pos]), np.nan)
        return {"loss": float(np.nanmean(traj))}, traj


@dataclass
class EnvSwapFraction(LossBase):
    """Fraction of frames where predicted environment differs from true environment."""

    name: ClassVar[str] = "env_swap"
    num_bins: int = 100

    def __call__(self, log_likelihood: np.ndarray, true_bin_idx: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        pred_bin_idx = np.argmax(log_likelihood, axis=1)
        pred_env = pred_bin_idx // self.num_bins
        true_env = true_bin_idx // self.num_bins
        traj = (pred_env != true_env).astype(float)
        return {"fraction": float(np.mean(traj))}, traj


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

_LIKELIHOOD_REGISTRY: dict[str, type[LikelihoodBase]] = {
    "poisson": PoissonLikelihood,
    "gaussian": GaussianLikelihood,
    "exponential_mean": ExponentialMeanLikelihood,
    # diag_gaussian is handled specially in _get_likelihood_methods (needs fitted variance)
}

_LOSS_REGISTRY: dict[str, Callable[..., LossBase]] = {
    "cross_entropy": CrossEntropyLoss,
    "rank_loss_logistic_mean": lambda: RankOrderingLoss(g="logistic", reduction="mean"),
    "rank_loss_logistic_sum": lambda: RankOrderingLoss(g="logistic", reduction="sum"),
    "rank_loss_hinge_mean": lambda: RankOrderingLoss(g="hinge", reduction="mean"),
    "rank_loss_hinge_sum": lambda: RankOrderingLoss(g="hinge", reduction="sum"),
    "rank_metric": RankOrderingMetric,
    "distance_error": DistanceErrorLoss,
    "env_swap": EnvSwapFraction,
}


def _get_likelihood_methods(
    names: tuple[str, ...],
    diag_gaussian_variance: np.ndarray | None = None,
) -> dict[str, LikelihoodBase]:
    out: dict[str, LikelihoodBase] = {}
    for name in names:
        if name == "diag_gaussian":
            out[name] = DiagonalGaussianLikelihood(variance=diag_gaussian_variance)
            continue
        if name not in _LIKELIHOOD_REGISTRY:
            raise ValueError(f"Unknown likelihood method: {name!r}. Valid: {list(_LIKELIHOOD_REGISTRY) + ['diag_gaussian']}")
        out[name] = _LIKELIHOOD_REGISTRY[name]()
    return out


def _get_loss_methods(names: tuple[str, ...], num_bins: int, dist_edges: np.ndarray) -> dict[str, LossBase]:
    out: dict[str, LossBase] = {}
    for name in names:
        if name not in _LOSS_REGISTRY:
            raise ValueError(f"Unknown loss method: {name!r}. Valid: {list(_LOSS_REGISTRY)}")
        if name == "distance_error":
            out[name] = _LOSS_REGISTRY[name](num_bins=num_bins, dist_edges=dist_edges)
        elif name == "env_swap":
            out[name] = _LOSS_REGISTRY[name](num_bins=num_bins)
        else:
            out[name] = _LOSS_REGISTRY[name]()
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocPredConfig(AnalysisConfigBase):
    """Configuration for estimating position from neural activity and placefields.

    Parameters
    ----------
    norm_method : str
        Normalization applied to spike counts using only training statistics.
        "zero-one": divide by per-cell max of training data.
        "none": no normalization.
    norm_scale : float
        Optional global scale applied after normalization.
    speed_threshold : float
        Minimum speed for including timepoints.
    num_bins : int
        Number of spatial bins for place field computation.
    train_test_split : tuple[float, float]
        Train/test proportions; must sum to 1.
    smooth_width : float or None
        Gaussian smoothing width for place fields.
    reliability_cutoff : float
        Minimum leave-one-out reliability for ROI inclusion.
    fraction_active_cutoff : float
        Minimum participation fraction for ROI inclusion.
    likelihood_methods : tuple[str, ...]
        Likelihood functions to evaluate.
        Options: "poisson", "gaussian", "diag_gaussian", "exponential_mean".
    loss_methods : tuple[str, ...]
        Loss/metric functions to evaluate.
        Options: "cross_entropy", "rank_loss_logistic_mean", "rank_loss_logistic_sum",
        "rank_loss_hinge_mean", "rank_loss_hinge_sum", "rank_metric",
        "distance_error", "env_swap".
    """

    schema_version: str = "v1"
    data_config_name: str = "default"

    norm_method: str = "zero-one"
    norm_scale: float = 1.0
    speed_threshold: float = 1.0
    num_bins: int = 100
    train_test_split: tuple[float, float] = (0.8, 0.2)
    smooth_width: float | None = 0.25
    reliability_cutoff: float = 0.1
    fraction_active_cutoff: float = 0.1
    likelihood_methods: tuple[str, ...] = ("poisson", "gaussian", "diag_gaussian")
    loss_methods: tuple[str, ...] = ("cross_entropy", "rank_loss_logistic_mean", "rank_metric", "distance_error", "env_swap")
    display_name: ClassVar[str] = "locprediction"

    @staticmethod
    def _param_grid() -> dict:
        return {}

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"norm_method={self.norm_method}",
            f"norm_scale={self.norm_scale}",
            f"speed_threshold={self.speed_threshold}",
            f"num_bins={self.num_bins}",
            f"train_test_split={self.train_test_split}",
            f"smooth_width={self.smooth_width}",
            f"reliability_cutoff={self.reliability_cutoff}",
            f"fraction_active_cutoff={self.fraction_active_cutoff}",
            f"likelihood_methods={','.join(self.likelihood_methods)}",
            f"loss_methods={','.join(self.loss_methods)}",
            self.schema_version,
        ]
        return "_".join(parts)

    def _select_rois(self, session: B2Session, spks: np.ndarray, frame_behavior: FrameBehavior, dist_edges: np.ndarray) -> np.ndarray:
        """Select reliable and active ROIs via leave-one-out reliability and fraction active."""
        _all_trials = get_placefield(
            spks,
            frame_behavior,
            dist_edges,
            average=False,
            use_fast_sampling=True,
            session=session,
        )
        pf_data = np.transpose(_all_trials.placefield, (2, 0, 1))
        idx_reliable = reliability_loo(pf_data) >= self.reliability_cutoff
        fraction_active = (
            FractionActive.compute(
                pf_data,
                activity_axis=2,
                fraction_axis=1,
                activity_method="rms",
                fraction_method="participation",
            )
            >= self.fraction_active_cutoff
        )
        return idx_reliable & fraction_active

    def _normalize_spks_train_test(
        self,
        spks_tr: np.ndarray,
        spks_te: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize using only training statistics, then apply the same transform to test."""
        if self.norm_method == "zero-one":
            norm_value = np.max(spks_tr, axis=0)
            norm_value = np.maximum(norm_value, 1e-12)  # guard divide-by-zero
            spks_tr = spks_tr / norm_value
            spks_te = spks_te / norm_value
        elif self.norm_method == "none":
            pass
        else:
            raise ValueError(f"Unknown norm_method: {self.norm_method!r}")

        if self.norm_scale != 1.0:
            spks_tr = spks_tr * self.norm_scale
            spks_te = spks_te * self.norm_scale

        return spks_tr, spks_te

    def process(self, session: B2Session, registry: PopulationRegistry, verbose: bool = False) -> dict:
        """Run Location Prediction analysis on a session."""
        frame_behavior = get_frame_behavior(session)
        idx_valid = frame_behavior.valid_frames()
        idx_fast = frame_behavior.speed >= self.speed_threshold
        idx_filter = idx_valid & idx_fast
        frame_behavior = frame_behavior.filter(idx_filter)

        trial_folds = cross_validate_trials(session.trial_environment, self.train_test_split)
        idx_train = np.isin(frame_behavior.trial, trial_folds[0])
        idx_test = np.isin(frame_behavior.trial, trial_folds[1])

        frame_behavior_tr = frame_behavior.filter(idx_train)
        frame_behavior_te = frame_behavior.filter(idx_test)

        spks = session.spks[idx_filter][:, session.idx_rois]
        spks_tr = spks[idx_train]
        spks_te = spks[idx_test]

        # Normalize using training statistics only
        spks_tr, spks_te = self._normalize_spks_train_test(spks_tr, spks_te)

        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)
        idx_keep_rois = self._select_rois(session, spks_tr, frame_behavior_tr, dist_edges)
        spks_tr = spks_tr[:, idx_keep_rois]
        spks_te = spks_te[:, idx_keep_rois]

        # Build train placefield
        placefield_tr = get_placefield(
            spks_tr,
            frame_behavior_tr,
            dist_edges,
            average=True,
            smooth_width=self.smooth_width,
            use_fast_sampling=True,
            session=session,
        )

        # Drop test frames whose environment didn't appear in training data.
        # cross_validate_trials stratifies, so this is rare but possible with few trials per env.
        idx_known_env_te = np.isin(frame_behavior_te.environment, placefield_tr.environment)
        frame_behavior_te = frame_behavior_te.filter(idx_known_env_te)
        spks_te = spks_te[idx_known_env_te]

        # True flat bin indices for test frames
        true_bins_te = _true_position_bins(frame_behavior_te, placefield_tr)

        # Estimate per-ROI residual variance from training data (for diag_gaussian)
        diag_gaussian_variance = _estimate_residual_variance(spks_tr, frame_behavior_tr, placefield_tr)

        # Build dispatchers
        lik_methods = _get_likelihood_methods(self.likelihood_methods, diag_gaussian_variance=diag_gaussian_variance)
        loss_methods = _get_loss_methods(self.loss_methods, self.num_bins, dist_edges)

        likelihood_matrix: dict[str, np.ndarray] = {}
        loss_trajectory: dict[str, np.ndarray] = {}
        loss_scalar: dict[str, float] = {}

        for lik_name, lik_fn in lik_methods.items():
            ll = lik_fn(spks_te, placefield_tr)  # (frames_te, total_bins)
            likelihood_matrix[lik_name] = ll

            for loss_name, loss_fn in loss_methods.items():
                scalars, traj = loss_fn(ll, true_bins_te)
                combo = f"{lik_name}_{loss_name}"
                loss_trajectory[combo] = traj
                for k, v in scalars.items():
                    # Single-scalar keys: keep combo name; multi-scalar: append sub-key
                    key = combo if k in ("loss", "fraction") else f"{combo}_{k}"
                    loss_scalar[key] = v

        return dict(
            likelihood_matrix=likelihood_matrix,
            loss_trajectory=loss_trajectory,
            loss_scalar=loss_scalar,
            true_position_bins_te=true_bins_te,
            idx_keep_rois=idx_keep_rois,
        )
