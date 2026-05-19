"""LocPrediction - a config for analyzing estimates of location from activity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Callable
from functools import partial
import numpy as np
import torch
from scipy.special import log_softmax
from vrAnalysis.sessions import B2Session, SpksTypes
from ..registry import PopulationRegistry
from vrAnalysis.processors.placefields import get_placefield, convert_position_to_bins
from vrAnalysis.processors.placefields import Placefield, FrameBehavior
from vrAnalysis.helpers import reliability_loo
from vrAnalysis.metrics import FractionActive
from dimilibi import measure_r2
from dimilibi.helpers import VectorizedGoldenSectionSearch
from ..pipeline.base import AnalysisConfigBase

if TYPE_CHECKING:
    from ..registry import SplitName


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


def _hard_e_step(
    spks: np.ndarray,
    placefield: Placefield,
    likelihood_fn: "LikelihoodBase",
    dist_edges: np.ndarray,
    frame_behavior: FrameBehavior,
) -> tuple[FrameBehavior, np.ndarray]:
    """Assign positions by argmax likelihood. Returns (new FrameBehavior, flat bin indices)."""
    ll = likelihood_fn(spks, placefield)  # (frames, total_bins)
    pred_bin_idx = np.argmax(ll, axis=1)  # (frames,)
    num_bins = len(dist_edges) - 1
    pred_env_idx = pred_bin_idx // num_bins
    pred_pos_idx = pred_bin_idx % num_bins
    bin_centers = (dist_edges[:-1] + dist_edges[1:]) / 2.0
    new_fb = FrameBehavior(
        position=bin_centers[pred_pos_idx],
        speed=frame_behavior.speed,
        environment=placefield.environment[pred_env_idx],
        trial=frame_behavior.trial,
        reward_delivery=frame_behavior.reward_delivery,
        reward_omitted=frame_behavior.reward_omitted,
        idx=frame_behavior.idx,
    )
    return new_fb, pred_bin_idx


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
class VonMisesFisherLikelihood(LikelihoodBase):
    """Cosine similarity decoder for directional tuning.

    Equivalent to a von Mises-Fisher decoder with fixed concentration κ:
        r_t | x ~ vMF(μ(x), κ)

    Score: s_t(x) = κ · (r_t / ‖r_t‖) · (μ(x) / ‖μ(x)‖)

    Since κ is constant across bins, it only scales the scores and doesn't change the rank order.
    Note: for validation, in principle we could optimize κ on the validation set, but in practice
    it just scales the scores and doesn't change the rank order, so we can absorb it into the
    temperature hyperparameter of the loss functions and therefore simplify the optimization system.
    """

    name: ClassVar[str] = "von_mises_fisher"
    kappa: float = 1.0

    def __call__(self, spks: np.ndarray, placefield: Placefield) -> np.ndarray:
        pf = _pf_flat(placefield)  # (total_bins, rois)
        spks_norm = np.linalg.norm(spks, axis=1, keepdims=True)  # (frames, 1)
        pf_norm = np.linalg.norm(pf, axis=1, keepdims=True)  # (total_bins, 1)
        spks_unit = spks / np.maximum(spks_norm, self.eps)
        pf_unit = pf / np.maximum(pf_norm, self.eps)
        return self.kappa * (spks_unit @ pf_unit.T)  # (frames, total_bins)


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


# ---------------------------------------------------------------------------
# Loss classes
# ---------------------------------------------------------------------------


@dataclass
class LossBase:
    name: ClassVar[str]
    has_hyperparameter: ClassVar[bool] = False
    hyperparameter_name: ClassVar[str | None] = None
    optimize_target: ClassVar[str | None] = None
    optimize_direction: ClassVar[str | None] = None
    optimize_init_range: ClassVar[tuple[float, float] | None] = None

    def __call__(
        self,
        log_likelihood: np.ndarray,
        true_bin_idx: np.ndarray,
        **kwargs,
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
    has_hyperparameter: ClassVar[bool] = True
    hyperparameter_name: ClassVar[str | None] = "temperature"
    optimize_target: ClassVar[str | None] = "loss"
    optimize_direction: ClassVar[str | None] = "minimize"
    optimize_init_range: ClassVar[tuple[float, float] | None] = (0.01, 100.0)
    temperature: float = 1.0

    def __call__(
        self,
        log_likelihood: np.ndarray,
        true_bin_idx: np.ndarray,
        *,
        temperature: float | None = None,
        **kwargs,
    ) -> tuple[dict[str, float], np.ndarray]:
        t = temperature if temperature is not None else self.temperature
        log_probs = log_softmax(log_likelihood / t, axis=1)  # (frames, total_bins)
        traj = -log_probs[np.arange(len(true_bin_idx)), true_bin_idx]  # (frames,)
        return {"loss": float(np.mean(traj))}, traj


@dataclass
class RankOrderingLoss(LossBase):
    """Pairwise surrogate for rank of true bin.

    For each frame, sums g(score_true - score_other) over all non-true bins.

    logistic: g(x) = log(1 + exp(-x / T))  [numerically stable via logaddexp]
    hinge:    g(x) = max(0, margin - x / T)

    reduction="mean" divides by (num_bins - 1) so losses are comparable
    across different numbers of environments/bins.
    """

    has_hyperparameter: ClassVar[bool] = True
    hyperparameter_name: ClassVar[str | None] = "temperature"
    optimize_target: ClassVar[str | None] = "loss"
    optimize_direction: ClassVar[str | None] = "minimize"
    optimize_init_range: ClassVar[tuple[float, float] | None] = (0.01, 100.0)
    temperature: float = 1.0
    g: str = "logistic"
    margin: float = 1.0
    reduction: str = "mean"

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"rank_loss_{self.g}_{self.reduction}"

    def __call__(
        self,
        log_likelihood: np.ndarray,
        true_bin_idx: np.ndarray,
        *,
        temperature: float | None = None,
        **kwargs,
    ) -> tuple[dict[str, float], np.ndarray]:
        n_frames, n_bins = log_likelihood.shape
        t = temperature if temperature is not None else self.temperature
        true_score = log_likelihood[np.arange(n_frames), true_bin_idx]  # (frames,)
        margin_to_other = (true_score[:, None] - log_likelihood) / t  # (frames, total_bins)

        if self.g == "logistic":
            values = np.logaddexp(0.0, -margin_to_other)  # (frames, total_bins)
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

    def __call__(
        self,
        log_likelihood: np.ndarray,
        true_bin_idx: np.ndarray,
        **kwargs,
    ) -> tuple[dict[str, float], np.ndarray]:
        n_frames = len(true_bin_idx)
        true_score = log_likelihood[np.arange(n_frames), true_bin_idx]  # (frames,)
        ranks = 1 + np.sum(log_likelihood > true_score[:, None], axis=1)  # (frames,)
        traj = ranks.astype(float)
        scalars = {
            "mean_rank": float(np.mean(traj)),
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
    dist_edges: np.ndarray

    def __call__(self, log_likelihood: np.ndarray, true_bin_idx: np.ndarray, **kwargs) -> tuple[dict[str, float], np.ndarray]:
        num_bins = len(self.dist_edges) - 1
        pred_bin_idx = np.argmax(log_likelihood, axis=1)  # (frames,)
        pred_env = pred_bin_idx // num_bins
        pred_pos = pred_bin_idx % num_bins
        true_env = true_bin_idx // num_bins
        true_pos = true_bin_idx % num_bins
        bin_centers = (self.dist_edges[:-1] + self.dist_edges[1:]) / 2.0
        same_env = pred_env == true_env
        traj = np.where(same_env, np.abs(bin_centers[pred_pos] - bin_centers[true_pos]), np.nan)
        return {"loss": float(np.nanmean(traj))}, traj


@dataclass
class EnvSwapFraction(LossBase):
    """Fraction of frames where predicted environment differs from true environment."""

    name: ClassVar[str] = "env_swap"
    dist_edges: np.ndarray

    def __call__(self, log_likelihood: np.ndarray, true_bin_idx: np.ndarray, **kwargs) -> tuple[dict[str, float], np.ndarray]:
        num_bins = len(self.dist_edges) - 1
        pred_bin_idx = np.argmax(log_likelihood, axis=1)
        pred_env = pred_bin_idx // num_bins
        true_env = true_bin_idx // num_bins
        traj = (pred_env != true_env).astype(float)
        return {"fraction": float(np.mean(traj))}, traj


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

_LIKELIHOOD_REGISTRY: dict[str, type[LikelihoodBase]] = {
    "poisson": PoissonLikelihood,
    "gaussian": GaussianLikelihood,
    "von_mises_fisher": VonMisesFisherLikelihood,
    # diag_gaussian is handled specially in _get_likelihood_methods (needs fitted variance)
}

_LOSS_REGISTRY: dict[str, Callable[..., LossBase]] = {
    "cross_entropy": CrossEntropyLoss,
    "rank_loss_logistic_mean": partial(RankOrderingLoss, g="logistic", reduction="mean"),
    "rank_loss_logistic_sum": partial(RankOrderingLoss, g="logistic", reduction="sum"),
    "rank_loss_hinge_mean": partial(RankOrderingLoss, g="hinge", reduction="mean"),
    "rank_loss_hinge_sum": partial(RankOrderingLoss, g="hinge", reduction="sum"),
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


_NEEDS_DIST_EDGES = {"distance_error", "env_swap"}


def _get_loss_methods(
    names: tuple[str, ...],
    dist_edges: np.ndarray,
    only_with_hyperparameters: bool = False,
) -> dict[str, LossBase]:
    out: dict[str, LossBase] = {}
    for name in names:
        if name not in _LOSS_REGISTRY:
            raise ValueError(f"Unknown loss method: {name!r}. Valid: {list(_LOSS_REGISTRY)}")
        cls = _LOSS_REGISTRY[name]
        out[name] = cls(dist_edges=dist_edges) if name in _NEEDS_DIST_EDGES else cls()
    if only_with_hyperparameters:
        out = {key: v for key, v in out.items() if v.has_hyperparameter}
    return out


# ---------------------------------------------------------------------------
# Optimization Helpers
# ---------------------------------------------------------------------------
def _optimize(
    loss_fn: LossBase,
    ll: np.ndarray,
    true_bins_vl: np.ndarray,
) -> float:
    # Check the loss_fn provided is a valid optimizable loss
    assert loss_fn.optimize_init_range is not None, f"{loss_fn.name} has no optimize_init_range"
    assert loss_fn.optimize_target is not None
    assert loss_fn.hyperparameter_name is not None
    target = loss_fn.optimize_target
    direction = loss_fn.optimize_direction
    init_range = loss_fn.optimize_init_range
    hyperparameter_name = loss_fn.hyperparameter_name
    gss = VectorizedGoldenSectionSearch(init_range[0], init_range[1])

    def objective(points: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        val = loss_fn(ll, true_bins_vl, **{hyperparameter_name: points[0].item()})[0][target]
        return torch.tensor([val], dtype=torch.float32)

    return gss.run(objective, verbose=False, maximize=(direction == "maximize"))[0].item()


# ---------------------------------------------------------------------------
# Shared ROI selection and data helpers
# ---------------------------------------------------------------------------


def _select_rois_fn(
    spks: np.ndarray,
    frame_behavior: FrameBehavior,
    dist_edges: np.ndarray,
    session: B2Session,
    reliability_cutoff: float,
    fraction_active_cutoff: float,
) -> np.ndarray:
    """Select reliable and active ROIs via leave-one-out reliability and fraction active.

    Parameters
    ----------
    spks : np.ndarray
        Activity array of shape (frames, rois).
    frame_behavior : FrameBehavior
    dist_edges : np.ndarray
        Spatial bin edges, shape (num_bins + 1,).
    session : B2Session
    reliability_cutoff : float
        Minimum leave-one-out reliability score.
    fraction_active_cutoff : float
        Minimum participation fraction.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (rois,).
    """
    _all_trials = get_placefield(
        spks,
        frame_behavior,
        dist_edges,
        average=False,
        use_fast_sampling=True,
        session=session,
    )
    pf_data = np.transpose(_all_trials.placefield, (2, 0, 1))
    idx_reliable = reliability_loo(pf_data) >= reliability_cutoff
    fraction_active = (
        FractionActive.compute(
            pf_data,
            activity_axis=2,
            fraction_axis=1,
            activity_method="rms",
            fraction_method="participation",
        )
        >= fraction_active_cutoff
    )
    return idx_reliable & fraction_active


def _get_source_target_arrays(
    session: B2Session,
    registry: "PopulationRegistry",
    spks_type: SpksTypes,
    split: "SplitName",
) -> tuple[np.ndarray, np.ndarray, FrameBehavior]:
    """Get source and target spike arrays for a registry split via the preset cell split.

    Parameters
    ----------
    session : B2Session
    registry : PopulationRegistry
    spks_type : SpksTypes
    split : SplitName

    Returns
    -------
    spks_source : np.ndarray
        Shape (frames, source_rois).
    spks_target : np.ndarray
        Shape (frames, target_rois).
    frame_behavior : FrameBehavior
        Filtered to the requested split.
    """
    population, frame_behavior = registry.get_population(session, spks_type)
    split_idx = registry.time_split[split]
    source_data, target_data = population.get_split_data(split_idx, center=False, scale=False)
    idx_orig = np.array(population.get_split_times(split_idx, within_idx_samples=False))
    spks_source = source_data.numpy().T  # (frames, source_rois)
    spks_target = target_data.numpy().T  # (frames, target_rois)
    return spks_source, spks_target, frame_behavior.filter(idx_orig)


# ---------------------------------------------------------------------------
# Fitted state
# ---------------------------------------------------------------------------


@dataclass
class LocPredFit:
    """Fitted state from LocPredConfig.fit().

    Parameters
    ----------
    placefield : Placefield
        Averaged place field from the last iteration (= iteration_placefields[-1]).
    diag_gaussian_variance : np.ndarray
        Per-ROI residual variance from the last iteration (= iteration_diag_variances[-1]).
    idx_keep_rois : np.ndarray
        Boolean mask over neurons in the population selecting reliable, active ROIs.
    dist_edges : np.ndarray
        Spatial bin edges. Shape (num_bins + 1,).
    norm_value : np.ndarray or None
        Per-ROI normalization constants from training data. None when norm_method="none".
    iteration_placefields : list[Placefield]
        Placefield from each EM iteration. Length = fit_iterations.
    iteration_diag_variances : list[np.ndarray]
        Per-ROI residual variance from each EM iteration. Length = fit_iterations.
    iteration_position_bins_tr : list[np.ndarray]
        Flat bin assignments on training data after each iteration. Length = fit_iterations.
    """

    placefield: Placefield
    diag_gaussian_variance: np.ndarray
    idx_keep_rois: np.ndarray
    dist_edges: np.ndarray
    norm_value: np.ndarray | None
    iteration_placefields: list[Placefield]
    iteration_diag_variances: list[np.ndarray]
    iteration_position_bins_tr: list[np.ndarray]


@dataclass
class LocPredCrossValFit:
    """Fitted state from LocPredCrossVal.fit().

    Parameters
    ----------
    pf_source : Placefield
        Place field built from source neurons on training data.
    pf_target : Placefield
        Place field built from target neurons on training data.
    diag_gaussian_variance : np.ndarray
        Per-source-ROI residual variance for diagonal-Gaussian decoding. Shape (n_source_kept,).
    idx_keep_source : np.ndarray
        Boolean mask over source neurons selecting reliable, active ROIs.
    idx_keep_target : np.ndarray
        Boolean mask over target neurons selecting reliable, active ROIs.
    dist_edges : np.ndarray
        Spatial bin edges. Shape (num_bins + 1,).
    norm_value : np.ndarray or None
        Per-ROI normalization constants from source training data. None when norm_method="none".
    norm_value_target : np.ndarray or None
        Per-ROI normalization constants from target training data. None when norm_method="none".
    """

    pf_source: Placefield
    pf_target: Placefield
    diag_gaussian_variance: np.ndarray
    idx_keep_source: np.ndarray
    idx_keep_target: np.ndarray
    dist_edges: np.ndarray
    norm_value: np.ndarray | None
    norm_value_target: np.ndarray | None


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
    num_bins : int
        Number of spatial bins for place field computation.
    smooth_width : float or None
        Gaussian smoothing width for place fields.
    reliability_cutoff : float
        Minimum leave-one-out reliability for ROI inclusion.
    fraction_active_cutoff : float
        Minimum participation fraction for ROI inclusion.
    spks_type : SpksTypes
        Spike type to retrieve from the registry.
    likelihood_methods : tuple[str, ...]
        Likelihood functions to evaluate.
        Options: "poisson", "gaussian", "diag_gaussian", "von_mises_fisher".
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
    num_bins: int = 100
    smooth_width: float | None = 0.25
    reliability_cutoff: float = 0.1
    fraction_active_cutoff: float = 0.1
    spks_type: SpksTypes = "oasis"
    likelihood_methods: tuple[str, ...] = ("poisson", "gaussian", "diag_gaussian", "von_mises_fisher")
    loss_methods: tuple[str, ...] = ("cross_entropy", "rank_loss_logistic_mean", "rank_loss_hinge_mean", "rank_metric", "distance_error", "env_swap")
    fit_iterations: int = 4
    predict_likelihood: str = "diag_gaussian"
    display_name: ClassVar[str] = "locprediction"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "norm_scale": [1.0, 50.0],
            "smooth_width": [None, 5.0],
        }

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"norm_method={self.norm_method}",
            f"norm_scale={self.norm_scale}",
            f"num_bins={self.num_bins}",
            f"smooth_width={self.smooth_width}",
            f"reliability_cutoff={self.reliability_cutoff}",
            f"fraction_active_cutoff={self.fraction_active_cutoff}",
            f"spks_type={self.spks_type}",
            f"likelihood_methods={','.join(self.likelihood_methods)}",
            f"loss_methods={','.join(self.loss_methods)}",
            f"fit_iterations={self.fit_iterations}",
            f"predict_likelihood={self.predict_likelihood}",
            self.schema_version,
        ]
        return "_".join(parts)

    def _select_rois(self, session: B2Session, spks: np.ndarray, frame_behavior: FrameBehavior, dist_edges: np.ndarray) -> np.ndarray:
        """Select reliable and active ROIs via leave-one-out reliability and fraction active."""
        return _select_rois_fn(spks, frame_behavior, dist_edges, session, self.reliability_cutoff, self.fraction_active_cutoff)

    def _compute_norm_value(self, spks_tr: np.ndarray) -> np.ndarray | None:
        if self.norm_method == "zero-one":
            return np.maximum(np.max(spks_tr, axis=0), 1e-12)
        elif self.norm_method == "none":
            return None
        else:
            raise ValueError(f"Unknown norm_method: {self.norm_method!r}")

    def _apply_norm(self, spks: np.ndarray, norm_value: np.ndarray | None) -> np.ndarray:
        if norm_value is not None:
            spks = spks / norm_value
        if self.norm_scale != 1.0:
            spks = spks * self.norm_scale
        return spks

    def _get_split_arrays(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        split: "SplitName",
    ) -> tuple[np.ndarray, FrameBehavior]:
        """Get raw neural activity and matching FrameBehavior for a registry split."""
        population, frame_behavior = registry.get_population(session, self.spks_type)
        split_idx = registry.time_split[split]
        idx_within = population.get_split_times(split_idx, within_idx_samples=True)
        spks = population.data[:, idx_within].numpy().T  # (frames, neurons)
        idx_orig = np.array(population.get_split_times(split_idx, within_idx_samples=False))
        return spks, frame_behavior.filter(idx_orig)

    def fit(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        split: "SplitName" = "train",
        idx_rois: np.ndarray | None = None,
    ) -> LocPredFit:
        """Build place fields and estimate residual variance from a training split.

        Parameters
        ----------
        session : B2Session
        registry : PopulationRegistry
        split : SplitName
            Registry split to use as training data. Default "train".
        idx_rois : np.ndarray or None
            Boolean mask over neurons to use. If None, ROIs are selected via reliability and
            activity criteria. Pass a precomputed mask to skip the selection step.

        Returns
        -------
        LocPredFit
        """
        spks_tr, frame_behavior_tr = self._get_split_arrays(session, registry, split)
        norm_value = self._compute_norm_value(spks_tr)
        spks_tr = self._apply_norm(spks_tr, norm_value)

        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)
        if idx_rois is None:
            idx_keep_rois = self._select_rois(session, spks_tr, frame_behavior_tr, dist_edges)
        else:
            idx_keep_rois = idx_rois
        spks_tr_roi = spks_tr[:, idx_keep_rois]

        # Iteration 0: standard placefield fit
        current_pf = get_placefield(
            spks_tr_roi,
            frame_behavior_tr,
            dist_edges,
            average=True,
            smooth_width=self.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        current_fb = frame_behavior_tr
        current_var = _estimate_residual_variance(spks_tr_roi, current_fb, current_pf)
        current_bins = _true_position_bins(current_fb, current_pf)

        iteration_placefields = [current_pf]
        iteration_diag_variances = [current_var]
        iteration_position_bins_tr = [current_bins]

        # Iterations 1..fit_iterations-1: E-step (argmax) then M-step (refit placefield)
        for _ in range(self.fit_iterations - 1):
            lik_fn = _get_likelihood_methods((self.predict_likelihood,), current_var)[self.predict_likelihood]
            current_fb, current_bins = _hard_e_step(spks_tr_roi, current_pf, lik_fn, dist_edges, current_fb)
            current_pf = get_placefield(
                spks_tr_roi,
                current_fb,
                dist_edges,
                average=True,
                smooth_width=self.smooth_width,
                use_fast_sampling=False,
                session=session,
            )
            current_var = _estimate_residual_variance(spks_tr_roi, current_fb, current_pf)
            iteration_placefields.append(current_pf)
            iteration_diag_variances.append(current_var)
            iteration_position_bins_tr.append(current_bins)

        return LocPredFit(
            placefield=iteration_placefields[-1],
            diag_gaussian_variance=iteration_diag_variances[-1],
            idx_keep_rois=idx_keep_rois,
            dist_edges=dist_edges,
            norm_value=norm_value,
            iteration_placefields=iteration_placefields,
            iteration_diag_variances=iteration_diag_variances,
            iteration_position_bins_tr=iteration_position_bins_tr,
        )

    def optimize(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        fit: LocPredFit,
        split: "SplitName" = "validation",
        iteration: int = -1,
    ):
        """Validation of hyperparameters for loss functions.

        As of implementation of this function, the only hyperparameters are the temperature for cross-entropy loss
        and the margin for hinge rank loss. (Kappa is a hyperparameter for von-mises Fisher, but we can absorb that into
        temperature and call it a day). To measure these, we compute loss on the validation split using golden section
        search.

        Parameters
        ----------
        fit : LocPredFit
            Fitted state from :meth:`fit`.
        session : B2Session
        registry : PopulationRegistry
        split : SplitName
            Registry split to use as validation data. Default "validation".
        iteration : int
            Which EM iteration's placefield to use. Default -1 (last iteration).
        """
        pf = fit.iteration_placefields[iteration]
        dvar = fit.iteration_diag_variances[iteration]

        spks_vl, frame_behavior_vl = self._get_split_arrays(session, registry, split)
        spks_vl = self._apply_norm(spks_vl, fit.norm_value)
        spks_vl = spks_vl[:, fit.idx_keep_rois]

        # Drop frames whose environment didn't appear in training data.
        idx_known = np.isin(frame_behavior_vl.environment, pf.environment)
        frame_behavior_vl = frame_behavior_vl.filter(idx_known)
        spks_vl = spks_vl[idx_known]

        true_bins_vl = _true_position_bins(frame_behavior_vl, pf)
        lik_methods = _get_likelihood_methods(self.likelihood_methods, dvar)

        # Only keep the ones with a hyperparameter to optimize over
        loss_methods = _get_loss_methods(self.loss_methods, fit.dist_edges, only_with_hyperparameters=True)

        hyperparameters: dict[str, dict[str, float]] = {}
        for lik_name, lik_fn in lik_methods.items():
            ll = lik_fn(spks_vl, pf)
            hyperparameters[lik_name] = {}
            for loss_name, loss_fn in loss_methods.items():
                hyperparameters[lik_name][loss_name] = _optimize(loss_fn, ll, true_bins_vl)

        return hyperparameters

    def score(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        fit: LocPredFit,
        hyperparameters: dict[str, dict[str, float]] | None = None,
        split: "SplitName" = "test",
        iteration: int = -1,
    ) -> dict:
        """Compute log-likelihoods and losses on a held-out split.

        Parameters
        ----------
        session : B2Session
        registry : PopulationRegistry
        fit : LocPredFit
            Fitted state from :meth:`fit`.
        hyperparameters : dict[str, dict[str, float]] | None
            Hyperparameters to use for evaluation. If None, uses the default hyperparameters.
        split : SplitName
            Registry split to evaluate on. Default "test".
        iteration : int
            Which EM iteration's placefield to use. Default -1 (last iteration).

        Returns
        -------
        dict
            likelihood_matrix : dict mapping likelihood method name to array of shape (frames, total_bins) with log-likelihoods for each bin
            loss_trajectory : dict mapping "likelihood_loss" method name to array of shape (frames,) with loss values for each frame
            loss_scalar : dict mapping "likelihood_loss" method name to scalar summary of the loss across frames (e.g. mean loss)
            true_bin_score : dict mapping "likelihood_method" to array of shape (frames,) with log-likelihood of the true position bin for each frame (to estimate prediction of activity)
            true_position_bins_te : array of shape (frames,) with true position bin index for each frame
            idx_keep_rois : boolean array of shape (rois,) indicating which ROIs were kept in the fit (for reference when interpreting likelihoods and losses)
        """
        pf = fit.iteration_placefields[iteration]
        dvar = fit.iteration_diag_variances[iteration]

        spks_te, frame_behavior_te = self._get_split_arrays(session, registry, split)
        spks_te = self._apply_norm(spks_te, fit.norm_value)
        spks_te = spks_te[:, fit.idx_keep_rois]

        # Drop test frames whose environment didn't appear in training data.
        idx_known = np.isin(frame_behavior_te.environment, pf.environment)
        frame_behavior_te = frame_behavior_te.filter(idx_known)
        spks_te = spks_te[idx_known]
        true_bins_te = _true_position_bins(frame_behavior_te, pf)

        # Use true position bins and learned PF to predict likelihood of each activity vector
        # likelihood returns p(x=bin | r, theta) = p(r | x=bin, theta) * p(x=bin | theta) / p(r | theta)

        # Compute likelihoods and losses for all requested methods.
        # Likelihood: use a model p(x=bin | r, theta) to predict the likelihood of each position bin given the neural activity and the PF learned in fit()
        # Loss: measure how well that likelihood estimates the true position bin
        lik_methods = _get_likelihood_methods(self.likelihood_methods, dvar)
        loss_methods = _get_loss_methods(self.loss_methods, fit.dist_edges)

        # Prepare our output dicts
        likelihood_matrix: dict[str, np.ndarray] = {}
        loss_trajectory: dict[str, np.ndarray] = {}
        loss_scalar: dict[str, float] = {}
        true_bin_score: dict[str, np.ndarray] = {}
        for lik_name, lik_fn in lik_methods.items():

            # Measure likelihood of each bin
            ll = lik_fn(spks_te, pf)
            likelihood_matrix[lik_name] = ll
            true_bin_score[lik_name] = ll[np.arange(len(true_bins_te)), true_bins_te]

            # Get hyperparameters for this likelihood, if any
            _hyperparams_for_lik = hyperparameters[lik_name] if hyperparameters is not None else {}
            for loss_name, loss_fn in loss_methods.items():

                # If this likelihood-loss combo has a hyperparameter, pass it in kwargs
                kwargs = {}
                if loss_fn.has_hyperparameter and loss_name in _hyperparams_for_lik:
                    kwargs[loss_fn.hyperparameter_name] = _hyperparams_for_lik[loss_name]

                # Copmute loss scalars, trajectory (across samples), add to summary dicts
                scalars, traj = loss_fn(ll, true_bins_te, **kwargs)
                combo = f"{lik_name}_{loss_name}"
                loss_trajectory[combo] = traj
                for k, v in scalars.items():
                    key = combo if k in ("loss", "fraction") else f"{combo}_{k}"
                    loss_scalar[key] = v

        return dict(
            likelihood_matrix=likelihood_matrix,
            loss_trajectory=loss_trajectory,
            loss_scalar=loss_scalar,
            true_bin_score=true_bin_score,
            true_position_bins_te=true_bins_te,
            idx_keep_rois=fit.idx_keep_rois,
        )

    def process(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        train_split: "SplitName" = "train",
        val_split: "SplitName" = "validation",
        test_split: "SplitName" = "test",
    ) -> dict:
        """Run Location Prediction analysis on a session.

        Iterates over all EM iterations, running optimize and score for each. Results are
        stacked across iterations: loss_scalar values are arrays of shape (fit_iterations,),
        loss_trajectory arrays are shape (fit_iterations, frames), and likelihood_matrix arrays
        are shape (fit_iterations, frames, total_bins).
        """
        fit = self.fit(session, registry, split=train_split)
        all_results = []
        for i in range(self.fit_iterations):
            hyperparameters = self.optimize(session, registry, fit, split=val_split, iteration=i)
            result = self.score(session, registry, fit, hyperparameters=hyperparameters, split=test_split, iteration=i)
            all_results.append(result)

        first = all_results[0]
        return dict(
            likelihood_matrix={k: np.stack([r["likelihood_matrix"][k] for r in all_results]) for k in first["likelihood_matrix"]},
            loss_trajectory={k: np.stack([r["loss_trajectory"][k] for r in all_results]) for k in first["loss_trajectory"]},
            loss_scalar={k: np.array([r["loss_scalar"][k] for r in all_results]) for k in first["loss_scalar"]},
            true_bin_score={k: np.stack([r["true_bin_score"][k] for r in all_results]) for k in first["true_bin_score"]},
            true_position_bins_te=first["true_position_bins_te"],
            idx_keep_rois=fit.idx_keep_rois,
            iteration_position_bins_tr=fit.iteration_position_bins_tr,
        )


# ---------------------------------------------------------------------------
# LocPredCrossVal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocPredCrossVal(AnalysisConfigBase):
    """Cross-population location prediction: decode position from source neurons, predict target activity.

    Fits separate place fields for source and target neurons (using the registry's preset cell split),
    then for each test frame:
    - Decodes position from source activity using each likelihood model (argmax).
    - Predicts target activity as target_PF[decoded_bin] (internal estimate).
    - Compares against target_PF[true_bin] as an oracle upper bound.

    The gap between oracle R² and internal R² measures information lost by going through
    an internal position estimate rather than true/extrinsic position.

    Parameters
    ----------
    norm_method : str
        Normalization applied to source spike counts using only training statistics.
        "zero-one": divide by per-cell max. "none": no normalization.
    norm_scale : float
        Optional global scale applied to source spikes after normalization.
    num_bins : int
        Number of spatial bins for place field computation.
    smooth_width : float or None
        Gaussian smoothing width for place fields.
    reliability_cutoff : float
        Minimum leave-one-out reliability for ROI inclusion (applied to source and target separately).
    fraction_active_cutoff : float
        Minimum participation fraction for ROI inclusion (applied to source and target separately).
    spks_type : SpksTypes
        Spike type to retrieve from the registry.
    likelihood_methods : tuple[str, ...]
        Likelihood functions to use for position decoding.
        Options: "poisson", "gaussian", "diag_gaussian", "von_mises_fisher".
    """

    schema_version: str = "v1"
    data_config_name: str = "default"

    norm_method: str = "zero-one"
    norm_scale: float = 1.0
    num_bins: int = 100
    smooth_width: float | None = 0.25
    reliability_cutoff: float = 0.1
    fraction_active_cutoff: float = 0.1
    spks_type: SpksTypes = "oasis"
    likelihood_methods: tuple[str, ...] = ("poisson", "gaussian", "diag_gaussian", "von_mises_fisher")

    display_name: ClassVar[str] = "locpred_crossval"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "norm_scale": [1.0, 50.0],
            "smooth_width": [None, 5.0],
        }

    def _compute_norm_value(self, spks_tr: np.ndarray) -> np.ndarray | None:
        if self.norm_method == "zero-one":
            return np.maximum(np.max(spks_tr, axis=0), 1e-12)
        elif self.norm_method == "none":
            return None
        else:
            raise ValueError(f"Unknown norm_method: {self.norm_method!r}")

    def _apply_norm(self, spks: np.ndarray, norm_value: np.ndarray | None) -> np.ndarray:
        if norm_value is not None:
            spks = spks / norm_value
        if self.norm_scale != 1.0:
            spks = spks * self.norm_scale
        return spks

    def fit(
        self,
        session: B2Session,
        registry: "PopulationRegistry",
        split: "SplitName" = "train",
    ) -> LocPredCrossValFit:
        """Fit source and target place fields from a training split.

        Parameters
        ----------
        session : B2Session
        registry : PopulationRegistry
        split : SplitName
            Registry split to use as training data. Default "train".

        Returns
        -------
        LocPredCrossValFit
        """
        spks_source_tr, spks_target_tr, frame_behavior_tr = _get_source_target_arrays(session, registry, self.spks_type, split)
        norm_value = self._compute_norm_value(spks_source_tr)
        spks_source_tr = self._apply_norm(spks_source_tr, norm_value)
        norm_value_target = self._compute_norm_value(spks_target_tr)
        spks_target_tr = self._apply_norm(spks_target_tr, norm_value_target)

        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)

        idx_keep_source = _select_rois_fn(
            spks_source_tr,
            frame_behavior_tr,
            dist_edges,
            session,
            self.reliability_cutoff,
            self.fraction_active_cutoff,
        )
        idx_keep_target = _select_rois_fn(
            spks_target_tr,
            frame_behavior_tr,
            dist_edges,
            session,
            self.reliability_cutoff,
            self.fraction_active_cutoff,
        )

        pf_source = get_placefield(
            spks_source_tr[:, idx_keep_source],
            frame_behavior_tr,
            dist_edges,
            average=True,
            smooth_width=self.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        pf_target = get_placefield(
            spks_target_tr[:, idx_keep_target],
            frame_behavior_tr,
            dist_edges,
            average=True,
            smooth_width=self.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        diag_gaussian_variance = _estimate_residual_variance(spks_source_tr[:, idx_keep_source], frame_behavior_tr, pf_source)
        return LocPredCrossValFit(
            pf_source=pf_source,
            pf_target=pf_target,
            diag_gaussian_variance=diag_gaussian_variance,
            idx_keep_source=idx_keep_source,
            idx_keep_target=idx_keep_target,
            dist_edges=dist_edges,
            norm_value=norm_value,
            norm_value_target=norm_value_target,
        )

    def score(
        self,
        session: B2Session,
        registry: "PopulationRegistry",
        fit: LocPredCrossValFit,
        split: "SplitName" = "test",
    ) -> dict:
        """Score source→target prediction on a held-out split.

        For each frame, decodes position from source activity (argmax of log-likelihood under each
        likelihood model), then predicts target activity as target_PF[decoded_bin]. Compares against
        oracle prediction target_PF[true_bin].

        Parameters
        ----------
        session : B2Session
        registry : PopulationRegistry
        fit : LocPredCrossValFit
            Fitted state from :meth:`fit`.
        split : SplitName
            Registry split to evaluate on. Default "test".

        Returns
        -------
        dict
            oracle_r2 : float
                Mean R² across target ROIs using true position bins (upper bound).
            oracle_r2_roi : np.ndarray
                Per-target-ROI R², shape (n_target_kept,).
            internal_r2 : dict[str, float]
                Mean R² per likelihood method using decoded position bins.
            internal_r2_roi : dict[str, np.ndarray]
                Per-target-ROI R² per likelihood method, shape (n_target_kept,).
            true_bins_te : np.ndarray
                True flat bin indices, shape (frames,).
            idx_keep_source : np.ndarray
                Boolean mask over source neurons used in fit.
            idx_keep_target : np.ndarray
                Boolean mask over target neurons used in fit.
        """
        spks_source_te, spks_target_te, frame_behavior_te = _get_source_target_arrays(session, registry, self.spks_type, split)
        spks_source_te = self._apply_norm(spks_source_te, fit.norm_value)
        spks_source_te = spks_source_te[:, fit.idx_keep_source]
        spks_target_te = self._apply_norm(spks_target_te, fit.norm_value_target)
        spks_target_te = spks_target_te[:, fit.idx_keep_target]

        # Drop frames whose environment didn't appear in training data.
        idx_known = np.isin(frame_behavior_te.environment, fit.pf_source.environment)
        frame_behavior_te = frame_behavior_te.filter(idx_known)
        spks_source_te = spks_source_te[idx_known]
        spks_target_te = spks_target_te[idx_known]

        true_bins_te = _true_position_bins(frame_behavior_te, fit.pf_source)
        pf_target_flat = _pf_flat(fit.pf_target)  # (total_bins, n_target_kept)

        # Oracle: predict target activity using true position bins
        target_oracle = pf_target_flat[true_bins_te]  # (frames, n_target_kept)
        oracle_r2 = measure_r2(target_oracle.T, spks_target_te.T, reduce="mean", dim=None)

        # Internal: decode position from source, predict target
        lik_methods = _get_likelihood_methods(self.likelihood_methods, fit.diag_gaussian_variance)
        internal_r2: dict[str, float] = {}
        for lik_name, lik_fn in lik_methods.items():
            ll = lik_fn(spks_source_te, fit.pf_source)  # (frames, total_bins)
            pred_bins = np.argmax(ll, axis=1)  # (frames,)
            target_internal = pf_target_flat[pred_bins]  # (frames, n_target_kept)
            internal_r2[lik_name] = measure_r2(target_internal.T, spks_target_te.T, reduce="mean", dim=None)

        return dict(
            oracle_r2=oracle_r2,
            internal_r2=internal_r2,
            true_bins_te=true_bins_te,
            idx_keep_source=fit.idx_keep_source,
            idx_keep_target=fit.idx_keep_target,
        )

    def process(
        self,
        session: B2Session,
        registry: "PopulationRegistry",
        train_split: "SplitName" = "train",
        test_split: "SplitName" = "test",
    ) -> dict:
        """Run source→target location prediction analysis on a session."""
        fit = self.fit(session, registry, split=train_split)
        return self.score(session, registry, fit, split=test_split)
