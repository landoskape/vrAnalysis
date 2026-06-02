"""Placefield structure — noise modelling and synthetic data generation.

Implements the composable system described in ``simulation_design.md``:
residual models (fit from data or parametric), optional nuisance signals,
and a :class:`PlacefieldDataGenerator` that produces ``(N, K, T)`` synthetic
datasets.  The :class:`PlacefieldStructureConfig` wires this into the
standard analysis pipeline so the generator can be constructed from a live
session or tested standalone in a notebook.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis

from vrAnalysis.helpers import reliability_loo
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.placefields import get_placefield
from vrAnalysis.sessions import B2Session, SpksTypes

from ..pipeline.base import AnalysisConfigBase
from ..registry import (
    ACTIVITY_PARAMETERS_NAMES,
    PopulationRegistry,
    get_activity_parameters,
)

if TYPE_CHECKING:
    from ..registry import SplitName


# ---------------------------------------------------------------------------
# Abstract bases
# ---------------------------------------------------------------------------


class ResidualConfig(ABC):
    """Abstract base for position-locked residual noise models.

    Subclasses know how to:

    - Be constructed from real session data via :meth:`from_data`.
    - Be constructed from explicit distribution parameters via
      :meth:`from_distribution`.
    - Draw residual tensors ``R`` of shape ``(N, K, T)`` via :meth:`sample`.

    ``P`` is passed at sample time so that heteroscedastic models can set
    local variance from the local mean.  All fitted subclasses expose a
    ``P_`` attribute (the estimated mean from fitting) for inspection.
    """

    @classmethod
    def from_data(cls, Y: np.ndarray) -> "ResidualConfig":
        """Fit noise model from a ``(N, K, T)`` data tensor.

        Computes ``P = Y.mean(axis=-1)`` then fits model parameters from
        residuals ``R = Y - P[:, :, None]``.

        Parameters
        ----------
        Y : np.ndarray
            Data tensor of shape ``(N, K, T)`` (neurons × positions × trials).

        Returns
        -------
        ResidualConfig
        """
        raise NotImplementedError

    @classmethod
    def from_distribution(cls, **params) -> "ResidualConfig":
        """Construct from explicit distribution parameters.

        Parameters
        ----------
        **params
            Distribution-specific keyword arguments (e.g. ``sigma``, ``a``,
            ``b``).

        Returns
        -------
        ResidualConfig
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, P: np.ndarray, n_trials: int) -> np.ndarray:
        """Draw residuals given a mean placefield matrix.

        Parameters
        ----------
        P : np.ndarray
            Mean placefield matrix, shape ``(N, K)``.
        n_trials : int
            Number of trials to generate.

        Returns
        -------
        np.ndarray
            Residuals of shape ``(N, K, T)``.
        """


class NuisanceConfig(ABC):
    """Abstract base for position-independent nuisance signals.

    Nuisance is additive in the unfolded ``(N, S)`` space, capturing
    structure that is not locked to spatial position — shared brain-state
    fluctuations, low-rank drift, global gain modulation, etc.
    """

    @abstractmethod
    def sample(self, n_neurons: int, n_samples: int) -> np.ndarray:
        """Draw a nuisance matrix in unfolded sample space.

        Parameters
        ----------
        n_neurons : int
        n_samples : int
            Number of unfolded samples (``K * n_trials``).

        Returns
        -------
        np.ndarray
            Shape ``(N, S)``.
        """


# ---------------------------------------------------------------------------
# Residual config implementations
# ---------------------------------------------------------------------------


@dataclass
class StationaryDiagonalResidualConfig(ResidualConfig):
    """Per-neuron scalar variance, constant across positions.

    Parameters
    ----------
    sigma : np.ndarray
        Standard deviations, shape ``(N,)``.
    P_ : np.ndarray or None
        Mean placefield estimated during fitting, shape ``(N, K)``.
        ``None`` when constructed via :meth:`from_distribution`.
    """

    sigma: np.ndarray
    P_: np.ndarray | None = None

    @classmethod
    def from_data(
        cls,
        Y: np.ndarray,
        P: np.ndarray | None = None,
    ) -> "StationaryDiagonalResidualConfig":
        """Fit from ``(N, K, T)`` data; ``σ_n = rms(R[n, :, :])``.

        Parameters
        ----------
        Y : np.ndarray
            Shape ``(N, K, T)``.
        P : np.ndarray or None
            Mean placefield, shape ``(N, K)``.  If provided (e.g. a
            cross-validated mean) it is used in place of ``Y.mean(axis=-1)``
            when computing residuals.
        """
        P_fit = Y.mean(axis=-1) if P is None else P
        R = Y - P_fit[:, :, None]
        sigma = np.sqrt(np.mean(R**2, axis=(1, 2)))
        return cls(sigma=sigma, P_=P_fit)

    @classmethod
    def from_distribution(cls, sigma: np.ndarray | float) -> "StationaryDiagonalResidualConfig":
        """Construct from per-neuron standard deviations.

        Parameters
        ----------
        sigma : np.ndarray or float
            Per-neuron standard deviation(s), broadcast to ``(N,)`` at sample time.
        """
        return cls(sigma=np.asarray(sigma))

    def sample(self, P: np.ndarray, n_trials: int) -> np.ndarray:
        N, K = P.shape
        return np.random.randn(N, K, n_trials) * self.sigma[:, None, None]


@dataclass
class StationaryLowRankResidualConfig(ResidualConfig):
    """Factor-model covariance ``Σ = W Wᵀ + diag(ψ)``, position-stationary.

    Shared noise (neuropil, hemodynamics) captured by a low-rank factor
    model.  Residuals are pooled across positions before fitting.

    Parameters
    ----------
    W : np.ndarray
        Factor loadings, shape ``(N, r)``.
    psi : np.ndarray
        Diagonal private noise variances, shape ``(N,)``.
    P_ : np.ndarray or None
        Mean placefield estimated during fitting, shape ``(N, K)``.
    """

    W: np.ndarray
    psi: np.ndarray
    P_: np.ndarray | None = None

    @classmethod
    def from_data(
        cls,
        Y: np.ndarray,
        P: np.ndarray | None = None,
        n_components: int = 5,
    ) -> "StationaryLowRankResidualConfig":
        """Fit via sklearn FactorAnalysis on pooled residuals.

        Parameters
        ----------
        Y : np.ndarray
            Shape ``(N, K, T)``.
        P : np.ndarray or None
            Mean placefield, shape ``(N, K)``.  If provided (e.g. a
            cross-validated mean) it is used in place of ``Y.mean(axis=-1)``
            when computing residuals.
        n_components : int
            Number of latent factors.
        """
        N, K, T = Y.shape
        P_fit = Y.mean(axis=-1) if P is None else P
        R = Y - P_fit[:, :, None]
        X = R.reshape(N, K * T).T  # (K*T, N) — samples × features for sklearn
        fa = FactorAnalysis(n_components=n_components)
        fa.fit(X)
        W = fa.components_.T  # (N, r)
        psi = fa.noise_variance_  # (N,)
        return cls(W=W, psi=psi, P_=P_fit)

    @classmethod
    def from_distribution(cls, W: np.ndarray, psi: np.ndarray) -> "StationaryLowRankResidualConfig":
        """Construct from factor loadings and diagonal variances.

        Parameters
        ----------
        W : np.ndarray
            Shape ``(N, r)``.
        psi : np.ndarray
            Shape ``(N,)``.
        """
        return cls(W=np.asarray(W), psi=np.asarray(psi))

    def sample(self, P: np.ndarray, n_trials: int) -> np.ndarray:
        N, K = P.shape
        r = self.W.shape[1]
        S = K * n_trials
        z = np.random.randn(r, S)
        shared = self.W @ z  # (N, S)
        noise = np.random.randn(N, S) * np.sqrt(np.maximum(self.psi[:, None], 0.0))
        return (shared + noise).reshape(N, K, n_trials)


@dataclass
class HeteroscedasticDiagonalResidualConfig(ResidualConfig):
    """Per-neuron, per-position variance: ``σ²[n, k]`` fitted from residuals.

    Parameters
    ----------
    sigma2 : np.ndarray
        Variance array, shape ``(N, K)``.
    P_ : np.ndarray or None
        Mean placefield estimated during fitting.
    """

    sigma2: np.ndarray
    P_: np.ndarray | None = None

    @classmethod
    def from_data(
        cls,
        Y: np.ndarray,
        P: np.ndarray | None = None,
    ) -> "HeteroscedasticDiagonalResidualConfig":
        """Fit per-bin variance ``σ²[n,k] = var_t R[n,k,:]``.

        Parameters
        ----------
        Y : np.ndarray
            Shape ``(N, K, T)``.
        P : np.ndarray or None
            Mean placefield, shape ``(N, K)``.  If provided (e.g. a
            cross-validated mean) it is used in place of ``Y.mean(axis=-1)``
            when computing residuals.
        """
        P_fit = Y.mean(axis=-1) if P is None else P
        R = Y - P_fit[:, :, None]
        sigma2 = np.var(R, axis=-1, ddof=1)  # (N, K)
        return cls(sigma2=sigma2, P_=P_fit)

    @classmethod
    def from_distribution(cls, sigma2: np.ndarray) -> "HeteroscedasticDiagonalResidualConfig":
        """Construct from a ``(N, K)`` variance array.

        Parameters
        ----------
        sigma2 : np.ndarray
            Per-neuron, per-position variance, shape ``(N, K)``.
        """
        return cls(sigma2=np.asarray(sigma2))

    def sample(self, P: np.ndarray, n_trials: int) -> np.ndarray:
        sigma = np.sqrt(np.maximum(self.sigma2, 0.0))  # (N, K)
        return np.random.randn(*P.shape, n_trials) * sigma[:, :, None]


@dataclass
class PoissonLikeResidualConfig(ResidualConfig):
    """Heteroscedastic parametric model: ``σ²[n,k] = a[n] · P[n,k] + b[n]``.

    Recommended default for calcium imaging data.  Variance scales linearly
    with mean (Poisson/Fano-factor-like), capturing shot noise and gain
    fluctuations.

    Parameters
    ----------
    a : np.ndarray
        Fano-factor-like slope per neuron, shape ``(N,)``.
    b : np.ndarray
        Baseline additive variance per neuron, shape ``(N,)``.
    min_variance : float
        Floor applied at sample time to prevent degenerate zero or negative
        variance when ``a`` or ``b`` are negative (e.g. from noisy regression).
    P_ : np.ndarray or None
        Mean placefield estimated during fitting.
    """

    a: np.ndarray
    b: np.ndarray
    min_variance: float = 1e-8
    P_: np.ndarray | None = None

    @classmethod
    def from_data(
        cls,
        Y: np.ndarray,
        P: np.ndarray | None = None,
        min_variance: float = 1e-8,
    ) -> "PoissonLikeResidualConfig":
        """Fit slope and intercept per neuron by OLS of ``σ²[n,k]`` on ``P[n,k]``.

        Parameters
        ----------
        Y : np.ndarray
            Shape ``(N, K, T)``.
        P : np.ndarray or None
            Mean placefield, shape ``(N, K)``.  If provided (e.g. a
            cross-validated mean) it is used in place of ``Y.mean(axis=-1)``
            when computing residuals.
        min_variance : float
            Floor for variance in :meth:`sample`.
        """
        P_fit = Y.mean(axis=-1) if P is None else P  # (N, K)
        R = Y - P_fit[:, :, None]
        sigma2 = np.var(R, axis=-1, ddof=1)  # (N, K)

        # Vectorised OLS: σ²[n,k] ~ a[n] * P[n,k] + b[n]
        P_mean = P_fit.mean(axis=1, keepdims=True)  # (N, 1)
        s2_mean = sigma2.mean(axis=1, keepdims=True)  # (N, 1)
        P_c = P_fit - P_mean  # (N, K)
        s2_c = sigma2 - s2_mean  # (N, K)
        P_var = np.maximum(np.sum(P_c**2, axis=1), 1e-12)  # (N,)
        a = np.sum(P_c * s2_c, axis=1) / P_var  # (N,)
        b = s2_mean.squeeze(axis=1) - a * P_mean.squeeze(axis=1)  # (N,)
        return cls(a=a, b=b, min_variance=min_variance, P_=P_fit)

    @classmethod
    def from_distribution(
        cls,
        a: np.ndarray | float,
        b: np.ndarray | float,
        min_variance: float = 1e-8,
    ) -> "PoissonLikeResidualConfig":
        """Construct from explicit slope and intercept parameters.

        Parameters
        ----------
        a : np.ndarray or float
            Fano-factor slope(s).  Scalar broadcasts to all neurons at sample
            time.
        b : np.ndarray or float
            Baseline additive variance.
        min_variance : float
            Variance floor for :meth:`sample`.
        """
        return cls(a=np.asarray(a), b=np.asarray(b), min_variance=min_variance)

    def sample(self, P: np.ndarray, n_trials: int) -> np.ndarray:
        variance = np.maximum(
            self.a[:, None] * np.maximum(P, 0.0) + self.b[:, None],
            self.min_variance,
        )  # (N, K)
        return np.random.randn(*P.shape, n_trials) * np.sqrt(variance)[:, :, None]


@dataclass
class FullCovariancePerPositionResidualConfig(ResidualConfig):
    """Full ``(N, N)`` covariance fitted per position (Ledoit-Wolf regularised).

    Only tractable when ``T >> N``.  Raises a warning when ``T ≤ N`` per
    position, as estimation is underdetermined.

    Parameters
    ----------
    sigmas : np.ndarray
        Covariance matrices per position, shape ``(K, N, N)``.
    P_ : np.ndarray or None
        Mean placefield estimated during fitting.
    """

    sigmas: np.ndarray  # (K, N, N)
    P_: np.ndarray | None = None

    @classmethod
    def from_data(
        cls,
        Y: np.ndarray,
        P: np.ndarray | None = None,
    ) -> "FullCovariancePerPositionResidualConfig":
        """Fit per-position covariance from residuals via Ledoit-Wolf.

        Parameters
        ----------
        Y : np.ndarray
            Shape ``(N, K, T)``.
        P : np.ndarray or None
            Mean placefield, shape ``(N, K)``.  If provided (e.g. a
            cross-validated mean) it is used in place of ``Y.mean(axis=-1)``
            when computing residuals.
        """
        N, K, T = Y.shape
        if T <= N:
            warnings.warn(
                f"FullCovariancePerPositionResidualConfig: T={T} ≤ N={N}; " "per-position covariance estimation is underdetermined.",
                UserWarning,
                stacklevel=2,
            )
        P_fit = Y.mean(axis=-1) if P is None else P
        R = Y - P_fit[:, :, None]  # (N, K, T)
        lw = LedoitWolf()
        sigmas = np.empty((K, N, N))
        for k in range(K):
            lw.fit(R[:, k, :].T)  # (T, N)
            sigmas[k] = lw.covariance_
        return cls(sigmas=sigmas, P_=P_fit)

    @classmethod
    def from_distribution(cls, sigmas: np.ndarray) -> "FullCovariancePerPositionResidualConfig":
        """Construct from explicit per-position covariance matrices.

        Parameters
        ----------
        sigmas : np.ndarray
            Shape ``(K, N, N)``.
        """
        return cls(sigmas=np.asarray(sigmas))

    def sample(self, P: np.ndarray, n_trials: int) -> np.ndarray:
        N, K = P.shape
        R = np.empty((N, K, n_trials))
        for k in range(K):
            try:
                L = np.linalg.cholesky(self.sigmas[k])
                R[:, k, :] = L @ np.random.randn(N, n_trials)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(self.sigmas[k])
                vals = np.maximum(vals, 0.0)
                z = np.random.randn(N, n_trials)
                R[:, k, :] = vecs @ (np.sqrt(vals[:, None]) * z)
        return R


# ---------------------------------------------------------------------------
# Nuisance config implementations
# ---------------------------------------------------------------------------


@dataclass
class LowRankNuisanceConfig(NuisanceConfig):
    """Position-independent low-rank signal ``Z = U V``, rescaled to variance.

    Models global brain-state fluctuations or task-correlated signals that
    are not place-field-locked.

    Parameters
    ----------
    rank : int
        Rank of the nuisance signal.
    variance : float
        Target per-element variance of ``Z``.
    """

    rank: int
    variance: float

    def sample(self, n_neurons: int, n_samples: int) -> np.ndarray:
        U = np.random.randn(n_neurons, self.rank)
        V = np.random.randn(self.rank, n_samples)
        Z = U @ V  # (N, S)
        current_var = np.var(Z)
        if current_var > 0:
            Z *= np.sqrt(self.variance / current_var)
        return Z


@dataclass
class StationaryGaussianNuisanceConfig(NuisanceConfig):
    """i.i.d. Gaussian nuisance in unfolded sample space.

    Parameters
    ----------
    sigma : float
        Standard deviation of the additive noise.
    """

    sigma: float

    def sample(self, n_neurons: int, n_samples: int) -> np.ndarray:
        return np.random.randn(n_neurons, n_samples) * self.sigma


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class PlacefieldDataGenerator:
    """Composable synthetic neural population data generator.

    Combines a mean placefield ``P``, a :class:`ResidualConfig`, and an
    optional :class:`NuisanceConfig` into a callable sampler.

    Parameters
    ----------
    P : np.ndarray
        Mean placefield matrix, shape ``(N, K)``.
    residual_config : ResidualConfig
        Noise model for position-locked residuals.
    nuisance_config : NuisanceConfig or None
        Optional position-independent additive signal in unfolded space.
    """

    P: np.ndarray
    residual_config: ResidualConfig
    nuisance_config: NuisanceConfig | None = None

    @property
    def true_spectrum(self) -> np.ndarray:
        """Squared singular values of ``P`` — the true signal spectrum.

        Returns
        -------
        np.ndarray
            Shape ``(min(N, K),)``.
        """
        _, s, _ = np.linalg.svd(self.P, full_matrices=False)
        return s**2

    def sample(self, n_trials: int, return_unfolded: bool = False) -> np.ndarray:
        """Generate a synthetic dataset.

        Steps:

        1. ``R = residual_config.sample(P, n_trials)``   — shape ``(N, K, T)``
        2. ``Y = P[:, :, None] + R``
        3. ``X = Y.reshape(N, K * n_trials)``
        4. (optional) ``X += nuisance_config.sample(N, K * n_trials)``
        5. Return ``X`` (unfolded) or ``X.reshape(N, K, n_trials)``.

        Parameters
        ----------
        n_trials : int
        return_unfolded : bool
            Return the unfolded ``(N, K * n_trials)`` array instead of the
            ``(N, K, T)`` tensor.

        Returns
        -------
        np.ndarray
            Shape ``(N, K, T)`` or ``(N, K * n_trials)`` when
            ``return_unfolded=True``.
        """
        N, K = self.P.shape
        R = self.residual_config.sample(self.P, n_trials)  # (N, K, T)
        Y = self.P[:, :, None] + R  # (N, K, T)
        X = Y.reshape(N, K * n_trials)  # (N, S)
        if self.nuisance_config is not None:
            X = X + self.nuisance_config.sample(N, K * n_trials)
        if return_unfolded:
            return X
        return X.reshape(N, K, n_trials)


# ---------------------------------------------------------------------------
# Residual type registry
# ---------------------------------------------------------------------------

_RESIDUAL_TYPE_MAP: dict[str, type[ResidualConfig]] = {
    "stationary_diagonal": StationaryDiagonalResidualConfig,
    "stationary_low_rank": StationaryLowRankResidualConfig,
    "heteroscedastic_diagonal": HeteroscedasticDiagonalResidualConfig,
    "poisson_like": PoissonLikeResidualConfig,
    "full_covariance": FullCovariancePerPositionResidualConfig,
}

# ---------------------------------------------------------------------------
# Fitted state
# ---------------------------------------------------------------------------


@dataclass
class PlacefieldStructureFit:
    """Fitted state returned by :meth:`PlacefieldStructureConfig.fit`.

    ``Y`` has environments stacked along the position axis so its shape is
    ``(N, E*K, T)`` where ``E`` is the number of qualifying environments,
    ``K = num_bins``, and ``T`` is the equalized trial count.  ``P`` is the
    trial-mean of ``Y``, also of shape ``(N, E*K)``.

    Parameters
    ----------
    P : np.ndarray
        Mean placefield matrix, shape ``(N, E*K)``.
    Y : np.ndarray
        Trial-wise data tensor, shape ``(N, E*K, T)``.
    idx_keep_rois : np.ndarray
        Boolean mask selecting reliable, active ROIs. Shape ``(rois,)``.
    dist_edges : np.ndarray
        Spatial bin edges for a single environment. Shape ``(num_bins + 1,)``.
    generator : PlacefieldDataGenerator
        Sampling-ready generator constructed from this session.
    environments : list[int]
        Sorted list of environment IDs included in ``Y``.
    n_trials_used : int
        Number of trials per environment (equalized across environments).
    """

    P: np.ndarray
    Y: np.ndarray
    idx_keep_rois: np.ndarray
    dist_edges: np.ndarray
    generator: PlacefieldDataGenerator
    environments: list
    n_trials_used: int

    @property
    def true_spectrum(self) -> np.ndarray:
        """Squared singular values of ``P``.

        Returns
        -------
        np.ndarray
            Shape ``(min(N, E*K),)``.
        """
        return self.generator.true_spectrum

    @property
    def residual_config(self) -> ResidualConfig:
        """The fitted residual config inside the generator."""
        return self.generator.residual_config

    @property
    def n_neurons(self) -> int:
        return self.P.shape[0]

    @property
    def n_bins(self) -> int:
        """Total position bins including all environments (``E * num_bins``)."""
        return self.P.shape[1]

    @property
    def n_envs(self) -> int:
        return len(self.environments)

    @property
    def n_trials(self) -> int:
        return self.n_trials_used


# ---------------------------------------------------------------------------
# Analysis config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacefieldStructureConfig(AnalysisConfigBase):
    """Fit a placefield noise model and build a :class:`PlacefieldDataGenerator`.

    Extracts a trial-wise data tensor ``Y`` of shape ``(N, E*K, T)`` from a
    session (environments stacked along the position axis), fits the requested
    residual model, and wraps everything in a :class:`PlacefieldDataGenerator`
    ready to produce synthetic data for CV-estimator benchmarks.

    Per-trial placefields have environments interleaved in presentation order.
    Environments with fewer than ``min_trials_per_env`` trials are silently
    skipped; the remaining environments are equalised to the same trial count
    by random subsampling.

    Parameters
    ----------
    num_bins : int
        Number of spatial bins per environment.
    smooth_width : float or None
        Gaussian smoothing width applied per-trial (cm).  ``None`` for raw
        (unsmoothed) per-trial placefields.
    reliability_cutoff : float
        Minimum leave-one-out reliability for ROI inclusion.
    fraction_active_cutoff : float
        Minimum participation fraction for ROI inclusion.
    spks_type : SpksTypes
        Spike type to retrieve from the registry.
    activity_parameters_name : str
        Name of the activity normalisation preset (``"default"``, ``"raw"``,
        ``"preserved"``).  Normalisation statistics are computed on the
        ``"train"`` split and applied to both splits.
    residual_type : str
        Which residual model to fit.  One of:
        ``"stationary_diagonal"``, ``"stationary_low_rank"``,
        ``"heteroscedastic_diagonal"``, ``"poisson_like"``,
        ``"full_covariance"``.
    n_components : int
        Number of latent factors when ``residual_type="stationary_low_rank"``.
    min_trials_per_env : int
        Environments with fewer trials than this are excluded from ``Y``.
    trial_subsample_seed : int
        RNG seed for reproducible trial subsampling across environments.

    Notes
    -----
    :meth:`fit` is cross-validated by default: ``Y`` is built from the
    registry ``"train"`` split and the mean placefield ``P`` used for
    residual computation is estimated from the ``"validation"`` split.
    This prevents noise-dominated residuals that arise when the mean and the
    residuals are computed from the same data.
    """

    schema_version: str = "v1"
    data_config_name: str = "even"

    num_bins: int = 100
    smooth_width: float | None = 5.0
    reliability_cutoff: float = 0.1
    fraction_active_cutoff: float = 0.1
    spks_type: SpksTypes = "oasis"
    activity_parameters_name: str = "raw"
    residual_type: str = "poisson_like"
    n_components: int = 5
    min_trials_per_env: int = 5
    trial_subsample_seed: int = 0

    display_name: ClassVar[str] = "placefield_structure"
    _result_handling: ClassVar[dict[str, str]] = {
        "idx_keep_rois": "skip",
    }

    def validate(self) -> None:
        if self.residual_type not in _RESIDUAL_TYPE_MAP:
            raise ValueError(f"Unknown residual_type {self.residual_type!r}. " f"Valid: {list(_RESIDUAL_TYPE_MAP)}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(f"Unknown activity_parameters_name {self.activity_parameters_name!r}. " f"Available: {list(ACTIVITY_PARAMETERS_NAMES)}")

    @staticmethod
    def _param_grid() -> dict:
        return {
            "residual_type": [
                "stationary_diagonal",
                "poisson_like",
                "full_covariance",
            ],
            "smooth_width": [None, 5.0],
            "activity_parameters_name": list(ACTIVITY_PARAMETERS_NAMES),
        }

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"bins={self.num_bins}",
            f"smooth={self.smooth_width}",
            f"rel={self.reliability_cutoff}",
            f"frac={self.fraction_active_cutoff}",
            f"spks={self.spks_type}",
            f"ap={self.activity_parameters_name}",
            f"residual={self.residual_type}",
            f"nc={self.n_components}",
            f"min_tr={self.min_trials_per_env}",
            f"data_config={self.data_config_name}",
            self.schema_version,
        ]
        return "_".join(parts)

    def _get_split_spks(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        split: str,
    ) -> tuple[np.ndarray, object]:
        """Load neural activity for a single registry split.

        Parameters
        ----------
        session : B2Session
        registry : PopulationRegistry
        split : str
            Registry split name (e.g. ``"train"``, ``"validation"``).

        Returns
        -------
        spks : np.ndarray
            Shape ``(frames, rois)`` — filtered to valid neurons in the
            population, at the timepoints belonging to ``split``.
        frame_behavior : FrameBehavior
            Aligned to the returned frames.
        """
        population, frame_behavior = registry.get_population(session, self.spks_type)
        split_idx = registry.time_split[split]
        idx_within = population.get_split_times(split_idx, within_idx_samples=True)
        spks = np.array(population.data[population.idx_neurons][:, idx_within]).T  # (frames, rois)
        idx_orig = np.array(population.get_split_times(split_idx, within_idx_samples=False))
        return spks, frame_behavior.filter(idx_orig)

    def _apply_activity_params(
        self,
        spks_tr: np.ndarray,
        spks_vl: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalise spike arrays using train-split statistics.

        Parameters
        ----------
        spks_tr : np.ndarray
            Shape ``(frames_tr, rois)``.  Statistics are computed from this.
        spks_vl : np.ndarray
            Shape ``(frames_vl, rois)``.  Same transform applied as for train.

        Returns
        -------
        spks_tr : np.ndarray
        spks_vl : np.ndarray
        """
        ap = get_activity_parameters(self.activity_parameters_name)

        if ap.center:
            mu = spks_tr.mean(axis=0, keepdims=True)
            spks_tr = spks_tr - mu
            spks_vl = spks_vl - mu

        if ap.scale:
            if ap.scale_type in (None, "std"):
                s = spks_tr.std(axis=0, keepdims=True)
                s[s == 0] = 1.0
            elif ap.scale_type == "max":
                s = spks_tr.max(axis=0, keepdims=True)
                s[s == 0] = 1.0
            elif ap.scale_type == "preserve":
                median_std = np.median(spks_tr.std(axis=0))
                s = max(float(median_std), 1e-12)
            elif ap.scale_type == "sqrt":
                s = np.sqrt(spks_tr.std(axis=0, keepdims=True))
                s[s == 0] = 1.0
            else:
                raise ValueError(f"Unsupported scale_type {ap.scale_type!r}")
            spks_tr = spks_tr / s
            spks_vl = spks_vl / s

        return spks_tr, spks_vl

    def _build_P_cv(
        self,
        spks_vl: np.ndarray,
        frame_behavior_vl,
        dist_edges: np.ndarray,
        included_envs: list,
        session: B2Session,
    ) -> np.ndarray:
        """Build cross-validated mean placefield from the validation split.

        Parameters
        ----------
        spks_vl : np.ndarray
            Shape ``(frames_vl, N)`` — validation split, filtered to kept ROIs.
        frame_behavior_vl : FrameBehavior
        dist_edges : np.ndarray
        included_envs : list[int]
            Sorted environment IDs selected by :meth:`_build_Y`.  Only these
            environments are included; ordering matches ``Y``'s position axis.
        session : B2Session

        Returns
        -------
        P_cv : np.ndarray
            Shape ``(N, E*K)``.
        """
        pf_avg = get_placefield(
            spks_vl,
            frame_behavior_vl,
            dist_edges,
            average=True,
            smooth_width=self.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        # pf_avg.placefield: (E_found, K, N)
        K = len(dist_edges) - 1
        N = spks_vl.shape[1]
        env_blocks: list[np.ndarray] = []
        for env in included_envs:
            mask = pf_avg.environment == env
            if np.any(mask):
                env_blocks.append(pf_avg.placefield[mask][0])  # (K, N)
            else:
                warnings.warn(
                    f"Environment {env} not found in validation split; " "using zeros for P_cv.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                env_blocks.append(np.zeros((K, N)))

        # (E*K, N) → (N, E*K)
        return np.concatenate(env_blocks, axis=0).T

    def _select_rois(
        self,
        spks: np.ndarray,
        frame_behavior,
        dist_edges: np.ndarray,
        session: B2Session,
    ) -> np.ndarray:
        """Return boolean mask of reliable, active ROIs.

        Parameters
        ----------
        spks : np.ndarray
            Shape ``(frames, rois)``.
        frame_behavior : FrameBehavior
        dist_edges : np.ndarray
        session : B2Session

        Returns
        -------
        np.ndarray
            Boolean mask, shape ``(rois,)``.
        """
        pf_all = get_placefield(
            spks,
            frame_behavior,
            dist_edges,
            average=False,
            use_fast_sampling=True,
            session=session,
        )
        # pf_all.placefield: (T_all, K, N) → (N, T_all, K) for reliability_loo
        pf_data = np.transpose(pf_all.placefield, (2, 0, 1))
        idx_reliable = reliability_loo(pf_data) >= self.reliability_cutoff
        idx_active = (
            FractionActive.compute(
                pf_data,
                activity_axis=2,
                fraction_axis=1,
                activity_method="rms",
                fraction_method="participation",
            )
            >= self.fraction_active_cutoff
        )
        return idx_reliable & idx_active

    def _build_Y(
        self,
        spks: np.ndarray,
        frame_behavior,
        dist_edges: np.ndarray,
        session: B2Session,
    ) -> tuple[np.ndarray, list, int]:
        """Build trial-wise data tensor ``Y`` of shape ``(N, E*K, T)``.

        Environments are stacked along the position axis in sorted order,
        mirroring what :meth:`vrAnalysis.processors.placefields.Placefield.flattened`
        does for averaged data.  Trial counts are equalised across environments
        by random subsampling.

        Parameters
        ----------
        spks : np.ndarray
            Shape ``(frames, N)`` — already filtered to kept ROIs.
        frame_behavior : FrameBehavior
        dist_edges : np.ndarray
        session : B2Session

        Returns
        -------
        Y : np.ndarray
            Shape ``(N, E*K, T)``.
        included_envs : list[int]
            Sorted environment IDs that met ``min_trials_per_env``.
        T_eq : int
            Equalized trial count per environment.
        """
        pf_all = get_placefield(
            spks,
            frame_behavior,
            dist_edges,
            average=False,
            smooth_width=self.smooth_width,
            use_fast_sampling=True,
            session=session,
        )
        # pf_all.placefield: (T_all, K, N)
        # pf_all.environment: (T_all,) — environment label for each trial row

        unique_envs = np.unique(pf_all.environment)
        env_row_indices: dict[int, np.ndarray] = {}
        for env in unique_envs:
            rows = np.where(pf_all.environment == env)[0]
            if len(rows) >= self.min_trials_per_env:
                env_row_indices[int(env)] = rows

        if not env_row_indices:
            counts = {int(e): int(np.sum(pf_all.environment == e)) for e in unique_envs}
            raise ValueError(f"No environment has >= {self.min_trials_per_env} trials. " f"Counts per environment: {counts}")

        included_envs = sorted(env_row_indices)
        T_eq = min(len(rows) for rows in env_row_indices.values())
        rng = np.random.default_rng(self.trial_subsample_seed)

        # Build one (T_eq, K, N) block per environment, then stack along K axis
        env_blocks: list[np.ndarray] = []
        for env in included_envs:
            rows = env_row_indices[env]
            chosen = rng.choice(rows, size=T_eq, replace=False)
            chosen.sort()  # preserve chronological order within each env
            env_blocks.append(pf_all.placefield[chosen])  # (T_eq, K, N)

        # Stack environments along K: (T_eq, E*K, N) → (N, E*K, T_eq)
        Y_stacked = np.concatenate(env_blocks, axis=1)
        Y = np.transpose(Y_stacked, (2, 1, 0))
        return Y, included_envs, T_eq

    def fit(
        self,
        session: B2Session,
        registry: PopulationRegistry,
        nuisance_config: NuisanceConfig | None = None,
    ) -> PlacefieldStructureFit:
        """Fit a noise model from session data and build a generator.

        Cross-validated by default: ``Y`` is built from the ``"train"`` registry
        split and the mean placefield ``P`` used for residual computation is
        estimated on the ``"validation"`` split to avoid noise-dominated
        residuals.

        Parameters
        ----------
        session : B2Session
        registry : PopulationRegistry
        nuisance_config : NuisanceConfig or None
            Optional position-independent nuisance component to attach to the
            generator.

        Returns
        -------
        PlacefieldStructureFit
            Contains ``P``, ``Y`` of shape ``(N, E*K, T)``, ``idx_keep_rois``,
            ``dist_edges``, ``environments``, ``n_trials_used``, and the
            ``generator`` ready to call ``generator.sample(n_trials)``.
        """
        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)

        # --- load both splits ------------------------------------------------
        spks_tr, fb_tr = self._get_split_spks(session, registry, "train")
        spks_vl, fb_vl = self._get_split_spks(session, registry, "validation")

        # --- normalise (stats from train split) -------------------------------
        spks_tr, spks_vl = self._apply_activity_params(spks_tr, spks_vl)

        # --- ROI selection on train data -------------------------------------
        idx_keep_rois = self._select_rois(spks_tr, fb_tr, dist_edges, session)
        spks_tr_roi = spks_tr[:, idx_keep_rois]
        spks_vl_roi = spks_vl[:, idx_keep_rois]

        # --- build trial-wise Y from train split -----------------------------
        Y, included_envs, T_eq = self._build_Y(spks_tr_roi, fb_tr, dist_edges, session)

        # --- cross-validated mean from validation split ----------------------
        P_cv = self._build_P_cv(spks_vl_roi, fb_vl, dist_edges, included_envs, session)

        # --- fit residual model with CV mean ---------------------------------
        residual_cls = _RESIDUAL_TYPE_MAP[self.residual_type]
        if self.residual_type == "stationary_low_rank":
            residual_config: ResidualConfig = residual_cls.from_data(Y, P=P_cv, n_components=self.n_components)
        else:
            residual_config = residual_cls.from_data(Y, P=P_cv)

        generator = PlacefieldDataGenerator(P_cv, residual_config, nuisance_config)
        return PlacefieldStructureFit(
            P=P_cv,
            Y=Y,
            idx_keep_rois=idx_keep_rois,
            dist_edges=dist_edges,
            generator=generator,
            environments=included_envs,
            n_trials_used=T_eq,
        )

    def process(
        self,
        session: B2Session,
        registry: PopulationRegistry,
    ) -> dict:
        """Fit noise model and return storable summary statistics.

        Returns
        -------
        dict
            ``true_spectrum`` : np.ndarray
                Squared singular values of ``P``, shape ``(min(N, K),)``.
            ``pca_spectrum`` : np.ndarray
                Empirical PCA eigenvalues of centred, unfolded ``Y``, shape
                ``(min(N, K*T),)`` — the raw observed spectrum before any
                noise correction.
            ``n_neurons`` : int
            ``n_bins`` : int
            ``n_trials`` : int
            ``idx_keep_rois`` : np.ndarray
                Boolean mask over the full population. Shape ``(rois,)``.
        """
        fit = self.fit(session, registry)
        N, K, T = fit.Y.shape
        X = fit.Y.reshape(N, K * T)
        X_c = X - X.mean(axis=1, keepdims=True)
        _, s_emp, _ = np.linalg.svd(X_c, full_matrices=False)
        pca_spectrum = (s_emp**2) / max(K * T - 1, 1)

        return {
            "true_spectrum": fit.true_spectrum,
            "pca_spectrum": pca_spectrum,
            "n_neurons": N,
            "n_bins": K,
            "n_envs": fit.n_envs,
            "n_trials": T,
            "idx_keep_rois": fit.idx_keep_rois,
        }
