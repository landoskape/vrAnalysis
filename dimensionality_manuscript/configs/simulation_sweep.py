"""AnalysisConfig wrappers for the simulation parameter sweeps.

Each config class flattens one simulation type into a naive Cartesian-product
grid so ResultsAggregator works without modification.

``SimulationSession`` is a stub satisfying the B2Session duck-type required by
``AnalysisPlan._execute_job``.  It lives here because it is only ever needed
by these configs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import torch

from ..pipeline.base import AnalysisConfigBase
from ..simulations import (
    PlacefieldFullConfig,
    PlacefieldFullGenerator,
    SmoothGPFieldConfig,
    StimFullConfig,
    StimFullGenerator,
    ThresholdedGPFieldConfig,
    TilburyFieldConfig,
    kappa_modes,
    sqrtm_spd,
)
from ..simulations.shared_variance import cv_kappa_modes

# ---------------------------------------------------------------------------
# Dummy session
# ---------------------------------------------------------------------------


@dataclass
class SimulationSession:
    """Minimal stub satisfying the B2Session duck-type for simulation configs.

    ``AnalysisPlan._execute_job`` sets ``session.params.spks_type`` and calls
    ``session.clear_cache()``.  Both are handled here without side-effects.
    """

    session_uid: str = "simulation_sweep"
    mouse_name: str = "simulation"
    params: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(spks_type="none"))

    def clear_cache(self, file_names=None):
        pass


SIMULATION_SESSION = SimulationSession()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _to_g(x: torch.Tensor) -> torch.Tensor:
    """Center rows and scale so x @ x.T equals the sample covariance. (N, K) -> (N, K)."""
    return (x - x.mean(dim=1, keepdim=True)) / (x.shape[1] - 1) ** 0.5


def _cvsvd(stim_train: torch.Tensor, stim_test: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Cross-validated singular values of stim_train.T @ proj, scored on stim_test.T @ proj."""
    cross_train = (stim_train.T @ proj).numpy()
    cross_test = (stim_test.T @ proj).numpy()
    U, _, Vt = np.linalg.svd(cross_train, full_matrices=False)
    return torch.from_numpy(np.sum(U * (cross_test @ Vt.T), axis=0).copy())


def _direct_svd(A: torch.Tensor, B: torch.Tensor, n_components: int | None = None) -> torch.Tensor:
    """Singular values of A.T @ B. Pass n_components to use randomized SVD."""
    cross = (A.T @ B).numpy()
    if n_components is not None:
        from sklearn.utils.extmath import randomized_svd as _rsvd

        return torch.from_numpy(_rsvd(cross, n_components=n_components)[1].copy())
    return torch.from_numpy(np.linalg.svd(cross, full_matrices=False, compute_uv=False).copy())


def _mean_stack(tensors: list[torch.Tensor]) -> npt.NDArray[np.floating]:
    return torch.mean(torch.stack(tensors), dim=0).numpy()


def _sorted_eigvals(M: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return np.sort(np.linalg.eigvalsh(M))[::-1].copy()


def _compute_stim_means(
    data: npt.NDArray[np.floating],
    stim_indices: npt.NDArray[np.intp],
    num_stimuli: int,
) -> npt.NDArray[np.floating]:
    """Trial-average data per stimulus/position. Returns (N, num_stimuli)."""
    N = data.shape[0]
    out = np.zeros((N, num_stimuli), dtype=data.dtype)
    for s in range(num_stimuli):
        mask = stim_indices == s
        out[:, s] = data[:, mask].mean(axis=1)
    return out


def _svd_estimators(
    sm: list[torch.Tensor],
    ga: list[torch.Tensor],
    n_neurons: int,
) -> dict[str, npt.NDArray[np.floating]]:
    """Five SVD estimators from 4 independent (stim_means, data) draw pairs.

    sm[i] : (N, K) torch float32 — _to_g(stim_means[i])
    ga[i] : (N, T) torch float32 — _to_g(data[i])
    """
    combos3 = list(itertools.combinations(range(4), 3))
    pairs = list(itertools.combinations(range(4), 2))
    return {
        "empirical_svd_ss_cv": _mean_stack([_cvsvd(sm[i], sm[j], sm[k]) for i, j, k in combos3]),
        "empirical_svd_sf_cv": _mean_stack([_cvsvd(sm[i], sm[j], ga[k]) for i, j, k in combos3]),
        "empirical_svd_ss": _mean_stack([_direct_svd(sm[i], sm[j]) for i, j in pairs]),
        "empirical_svd_sf": _mean_stack([_direct_svd(sm[i], ga[j]) for i, j in pairs]),
        "empirical_svd_ff": _mean_stack([_direct_svd(ga[i], ga[j], n_neurons) for i, j in pairs]),
    }


def _per_draw_covariances(
    stim_means_list: list[npt.NDArray[np.floating]],
    data_list: list[npt.NDArray[np.floating]],
) -> tuple[list[npt.NDArray[np.floating]], list[npt.NDArray[np.floating]]]:
    """Sample covariance matrices per draw. Returns (sigma_stim_list, sigma_full_list)."""
    sigma_stim_list, sigma_full_list = [], []
    for stim_means, data in zip(stim_means_list, data_list):
        K = stim_means.shape[1]
        T = data.shape[1]
        stim_c = stim_means - stim_means.mean(axis=1, keepdims=True)
        data_c = data - data.mean(axis=1, keepdims=True)
        sigma_stim_list.append(stim_c @ stim_c.T / (K - 1))
        sigma_full_list.append(data_c @ data_c.T / (T - 1))
    return sigma_stim_list, sigma_full_list


def _kappa_estimators(
    stim_means_list: list[npt.NDArray[np.floating]],
    data_list: list[npt.NDArray[np.floating]],
) -> dict[str, npt.NDArray[np.floating]]:
    """Cross-validated kappa modes averaged over draw pairs (0,1) and (2,3).

    Within each pair, fold 1 provides the A-root, fold 2 provides the B-root.
    Stim and full covariances are estimated from the same fold.
    """
    sigma_stim_list, sigma_full_list = _per_draw_covariances(stim_means_list, data_list)
    L_s = [sqrtm_spd(s) for s in sigma_stim_list]
    L_f = [sqrtm_spd(s) for s in sigma_full_list]

    return {
        "empirical_kappa_ss": 0.5 * (cv_kappa_modes(L_s[0], L_s[1]) + cv_kappa_modes(L_s[2], L_s[3])),
        "empirical_kappa_sf": 0.5 * (cv_kappa_modes(L_s[0], L_f[1]) + cv_kappa_modes(L_s[2], L_f[3])),
        "empirical_kappa_ff": 0.5 * (cv_kappa_modes(L_f[0], L_f[1]) + cv_kappa_modes(L_f[2], L_f[3])),
    }


def _oracle_metrics(
    sigma_stim: npt.NDArray[np.floating],
    sigma_nuisance: npt.NDArray[np.floating],
    sigma_eps: npt.NDArray[np.floating],
    noise_variance: float,
) -> dict[str, npt.NDArray[np.floating]]:
    """Oracle eigenvalue spectra, kappa modes, and SVD from true covariance matrices."""
    N = sigma_stim.shape[0]
    sigma_sn = sigma_stim + sigma_nuisance
    sigma_full = sigma_sn + sigma_eps + noise_variance * np.eye(N)

    L_s = sqrtm_spd(sigma_stim)
    L_f = sqrtm_spd(sigma_full)

    return {
        # Eigenvalue spectra
        "oracle_lambda_stim": _sorted_eigvals(sigma_stim),
        "oracle_lambda_stim_nuisance": _sorted_eigvals(sigma_sn),
        "oracle_lambda_full": _sorted_eigvals(sigma_full),
        # Kappa modes: sqrt(eigvals(sqrtm(A) @ B @ sqrtm(A)))
        "oracle_kappa_ss": kappa_modes(sigma_stim, sigma_stim),
        "oracle_kappa_sf": kappa_modes(sigma_stim, sigma_full),
        "oracle_kappa_ff": kappa_modes(sigma_full, sigma_full),
        # SVD of matrix square root products (oracle_svd_sf == oracle_kappa_sf)
        "oracle_svd_ss": np.linalg.svd(L_s @ L_s, compute_uv=False),
        "oracle_svd_sf": np.linalg.svd(L_s @ L_f, compute_uv=False),
        "oracle_svd_ff": np.linalg.svd(L_f @ L_f, compute_uv=False),
    }


# ---------------------------------------------------------------------------
# StimFull
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StimFullSweepConfig(AnalysisConfigBase):
    """Sweep over StimFullConfig parameters."""

    display_name: ClassVar[str] = "stim_full_sweep"
    _result_handling: ClassVar[dict] = {}

    # --- grid axes ---
    num_neurons: int = 200
    stim_dim: int = 5
    alpha_stim: float = 1.0
    nuisance_dim: int = 5
    nuisance_scale: float = 0.1
    nuisance_alignment: str = "orthogonal"
    noise_scale: float = 0.0

    # --- fixed (not in grid) ---
    num_stimuli: int = 100
    alpha_nuisance: float = 1.0

    # --- run control (not in grid) ---
    num_samples: int = 5000
    base_seed: int = 0
    noise_variance: float = 0.0

    @staticmethod
    def _param_grid() -> dict:
        return {
            "num_neurons": [200, 500, 1000],
            "stim_dim": [5, 10, 20],
            "alpha_stim": [1.0, 3.0],
            "nuisance_dim": [5, 20],
            "nuisance_scale": [0.1, 0.5, 1.0, 2.0],
            "nuisance_alignment": ["orthogonal", "random"],
            "noise_scale": [0.0, 0.1, 0.3, 1.0],
        }

    def process(self, session, registry) -> dict:
        cfg = StimFullConfig(
            num_neurons=self.num_neurons,
            num_stimuli=self.num_stimuli,
            stim_dim=self.stim_dim,
            alpha_stim=self.alpha_stim,
            nuisance_dim=self.nuisance_dim,
            alpha_nuisance=self.alpha_nuisance,
            nuisance_scale=self.nuisance_scale,
            nuisance_alignment=self.nuisance_alignment,
            noise_scale=self.noise_scale,
            rng=np.random.default_rng(self.base_seed),
        )
        generator = StimFullGenerator(cfg)

        sigma_stim, sigma_nuisance, sigma_eps = generator.true_covariance()
        oracle = _oracle_metrics(sigma_stim, sigma_nuisance, sigma_eps, self.noise_variance)

        draws = [generator.generate(self.num_samples, noise_variance=self.noise_variance, return_extras=True) for _ in range(4)]
        data_list = [d[0] for d in draws]
        stim_means_list = [_compute_stim_means(d[0], d[2]["stim_indices"], self.num_stimuli) for d in draws]

        sm = [_to_g(torch.tensor(s, dtype=torch.float32)) for s in stim_means_list]
        ga = [_to_g(torch.tensor(d, dtype=torch.float32)) for d in data_list]

        return {
            **oracle,
            **_svd_estimators(sm, ga, self.num_neurons),
            **_kappa_estimators(stim_means_list, data_list),
            "n_neurons": self.num_neurons,
            "n_stimuli": self.num_stimuli,
        }


# ---------------------------------------------------------------------------
# Shared placefield base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlacefieldSweepBase(AnalysisConfigBase):
    """Shared fields and process() logic for all placefield sweep configs.

    Subclasses must implement ``_param_grid()`` (adding their field-model axes)
    and ``_build_field_model()``.  This class is never registered or instantiated
    directly.
    """

    _result_handling: ClassVar[dict] = {}

    # --- shared grid axes ---
    num_neurons: int = 200
    repeat_noise_alpha: float = 0.0
    noise_level: float = 0.0
    nuisance_scale: float = 0.0

    # --- fixed ---
    num_positions: int = 100
    alpha_nuisance: float = 1.0

    # --- run control ---
    num_samples: int = 5000
    base_seed: int = 0
    noise_variance: float = 0.0

    @staticmethod
    def _shared_pf_grid() -> dict:
        return {
            "num_neurons": [200, 500, 1000],
            "repeat_noise_alpha": [0.0, 0.3, 1.0],
            "noise_level": [0.0, 0.3, 1.0],
            "nuisance_scale": [0.0, 0.3, 1.0, 3.0],
        }

    def _build_field_model(self):
        raise NotImplementedError

    def process(self, session, registry) -> dict:
        if self.num_samples % self.num_positions != 0:
            raise ValueError(f"num_samples ({self.num_samples}) must be a multiple of num_positions ({self.num_positions})")
        n_repeats = self.num_samples // self.num_positions
        cfg = PlacefieldFullConfig(
            field_model=self._build_field_model(),
            num_neurons=self.num_neurons,
            num_positions=self.num_positions,
            n_repeats=n_repeats,
            repeat_noise_alpha=self.repeat_noise_alpha,
            alpha_nuisance=self.alpha_nuisance,
            nuisance_scale=self.nuisance_scale,
            noise_level=self.noise_level,
            rng=np.random.default_rng(self.base_seed),
        )
        generator = PlacefieldFullGenerator(cfg)

        sigma_stim, sigma_nuisance, sigma_eps = generator.true_covariance()
        oracle = _oracle_metrics(sigma_stim, sigma_nuisance, sigma_eps, self.noise_variance)

        draws = [generator.generate(self.num_samples, noise_variance=self.noise_variance, return_extras=True) for _ in range(4)]
        data_list = [d[0] for d in draws]
        stim_means_list = [_compute_stim_means(d[0], d[2]["stim_indices"], self.num_positions) for d in draws]

        sm = [_to_g(torch.tensor(s, dtype=torch.float32)) for s in stim_means_list]
        ga = [_to_g(torch.tensor(d, dtype=torch.float32)) for d in data_list]

        return {
            **oracle,
            **_svd_estimators(sm, ga, self.num_neurons),
            **_kappa_estimators(stim_means_list, data_list),
            "n_neurons": self.num_neurons,
            "n_stimuli": self.num_positions,
        }


# ---------------------------------------------------------------------------
# ThresholdedGP placefield
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdedGPSweepConfig(_PlacefieldSweepBase):
    """Sweep over PlacefieldFullConfig + ThresholdedGPFieldConfig parameters."""

    display_name: ClassVar[str] = "placefield_thresholded_sweep"

    # --- field-model grid axes ---
    lengthscale: float = 4.0
    threshold_pct: float = 30.0

    # --- fixed ---
    amplitude: float = 2.0

    @staticmethod
    def _param_grid() -> dict:
        return {
            **_PlacefieldSweepBase._shared_pf_grid(),
            "lengthscale": [4.0, 8.0],
            "threshold_pct": [30.0, 60.0],
        }

    def _build_field_model(self):
        return ThresholdedGPFieldConfig(
            lengthscale=self.lengthscale,
            threshold_pct=self.threshold_pct,
            amplitude=self.amplitude,
        )


# ---------------------------------------------------------------------------
# SmoothGP placefield
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothGPSweepConfig(_PlacefieldSweepBase):
    """Sweep over PlacefieldFullConfig + SmoothGPFieldConfig parameters."""

    display_name: ClassVar[str] = "placefield_smooth_sweep"

    # --- field-model grid axes ---
    lengthscale: float = 4.0

    # --- fixed ---
    amplitude: float = 2.0

    @staticmethod
    def _param_grid() -> dict:
        return {
            **_PlacefieldSweepBase._shared_pf_grid(),
            "lengthscale": [4.0, 8.0],
        }

    def _build_field_model(self):
        return SmoothGPFieldConfig(lengthscale=self.lengthscale, amplitude=self.amplitude)


# ---------------------------------------------------------------------------
# Tilbury placefield
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TilburySweepConfig(_PlacefieldSweepBase):
    """Sweep over PlacefieldFullConfig + TilburyFieldConfig parameters."""

    display_name: ClassVar[str] = "placefield_tilbury_sweep"

    # --- field-model grid axes ---
    sigma_mean: float = 4.0
    exponent_mean: float = 1.0
    exponent_spread: float = 0.0

    # --- fixed ---
    amplitude_mean: float = 2.0

    @staticmethod
    def _param_grid() -> dict:
        return {
            **_PlacefieldSweepBase._shared_pf_grid(),
            "sigma_mean": [4.0, 8.0],
            "exponent_mean": [1.0, 2.0, 2.5],
            "exponent_spread": [0.0, 0.5],
        }

    def _build_field_model(self):
        return TilburyFieldConfig(
            amplitude_mean=self.amplitude_mean,
            sigma_mean=self.sigma_mean,
            exponent_mean=self.exponent_mean,
            exponent_spread=self.exponent_spread,
        )
