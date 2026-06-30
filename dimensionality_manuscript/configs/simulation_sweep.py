"""AnalysisConfig wrappers for the simulation parameter sweeps.

Each config class flattens one simulation type into a naive Cartesian-product
grid so ResultsAggregator works without modification.

``SimulationSession`` is a stub satisfying the B2Session duck-type required by
``AnalysisPlan._execute_job``.  It lives here because it is only ever needed
by these configs.

RNG note: each ``process()`` call submits one config to ``run_manual_sweep``
(n_cases=1).  The geometry and sample seeds come from
``SeedSequence(base_seed)``; all configs in the same grid therefore draw from
the same seed sequence.  This matches the existing notebook behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import ClassVar, Literal

import numpy as np

from ..pipeline.base import AnalysisConfigBase
from ..simulations import (
    PlacefieldFullConfig,
    SmoothGPFieldConfig,
    StimFullConfig,
    ThresholdedGPFieldConfig,
    TilburyFieldConfig,
)
from ..simulations.estimation_sweep import run_manual_sweep
from ..simulations.shared_variance import ComputeFlags

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
# StimFull
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StimFullSweepConfig(AnalysisConfigBase):
    """Sweep over StimFullConfig parameters.

    Grid size: 3 × 3 × 2 × 2 × 4 × 2 × 4 = 1 152 variations.
    """

    display_name: ClassVar[str] = "stim_full_sweep"
    _result_handling: ClassVar[dict] = {"empirical_block": "skip"}

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
    n_seeds: int = 20
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

    @staticmethod
    def _compute_flags() -> ComputeFlags:
        return ComputeFlags(cv_energy=False, cv_stimstim=False, cv_rcvpca=False, roundhouse=False, mtfa=False)

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
        )
        sweep = run_manual_sweep(
            [cfg],
            n_seeds=self.n_seeds,
            num_samples=self.num_samples,
            base_seed=self.base_seed,
            noise_variance=self.noise_variance,
            compute=self._compute_flags(),
        )
        return {
            "oracle_kappa": sweep.oracle_kappa[0],
            "oracle_energy": sweep.oracle_energy[0],
            "empirical_kappa": sweep.empirical_kappa[0],
            "empirical_cv_kappa": sweep.empirical_cv_kappa[0],
            "empirical_cv_variance_scale": sweep.empirical_cv_variance_scale[0],
            "empirical_block": sweep.empirical_blocks[0],
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

    _result_handling: ClassVar[dict] = {"empirical_block": "skip"}

    # --- shared grid axes ---
    num_neurons: int = 200
    repeat_noise_alpha: float = 0.0
    noise_level: float = 0.0
    nuisance_scale: float = 0.0

    # --- fixed ---
    num_positions: int = 100
    alpha_nuisance: float = 1.0

    # --- run control ---
    n_seeds: int = 20
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

    @staticmethod
    def _compute_flags() -> ComputeFlags:
        return ComputeFlags(cv_energy=False, cv_stimstim=False, roundhouse=False, mtfa=False, cv_rcvpca=False)

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
            nuisance_scale=self.nuisance_scale,
            noise_level=self.noise_level,
        )
        sweep = run_manual_sweep(
            [cfg],
            n_seeds=self.n_seeds,
            num_samples=self.num_samples,
            base_seed=self.base_seed,
            noise_variance=self.noise_variance,
            compute=self._compute_flags(),
        )
        return {
            "oracle_kappa": sweep.oracle_kappa[0],
            "oracle_energy": sweep.oracle_energy[0],
            "empirical_kappa": sweep.empirical_kappa[0],
            "empirical_cv_kappa": sweep.empirical_cv_kappa[0],
            "empirical_cv_variance_scale": sweep.empirical_cv_variance_scale[0],
            "empirical_cv_rcvpca": sweep.empirical_cv_rcvpca[0],
            "empirical_block": sweep.empirical_blocks[0],
        }


# ---------------------------------------------------------------------------
# ThresholdedGP placefield
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdedGPSweepConfig(_PlacefieldSweepBase):
    """Sweep over PlacefieldFullConfig + ThresholdedGPFieldConfig parameters.

    Grid size: 3 × 3 × 3 × 4 × 2 × 2 = 432 variations.
    """

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
    """Sweep over PlacefieldFullConfig + SmoothGPFieldConfig parameters.

    Grid size: 3 × 3 × 3 × 4 × 2 = 216 variations.
    """

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
    """Sweep over PlacefieldFullConfig + TilburyFieldConfig parameters.

    Grid size: 3 × 3 × 3 × 4 × 2 × 3 × 2 = 1 296 variations.
    """

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
