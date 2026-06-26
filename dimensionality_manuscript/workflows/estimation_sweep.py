"""Manual grid sweeps over SVR estimators: stim_full + three placefield field models.

Builds explicit parameter grids (lists -> itertools.product -> configs), then runs each
grid through ``estimation_sweep.run_manual_sweep``. Edit the lists below to change the grid;
each config gets ``n_seeds`` independent geometry draws.

Placefield ``n_repeats`` is NOT a swept list: it's derived per-config from
``num_samples // num_positions`` (the same quantity that determines repeat count inside
``PlacefieldFullGenerator.generate``), so the rCVPCA estimator's repeat count always matches
what the rest of the pipeline actually uses.

Four grids:
  - stim_full grid       : StimFullConfig product over STIM_* lists
  - thresholded grid     : PlacefieldFullConfig(field_model=ThresholdedGPFieldConfig) over
                            shared PF_* lists x THRESHOLDED_* lists
  - smooth grid           : PlacefieldFullConfig(field_model=SmoothGPFieldConfig) over
                            shared PF_* lists x SMOOTH_* lists
  - tilbury grid          : PlacefieldFullConfig(field_model=TilburyFieldConfig) over
                            shared PF_* lists x TILBURY_* lists
"""

from __future__ import annotations

from itertools import product

import numpy as np

from ..simulations.estimation_sweep import SweepResults, run_manual_sweep
from ..simulations.generators import StimFullConfig
from ..simulations.placefield_full import (
    PlacefieldFullConfig,
    SmoothGPFieldConfig,
    TilburyFieldConfig,
    ThresholdedGPFieldConfig,
)

# ---------------------------------------------------------------------------
# stim_full grid
# ---------------------------------------------------------------------------

STIM_NUM_NEURONS = [200]
STIM_NUM_STIMULI = [100]
STIM_STIM_DIM = [5, 10, 20]
STIM_ALPHA_STIM = [1.0, 3.0]
STIM_NUISANCE_DIM = [5, 20]
STIM_ALPHA_NUISANCE = [1.0]
STIM_NUISANCE_SCALE = [0.1, 0.5, 1.0, 2.0]
STIM_NUISANCE_ALIGNMENT = ["orthogonal", "random"]
STIM_NOISE_SCALE = [0.0, 0.1, 0.3, 1.0]


def build_stim_full_grid() -> tuple[list[StimFullConfig], list[str]]:
    """Build the stim_full parameter grid via itertools.product over the STIM_* lists."""
    configs: list[StimFullConfig] = []
    labels: list[str] = []
    for num_neurons, num_stimuli, stim_dim, alpha_stim, nuisance_dim, alpha_nuisance, nuisance_scale, nuisance_alignment, noise_scale in product(
        STIM_NUM_NEURONS,
        STIM_NUM_STIMULI,
        STIM_STIM_DIM,
        STIM_ALPHA_STIM,
        STIM_NUISANCE_DIM,
        STIM_ALPHA_NUISANCE,
        STIM_NUISANCE_SCALE,
        STIM_NUISANCE_ALIGNMENT,
        STIM_NOISE_SCALE,
    ):
        configs.append(
            StimFullConfig(
                num_neurons=num_neurons,
                num_stimuli=num_stimuli,
                stim_dim=stim_dim,
                alpha_stim=alpha_stim,
                nuisance_dim=nuisance_dim,
                alpha_nuisance=alpha_nuisance,
                nuisance_scale=nuisance_scale,
                nuisance_alignment=nuisance_alignment,
                noise_scale=noise_scale,
            )
        )
        labels.append(f"stim_dim={stim_dim},nuisance_dim={nuisance_dim},nuisance_scale={nuisance_scale}," f"nuisance_alignment={nuisance_alignment}")
    return configs, labels


# ---------------------------------------------------------------------------
# placefield grids: shared geometry params + per-field-model special params
# ---------------------------------------------------------------------------

PF_NUM_NEURONS = [200]
PF_NUM_POSITIONS = [100]
PF_REPEAT_NOISE_ALPHA = [0.0, 0.3, 1.0]
PF_NOISE_LEVEL = [0.0, 0.3, 1.0]
PF_NUISANCE_SCALE = [0.0, 0.3, 1.0, 3.0]

THRESHOLDED_LENGTHSCALE = [4.0, 8.0]
THRESHOLDED_THRESHOLD_PCT = [30.0, 60.0]
THRESHOLDED_AMPLITUDE = [2.0]

SMOOTH_LENGTHSCALE = [4.0, 8.0]
SMOOTH_AMPLITUDE = [2.0]

TILBURY_AMPLITUDE_MEAN = [2.0]
TILBURY_SIGMA_MEAN = [4.0, 8.0]
TILBURY_EXPONENT_MEAN = [1.0, 2.0, 2.5]
TILBURY_EXPONENT_SPREAD = [0.0, 0.5]


def _shared_pf_grid():
    return product(PF_NUM_NEURONS, PF_NUM_POSITIONS, PF_REPEAT_NOISE_ALPHA, PF_NOISE_LEVEL, PF_NUISANCE_SCALE)


def _derive_n_repeats(num_samples: int, num_positions: int) -> int:
    """n_repeats (used by the rCVPCA estimator) derived from num_samples // num_positions."""
    if num_samples % num_positions != 0:
        raise ValueError(f"num_samples ({num_samples}) must be a multiple of num_positions ({num_positions})")
    return num_samples // num_positions


def build_thresholded_grid(num_samples: int) -> tuple[list[PlacefieldFullConfig], list[str]]:
    """Build the thresholded-GP placefield grid: shared PF_* lists x THRESHOLDED_* lists."""
    configs: list[PlacefieldFullConfig] = []
    labels: list[str] = []
    for (num_neurons, num_positions, repeat_noise_alpha, noise_level, nuisance_scale), (
        lengthscale,
        threshold_pct,
        amplitude,
    ) in product(_shared_pf_grid(), product(THRESHOLDED_LENGTHSCALE, THRESHOLDED_THRESHOLD_PCT, THRESHOLDED_AMPLITUDE)):
        configs.append(
            PlacefieldFullConfig(
                field_model=ThresholdedGPFieldConfig(lengthscale=lengthscale, threshold_pct=threshold_pct, amplitude=amplitude),
                num_neurons=num_neurons,
                num_positions=num_positions,
                n_repeats=_derive_n_repeats(num_samples, num_positions),
                repeat_noise_alpha=repeat_noise_alpha,
                noise_level=noise_level,
                nuisance_scale=nuisance_scale,
            )
        )
        labels.append(
            f"thresholded:lengthscale={lengthscale},threshold_pct={threshold_pct}," f"noise_level={noise_level},nuisance_scale={nuisance_scale}"
        )
    return configs, labels


def build_smooth_grid(num_samples: int) -> tuple[list[PlacefieldFullConfig], list[str]]:
    """Build the smooth-GP placefield grid: shared PF_* lists x SMOOTH_* lists."""
    configs: list[PlacefieldFullConfig] = []
    labels: list[str] = []
    for (num_neurons, num_positions, repeat_noise_alpha, noise_level, nuisance_scale), (
        lengthscale,
        amplitude,
    ) in product(_shared_pf_grid(), product(SMOOTH_LENGTHSCALE, SMOOTH_AMPLITUDE)):
        configs.append(
            PlacefieldFullConfig(
                field_model=SmoothGPFieldConfig(lengthscale=lengthscale, amplitude=amplitude),
                num_neurons=num_neurons,
                num_positions=num_positions,
                n_repeats=_derive_n_repeats(num_samples, num_positions),
                repeat_noise_alpha=repeat_noise_alpha,
                noise_level=noise_level,
                nuisance_scale=nuisance_scale,
            )
        )
        labels.append(f"smooth:lengthscale={lengthscale},noise_level={noise_level},nuisance_scale={nuisance_scale}")
    return configs, labels


def build_tilbury_grid(num_samples: int) -> tuple[list[PlacefieldFullConfig], list[str]]:
    """Build the Tilbury placefield grid: shared PF_* lists x TILBURY_* lists."""
    configs: list[PlacefieldFullConfig] = []
    labels: list[str] = []
    for (num_neurons, num_positions, repeat_noise_alpha, noise_level, nuisance_scale), (
        amplitude_mean,
        sigma_mean,
        exponent_mean,
        exponent_spread,
    ) in product(_shared_pf_grid(), product(TILBURY_AMPLITUDE_MEAN, TILBURY_SIGMA_MEAN, TILBURY_EXPONENT_MEAN, TILBURY_EXPONENT_SPREAD)):
        configs.append(
            PlacefieldFullConfig(
                field_model=TilburyFieldConfig(
                    amplitude_mean=amplitude_mean, sigma_mean=sigma_mean, exponent_mean=exponent_mean, exponent_spread=exponent_spread
                ),
                num_neurons=num_neurons,
                num_positions=num_positions,
                n_repeats=_derive_n_repeats(num_samples, num_positions),
                repeat_noise_alpha=repeat_noise_alpha,
                noise_level=noise_level,
                nuisance_scale=nuisance_scale,
            )
        )
        labels.append(
            f"tilbury:sigma_mean={sigma_mean},exponent_mean={exponent_mean},exponent_spread={exponent_spread},"
            f"noise_level={noise_level},nuisance_scale={nuisance_scale}"
        )
    return configs, labels


# ---------------------------------------------------------------------------
# Run all four grids
# ---------------------------------------------------------------------------


def run_all_grids(
    *,
    n_seeds: int = 20,
    num_samples: int = 2000,
    noise_variance: float = 0.0,
    base_seed: int = 0,
    dtype: np.dtype = np.float64,
) -> dict[str, SweepResults]:
    """Run the stim_full grid and the three placefield grids through run_manual_sweep."""
    results: dict[str, SweepResults] = {}
    grid_builders = {
        "stim_full": lambda: build_stim_full_grid(),
        "placefield_thresholded": lambda: build_thresholded_grid(num_samples),
        "placefield_smooth": lambda: build_smooth_grid(num_samples),
        "placefield_tilbury": lambda: build_tilbury_grid(num_samples),
    }
    for name, builder in grid_builders.items():
        configs, labels = builder()
        results[name] = run_manual_sweep(
            configs,
            case_labels=labels,
            n_seeds=n_seeds,
            num_samples=num_samples,
            noise_variance=noise_variance,
            base_seed=base_seed,
            dtype=dtype,
        )
    return results


if __name__ == "__main__":
    run_all_grids()
