"""Tests for the place-field stim_full atlas generator and its analysis routing."""

import numpy as np
import pytest

from dimensionality_manuscript.simulations import (
    PlacefieldFullConfig,
    PlacefieldFullGenerator,
    SmoothGPFieldConfig,
    ThresholdedGPFieldConfig,
    TilburyFieldConfig,
    process,
)

N = 40
P = 20


def _config(field_model, **kwargs):
    return PlacefieldFullConfig(
        field_model=field_model,
        num_neurons=N,
        num_positions=P,
        n_repeats=4,
        rng=np.random.default_rng(0),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Generator contract
# ---------------------------------------------------------------------------


def test_generate_shapes_and_balanced_labels():
    gen = PlacefieldFullGenerator(_config(SmoothGPFieldConfig()))
    n_repeats = 5
    data, stim_data, extras = gen.generate(n_repeats * P, rng=np.random.default_rng(1), return_extras=True)

    assert data.shape == (N, n_repeats * P)
    assert stim_data.shape == (N, n_repeats * P)
    assert extras["stim_indices"].shape == (n_repeats * P,)
    # Balanced design: every position appears exactly n_repeats times.
    counts = np.bincount(extras["stim_indices"], minlength=P)
    assert np.all(counts == n_repeats)
    assert len(extras["repeat_maps"]) == n_repeats
    assert all(m.shape == (N, P) for m in extras["repeat_maps"])
    # repeat_indices label each column's repeat block.
    assert np.array_equal(extras["repeat_indices"], np.repeat(np.arange(n_repeats), P))


def test_generate_requires_multiple_of_positions():
    gen = PlacefieldFullGenerator(_config(SmoothGPFieldConfig()))
    with pytest.raises(ValueError):
        gen.generate(P + 1)


def test_num_stimuli_matches_positions():
    cfg = _config(SmoothGPFieldConfig())
    assert cfg.num_stimuli == P


# ---------------------------------------------------------------------------
# true_covariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_model", [SmoothGPFieldConfig(), ThresholdedGPFieldConfig(), TilburyFieldConfig()])
def test_true_covariance_shapes_and_psd(field_model):
    gen = PlacefieldFullGenerator(_config(field_model))
    sigma_stim, sigma_nuisance, sigma_eps = gen.true_covariance()

    for sigma in (sigma_stim, sigma_nuisance, sigma_eps):
        assert sigma.shape == (N, N)
        evals = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))
        assert evals.min() > -1e-8  # PSD up to numerical tolerance

    # Stimulus covariance comes from P positions, so its rank cannot exceed min(N, P).
    assert np.linalg.matrix_rank(sigma_stim, tol=1e-8) <= min(N, P)

    # Isotropic nuisance / noise contributions.
    cfg = gen.config
    assert np.allclose(sigma_nuisance, cfg.repeat_noise_alpha**2 * np.eye(N))
    assert np.allclose(sigma_eps, cfg.noise_level**2 * np.eye(N))


def test_stim_space_reconstruction():
    """stim_space @ diag(sqrt(stim_spectrum)) @ stim_latents reconstructs the source."""
    gen = PlacefieldFullGenerator(_config(ThresholdedGPFieldConfig()))
    recon = gen.stim_space @ np.diag(np.sqrt(gen.stim_spectrum)) @ gen.stim_latents
    assert np.allclose(recon, gen.source, atol=1e-6)


# ---------------------------------------------------------------------------
# Smooth vs thresholded structure
# ---------------------------------------------------------------------------


def test_smooth_has_no_hard_zeros_thresholded_does():
    smooth = PlacefieldFullGenerator(_config(SmoothGPFieldConfig()))
    thresholded = PlacefieldFullGenerator(_config(ThresholdedGPFieldConfig()))

    # Smooth GP fields are sign-free (no rectification): they have negative values and no
    # block of exact zeros.
    assert smooth.source.min() < 0.0
    assert not np.any(smooth.source == 0.0)

    # Thresholded fields are rectified: a substantial fraction of entries are exactly zero.
    assert np.mean(thresholded.source == 0.0) > 0.3


# ---------------------------------------------------------------------------
# Analysis routing through the atlas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_model", [SmoothGPFieldConfig(), ThresholdedGPFieldConfig(), TilburyFieldConfig()])
def test_process_populates_population_and_empirical(field_model):
    cfg = _config(field_model)
    result = process(cfg, num_samples=6 * P, sample_seed=0)

    # Population block.
    assert result.population.kappa.ratio == pytest.approx(result.population.kappa.ratio)  # finite
    assert np.isfinite(result.population.kappa.ratio)
    assert np.isfinite(result.population.energy.ratio)
    assert result.population.geometry.cka == pytest.approx(result.population.geometry.cka)

    # Empirical block with all CV variants, including the new rCVPCA estimator.
    emp = result.empirical
    assert emp is not None
    assert np.isfinite(emp.kappa.ratio)
    assert emp.cv_kappa is not None and np.isfinite(emp.cv_kappa.ratio)
    assert emp.cv_stimstim is not None and np.isfinite(emp.cv_stimstim.ratio)
    assert emp.cv_energy is not None and np.isfinite(emp.cv_energy.ratio)
    assert emp.cv_rcvpca is not None and np.isfinite(emp.cv_rcvpca.ratio)


def test_smooth_and_thresholded_give_different_stim_spectra():
    smooth = PlacefieldFullGenerator(_config(SmoothGPFieldConfig()))
    thresholded = PlacefieldFullGenerator(_config(ThresholdedGPFieldConfig()))
    sigma_smooth, _, _ = smooth.true_covariance()
    sigma_thr, _, _ = thresholded.true_covariance()
    spec_smooth = np.sort(np.linalg.eigvalsh(sigma_smooth))[::-1]
    spec_thr = np.sort(np.linalg.eigvalsh(sigma_thr))[::-1]
    assert not np.allclose(spec_smooth, spec_thr)


def test_population_only_when_num_samples_none():
    cfg = _config(SmoothGPFieldConfig())
    result = process(cfg)
    assert result.empirical is None
    assert result.population is not None
