"""Tests for the round-the-house empirical estimator."""

import numpy as np
import pytest

from dimensionality_manuscript.simulations import analyze_atlas_case
from dimensionality_manuscript.simulations.shared_variance import (
    ModeComparison,
    NeuronSplit,
    _center_rows,
    _comparison,
    _neuron_split,
    _roundhouse_kernel,
    _roundhouse_modes,
    _roundhouse_modes_asymmetric,
    _roundhouse_modes_from_matrices,
)


def test_neuron_split_midpoint():
    split = _neuron_split(10)
    assert split.source_idx.shape == (5,)
    assert split.target_idx.shape == (5,)
    assert np.array_equal(split.source_idx, np.arange(5))
    assert np.array_equal(split.target_idx, np.arange(5, 10))


def test_neuron_split_requires_two_neurons():
    with pytest.raises(ValueError, match="at least 2 neurons"):
        _neuron_split(1)


def test_center_rows_zeros_row_means():
    matrix = np.array([[1.0, 3.0], [10.0, 20.0]])
    centered = _center_rows(matrix)
    np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=1e-12)


def test_roundhouse_kernel_shape_and_modes():
    rng = np.random.default_rng(0)
    n0, n1, t0, t1 = 4, 5, 30, 25
    m00 = rng.standard_normal((n0, t0))
    m10 = rng.standard_normal((n1, t0))
    m11 = rng.standard_normal((n1, t1))
    m01 = rng.standard_normal((n0, t1))

    kernel = _roundhouse_kernel(m00, m10, m11, m01)
    assert kernel.shape == (n0, n0)

    modes_a = _roundhouse_modes(kernel)
    modes_b = _roundhouse_modes(kernel)
    assert modes_a.shape == (n0,)
    assert np.all(modes_a >= 0.0)
    np.testing.assert_allclose(modes_a, modes_b)


def test_roundhouse_modes_from_matrices_uses_split():
    rng = np.random.default_rng(1)
    n, t0, t1 = 8, 20, 15
    split = _neuron_split(n)
    full0 = rng.standard_normal((n, t0))
    full1 = rng.standard_normal((n, t1))

    modes = _roundhouse_modes_from_matrices(split, full0, full0, full1, full1)
    assert modes.shape == (split.source_idx.shape[0],)
    assert np.all(np.isfinite(modes))


def test_roundhouse_comparison_ratio_one_when_modes_match():
    rng = np.random.default_rng(2)
    n, t0, t1 = 20, 80, 90
    split = _neuron_split(n)
    full0 = rng.standard_normal((n, t0))
    full1 = rng.standard_normal((n, t1))
    modes = _roundhouse_modes_from_matrices(split, full0, full0, full1, full1)
    comparison = _comparison(modes, modes, metric="energy")
    assert comparison.ratio == pytest.approx(1.0)


def test_empirical_roundhouse_smoke_stim_full():
    result = analyze_atlas_case(
        "stim_full.identity",
        seed=0,
        num_samples=4000,
        sample_seed=1,
    )
    assert result.empirical is not None
    assert result.empirical.roundhouse_sym is not None
    assert result.empirical.roundhouse_sym.metric == "energy"
    assert np.isfinite(result.empirical.roundhouse_sym.ratio)
    assert not hasattr(result.population, "roundhouse")
    assert isinstance(result.extras["neuron_split"], NeuronSplit)


def test_empirical_roundhouse_identity_positive_finite():
    """Stim-means lead leg has fewer columns than raw time data, so ratio is not ~1."""
    result = analyze_atlas_case(
        "stim_full.identity",
        seed=0,
        num_samples=8000,
        sample_seed=2,
    )
    assert result.empirical is not None
    assert result.empirical.roundhouse_sym is not None
    assert result.empirical.roundhouse_sym.ratio > 0.0
    assert np.isfinite(result.empirical.roundhouse_sym.ratio)


def test_empirical_roundhouse_context_identical_ratio_near_one():
    result = analyze_atlas_case(
        "context.identical",
        seed=0,
        num_samples=2000,
        sample_seed=3,
    )
    assert result.empirical is not None
    assert result.empirical.roundhouse_sym is not None
    assert result.empirical.roundhouse_sym.ratio == pytest.approx(1.0, rel=0.15, abs=0.05)


def test_empirical_roundhouse_asymmetric_smoke_stim_full():
    result = analyze_atlas_case(
        "stim_full.identity",
        seed=0,
        num_samples=4000,
        sample_seed=1,
    )
    assert result.empirical is not None
    assert result.empirical.roundhouse is not None
    assert result.empirical.roundhouse.metric == "energy_signed"
    assert np.isfinite(result.empirical.roundhouse.ratio)
    assert np.all(np.isfinite(result.empirical.roundhouse.candidate_modes))
    assert np.all(np.isfinite(result.empirical.roundhouse.reference_modes))


def test_roundhouse_modes_asymmetric_keeps_negative_eigenvalues():
    """A real, non-symmetric matrix with a negative real eigenvalue must survive unclipped."""
    # Eigenvalues 4, -3 (trace=1, det=-12), kept asymmetric by construction.
    kernel = np.array([[1.0, 4.0], [3.0, 0.0]])
    modes = _roundhouse_modes_asymmetric(kernel)
    np.testing.assert_allclose(np.sort(modes.real), [-3.0, 4.0])
    assert np.any(modes.real < 0.0)


def test_roundhouse_modes_asymmetric_allows_genuine_complex_eigenvalues():
    """A rotation-like asymmetric matrix has genuinely complex eigenvalues; they must pass through, not raise."""
    kernel = np.array([[0.0, -1.0], [1.0, 0.0]])
    modes = _roundhouse_modes_asymmetric(kernel)
    assert np.iscomplexobj(modes)
    np.testing.assert_allclose(np.sort_complex(modes), [-1j, 1j])


def test_as_variance_scale_sign_preserving_for_energy_signed():
    comparison = ModeComparison(
        candidate_modes=np.array([4.0, -4.0]),
        reference_modes=np.array([9.0, -9.0]),
        ratio=np.nan,
        cumulative_ratio=np.array([np.nan, np.nan]),
        metric="energy_signed",
    )
    scaled = comparison.as_variance_scale()
    np.testing.assert_allclose(scaled.candidate_modes, [2.0, -2.0])
    np.testing.assert_allclose(scaled.reference_modes, [3.0, -3.0])
    assert scaled.metric == "kappa"
