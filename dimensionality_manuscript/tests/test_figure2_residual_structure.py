from unittest.mock import patch

import numpy as np

from dimensionality_manuscript.figure_scripts.figure2.residual_structure import (
    logistic_multiplicative_residual,
    mean_by_unit,
    pairwise_nan_covariance,
    prediction_environment_sort,
    rastermap_sort,
)


def test_logistic_multiplicative_residual_is_centered_on_unit_gain():
    gain = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 0.0, -1.0])
    transformed = logistic_multiplicative_residual(gain)
    np.testing.assert_allclose(transformed[:5], [-0.6, -1 / 3, 0.0, 1 / 3, 0.6])
    assert np.all(np.isnan(transformed[5:]))


def test_mean_by_unit_preserves_roi_rows_and_ignores_nan():
    data = np.array([[1.0, 3.0, 10.0, 14.0], [np.nan, 2.0, 4.0, 8.0]])
    result = mean_by_unit(data, np.array([0, 0, 1, 1]), 2)
    np.testing.assert_allclose(result, [[2.0, 12.0], [2.0, 6.0]])


def test_pairwise_nan_covariance_matches_numpy_without_missing_values():
    data = np.array([[1.0, 2.0, 4.0, 8.0], [2.0, 1.0, 3.0, 7.0]])
    np.testing.assert_allclose(pairwise_nan_covariance(data), np.cov(data))


def test_pairwise_nan_covariance_uses_shared_finite_samples():
    data = np.array([[1.0, 2.0, np.nan, 4.0], [1.0, np.nan, 3.0, 5.0]])
    result = pairwise_nan_covariance(data)
    np.testing.assert_allclose(result[0, 1], np.cov([1.0, 4.0], [1.0, 5.0])[0, 1])
    np.testing.assert_allclose(result, result.T)


def test_prediction_environment_sort_uses_environment_then_peak_position():
    position = np.tile(np.arange(4), 2)
    environment = np.repeat([10, 20], 4)
    prediction = np.array(
        [
            [0, 0, 0, 0, 0, 0, 5, 0],  # environment 20, position 2
            [0, 0, 3, 0, 0, 0, 0, 0],  # environment 10, position 2
            [0, 4, 0, 0, 0, 0, 0, 0],  # environment 10, position 1
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(
        prediction_environment_sort(prediction, position, environment, num_position_bins=4),
        [2, 1, 0],
    )


def test_rastermap_sort_excludes_nan_and_constant_rows_from_fit():
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],
            [4.0, 4.0, 4.0],
            [3.0, np.nan, 1.0],
        ]
    )

    class _FakeRastermap:
        def fit(self, fitted):
            assert np.all(np.isfinite(fitted))
            assert fitted.shape == (2, 3)
            self.isort = np.array([1, 0])
            return self

    with patch("dimensionality_manuscript.figure_scripts.figure2.residual_structure.Rastermap", _FakeRastermap):
        np.testing.assert_array_equal(rastermap_sort(data), [3, 0, 1, 2])
