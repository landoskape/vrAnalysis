from __future__ import annotations

import numpy as np
import pytest

from dimensionality_manuscript.configs.regression import (
    RegressionConfig,
    RegressionPlacefieldResidualConfig,
    _placefield_weighted_residual_rms,
)


def test_placefield_weighted_residual_rms_separates_inside_and_outside():
    residual = np.array([[1.0, 2.0, 3.0]])
    pf_prediction = np.array([[1.0, 0.5, 0.0]])

    within, outside = _placefield_weighted_residual_rms(
        residual,
        pf_prediction,
        placefield_peak=np.array([1.0]),
    )

    assert within == pytest.approx([np.sqrt(2.0)])
    assert outside == pytest.approx([np.sqrt(11.0 / 1.5)])


def test_placefield_weighted_residual_rms_ignores_nan_frames_per_roi():
    residual = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    pf_prediction = np.array([[1.0, 0.5, 0.0], [np.nan, 1.0, 0.0]])

    within, outside = _placefield_weighted_residual_rms(
        residual,
        pf_prediction,
        placefield_peak=np.array([1.0, 1.0]),
    )

    assert within == pytest.approx([1.0, 5.0])
    assert outside == pytest.approx([3.0, 6.0])


def test_placefield_weighted_residual_rms_marks_undefined_regions_nan():
    residual = np.ones((2, 3))
    pf_prediction = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])

    within, outside = _placefield_weighted_residual_rms(
        residual,
        pf_prediction,
        placefield_peak=np.array([0.0, 2.0]),
    )

    assert np.isnan(within[0])
    assert np.isnan(outside[0])
    assert within[1] == pytest.approx(1.0)
    assert np.isnan(outside[1])


def test_placefield_residual_grid_mirrors_regression_grid():
    regression = {
        (cfg.model_name, cfg.activity_parameters_name)
        for cfg in RegressionConfig.generate_variations()
    }
    residual = {
        (cfg.model_name, cfg.activity_parameters_name)
        for cfg in RegressionPlacefieldResidualConfig.generate_variations()
    }
    assert residual == regression


def test_placefield_residual_quality_thresholds_are_fixed_defaults_not_grid_axes():
    config = RegressionPlacefieldResidualConfig()
    assert config.reliability_threshold == pytest.approx(0.3)
    assert config.fraction_active_threshold == pytest.approx(0.1)
    assert "reliability_threshold" not in config._param_grid()
    assert "fraction_active_threshold" not in config._param_grid()
