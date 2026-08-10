from __future__ import annotations

import numpy as np
import pytest

from dimilibi import measure_r2

from dimensionality_manuscript.configs.regression import (
    RESIDUAL_FOLDS,
    RESIDUAL_METRICS,
    RESIDUAL_REGIONS,
    RESIDUAL_STATISTICS,
    RESIDUAL_SUBSETS,
    RegressionConfig,
    RegressionPlacefieldResidualConfig,
    _placefield_weighted_residual_rms,
    _residual_fold_metrics,
    _residual_summary_scalars,
    _weighted_mse,
    _weighted_r2,
    residual_per_roi_keys,
    residual_summary_keys,
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


def test_weighted_r2_with_uniform_weights_matches_unweighted_r2():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(5, 60))
    prediction = target + 0.4 * rng.normal(size=(5, 60))

    weighted = _weighted_r2(target, prediction, np.ones_like(target))
    unweighted = np.asarray(measure_r2(prediction, target, reduce="none", dim=1))

    np.testing.assert_allclose(weighted, unweighted, rtol=1e-6)


def test_weighted_r2_matches_hand_computed_value():
    target = np.array([[0.0, 1.0, 2.0, 3.0]])
    prediction = np.array([[0.0, 0.0, 0.0, 0.0]])
    weights = np.array([[3.0, 1.0, 0.0, 0.0]])

    # Weighted mean over the two weighted samples is (3*0 + 1*1) / 4 = 0.25.
    ss_res = 3.0 * 0.0**2 + 1.0 * 1.0**2
    ss_tot = 3.0 * 0.25**2 + 1.0 * 0.75**2

    assert _weighted_r2(target, prediction, weights) == pytest.approx([1.0 - ss_res / ss_tot])


def test_weighted_metrics_are_nan_without_weight_or_variance():
    target = np.array([[1.0, 1.0, 1.0], [0.0, 5.0, 9.0]])
    prediction = np.zeros_like(target)

    # Row 0 has weight but no weighted variance; row 1 has no weight at all.
    weights = np.array([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]])

    assert np.all(np.isnan(_weighted_r2(target, prediction, weights)))
    assert np.isnan(_weighted_mse(target - prediction, weights)[1])


def test_r2_shared_is_one_minus_normalized_rms_squared():
    rng = np.random.default_rng(1)
    target = np.abs(rng.normal(size=(4, 40)))
    prediction = np.abs(rng.normal(size=(4, 40)))
    pf_prediction = np.abs(rng.normal(size=(4, 40)))

    metrics = _residual_fold_metrics(
        target,
        prediction,
        pf_prediction,
        placefield_peak=pf_prediction.max(axis=1),
        variance_pf=np.var(target, axis=1),
    )

    for region in RESIDUAL_REGIONS:
        np.testing.assert_allclose(
            metrics[f"{region}_pf_r2_shared"],
            1.0 - metrics[f"{region}_pf_normalized_rms"] ** 2,
        )


def test_residual_fold_metrics_emit_every_region_metric_and_difference():
    target = np.ones((2, 5))
    prediction = np.zeros((2, 5))
    pf_prediction = np.tile(np.linspace(0.0, 1.0, 5), (2, 1))

    metrics = _residual_fold_metrics(
        target,
        prediction,
        pf_prediction,
        placefield_peak=np.ones(2),
        variance_pf=np.ones(2),
    )

    expected = {f"{region}_pf_{metric}" for region in RESIDUAL_REGIONS for metric in RESIDUAL_METRICS}
    expected |= {f"outside_minus_within_pf_{metric}" for metric in RESIDUAL_METRICS}
    assert set(metrics) == expected
    for metric in RESIDUAL_METRICS:
        np.testing.assert_allclose(
            metrics[f"outside_minus_within_pf_{metric}"],
            metrics[f"outside_pf_{metric}"] - metrics[f"within_pf_{metric}"],
        )


def test_residual_keys_cover_every_fold_region_metric_and_subset():
    per_roi = residual_per_roi_keys()
    assert len(per_roi) == len(RESIDUAL_FOLDS) * (len(RESIDUAL_REGIONS) + 1) * len(RESIDUAL_METRICS) + 1
    for fold in RESIDUAL_FOLDS:
        for metric in RESIDUAL_METRICS:
            for region in RESIDUAL_REGIONS:
                assert f"{fold}_{region}_pf_{metric}" in per_roi
            assert f"{fold}_outside_minus_within_pf_{metric}" in per_roi
    assert "variance_pf" in per_roi

    summary = residual_summary_keys()
    assert len(summary) == len(RESIDUAL_STATISTICS) * len(RESIDUAL_SUBSETS) * len(per_roi)
    assert set(summary) == {
        f"{statistic}_{subset}{key}" for statistic in RESIDUAL_STATISTICS for subset in RESIDUAL_SUBSETS for key in per_roi
    }


def test_residual_summary_scalars_split_on_the_quality_mask():
    per_roi = {"xval_within_pf_rms": np.array([1.0, 3.0, 10.0, np.nan])}
    mask = np.array([True, True, False, False])

    scalars = _residual_summary_scalars(per_roi, mask)

    assert scalars["mean_xval_within_pf_rms"] == pytest.approx(14.0 / 3.0)
    assert scalars["mean_quality_filtered_xval_within_pf_rms"] == pytest.approx(2.0)
    assert scalars["mean_notquality_filtered_xval_within_pf_rms"] == pytest.approx(10.0)


def test_residual_summary_reports_a_median_that_survives_the_r2_tail():
    # One ROI with a near-zero weighted baseline blows its weighted R2 up; the mean follows it
    # off the scale, the median does not.
    per_roi = {"xval_within_pf_r2_weighted": np.array([-0.1, -0.2, -0.3, -1e5])}
    mask = np.ones(4, dtype=bool)

    scalars = _residual_summary_scalars(per_roi, mask)

    assert scalars["mean_xval_within_pf_r2_weighted"] < -100.0
    assert scalars["median_xval_within_pf_r2_weighted"] == pytest.approx(-0.25)


def test_residual_summary_ignores_nan_rois_in_both_statistics():
    per_roi = {"xval_within_pf_rms": np.array([np.nan, 2.0, 4.0, np.nan])}

    scalars = _residual_summary_scalars(per_roi, np.ones(4, dtype=bool))

    assert scalars["mean_xval_within_pf_rms"] == pytest.approx(3.0)
    assert scalars["median_xval_within_pf_rms"] == pytest.approx(3.0)
