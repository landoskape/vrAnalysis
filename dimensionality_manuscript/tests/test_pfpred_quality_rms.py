from types import SimpleNamespace

import numpy as np
import pytest

from dimensionality_manuscript.configs import pfpred_quality
from dimensionality_manuscript.configs.pfpred_quality import (
    _binned_rms,
    _kde_rms,
    _normalized_rms_error,
    _r2_by_slot,
    _rms_error,
    _summarize_trial_activity,
    _summarize_trial_activity_across_rois,
    _trial_activity_by_slot,
    _trial_activity_measurements,
    _variance_components,
    PFPredQualityConfig,
    trial_activity_population_summary_keys,
    trial_activity_summary_keys,
)


def test_rms_error_is_computed_per_roi():
    activity = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    prediction = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 8.0]])

    expected = np.sqrt(np.mean((prediction - activity) ** 2, axis=0))

    np.testing.assert_allclose(_rms_error(prediction, activity), expected)


def test_normalized_rms_divides_by_each_rois_activity_standard_deviation():
    activity = np.array([[0.0, 2.0, 1.0], [2.0, 4.0, 1.0], [4.0, 6.0, 1.0]])
    prediction = np.array([[0.0, 3.0, 0.0], [1.0, 4.0, 1.0], [2.0, 7.0, 2.0]])

    expected = _rms_error(prediction, activity)[:2] / np.std(activity[:, :2], axis=0)
    normalized = _normalized_rms_error(prediction, activity)
    np.testing.assert_allclose(normalized[:2], expected)
    assert np.isnan(normalized[2])


def test_variance_components_are_computed_per_roi():
    activity = np.array([[0.0, 2.0, 1.0], [2.0, 4.0, 1.0], [4.0, 8.0, 1.0]])
    prediction = np.array([[0.0, 3.0, 0.5], [1.0, 4.0, 1.0], [3.0, 6.0, 1.5]])

    result = _variance_components(prediction, activity)

    expected_total = np.var(activity, axis=0)
    expected_pred = np.var(prediction, axis=0)
    expected_residual = np.var(activity - prediction, axis=0)
    assert set(result) == {"var_total", "var_pred", "var_residual", "frac_var_pred"}
    np.testing.assert_allclose(result["var_total"], expected_total)
    np.testing.assert_allclose(result["var_pred"], expected_pred)
    np.testing.assert_allclose(result["var_residual"], expected_residual)
    np.testing.assert_allclose(result["frac_var_pred"][:2], expected_pred[:2] / expected_total[:2])
    assert np.isnan(result["frac_var_pred"][2])


def test_rms_binned_and_kde_results_use_rms_key_names():
    rms = np.array([1.0, 3.0, 5.0, np.nan])
    reliability = np.array([-0.75, -0.25, 0.75, 0.25])
    bin_edges = np.array([-1.0, 0.0, 1.0])
    grid = np.array([-0.5, 0.5])

    binned = _binned_rms(rms, reliability, bin_edges)
    kde = _kde_rms(rms, reliability, grid, bw=0.2)

    assert set(binned) == {"rms_bin_mean", "rms_bin_sem", "rms_bin_n"}
    np.testing.assert_allclose(binned["rms_bin_mean"], [2.0, 5.0])
    np.testing.assert_allclose(binned["rms_bin_n"], [2.0, 1.0])
    assert set(kde) == {"rms_kde_grid", "rms_kde_mean"}
    np.testing.assert_array_equal(kde["rms_kde_grid"], grid)
    assert np.all(np.isfinite(kde["rms_kde_mean"]))


def test_rms_is_propagated_to_slot_and_pooled_kde_results(monkeypatch):
    monkeypatch.setattr(pfpred_quality, "load_env_order", lambda: {"mouse": [11, 22]})
    session = SimpleNamespace(mouse_name="mouse")
    spks = np.array([[0.0, 1.0], [2.0, 3.0], [1.0, 4.0], [3.0, 6.0]])
    prediction = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0], [3.0, 4.0]])
    extras = {
        "idx_valid": np.ones(4, dtype=bool),
        "frame_environment_index": np.array([0, 0, 1, 1]),
    }
    reliability = SimpleNamespace(values=np.array([[0.2, 0.4], [0.6, 0.8]]))
    env_maps = SimpleNamespace(environments=np.array([11, 22]))
    grid = np.array([0.25, 0.75])

    result = _r2_by_slot(session, spks, prediction, extras, reliability, env_maps, best_env=1, kde_grid=grid)

    expected_slot_0 = _rms_error(prediction[:2], spks[:2])
    expected_slot_1 = _rms_error(prediction[2:], spks[2:])
    np.testing.assert_allclose(result["rms_slot"][0], expected_slot_0)
    np.testing.assert_allclose(result["rms_slot"][1], expected_slot_1)
    np.testing.assert_allclose(result["norm_rms_slot"][0], expected_slot_0 / np.std(spks[:2], axis=0))
    np.testing.assert_allclose(result["norm_rms_slot"][1], expected_slot_1 / np.std(spks[2:], axis=0))
    for slot, idx in [(0, slice(None, 2)), (1, slice(2, None))]:
        expected_variance = _variance_components(prediction[idx], spks[idx])
        for key in ("var_total", "var_pred", "var_residual", "frac_var_pred"):
            np.testing.assert_allclose(result[f"{key}_slot"][slot], expected_variance[key])
    np.testing.assert_allclose(
        result["rms_kde_slot"][0],
        _kde_rms(expected_slot_0, reliability.values[0], grid)["rms_kde_mean"],
    )
    np.testing.assert_allclose(
        result["rms_kde_pooled"],
        _kde_rms(result["rms_slot"].reshape(-1), result["reliability_slot"].reshape(-1), grid)["rms_kde_mean"],
    )


def test_trial_activity_measurements_include_rms_mean_pf_dot_and_gain(monkeypatch):
    spkmap = np.array(
        [
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            [[0.0, np.nan, 2.0], [2.0, 4.0, np.nan]],
        ]
    )
    expected_gain = np.array([[0.5, 1.5], [2.0, 3.0]])

    def fake_gain(values, positions):
        np.testing.assert_array_equal(values, spkmap)
        np.testing.assert_array_equal(positions, [0.0, 1.0, 2.0])
        return expected_gain, None, None

    monkeypatch.setattr(pfpred_quality, "gaussian_gain_matrix", fake_gain)
    result = _trial_activity_measurements(spkmap, np.arange(3.0))

    np.testing.assert_allclose(result["trial_rms"][0], np.sqrt([[14 / 3], [14 / 3]]).ravel())
    np.testing.assert_allclose(result["trial_rms"][1], [np.sqrt(2.0), np.sqrt(10.0)])
    np.testing.assert_allclose(result["trial_mean"], [[2.0, 2.0], [1.0, 3.0]])
    mean_pf = np.array([[2.0, 2.0, 2.0], [1.0, 4.0, 2.0]])
    unit_pf = mean_pf / np.linalg.norm(mean_pf, axis=1, keepdims=True)
    expected_dot = np.array(
        [
            [np.dot(spkmap[0, 0], unit_pf[0]), np.dot(spkmap[0, 1], unit_pf[0])],
            [np.nansum(spkmap[1, 0] * unit_pf[1]), np.nansum(spkmap[1, 1] * unit_pf[1])],
        ]
    )
    np.testing.assert_allclose(result["trial_pf_dot"], expected_dot)
    np.testing.assert_array_equal(result["trial_gain"], expected_gain)


def test_trial_activity_summaries_and_completed_key_names():
    metric_values = {
        "trial_rms": np.array([[1.0, 3.0, np.nan], [2.0, 2.0, 2.0]]),
        "trial_mean": np.array([[-1.0, 1.0, np.nan], [1.0, 3.0, 5.0]]),
        "trial_pf_dot": np.array([[2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]),
        "trial_gain": np.array([[0.5, 1.0, 1.5], [np.nan, np.nan, np.nan]]),
    }
    result = _summarize_trial_activity(metric_values)

    assert trial_activity_summary_keys() == [
        "trial_rms_mean",
        "trial_rms_variance",
        "trial_rms_cv",
        "trial_mean_mean",
        "trial_mean_variance",
        "trial_mean_cv",
        "trial_pf_dot_mean",
        "trial_pf_dot_variance",
        "trial_pf_dot_cv",
        "trial_gain_mean",
        "trial_gain_variance",
        "trial_gain_cv",
    ]
    assert set(result) == set(trial_activity_summary_keys())
    np.testing.assert_allclose(result["trial_rms_mean"], [2.0, 2.0])
    np.testing.assert_allclose(result["trial_rms_variance"], [1.0, 0.0])
    np.testing.assert_allclose(result["trial_rms_cv"], [0.5, 0.0])
    assert np.isnan(result["trial_mean_cv"][0])
    np.testing.assert_allclose(result["trial_mean_cv"][1], np.std([1.0, 3.0, 5.0]) / 3.0)
    assert np.isnan(result["trial_gain_mean"][1])
    assert np.isnan(result["trial_gain_variance"][1])
    assert np.isnan(result["trial_gain_cv"][1])


def test_trial_activity_population_summaries_match_quality_naming_and_ignore_nans():
    per_roi = {key: np.array([1.0, 3.0, 5.0, 7.0]) for key in trial_activity_summary_keys()}
    per_roi["trial_gain_mean"] = np.array([np.nan, 3.0, np.nan, 7.0])
    quality_filtered = np.array([True, True, False, False])

    result = _summarize_trial_activity_across_rois(per_roi, quality_filtered)

    assert set(result) == set(trial_activity_population_summary_keys())
    assert len(result) == 2 * 3 * 12
    assert result["mean_trial_gain_mean"] == pytest.approx(5.0)
    assert result["mean_quality_filtered_trial_gain_mean"] == pytest.approx(3.0)
    assert result["mean_notquality_filtered_trial_gain_mean"] == pytest.approx(7.0)
    assert result["median_quality_filtered_trial_rms_variance"] == pytest.approx(2.0)
    assert result["median_notquality_filtered_trial_rms_variance"] == pytest.approx(6.0)


def test_pfpred_quality_defaults_and_threshold_validation():
    config = PFPredQualityConfig()

    assert config.schema_version == "v9"
    assert config.reliability_threshold == pytest.approx(0.3)
    assert config.fraction_active_threshold == pytest.approx(0.1)
    assert "rel=0.3_fa=0.1" in config.summary()
    with pytest.raises(ValueError, match="reliability_threshold"):
        PFPredQualityConfig(reliability_threshold=1.1)
    with pytest.raises(ValueError, match="fraction_active_threshold"):
        PFPredQualityConfig(fraction_active_threshold=-0.1)


def test_trial_activity_by_slot_stores_fraction_active_masks_and_subset_summaries(monkeypatch):
    class FakePlacefields:
        environment = np.array([11, 11, 22, 22])
        placefield = np.array(
            [
                [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]],
                [[2.0, 1.0], [4.0, 2.0], [6.0, 3.0], [8.0, 4.0]],
                [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0]],
                [[3.0, 1.0], [3.0, 1.0], [3.0, 1.0], [3.0, 1.0]],
            ]
        )

        def filter_by_environment(self, environment):
            return SimpleNamespace(placefield=self.placefield[self.environment == environment])

    monkeypatch.setattr(pfpred_quality, "get_frame_behavior", lambda session: object())
    monkeypatch.setattr(pfpred_quality, "get_placefield", lambda *args, **kwargs: FakePlacefields())
    monkeypatch.setattr(pfpred_quality, "load_env_order", lambda: {"mouse": [11, 22]})
    monkeypatch.setattr(
        pfpred_quality,
        "gaussian_gain_matrix",
        lambda spkmap, positions: (np.ones(spkmap.shape[:2]), None, None),
    )
    monkeypatch.setattr(
        pfpred_quality.FractionActive,
        "compute",
        staticmethod(lambda spkmap, **kwargs: np.array([0.2, 0.05])),
    )
    session = SimpleNamespace(mouse_name="mouse")
    smp = SimpleNamespace(dist_edges=np.arange(5.0), params=SimpleNamespace(speed_threshold=1.0, smooth_width=None))
    reliability = SimpleNamespace(values=np.array([[0.4, 0.9], [0.2, 0.8]]))
    env_maps = SimpleNamespace(environments=np.array([11, 22]))

    result = _trial_activity_by_slot(
        session,
        np.zeros((10, 2)),
        smp,
        reliability,
        env_maps,
        best_env=0,
        reliability_threshold=0.3,
        fraction_active_threshold=0.1,
    )

    np.testing.assert_allclose(result["fraction_active_slot"][:2], [[0.2, 0.05], [0.2, 0.05]])
    np.testing.assert_array_equal(result["quality_filtered_roi_mask_slot"][:2], [[True, False], [False, False]])
    np.testing.assert_array_equal(result["quality_filtered_roi_mask"], [True, False])
    np.testing.assert_allclose(result["num_quality_filtered_rois_slot"][:2], [1.0, 0.0])
    assert result["mean_quality_filtered_trial_rms_mean"][0] == pytest.approx(result["trial_rms_mean"][0, 0])
    assert result["mean_notquality_filtered_trial_rms_mean"][0] == pytest.approx(result["trial_rms_mean"][0, 1])
    assert np.isnan(result["mean_quality_filtered_trial_rms_mean"][1])
