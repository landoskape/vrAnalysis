from types import SimpleNamespace

import numpy as np

from dimensionality_manuscript.configs import pfpred_quality
from dimensionality_manuscript.configs.pfpred_quality import (
    _binned_rms,
    _kde_rms,
    _normalized_rms_error,
    _r2_by_slot,
    _rms_error,
    _variance_components,
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
