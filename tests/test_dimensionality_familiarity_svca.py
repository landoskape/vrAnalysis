"""StimspaceSVCA result-key coverage for DimensionalityFamiliarityViewer."""

import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure4.familiarity import DimensionalityFamiliarityViewer


class _FakeAggregator:
    def __init__(self, arrays):
        self._arrays = arrays
        self.param_axes = {}
        self.mouse_names = np.array(["mouse-a"])
        self.session_ids = ["session-a"]

    def sel(self, *, keys, **_kwargs):
        return {key: self._arrays[key] for key in keys}


def _viewer(svca_arrays):
    viewer = object.__new__(DimensionalityFamiliarityViewer)
    viewer.results_svca = _FakeAggregator(svca_arrays)
    viewer.results_subspace = viewer.results_svca
    viewer._svca_results = viewer.results_svca
    viewer._svca_pf_keys = {"SVCA": "ss", "SVCA_PRED": "ss_pred"}
    viewer._svca_pf_env_keys = {"SVCA": "ss_env", "SVCA_PRED": "ss_pred_env"}
    viewer._svca_full_keys = {"SVCA": "ff", "SVCA_RES": "ff_res"}
    viewer._svca_full_env_keys = {"SVCA": "ff_env", "SVCA_RES": "ff_res_env"}
    viewer._agg = {"svca": viewer._svca_results}
    viewer._tuple_labels = {}
    return viewer


def _legacy_viewer(arrays):
    viewer = _viewer({})
    viewer.results_svca = None
    viewer.results_subspace = _FakeAggregator(arrays)
    viewer._svca_results = viewer.results_subspace
    viewer._svca_pf_keys = {"SVCA": "variance_placefield_placefield", "SVCA_PRED": "variance_placefield_prediction"}
    viewer._svca_pf_env_keys = {
        "SVCA": "variance_placefield_placefield_env",
        "SVCA_PRED": "variance_placefield_prediction_env",
    }
    viewer._svca_full_keys = {"SVCA": "variance_activity", "SVCA_RES": "variance_activity_residual"}
    viewer._svca_full_env_keys = {"SVCA": "variance_activity_env", "SVCA_RES": "variance_activity_residual_env"}
    viewer._agg["svca"] = viewer._svca_results
    return viewer


@pytest.mark.parametrize("public_key, stored_key", [("SVCA", "ss"), ("SVCA_PRED", "ss_pred")])
def test_source_svca_keys(public_key, stored_key):
    spectrum = np.array([[8.0, 4.0, 2.0]])
    viewer = _viewer({stored_key: spectrum})

    actual, aggregator = viewer._spectrum_sessions({"clip_negative": False}, public_key)

    np.testing.assert_allclose(actual, spectrum)
    assert aggregator is viewer.results_svca


@pytest.mark.parametrize("public_key, stored_key", [("SVCA", "ff"), ("SVCA_RES", "ff_res")])
def test_full_svca_keys(public_key, stored_key):
    spectrum = np.array([[9.0, 3.0, 1.0]])
    viewer = _viewer({stored_key: spectrum})

    actual, aggregator = viewer._full_spectrum_sessions({"full_key": public_key, "clip_negative": False})

    np.testing.assert_allclose(actual, spectrum)
    assert aggregator is viewer.results_svca


def test_per_environment_svca_keys():
    assert DimensionalityFamiliarityViewer._per_env_result_keys("SVCA", "SVCA_RES", "full1") == (
        "ss_env",
        "ff_res_env",
    )
    assert DimensionalityFamiliarityViewer._per_env_result_keys("SVCA_PRED", "SVCA", "fullall") == (
        "ss_pred_env",
        "ff_env",
    )


def test_legacy_subspace_keys_still_work():
    spectrum = np.array([[9.0, 3.0, 1.0]])
    viewer = _legacy_viewer({"variance_activity_residual": spectrum})

    actual, aggregator = viewer._full_spectrum_sessions({"full_key": "SVCA_RES", "clip_negative": False})

    np.testing.assert_allclose(actual, spectrum)
    assert aggregator is viewer.results_subspace
    assert viewer._selected_per_env_result_keys("SVCA", "SVCA_RES", "full1") == (
        "variance_placefield_placefield_env",
        "variance_activity_residual_env",
    )
