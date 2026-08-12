"""Focused tests for SpectrumFigureViewer's per-environment source mode."""

import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure4._alpha_config import SpectrumSmoothingConfig
from dimensionality_manuscript.figure_scripts.figure4._spectrum_math import _signed_participation_ratio
from dimensionality_manuscript.figure_scripts.figure4.spectrum_figure import SpectrumFigureViewer
from dimensionality_manuscript import average_by_mouse


class _FakeAggregator:
    def __init__(self, arrays, mouse_names):
        self._arrays = arrays
        self.arrays = arrays
        self.param_axes = {}
        self.mouse_names = np.asarray(mouse_names)
        self.session_ids = [f"session-{i}" for i in range(len(mouse_names))]

    def sel(self, *, keys, avg_by_mouse=False, **_kwargs):
        selected = {key: self._arrays[key] for key in keys}
        if avg_by_mouse:
            return {key: average_by_mouse(value, self.mouse_names) for key, value in selected.items()}
        return selected


def _viewer(arrays, mouse_names, *, subspace_arrays=None, subspace_mouse_names=None):
    viewer = object.__new__(SpectrumFigureViewer)
    viewer.results = _FakeAggregator(arrays, mouse_names)
    viewer.results_svca = (
        _FakeAggregator(subspace_arrays, subspace_mouse_names)
        if subspace_arrays is not None
        else None
    )
    viewer.results_subspace = viewer.results_svca
    viewer._svca_results = viewer.results_svca
    viewer._svca_pf_keys = {"SVCA": "ss", "SVCA_PRED": "ss_pred"}
    viewer._svca_full_keys = {"SVCA": "ff", "SVCA_RES": "ff_res"}
    viewer._agg = {"stimspace": viewer.results, "cvpca": None, "svca": viewer._svca_results}
    viewer._tuple_labels = {}
    return viewer


def _state(**overrides):
    return {
        "source_mode": "avg_env",
        "normalize": False,
        "clip_negative": False,
        **overrides,
    }


def test_avg_env_averages_environment_then_session_within_mouse_before_pr():
    per_env = np.array(
        [
            [[8.0, 4.0, 2.0], [4.0, 2.0, np.nan]],
            [[2.0, 1.0, 0.5], [np.nan, np.nan, np.nan]],
            [[10.0, 5.0, 2.0], [6.0, 3.0, 2.0]],
        ]
    )
    viewer = _viewer({"ss_cv_env": per_env}, ["mouse-a", "mouse-a", "mouse-b"])
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    spectra = viewer._avg_env_spectrum(_state(), "ss_cv", cfg)

    expected = np.array([[4.0, 2.0, 1.25], [8.0, 4.0, 2.0]])
    np.testing.assert_allclose(spectra, expected)
    np.testing.assert_allclose(_signed_participation_ratio(spectra), _signed_participation_ratio(expected))


def test_avg_env_clips_each_environment_before_averaging():
    per_env = np.array([[[6.0, 3.0, -1.0], [2.0, 1.0, 0.5]]])
    viewer = _viewer({"ss_cv_env": per_env}, ["mouse-a"])
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    spectra = viewer._avg_env_spectrum(_state(clip_negative=True), "ss_cv", cfg)

    np.testing.assert_allclose(spectra, [[4.0, 2.0, 0.5]])


def test_avg_env_supports_svca_placefield_environment_spectra():
    per_env = np.array(
        [
            [[8.0, 4.0, 2.0], [4.0, 2.0, np.nan]],
            [[2.0, 1.0, 0.5], [np.nan, np.nan, np.nan]],
            [[10.0, 5.0, 2.0], [6.0, 3.0, 2.0]],
        ]
    )
    viewer = _viewer(
        {},
        ["stim-mouse"],
        subspace_arrays={"ss_env": per_env},
        subspace_mouse_names=["mouse-a", "mouse-a", "mouse-b"],
    )
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    spectra = viewer._avg_env_spectrum(_state(), "SVCA", cfg)

    expected = np.array([[4.0, 2.0, 1.25], [8.0, 4.0, 2.0]])
    np.testing.assert_allclose(spectra, expected)


def test_all_supports_svca_placefield_prediction_spectra():
    prediction = np.array(
        [
            [8.0, 4.0, 2.0],
            [2.0, 1.0, 0.5],
            [10.0, 5.0, 2.0],
        ]
    )
    viewer = _viewer(
        {},
        ["stim-mouse"],
        subspace_arrays={"ss_pred": prediction},
        subspace_mouse_names=["mouse-a", "mouse-a", "mouse-b"],
    )
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    spectra = viewer._spectrum(_state(source_mode="all"), "SVCA_PRED", cfg)
    raw, smoothed, mouse_names, _ = viewer._spectrum_sessions(
        _state(source_mode="all"),
        "SVCA_PRED",
        cfg,
    )

    np.testing.assert_allclose(spectra, [[5.0, 2.5, 1.25], [10.0, 5.0, 2.0]])
    np.testing.assert_allclose(raw, prediction)
    np.testing.assert_allclose(smoothed, prediction)
    np.testing.assert_array_equal(mouse_names, ["mouse-a", "mouse-a", "mouse-b"])


def test_avg_env_supports_svca_placefield_prediction_spectra():
    per_env = np.array(
        [
            [[8.0, 4.0, 2.0], [4.0, 2.0, np.nan]],
            [[2.0, 1.0, 0.5], [np.nan, np.nan, np.nan]],
            [[10.0, 5.0, 2.0], [6.0, 3.0, 2.0]],
        ]
    )
    viewer = _viewer(
        {},
        ["stim-mouse"],
        subspace_arrays={"ss_pred_env": per_env},
        subspace_mouse_names=["mouse-a", "mouse-a", "mouse-b"],
    )
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    spectra = viewer._avg_env_spectrum(_state(), "SVCA_PRED", cfg)
    raw, smoothed, mouse_names, _ = viewer._spectrum_sessions(_state(), "SVCA_PRED", cfg)

    np.testing.assert_allclose(spectra, [[4.0, 2.0, 1.25], [8.0, 4.0, 2.0]])
    np.testing.assert_allclose(raw, [[6.0, 3.0, 2.0], [2.0, 1.0, 0.5], [8.0, 4.0, 2.0]])
    np.testing.assert_allclose(smoothed, raw)
    np.testing.assert_array_equal(mouse_names, ["mouse-a", "mouse-a", "mouse-b"])


def test_avg_env_svca_session_path_uses_subspace_session_metadata():
    per_env = np.array([[[6.0, 3.0, -1.0], [2.0, 1.0, 0.5]]])
    viewer = _viewer(
        {},
        ["stim-mouse"],
        subspace_arrays={"ss_env": per_env},
        subspace_mouse_names=["svca-mouse"],
    )
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    raw, smoothed, mouse_names, session_ids = viewer._spectrum_sessions(
        _state(clip_negative=True),
        "SVCA",
        cfg,
    )

    np.testing.assert_allclose(raw, [[4.0, 2.0, 0.5]])
    np.testing.assert_allclose(smoothed, raw)
    np.testing.assert_array_equal(mouse_names, ["svca-mouse"])
    assert session_ids == ["session-0"]


def test_avg_env_rejects_source_without_per_environment_results():
    viewer = _viewer({}, ["mouse-a"])
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    with pytest.raises(ValueError, match="no independently computed per-environment spectrum"):
        viewer._avg_env_spectrum(_state(), "reg_covariances_fixed", cfg)


def test_avg_env_does_not_change_full_source_spectrum():
    viewer = _viewer({"ff": np.array([[9.0, 3.0, 1.0]])}, ["mouse-a"])
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    spectra = viewer._ff_spectrum({**_state(), "full_source_key": "SVD"}, cfg)

    np.testing.assert_allclose(spectra, [[9.0, 3.0, 1.0]])


@pytest.mark.parametrize("public_key, stored_key", [("SVCA", "ff"), ("SVCA_RES", "ff_res")])
def test_full_svca_sources_use_stimspace_svca_keys(public_key, stored_key):
    spectrum = np.array([[9.0, 3.0, 1.0]])
    viewer = _viewer(
        {},
        ["stim-mouse"],
        subspace_arrays={stored_key: spectrum},
        subspace_mouse_names=["svca-mouse"],
    )
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    actual = viewer._ff_spectrum({**_state(), "full_source_key": public_key}, cfg)

    np.testing.assert_allclose(actual, spectrum)


def test_full_svca_source_still_supports_legacy_subspace_key():
    spectrum = np.array([[9.0, 3.0, 1.0]])
    viewer = _viewer(
        {},
        ["stim-mouse"],
        subspace_arrays={"variance_activity_residual": spectrum},
        subspace_mouse_names=["svca-mouse"],
    )
    viewer.results_svca = None
    viewer.results_subspace = viewer._svca_results
    viewer._svca_full_keys = {"SVCA": "variance_activity", "SVCA_RES": "variance_activity_residual"}
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    actual = viewer._ff_spectrum({**_state(), "full_source_key": "SVCA_RES"}, cfg)

    np.testing.assert_allclose(actual, spectrum)


def test_white_svd_source_uses_ffres_white_and_ignores_source_mode():
    spectrum = np.array(
        [
            [8.0, 4.0, 2.0],
            [2.0, 1.0, 0.5],
            [10.0, 5.0, 2.0],
        ]
    )
    viewer = _viewer({"ffres_white": spectrum}, ["mouse-a", "mouse-a", "mouse-b"])
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    actual = viewer._white_spectrum(_state(white_source_key="SVD"), cfg)

    np.testing.assert_allclose(actual, [[5.0, 2.5, 1.25], [10.0, 5.0, 2.0]])


def test_white_svca_source_uses_ff_res_white():
    spectrum = np.array(
        [
            [8.0, 4.0, 2.0],
            [2.0, 1.0, 0.5],
            [10.0, 5.0, 2.0],
        ]
    )
    viewer = _viewer(
        {},
        ["stim-mouse"],
        subspace_arrays={"ff_res_white": spectrum},
        subspace_mouse_names=["mouse-a", "mouse-a", "mouse-b"],
    )
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    actual = viewer._white_spectrum(_state(white_source_key="SVCA"), cfg)
    raw, smoothed, mouse_names, _ = viewer._white_spectrum_sessions(
        _state(white_source_key="SVCA"),
        cfg,
    )

    np.testing.assert_allclose(actual, [[5.0, 2.5, 1.25], [10.0, 5.0, 2.0]])
    np.testing.assert_allclose(raw, spectrum)
    np.testing.assert_allclose(smoothed, spectrum)
    np.testing.assert_array_equal(mouse_names, ["mouse-a", "mouse-a", "mouse-b"])


def test_white_none_hides_spectrum_and_sessions():
    viewer = _viewer({}, ["mouse-a"])
    cfg = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

    assert viewer._white_spectrum(_state(white_source_key="none"), cfg) is None
    assert viewer._white_spectrum_sessions(_state(white_source_key="none"), cfg) is None
