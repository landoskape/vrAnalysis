"""Sign classification, hierarchical weighting, and plotting of saved spectra."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure4 import SpectrumNegativeViewer
from dimensionality_manuscript.figure_scripts.figure4.spectrum_negative import _mouse_summary


class _Aggregator:
    def __init__(self, arrays, mice, param_axes=None):
        self.arrays = arrays
        self.mouse_names = np.asarray(mice)
        self.session_ids = [f"s{i}" for i in range(len(mice))]
        self.param_axes = param_axes or {}
        self.last_selection = None

    def sel(self, **kwargs):
        self.last_selection = kwargs
        return {key: self.arrays[key] for key in kwargs["keys"]}


def test_binary_signs_preserve_zero_missing_values_and_mouse_weighting():
    agg = _Aggregator({"ss_cv": np.array([[2, -1, np.nan], [-4, -2, np.nan], [3, 0, -1]])}, ["a", "a", "b"])
    viewer = SpectrumNegativeViewer(agg)
    np.testing.assert_allclose(viewer._heatmap, [[0, 1, np.nan], [1, 1, np.nan], [0, 0, 1]], equal_nan=True)
    np.testing.assert_allclose(viewer._mouse_probabilities, [[0.5, 1, np.nan], [0, 0, 1]], equal_nan=True)
    np.testing.assert_allclose(viewer._mean, [0.25, 0.5, 1])
    np.testing.assert_allclose(viewer._sem, [0.25, 0.5, np.nan], equal_nan=True)
    assert agg.last_selection["squeeze_ones"] is False
    assert agg.last_selection["avg_by_mouse"] is False


def test_environment_indicators_then_sessions_then_mice_with_unequal_counts():
    # Mouse a has two sessions with two and one environments; mouse b has one.
    # Pooling rows or signs after averaging spectrum magnitudes gives wrong answers.
    spectra = np.array([[[100.0], [-1.0]], [[-1.0], [np.nan]], [[1.0], [1.0]]])
    agg = _Aggregator({"ss_cv_env": spectra}, ["a", "a", "b"])
    viewer = SpectrumNegativeViewer(agg, source_mode="avg_env")
    np.testing.assert_allclose(viewer._heatmap[:, 0], [0, 1, 1, 0, 0])
    np.testing.assert_allclose(viewer._mouse_probabilities, [[0.75], [0]])
    np.testing.assert_allclose(viewer._mean, [0.375])
    np.testing.assert_allclose(viewer._sem, [0.375])
    assert len(viewer._row_labels) == 5


@pytest.mark.parametrize("key", ["ss_cv", "ss_direct", "ss_cvpca", "sf_cv", "sf_direct"])
def test_stimspace_sources(key):
    viewer = SpectrumNegativeViewer(_Aggregator({key: np.array([[-1.0]])}, ["a"]), source_key=key)
    np.testing.assert_equal(viewer._heatmap, [[1]])


@pytest.mark.parametrize("key,stored", [("ss_cv", "ss_cv_env"), ("ss_direct", "ss_direct_env"),
                                       ("ss_cvpca", "ss_cvpca_env"), ("sf_cv", "sf_cv_env_full1"),
                                       ("sf_direct", "sf_direct_env_full1")])
def test_stimspace_environment_key_mapping(key, stored):
    viewer = SpectrumNegativeViewer(_Aggregator({stored: np.array([[[-1.0]]])}, ["a"]),
                                    source_key=key, source_mode="avg_env")
    np.testing.assert_equal(viewer._heatmap, [[1]])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("key,compact,old", [("SVCA", "ss", "variance_placefield_placefield"),
                                          ("SVCA_PRED", "ss_pred", "variance_placefield_prediction")])
@pytest.mark.parametrize("mode", ["all", "avg_env"])
def test_svca_uses_own_session_metadata_and_supports_legacy_keys(legacy, key, compact, old, mode):
    stored = old if legacy else compact
    data = np.array([[-1.0, 1.0]])
    if mode == "avg_env":
        stored += "_env"
        data = data[:, None, :]
    svca = _Aggregator({stored: data}, ["svca-mouse"])
    kwargs = {"results_subspace" if legacy else "results_svca": svca}
    viewer = SpectrumNegativeViewer(_Aggregator({}, ["other"]), source_key=key, source_mode=mode, **kwargs)
    assert viewer._mice == ["svca-mouse"]
    assert viewer._row_labels[0].startswith("svca-mouse:")
    np.testing.assert_equal(viewer._mean, [1, 0])


@pytest.mark.parametrize("mode", ["all", "avg_env"])
def test_cvpca_tuple_selection_and_ignored_env_mode(mode):
    cvpca = _Aggregator({"reg_covariances_fixed": np.array([[-1.0]])}, ["a"],
                        {"smooth_widths": [(None, None), (5.0, None)]})
    viewer = SpectrumNegativeViewer(_Aggregator({}, ["other"]), results_cvpca=cvpca,
                                    source_key="reg_covariances_fixed", source_mode=mode, smooth_widths=(5.0, None))
    assert cvpca.last_selection["smooth_widths"] == (5.0, None)
    viewer.refresh_data({**viewer.state, "source_mode": "avg_env"})
    assert cvpca.last_selection["keys"] == ["reg_covariances_fixed"]
    np.testing.assert_equal(viewer._heatmap, [[1]])
    np.testing.assert_equal(viewer._mean, [1])
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert fig.axes[0].get_ylabel() == "Session"
    assert " | all | " in fig.axes[0].get_title()


def test_single_mouse_sem_and_missing_rank_are_undefined():
    mean, sem = _mouse_summary(np.array([[0.5, np.nan]]))
    np.testing.assert_allclose(mean, [0.5, np.nan], equal_nan=True)
    assert np.isnan(sem).all()


@pytest.mark.parametrize("mode", ["all", "avg_env"])
@pytest.mark.parametrize("xscale", ["linear", "log"])
def test_plot_renders_binary_heatmap_and_mean_curve(mode, xscale):
    data = np.array([[[-1.0, 0, np.nan], [1, -2, np.nan]], [[1.0, 2, 1], [np.nan, np.nan, np.nan]]])
    agg = _Aggregator({"ss_cv": data[:, 0], "ss_cv_env": data}, ["a", "b"])
    viewer = SpectrumNegativeViewer(agg, source_mode=mode, xscale=xscale)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 3  # heatmap, curve, and binary colorbar
    assert fig.axes[1].get_xscale() == xscale
    np.testing.assert_allclose(fig.axes[1].lines[2].get_ydata(), viewer._mean)


def test_empty_selection_can_render_without_counting_missing_as_nonnegative():
    viewer = SpectrumNegativeViewer(_Aggregator({"ss_cv": np.full((1, 2), np.nan)}, ["a"]))
    assert viewer._heatmap.shape == (0, 2)
    assert np.isnan(viewer._mean).all()
    viewer.plot(viewer.state).canvas.draw()
