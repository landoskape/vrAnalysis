from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure2 import rasters


class _FakeSession:
    session_name = "fake_session"

    def session_print(self):
        return "fake session"

    def loadone(self, name):
        assert name == "mpci.times"
        return np.arange(8, dtype=float)


class _FakeResults:
    param_axes = {"model_name": list(rasters.PREDICTION_FIGURE_MODELS)}
    sessions = [_FakeSession()]
    unique_mice = np.array(["m1"])
    mouse_names = np.array(["m1"])
    config_class = SimpleNamespace(spks_type="oasis", activity_parameters_name="default")


def test_prediction_figure_adds_structured_additive_between_gain_and_peer(monkeypatch):
    calls = []

    def fake_predictions(model_name, *args, **kwargs):
        calls.append(model_name)
        target = np.arange(24, dtype=float).reshape(3, 8)
        prediction = target - (1 + rasters.PREDICTION_FIGURE_MODELS.index(model_name))
        return target, prediction

    monkeypatch.setattr(rasters, "get_model_predictions", fake_predictions)
    viewer = rasters.ModelPredictionsFigureViewer(
        _FakeResults(),
        registry=object(),
        sort_method="activity",
        scale_bar_seconds=0,
        figsize=(8, 4),
        include_structured_additive=True,
    )

    fig = viewer.plot(viewer.state)
    main_axes = fig.axes[:8]

    assert rasters.PREDICTION_FIGURE_MODELS == (
        "internal_placefield_1d_gain",
        "internal_placefield_1d_structured_additive",
        "rrr",
    )
    assert len(main_axes) == 8
    assert [[text.get_text() for text in axis.texts] for axis in main_axes[:4]] == [
        ["Target Data"],
        ["Global Gain"],
        ["Residual"],
        ["Peer Prediction"],
    ]
    assert set(rasters.PREDICTION_FIGURE_MODELS).issubset(calls)
    plt.close(fig)


def test_prediction_figure_can_hide_only_the_structured_additive_rasters(monkeypatch):
    calls = []

    def fake_predictions(model_name, *args, **kwargs):
        calls.append(model_name)
        target = np.arange(24, dtype=float).reshape(3, 8)
        return target, target - 1

    monkeypatch.setattr(rasters, "get_model_predictions", fake_predictions)
    viewer = rasters.ModelPredictionsFigureViewer(
        _FakeResults(),
        registry=object(),
        sort_method="activity",
        scale_bar_seconds=0,
        figsize=(8, 4),
        include_structured_additive=False,
    )

    fig = viewer.plot(viewer.state)
    main_axes = fig.axes[:6]

    assert viewer.model_names == ("internal_placefield_1d_gain", "rrr")
    assert len(main_axes) == 6
    assert [[text.get_text() for text in axis.texts] for axis in main_axes[:3]] == [
        ["Target Data"],
        ["Global Gain"],
        ["Peer Prediction"],
    ]
    assert "internal_placefield_1d_structured_additive" not in calls
    plt.close(fig)


def test_zoom_figure_reuses_sorted_overview_and_boxes_absolute_window(monkeypatch):
    target = np.arange(40, dtype=float).reshape(5, 8)
    monkeypatch.setattr(rasters, "get_model_predictions", lambda *args, **kwargs: (target, target - 2))
    viewer = rasters.ModelPredictionsZoomFigureViewer(
        _FakeResults(), registry=object(), sort_method="activity", xslice=slice(1, 8),
        zoom_xslice=slice(3, 6), zoom_yslice=slice(1, 4), zscore=True,
        scale_bar_num_rois=2, scale_bar_seconds=1,
    )
    fig = viewer.plot(viewer.state)
    assert viewer.model_names == ("internal_placefield_1d_gain", "rrr")
    assert not any(name in viewer.state for name in rasters.ModelZooUltraCondensed._TUNABLES)
    assert len(fig.axes) == 6
    expected = rasters.transform_raster_rows(target, zscore=True, subtract_median=False)[::-1, 1:8]
    assert viewer.state["x_ratio"] == 0.3
    assert fig.axes[0].get_gridspec().get_geometry() == (3, 2)
    assert fig.axes[0].get_gridspec().get_width_ratios() == [1, 0.3]
    for row in range(3):
        overview, zoom = fig.axes[2 * row:2 * row + 2]
        np.testing.assert_array_equal(overview.images[0].get_array(), zoom.images[0].get_array())
        assert overview.images[0].get_clim() == zoom.images[0].get_clim()
        assert zoom.get_xlim() == (2.5, 5.5)
        assert zoom.get_ylim() == (3.5, 0.5)
        box = overview.patches[0]
        assert box.get_bbox().bounds == (2.5, 0.5, 3, 3)
        assert not zoom.lines
        assert not zoom.child_axes
    np.testing.assert_allclose(fig.axes[0].images[0].get_array(), expected)
    assert len(fig.axes[0].lines) == 1
    fig.canvas.draw()
    viewer.update_float("x_ratio", value=0.5)
    fig = viewer.plot(viewer.state)
    assert fig.axes[0].get_gridspec().get_width_ratios() == [1, 0.5]
    plt.close(fig)


def test_zoom_defaults_to_all_cells_and_clamps_after_data_change(monkeypatch):
    shape = [5, 8]

    def predictions(*args, **kwargs):
        target = np.arange(np.prod(shape), dtype=float).reshape(shape)
        return target, target - 1

    monkeypatch.setattr(rasters, "get_model_predictions", predictions)
    viewer = rasters.ModelPredictionsZoomFigureViewer(
        _FakeResults(), registry=object(), sort_method="activity", zoom_xslice=slice(4, 8),
    )
    assert viewer.state["zoom_all_cells"]
    fig = viewer.plot(viewer.state)
    assert fig.axes[1].get_ylim() == (4.5, -0.5)
    shape[:] = [2, 3]
    viewer._update_frame_bounds(viewer.state)
    assert viewer.state["xslice_stop"] == 3
    assert viewer.state["zoom_xslice_start"] == 2
    assert viewer.state["zoom_xslice_stop"] == 3
    assert viewer.state["zoom_yslice_stop"] == 2
    plt.close(fig)


@pytest.mark.parametrize("include_structured_additive", [False, True])
def test_zoom_external_pf_is_first_prediction(monkeypatch, include_structured_additive):
    calls = []

    def predictions(model_name, *args, **kwargs):
        calls.append(model_name)
        target = np.arange(24, dtype=float).reshape(3, 8)
        return target, target - 1

    monkeypatch.setattr(rasters, "get_model_predictions", predictions)
    results = _FakeResults()
    results.param_axes = {"model_name": ["external_placefield_1d", *rasters.PREDICTION_FIGURE_MODELS]}
    viewer = rasters.ModelPredictionsZoomFigureViewer(
        results, registry=object(), sort_method="activity", include_external_pf=True,
        include_structured_additive=include_structured_additive,
    )
    fig = viewer.plot(viewer.state)
    expected = ["Target Data", "External Placefield", "Global Gain"]
    if include_structured_additive:
        expected.append("Residual")
    expected.append("Peer Prediction")
    assert [axis.texts[0].get_text() for axis in fig.axes[::2]] == expected
    assert viewer.model_names[0] == "external_placefield_1d"
    assert "external_placefield_1d" in calls
    for overview, zoom in zip(fig.axes[::2], fig.axes[1::2]):
        assert len(overview.patches) == 1
        np.testing.assert_array_equal(overview.images[0].get_array(), zoom.images[0].get_array())
    plt.close(fig)
