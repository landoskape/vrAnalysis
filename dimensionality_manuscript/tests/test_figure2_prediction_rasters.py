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


def test_prediction_figure_adds_structured_additive_before_peer(monkeypatch):
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
    main_axes = fig.axes[:10]

    assert rasters.PREDICTION_FIGURE_MODELS == (
        "internal_placefield_1d",
        "internal_placefield_1d_gain",
        "internal_placefield_1d_structured_additive",
        "rrr",
    )
    assert len(main_axes) == 10
    assert [[text.get_text() for text in axis.texts] for axis in main_axes[:5]] == [
        ["Target Data"],
        ["Int. Pos. PF"],
        ["Global Gain"],
        ["Shared Residual"],
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
    main_axes = fig.axes[:8]

    assert viewer.model_names == ("internal_placefield_1d", "internal_placefield_1d_gain", "rrr")
    assert len(main_axes) == 8
    assert [[text.get_text() for text in axis.texts] for axis in main_axes[:4]] == [
        ["Target Data"],
        ["Int. Pos. PF"],
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
    assert viewer.model_names == ("internal_placefield_1d", "internal_placefield_1d_gain", "rrr")
    assert not any(name in viewer.state for name in rasters.ModelZooUltraCondensed._TUNABLES)
    assert len(fig.axes) == 8
    expected = rasters.transform_raster_rows(target, zscore=True, subtract_median=False)[::-1, 1:8]
    assert viewer.state["x_ratio"] == 0.3
    assert fig.axes[0].get_gridspec().get_geometry() == (4, 2)
    assert fig.axes[0].get_gridspec().get_width_ratios() == [1, 0.3]
    for row in range(4):
        overview, zoom = fig.axes[2 * row:2 * row + 2]
        np.testing.assert_array_equal(overview.images[0].get_array()[1:4, 2:5], zoom.images[0].get_array())
        assert tuple(zoom.images[0].get_extent()) == (2.5, 5.5, 3.5, 0.5)
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
    np.testing.assert_array_equal(fig.axes[1].images[0].get_array(), fig.axes[0].images[0].get_array()[:, 4:8])
    shape[:] = [2, 3]
    viewer._update_frame_bounds(viewer.state)
    assert viewer.state["xslice_stop"] == 3
    assert viewer.state["zoom_xslice_start"] == 2
    assert viewer.state["zoom_xslice_stop"] == 3
    assert viewer.state["zoom_yslice_stop"] == 2
    plt.close(fig)


@pytest.mark.parametrize("position_height", [0.1, 0.5, 1.0])
def test_zoom_position_row_matches_held_out_frames_and_zoom(monkeypatch, position_height):
    target = np.arange(24, dtype=float).reshape(3, 8)
    monkeypatch.setattr(rasters, "get_model_predictions", lambda *args, **kwargs: (target, target - 1))
    indices = np.array([1, 2, 3, 5, 6, 7, 8, 9])
    position = np.array([90, 10, 20, 30, 40, 50, 60, 70], dtype=float)
    environment = np.array([1, 1, 1, 1, 3, 3, 4, 4])

    def filter_behavior(selected):
        np.testing.assert_array_equal(selected, indices)
        return SimpleNamespace(position=position, environment=environment)

    def split_times(split, within_idx_samples):
        assert split == "test split"
        assert not within_idx_samples
        return indices

    registry = SimpleNamespace(
        time_split={"test": "test split"},
        get_population=lambda *args: (
            SimpleNamespace(get_split_times=split_times), SimpleNamespace(filter=filter_behavior),
        ),
    )
    monkeypatch.setattr(_FakeSession, "env_length", [100], raising=False)
    viewer = rasters.ModelPredictionsZoomFigureViewer(
        _FakeResults(), registry=registry, sort_method="activity", show_position=True,
        position_height=position_height, xslice=slice(1, 8), zoom_xslice=slice(3, 6),
        scale_bar_seconds=0,
    )
    fig = viewer.plot(viewer.state)
    overview, zoom = fig.axes[:2]
    assert len(fig.axes) == 10
    assert overview.get_gridspec().get_height_ratios() == [position_height, 1, 1, 1, 1]
    assert [line.get_color() for line in overview.lines] == ["k", "blue", "green"]
    assert overview.get_xlim() == fig.axes[2].get_xlim() == (0.5, 7.5)
    assert zoom.get_xlim() == fig.axes[3].get_xlim() == (2.5, 5.5)
    assert overview.get_ylim() == zoom.get_ylim()
    assert overview.yaxis_inverted()
    # Lap reset and a gap in the held-out split both interrupt the trace.
    np.testing.assert_equal(overview.lines[0].get_xdata(), [0, np.nan, 1, 2, np.nan, 3, 4, 5, 6, 7])
    for full_line, zoom_line in zip(overview.lines, zoom.lines):
        np.testing.assert_equal(full_line.get_data(), zoom_line.get_data())
    fig.canvas.draw()
    assert overview.get_position().height / fig.axes[2].get_position().height == pytest.approx(position_height)
    viewer.update_boolean("show_position", value=False)
    fig = viewer.plot(viewer.state)
    assert len(fig.axes) == 8
    assert all(axis.images for axis in fig.axes)
    plt.close(fig)


def test_zoom_bounds_only_request_the_first_model(monkeypatch):
    calls = []
    target = np.arange(40, dtype=float).reshape(5, 8)

    def predictions(model_name, *args, **kwargs):
        calls.append(model_name)
        return target, target - 1

    monkeypatch.setattr(rasters, "get_model_predictions", predictions)
    viewer = rasters.ModelPredictionsZoomFigureViewer(
        _FakeResults(), registry=object(), sort_method="activity",
        zoom_xslice=slice(3, 4), zoom_yslice=slice(2, 3),
        include_structured_additive=True,
    )
    assert set(calls) == {viewer.model_names[0]}
    calls.clear()
    viewer._update_zoom_bounds(viewer.state)
    assert set(calls) == {viewer.model_names[0]}

    calls.clear()
    fig = viewer.plot(viewer.state)
    assert set(calls) == set(viewer.model_names)
    for overview, zoom in zip(fig.axes[::2], fig.axes[1::2]):
        np.testing.assert_array_equal(zoom.images[0].get_array(), overview.images[0].get_array()[2:3, 3:4])
        assert tuple(zoom.images[0].get_extent()) == (2.5, 3.5, 2.5, 1.5)
    fig.canvas.draw()
    plt.close(fig)


@pytest.mark.parametrize("include_structured_additive", [False, True])
@pytest.mark.parametrize("include_external_pf", [False, True])
def test_zoom_matches_performance_models(monkeypatch, include_structured_additive, include_external_pf):
    from dimensionality_manuscript.figure_scripts.figure2.performance import ModelPerformanceViewer

    calls = []

    def predictions(model_name, *args, **kwargs):
        calls.append(model_name)
        target = np.arange(24, dtype=float).reshape(3, 8)
        return target, target - 1

    monkeypatch.setattr(rasters, "get_model_predictions", predictions)
    results = _FakeResults()
    results.param_axes = {"model_name": ["external_placefield_1d", *rasters.PREDICTION_FIGURE_MODELS]}
    results.sel = lambda **kwargs: {"r2": np.array([0.1, 0.2])}
    performance = ModelPerformanceViewer(
        results, include_external_placefield=include_external_pf,
        include_structured_additive=include_structured_additive,
    )
    viewer = rasters.ModelPredictionsZoomFigureViewer(
        results, registry=object(), sort_method="activity", include_external_pf=include_external_pf,
        include_structured_additive=include_structured_additive,
    )
    fig = viewer.plot(viewer.state)
    expected = ["Target Data", "Int. Pos. PF", "Global Gain", "Peer Prediction"]
    if include_external_pf:
        expected.insert(1, "Placefield")
    if include_structured_additive:
        expected.insert(-1, "Shared Residual")
    assert [axis.texts[0].get_text() for axis in fig.axes[::2]] == expected
    assert list(viewer.model_names) == performance.model_names
    assert expected[1:] == [label.replace("\n", " ") for label in performance.model_labels]
    assert ("external_placefield_1d" in calls) == include_external_pf
    for overview, zoom in zip(fig.axes[::2], fig.axes[1::2]):
        assert len(overview.patches) == 1
        xstart = viewer.state["zoom_xslice_start"] - viewer.state["xslice_start"]
        xstop = viewer.state["zoom_xslice_stop"] - viewer.state["xslice_start"]
        np.testing.assert_array_equal(overview.images[0].get_array()[:, xstart:xstop], zoom.images[0].get_array())
    plt.close(fig)
