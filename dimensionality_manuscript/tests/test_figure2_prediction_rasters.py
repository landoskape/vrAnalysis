from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

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
        ["PF+Gain"],
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
    main_axes = fig.axes[:6]

    assert viewer.model_names == ("internal_placefield_1d_gain", "rrr")
    assert len(main_axes) == 6
    assert [[text.get_text() for text in axis.texts] for axis in main_axes[:3]] == [
        ["Target Data"],
        ["PF+Gain"],
        ["Peer Prediction"],
    ]
    assert "internal_placefield_1d_structured_additive" not in calls
    plt.close(fig)
