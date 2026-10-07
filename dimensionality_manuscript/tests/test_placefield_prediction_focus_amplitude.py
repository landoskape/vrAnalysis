"""Focused tests for the optional trial-amplitude column in PlaceFieldPredictionFocus."""

from pathlib import Path
from types import SimpleNamespace
import sys
import types

import matplotlib
import numpy as np

matplotlib.use("Agg")

# Avoid importing every figure-1 panel through figure1.__init__; some optional panels have heavy
# dependencies unrelated to this focused viewer test.
package_name = "dimensionality_manuscript.figure_scripts.figure1"
if package_name not in sys.modules:
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[1] / "figure_scripts" / "figure1")]
    sys.modules[package_name] = package

from dimensionality_manuscript.figure_scripts.figure1 import placefield
from dimensionality_manuscript.figure_scripts.figure1.placefield import PlaceFieldPredictionFocus


def _viewer(monkeypatch, *, include_amplitude_summary=True):
    session = SimpleNamespace(
        mouse_name="M1",
        date="2020-01-01",
        session_id="1",
        params=SimpleNamespace(spks_type="oasis"),
        session_print=lambda: "M1/2020-01-01/1",
    )
    results = SimpleNamespace(
        mouse_names=np.array(["M1"]),
        sessions=[session],
        param_axes={},
    )
    spkmap = np.array(
        [
            [
                [0.0, 1.0, 3.0, 1.0, 0.0],
                [0.0, 2.0, 6.0, 2.0, 0.0],
                [0.0, 0.5, 1.5, 0.5, 0.0],
            ]
        ]
    )
    env_maps = SimpleNamespace(
        environments=np.array([7]),
        distcenters=np.linspace(0.0, 200.0, 5),
        spkmap=[spkmap],
    )
    measured = placefield._trial_activity_measurements(spkmap, env_maps.distcenters)
    summaries = placefield._summarize_trial_activity(measured)
    precomputed = {
        "env_slot_ids": np.array([[7.0]]),
        "reliability_slot": np.array([[[placefield._trial_consistency(spkmap[0])[3]]]]),
        **{key: values[None, None, :] for key, values in summaries.items()},
    }
    results.sel = lambda *, keys, **kwargs: {key: precomputed[key] for key in keys}
    monkeypatch.setattr(placefield.session_cache, "get_env_maps", lambda selected_session: env_maps)
    return PlaceFieldPredictionFocus(
        results,
        mouse="M1",
        example_session="M1/2020-01-01/1",
        include_amplitude_summary=include_amplitude_summary,
        show_prediction=False,
    )


def test_amplitude_column_uses_v9_metrics_and_dropdown_defaults(monkeypatch):
    viewer = _viewer(monkeypatch)

    assert viewer.state["include_amplitude_summary"] is True
    assert viewer.state["amplitude_metric"] == "trial_rms"
    assert viewer.state["amplitude_statistic"] == "mean"
    np.testing.assert_allclose(
        viewer.trial_activity["trial_rms"],
        np.sqrt(np.mean(viewer.spkmap**2, axis=1)),
    )
    assert viewer.trial_activity_summary["trial_rms_mean"] == np.mean(viewer.trial_activity["trial_rms"])


def test_amplitude_column_draws_after_reliability_and_honors_statistic_selection(monkeypatch):
    viewer = _viewer(monkeypatch)
    state = {**viewer.state, "amplitude_metric": "trial_rms", "amplitude_statistic": "variance"}

    fig = viewer.plot(state)
    amplitude_axis = next(axis for axis in fig.axes if axis.get_ylabel() == "Trial RMS")
    expected = viewer.trial_activity_summary["trial_rms_variance"]
    summary_axis = next(axis for axis in fig.axes if any(text.get_text() == "Variance" for text in axis.texts))
    consistency_axis = next(axis for axis in fig.axes if any("corr" in text.get_text() for text in axis.texts))

    assert consistency_axis.get_position().x0 < amplitude_axis.get_position().x0
    assert consistency_axis.get_xlim() == (-1.1, 1.1)
    assert amplitude_axis.spines["bottom"].get_visible()
    assert amplitude_axis.get_xlim() == (0.0, np.ceil(np.nanmax(viewer.trial_activity["trial_rms"])))
    assert len(amplitude_axis.get_xticks()) == 2
    assert summary_axis.get_xlabel() == ""
    assert summary_axis.get_xlim() == (
        0.0,
        np.ceil(np.nanmax(viewer.session_activity_summary["trial_rms_variance"])),
    )
    assert len(summary_axis.patches) == state["nbins"]
    point_lines = [line for line in summary_axis.lines if line.get_marker() == "o"]
    assert len(point_lines) == 1
    np.testing.assert_allclose(point_lines[0].get_xdata(), [expected])
    assert point_lines[0].get_linestyle() == "None"


def test_bottom_histograms_share_nbins_and_mark_selected_roi(monkeypatch):
    viewer = _viewer(monkeypatch)
    state = {**viewer.state, "nbins": 7}

    fig = viewer.plot(state)

    reliability_axis = next(axis for axis in fig.axes if any(text.get_text() == "W. Avg." for text in axis.texts))
    summary_axis = next(axis for axis in fig.axes if any(text.get_text() == "Mean" for text in axis.texts))
    assert reliability_axis.get_xlim() == (-1.1, 1.1)
    assert len(reliability_axis.patches) == 7
    assert len(summary_axis.patches) == 7
    assert all(line.get_linestyle() == "None" for axis in (reliability_axis, summary_axis) for line in axis.lines)


def test_amplitude_column_is_absent_when_disabled(monkeypatch):
    viewer = _viewer(monkeypatch, include_amplitude_summary=False)

    fig = viewer.plot(viewer.state)

    assert not any(axis.get_ylabel() == "Trial RMS" for axis in fig.axes)
