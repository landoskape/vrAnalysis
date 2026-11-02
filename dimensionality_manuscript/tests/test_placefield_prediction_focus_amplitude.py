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
        session_uid="M1/2020-01-01/1",
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
    from dimensionality_manuscript.configs.pfpred_quality import PFPredQualityConfig
    results.config_class = PFPredQualityConfig
    spkmap = np.array(
        [
            [
                [0.0, 1.0, 3.0, 1.0, 0.0],
                [0.0, 2.0, 6.0, 2.0, 0.0],
                [0.0, 0.5, 1.5, 0.5, 0.0],
                [1.0, 3.0, 4.0, 1.0, 0.0],
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
    from dimensionality_manuscript.configs.pfpred_quality import _placefield_rms_maps
    spatial = _placefield_rms_maps(spkmap)
    precomputed = {
        "env_slot_ids": np.array([7.0]),
        "reliability_slot": np.array([[placefield._trial_consistency(spkmap[0])[3]]]),
        **{key: values[None, :] for key, values in summaries.items()},
        **{f"{key}_slot": values[None] for key, values in spatial.items()},
        "pf_rms_distcenters": env_maps.distcenters,
    }
    results.store = SimpleNamespace(get=lambda session_uid, config: precomputed)
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


def test_pf_rms_scatter_histogram_and_overlay_follow_comparison_selection(monkeypatch):
    viewer = _viewer(monkeypatch, include_amplitude_summary=False)
    state = {**viewer.state, "pf_rms_comparison": "independent_f1", "nbins": 6}
    fig = viewer.plot(state)
    fig.canvas.draw()
    pf = viewer.session_pf_rms["pf_f1"][0]
    rms = viewer.session_pf_rms["rms_f2_from_f2"][0]
    r = viewer.session_pf_rms["pf_rms_r_independent_f1"][0]
    scatter = next(axis for axis in fig.axes if axis.get_title().startswith("r ="))
    np.testing.assert_allclose(scatter.collections[0].get_offsets(), np.column_stack([pf, rms]))
    assert scatter.get_title() == f"r = {r:.3f}"
    histogram = next(axis for axis in fig.axes if any(text.get_text() == "PF–RMS r" for text in axis.texts))
    assert len(histogram.patches) == 6
    assert histogram.lines[0].get_xdata()[0] == r
    overlay = next(axis for axis in fig.axes if any(text.get_text() == "PF f1" for text in axis.texts))
    np.testing.assert_allclose(overlay.lines[0].get_ydata(), pf)
    np.testing.assert_allclose(overlay.lines[1].get_ydata(), rms)
    assert sum(len(axis.images) for axis in fig.axes) == 1


def test_focus_reuses_one_session_result_without_aggregating_maps(monkeypatch):
    viewer = _viewer(monkeypatch)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Changing ROI must reuse the cached session result")

    monkeypatch.setattr(viewer.results.store, "get", unexpected_load)
    viewer.refresh_data(viewer.state)
    assert viewer.session_pf_rms["pf_f1"].shape == (1, 5)
