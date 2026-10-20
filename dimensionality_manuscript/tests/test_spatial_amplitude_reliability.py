"""Focused tests for the per-session spatial/amplitude reliability scatter viewer."""

from pathlib import Path
from types import SimpleNamespace
import sys
import types

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

package_name = "dimensionality_manuscript.figure_scripts.figure1"
if package_name not in sys.modules:
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[1] / "figure_scripts" / "figure1")]
    sys.modules[package_name] = package

from dimensionality_manuscript.figure_scripts.figure1 import placefield
from dimensionality_manuscript.figure_scripts.figure1.placefield import SpatialAmplitudeReliability


def _viewer(monkeypatch):
    session = SimpleNamespace(
        mouse_name="M1",
        date="2020-01-01",
        session_id="1",
        params=SimpleNamespace(spks_type="oasis"),
        session_print=lambda: "M1/2020-01-01/1",
    )
    results = SimpleNamespace(mouse_names=np.array(["M1"]), sessions=[session], param_axes={})
    arrays = {
        "trial_rms_cv": np.array([[[0.5, 0.25]]]),
        "trial_mean_variance": np.array([[[2.0, 3.0]]]),
        "reliability_slot": np.array([[[0.8, 0.25]]]),
        "fraction_active_slot": np.array([[[0.8, 0.2]]]),
        "env_slot_ids": np.array([[7.0]]),
    }

    def sel(*, keys, **kwargs):
        return {key: arrays[key] for key in keys}

    results.sel = sel
    monkeypatch.setattr(placefield.session_cache, "get_env_maps", lambda *args: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(placefield.session_cache, "get_reliability", lambda *args: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(placefield.session_cache, "get_fraction_active", lambda *args: (_ for _ in ()).throw(AssertionError))
    return SpatialAmplitudeReliability(
        results,
        mouse="M1",
        example_session="M1/2020-01-01/1",
        env=0,
    )


def test_scatter_has_trial_and_summary_dropdowns_and_one_point_per_finite_roi(monkeypatch):
    viewer = _viewer(monkeypatch)

    assert viewer.state["amplitude_metric"] == "trial_rms"
    assert viewer.state["amplitude_statistic"] == "cv"
    assert viewer.state["fraction_active_threshold"] == 0.1
    np.testing.assert_allclose(viewer.spatial_reliability, [0.8, 0.25])

    state = dict(viewer.state)
    fig = viewer.plot(state)
    offsets = fig.axes[0].collections[0].get_offsets()

    np.testing.assert_allclose(offsets[:, 0], viewer.spatial_reliability)
    np.testing.assert_allclose(offsets[:, 1], viewer.amplitude_summaries["trial_rms_cv"])
    assert fig.axes[0].get_xlabel() == "Spatial Reliability"
    assert "Trial RMS CV" in fig.axes[0].get_ylabel()


def test_fraction_active_threshold_filters_rois_in_the_selected_environment(monkeypatch):
    viewer = _viewer(monkeypatch)
    state = {
        **viewer.state,
        "fraction_active_threshold": 0.5,
    }

    fig = viewer.plot(state)
    offsets = fig.axes[0].collections[0].get_offsets()

    np.testing.assert_allclose(offsets[:, 0], [viewer.spatial_reliability[0]])
    np.testing.assert_allclose(offsets[:, 1], [viewer.amplitude_summaries["trial_rms_cv"][0]])
    assert "n=1" in fig.axes[0].get_title()


def test_changing_amplitude_selection_reloads_the_stored_key(monkeypatch):
    viewer = _viewer(monkeypatch)
    state = {**viewer.state, "amplitude_metric": "trial_mean", "amplitude_statistic": "variance"}

    viewer.reload_arrays(state)

    assert viewer.amplitude_key == "trial_mean_variance"
    np.testing.assert_allclose(viewer.amplitude_summaries["trial_mean_variance"], [2.0, 3.0])


@pytest.mark.parametrize("padded_key", ["trial_rms_cv", "reliability_slot", "fraction_active_slot"])
def test_load_arrays_trims_only_surplus_nan_padding(monkeypatch, padded_key):
    viewer = _viewer(monkeypatch)
    values = np.array([[[0.8, np.nan, 0.2]], [[0.4, np.nan, 0.6]]])
    arrays = {key: values.copy() for key in ("trial_rms_cv", "reliability_slot", "fraction_active_slot")}
    arrays[padded_key] = np.pad(values, ((0, 0), (0, 0), (0, 4)), constant_values=np.nan)
    arrays["env_slot_ids"] = np.array([[7.0], [7.0]])
    viewer.results.sel = lambda **kwargs: arrays

    viewer._load_arrays(viewer.state)

    for array in (viewer.amplitude_slot, viewer.reliability_slot, viewer.fraction_active_slot):
        np.testing.assert_allclose(array, values)
    assert arrays[padded_key].shape == (2, 1, 7)


def test_load_arrays_rejects_surplus_non_nan_rois(monkeypatch):
    viewer = _viewer(monkeypatch)
    original_sel = viewer.results.sel
    arrays = original_sel(keys=["trial_rms_cv", "reliability_slot", "fraction_active_slot", "env_slot_ids"])
    arrays["reliability_slot"] = np.array([[[0.8, 0.25, 0.9]]])
    viewer.results.sel = lambda **kwargs: arrays

    with pytest.raises(ValueError, match="matching shapes"):
        viewer._load_arrays(viewer.state)
