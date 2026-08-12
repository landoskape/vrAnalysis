"""Focused tests for the direct-session place-field gain viewer helpers."""

from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt

# Avoid importing every Figure 2 viewer (several eagerly import Rastermap) in this
# numerical helper test.
package_name = "dimensionality_manuscript.figure_scripts.figure2"
if package_name not in sys.modules:
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[1] / "figure_scripts" / "figure2")]
    sys.modules[package_name] = package

from dimensionality_manuscript.figure_scripts.figure2.placefield_gain import (
    PlacefieldGainViewer,
    _full_trial_indices,
    _values_by_mouse,
    gaussian_gain_matrix,
    standardize_by_std,
    threshold_gain_matrix,
)


def test_values_by_mouse_flattens_session_and_environment_scores():
    values = np.array([[0.1, np.nan], [0.2, 0.3], [0.4, 0.5]])
    mice, grouped = _values_by_mouse(values, np.array(["m1", "m2", "m1"]))
    assert mice == ["m1", "m2"]
    np.testing.assert_allclose(grouped[0], [0.1, 0.4, 0.5])
    np.testing.assert_allclose(grouped[1], [0.2, 0.3])


def test_values_by_mouse_requires_matching_session_axis():
    with np.testing.assert_raises_regex(ValueError, "same session axis"):
        _values_by_mouse(np.ones((2, 3)), np.array(["m1"]))


def test_summary_selection_filters_trials_and_locates_selected_result():
    class Results:
        mouse_names = np.array(["m1", "m2"])
        session_ids = ["s1", "s2"]

        def sel(self, **kwargs):
            self.selection = kwargs
            return {
                "r2_test": np.array([[0.1, 0.4], [0.2, 0.3]]),
                "r2_test_null": np.array([[-0.1, -0.4], [-0.2, -0.3]]),
                "n_trials_env": np.array([[12, 4], [20, 20]]),
                "env_slot_ids": np.array([[1, 2], [1, 2]]),
            }

    viewer = object.__new__(PlacefieldGainViewer)
    viewer.results = Results()
    viewer.session = SimpleNamespace(session_uid="s1")
    viewer.refresh_summary({"gain_transform": "sqrt", "placefield_split": "all", "min_trial": 10, "environment": 1})

    assert viewer.results.selection == {
        "keys": ["r2_test", "r2_test_null", "n_trials_env", "env_slot_ids"],
        "squeeze_ones": False,
        "gain_transform": "sqrt",
        "placefield_split": "all",
    }
    assert viewer.summary_mice == ["m1", "m2"]
    np.testing.assert_allclose(viewer.summary_values[0], [0.1])
    np.testing.assert_allclose(viewer.summary_values[1], [0.2, 0.3])
    np.testing.assert_allclose(viewer.summary_null_values[0], [-0.1])
    np.testing.assert_allclose(viewer.summary_null_values[1], [-0.2, -0.3])
    assert viewer.selected_result == (0, 0, 0.1)


def test_plot_adds_summary_and_optional_familiarity_panels():
    sessions = [
        SimpleNamespace(date="2023-01-01", session_id="1"),
        SimpleNamespace(date="2023-01-02", session_id="2"),
    ]
    viewer = object.__new__(PlacefieldGainViewer)
    viewer.figsize = (8.0, 3.0)
    viewer.gain = np.arange(12, dtype=float).reshape(3, 4)
    viewer.covariance = np.eye(3)
    viewer.results = SimpleNamespace(mouse_names=np.array(["m1", "m2"]), sessions=sessions)
    viewer.summary_mice = ["m1", "m2"]
    viewer.summary_values = [np.array([0.1, 0.2]), np.array([0.3])]
    viewer.summary_null_values = [np.array([-0.1, -0.2]), np.array([-0.3])]
    viewer.summary_scores = np.array([[0.1, 0.2, np.nan], [0.3, np.nan, np.nan]])
    viewer.summary_null_scores = np.array([[-0.1, -0.2, np.nan], [-0.3, np.nan, np.nan]])
    viewer.summary_trials = np.array([[20.0, 20.0, np.nan], [20.0, np.nan, np.nan]])
    viewer.selected_result = (0, 0, 0.1)
    state = {
        "gain_vmax": 4.0,
        "covariance_vmax": 0.5,
        "interpolation": "none",
        "fontsize": 8.0,
        "beewidth": 0.2,
        "swarm_mode": "by_mouse",
        "include_covariance": True,
        "include_familiarity": True,
        "by_env": False,
    }

    fig = viewer.plot(state)
    try:
        assert len(fig.axes) == 4
        assert fig.axes[2].get_ylabel() == r"Gain Prediction $R^2$"
        assert [tick.get_text() for tick in fig.axes[2].get_xticklabels()] == ["m1", "m2"]
        np.testing.assert_allclose(np.diff(fig.axes[2].get_yticks()), 0.1)
        np.testing.assert_allclose(fig.axes[2].lines[1].get_ydata(), [0.15, 0.15])
        np.testing.assert_allclose(fig.axes[2].lines[2].get_ydata(), [-0.15, -0.15])
        assert fig.axes[2].lines[2].get_color() == "red"
        assert fig.axes[2].lines[-1].get_color() == "red"
        assert fig.axes[3].get_xlabel() == "Overall Session #"
    finally:
        plt.close(fig)


def test_plot_can_omit_covariance_panel():
    viewer = object.__new__(PlacefieldGainViewer)
    viewer.figsize = (8.0, 3.0)
    viewer.gain = np.arange(12, dtype=float).reshape(3, 4)
    viewer.covariance = np.eye(3)
    viewer.results = SimpleNamespace(mouse_names=np.array(["m1"]), sessions=[])
    viewer.summary_mice = ["m1"]
    viewer.summary_values = [np.array([0.1])]
    viewer.summary_null_values = [np.array([-0.1])]
    viewer.summary_scores = np.array([[0.1]])
    viewer.summary_null_scores = np.array([[-0.1]])
    viewer.summary_trials = np.array([[20.0]])
    viewer.selected_result = None
    state = {
        "gain_vmax": 4.0,
        "covariance_vmax": 0.5,
        "interpolation": "none",
        "fontsize": 8.0,
        "beewidth": 0.2,
        "swarm_mode": "by_mouse",
        "include_covariance": False,
        "include_familiarity": False,
        "by_env": False,
    }

    fig = viewer.plot(state)
    try:
        assert len(fig.axes) == 2
        assert fig.axes[0].get_xlabel() == "Trials"
        assert fig.axes[0].get_box_aspect() is None
        assert fig.axes[1].get_ylabel() == r"Gain Prediction $R^2$"
        assert all(text.get_text() != "Gain Covariance" for ax in fig.axes for text in ax.texts)
    finally:
        plt.close(fig)


def test_pooled_summary_averages_envs_then_sessions_then_mice():
    viewer = object.__new__(PlacefieldGainViewer)
    viewer.results = SimpleNamespace(mouse_names=np.array(["m1", "m1", "m2"]))
    viewer.summary_mice = ["m1", "m2"]
    viewer.summary_scores = np.array([[0.1, 0.3], [0.5, np.nan], [0.7, 0.9]])
    viewer.summary_null_scores = np.array([[-0.1, -0.3], [-0.5, np.nan], [-0.7, -0.9]])
    viewer.selected_result = None
    fig, ax = plt.subplots()
    try:
        viewer._draw_summary(ax, {"swarm_mode": "pooled", "beewidth": 0.2, "fontsize": 8.0})
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.35, 0.8])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.575, 0.575])
        np.testing.assert_allclose(ax.lines[2].get_ydata(), [-0.35, -0.8])
        np.testing.assert_allclose(ax.lines[3].get_ydata(), [-0.575, -0.575])
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Data", "Roll Null"]
    finally:
        plt.close(fig)


def test_by_env_familiarity_uses_chronology_and_sessions_within_env():
    viewer = object.__new__(PlacefieldGainViewer)
    viewer.results = SimpleNamespace(
        mouse_names=np.array(["m1", "m1", "m2"]),
        sessions=[
            SimpleNamespace(date="2023-01-02", session_id="2"),
            SimpleNamespace(date="2023-01-01", session_id="1"),
            SimpleNamespace(date="2023-01-03", session_id="1"),
        ],
    )
    viewer.summary_scores = np.array([[0.3, 0.4, np.nan], [0.1, np.nan, np.nan], [0.6, 0.8, 0.9]])
    viewer.summary_trials = np.array([[20.0, 20.0, np.nan], [20.0, np.nan, np.nan], [20.0, 20.0, 20.0]])

    curves = viewer._familiarity_curves(by_env=True)
    assert [label for label, _, _ in curves] == ["Env #1", "Env #2", "Env #3"]
    np.testing.assert_allclose(curves[0][2]["m1"], [0.1, 0.3])
    np.testing.assert_allclose(curves[1][2]["m1"], [0.4])
    np.testing.assert_allclose(curves[2][2]["m2"], [0.9])


def test_full_trial_filter_uses_required_bins_and_allows_edge_flexibility():
    occupancy = np.array(
        [
            [np.nan, 0.0, 1.0, 1.0, 0.0, np.nan],
            [np.nan, 0.0, 1.0, np.nan, 0.0, np.nan],
            [np.nan, np.nan, 1.0, 1.0, 0.0, np.nan],
        ]
    )
    # Edge bins 0 and 5 are outside the required range. Zero occupancy is valid:
    # Spkmap's full-trial test concerns positional coverage (NaNs), not activity.
    np.testing.assert_array_equal(_full_trial_indices(occupancy, np.arange(1, 5)), [0])


def test_standardize_by_std_does_not_center_activity():
    activity = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 6.0], [5.0, 5.0, 10.0]])
    scaled, valid = standardize_by_std(activity)
    np.testing.assert_array_equal(valid, [True, False, True])
    np.testing.assert_allclose(np.nanstd(scaled[:, valid], axis=0), 1.0)
    np.testing.assert_allclose(np.nanmean(scaled[:, 0]), np.mean(activity[:, 0]) / np.std(activity[:, 0]))
    assert np.all(np.isnan(scaled[:, 1]))


def test_threshold_gain_uses_only_bins_above_threshold():
    prediction = np.array([[0.5, 2.0, 4.0, 0.25], [0.1, 0.2, 0.3, 0.4]])
    trial_maps = np.array(
        [
            [[99.0, 1.0, 2.0, 99.0], [99.0, 4.0, 8.0, 99.0]],
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
        ]
    )
    gain = threshold_gain_matrix(trial_maps, prediction, threshold=1.0)
    np.testing.assert_allclose(gain[0], [0.5, 2.0])
    assert np.all(np.isnan(gain[1]))


def test_gaussian_gain_recovers_multipliers():
    positions = np.linspace(0.0, 10.0, 51)
    prediction = 0.2 + 2.5 * np.exp(-0.5 * ((positions - 4.0) / 1.2) ** 2)
    multipliers = np.array([0.5, 1.0, 1.75])
    trial_maps = multipliers[:, None] * prediction[None, :]
    gain, fitted = gaussian_gain_matrix(trial_maps[None, :, :], prediction[None, :], positions)
    np.testing.assert_allclose(fitted[0], prediction, atol=1e-5)
    np.testing.assert_allclose(gain[0], multipliers, atol=1e-5)
