"""Focused tests for the direct-session place-field gain viewer helpers."""

from pathlib import Path
import inspect
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
    PlacefieldGainSortViewer,
    PlacefieldGainPerformanceViewer,
    gain_performance_comparison,
    PlacefieldGainViewer,
    _full_trial_indices,
    _values_by_mouse,
    gaussian_gain_matrix,
    least_squares_gain_matrix,
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
                "r2_test_raw": np.array([[0.1, 0.4], [0.2, 0.3]]),
                "r2_test_raw_null": np.array([[-0.1, -0.4], [-0.2, -0.3]]),
                "n_trials_env": np.array([[12, 4], [20, 20]]),
                "env_slot_ids": np.array([[1, 2], [1, 2]]),
            }

    viewer = object.__new__(PlacefieldGainViewer)
    viewer.results = Results()
    viewer.session = SimpleNamespace(session_uid="s1")
    viewer.refresh_summary(
        {
            "gain_estimator": "least_squares",
            "gain_transform": "sqrt",
            "min_trial": 10,
            "environment": 1,
        }
    )

    assert viewer.results.selection == {
        "keys": ["r2_test_raw", "r2_test_raw_null", "n_trials_env", "env_slot_ids"],
        "squeeze_ones": False,
        "gain_estimator": "least_squares",
        "gain_transform": "sqrt",
    }
    assert viewer.summary_mice == ["m1", "m2"]
    np.testing.assert_allclose(viewer.summary_values[0], [0.1])
    np.testing.assert_allclose(viewer.summary_values[1], [0.2, 0.3])
    np.testing.assert_allclose(viewer.summary_null_values[0], [-0.1])
    np.testing.assert_allclose(viewer.summary_null_values[1], [-0.2, -0.3])
    assert viewer.selected_result == (0, 0, 0.1)

    viewer.results.param_axes = {"trial_shift": list(range(-3, 4))}
    viewer.refresh_summary(
        {"gain_estimator": "least_squares", "gain_transform": "sqrt", "min_trial": 10, "environment": 1}
    )
    assert viewer.results.selection["trial_shift"] == 0


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
        assert fig.axes[2].get_ylabel() == r"Gain Prediction $R^2$ (raw gain)"
        assert [tick.get_text() for tick in fig.axes[2].get_xticklabels()] == ["m1", "m2"]
        np.testing.assert_allclose(np.diff(fig.axes[2].get_yticks()), 0.1)
        np.testing.assert_allclose(fig.axes[2].lines[1].get_ydata(), [0.15, 0.15])
        np.testing.assert_allclose(fig.axes[2].lines[2].get_ydata(), [-0.15, -0.15])
        assert fig.axes[2].lines[2].get_color() == "red"
        assert fig.axes[2].lines[-1].get_color() == "red"
        assert fig.axes[3].get_xlabel() == "Overall Session #"
    finally:
        plt.close(fig)


def test_gain_image_interpolates_each_discrete_trial_separately():
    values = np.arange(12, dtype=float).reshape(3, 4)
    fig, ax = plt.subplots()
    try:
        PlacefieldGainViewer._draw_trial_image(ax, values, cmap="coolwarm", interpolation="bilinear")
        assert len(ax.images) == values.shape[1]
        assert all(image.get_array().shape == (values.shape[0], 1) for image in ax.images)
        np.testing.assert_allclose([image.get_extent() for image in ax.images], [
            (-0.5, 0.5, 2.5, -0.5),
            (0.5, 1.5, 2.5, -0.5),
            (1.5, 2.5, 2.5, -0.5),
            (2.5, 3.5, 2.5, -0.5),
        ])
        np.testing.assert_allclose(ax.get_xlim(), (-0.5, 3.5))
        np.testing.assert_allclose(ax.get_ylim(), (2.5, -0.5))
    finally:
        plt.close(fig)


def test_sort_viewer_draws_two_fixed_orderings_with_aligned_positions():
    viewer = object.__new__(PlacefieldGainSortViewer)
    viewer.figsize = (5.0, 3.0)
    viewer.data = {
        "positions": np.linspace(0.0, 100.0, 11),
        "session": SimpleNamespace(env_length=np.array([100.0])),
    }
    viewer.sorted_gains = {
        "environment": np.arange(12, dtype=float).reshape(3, 4),
        "rastermap_gain": np.arange(12, dtype=float).reshape(3, 4)[[2, 0, 1]],
    }
    viewer.sorted_placefield_positions = {
        "environment": np.array([10.0, 40.0, 80.0]),
        "rastermap_gain": np.array([80.0, 10.0, 40.0]),
    }
    viewer._refresh_trial_order({"sort_trials": False})

    fig = viewer.plot({"gain_vmax": 4.0, "interpolation": "bilinear", "markersize": 4.0, "fontsize": 8.0, "sort_trials": False})
    try:
        assert len(fig.axes) == 4
        assert [text.get_text() for text in fig.axes[0].texts] == ["Place-field position sort"]
        assert [text.get_text() for text in fig.axes[2].texts] == ["Rastermap gain sort"]
        np.testing.assert_allclose(fig.axes[1].collections[0].get_offsets(), [[10.0, 0.0], [40.0, 1.0], [80.0, 2.0]])
        np.testing.assert_allclose(fig.axes[3].collections[0].get_offsets(), [[80.0, 0.0], [10.0, 1.0], [40.0, 2.0]])
        assert len(fig.axes[0].images) == 4
        assert len(fig.axes[2].images) == 4
        assert "sort_method" not in inspect.signature(PlacefieldGainSortViewer).parameters
        assert "include_covariance" not in inspect.signature(PlacefieldGainSortViewer).parameters
        np.testing.assert_allclose(fig.axes[1].collections[0].get_sizes(), [16.0])
        assert fig.axes[0].get_xlabel() == ""
        assert fig.axes[1].get_xlabel() == ""
        assert fig.axes[2].get_xlabel() == "Trials"
        assert fig.axes[3].get_xlabel() == "PF Pos (cm)"
        assert fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[2])
        assert fig.axes[1].get_shared_x_axes().joined(fig.axes[1], fig.axes[3])
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
        assert fig.axes[1].get_ylabel() == r"Gain Prediction $R^2$ (raw gain)"
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
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Data", "Shuffle"]
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


def test_least_squares_gain_recovers_multipliers():
    prediction = np.array([[0.0, 1.0, 3.0, 2.0]])
    multipliers = np.array([0.25, 1.0, 2.5])
    trial_maps = prediction[:, None, :] * multipliers[None, :, None]
    gain, fitted = least_squares_gain_matrix(trial_maps, prediction)
    np.testing.assert_allclose(gain[0], multipliers)
    np.testing.assert_allclose(fitted, prediction)



def _comparison_results():
    from dimensionality_manuscript.pipeline import ResultsAggregator
    axes = {"gain_estimator": ["gaussian", "least_squares"], "gain_transform": ["raw", "sqrt"], "trial_shift": list(range(-3, 4))}
    shape = (3, 2, 2, 7, 3)
    score = np.full(shape, -999.)  # Any accidental ±2/±3 selection will be obvious.
    null = np.full(shape, -888.)
    trials = np.full(shape, 30.)
    base = np.array([[.1, .3, np.nan], [.5, .7, np.nan], [.9, 1.1, np.nan]])
    for estimator in range(2):
        for transform in range(2):
            for shift, offset in ((0, 0), (-1, -.04), (1, -.08)):
                index = (slice(None), estimator, transform, shift + 3, slice(None))
                score[index] = base + transform * .01 + estimator * .001 + offset
                null[index] = base + transform * .01 + estimator * .001 - .09
    trials[1, :, :, :, 1] = 5  # Threshold excludes the second environment of session 2.
    score[2, :, 1, 4, 1] = np.nan  # Missing sqrt +1 excludes this observation from ALL groups.
    return ResultsAggregator._from_data(
        {"r2_test_raw": score, "r2_test_raw_null": null, "n_trials_env": trials,
         "r2_test": np.full(shape, 1234.)},
        axes, ["s1", "s2", "s3"], np.array(["m1", "m1", "m2"]),
    )


def test_gain_comparison_pairs_transforms_lags_and_rolls_in_raw_units():
    results = _comparison_results()
    calls = []
    original_sel = results.sel
    def recording_sel(**kwargs):
        calls.append(kwargs)
        return original_sel(**kwargs)
    results.sel = recording_sel
    scores = gain_performance_comparison(results, "gaussian", min_trial=10)
    assert scores.shape == (3, 3, 2, 3)
    np.testing.assert_allclose(scores[0, 0], [[.1, .04, .01], [.11, .05, .02]], atol=1e-12)
    assert np.all(np.isnan(scores[1, 1]))  # Threshold.
    assert np.all(np.isnan(scores[2, 1]))  # Missing one lag is never treated as a two-lag mean.
    assert np.all(np.isnan(scores[:, 2]))  # Unvisited environment slot.
    assert {call["trial_shift"] for call in calls} == {-1, 0, 1}
    assert all("r2_test" not in call["keys"] for call in calls)
    assert all(("r2_test_raw_null" in call["keys"]) == (call["trial_shift"] == 0) for call in calls)
    assert len(calls) == 6
    # Choosing the estimator must affect the data rather than silently pooling it.
    least_squares = gain_performance_comparison(_comparison_results(), "least_squares", 10)
    np.testing.assert_allclose(least_squares[0, 0], scores[0, 0] + .001)


def test_gain_comparison_viewer_hierarchical_averages_and_mouse_filter():
    viewer = PlacefieldGainPerformanceViewer(_comparison_results())
    subjects, values = viewer.subject_scores(viewer.state)
    assert subjects == ["m1", "m2"]
    # m1: average environments within each session (.2 and .5), then sessions (.35).
    np.testing.assert_allclose(values[:, 0], [.35, 1.])
    np.testing.assert_allclose(values[:, 1], [.29, .94])
    np.testing.assert_allclose(values[:, 2], [.26, .91])
    assert values.shape == (2, 3)
    state = dict(viewer.state, avg_by_mouse=False, mouse="m1")
    subjects, values = viewer.subject_scores(state)
    assert subjects == ["s1", "s2"]
    np.testing.assert_allclose(values[:, 0], [.2, .5])
    fig = viewer.plot(viewer.state)
    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.get_xticklabels()) == 3
        assert "raw gain" in ax.get_ylabel()
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Same trial", "Mean lag ±1", "Random roll"]
        assert ax.get_title() == "Raw gain fit"
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [.35, 1.])
        np.testing.assert_allclose(ax.lines[2].get_ydata(), [.29, .94])
        np.testing.assert_allclose(ax.lines[4].get_ydata(), [.26, .91])
    finally:
        plt.close(fig)


def test_gain_comparison_missing_lag_grid_and_empty_results():
    results = _comparison_results()
    results.param_axes["trial_shift"] = [0]
    with np.testing.assert_raises_regex(ValueError, "trial_shift"):
        gain_performance_comparison(results, "gaussian")
    viewer = PlacefieldGainPerformanceViewer(_comparison_results(), min_trial=100)
    subjects, values = viewer.subject_scores(viewer.state)
    assert subjects == ["m1", "m2"]
    assert np.all(np.isnan(values))
    fig = viewer.plot(viewer.state)
    plt.close(fig)



def test_gain_comparison_selects_sqrt_score_scale_for_both_fits_and_controls():
    results = _comparison_results()
    results.arrays["r2_test_sqrt"] = results.arrays["r2_test_raw"] + .2
    results.arrays["r2_test_sqrt_null"] = results.arrays["r2_test_raw_null"] + .3
    raw = gain_performance_comparison(results, "gaussian")
    root = gain_performance_comparison(results, "gaussian", score_scale="sqrt")
    np.testing.assert_allclose(root[..., :2], raw[..., :2] + .2)
    np.testing.assert_allclose(root[..., 2], raw[..., 2] + .3)
    viewer = PlacefieldGainPerformanceViewer(results, score_scale="sqrt")
    fig = viewer.plot(viewer.state)
    try:
        assert "sqrt gain" in fig.axes[0].get_ylabel()
        np.testing.assert_allclose(
            viewer.comparison_scores,
            gain_performance_comparison(results, "gaussian", score_scale="sqrt", gain_transform="raw"),
        )
    finally:
        plt.close(fig)



def test_gain_comparison_loads_only_selected_fit_and_its_controls():
    results = _comparison_results()
    calls = []
    original_sel = results.sel
    def recording_sel(**kwargs):
        calls.append(kwargs)
        return original_sel(**kwargs)
    results.sel = recording_sel
    viewer = PlacefieldGainPerformanceViewer(results, gain_transform="sqrt")
    assert len(calls) == 3
    assert all(call["gain_transform"] == "sqrt" for call in calls)
    subjects, scores = viewer.subject_scores(viewer.state)
    np.testing.assert_allclose(scores[:, 0], [.36, .91])
    fig = viewer.plot(viewer.state)
    try:
        assert fig.axes[0].get_title() == "Sqrt gain fit"
        assert len(fig.axes[0].get_xticklabels()) == 3
    finally:
        plt.close(fig)
    raw = gain_performance_comparison(results, "gaussian", gain_transform="raw")
    assert raw.shape == (3, 3, 1, 3)
    assert np.isfinite(raw[2, 1]).all()  # Missing sqrt lag must not hide valid raw data.
