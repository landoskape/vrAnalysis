"""Stored score selection, mouse weighting, and layout of the gain viewer."""

from pathlib import Path
import sys
import types

import numpy as np
import pytest
from matplotlib import pyplot as plt

# Keep unrelated Figure 2 viewers and their optional dependencies out of this test.
package_name = "dimensionality_manuscript.figure_scripts.figure2"
if package_name not in sys.modules:
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[1] / "figure_scripts" / "figure2")]
    sys.modules[package_name] = package

from dimensionality_manuscript.figure_scripts.figure2.gain_regression import GainRegressionViewer
from dimensionality_manuscript.pipeline import ResultsAggregator


def _results():
    axes = dict(placefield_split=["all", "train"], gain_estimator=["gaussian", "least_squares"],
                gain_transform=["raw", "sqrt"], trial_shift=list(range(-3, 4)))
    shape = (3, 2, 2, 2, 7, 2)
    score = np.empty(shape)
    for pf in range(2):
        for estimator in range(2):
            for transform in range(2):
                for lag in range(7):
                    score[:, pf, estimator, transform, lag] = (
                        np.array([[1., 3.], [5., 7.], [9., 11.]])
                        + pf * 100 + estimator * 10 + transform * 1000 + lag
                    )
    trials = np.full(shape, 30.)
    trials[1, ..., 1] = 5  # One environment excluded, before session averaging.
    return ResultsAggregator._from_data(
        {"r2_test_raw": score, "r2_test_raw_null": score - 2,
         "r2_test_sqrt": score + 0.5, "r2_test_sqrt_null": score - 1.5,
         "mse_test_raw": score + 20, "mse_test_raw_null": score + 22,
         "n_trials_env": trials}, axes, ["s1", "s2", "s3"], np.array(["m1", "m1", "m2"]),
    )


def test_selection_and_hierarchical_mouse_weighting():
    viewer = GainRegressionViewer(_results(), placefield_split="train", gain_estimator="least_squares",
                                  gain_transform="sqrt", score_scale="sqrt")
    mice, comparison, trajectory = viewer.mouse_scores(viewer.state)
    assert mice == ["m1", "m2"]
    # m1 averages session means 2 and 5, rather than pooling three environment values.
    np.testing.assert_allclose(trajectory, np.array([[3.5], [10.]]) + np.arange(7) + 1110.5)
    np.testing.assert_allclose(comparison, trajectory[:, 3, None] + np.array([0., -2.]))
    state = dict(viewer.state, mouse="m2")
    assert viewer.mouse_scores(state)[0] == ["m2"]


def test_pairing_is_independent_between_rows():
    results = _results()
    results.arrays["r2_test_raw"][0, 0, 0, 0, 0, 0] = np.nan
    results.arrays["r2_test_raw_null"][2, 0, 0, 0, 3, 1] = np.nan
    viewer = GainRegressionViewer(results)
    assert np.all(np.isnan(viewer.trajectory_scores[0, 0]))
    assert np.all(np.isfinite(viewer.comparison_scores[0, 0]))
    assert np.all(np.isnan(viewer.comparison_scores[2, 1]))
    assert np.all(np.isfinite(viewer.trajectory_scores[2, 1]))


@pytest.mark.parametrize("display", ["each", "errorPlot"])
def test_two_rows_render_with_horizontal_labels_and_full_lag_range(display):
    viewer = GainRegressionViewer(_results(), display=display)
    fig = viewer.plot(viewer.state)
    try:
        fig.canvas.draw()
        assert len(fig.axes) == 2
        top, bottom = fig.axes
        assert [label.get_text() for label in top.get_xticklabels()] == ["Data", "Shuffle"]
        assert all(label.get_rotation() == 0 for label in top.get_xticklabels())
        np.testing.assert_equal(bottom.get_xticks(), np.arange(-3, 4))
        assert [label.get_text() for label in bottom.get_xticklabels()] == ["-3", "", "", "0", "", "", "3"]
        assert bottom.get_xlabel() == "Trial Lag"
        assert bottom.get_ylabel() == r"Gain Pred. $R^2$"
        assert all(line.get_color() == "k" for line in top.lines)
        if display == "errorPlot":
            assert len(bottom.collections) == 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("plot_structure", ["split", "condensed"])
def test_empty_results_and_single_mouse_render(plot_structure):
    results = _results()
    viewer = GainRegressionViewer(results, mouse="m2", display="errorPlot", plot_structure=plot_structure)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    results.arrays["n_trials_env"][:] = 0
    viewer.refresh_data(viewer.state)
    fig = viewer.plot(viewer.state)
    try:
        fig.canvas.draw()
        assert all(any(text.get_text() == "No matched results" for text in ax.texts) for ax in fig.axes)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("display", ["each", "errorPlot"])
def test_condensed_shuffle_controls_and_label(display):
    viewer = GainRegressionViewer(_results(), display=display, plot_structure="condensed")
    _, comparison, _ = viewer.mouse_scores(viewer.state)
    shuffle = comparison[:, 1]
    fig = viewer.plot(viewer.state)
    try:
        fig.canvas.draw()
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        dotted = [line for line in ax.lines if line.get_linestyle() == ":"]
        if display == "each":
            assert len(dotted) == 2
            np.testing.assert_allclose([line.get_ydata()[0] for line in dotted], shuffle)
            lower = np.min(shuffle)
        else:
            assert len(dotted) == 1
            np.testing.assert_allclose(dotted[0].get_ydata(), np.mean(shuffle))
            assert len(ax.collections) == 2  # True and shuffle SE bands.
            lower = np.mean(shuffle) - np.std(shuffle) / np.sqrt(len(shuffle))
        label = next(text for text in ax.texts if text.get_text() == "shuffle")
        assert label.xy == (0, lower)
        assert label.get_position()[1] < 0
    finally:
        plt.close(fig)


def test_metric_selection_and_missing_lags():
    results = _results()
    viewer = GainRegressionViewer(results, metric="mse")
    _, comparison, trajectory = viewer.mouse_scores(viewer.state)
    np.testing.assert_allclose(comparison[:, 1], comparison[:, 0] + 2)
    np.testing.assert_allclose(trajectory[:, 3], comparison[:, 0])
    results.param_axes["trial_shift"] = [0]
    with pytest.raises(ValueError, match="-3 through"):
        GainRegressionViewer(results)


@pytest.mark.parametrize("plot_structure", ["split", "condensed"])
@pytest.mark.parametrize("display", ["each", "errorPlot"])
def test_relative_uses_each_mouse_shuffle_before_population_summary(plot_structure, display):
    viewer = GainRegressionViewer(_results(), plot_structure=plot_structure, display=display)
    _, absolute, trajectory = viewer.mouse_scores(viewer.state)
    state = dict(viewer.state, value="relative")
    _, relative, relative_trajectory = viewer.mouse_scores(state)
    np.testing.assert_allclose(relative, absolute[:, 1:2] - absolute)
    np.testing.assert_allclose(relative_trajectory, absolute[:, 1:2] - trajectory)
    np.testing.assert_allclose(relative[:, 1], 0)
    fig = viewer.plot(state)
    try:
        fig.canvas.draw()
        ax = fig.axes[-1]
        assert ax.get_ylabel() == r"Gain Pred. $\Delta$ $R^2$"
        if plot_structure == "condensed":
            for line in ax.lines:
                if line.get_linestyle() == ":":
                    np.testing.assert_allclose(line.get_ydata(), 0)
        for axis in (ax.xaxis, ax.yaxis):
            tick = axis.get_major_ticks()[0]
            assert tick.tick1line.get_markersize() == 2.0
            assert tick.tick1line.get_markeredgewidth() == 0.6
            assert tick.label1.get_fontsize() == 8.0
    finally:
        plt.close(fig)


def test_relative_excludes_mouse_without_shuffle():
    results = _results()
    results.arrays["r2_test_raw_null"][2] = np.nan
    viewer = GainRegressionViewer(results, value="relative")
    _, comparison, trajectory = viewer.mouse_scores(viewer.state)
    assert np.all(np.isnan(comparison[1]))
    assert np.all(np.isnan(trajectory[1]))
