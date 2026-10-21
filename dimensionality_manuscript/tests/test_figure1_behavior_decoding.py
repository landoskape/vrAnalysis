import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure1 import SpeedEnvironmentDecoding


class _Results:
    param_axes = {"regularization": [0.1, 1.0]}
    mouse_names = np.array(["a", "a", "b", "a", "c", "d", "e", "f"])

    def __init__(self):
        # The canonical environment axis always has three columns, even in 1/2-env sessions.
        self.arrays = {
            "n_trials_per_env": np.array([[10, 10, 0]] * 3 + [[10, 10, 10]] * 2 + [[10, 0, 0], [10, 10, 10], [10, 10, 0]]),
            "env_used": np.array([[1, 1, 0]] * 3 + [[1, 1, 1]] * 2 + [[np.nan] * 3, [1, 1, 0], [1, 1, 0]]),
            "acc_test_random": np.array([0.6, 0.8, 0.9, 0.7, 0.9, np.nan, 0.99, np.nan]),
            "acc_test_block": np.array([0.5, 0.7, np.nan, 0.6, 0.8, np.nan, 0.99, 0.4]),
            "chance_random": np.array([0.55, 0.65, 0.7, 1 / 3, 0.4, np.nan, 0.5, np.nan]),
            "chance_block": np.array([0.5, 0.5, np.nan, 0.3, 0.4, np.nan, 0.5, 0.5]),
        }
        for split in ("random", "block"):
            self.arrays[f"bal_acc_test_{split}"] = self.arrays[f"acc_test_{split}"] - 0.05
        self.selections = []

    def sel(self, *, keys, squeeze_ones, regularization):
        assert squeeze_ones is False
        self.selections.append(regularization)
        return {key: self.arrays[key] for key in keys}


def test_summary_weights_mice_equally_and_omits_missing_splits_independently():
    viewer = SpeedEnvironmentDecoding(_Results(), regularization=1.0)
    random_two, block_two, random_three, block_three = viewer.summarize(viewer.state)
    assert random_two["mice"] == ["a", "b"]
    np.testing.assert_allclose(random_two["scores"], [0.7, 0.9])
    np.testing.assert_allclose(random_two["chance"], [0.6, 0.7])
    assert np.mean(random_two["scores"]) == pytest.approx(0.8)
    assert len(random_two["session_scores"]) == 3
    assert block_two["mice"] == ["a", "f"]
    np.testing.assert_allclose(block_two["scores"], [0.6, 0.4])
    # A 3-env session decoded using only 2 environments contributes to neither group.
    assert random_three["mice"] == block_three["mice"] == ["a", "c"]
    assert viewer.results.selections == [1.0]


def test_balanced_accuracy_uses_uniform_chance():
    results = _Results()
    results.arrays["chance_block"][:] = np.nan
    viewer = SpeedEnvironmentDecoding(results, metric="balanced_accuracy", splits=["block"])
    two, three = viewer.summarize(viewer.state)
    np.testing.assert_allclose(two["scores"], [0.55, 0.35])
    np.testing.assert_allclose(two["chance"], [0.5, 0.5])
    np.testing.assert_allclose(three["chance"], [1 / 3, 1 / 3])


@pytest.mark.parametrize("splits", [["random"], ["block"], ["random", "block"], []])
@pytest.mark.parametrize("counts", [[2], [3], [2, 3], []])
def test_standalone_plot_controls_and_empty_selection(splits, counts):
    viewer = SpeedEnvironmentDecoding(_Results(), splits=splits, environment_counts=counts, show_sessions=True)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    ax = fig.axes[0]
    assert [label.get_text() for label in ax.get_xticklabels()] == [f"{n} environments" for n in counts]
    assert ax.get_ylabel() == "Test accuracy"
    plt.close(fig)


def test_single_session_keeps_session_axis_and_missing_results_render():
    results = _Results()
    results.mouse_names = results.mouse_names[:1]
    results.arrays = {key: value[:1] for key, value in results.arrays.items()}
    viewer = SpeedEnvironmentDecoding(results)
    assert viewer.summarize(viewer.state)[0]["scores"].shape == (1,)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    plt.close(fig)
    for split in ("random", "block"):
        results.arrays[f"acc_test_{split}"][:] = np.nan
    viewer.refresh_data(viewer.state)
    fig = viewer.plot(viewer.state)
    assert "No valid decoding results" in [text.get_text() for text in fig.axes[0].texts]
    plt.close(fig)
