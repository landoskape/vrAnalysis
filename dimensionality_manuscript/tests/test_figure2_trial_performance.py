import inspect
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure2.trial_performance import (
    TRIAL_PERFORMANCE_MODEL_NAMES,
    TrialModelPerformanceViewer,
    trial_performance_scores,
)


class _FakeTrialResults:
    model_names = ("external_placefield_1d", *TRIAL_PERFORMANCE_MODEL_NAMES)
    param_axes = {"model_name": list(model_names)}
    config_class = SimpleNamespace(
        spks_type="sigrebase",
        activity_parameters_name="std",
        reliability_fraction_active_threshold=(0.3, 0.1),
        trial_split_seed=0,
    )
    mouse_names = np.array(["m1", "m1", "m2"])

    def __init__(self):
        self.calls = []

    def sel(self, *, model_name, avg_by_mouse, **selection):
        self.calls.append((model_name, avg_by_mouse, selection))
        model_index = _FakeTrialResults.model_names.index(model_name) - 1
        base = np.array([[0.10, 0.20, np.nan], [0.30, 0.50, 0.70], [0.80, np.nan, 1.00]])
        return {
            "r2": base + 0.01 * model_index,
            "mse": 1.0 - base - 0.01 * model_index,
        }


def test_trial_performance_uses_temporal_trial_model_comparison():
    results = _FakeTrialResults()
    viewer = TrialModelPerformanceViewer(results)

    assert tuple(viewer.model_names) == TRIAL_PERFORMANCE_MODEL_NAMES
    assert "external_placefield_1d_gain" not in viewer.model_names
    assert viewer.model_labels == [
        "Placefield",
        "Global\nGain",
        "Trial\nGain",
        "Peer\nPrediction",
    ]
    assert [call[0] for call in results.calls] == list(TRIAL_PERFORMANCE_MODEL_NAMES)
    assert viewer.state["metric"] == "r2"
    assert viewer.state["environment"] == "Average"
    assert viewer.state["aggregation"] == "mean"
    assert viewer.parameters["metric"].options == ["r2", "mse"]
    assert viewer.parameters["aggregation"].options == ["mean", "median"]
    # Environment mean is within session first: m1 = mean([.15, .50]), not a mean over all
    # five finite environment/session cells for m1.
    np.testing.assert_allclose(viewer._scores[0], [0.325, 0.9])

    slot_state = dict(viewer.state)
    slot_state["environment"] = "Slot 2"
    viewer.refresh_data(slot_state)
    np.testing.assert_allclose(viewer._scores[0], [0.35, np.nan], equal_nan=True)

    fig = viewer.plot(viewer.state)
    assert [label.get_text() for label in fig.axes[0].get_xticklabels()] == viewer.model_labels
    assert len(fig.axes[0].child_axes) == 1  # absolute-score inset
    plt.close(fig)


def test_trial_performance_median_reduces_environments_then_sessions_per_mouse():
    results = _FakeTrialResults()
    results.mouse_names = np.array(["m1", "m1", "m1", "m2"])

    def sel(*, model_name, avg_by_mouse, **selection):
        model_index = _FakeTrialResults.model_names.index(model_name) - 1
        base = np.array(
            [
                [0.10, 0.20, np.nan],
                [0.30, 0.50, 0.70],
                [-9.00, 0.80, 1.00],
                [0.80, np.nan, 1.00],
            ]
        )
        return {"r2": base + 0.01 * model_index}

    results.sel = sel
    scores = trial_performance_scores(
        results,
        TRIAL_PERFORMANCE_MODEL_NAMES,
        "r2",
        {},
        "Average",
        True,
        "median",
    )

    # Session medians for m1 are .15, .50, and .80; their mouse median is .50.
    # m2 has a single session median of .90.
    np.testing.assert_allclose(scores[0], [0.50, 0.90])

    viewer = TrialModelPerformanceViewer(results, aggregation="median")
    fig = viewer.plot(viewer.state)
    # The thick center trace is also the across-mouse median, not a mean of median-reduced mice.
    center_line = next(line for line in fig.axes[0].lines if line.get_linewidth() == viewer.state["mean_linewidth"])
    np.testing.assert_allclose(center_line.get_ydata(), np.nanmedian(viewer._scores - viewer._scores[0], axis=1))
    plt.close(fig)


def test_trial_performance_rejects_unknown_aggregation():
    with pytest.raises(ValueError, match="aggregation"):
        TrialModelPerformanceViewer(_FakeTrialResults(), aggregation="trimmed")


def test_trial_performance_rejects_all_cell_results():
    results = _FakeTrialResults()
    results.config_class = SimpleNamespace(display_name="trial_split_regression")
    with pytest.raises(ValueError, match="TrialPlacecellRegressionConfig"):
        TrialModelPerformanceViewer(results)


@pytest.mark.parametrize("include_external", [False, True])
@pytest.mark.parametrize("metric", ["r2", "mse"])
def test_trial_performance_placefield_baseline(include_external, metric):
    results = _FakeTrialResults()
    viewer = TrialModelPerformanceViewer(
        results, include_external_placefield=include_external, metric=metric,
    )
    expected_names = list(TRIAL_PERFORMANCE_MODEL_NAMES)
    expected_labels = ["Placefield", "Global\nGain", "Trial\nGain", "Peer\nPrediction"]
    expected_colors = ["#000000", "#c00000", "#008000", "#0000cd"]
    if include_external:
        expected_names.insert(0, "external_placefield_1d")
        expected_labels[0] = "Internal\nPlacefield"
        expected_labels.insert(0, "Placefield")
        expected_colors.insert(0, "#666666")
    assert viewer.model_names == expected_names
    assert viewer.model_labels == expected_labels
    assert viewer.model_colors == expected_colors
    assert [call[0] for call in results.calls] == expected_names
    assert "reference_model" not in viewer.parameters
    assert "external_placefield_1d_gain" not in viewer.model_names

    fig = viewer.plot(viewer.state)
    ax = fig.axes[0]
    center = next(line for line in ax.lines if line.get_linewidth() == viewer.state["mean_linewidth"])
    expected_scores = np.mean(viewer._scores - viewer._scores[0], axis=1)
    np.testing.assert_allclose(center.get_ydata(), expected_scores)
    inset = ax.child_axes[0]
    inset_center = next(line for line in inset.lines if line.get_linewidth() == viewer.state["mean_linewidth"])
    np.testing.assert_allclose(inset_center.get_ydata(), np.mean(viewer._scores, axis=1))
    plt.close(fig)


def test_trial_performance_exposes_parent_parameters():
    from dimensionality_manuscript.figure_scripts.figure2.performance import ModelPerformanceViewer

    parent = inspect.signature(ModelPerformanceViewer)
    trial = inspect.signature(TrialModelPerformanceViewer)
    for name, parameter in parent.parameters.items():
        assert trial.parameters[name] == parameter
    assert "kwargs" not in trial.parameters
    assert "reference_model" not in trial.parameters
