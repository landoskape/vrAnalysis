from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from dimensionality_manuscript.figure_scripts.figure2.trial_performance import (
    TRIAL_PERFORMANCE_MODEL_NAMES,
    TrialModelPerformanceViewer,
)


class _FakeTrialResults:
    param_axes = {"model_name": list(TRIAL_PERFORMANCE_MODEL_NAMES)}
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
        model_index = TRIAL_PERFORMANCE_MODEL_NAMES.index(model_name)
        base = np.array([[0.10, 0.20, np.nan], [0.30, 0.50, 0.70], [0.80, np.nan, 1.00]])
        return {
            "r2": base + 0.01 * model_index,
            "mse": 1.0 - base - 0.01 * model_index,
        }


def test_trial_performance_uses_temporal_trial_model_comparison():
    results = _FakeTrialResults()
    viewer = TrialModelPerformanceViewer(results)

    assert tuple(viewer.model_names) == TRIAL_PERFORMANCE_MODEL_NAMES
    assert viewer.model_labels == [
        "External\nPF",
        "External\nGlobal Gain",
        "Internal\nPF",
        "Internal\nGlobal Gain",
        "Trial\nGain",
        "Peer\nPrediction",
    ]
    assert [call[0] for call in results.calls] == list(TRIAL_PERFORMANCE_MODEL_NAMES)
    assert viewer.state["metric"] == "r2"
    assert viewer.state["environment"] == "Average"
    assert viewer.parameters["metric"].options == ["r2", "mse"]
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
