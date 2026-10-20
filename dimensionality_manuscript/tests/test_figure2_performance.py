import matplotlib.pyplot as plt
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure2.performance import (
    PERFORMANCE_MODEL_NAMES,
    ModelPerformanceViewer,
)


class _FakeResults:
    model_names = [
        "external_placefield_1d",
        *PERFORMANCE_MODEL_NAMES,
        "internal_placefield_1d_structured_additive",
    ]
    param_axes = {"model_name": model_names}

    def sel(self, *, model_name, avg_by_mouse, **selection):
        base = np.array([0.1, 0.2]) + 0.03 * self.model_names.index(model_name)
        return {"r2": base, "mse": 1.0 - base}


@pytest.mark.parametrize("include_external", [False, True])
@pytest.mark.parametrize("include_structured", [False, True])
@pytest.mark.parametrize("metric", ["r2", "mse"])
def test_performance_placefield_reference(include_external, include_structured, metric):
    viewer = ModelPerformanceViewer(
        _FakeResults(),
        metric=metric,
        include_external_placefield=include_external,
        include_structured_additive=include_structured,
    )
    expected_names = list(PERFORMANCE_MODEL_NAMES)
    expected_labels = ["Placefield", "Global\nGain", "Peer\nPrediction"]
    if include_external:
        expected_names.insert(0, "external_placefield_1d")
        expected_labels[0] = "Internal\nPlacefield"
        expected_labels.insert(0, "Placefield")
    if include_structured:
        expected_names.insert(-1, "internal_placefield_1d_structured_additive")
        expected_labels.insert(-1, "Shared\nResidual")
    assert viewer.model_names == expected_names
    assert viewer.model_labels == expected_labels
    assert "reference_model" not in viewer.parameters

    fig = viewer.plot(viewer.state)
    ax = fig.axes[0]
    assert [label.get_text() for label in ax.get_xticklabels()] == expected_labels
    center = next(line for line in ax.lines if line.get_linewidth() == viewer.state["mean_linewidth"])
    np.testing.assert_allclose(center.get_ydata(), np.mean(viewer._scores - viewer._scores[0], axis=1))
    inset = ax.child_axes[0]
    inset_center = next(line for line in inset.lines if line.get_linewidth() == viewer.state["mean_linewidth"])
    np.testing.assert_allclose(inset_center.get_ydata(), np.mean(viewer._scores, axis=1))
    plt.close(fig)
