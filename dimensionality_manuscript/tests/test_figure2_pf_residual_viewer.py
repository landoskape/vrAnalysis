from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from dimensionality_manuscript.figure_scripts.figure2.pf_residual import (
    ModelPlacefieldResidualViewer,
    ModelPlacefieldResidualFamiliarityViewer,
)


class _FakeResidualResults:
    param_axes = {
        "model_name": [
            "internal_placefield_1d",
            "internal_placefield_1d_gain",
            "rrr",
        ]
    }
    unique_mice = np.array(["m1", "m2"])
    mouse_names = np.array(["m1", "m1", "m2", "m2"])
    sessions = [
        SimpleNamespace(date="2024-01-01", environments=np.array([0])),
        SimpleNamespace(date="2024-01-02", environments=np.array([0])),
        SimpleNamespace(date="2024-01-01", environments=np.array([0])),
        SimpleNamespace(date="2024-01-02", environments=np.array([0])),
    ]

    def sel(self, *, model_name, keys, avg_by_mouse, **selection):
        assert avg_by_mouse is False
        model_offset = self.param_axes["model_name"].index(model_name) * 0.05
        key = keys[0]
        metric_offset = 0.2 if key.endswith("outside_pf_rms") else 0.0
        return {key: np.array([0.4, 0.5, 0.45, 0.55]) + model_offset + metric_offset}


class _FakeResidualSummaryResults:
    param_axes = _FakeResidualResults.param_axes

    def __init__(self):
        self.requested_keys = []

    def sel(self, *, model_name, keys, avg_by_mouse, **selection):
        assert avg_by_mouse is True
        key = keys[0]
        self.requested_keys.append(key)
        if "notquality_filtered" in key:
            subset_offset = 1.0
        elif "quality_filtered" in key:
            subset_offset = 0.5
        else:
            subset_offset = 0.0
        model_offset = self.param_axes["model_name"].index(model_name) * 0.05
        return {key: np.array([0.4, 0.5]) + subset_offset + model_offset}


def test_pf_residual_summary_selects_main_and_inset_roi_subsets():
    results = _FakeResidualSummaryResults()
    viewer = ModelPlacefieldResidualViewer(
        results,
        main_show="not quality",
        inset_show="all",
    )

    assert viewer.state["main_show"] == "not quality"
    assert viewer.state["inset_show"] == "all"
    assert "mean_notquality_filtered_within_pf_rms" in results.requested_keys
    assert "mean_within_pf_rms" in results.requested_keys
    np.testing.assert_allclose(viewer._scores["within_pf_rms"][:, 0], [1.4, 1.45, 1.5])
    np.testing.assert_allclose(viewer._inset_scores["within_pf_rms"][:, 0], [0.4, 0.45, 0.5])


def test_pf_residual_summary_notquality_normalized_key_order():
    results = _FakeResidualSummaryResults()
    ModelPlacefieldResidualViewer(
        results,
        normalized=True,
        main_show="not quality",
        include_inset=False,
    )

    assert "mean_notquality_filtered_normalized_within_pf_rms" in results.requested_keys


def test_pf_residual_familiarity_within_only_uses_one_axis_and_compact_labels():
    viewer = ModelPlacefieldResidualFamiliarityViewer(
        _FakeResidualResults(),
        within_only=True,
        min_mice=1,
    )

    fig = viewer.plot(viewer.state)

    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylabel() == "Within-PF residual RMS"
    assert [text.get_text() for text in fig.axes[0].get_legend().get_texts()] == [
        "PF",
        "PF+Gain",
        "Peer Prediction",
    ]
    plt.close(fig)


def test_pf_residual_familiarity_two_axis_layout_has_one_legend():
    viewer = ModelPlacefieldResidualFamiliarityViewer(
        _FakeResidualResults(),
        within_only=False,
        min_mice=1,
    )

    fig = viewer.plot(viewer.state)

    assert len(fig.axes) == 2
    assert fig.axes[0].get_legend() is not None
    assert fig.axes[1].get_legend() is None
    plt.close(fig)
