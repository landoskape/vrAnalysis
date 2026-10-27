"""Standalone component modes retain the composite's data and compact layout."""

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import pyplot as plt

from dimensionality_manuscript.figure_scripts.figure3 import PlacefieldComponentFamiliarityViewer


class _Results:
    param_axes = {"smooth_widths": [(5.0, None)]}
    unique_mice = ["a", "b"]
    mouse_names = np.array(["a", "b", "a", "b"])
    sessions = [SimpleNamespace(date=date) for date in [2, 2, 1, 1]]

    def sel(self, *, keys, **kwargs):
        assert kwargs["smooth_widths"] == (5.0, None)
        sf = np.tile([4.0, 1.0, 1.0], (4, 3, 1))
        sf[:2] *= 0.5
        arrays = {"sf_cv_env_full1": sf, "ff_env_full1": np.tile([8.0, 4.0, 4.0], (4, 3, 1))}
        return {key: arrays[key] for key in keys}


@pytest.mark.parametrize("component,labels", [("head", []), ("full", []), ("both", ["First 1", "Full"])])
def test_modes_data_labels_and_shared_session_axis(component, labels):
    viewer = PlacefieldComponentFamiliarityViewer(_Results(), component=component, first10=1)
    if component != "full":
        np.testing.assert_allclose(viewer._curves["head"]["Env #1"]["svr"]["a"], [0.5, 0.25])
    if component != "head":
        np.testing.assert_allclose(viewer._curves["full"]["Env #1"]["svr"]["a"], [0.375, 0.1875])
    fig = viewer.plot(viewer.state)
    try:
        fig.canvas.draw()
        assert len(fig.axes) == 1
        assert [text.get_text() for ax in fig.axes for text in ax.texts] == labels
        assert all(not ax.spines["left"].get_visible() and not ax.get_yticklabels() for ax in fig.axes)
        assert all(ax.get_legend() is None for ax in fig.axes)
        assert fig.axes[-1].get_xlabel() == "Env Session #"
        if component == "both":
            ax = fig.axes[0]
            assert len(ax.lines) == 6
            np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.5, 0.25])
            np.testing.assert_allclose(ax.lines[3].get_ydata(), [0.375, 0.1875])
    finally:
        plt.close(fig)


def test_percent_yaxis_and_legend():
    viewer = PlacefieldComponentFamiliarityViewer(
        _Results(), component="full", show_yaxis=True, show_legend=True, as_percent=True,
        shared_variance_ratio_ylims=(0.1, 0.8),
    )
    fig = viewer.plot(viewer.state)
    try:
        ax = fig.axes[0]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [37.5, 18.75])
        assert ax.get_ylim() == (10.0, 80.0)
        assert ax.get_ylabel() == "Placefield Component (%)"
        assert [text.get_text() for text in ax.get_legend().texts] == ["Env #1", "Env #2", "Env #3"]
    finally:
        plt.close(fig)


def test_switching_component_rebuilds_selected_curves():
    viewer = PlacefieldComponentFamiliarityViewer(_Results(), component="full")
    viewer.set_parameter_value("component", "head")
    assert set(viewer._curves) == {"head"}
    viewer.set_parameter_value("component", "both")
    assert set(viewer._curves) == {"head", "full"}


def test_both_label_positions_and_unique_legend():
    viewer = PlacefieldComponentFamiliarityViewer(
        _Results(), head_text_xy=(0.2, 0.8), full_text_xy=(0.3, 0.4), show_legend=True,
    )
    fig = viewer.plot(viewer.state)
    try:
        ax = fig.axes[0]
        assert [text.get_position() for text in ax.texts] == [(0.2, 0.8), (0.3, 0.4)]
        assert all(text.get_transform() is ax.transAxes for text in ax.texts)
        assert [text.get_text() for text in ax.get_legend().texts] == ["Env #1", "Env #2", "Env #3"]
    finally:
        plt.close(fig)
