"""Check trial-set routing and physical reward alignment in the figure-1 layouts."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure1 import schematic as schematic_module
from dimensionality_manuscript.figure_scripts.figure1.schematic import VREnvironmentSchematic, vr_schematic_and_speed
from dimensionality_manuscript.figure_scripts.figure1.speed import MouseSpeedFocus


def make_speed(**kwargs):
    bins = 20
    x = np.linspace(0.025, 0.975, bins)
    all_curves = np.stack([20 + 5 * np.sin(x * 6 + i) for i in range(3)])[None]
    data = dict(environments=np.array([[1, 3, 4]]), dist_fraction_centers=x[None],
                speed_curve_all=all_curves, speed_curve_first=all_curves * 0.5)
    results = SimpleNamespace(arrays=data, param_axes={}, config_class=SimpleNamespace(num_bins=bins),
                              mouse_names=["mouse"], sel=lambda **kwargs: data)
    return MouseSpeedFocus(results, show_significance=False, **kwargs)


def make_schematic(monkeypatch):
    monkeypatch.setattr(schematic_module, "load_vr_room_images",
                        lambda *args: [np.ones((20, 30, 3)) * 0.7] * 4)
    return VREnvironmentSchematic()


@pytest.mark.parametrize("split", [False, True])
def test_standalone_trial_sets_and_reward_labels(split):
    viewer = make_speed(split_speed=split)
    fig = viewer.plot(viewer.state)
    try:
        fig.canvas.draw()
        assert len(fig.axes) == (2 if split else 1)
        top = fig.axes[0]
        if split:
            bottom = fig.axes[1]
            assert len(bottom.collections) == len(top.collections) == 3
            assert top.get_shared_x_axes().joined(top, bottom)
            assert all(axis.get_legend() is None for axis in fig.axes)
            assert "All Trials" in [text.get_text() for text in top.texts]
            assert "1st in Block" in [text.get_text() for text in bottom.texts]
            np.testing.assert_allclose(bottom.lines[0].get_ydata(), top.lines[0].get_ydata() * 0.5)
            assert "reward location" in [text.get_text() for text in bottom.texts]
            assert not any(label.get_visible() for label in top.get_xticklabels())
        else:
            assert [text.get_text() for text in top.get_legend().texts] == ["All Trials", "1st in Block", "reward location"]
        for axis in fig.axes:
            arrows = [text for text in axis.texts if isinstance(text, Annotation) and text.arrow_patch is not None]
            assert [arrow.xy[0] for arrow in arrows] == [150, 100, 50]
            assert all(arrow.xy[1] == 0.14 for arrow in arrows)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("split", [False, True])
def test_combined_alignment_and_no_duplicate_reward_legend(monkeypatch, split):
    schematic = make_schematic(monkeypatch)
    speed = make_speed()
    fig = vr_schematic_and_speed(schematic, speed, split_speed=split)
    try:
        fig.canvas.draw()
        assert len(fig.axes) == (3 if split else 2)
        schem_ax, speed_ax = fig.axes[:2]
        labels = [text for text in schem_ax.texts if text.get_text().startswith("Env ")]
        assert [text.get_text() for text in labels] == ["Env A", "Env B", "Env C"]
        assert all(text.get_ha() == "right" and text.get_rotation() == 0 for text in labels)
        track_w = schematic.layout_metrics(schematic.state)["track_w"]
        for axis in fig.axes[1:]:
            for cm in (0, 50, 100, 150, 200):
                assert axis.transData.transform((cm, 0))[0] == pytest.approx(
                    schem_ax.transData.transform((track_w * cm / 200, 0))[0])
            if axis.get_legend() is not None:
                assert [text.get_text() for text in axis.get_legend().texts] == ["All Trials", "1st in Block"]
        if split:
            assert speed_ax.get_shared_x_axes().joined(speed_ax, fig.axes[2])
    finally:
        plt.close(fig)


def test_split_without_first_draws_only_all_trials():
    viewer = make_speed(split_speed=True, show_first=False)
    fig = viewer.plot(viewer.state)
    try:
        assert len(fig.axes) == 1
        assert len(fig.axes[0].lines) == 3
    finally:
        plt.close(fig)
