"""Numerical and shared-session layout checks for the three-row familiarity viewer."""

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import pyplot as plt

from dimensionality_manuscript.figure_scripts.figure3 import DimensionalityComponentFamiliarityViewer
from dimensionality_manuscript.figure_scripts.figure4._spectrum_math import _signed_participation_ratio


class _Results:
    param_axes = {
        "smooth_widths": [(5.0, None)],
        "activity_parameters_name": ["default"],
        "reliability_fraction_active_thresholds": [(None, None)],
    }
    unique_mice = ["a", "b"]
    mouse_names = np.array(["a", "b", "a", "b", "a", "b"])

    def __init__(self):
        self.sessions = [
            SimpleNamespace(date=date, has_spontaneous=lambda: False)
            for date in [3, 3, 1, 1, 2, 2]
        ]
        self.sf = np.tile([4.0, 2.0, -1.0], (6, 3, 1))
        self.ff = np.tile([8.0, 3.0, -0.5], (6, 3, 1))
        self.sf[:2, 1] = np.nan  # No second environment in the last session.
        self.calls = []

    def sel(self, *, keys, **kwargs):
        self.calls.append((keys, kwargs))
        arrays = {
            "sf_cv_env_full1": self.sf,
            "ff_env_full1": self.ff,
            "sf_cv_env_fullall": self.sf,
            "ff_env_full1_fullall": self.ff * 2,
            "ff": self.ff[:, 0] * 3,
        }
        return {key: arrays[key] for key in keys}


def test_raw_pr_and_component_use_environment_cross_spectra():
    results = _Results()
    viewer = DimensionalityComponentFamiliarityViewer(results)
    full, pf, component = viewer._curves
    np.testing.assert_allclose(full[0]["a"], _signed_participation_ratio(results.ff[[2, 4, 0], 0]))
    np.testing.assert_allclose(pf[0]["a"], _signed_participation_ratio(results.sf[[2, 4, 0], 0]))
    np.testing.assert_allclose(component[0]["a"], 5.0 / 10.5)
    assert len(component[1]["a"]) == 2
    keys, kwargs = results.calls[-1]
    assert keys == ["sf_cv_env_full1", "ff_env_full1"]
    assert kwargs["include_iti"] is False
    assert kwargs["smooth_widths"] == (5.0, None)
    assert kwargs["reliability_fraction_active_thresholds"] == (None, None)


def test_undefined_pr_preserves_chronology_across_rows():
    results = _Results()
    results.sf[4, 0] = 0  # Middle session's PR is undefined, component is zero.
    results.sf[0, 0] *= 2  # Last session must remain at x=2.
    viewer = DimensionalityComponentFamiliarityViewer(results)
    assert np.isnan(viewer._curves[1][0]["a"][1])
    np.testing.assert_allclose(viewer._curves[2][0]["a"], [5 / 10.5, 0, 10 / 10.5])
    assert len(viewer._curves[0][0]["a"]) == 3


def test_pr_clipping_and_smoothing_do_not_change_component():
    results = _Results()
    raw = DimensionalityComponentFamiliarityViewer(results)
    clipped = DimensionalityComponentFamiliarityViewer(results, clip_negative=True)
    smoothed = DimensionalityComponentFamiliarityViewer(
        results, pr_pre_smooth=False, pf_smooth_method="boxcar", full_smooth_method="boxcar"
    )
    np.testing.assert_allclose(clipped._curves[0][0]["a"], 121 / 73)
    assert not np.allclose(raw._curves[0][0]["a"], clipped._curves[0][0]["a"])
    assert not np.allclose(raw._curves[1][0]["a"], smoothed._curves[1][0]["a"])
    for viewer in [clipped, smoothed]:
        np.testing.assert_allclose(raw._curves[2][0]["a"], viewer._curves[2][0]["a"])


@pytest.mark.parametrize("full_within_env,denominator", [(False, 31.5), (True, 21.0)])
def test_outside_scope_reuses_selected_full_denominator(full_within_env, denominator):
    viewer = DimensionalityComponentFamiliarityViewer(
        _Results(), env_full_scope="outside_env", full_within_env=full_within_env
    )
    np.testing.assert_allclose(viewer._curves[2][0]["a"], 5 / denominator)


def test_spontaneous_scope_applies_same_session_mask_to_all_rows():
    results = _Results()
    for row in [0, 1]:
        results.sessions[row].has_spontaneous = lambda: True
    viewer = DimensionalityComponentFamiliarityViewer(results, env_full_scope="with_spontaneous")
    assert results.calls[-1][1]["include_iti"] is True
    for curves in viewer._curves:
        assert len(curves[0]["a"]) == 1
        assert len(curves[1]["a"]) == 0


def test_log_scale_applies_only_to_dimensionality_and_legend_can_be_hidden():
    viewer = DimensionalityComponentFamiliarityViewer(_Results(), log_y=True, env_legend={"visible": False})
    fig = viewer.plot(viewer.state)
    try:
        fig.canvas.draw()
        assert [ax.get_yscale() for ax in fig.axes] == ["log", "log", "linear"]
        assert fig.axes[-1].get_legend() is None
    finally:
        plt.close(fig)


@pytest.mark.parametrize("display", ["each", "errorPlot"])
def test_three_rows_share_x_and_only_bottom_shows_x_axis(display):
    viewer = DimensionalityComponentFamiliarityViewer(_Results(), display=display, as_percent=True)
    fig = viewer.plot(viewer.state)
    try:
        fig.canvas.draw()
        top, middle, bottom = fig.axes
        assert top.get_shared_x_axes().joined(top, bottom)
        assert middle.get_shared_x_axes().joined(middle, bottom)
        assert top.get_ylabel() == "Full CA1"
        assert middle.get_ylabel() == "PF Fraction"
        assert bottom.get_ylabel() == "Placefield\nComponent (%)"
        assert bottom.get_ylim() == (0, 100)
        np.testing.assert_allclose(bottom.get_yticks(), [0, 50, 100])
        assert bottom.get_xlabel() == "Env Session #"
        for ax in [top, middle]:
            assert not ax.spines["bottom"].get_visible()
            assert not ax.get_xticklabels()
            assert not ax.texts
        assert top.get_legend() is None
        assert bottom.get_legend() is None
        assert [text.get_text() for text in middle.get_legend().texts] == ["Env #1", "Env #2", "Env #3"]
        assert bottom.get_xlim() == (0, 2)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("sharey", [False, True])
@pytest.mark.parametrize("log_y", [False, True])
def test_dimensionality_y_sharing_and_decade_ticks(sharey, log_y):
    viewer = DimensionalityComponentFamiliarityViewer(_Results(), sharey=sharey, log_y=log_y)
    # Distinct panel ranges make accidental sharing or clipping visible.
    for per_mouse in viewer._curves[0].values():
        for mouse in per_mouse:
            per_mouse[mouse] *= 100
    fig = viewer.plot(viewer.state)
    try:
        fig.canvas.draw()
        top, middle, bottom = fig.axes
        assert top.get_shared_y_axes().joined(top, middle) == sharey
        assert not top.get_shared_y_axes().joined(top, bottom)
        if sharey:
            assert top.get_ylim() == middle.get_ylim()
            assert top.get_ylim()[0] < 1.2
            assert top.get_ylim()[1] > 150
        if log_y:
            for ax in (top, middle):
                ticks = ax.get_yticks()
                np.testing.assert_allclose(np.log10(ticks), np.round(np.log10(ticks)))
                minor_ticks = ax.get_yticks(minor=True)
                assert len(minor_ticks) > 0
                mantissas = minor_ticks / 10.0 ** np.floor(np.log10(minor_ticks))
                np.testing.assert_allclose(mantissas, np.round(mantissas))
                assert np.all((mantissas >= 2) & (mantissas <= 9))
                assert all(label.get_text() == "" for label in ax.get_yticklabels(minor=True))
                np.testing.assert_allclose(np.log10(ax.get_ylim()), np.round(np.log10(ax.get_ylim())))
        np.testing.assert_allclose(bottom.get_yticks(), [0, 0.5, 1])
    finally:
        plt.close(fig)


def test_log_y_is_default():
    viewer = DimensionalityComponentFamiliarityViewer(_Results())
    assert viewer.state["log_y"] is True
    assert viewer.state["sharey"] is False
    assert viewer.state["legend_panel"] == 1


@pytest.mark.parametrize("panel", [0, 1, 2])
def test_environment_legend_uses_selected_panel(panel):
    viewer = DimensionalityComponentFamiliarityViewer(_Results(), legend_panel=panel)
    fig = viewer.plot(viewer.state)
    try:
        assert [ax.get_legend() is not None for ax in fig.axes] == [row == panel for row in range(3)]
    finally:
        plt.close(fig)
