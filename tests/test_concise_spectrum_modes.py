"""Focused tests for ConciseSpectrumViewer spectrum and panel modes."""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from dimensionality_manuscript.figure_scripts.figure3 import concise_spectrum
from dimensionality_manuscript.figure_scripts.figure3._familiarity import render_familiarity_ratio_panel
from dimensionality_manuscript.figure_scripts.figure3._ratios import (
    plot_ratios_beeswarms_concise,
    plot_ratios_spectrum,
    ratios_arrays,
    spectrum_display_arrays,
)
from dimensionality_manuscript.figure_scripts.figure3.concise_spectrum import ConciseSpectrumViewer


class _Session:
    def __init__(self, spontaneous=False):
        self._spontaneous = spontaneous

    def has_spontaneous(self):
        return self._spontaneous


class _FakeAggregator:
    def __init__(self, arrays, mouse_names, spontaneous=None):
        self.arrays = arrays
        self.mouse_names = np.asarray(mouse_names)
        spontaneous = spontaneous or [False] * len(mouse_names)
        self.sessions = [_Session(value) for value in spontaneous]
        self.calls = []

    def sel(self, *, keys, **kwargs):
        self.calls.append((tuple(keys), kwargs))
        return {key: self.arrays[key] for key in keys}


def _mode_arrays():
    return {
        "sf_cv": np.array([[3.0, 1.0, 0.5], [2.0, 1.0, 0.5], [4.0, 2.0, 1.0]]),
        "ff": np.array([[6.0, 3.0, 1.0], [4.0, 2.0, 1.0], [8.0, 4.0, 2.0]]),
        "sf_cv_env_full1": np.array(
            [
                [[8.0, 4.0, 2.0], [4.0, 2.0, np.nan]],
                [[2.0, 1.0, 0.5], [np.nan, np.nan, np.nan]],
                [[10.0, 5.0, 2.0], [6.0, 3.0, 2.0]],
            ]
        ),
        "ff_env_full1": np.array(
            [
                [[12.0, 6.0, 3.0], [8.0, 4.0, 1.0]],
                [[6.0, 3.0, 1.0], [np.nan, np.nan, np.nan]],
                [[14.0, 7.0, 3.0], [10.0, 5.0, 1.0]],
            ]
        ),
    }


def test_avg_env_spectrum_averages_env_then_sessions_within_mouse():
    results = _FakeAggregator(_mode_arrays(), ["mouse-a", "mouse-a", "mouse-b"])

    arrays = ratios_arrays(results, {}, spectrum_mode="avg_env", env_full_scope="within_env", first_n=2)
    display = spectrum_display_arrays(
        arrays,
        "none",
        0.0,
        "none",
        0.0,
        clip_negative=False,
        normalize=False,
    )

    np.testing.assert_allclose(display["sf_cv"], [[4.0, 2.0, 1.25], [8.0, 4.0, 2.0]])
    np.testing.assert_allclose(display["ff"], [[8.0, 4.0, 1.5], [12.0, 6.0, 2.0]])
    np.testing.assert_allclose(
        arrays["sf_cv_total"],
        [((11.0 / 17.0) + (3.5 / 10.0)) / 2.0, 14.0 / 20.0],
    )
    np.testing.assert_allclose(
        arrays["sf_cv_total_10"],
        [((9.0 / 15.0) + (3.0 / 9.0)) / 2.0, 12.0 / 18.0],
    )
    env_call = next(call for call in results.calls if call[0] == ("sf_cv_env_full1", "ff_env_full1"))
    assert env_call[1]["include_iti"] is False


def test_avg_env_spectrum_reuses_by_env_scope_keys_and_session_mask():
    arrays = _mode_arrays()
    arrays["sf_cv_env_fullall"] = arrays["sf_cv_env_full1"]
    arrays["ff_env_full1_fullall"] = arrays["ff_env_full1"]
    results = _FakeAggregator(
        arrays,
        ["mouse-a", "mouse-a", "mouse-b"],
        spontaneous=[False, False, True],
    )

    selected = ratios_arrays(
        results,
        {},
        spectrum_mode="avg_env",
        env_full_scope="with_spontaneous",
        full_within_env=True,
    )

    np.testing.assert_allclose(selected["_sf_cv_sessions_raw"], [[8.0, 4.0, 2.0]])
    np.testing.assert_allclose(selected["_ff_sessions_raw"], [[12.0, 6.0, 2.0]])
    np.testing.assert_array_equal(selected["_mouse_names"], ["mouse-b"])
    np.testing.assert_allclose(selected["sf_cv_total"], [14.0 / 20.0])
    env_call = next(
        call
        for call in results.calls
        if call[0] == ("sf_cv_env_fullall", "ff_env_full1_fullall")
    )
    assert env_call[1]["include_iti"] is True


def test_avg_env_beeswarm_uses_selected_full_scope_denominator():
    arrays = _mode_arrays()
    arrays["sf_cv_env_fullall"] = arrays["sf_cv_env_full1"]
    arrays["ff_env_full1_fullall"] = 2.0 * arrays["ff_env_full1"]
    results = _FakeAggregator(arrays, ["mouse-a", "mouse-a", "mouse-b"])

    whole_session_full = ratios_arrays(
        results,
        {},
        spectrum_mode="avg_env",
        env_full_scope="outside_env",
        full_within_env=False,
    )
    within_env_full = ratios_arrays(
        results,
        {},
        spectrum_mode="avg_env",
        env_full_scope="outside_env",
        full_within_env=True,
    )

    np.testing.assert_allclose(whole_session_full["sf_cv_total"], [0.8, 1.0])
    np.testing.assert_allclose(
        within_env_full["sf_cv_total"],
        [((11.0 / 34.0) + (3.5 / 20.0)) / 2.0, 14.0 / 40.0],
    )


def test_concise_viewer_exposes_and_decodes_smooth_widths(monkeypatch):
    class _SmoothWidthResults:
        param_axes = {"smooth_widths": [(5.0, None), (None, None)]}

    def fake_refresh_data(self, state):
        pass

    monkeypatch.setattr(ConciseSpectrumViewer, "refresh_data", fake_refresh_data)
    viewer = ConciseSpectrumViewer(_SmoothWidthResults(), show_slope_panel=False)

    assert "smooth_widths" in viewer.selection_names
    assert viewer.state["smooth_widths"] == "5.0-None"
    assert viewer.state["first10"] == 10
    assert viewer._sel_params(viewer.state)["smooth_widths"] == (5.0, None)


def _plot_state(
    include_first10_beeswarm,
    include_first10_by_env,
    include_beeswarm,
    by_env_layout="separate",
):
    return {
        "first10": 7,
        "include_first10_beeswarm": include_first10_beeswarm,
        "include_first10_by_env": include_first10_by_env,
        "include_beeswarm": include_beeswarm,
        "by_env_layout": by_env_layout,
        "fontsize": 8.0,
        "ax1_width_ratio": 0.8,
        "ax23_width_ratio": 1.0,
        "pf_smooth_method": "none",
        "pf_smooth_width": 0.0,
        "full_smooth_method": "none",
        "full_smooth_width": 0.0,
        "clip_negative": False,
        "as_percent": False,
        "normalize": False,
        "ylims": (-5.0, -1.0),
        "shared_variance_ratio_ylims": (0.1, 0.9),
        "annotation_mode": "X",
        "annotation_x": 0.7,
        "annotation_y": 0.9,
        "annotation_yoffset": -0.1,
        "annotation_line_yoffset": 0.0,
        "annotation_line_xpad": 0.2,
        "style_by_env": "all",
        "env_legend_fontsize_scale": 1.0,
        "first10_text_x": 0.1,
        "first10_text_y": 0.9,
        "first10_text_ha": "left",
        "first10_text_va": "top",
        "all_text_x": 0.1,
        "all_text_y": 0.9,
        "all_text_ha": "left",
        "all_text_va": "top",
    }


@pytest.mark.parametrize(
    (
        "include_first10_beeswarm",
        "include_first10_by_env",
        "include_beeswarm",
        "by_env_layout",
        "as_percent",
        "expected_axes",
        "expected_legend",
    ),
    [
        (True, True, True, "separate", False, 4, ["1st", "All"]),
        (True, True, True, "shared", True, 3, ["1st", "All"]),
        (False, True, True, "separate", False, 4, ["1st", "All"]),
        (True, False, True, "shared", False, 3, ["1st", "All"]),
        (False, False, True, "separate", False, 3, ["1st", "All"]),
        (True, True, False, "shared", False, 2, ["1st"]),
        (False, False, False, "shared", False, 2, ["1st"]),
    ],
)
def test_panel_options_are_independent(
    monkeypatch,
    include_first10_beeswarm,
    include_first10_by_env,
    include_beeswarm,
    by_env_layout,
    as_percent,
    expected_axes,
    expected_legend,
):
    viewer = object.__new__(ConciseSpectrumViewer)
    viewer._previous_figure = None
    viewer.figsize = (5.0, 2.0)
    viewer.show_slope_panel = False
    viewer._ratios_arrays = {}
    viewer._curves_all = {}
    viewer._curves_first10 = {}
    env_legend_labels = []
    first10_indicator = []
    familiarity_call_axes = []
    ratio_value_scales = []
    ratio_leading_options = []
    familiarity_value_scales = []

    monkeypatch.setattr(concise_spectrum, "spectrum_display_arrays", lambda *_args, **_kwargs: {})

    def fake_spectrum(*_args, **kwargs):
        first10_indicator.append(kwargs["show_first10_indicator"])

    monkeypatch.setattr(concise_spectrum, "plot_ratios_spectrum", fake_spectrum)
    monkeypatch.setattr(concise_spectrum, "draw_shared_reliable_annotation", lambda *_args, **_kwargs: None)

    def fake_ratio(axis, *_args, **_kwargs):
        ratio_value_scales.append(_kwargs["value_scale"])
        ratio_leading_options.append((_kwargs["include_first10"], _kwargs["leading_dims"]))
        axis.plot([0, 1], [0.2, 0.2], label="All")

    def fake_familiarity(axis, *_args, **_kwargs):
        familiarity_call_axes.append(axis)
        familiarity_value_scales.append(_kwargs["value_scale"])
        axis.plot([0, 1], [0.2, 0.3], label="Env #1")
        axis.legend()

    def fake_apply(axis, _state, _fontsize, *, prefix, handles=None, labels=None, **_kwargs):
        if prefix == "env_legend":
            env_legend_labels.extend(labels)
            axis.legend(handles, labels)

    monkeypatch.setattr(concise_spectrum, "plot_ratios_beeswarms_concise", fake_ratio)
    monkeypatch.setattr(concise_spectrum, "render_familiarity_ratio_panel", fake_familiarity)
    monkeypatch.setattr(concise_spectrum, "apply_legend", fake_apply)

    state = _plot_state(
        include_first10_beeswarm,
        include_first10_by_env,
        include_beeswarm,
        by_env_layout,
    )
    state["as_percent"] = as_percent
    fig = viewer.plot(state)

    assert len(fig.axes) == expected_axes
    assert env_legend_labels == expected_legend
    expected_indicator = (include_beeswarm and include_first10_beeswarm) or include_first10_by_env
    assert first10_indicator == [expected_indicator]
    leading_ratio_axis = fig.axes[1]
    expected_scale = 100.0 if as_percent else 1.0
    expected_ylabel = "Placefield Component (%)" if as_percent else "Placefield Component"
    assert leading_ratio_axis.get_ylabel() == expected_ylabel
    assert leading_ratio_axis.spines["left"].get_visible()
    assert leading_ratio_axis.get_ylim() == pytest.approx((0.1 * expected_scale, 0.9 * expected_scale))
    assert leading_ratio_axis.spines["left"].get_bounds() == pytest.approx((0.1 * expected_scale, 0.9 * expected_scale))
    assert ratio_value_scales == ([expected_scale] if include_beeswarm else [])
    assert ratio_leading_options == ([(include_first10_beeswarm, 7)] if include_beeswarm else [])
    assert familiarity_value_scales == [expected_scale] * (2 if include_first10_by_env else 1)
    familiarity_axes = fig.axes[2:] if include_beeswarm else fig.axes[1:]
    assert sum(len(axis.texts) for axis in familiarity_axes) == (2 if include_first10_by_env else 0)
    assert len(familiarity_axes) == (1 if by_env_layout == "shared" or not include_first10_by_env else 2)
    assert len(familiarity_call_axes) == (2 if include_first10_by_env else 1)
    if include_first10_by_env:
        assert (familiarity_call_axes[0] is familiarity_call_axes[1]) == (by_env_layout == "shared")
        assert any(text.get_text() == "Leading 7" for axis in familiarity_axes for text in axis.texts)

    expected_width_ratios = [1.0]
    if include_beeswarm:
        expected_width_ratios.append(0.8)
    if include_first10_by_env and by_env_layout == "separate":
        expected_width_ratios.append(1.0)
    expected_width_ratios.append(1.0)
    gridspec = fig.axes[0].get_subplotspec().get_gridspec()
    assert gridspec.get_width_ratios() == pytest.approx(expected_width_ratios)
    plt.close(fig)


def test_percent_scale_is_applied_by_beeswarm_and_familiarity_renderers():
    fig, (beeswarm_axis, familiarity_axis) = plt.subplots(1, 2)
    plot_ratios_beeswarms_concise(
        beeswarm_axis,
        {
            "sf_cv_total_10": np.array([0.2, 0.4]),
            "sf_cv_total": np.array([0.3, 0.5]),
        },
        fontsize=8.0,
        value_scale=100.0,
        leading_dims=7,
    )
    curves = {
        "Env #1": {
            "svr": {"mouse-a": np.array([0.2, 0.4])},
            "total": {"mouse-a": np.array([1.0, 1.0])},
        }
    }
    render_familiarity_ratio_panel(
        familiarity_axis,
        curves,
        "by_env",
        "all",
        fontsize=8.0,
        value_scale=100.0,
    )

    np.testing.assert_allclose(beeswarm_axis.lines[0].get_ydata(), [20.0, 40.0])
    np.testing.assert_allclose(beeswarm_axis.get_yticks(), [0.0, 50.0, 100.0])
    assert [tick.get_text() for tick in beeswarm_axis.get_xticklabels()] == ["Leading 7", "All"]
    np.testing.assert_allclose(familiarity_axis.lines[0].get_ydata(), [20.0, 40.0])
    plt.close(fig)


def test_spectrum_indicator_uses_leading_dimension_label():
    fig, axis = plt.subplots()
    spectra = np.linspace(0.2, 0.01, 12)[None, :]
    plot_ratios_spectrum(
        axis,
        {"sf_cv": spectra, "ff": spectra},
        fontsize=8.0,
        leading_dims=7,
        include_legend=False,
    )

    assert any(text.get_text() == "Leading 7" for text in axis.texts)
    plt.close(fig)
