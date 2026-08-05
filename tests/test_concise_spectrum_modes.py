"""Focused tests for ConciseSpectrumViewer spectrum and panel modes."""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from dimensionality_manuscript.figure_scripts.figure3 import concise_spectrum
from dimensionality_manuscript.figure_scripts.figure3._ratios import ratios_arrays, spectrum_display_arrays
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

    arrays = ratios_arrays(results, {}, spectrum_mode="avg_env", env_full_scope="within_env")
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
    env_call = next(
        call
        for call in results.calls
        if call[0] == ("sf_cv_env_fullall", "ff_env_full1_fullall")
    )
    assert env_call[1]["include_iti"] is True


def _plot_state(include):
    return {
        "include": include,
        "fontsize": 8.0,
        "ax1_width_ratio": 0.8,
        "ax23_width_ratio": 1.0,
        "pf_smooth_method": "none",
        "pf_smooth_width": 0.0,
        "full_smooth_method": "none",
        "full_smooth_width": 0.0,
        "clip_negative": False,
        "normalize": False,
        "ylims": (-5.0, -1.0),
        "shared_variance_ratio_ylims": (0.1, 0.9),
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
    ("include", "expected_axes", "expected_legend", "expected_first10_indicator"),
    [
        ("both", 4, ["1st", "All"], True),
        ("all_only", 3, ["1st", "All"], False),
        ("none", 2, ["1st"], False),
    ],
)
def test_include_controls_panels_indicator_and_all_legend_entry(
    monkeypatch,
    include,
    expected_axes,
    expected_legend,
    expected_first10_indicator,
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

    monkeypatch.setattr(concise_spectrum, "spectrum_display_arrays", lambda *_args, **_kwargs: {})

    def fake_spectrum(*_args, **kwargs):
        first10_indicator.append(kwargs["show_first10_indicator"])

    monkeypatch.setattr(concise_spectrum, "plot_ratios_spectrum", fake_spectrum)
    monkeypatch.setattr(concise_spectrum, "draw_shared_reliable_annotation", lambda *_args, **_kwargs: None)

    def fake_ratio(axis, *_args, **_kwargs):
        axis.plot([0, 1], [0.2, 0.2], label="All")

    def fake_familiarity(axis, *_args, **_kwargs):
        axis.plot([0, 1], [0.2, 0.3], label="Env #1")
        axis.legend()

    def fake_apply(axis, _state, _fontsize, *, prefix, handles=None, labels=None, **_kwargs):
        if prefix == "env_legend":
            env_legend_labels.extend(labels)
            axis.legend(handles, labels)

    monkeypatch.setattr(concise_spectrum, "plot_ratios_beeswarms_concise", fake_ratio)
    monkeypatch.setattr(concise_spectrum, "render_familiarity_ratio_panel", fake_familiarity)
    monkeypatch.setattr(concise_spectrum, "apply_legend", fake_apply)

    fig = viewer.plot(_plot_state(include))

    assert len(fig.axes) == expected_axes
    assert env_legend_labels == expected_legend
    assert first10_indicator == [expected_first10_indicator]
    if include == "none":
        assert fig.axes[1].get_ylabel() == "Shared Variance Ratio"
        assert fig.axes[1].spines["left"].get_visible()
        assert fig.axes[1].get_ylim() == pytest.approx((0.1, 0.9))
        assert fig.axes[1].spines["left"].get_bounds() == pytest.approx((0.1, 0.9))
    else:
        ratio_axis = fig.axes[1]
        assert ratio_axis.get_ylim() == pytest.approx((0.1, 0.9))
        assert ratio_axis.spines["left"].get_bounds() == pytest.approx((0.1, 0.9))
    plt.close(fig)
