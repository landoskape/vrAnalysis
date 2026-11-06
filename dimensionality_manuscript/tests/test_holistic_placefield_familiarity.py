"""Stored-data aggregation, exposure indexing, and embedded timeline regression tests."""

from pathlib import Path
import sys
import types
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

package_name = "dimensionality_manuscript.figure_scripts.figure1"
if package_name not in sys.modules:
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[1] / "figure_scripts" / "figure1")]
    sys.modules[package_name] = package

from dimensionality_manuscript.figure_scripts.figure1.holistic_placefield_familiarity import HolisticPlacefieldFamiliarityViewer
from dimensionality_manuscript.figure_scripts.figure1.timeline import ExperimentTimeline


class Results:
    param_axes = {}
    config_class = None

    def __init__(self):
        self.sessions = [SimpleNamespace(mouse_name=mouse, date=f"2020-01-0{day}", session_id="1",
                                         environments=[1] if day == 1 else [1, 2])
                         for mouse in ("M1", "M2") for day in (3, 1, 2)]
        self.mouse_names = np.array([s.mouse_name for s in self.sessions])
        reliability = np.tile([0.8, 0.7, 0.9], (6, 2, 1))
        reliability[[1, 4], 1] = np.nan
        reliability[[2, 5], 1] = 0.6  # Present, but no quality cells on first exposure.
        # Real results retain the mouse's full slot mapping in every session, even
        # before the second environment is introduced.
        ids = np.tile([1, 2], (6, 1))
        self.arrays = {
            "reliability_slot": reliability,
            "env_slot_ids": ids,
            "fraction_active_slot": np.tile([0.3, 0.9, 0.1], (6, 2, 1)),
            "pf_rms_r_f2_from_f1_slot": np.tile([0.2, 0.4, 0.8], (6, 2, 1)),
            "pf_rms_r_independent_cv_slot": np.tile([-0.2, -0.4, -0.8], (6, 2, 1)),
            "trial_rms_cv": np.tile([0.0, 1.0, 2.0], (6, 2, 1)),
            "trial_mean_mean": np.tile([2.0, 4.0, 6.0], (6, 2, 1)),
        }

    def sel(self, *, keys, **kwargs):
        return {key: self.arrays[key] for key in keys if key in self.arrays}


def test_quality_partition_and_exposure_index_survive_empty_selected_sessions():
    viewer = HolisticPlacefieldFamiliarityViewer(Results(), cells="split")
    absolute = viewer.metric_stacks["absolute"]
    relative = viewer.metric_stacks["relative"]
    np.testing.assert_allclose(absolute["quality"]["correlation"][0], [[0.2] * 3] * 2)
    # Equality to the spatial cutoff belongs to not quality; low active cells do too.
    np.testing.assert_allclose(absolute["not quality"]["correlation"][0], [[0.6] * 3] * 2)
    np.testing.assert_allclose(absolute["quality"]["reliability"][1], [[np.nan, np.nan, 0.8]] * 2)
    np.testing.assert_allclose(relative["quality"]["reliability"][1], [[np.nan, 0.8]] * 2)
    np.testing.assert_allclose(relative["not quality"]["correlation"][1], [[1.4 / 3, 0.6]] * 2)
    state = {**viewer.state, "fraction_active_threshold": 0.3}
    viewer.refresh_summary_data(state)
    assert np.isnan(viewer.metric_stacks["relative"]["quality"]["correlation"][0]).all()


def test_metric_switches_and_transform_before_cell_summary():
    viewer = HolisticPlacefieldFamiliarityViewer(Results())
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["amplitude"][0], [[1.7 / 3] * 3] * 2)
    state = {**viewer.state, "amplitude_transform": "inverse", "summary_stat": "median"}
    viewer.refresh_summary_data(state)
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["amplitude"][0], [[0.75] * 3] * 2)
    state.update(pf_rms_comparison="independent_cv", amplitude_metric="trial_mean", amplitude_statistic="mean", amplitude_transform="raw")
    viewer.reload_arrays(state)
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["amplitude"][0], [[4.0] * 3] * 2)
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["correlation"][0], [[-0.4] * 3] * 2)


def test_metric_nan_retains_exposure_but_absent_environment_does_not():
    results = Results()
    results.arrays["pf_rms_r_f2_from_f1_slot"][[2, 5], 0] = np.nan
    for row in (2, 5):
        results.sessions[row].environments = [1]
    viewer = HolisticPlacefieldFamiliarityViewer(results)
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["correlation"][0], [[1.4 / 3, np.nan, 1.4 / 3]] * 2)
    np.testing.assert_allclose(viewer.metric_stacks["absolute"]["all"]["correlation"][1], [[np.nan, np.nan, 1.4 / 3]] * 2)
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["correlation"][1], [[1.4 / 3]] * 2)


def test_timeline_snapshot_and_both_column_geometries(tmp_path):
    results = Results()
    source = ExperimentTimeline(results.sessions, num_training=2, merge_gaps=False, fontsize=7, mode="relative")
    source_state = dict(source.state)
    viewer = HolisticPlacefieldFamiliarityViewer(results, schematic_config=source, cells="split")
    source.update_integer("num_training", value=8)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 8
    assert viewer.timeline.state["num_training"] == 2
    assert source.state["mode"] == source_state["mode"]
    absolute, relative = fig.axes[:2]
    # Phase bars precede the environment rectangles. The second env starts later only
    # in absolute order and has two imaging exposures in both representations.
    abs_env = absolute.patches[-1]
    rel_env = relative.patches[-1]
    assert abs_env.get_x() == pytest.approx(1 / 3)
    assert rel_env.get_x() == 0
    assert abs_env.get_width() == pytest.approx(2 / 3)
    assert rel_env.get_width() == pytest.approx(2 / 3)
    np.testing.assert_allclose(absolute.get_xlim(), [1 / 6, 5 / 6])
    np.testing.assert_allclose(relative.get_xlim(), [1 / 6, 5 / 6])
    assert [text.get_text() for text in absolute.texts] == ["Imaging Timeline", "1st Env", "2nd Env"]
    assert [text.get_text() for text in relative.texts] == ["Relative Timeline", "1st Env", "2nd Env"]
    assert absolute.get_title() == relative.get_title() == ""
    for ax in fig.axes[2:6]:
        assert ax.get_xlabel() == ""
        assert not ax.spines["bottom"].get_visible()
        assert not ax.get_xticklabels()
    assert fig.axes[6].get_xlabel() == "Overall session #"
    assert fig.axes[7].get_xlabel() == "Env session #"
    assert fig.axes[6].get_ylabel() == "Amplitude Reliability"
    assert all(ax.spines["bottom"].get_visible() for ax in fig.axes[6:])
    for ax in fig.axes[3::2]:
        assert not ax.spines["left"].get_visible()
        assert not ax.get_yticklabels()
        assert all(not tick.tick1line.get_visible() and not tick.tick2line.get_visible()
                   for tick in ax.yaxis.get_major_ticks())
    assert all(ax.spines["left"].get_visible() for ax in fig.axes[2::2])
    for ax in fig.axes[2:4]:
        np.testing.assert_allclose(ax.get_ylim(), [-1, 1])
    for ax in fig.axes[4:]:
        np.testing.assert_allclose(ax.get_ylim(), [0, 1])
    assert fig.axes[2].get_ylabel() == "PF-Var Alignment"
    assert fig.axes[2].yaxis.label.get_color() != "none"
    absolute_env = next(line for line in fig.axes[2].lines if line.get_label() == "2nd Env (not quality)")
    relative_env = next(line for line in fig.axes[3].lines if line.get_label() == "2nd Env (not quality)")
    np.testing.assert_array_equal(absolute_env.get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(relative_env.get_xdata(), [1, 2])
    assert any(line.get_linestyle() == "--" for line in fig.axes[2].lines)
    assert any(line.get_marker() == "." for line in fig.axes[2].lines)
    assert len(fig.axes[2].get_legend().get_texts()) == 4
    fig.savefig(tmp_path / "holistic.png")


def test_incompatible_roi_arrays_and_missing_keys_report_useful_errors():
    results = Results()
    results.arrays["trial_rms_cv"] = np.pad(results.arrays["trial_rms_cv"], ((0, 0), (0, 0), (0, 1)), constant_values=np.nan)
    HolisticPlacefieldFamiliarityViewer(results)  # Different trailing NaN padding is safe.
    results.arrays["trial_rms_cv"][0, 0, -1] = 1
    with pytest.raises(ValueError, match="ROI dimensions"):
        HolisticPlacefieldFamiliarityViewer(results)
    del results.arrays["trial_rms_cv"]
    with pytest.raises(KeyError, match="schema v11"):
        HolisticPlacefieldFamiliarityViewer(results)


@pytest.mark.parametrize("plot_style,hide_error", [("each", False), ("errorPlot", False), ("errorPlot", True)])
def test_curves_require_two_mice_at_each_xvalue(plot_style, hide_error):
    viewer = HolisticPlacefieldFamiliarityViewer(Results(), cells="split", plot_style=plot_style, hide_error=hide_error)
    supported = np.array([[0.2, 0.3, 0.4], [0.6, np.nan, 0.8]])
    unsupported = np.array([[np.nan, 0.9, np.nan], [np.nan, np.nan, np.nan]])
    for groups in viewer.metric_stacks.values():
        for group, metrics in groups.items():
            for metric in metrics:
                metrics[metric] = {0: supported if group == "quality" else unsupported}
    fig = viewer.plot(viewer.state)
    for ax in fig.axes[2:]:
        # Mask the single-mouse point in individual curves as well as the mean/band.
        assert all(np.isnan(line.get_ydata()[1]) for line in ax.lines if line.get_linestyle() != ":")
        assert not any("not quality" in line.get_label() for line in ax.lines)
        mean = next(line for line in ax.lines if line.get_label() == "1st Env (quality)")
        np.testing.assert_array_equal(mean.get_xdata(), [1, 2, 3])
        np.testing.assert_allclose(mean.get_ydata(), [0.4, np.nan, 0.6])
    assert supported[0, 1] == 0.3  # Plotting doesn't discard stored summaries.


@pytest.mark.parametrize("transform", ["raw", "inverse", "bounded"])
def test_full_row_limits_and_autoscaling(transform):
    viewer = HolisticPlacefieldFamiliarityViewer(Results(), amplitude_transform=transform, hide_error=True)
    # Use data exceeding 1 for unbounded transforms to detect accidental clipping.
    values = np.tile([2.0, 3.0, 4.0] if transform != "bounded" else [0.2, 0.3, 0.4], (2, 1))
    for groups in viewer.metric_stacks.values():
        groups["all"]["amplitude"] = {0: values}
    full = viewer.plot(viewer.state)
    for ax in full.axes[6:]:
        if transform == "bounded":
            np.testing.assert_allclose(ax.get_ylim(), [0, 1])
        else:
            assert ax.get_ylim()[0] < 2 and ax.get_ylim()[1] > 4
    state = {**viewer.state, "ylim_full_correlation": False, "ylim_full_reliability": False, "ylim_full_amplitude": False}
    auto = viewer.plot(state)
    for ax in auto.axes[2:4]:
        low, high = ax.get_ylim()
        assert -1 < low < high < 1
    for ax in auto.axes[4:6]:
        low, high = ax.get_ylim()
        assert 0 < low < high < 1
    for ax in auto.axes[6:]:
        if transform == "bounded":
            low, high = ax.get_ylim()
            assert 0 < low < high < 1
        else:
            assert ax.get_ylim()[0] < 2 and ax.get_ylim()[1] > 4


def test_min_mice_control_trims_unsupported_tails_with_shared_column_limits():
    viewer = HolisticPlacefieldFamiliarityViewer(Results(), hide_error=True)
    stack = np.array([[0.1, 0.2, 0.3, np.nan, 0.5], [np.nan, 0.4, 0.5, np.nan, np.nan]])
    for groups in viewer.metric_stacks.values():
        for metrics in groups.values():
            for metric in metrics:
                metrics[metric] = {0: stack}
    fig = viewer.plot(viewer.state)
    assert all(ax.get_xlim() == (2, 3) for ax in fig.axes[2:])
    viewer.update_integer("min_mice", value=1)
    fig = viewer.plot(viewer.state)
    for ax in fig.axes[2:]:
        assert ax.get_xlim() == (1, 5)
        line = next(line for line in ax.lines if line.get_label() == "1st Env")
        np.testing.assert_allclose(line.get_ydata(), [0.1, 0.3, 0.4, np.nan, 0.5])


def test_optional_mean_std_row_is_removed_and_excluded_from_column_limits():
    viewer = HolisticPlacefieldFamiliarityViewer(Results(), include_mean_std=False, hide_error=True)
    for groups in viewer.metric_stacks.values():
        groups["all"]["correlation"] = {0: np.ones((2, 8))}
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 6
    assert [ax.get_ylabel() for ax in fig.axes[2:]] == ["Spatial Reliability", "", "Amplitude Reliability", ""]
    assert all(ax.get_xlim() == (1, 3) for ax in fig.axes[2:])
    assert not fig.axes[2].spines["bottom"].get_visible()
    assert not fig.axes[3].spines["left"].get_visible()
    assert not fig.axes[5].spines["left"].get_visible()
    assert fig.axes[4].get_xlabel() == "Overall session #"
    assert fig.axes[5].get_xlabel() == "Env session #"
    assert fig.axes[2].get_legend() is not None

    state = {**viewer.state, "include_mean_std": True}
    fig = viewer.plot(state)
    assert len(fig.axes) == 8
    assert fig.axes[2].get_ylabel() == "PF-Var Alignment"
    assert all(ax.get_xlim() == (1, 8) for ax in fig.axes[2:])


@pytest.mark.parametrize("include_mean_std", [True, False])
@pytest.mark.parametrize("cells", ["all", "quality", "not quality", "split"])
def test_spatial_reliability_quality_threshold_line(include_mean_std, cells):
    viewer = HolisticPlacefieldFamiliarityViewer(Results(), cells=cells, include_mean_std=include_mean_std, hide_error=True)
    for threshold in (0.7, 0.4):
        fig = viewer.plot({**viewer.state, "reliability_threshold": threshold})
        row = 2 if include_mean_std else 1
        for index, ax in enumerate(fig.axes[2:], start=2):
            reference = [line for line in ax.lines if line.get_linestyle() == ":"]
            labels = [text for text in ax.texts if text.get_text() == "Placecell Cutoff"]
            if cells == "split" and index in (2 * row, 2 * row + 1):
                assert len(reference) == 1
                np.testing.assert_allclose(reference[0].get_ydata(), [threshold, threshold])
                np.testing.assert_allclose(reference[0].get_xdata(), [0, 1])
                assert len(labels) == 1
                assert labels[0].xy == (0.5, threshold)
                assert labels[0].get_ha() == "center"
                assert labels[0].get_va() == "bottom"
                assert labels[0].get_position() == (0, 3)
            else:
                assert not reference
                assert not labels


@pytest.mark.parametrize("cells", ["all", "quality", "not quality", "split"])
@pytest.mark.parametrize("baseline_sessions", [1, 2, 5])
def test_relative_values_normalize_each_mouse_metric_and_group(cells, baseline_sessions):
    results = Results()
    factors = np.array([3, 1, 2, 6, 2, 4])[:, None, None]
    results.arrays["pf_rms_r_f2_from_f1_slot"] *= factors
    results.arrays["trial_rms_cv"] = (results.arrays["trial_rms_cv"] + 0.5) * factors
    raw = HolisticPlacefieldFamiliarityViewer(results, cells=cells, amplitude_transform="raw", summary_stat="median")
    viewer = HolisticPlacefieldFamiliarityViewer(results, cells=cells, amplitude_transform="raw", summary_stat="median",
                                                relative_values=True, baseline_sessions=baseline_sessions)
    for group, metrics in raw.metric_stacks["relative"].items():
        for metric, stacks in metrics.items():
            baseline = np.mean(stacks[0][:, :baseline_sessions], axis=1)
            np.testing.assert_allclose(viewer.baselines[group][metric], baseline)
            for mode in ("absolute", "relative"):
                for slot, values in raw.metric_stacks[mode][group][metric].items():
                    np.testing.assert_allclose(viewer.metric_stacks[mode][group][metric][slot], values / baseline[:, None])
    # Disabling the widget restores the original summaries, without reloading arrays.
    viewer.update_boolean("relative_values", value=False)
    viewer.refresh_summary_data(viewer.state)
    for mode, groups in raw.metric_stacks.items():
        for group, metrics in groups.items():
            for metric, stacks in metrics.items():
                for slot, values in stacks.items():
                    np.testing.assert_allclose(viewer.metric_stacks[mode][group][metric][slot], values)


def test_invalid_baselines_keep_gaps_and_normalized_plots_autoscale():
    results = Results()
    results.arrays["pf_rms_r_f2_from_f1_slot"][1, 0] = np.nan  # First M1 exposure.
    results.arrays["pf_rms_r_f2_from_f1_slot"][4, 0] = 0       # First M2 exposure.
    results.arrays["trial_rms_cv"][[0, 3]] = 0.1  # Higher bounded amplitude after baseline.
    viewer = HolisticPlacefieldFamiliarityViewer(results, cells="split", relative_values=True, hide_error=True)
    for groups in viewer.metric_stacks.values():
        for metrics in groups.values():
            assert all(np.isnan(values).all() for values in metrics["correlation"].values())
    fig = viewer.plot(viewer.state)
    assert not any(text.get_text() == "Placecell Cutoff" for ax in fig.axes for text in ax.texts)
    assert not any(line.get_linestyle() == ":" for ax in fig.axes for line in ax.lines)
    assert fig.axes[6].get_ylim()[1] > 1  # Full-limit toggles don't clip ratios.
    viewer.update_integer("baseline_sessions", value=2)
    viewer.refresh_summary_data(viewer.state)
    # Missing first exposure doesn't extend the window into the third exposure.
    for group, expected in (("quality", 0.2), ("not quality", 0.6)):
        np.testing.assert_allclose(viewer.baselines[group]["correlation"], [expected, expected / 2])
