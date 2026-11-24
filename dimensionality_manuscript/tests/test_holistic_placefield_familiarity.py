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


def test_gamma_controls_use_paired_cells_transforms_and_dotted_curves():
    results = Results()
    results.arrays["gamma_mean_pf_rms_r_f2_from_f1_slot"] = np.tile([0.1, 0.3, 0.5], (6, 2, 1))
    results.arrays["gamma_mean_trial_rms_cv_slot"] = np.tile([1., 2., 4.], (6, 2, 1))
    results.arrays["trial_residual_rms_mean_all_from_all"] = np.tile([1., np.nan, 4.], (6, 2, 1))
    results.arrays["gamma_mean_trial_residual_rms_mean_all_from_all_slot"] = np.tile([2., 100., 4.], (6, 2, 1))
    viewer = HolisticPlacefieldFamiliarityViewer(
        results, include_gamma_control=True, include_residual_rms=True,
        residual_rms_transform="inverse", show_schematic=False,
    )
    stacks = viewer.metric_stacks["relative"]["all"]
    np.testing.assert_allclose(stacks["gamma_amplitude"][0], np.mean(1 / (1 + np.array([1., 2., 4.])**2)))
    np.testing.assert_allclose(stacks["gamma_residual_rms"][0], np.mean([0.5, 0.25]))
    assert "gamma_reliability" not in stacks
    fig = viewer.plot(viewer.state)
    try:
        for ax in fig.axes[:3] + fig.axes[6:]:
            gamma = [line for line in ax.lines if "(Gamma)" in line.get_label()]
            assert gamma
            assert all(line.get_linestyle() == ":" for line in gamma)
        assert all("(Gamma)" not in line.get_label() for ax in fig.axes[3:6] for line in ax.lines)
    finally:
        from matplotlib import pyplot as plt
        plt.close(fig)
    viewer.update_boolean("relative_values", value=True)
    viewer.refresh_summary_data(viewer.state)
    for metric in ("gamma_amplitude", "gamma_residual_rms", "gamma_correlation"):
        np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"][metric][0], 1.)


def test_gamma_controls_are_suppressed_for_split_and_reload_when_cells_change():
    results = Results()
    viewer = HolisticPlacefieldFamiliarityViewer(results, include_gamma_control=True, cells="split")
    assert not any(key.startswith("gamma_") for key in viewer.arrays)
    results.arrays["gamma_mean_pf_rms_r_f2_from_f1_slot"] = results.arrays["pf_rms_r_f2_from_f1_slot"]
    results.arrays["gamma_mean_trial_rms_cv_slot"] = results.arrays["trial_rms_cv"]
    viewer.update_selection("cells", value="all")
    viewer.reload_arrays(viewer.state)
    assert "gamma_amplitude" in viewer.metric_stacks["relative"]["all"]
    viewer.update_selection("cells", value="split")
    viewer.reload_arrays(viewer.state)
    assert not any(key.startswith("gamma_") for key in viewer.arrays)
    viewer.update_boolean("include_gamma_control", value=False)
    viewer.update_selection("cells", value="all")
    viewer.reload_arrays(viewer.state)
    assert not any(key.startswith("gamma_") for key in viewer.arrays)


def test_gamma_fraction_uses_the_selected_fold_and_missing_controls_require_v15():
    with pytest.raises(KeyError, match="schema v15"):
        HolisticPlacefieldFamiliarityViewer(Results(), include_gamma_control=True)
    results = Results()
    results.arrays["gamma_mean_pf_rms_r_f2_from_f1_slot"] = results.arrays["pf_rms_r_f2_from_f1_slot"]
    results.arrays["gamma_mean_trial_rms_cv_slot"] = results.arrays["trial_rms_cv"]
    results.arrays["pf_frac_unexplained_cv_slot"] = np.full((6, 2, 3), 1.5)
    results.arrays["gamma_mean_pf_frac_unexplained_cv_slot"] = np.full((6, 2, 3), 0.5)
    viewer = HolisticPlacefieldFamiliarityViewer(
        results, include_gamma_control=True, include_residual_rms=True,
        residual_metric="unexplained_fraction", residual_rms_fold="cv",
    )
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["gamma_unexplained_fraction"][0], 0.5)


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


@pytest.mark.parametrize("statistic", ["mean", "variance", "cv"])
@pytest.mark.parametrize("transform", ["raw", "inverse", "bounded", "inv"])
def test_residual_rms_independent_controls_transform_before_cell_summary(statistic, transform):
    from dimensionality_manuscript.configs.pfpred_quality import PF_VARIANCE_COMPARISONS
    results = Results()
    for index, fold in enumerate(PF_VARIANCE_COMPARISONS, 1):
        results.arrays[f"trial_residual_rms_{statistic}_{fold}"] = np.tile(np.array([1., 2., 4.]) * index, (6, 2, 1))
    viewer = HolisticPlacefieldFamiliarityViewer(
        results, include_residual_rms=True, residual_rms_statistic=statistic, residual_rms_transform=transform)
    for index, fold in enumerate(PF_VARIANCE_COMPARISONS, 1):
        state = {**viewer.state, "residual_rms_fold": fold}
        viewer.reload_arrays(state)
        values = np.array([1., 2., 4.]) * index
        expected = {"raw": values, "inverse": 1 / values, "inv": 1 / values,
                    "bounded": 1 / (1 + values**2)}[transform]
        np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["residual_rms"][0], expected.mean())
        # Amplitude settings and values remain independent of residual settings.
        np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["amplitude"][0], 1.7 / 3)


@pytest.mark.parametrize("include_mean_std,show_schematic", [(True, True), (False, False)])
def test_residual_rms_is_bottom_row_with_session_labels(include_mean_std, show_schematic):
    results = Results()
    results.arrays["trial_residual_rms_mean_all_from_all"] = np.tile([1., 2., 4.], (6, 2, 1))
    viewer = HolisticPlacefieldFamiliarityViewer(
        results, include_residual_rms=True, residual_rms_transform="bounded",
        include_mean_std=include_mean_std, show_schematic=show_schematic, cells="split")
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == (3 + int(include_mean_std) + int(show_schematic)) * 3
    assert fig.axes[-3].get_ylabel() == "Bounded Residual RMS Mean"
    assert fig.axes[-3].get_xlabel() == "Overall session #"
    assert fig.axes[-2].get_xlabel() == fig.axes[-1].get_xlabel() == "Env session #"
    assert all(ax.get_xlabel() == "" for ax in fig.axes[:-3])
    np.testing.assert_allclose(fig.axes[-3].get_ylim(), [0, 1])
    assert set(viewer.metric_stacks["relative"]) == {"quality", "not quality"}
    viewer.reload_arrays({**viewer.state, "include_residual_rms": False})
    assert "residual_rms" not in viewer.arrays
    assert "residual_rms" not in viewer.metric_stacks["relative"]["quality"]


def test_residual_rms_requires_new_results_only_when_enabled():
    HolisticPlacefieldFamiliarityViewer(Results())
    with pytest.raises(KeyError, match="schema v13"):
        HolisticPlacefieldFamiliarityViewer(Results(), include_residual_rms=True)


def test_unexplained_fraction_uses_selected_fold_and_shows_negative_r2():
    results = Results()
    results.arrays["pf_frac_unexplained_cv_slot"] = np.tile([0.5, 1.5, 2.5], (6, 2, 1))
    viewer = HolisticPlacefieldFamiliarityViewer(
        results, include_residual_rms=True, residual_metric="unexplained_fraction", residual_rms_fold="cv",
        show_schematic=False,
    )
    fig = viewer.plot(viewer.state)
    try:
        axis = fig.axes[-3]
        assert axis.get_xlabel() == "Overall session #"
        assert axis.get_ylabel() == r"PF × Gain $1-R^2$"
        assert axis.get_ylim()[1] > 1.5
        line = next(line for line in axis.lines if line.get_label() == "1st Env")
        np.testing.assert_allclose(line.get_ydata(), 1.5)
    finally:
        from matplotlib import pyplot as plt
        plt.close(fig)


@pytest.mark.parametrize("plot_style", ["each", "errorPlot"])
def test_mouse_selection_controls_example_columns_without_changing_average(plot_style):
    results = Results()
    results.arrays["pf_rms_r_f2_from_f1_slot"][3:] *= 2
    # Only M1 contributes to the final chronological / relative exposure.
    results.arrays["pf_rms_r_f2_from_f1_slot"][3] = np.nan
    viewer = HolisticPlacefieldFamiliarityViewer(results, plot_style=plot_style)
    for mouse, expected in (("M1", [1.4 / 3] * 3), ("M2", [2.8 / 3, 2.8 / 3, np.nan])):
        fig = viewer.plot({**viewer.state, "mouse": mouse})
        for ax in fig.axes[3:5]:
            line = next(line for line in ax.lines if line.get_label() == "1st Env")
            np.testing.assert_allclose(line.get_ydata(), expected)
            assert not ax.collections  # Single mouse curves have no across-mouse band.
            assert ax.get_xlim() == (1, 3)  # Other metrics still support exposure 3.
        line = next(line for line in fig.axes[5].lines if line.get_label() == "1st Env")
        np.testing.assert_allclose(line.get_ydata(), [0.7, 0.7, np.nan])
        assert fig.axes[5].get_xlim() == (1, 3)  # Other metrics still support exposure 3.
        assert all(ax.get_title() == "" for ax in fig.axes[:3])


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
    assert len(fig.axes) == 12
    assert viewer.timeline.state["num_training"] == 2
    assert source.state["mode"] == source_state["mode"]
    absolute, relative, average = fig.axes[:3]
    # Only environment rectangles remain. The second env starts later only
    # in absolute order and has two imaging exposures in both representations.
    abs_env = absolute.patches[-1]
    rel_env = relative.patches[-1]
    assert abs_env.get_x() == pytest.approx(1 / 3)
    assert rel_env.get_x() == 0
    assert abs_env.get_width() == pytest.approx(2 / 3)
    assert rel_env.get_width() == pytest.approx(2 / 3)
    np.testing.assert_allclose(absolute.get_xlim(), [1 / 6, 5 / 6])
    np.testing.assert_allclose(relative.get_xlim(), [1 / 6, 5 / 6])
    assert [text.get_text() for text in absolute.texts] == ["Chronological Time", "1st Env", "2nd Env"]
    assert [text.get_text() for text in relative.texts] == ["Relative Time", "1st Env", "2nd Env"]
    assert [text.get_text() for text in average.texts] == ["Relative Time", "1st Env", "2nd Env"]
    assert all(len(ax.patches) == 2 for ax in fig.axes[:3])
    assert absolute.get_title() == relative.get_title() == average.get_title() == ""
    for ax in fig.axes[3:9]:
        assert ax.get_xlabel() == ""
        assert not ax.spines["bottom"].get_visible()
        assert not ax.get_xticklabels()
    assert fig.axes[9].get_xlabel() == "Overall session #"
    assert fig.axes[10].get_xlabel() == fig.axes[11].get_xlabel() == "Env session #"
    assert fig.axes[9].get_ylabel() == "Amplitude Reliability"
    assert all(ax.spines["bottom"].get_visible() for ax in fig.axes[9:])
    for ax in [ax for index, ax in enumerate(fig.axes[3:], start=3) if index % 3]:
        assert not ax.spines["left"].get_visible()
        assert not ax.get_yticklabels()
        assert all(not tick.tick1line.get_visible() and not tick.tick2line.get_visible()
                   for tick in ax.yaxis.get_major_ticks())
    assert all(ax.spines["left"].get_visible() for ax in fig.axes[3::3])
    for ax in fig.axes[3:6]:
        np.testing.assert_allclose(ax.get_ylim(), [-1, 1])
    for ax in fig.axes[6:]:
        np.testing.assert_allclose(ax.get_ylim(), [0, 1])
    assert fig.axes[3].get_ylabel() == "PF-Var Alignment"
    assert fig.axes[3].yaxis.label.get_color() != "none"
    absolute_env = next(line for line in fig.axes[3].lines if line.get_label() == "2nd Env (not quality)")
    relative_env = next(line for line in fig.axes[4].lines if line.get_label() == "2nd Env (not quality)")
    np.testing.assert_array_equal(absolute_env.get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(relative_env.get_xdata(), [1, 2])
    assert any(line.get_linestyle() == "--" for line in fig.axes[3].lines)
    assert any(line.get_marker() == "." for line in fig.axes[3].lines)
    assert len(fig.axes[3].get_legend().get_texts()) == 4
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
    for ax in fig.axes[5::3]:
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
    for ax in full.axes[9:]:
        if transform == "bounded":
            np.testing.assert_allclose(ax.get_ylim(), [0, 1])
        else:
            assert ax.get_ylim()[0] < 2 and ax.get_ylim()[1] > 4
    state = {**viewer.state, "ylim_full_correlation": False, "ylim_full_reliability": False, "ylim_full_amplitude": False}
    auto = viewer.plot(state)
    for ax in auto.axes[3:6]:
        low, high = ax.get_ylim()
        assert -1 < low < high < 1
    for ax in auto.axes[6:9]:
        low, high = ax.get_ylim()
        assert 0 < low < high < 1
    for ax in auto.axes[9:]:
        if transform == "bounded":
            low, high = ax.get_ylim()
            assert 0 < low < high < 1
        else:
            assert ax.get_ylim()[0] < 2 and ax.get_ylim()[1] > 4


@pytest.mark.parametrize("include_mean_std", [True, False])
def test_bottom_spines_stay_below_panels_when_later_columns_expand_shared_limits(include_mean_std):
    viewer = HolisticPlacefieldFamiliarityViewer(Results(), include_mean_std=include_mean_std,
                                                ylim_full_amplitude=False)
    for groups in viewer.metric_stacks.values():
        groups["all"]["amplitude"] = {0: np.array([[0.2, 0.3, 0.4], [0.1, 0.5, 0.9]])}
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    for ax in fig.axes[-3:]:
        assert ax.get_ylim()[0] < 0.1 and ax.get_ylim()[1] > 0.9
        spine = ax.spines["bottom"]
        vertices = ax.transAxes.inverted().transform(spine.get_transform().transform(spine.get_path().vertices))
        np.testing.assert_allclose(vertices[:, 1], -0.02)


def test_min_mice_control_trims_unsupported_tails_with_shared_column_limits():
    viewer = HolisticPlacefieldFamiliarityViewer(Results(), hide_error=True)
    stack = np.array([[0.1, 0.2, 0.3, np.nan, 0.5], [np.nan, 0.4, 0.5, np.nan, np.nan]])
    for groups in viewer.metric_stacks.values():
        for metrics in groups.values():
            for metric in metrics:
                metrics[metric] = {0: stack}
    fig = viewer.plot(viewer.state)
    assert all(ax.get_xlim() == (2, 3) for ax in fig.axes[5::3])
    assert all(ax.get_xlim() == (1, 5) for ax in fig.axes[3::3] + fig.axes[4::3])
    selected = viewer.plot({**viewer.state, "mouse": "M2"})
    assert all(ax.get_xlim() == (2, 3) for ax in selected.axes[3:])
    viewer.update_integer("min_mice", value=1)
    fig = viewer.plot(viewer.state)
    for ax in fig.axes[5::3]:
        assert ax.get_xlim() == (1, 5)
        line = next(line for line in ax.lines if line.get_label() == "1st Env")
        np.testing.assert_allclose(line.get_ydata(), [0.1, 0.3, 0.4, np.nan, 0.5])


def test_optional_mean_std_row_is_removed_and_excluded_from_column_limits():
    viewer = HolisticPlacefieldFamiliarityViewer(Results(), include_mean_std=False, hide_error=True)
    for groups in viewer.metric_stacks.values():
        groups["all"]["correlation"] = {0: np.ones((2, 8))}
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 9
    assert [ax.get_ylabel() for ax in fig.axes[3:]] == ["Spatial Reliability", "", "", "Amplitude Reliability", "", ""]
    assert all(ax.get_xlim() == (1, 3) for ax in fig.axes[3:])
    assert not fig.axes[3].spines["bottom"].get_visible()
    assert not fig.axes[4].spines["left"].get_visible()
    assert not fig.axes[8].spines["left"].get_visible()
    assert fig.axes[6].get_xlabel() == "Overall session #"
    assert fig.axes[7].get_xlabel() == fig.axes[8].get_xlabel() == "Env session #"
    assert fig.axes[3].get_legend() is not None

    state = {**viewer.state, "include_mean_std": True}
    fig = viewer.plot(state)
    assert len(fig.axes) == 12
    assert fig.axes[3].get_ylabel() == "PF-Var Alignment"
    assert all(ax.get_xlim() == (1, 8) for ax in fig.axes[3:])


@pytest.mark.parametrize("include_mean_std", [True, False])
@pytest.mark.parametrize("cells", ["all", "quality", "not quality", "split"])
def test_spatial_reliability_quality_threshold_line(include_mean_std, cells):
    viewer = HolisticPlacefieldFamiliarityViewer(Results(), cells=cells, include_mean_std=include_mean_std, hide_error=True)
    for threshold in (0.7, 0.4):
        fig = viewer.plot({**viewer.state, "reliability_threshold": threshold})
        row = 2 if include_mean_std else 1
        for index, ax in enumerate(fig.axes[3:], start=3):
            reference = [line for line in ax.lines if line.get_linestyle() == ":"]
            labels = [text for text in ax.texts if text.get_text() == "Placecell Cutoff"]
            if cells == "split" and index // 3 == row:
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
    assert fig.axes[9].get_ylim()[1] > 1  # Full-limit toggles don't clip ratios.
    viewer.update_integer("baseline_sessions", value=2)
    viewer.refresh_summary_data(viewer.state)
    # Missing first exposure doesn't extend the window into the third exposure.
    for group, expected in (("quality", 0.2), ("not quality", 0.6)):
        np.testing.assert_allclose(viewer.baselines[group]["correlation"], [expected, expected / 2])
