"""Fold selection and shared holistic familiarity behavior for variance fractions."""

from pathlib import Path
from types import SimpleNamespace
import sys
import types

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

package_name = "dimensionality_manuscript.figure_scripts.figure1"
if package_name not in sys.modules:
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[1] / "figure_scripts" / "figure1")]
    sys.modules[package_name] = package

from dimensionality_manuscript.configs.pfpred_quality import PF_VARIANCE_COMPARISONS
from dimensionality_manuscript.figure_scripts.figure1.placefield_breakdown_familiarity import PlacefieldBreakdownFamiliarityViewer
from dimensionality_manuscript.figure_scripts.figure1.timeline import ExperimentTimeline


class Results:
    param_axes = {"spks_type": ["sigrebase", "oasis"]}
    config_class = None

    def __init__(self):
        self.sessions = [SimpleNamespace(
            mouse_name=mouse, date=f"2020-01-0{day}", session_id="1", environments=[1] if day == 1 else [1, 2],
        ) for mouse in ("M1", "M2") for day in (3, 1, 2)]
        self.mouse_names = np.array([s.mouse_name for s in self.sessions])
        self.arrays = {
            "env_slot_ids": np.tile([1, 2], (6, 1)),
            "reliability_slot": np.tile([0.9, 0.7, 0.8], (6, 2, 1)),
            "fraction_active_slot": np.tile([0.4, 0.5, 0.1], (6, 2, 1)),
        }
        self.arrays["reliability_slot"][[2, 5], 1] = 0.5
        for index, fold in enumerate(PF_VARIANCE_COMPARISONS[:5], 1):
            for metric, scale in (("template", 0.1), ("gain", 0.2), ("additive", 0.3)):
                self.arrays[f"pf_frac_var_{metric}_{fold}_slot"] = np.tile(np.array([1., 2., 3.]) * scale * index, (6, 2, 1))
        self.average_folds()
        self.selections = []

    def average_folds(self):
        for average, directions in (("cv", ("f2_from_f1", "f1_from_f2")),
                                    ("within_cv", ("f1_from_f1", "f2_from_f2"))):
            for metric in ("template", "gain", "additive"):
                left, right = [self.arrays[f"pf_frac_var_{metric}_{fold}_slot"] for fold in directions]
                self.arrays[f"pf_frac_var_{metric}_{average}_slot"] = (left + right) / 2

    def sel(self, *, keys, **kwargs):
        self.selections.append(kwargs)
        return {key: self.arrays[key] for key in keys if key in self.arrays}


@pytest.mark.parametrize("fold", PF_VARIANCE_COMPARISONS)
def test_all_fold_selections_load_component_fractions_and_forward_configuration(fold):
    results = Results()
    viewer = PlacefieldBreakdownFamiliarityViewer(results, fold=fold, spks_type="oasis")
    assert viewer.state["fold"] == fold
    assert results.selections[-1] == {"squeeze_ones": False, "spks_type": "oasis"}
    for metric in viewer._METRICS:
        np.testing.assert_allclose(viewer.arrays[metric], results.arrays[f"pf_frac_var_{metric}_{fold}_slot"])
    assert not {"amplitude_transform", "pf_rms_comparison", "include_mean_std"} & viewer.state.keys()
    # A live fold change reloads arrays and the summaries used by every panel.
    viewer.reload_arrays({**viewer.state, "fold": "all_from_all"})
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["template"][0], 0.2)


@pytest.mark.parametrize("summary_stat", ["mean", "median"])
def test_cv_averages_folds_per_cell_before_summary_and_preserves_missing_pairs(summary_stat):
    results = Results()
    results.arrays["pf_frac_var_template_f2_from_f1_slot"][:] = [0., 10., np.nan]
    results.arrays["pf_frac_var_template_f1_from_f2_slot"][:] = [10., 0., 4.]
    results.average_folds()
    viewer = PlacefieldBreakdownFamiliarityViewer(results, summary_stat=summary_stat)
    assert viewer.state["fold"] == "cv"
    np.testing.assert_allclose(viewer.arrays["template"][0, 0], [5., 5., np.nan])
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["all"]["template"][0], 5.)


@pytest.mark.parametrize("plot_style", ["each", "errorPlot"])
def test_layout_quality_groups_exposure_gaps_and_unclipped_fractions(plot_style, tmp_path):
    results = Results()
    timeline = ExperimentTimeline(results.sessions, num_training=2, fontsize=7)
    original_state = dict(timeline.state)
    viewer = PlacefieldBreakdownFamiliarityViewer(results, schematic_config=timeline, cells="split", plot_style=plot_style, ylim_full_components=False)
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["quality"]["template"][1], [[np.nan, 0.35]] * 2)
    # Reliability equal to its cutoff and insufficient fraction active both fail quality.
    np.testing.assert_allclose(viewer.metric_stacks["relative"]["not quality"]["template"][0], 0.875)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 12
    assert [ax.get_ylabel() for ax in fig.axes[3::3]] == ["PF Component", "Gain Component", "Residual Component"]
    assert [ax.get_xlabel() for ax in fig.axes[-3:]] == ["Overall session #", "Env session #", "Env session #"]
    assert [text.get_text() for text in fig.axes[0].texts][:1] == ["Chronological Time"]
    assert [text.get_text() for text in fig.axes[1].texts][:1] == ["Relative Time"]
    assert all(ax.get_xlabel() == "" for ax in fig.axes[3:9])
    assert all(not ax.spines["left"].get_visible() for ax in fig.axes[4::3] + fig.axes[5::3])
    assert fig.axes[9].get_ylim()[1] > 2  # Residual fractions are not bounded by one.
    assert any(line.get_linestyle() == "--" for line in fig.axes[3].lines)
    assert not any(text.get_text() == "Placecell Cutoff" for ax in fig.axes for text in ax.texts)
    assert timeline.state == original_state
    fig.savefig(tmp_path / "breakdown.png")


def test_equal_mouse_weight_and_minimum_support_preserve_session_coordinates():
    results = Results()
    results.arrays["pf_frac_var_template_cv_slot"][1, 0] = [0., 0., 1.]
    results.arrays["pf_frac_var_template_cv_slot"][4, 0] = [1., np.nan, np.nan]
    # Only M1 supports the last first-environment exposure for every metric.
    for metric in ("template", "gain", "additive"):
        results.arrays[f"pf_frac_var_{metric}_cv_slot"][3, 0] = np.nan
    viewer = PlacefieldBreakdownFamiliarityViewer(results, hide_error=True)
    fig = viewer.plot(viewer.state)
    line = next(line for line in fig.axes[5].lines if line.get_label() == "1st Env")
    np.testing.assert_allclose(line.get_ydata(), [2 / 3, 0.7, np.nan])
    assert fig.axes[3].get_xlim() == (1, 3)
    assert fig.axes[5].get_xlim() == (1, 2)
    fig = viewer.plot({**viewer.state, "mouse": "M2", "min_mice": 1})
    line = next(line for line in fig.axes[3].lines if line.get_label() == "1st Env")
    np.testing.assert_allclose(line.get_ydata(), [1., 0.7, np.nan])
    assert fig.axes[5].get_xlim() == (1, 3)


def test_baseline_normalization_applies_to_each_component_and_group():
    results = Results()
    for metric in ("template", "gain", "additive"):
        results.arrays[f"pf_frac_var_{metric}_cv_slot"] *= np.array([3., 1., 2., 6., 2., 4.])[:, None, None]
    raw = PlacefieldBreakdownFamiliarityViewer(results, cells="split")
    viewer = PlacefieldBreakdownFamiliarityViewer(results, cells="split", relative_values=True,
                                                baseline_sessions=2, ylim_full_components=True)
    for group in ("quality", "not quality"):
        for metric in viewer._METRICS:
            baseline = raw.metric_stacks["relative"][group][metric][0][:, :2].mean(axis=1)
            np.testing.assert_allclose(viewer.baselines[group][metric], baseline)
            for mode in ("absolute", "relative"):
                for slot, values in raw.metric_stacks[mode][group][metric].items():
                    np.testing.assert_allclose(viewer.metric_stacks[mode][group][metric][slot], values / baseline[:, None])
    fig = viewer.plot(viewer.state)
    assert all(ax.get_ylim()[1] > 1 for ax in fig.axes[3:])
    fig = raw.plot({**raw.state, "ylim_full_components": True})
    assert all(ax.get_ylim() == (0., 1.) for ax in fig.axes[3:])


def test_array_alignment_and_missing_v12_keys():
    results = Results()
    key = "pf_frac_var_gain_cv_slot"
    results.arrays[key] = np.pad(results.arrays[key], ((0, 0), (0, 0), (0, 1)), constant_values=np.nan)
    PlacefieldBreakdownFamiliarityViewer(results)
    results.arrays[key][0, 0, -1] = 1.
    with pytest.raises(ValueError, match="ROI dimensions"):
        PlacefieldBreakdownFamiliarityViewer(results)
    del results.arrays[key]
    with pytest.raises(KeyError, match="schema v12"):
        PlacefieldBreakdownFamiliarityViewer(results)


def test_default_component_limits_only_change_display():
    viewer = PlacefieldBreakdownFamiliarityViewer(Results())
    assert viewer.state["ylim_full_components"] is True
    assert viewer.arrays["gain"].max() > 1
    fig = viewer.plot(viewer.state)
    assert all(ax.get_ylim() == (0., 1.) for ax in fig.axes[3:])
    fig = viewer.plot({**viewer.state, "ylim_full_components": False})
    assert fig.axes[6].get_ylim()[1] > 1


@pytest.mark.parametrize("plot_style", ["each", "errorPlot"])
def test_summary_schematic_uses_supported_curve_endpoints_independent_of_mouse(plot_style):
    viewer = PlacefieldBreakdownFamiliarityViewer(Results(), plot_style=plot_style, cells="split")
    # Remove the selected mouse's second environment from its illustrative timeline.
    for session in viewer.results.sessions[:3]:
        session.environments = [1]
    for groups in viewer.metric_stacks.values():
        for metrics in groups.values():
            for metric in metrics:
                metrics[metric] = {
                    0: np.array([[1., 1., np.nan, 1., 1.], [1., 1., np.nan, 1., np.nan]]),
                    1: np.array([[np.nan, 1., 1.], [np.nan, 1., 1.]]),
                }
    for minimum, expected in ((2, [(1, 4), (2, 3)]), (1, [(1, 5), (2, 3)])):
        state = {**viewer.state, "min_mice": minimum}
        assert list(viewer._summary_timeline_ranges(state).values()) == expected
        for mouse in viewer.mice:
            fig = viewer.plot({**state, "mouse": mouse})
            fig.canvas.draw()
            schematic, curves = fig.axes[2], fig.axes[5]
            assert len(schematic.patches) == 2
            for patch, (first, last) in zip(schematic.patches, expected):
                # Compare endpoints in display coordinates; axes use different data units.
                actual = schematic.transData.transform([
                    [patch.get_x(), 0], [patch.get_x() + patch.get_width(), 0],
                ])[:, 0]
                target = curves.transData.transform([[first, 0], [last, 0]])[:, 0]
                np.testing.assert_allclose(actual, target)
