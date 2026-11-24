"""Frame alignment, full-trial fitting, exposure indexing, and population weighting."""

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

from dimensionality_manuscript.configs.pfpred_quality import PFPredQualityConfig
from dimensionality_manuscript.figure_scripts.figure1 import pf_variance as panels


class Results:
    param_axes = {}
    config_class = PFPredQualityConfig

    def __init__(self):
        self.sessions = [SimpleNamespace(
            mouse_name=mouse, date=f"2020-01-0{day}", session_id="1", environments=[1] if day == 1 else [1, 2],
            session_print=lambda mouse=mouse, day=day: f"{mouse}/2020-01-0{day}/1",
        ) for mouse in ("M1", "M2") for day in (3, 1, 2)]
        self.mouse_names = np.array([s.mouse_name for s in self.sessions])
        shape = (6, 2, 3)
        self.arrays = {
            "best_env_slot": np.zeros(6),
            "env_slot_ids": np.tile([1, 2], (6, 1)),
            "reliability_slot": np.tile([0.9, 0.5, 0.8], (6, 2, 1)),
            "fraction_active_slot": np.tile([0.4, 0.5, 0.1], (6, 2, 1)),
            **{f"pf_frac_var_{name}_cv_slot": np.full(shape, value) for name, value in
               (("template", 0.4), ("gain", 0.2), ("additive", 0.5), ("covariance", -0.1))},
        }
        self.arrays.update({key.replace("_cv_slot", "_all_from_all_slot"): value
                            for key, value in list(self.arrays.items()) if key.endswith("_cv_slot")})
        # First exposure to env 2 has no quality cells; do not remove its time point.
        self.arrays["reliability_slot"][[2, 5], 1] = 0.5

    def sel(self, *, keys, **kwargs):
        return {key: self.arrays[key] for key in keys if key in self.arrays}


def test_familiarity_preserves_empty_exposures_and_weights_mice_equally():
    viewer = panels.PlacefieldVarianceFamiliarity(Results(), cells="split")
    curves = viewer.exposure_curves()
    np.testing.assert_allclose(curves["quality", "template", 1], [[np.nan, 0.4], [np.nan, 0.4]], equal_nan=True)
    np.testing.assert_allclose(curves["not quality", "covariance", 0], -0.1)
    stats = viewer.session_statistics()
    empty = stats[(stats.group == "quality") & (stats.env_slot == 1) & (stats.exposure == 1)]
    assert (empty.n_rois == 0).all()
    assert empty["median"].isna().all()
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 8
    first_mean = fig.axes[4].lines[0]
    np.testing.assert_allclose(first_mean.get_ydata(), 0.4)


def test_summary_keeps_signed_and_large_ratios_in_stats_and_scatter():
    results = Results()
    results.arrays["pf_frac_var_gain_cv_slot"][1, 0, 0] = 2.3
    viewer = panels.PlacefieldVarianceSummary(results, session="M1/2020-01-01/1", display_percentile=90.)
    stats = viewer.session_statistics()
    selected = stats[(stats.row == 1) & (stats.env_slot == 0) & (stats.component == "gain")]
    np.testing.assert_allclose(selected["mean"], 0.9)
    selected = stats[(stats.row == 1) & (stats.env_slot == 0) & (stats.component == "covariance")]
    np.testing.assert_allclose(selected["mean"], -0.1)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 12
    gain_scatter = fig.axes[5].collections[0].get_offsets()
    assert gain_scatter[:, 1].max() == 2.3
    assert "outside: 1" in fig.axes[1].texts[0].get_text()
    assert "outside: 1" in fig.axes[5].texts[0].get_text()
    assert fig.axes[5].get_ylim()[1] < 2.3


def test_population_requires_new_metric_keys():
    results = Results()
    del results.arrays["pf_frac_var_covariance_cv_slot"]
    with pytest.raises(KeyError, match="schema v12"):
        panels.PlacefieldVarianceSummary(results)


def test_mouse_average_is_not_cell_pooled_and_requires_minimum_mouse_count():
    results = Results()
    values = results.arrays["pf_frac_var_template_cv_slot"]
    values[1, 0] = [0., 0., 1.]  # First M1 exposure, mean 1/3 from three cells.
    values[4, 0] = [1., np.nan, np.nan]  # First M2 exposure, mean 1 from one cell.
    viewer = panels.PlacefieldVarianceFamiliarity(results, summary_stat="mean", min_mice=2)
    fig = viewer.plot(viewer.state)
    assert fig.axes[4].lines[0].get_ydata()[0] == pytest.approx(2 / 3)
    values[4, 0] = np.nan
    viewer.reload_arrays(viewer.state)
    fig = viewer.plot(viewer.state)
    assert np.isnan(fig.axes[4].lines[0].get_ydata()[0])
    fig = viewer.plot({**viewer.state, "min_mice": 1})
    assert fig.axes[4].lines[0].get_ydata()[0] == pytest.approx(1 / 3)


def _focus_inputs(monkeypatch):
    session = SimpleNamespace(
        mouse_name="M1", date="2020-01-01", session_id="1", session_uid="session",
        session_print=lambda: "M1/2020-01-01/1", params=SimpleNamespace(spks_type="original"),
        timestamps=np.arange(10.) / 4,
    )
    templates = np.array([[[1., 2., 3.], [2., 4., 6.]], [[2., 1., 0.], [4., 2., 0.]]])
    stored = dict(env_slot_ids=np.array([1, 2]), pf_all_slot=templates, pf_rms_distcenters=np.arange(3.) + 0.5)
    stored.update(reliability_slot=np.array([[.9, .8], [.8, .4]]),
                  fraction_active_slot=np.array([[.4, .6], [.5, .6]]),
                  pf_frac_var_gain_cv_slot=np.array([[.1, .9], [.5, .1]]),
                  pf_frac_var_additive_cv_slot=np.array([[.8, .1], [.2, .8]]),
                  pf_frac_var_template_cv_slot=np.array([[.2, .4], [.3, .2]]))
    results = SimpleNamespace(sessions=[session], mouse_names=np.array(["M1"]), param_axes={},
                              config_class=PFPredQualityConfig, store=SimpleNamespace(get=lambda *args: stored))
    behavior = SimpleNamespace(position=np.array([.5, 1.5, 2.5, np.nan, .5, 1.5, 2.5, .5, 1.5, 2.5]),
                               environment=np.array([1, 1, 1, -1, 1, 1, 1, 2, 2, 2]),
                               trial=np.array([0, 0, 0, np.nan, 1, 1, 1, 2, 2, 2]),
                               speed=np.array([2, 2, 2, 2, 2, 0, 2, 2, 2, 2]))
    y = np.array([1., 5., 2., 1000., 4., 0., 7., 1., 1., 2.])

    def get_spks(selected_session):
        assert selected_session.params.spks_type == "sigrebase"
        return np.column_stack((y, y * 2))

    monkeypatch.setattr(panels.session_cache, "get_spks", get_spks)
    monkeypatch.setattr(panels.session_cache, "get_smp", lambda s: SimpleNamespace(
        dist_edges=np.arange(4.), params=SimpleNamespace(speed_threshold=1.)))
    monkeypatch.setattr(panels, "get_frame_behavior", lambda s: behavior)
    return results, session, behavior


def test_focus_uses_full_trial_gains_actual_time_and_common_vr_frames(monkeypatch):
    results, session, behavior = _focus_inputs(monkeypatch)
    viewer = panels.PlacefieldVarianceFocus(results, xslice=slice(0, 2))
    assert session.params.spks_type == "original"
    np.testing.assert_array_equal(viewer.frames, [0, 1, 2, 4, 6, 7, 8, 9])
    np.testing.assert_allclose(sum(viewer.components.values()), viewer.data)
    np.testing.assert_allclose(viewer.components["gain"][:3], np.array([1., 2., 3.]) * (3 / 14))
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 4
    assert fig.axes[2].get_xlabel() == "Time (s)"
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [0., 0.25])
    for name, values in viewer.components.items():
        assert viewer.display_variances[name] == pytest.approx(np.var(values[:2]))
    full = viewer.plot({**viewer.state, "xslice_stop": len(viewer.frames), "variance_scope": "session"})
    assert np.isnan(full.axes[0].lines[0].get_xdata()).sum() == 2
    for name, values in viewer.components.items():
        assert viewer.display_variances[name] == pytest.approx(np.var(values))

    viewer.refresh_data({**viewer.state, "all_environments": False, "env_slot": 1})
    np.testing.assert_array_equal(viewer.frames, [7, 8, 9])


def test_trial_focus_filters_sorts_real_roi_ids_and_uses_position_coordinates(monkeypatch):
    results, session, behavior = _focus_inputs(monkeypatch)
    viewer = panels.PlacefieldVarianceTrialFocus(results, xslice=slice(0, 1), column_ratio=3.)
    assert viewer.roi_options == [1, 0]
    assert viewer.state["roi"] == 1
    assert session.params.spks_type == "original"
    np.testing.assert_allclose(viewer.trial_maps["data"][0], [2., 10., 4.])
    np.testing.assert_allclose(sum(viewer.trial_maps[name] for name in panels.COMPONENTS), viewer.trial_maps["data"])
    np.testing.assert_allclose(viewer.components["gain"][:3], np.array([2., 4., 6.]) * (3 / 14))
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    axes = fig.axes
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), np.array([.5, 1.5, 2.5]) / 3)
    np.testing.assert_allclose(axes[2].get_xticks(), [.5])
    assert [tick.get_text() for tick in axes[2].get_xticklabels()] == ["0"]
    assert axes[2].get_xlabel() == "Trials"
    assert axes[0].get_ylabel() == axes[1].get_ylabel() == "Norm. Fluor."
    assert axes[0].get_legend().get_texts()[0].get_text() == "data"
    assert len(axes[1].get_legend().get_texts()) == 3
    assert axes[0].get_subplotspec().get_gridspec().get_width_ratios() == [3., 1]
    for name, data in viewer.components.items():
        assert viewer.display_variances[name] == pytest.approx(np.var(data[:3]))

    state = {**viewer.state, "sort_by": "residual", "roi": None}
    viewer.update_roi_selection(state)
    assert viewer.roi_options == [0, 1]
    assert viewer.state["roi"] == 0
    viewer.update_roi_selection({**viewer.state, "env_slot": 1})
    assert viewer.roi_options == [0]
    np.testing.assert_array_equal(viewer.frames, [7, 8, 9])
    viewer.update_roi_selection({**viewer.state, "reliability_threshold": 1.})
    assert viewer.state["roi"] is None
    fig = viewer.plot(viewer.state)
    assert "No ROIs" in fig.axes[0].texts[0].get_text()


def test_distribution_overview_weights_sessions_then_mice_not_cells():
    results = Results()
    values = results.arrays["pf_frac_var_template_cv_slot"]
    values[:3, 0] = [[1., np.nan, np.nan], [0., 0., 0.], [1., np.nan, np.nan]]
    values[3:, 0] = 1.
    viewer = panels.PlacefieldVarianceDistributions(results, nbins=5)
    data = viewer.distributions()["template"]
    width = np.diff(data["edges"])[0]
    np.testing.assert_allclose(data["overview"]["all"][0, [0, -1]], np.array([1 / 3, 2 / 3]) / width)
    np.testing.assert_allclose(data["overview"]["all"][1, [0, -1]], np.array([0., 1.]) / width)
    np.testing.assert_allclose(np.sum(data["overview"]["all"] * width, axis=1), 1.)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 6
    assert not any(ax.texts for ax in fig.axes)
    fig = viewer.plot({**viewer.state, "include_covariance": True, "plot_style": "each"})
    assert len(fig.axes) == 8
    assert all(ax.get_xlim() == (0., 1.) for ax in fig.axes)


def test_distributions_defaults_to_all_from_all_and_can_switch_to_cv():
    results = Results()
    results.arrays["pf_frac_var_template_all_from_all_slot"] = np.full((6, 2, 3), .123)
    viewer = panels.PlacefieldVarianceDistributions(results)
    np.testing.assert_allclose(viewer.arrays["template"], .123)
    assert all("all_from_all" in key for key in viewer._variance_metric_keys())
    viewer.set_parameter_value("comparison", "cv")
    viewer.reload_arrays(viewer.state)
    np.testing.assert_allclose(viewer.arrays["template"], .4)
    assert all("_cv_slot" in key for key in viewer._variance_metric_keys())
    familiarity = panels.PlacefieldVarianceFamiliarity(results)
    np.testing.assert_allclose(familiarity.arrays["template"], .4)


def test_trial_curves_join_observed_bins_but_variances_keep_temporal_weights(monkeypatch):
    results, session, behavior = _focus_inputs(monkeypatch)
    behavior.position[1] = .5  # Two temporal observations in the first bin; middle bin unobserved.
    viewer = panels.PlacefieldVarianceTrialFocus(results, xslice=slice(0, 1), legend_options={
        "ncols": 1, "x_offset": .02, "y_offset": -.02,
    })
    np.testing.assert_allclose(viewer.trial_maps["data"][0], [6., np.nan, 4.], equal_nan=True)
    fig = viewer.plot(viewer.state)
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [6., 4.])
    assert viewer.state["legend_x_offset"] == .02
    for name, data in viewer.components.items():
        assert viewer.display_variances[name] == pytest.approx(np.var(data[:3]))
    assert viewer.display_variances["template"] != np.nanvar(viewer.trial_maps["template"][0])


def test_distribution_selections_are_independent_and_nonquality_is_dotted():
    viewer = panels.PlacefieldVarianceDistributions(Results(), session_cells="quality", overview_cells="non-quality")
    data = viewer.distributions()["gain"]
    assert set(data["session"]) == {"quality"}
    assert set(data["overview"]) == {"non-quality"}
    fig = viewer.plot({**viewer.state, "session_cells": "split", "overview_cells": "split"})
    fig.canvas.draw()
    assert [patch.get_linestyle() for patch in fig.axes[0].patches] == ["-", ":"]
    assert fig.axes[1].lines[-1].get_linestyle() == ":"
    legend = fig.axes[0].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["Placecells", "Non-Placecells"]
    assert all(handle.get_color() == "k" for handle in legend.legend_handles)
    right = viewer.plot({**viewer.state, "overview_cells": "split"})
    assert right.axes[0].get_legend() is not None
    assert right.axes[1].get_legend() is not None
    assert [t.get_text() for t in right.axes[1].get_legend().get_texts()] == [
        "Placefield", "Gain", "Residual"]
    assert right.axes[1].get_legend().get_title().get_text() == "Component"
    assert right.axes[1].get_legend().get_title().get_color() == "black"
    assert "legend_axis" not in viewer.state
    for ax in right.axes[:-2]:
        assert ax.get_xlabel() == ""
        assert not ax.get_xticklabels()
        assert ax.get_xticks().size
        assert ax.spines["bottom"].get_visible()
    assert all(ax.get_xlabel() == "Variance Fraction" for ax in right.axes[-2:])
    each = viewer.plot({**viewer.state, "plot_style": "each"})
    assert len(each.axes[1].lines) > len(right.axes[1].lines)


def test_trial_boundaries_insert_nan_without_shifting_samples(monkeypatch):
    results, _, _ = _focus_inputs(monkeypatch)
    viewer = panels.PlacefieldVarianceTrialFocus(results, xslice=slice(0, 2), component_legend_horizontal=True)
    fig = viewer.plot(viewer.state)
    for ax in fig.axes[:3]:
        for line in ax.lines:
            x, y = line.get_xdata(), line.get_ydata()
            assert np.isnan(x).sum() == 1
            assert np.isnan(y).sum() == 1
            assert np.allclose(x[np.isfinite(x)] * 6 % 1, 0, atol=1e-12)
    assert fig.axes[1].get_legend()._ncols == 3


def test_all_environment_overview_averages_environments_before_sessions():
    results = Results()
    values = results.arrays["pf_frac_var_template_cv_slot"]
    values[:] = np.nan
    values[0, 0] = [0., 0., 0.]
    values[0, 1] = [1., np.nan, np.nan]
    viewer = panels.PlacefieldVarianceDistributions(results, overview_environment="all", nbins=5)
    data = viewer.distributions()["template"]
    width = np.diff(data["edges"])[0]
    np.testing.assert_allclose(data["overview"]["all"][0, [0, -1]], np.array([.5, .5]) / width)
    assert np.isnan(data["overview"]["all"][1]).all()
    assert viewer._overview_slots(1, viewer.state) == [0]  # Env 2 was not present on day 1.
