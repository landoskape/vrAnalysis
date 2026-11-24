"""Tracked identity, paired selection, ranking, and stacked focus rendering."""

from pathlib import Path
from types import SimpleNamespace
import sys
import types

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt

package_name = "dimensionality_manuscript.figure_scripts.figure1"
if package_name not in sys.modules:
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[1] / "figure_scripts" / "figure1")]
    sys.modules[package_name] = package

from dimensionality_manuscript.configs.pfpred_quality import PFPredQualityConfig, _placefield_rms_maps
from dimensionality_manuscript.figure_scripts.figure1 import placefield, tracked_placefield
from dimensionality_manuscript.figure_scripts.figure1.tracked_placefield import (
    TrackedPlaceFieldPredictionFocus,
    TrackedPlacefieldBreakdownFocus,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def example(monkeypatch):
    def session(day, valid, environments):
        label = f"M1/2020-01-{day:02d}/1"
        return SimpleNamespace(
            session_uid=label, mouse_name="M1", date=f"2020-01-{day:02d}", session_id="1",
            session_print=lambda: label, idx_rois=np.isin(np.arange(10), valid),
            environments=np.array(environments), params=SimpleNamespace(spks_type="oasis"),
        )

    early = session(1, [1, 3, 4, 6, 7, 8], [3, 7])
    late = session(3, [0, 2, 4, 5, 7, 9], [7, 3])
    other = session(2, [0], [3])
    tracked = np.array([[3, 1, 4, 6, 7, 8, 0], [4, 2, 7, 0, 5, 9, 1]])
    calls = []

    def get_tracked_idx(*, idx_ses, use_session_filters):
        calls.append((idx_ses, use_session_filters))
        assert idx_ses == [2, 0]  # tracker list order differs from results/chronology
        assert use_session_filters is False
        return tracked, {"cluster_ids": np.arange(10, 80, 10)}

    tracker = SimpleNamespace(mouse_name="M1", sessions=[late, other, early], get_tracked_idx=get_tracked_idx)
    results = SimpleNamespace(
        mouse_names=np.array(["M1", "M1"]), sessions=[late, early],
        param_axes={}, config_class=PFPredQualityConfig,
    )
    stored, maps = {}, {}
    for session_index, s in enumerate((early, late)):
        shape = np.array([[0, 1, 3, 1, 0], [0, 2, 4, 1, 0], [0, 1, 2, 2, 0], [0, 1, 5, 1, 0]], dtype=float)
        spkmap = np.stack([shape * (i + 1 + session_index) for i in range(6)])
        # Environment 3 is deliberately different and in a different map/slot order.
        env_spkmaps = [spkmap if env == 7 else spkmap * 0.1 for env in s.environments]
        maps[s.session_uid] = SimpleNamespace(environments=s.environments, distcenters=np.linspace(0, 200, 5), spkmap=env_spkmaps)
        measured = placefield._trial_activity_measurements(spkmap, np.linspace(0, 200, 5))
        summaries = placefield._summarize_trial_activity(measured)
        spatial = _placefield_rms_maps(spkmap)
        values = {
            "env_slot_ids": s.environments.copy(),
            "reliability_slot": np.full((2, 6), 0.8),
            "fraction_active_slot": np.full((2, 6), 0.5),
            "pf_rms_distcenters": np.linspace(0, 200, 5),
            **{key: np.tile(value, (2, 1)) for key, value in summaries.items()},
            **{f"{key}_slot": np.stack([value, value]) for key, value in spatial.items()},
        }
        slot = list(s.environments).index(7)
        if s is early:
            values["trial_rms_cv"][slot] = [1, 2, 1, 1, np.nan, 0]
            values["reliability_slot"][slot, 2] = 0.3
        else:
            values["trial_rms_cv"][slot] = [1, 2, 0, 1, 1, 1]
            values["fraction_active_slot"][slot, 0] = 0.1
        values["trial_gain_least_squares_cv"] = values["trial_rms_cv"].copy()
        stored[s.session_uid] = values
    loads = []

    def get(uid, config):
        loads.append((uid, config))
        return stored[uid]

    results.store = SimpleNamespace(get=get)
    monkeypatch.setattr(placefield.session_cache, "get_env_maps", lambda s: maps[s.session_uid])
    monkeypatch.setattr(tracked_placefield, "Tracker", lambda mouse: tracker)
    msm = SimpleNamespace(tracker=tracker)
    viewer = TrackedPlaceFieldPredictionFocus(results, msm=msm, environment=7)
    return SimpleNamespace(viewer=viewer, results=results, msm=msm, early=early, late=late,
                           stored=stored, maps=maps, calls=calls, loads=loads)


def test_identity_mapping_and_both_session_quality(example):
    table = example.viewer.candidate_table()
    np.testing.assert_array_equal(table.cluster_id, [10, 20, 60])
    np.testing.assert_array_equal(table.original_roi_early, [3, 1, 8])
    np.testing.assert_array_equal(table.original_roi_late, [4, 2, 9])
    np.testing.assert_array_equal(table.roi_early, [1, 0, 5])
    np.testing.assert_array_equal(table.roi_late, [2, 1, 5])
    np.testing.assert_allclose(table.amplitude_change, [0.8, -0.3, -0.5])
    assert example.viewer.parameters["candidate"].options == ["ROI 1 → ROI 2", "ROI 0 → ROI 1", "ROI 5 → ROI 5"]
    assert example.viewer.state["candidate"] == "ROI 1 → ROI 2"
    assert example.early.params.spks_type == example.late.params.spks_type == "oasis"


def test_reverse_pair_keeps_chronology_and_reuses_pair_results(example):
    viewer = example.viewer
    state = {**viewer.state, "early_session": example.late.session_print(), "late_session": example.early.session_print()}
    np.testing.assert_array_equal(viewer.candidate_table(state).cluster_id, [10, 20, 60])
    assert len(example.calls) == 1
    assert len(example.loads) == 2


def test_sorting_thresholds_and_rank_bounds(example):
    viewer = example.viewer
    np.testing.assert_array_equal(viewer.candidate_table({**viewer.state, "descending": False}).cluster_id, [60, 20, 10])
    # Spatial-reliability sorting has deterministic ties.
    np.testing.assert_array_equal(viewer.candidate_table({**viewer.state, "sort_by": "spatial_change"}).cluster_id, [10, 20, 60])
    viewer.set_parameter_value("candidate", "ROI 5 → ROI 5")
    viewer.set_parameter_value("fraction_active_threshold", 0.5)
    assert viewer.state["candidate"] == "No matching ROIs"
    assert viewer.parameters["candidate"].options == ["No matching ROIs"]
    assert len(example.calls) == 1
    assert len(example.loads) == 2


@pytest.mark.parametrize("change", [{"fraction_active_threshold": 0.5}, {"early_session": "M1/2020-01-03/1"}])
def test_no_candidates_render_without_stale_neuron(example, change):
    viewer = example.viewer
    state = {**viewer.state, **change}
    assert viewer.candidate_table(state).empty
    fig = viewer.plot(state)
    assert not any(ax.images for ax in fig.axes)
    assert fig.axes[0].texts


@pytest.mark.parametrize("show_correlation", [False, True])
def test_stack_uses_same_cluster_correct_environment_and_bounded_summaries(example, tmp_path, show_correlation):
    viewer = example.viewer
    assert viewer.state["show_correlation"] is False
    viewer.set_parameter_value("show_correlation", show_correlation)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    images = [ax.images[0].get_array() for ax in fig.axes if ax.images]
    assert len(images) == 2
    np.testing.assert_allclose(images[0], example.maps[example.early.session_uid].spkmap[1][1])
    np.testing.assert_allclose(images[1], example.maps[example.late.session_uid].spkmap[0][2])
    titles = [ax.get_title() for ax in fig.axes]
    summaries = [artist.get_label() for ax in fig.axes for artist in ax.artists]
    assert "GR=0.20" in summaries
    assert "GR=1.00" in summaries
    overlays = [ax for ax in fig.axes if ax.get_ylabel() == "N.Fluor"]
    assert len(overlays) == 2
    for overlay, suffix in zip(overlays, ("early", "late")):
        if show_correlation:
            panel = viewer._focus_panels[suffix][1]
            r = panel.session_pf_rms[f"pf_rms_r_{viewer.state['pf_rms_comparison']}"][panel.state["roi"]]
            title, = overlay.artists
            assert title.get_label() == f"Mean\nSTD\nr(M,S)={r:.2f}"
            renderer = fig.canvas.get_renderer()
            box = title.get_window_extent(renderer)
            assert box.y1 <= overlay.bbox.y1
            assert box.y0 >= overlay.bbox.y0
        else:
            assert overlay.artists[0].get_label() == "Mean\nSTD"
    assert all(title == "" for title in titles)
    assert fig._suptitle is None
    assert not any(text.get_text().startswith(("Early:", "Late:")) for ax in fig.axes for text in ax.texts)
    assert len(fig.axes) == 12
    assert all(not ax.get_xlabel() for ax in fig.axes[:6])
    assert [ax.get_xlabel() for ax in fig.axes[6:] if ax.get_xlabel()] == [
        "Position (cm)", "Trial\nSimilarity", "Trial\nGain",
    ]
    fig.savefig(tmp_path / "tracked_focus.png", dpi=100)
    # Moving rank must select the corresponding original neuron in both sessions.
    fig2 = viewer.plot({**viewer.state, "candidate": "ROI 0 → ROI 1", "show_prediction": True})
    fig2.canvas.draw()
    image_axes = [ax for ax in fig2.axes if ax.images]
    assert len(image_axes) == 4
    np.testing.assert_allclose(image_axes[0].images[0].get_array(), example.maps[example.early.session_uid].spkmap[1][0])
    np.testing.assert_allclose(image_axes[2].images[0].get_array(), example.maps[example.late.session_uid].spkmap[0][1])
    assert fig2._suptitle is None
    assert all(not ax.get_xlabel() for ax in fig2.axes[:8])
    assert [ax.get_xlabel() for ax in fig2.axes[8:] if ax.get_xlabel()] == [
        "Position (cm)", "Trial\nSimilarity", "Trial\nGain", "Position (cm)",
    ]


def test_mapping_and_lazy_tracker_inputs(example):
    for kwargs in ({"msm": {"M1": example.msm}}, {}):
        viewer = TrackedPlaceFieldPredictionFocus(example.results, environment=7, **kwargs)
        np.testing.assert_array_equal(viewer.candidate_table().cluster_id, [10, 20, 60])


def test_environment_change_updates_session_options(example):
    viewer = example.viewer
    viewer.set_parameter_value("environment", 3)
    table = viewer.candidate_table()
    assert len(table) == 6
    assert viewer.state["candidate"] == "ROI 1 → ROI 2"


def test_missing_results_and_filter_mismatch_are_explicit(example):
    example.stored[example.early.session_uid]["trial_gain_least_squares_cv"] = np.ones((2, 5))
    example.viewer._pair_cache = None
    with pytest.raises(ValueError, match="ROI filters"):
        example.viewer.candidate_table()
    example.stored[example.early.session_uid].pop("trial_gain_least_squares_cv")
    with pytest.raises(KeyError, match="schema v11"):
        example.viewer.candidate_table()


def test_data_selection_refreshes_results_and_uses_selected_spike_type(example, monkeypatch):
    example.results.param_axes = {"spks_type": ["sigrebase", "oasis"]}
    seen = []

    def get_maps(session):
        seen.append(session.params.spks_type)
        return example.maps[session.session_uid]

    monkeypatch.setattr(placefield.session_cache, "get_env_maps", get_maps)
    viewer = TrackedPlaceFieldPredictionFocus(example.results, msm=example.msm, environment=7)
    viewer.plot(viewer.state)
    assert set(seen) == {"sigrebase"}
    seen.clear()
    viewer.set_parameter_value("spks_type", "oasis")
    viewer.plot(viewer.state)
    assert set(seen) == {"oasis"}
    assert all(config.spks_type == "oasis" for _, config in example.loads[-4:])
    assert example.early.params.spks_type == example.late.params.spks_type == "oasis"


def test_no_shared_clusters(example):
    example.msm.tracker.get_tracked_idx = lambda **kwargs: (
        np.empty((2, 0), dtype=int), {"cluster_ids": np.array([], dtype=int)},
    )
    example.viewer._pair_cache = None
    table = example.viewer.candidate_table()
    assert table.empty
    assert "amplitude_change" in table
    assert not any(ax.images for ax in example.viewer.plot(example.viewer.state).axes)


def test_quality_cutoffs_do_not_select_different_stored_results(example):
    example.results.param_axes = {"reliability_threshold": [0.9], "fraction_active_threshold": [0.2]}
    viewer = TrackedPlaceFieldPredictionFocus(
        example.results, msm=example.msm, environment=7,
        reliability_threshold=0.7, fraction_active_threshold=0.4,
    )
    assert len(viewer.candidate_table()) == 3
    viewer.plot(viewer.state)
    for _, config in example.loads[-4:]:
        assert config.reliability_threshold == 0.9
        assert config.fraction_active_threshold == 0.2


def test_tracked_layout_controls_reach_both_sessions(example):
    viewer = example.viewer
    viewer.set_parameter_value("bottom_text_location", "upper right")
    viewer.set_parameter_value("colorbar_width", 0.05)
    viewer.set_parameter_value("colorbar_height", 0.4)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    for ax in fig.axes:
        for inset in ax.child_axes:
            bounds = ax.transAxes.inverted().transform_bbox(inset.bbox).bounds
            np.testing.assert_allclose(bounds, [.225, .15, .05, .4])
        for text in ax.artists:
            box = text.get_window_extent(fig.canvas.get_renderer())
            assert box.x1 == pytest.approx(ax.bbox.x1 - .02 * ax.bbox.width)


def test_amplitude_defaults_and_controls_update_ranking_and_both_panels(example):
    viewer = example.viewer
    assert viewer.state["amplitude_metric"] == "trial_gain_least_squares"
    assert viewer.state["amplitude_statistic"] == "cv"
    assert viewer.state["amplitude_transform"] == "bounded"
    # Change measurement and statistic after the pair cache has already been populated.
    for session, values in ((example.early, [2, 4, 6, 8, 10, 12]),
                            (example.late, [10, 20, 30, 40, 50, 60])):
        slot = list(session.environments).index(7)
        example.stored[session.session_uid]["trial_mean_mean"][slot] = values
    viewer.set_parameter_value("candidate", "ROI 5 → ROI 5")
    viewer.set_parameter_value("amplitude_metric", "trial_mean")
    assert viewer.state["candidate"] == "ROI 5 → ROI 5"
    viewer.set_parameter_value("amplitude_statistic", "mean")
    viewer.set_parameter_value("amplitude_transform", "raw")
    table = viewer.candidate_table()
    np.testing.assert_array_equal(table.cluster_id, [60, 50, 10, 20])
    np.testing.assert_allclose(table.amplitude_change, [48, 30, 26, 18])
    assert len(example.calls) == 1
    fig = viewer.plot(viewer.state)
    summaries = [artist.get_label() for ax in fig.axes for artist in ax.artists]
    assert "Mean=12.00" in summaries
    assert "Mean=60.00" in summaries
    assert any(ax.get_xlabel() == "Trial\nMean" for ax in fig.axes)
    viewer.set_parameter_value("candidate", "ROI 1 → ROI 2")
    viewer.set_parameter_value("amplitude_transform", "inverse")
    assert viewer.state["candidate"] == "ROI 1 → ROI 2"
    table = viewer.candidate_table()
    expected_early = {20: 2, 10: 4, 50: 10, 60: 12}
    np.testing.assert_allclose(table.amplitude_reliability_early,
                               [1 / expected_early[cluster] for cluster in table.cluster_id])


def test_default_plot_uses_least_squares_gain(example):
    viewer = example.viewer
    fig = viewer.plot(viewer.state)
    amplitude_axes = [ax for ax in fig.axes if ax.collections and ax.get_xlim()[0] == 0]
    assert len(amplitude_axes) == 2
    for ax, suffix in zip(amplitude_axes, ("early", "late")):
        panel = viewer._focus_panels[suffix][1]
        values = panel.trial_activity["trial_gain_least_squares"]
        np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 0], values[np.isfinite(values)])
    assert any(ax.get_xlabel() == "Trial\nGain" for ax in fig.axes)


def test_roi_dropdown_follows_preference_and_preserves_selected_identity(example):
    viewer = example.viewer
    viewer.set_parameter_value("descending", False)
    assert viewer.parameters["candidate"].options == ["ROI 5 → ROI 5", "ROI 0 → ROI 1", "ROI 1 → ROI 2"]
    assert viewer.state["candidate"] == "ROI 1 → ROI 2"
    fig = viewer.plot(viewer.state)
    images = [ax.images[0].get_array() for ax in fig.axes if ax.images]
    np.testing.assert_allclose(images[0], example.maps[example.early.session_uid].spkmap[1][1])
    np.testing.assert_allclose(images[1], example.maps[example.late.session_uid].spkmap[0][2])


def test_constructor_accepts_roi_pair_selection(example):
    viewer = TrackedPlaceFieldPredictionFocus(
        example.results, msm=example.msm, environment=7, candidate="ROI 0 → ROI 1",
    )
    assert viewer.state["candidate"] == "ROI 0 → ROI 1"


def test_tracked_breakdown_draws_the_selected_pair_in_chronological_order(example):
    for stored in example.stored.values():
        # Ranking needs quality summaries; the decomposition is fit from local maps.
        for key in list(stored):
            if key not in {"env_slot_ids", "reliability_slot", "fraction_active_slot", "trial_gain_least_squares_cv"}:
                del stored[key]
    viewer = TrackedPlacefieldBreakdownFocus(
        example.results,
        msm=example.msm,
        environment=7,
        early_session=example.late.session_print(),
        late_session=example.early.session_print(),
        candidate="ROI 0 → ROI 1",
        include_spatial_reliability=True,
        include_trial_gain=True,
        include_residual_rms=True,
        residual_metric="unexplained_fraction",
    )
    assert viewer.candidate_table().cluster_id.tolist() == [10, 20, 60]
    viewer.set_parameter_value("descending", False)
    assert viewer.candidate_table().cluster_id.tolist() == [60, 20, 10]
    assert viewer.state["candidate"] == "ROI 0 → ROI 1"
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 12
    for axes, session, env_index, roi in (
        (fig.axes[:6], example.early, 1, 0),
        (fig.axes[6:], example.late, 0, 1),
    ):
        activity, prediction, residual = [ax.images[0].get_array() for ax in axes if ax.images]
        expected = example.maps[session.session_uid].spkmap[env_index][roi]
        template = expected.mean(axis=0)
        gains = (expected @ template) / (template @ template)
        np.testing.assert_allclose(activity, expected)
        np.testing.assert_allclose(prediction, gains[:, None] * template)
        np.testing.assert_allclose(residual, expected - prediction)
        np.testing.assert_allclose(axes[3].collections[0].get_offsets()[:, 0], gains)
        np.testing.assert_allclose(axes[5].collections[0].get_offsets()[:, 0],
                                   np.sum(residual**2, axis=1) / np.sum((expected - expected.mean(axis=1, keepdims=True))**2, axis=1))
    assert all(not ax.get_xlabel() for ax in fig.axes[:6])
    assert fig.axes[-1].get_xlabel() == r"$1-R^2$"

    # Removing the selected ROI pair still picks the first eligible one.
    example.stored[example.early.session_uid]["fraction_active_slot"][1, 0] = 0.2
    viewer.set_parameter_value("fraction_active_threshold", 0.3)
    assert viewer.state["candidate"] == "ROI 5 → ROI 5"
