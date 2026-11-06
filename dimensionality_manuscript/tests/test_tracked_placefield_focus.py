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
from dimensionality_manuscript.figure_scripts.figure1.tracked_placefield import TrackedPlaceFieldPredictionFocus


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
    assert example.viewer.parameters["candidate"].max == 2
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
    viewer.set_parameter_value("candidate", 2)
    viewer.set_parameter_value("fraction_active_threshold", 0.5)
    assert viewer.state["candidate"] == 0
    assert viewer.parameters["candidate"].max == 0
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
    assert "AmpRel\n=0.20" in summaries
    assert "AmpRel\n=1.00" in summaries
    overlays = [ax for ax in fig.axes if ax.get_ylabel() == "N.Fluor"]
    assert len(overlays) == 2
    for overlay, suffix in zip(overlays, ("early", "late")):
        if show_correlation:
            panel = viewer._focus_panels[suffix][1]
            r = panel.session_pf_rms[f"pf_rms_r_{viewer.state['pf_rms_comparison']}"][panel.state["roi"]]
            title, = overlay.artists
            assert title.get_label() == f"PF-Var Alignment\n={r:.2f}"
            label, value = title.offsetbox.get_children()
            assert isinstance(label, placefield._UnderlinedTextArea)
            renderer = fig.canvas.get_renderer()
            assert value.get_window_extent(renderer).y1 < label.get_window_extent(renderer).y0
            assert title.get_window_extent(renderer).y0 > overlay.bbox.y1
        else:
            assert not overlay.artists
    for title in ("Placefield\nStructure", "Spatial\nReliability", "Amplitude\nReliability"):
        assert titles.count(title) == 1
        assert title not in titles[6:]
    assert fig._suptitle is None
    assert not any(text.get_text().startswith(("Early:", "Late:")) for ax in fig.axes for text in ax.texts)
    assert len(fig.axes) == 12
    assert all(not ax.get_xlabel() for ax in fig.axes[:6])
    assert [ax.get_xlabel() for ax in fig.axes[6:] if ax.get_xlabel()] == [
        "Position (cm)", "Trial\nSimilarity", "Trial\nRMS",
    ]
    fig.savefig(tmp_path / "tracked_focus.png", dpi=100)
    # Moving rank must select the corresponding original neuron in both sessions.
    fig2 = viewer.plot({**viewer.state, "candidate": 1, "show_prediction": True})
    fig2.canvas.draw()
    image_axes = [ax for ax in fig2.axes if ax.images]
    assert len(image_axes) == 4
    np.testing.assert_allclose(image_axes[0].images[0].get_array(), example.maps[example.early.session_uid].spkmap[1][0])
    np.testing.assert_allclose(image_axes[2].images[0].get_array(), example.maps[example.late.session_uid].spkmap[0][1])
    assert fig2._suptitle is None
    assert all(not ax.get_xlabel() for ax in fig2.axes[:8])
    assert [ax.get_xlabel() for ax in fig2.axes[8:] if ax.get_xlabel()] == [
        "Position (cm)", "Trial\nSimilarity", "Trial\nRMS", "Position (cm)",
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
    assert viewer.state["candidate"] == 0


def test_missing_results_and_filter_mismatch_are_explicit(example):
    example.stored[example.early.session_uid]["trial_rms_cv"] = np.ones((2, 5))
    example.viewer._pair_cache = None
    with pytest.raises(ValueError, match="ROI filters"):
        example.viewer.candidate_table()
    example.stored[example.early.session_uid].pop("trial_rms_cv")
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
