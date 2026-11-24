"""Population covariance and shared-axis checks without session data access."""

from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure1 import PlacefieldActivityFocus
from dimensionality_manuscript.figure_scripts.figure1 import placefield_activity as module


def test_population_covariance_matches_numpy():
    rng = np.random.default_rng(21)
    pf, activity = rng.normal(size=(7, 4)), rng.normal(size=(7, 9))
    expected = np.cov(pf.T, activity.T)[:4, 4:]
    np.testing.assert_allclose(module._population_covariance(pf, activity), expected)
    permutation = rng.permutation(7)
    np.testing.assert_allclose(module._population_covariance(pf[permutation], activity[permutation]), expected)
    assert np.isnan(module._population_covariance(pf[:1], activity[:1])).all()


@pytest.fixture
def setup_viewer(monkeypatch):
    # Peaks sort as ROI 1, ROI 2, ROI 0 across the concatenated environments.
    maps = [np.array([[0., 0.], [4., 0.], [0., 3.]]),
            np.array([[0., 5.], [0., 0.], [0., 0.]])]
    activity = np.arange(18.).reshape(6, 3)
    sessions = [SimpleNamespace(session_uid=str(i), date=f"2024-01-0{i + 1}", session_id=i,
                               params=SimpleNamespace(spks_type="oasis"), env_length=[100],
                               timestamps=np.arange(6.),
                               session_print=lambda i=i: f"mouse/session{i}") for i in range(2)]
    monkeypatch.setattr(module.session_cache, "get_env_maps", lambda session: SimpleNamespace(
        spkmap=maps if session is sessions[0] else maps[:1],
        environments=[1, 3] if session is sessions[0] else [1], average_trials=lambda: None))
    monkeypatch.setattr(module.session_cache, "get_spks", lambda session: activity)
    monkeypatch.setattr(module.session_cache, "get_reliability", lambda session: SimpleNamespace(
        environments=[1, 3], values=np.array([[0.1, 0.9, 0.8], [0.95, 0.1, 0.1]])))
    monkeypatch.setattr(module.session_cache, "get_smp", lambda session: SimpleNamespace(
        dist_edges=np.arange(3), get_frame_behavior=lambda: (
            np.arange(6.) * 10, None, np.array([1, 1, -1, 3, 3, -1]), None)))
    return SimpleNamespace(mouse_names=["mouse", "mouse"], sessions=sessions), maps, activity


@pytest.mark.parametrize("transpose", [False, True])
def test_full_frames_shared_axes_and_color_limits(setup_viewer, transpose):
    results, maps, activity = setup_viewer
    viewer = PlacefieldActivityFocus(results, transpose_pf=transpose, xslice=slice(0, 6),
                                    vmax_pf=2, vmax_activity=4, vmax_covariance=8,
                                    show_position=True, show_scalebar=True, scalebar_seconds=1)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    pf_ax, act_ax, cov_ax, pos_ax = fig.axes
    expected_pf = np.concatenate(maps, axis=1)[[1, 2, 0]]
    np.testing.assert_array_equal(pf_ax.images[0].get_array(), expected_pf.T if transpose else expected_pf)
    np.testing.assert_array_equal(act_ax.images[0].get_array(), activity.T[[1, 2, 0]])
    np.testing.assert_allclose(cov_ax.images[0].get_array(), module._population_covariance(expected_pf, activity.T[[1, 2, 0]]))
    assert act_ax.get_shared_x_axes().joined(act_ax, cov_ax)
    assert act_ax.get_shared_x_axes().joined(act_ax, pos_ax)
    if transpose:
        assert pf_ax.get_shared_y_axes().joined(pf_ax, cov_ax)
    else:
        assert pf_ax.get_shared_y_axes().joined(pf_ax, act_ax)
    assert [ax.images[0].get_clim() for ax in (pf_ax, act_ax, cov_ax)] == [(0, 2), (0, 4), (-8, 8)]


def test_single_environment_filter_and_session_switch(setup_viewer):
    results, maps, activity = setup_viewer
    viewer = PlacefieldActivityFocus(results, all_environments=False, env_slot=1)
    np.testing.assert_array_equal(viewer._roi_indices, [0])
    np.testing.assert_array_equal(viewer._placefield, maps[1][:1])
    assert viewer._activity.shape == (1, 6)
    # Loading a session with fewer environments clamps the selector.
    viewer.update_selection("session", value="mouse/session1")
    viewer.load_session(viewer.state)
    assert viewer.state["env_slot"] == 0
    np.testing.assert_array_equal(viewer._roi_indices, [1, 2])
    viewer.update_float("reliability_threshold", value=1.0)
    viewer.refresh_data(viewer.state)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert not fig.axes[0].images


def test_sample_sort_applied_to_activity_covariance_and_position(setup_viewer, monkeypatch):
    results, maps, activity = setup_viewer
    order = np.array([5, 2, 0, 4, 1, 3])
    fit = Mock(return_value=SimpleNamespace(isort=order))
    monkeypatch.setattr(module, "Rastermap", lambda: SimpleNamespace(fit=fit))
    viewer = PlacefieldActivityFocus(results, sample_sort="rastermap", xslice=slice(1, 4),
                                    show_position=True, show_scalebar=True)
    fig = viewer.plot(viewer.state)
    pf_ax, act_ax, cov_ax, pos_ax = fig.axes
    selected = order[1:4]
    np.testing.assert_array_equal(act_ax.images[0].get_array(), activity.T[[1, 2, 0]][:, selected])
    np.testing.assert_allclose(cov_ax.images[0].get_array(),
                               module._population_covariance(viewer._placefield, viewer._activity[:, selected]))
    assert len(cov_ax.lines) == len(viewer._position_boundaries)  # No temporal scalebar.
    assert all(line.get_linestyle() == "None" for line in pos_ax.lines)
    viewer.plot(viewer.state)
    assert fit.call_count == 1


def test_roi_rastermap_order_is_shared(setup_viewer, monkeypatch):
    results, maps, activity = setup_viewer
    monkeypatch.setattr(module, "Rastermap", lambda: SimpleNamespace(
        fit=lambda data: SimpleNamespace(isort=np.array([2, 0, 1]))))
    viewer = PlacefieldActivityFocus(results, roi_sort="rastermap")
    np.testing.assert_array_equal(viewer._activity, activity.T[[2, 0, 1]])
    np.testing.assert_array_equal(viewer._placefield, np.concatenate(maps, axis=1)[[2, 0, 1]])


def test_transpose_toggle_preserves_activity_and_covariance(setup_viewer):
    results, _, _ = setup_viewer
    viewer = PlacefieldActivityFocus(results, show_position=True)
    state = dict(viewer.state)
    before = viewer.plot(state)
    before.canvas.draw()
    snapshots = [(np.array(ax.images[0].get_array()), ax.images[0].get_clim(),
                  ax.get_xlim(), ax.get_ylim()) for ax in before.axes[1:3]]
    after = viewer.plot({**state, "transpose_pf": True})
    after.canvas.draw()
    for ax, (data, clim, xlim, ylim) in zip(after.axes[1:3], snapshots):
        np.testing.assert_array_equal(ax.images[0].get_array(), data)
        assert ax.images[0].get_clim() == clim
        assert ax.get_xlim() == xlim
        assert ax.get_ylim() == ylim
