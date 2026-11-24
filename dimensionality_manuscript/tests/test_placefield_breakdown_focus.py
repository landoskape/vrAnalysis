"""The focus viewer shows an exact gain/template/additive decomposition in one row."""

from itertools import product

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from test_placefield_prediction_focus_amplitude import _viewer
from dimensionality_manuscript.figure_scripts.figure1.placefield_breakdown import PlacefieldBreakdownFocus


@pytest.fixture
def viewer(monkeypatch):
    original = _viewer(monkeypatch)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Breakdown should require only local maps")

    monkeypatch.setattr(original.results.store, "get", unexpected_read)
    instance = PlacefieldBreakdownFocus(original.results)
    yield instance
    from matplotlib import pyplot as plt
    plt.close("all")


def test_gain_prediction_and_residual_are_the_same_decomposition(viewer):
    template = viewer.spkmap.mean(axis=0)
    gain = (viewer.spkmap @ template) / (template @ template)
    np.testing.assert_allclose(viewer.placefield, template)
    np.testing.assert_allclose(viewer.trial_gain, gain)
    np.testing.assert_allclose(viewer.pred_spkmap, gain[:, None] * template)
    np.testing.assert_allclose(viewer.pred_spkmap + viewer.residual_spkmap, viewer.spkmap)
    np.testing.assert_allclose(viewer.residual_rms, np.sqrt(np.mean(viewer.residual_spkmap**2, axis=1)))
    # A least-squares remainder is orthogonal to the template, and stays signed.
    np.testing.assert_allclose(viewer.residual_spkmap @ template, 0, atol=1e-13)
    assert np.any(viewer.residual_spkmap < 0)
    assert np.any(viewer.residual_spkmap > 0)


@pytest.mark.parametrize("reliability,gain,rms", list(product([False, True], repeat=3)))
def test_optional_columns_follow_their_maps_in_one_row(viewer, reliability, gain, rms):
    state = {**viewer.state, "include_spatial_reliability": reliability,
             "include_trial_gain": gain, "include_residual_rms": rms}
    fig = viewer.plot(state)
    fig.canvas.draw()
    labels = ["Placefield map"]
    if reliability:
        labels.append("Spatial reliability")
    labels.append("Gain × PF")
    if gain:
        labels.append("Trial Gain")
    labels.append("Remaining additive component")
    if rms:
        labels.append("Resid.\nRMS")
    assert [ax.get_label() for ax in fig.axes] == labels
    assert all(ax.get_title() == "" for ax in fig.axes)
    assert len(fig.axes) == 3 + reliability + gain + rms
    assert all(ax.get_subplotspec().get_gridspec().nrows == 1 for ax in fig.axes)
    assert all(ax.get_ylim() == fig.axes[0].get_ylim() for ax in fig.axes)
    assert fig.axes[0].get_yticks().tolist() == [0, viewer.spkmap.shape[0]]
    assert all(not any(tick.get_visible() for tick in ax.get_yticklabels()) for ax in fig.axes[1:])
    images = [ax.images[0] for ax in fig.axes if ax.images]
    for im, expected in zip(images, [viewer.spkmap, viewer.pred_spkmap, viewer.residual_spkmap]):
        np.testing.assert_allclose(im.get_array(), expected)
    assert images[0].get_clim() == images[1].get_clim() == (0, state["vmax"])
    assert images[2].get_clim() == (-state["vmax"], state["vmax"])
    for ax in fig.axes:
        if ax.get_label() == "Spatial reliability":
            assert ax.get_xlim() == (-1.1, 1.1)
            assert ax.get_xlabel() == "Trial\nSim."
        if ax.get_label() in ("Trial Gain", "Resid.\nRMS"):
            expected = viewer.trial_gain if ax.get_label() == "Trial Gain" else viewer.residual_rms
            np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 0], expected)
            np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 1], np.arange(len(expected)))


def test_missing_bins_remain_missing_and_rms_uses_only_observed_bins(viewer, monkeypatch):
    maps = viewer._get_env_maps(viewer.state)
    maps.spkmap[0][0, 0, 1] = np.nan
    monkeypatch.setattr(viewer, "_get_env_maps", lambda state: maps)
    viewer.refresh_data(viewer.state)
    assert np.isnan(viewer.pred_spkmap[0, 1])
    assert np.isnan(viewer.residual_spkmap[0, 1])
    template = np.nanmean(viewer.spkmap, axis=0)
    valid = np.isfinite(viewer.spkmap[0])
    expected_gain = viewer.spkmap[0, valid] @ template[valid] / (template[valid] @ template[valid])
    assert viewer.trial_gain[0] == pytest.approx(expected_gain)
    np.testing.assert_allclose(viewer.residual_rms, np.sqrt(np.nanmean(viewer.residual_spkmap**2, axis=1)))


def test_unexplained_fraction_scatter_uses_each_trials_variance(viewer):
    viewer.spkmap = np.array([[0., 2., np.nan, 1., 1.], [0., 0., 0., 0., 0.]])
    viewer.residual_spkmap = np.array([[1., -1., np.nan, 1., -1.], [0., 0., 0., 0., 0.]])
    fig = viewer.plot({**viewer.state, "include_residual_rms": True, "residual_metric": "unexplained_fraction"})
    axis = fig.axes[-1]
    assert axis.get_xlabel() == r"$1-R^2$"
    assert axis.get_xlim() == (-0.05, 1.05)
    assert axis.spines["bottom"].get_bounds() == (0, 1)
    np.testing.assert_array_equal(axis.get_xticks(), [0, 1])
    # First trial has SSE 4 and SST 2. The constant trial is undefined.
    np.testing.assert_allclose(axis.collections[0].get_offsets(), [[2., 0.]])


def test_silent_cell_has_undefined_gain_and_empty_reliability(viewer, monkeypatch):
    maps = viewer._get_env_maps(viewer.state)
    maps.spkmap[0][:] = 0
    monkeypatch.setattr(viewer, "_get_env_maps", lambda state: maps)
    viewer.refresh_data(viewer.state)
    assert np.all(np.isnan(viewer.trial_gain))
    assert np.all(np.isnan(viewer.residual_rms))
    fig = viewer.plot({**viewer.state, "include_spatial_reliability": True,
                       "include_trial_gain": True, "include_residual_rms": True})
    fig.canvas.draw()
    assert len(fig.axes) == 6


def test_refresh_reloads_selected_roi_and_environment(viewer, monkeypatch):
    maps = viewer._get_env_maps(viewer.state)
    from types import SimpleNamespace
    selected = np.array([[1., 2., 4., 2., 1.], [2., 4., 8., 4., 2.]])
    maps = SimpleNamespace(distcenters=maps.distcenters, spkmap=[maps.spkmap[0], np.stack([selected * 3, selected])])
    monkeypatch.setattr(viewer, "_get_env_maps", lambda state: maps)
    viewer.refresh_data({**viewer.state, "env": 1, "roi": 1})
    np.testing.assert_allclose(viewer.spkmap, selected)
    np.testing.assert_allclose(viewer.residual_spkmap, 0, atol=1e-14)


@pytest.mark.parametrize("summary_width,support,upper", [(0.0, 199.5, 200), (0.3, 200, 200), (1.0, 213, 250)])
def test_layout_controls_and_position_bounds(viewer, summary_width, support, upper):
    viewer.distcenters = np.linspace(0.5, support, viewer.spkmap.shape[1])
    colorbar_bounds = (0.4, 0.25, 0.08, 0.45)
    state = {**viewer.state, "summary_width": summary_width, "markersize": 9.0,
             "include_spatial_reliability": True, "include_trial_gain": True,
             "include_residual_rms": True, "legend_x": 0.9, "legend_y": 0.15,
             "legend_ha": "right", "legend_va": "bottom",
             **dict(zip(("colorbar_x", "colorbar_y", "colorbar_width", "colorbar_height"), colorbar_bounds))}
    with matplotlib.rc_context({"axes.facecolor": "0.9"}):
        fig = viewer.plot(state)
    fig.canvas.draw()
    assert len(fig.axes) == (3 if summary_width == 0 else 6)
    map_axes = [ax for ax in fig.axes if ax.images]
    for ax in map_axes:
        assert ax.get_xlim() == (0, upper)
        np.testing.assert_array_equal(ax.get_xticks(), [0, upper])
        # Expanded display bounds must not stretch the image's actual support.
        np.testing.assert_allclose(ax.images[0].get_extent()[:2], [0.5, support])
    for ax in (map_axes[0], map_axes[-1]):
        inset, = ax.child_axes
        np.testing.assert_allclose(ax.transAxes.inverted().transform_bbox(inset.bbox).bounds, colorbar_bounds)
    for ax, text, color in zip(map_axes, ["Placefield", "Gain x PF\nModel", "Gain x PF\nResidual"],
                               ["#666666", "#c00000", "#c06000"]):
        legend, = ax.texts
        assert legend.get_text() == text
        assert legend.get_color() == color
        assert legend.get_position() == (0.9, 0.15)
        assert legend.get_transform() == ax.transAxes
        assert legend.get_ha() == "right"
        assert legend.get_va() == "bottom"
        assert ax.get_title() == ""
    assert map_axes[-1].get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    residual_image, = map_axes[-1].images
    np.testing.assert_allclose(residual_image.cmap(residual_image.norm(0.0)), [1, 1, 1, 1])
    for ax in fig.axes:
        if ax.collections:
            np.testing.assert_allclose(ax.collections[0].get_sizes(), [9.0])
            assert ax.get_position().width / map_axes[0].get_position().width == pytest.approx(summary_width)
