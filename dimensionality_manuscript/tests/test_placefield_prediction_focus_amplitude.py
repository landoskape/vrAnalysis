"""Focused tests for the optional trial-amplitude column in PlaceFieldPredictionFocus."""

from pathlib import Path
from types import SimpleNamespace
import sys
import types

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

# Avoid importing every figure-1 panel through figure1.__init__; some optional panels have heavy
# dependencies unrelated to this focused viewer test.
package_name = "dimensionality_manuscript.figure_scripts.figure1"
if package_name not in sys.modules:
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[1] / "figure_scripts" / "figure1")]
    sys.modules[package_name] = package

from dimensionality_manuscript.figure_scripts.figure1 import placefield
from dimensionality_manuscript.figure_scripts.figure1.placefield import PlaceFieldPredictionFocus


def _summary_title(axis):
    return next(artist for artist in axis.artists if artist.get_label().startswith(("SpRel\n=", "AmpRel\n=", "CoV\n=", "PF-Var Alignment\n=")))


def _summary_axis(fig, label):
    return next(axis for axis in fig.axes if any(artist.get_label().startswith(label) for artist in axis.artists))


def _assert_summary_layout(axis, renderer):
    title = _summary_title(axis)
    label, value = title.offsetbox.get_children()
    label_box = label.get_window_extent(renderer)
    value_box = value.get_window_extent(renderer)
    center = (axis.bbox.x0 + axis.bbox.x1) / 2
    assert (label_box.x0 + label_box.x1) / 2 == pytest.approx(center, abs=0.5)
    assert (value_box.x0 + value_box.x1) / 2 == pytest.approx(center, abs=0.5)
    assert value_box.y1 < label_box.y0
    assert title.get_window_extent(renderer).y0 > axis.bbox.y1


def _viewer(monkeypatch, *, include_amplitude_summary=True):
    session = SimpleNamespace(
        session_uid="M1/2020-01-01/1",
        mouse_name="M1",
        date="2020-01-01",
        session_id="1",
        params=SimpleNamespace(spks_type="oasis"),
        session_print=lambda: "M1/2020-01-01/1",
    )
    results = SimpleNamespace(
        mouse_names=np.array(["M1"]),
        sessions=[session],
        param_axes={},
    )
    from dimensionality_manuscript.configs.pfpred_quality import PFPredQualityConfig
    results.config_class = PFPredQualityConfig
    spkmap = np.array(
        [
            [
                [0.0, 1.0, 3.0, 1.0, 0.0],
                [0.0, 2.0, 6.0, 2.0, 0.0],
                [0.0, 0.5, 1.5, 0.5, 0.0],
                [1.0, 3.0, 4.0, 1.0, 0.0],
            ]
        ]
    )
    env_maps = SimpleNamespace(
        environments=np.array([7]),
        distcenters=np.linspace(0.0, 200.0, 5),
        spkmap=[spkmap],
    )
    measured = placefield._trial_activity_measurements(spkmap, env_maps.distcenters)
    summaries = placefield._summarize_trial_activity(measured)
    from dimensionality_manuscript.configs.pfpred_quality import _placefield_rms_maps
    spatial = _placefield_rms_maps(spkmap)
    precomputed = {
        "env_slot_ids": np.array([7.0]),
        "reliability_slot": np.array([[placefield._trial_consistency(spkmap[0])[3]]]),
        **{key: values[None, :] for key, values in summaries.items()},
        **{f"{key}_slot": values[None] for key, values in spatial.items()},
        "pf_rms_distcenters": env_maps.distcenters,
    }
    results.store = SimpleNamespace(get=lambda session_uid, config: precomputed)
    monkeypatch.setattr(placefield.session_cache, "get_env_maps", lambda selected_session: env_maps)
    return PlaceFieldPredictionFocus(
        results,
        mouse="M1",
        example_session="M1/2020-01-01/1",
        include_amplitude_summary=include_amplitude_summary,
        show_prediction=False,
    )


def test_amplitude_column_uses_v9_metrics_and_dropdown_defaults(monkeypatch):
    viewer = _viewer(monkeypatch)

    assert viewer.state["include_amplitude_summary"] is True
    assert viewer.state["amplitude_metric"] == "trial_rms"
    assert viewer.state["amplitude_transform"] == "raw"
    assert viewer.state["show_correlation"] is False
    assert viewer.state["row_ratio"] == 2.0
    np.testing.assert_allclose(
        viewer.trial_activity["trial_rms"],
        np.sqrt(np.mean(viewer.spkmap**2, axis=1)),
    )
    assert viewer.trial_activity_summary["trial_rms_mean"] == np.mean(viewer.trial_activity["trial_rms"])


def test_amplitude_column_draws_after_reliability_with_centered_cov(monkeypatch):
    viewer = _viewer(monkeypatch)
    state = {**viewer.state, "amplitude_metric": "trial_mean"}

    fig = viewer.plot(state)
    amplitude_axis = next(axis for axis in fig.axes if axis.get_title() == "Amplitude\nReliability")
    expected = viewer.trial_activity_summary["trial_mean_cv"]
    summary_axis = _summary_axis(fig, "CoV\n=")
    consistency_axis = next(axis for axis in fig.axes if axis.get_title() == "Spatial\nReliability")

    assert consistency_axis.get_position().x0 < amplitude_axis.get_position().x0
    assert consistency_axis.get_xlim() == (-1.1, 1.1)
    assert not amplitude_axis.spines["bottom"].get_visible()
    assert amplitude_axis.get_ylabel() == ""
    assert amplitude_axis.get_xlim() == (0.0, np.ceil(np.nanmax(viewer.trial_activity["trial_mean"])))
    assert len(amplitude_axis.get_xticks()) == 0
    assert summary_axis.get_xlabel() == "Trial\nMean"
    assert summary_axis.axison
    assert len(summary_axis.patches) == state["nbins"]
    assert not summary_axis.lines
    counts, _ = np.histogram(viewer.trial_activity["trial_mean"], bins=state["nbins"],
                             range=amplitude_axis.get_xlim())
    np.testing.assert_array_equal([patch.get_height() for patch in summary_axis.patches], counts)
    assert not summary_axis.texts
    text = _summary_title(summary_axis)
    assert text.get_label() == f"CoV\n={expected:.2f}"
    fig.canvas.draw()
    _assert_summary_layout(summary_axis, fig.canvas.get_renderer())


@pytest.mark.parametrize("transform,expected_text", [
    ("raw", "CoV\n=2.00"),
    ("inverse", "AmpRel\n=0.50"),
    ("bounded", "AmpRel\n=0.20"),
])
def test_cov_transform_changes_summary_without_transforming_trials(monkeypatch, transform, expected_text):
    viewer = _viewer(monkeypatch)
    viewer.trial_activity_summary["trial_rms_cv"] = 2.0
    values = viewer.trial_activity["trial_rms"].copy()
    fig = viewer.plot({**viewer.state, "amplitude_transform": transform})
    summary = _summary_axis(fig, expected_text)
    assert not summary.texts
    trials = next(axis for axis in fig.axes if axis.get_title() == "Amplitude\nReliability")
    np.testing.assert_allclose(trials.collections[0].get_offsets()[:, 0], values)
    counts, _ = np.histogram(values, bins=viewer.state["nbins"], range=trials.get_xlim())
    np.testing.assert_array_equal([patch.get_height() for patch in summary.patches], counts)
    assert viewer.trial_activity_summary["trial_rms_cv"] == 2.0


@pytest.mark.parametrize("transform,expected_text", [
    ("inverse", "AmpRel\n=nan"), ("bounded", "AmpRel\n=1.00"),
])
def test_zero_cov_transform_summary(monkeypatch, transform, expected_text):
    viewer = _viewer(monkeypatch)
    viewer.trial_activity_summary["trial_rms_cv"] = 0.0
    fig = viewer.plot({**viewer.state, "amplitude_transform": transform})
    assert _summary_axis(fig, expected_text) is not None


def test_reliability_histogram_uses_example_trials_not_session_rois(monkeypatch):
    viewer = _viewer(monkeypatch)
    viewer.session_reliability = np.full(100, -0.9)
    state = {**viewer.state, "nbins": 7}

    fig = viewer.plot(state)

    reliability_axis = next(axis for axis in fig.axes if axis.get_xlabel() == "Trial\nSimilarity")
    assert reliability_axis.get_xlim() == (-1.1, 1.1)
    assert len(reliability_axis.patches) == 7
    expected, _ = np.histogram(viewer.consistency[2], bins=7, range=(-1, 1))
    np.testing.assert_array_equal([patch.get_height() for patch in reliability_axis.patches], expected)
    assert not reliability_axis.lines
    assert not reliability_axis.texts
    text = _summary_title(reliability_axis)
    assert text.get_label() == f"SpRel\n={viewer.consistency[3]:.2f}"
    fig.canvas.draw()
    _assert_summary_layout(reliability_axis, fig.canvas.get_renderer())


def test_amplitude_column_is_absent_when_disabled(monkeypatch):
    viewer = _viewer(monkeypatch, include_amplitude_summary=False)

    fig = viewer.plot(viewer.state)

    assert not any(axis.get_title() == "Amplitude\nReliability" for axis in fig.axes)


@pytest.mark.parametrize("figsize,fontsize", [((12, 6), 12), ((3.5, 2.25), 7)])
@pytest.mark.parametrize("include_amplitude_summary", [False, True])
def test_overlay_and_optional_correlation_follow_comparison_without_extra_column(
    monkeypatch, figsize, fontsize, include_amplitude_summary
):
    viewer = _viewer(monkeypatch, include_amplitude_summary=include_amplitude_summary)
    viewer.figsize = figsize
    state = {**viewer.state, "pf_rms_comparison": "independent_f1", "show_correlation": True, "fontsize": fontsize}
    fig = viewer.plot(state)
    fig.canvas.draw()
    pf = viewer.session_pf_rms["pf_f1"][0]
    rms = viewer.session_pf_rms["rms_f2_from_f2"][0]
    r = viewer.session_pf_rms["pf_rms_r_independent_f1"][0]
    overlay = next(axis for axis in fig.axes if axis.get_ylabel() == "N.Fluor")
    np.testing.assert_allclose(overlay.lines[0].get_ydata(), pf)
    np.testing.assert_allclose(overlay.lines[1].get_ydata(), rms)
    assert sum(len(axis.images) for axis in fig.axes) == 1
    assert len(fig.axes) == (6 if include_amplitude_summary else 4)
    assert [text.get_text() for text in overlay.texts] == ["Pos Mean", "Pos STD"]
    title = overlay.artists[0]
    assert title.get_label() == f"PF-Var Alignment\n={r:.2f}"
    pieces = title.offsetbox.get_children()
    assert isinstance(pieces[0], placefield._UnderlinedTextArea)
    assert pieces[0].get_text() == "PF-Var Alignment"
    assert pieces[1].get_text() == f"=${r:.2f}$"
    _assert_summary_layout(overlay, fig.canvas.get_renderer())
    assert title.get_window_extent(fig.canvas.get_renderer()).y0 > overlay.bbox.y1
    assert overlay.get_xlabel() == "Position (cm)"
    ymax = overlay.get_ylim()[1]
    assert ymax == np.ceil(max(pf.max(), rms.max()) * 1.25)
    assert ymax == int(ymax)
    np.testing.assert_allclose(overlay.get_ylim(), [0, ymax])
    np.testing.assert_allclose(overlay.get_yticks(), [0, ymax])
    np.testing.assert_allclose(overlay.spines["left"].get_bounds(), [0, ymax])
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer) for text in overlay.texts]
    assert boxes[0].y0 > boxes[1].y1
    assert boxes[-1].y0 > overlay.bbox.y0
    assert overlay.get_yticklabels()[-1].get_text() == str(int(ymax))


def test_labels_and_trial_end_ticks_without_correlation(monkeypatch):
    viewer = _viewer(monkeypatch)
    fig = viewer.plot(viewer.state)
    overlay = next(axis for axis in fig.axes if axis.get_ylabel() == "N.Fluor")
    assert [text.get_text() for text in overlay.texts] == ["Pos Mean", "Pos STD"]
    ymax = np.ceil(max(np.max(line.get_ydata()) for line in overlay.lines) * 1.25)
    np.testing.assert_allclose(overlay.get_ylim(), [0, ymax])
    np.testing.assert_allclose(overlay.get_yticks(), [0, ymax])
    trials = next(axis for axis in fig.axes if axis.get_ylabel() == "Trials")
    assert trials.get_title() == "Placefield\nStructure"
    consistency = next(axis for axis in fig.axes if axis.get_title() == "Spatial\nReliability")
    assert not consistency.texts
    assert not overlay.artists
    np.testing.assert_array_equal(trials.get_yticks(), [0, viewer.spkmap.shape[0]])
    assert any(axis.get_ylabel() == "Norm. Fluor." for axis in trials.child_axes)
    for axis in fig.axes:
        if "Reliability" in axis.get_title():
            assert axis.title.get_ha() == "center"
            assert axis.title.get_rotation() == 0


def test_optional_prediction_has_no_pf_rms_comparison_column(monkeypatch):
    viewer = _viewer(monkeypatch)
    viewer.avg_prediction = np.full_like(viewer.avg_prediction, 100.0)
    fig = viewer.plot({**viewer.state, "show_prediction": True})
    assert len(fig.axes) == 8
    assert sum(len(axis.images) for axis in fig.axes) == 2
    overlay = next(axis for axis in fig.axes if axis.get_ylabel() == "N.Fluor")
    assert overlay.get_ylim()[1] == np.ceil(max(np.max(line.get_ydata()) for line in overlay.lines) * 1.25)
    prediction = next(axis for axis in fig.axes if any(text.get_text() == "PF Prediction" for text in axis.texts))
    assert prediction.get_ylim()[1] == 125.0


def test_focus_reuses_one_session_result_without_aggregating_maps(monkeypatch):
    viewer = _viewer(monkeypatch)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Changing ROI must reuse the cached session result")

    monkeypatch.setattr(viewer.results.store, "get", unexpected_load)
    viewer.refresh_data(viewer.state)
    assert viewer.session_pf_rms["pf_f1"].shape == (1, 5)


@pytest.mark.parametrize("row_ratio", [1.0, 4.0])
def test_row_ratio_controls_relative_panel_height(monkeypatch, row_ratio):
    viewer = _viewer(monkeypatch)
    fig = viewer.plot({**viewer.state, "row_ratio": row_ratio})
    fig.canvas.draw()
    top = next(axis for axis in fig.axes if axis.get_ylabel() == "Trials")
    bottom = next(axis for axis in fig.axes if axis.get_ylabel() == "N.Fluor")
    assert top.get_position().height / bottom.get_position().height == pytest.approx(row_ratio)


@pytest.mark.parametrize("dpi", [100, 200])
def test_peak_away_from_labels_does_not_inflate_correlation_headroom(monkeypatch, dpi):
    viewer = _viewer(monkeypatch)
    viewer.figsize = (3.5, 2.25)
    positions = np.linspace(0, 200, 201)
    viewer.pf_rms_distcenters = positions
    peak = 4.8 * np.exp(-((positions - 155) / 12)**2)
    viewer.session_pf_rms["pf_f1"] = peak[None, :]
    viewer.session_pf_rms["rms_f2_from_f1"] = (peak * 0.9)[None, :]
    with matplotlib.rc_context({"figure.dpi": dpi}):
        fig = viewer.plot({**viewer.state, "fontsize": 7, "show_correlation": True})
        overlay = next(axis for axis in fig.axes if axis.get_ylabel() == "N.Fluor")
        assert overlay.get_ylim() == (0, 6)


@pytest.mark.parametrize("dpi", [100, 200])
def test_lower_spines_and_label_tops_align_at_manuscript_size(monkeypatch, dpi):
    viewer = _viewer(monkeypatch)
    viewer.figsize = (3.5, 2.75)
    with matplotlib.rc_context({"figure.dpi": dpi}):
        fig = viewer.plot({**viewer.state, "fontsize": 7, "show_correlation": True, "row_ratio": 3})
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lower = [axis for axis in fig.axes if axis.get_subplotspec().rowspan.start == 1]
        assert [axis.get_xlabel() for axis in lower] == ["Position (cm)", "Trial\nSimilarity", "Trial\nRMS"]
        for axis in lower[1:]:
            _assert_summary_layout(axis, renderer)
        spine_y = [axis.spines["bottom"].get_transform().transform(axis.spines["bottom"].get_path().vertices)[0, 1]
                   for axis in lower]
        np.testing.assert_allclose(spine_y, spine_y[0], atol=0.5)
        label_tops = [axis.xaxis.label.get_window_extent(renderer).y1 for axis in lower]
        np.testing.assert_allclose(label_tops, label_tops[0], atol=0.5)
        left = [axis for axis in fig.axes if axis.get_ylabel() in ("Trials", "N.Fluor")]
        label_centers = [axis.yaxis.label.get_window_extent(renderer).x0 for axis in left]
        np.testing.assert_allclose(label_centers, label_centers[0], atol=0.5)
