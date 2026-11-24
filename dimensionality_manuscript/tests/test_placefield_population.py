"""Population viewer weighting, neuron pairing, and manuscript layout checks."""

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

from dimensionality_manuscript.figure_scripts.figure1.placefield import (
    PlaceFieldPopulationDistributions,
    PlaceFieldReliabilityRelationships,
    _finite_column_summary,
)


def _results():
    """Two unequal-sized sessions for A, one for B, and one unusable best slot."""
    r = np.array([
        [-0.8, -0.6, -0.4, -0.2, np.nan, np.nan],
        [0.2, 0.4, np.nan, np.nan, np.nan, np.nan],
        [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        [0.5] * 6,
    ])
    spatial = r.copy()
    amplitude = 1 - r
    amplitude[2, -1] = 9  # Overflow belongs in the last amplitude bin.
    arrays = {"best_env_slot": np.array([1, 0, 1, np.nan])}
    for key, values in {
        "pf_rms_r_cv_slot": r,
        "pf_rms_r_all_from_all_slot": r * -1,
        "pf_rms_r_independent_cv_slot": r * 0.5,
        "reliability_slot": spatial,
        "trial_rms_cv": amplitude,
        "trial_mean_variance": amplitude * 2,
        "fraction_active_slot": np.where(np.isfinite(r), 0.8, np.nan),
    }.items():
        # Non-selected slots contain sentinel values to catch pooling of environments.
        packed = np.full((4, 2, 6), 0.99)
        for row, slot in enumerate([1, 0, 1, 0]):
            packed[row, slot] = values[row]
        arrays[key] = packed
    calls = []

    def sel(*, keys, squeeze_ones, **kwargs):
        calls.append(keys)
        return {key: arrays[key] for key in keys}

    sessions = [SimpleNamespace(session_uid=f"session-{row}", environments=np.array([20, 10]), date=f"2023-01-{row + 1:02d}", session_id=1,
                                session_print=lambda row=row, mouse=mouse: f"{mouse}/2023-01-{row + 1:02d}/1",
                                params=SimpleNamespace(spks_type="original"))
                for row, mouse in enumerate(["A", "A", "B", "B"])]
    arrays["env_slot_ids"] = np.tile([10, 20], (4, 1))
    store_calls = []

    def get(session_uid, config):
        store_calls.append((session_uid, config.key()))
        row = next(row for row, session in enumerate(sessions) if session.session_uid == session_uid)
        return {key: values[row] for key, values in arrays.items()}

    return SimpleNamespace(mouse_names=np.array(["A", "A", "B", "B"]), param_axes={}, sel=sel,
                           sessions=sessions, store=SimpleNamespace(get=get, calls=store_calls),
                           config_class=lambda **kwargs: SimpleNamespace(spks_type="selected", key=lambda: tuple(sorted(kwargs.items())))), arrays, calls


def test_distributions_average_sessions_then_mice_and_integrate_to_one():
    results, _, _ = _results()
    viewer = PlaceFieldPopulationDistributions(results, nbins=2)
    edges, curves = viewer.distributions(viewer.state)["correlation"]
    # Mouse A gives its 4-neuron negative and 2-neuron positive sessions equal weight.
    np.testing.assert_allclose(edges, [-1, 0, 1])
    np.testing.assert_allclose(curves, [[0.5, 0.5], [0, 1]])
    np.testing.assert_allclose(np.sum(curves * np.diff(edges), axis=1), 1)
    mean, std, count = _finite_column_summary(curves)
    np.testing.assert_allclose(mean, [0.25, 0.75])
    np.testing.assert_allclose(std, np.sqrt(0.125))  # STD, not SEM (0.25).
    np.testing.assert_array_equal(count, [2, 2])


def test_selection_reloads_only_selected_summary_arrays_and_activity_filter_is_live():
    results, arrays, calls = _results()
    viewer = PlaceFieldPopulationDistributions(results, nbins=2)
    viewer.distributions(viewer.state)
    viewer.distributions(viewer.state)
    assert len(calls) == 1
    assert not any("distcenters" in key or key.startswith("pf_f") for key in calls[0])
    state = {**viewer.state, "comparison": "shared", "amplitude_metric": "trial_mean", "amplitude_statistic": "variance"}
    values = viewer._population_sessions(state)
    np.testing.assert_allclose(values[0][1]["correlation"][:4], [0.8, 0.6, 0.4, 0.2])
    np.testing.assert_allclose(values[0][1]["amplitude"][:4], [3.6, 3.2, 2.8, 2.4])
    assert "pf_rms_r_all_from_all_slot" in calls[-1]
    assert "trial_mean_variance" in calls[-1]
    empty = viewer.distributions({**state, "fraction_active_threshold": 0.9})
    assert all(np.isnan(curves).all() for _, curves in empty.values())
    assert len(calls) == 2  # Filtering never reloads stored arrays.


def test_relationship_pairing_best_slot_and_mouse_level_associations():
    results, _, _ = _results()
    viewer = PlaceFieldReliabilityRelationships(results, nbins=2, min_cells=2, min_mice=1)
    relationships = viewer.relationships(viewer.state)
    edges, curves, rho = relationships["spatial"]
    np.testing.assert_allclose(curves[0], [-0.5, 0.3])
    np.testing.assert_allclose(curves[1], [np.nan, 3.5 / 6], equal_nan=True)
    np.testing.assert_allclose(rho, [1, 1])
    # A has a perfectly inverse amplitude relationship in both sessions.
    np.testing.assert_allclose(relationships["amplitude"][2][0], -1)
    assert np.isposinf(relationships["amplitude"][0][-1])
    # A single overflow neuron does not pass the min_cells criterion.
    assert np.isnan(relationships["amplitude"][1][1, -1])


def test_missing_pairs_and_constant_inputs_do_not_create_associations():
    results, arrays, _ = _results()
    arrays["reliability_slot"][:] = 0.5
    arrays["trial_rms_cv"][:] = np.nan
    viewer = PlaceFieldReliabilityRelationships(results, min_cells=2, min_mice=1)
    relationships = viewer.relationships(viewer.state)
    assert np.isnan(relationships["spatial"][2]).all()
    assert np.isnan(relationships["amplitude"][2]).all()
    assert np.isnan(relationships["amplitude"][1]).all()


def test_padding_is_trimmed_without_changing_neuron_pairing():
    results, arrays, _ = _results()
    arrays["trial_rms_cv"] = np.pad(arrays["trial_rms_cv"], ((0, 0), (0, 0), (0, 2)), constant_values=np.nan)
    viewer = PlaceFieldPopulationDistributions(results)
    assert viewer._population_sessions(viewer.state)[0][1]["amplitude"].size == 6
    arrays["trial_rms_cv"][..., -1] = 1
    viewer._population_cache = None
    with pytest.raises(ValueError, match="mismatched ROI widths"):
        viewer._population_sessions(viewer.state)


@pytest.mark.parametrize("association", [False, True])
@pytest.mark.parametrize("row_ratio", [1.0, 2.5, 5.0])
def test_relationship_layout_and_mouse_histograms(association, row_ratio):
    results, _, _ = _results()
    viewer = PlaceFieldReliabilityRelationships(results, nbins=4, min_cells=2, min_mice=1,
                                               row_ratio=row_ratio, figsize=(5.2, 4.0))
    fig = viewer.plot({**viewer.state, "include_association_strength": association})
    fig.canvas.draw()
    np.testing.assert_allclose(fig.get_size_inches(), viewer.figsize)
    assert len(fig.axes) == (4 if association else 2)
    if association:
        assert fig.axes[0].get_position().height / fig.axes[2].get_position().height == pytest.approx(row_ratio)
    assert all(ax.get_title() == "" for ax in fig.axes)
    assert fig.axes[1].get_xlabel() == "Amplitude Reliability"
    assert fig.axes[0].get_ylabel() == "PF-Var Alignment"
    for ax in fig.axes:
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["bottom"].get_position()[0] == "data"
    if association:
        for ax in fig.axes[2:]:
            assert sum(patch.get_height() for patch in ax.patches) == 2
            assert ax.get_xlabel() == "Spearman ρ"
        assert fig.axes[2].get_ylabel() == "Mice"


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_distribution_layout_and_band_uses_mouse_std(orientation):
    results, _, _ = _results()
    figsize = (2.5, 7.5) if orientation == "vertical" else (7.5, 2.5)
    viewer = PlaceFieldPopulationDistributions(results, nbins=2, orientation=orientation, figsize=figsize)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert len(fig.axes) == 3
    assert [ax.get_xlabel() for ax in fig.axes] == ["r(M,S)", "Spatial\nReliability", "Amplitude\nReliability"]
    assert fig.axes[0].get_ylabel() == "prob. density"
    np.testing.assert_allclose(fig.get_size_inches(), figsize)
    if orientation == "vertical":
        assert all(ax.get_ylabel() == "prob. density" for ax in fig.axes)
        assert fig.axes[0].get_position().y0 > fig.axes[1].get_position().y1
        assert fig.axes[1].get_position().y0 > fig.axes[2].get_position().y1
    else:
        assert fig.axes[0].get_position().x1 < fig.axes[1].get_position().x0
        assert fig.axes[1].get_position().x1 < fig.axes[2].get_position().x0
    for ax in fig.axes:
        assert ax.get_title() == ""
        ticks = ax.get_yticks()
        assert len(ticks) == 2 and ticks[0] == 0 and ticks[1] >= 1
        assert ticks[1] == np.ceil(ticks[1])
    np.testing.assert_allclose(fig.axes[0].lines[2].get_ydata(), [0.25, 0.75])
    vertices = fig.axes[0].collections[0].get_paths()[0].vertices
    assert np.max(vertices[:, 1]) == pytest.approx(0.75 + np.sqrt(0.125))
    assert np.min(vertices[:, 1]) == pytest.approx(0.25 - np.sqrt(0.125))


def test_reciprocal_distributions_transform_neurons_before_histogramming_and_preserve_cache():
    results, arrays, calls = _results()
    arrays["trial_rms_cv"][0, 1] = [0.25, 0.5, 1, 2, 0, np.inf]
    arrays["trial_rms_cv"][2, 1] = 1
    arrays["fraction_active_slot"][0, 1] = 0.8
    viewer = PlaceFieldPopulationDistributions(results, amplitude_transform="inverse", nbins=2)
    selected = viewer._selected_sessions(viewer.state)
    np.testing.assert_allclose(selected[0][1]["amplitude"], [4, 2, 1, 0.5, np.nan, np.nan], equal_nan=True)
    edges, curves = viewer.distributions(viewer.state)["amplitude"]
    np.testing.assert_allclose(edges, [0, 2, 4])
    # First session: half of the four valid reciprocal values in each bin.
    # Second session: both reciprocal values in the first bin. Equal-session mean:
    np.testing.assert_allclose(curves[0], [0.375, 0.125])
    raw_state = {**viewer.state, "amplitude_transform": "raw"}
    raw = viewer._selected_sessions(raw_state)[0][1]["amplitude"]
    np.testing.assert_allclose(raw, [0.25, 0.5, 1, 2, 0, np.inf])
    assert len(calls) == 1
    np.testing.assert_allclose(
        viewer.distributions(viewer.state)["spatial"][1], viewer.distributions(raw_state)["spatial"][1], equal_nan=True
    )


def test_association_recomputed_after_reciprocal_and_before_mouse_averaging():
    results, arrays, _ = _results()
    # Across zero, inversion is not a simple reversal of the original ranking.
    arrays["trial_rms_cv"][0, 1] = [-2, -1, 1, 2, np.nan, np.nan]
    arrays["trial_rms_cv"][1, 0] = [1, 2, np.nan, np.nan, np.nan, np.nan]
    viewer = PlaceFieldReliabilityRelationships(results, amplitude_transform="inverse", min_cells=2, min_mice=1)
    inverted = viewer.relationships(viewer.state)
    raw = viewer.relationships({**viewer.state, "amplitude_transform": "raw"})
    # Session A1: Spearman(-.8,-.6,-.4,-.2 vs -.5,-1,1,.5) = .6.
    # Session A2: Spearman(.2,.4 vs 1,.5) = -1. Mouse A: (.6-1)/2 = -.2.
    assert inverted["amplitude"][2][0] == pytest.approx(-0.2)
    assert raw["amplitude"][2][0] == pytest.approx(1)
    np.testing.assert_allclose(inverted["spatial"][2], raw["spatial"][2])


@pytest.mark.parametrize("kind", [PlaceFieldPopulationDistributions, PlaceFieldReliabilityRelationships])
def test_inverted_amplitude_labels_and_association_histogram(kind):
    results, _, _ = _results()
    kwargs = {} if kind is PlaceFieldPopulationDistributions else {
        "include_association_strength": True, "min_cells": 2, "min_mice": 1,
    }
    viewer = kind(results, amplitude_transform="inverse", **kwargs)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    amplitude_axis = fig.axes[2] if kind is PlaceFieldPopulationDistributions else fig.axes[1]
    expected = "Amplitude\nReliability" if kind is PlaceFieldPopulationDistributions else "Amplitude Reliability"
    assert amplitude_axis.get_xlabel() == expected
    if kind is PlaceFieldReliabilityRelationships:
        assert fig.axes[3].get_xlabel() == "Spearman ρ"
        # Positive raw inputs reverse rank on inversion; mouse A's rho moves to +1.
        last_bar = fig.axes[3].patches[-1]
        assert last_bar.get_height() == 1


def test_bounded_transform_handles_zero_large_values_and_recomputes_association():
    results, arrays, calls = _results()
    arrays["trial_rms_cv"][0, 1] = [0, 1, 2, 1e300, np.nan, np.inf]
    arrays["fraction_active_slot"][0, 1] = 0.8
    viewer = PlaceFieldReliabilityRelationships(results, amplitude_transform="bounded", min_cells=2, min_mice=1, nbins=2)
    with np.errstate(all="raise"):
        selected = viewer._selected_sessions(viewer.state)[0][1]["amplitude"]
    np.testing.assert_allclose(selected, [1, 0.5, 0.2, 0, np.nan, np.nan], equal_nan=True)
    edges, curves, association = viewer.relationships(viewer.state)["amplitude"]
    np.testing.assert_allclose(edges, [0, 0.5, 1])
    # Both finite values in each bin are counted, including the upper endpoint 1.
    np.testing.assert_allclose(curves[0], [-0.3, -0.2])
    assert association[0] == pytest.approx(0)  # A1: -1, A2: +1, averaged within mouse.
    raw = viewer._selected_sessions({**viewer.state, "amplitude_transform": "raw"})[0][1]["amplitude"]
    np.testing.assert_allclose(raw, [0, 1, 2, 1e300, np.nan, np.inf])
    assert len(calls) == 1


@pytest.mark.parametrize("kind", [PlaceFieldPopulationDistributions, PlaceFieldReliabilityRelationships])
def test_bounded_amplitude_axes_span_zero_to_one(kind):
    results, _, _ = _results()
    viewer = kind(results, amplitude_transform="bounded")
    assert "invert_amplitude" not in viewer.state
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    ax = fig.axes[2] if kind is PlaceFieldPopulationDistributions else fig.axes[1]
    np.testing.assert_allclose(ax.spines["bottom"].get_bounds(), [0, 1])
    expected = "Amplitude\nReliability" if kind is PlaceFieldPopulationDistributions else "Amplitude Reliability"
    assert ax.get_xlabel() == expected
    if kind is PlaceFieldReliabilityRelationships:
        np.testing.assert_array_equal(ax.get_xticks(), [0, 1])
    if kind is PlaceFieldPopulationDistributions:
        edges, densities = viewer.distributions(viewer.state)["amplitude"]
        np.testing.assert_allclose(np.sum(densities * np.diff(edges), axis=1), 1)


def test_session_histograms_keep_each_session_and_preserve_summary_curves():
    results, arrays, _ = _results()
    arrays["reliability_slot"][1, 0] *= -1
    viewer = PlaceFieldReliabilityRelationships(results, min_cells=2, min_mice=1, nbins=4,
                                               include_association_strength=True)
    averaged = viewer.relationships(viewer.state)
    state = {**viewer.state, "association_level": "all_sessions"}
    sessions = viewer.relationships(state)
    for predictor in averaged:
        np.testing.assert_allclose(sessions[predictor][1], averaged[predictor][1], equal_nan=True)
        assert sessions[predictor][2].size == 3
        assert np.isfinite(averaged[predictor][2]).sum() == 2
    np.testing.assert_allclose(averaged["spatial"][2], [0, 1], atol=1e-12)
    np.testing.assert_allclose(sessions["spatial"][2], [1, -1, 1])
    fig = viewer.plot(state)
    for ax in fig.axes[2:]:
        assert sum(patch.get_height() for patch in ax.patches) == 3
    assert fig.axes[2].get_ylabel() == "Sessions"


@pytest.mark.parametrize("display", ["example", "summary", "both"])
@pytest.mark.parametrize("histogram", [False, True])
def test_relationship_display_rows_and_exact_example_selection(monkeypatch, display, histogram):
    results, arrays, _ = _results()
    from dimensionality_manuscript.figure_scripts.figure1 import placefield

    def get_env_maps(session):
        pytest.fail("Example lookup must not build activity maps")

    monkeypatch.setattr(placefield.session_cache, "get_env_maps", get_env_maps)
    viewer = PlaceFieldReliabilityRelationships(
        results, mouse="A", example_session="A/2023-01-01/1", env=0,
        display=display, include_association_strength=histogram, row_ratio=3.0,
        min_cells=2, min_mice=1, figsize=(6, 7), amplitude_transform="bounded",
    )
    example = viewer.example_data(viewer.state)
    np.testing.assert_allclose(example["correlation"], [-0.8, -0.6, -0.4, -0.2])
    assert results.sessions[0].params.spks_type == "original"
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    ndata = 2 if display == "both" else 1
    assert len(fig.axes) == 2 * (ndata + histogram)
    np.testing.assert_allclose(fig.get_size_inches(), (6, 7))
    np.testing.assert_allclose(fig.axes[0].get_gridspec().get_height_ratios(),
                               [3.0] * ndata + ([1] if histogram else []))
    if display != "summary":
        np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets(),
                                   np.column_stack((example["spatial"], example["correlation"])))
        assert fig.axes[0].get_ylabel() == "PF-Var Alignment"
        np.testing.assert_array_equal(fig.axes[1].get_xticks(), [0, 1])
    for ax in fig.axes[:2]:
        assert ax.xaxis.get_visible() == (display != "both")
        assert ax.spines["bottom"].get_visible() == (display != "both")
    if display == "both":
        for ax in fig.axes[2:4]:
            assert ax.xaxis.get_visible()
            assert ax.spines["bottom"].get_visible()
    # env follows the session environment list, not the stored experience-order slots.
    other = viewer.example_data({**viewer.state, "env": 1})
    np.testing.assert_allclose(other["correlation"], 0.99)
    viewer.update_example_session({**viewer.state, "mouse": "B"})
    assert viewer.state["example_session"] == "B/2023-01-03/1"


def test_example_filter_transforms_missing_pairs_and_cache(monkeypatch):
    results, arrays, calls = _results()
    from dimensionality_manuscript.figure_scripts.figure1 import placefield
    monkeypatch.setattr(placefield.session_cache, "get_env_maps",
                        lambda session: pytest.fail("Example lookup must not build activity maps"))
    arrays["trial_rms_cv"][0, 1, :4] = [0, 2, np.inf, 4]
    viewer = PlaceFieldReliabilityRelationships(results, display="example", amplitude_transform="inverse")
    data = viewer.example_data(viewer.state)
    np.testing.assert_allclose(data["amplitude"], [np.nan, 0.5, np.nan, 0.25], equal_nan=True)
    viewer.example_data({**viewer.state, "env": 1})
    assert len(results.store.calls) == 1
    assert calls == []
    fig = viewer.plot(viewer.state)
    assert len(fig.axes[1].collections[0].get_offsets()) == 2
    data = viewer.example_data({**viewer.state, "fraction_active_threshold": 0.9})
    assert data["correlation"].size == 0
    assert len(results.store.calls) == 1
    assert calls == []


@pytest.mark.parametrize("level", ["all_sessions", "mouse_averages"])
def test_pearson_histogram_uses_values_instead_of_ranks(level):
    results, arrays, _ = _results()
    arrays["reliability_slot"][0, 1, :4] = [0, 0.1, 0.2, 0.9]
    viewer = PlaceFieldReliabilityRelationships(
        results, min_cells=2, min_mice=1, association_level=level,
        association_method="pearson", include_association_strength=True,
    )
    pearson = viewer.relationships(viewer.state)
    spearman = viewer.relationships({**viewer.state, "association_method": "spearman"})
    first = np.corrcoef([0, 0.1, 0.2, 0.9], [-0.8, -0.6, -0.4, -0.2])[0, 1]
    expected = [first, 1, 1] if level == "all_sessions" else [(first + 1) / 2, 1]
    np.testing.assert_allclose(pearson["spatial"][2], expected)
    assert pearson["spatial"][2][0] < spearman["spatial"][2][0]
    for predictor in pearson:
        np.testing.assert_allclose(pearson[predictor][1], spearman[predictor][1], equal_nan=True)
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    assert all(ax.get_xlabel() == "Pearson r" for ax in fig.axes[2:])


def test_example_load_is_lazy_session_scoped_and_tracks_config():
    results, _, calls = _results()
    results.param_axes = {"smooth_width": [5.0, 10.0]}
    results._schema_version = "v11"
    viewer = PlaceFieldReliabilityRelationships(results, display="example", smooth_width=5.0)
    assert results.store.calls == []  # Constructor reads only environment metadata.
    original = viewer.example_data(viewer.state)
    assert results.store.calls == [("session-0", (("schema_version", "v11"), ("smooth_width", 5.0)))]
    for changes in [{"env": 1}, {"comparison": "shared"}, {"amplitude_metric": "trial_mean", "amplitude_statistic": "variance"},
                    {"amplitude_transform": "bounded"}, {"fraction_active_threshold": 0.9}]:
        viewer.example_data({**viewer.state, **changes})
    assert len(results.store.calls) == 1
    viewer.plot(viewer.state)
    assert calls == []  # An example without histograms never aggregates other sessions.
    viewer.example_data({**viewer.state, "example_session": "A/2023-01-02/1"})
    assert results.store.calls[-1][0] == "session-1"
    viewer.example_data({**viewer.state, "smooth_width": 10.0})
    assert results.store.calls[-1] == ("session-0", (("schema_version", "v11"), ("smooth_width", 10.0)))
    np.testing.assert_allclose(viewer.example_data(viewer.state)["correlation"], original["correlation"])
