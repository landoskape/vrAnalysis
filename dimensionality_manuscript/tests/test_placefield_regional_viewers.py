"""Paired regional PF × Gain summaries and exposure-aligned plots."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from dimensionality_manuscript.figure_scripts.figure1.placefield_regional import (
    PlacefieldRegionalFamiliarityViewer,
    PlacefieldRegionalScatter,
)


class Results:
    param_axes = {"spks_type": ["sigrebase", "oasis"]}

    def __init__(self):
        # Unsorted rows, different environment IDs, and different session counts.
        self.sessions = [
            SimpleNamespace(mouse_name=mouse, date=f"2020-01-0{day}", session_id="1", environments=envs)
            for mouse, day, envs in (
                ("M1", 3, [1, 2]), ("M1", 1, [1]), ("M1", 2, [1, 2]),
                ("M2", 1, [3]), ("M2", 2, [3, 4]),
            )
        ]
        self.mouse_names = [session.mouse_name for session in self.sessions]
        scores = np.array([[3., 30.], [1., 100.], [np.nan, 20.], [10., 100.], [np.nan, 40.]])
        self.arrays = {
            "env_slot_ids": np.array([[1, 2]] * 3 + [[3, 4]] * 2),
            "reliability_slot": np.tile([0.9, 0.7, 0.9, 0.9], (5, 2, 1)),
            "fraction_active_slot": np.tile([0.4, 0.4, 0.2, 0.4], (5, 2, 1)),
            "pf_r2_within_shared_cv_slot": np.stack(
                [scores, np.full_like(scores, 300), np.full_like(scores, 400), np.full_like(scores, 500)], axis=-1,
            ),
            "pf_r2_outside_shared_cv_slot": np.stack(
                [-scores, np.full_like(scores, -300), np.full_like(scores, -400), np.full_like(scores, np.nan)], axis=-1,
            ),
        }
        self.selection = None

    def sel(self, *, keys, **selection):
        self.selection = selection
        return {key: self.arrays[key] for key in keys if key in self.arrays}


def test_scatter_uses_paired_quality_cells_and_equal_session_weights():
    results = Results()
    viewer = PlacefieldRegionalScatter(results, spks_type="oasis")
    assert results.selection == {"squeeze_ones": False, "spks_type": "oasis"}
    expected = np.array([16.5, 1., 20., 10., 40.])
    np.testing.assert_allclose(viewer.points({**viewer.state, "unit": "session"})[:, 0], expected)
    mouse_means = [expected[:3].mean(), expected[3:].mean()]
    np.testing.assert_allclose(viewer.points(viewer.state), np.column_stack([mouse_means, -np.array(mouse_means)]))
    state = {**viewer.state, "unit": "session", "env_slot": 0}
    ax, = viewer.plot(state).axes
    np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], [3, 1, 10])
    np.testing.assert_allclose(ax.collections[1].get_offsets()[:, 1], [-3, -1, -10])
    assert viewer.points({**state, "reliability_threshold": 1.0}).shape == (0, 2)


def test_familiarity_uses_equal_mouse_weights_and_keeps_missing_exposures():
    viewer = PlacefieldRegionalFamiliarityViewer(Results(), min_mice=1)
    ax, = viewer.plot(viewer.state).axes
    inside, outside, second_inside, second_outside = ax.lines
    np.testing.assert_array_equal(inside.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(inside.get_ydata(), [5.5, np.nan, 3.])
    np.testing.assert_allclose(outside.get_ydata(), [-5.5, np.nan, -3.])
    assert [inside.get_linestyle(), outside.get_linestyle()] == ["-", "--"]
    np.testing.assert_allclose(second_inside.get_ydata(), [30., 30.])
    np.testing.assert_allclose(second_outside.get_ydata(), [-30., -30.])
    assert len(ax.collections) == 4
    ax, = viewer.plot({**viewer.state, "min_mice": 2, "hide_error": True}).axes
    assert [list(line.get_xdata()) for line in ax.lines] == [[1], [1], [1], [1]]
    assert not ax.collections


def test_regional_metric_and_fold_select_the_stored_keys():
    results = Results()
    for region, value in (("within", 0.2), ("outside", 0.7)):
        key = f"pf_frac_unexplained_{region}_weighted_all_from_all_slot"
        results.arrays[key] = np.full((5, 2, 4), value)
    viewer = PlacefieldRegionalScatter(results)
    state = {**viewer.state, "metric": "frac_unexplained_weighted", "fold": "all_from_all"}
    viewer.reload_arrays(state)
    np.testing.assert_allclose(viewer.points(state), [[0.2, 0.7], [0.2, 0.7]])
    with pytest.raises(KeyError, match="schema v15"):
        viewer.reload_arrays({**state, "fold": "within_cv"})


@pytest.mark.parametrize("viewer_class", [PlacefieldRegionalScatter, PlacefieldRegionalFamiliarityViewer])
def test_residual_rms_requires_v16_and_plots_in_activity_units(viewer_class):
    results = Results()
    with pytest.raises(KeyError, match="schema v16"):
        viewer_class(results, metric="residual_rms")

    for region, rms in (("within", 0.5), ("outside", 2.0)):
        results.arrays[f"pf_residual_rms_{region}_cv_slot"] = np.full((5, 2, 4), rms)
    viewer = viewer_class(results, metric="residual_rms")
    ax, = viewer.plot(viewer.state).axes
    assert ax.get_ylabel() == "PF × Gain residual RMS"
    if viewer_class is PlacefieldRegionalScatter:
        np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], 0.5)
        np.testing.assert_allclose(ax.collections[1].get_offsets()[:, 1], 2.0)
    else:
        np.testing.assert_allclose(ax.lines[0].get_ydata(), 0.5)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), 2.0)


def test_weighted_r2_is_the_complement_of_stored_unexplained_fraction_in_both_viewers():
    results = Results()
    for region, unexplained in (("within", 0.25), ("outside", 1.25)):
        results.arrays[f"pf_frac_unexplained_{region}_weighted_cv_slot"] = np.full((5, 2, 4), unexplained)

    scatter = PlacefieldRegionalScatter(results, metric="r2_weighted")
    ax, = scatter.plot(scatter.state).axes
    assert ax.get_ylabel() == r"Regional PF × Gain $R^2$"
    for center, score, collection in zip((1, 2), (0.75, -0.25), ax.collections):
        x, y = collection.get_offsets().T
        np.testing.assert_allclose(np.sort(x), [center - 0.1, center + 0.1])
        np.testing.assert_allclose(y, score)

    ax, = scatter.plot({**scatter.state, "swarm_width": 0.0}).axes
    for center, collection in enumerate(ax.collections, 1):
        np.testing.assert_allclose(collection.get_offsets()[:, 0], center)

    familiarity = PlacefieldRegionalFamiliarityViewer(results, metric="r2_weighted")
    ax, = familiarity.plot(familiarity.state).axes
    np.testing.assert_allclose(ax.lines[0].get_ydata(), 0.75)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), -0.25)
