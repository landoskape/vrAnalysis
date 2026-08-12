from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from dimensionality_manuscript.configs.regression import residual_summary_keys
from dimensionality_manuscript.figure_scripts.figure2.pf_residual import (
    ModelPlacefieldResidualExplorer,
    ModelPlacefieldResidualFamiliarityViewer,
    ModelPlacefieldResidualInsetViewer,
    ModelPlacefieldResidualViewer,
    _bounded_ticks,
)


class _FakeResidualResults:
    param_axes = {
        "model_name": [
            "internal_placefield_1d",
            "internal_placefield_1d_gain",
            "rrr",
        ]
    }
    unique_mice = np.array(["m1", "m2"])
    mouse_names = np.array(["m1", "m1", "m2", "m2"])
    sessions = [
        SimpleNamespace(date="2024-01-01", environments=np.array([0])),
        SimpleNamespace(date="2024-01-02", environments=np.array([0])),
        SimpleNamespace(date="2024-01-01", environments=np.array([0])),
        SimpleNamespace(date="2024-01-02", environments=np.array([0])),
    ]

    def sel(self, *, model_name, keys, avg_by_mouse, **selection):
        assert avg_by_mouse is False
        model_offset = self.param_axes["model_name"].index(model_name) * 0.05
        key = keys[0]
        assert key in residual_summary_keys(), f"{key!r} is not a key the residual config emits"
        metric_offset = 0.2 if "outside_pf" in key else 0.0
        return {key: np.array([0.4, 0.5, 0.45, 0.55]) + model_offset + metric_offset}


class _FakeResidualSummaryResults:
    param_axes = _FakeResidualResults.param_axes

    def __init__(self):
        self.requested_keys = []

    def sel(self, *, model_name, keys, avg_by_mouse, **selection):
        assert avg_by_mouse is True
        key = keys[0]
        assert key in residual_summary_keys(), f"{key!r} is not a key the residual config emits"
        self.requested_keys.append(key)
        if "notquality_filtered" in key:
            subset_offset = 1.0
        elif "quality_filtered" in key:
            subset_offset = 0.5
        else:
            subset_offset = 0.0
        model_offset = self.param_axes["model_name"].index(model_name) * 0.05
        return {key: np.array([0.4, 0.5]) + subset_offset + model_offset}


def test_pf_residual_summary_selects_main_and_inset_roi_subsets():
    results = _FakeResidualSummaryResults()
    viewer = ModelPlacefieldResidualViewer(
        results,
        main_show="not quality",
        inset_show="all",
    )

    assert viewer.state["main_show"] == "not quality"
    assert viewer.state["inset_show"] == "all"
    assert "mean_notquality_filtered_xval_within_pf_rms" in results.requested_keys
    assert "mean_xval_within_pf_rms" in results.requested_keys
    np.testing.assert_allclose(viewer._scores["within"][:, 0], [1.4, 1.45, 1.5])
    np.testing.assert_allclose(viewer._inset_scores["within"][:, 0], [0.4, 0.45, 0.5])


def test_pf_residual_summary_notquality_normalized_key_order():
    results = _FakeResidualSummaryResults()
    ModelPlacefieldResidualViewer(
        results,
        metric="normalized_rms",
        main_show="not quality",
        include_inset=False,
    )

    assert "mean_notquality_filtered_xval_within_pf_normalized_rms" in results.requested_keys


def test_pf_residual_summary_fold_metric_and_statistic_select_the_matching_keys():
    for fold in ("xval", "infold"):
        for metric in ("rms", "normalized_rms", "r2_weighted", "r2_shared"):
            for statistic in ("mean", "median"):
                results = _FakeResidualSummaryResults()
                ModelPlacefieldResidualViewer(
                    results,
                    fold=fold,
                    metric=metric,
                    statistic=statistic,
                    include_inset=False,
                )

                assert f"{statistic}_{fold}_within_pf_{metric}" in results.requested_keys
                assert f"{statistic}_{fold}_outside_pf_{metric}" in results.requested_keys


def test_pf_residual_summary_main_and_inset_relative_are_independent():
    viewer = ModelPlacefieldResidualViewer(
        _FakeResidualSummaryResults(),
        main_relative=True,
        inset_relative=False,
    )

    fig = viewer.plot(viewer.state)
    main_axis = fig.axes[0]
    inset_axis = main_axis.child_axes[0]

    assert main_axis.get_ylabel() == r"$\Delta$ Residual RMS"
    np.testing.assert_allclose(main_axis.lines[1].get_ydata(), [0.0, 0.05, 0.1])
    np.testing.assert_allclose(
        inset_axis.lines[0].get_ydata(),
        [0.9, 0.95, 1.0],
    )
    plt.close(fig)


def test_pf_residual_summary_main_and_inset_sharey_are_independent():
    viewer = ModelPlacefieldResidualViewer(
        _FakeResidualSummaryResults(),
        main_sharey=False,
        inset_sharey=True,
    )

    fig = viewer.plot(viewer.state)
    main_axes = fig.axes[:2]
    inset_axes = [axis.child_axes[0] for axis in main_axes]

    assert not main_axes[0].get_shared_y_axes().joined(*main_axes)
    assert inset_axes[0].get_shared_y_axes().joined(*inset_axes)
    assert main_axes[1].yaxis.get_visible()
    assert inset_axes[1].yaxis.get_visible()
    assert inset_axes[1].spines["left"].get_visible()
    plt.close(fig)


def test_pf_residual_summary_absolute_and_relative_tick_spacing():
    absolute_viewer = ModelPlacefieldResidualViewer(
        _FakeResidualSummaryResults(),
        main_show="not quality",
    )
    absolute_fig = absolute_viewer.plot(absolute_viewer.state)
    absolute_main = absolute_fig.axes[0]
    absolute_inset = absolute_main.child_axes[0]

    relative_viewer = ModelPlacefieldResidualViewer(
        _FakeResidualSummaryResults(),
        main_relative=True,
        inset_relative=True,
    )
    relative_fig = relative_viewer.plot(relative_viewer.state)
    relative_main = relative_fig.axes[0]
    relative_inset = relative_main.child_axes[0]

    np.testing.assert_allclose(np.diff(absolute_main.get_yticks()), 1.0)
    np.testing.assert_allclose(np.diff(absolute_inset.get_yticks()), 1.0)
    np.testing.assert_allclose(np.diff(relative_main.get_yticks()), 0.1)
    np.testing.assert_allclose(np.diff(relative_inset.get_yticks()), 0.1)
    for axis in (absolute_main, absolute_inset, relative_main, relative_inset):
        ymin, ymax = axis.get_ylim()
        assert ymin <= 0.0 <= ymax
    plt.close(absolute_fig)
    plt.close(relative_fig)


def test_pf_residual_summary_ticks_keep_preferred_spacing_but_cap_extreme_ranges():
    ordinary = _bounded_ticks(-0.31, 0.31, preferred_step=0.1)
    extreme_absolute = _bounded_ticks(-938.4, 0.2, preferred_step=1.0)
    extreme_relative = _bounded_ticks(-938.4, 0.2, preferred_step=0.1)

    np.testing.assert_allclose(np.diff(ordinary), 0.1)
    assert ordinary.size <= 10
    for extreme, preferred_step in ((extreme_absolute, 1.0), (extreme_relative, 0.1)):
        assert extreme.size <= 10
        assert np.all(np.diff(extreme) >= preferred_step)
        step_multiples = np.diff(extreme) / preferred_step
        np.testing.assert_allclose(step_multiples, np.round(step_multiples))


def test_pf_residual_summary_legacy_relative_alias_controls_both_regions():
    viewer = ModelPlacefieldResidualViewer(
        _FakeResidualSummaryResults(),
        relative=True,
    )

    assert viewer.state["main_relative"] is True
    assert viewer.state["inset_relative"] is True


def test_pf_residual_summary_inset_label_matches_roi_subset():
    expected_labels = {
        "all": [],
        "quality": ["placecells"],
        "not quality": ["non placecells"],
    }

    for subset, expected in expected_labels.items():
        viewer = ModelPlacefieldResidualViewer(
            _FakeResidualSummaryResults(),
            inset_show=subset,
        )
        fig = viewer.plot(viewer.state)
        inset = fig.axes[0].child_axes[0]

        assert [text.get_text() for text in inset.texts] == expected
        plt.close(fig)


def test_pf_residual_summary_inset_ylabel_tracks_relative_mode_and_font_scale():
    absolute_viewer = ModelPlacefieldResidualViewer(
        _FakeResidualSummaryResults(),
        fontsize=10.0,
        inset_ylabel_fontsize_scale=0.8,
    )
    absolute_fig = absolute_viewer.plot(absolute_viewer.state)
    absolute_inset = absolute_fig.axes[0].child_axes[0]

    relative_viewer = ModelPlacefieldResidualViewer(
        _FakeResidualSummaryResults(),
        inset_relative=True,
    )
    relative_fig = relative_viewer.plot(relative_viewer.state)
    relative_inset = relative_fig.axes[0].child_axes[0]

    assert absolute_inset.get_ylabel() == "Res.\nRMS"
    assert absolute_inset.yaxis.label.get_fontsize() == 8.0
    assert absolute_inset.yaxis.majorTicks[0].tick1line.get_markersize() == 2.0
    assert relative_inset.get_ylabel() == "$\\Delta$ Res.\nRMS"
    plt.close(absolute_fig)
    plt.close(relative_fig)


def test_pf_residual_inset_viewer_uses_one_subset_for_within_and_outside():
    results = _FakeResidualSummaryResults()
    viewer = ModelPlacefieldResidualInsetViewer(results, show="not quality")

    assert viewer.state["show"] == "not quality"
    assert not any(name in viewer.state for name in ("main_show", "inset_show", "main_relative", "inset_relative"))
    assert "mean_notquality_filtered_xval_within_pf_rms" in results.requested_keys
    assert "mean_notquality_filtered_xval_outside_pf_rms" in results.requested_keys
    np.testing.assert_allclose(viewer._scores["within"][:, 0], [1.4, 1.45, 1.5])
    np.testing.assert_allclose(viewer._scores["outside"][:, 0], [1.4, 1.45, 1.5])


def test_pf_residual_inset_viewer_draws_outside_as_an_always_present_inset():
    viewer = ModelPlacefieldResidualInsetViewer(
        _FakeResidualSummaryResults(),
        sharey=True,
        relative=True,
    )

    fig = viewer.plot(viewer.state)
    main = fig.axes[0]
    (inset,) = main.child_axes

    assert [text.get_text() for text in main.texts] == ["Within PF"]
    assert [text.get_text() for text in inset.texts] == ["Outside PF"]
    assert main.get_shared_y_axes().joined(main, inset)
    assert main.get_ylabel() == r"$\Delta$ Residual RMS"
    assert inset.get_ylabel() == "$\\Delta$ Res.\nRMS"
    np.testing.assert_allclose(main.lines[1].get_ydata(), [0.0, 0.05, 0.1])
    np.testing.assert_allclose(inset.lines[1].get_ydata(), [0.0, 0.05, 0.1])
    plt.close(fig)


def test_pf_residual_inset_viewer_is_exported_from_figure2():
    from dimensionality_manuscript.figure_scripts.figure2 import ModelPlacefieldResidualInsetViewer as exported

    assert exported is ModelPlacefieldResidualInsetViewer


def test_pf_residual_familiarity_within_only_uses_one_axis_and_compact_labels():
    viewer = ModelPlacefieldResidualFamiliarityViewer(
        _FakeResidualResults(),
        within_only=True,
        min_mice=1,
    )

    fig = viewer.plot(viewer.state)

    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylabel() == "Within-PF residual RMS"
    assert [text.get_text() for text in fig.axes[0].get_legend().get_texts()] == [
        "PF",
        "PF+Gain",
        "Peer Prediction",
    ]
    plt.close(fig)


class _FakeResidualMenuResults:
    """Serves any summary key, with a distinct value per axis level so keys are traceable."""

    param_axes = _FakeResidualResults.param_axes

    def __init__(self):
        self.calls = []

    @staticmethod
    def _value(key):
        value = 0.4
        value += 0.2 if "outside_pf" in key else 0.0
        value += 0.01 if "infold" in key else 0.0
        value += 0.002 if key.startswith("median") else 0.0
        if "notquality_filtered" in key:
            value += 1.0
        elif "quality_filtered" in key:
            value += 0.5
        return np.array([value, value + 0.1])

    def sel(self, *, model_name, keys, avg_by_mouse, **selection):
        self.calls.append((model_name, tuple(keys), avg_by_mouse))
        for key in keys:
            assert key in residual_summary_keys(), f"{key!r} is not a key the residual config emits"
        model_offset = self.param_axes["model_name"].index(model_name) * 0.05
        return {key: self._value(key) + model_offset for key in keys}


def _requested_keys(results):
    return {key for _, keys, _ in results.calls for key in keys}


def test_explorer_draws_the_product_of_every_selected_axis():
    results = _FakeResidualMenuResults()
    viewer = ModelPlacefieldResidualExplorer(
        results,
        folds=("xval", "infold"),
        statistics=("median",),
        subsets=("quality", "not quality"),
        metrics=("rms",),
    )

    assert viewer._series == [
        ("xval", "median", "quality", "rms"),
        ("xval", "median", "not quality", "rms"),
        ("infold", "median", "quality", "rms"),
        ("infold", "median", "not quality", "rms"),
    ]
    # Four series x two regions, and one aggregator call per model carrying all eight keys.
    assert len(_requested_keys(results)) == 8
    assert {model for model, _, _ in results.calls} == set(viewer._models)
    assert "median_quality_filtered_xval_within_pf_rms" in _requested_keys(results)
    assert "median_notquality_filtered_infold_outside_pf_rms" in _requested_keys(results)


def test_explorer_panels_are_always_within_then_outside():
    results = _FakeResidualMenuResults()
    viewer = ModelPlacefieldResidualExplorer(
        results,
        folds=("xval", "infold"),
        metrics=("rms", "r2_shared"),
        show_subjects=False,
    )

    fig = viewer.plot(viewer.state)
    within_axis, outside_axis = fig.axes[:2]

    for index, series in enumerate(viewer._series):
        for axis, region in ((within_axis, "within"), (outside_axis, "outside")):
            expected = np.nanmean(viewer._scores[f"median_{series[0]}_{region}_pf_{series[3]}"], axis=1)
            np.testing.assert_allclose(axis.lines[index].get_ydata(), expected)
    plt.close(fig)


def test_explorer_legend_names_only_the_varying_axes():
    viewer = ModelPlacefieldResidualExplorer(
        _FakeResidualMenuResults(),
        folds=("xval", "infold"),
        statistics=("median",),
        subsets=("all",),
        metrics=("rms",),
    )

    fig = viewer.plot(viewer.state)
    (legend,) = fig.legends

    assert [text.get_text() for text in legend.get_texts()] == ["xval", "infold"]
    assert legend.get_title().get_text() == "median all rms"
    # The legend belongs to the figure, not to a panel, so it cannot cover either one.
    assert all(axis.get_legend() is None for axis in fig.axes)
    plt.close(fig)


def test_explorer_lone_series_is_named_in_full():
    viewer = ModelPlacefieldResidualExplorer(
        _FakeResidualMenuResults(),
        folds=("infold",),
        statistics=("mean",),
        subsets=("quality",),
        metrics=("r2_weighted",),
    )

    fig = viewer.plot(viewer.state)
    (legend,) = fig.legends

    assert [text.get_text() for text in legend.get_texts()] == ["infold mean quality r2_weighted"]
    assert legend.get_title().get_text() == ""
    plt.close(fig)


def test_explorer_relative_subtracts_the_first_selected_model():
    viewer = ModelPlacefieldResidualExplorer(
        _FakeResidualMenuResults(),
        relative=True,
        show_subjects=False,
    )

    fig = viewer.plot(viewer.state)
    within_axis = fig.axes[0]

    # The fake offsets every model by 0.05, so the relative curve is 0, 0.05, 0.1.
    np.testing.assert_allclose(within_axis.lines[0].get_ydata(), [0.0, 0.05, 0.1], atol=1e-12)
    plt.close(fig)


def test_explorer_empty_axis_selection_draws_empty_panels():
    results = _FakeResidualMenuResults()
    viewer = ModelPlacefieldResidualExplorer(results, metrics=())

    assert viewer._series == []
    assert results.calls == []

    fig = viewer.plot(viewer.state)
    assert len(fig.axes) == 2
    assert all(not axis.lines for axis in fig.axes)
    plt.close(fig)


def test_explorer_model_selection_follows_the_aggregators_order():
    results = _FakeResidualMenuResults()
    viewer = ModelPlacefieldResidualExplorer(results, model_names=["rrr", "internal_placefield_1d"])

    assert viewer._models == ["internal_placefield_1d", "rrr"]

    fig = viewer.plot(viewer.state)
    assert [label.get_text() for label in fig.axes[0].get_xticklabels()] == ["internal_placefield_1d", "rrr"]
    plt.close(fig)


def test_pf_residual_familiarity_two_axis_layout_has_one_legend():
    viewer = ModelPlacefieldResidualFamiliarityViewer(
        _FakeResidualResults(),
        within_only=False,
        min_mice=1,
    )

    fig = viewer.plot(viewer.state)

    assert len(fig.axes) == 2
    assert fig.axes[0].get_legend() is not None
    assert fig.axes[1].get_legend() is None
    plt.close(fig)
