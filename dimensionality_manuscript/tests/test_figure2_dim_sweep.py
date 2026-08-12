import numpy as np
import matplotlib.pyplot as plt

from dimensionality_manuscript.configs.regression import RANK_VALUES
from dimensionality_manuscript.figure_scripts.figure2.dim_sweep import (
    RANK_SWEEP_MODEL_COLORS,
    RankModelsSweepViewer,
    RegressionDimSweepViewer,
    dim_sweep_curve,
)


class _FakeDimSweepResults:
    param_axes = {
        "model_name": ["internal_placefield_1d", "internal_placefield_1d_gain", "rrr"],
    }
    mouse_names = np.array(["m1", "m1", "m2"])

    def __init__(self):
        self.selected_models = []

    def sel(self, *, model_name, squeeze_ones, **selection):
        assert squeeze_ones is False
        self.selected_models.append(model_name)
        # Every model sweeps projection rank, so every model reads the same keys. The third row's
        # trailing NaN stands in for a session that reached a lower maximum rank than the others.
        prefix = "rank"
        return {
            f"{prefix}_dim": np.array(
                [
                    [2.0, 4.0, 6.0],
                    [2.0, 6.0, 8.0],
                    [4.0, 6.0, np.nan],
                ]
            ),
            f"{prefix}_r2": np.array(
                [
                    [0.1, 0.4, 0.5],
                    [0.3, np.nan, 0.75],
                    [0.8, 0.9, 1.0],
                ]
            ),
        }


def test_dim_sweep_curve_groups_true_dimensionalities_and_averages_by_mouse():
    x, y = dim_sweep_curve(_FakeDimSweepResults(), "internal_placefield_1d", "r2", {})

    np.testing.assert_array_equal(x, [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(
        y,
        [
            [0.2, 0.4, 0.5, 0.75],
            [np.nan, 0.8, 0.9, np.nan],
        ],
        equal_nan=True,
    )


def test_dim_sweep_curve_rejects_mismatched_dim_and_metric_shapes():
    results = _FakeDimSweepResults()
    original_sel = results.sel

    def mismatched_sel(**kwargs):
        selected = original_sel(**kwargs)
        selected["rank_r2"] = selected["rank_r2"][:, :-1]
        return selected

    results.sel = mismatched_sel

    with np.testing.assert_raises_regex(ValueError, "arrays must match"):
        dim_sweep_curve(results, "internal_placefield_1d", "r2", {})


def test_dim_sweep_curve_can_restrict_grid_and_minimum_mouse_support():
    x, y = dim_sweep_curve(
        _FakeDimSweepResults(),
        "internal_placefield_1d",
        "r2",
        {},
        allowed_dimensions=np.array([2.0, 4.0, 6.0]),
        min_included=2,
    )

    np.testing.assert_array_equal(x, [4.0, 6.0])
    np.testing.assert_allclose(y, [[0.4, 0.5], [0.8, 0.9]])


def test_dim_sweep_viewer_each_draws_each_mouse_and_mean_for_every_model():
    viewer = RegressionDimSweepViewer(_FakeDimSweepResults(), plot_style="each", xlog=False)

    fig = viewer.plot(viewer.state)
    ax = fig.axes[0]

    assert viewer.state["plot_style"] == "each"
    assert len(ax.lines) == 9  # two mouse curves and one mean for each of three models
    assert [line.get_label() for line in ax.lines if not line.get_label().startswith("_")] == ["PF", "PF+Gain", "Peer"]
    plt.close(fig)


def test_rank_models_sweep_viewer_selects_one_additive_and_rrr_with_performance_colors():
    results = _FakeDimSweepResults()
    results.param_axes = {
        "model_name": [
            "external_placefield_1d_structured_additive",
            "internal_placefield_1d_structured_additive",
            "rrr",
        ]
    }
    viewer = RankModelsSweepViewer(
        results,
        structured_additive="internal",
        min_included=1,
        plot_style="each",
        xlog=False,
    )

    fig = viewer.plot(viewer.state)
    ax = fig.axes[0]

    assert results.selected_models == ["internal_placefield_1d_structured_additive", "rrr"]
    mean_lines = [line for line in ax.lines if not line.get_label().startswith("_")]
    assert [line.get_label() for line in mean_lines] == ["Int. Structured Additive", "Peer"]
    assert [line.get_color() for line in mean_lines] == list(RANK_SWEEP_MODEL_COLORS)
    assert all(np.all(np.isin(line.get_xdata(), RANK_VALUES)) for line in ax.lines)
    assert all(not np.any(np.isin(line.get_xdata(), [6.0, 8.0])) for line in ax.lines)
    assert len(ax.lines) == 6  # two mouse curves and one mean for each of two models
    plt.close(fig)


def test_rank_models_sweep_viewer_exposes_dim_sweep_display_controls():
    viewer = RankModelsSweepViewer(_FakeDimSweepResults())

    for name in ("min_included", "metric", "plot_style", "se", "xlog", "linewidth", "fill_alpha", "fontsize"):
        assert name in viewer.state
