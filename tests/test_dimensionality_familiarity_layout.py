"""Layout tests for the figure-4 dimensionality familiarity viewer."""

from matplotlib import pyplot as plt
import numpy as np

from dimensionality_manuscript.figure_scripts.figure4.familiarity import DimensionalityFamiliarityViewer


def _viewer() -> DimensionalityFamiliarityViewer:
    viewer = object.__new__(DimensionalityFamiliarityViewer)
    viewer._previous_figure = None
    viewer.figsize = (4.0, 4.0)
    viewer.pf_label = "Placefields"
    viewer.ff_label = "Full CA1"
    viewer._ylabel = "Dimensionality"
    curves = {0: {"mouse-a": np.array([2.0, 3.0]), "mouse-b": np.array([3.0, 4.0])}}
    viewer._pf_curves = curves
    viewer._ff_curves = curves
    return viewer


def _state(**overrides) -> dict:
    return {
        "by_env_layout": "row",
        "full_key": "SVCA_RES",
        "sharey": False,
        "log_y": False,
        "ylim_start_at_one": False,
        "display": "errorPlot",
        "session_alignment": "within_env",
        "pf_text_x": 0.05,
        "pf_text_y": 0.9,
        "ff_text_x": 0.05,
        "ff_text_y": 0.9,
        "legend_panel": "top",
        "legend_anchor_x": 0.0,
        "legend_anchor_y": 0.0,
        "legend_visible": False,
        **overrides,
    }


def test_by_env_column_layout_puts_residual_above_pfs_and_shares_x():
    viewer = _viewer()

    fig = viewer._plot_by_env(_state(by_env_layout="col"), fontsize=8.0)
    top, bottom = fig.axes

    assert top.texts[0].get_text() == "Residual CA1"
    assert bottom.texts[0].get_text() == "Placefields"
    assert top.get_shared_x_axes().joined(top, bottom)
    assert not top.spines["bottom"].get_visible()
    assert not top.xaxis.get_visible()
    assert top.get_xlabel() == ""
    assert bottom.spines["bottom"].get_visible()
    assert bottom.get_xlabel() == "Env session #"
    plt.close(fig)


def test_by_env_row_layout_keeps_placefields_left_and_ca1_right():
    viewer = _viewer()

    fig = viewer._plot_by_env(_state(), fontsize=8.0)
    left, right = fig.axes

    assert left.texts[0].get_text() == "Placefields"
    assert right.texts[0].get_text() == "Residual CA1"
    assert not left.get_shared_x_axes().joined(left, right)
    assert left.spines["bottom"].get_visible()
    assert right.spines["bottom"].get_visible()
    assert left.get_xlabel() == "Env session #"
    assert right.get_xlabel() == "Env session #"
    plt.close(fig)


def test_by_env_column_legend_can_anchor_above_lower_panel():
    viewer = _viewer()
    reference = viewer._plot_by_env(_state(by_env_layout="col"), fontsize=8.0)
    reference.canvas.draw()
    reference_positions = [axis.get_position().bounds for axis in reference.axes]
    plt.close(reference)

    legend_state = {
        "legend_visible": True,
        "legend_loc": "upper left",
        "legend_ncols": 1,
        "legend_fontsize_scale": 1.0,
        "legend_handlelength": 2.0,
        "legend_handletextpad": 0.8,
        "legend_labelspacing": 0.5,
        "legend_borderpad": 0.4,
        "legend_borderaxespad": 0.5,
        "legend_markerfirst": True,
        "legend_frameon": False,
    }

    fig = viewer._plot_by_env(
        _state(
            by_env_layout="col",
            legend_panel="bottom",
            legend_anchor_x=0.1,
            legend_anchor_y=0.2,
            **legend_state,
        ),
        fontsize=8.0,
    )
    top, bottom = fig.axes
    legend = bottom.get_legend()
    fig.canvas.draw()

    assert top.get_legend() is None
    assert legend is not None
    assert not legend.get_in_layout()
    anchor = legend.get_bbox_to_anchor().transformed(bottom.transAxes.inverted())
    np.testing.assert_allclose(anchor.bounds, (0.1, 0.2, 1.0, 1.0))
    for axis, reference_position in zip(fig.axes, reference_positions):
        np.testing.assert_allclose(axis.get_position().bounds, reference_position)
    plt.close(fig)
