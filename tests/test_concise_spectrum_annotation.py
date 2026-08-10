"""Focused tests for the concise spectrum's shared/reliable annotation."""

import pytest
from matplotlib.figure import Figure

from dimensionality_manuscript.figure_scripts.figure3._familiarity import CONDITION_COLORS
from dimensionality_manuscript.figure_scripts.figure3.concise_spectrum import draw_shared_reliable_annotation


def test_draw_shared_reliable_annotation_centers_rows_and_separator_in_axes_coordinates():
    ax = Figure().subplots()

    top, bottom, separator = draw_shared_reliable_annotation(
        ax,
        xy=(0.7, 0.9),
        yoffset=-0.2,
        line_yoffset=0.01,
        line_xpad=0.25,
        fontsize=8.0,
    )

    assert [top.get_text(), bottom.get_text()] == [
        r" PF $\times$ CA1",
        r"CA1 $\times$ CA1",
    ]
    assert [top.get_position(), bottom.get_position()] == pytest.approx([(0.7, 0.9), (0.7, 0.7)])
    assert top.get_ha() == bottom.get_ha() == "center"
    assert top.get_transform() == bottom.get_transform() == ax.transAxes
    assert separator.get_xdata() == pytest.approx([0.45, 0.95])
    assert separator.get_ydata() == pytest.approx([0.81, 0.81])
    assert separator.get_transform() == ax.transAxes
    assert separator.get_linestyle() == "-"
    assert top.get_color() == CONDITION_COLORS["behaving"]
    assert bottom.get_color() == separator.get_color() == "black"
    assert top.get_fontweight() == bottom.get_fontweight() == "bold"


def test_draw_shared_reliable_annotation_accepts_positive_row_offset_and_curve_colors():
    ax = Figure().subplots()

    top, bottom, separator = draw_shared_reliable_annotation(
        ax,
        xy=(0.4, 0.2),
        yoffset=0.1,
        line_yoffset=-0.02,
        line_xpad=0.1,
        fontsize=10.0,
        numerator_color="red",
        denominator_color="blue",
    )

    assert bottom.get_position() == pytest.approx((0.4, 0.3))
    assert separator.get_ydata() == pytest.approx([0.23, 0.23])
    assert top.get_color() == "red"
    assert bottom.get_color() == separator.get_color() == "blue"


def test_draw_shared_reliable_annotation_supports_xcov_svr_definition():
    ax = Figure().subplots()

    top, bottom, separator, svr = draw_shared_reliable_annotation(
        ax,
        xy=(0.4, 0.8),
        yoffset=-0.2,
        line_yoffset=0.01,
        line_xpad=0.25,
        fontsize=9.0,
        mode="X-Cov",
    )

    assert top.get_text() == r"$\mathbf{xcov}(\mathbf{PF}_{j};\,\mathbf{CA1}_{k})$"
    assert bottom.get_text() == r"$\mathbf{xcov}(\mathbf{CA1}_{j};\,\mathbf{CA1}_{k})$"
    assert svr.get_text() == r"$=\mathrm{SVR}$"
    assert svr.get_position() == pytest.approx((0.665, 0.71))
    assert svr.get_ha() == "left"
    assert svr.get_va() == "center"
    assert svr.get_transform() == ax.transAxes
    assert svr.get_color() == separator.get_color() == "black"


def test_draw_shared_reliable_annotation_none_mode_draws_nothing():
    ax = Figure().subplots()

    artists = draw_shared_reliable_annotation(
        ax,
        xy=(0.4, 0.8),
        yoffset=-0.2,
        line_yoffset=0.0,
        line_xpad=0.25,
        fontsize=9.0,
        mode="None",
    )

    assert artists == ()
    assert not ax.texts
    assert not ax.lines
