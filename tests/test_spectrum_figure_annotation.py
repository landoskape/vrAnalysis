"""Focused tests for the reliable-spectrum text annotation."""

import numpy as np
from matplotlib.figure import Figure

from dimensionality_manuscript.figure_scripts.figure4.spectrum_figure import (
    SpectrumFigureViewer,
    draw_dimensionality_equation_annotation,
    draw_reliable_spectrum_annotation,
    draw_xcov_spectrum_annotation,
)


def test_draw_reliable_spectrum_annotation_positions_math_text_in_axes_coordinates():
    ax = Figure().subplots()

    artists = draw_reliable_spectrum_annotation(
        ax,
        xy=(0.4, 0.9),
        y_spacing=0.1,
        definition_x_offset=0.2,
        fontsize=8.0,
    )

    assert [artist.get_text() for artist in artists] == [
        r"$ROIs_A$",
        r"$ROIs_B$",
        r"$\times$ = reliable spectrum",
    ]
    np.testing.assert_allclose(
        [artist.get_position() for artist in artists],
        [(0.4, 0.9), (0.4, 0.7), (0.6, 0.8)],
    )
    assert all(artist.get_transform() == ax.transAxes for artist in artists)
    assert all(artist.get_fontsize() == 8.0 for artist in artists)


def test_draw_reliable_spectrum_annotation_allows_atomic_text_customization():
    ax = Figure().subplots()

    artists = draw_reliable_spectrum_annotation(
        ax,
        xy=(0.0, 1.0),
        y_spacing=0.2,
        definition_x_offset=0.3,
        fontsize=10.0,
        top_label="source A",
        bottom_label="source B",
        definition_label="shared component",
        color="red",
    )

    assert [artist.get_text() for artist in artists] == [
        "source A",
        "source B",
        r"$\times$ = shared component",
    ]
    assert all(artist.get_color() == "red" for artist in artists)


def test_draw_xcov_spectrum_annotation_uses_two_left_aligned_lines():
    ax = Figure().subplots()

    artists = draw_xcov_spectrum_annotation(
        ax,
        xy=(0.25, 0.8),
        y_spacing=0.15,
        fontsize=9.0,
    )

    assert [artist.get_text() for artist in artists] == [
        "x-cov(source, target)",
        r"$\rightarrow$ reliable spectrum",
    ]
    np.testing.assert_allclose(
        [artist.get_position() for artist in artists],
        [(0.25, 0.8), (0.25, 0.65)],
    )
    assert all(artist.get_horizontalalignment() == "left" for artist in artists)


def test_none_annotation_mode_draws_no_text():
    ax = Figure().subplots()
    viewer = object.__new__(SpectrumFigureViewer)
    viewer.pf_label = "Placefields"
    viewer.pf_color = "purple"
    state = {
        "show_annotation": "None",
        "show_equation": False,
        "annotation_x": 0.2,
        "annotation_y": 0.8,
        "annotation_y_spacing": 0.1,
        "annotation_definition_x_offset": 0.2,
    }

    viewer._draw_spectrum_annotation(ax, state, fontsize=9.0, ff_label="PF Residual", ff_color="brown")

    assert len(ax.texts) == 0


def test_dimensionality_equation_uses_term_bounds_and_plot_colors():
    ax = Figure().subplots()

    artists = draw_dimensionality_equation_annotation(
        ax,
        xy=(0.1, 0.9),
        fontsize=10.0,
        pf_label="Placefields",
        ff_label="PF Residual",
        pf_color="purple",
        ff_color="brown",
        linewidth=2.0,
        yoffset=0.02,
        yheight=0.12,
        arrow_mutation_scale=14.0,
    )

    texts = artists[:5]
    bars = (artists[5], artists[8])
    dimensions = (artists[7], artists[10])
    assert [artist.get_text() for artist in texts] == ["CA1", "=", "Placefields", "+", "PF Residual"]
    assert [artist.get_fontweight() for artist in texts] == ["bold", "normal", "bold", "normal", "bold"]
    assert [bar.get_color() for bar in bars] == ["purple", "brown"]
    assert [dimension.get_color() for dimension in dimensions] == ["purple", "brown"]
    assert all(dimension.get_text() == "dim?" for dimension in dimensions)
    assert all(np.isclose(bar.get_linewidth(), 2.0) for bar in bars)
