"""Focused tests for the reliable-spectrum text annotation."""

import numpy as np
from matplotlib.figure import Figure

from dimensionality_manuscript.figure_scripts.figure4.spectrum_figure import draw_reliable_spectrum_annotation


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
        "ROI_A",
        r"$\times$",
        "ROI_B",
        r"$\times$ = reliable spectrum",
    ]
    np.testing.assert_allclose(
        [artist.get_position() for artist in artists],
        [(0.4, 0.9), (0.4, 0.8), (0.4, 0.7), (0.6, 0.8)],
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
        r"$\times$",
        "source B",
        r"$\times$ = shared component",
    ]
    assert all(artist.get_color() == "red" for artist in artists)
