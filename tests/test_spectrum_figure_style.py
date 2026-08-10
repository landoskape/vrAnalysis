"""Focused tests for key-aware SpectrumFigureViewer display styling."""

from dimensionality_manuscript.figure_scripts.figure4.spectrum_figure import SpectrumFigureViewer


def _viewer() -> SpectrumFigureViewer:
    viewer = object.__new__(SpectrumFigureViewer)
    viewer.ff_label = "Full CA1"
    viewer.ff_color = "black"
    return viewer


def test_ff_plot_style_uses_configured_defaults_for_full_spectrum():
    assert _viewer()._ff_plot_style({"full_source_key": "SVD"}) == ("Full CA1", "black")


def test_ff_plot_style_identifies_residual_sources():
    viewer = _viewer()

    assert viewer._ff_plot_style({"full_source_key": "SVD_RES"}) == ("PF Residual", "brown")
    assert viewer._ff_plot_style({"full_source_key": "SVCA_RES"}) == ("PF Residual", "brown")
