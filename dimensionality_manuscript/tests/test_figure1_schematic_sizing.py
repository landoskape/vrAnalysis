from unittest.mock import Mock

import matplotlib.pyplot as plt
import pytest

from dimensionality_manuscript.figure_scripts.figure1.schematic import VREnvironmentSchematic, vr_schematic_and_speed


def _schematic_and_state():
    schematic = object.__new__(VREnvironmentSchematic)
    schematic.num_rooms = 4
    schematic.envs = (1, 3, 4)
    state = {
        "panel_aspect": 1.6,
        "room_gap": 0.05,
        "env_gap": 0.34,
        "arrow_gap": 0.14,
        "track_height": 0.16,
        "margin": 0.06,
        "show_legend": True,
        "legend_yoffset": 0.0,
        "fig_width": 7.0,
        "fig_height": 4.0,
        "force_width": False,
    }
    return schematic, state


def test_fitted_layout_force_width_derives_figure_height():
    schematic, state = _schematic_and_state()
    metrics = schematic.layout_metrics(state)

    fitted = schematic.fitted_figure_layout(state, figsize=(7.0, 1.0), force_width=True)

    unit_in = 7.0 / metrics["width"]
    assert fitted["unit_in"] == pytest.approx(unit_in)
    assert fitted["figsize"] == pytest.approx((7.0, metrics["height"] * unit_in))
    assert fitted["axes_bounds"] == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_fitted_layout_uses_more_constraining_height_and_centers_width():
    schematic, state = _schematic_and_state()
    metrics = schematic.layout_metrics(state)

    fitted = schematic.fitted_figure_layout(state)

    unit_in = state["fig_height"] / metrics["height"]
    axes_width = metrics["width"] * unit_in / state["fig_width"]
    assert fitted["unit_in"] == pytest.approx(unit_in)
    assert fitted["figsize"] == pytest.approx((state["fig_width"], state["fig_height"]))
    assert fitted["axes_bounds"] == pytest.approx(((1.0 - axes_width) / 2, 0.0, axes_width, 1.0))


def test_fitted_layout_uses_more_constraining_width_and_centers_height():
    schematic, state = _schematic_and_state()
    metrics = schematic.layout_metrics(state)
    state.update(fig_width=4.0, fig_height=7.0)

    fitted = schematic.fitted_figure_layout(state)

    unit_in = state["fig_width"] / metrics["width"]
    axes_height = metrics["height"] * unit_in / state["fig_height"]
    assert fitted["unit_in"] == pytest.approx(unit_in)
    assert fitted["axes_bounds"] == pytest.approx((0.0, (1.0 - axes_height) / 2, 1.0, axes_height))


def test_report_figsize_returns_requested_or_width_derived_size():
    schematic, state = _schematic_and_state()

    assert schematic.report_figsize(state) == pytest.approx((7.0, 4.0))

    state["force_width"] = True
    metrics = schematic.layout_metrics(state)
    expected_height = metrics["height"] * state["fig_width"] / metrics["width"]
    assert schematic.report_figsize(state) == pytest.approx((state["fig_width"], expected_height))


def test_combined_figure_reuses_width_forced_schematic_sizing():
    _, state = _schematic_and_state()

    class ComposableSchematic:
        num_rooms = 4
        envs = (1, 3, 4)
        layout_metrics = VREnvironmentSchematic.layout_metrics
        fitted_figure_layout = VREnvironmentSchematic.fitted_figure_layout

        def __init__(self):
            self.state = state
            self.draw = Mock()

    schematic = ComposableSchematic()
    schematic.fitted_figure_layout = Mock(wraps=schematic.fitted_figure_layout)

    speed = Mock()
    speed.state = {}
    speed.draw = Mock()

    fig = vr_schematic_and_speed(schematic, speed, fig_width=3.5, fig_height=6.0)
    try:
        schematic.fitted_figure_layout.assert_called_once_with(
            state,
            figsize=((0.98 - 0.16) * 3.5, 6.0),
            force_width=True,
        )
        assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 6.0))
        schematic.draw.assert_called_once()
        speed.draw.assert_called_once()
    finally:
        plt.close(fig)
