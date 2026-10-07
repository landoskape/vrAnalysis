"""Focused tests for the direct-store cross-validated place-field viewer."""

from types import SimpleNamespace
from pathlib import Path
import sys
import types

import matplotlib
import numpy as np

matplotlib.use("Agg")

# Loading ``figure1.__init__`` imports every panel, including Rastermap.  This focused test only
# needs the cross-validation module, so install a normal namespace-package shell for that folder.
package_name = "dimensionality_manuscript.figure_scripts.figure1"
if package_name not in sys.modules:
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[1] / "figure_scripts" / "figure1")]
    sys.modules[package_name] = package

from dimensionality_manuscript.figure_scripts.figure1.crossval import CrossValidatedPlacefieldFocus, _BWR_POSITIVE_CMAP
from dimensionality_manuscript.configs.placefield_structure import CrossValidatedPlacefieldsConfig, _rms_prediction_error


class _Store:
    def __init__(self, result):
        self.result = result
        self.request = None

    def get(self, session_uid, config):
        self.request = (session_uid, config)
        return self.result

    def summary_table(self, **kwargs):
        config = CrossValidatedPlacefieldsConfig()
        return [
            {
                "session_id": "M1.2020-01-01.1",
                "analysis_key": config.key(),
            }
        ]


def _session():
    return SimpleNamespace(
        mouse_name="M1",
        date="2020-01-01",
        session_id="1",
        session_uid="M1.2020-01-01.1",
        session_print=lambda: "M1/2020-01-01/1",
        env_length=np.array([200.0]),
    )


def _results(store):
    session = _session()
    return SimpleNamespace(
        store=store,
        sessions=[session],
        session_ids=[session.session_uid],
        mouse_names=np.array([session.mouse_name]),
    )


def _result():
    nan_map = np.full((3, 4), np.nan)
    even = np.stack(
        [
            np.array([[0, 0, 1, 4], [4, 1, 0, 0], [0, 3, 1, 0]], dtype=float),
            nan_map,
        ]
    )
    odd = np.stack(
        [
            np.array([[0, 0, 2, 5], [3, 2, 0, 0], [0, 2, 2, 0]], dtype=float),
            nan_map,
        ]
    )
    return {
        "even_placefield": even,
        "odd_placefield": odd,
        "odd_rms_error": np.stack(
            [
                np.array([[0, 0, 3, 4], [2, 4, 0, 0], [0, 1, 2, 0]], dtype=float),
                nan_map,
            ]
        ),
        "reliability": np.array([[0.9, 0.8, 0.1], [np.nan, np.nan, np.nan]]),
        "fraction_active": np.array([[0.8, 0.7, 0.9], [np.nan, np.nan, np.nan]]),
        "env_slot_ids": np.array([7.0, np.nan]),
        "spks_std": np.array([1.0, 2.0, 1.0]),
    }


def test_focus_loads_one_session_and_cross_validates_sort_and_rms_error():
    store = _Store(_result())
    results = _results(store)

    viewer = CrossValidatedPlacefieldFocus(
        results,
        mouse="M1",
        example_session="M1/2020-01-01/1",
        env=0,
        reliability_threshold=0.5,
        fraction_active_threshold=0.5,
    )

    assert store.request[0] == results.sessions[0].session_uid
    assert viewer.state["mouse"] == "M1"
    assert viewer.state["example_session"] == "M1/2020-01-01/1"
    assert viewer.state["env"] == 0
    assert viewer.environments == [7]
    # Neuron 1 peaks first in the even fold, followed by neuron 0; neuron 2 is unreliable.
    np.testing.assert_array_equal(viewer.roi_indices, [1, 0])
    np.testing.assert_allclose(viewer.placefield, [[1.5, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 5.0]])
    np.testing.assert_allclose(viewer.rms_error, [[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0]])


def test_focus_plot_has_two_main_population_axes():
    store = _Store(_result())
    viewer = CrossValidatedPlacefieldFocus(
        _results(store),
        mouse="M1",
        example_session="M1/2020-01-01/1",
        env=0,
        text_xy=(0.9, 0.85),
    )

    fig = viewer.plot(viewer.state)

    main_axes = fig.axes[:2]
    assert [axis.get_title() for axis in main_axes] == ["", ""]
    assert [axis.texts[0].get_text() for axis in main_axes] == ["Placefields", "RMS error"]
    assert [axis.texts[0].get_position() for axis in main_axes] == [(0.9, 0.85), (0.9, 0.85)]
    assert all(axis.texts[0].get_ha() == "right" and axis.texts[0].get_va() == "top" for axis in main_axes)
    assert main_axes[0].get_ylabel() == "ROIs"
    assert [axis.get_xlabel() for axis in main_axes] == ["VR Position (cm)", "VR Position (cm)"]
    assert all(list(axis.get_xticks()) == [0, 200] for axis in main_axes)
    assert all(list(axis.get_yticks()) == [] for axis in main_axes)
    assert main_axes[0].spines["left"].get_visible()
    assert not main_axes[1].spines["left"].get_visible()
    assert all(axis.spines["bottom"].get_visible() for axis in main_axes)
    assert main_axes[0].child_axes[0].get_ylabel() == r"Fluorescence ($\sigma$)"


def test_focus_reward_zone_uses_environment_style_controls_and_optional_legend():
    result = _result()
    result["env_slot_ids"] = np.array([3.0, np.nan])
    viewer = CrossValidatedPlacefieldFocus(
        _results(_Store(result)),
        reward_linewidth=2.5,
        reward_linestyle="--",
        reward_alpha=0.4,
        show_reward_legend=True,
    )

    fig = viewer.plot(viewer.state)
    main_axes = fig.axes[:2]

    for axis in main_axes:
        reward_line = axis.lines[0]
        np.testing.assert_allclose(reward_line.get_xdata(), [100, 100])
        assert reward_line.get_color() == "blue"
        assert reward_line.get_linewidth() == 2.5
        assert reward_line.get_linestyle() == "--"
        assert reward_line.get_alpha() == 0.4
    assert main_axes[0].get_legend() is not None
    assert [text.get_text() for text in main_axes[0].get_legend().get_texts()] == ["Reward zone"]
    assert main_axes[1].get_legend() is None


def test_rms_error_squares_each_heldout_trial_before_averaging():
    spkmap = np.array([[[99.0], [1.0], [99.0], [3.0]]])
    prediction = np.array([[0.0]])

    error = _rms_prediction_error(spkmap, prediction, np.array([1, 3]))

    np.testing.assert_allclose(error, [[np.sqrt(5.0)]])
    assert error[0, 0] != 2.0  # abs(mean residual) would incorrectly give 2


def test_positive_bwr_colormap_is_exactly_the_white_to_red_half():
    bwr = matplotlib.colormaps["bwr"]

    np.testing.assert_allclose(_BWR_POSITIVE_CMAP(0.0), bwr(0.5))
    np.testing.assert_allclose(_BWR_POSITIVE_CMAP(1.0), bwr(1.0))
