"""Focused tests for the reliability/prediction-quality familiarity viewer."""

from pathlib import Path
import sys
import types
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

# Avoid importing every Figure 1 viewer (notably Rastermap) during this focused test.
package_name = "dimensionality_manuscript.figure_scripts.figure1"
if package_name not in sys.modules:
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[1] / "figure_scripts" / "figure1")]
    sys.modules[package_name] = package

from dimensionality_manuscript.figure_scripts.figure1.reliability_prediction import (
    ReliabilityPredictionFamiliarity,
    ReliabilityPredictionFocus,
    _prediction_quality_by_region,
)


class _Results:
    param_axes = {}
    config_class = None

    def __init__(self):
        self.sessions = [
            self._session("M1", "2020-01-01", "1"),
            self._session("M1", "2020-01-02", "1"),
            self._session("M2", "2020-01-01", "1"),
            self._session("M2", "2020-01-02", "1"),
        ]
        self.mouse_names = np.array([session.mouse_name for session in self.sessions])
        self.arrays = {
            "reliability_slot": np.array(
                [
                    [[0.2, 0.8, 0.9], [np.nan, np.nan, np.nan]],
                    [[0.4, 0.7, 1.0], [0.5, 0.6, np.nan]],
                    [[0.3, 0.8, 0.7], [np.nan, np.nan, np.nan]],
                    [[0.5, 0.9, 0.6], [0.4, 0.7, np.nan]],
                ]
            ),
            "r2_slot": np.array(
                [
                    [[0.1, 0.4, 0.7], [np.nan, np.nan, np.nan]],
                    [[0.2, 0.5, 0.8], [0.2, 0.4, np.nan]],
                    [[0.0, 0.3, 0.6], [np.nan, np.nan, np.nan]],
                    [[0.3, 0.6, 0.9], [0.1, 0.5, np.nan]],
                ]
            ),
            "rms_slot": np.array(
                [
                    [[1.4, 0.8, 0.5], [np.nan, np.nan, np.nan]],
                    [[1.2, 0.7, 0.4], [1.0, 0.8, np.nan]],
                    [[1.5, 0.9, 0.6], [np.nan, np.nan, np.nan]],
                    [[1.1, 0.6, 0.3], [1.2, 0.7, np.nan]],
                ]
            ),
            "norm_rms_slot": np.array(
                [
                    [[1.0, 0.5, 0.25], [np.nan, np.nan, np.nan]],
                    [[0.9, 0.4, 0.2], [0.8, 0.6, np.nan]],
                    [[1.1, 0.6, 0.3], [np.nan, np.nan, np.nan]],
                    [[0.8, 0.3, 0.15], [0.9, 0.5, np.nan]],
                ]
            ),
            "pf_peak": np.array(
                [
                    [[0.4, 1.5, 3.0], [np.nan, np.nan, np.nan]],
                    [[0.5, 2.0, 3.5], [1.0, 2.5, np.nan]],
                    [[0.6, 1.0, 2.0], [np.nan, np.nan, np.nan]],
                    [[0.8, 2.5, 3.0], [1.5, 3.5, np.nan]],
                ]
            ),
            "frac_var_pred_slot": np.array(
                [
                    [[0.1, 0.4, 0.8], [np.nan, np.nan, np.nan]],
                    [[0.2, 0.5, 1.2], [0.3, 0.7, np.nan]],
                    [[0.2, 0.6, 0.9], [np.nan, np.nan, np.nan]],
                    [[0.3, 0.7, 1.1], [0.4, 0.8, np.nan]],
                ]
            ),
            "env_slot_ids": np.array([[1, np.nan], [1, 2], [3, np.nan], [3, 4]], dtype=float),
        }

    @staticmethod
    def _session(mouse, date, session_id):
        return SimpleNamespace(
            mouse_name=mouse,
            date=date,
            session_id=session_id,
            session_print=lambda m=mouse, d=date, s=session_id: f"{m}/{d}/{s}",
        )

    def sel(self, *, keys, **kwargs):
        return {key: self.arrays[key] for key in keys}


def test_example_selection_metric_switch_and_paired_filters():
    viewer = ReliabilityPredictionFamiliarity(
        _Results(),
        mouse="M1",
        example_session="M1/2020-01-01/1",
        metric="r2",
        filter_by_reliability=True,
        reliability_threshold=0.7,
        filter_by_metric=True,
        r2_filter_range=(0.3, 0.6),
    )

    # The reliability threshold is inclusive; the metric filter removes the 0.7 R² ROI.
    np.testing.assert_allclose(viewer.example_reliability, [0.8])
    np.testing.assert_allclose(viewer.example_metric, [0.4])
    assert viewer.environments == [1]

    state = {**viewer.state, "metric": "rms", "filter_by_metric": False}
    viewer.refresh_data(state)
    np.testing.assert_allclose(viewer.example_reliability, [0.8, 0.9])
    np.testing.assert_allclose(viewer.example_metric, [0.8, 0.5])

    state = {**viewer.state, "metric": "norm_rms", "filter_by_metric": False}
    viewer.refresh_data(state)
    np.testing.assert_allclose(viewer.example_reliability, [0.8, 0.9])
    np.testing.assert_allclose(viewer.example_metric, [0.5, 0.25])
    fig = viewer.plot(state)
    assert fig.axes[0].get_ylabel() == "Normalized RMS error"

    viewer.refresh_data({**state, "filter_by_metric": True, "norm_rms_filter_range": (0.3, 0.6)})
    np.testing.assert_allclose(viewer.example_reliability, [0.8])
    np.testing.assert_allclose(viewer.example_metric, [0.5])

    state = {**viewer.state, "metric": "pf_peak", "filter_by_metric": True, "pf_peak_filter_range": (1.0, 2.0)}
    viewer.refresh_data(state)
    np.testing.assert_allclose(viewer.example_reliability, [0.8])
    np.testing.assert_allclose(viewer.example_metric, [1.5])
    viewer.refresh_data({**state, "filter_by_metric": False})
    fig = viewer.plot({**state, "filter_by_metric": False})
    assert fig.axes[0].get_ylabel() == r"Peak place-field amplitude ($\sigma$)"

    state = {
        **viewer.state,
        "metric": "fraction_variance",
        "filter_by_metric": True,
        "fraction_variance_filter_range": (0.3, 0.6),
    }
    viewer.refresh_data(state)
    np.testing.assert_allclose(viewer.example_reliability, [0.8])
    np.testing.assert_allclose(viewer.example_metric, [0.4])
    viewer.refresh_data({**state, "filter_by_metric": False})
    fig = viewer.plot({**state, "filter_by_metric": False})
    assert fig.axes[0].get_ylabel() == "Fraction of variance in prediction"


def test_by_env_curves_match_r2_familiarity_experience_indexing_and_plot_three_axes():
    viewer = ReliabilityPredictionFamiliarity(_Results(), metric="rms", summary_stat="mean")

    # Slot 0 has two chronological points per mouse; slot 1 only has one and therefore is
    # omitted by the same two-mouse support rule used by R2Familiarity's final panel.
    np.testing.assert_allclose(viewer.reliability_stacks[0], [[1.9 / 3, 2.1 / 3], [1.8 / 3, 2.0 / 3]])
    np.testing.assert_allclose(viewer.metric_stacks[0], [[2.7 / 3, 2.3 / 3], [3.0 / 3, 2.0 / 3]])

    fig = viewer.plot(viewer.state)
    assert len(fig.axes) == 3
    assert [axis.get_xlabel() for axis in fig.axes] == ["Spatial reliability", "Env session #", "Env session #"]
    assert fig.axes[0].get_ylabel() == "RMS error"
    assert fig.axes[2].get_legend().get_title().get_text() == "Env"


def test_prediction_quality_regions_preserve_overall_definition_and_soft_weight_regions():
    prediction = np.array([[0.0], [0.0], [2.0], [2.0]])
    activity = np.array([[0.0], [1.0], [1.0], [3.0]])

    quality = _prediction_quality_by_region(activity, prediction)

    # SSE=3 and SST=4.75 for the existing PFPredQualityConfig overall definition.
    np.testing.assert_allclose(quality["overall"]["r2"], [1 - 3 / 4.75])
    np.testing.assert_allclose(quality["overall"]["rms"], [np.sqrt(3 / 4)])
    # Prediction/peak gives hard 0/1 memberships in this example.
    np.testing.assert_allclose(quality["within"]["rms"], [1.0])
    np.testing.assert_allclose(quality["outside"]["rms"], [np.sqrt(1 / 2)])


class _FocusResults:
    param_axes = {}
    config_class = SimpleNamespace(spks_type="sigrebase")

    def __init__(self):
        session = SimpleNamespace(
            mouse_name="M1",
            date="2020-01-01",
            session_id="1",
            params=SimpleNamespace(spks_type="oasis"),
            session_print=lambda: "M1/2020-01-01/1",
        )
        self.sessions = [session]
        self.mouse_names = np.array(["M1"])


def test_focus_sorts_roi_rank_best_to_worst_and_draws_requested_five_axes(monkeypatch):
    activity = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    prediction = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
        ]
    )
    env_maps = SimpleNamespace(
        environments=[7],
        distcenters=np.arange(2, dtype=float),
        spkmap=[np.transpose(activity.reshape(2, 2, 3), (2, 0, 1))],
    )
    extras = {"idx_valid": np.ones(4, dtype=bool), "frame_environment_index": np.zeros(4, dtype=int)}

    fake_cache = SimpleNamespace(
        get_env_maps=lambda session: env_maps,
        get_reliability=lambda session: SimpleNamespace(values=np.array([[0.3, 0.9, 0.6]])),
        get_fraction_active=lambda session: np.array([[0.8, 0.9, 0.6]]),
        get_prediction=lambda session, mode: prediction,
        get_prediction_extras=lambda session: extras,
        get_spks=lambda session: activity,
    )
    module = sys.modules["dimensionality_manuscript.figure_scripts.figure1.reliability_prediction"]
    monkeypatch.setattr(module, "session_cache", fake_cache)

    viewer = ReliabilityPredictionFocus(_FocusResults(), sort_by="reliability")
    assert viewer.state["roi"] == 0
    assert viewer.roi_index == 1

    viewer.refresh_metrics({**viewer.state, "sort_by": "rms", "quality_region": "overall"})
    assert viewer.roi_index == 0

    viewer.refresh_metrics({**viewer.state, "sort_by": "rms", "fraction_active_threshold": 0.85})
    np.testing.assert_array_equal(viewer.roi_indices, [1])
    assert viewer.roi_index == 1

    viewer.refresh_metrics({**viewer.state, "sort_by": "rms"})
    fig = viewer.plot({**viewer.state, "sort_by": "rms"})
    assert len(fig.axes) == 5
    assert [axis.get_xlabel() for axis in fig.axes] == [
        "VR position",
        "VR position",
        "Reliability",
        r"$R^2$ (overall)",
        "RMS (overall)",
    ]
