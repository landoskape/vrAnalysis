from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure2 import trial_gain, trial_predictions, _trial_prediction_base as trial_base
from dimensionality_manuscript.configs.trial_regression import TrialPlacecellRegressionConfig


def test_bin_prediction_excludes_missing_and_out_of_range_frames():
    curve = trial_predictions.bin_prediction(
        np.array([2., 4., np.nan, 8., 100., 100.]),
        np.array([.1, .4, .7, 1.5, -1., 3.]),
        np.array([0., 1., 2., 3.]),
    )
    np.testing.assert_allclose(curve, [3., 8., np.nan], equal_nan=True)


@pytest.mark.parametrize("version,folds", [("v1", [None]), ("v2", [None, 0, 1])])
@pytest.mark.parametrize("drop_frame", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_replay_uses_saved_params_targets_and_scored_folds(monkeypatch, version, folds, drop_frame, temporal, reference):
    calls = []
    session = SimpleNamespace(idx_rois=np.array([False, True, False, True]))
    config = TrialPlacecellRegressionConfig(schema_version=version, model_name="rrr")
    saved = dict(env_slot_ids=np.array([7.]), selected_hyperparameters=[{"rank": 2}],
                 selected_roi_indices=[np.array([1, 3])], target_roi_indices=[np.array([3])])
    monkeypatch.setattr(trial_predictions, "PopulationRegistry", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(trial_predictions.session_cache, "get_spks",
                        lambda session: np.array([[0., 10.], [0., 14.]]))

    def registry(base, environment, selected, params, **kwargs):
        assert environment == 7
        np.testing.assert_array_equal(selected, [1, 3])
        return SimpleNamespace(fold=kwargs.get("cv_test_fold"),
            get_population=lambda *args: (SimpleNamespace(
                idx_neurons=np.array([1, 3]), cell_split_indices=[np.array([0]), np.array([1])]), None))

    monkeypatch.setattr(trial_predictions, "EnvironmentTrialRegistry", registry)

    def make_model(name, registry, **kwargs):
        trial = 3 if registry.fold is None else registry.fold
        behavior = SimpleNamespace(trial=np.array([trial, trial]),
                                   position=np.array([.2, 1.2]), idx=np.array([0, 1]))
        class Behavior(SimpleNamespace):
            def __len__(self):
                return len(self.trial)

            def filter(self, indices):
                return Behavior(**{key: value[indices] for key, value in vars(self).items()})
        behavior = Behavior(**vars(behavior))

        def process(session, spks_type, **kwargs):
            assert kwargs == dict(train_split="train", test_split="test", hyperparameters={"rank": 2})
            calls.append(registry.fold)
            if drop_frame:
                return SimpleNamespace(trained_model=("target_field", "source_field"), predicted_data=np.array([[4.]]), extras={
                    "predictions_were_filtered": True, "idx_valid_predictions": np.array([1]),
                })
            return SimpleNamespace(trained_model=("target_field", "source_field"), predicted_data=np.array([[2., 4.]]), extras={})

        return SimpleNamespace(
            decode_internal_frame_behavior=lambda source, field, behavior: Behavior(
                **{**vars(behavior), "position": behavior.position + 10}),
            activity_parameters=SimpleNamespace(presplit=True),
            _model_hyperparameters=SimpleNamespace(from_dict=lambda params: params),
            process=process,
            get_session_data=lambda *args: (None, np.array([[1., 3.]]), behavior),
        )

    monkeypatch.setattr(trial_predictions, "get_model", make_model)
    monkeypatch.setattr("vrAnalysis.processors.placefields.get_placefield_prediction",
                        lambda field, behavior: ((behavior.position + 5)[:, None], {}))
    curves = trial_predictions.replay_predictions(
        config, saved, session, np.array([0., 1., 2.]), temporal=temporal, reference=reference,
    )
    assert calls == folds
    assert set(curves) == {(3, 3 if fold is None else fold) for fold in folds}
    for curve in curves.values():
        if reference:
            indices, external, internal = curve
            np.testing.assert_array_equal(indices, [0, 1])
            np.testing.assert_allclose(external, [18.4, 20.4])
            np.testing.assert_allclose(internal, [38.4, 40.4])
        elif temporal:
            indices, values = curve
            np.testing.assert_array_equal(indices, [1] if drop_frame else [0, 1])
            np.testing.assert_allclose(values, [16.] if drop_frame else [12., 16.])
        else:
            np.testing.assert_allclose(curve, [np.nan if drop_frame else 12., 16.], equal_nan=True)


@pytest.fixture
def viewer_setup(monkeypatch):
    session = SimpleNamespace(session_uid="s", session_id="1", date="2025-01-01",
        params=SimpleNamespace(spks_type="original"),
        idx_rois=np.array([False, True, False, True]),
        trial_environment=np.array([7, 7, 7]),
        session_print=lambda: "session")
    names = ["internal_placefield_1d", "rrr"]
    configs = {name: TrialPlacecellRegressionConfig(model_name=name) for name in names}
    refs = {(0, (i,)): SimpleNamespace(row={"analysis_key": name}) for i, name in enumerate(names)}
    for config in configs.values():
        monkeypatch.setattr(type(config), "get_result", lambda *args: {"saved": True})
    results = SimpleNamespace(
        config_class=TrialPlacecellRegressionConfig, param_axes={"model_name": names},
        _param_names=["model_name"], _cells=refs, _key_to_config=configs,
        _session_index={"s": 0}, store=None, mouse_names=np.array(["mouse"]), sessions=[session],
    )
    maps = SimpleNamespace(environments=[7], distcenters=np.array([.5, 1.5]),
                           spkmap=[np.array([[[1., 2.], [2., 3.]], [[3., 4.], [4., 5.]]])])
    monkeypatch.setattr(trial_gain.session_cache, "get_env_maps", lambda session: maps)
    smp = SimpleNamespace(
        dist_edges=np.array([0., 1., 2.]),
        _idx_required_position_bins=lambda: np.array([0, 1]),
        get_processed_maps=lambda: SimpleNamespace(occmap=np.array([[np.nan, 1.], [1., 1.], [1., 1.]])),
    )
    monkeypatch.setattr(trial_gain.session_cache, "get_smp", lambda session: smp)
    calls = []

    def replay(config, saved, session, edges):
        calls.append(config.model_name)
        return {(1, 1): np.array([7., 8.]), (3, 2): np.array([9., 10.])}

    monkeypatch.setattr(trial_base._TrialPredictionBase, "_replay_predictions",
                        lambda self, *args: replay(*args))
    viewer = trial_base._TrialPredictionBase(results, models=tuple(names))
    return SimpleNamespace(viewer=viewer, calls=calls, names=names, session=session, configs=configs)


def test_viewer_cache_raw_roi_trial_alignment_and_unavailable_annotations(viewer_setup):
    viewer, calls, names, session = (
        viewer_setup.viewer, viewer_setup.calls, viewer_setup.names, viewer_setup.session,
    )
    assert calls == names
    np.testing.assert_array_equal(viewer.prediction_curves["rrr"], [7., 8.])
    assert session.params.spks_type == "original"
    state = {**viewer.state, "roi": 1, "trial": 1, "models": ["rrr"]}
    viewer.refresh_data(state)
    np.testing.assert_array_equal(viewer.prediction_curves["rrr"], [9., 10.])
    viewer.refresh_data({**state, "models": []})
    viewer.refresh_data(state)
    assert calls == names
    viewer.refresh_data({**state, "roi": 0})
    assert not viewer.prediction_curves
    assert "outside scored target set" in viewer.prediction_notes[0]
    viewer.clear_prediction_cache()
    viewer.refresh_data(state)
    assert calls == [*names, "rrr"]



def test_target_filter_limits_pairs_and_restores_original_selection(viewer_setup):
    viewer = viewer_setup.viewer
    # Both coordinates are valid individually, but this pair has no prediction.
    viewer.set_parameter_value("trial", 1)
    assert viewer.prediction_notes
    viewer.set_parameter_value("filter_to_target_set", True)
    assert viewer.parameters["roi"].options == [0, 1]
    assert viewer.parameters["trial"].options == [0]
    assert (viewer.state["roi"], viewer.state["trial"]) == (0, 0)
    assert not viewer.prediction_notes

    # The trial choices follow the selected ROI, preserving the original indices.
    viewer.set_parameter_value("roi", 1)
    assert viewer.parameters["trial"].options == [1]
    assert viewer.state["trial"] == 1
    np.testing.assert_array_equal(viewer.trial_activity, [4., 5.])
    assert set(viewer.prediction_curves) == set(viewer_setup.names)

    viewer.set_parameter_value("filter_to_target_set", False)
    assert viewer.parameters["roi"].options == [0, 1]
    assert viewer.parameters["trial"].options == [0, 1]
    assert (viewer.state["roi"], viewer.state["trial"]) == (0, 1)
    assert viewer_setup.calls == viewer_setup.names


def test_target_filter_intersects_models_and_handles_no_common_pairs(viewer_setup):
    viewer = viewer_setup.viewer
    config = viewer_setup.configs["rrr"]
    key = ("s", config.key())
    viewer._prediction_cache[key] = {(3, 2): np.array([9., 10.])}
    viewer.set_parameter_value("filter_to_target_set", True)
    assert viewer.parameters["roi"].options == [1]
    assert viewer.parameters["trial"].options == [1]
    assert not viewer.prediction_notes

    viewer._prediction_cache[key] = {}
    viewer.refresh_data(viewer.state)
    assert viewer.parameters["roi"].options == [None]
    assert viewer.parameters["trial"].options == [None]
    assert not viewer.prediction_curves
    assert "No ROI/trial pairs" in viewer.prediction_notes[0]

    # Removing the empty model recovers valid choices from the remaining cache.
    viewer.set_parameter_value("models", ["internal_placefield_1d"])
    assert viewer.parameters["roi"].options == [0, 1]
    assert viewer.parameters["trial"].options == [0]
    assert viewer.prediction_curves
    viewer.set_parameter_value("models", [])
    assert viewer.parameters["trial"].options == [0, 1]
    viewer.set_parameter_value("filter_to_target_set", False)
    assert viewer.state["roi"] == 0
    assert viewer.state["trial"] == 0


def test_trial_gain_focus_retains_original_controls_and_two_panel_plot(viewer_setup):
    viewer = trial_gain.TrialGainFocus(viewer_setup.viewer.results)
    assert set(viewer.parameters) == {"mouse", "example_session", "env", "roi", "trial", "fontsize"}
    assert viewer.parameters["roi"].max == 1
    assert viewer.parameters["trial"].max == 1
    viewer.set_parameter_value("trial", 1)
    np.testing.assert_array_equal(viewer.trial_activity, [2., 3.])
    fig = viewer.plot(viewer.state)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_xlabel() == "VR Position"
    assert fig.axes[0].get_legend_handles_labels()[1] == ["Average place field", "Trial 1"]
    assert "Gain (slope)" in fig.axes[1].texts[0].get_text()
    plt.close(fig)


def test_time_viewer_contracts_shared_gaps_preserving_model_gaps_and_cache(viewer_setup, monkeypatch):
    from dimensionality_manuscript.figure_scripts.figure2 import TrialTimeFocus, trial_time

    session = viewer_setup.session
    session.timestamps = np.array([10., 10.1, 10.4, 11., 11.2])
    behavior = SimpleNamespace(
        trial=np.array([1, 1, 1, 2, 2]),
        position=np.array([.5, 1., 1.5, .5, 1.5]),
        speed=np.array([2., 0., 2., 2., 2.]),
        environment=np.full(5, 7),
        valid_frames=lambda full_check: np.ones(5, dtype=bool),
    )
    activity = np.arange(10., dtype=float).reshape(5, 2)
    monkeypatch.setattr(trial_time, "get_frame_behavior", lambda session: behavior)
    monkeypatch.setattr(trial_time.session_cache, "get_spks", lambda session: activity)
    calls = []

    def replay(config, saved, session, edges, *, temporal, reference=False):
        assert temporal
        calls.append((config.model_name, reference))
        if reference:
            assert config.model_name == "internal_placefield_1d"
            return {
                (1, 1): (np.array([0, 1, 2]), np.array([1., 2., 3.]), np.array([3., 2., np.nan])),
                (3, 2): (np.array([3, 4]), np.array([4., 5.]), np.array([5., 4.])),
            }
        if config.model_name == "rrr":
            return {
                (1, 1): (np.array([0]), np.array([7.])),
                (3, 2): (np.array([3, 4]), np.array([9., 10.])),
            }
        return {
            (1, 1): (np.array([0, 2]), np.array([7., 8.])),
            (3, 2): (np.array([3, 4]), np.array([9., 10.])),
        }

    monkeypatch.setattr(trial_time, "replay_predictions", replay)
    viewer = TrialTimeFocus(viewer_setup.viewer.results, models=tuple(viewer_setup.names),
                           filter_to_target_set=True)
    np.testing.assert_allclose(viewer.time, [0., .25])
    np.testing.assert_array_equal(viewer.frame_indices, [0, 2])
    np.testing.assert_allclose(viewer.observed_activity, [0., 4.], equal_nan=True)
    np.testing.assert_allclose(viewer.temporal_predictions["rrr"], [7., np.nan], equal_nan=True)
    np.testing.assert_allclose(viewer.temporal_predictions["internal_placefield_1d"], [7., 8.], equal_nan=True)
    np.testing.assert_allclose(viewer.placefield_activity, [1., 3.], equal_nan=True)
    assert viewer.absolute_trial == 1
    fig = viewer.plot(viewer.state)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == "Retained trial time (s)"
    for line in fig.axes[0].lines:
        np.testing.assert_allclose(line.get_xdata(), [0., .25])
        assert line.get_linestyle() == "-"
    plt.close(fig)

    initial_calls = list(calls)
    viewer.set_parameter_value("internal4reference", True)
    np.testing.assert_allclose(viewer.placefield_activity, [3., np.nan], equal_nan=True)
    np.testing.assert_allclose(viewer.observed_activity, [0., 4.], equal_nan=True)
    assert calls == initial_calls
    viewer.set_parameter_value("internal4reference", False)

    viewer.set_parameter_value("roi", 1)
    assert viewer.state["trial"] == 1
    assert viewer.absolute_trial == 2
    np.testing.assert_allclose(viewer.time, [0., .25])
    np.testing.assert_array_equal(viewer.observed_activity, [7., 9.])
    np.testing.assert_array_equal(viewer.temporal_predictions["rrr"], [9., 10.])
    assert calls == [(name, False) for name in viewer_setup.names] + [("internal_placefield_1d", True)]
    assert session.params.spks_type == "original"

    viewer.set_parameter_value("models", [])
    assert not viewer.temporal_predictions
    assert viewer.parameters["trial"].options == [0, 1]
    viewer.clear_prediction_cache()
    assert not viewer._temporal_session_cache
    assert not viewer._reference_cache
