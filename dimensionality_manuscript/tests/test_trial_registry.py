"""Tests for the isolated whole-trial regression registry and config."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dimilibi import Population
from vrAnalysis.processors.placefields import FrameBehavior

from dimensionality_manuscript.configs import ANALYSIS_CONFIG_CLASSES
from dimensionality_manuscript.configs.trial_regression import (
    ALL_CELL_TRIAL_MODELS,
    TrialPlacecellRegressionConfig,
    TrialSplitRegressionConfig,
    _placecell_mask,
)
from dimensionality_manuscript.registry import RegistryParameters
from dimensionality_manuscript.regression_models.models import ReducedRankRegressionModel, make_gain_unit_index
from dimensionality_manuscript.trial_registry import (
    EnvironmentTrialRegistry,
    TrialRegistry,
    TrialRegistryParameters,
    TrialTimeSplit,
    split_trials_by_environment,
)


def test_even_trial_split_is_reproducible_stratified_and_restores_rng_state():
    trial_environment = np.repeat([1, 2, 3], 8)
    np.random.seed(91)
    expected_next = np.random.random()
    np.random.seed(91)

    train, test = split_trials_by_environment(trial_environment, (1, 1), seed=7)
    actual_next = np.random.random()
    train_again, test_again = split_trials_by_environment(trial_environment, (1, 1), seed=7)

    assert actual_next == expected_next
    np.testing.assert_array_equal(train, train_again)
    np.testing.assert_array_equal(test, test_again)
    assert not np.intersect1d(train, test).size
    np.testing.assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(len(trial_environment)))
    for environment in np.unique(trial_environment):
        assert np.sum(trial_environment[train] == environment) == 4
        assert np.sum(trial_environment[test] == environment) == 4


def test_trial_time_split_has_no_validation_fold():
    split = TrialTimeSplit()
    assert split["train"] == 0
    assert split["test"] == 1
    assert split["full"] == (0, 1)
    with pytest.raises(ValueError, match="no validation fold"):
        split["validation"]


def test_even_trial_split_rejects_an_environment_that_cannot_reach_both_folds():
    with pytest.raises(ValueError, match="too few to appear"):
        split_trials_by_environment(np.array([1, 2, 2]), (1, 1), seed=0)


def test_three_way_trial_split_is_disjoint_exhaustive_reproducible_and_stratified():
    trial_environment = np.repeat([1, 2, 3], 8)
    folds = split_trials_by_environment(trial_environment, (2, 1, 1), seed=11)
    folds_again = split_trials_by_environment(trial_environment, (2, 1, 1), seed=11)
    assert len(folds) == 3
    for actual, expected in zip(folds, folds_again):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(trial_environment.size))
    assert all(not np.intersect1d(a, b).size for i, a in enumerate(folds) for b in folds[i + 1 :])
    for environment in np.unique(trial_environment):
        assert [np.sum(trial_environment[fold] == environment) for fold in folds] == [4, 2, 2]


def test_placecell_mask_requires_both_metrics_in_the_same_environment():
    reliability = np.array([0.5, 0.5, 0.1, np.nan])
    fraction_active = np.array([0.2, 0.05, 0.2, 0.2])
    np.testing.assert_array_equal(
        _placecell_mask(reliability, fraction_active, (0.3, 0.1)),
        [True, False, False, False],
    )


def _frame_behavior(num_trials: int, frames_per_trial: int) -> FrameBehavior:
    trial = np.repeat(np.arange(num_trials), frames_per_trial)
    environment = np.repeat(np.repeat([1, 2], num_trials // 2), frames_per_trial)
    num_frames = len(trial)
    return FrameBehavior(
        position=np.tile(np.linspace(0, 100, frames_per_trial), num_trials),
        speed=np.full(num_frames, 5.0),
        environment=environment,
        trial=trial.astype(float),
        reward_delivery=np.zeros(num_frames, dtype=bool),
        reward_omitted=np.zeros(num_frames, dtype=bool),
    )


def test_trial_registry_reuses_base_cells_and_assigns_whole_trials(monkeypatch, tmp_path):
    num_trials = 8
    frames_per_trial = 5
    num_frames = num_trials * frames_per_trial
    num_neurons = 8
    behavior = _frame_behavior(num_trials, frames_per_trial)
    spks = np.arange(num_frames * num_neurons, dtype=float).reshape(num_frames, num_neurons)

    base_population = Population(
        spks.T,
        generate_splits=False,
        idx_samples=np.arange(num_frames),
        idx_neurons=np.array([0, 2, 3, 5, 6, 7]),
    )
    base_population.cell_split_indices = [torch.tensor([0, 3, 4]), torch.tensor([1, 2, 5])]
    base_population.time_split_indices = [torch.arange(num_frames)]

    base_registry = SimpleNamespace(
        registry_paths=SimpleNamespace(registry_path=tmp_path / "population-registry"),
        registry_params=RegistryParameters(time_split_relative_size=(1, 1, 1, 1)),
        get_population=lambda session, spks_type=None: (base_population, behavior),
    )
    (tmp_path / "population-registry").mkdir()
    session = SimpleNamespace(
        spks=spks,
        params=SimpleNamespace(spks_type="sigrebase"),
        trial_environment=np.repeat([1, 2], num_trials // 2),
        session_uid="mouse.date.1",
        session_name=("mouse", "date", "1"),
        session_print=lambda: "mouse/date/1",
    )
    monkeypatch.setattr("dimensionality_manuscript.trial_registry.get_frame_behavior", lambda session, clear_one_cache=True: behavior)

    registry = TrialRegistry(
        base_registry,
        TrialRegistryParameters(split_seed=3),
        autosave=False,
    )
    population, trial_behavior = registry.get_population(session, "sigrebase")

    np.testing.assert_array_equal(population.idx_neurons, base_population.idx_neurons)
    for actual, expected in zip(population.cell_split_indices, base_population.cell_split_indices):
        torch.testing.assert_close(actual, expected)

    fold_trials = []
    for fold_index in population.time_split_indices:
        trials = np.asarray(trial_behavior.trial)[fold_index.numpy()].astype(int)
        fold_trials.append(np.unique(trials))
        for trial in np.unique(trials):
            assert np.sum(trials == trial) == frames_per_trial
        for environment in (1, 2):
            assert np.sum(session.trial_environment[np.unique(trials)] == environment) == 2
    assert not np.intersect1d(*fold_trials).size
    np.testing.assert_array_equal(np.sort(np.concatenate(fold_trials)), np.arange(num_trials))

    trial_path = registry._get_population_path(session)
    assert trial_path.parent == tmp_path / "population-registry" / "trial"
    assert trial_path != tmp_path / "population-registry" / f"{session.session_uid}.joblib"

    # Trial mode returns one constant chunk. The existing (chunk, trial) helper must therefore
    # produce exactly one unit per held-out trial, even when adjacent trials share a fold.
    model = ReducedRankRegressionModel(registry)
    _, _, test_behavior = model.get_session_data(session, "sigrebase", "test")
    chunks = model.get_split_chunk_index(session, "sigrebase", "test")
    np.testing.assert_array_equal(chunks, np.zeros(len(test_behavior), dtype=np.int64))
    units, num_units = make_gain_unit_index(chunks, test_behavior.trial)
    assert num_units == np.unique(test_behavior.trial).size
    for trial in np.unique(test_behavior.trial):
        assert np.unique(units[test_behavior.trial == trial]).size == 1

    # A manually regenerated base population could have a different random source/target split
    # under the same RegistryParameters. Its old trial population must not then be reused.
    base_population.cell_split_indices = [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])]
    assert registry._get_population_path(session) != trial_path

    env_registry = EnvironmentTrialRegistry(
        base_registry,
        environment=1,
        selected_roi_indices=np.array([0, 2, 5, 6]),
        registry_params=TrialRegistryParameters(relative_size=(2, 1, 1), split_seed=3),
        autosave=False,
    )
    env_population, env_behavior = env_registry.get_population(session, "sigrebase")
    np.testing.assert_array_equal(env_population.idx_neurons, [0, 2, 5, 6])
    np.testing.assert_array_equal(env_population.cell_split_indices[0], [0, 1])
    np.testing.assert_array_equal(env_population.cell_split_indices[1], [2, 3])
    assert np.all(np.asarray(env_behavior.environment) == 1)


def test_trial_regression_config_is_even_only_and_grids_only_models():
    config = TrialSplitRegressionConfig()
    assert config.data_config_name == "even"
    assert config._param_grid() == {"model_name": list(ALL_CELL_TRIAL_MODELS)}
    assert len(config.generate_variations()) == len(ALL_CELL_TRIAL_MODELS)
    assert ANALYSIS_CONFIG_CLASSES[config.display_name] is TrialSplitRegressionConfig
    assert ANALYSIS_CONFIG_CLASSES[TrialPlacecellRegressionConfig.display_name] is TrialPlacecellRegressionConfig
    with pytest.raises(ValueError, match="data_config_name='even'"):
        TrialSplitRegressionConfig(data_config_name="default")


def test_legacy_trial_regression_inherits_even_hyperparameters_without_trial_optimization(monkeypatch):
    config = TrialSplitRegressionConfig(schema_version="v1", model_name="external_placefield_1d")
    base_registry = SimpleNamespace(registry_params=config.data_config.to_registry_params())
    inherited_hyperparameters = SimpleNamespace(marker="ordinary-even")

    class BaseModel:
        def get_best_hyperparameters(self, session, spks_type, method):
            return inherited_hyperparameters, 0.0, None

    target = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 2.0, 1.0]])
    prediction = target * 0.9
    train_behavior = _frame_behavior(2, 2)
    test_behavior = _frame_behavior(2, 2)

    class TrialModel:
        def process(self, session, spks_type, train_split, test_split, hyperparameters):
            assert hyperparameters is inherited_hyperparameters
            assert (train_split, test_split) == ("train", "test")
            return SimpleNamespace(
                target_data=target,
                predicted_data=prediction,
                extras={},
                trained_model=object(),
            )

        def get_session_data(self, session, spks_type, split):
            behavior = train_behavior if split == "train" else test_behavior
            return torch.zeros((1, len(behavior))), torch.zeros((2, len(behavior))), behavior

    trial_registry = object()
    monkeypatch.setattr("dimensionality_manuscript.configs.trial_regression.TrialRegistry", lambda *args, **kwargs: trial_registry)

    def fake_get_model(model_name, registry, activity_parameters):
        return BaseModel() if registry is base_registry else TrialModel()

    monkeypatch.setattr("dimensionality_manuscript.configs.trial_regression.get_model", fake_get_model)
    monkeypatch.setattr(
        "dimensionality_manuscript.configs.trial_regression._regression_quality_filter",
        lambda *args, **kwargs: {
            "reliability": np.ones((1, 2)),
            "fraction_active": np.ones((1, 2)),
            "quality_environments": np.array([1]),
            "quality_filtered_roi_mask": np.array([True, False]),
        },
    )
    monkeypatch.setattr(
        "dimensionality_manuscript.configs.trial_regression._per_environment_scores",
        lambda *args, **kwargs: {"per_environment_sentinel": np.array([1.0])},
    )
    session = SimpleNamespace()

    result = config.process(session, base_registry)

    assert result["n_trials_train"] == 2
    assert result["n_trials_test"] == 2
    assert np.isfinite(result["r2"])
    assert np.isfinite(result["r2_quality_filtered"])
    assert np.isfinite(result["r2_notquality_filtered"])
    assert result["per_environment_sentinel"] == 1.0
