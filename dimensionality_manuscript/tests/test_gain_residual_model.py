"""Trial-template decomposition, held-out prediction, and additive score accounting."""

from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dimensionality_manuscript.configs.trial_regression import TrialPlacecellRegressionConfig
from dimensionality_manuscript.regression_models.gain_residual import (
    PlaceFieldGainResidualModel,
    gain_residual_scores,
)
from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldGainResidualHyperparameters
from dimensionality_manuscript.regression_models.models import PlaceFieldModel, get_regressor_dimensionality_from_hyperparameters
from dimensionality_manuscript.registry import MODEL_NAMES, get_model


class Behavior:
    def __init__(self, split, trials, frames):
        self.split = split
        self.trial = np.repeat(np.arange(trials), frames)
        self.idx = np.arange(trials * frames)
        self.position = np.tile(np.arange(frames), trials)

    def filter(self, indices):
        result = copy(self)
        for name in ("trial", "idx", "position"):
            setattr(result, name, getattr(self, name)[indices])
        return result


@pytest.fixture
def synthetic(monkeypatch):
    rng = np.random.default_rng(14)
    datasets, templates = {}, {}
    weights_gain = np.array([[0.5, 0.2], [0.2, 0.6]])
    weights_residual = np.array([[0.8, -0.2], [-0.3, 0.4]])
    for split, n_trials in (("train", 16), ("validation", 5), ("test", 6)):
        behavior = Behavior(split, n_trials, 6)
        p = np.tile([0.3, 1, 2, 3, 1, 0.5], n_trials)
        source_pf = np.array([1, 1.5])[:, None] * p
        target_pf = np.array([0.8, 1.2])[:, None] * p
        source_gain = rng.uniform(1, 3, (n_trials, 2))
        target_gain = source_gain @ weights_gain + [0.2, 0.3]
        residual = rng.normal(0, 0.1, (2, n_trials, 6))
        trial_p = p.reshape(n_trials, 6)
        residual -= (np.sum(residual * trial_p, axis=2, keepdims=True) / np.sum(trial_p**2, axis=1)[None, :, None]) * trial_p
        residual = residual.reshape(2, -1)
        units = behavior.trial
        source = source_pf * source_gain.T[:, units] + residual
        target = target_pf * target_gain.T[:, units] + weights_residual.T @ residual
        datasets[split] = (torch.tensor(source), torch.tensor(target), behavior)
        templates[split] = {"source": source_pf, "target": target_pf}

    model = PlaceFieldGainResidualModel(None, min_frames_per_unit=1)
    monkeypatch.setattr(model, "get_session_data", lambda session, spks_type, split: datasets[split])
    monkeypatch.setattr(model, "get_split_chunk_index", lambda session, spks_type, split: np.zeros(len(datasets[split][2].trial)))
    monkeypatch.setattr(PlaceFieldModel, "train", lambda *args: ("target", "source"))
    monkeypatch.setattr(
        "dimensionality_manuscript.regression_models.gain_residual.get_placefield_prediction",
        lambda pf, behavior: (templates[behavior.split][pf][:, behavior.idx].T, None),
    )
    hp = PlaceFieldGainResidualHyperparameters(alpha=0, rank=2, alpha_residual=0, rank_residual=2)
    return model, hp, datasets, templates


def test_exact_projection_and_held_out_reconstruction_without_target_access(synthetic):
    model, hp, datasets, _ = synthetic
    inputs = model.decomposition_inputs(None, "target", "source", split="train", include_target=True)
    source, target, _ = datasets["train"]
    np.testing.assert_allclose(inputs["target_gain_activity"] + inputs["target_residual"], target)
    for unit in range(inputs["num_units"]):
        keep = inputs["unit_index"] == unit
        np.testing.assert_allclose(
            np.sum(inputs["target_prediction"][:, keep] * inputs["target_residual"][:, keep], axis=1),
            0,
            atol=1e-12,
        )
    coefficients = model.train(None, hyperparameters=hp)
    expected = datasets["test"][1].numpy().copy()
    # A held-out target full of NaN cannot alter predictions or select test frames.
    datasets["test"] = (datasets["test"][0], torch.full_like(datasets["test"][1], torch.nan), datasets["test"][2])
    prediction, extras = model.predict(None, coefficients, hyperparameters=hp)
    np.testing.assert_allclose(prediction, expected, atol=1e-11)
    np.testing.assert_allclose(prediction, extras["gain_prediction"] + extras["residual_prediction"])
    assert np.any(extras["residual_prediction"] < 0)
    assert extras["gain_source"].shape == (2, 6)
    assert extras["residual_source"].shape == (2, 36)


def test_common_frame_filter_and_empty_trial_remapping(synthetic):
    model, _, datasets, templates = synthetic
    templates["test"]["target"][:, :6] = np.nan
    templates["test"]["source"][:, 8] = np.nan
    inputs = model.decomposition_inputs(None, "target", "source", split="test", include_target=True)
    assert inputs["num_units"] == 5
    assert inputs["was_filtered"]
    expected = np.setdiff1d(np.arange(36), [0, 1, 2, 3, 4, 5, 8])
    np.testing.assert_array_equal(inputs["idx_valid_prediction"], expected)
    np.testing.assert_allclose(
        inputs["target_gain_activity"] + inputs["target_residual"],
        datasets["test"][1][:, expected],
    )
    with pytest.raises(ValueError, match="Non-finite"):
        model.decomposition_inputs(None, "target", "source", split="test", nan_safe=True)
    coefficients = model.train(None, hyperparameters=PlaceFieldGainResidualHyperparameters())
    prediction, extras = model.predict(None, coefficients)
    assert prediction.shape == (2, len(expected))
    np.testing.assert_array_equal(extras["idx_valid_predictions"], expected)


def test_internal_uses_frozen_maps_at_decoded_position(synthetic, monkeypatch):
    model, _, _, _ = synthetic
    model.internal = True
    calls = []

    def decode(source, source_pf, behavior):
        assert source_pf == "source"
        decoded = copy(behavior)
        decoded.position = np.full_like(behavior.position, 65)
        return decoded

    def lookup(pf, behavior):
        calls.append((pf, behavior.position.copy()))
        return np.ones((len(behavior.idx), 2)) * (2 if pf == "source" else 3), None

    monkeypatch.setattr(model, "decode_internal_frame_behavior", decode)
    monkeypatch.setattr("dimensionality_manuscript.regression_models.gain_residual.get_placefield_prediction", lookup)
    inputs = model.decomposition_inputs(None, "target", "source", split="train", include_target=True)
    assert [pf for pf, _ in calls] == ["source", "target"]
    assert all(np.all(positions == 65) for _, positions in calls)
    assert np.all(inputs["frame_behavior_internal"].position == 65)
    assert not np.all(inputs["frame_behavior"].position == 65)


def test_validation_tunes_two_branches_without_test_data(synthetic, monkeypatch):
    model, hp, datasets, _ = synthetic
    datasets.pop("test")
    monkeypatch.setattr(model, "inherited_placefield_hyperparameters", lambda *args: hp)

    def small_search(func, a, b, **kwargs):
        candidates = [a, b]
        scores = [func(value) for value in candidates]
        return candidates[int(np.argmin(scores))], min(scores)

    monkeypatch.setattr("dimensionality_manuscript.regression_models.gain_residual.golden_section_search", small_search)
    params, score, table = model.optimize(None, "sigrebase", method="preferred")
    assert set(table.branch) == {"gain", "residual"}
    assert {"alpha", "rank", "alpha_residual", "rank_residual"} <= params.keys()
    assert np.isfinite(score)


def test_unconstrained_residual_prediction_is_not_rectified_or_projected(synthetic):
    model, hp, _, _ = synthetic

    class FixedModel:
        max_rank = 1

        def predict(self, x, rank, nonnegative):
            return torch.full((x.shape[0], 2), 0.25 if nonnegative else -4.0, dtype=torch.float64)

    inputs = model.decomposition_inputs(None, "target", "source", split="test")
    prediction, extras = model.apply_models(inputs, FixedModel(), FixedModel(), hp)
    np.testing.assert_allclose(extras["residual_prediction"], -4)
    assert np.any(prediction < 0)
    assert np.any(np.sum(extras["residual_prediction"] * inputs["target_prediction"], axis=1) != 0)


def test_component_scores_share_units_and_increments_telescope(synthetic):
    model, hp, datasets, _ = synthetic
    coefficients = model.train(None, hyperparameters=replace(hp, alpha=1, alpha_residual=1))
    prediction, extras = model.predict(None, coefficients, hyperparameters=hp)
    report = SimpleNamespace(target_data=datasets["test"][1].numpy(), extras=extras)
    scores = gain_residual_scores(report)
    total_r2 = 1 - scores["sse_combined"] / scores["sst_activity"]
    assert scores["r2_template"] + scores["delta_r2_gain"] + scores["delta_r2_residual"] == pytest.approx(total_r2)
    assert scores["sse_combined"] == pytest.approx(np.sum((report.target_data - prediction) ** 2))


def test_trial_config_saves_component_folds_and_additive_pooled_scores(synthetic, monkeypatch):
    model, hp, datasets, _ = synthetic
    config = TrialPlacecellRegressionConfig(model_name="external_placefield_1d_structured_gain_residual")
    registry = SimpleNamespace(registry_params=config.data_config.to_registry_params())
    session = SimpleNamespace(
        mouse_name="mouse",
        session_uid="mouse.date.1",
        environments=np.array([1]),
        trial_environment=np.ones(8, dtype=int),
    )
    optimized = []

    def best(*args, **kwargs):
        optimized.append(kwargs)
        return hp, 0, None

    monkeypatch.setattr(model, "get_best_hyperparameters", best)
    monkeypatch.setattr("dimensionality_manuscript.configs.trial_regression.load_env_order", lambda: {"mouse": [1]})
    monkeypatch.setattr(
        "dimensionality_manuscript.configs.trial_regression._environment_placecell_rois",
        lambda *args: dict(roi_indices=np.arange(4), source_roi_indices=np.arange(2), target_roi_indices=np.arange(2, 4)),
    )
    monkeypatch.setattr("dimensionality_manuscript.configs.trial_regression.EnvironmentTrialRegistry", lambda *args, **kwargs: registry)
    monkeypatch.setattr("dimensionality_manuscript.configs.trial_regression.get_model", lambda *args, **kwargs: model)
    result = config.process(session, registry)
    assert len(optimized) == 1
    assert np.isfinite(result["r2_gain_folds"][0]).all()
    assert np.isfinite(result["r2_residual_folds"][0]).all()
    assert result["residual_effective_rank"][0] == 2
    assert result["r2_pooled"][0] == pytest.approx(1)
    assert (result["r2_template_pooled"][0] + result["delta_r2_gain_pooled"][0] + result["delta_r2_residual_pooled"][0]) == pytest.approx(
        result["r2_pooled"][0]
    )
    assert result["sst_activity"][0] == pytest.approx(result["sst_activity_folds"][0].sum())


@pytest.mark.parametrize("internal", [False, True])
def test_registry_cache_and_schema_identity(internal):
    prefix = "internal" if internal else "external"
    name = f"{prefix}_placefield_1d_structured_gain_residual"
    config = TrialPlacecellRegressionConfig(model_name=name)
    assert config.schema_version == "v2"
    assert MODEL_NAMES.index(name) >= 50
    registry = SimpleNamespace(registry_params=config.data_config.to_registry_params())
    new = get_model(name, registry)
    old = get_model(f"{prefix}_placefield_1d_structured_gain", registry)
    session = SimpleNamespace(session_name=("mouse", "date", "1"))
    assert new._get_model_name() == name
    assert new._get_hyperparameter_cache_key(session, "sigrebase") != old._get_hyperparameter_cache_key(session, "sigrebase")
    assert new._get_score_cache_key(session, "sigrebase") != old._get_score_cache_key(session, "sigrebase")
    assert new.regressor_dimensionality() == get_regressor_dimensionality_from_hyperparameters(new.hyperparameters)
    baseline_name = old._get_model_name() if internal else "external_placefield_1d"
    assert config.key() != replace(config, model_name=baseline_name).key()
    assert "internal_placefield_1d_structured_gain_residual" in TrialPlacecellRegressionConfig._param_grid()["model_name"]
