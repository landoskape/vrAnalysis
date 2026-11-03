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
    template_projection_energy,
)
from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldGainResidualHyperparameters
from dimensionality_manuscript.regression_models.models import (
    PlaceFieldModel,
    PlaceFieldStructuredGainModel,
    get_regressor_dimensionality_from_hyperparameters,
)
from dimensionality_manuscript.regression_models.base import RegressionModel
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

    def __len__(self):
        return len(self.trial)


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
    monkeypatch.setattr(
        "dimensionality_manuscript.regression_models.models.get_placefield_prediction",
        lambda pf, behavior: (templates[behavior.split][pf][:, behavior.idx].T, None),
    )
    hp = PlaceFieldGainResidualHyperparameters(alpha=0, rank=2, alpha_residual=0, rank_residual=2)
    return model, hp, datasets, templates


def test_regularized_decomposition_and_target_independent_prediction(synthetic):
    model, hp, datasets, _ = synthetic
    inputs = model.decomposition_inputs(None, "target", "source", split="train", include_target=True)
    source, target, _ = datasets["train"]
    np.testing.assert_allclose(inputs["target_gain_activity"] + inputs["target_residual"], target)
    assert model.gain_regularization == 0.01
    projection = template_projection_energy(inputs["target_residual"], inputs["target_prediction"], inputs["unit_index"])
    assert 0 < projection["gain_energy_fraction"] < 1
    coefficients = model.train(None, hyperparameters=hp)
    expected = model.predict(None, coefficients, hyperparameters=hp)[0]
    # A held-out target full of NaN cannot alter predictions or select test frames.
    datasets["test"] = (datasets["test"][0], torch.full_like(datasets["test"][1], torch.nan), datasets["test"][2])
    prediction, extras = model.predict(None, coefficients, hyperparameters=hp)
    np.testing.assert_allclose(prediction, expected, atol=1e-11)
    np.testing.assert_allclose(prediction, extras["gain_prediction"] + extras["residual_prediction"])
    assert np.any(extras["residual_prediction"] < 0)
    assert extras["gain_source"].shape == (2, 6)
    assert extras["residual_source"].shape == (2, 36)


def test_target_frame_filter_preserves_baseline_source_gain_measurements(synthetic, monkeypatch):
    model, _, datasets, templates = synthetic
    templates["test"]["target"][:, :6] = np.nan
    templates["test"]["source"][:, 8] = np.nan
    inputs = model.decomposition_inputs(None, "target", "source", split="test", include_target=True)
    assert inputs["num_units"] == 6
    assert inputs["was_filtered"]
    expected = np.setdiff1d(np.arange(36), [0, 1, 2, 3, 4, 5, 8])
    np.testing.assert_array_equal(inputs["idx_valid_prediction"], expected)
    np.testing.assert_allclose(
        inputs["target_gain_activity"] + inputs["target_residual"],
        datasets["test"][1][:, expected],
    )
    reference = PlaceFieldStructuredGainModel(None, min_frames_per_unit=1)
    monkeypatch.setattr(reference, "get_session_data", model.get_session_data)
    monkeypatch.setattr(reference, "get_split_chunk_index", model.get_split_chunk_index)
    original_inputs = reference.prediction_inputs(None, "target", "source", split="test")
    np.testing.assert_array_equal(inputs["source_gain"], original_inputs["source_gain"])
    with pytest.raises(ValueError, match="nan"):
        model.decomposition_inputs(SimpleNamespace(session_print=lambda: "synthetic"), "target", "source", split="test", nan_safe=True)
    coefficients = model.train(None, hyperparameters=PlaceFieldGainResidualHyperparameters())
    prediction, extras = model.predict(None, coefficients)
    assert prediction.shape == (2, len(expected))
    np.testing.assert_array_equal(extras["idx_valid_predictions"], expected)
    reference_prediction, reference_extras = reference.predict(None, coefficients[:3], hyperparameters=model.hyperparameters)
    np.testing.assert_array_equal(extras["gain_prediction"], reference_prediction)
    np.testing.assert_array_equal(extras["idx_valid_predictions"], reference_extras["idx_valid_predictions"])


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
    monkeypatch.setattr("dimensionality_manuscript.regression_models.models.get_placefield_prediction", lookup)
    inputs = model.decomposition_inputs(None, "target", "source", split="train", include_target=True)
    assert [pf for pf, _ in calls] == ["source", "target", "source"]
    assert all(np.all(positions == 65) for _, positions in calls)
    assert np.all(inputs["frame_behavior_internal"].position == 65)
    assert not np.all(inputs["frame_behavior"].position == 65)


@pytest.mark.parametrize("method", ["preferred", "golden", "grid", "optuna"])
def test_validation_inherits_gain_hyperparameters_and_tunes_only_residual(synthetic, monkeypatch, method):
    model, hp, datasets, _ = synthetic
    datasets.pop("test")
    inherited = []

    def reference_parameters(*args):
        inherited.append(args)
        return hp

    monkeypatch.setattr(model, "inherited_gain_hyperparameters", reference_parameters)

    def small_search(func, a, b, **kwargs):
        candidates = [a, b]
        scores = [func(value) for value in candidates]
        return candidates[int(np.argmin(scores))], min(scores)

    monkeypatch.setattr("dimensionality_manuscript.regression_models.gain_residual.golden_section_search", small_search)
    params, score, table = model.optimize(None, "sigrebase", method=method)
    assert len(inherited) == 1 and inherited[0][-1] == ("golden" if method == "preferred" else method)
    assert params["alpha"] == hp.alpha and params["rank"] == hp.rank
    assert np.all(table.alpha == hp.alpha) and np.all(table["rank"] == hp.rank)
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
    assert result["r2_pooled"][0] > 0.99
    assert (result["r2_template_pooled"][0] + result["delta_r2_gain_pooled"][0] + result["delta_r2_residual_pooled"][0]) == pytest.approx(
        result["r2_pooled"][0]
    )
    assert result["sst_activity"][0] == pytest.approx(result["sst_activity_folds"][0].sum())
    for prefix in ("source_residual", "target_residual", "predicted_residual"):
        energy = result[prefix + "_energy"][0]
        gain_energy = result[prefix + "_gain_energy"][0]
        assert energy == pytest.approx(result[prefix + "_energy_folds"][0].sum())
        assert gain_energy + result[prefix + "_off_template_energy"][0] == pytest.approx(energy)
        assert result[prefix + "_gain_energy_fraction"][0] == pytest.approx(gain_energy / energy)


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
    old_new_cache_key = RegressionModel._get_hyperparameter_cache_key(new, session, "sigrebase")
    assert new._get_hyperparameter_cache_key(session, "sigrebase") == old_new_cache_key + "_regularized_v2"
    assert "internal_placefield_1d_structured_gain_residual" in TrialPlacecellRegressionConfig._param_grid()["model_name"]


@pytest.mark.parametrize("scale", [1.0, 1e-200])
def test_posthoc_projection_is_exact_and_stable_for_tiny_templates(scale):
    template = np.array([[1.0, 2.0], [0.0, 0.0]]) * scale
    activity = np.array([[1.0, 7.0], [2.0, -3.0]])
    result = template_projection_energy(activity, template, np.array([0, 0]))
    assert result["gain_energy"] == pytest.approx(45)
    assert result["off_template_energy"] == pytest.approx(18)
    assert result["gain_energy"] + result["off_template_energy"] == pytest.approx(result["energy"])
    assert result["gain_energy_fraction"] == pytest.approx(45 / 63)


@pytest.mark.parametrize("internal", [False, True])
def test_gain_predictions_match_standalone_model_exactly(synthetic, monkeypatch, internal):
    model, hp, _, _ = synthetic
    model.internal = internal
    reference = PlaceFieldStructuredGainModel(None, internal=internal, min_frames_per_unit=1)
    monkeypatch.setattr(reference, "get_session_data", model.get_session_data)
    monkeypatch.setattr(reference, "get_split_chunk_index", model.get_split_chunk_index)
    if internal:
        monkeypatch.setattr(model, "decode_internal_frame_behavior", lambda source, pf, behavior: behavior)
        monkeypatch.setattr(reference, "decode_internal_frame_behavior", model.decode_internal_frame_behavior)
    baseline_coefficients = reference.train(None, hyperparameters=hp)
    coefficients = model.train(None, hyperparameters=hp)
    baseline, _ = reference.predict(None, baseline_coefficients, hyperparameters=hp)
    _, extras = model.predict(None, coefficients, hyperparameters=hp)
    np.testing.assert_array_equal(extras["gain_prediction"], baseline)


def test_reference_optimization_uses_identical_settings(synthetic, monkeypatch):
    model, hp, _, _ = synthetic
    model.internal = True
    calls = []

    def best(reference, session, **kwargs):
        assert type(reference) is PlaceFieldStructuredGainModel
        assert reference.internal and reference.gain_regularization == model.gain_regularization
        assert reference.activity_parameters == model.activity_parameters
        assert reference.min_frames_per_unit == model.min_frames_per_unit
        calls.append(kwargs)
        return hp, 0, None

    monkeypatch.setattr(PlaceFieldStructuredGainModel, "get_best_hyperparameters", best)
    assert model.inherited_gain_hyperparameters(None, "sigrebase", "train", "validation", "golden") is hp
    assert calls == [dict(spks_type="sigrebase", train_split="train", validation_split="validation", method="golden")]


def test_revised_models_keep_one_schema_and_support_targeted_invalidation():
    variations = TrialPlacecellRegressionConfig.generate_variations()
    assert {cfg.schema_version for cfg in variations} == {"v2"}
    assert len(variations) == 9
    for prefix in ("internal", "external"):
        name = f"{prefix}_placefield_1d_structured_gain_residual"
        configs = TrialPlacecellRegressionConfig.generate_variations_matching({"model_name": name, "schema_version": "v2"})
        assert len(configs) == 1 and configs[0].model_name == name
        assert configs[0].schema_version == "v2"
    baseline = TrialPlacecellRegressionConfig(model_name="rrr")
    assert baseline.key() == replace(baseline, schema_version="v2").key()
    with pytest.raises(ValueError, match="supports schema_version"):
        TrialPlacecellRegressionConfig(model_name="internal_placefield_1d_structured_gain_residual", schema_version="v3")
    with pytest.raises(ValueError, match="one reference optimization method"):
        TrialPlacecellRegressionConfig(model_name="internal_placefield_1d_structured_gain_residual", method="best")
