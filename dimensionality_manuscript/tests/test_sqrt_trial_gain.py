"""Square-root gain regression must reconstruct activity in the original units."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from dimensionality_manuscript.configs.trial_regression import (
    STRUCTURED_GAIN_MODELS,
    TrialPlacecellRegressionConfig,
)
from dimensionality_manuscript.regression_models.models import PlaceFieldStructuredGainModel
from dimensionality_manuscript.registry import MODEL_METADATA, get_model, short_model_name
from dimensionality_manuscript.figure_scripts.figure2.trial_performance import (
    SQRT_GAIN_MODEL_NAME,
    TRIAL_PERFORMANCE_MODEL_NAMES,
    TrialModelPerformanceViewer,
)


def test_sqrt_gain_fit_and_held_out_activity_reconstruction(monkeypatch):
    """A linear relation in sqrt space generalizes through the full gain pipeline."""
    rng = np.random.default_rng(23)
    source_root = rng.uniform(0.2, 2.0, (12, 2))
    weights = np.array([[0.4, 0.2], [0.3, 0.6]])
    intercept = np.array([0.3, 0.1])
    target_root = source_root @ weights + intercept
    behavior = SimpleNamespace(trial=np.repeat(np.arange(12), 2))
    source = torch.tensor(np.repeat(source_root.T**2, 2, axis=1))
    target = torch.tensor(np.repeat(target_root.T**2, 2, axis=1))
    model = PlaceFieldStructuredGainModel(
        None, gain_transform="sqrt", gain_regularization=0, min_frames_per_unit=1,
    )
    monkeypatch.setattr(model, "get_session_data", lambda *args: (source, target, behavior))
    monkeypatch.setattr(model, "get_split_chunk_index", lambda *args: np.zeros(24))
    monkeypatch.setattr(
        "dimensionality_manuscript.regression_models.models.get_placefield_prediction",
        lambda *args: (np.ones((24, 2)), None),
    )
    fitted = model.fit_gain_model(None, None, None, alpha=0)

    held_out_root = np.array([[0.0, 0.8], [1.2, 1.8], [0.6, 0.0]])
    source_gain = held_out_root.T**2
    expected_gain = (held_out_root @ weights + intercept).T**2
    unit_index = np.repeat(np.arange(3), 2)
    placefield = rng.uniform(0.2, 3.0, (2, 6))
    inputs = dict(
        source_gain=source_gain, target_prediction=placefield, unit_index=unit_index,
        frame_behavior=None, idx_valid_prediction=np.arange(6), was_filtered=False,
    )
    prediction, extras = model.apply_gain_model(inputs, fitted, rank=2)

    np.testing.assert_allclose(extras["gain_predicted"], expected_gain, atol=1e-10)
    np.testing.assert_allclose(prediction, placefield * expected_gain[:, unit_index], atol=1e-10)
    np.testing.assert_array_equal(extras["gain_source"], source_gain)
    np.testing.assert_array_equal(inputs["source_gain"], source_gain)
    assert extras["predictions_were_filtered"] is False


@pytest.mark.parametrize("transform", ["raw", "sqrt"])
def test_negative_predictions_are_clipped_before_inverse_transform(transform):
    class FixedRegression:
        max_rank = 1

        def predict(self, x, *, rank, nonnegative):
            np.testing.assert_allclose(x.numpy(), [[0.0], [2.0 if transform == "sqrt" else 4.0]])
            assert nonnegative
            assert rank == 1
            return torch.clamp(torch.tensor([[-2.0], [3.0]]), min=0)

    inputs = dict(
        source_gain=np.array([[0.0, 4.0]]), target_prediction=np.array([[2.0, 5.0]]),
        unit_index=np.array([0, 1]), frame_behavior=None,
        idx_valid_prediction=np.arange(2), was_filtered=False,
    )
    model = PlaceFieldStructuredGainModel(None, gain_transform=transform)
    prediction, extras = model.apply_gain_model(inputs, FixedRegression(), rank=100)
    expected_gain = np.array([[0.0, 9.0 if transform == "sqrt" else 3.0]])
    np.testing.assert_array_equal(extras["gain_predicted"], expected_gain)
    np.testing.assert_array_equal(prediction, inputs["target_prediction"] * expected_gain)


def test_sqrt_gain_rejects_negative_measurements_and_unknown_transforms():
    with pytest.raises(ValueError, match="gain_transform"):
        PlaceFieldStructuredGainModel(None, gain_transform="log")
    with pytest.raises(ValueError, match="nonnegative measured gains"):
        PlaceFieldStructuredGainModel(None, gain_transform="sqrt")._transform_gain(np.array([-0.1]))


@pytest.mark.parametrize("internal", [False, True])
def test_registry_and_cache_identities_separate_raw_and_sqrt(internal):
    prefix = "internal" if internal else "external"
    raw_name = f"{prefix}_placefield_1d_structured_gain"
    sqrt_name = raw_name + "_sqrt"
    registry = SimpleNamespace(registry_params=TrialPlacecellRegressionConfig().data_config.to_registry_params())
    session = SimpleNamespace(session_name=("mouse", "2026-10-26", "1"))
    raw = get_model(raw_name, registry)
    sqrt = get_model(sqrt_name, registry)
    assert raw.gain_transform == "raw"
    assert sqrt.gain_transform == "sqrt"
    assert raw._get_model_name() == raw_name
    assert sqrt._get_model_name() == sqrt_name
    assert raw._get_hyperparameter_cache_key(session, "sigrebase") != sqrt._get_hyperparameter_cache_key(session, "sigrebase")
    assert raw._get_score_cache_key(session, "sigrebase") != sqrt._get_score_cache_key(session, "sigrebase")
    assert MODEL_METADATA[sqrt_name].reference == raw_name
    assert "Sqrt" in short_model_name(sqrt_name)


def test_trial_grid_includes_both_gain_models_with_distinct_result_keys():
    configs = {c.model_name: c for c in TrialPlacecellRegressionConfig.generate_variations()}
    raw = configs["internal_placefield_1d_structured_gain"]
    sqrt = configs[SQRT_GAIN_MODEL_NAME]
    assert raw.key() != sqrt.key()
    assert SQRT_GAIN_MODEL_NAME in STRUCTURED_GAIN_MODELS


def test_trial_viewer_compares_sqrt_activity_scores_without_changing_defaults():
    class Results:
        config_class = TrialPlacecellRegressionConfig
        param_axes = {"model_name": [*TRIAL_PERFORMANCE_MODEL_NAMES, SQRT_GAIN_MODEL_NAME]}
        mouse_names = np.array(["mouse"])

        def sel(self, *, model_name, **kwargs):
            value = 0.2 if model_name == SQRT_GAIN_MODEL_NAME else 0.1
            return {"r2": np.array([[value, value]])}

    viewer = TrialModelPerformanceViewer(Results(), include_sqrt_gain=True)
    assert viewer.model_names[-2] == SQRT_GAIN_MODEL_NAME
    assert viewer.model_labels[-2] == "Trial Gain\n(sqrt)"
    np.testing.assert_allclose(viewer._scores[-2], [0.2])
    fig = viewer.plot(viewer.state)
    assert "Trial Gain\n(sqrt)" in [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    plt.close(fig)
    assert tuple(TrialModelPerformanceViewer(Results()).model_names) == TRIAL_PERFORMANCE_MODEL_NAMES
