from __future__ import annotations

import inspect

import numpy as np
import pytest

from dimensionality_manuscript.configs import ANALYSIS_CONFIG_CLASSES
from dimensionality_manuscript.configs.regression import (
    RESIDUAL_STRUCTURE_KEYS,
    KEY_FIGURE_MODELS,
    RegressionResidualStructureConfig,
)
from dimensionality_manuscript.regression_models.models import least_squares_gain, make_gain_unit_index


def _multiplicative_residual(target: np.ndarray, prediction: np.ndarray, eps_scale: float) -> np.ndarray:
    """The multiplicative residual exactly as ``process`` computes it."""
    rectified = np.clip(prediction, 0, None)
    eps = eps_scale * np.mean(rectified, axis=1, keepdims=True)
    return np.divide(target, rectified + eps, out=np.full_like(target, np.nan), where=eps > 0)


def test_multiplicative_residual_uses_a_relative_per_neuron_floor():
    target = np.array([[2.0, 4.0, 6.0]])
    prediction = np.array([[5.0, 5.0, 5.0]])

    residual = _multiplicative_residual(target, prediction, eps_scale=0.01)

    np.testing.assert_allclose(residual, target / (5.0 + 0.05))


def test_multiplicative_residual_is_nan_when_a_neuron_predicts_nothing():
    target = np.array([[1.0, 2.0], [1.0, 2.0]])
    prediction = np.array([[0.0, 0.0], [1.0, 1.0]])

    residual = _multiplicative_residual(target, prediction, eps_scale=0.01)

    assert np.all(np.isnan(residual[0]))
    assert np.all(np.isfinite(residual[1]))


def test_multiplicative_residual_rectifies_negative_predictions():
    # Only the structured-gain model can predict below zero; the clip keeps the ratio's sign
    # meaningful instead of flipping it.
    target = np.array([[1.0, 1.0]])
    prediction = np.array([[-4.0, 4.0]])

    residual = _multiplicative_residual(target, prediction, eps_scale=0.01)

    assert np.all(residual > 0)


def test_chunk_gain_recovers_a_known_per_unit_scaling():
    # Two chunks x two trials -> four units, each scaling the prediction by a known factor.
    chunk_index = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    trial = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    unit_index, num_units = make_gain_unit_index(chunk_index, trial)
    assert num_units == 4

    prediction = np.tile(np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]), (2, 1))
    expected = np.array([[0.5, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5]])
    target = prediction * expected[:, unit_index]

    gain = least_squares_gain(target, prediction, unit_index, num_units, regularization=0.0)

    np.testing.assert_allclose(gain, expected)


def test_chunk_gain_regularization_pulls_an_unsupported_unit_to_one():
    unit_index = np.array([0, 0, 1, 1])
    prediction = np.array([[3.0, 3.0, 0.0, 0.0]])
    target = np.array([[6.0, 6.0, 4.0, 4.0]])

    gain = least_squares_gain(target, prediction, unit_index, num_units=2, regularization=0.01)

    assert gain[0, 0] == pytest.approx(2.0, rel=1e-2)
    assert gain[0, 1] == pytest.approx(1.0)


def test_every_returned_key_is_excluded_from_aggregation():
    handling = RegressionResidualStructureConfig._result_handling
    assert set(handling) == set(RESIDUAL_STRUCTURE_KEYS)
    assert set(handling.values()) == {"skip"}

    # Guard against a key being added to process() without being registered as skip: the returned
    # dict literal must not name anything outside RESIDUAL_STRUCTURE_KEYS.
    source = inspect.getsource(RegressionResidualStructureConfig.process)
    returned = {line.split('"')[1] for line in source.splitlines() if line.strip().startswith('"') and '":' in line}
    assert returned == set(RESIDUAL_STRUCTURE_KEYS)


def test_grid_sweeps_models_only():
    assert set(RegressionResidualStructureConfig._param_grid()) == {"model_name"}
    variations = RegressionResidualStructureConfig.generate_variations()
    assert [cfg.model_name for cfg in variations] == KEY_FIGURE_MODELS
    # The stored matrices are large, so the activity preset is pinned rather than swept.
    assert {cfg.activity_parameters_name for cfg in variations} == {"std"}


def test_config_is_registered_under_its_display_name():
    assert ANALYSIS_CONFIG_CLASSES["regression_residual_structure"] is RegressionResidualStructureConfig


def test_validate_rejects_nonpositive_eps_and_negative_regularization():
    with pytest.raises(ValueError, match="multiplicative_eps_scale"):
        RegressionResidualStructureConfig(multiplicative_eps_scale=0.0)
    with pytest.raises(ValueError, match="chunk_gain_regularization"):
        RegressionResidualStructureConfig(chunk_gain_regularization=-1.0)
