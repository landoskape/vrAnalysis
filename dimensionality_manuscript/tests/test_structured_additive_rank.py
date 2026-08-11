"""Tests for the structured-additive rank sweep config.

The sweep never calls ``ReducedRankRegression.predict`` per rank -- it projects the model's own
full-rank residual prediction onto the leading columns of that prediction's basis. That identity is
the whole reason the config is cheap enough to run densely, so it is what the tests here check,
along with the offset bookkeeping that keeps the place-field part of the prediction out of the
projection.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dimilibi import ReducedRankRegression

from dimensionality_manuscript.configs import ANALYSIS_CONFIG_CLASSES
from dimensionality_manuscript.configs.regression import (
    STRUCTURED_ADDITIVE_MODELS,
    StructuredAdditiveRankConfig,
    _prediction_basis,
    _score_rank_projection,
)


def _fitted_regression(num_frames: int = 200, num_sources: int = 12, num_targets: int = 8):
    """A reduced-rank regression fit on random data, plus the source frames it saw."""
    rng = np.random.default_rng(0)
    source = rng.normal(size=(num_frames, num_sources))
    weights = rng.normal(size=(num_sources, num_targets))
    target = source @ weights + 0.1 * rng.normal(size=(num_frames, num_targets))
    source_tensor = torch.tensor(source, dtype=torch.float32)
    model = ReducedRankRegression(alpha=1.0, fit_intercept=True).fit(
        source_tensor,
        torch.tensor(target, dtype=torch.float32),
    )
    return model, source_tensor


def test_projecting_the_full_rank_prediction_reproduces_every_lower_rank():
    model, source = _fitted_regression()
    # (targets, frames), the orientation the config works in.
    full = model.predict(source, rank=model.max_rank).numpy().T
    basis = _prediction_basis(full)
    assert basis.shape[1] == model.max_rank

    for rank in range(1, model.max_rank + 1):
        expected = model.predict(source, rank=rank).numpy().T
        projected = basis[:, :rank] @ (basis[:, :rank].T @ full)
        np.testing.assert_allclose(projected, expected, atol=1e-4)


class _MSEOnlyModel:
    """Minimal stand-in exposing the ``evaluate`` the sweep calls."""

    @staticmethod
    def evaluate(prediction, target):
        return {"mse": float(np.mean((prediction - target) ** 2)), "r2": np.nan}


def test_offset_is_added_back_before_rectifying():
    # A single target whose place-field prediction is well above zero and whose residual pulls it
    # below: rectifying the projection alone rather than the sum would give a different answer.
    basis = np.array([[1.0]])
    residual_prediction = np.array([[-3.0, 1.0]])
    offset = np.array([[2.0, 2.0]])
    target = np.array([[0.0, 3.0]])

    mse_arr, _ = _score_rank_projection(_MSEOnlyModel, basis, residual_prediction, target, np.array([1]), offset=offset)

    # clip(offset + residual) = [0, 3] -> exactly the target.
    assert mse_arr[0] == pytest.approx(0.0)


def test_omitting_the_offset_leaves_the_projection_only_behavior_unchanged():
    basis = np.array([[1.0]])
    prediction = np.array([[-3.0, 1.0]])
    target = np.array([[0.0, 1.0]])

    mse_arr, _ = _score_rank_projection(_MSEOnlyModel, basis, prediction, target, np.array([1]))

    # clip(prediction) = [0, 1], which is the target: the rank sweep's original contract.
    assert mse_arr[0] == pytest.approx(0.0)


def test_config_is_registered_under_its_display_name():
    assert ANALYSIS_CONFIG_CLASSES["structured_additive_rank"] is StructuredAdditiveRankConfig


def test_grid_sweeps_both_structured_additive_models_only():
    assert set(StructuredAdditiveRankConfig._param_grid()) == {"model_name"}
    variations = StructuredAdditiveRankConfig.generate_variations()
    assert [cfg.model_name for cfg in variations] == STRUCTURED_ADDITIVE_MODELS
    assert {cfg.activity_parameters_name for cfg in variations} == {"std"}
    assert {cfg.spks_type for cfg in variations} == {"sigrebase"}


def test_validate_rejects_models_without_a_residual_regression():
    with pytest.raises(ValueError, match="model_name"):
        StructuredAdditiveRankConfig(model_name="external_placefield_1d")
    with pytest.raises(ValueError, match="activity_parameters_name"):
        StructuredAdditiveRankConfig(activity_parameters_name="nonexistent")


def test_summary_distinguishes_the_two_models():
    summaries = {cfg.summary() for cfg in StructuredAdditiveRankConfig.generate_variations()}
    assert len(summaries) == len(STRUCTURED_ADDITIVE_MODELS)
    assert all(s.startswith("structured_additive_rank_") for s in summaries)
    assert all(s.endswith(StructuredAdditiveRankConfig.schema_version) for s in summaries)
