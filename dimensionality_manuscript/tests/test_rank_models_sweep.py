"""Tests for the regression-rank sweep config.

The sweep never rebuilds reduced-rank coefficients per rank -- it projects the model's own
full-rank regression prediction onto the leading columns of that prediction's basis. What is new
here relative to ``test_structured_additive_rank`` is the RRR branch, which reaches past
``ReducedRankRegressionModel.predict`` to the fitted regression precisely because that method
rectifies, and a rectified prediction is not what the basis diagonalizes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dimilibi import ReducedRankRegression

from dimensionality_manuscript.configs import ANALYSIS_CONFIG_CLASSES
from dimensionality_manuscript.configs.regression import (
    RANK_MODELS,
    RankModelsSweepConfig,
    _prediction_basis,
    _rank_grid,
    _score_rank_projection,
)


def _fitted_regression(num_frames: int = 200, num_sources: int = 12, num_targets: int = 8):
    """A reduced-rank regression on random data with a signed target, plus the frames it saw."""
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


def test_projecting_the_unrectified_prediction_reproduces_every_lower_rank():
    model, source = _fitted_regression()
    full = model.predict(source, rank=model.max_rank).numpy().T
    basis = _prediction_basis(full)
    assert basis.shape[1] == model.max_rank

    for rank in range(1, model.max_rank + 1):
        expected = model.predict(source, rank=rank).numpy().T
        projected = basis[:, :rank] @ (basis[:, :rank].T @ full)
        np.testing.assert_allclose(projected, expected, atol=1e-4)


def test_projecting_a_rectified_prediction_does_not_reproduce_lower_ranks():
    # Why the RRR branch bypasses ReducedRankRegressionModel.predict: the ReLU is nonlinear, so the
    # rectified prediction is not the matrix the basis diagonalizes and the projection identity is
    # lost. Rectification has to happen once per rank, after the projection.
    model, source = _fitted_regression()
    rectified = model.predict(source, rank=model.max_rank, nonnegative=True).numpy().T
    basis = _prediction_basis(rectified)

    rank = 1
    expected = np.clip(model.predict(source, rank=rank).numpy().T, 0, None)
    projected = np.clip(basis[:, :rank] @ (basis[:, :rank].T @ rectified), 0, None)
    assert not np.allclose(projected, expected, atol=1e-3)


class _MetricModel:
    """Minimal stand-in exposing the ``evaluate`` the sweep calls."""

    @staticmethod
    def evaluate(prediction, target):
        return {"mse": float(np.mean((prediction - target) ** 2)), "r2": np.nan}


def test_rrr_path_scores_the_rectified_projection_with_no_offset():
    model, source = _fitted_regression()
    full = model.predict(source, rank=model.max_rank).numpy().T
    basis = _prediction_basis(full)
    target = np.zeros_like(full)
    ranks = _rank_grid(basis.shape[1])

    mse_arr, _ = _score_rank_projection(_MetricModel, basis, full, target, ranks, offset=None)

    for i, rank in enumerate(ranks):
        expected = np.clip(model.predict(source, rank=int(rank)).numpy().T, 0, None)
        assert mse_arr[i] == pytest.approx(float(np.mean(expected**2)), rel=1e-4)


def test_config_is_registered_under_its_display_name():
    assert ANALYSIS_CONFIG_CLASSES["rank_models_sweep"] is RankModelsSweepConfig


def test_grid_sweeps_both_structured_additive_models_and_rrr():
    assert set(RankModelsSweepConfig._param_grid()) == {"model_name"}
    variations = RankModelsSweepConfig.generate_variations()
    assert [cfg.model_name for cfg in variations] == RANK_MODELS
    assert "rrr" in RANK_MODELS
    assert {cfg.activity_parameters_name for cfg in variations} == {"std"}
    assert {cfg.spks_type for cfg in variations} == {"sigrebase"}


def test_validate_rejects_models_without_a_reduced_rank_regression():
    with pytest.raises(ValueError, match="model_name"):
        RankModelsSweepConfig(model_name="external_placefield_1d")
    with pytest.raises(ValueError, match="activity_parameters_name"):
        RankModelsSweepConfig(activity_parameters_name="nonexistent")


def test_summary_distinguishes_every_model():
    summaries = {cfg.summary() for cfg in RankModelsSweepConfig.generate_variations()}
    assert len(summaries) == len(RANK_MODELS)
    assert all(s.startswith("rank_models_sweep_") for s in summaries)
    assert all(s.endswith(RankModelsSweepConfig.schema_version) for s in summaries)
