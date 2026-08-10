"""Tests for the place-field + structured-additive model.

The claim the model makes is narrow and checkable without session data: the second stage sees the
place-field *residual* on both sides, and its prediction is added back to the target place field
rather than multiplied into it. So the tests here drive :meth:`fit_residual_model` and
:meth:`apply_residual_model` with plain arrays -- including a noiseless low-rank residual, which
the regression must recover exactly -- plus the bookkeeping (rank clipping, rectification, dropped
frames) that keeps a prediction aligned with its target.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldStructuredAdditiveHyperparameters
from dimensionality_manuscript.regression_models.models import PlaceFieldStructuredAdditiveModel


class _FakeSplitPopulation:
    """Stands in for ``Population``, returning canned split sample indices."""

    def __init__(self, split_times: dict[object, np.ndarray]):
        self._split_times = split_times

    def get_split_times(self, time_idx, within_idx_samples: bool = True) -> np.ndarray:
        return self._split_times[time_idx]


class _FakeRegistry:
    """Stands in for ``PopulationRegistry``."""

    def __init__(self, split_times: dict[str, np.ndarray], num_buffer: int = 3):
        self.registry_params = SimpleNamespace(time_split_num_buffer=num_buffer)
        self.time_split = {name: name for name in split_times}
        self._population = _FakeSplitPopulation(split_times)

    def get_population(self, session, spks_type=None):
        return self._population, None


class _FakeFrameBehavior:
    """Stands in for ``FrameBehavior``; only ``filter`` is exercised here."""

    def __init__(self, frames: np.ndarray):
        self.frames = np.asarray(frames)

    def filter(self, idx) -> "_FakeFrameBehavior":
        return _FakeFrameBehavior(self.frames[idx])


def _model(**kwargs) -> PlaceFieldStructuredAdditiveModel:
    return PlaceFieldStructuredAdditiveModel(_FakeRegistry({"test": np.arange(4)}), **kwargs)


def _low_rank_residuals(
    num_source: int = 20,
    num_target: int = 8,
    num_frames: int = 400,
    rank: int = 3,
    placefield_level: float = 50.0,
    seed: int = 0,
) -> dict:
    """Residuals of an exactly rank-``rank`` shared subspace, on a positive place field.

    ``placefield_level`` sits well above the residual scale by default, so the default
    rectification never bites and the additive arithmetic can be checked exactly. The
    rectification tests zero it out on purpose.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(rank, num_frames))
    return {
        "target_prediction": np.abs(rng.normal(placefield_level, 0.3, size=(num_target, num_frames))),
        "source_residual": (rng.normal(size=(num_source, rank)) @ latent).astype(np.float32),
        "target_residual": (rng.normal(size=(num_target, rank)) @ latent).astype(np.float32),
        "frame_behavior": _FakeFrameBehavior(np.arange(num_frames)),
        "idx_valid_prediction": np.arange(num_frames),
        "was_filtered": False,
    }


# ---------------------------------------------------------------------------
# The residual regression
# ---------------------------------------------------------------------------


def test_a_noiseless_low_rank_residual_is_recovered_at_its_own_rank():
    """The point of the model: the source residual carries the target residual, no SVD needed."""
    model = _model()
    residuals = _low_rank_residuals(rank=3)

    residual_model = model.fit_residual_model(residuals, alpha=1e-3)
    _, extras = model.apply_residual_model(residuals, residual_model, rank=3)

    assert np.allclose(extras["residual_predicted"], residuals["target_residual"], atol=1e-3)


def test_prediction_is_the_place_field_plus_the_predicted_residual():
    """Additive, not multiplicative -- and the place field is passed through untouched."""
    model = _model()
    residuals = _low_rank_residuals()
    residual_model = model.fit_residual_model(residuals, alpha=1.0)

    prediction, extras = model.apply_residual_model(residuals, residual_model, rank=2)

    assert np.allclose(prediction, residuals["target_prediction"] + extras["residual_predicted"], atol=1e-4)
    assert np.allclose(extras["placefield_prediction"], residuals["target_prediction"])
    assert extras["predictions_were_filtered"] is False


def test_truncating_the_rank_degrades_the_fit_monotonically():
    """Rank is a real knob on the residual map, not a no-op."""
    model = _model()
    residuals = _low_rank_residuals(rank=5)
    residual_model = model.fit_residual_model(residuals, alpha=1e-3)

    errors = []
    for rank in (1, 2, 3, 5):
        _, extras = model.apply_residual_model(residuals, residual_model, rank=rank)
        errors.append(float(np.mean((extras["residual_predicted"] - residuals["target_residual"]) ** 2)))

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-6


def test_requested_rank_is_clipped_to_what_the_regression_supports():
    """The sweep asks for ranks up to 2000; the map is bounded by the smaller population."""
    model = _model()
    residuals = _low_rank_residuals(num_source=20, num_target=8)
    residual_model = model.fit_residual_model(residuals, alpha=1.0)

    assert residual_model.max_rank == 8
    prediction, _ = model.apply_residual_model(residuals, residual_model, rank=2000)
    assert prediction.shape == residuals["target_prediction"].shape
    assert np.all(np.isfinite(prediction))


def test_max_residual_rank_matches_the_bound_the_regression_applies():
    residuals = _low_rank_residuals(num_source=20, num_target=8, num_frames=400)
    assert _model().max_residual_rank(residuals) == 8
    assert _model(fit_intercept=False).max_residual_rank(residuals) == 8

    narrow = _low_rank_residuals(num_source=3, num_target=50, num_frames=400)
    assert _model().max_residual_rank(narrow) == 4
    assert _model(fit_intercept=False).max_residual_rank(narrow) == 3


# ---------------------------------------------------------------------------
# Rectification
# ---------------------------------------------------------------------------


def test_the_summed_prediction_is_rectified_but_the_offset_is_not():
    """A residual must be free to go negative; a firing rate must not."""
    model = _model()
    residuals = _low_rank_residuals()
    residual_model = model.fit_residual_model(residuals, alpha=1.0)
    # A place field small enough that the offset drives the sum negative somewhere.
    negative = {**residuals, "target_prediction": np.zeros_like(residuals["target_prediction"])}

    prediction, extras = model.apply_residual_model(negative, residual_model, rank=3)
    assert np.all(prediction >= 0.0)
    assert np.any(extras["residual_predicted"] < 0.0)


def test_rectification_can_be_turned_off():
    model = _model(nonnegative=False)
    residuals = _low_rank_residuals()
    residual_model = model.fit_residual_model(residuals, alpha=1.0)
    negative = {**residuals, "target_prediction": np.zeros_like(residuals["target_prediction"])}

    prediction, _ = model.apply_residual_model(negative, residual_model, rank=3)
    assert np.any(prediction < 0.0)


# ---------------------------------------------------------------------------
# Filtering bookkeeping
# ---------------------------------------------------------------------------


def test_filtering_upstream_of_the_residual_stage_is_reported_to_the_caller():
    """Frames dropped for an undefined place field must still be reported, or targets misalign."""
    model = _model()
    residuals = _low_rank_residuals(num_frames=4)
    residual_model = model.fit_residual_model(_low_rank_residuals(), alpha=1.0)
    filtered = {**residuals, "idx_valid_prediction": np.array([1, 3, 5, 7]), "was_filtered": True}

    _, extras = model.apply_residual_model(filtered, residual_model, rank=2)
    assert extras["predictions_were_filtered"] is True
    assert extras["idx_valid_predictions"].tolist() == [1, 3, 5, 7]


def test_nan_in_the_place_field_prediction_drops_the_frame_and_reports_it():
    model = _model()
    residuals = _low_rank_residuals(num_frames=4)
    residual_model = model.fit_residual_model(_low_rank_residuals(), alpha=1.0)
    residuals["target_prediction"][0, 2] = np.nan

    prediction, extras = model.apply_residual_model(residuals, residual_model, rank=2)
    assert prediction.shape[1] == 3
    assert extras["predictions_were_filtered"] is True
    assert extras["idx_valid_predictions"].tolist() == [0, 1, 3]
    assert extras["residual_predicted"].shape[1] == 3


def test_nan_safe_raises_instead_of_dropping():
    model = _model()
    residuals = _low_rank_residuals(num_frames=4)
    residual_model = model.fit_residual_model(_low_rank_residuals(), alpha=1.0)
    residuals["target_prediction"][0, 2] = np.nan

    with pytest.raises(ValueError, match="NaN values in prediction"):
        model.apply_residual_model(residuals, residual_model, rank=2, nan_safe=True)


# ---------------------------------------------------------------------------
# Hyperparameters and naming
# ---------------------------------------------------------------------------


def test_dimensionality_counts_spatial_bins_and_residual_latents():
    hyperparameters = PlaceFieldStructuredAdditiveHyperparameters(num_bins=40, rank=7)
    assert _model().regressor_dimensionality(3, hyperparameters=hyperparameters) == 40 * 3 + 7


def test_hyperparameters_round_trip_through_the_optimizer_dict_form():
    hyperparameters = PlaceFieldStructuredAdditiveHyperparameters(num_bins=25, smooth_width=None, rank=4, alpha=12.5)
    assert PlaceFieldStructuredAdditiveHyperparameters.from_dict(dict(vars(hyperparameters))) == hyperparameters


def test_hyperparameters_survive_dataclasses_replace():
    """RegressionDimensionalitySweepConfig sweeps num_bins by replacing fields in place."""
    hyperparameters = PlaceFieldStructuredAdditiveHyperparameters(num_bins=100, smooth_width=1.0, rank=9, alpha=50.0)
    swept = replace(hyperparameters, num_bins=10, smooth_width=2.0)
    assert (swept.num_bins, swept.smooth_width) == (10, 2.0)
    assert (swept.rank, swept.alpha) == (9, 50.0)


def test_search_space_separates_training_and_prediction_hyperparameters():
    """Rank is a prediction-time knob; everything else changes the fit."""
    space = PlaceFieldStructuredAdditiveHyperparameters.get_search_space()
    assert set(space) == {"training", "prediction"}
    assert set(space["training"]) == {"num_bins", "smooth_width", "alpha"}
    assert set(space["prediction"]) == {"rank"}


def test_model_names_follow_the_registry_convention():
    assert _model()._get_model_name() == "external_placefield_1d_structured_additive"
    assert _model(internal=True)._get_model_name() == "internal_placefield_1d_structured_additive"
    assert _model(fit_intercept=False)._get_model_name() == "external_placefield_1d_structured_additive_no_intercept"
