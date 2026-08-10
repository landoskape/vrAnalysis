"""Tests for the place-field + structured-gain model.

The pieces that carry the model's cross-validation guarantee are the gain unit definition
(a contiguous time chunk intersected with a trial) and the chunk index it is built from, so those
get the most attention here. Everything is exercised with plain arrays and light stand-ins for the
population registry -- no session data required.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dimilibi import ReducedRankRegression
from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldStructuredGainHyperparameters
from dimensionality_manuscript.regression_models.models import (
    PlaceFieldStructuredGainModel,
    least_squares_gain,
    make_gain_unit_index,
)


class _FakeSplitPopulation:
    """Stands in for ``Population``, returning canned split sample indices."""

    def __init__(self, split_times: dict[object, np.ndarray]):
        self._split_times = split_times

    def get_split_times(self, time_idx, within_idx_samples: bool = True) -> np.ndarray:
        return self._split_times[time_idx]


class _FakeRegistry:
    """Stands in for ``PopulationRegistry`` for chunk-index lookups."""

    def __init__(self, split_times: dict[str, np.ndarray], num_buffer: int = 3):
        self.registry_params = SimpleNamespace(time_split_num_buffer=num_buffer)
        self.time_split = {name: name for name in split_times}
        self._population = _FakeSplitPopulation(split_times)

    def get_population(self, session, spks_type=None):
        return self._population, None


def _model(registry) -> PlaceFieldStructuredGainModel:
    return PlaceFieldStructuredGainModel(registry)


# ---------------------------------------------------------------------------
# least_squares_gain
# ---------------------------------------------------------------------------


def test_gain_recovers_the_scale_factor_of_each_cell_and_unit():
    """When activity is a scaled place field, the gain is exactly that scale."""
    rng = np.random.default_rng(0)
    unit_index = np.repeat([0, 1, 2], 10)
    prediction = rng.uniform(0.5, 2.0, size=(4, 30))
    true_gain = np.array([[1.0, 0.5, 2.0], [0.25, 1.5, 1.0], [3.0, 1.0, 0.75], [1.0, 1.0, 1.0]])
    activity = prediction * true_gain[:, unit_index]

    gain = least_squares_gain(activity, prediction, unit_index, num_units=3)
    assert gain.shape == (4, 3)
    assert np.allclose(gain, true_gain)


def test_gain_ignores_frames_with_undefined_place_fields():
    """NaN place-field predictions (unvisited bins) drop out of both sums."""
    unit_index = np.zeros(4, dtype=np.int64)
    prediction = np.array([[1.0, 1.0, np.nan, 1.0]])
    activity = np.array([[2.0, 2.0, 1000.0, 2.0]])

    gain = least_squares_gain(activity, prediction, unit_index, num_units=1)
    assert gain == pytest.approx(2.0)


def test_gain_is_neutral_where_the_place_field_has_no_energy():
    """A cell with no place-field energy in a unit gets 1.0, not a divide-by-zero."""
    unit_index = np.zeros(3, dtype=np.int64)
    prediction = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    activity = np.array([[5.0, 5.0, 5.0], [3.0, 3.0, 3.0]])

    gain = least_squares_gain(activity, prediction, unit_index, num_units=1)
    assert gain[0, 0] == 1.0
    assert gain[1, 0] == pytest.approx(3.0)


def test_regularization_bounds_gains_that_would_otherwise_diverge():
    """A cell whose place field predicts almost nothing over a unit must not blow up.

    Unregularized this ratio reaches ~1e30 on real sessions -- large enough to make the gain
    regression non-finite once it is rounded to single precision.
    """
    unit_index = np.array([0, 0, 1, 1])
    # Unit 0 is well measured; unit 1 has a vanishing place-field prediction.
    prediction = np.array([[1.0, 1.0, 1e-18, 1e-18]])
    activity = np.array([[2.0, 2.0, 1.0, 1.0]])

    unregularized = least_squares_gain(activity, prediction, unit_index, num_units=2)
    assert unregularized[0, 1] > 1e15

    regularized = least_squares_gain(activity, prediction, unit_index, num_units=2, regularization=0.01)
    assert regularized[0, 1] < 1e3
    # The well-measured unit is barely touched.
    assert regularized[0, 0] == pytest.approx(2.0, rel=0.02)


def test_regularization_leaves_unmeasurable_units_neutral():
    """With no place-field energy at all, the shrinkage target itself is the answer."""
    unit_index = np.array([0, 0, 1, 1])
    prediction = np.array([[1.0, 1.0, 0.0, 0.0]])
    activity = np.array([[2.0, 2.0, 5.0, 5.0]])

    gain = least_squares_gain(activity, prediction, unit_index, num_units=2, regularization=0.01)
    assert gain[0, 1] == pytest.approx(1.0)


def test_regularization_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        least_squares_gain(np.ones((1, 2)), np.ones((1, 2)), np.zeros(2, dtype=np.int64), num_units=1, regularization=-1.0)


def test_gain_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        least_squares_gain(np.ones((2, 5)), np.ones((2, 4)), np.zeros(5, dtype=np.int64), num_units=1)
    with pytest.raises(ValueError):
        least_squares_gain(np.ones((2, 5)), np.ones((2, 5)), np.zeros(4, dtype=np.int64), num_units=1)


# ---------------------------------------------------------------------------
# Chunk detection and gain units
# ---------------------------------------------------------------------------


def test_chunk_index_splits_on_gaps_in_the_split_samples():
    """Consecutive samples are one chunk; a gap (the split buffer) starts a new one."""
    split_times = {"test": np.array([0, 1, 2, 7, 8, 9, 20, 21])}
    model = _model(_FakeRegistry(split_times))

    chunk_index = model.get_split_chunk_index(session=None, split="test")
    assert chunk_index.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]


def test_chunk_index_requires_a_buffer_between_chunks():
    """Without a buffer, adjacent chunks are indistinguishable from one chunk."""
    model = _model(_FakeRegistry({"test": np.arange(5)}, num_buffer=0))
    with pytest.raises(ValueError, match="buffer"):
        model.get_split_chunk_index(session=None, split="test")


def test_chunk_index_is_empty_for_an_empty_split():
    model = _model(_FakeRegistry({"test": np.array([], dtype=int)}))
    assert model.get_split_chunk_index(session=None, split="test").shape == (0,)


def test_gain_unit_splits_a_chunk_that_spans_a_trial_boundary():
    """One chunk covering two trials yields one unit per trial."""
    chunk_index = np.array([0, 0, 0, 0, 1, 1])
    trial = np.array([7.0, 7.0, 8.0, 8.0, 8.0, 8.0])

    unit_index, num_units = make_gain_unit_index(chunk_index, trial)
    assert num_units == 3
    # (chunk 0, trial 7), (chunk 0, trial 8), (chunk 1, trial 8)
    assert unit_index.tolist() == [0, 0, 1, 1, 2, 2]


def test_gain_unit_never_merges_the_same_trial_across_chunks():
    """A trial appearing in two chunks yields two units, so folds never share a unit."""
    unit_index, num_units = make_gain_unit_index(np.array([0, 0, 1, 1]), np.array([3.0, 3.0, 3.0, 3.0]))
    assert num_units == 2
    assert unit_index.tolist() == [0, 0, 1, 1]


def test_gain_unit_requires_a_trial_number_for_every_frame():
    with pytest.raises(ValueError, match="trial number"):
        make_gain_unit_index(np.array([0, 0]), np.array([1.0, np.nan]))


def test_gain_units_are_disjoint_between_splits():
    """The leak invariant: no gain unit draws frames from more than one split.

    Units are keyed by (chunk, trial) and chunks belong to exactly one split, so a trial spanning
    the train/test boundary contributes separate units to each -- which is the whole point of
    grouping by chunk rather than by trial.
    """
    # One trial (trial 4) spread over three chunks that land in different splits.
    split_times = {
        "train": np.array([0, 1, 2, 3, 10, 11, 12, 13]),
        "test": np.array([20, 21, 22, 23]),
    }
    registry = _FakeRegistry(split_times)
    model = _model(registry)
    trial = {"train": np.full(8, 4.0), "test": np.full(4, 4.0)}

    units = {}
    for split in ("train", "test"):
        chunk_index = model.get_split_chunk_index(session=None, split=split)
        unit_index, _ = make_gain_unit_index(chunk_index, trial[split])
        # Key units by the actual sample indices they cover, which is what "the same data" means.
        units[split] = {tuple(split_times[split][unit_index == unit]) for unit in np.unique(unit_index)}

    train_samples = {sample for unit in units["train"] for sample in unit}
    test_samples = {sample for unit in units["test"] for sample in unit}
    assert not (train_samples & test_samples)
    assert not (units["train"] & units["test"])


# ---------------------------------------------------------------------------
# Applying the gain model
# ---------------------------------------------------------------------------


def _fit_gain_model(num_units: int = 12, num_source: int = 5, num_target: int = 3, alpha: float = 1.0):
    """Fit a gain regression the way the model does -- in double precision (see fit_gain_model)."""
    rng = np.random.default_rng(1)
    source_gain = rng.normal(1.0, 0.2, size=(num_source, num_units))
    target_gain = rng.normal(1.0, 0.2, size=(num_target, num_units))
    model = ReducedRankRegression(alpha=alpha, fit_intercept=True)
    return model.fit(
        torch.tensor(source_gain.T, dtype=torch.float64),
        torch.tensor(target_gain.T, dtype=torch.float64),
    ), source_gain


def test_requested_rank_is_clipped_to_what_the_gain_model_supports():
    """Gain units are few, so the requested rank routinely exceeds the achievable rank."""
    gain_model, source_gain = _fit_gain_model()
    model = _model(_FakeRegistry({"test": np.arange(6)}))
    unit_index = np.repeat(np.arange(source_gain.shape[1]), 2)[:6]
    inputs = {
        "target_prediction": np.ones((3, 6)),
        "source_gain": source_gain,
        "unit_index": unit_index,
        "num_units": source_gain.shape[1],
        "frame_behavior": None,
        "idx_valid_prediction": np.arange(6),
        "was_filtered": False,
    }

    assert gain_model.max_rank < 2000
    prediction, extras = model.apply_gain_model(inputs, gain_model, rank=2000)
    assert prediction.shape == (3, 6)
    assert np.all(np.isfinite(prediction))
    # A unit's gain applies to every frame of that unit.
    assert np.allclose(extras["gain_applied"], extras["gain_predicted"][:, unit_index])


def test_prediction_is_the_place_field_scaled_by_the_predicted_gain():
    gain_model, source_gain = _fit_gain_model()
    model = _model(_FakeRegistry({"test": np.arange(4)}))
    rng = np.random.default_rng(2)
    target_prediction = rng.uniform(0.5, 2.0, size=(3, 4))
    unit_index = np.array([0, 0, 1, 1])
    inputs = {
        "target_prediction": target_prediction,
        "source_gain": source_gain,
        "unit_index": unit_index,
        "num_units": source_gain.shape[1],
        "frame_behavior": None,
        "idx_valid_prediction": np.arange(4),
        "was_filtered": False,
    }

    prediction, extras = model.apply_gain_model(inputs, gain_model, rank=2)
    assert np.allclose(prediction, target_prediction * extras["gain_predicted"][:, unit_index])
    assert extras["predictions_were_filtered"] is False


def test_gain_stage_filtering_is_reported_to_the_caller():
    """Frames dropped before the gain stage must still be reported, or targets misalign."""
    gain_model, source_gain = _fit_gain_model()
    model = _model(_FakeRegistry({"test": np.arange(4)}))
    inputs = {
        "target_prediction": np.ones((3, 4)),
        "source_gain": source_gain,
        "unit_index": np.array([0, 0, 1, 1]),
        "num_units": source_gain.shape[1],
        "frame_behavior": None,
        "idx_valid_prediction": np.array([1, 3, 5, 7]),
        "was_filtered": True,
    }

    _, extras = model.apply_gain_model(inputs, gain_model, rank=2)
    assert extras["predictions_were_filtered"] is True
    assert extras["idx_valid_predictions"].tolist() == [1, 3, 5, 7]


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


def test_dimensionality_counts_spatial_bins_and_gain_latents():
    model = _model(_FakeRegistry({"test": np.arange(4)}))
    hyperparameters = PlaceFieldStructuredGainHyperparameters(num_bins=40, rank=7)
    assert model.regressor_dimensionality(3, hyperparameters=hyperparameters) == 40 * 3 + 7


def test_hyperparameters_round_trip_through_the_optimizer_dict_form():
    hyperparameters = PlaceFieldStructuredGainHyperparameters(num_bins=25, smooth_width=None, rank=4, alpha=12.5)
    assert PlaceFieldStructuredGainHyperparameters.from_dict(dict(vars(hyperparameters))) == hyperparameters


def test_hyperparameters_survive_dataclasses_replace():
    """RegressionDimensionalitySweepConfig sweeps num_bins by replacing fields in place."""
    hyperparameters = PlaceFieldStructuredGainHyperparameters(num_bins=100, smooth_width=1.0, rank=9, alpha=50.0)
    swept = replace(hyperparameters, num_bins=10, smooth_width=2.0)
    assert (swept.num_bins, swept.smooth_width) == (10, 2.0)
    assert (swept.rank, swept.alpha) == (9, 50.0)


def test_search_space_separates_training_and_prediction_hyperparameters():
    """Rank is a prediction-time knob; everything else changes the fit."""
    space = PlaceFieldStructuredGainHyperparameters.get_search_space()
    assert set(space) == {"training", "prediction"}
    assert set(space["training"]) == {"num_bins", "smooth_width", "alpha"}
    assert set(space["prediction"]) == {"rank"}


def test_model_names_follow_the_registry_convention():
    registry = _FakeRegistry({"test": np.arange(4)})
    assert PlaceFieldStructuredGainModel(registry)._get_model_name() == "external_placefield_1d_structured_gain"
    assert PlaceFieldStructuredGainModel(registry, internal=True)._get_model_name() == "internal_placefield_1d_structured_gain"
