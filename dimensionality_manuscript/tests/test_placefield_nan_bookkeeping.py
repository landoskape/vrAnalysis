"""Regression tests for place-field prediction NaN bookkeeping."""

from types import MethodType

import numpy as np
import pytest
import torch

from vrAnalysis.processors.placefields import FrameBehavior

from dimensionality_manuscript.regression_models.models import PlaceFieldModel


def _frame_behavior(num_samples: int) -> FrameBehavior:
    values = np.arange(num_samples, dtype=float)
    return FrameBehavior(
        position=values,
        speed=values,
        environment=np.zeros(num_samples),
        trial=np.zeros(num_samples),
        reward_delivery=np.zeros(num_samples),
        reward_omitted=np.zeros(num_samples),
    )


def test_score_nan_fraction_uses_sample_axis():
    model = PlaceFieldModel(None, autosave=False)
    prediction = np.ones((2, 9))
    target = torch.ones((100, 10))
    extras = {
        "predictions_were_filtered": True,
        "idx_valid_predictions": np.arange(9),
    }
    model.predict = MethodType(lambda self, *args, **kwargs: (prediction, extras), model)
    model.get_session_data = MethodType(lambda self, *args, **kwargs: (torch.empty(0), target, None), model)
    model.evaluate = MethodType(
        lambda self, prediction, filtered_target, **kwargs: {
            "mse": float(filtered_target.shape[1] != prediction.shape[1])
        },
        model,
    )

    assert model.score(object(), object(), nan_safe=False) == pytest.approx(0.0)


def test_score_nan_error_reports_nan_count_and_sample_count():
    model = PlaceFieldModel(None, autosave=False)
    prediction = np.ones((2, 8))
    target = torch.ones((100, 10))
    extras = {
        "predictions_were_filtered": True,
        "idx_valid_predictions": np.arange(8),
    }
    model.predict = MethodType(lambda self, *args, **kwargs: (prediction, extras), model)
    model.get_session_data = MethodType(lambda self, *args, **kwargs: (torch.empty(0), target, None), model)

    with pytest.raises(ValueError, match=r"2 / 10 samples have NaN values"):
        model.score(object(), object(), nan_safe=False)


def test_gain_stage_filter_is_preserved_when_target_has_no_additional_nans(monkeypatch):
    model = PlaceFieldModel(None, gain=True, autosave=False)
    behavior = _frame_behavior(6)
    source_data = torch.ones((2, 6))
    target_data = torch.ones((3, 6))
    model.get_session_data = MethodType(lambda self, *args, **kwargs: (source_data, target_data, behavior), model)

    source_placefield = object()
    target_placefield = object()

    def fake_placefield_prediction(placefield, frame_behavior):
        if placefield is source_placefield:
            prediction = np.ones((len(frame_behavior), 2))
            prediction[1] = np.nan
        else:
            prediction = np.ones((len(frame_behavior), 3))
        return prediction, {}

    monkeypatch.setattr(
        "dimensionality_manuscript.regression_models.models.get_placefield_prediction",
        fake_placefield_prediction,
    )

    prediction, extras = model.predict(
        object(),
        (target_placefield, source_placefield),
        nan_safe=False,
    )

    assert prediction.shape == (3, 5)
    assert extras["predictions_were_filtered"] is True
    np.testing.assert_array_equal(extras["idx_valid_predictions"], [0, 2, 3, 4, 5])
    np.testing.assert_array_equal(extras["frame_behavior"].idx, [0, 2, 3, 4, 5])


def test_gain_prediction_reports_false_when_neither_stage_filters(monkeypatch):
    model = PlaceFieldModel(None, gain=True, autosave=False)
    behavior = _frame_behavior(6)
    source_data = torch.ones((2, 6))
    target_data = torch.ones((3, 6))
    model.get_session_data = MethodType(lambda self, *args, **kwargs: (source_data, target_data, behavior), model)

    source_placefield = object()
    target_placefield = object()

    def fake_placefield_prediction(placefield, frame_behavior):
        num_rois = 2 if placefield is source_placefield else 3
        return np.ones((len(frame_behavior), num_rois)), {}

    monkeypatch.setattr(
        "dimensionality_manuscript.regression_models.models.get_placefield_prediction",
        fake_placefield_prediction,
    )

    prediction, extras = model.predict(
        object(),
        (target_placefield, source_placefield),
        nan_safe=False,
    )

    assert prediction.shape == (3, 6)
    assert extras["predictions_were_filtered"] is False
    assert "idx_valid_predictions" not in extras
    np.testing.assert_array_equal(extras["frame_behavior"].idx, np.arange(6))
