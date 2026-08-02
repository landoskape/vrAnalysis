from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vrAnalysis.processors.placefields import FrameBehavior, Placefield

from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldHyperparameters
from dimensionality_manuscript.subspace_analysis.base import Subspace
from dimensionality_manuscript.subspace_analysis.subspaces import (
    CovCovSubspace,
    SVCASubspace,
    _cov_or_gram,
)

# Environments in experience order [20, 10, 30]; the test frames visit 20 twice then 10 three times.
ENVIRONMENTS = np.array([20, 20, 10, 10, 10])
POSITIONS = np.array([0.5, 1.5, 0.5, 1.5, 2.5])
# Place field rows are environment-sorted (as get_placefield returns them), so row 0 is env 10.
PF_ENVIRONMENTS = (10, 20)
PF_ROWS = np.array([1, 1, 0, 0, 0])
PF_BINS = np.array([0, 1, 0, 1, 2])


def _placefield(values: np.ndarray, environments: tuple[int, ...], count: np.ndarray | None = None) -> Placefield:
    num_bins = values.shape[1]
    return Placefield(
        placefield=values,
        dist_edges=np.arange(num_bins + 1),
        environment=np.asarray(environments),
        count=np.ones(values.shape[:2]) if count is None else count,
    )


def _frame_behavior() -> FrameBehavior:
    num_frames = len(ENVIRONMENTS)
    return FrameBehavior(
        position=POSITIONS,
        speed=np.ones(num_frames),
        environment=ENVIRONMENTS,
        trial=np.arange(num_frames),
        reward_delivery=np.zeros(num_frames),
        reward_omitted=np.zeros(num_frames),
    )


def _expected_prediction(
    placefield: Placefield,
    frames: np.ndarray | slice = slice(None),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Place field prediction for the shared test frames, as (num_rois, num_frames)."""
    prediction = placefield.placefield[PF_ROWS, PF_BINS]  # (frames, rois)
    return torch.tensor(prediction[frames].T, dtype=dtype)


class _ShapeScore:
    """Minimal SVCA-like scorer whose output exposes the number of selected samples."""

    def score(self, source: torch.Tensor, target: torch.Tensor):
        score = torch.tensor(
            [float(source.shape[1]), float(source.sum() + target.sum())],
            dtype=source.dtype,
        )
        return score, score


class _IdentityPCA:
    def __init__(self, num_neurons: int):
        self._components = torch.eye(num_neurons, dtype=torch.float64)
        self._eigenvalues = torch.ones(num_neurons, dtype=torch.float64)

    def get_components(self):
        return self._components

    def get_eigenvalues(self):
        return self._eigenvalues


def _session():
    return SimpleNamespace(
        mouse_name="mouse_a",
        session_name=("mouse_a", "2020-01-01", "1"),
        session_uid="mouse_a_2020-01-01_1",
        env_length=(100,),
    )


def test_svca_score_filters_activity_and_placefields_by_environment_slot():
    session = _session()
    frame_behavior = _frame_behavior()
    source = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    target = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    pf_source = _placefield(np.arange(12, dtype=float).reshape(2, 3, 2), PF_ENVIRONMENTS)
    pf_target = _placefield(np.arange(18, dtype=float).reshape(2, 3, 3), PF_ENVIRONMENTS)

    model = SVCASubspace(registry=None, centered=True, autosave=False)
    model.get_session_data = lambda *args, **kwargs: ((source, target), frame_behavior, (2, 3))
    subspace = Subspace(
        _ShapeScore(),
        _ShapeScore(),
        dict(placefield_source=pf_source, placefield_target=pf_target),
    )
    hyperparameters = PlaceFieldHyperparameters(num_bins=100, smooth_width=5.0)

    with (
        patch(
            "dimensionality_manuscript.subspace_analysis.subspaces.get_placefield",
            side_effect=(pf_source, pf_target),
        ) as get_placefield_mock,
        patch(
            "dimensionality_manuscript.subspace_analysis.subspaces.load_env_order",
            return_value={"mouse_a": [20, 10, 30]},
        ),
    ):
        result = model.score(session, subspace, hyperparameters=hyperparameters)

    np.testing.assert_array_equal(result["env_slot_ids"], [20, 10, 30])
    torch.testing.assert_close(result["variance_activity_env"][:2, 0], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(result["variance_placefields_env"][:2, 0], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(result["variance_placefield_placefield_env"][:2, 0], torch.tensor([3.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(result["variance_activity_residual_env"][:2, 0], torch.tensor([2.0, 3.0]))
    assert torch.isnan(result["variance_activity_env"][2]).all()
    assert torch.isnan(result["variance_activity_residual_env"][2]).all()
    assert result["variance_activity"][0] == 5
    assert result["variance_activity_residual"][0] == 5
    assert result["variance_placefield_placefield"][0] == 6
    assert all(call.kwargs["smooth_width"] == 5.0 for call in get_placefield_mock.call_args_list)


def test_svca_residual_subtracts_placefield_prediction_and_drops_unvisited_bins():
    session = _session()
    frame_behavior = _frame_behavior()
    source = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    target = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    # Env 10 (row 0) never visited its last bin, so the frame predicted from it is dropped.
    count = np.ones((2, 3))
    count[0, 2] = 0
    pf_source = _placefield(np.arange(12, dtype=float).reshape(2, 3, 2), PF_ENVIRONMENTS, count=count)
    pf_target = _placefield(np.arange(18, dtype=float).reshape(2, 3, 3), PF_ENVIRONMENTS, count=count)

    calls = []

    class _CaptureScore:
        def score(self, source: torch.Tensor, target: torch.Tensor):
            calls.append((source, target))
            score = torch.tensor([float(source.shape[1])], dtype=source.dtype)
            return score, score

    model = SVCASubspace(registry=None, centered=True, autosave=False)
    model.get_session_data = lambda *args, **kwargs: ((source, target), frame_behavior, (2, 3))
    subspace = Subspace(
        _CaptureScore(),
        _ShapeScore(),
        dict(placefield_source=pf_source, placefield_target=pf_target),
    )

    with (
        patch(
            "dimensionality_manuscript.subspace_analysis.subspaces.get_placefield",
            side_effect=(pf_source, pf_target),
        ),
        patch(
            "dimensionality_manuscript.subspace_analysis.subspaces.load_env_order",
            return_value={"mouse_a": [20, 10, 30]},
        ),
    ):
        result = model.score(session, subspace)

    kept = np.array([0, 1, 2, 3])
    residual_source, residual_target = calls[1]  # calls[0] is the raw-activity score
    torch.testing.assert_close(residual_source, source[:, kept] - _expected_prediction(pf_source, kept))
    torch.testing.assert_close(residual_target, target[:, kept] - _expected_prediction(pf_target, kept))
    assert result["variance_activity_residual"][0] == 4
    # Env 20 keeps both frames; env 10 keeps two of three.
    torch.testing.assert_close(result["variance_activity_residual_env"][:2, 0], torch.tensor([2.0, 2.0]))


def test_covcov_score_uses_environment_specific_covariances():
    session = _session()
    frame_behavior = _frame_behavior()
    test_data = torch.tensor(
        [
            [0.0, 2.0, 0.0, 2.0, 5.0],
            [1.0, 3.0, 4.0, 1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    pf_values = np.array(
        [
            [[0.0, 1.0], [1.0, 1.0], [3.0, 2.0]],
            [[0.0, 0.0], [2.0, 1.0], [5.0, 3.0]],
        ]
    )
    placefield = _placefield(pf_values, PF_ENVIRONMENTS)

    model = CovCovSubspace(registry=None, centered=True, autosave=False)
    model.get_session_data = lambda *args, **kwargs: (test_data, frame_behavior, 2)
    subspace = Subspace(_IdentityPCA(2), _IdentityPCA(2), dict(placefield=placefield))

    with (
        patch(
            "dimensionality_manuscript.subspace_analysis.subspaces.get_placefield",
            return_value=placefield,
        ),
        patch(
            "dimensionality_manuscript.subspace_analysis.subspaces.load_env_order",
            return_value={"mouse_a": [20, 10, 30]},
        ),
    ):
        result = model.score(session, subspace)

    expected = []
    for env in (20, 10):
        idx_env = torch.as_tensor(frame_behavior.environment == env)
        eigenvalues = torch.linalg.eigvalsh(_cov_or_gram(test_data[:, idx_env], centered=True))
        expected.append(torch.sqrt(torch.clamp_min(torch.flipud(eigenvalues), 0.0)))

    np.testing.assert_array_equal(result["env_slot_ids"], [20, 10, 30])
    torch.testing.assert_close(result["variance_activity_env"][0], expected[0])
    torch.testing.assert_close(result["variance_activity_env"][1], expected[1])
    torch.testing.assert_close(result["variance_placefields_env"][:2], result["variance_activity_env"][:2])
    assert torch.isfinite(result["variance_placefield_placefield_env"][:2]).all()
    assert torch.isnan(result["variance_activity_env"][2]).all()

    # With an identity subspace the residual score is just the spectrum of the residual covariance.
    residual = test_data - _expected_prediction(placefield, dtype=torch.float64)
    expected_residual = torch.sqrt(torch.clamp_min(torch.flipud(torch.linalg.eigvalsh(_cov_or_gram(residual, centered=True))), 0.0))
    torch.testing.assert_close(result["variance_activity_residual"], expected_residual)
    assert torch.isnan(result["variance_activity_residual_env"][2]).all()


def test_affected_subspaces_version_their_score_cache_keys():
    registry = SimpleNamespace(registry_params={}, registry_paths=SimpleNamespace())
    hyperparameters = PlaceFieldHyperparameters(num_bins=100, smooth_width=5.0)

    for model_class in (SVCASubspace, CovCovSubspace):
        model = model_class(registry, autosave=False)
        cache_key = model._get_score_from_hyps_cache_key(
            _session(),
            "sigrebase",
            "train",
            "not_train",
            hyperparameters,
        )
        assert cache_key.endswith("_residual-v1")
