"""Tests for StimSpaceSubspace fold preparation."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from dimensionality_manuscript.subspace_analysis.stimspace import StimSpacePrepState, StimSpaceSubspace
from vrAnalysis.processors.placefields import Placefield
from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldHyperparameters


def _synthetic_placefield(num_neurons: int = 4, num_bins: int = 8, environment: int = 0) -> Placefield:
    """Placefield with no NaNs (all bins valid)."""
    pf = np.ones((1, num_bins, num_neurons), dtype=np.float32)
    count = np.ones((1, num_bins), dtype=np.float32)
    dist_edges = np.linspace(0, num_bins, num_bins + 1)
    return Placefield(
        placefield=pf,
        dist_edges=dist_edges,
        environment=np.array([environment]),
        count=count,
    )


def _make_model() -> StimSpaceSubspace:
    return StimSpaceSubspace(MagicMock(), normalize=False, use_fast_sampling=False)


def test_prep_state_roundtrip_extras():
    prep = StimSpacePrepState(
        idx_keep=np.array([True, False, True]),
        valid_environments=np.array([0]),
        valid_positions=np.array([True, True, False, True]),
        idx_test_split0=torch.tensor([0, 2]),
        idx_test_split1=torch.tensor([1, 3]),
    )
    restored = StimSpacePrepState.from_extras(prep.to_extras())
    assert np.array_equal(restored.idx_keep, prep.idx_keep)
    assert np.array_equal(restored.valid_environments, prep.valid_environments)
    assert np.array_equal(restored.valid_positions, prep.valid_positions)
    assert torch.equal(restored.idx_test_split0, prep.idx_test_split0)
    assert torch.equal(restored.idx_test_split1, prep.idx_test_split1)


def test_get_processed_folds_fit_then_score_reuses_prep():
    model = _make_model()
    session = MagicMock()
    session.environments = [0]
    hyps = PlaceFieldHyperparameters(num_bins=8, smooth_width=None)

    split_sizes = {
        "train0": 20,
        "train1": 16,
        "validation": 12,
        "test": 10,
    }
    num_neurons = 5

    def mock_get_session_data(sess, spks_type, split, use_cell_split=False):
        t = split_sizes[split]
        data = torch.randn(num_neurons, t)
        fb = MagicMock()
        fb.filter = MagicMock(side_effect=lambda idx: fb)
        return data, fb, num_neurons

    def mock_compute_placefield_folds(fold_specs, dist_edges, session):
        n = next(data for data, _, _ in fold_specs.values()).shape[0]
        return {name: _synthetic_placefield(num_neurons=n) for name in fold_specs}

    fit_specs = model._fit_fold_specs(hyps)
    score_specs = model._score_fold_specs(hyps)

    with (
        patch.object(model, "get_session_data", side_effect=mock_get_session_data),
        patch.object(model, "_get_placefield_dist_edges", return_value=np.linspace(0, 8, 9)),
        patch.object(
            model,
            "_neuron_filter",
            autospec=True,
            return_value=np.ones(num_neurons, dtype=bool),
        ) as neuron_filter,
        patch.object(model, "_compute_placefield_folds", side_effect=mock_compute_placefield_folds),
    ):
        folds_fit, prep_fit = model.get_processed_folds(session, "oasis", fit_specs, hyps)
        neuron_filter.assert_called_once()

        prep_from_extras = StimSpacePrepState.from_extras(prep_fit.to_extras())
        folds_score, prep_score = model.get_processed_folds(
            session, "oasis", score_specs, hyps, prep_state=prep_from_extras
        )

    assert neuron_filter.call_count == 1
    assert np.array_equal(prep_score.valid_positions, prep_fit.valid_positions)
    assert np.array_equal(prep_score.valid_environments, prep_fit.valid_environments)
    for name in ("cv1", "cv2", "test"):
        assert name in folds_score.activity
        assert name in folds_score.pf_matrices
        assert folds_score.pf_matrices[name].shape[1] == folds_fit.pf_matrices["train"].shape[1]
