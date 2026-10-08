from types import SimpleNamespace

import pytest
import torch

from dimensionality_manuscript.regression_models.hyperparameters import ReducedRankRegressionHyperparameters
from dimensionality_manuscript.regression_models.models import ReducedRankRegressionModel


@pytest.mark.parametrize(
    ("num_source", "num_target", "num_samples", "max_rank"),
    [
        (1, 4, 10, 1),
        (4, 3, 10, 3),
        (4, 6, 2, 2),
    ],
)
def test_rrr_golden_alpha_search_uses_achievable_rank(
    monkeypatch,
    num_source,
    num_target,
    num_samples,
    max_rank,
):
    used_ranks = []
    model = SimpleNamespace()
    model.get_session_data = lambda *args, **kwargs: (
        torch.zeros((num_source, num_samples)),
        torch.zeros((num_target, num_samples)),
        None,
    )

    def train(*args, hyperparameters, **kwargs):
        assert 1 <= hyperparameters.rank <= max_rank
        used_ranks.append(hyperparameters.rank)
        return object()

    model.train = train
    model.score = lambda *args, hyperparameters, **kwargs: float(hyperparameters.rank)

    def fake_golden_section_search(func, a, b, **kwargs):
        value = a
        return value, func(value), []

    monkeypatch.setattr(
        "dimensionality_manuscript.regression_models.models.golden_section_search",
        fake_golden_section_search,
    )

    best_params, _, results = ReducedRankRegressionModel._optimize_golden(
        model,
        session=object(),
        spks_type="sigrebase",
        train_split="train",
        validation_split="validation",
    )

    assert used_ranks[0] == max_rank
    assert max(used_ranks) <= max_rank
    assert results.iloc[0]["rank"] == max_rank
    assert best_params["rank"] <= max_rank


def test_rrr_prediction_clips_requested_rank_to_fitted_capacity():
    used_ranks = []
    num_samples = 5
    num_targets = 3
    model = SimpleNamespace(nonnegative=True)
    model.get_session_data = lambda *args, **kwargs: (
        torch.zeros((4, num_samples)),
        torch.zeros((num_targets, num_samples)),
        None,
    )

    fitted = SimpleNamespace(max_rank=2)

    def predict(source, rank, nonnegative=False):
        used_ranks.append(rank)
        return torch.zeros((source.shape[0], num_targets))

    def predict_latent(source, rank):
        used_ranks.append(rank)
        return torch.zeros((source.shape[0], rank))

    fitted.predict = predict
    fitted.predict_latent = predict_latent

    prediction, extras = ReducedRankRegressionModel.predict(
        model,
        session=object(),
        rrr_model=fitted,
        spks_type="sigrebase",
        split="test",
        hyperparameters=ReducedRankRegressionHyperparameters(alpha=1.0, rank=200),
    )

    assert prediction.shape == (num_targets, num_samples)
    assert used_ranks == [2, 2]
    assert extras["requested_rank"] == 200
    assert extras["effective_rank"] == 2
    assert extras["max_rank"] == 2
