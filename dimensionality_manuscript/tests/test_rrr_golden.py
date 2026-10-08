from types import SimpleNamespace

import pytest
import torch

from dimensionality_manuscript.regression_models.models import ReducedRankRegressionModel


@pytest.mark.parametrize("max_rank", [1, 3])
def test_rrr_golden_alpha_search_uses_achievable_rank(monkeypatch, max_rank):
    used_ranks = []
    model = SimpleNamespace()
    model.get_session_data = lambda *args, **kwargs: (
        torch.zeros((4, 10)),
        torch.zeros((max_rank, 10)),
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
