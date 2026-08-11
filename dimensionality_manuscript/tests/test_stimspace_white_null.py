"""Unit tests for the white-noise null used by ``ffres_white`` / ``ff_res_white``.

These exercise the surrogate construction and the estimator path it feeds directly, with
synthetic matrices, so they need no session data or mounted store.
"""

from __future__ import annotations

import numpy as np
import torch

from dimensionality_manuscript.configs.stimspace import _direct_svd, _to_g, _white_generator, _white_like


class _FakeSession:
    """Minimal stand-in exposing the only attribute ``_white_generator`` reads."""

    def __init__(self, session_name: tuple[str, ...]):
        self.session_name = session_name


def _low_rank_plus_noise(n: int, t: int, rank: int, seed: int) -> torch.Tensor:
    """(n, t) matrix with ``rank`` shared latents across neurons plus private noise."""
    generator = torch.Generator().manual_seed(seed)
    loadings = torch.randn(n, rank, generator=generator)
    latents = torch.randn(rank, t, generator=generator)
    return loadings @ latents + 0.3 * torch.randn(n, t, generator=generator)


def test_white_generator_is_deterministic_per_session():
    name = ("ATL022", "2023-04-12", "701")
    assert _white_generator(_FakeSession(name)).initial_seed() == _white_generator(_FakeSession(name)).initial_seed()
    other = ("CR_Hippocannula6", "2022-08-30", "701")
    assert _white_generator(_FakeSession(name)).initial_seed() != _white_generator(_FakeSession(other)).initial_seed()


def test_white_like_matches_shape_dtype_and_per_neuron_variance():
    generator = _white_generator(_FakeSession(("m", "d", "s")))
    scales = torch.tensor([0.5, 2.0, 10.0]).unsqueeze(1)
    x = torch.randn(3, 20000) * scales + torch.tensor([1.0, -3.0, 7.0]).unsqueeze(1)

    white = _white_like(x, generator)

    assert white.shape == x.shape
    assert white.dtype == x.dtype
    # The diagonal of the covariance is preserved (sampling error over 20k columns is ~1%).
    assert torch.allclose(white.std(dim=1), x.std(dim=1), rtol=0.05)


def test_white_like_discards_cross_neuron_covariance():
    """The surrogate keeps marginal variances but drops the shared structure it is a null for."""
    generator = _white_generator(_FakeSession(("m", "d", "s")))
    x = _low_rank_plus_noise(n=8, t=6000, rank=2, seed=0)
    white = _white_like(x, generator)

    def max_abs_offdiag(m: torch.Tensor) -> float:
        cov = _to_g(m) @ _to_g(m).T
        scale = torch.sqrt(torch.diag(cov)).unsqueeze(1)
        corr = (cov / (scale * scale.T)).fill_diagonal_(0.0)
        return float(corr.abs().max())

    assert max_abs_offdiag(x) > 0.3
    assert max_abs_offdiag(white) < 0.1


def test_white_like_leaves_the_global_torch_rng_untouched():
    """``_white_like`` draws from its own Generator, so it must not advance the global stream."""
    generator = _white_generator(_FakeSession(("m", "d", "s")))
    x = torch.ones(4, 100) * torch.arange(1.0, 5.0).unsqueeze(1) + torch.randn(4, 100)

    torch.manual_seed(1234)
    expected = torch.randn(5)
    torch.manual_seed(1234)
    _white_like(x, generator)
    assert torch.allclose(torch.randn(5), expected)


def test_ffres_white_spectrum_falls_below_the_structured_spectrum():
    """The null runs through the same _to_g + _direct_svd path and must lack the leading spectrum."""
    generator = _white_generator(_FakeSession(("m", "d", "s")))
    a = _low_rank_plus_noise(n=30, t=800, rank=3, seed=1)
    b = _low_rank_plus_noise(n=30, t=800, rank=3, seed=1)  # same latents -> genuine shared structure

    n_components = 30
    real = _direct_svd(_to_g(a), _to_g(b), n_components).numpy()
    white = _direct_svd(_to_g(_white_like(a, generator)), _to_g(_white_like(b, generator)), n_components).numpy()

    assert real.shape == white.shape
    assert np.all(np.isfinite(white))
    assert white[:3].max() < real[:3].min(), "white null reaches into the structured leading components"
