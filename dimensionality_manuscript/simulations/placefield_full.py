"""
Place field generator for the ``stim_full`` shared-variance atlas (numpy).

This module lets place-field simulations flow through the *canonical* shared-variance
operators in ``shared_variance.py`` / ``operators.py`` instead of the bespoke inline math in
``simulations/placefield_*.py`` and the top-level ``study_cvpca_vs_stimspace.py``. It exposes a
config + generator pair that satisfies the ``stim_full`` generator contract (see
``shared_variance.py``), so a place-field condition behaves exactly like the other
``stim_full`` atlas cases: population kappa / energy / geometry, 3-fold cvSER, 4-fold cv-kappa,
and 5-draw cv-stimstim are all computed by the shared engine with no estimator math duplicated
here.

Generative model (mirrors the "stimulus = position" reading of place fields)
---------------------------------------------------------------------------
Each neuron has a fixed *clean* tuning curve over ``P`` positions: ``source`` is an
``(N, P)`` matrix. One "trial repeat" is the clean source plus:

  1. per-repeat spatial noise: an RBF Gaussian-process draw over positions, scaled by
     ``repeat_noise_alpha`` and shared across positions *within* a repeat (independent across
     repeats and neurons),
  2. a field-model rectification (ReLU / per-repeat threshold for the thresholded and Tilbury
     models; identity for the smooth-GP model), and
  3. ``noise_level`` i.i.d. Gaussian noise, independent per neuron / position / repeat.

``generate(num_samples)`` builds ``R = num_samples / P`` independent repeats and lays them out
as columns (one column per (repeat, position)), labelling each column with its position via
``extras["stim_indices"]``. Positions play the role of "stimuli", repeats play the role of
"samples per stimulus" — exactly the structure the atlas's stimulus-balanced fold machinery and
cross-validated estimators expect.

Three field models (discriminated by type, like PlacefieldConfig vs TilburyConfig elsewhere):
  - ``ThresholdedGPFieldConfig`` : RBF-GP fields, percentile threshold + ReLU, optional peak warp.
  - ``SmoothGPFieldConfig``      : pure RBF-GP fields, NO threshold / NO ReLU (no sharp
                                    discontinuities) — the new "smooth" option.
  - ``TilburyFieldConfig``       : per-neuron double generalized-Gaussian tuning curves.

NOTE (deferred estimator): the canonical *cross-validated place-field-kernel* estimator
implemented in ``dimensionality_manuscript/subspace_analysis/stimspace.py`` (``StimSpaceSubspace``,
driven by ``configs/stimspace.py``) is NOT yet ported into this atlas path. The atlas currently
supplies population/CV kappa, energy,
cvSER, cv-stimstim, and (in ``shared_variance.py``) an rCVPCA estimator — but not that specific
cross-validated-kernel form. See the matching note beside ``_stim_full_rcvpca_result``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from .utilities import generate_orthonormal, rotate_subspace_by_angle

# ---------------------------------------------------------------------------
# Low-level numpy place-field math (ports of placefield_generator.py)
# ---------------------------------------------------------------------------


def _rbf_kernel(num_positions: int, lengthscale: float, exponent: Optional[float], dtype: np.dtype) -> npt.NDArray[np.floating]:
    """RBF (optionally generalized-exponent) kernel over positions, shape (P, P).

    Exponents >= 2 collapse to the standard squared-exponential (the only PSD choice); a smaller
    exponent roughens the kernel. Exponents > 2 are handled by ``peaky_transform`` instead,
    because the corresponding kernel is not positive semi-definite.
    """
    pos = np.arange(num_positions, dtype=dtype)
    diff = pos[:, None] - pos[None, :]
    exp = 2.0 if (exponent is None or exponent >= 2.0) else float(exponent)
    return np.exp(-np.abs(diff / lengthscale) ** exp).astype(dtype, copy=False)


def _chol(kernel: npt.NDArray[np.floating], jitter: float = 1e-5) -> npt.NDArray[np.floating]:
    """Cholesky factor of ``kernel`` with diagonal jitter for numerical stability."""
    P = kernel.shape[0]
    return np.linalg.cholesky(kernel + jitter * np.eye(P, dtype=kernel.dtype))


def _sample_gp(chol: npt.NDArray[np.floating], n_samples: int, rng: np.random.Generator, dtype: np.dtype) -> npt.NDArray[np.floating]:
    """Draw ``n_samples`` independent GP samples given a Cholesky factor. Returns (n_samples, P)."""
    P = chol.shape[0]
    z = rng.standard_normal((P, n_samples)).astype(dtype, copy=False)
    return (chol @ z).T


def peaky_transform(
    fields: npt.NDArray[np.floating],
    peak_exponent: float,
    sigma_scale: float = 1.0,
    eps: float = 1e-8,
) -> npt.NDArray[np.floating]:
    """Warp non-negative GP fields toward generalized-Gaussian peak shapes.

    Preserves each field's peak location and multi-peak structure while reshaping the local
    amplitude profile. Identity on positive values when ``peak_exponent == 2`` and
    ``sigma_scale == 1``.
    """
    fields = np.maximum(fields, 0.0)
    peak_amp = np.maximum(fields.max(axis=1, keepdims=True), eps)
    normalized = np.clip(fields / peak_amp, eps, 1.0)
    scale = max(float(sigma_scale), eps)
    warped_distance = -np.log(normalized)
    warped = np.exp(-(warped_distance ** (peak_exponent / 2.0)) / (scale**peak_exponent))
    return np.where(fields > 0, peak_amp * warped, 0.0)


def _sample_tilbury_params(cfg: "TilburyFieldConfig", num_neurons: int, num_positions: int, rng: np.random.Generator, dtype: np.dtype) -> dict:
    """Sample per-neuron double generalized-Gaussian parameters. Each value has shape (N, 2)."""
    N, P = num_neurons, num_positions

    # Primary amplitude A1 ~ LogNormal(log(mean), spread); A2 = A1 * Beta(1, beta) ratio.
    if cfg.amplitude_spread > 0:
        A1 = np.exp(np.log(cfg.amplitude_mean) + cfg.amplitude_spread * rng.standard_normal(N))
    else:
        A1 = np.full(N, float(cfg.amplitude_mean))
    U = rng.random(N)
    A2 = (1.0 - U ** (1.0 / max(cfg.amplitude_ratio_beta, 1e-6))) * A1

    # Primary peak uniform over the track; secondary peak offset by Normal(0, separation).
    phi1 = rng.random(N) * P
    if cfg.peak_separation_scale > 0:
        phi2 = np.clip(phi1 + cfg.peak_separation_scale * rng.standard_normal(N), 0.0, float(P - 1))
    else:
        phi2 = phi1.copy()

    def _sigma_base() -> npt.NDArray[np.floating]:
        if cfg.sigma_spread > 0:
            return np.exp(np.log(cfg.sigma_mean) + cfg.sigma_spread * rng.standard_normal(N))
        return np.full(N, float(cfg.sigma_mean))

    sb1, sb2 = _sigma_base(), _sigma_base()

    # Left/right width asymmetry: log(sigma_right / sigma_left) ~ Normal(0, asym_std).
    def _asym() -> npt.NDArray[np.floating]:
        if cfg.sigma_asym_std > 0:
            return np.exp(cfg.sigma_asym_std * rng.standard_normal(N))
        return np.ones(N)

    asym1, asym2 = _asym(), _asym()

    def _exponent() -> npt.NDArray[np.floating]:
        if cfg.exponent_spread > 0:
            return np.exp(np.log(cfg.exponent_mean) + cfg.exponent_spread * rng.standard_normal(N))
        return np.full(N, float(cfg.exponent_mean))

    params = {
        "phi": np.stack([phi1, phi2], axis=1),
        "A": np.stack([A1, A2], axis=1),
        "sigma_left": np.stack([sb1 / np.sqrt(asym1), sb2 / np.sqrt(asym2)], axis=1),
        "sigma_right": np.stack([sb1 * np.sqrt(asym1), sb2 * np.sqrt(asym2)], axis=1),
        "p": np.stack([_exponent(), _exponent()], axis=1),
    }
    return {k: v.astype(dtype, copy=False) for k, v in params.items()}


def _eval_tilbury(
    theta: npt.NDArray[np.floating],
    phi: npt.NDArray[np.floating],
    A: npt.NDArray[np.floating],
    sigma_left: npt.NDArray[np.floating],
    sigma_right: npt.NDArray[np.floating],
    p: npt.NDArray[np.floating],
    baseline: float,
) -> npt.NDArray[np.floating]:
    """Evaluate double generalized-Gaussian fields for all neurons. Returns (N, P)."""
    diff = theta[None, None, :] - phi[:, :, None]  # (N, 2, P)
    sigma = np.where(diff < 0, sigma_left[:, :, None], sigma_right[:, :, None])
    bumps = A[:, :, None] * np.exp(-((np.abs(diff) / np.clip(sigma, 1e-6, None)) ** p[:, :, None]))
    return baseline + bumps.sum(axis=1)


# ---------------------------------------------------------------------------
# Field-model configs
# ---------------------------------------------------------------------------
#
# Each field model knows how to (a) build the clean (N, P) source from a construction RNG and
# (b) rectify a noisy repeat (source + GP noise). The rectification is what distinguishes the
# models at noise-handling time: ReLU for thresholded, percentile-threshold+ReLU for Tilbury,
# and identity for the smooth model.


@dataclass
class ThresholdedGPFieldConfig:
    """RBF Gaussian-process fields with percentile threshold + ReLU (sparse, localized).

    Parameters
    ----------
    lengthscale : float
        RBF kernel lengthscale in position-bin units.
    threshold_pct : float
        Percentile of the GP sample used as the rectification threshold (higher -> sparser).
    amplitude : float
        Scale applied after thresholding.
    peak_exponent : float or None
        When > 2, warp amplitudes toward a generalized-Gaussian peak via ``peaky_transform``
        (the kernel itself stays squared-exponential). Values in (0, 2) roughen the kernel.
        None -> standard smooth GP bumps.
    peak_sigma_scale : float
        Width-like scale for the amplitude warp (only used when ``peak_exponent`` > 2).
    """

    lengthscale: float = 8.0
    threshold_pct: float = 60.0
    amplitude: float = 10.0
    peak_exponent: Optional[float] = None
    peak_sigma_scale: float = 1.0

    def build_source(self, num_neurons: int, num_positions: int, rng: np.random.Generator, dtype: np.dtype) -> npt.NDArray[np.floating]:
        kernel = _rbf_kernel(num_positions, self.lengthscale, self.peak_exponent, dtype)
        raw = _sample_gp(_chol(kernel), num_neurons, rng, dtype)
        threshold = np.quantile(raw, self.threshold_pct / 100.0, axis=1, keepdims=True)
        source = self.amplitude * np.maximum(raw - threshold, 0.0)
        if self.peak_exponent is not None and self.peak_exponent > 2.0:
            source = peaky_transform(source, self.peak_exponent, self.peak_sigma_scale)
        return source.astype(dtype, copy=False)

    def rectify_repeat(self, noisy: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return np.maximum(noisy, 0.0)


@dataclass
class SmoothGPFieldConfig:
    """Pure RBF Gaussian-process fields: NO threshold, NO ReLU (smooth, no discontinuities).

    This is the new linear / non-thresholded option. With no rectification, repeats are exactly
    ``source + alpha * GP_noise + iid``, so the additive decomposition x = g(s) + h + eps holds
    exactly and the population Sigma_stim / Sigma_nuisance / Sigma_eps are exact (not approximate).

    Parameters
    ----------
    lengthscale : float
        RBF kernel lengthscale in position-bin units.
    amplitude : float
        Scale applied to the GP draw.
    """

    lengthscale: float = 8.0
    amplitude: float = 10.0

    def build_source(self, num_neurons: int, num_positions: int, rng: np.random.Generator, dtype: np.dtype) -> npt.NDArray[np.floating]:
        kernel = _rbf_kernel(num_positions, self.lengthscale, exponent=None, dtype=dtype)
        raw = _sample_gp(_chol(kernel), num_neurons, rng, dtype)
        return (self.amplitude * raw).astype(dtype, copy=False)

    def rectify_repeat(self, noisy: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return noisy  # no rectification: keep the smooth, sign-free GP structure


@dataclass
class TilburyFieldConfig:
    """Per-neuron double generalized-Gaussian fields (Tilbury model).

    Each neuron's clean tuning curve is
        f(theta) = baseline + A1 * exp(-(|theta-phi1|/sigma1+-)^p1) + A2 * exp(-(|theta-phi2|/sigma2+-)^p2),
    with sigma+- piecewise (left/right of the peak). All per-neuron parameters are sampled from
    the distributions below. Repeats apply a per-neuron percentile threshold + ReLU.

    Parameters mirror ``placefield_generator.TilburyConfig`` (see there for full detail):
    amplitude_mean/spread, amplitude_ratio_beta, peak_separation_scale, sigma_mean/spread,
    sigma_asym_std, exponent_mean/spread, baseline.
    """

    amplitude_mean: float = 10.0
    amplitude_spread: float = 0.5
    amplitude_ratio_beta: float = 5.0
    peak_separation_scale: float = 20.0
    sigma_mean: float = 8.0
    sigma_spread: float = 0.3
    sigma_asym_std: float = 0.0
    exponent_mean: float = 2.0
    exponent_spread: float = 0.0
    baseline: float = 0.0

    def build_source(self, num_neurons: int, num_positions: int, rng: np.random.Generator, dtype: np.dtype) -> npt.NDArray[np.floating]:
        params = _sample_tilbury_params(self, num_neurons, num_positions, rng, dtype)
        theta = np.arange(num_positions, dtype=dtype)
        source = _eval_tilbury(theta, baseline=self.baseline, **params)
        return source.astype(dtype, copy=False)

    def rectify_repeat(self, noisy: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return np.maximum(noisy, 0.0)


FieldModelConfig = Union[ThresholdedGPFieldConfig, SmoothGPFieldConfig, TilburyFieldConfig]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class PlacefieldFullConfig:
    """Configuration for a place-field ``stim_full`` atlas condition.

    Positions are the "stimuli" (``num_stimuli == num_positions``); repeats are the "samples per
    stimulus". ``generate`` requires ``num_samples`` to be a multiple of ``num_positions``.

    Parameters
    ----------
    field_model : ThresholdedGPFieldConfig | SmoothGPFieldConfig | TilburyFieldConfig
        Clean-field generative model.
    num_neurons : int
        Population size N.
    num_positions : int
        Track length P (= number of stimuli).
    n_repeats : int
        Repeats used by the rCVPCA estimator (must be >= 4 for its 4-fold r0/r1/r2/r3 split).
    repeat_noise_alpha : float
        Amplitude of per-repeat RBF-GP spatial noise relative to the source.
    repeat_noise_lengthscale : float
        Lengthscale of the per-repeat GP noise kernel.
    noise_level : float
        Std of intrinsic i.i.d. Gaussian noise -> Sigma_eps = noise_level^2 * I.
    normalize : bool
        If True, divide the source by a deterministic per-neuron clean-source max so all
        downstream quantities (source, noises, covariances) live in normalized units. This is a
        deterministic normalizer (unlike the data-dependent max used in placefield_generator.py),
        chosen so the population covariances stay analytic.
    rcvpca_smooth_width : float or None
        Gaussian smoothing width (bins) for the rCVPCA fit repeat; None disables smoothing.
    rcvpca_center : bool
        Mean-centering flag passed to CVPCA in the rCVPCA estimator.
    rcvpca_num_components : int or None
        Number of components for the rCVPCA estimator; None -> min(N-1, P-1).
    rng : np.random.Generator or None
        Construction RNG (controls the clean source + per-neuron parameters).
    """

    field_model: FieldModelConfig = field(default_factory=SmoothGPFieldConfig)
    num_neurons: int = 200
    num_positions: int = 50
    n_repeats: int = 4
    repeat_noise_alpha: float = 0.3
    repeat_noise_lengthscale: float = 5.0

    # nuisance component (not related to position)
    nuisance_dim: int = 1  # dimension of stimulus-independent nuisance subspace
    alpha_nuisance: float = 1.0  # power-law exponent for nuisance spectrum
    nuisance_scale: float = 0.0  # multiplicative scale for nuisance component

    # Some extra things
    noise_level: float = 0.3
    normalize: bool = False
    rcvpca_smooth_width: Optional[float] = 3.0
    rcvpca_center: bool = True
    rcvpca_num_components: Optional[int] = None
    rng: Optional[np.random.Generator] = None

    @property
    def num_stimuli(self) -> int:
        """Stimulus count seen by the atlas pipeline (positions act as stimuli)."""
        return self.num_positions

    def __post_init__(self):
        if self.nuisance_dim == 0:
            # Need to have at least one dimension for generating the component, even if not used
            self.nuisance_scale = 0.0
            self.nuisance_dim = 1


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class PlacefieldFullGenerator:
    """Place-field generator conforming to the ``stim_full`` atlas contract.

    Exposes ``config``, ``true_covariance()``, ``stim_space`` / ``stim_spectrum`` /
    ``stim_latents``, and ``generate(...)`` so the shared analysis engine treats it like any other
    ``stim_full`` generator. ``supports_rcvpca`` advertises the extra rCVPCA estimator (the engine
    only runs rCVPCA for generators that opt in and emit per-repeat maps).
    """

    supports_rcvpca: bool = True

    def __init__(self, config: PlacefieldFullConfig, dtype: npt.DTypeLike = np.float64):
        self.config = config
        self.dtype = np.dtype(dtype)
        rng = config.rng if config.rng is not None else np.random.default_rng()

        N, P = config.num_neurons, config.num_positions

        # Clean source tuning curves (N, P), built once from the construction RNG.
        source = config.field_model.build_source(N, P, rng, self.dtype)

        # Optional deterministic per-neuron normalizer (clean-source max), applied to the source
        # here so EVERY downstream quantity (repeats, noises, covariances) is in the same space.
        if config.normalize:
            self.neuron_normalizer = np.maximum(np.abs(source).max(axis=1, keepdims=True), 1e-8).astype(self.dtype)
            source = source / self.neuron_normalizer
        else:
            self.neuron_normalizer = None
        self.source = source.astype(self.dtype, copy=False)

        # Cholesky factor of the per-repeat GP noise kernel, reused across generate() calls.
        self._noise_kernel = _rbf_kernel(P, config.repeat_noise_lengthscale, exponent=None, dtype=self.dtype)
        self._noise_chol = _chol(self._noise_kernel)

        # stim_space / stim_spectrum / stim_latents from the economy SVD of the (uncentered)
        # source so that stim_space @ diag(sqrt(stim_spectrum)) @ stim_latents == source exactly.
        # The population block re-centers this product via _precov, so it recovers Sigma_stim.
        U, S, Vt = np.linalg.svd(self.source, full_matrices=False)
        self.stim_space = U.astype(self.dtype, copy=False)  # (N, r), orthonormal columns
        self.stim_spectrum = (S**2).astype(self.dtype, copy=False)  # (r,), sqrt -> singular values
        self.stim_latents = Vt.astype(self.dtype, copy=False)  # (r, P), orthonormal rows

        # Build nuisance space
        self.nuisance_space = generate_orthonormal(N, config.nuisance_dim, rng=rng).astype(self.dtype)
        self.nuisance_spectrum = np.arange(1, config.nuisance_dim + 1, dtype=self.dtype) ** (-config.alpha_nuisance)

    def true_covariance(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Population (Sigma_stim, Sigma_nuisance, Sigma_eps), each (N, N).

        Sigma_stim is the covariance of the clean source over positions. Sigma_nuisance and
        Sigma_eps are isotropic: the per-repeat GP noise contributes ``alpha^2 * I`` marginally
        (exact for the smooth-GP model; an approximation under threshold/ReLU/Tilbury
        nonlinearity), and the intrinsic i.i.d. noise contributes ``noise_level^2 * I``.

        NOTE: The sigma_nuisance and sigma_eps are approximate because noise is rectified.
        """
        N = self.config.num_neurons
        sigma_stim = np.cov(self.source, rowvar=True).astype(self.dtype, copy=False)

        # Per repeat noise covariance from GP noise
        P = self.config.num_positions
        H = np.eye(P) - np.ones((P, P)) / P
        sigma_repeat_noise = self.config.repeat_noise_alpha**2 * np.trace(H @ self._noise_kernel) / (P - 1)
        sigma_nuisance = (self.config.nuisance_scale**2) * self.nuisance_space @ np.diag(self.nuisance_spectrum) @ self.nuisance_space.T
        sigma_eps = (self.config.noise_level**2) * np.eye(N, dtype=self.dtype)
        return sigma_stim, sigma_nuisance + sigma_repeat_noise, sigma_eps

    def _sample_repeat_noise(self, num_neurons: int, rng: np.random.Generator) -> npt.NDArray[np.floating]:
        """One per-repeat GP noise field per neuron, shape (N, P)."""
        return _sample_gp(self._noise_chol, num_neurons, rng, self.dtype)

    def generate(
        self,
        num_samples: int,
        noise_variance: float = 0.0,
        rotation_angle: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        return_extras: bool = False,
    ) -> tuple[npt.NDArray[np.floating], ...]:
        """Generate balanced repeat blocks of place-field activity.

        Parameters
        ----------
        num_samples : int
            Total samples; must be a multiple of ``num_positions`` (one column per
            (repeat, position)). ``R = num_samples / P`` repeats are built.
        noise_variance : float
            Variance of additional i.i.d. Gaussian noise on top of the intrinsic ``noise_level``.
        rotation_angle : float
            No-op for place fields (no session-to-session drift analog); accepted for contract
            compatibility with the stim_full pipeline.
        rng : np.random.Generator, optional
            Sampling RNG; falls back to ``config.rng`` then a fresh default.
        return_extras : bool
            If True, also return an extras dict with ``stim_indices`` (position label per column),
            ``repeat_indices`` (repeat label per column), and ``repeat_maps`` (list of R (N, P)
            arrays) for estimators like rCVPCA that need the explicit repeat structure.

        Returns
        -------
        data : (N, R*P) ndarray
            Full activity, columns ordered repeat-major then position.
        stim_data : (N, R*P) ndarray
            Clean source tiled across repeats (the position signal g(s_t)).
        extras : dict, optional
        """
        rng = rng if rng is not None else self.config.rng
        rng = np.random.default_rng() if rng is None else rng

        N, P = self.config.num_neurons, self.config.num_positions
        if num_samples % P != 0:
            raise ValueError(f"num_samples ({num_samples}) must be a multiple of num_positions ({P}) for balanced repeat blocks.")
        n_repeats = num_samples // P

        field_model = self.config.field_model
        alpha = self.config.repeat_noise_alpha

        # Prepare nuisance components
        nuisance_space = self.nuisance_space.copy()
        if rotation_angle != 0.0:
            nuisance_space = rotate_subspace_by_angle(nuisance_space, rotation_angle, rng)

        nuisance_loadings = (
            self.config.nuisance_scale
            * np.diag(np.sqrt(self.nuisance_spectrum))
            @ rng.standard_normal((self.config.nuisance_dim, num_samples)).astype(self.dtype)
        )
        nuisance_component = nuisance_space @ nuisance_loadings  # (N, T)
        nuisance_repeats = np.reshape(nuisance_component, (N, n_repeats, P)).transpose(1, 0, 2)  # (R, N, P)]

        repeat_maps: list[npt.NDArray[np.floating]] = []
        for r in range(n_repeats):
            noisy = self.source.copy()
            if alpha > 0:
                noisy = noisy + alpha * self._sample_repeat_noise(N, rng)
            if self.config.noise_level > 0:
                noisy = noisy + self.config.noise_level * rng.standard_normal((N, P)).astype(self.dtype, copy=False)
            if noise_variance > 0:
                noisy = noisy + np.sqrt(noise_variance) * rng.standard_normal((N, P)).astype(self.dtype, copy=False)
            noisy = noisy + nuisance_repeats[r]

            repeat = field_model.rectify_repeat(noisy)
            repeat_maps.append(repeat.astype(self.dtype, copy=False))

        data = np.concatenate(repeat_maps, axis=1)  # (N, R*P)
        stim_data = np.tile(self.source, (1, n_repeats))  # (N, R*P)

        if return_extras:
            extras: dict[str, Any] = {
                "stim_indices": np.tile(np.arange(P), n_repeats),
                "repeat_indices": np.repeat(np.arange(n_repeats), P),
                "repeat_maps": repeat_maps,
                "nuisance_space": nuisance_space,
                "nuisance_spectrum": self.nuisance_spectrum.copy(),
                "nuisance_component": nuisance_component,
            }
            return data, stim_data, extras
        return data, stim_data
