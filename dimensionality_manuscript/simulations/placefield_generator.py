"""
Place field generation: configs, GP/Tilbury generators, assembly utilities.

Public API
----------
PlacefieldConfig     -- GP-based (RBF kernel + threshold + optional peak warp)
TilburyConfig        -- double generalized-Gaussian per neuron
SimConfig            -- top-level config; generator field is PlacefieldConfig | TilburyConfig

generate_repeats     -- unified entry point; returns noise-added, optionally normalized repeats
generate_noise       -- IID Gaussian noise list
assemble             -- add spatial repeats + noise
normalize_by_max     -- divide each neuron by its max across repeats
estimate_spatial_rank -- 90%-variance threshold on source SVD
peaky_transform      -- post-GP amplitude warp toward generalized-Gaussian peaks
"""

from dataclasses import dataclass, field

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacefieldConfig:
    """GP-based (RBF kernel) place field generation.

    Parameters
    ----------
    n_neurons, n_positions : int
        Population and track size.
    lengthscale : float
        RBF kernel lengthscale in bin units.
    threshold_pct : float
        Percentile of the GP sample used as rectification threshold. Higher → sparser.
    amplitude : float
        Scaling applied after thresholding.
    repeat_noise_alpha : float
        Amplitude of per-repeat GP noise relative to source. 0 = identical repeats.
    repeat_noise_lengthscale : float
        Lengthscale of per-repeat GP noise kernel.
    peak_exponent : float or None
        When > 2, warp GP amplitudes toward a generalized-Gaussian shape at each peak.
        None = smooth GP bumps without warp.
    peak_sigma_scale : float
        Width-like scale for the amplitude warp (only used when peak_exponent > 2).
    """

    n_neurons: int = 500
    n_positions: int = 100
    lengthscale: float = 8.0
    threshold_pct: float = 60.0
    amplitude: float = 10.0
    repeat_noise_alpha: float = 0.3
    repeat_noise_lengthscale: float = 5.0
    peak_exponent: float | None = None
    peak_sigma_scale: float = 1.0


@dataclass(frozen=True)
class TilburyConfig:
    """Double generalized-Gaussian place field generation (Tilbury model).

    Each neuron's firing-rate curve is:
        f(θ) = b + A₁·exp(-(|θ−φ₁|/σ₁±)^p₁) + A₂·exp(-(|θ−φ₂|/σ₂±)^p₂)

    where σ± is piecewise: σ_left when θ < φ, σ_right when θ ≥ φ. Setting
    ``amplitude_ratio_beta`` large and ``peak_separation_scale`` 0 recovers a
    single symmetric-Gaussian limit.

    Parameters
    ----------
    n_neurons, n_positions : int
        Population and track size.
    amplitude_mean : float
        Mean primary amplitude A₁ (log-normal).
    amplitude_spread : float
        Std of log(A₁). 0 → all neurons share ``amplitude_mean``.
    amplitude_ratio_beta : float
        Shape β of Beta(1, β) for A₂/A₁. Large β → A₂ ≈ 0 (single bump).
    peak_separation_scale : float
        Std (bins) of φ₂ − φ₁ ~ Normal(0, scale). 0 → bumps coincide.
    sigma_mean : float
        Mean bump width σ (log-normal, shared prior for both bumps).
    sigma_spread : float
        Std of log(σ). 0 → all bumps share ``sigma_mean``.
    sigma_asym_std : float
        Std of log(σ_right / σ_left). 0 → symmetric bumps.
    exponent_mean : float
        Mean exponent p (log-normal). 2 → standard Gaussian.
    exponent_spread : float
        Std of log(p). 0 → all neurons get ``exponent_mean``.
    baseline : float
        Additive baseline b.
    repeat_noise_alpha : float
        Amplitude of per-repeat RBF-GP noise relative to source.
    repeat_noise_lengthscale : float
        Lengthscale of per-repeat GP noise kernel.
    repeat_noise_threshold_pct : float
        Percentile threshold applied per-neuron to each noisy repeat (0 = no threshold).
    """

    n_neurons: int = 500
    n_positions: int = 100
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
    repeat_noise_alpha: float = 0.3
    repeat_noise_lengthscale: float = 5.0
    repeat_noise_threshold_pct: float = 60.0


@dataclass(frozen=True)
class SimConfig:
    """Top-level simulation config.

    Parameters
    ----------
    generator : PlacefieldConfig | TilburyConfig
        Controls which generation model to use and all its parameters.
        Type-dispatch replaces the old (placefield + tilbury + generation_path) triple.
    noise_level : float
        Std of IID additive Gaussian noise applied after spatial generation.
    n_repeats : int
        Number of trial repeats to generate. Must be >= 4 for the rCVPCA-vs-stimspace
        workflow (uses four folds r0/r1/r2/r3).
    normalize : bool
        If True, divide each neuron by its max value across all repeats.
    center : bool
        Passed to CVPCA (mean-center before PCA).
    smooth_width : float or None
        Gaussian smoothing kernel width (bins) applied to the training repeat.
        None → no smoothing.
    n_components : int
        Number of PCA components.
    seed : int
        Base random seed (individual runs offset from this).
    """

    generator: PlacefieldConfig | TilburyConfig = field(default_factory=PlacefieldConfig)
    noise_level: float = 1.0
    n_repeats: int = 4
    normalize: bool = True
    center: bool = True
    smooth_width: float | None = None
    n_components: int = 80
    seed: int = 42


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _n_neurons_positions(cfg: SimConfig) -> tuple[int, int]:
    return cfg.generator.n_neurons, cfg.generator.n_positions


def _rbf_kernel(n_positions: int, lengthscale: float, device: str, exponent: float | None = None) -> torch.Tensor:
    pos = torch.arange(n_positions, dtype=torch.float32, device=device)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    exp = 2.0 if (exponent is None or exponent >= 2.0) else exponent
    return torch.exp(-torch.abs(diff / lengthscale) ** exp)


def _sample_gp(kernel: torch.Tensor, n_samples: int, generator: torch.Generator) -> torch.Tensor:
    """Sample n_samples draws from GP(0, kernel). Returns (n_samples, P)."""
    P = kernel.shape[0]
    jitter = 1e-5 * torch.eye(P, device=kernel.device)
    L = torch.linalg.cholesky(kernel + jitter)
    z = torch.randn(P, n_samples, generator=generator, device=kernel.device)
    return (L @ z).T


def _generate_rbf(cfg: PlacefieldConfig, n_repeats: int, seed: int, device: str) -> dict:
    """GP-based generation. Returns {source, repeats (spatial only), spatial_rank}."""

    def rng(s: int) -> torch.Generator:
        return torch.Generator(device=device).manual_seed(s)

    K_source = _rbf_kernel(cfg.n_positions, cfg.lengthscale, device, exponent=cfg.peak_exponent)
    K_noise = _rbf_kernel(cfg.n_positions, cfg.repeat_noise_lengthscale, device)

    raw = _sample_gp(K_source, cfg.n_neurons, rng(seed))
    threshold = torch.quantile(raw, cfg.threshold_pct / 100.0, dim=1, keepdim=True)
    source = cfg.amplitude * torch.relu(raw - threshold)

    if cfg.peak_exponent is not None and cfg.peak_exponent > 2.0:
        source = peaky_transform(source, cfg.peak_exponent, sigma_scale=cfg.peak_sigma_scale)

    repeats = []
    for r in range(n_repeats):
        if cfg.repeat_noise_alpha > 0:
            noise = _sample_gp(K_noise, cfg.n_neurons, rng(seed + 100 * (r + 1)))
            repeat = torch.relu(source + cfg.repeat_noise_alpha * noise)
        else:
            repeat = source.clone()
        repeats.append(repeat)

    return {"source": source, "repeats": repeats, "spatial_rank": estimate_spatial_rank(source)}


def _sample_tilbury_params(cfg: TilburyConfig, seed: int, device: str) -> dict:
    """Sample per-neuron parameters from TilburyConfig distributions."""
    N = cfg.n_neurons
    P = cfg.n_positions

    def _rng(s: int) -> torch.Generator:
        return torch.Generator().manual_seed(s)

    A1 = (
        torch.exp(float(np.log(cfg.amplitude_mean)) + cfg.amplitude_spread * torch.randn(N, generator=_rng(seed)))
        if cfg.amplitude_spread > 0
        else torch.full((N,), float(cfg.amplitude_mean))
    )

    U = torch.rand(N, generator=_rng(seed + 1))
    A2 = (1.0 - U.pow(1.0 / max(cfg.amplitude_ratio_beta, 1e-6))) * A1

    phi1 = torch.rand(N, generator=_rng(seed + 2)) * P
    if cfg.peak_separation_scale > 0:
        phi2 = (phi1 + cfg.peak_separation_scale * torch.randn(N, generator=_rng(seed + 3))).clamp(0.0, float(P - 1))
    else:
        phi2 = phi1.clone()

    def _sigma_base(s: int) -> torch.Tensor:
        return (
            torch.exp(float(np.log(cfg.sigma_mean)) + cfg.sigma_spread * torch.randn(N, generator=_rng(s)))
            if cfg.sigma_spread > 0
            else torch.full((N,), float(cfg.sigma_mean))
        )

    sb1, sb2 = _sigma_base(seed + 4), _sigma_base(seed + 5)

    def _asym(s: int) -> torch.Tensor:
        return torch.exp(cfg.sigma_asym_std * torch.randn(N, generator=_rng(s))) if cfg.sigma_asym_std > 0 else torch.ones(N)

    asym1, asym2 = _asym(seed + 6), _asym(seed + 7)

    def _exponent(s: int) -> torch.Tensor:
        return (
            torch.exp(float(np.log(cfg.exponent_mean)) + cfg.exponent_spread * torch.randn(N, generator=_rng(s)))
            if cfg.exponent_spread > 0
            else torch.full((N,), float(cfg.exponent_mean))
        )

    params = {
        "phi": torch.stack([phi1, phi2], dim=1),
        "A": torch.stack([A1, A2], dim=1),
        "sigma_left": torch.stack([sb1 / asym1.sqrt(), sb2 / asym2.sqrt()], dim=1),
        "sigma_right": torch.stack([sb1 * asym1.sqrt(), sb2 * asym2.sqrt()], dim=1),
        "p": torch.stack([_exponent(seed + 8), _exponent(seed + 9)], dim=1),
    }
    return {k: v.to(device) for k, v in params.items()}


def _eval_tilbury(
    theta: torch.Tensor,
    phi: torch.Tensor,
    A: torch.Tensor,
    sigma_left: torch.Tensor,
    sigma_right: torch.Tensor,
    p: torch.Tensor,
    baseline: float = 0.0,
) -> torch.Tensor:
    """Evaluate double generalized-Gaussian fields. Returns (N, P)."""
    diff = theta[None, None, :] - phi[:, :, None]  # (N, 2, P)
    sigma = torch.where(diff < 0, sigma_left[:, :, None], sigma_right[:, :, None])
    bumps = A[:, :, None] * torch.exp(-((torch.abs(diff) / sigma.clamp(min=1e-6)) ** p[:, :, None]))
    return baseline + bumps.sum(dim=1)  # (N, P)


def _generate_tilbury(cfg: TilburyConfig, n_repeats: int, seed: int, device: str) -> dict:
    """Tilbury generation. Returns {source, repeats (spatial only), spatial_rank}."""

    def rng(s: int) -> torch.Generator:
        return torch.Generator(device=device).manual_seed(s)

    P = cfg.n_positions
    theta = torch.arange(P, dtype=torch.float32, device=device)
    params = _sample_tilbury_params(cfg, seed, device)
    source = _eval_tilbury(theta, params["phi"], params["A"], params["sigma_left"], params["sigma_right"], params["p"], cfg.baseline)

    K_noise = _rbf_kernel(P, cfg.repeat_noise_lengthscale, device)
    repeats = []
    for r in range(n_repeats):
        if cfg.repeat_noise_alpha > 0:
            noise = _sample_gp(K_noise, source.shape[0], rng(seed + 100 * (r + 1)))
            noisy = source + cfg.repeat_noise_alpha * noise
        else:
            noisy = source
        if cfg.repeat_noise_threshold_pct > 0:
            thr = torch.quantile(noisy, cfg.repeat_noise_threshold_pct / 100.0, dim=1, keepdim=True)
            repeat = torch.relu(noisy - thr)
        else:
            repeat = torch.relu(noisy)
        repeats.append(repeat)

    return {"source": source, "repeats": repeats, "spatial_rank": estimate_spatial_rank(source)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def peaky_transform(
    fields: torch.Tensor,
    peak_exponent: float,
    sigma_scale: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Warp thresholded GP fields toward generalized-Gaussian peak shapes.

    Preserves peak locations and multi-peak structure from the GP sample while
    reshaping the local amplitude profile. For ``peak_exponent=2`` and
    ``sigma_scale=1``, the transform is the identity on positive values.

    Parameters
    ----------
    fields : torch.Tensor
        Non-negative fields, shape (n_neurons, n_positions).
    peak_exponent : float
        Target generalized-Gaussian exponent.
    sigma_scale : float
        Width-like scale. Larger values broaden peaks.
    eps : float
        Small constant for numerical stability.
    """
    fields = fields.clamp(min=0)
    peak_amp = fields.amax(dim=1, keepdim=True).clamp(min=eps)
    normalized = (fields / peak_amp).clamp(min=eps, max=1.0)
    scale = max(float(sigma_scale), eps)
    warped_distance = -torch.log(normalized)
    warped = torch.exp(-(warped_distance ** (peak_exponent / 2.0)) / (scale**peak_exponent))
    return torch.where(fields > 0, peak_amp * warped, torch.zeros_like(fields))


def estimate_spatial_rank(source: torch.Tensor, var_threshold: float = 0.90) -> int:
    """Minimum number of PCs of source that explain >= var_threshold of total variance."""
    _, S, _ = torch.linalg.svd(source, full_matrices=False)
    cumvar = torch.cumsum(S**2, dim=0) / (S**2).sum()
    n = int((cumvar < var_threshold).sum()) + 1
    return max(1, min(n, len(S)))


def generate_noise(
    noise_level: float,
    n_neurons: int,
    n_positions: int,
    n_repeats: int,
    seed: int,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """IID Gaussian noise, one tensor per repeat. Each tensor is (n_neurons, n_positions)."""

    def rng(s: int) -> torch.Generator:
        return torch.Generator(device=device).manual_seed(s)

    return [noise_level * torch.randn(n_neurons, n_positions, generator=rng(seed + r), device=device) for r in range(n_repeats)]


def assemble(spatial: list[torch.Tensor], noise: list[torch.Tensor]) -> list[torch.Tensor]:
    """Add spatial fields and noise element-wise. Returns list of (N, P) tensors."""
    return [s + e for s, e in zip(spatial, noise)]


def normalize_by_max(repeats: list[torch.Tensor]) -> list[torch.Tensor]:
    """Divide each neuron by its max value across all repeats and positions."""
    max_val = torch.stack(repeats, dim=0).amax(dim=(0, 2), keepdim=False)  # (N,)
    max_val = max_val.unsqueeze(1).clamp(min=1e-8)
    return [r / max_val for r in repeats]


def generate_repeats(cfg: SimConfig, seed: int, device: str = "cpu") -> dict:
    """Generate analysis-ready place field repeats for one seed.

    Dispatches on type of cfg.generator, adds IID noise, and applies normalization.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.
    seed : int
        Random seed for this run (offsets internally for noise).
    device : str
        Torch device.

    Returns
    -------
    dict with:
        source  : (N, P) torch.Tensor — clean source fields (before IID noise)
        repeats : list of n_repeats (N, P) tensors — assembled and normalized
        spatial_rank : int — 90%-variance threshold on source
    """
    N, P = _n_neurons_positions(cfg)

    if isinstance(cfg.generator, TilburyConfig):
        pf_data = _generate_tilbury(cfg.generator, cfg.n_repeats, seed, device)
    else:
        pf_data = _generate_rbf(cfg.generator, cfg.n_repeats, seed, device)

    noise = generate_noise(cfg.noise_level, N, P, cfg.n_repeats, seed + 2000, device)
    repeats = assemble(pf_data["repeats"], noise)

    if cfg.normalize:
        repeats = normalize_by_max(repeats)

    return {"source": pf_data["source"], "repeats": repeats, "spatial_rank": pf_data["spatial_rank"]}
