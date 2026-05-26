"""
Simulation: neuron-space vs position-space cvPCA with GP place fields.

Generative model
----------------
    D_r = S_r + noise_r

    S_r : spatial signal (n_neurons, n_positions)
        Thresholded GP place fields. Each neuron has a source field drawn from
        GP(0, K_source). Optional peaky_transform reshapes each field to a
        generalized Gaussian (exponent p) at its GP peak. Optional per-repeat
        noise adds GP(0, K_noise) variation.

    noise_r : IID Gaussian, independent per neuron / position / repeat.
"""

import argparse
from dataclasses import dataclass, field, replace

import matplotlib.pyplot as plt
import numpy as np
import optuna
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from dimilibi.cvpca import CVPCA
from dimilibi.helpers import gaussian_filter


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class PlacefieldConfig:
    """GP-based place field generation.

    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    n_positions : int
        Number of spatial bins.
    lengthscale : float
        RBF lengthscale of the source GP (in bin units).
    threshold_pct : float
        Percentile of the source GP sample used as the rectification threshold.
        Higher values -> sparser, more localized fields.
    amplitude : float
        Scaling factor applied after thresholding.
    repeat_noise_alpha : float
        Amplitude of per-repeat GP noise relative to the source field.
        0 = perfectly reproducible repeats; 1 = noise as large as source.
    repeat_noise_lengthscale : float
        Lengthscale for per-repeat GP noise kernel.
    peak_exponent : float or None
        If set, reshape each thresholded field to a generalized Gaussian
        A * exp(-(|theta - phi| / sigma)^p) centered on its GP peak (phi).
        p=2 is Gaussian-like; p=1 is Laplace-like (pointy); p<1 is sharper.
        None disables the transform (default GP bump shape after thresholding).
    peak_sigma_scale : float
        Multiplier on the width (sigma) estimated from each field before reshaping.
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


@dataclass
class SimulationConfig:
    """Top-level simulation config.

    Parameters
    ----------
    placefield : PlacefieldConfig
    noise_level : float
        Std of IID additive noise.
    n_repeats : int
        Number of trial repeats to generate.
    normalize : bool
        Normalize each neuron by its max firing rate across repeats.
    center : bool
        Passed to CVPCA / RegularizedCVPCA.
    n_components : int
        Number of PCA components.
    seed : int
        Base random seed.
    """

    placefield: PlacefieldConfig = field(default_factory=PlacefieldConfig)
    noise_level: float = 1.0
    n_repeats: int = 3
    normalize: bool = True
    center: bool = True
    smooth_width: float = 3.0
    n_components: int = 80
    n_simulations: int = 10
    seed: int = 42


# ---------------------------------------------------------------------------
# GP utilities
# ---------------------------------------------------------------------------


def _rbf_kernel(n_positions: int, lengthscale: float, device: str, exponent: float | None = None) -> torch.Tensor:
    pos = torch.arange(n_positions, dtype=torch.float32, device=device)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    if exponent is None or exponent >= 2.0:
        exponent = 2.0
    return torch.exp(-torch.abs(diff / lengthscale) ** exponent)


def _sample_gp(kernel: torch.Tensor, n_samples: int, generator: torch.Generator) -> torch.Tensor:
    """Sample n_samples independent draws from GP(0, kernel). Returns (n_samples, P)."""
    P = kernel.shape[0]
    jitter = 1e-5 * torch.eye(P, device=kernel.device)
    L = torch.linalg.cholesky(kernel + jitter)
    z = torch.randn(P, n_samples, generator=generator, device=kernel.device)
    return (L @ z).T  # (n_samples, P)


def estimate_spatial_rank(source: torch.Tensor, var_threshold: float = 0.90) -> int:
    """Minimum n PCs of source that explain >= var_threshold of total variance."""
    _, S, _ = torch.linalg.svd(source, full_matrices=False)
    cumvar = torch.cumsum(S**2, dim=0) / (S**2).sum()
    n = int((cumvar < var_threshold).sum()) + 1
    return max(1, min(n, len(S)))


def peaky_transform(
    fields: torch.Tensor,
    peak_exponent: float,
    sigma_scale: float = 1.0,
    min_sigma: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reshape place fields to pointy generalized Gaussians at each row's peak.

      Keeps peak location and amplitude from the input (typically a thresholded GP
    sample) but replaces the spatial profile with
      A * exp(-(|theta - phi| / sigma)^p).

      Parameters
      ----------
      fields : torch.Tensor
          Non-negative fields, shape (n_neurons, n_positions).
      peak_exponent : float
          Exponent p in the generalized Gaussian. Smaller p yields pointier peaks.
      sigma_scale : float
          Multiplier on per-field width estimated from the input profile.
      min_sigma : float
          Lower bound on sigma in bin units.
      eps : float
          Small constant for numerical stability.

      Returns
      -------
      torch.Tensor
          Transformed fields, same shape as ``fields``.
    """
    n_positions = fields.shape[1]
    positions = torch.arange(n_positions, dtype=fields.dtype, device=fields.device)

    peak_idx = fields.argmax(dim=1)
    peak_amp = fields.gather(1, peak_idx.unsqueeze(1)).squeeze(1).clamp(min=eps)
    phi = peak_idx.to(fields.dtype)

    dist = torch.abs(positions.unsqueeze(0) - phi.unsqueeze(1))
    weights = fields.clamp(min=0)
    mass = weights.sum(dim=1).clamp(min=eps)
    mean_d = (weights * dist).sum(dim=1) / mass
    var_d = (weights * (dist - mean_d.unsqueeze(1)) ** 2).sum(dim=1) / mass
    sigma = (var_d.clamp(min=eps).sqrt() * sigma_scale).clamp(min=min_sigma)

    profile = torch.exp(-((dist / sigma.unsqueeze(1)) ** peak_exponent))
    return peak_amp.unsqueeze(1) * profile


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------


def generate_placefields(cfg: PlacefieldConfig, n_repeats: int, seed: int, device: str = "cpu") -> dict:
    """Generate GP-based place field repeats.

    Returns
    -------
    dict with:
        source : (n_neurons, n_positions) -- thresholded source fields
        repeats : list of n_repeats tensors, each (n_neurons, n_positions)
    """

    def rng(s):
        return torch.Generator(device=device).manual_seed(s)

    K_source = _rbf_kernel(cfg.n_positions, cfg.lengthscale, device, exponent=cfg.peak_exponent)
    K_noise = _rbf_kernel(cfg.n_positions, cfg.repeat_noise_lengthscale, device)

    raw = _sample_gp(K_source, cfg.n_neurons, rng(seed))  # (N, P)
    threshold = torch.quantile(raw, cfg.threshold_pct / 100.0, dim=1, keepdim=True)
    source = cfg.amplitude * torch.relu(raw - threshold)  # (N, P)

    # Need this adjustment for flat peaks (p > 2.0), will remove any multi-peaked gaussians!
    if cfg.peak_exponent is not None and cfg.peak_exponent > 2.0:
        source = peaky_transform(source, cfg.peak_exponent, sigma_scale=cfg.peak_sigma_scale)

    repeats = []
    for r in range(n_repeats):
        if cfg.repeat_noise_alpha > 0:
            noise = _sample_gp(K_noise, cfg.n_neurons, rng(seed + 100 * (r + 1)))  # (N, P)
            repeat = torch.relu(source + cfg.repeat_noise_alpha * noise)
        else:
            repeat = source.clone()
        repeats.append(repeat)

    return {"source": source, "repeats": repeats, "spatial_rank": estimate_spatial_rank(source)}


def generate_noise(noise_level: float, n_neurons: int, n_positions: int, n_repeats: int, seed: int, device: str = "cpu") -> list[torch.Tensor]:
    def rng(s):
        return torch.Generator(device=device).manual_seed(s)

    return [noise_level * torch.randn(n_neurons, n_positions, generator=rng(seed + r), device=device) for r in range(n_repeats)]


# ---------------------------------------------------------------------------
# Assembly and normalization
# ---------------------------------------------------------------------------


def assemble(spatial: list, noise: list) -> list[torch.Tensor]:
    return [s + e for s, e in zip(spatial, noise)]


def normalize_by_max(repeats: list[torch.Tensor]) -> list[torch.Tensor]:
    """Normalize each neuron by its max value across all repeats."""
    max_val = torch.stack(repeats, dim=0).amax(dim=(0, 2), keepdim=False)  # (N,)
    max_val = max_val.unsqueeze(1).clamp(min=1e-8)
    return [r / max_val for r in repeats]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def run_analysis(repeats: list[torch.Tensor], source: torch.Tensor, normalize: bool, center: bool, smooth_width: float, n_components: int) -> dict:
    """Run CVPCA (raw) and CVPCA (fixed smoothing) in both neuron and position space.

    Uses 3-fold cross-validation across repeats, averaging covariance spectra.

    Returns
    -------
    dict with keys:
        cov_neuron, cov_position : raw CVPCA covariances (n_components,)
        smooth_cov_neuron, smooth_cov_position : fixed-smooth CVPCA covariances (n_components,)
    """
    if normalize:
        repeats = normalize_by_max(repeats)

    n_rep = len(repeats)
    n_comp = min(n_components, repeats[0].shape[0] - 1, repeats[0].shape[1] - 1)

    cov_n, cov_p = [], []
    smooth_cov_n, smooth_cov_p = [], []

    components_n, components_p = [], []
    smooth_components_n, smooth_components_p = [], []

    for ref in range(n_rep):
        c0 = repeats[ref]
        c1 = repeats[(ref + 1) % n_rep]
        c2 = repeats[(ref + 2) % n_rep]
        c0s = gaussian_filter(c0, smooth_width, axis=1)

        _cov_n = CVPCA(num_components=n_comp, center=center, on_stimuli=False).fit(c0)
        _cov_p = CVPCA(num_components=n_comp, center=center, on_stimuli=True).fit(c0)
        _cov_sn = CVPCA(num_components=n_comp, center=center, on_stimuli=False).fit(c0s)
        _cov_sp = CVPCA(num_components=n_comp, center=center, on_stimuli=True).fit(c0s)
        _sn = _cov_n.score(c1, c2)
        _sp = _cov_p.score(c1, c2)
        _ssn = _cov_sn.score(c1, c2)
        _ssp = _cov_sp.score(c1, c2)

        cov_n.append(_sn)
        cov_p.append(_sp)
        smooth_cov_n.append(_ssn)
        smooth_cov_p.append(_ssp)

        components_n.append(_cov_n.pca.get_components())
        components_p.append(_cov_p.pca.get_components())
        smooth_components_n.append(_cov_sn.pca.get_components())
        smooth_components_p.append(_cov_sp.pca.get_components())

    def _mean(lst):
        return torch.stack(lst).mean(dim=0).cpu().numpy()

    eig_neuron = np.sort(torch.linalg.eigvalsh(torch.cov(source)).cpu().numpy())[::-1][:n_comp]
    eig_position = np.sort(torch.linalg.eigvalsh(torch.cov(source.T)).cpu().numpy())[::-1][:n_comp]
    svals = torch.linalg.svdvals(source).cpu().numpy()[:n_comp]
    u, s, v = torch.linalg.svd(source, full_matrices=False)
    source_components_n = u[:, :n_comp]
    source_components_p = v.T[:, :n_comp]

    return {
        "cov_neuron": _mean(cov_n),
        "cov_position": _mean(cov_p),
        "smooth_cov_neuron": _mean(smooth_cov_n),
        "smooth_cov_position": _mean(smooth_cov_p),
        "eig_neuron": eig_neuron,
        "eig_position": eig_position,
        "svals": svals,
        "components_n": torch.stack(components_n),
        "components_p": torch.stack(components_p),
        "smooth_components_n": torch.stack(smooth_components_n),
        "smooth_components_p": torch.stack(smooth_components_p),
        "source_components_n": source_components_n,
        "source_components_p": source_components_p,
    }


def run_simulation(cfg: SimulationConfig, device: str = "cpu") -> dict:
    N = cfg.placefield.n_neurons
    P = cfg.placefield.n_positions
    R = cfg.n_repeats

    pf_data = generate_placefields(cfg.placefield, R, cfg.seed, device)
    noise = generate_noise(cfg.noise_level, N, P, R, cfg.seed + 2000, device)

    repeats = assemble(pf_data["repeats"], noise)
    result = run_analysis(repeats, pf_data["source"], cfg.normalize, cfg.center, cfg.smooth_width, cfg.n_components)
    result["spatial_rank"] = pf_data["spatial_rank"]
    result["pf_data"] = pf_data
    result["noise"] = noise
    return result


def run_simulations(cfg: SimulationConfig, device: str = "cpu") -> dict:
    """Run cfg.n_simulations independent simulations (varying seed) and stack results.

    Returns
    -------
    dict with arrays of shape (n_simulations, n_components) for each covariance key,
    and corresponding frac_neg keys computed per simulation.
    """
    all_results = [run_simulation(replace(cfg, seed=cfg.seed + i), device) for i in tqdm(range(cfg.n_simulations), desc="simulations", leave=False)]

    cov_keys = ["cov_neuron", "cov_position", "smooth_cov_neuron", "smooth_cov_position"]
    stacked = {k: np.stack([r[k] for r in all_results]) for k in cov_keys}
    stacked["pf_data"] = [r["pf_data"] for r in all_results]
    stacked["eig_neuron"] = np.stack([r["eig_neuron"] for r in all_results])
    stacked["eig_position"] = np.stack([r["eig_position"] for r in all_results])
    stacked["svals"] = np.stack([r["svals"] for r in all_results])
    stacked["components_n"] = np.stack([r["components_n"] for r in all_results])
    stacked["components_p"] = np.stack([r["components_p"] for r in all_results])
    stacked["smooth_components_n"] = np.stack([r["smooth_components_n"] for r in all_results])
    stacked["smooth_components_p"] = np.stack([r["smooth_components_p"] for r in all_results])
    stacked["source_components_n"] = np.stack([r["source_components_n"] for r in all_results])
    stacked["source_components_p"] = np.stack([r["source_components_p"] for r in all_results])

    for ck in cov_keys:
        stacked[f"frac_neg_{ck}"] = np.mean(stacked[ck] < 0, axis=0)

    mean_spatial_rank = int(np.round(np.mean([r["spatial_rank"] for r in all_results])))
    stacked["burn_in"] = mean_spatial_rank

    return stacked


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_placefields(cfg: SimulationConfig, device: str = "cpu") -> plt.Figure:
    """Heatmaps of train and test placefields (with all noise), sorted by train peak position."""
    N, P, R = cfg.placefield.n_neurons, cfg.placefield.n_positions, cfg.n_repeats

    pf_data = generate_placefields(cfg.placefield, R, cfg.seed, device)
    noise = generate_noise(cfg.noise_level, N, P, R, cfg.seed + 2000, device)
    repeats = assemble(pf_data["repeats"], noise)

    if cfg.normalize:
        repeats = normalize_by_max(repeats)

    train = repeats[0].cpu().numpy()
    test = repeats[1].cpu().numpy()
    sort_idx = np.argsort(np.argmax(train, axis=1))

    vmax = max(train.max(), test.max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in [(axes[0], train, "Train (repeat 1)"), (axes[1], test, "Test (repeat 2)")]:
        im = ax.imshow(data[sort_idx], aspect="auto", interpolation="none", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Position bin")
        ax.set_ylabel("Neuron (sorted by train peak)")
        plt.colorbar(im, ax=ax, label="activity")

    burn_in = pf_data["spatial_rank"]
    fig.suptitle(f"spatial_rank={pf_data['spatial_rank']} burn_in={burn_in}", fontsize=10)
    plt.tight_layout()
    return fig


def plot_population(repeats: list[torch.Tensor], extras: dict, cfg: SimulationConfig) -> plt.Figure:
    """Four-panel plot: spatial source, train repeat, test repeat.

    Neurons are sorted by their peak position in the source place fields so
    spatial structure appears as a diagonal in the left panels.
    """
    source = extras["source"].cpu().numpy()  # (N, P), non-negative
    train = repeats[0].cpu().numpy()
    test = repeats[1].cpu().numpy()

    sort_idx = np.argsort(np.argmax(source, axis=1))

    panels = [
        (source[sort_idx], "Spatial source\n(place fields)", "viridis", 0, source.max()),
        (train[sort_idx], "Train (repeat 1)", "RdBu_r", None, None),
        (test[sort_idx], "Test (repeat 2)", "RdBu_r", None, None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        if vmin is None:
            vmax = np.abs(data).max()
            vmin = -vmax
        im = ax.imshow(data, aspect="auto", interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Position bin")
        plt.colorbar(im, ax=ax, shrink=0.75)
    axes[0].set_ylabel("Neuron (sorted by source peak)")

    sr = extras["spatial_rank"]
    fig.suptitle(
        f"spatial_rank={sr}  "
        f"pf_amp={cfg.placefield.amplitude:.1f} pf_ls={cfg.placefield.lengthscale:.1f}  "
        f"peak_p={cfg.placefield.peak_exponent}  "
        f"noise={cfg.noise_level:.1f}  normalize={cfg.normalize}",
        fontsize=10,
    )
    plt.tight_layout()
    return fig


_SERIES = [
    ("cov_neuron", "frac_neg_cov_neuron", "steelblue", "-", 1.8, "neuron cvpca"),
    ("cov_position", "frac_neg_cov_position", "tomato", "-", 1.8, "position cvpca"),
    ("smooth_cov_neuron", "frac_neg_smooth_cov_neuron", "steelblue", "--", 1.2, "smooth neuron cvpca"),
    ("smooth_cov_position", "frac_neg_smooth_cov_position", "tomato", "--", 1.2, "smooth position cvpca"),
]


def plot_results(results: dict[str, dict], suptitle: str = "") -> plt.Figure:
    """Plot log covariance (mean) and fraction-negative (mean ± 1 std) across simulations.

    Parameters
    ----------
    results : dict[str, stacked_dict]
        Keys are condition labels; values are outputs of run_simulations().
        Each value has arrays of shape (n_simulations, n_components).
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    for col, (label, r) in enumerate(results.items()):
        n_comp = r["cov_neuron"].shape[1]
        dims = np.arange(1, n_comp + 1)
        ax_c, ax_f = axes[0, col], axes[1, col]

        # Plot true spectrum of source place field data
        svals = (r["svals"] ** 2).mean(axis=0)
        ax_c.plot(dims, svals / svals.sum(), color="black", lw=1.0, label="True $\lambda$")

        # --- Log covariance (mean; negatives naturally absent from log scale) ---
        for ck, _fk, color, ls, lw, name in _SERIES:
            if ck not in r:
                continue
            mean_cov = r[ck].mean(axis=0)
            sum_cov = np.nansum(mean_cov)
            pos = np.where(mean_cov > 0, mean_cov, np.nan)
            ax_c.plot(dims, pos / sum_cov, color=color, ls=ls, lw=lw, label=name)

        ax_c.set_yscale("log")
        ax_c.set_title(label, fontsize=10)
        ax_c.set_xlabel("Dimension")
        if col == 0:
            ax_c.set_ylabel("CV covariance (log)")
        ax_c.legend(fontsize=7)

        # --- Fraction negative (mean ± 1 std across simulations) ---
        ax_f.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        for _ck, fk, color, ls, lw, name in _SERIES:
            if fk not in r:
                continue
            mean_fn = r[fk]
            ax_f.plot(dims, mean_fn, color=color, ls=ls, lw=lw, label=name)
            # ax_f.fill_between(dims, mean_fn - std_fn, mean_fn + std_fn, color=color, alpha=0.15)
        ax_f.set_ylim(-0.05, 1.05)
        ax_f.set_xlabel("Dimension")
        if col == 0:
            ax_f.set_ylabel("Fraction < 0 (cumulative)")
        if col == 0:
            ax_f.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_components(results: dict[str, dict], suptitle: str = "") -> plt.Figure:
    """Plot log covariance (mean) and fraction-negative (mean ± 1 std) across simulations.

    Parameters
    ----------
    results : dict[str, stacked_dict]
        Keys are condition labels; values are outputs of run_simulations().
        Each value has arrays of shape (n_simulations, n_components).
    """
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), height_ratios=[1, 0.5], squeeze=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    source = [results["source_components_n"], results["source_components_p"], results["source_components_n"], results["source_components_p"]]
    target = [results["components_n"], results["components_p"], results["smooth_components_n"], results["smooth_components_p"]]
    names = ["neuron", "position", "smooth neuron", "smooth position"]
    for col, (s, t) in enumerate(zip(source, target)):
        n_comp = s.shape[2]
        extent = [0.5, n_comp + 0.5, n_comp + 0.5, 0.5]
        xvals = np.arange(1, n_comp + 1)
        ax_c = axes[0, col]
        ax_m = axes[1, col]

        cross = np.einsum("sda, srdb -> srab", s, t)
        subspace = np.sum(cross**2, axis=3)

        ax_c.imshow(np.abs(cross[0, 0]), aspect="auto", interpolation="none", extent=extent, cmap="gray_r", vmin=0, vmax=1)
        ax_m.plot(xvals, np.mean(subspace, axis=(0, 1)), color="black")
        ax_c.set_title(names[col], fontsize=10)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Optuna study
# ---------------------------------------------------------------------------

_PARAM_NAMES = [
    "pf_amplitude",
    "pf_lengthscale",
    "pf_threshold_pct",
    "pf_repeat_noise_alpha",
    "peak_exponent",
    "noise_level",
    "smooth_width",
]
_PARAM_LABELS = ["pf amp", "pf ls", "threshold %", "repeat α", "peak p", "noise", "smooth"]


def frac_neg(cov: np.ndarray) -> np.ndarray:
    """Cumulative fraction of negative values among the first k components."""
    return np.array([(cov[:k] < 0).mean() for k in range(1, len(cov) + 1)])


def _stack_cumulative_frac_neg(stacked: dict, cov_key: str) -> np.ndarray:
    """Per-simulation cumulative frac_neg curves, shape (n_simulations, n_components)."""
    return np.stack([frac_neg(row) for row in stacked[cov_key]])


def _compute_asymmetry(stacked: dict) -> float:
    """Single objective: mean asymmetry after burn_in dims, averaged over raw + smooth variants.

    asymmetry = frac_neg_position[burn_in:] - frac_neg_neuron[burn_in:]

    Positive when position-space hits the noise floor faster than neuron-space.
    Computed from covariance stacks (not the 1D frac_neg keys in ``run_simulations``).
    """
    burn_in = min(int(stacked["burn_in"]), stacked["cov_neuron"].shape[1] - 1)
    fn_n = _stack_cumulative_frac_neg(stacked, "cov_neuron")
    fn_p = _stack_cumulative_frac_neg(stacked, "cov_position")
    fn_sn = _stack_cumulative_frac_neg(stacked, "smooth_cov_neuron")
    fn_sp = _stack_cumulative_frac_neg(stacked, "smooth_cov_position")
    asym_raw = fn_p[:, burn_in:].mean() - fn_n[:, burn_in:].mean()
    asym_smooth = fn_sp[:, burn_in:].mean() - fn_sn[:, burn_in:].mean()
    return float((asym_raw + asym_smooth) / 2)


def _suggest_config(trial: optuna.Trial, base: SimulationConfig) -> SimulationConfig:
    pf = replace(
        base.placefield,
        amplitude=trial.suggest_float("pf_amplitude", 2.0, 30.0, log=True),
        lengthscale=trial.suggest_float("pf_lengthscale", 2.0, 20.0, log=True),
        threshold_pct=trial.suggest_float("pf_threshold_pct", 20.0, 85.0),
        repeat_noise_alpha=trial.suggest_float("pf_repeat_noise_alpha", 0.0, 0.9),
        peak_exponent=trial.suggest_float("peak_exponent", 0.1, 4.0),
    )
    return replace(
        base,
        placefield=pf,
        noise_level=trial.suggest_float("noise_level", 0.1, 5.0, log=True),
        smooth_width=trial.suggest_float("smooth_width", 0.5, 20.0, log=True),
    )


def config_from_optuna_params(params: dict, base: SimulationConfig) -> SimulationConfig:
    """Rebuild ``SimulationConfig`` from Optuna trial params, keeping non-tuned fields from ``base``."""
    pf = replace(
        base.placefield,
        amplitude=params["pf_amplitude"],
        lengthscale=params["pf_lengthscale"],
        threshold_pct=params["pf_threshold_pct"],
        repeat_noise_alpha=params["pf_repeat_noise_alpha"],
        peak_exponent=params["peak_exponent"],
    )
    return replace(
        base,
        placefield=pf,
        noise_level=params["noise_level"],
        smooth_width=params["smooth_width"],
    )


def run_optuna_study(
    base: SimulationConfig,
    n_trials: int = 200,
    n_sims_per_trial: int = 5,
    device: str = "cpu",
    seed: int = 0,
) -> optuna.Study:
    """Single-objective Optuna study maximizing asymmetry after burn_in dims.

    asymmetry = mean(frac_neg_position - frac_neg_neuron) over dims > burn_in,
    averaged across raw and smooth variants.
    Uses TPE. n_sims_per_trial controls seed-averaging noise within each trial.
    """
    trial_base = replace(base, n_simulations=n_sims_per_trial)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial: optuna.Trial) -> float:
        cfg = _suggest_config(trial, trial_base)
        stacked = run_simulations(cfg, device)
        trial.set_user_attr("burn_in", int(stacked["burn_in"]))
        return _compute_asymmetry(stacked)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def _completed_trials(study: optuna.Study) -> tuple[list, np.ndarray]:
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return trials, np.array([t.value for t in trials])


def plot_optuna_results(study: optuna.Study) -> dict[str, go.Figure]:
    """Return 'history' and 'parallel' Plotly figures for a single-objective study."""
    trials, values = _completed_trials(study)
    trial_nums = np.array([t.number for t in trials])
    running_best = np.maximum.accumulate(values)

    hover = [
        (
            f"trial={t.number}  asymmetry={t.value:.4f}  burn_in={t.user_attrs.get('burn_in', '?')}<br>"
            f"pf_amp={t.params['pf_amplitude']:.2f}  ls={t.params['pf_lengthscale']:.2f}  thr={t.params['pf_threshold_pct']:.1f}<br>"
            f"pf_alpha={t.params['pf_repeat_noise_alpha']:.2f}  peak_p={t.params['peak_exponent']:.2f}<br>"
            f"noise={t.params['noise_level']:.2f}  smooth={t.params['smooth_width']:.2f}"
        )
        for t in trials
    ]

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=trial_nums,
            y=values,
            mode="markers",
            marker=dict(color=values, colorscale="Viridis", size=6, opacity=0.7, showscale=True, colorbar=dict(title="asymmetry")),
            text=hover,
            hoverinfo="text",
            name="trials",
        )
    )
    fig_hist.add_trace(
        go.Scatter(
            x=trial_nums,
            y=running_best,
            mode="lines",
            line=dict(color="red", width=2),
            name="running best",
        )
    )
    fig_hist.update_layout(
        title="Optimization history — asymmetry",
        xaxis_title="Trial",
        yaxis_title="Asymmetry",
        height=400,
        width=800,
    )

    all_params = {name: [t.params[name] for t in trials] for name in _PARAM_NAMES}
    dimensions = [dict(label=lbl, values=all_params[name]) for name, lbl in zip(_PARAM_NAMES, _PARAM_LABELS)]
    dimensions.append(dict(label="asymmetry", values=values.tolist()))

    fig_par = go.Figure(
        go.Parcoords(
            line=dict(color=values, colorscale="Viridis", showscale=True, colorbar=dict(title="asymmetry")),
            dimensions=dimensions,
        )
    )
    fig_par.update_layout(title="Parallel coordinates — colored by asymmetry", height=500, width=1100)

    return {"history": fig_hist, "parallel": fig_par}


def print_best_configs(study: optuna.Study, top_n: int = 5) -> None:
    """Print top_n trials sorted by asymmetry."""
    trials, _ = _completed_trials(study)
    top = sorted(trials, key=lambda t: t.value, reverse=True)[:top_n]

    print(f"\n=== Top {top_n} configs by asymmetry ===")
    for rank, t in enumerate(top, 1):
        p = t.params
        burn_in = t.user_attrs.get("burn_in", "?")
        print(
            f"#{rank}  asymmetry={t.value:.4f}  trial={t.number}  burn_in={burn_in}\n"
            f"    pf_amp={p['pf_amplitude']:.2f}  ls={p['pf_lengthscale']:.2f}  "
            f"thr={p['pf_threshold_pct']:.1f}  alpha={p['pf_repeat_noise_alpha']:.2f}  "
            f"peak_p={p['peak_exponent']:.2f}\n"
            f"    noise={p['noise_level']:.2f}  smooth={p['smooth_width']:.2f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

BASE = SimulationConfig(
    placefield=PlacefieldConfig(
        n_neurons=500,
        n_positions=100,
        lengthscale=8.0,
        threshold_pct=60.0,
        amplitude=10.0,
        repeat_noise_alpha=0.3,
        repeat_noise_lengthscale=5.0,
    ),
    noise_level=1.0,
    n_repeats=3,
    normalize=True,
    center=True,
    smooth_width=3.0,
    n_components=80,
    n_simulations=20,
    seed=42,
)


def build_parser(defaults: SimulationConfig = BASE) -> argparse.ArgumentParser:
    """Build CLI parser; defaults match ``defaults`` (``BASE`` by default)."""
    pf = defaults.placefield
    parser = argparse.ArgumentParser(
        description="Neuron vs position cvPCA simulation with GP place fields.",
    )
    g_pf = parser.add_argument_group("place fields")
    g_pf.add_argument("--n-neurons", type=int, default=pf.n_neurons)
    g_pf.add_argument("--n-positions", type=int, default=pf.n_positions)
    g_pf.add_argument("--lengthscale", type=float, default=pf.lengthscale)
    g_pf.add_argument("--threshold-pct", type=float, default=pf.threshold_pct)
    g_pf.add_argument("--amplitude", type=float, default=pf.amplitude)
    g_pf.add_argument("--repeat-noise-alpha", type=float, default=pf.repeat_noise_alpha)
    g_pf.add_argument("--repeat-noise-lengthscale", type=float, default=pf.repeat_noise_lengthscale)
    g_pf.add_argument(
        "--peak-exponent",
        type=float,
        default=pf.peak_exponent,
        metavar="P",
        help="Generalized Gaussian exponent p (pointy peaks; omit arg default=None = smooth GP bumps)",
    )
    g_pf.add_argument("--peak-sigma-scale", type=float, default=pf.peak_sigma_scale)

    g_sim = parser.add_argument_group("simulation")
    g_sim.add_argument("--noise-level", type=float, default=defaults.noise_level)
    g_sim.add_argument("--n-repeats", type=int, default=defaults.n_repeats)
    g_sim.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalize,
        help="Normalize each neuron by max firing rate across repeats",
    )
    g_sim.add_argument(
        "--center",
        action=argparse.BooleanOptionalAction,
        default=defaults.center,
        help="Center data in CVPCA",
    )
    g_sim.add_argument("--smooth-width", type=float, default=defaults.smooth_width)
    g_sim.add_argument("--n-components", type=int, default=defaults.n_components)
    g_sim.add_argument("--n-simulations", type=int, default=defaults.n_simulations)
    g_sim.add_argument("--seed", type=int, default=defaults.seed)
    g_sim.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Torch device for generation and analysis",
    )

    g_run = parser.add_argument_group("run control")
    g_run.add_argument(
        "--skip-simulations",
        action="store_true",
        help="Only plot placefield heatmaps (skip cvPCA ensemble)",
    )
    g_run.add_argument(
        "--compare-peaky",
        action="store_true",
        help="After main heatmaps, also plot smooth vs peak_exponent=1 comparison",
    )
    g_run.add_argument(
        "--optimize",
        action="store_true",
        help="After the normal run, maximize neuron/position asymmetry with Optuna, then re-run with the best trial",
    )
    g_run.add_argument("--n-trials", type=int, default=200, help="Optuna trials when --optimize is set")
    g_run.add_argument(
        "--n-sims-per-trial",
        type=int,
        default=5,
        help="Simulations per Optuna trial (seed averaging); does not change --n-simulations for normal runs",
    )
    g_run.add_argument("--optuna-seed", type=int, default=0, help="TPE sampler seed when --optimize is set")
    return parser


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    """Construct ``SimulationConfig`` from parsed CLI arguments."""
    placefield = PlacefieldConfig(
        n_neurons=args.n_neurons,
        n_positions=args.n_positions,
        lengthscale=args.lengthscale,
        threshold_pct=args.threshold_pct,
        amplitude=args.amplitude,
        repeat_noise_alpha=args.repeat_noise_alpha,
        repeat_noise_lengthscale=args.repeat_noise_lengthscale,
        peak_exponent=args.peak_exponent,
        peak_sigma_scale=args.peak_sigma_scale,
    )
    return SimulationConfig(
        placefield=placefield,
        noise_level=args.noise_level,
        n_repeats=args.n_repeats,
        normalize=args.normalize,
        center=args.center,
        smooth_width=args.smooth_width,
        n_components=args.n_components,
        n_simulations=args.n_simulations,
        seed=args.seed,
    )


def _run_default_pipeline(cfg: SimulationConfig, args: argparse.Namespace, suptitle: str) -> None:
    """Ensemble cvPCA plots, placefield heatmaps, and optional peaky comparison."""
    device = args.device

    if not args.skip_simulations:
        print(f"Simulation ({cfg.n_simulations} runs, device={device})...")
        result = run_simulations(cfg, device=device)
        print(f"  burn_in={result['burn_in']}  (spatial_rank)")
        dim_hi = min(79, cfg.n_components - 1)
        dim_mid = min(39, cfg.n_components - 1)
        for key, label in [("frac_neg_cov_neuron", "neuron  "), ("frac_neg_cov_position", "position")]:
            fn = result[key]
            print(f"  {label} frac<0 @dim{dim_mid + 1}/dim{dim_hi + 1}: {fn[dim_mid]:.2f} / {fn[dim_hi]:.2f}")
        plot_results({"run": result}, suptitle=suptitle)
        plt.show()

    print("\nPlacefield heatmaps...")
    plot_placefields(cfg, device=device)
    plt.show()

    if args.compare_peaky and cfg.placefield.peak_exponent is None:
        peaky_cfg = replace(cfg, placefield=replace(cfg.placefield, peak_exponent=1.0))
        print("\nPlacefield heatmaps (peaky comparison, p=1)...")
        plot_placefields(peaky_cfg, device=device)
        plt.show()


def main(argv: list[str] | None = None) -> None:
    """Run simulations and/or placefield plots from CLI arguments."""
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    _run_default_pipeline(cfg, args, suptitle="Neuron vs position cvPCA")

    if not args.optimize:
        return

    print("\nRunning Optuna study...")
    study = run_optuna_study(
        cfg,
        n_trials=args.n_trials,
        n_sims_per_trial=args.n_sims_per_trial,
        device=args.device,
        seed=args.optuna_seed,
    )
    print_best_configs(study)

    figs = plot_optuna_results(study)
    figs["history"].show()
    figs["parallel"].show()

    cfg_best = config_from_optuna_params(study.best_trial.params, cfg)
    _run_default_pipeline(cfg_best, args, suptitle="Neuron vs position cvPCA (best Optuna trial)")


if __name__ == "__main__":
    main()
