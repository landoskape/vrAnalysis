"""
Simulation: neuron-space vs position-space cvPCA with GP place fields.

Generative model
----------------
    D_r = S_r + noise_r

    S_r : spatial signal (n_neurons, n_positions)
        Thresholded GP place fields. Each neuron has a source field drawn from
        GP(0, K_source). Optional per-repeat noise adds GP(0, K_noise) variation.

    noise_r : IID Gaussian, independent per neuron / position / repeat.
"""

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    """

    n_neurons: int = 500
    n_positions: int = 100
    lengthscale: float = 8.0
    threshold_pct: float = 60.0
    amplitude: float = 10.0
    repeat_noise_alpha: float = 0.3
    repeat_noise_lengthscale: float = 5.0


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


def _rbf_kernel(n_positions: int, lengthscale: float, device: str) -> torch.Tensor:
    pos = torch.arange(n_positions, dtype=torch.float32, device=device)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    return torch.exp(-0.5 * diff**2 / lengthscale**2)


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

    K_source = _rbf_kernel(cfg.n_positions, cfg.lengthscale, device)
    K_noise = _rbf_kernel(cfg.n_positions, cfg.repeat_noise_lengthscale, device)

    raw = _sample_gp(K_source, cfg.n_neurons, rng(seed))  # (N, P)
    threshold = torch.quantile(raw, cfg.threshold_pct / 100.0, dim=1, keepdim=True)
    source = torch.relu(raw - threshold)  # (N, P)

    repeats = []
    for r in range(n_repeats):
        if cfg.repeat_noise_alpha > 0:
            noise = _sample_gp(K_noise, cfg.n_neurons, rng(seed + 100 * (r + 1)))  # (N, P)
            repeat = cfg.amplitude * torch.relu(source + cfg.repeat_noise_alpha * noise)
        else:
            repeat = cfg.amplitude * source.clone()
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
# Population generator
# ---------------------------------------------------------------------------


def generate_population(cfg: SimulationConfig, device: str = "cpu") -> tuple[list[torch.Tensor], dict]:
    """Assemble one population dataset and return its component signals.

    Returns
    -------
    repeats : list of n_repeats (n_neurons, n_positions) tensors
        Final assembled (and optionally normalized) data.
    extras : dict
        source : (n_neurons, n_positions)  thresholded GP source fields
        spatial_repeats : list of (n_neurons, n_positions)  spatial signal per repeat
        noise_repeats : list of (n_neurons, n_positions)  IID noise per repeat
        spatial_rank : int
    """
    N = cfg.placefield.n_neurons
    P = cfg.placefield.n_positions
    R = cfg.n_repeats

    pf_data = generate_placefields(cfg.placefield, R, cfg.seed, device)
    noise_repeats = generate_noise(cfg.noise_level, N, P, R, cfg.seed + 2000, device)

    spatial_repeats = pf_data["repeats"]
    repeats = assemble(spatial_repeats, noise_repeats)
    if cfg.normalize:
        repeats = normalize_by_max(repeats)

    extras = {
        "source": pf_data["source"],
        "spatial_repeats": spatial_repeats,
        "noise_repeats": noise_repeats,
        "spatial_rank": pf_data["spatial_rank"],
    }
    return repeats, extras


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


if __name__ == "__main__":
    # Base result: fraction-negative curves + placefield heatmaps
    print(f"Base simulation ({BASE.n_simulations} runs)...")
    base_result = run_simulations(BASE)
    print(f"  burn_in={base_result['burn_in']}  (spatial_rank + ns_modes)")
    for key, label in [("frac_neg_cov_neuron", "neuron  "), ("frac_neg_cov_position", "position")]:
        fn = base_result[key]
        print(f"  {label} frac<0 @dim40/dim79: {fn[39]:.2f} / {fn[78]:.2f}")
    plot_results({"base": base_result}, suptitle="Neuron vs position cvPCA (base config)")
    plt.show()

    print("\nPlacefield heatmaps...")
    plot_placefields(BASE)
    plt.show()
