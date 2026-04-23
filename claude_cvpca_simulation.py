"""
Simulation: neuron-space vs position-space cvPCA

Reproduces the observed asymmetry:
  - neuron-space cvPCA: fraction-negative stays LOW (positive signal in late dims)
  - position-space cvPCA: fraction-negative approaches ~50% after spatial signal exhausted

Generative model
----------------
    X = spatial_signal + ns_signal + noise

    spatial_signal : (n_neurons, n_positions), rank K_s
        Gaussian place field bumps with power-law amplitude decay.
        neuron_loadings (N, K_s) @ spatial_basis (K_s, P)
        CRITICAL: do NOT normalize neuron_loadings per-neuron -- that kills SNR.
        Use amplitude=15.0 so spatial SVs (~140-1080) dominate noise SVs (~30).

    ns_signal : (n_neurons, n_positions), rank K_ns
        Low-frequency (slow oscillation) position pattern shared across repeats,
        with small per-repeat variation. Survives Gaussian smoothing.
        Captured by neuron-space eigenvectors (positive CV cov) but NOT well by
        position-space eigenvectors (not a place-field pattern -> hits noise floor).

    noise : IID Gaussian, independent per neuron/position/repeat.

Root cause of flat spectrum in original script
----------------------------------------------
1. neuron_loadings normalized per-neuron (/= row_norm) reduced signal variance
   by ~1/sqrt(n_neurons). SNR was 0.25x at dim 1 -- completely below noise bulk.
2. Equal-amplitude sinusoidal basis: no power-law decay -> flat signal spectrum.
3. NS per-repeat amplitudes drawn independently -> zero cross-repeat correlation
   -> zero CV covariance contribution from NS.
"""

import torch
import numpy as np
import matplotlib
from tqdm import tqdm

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dimilibi.cvpca import CVPCA, RegularizedCVPCA


# ---------------------------------------------------------------------------
# Gaussian smoothing
# ---------------------------------------------------------------------------


def smooth_gaussian(data: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian smooth along position axis (dim=1). data: (N, P)."""
    if sigma < 0.5:
        return data.clone()
    P = data.shape[1]
    pos = torch.arange(P, dtype=data.dtype, device=data.device)
    kernel = torch.exp(-0.5 * (pos.unsqueeze(0) - pos.unsqueeze(1)) ** 2 / sigma**2)
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    return data @ kernel.T


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_population(
    n_neurons: int = 500,
    n_positions: int = 100,
    n_spatial_dims: int = 8,
    n_ns_modes: int = 30,
    spatial_amplitude: float = 15.0,
    ns_amplitude: float = 4.0,
    ns_variation: float = 0.3,
    noise_level: float = 1.0,
    smoothing_sigma: float = 8.0,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """
    Generate three repeats of a neural population response.

    Returns
    -------
    data : dict
        r1, r2, r3 : (n_neurons, n_positions) assembled repeats
        r1_smooth  : Gaussian-smoothed r1 for fitting eigenvectors
        spatial_rank : int
    extras : dict
        spatial_mean    : (n_neurons, n_positions) shared spatial signal
        spatial_basis   : (n_spatial_dims, n_positions) Gaussian bump basis
        neuron_loadings : (n_neurons, n_spatial_dims) random neural weights
        ns_template     : (n_neurons, n_positions) shared nonsense signal
        ns_modes        : (n_neurons, n_ns_modes) neural modes for nonsense
        ns_pos_template : (n_ns_modes, n_positions) sinusoidal position patterns
        ns_amplitude    : float
        n_ns_modes      : int
        noise_level     : float
    """

    def rng(s):
        return torch.Generator(device=device).manual_seed(seed + s)

    pos_idx = torch.arange(n_positions, dtype=torch.float32, device=device)

    # --- Spatial: Gaussian place fields, power-law amplitude decay ---
    centers = torch.linspace(3, n_positions - 3, n_spatial_dims, device=device)
    field_width = n_positions / 15.0
    alphas = torch.tensor(
        [(1.0 / (k + 1)) ** 0.8 for k in range(n_spatial_dims)],
        dtype=torch.float32,
        device=device,
    )
    spatial_basis = torch.stack(
        [alphas[k] * torch.exp(-0.5 * ((pos_idx - centers[k]) / field_width) ** 2) for k in range(n_spatial_dims)]
    )  # (K_s, P)

    # CRITICAL: do NOT normalize neuron_loadings -- normalization kills SNR
    neuron_loadings = torch.randn(n_neurons, n_spatial_dims, generator=rng(1), device=device)
    spatial_mean = spatial_amplitude * neuron_loadings @ spatial_basis  # (N, P)

    # --- NS: low-frequency position modulation, shared + per-repeat variation ---
    ns_modes = torch.randn(n_neurons, n_ns_modes, generator=rng(2), device=device)
    t = pos_idx / n_positions
    ns_pos_template = torch.stack(
        [torch.sin(2 * torch.pi * (k + 1) * t / max(1, n_ns_modes // 3)) for k in range(n_ns_modes)]
    )  # (K_ns, P) -- slow oscillations, NOT place-field-like
    ns_template = ns_modes @ ns_pos_template  # (N, P) -- shared structure

    def make_repeat(seed_ns: int, seed_noise: int) -> torch.Tensor:
        ns_var = ns_variation * ns_modes @ torch.randn(n_ns_modes, n_positions, generator=rng(seed_ns), device=device)
        ns = ns_amplitude * ns_template + ns_var
        noise = noise_level * torch.randn(n_neurons, n_positions, generator=rng(seed_noise), device=device)
        return spatial_mean + ns + noise

    r1, r2, r3 = make_repeat(100, 200), make_repeat(300, 400), make_repeat(500, 600)
    r1_smooth = smooth_gaussian(r1, smoothing_sigma)

    data = {
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r1_smooth": r1_smooth,
        "spatial_rank": n_spatial_dims,
    }
    extras = {
        "spatial_mean": spatial_mean,
        "spatial_basis": spatial_basis,
        "neuron_loadings": neuron_loadings,
        "ns_template": ns_template,
        "ns_modes": ns_modes,
        "ns_pos_template": ns_pos_template,
        "ns_amplitude": ns_amplitude,
        "n_ns_modes": n_ns_modes,
        "noise_level": noise_level,
    }
    return data, extras


# ---------------------------------------------------------------------------
# cvPCA wrappers (adjust these if CVPCA interface differs)
# ---------------------------------------------------------------------------


def run_cvpca_neuron_space(
    r1_smooth: torch.Tensor,
    r2: torch.Tensor,
    r3: torch.Tensor,
    n_components: int,
) -> np.ndarray:
    """Fit eigenvectors over neurons (N, K), score on r2/r3."""
    cvpca = CVPCA(num_components=n_components, center=True)
    cvpca.fit(r1_smooth)
    return cvpca.score(r2, r3).cpu().numpy()


def run_cvpca_position_space(
    r1_smooth: torch.Tensor,
    r2: torch.Tensor,
    r3: torch.Tensor,
    n_components: int,
) -> np.ndarray:
    """Fit eigenvectors over positions (P, K) by transposing data."""
    cvpca = CVPCA(num_components=n_components, center=True)
    cvpca.fit(r1_smooth.T.contiguous())
    return cvpca.score(r2.T.contiguous(), r3.T.contiguous()).cpu().numpy()


def frac_neg_cumulative(cov: np.ndarray) -> np.ndarray:
    return np.array([(cov[:k] < 0).mean() for k in range(1, len(cov) + 1)])


# ---------------------------------------------------------------------------
# Sanity check: verify data structure without CVPCA
# Call this first to confirm SNR before debugging CVPCA interface.
# ---------------------------------------------------------------------------


def sanity_check(data: dict, extras: dict):
    """Print SVD-based SNR check. Run this before using CVPCA."""
    spatial = extras["spatial_mean"]
    r1, r2, r3 = data["r1"], data["r2"], data["r3"]

    _, S_sig, _ = torch.linalg.svd(spatial, full_matrices=False)
    noise = r1 - spatial
    _, S_noise, _ = torch.linalg.svd(noise, full_matrices=False)

    print("=== Sanity check ===")
    print(f"Signal SVs (first 10): {S_sig[:10].numpy().round(1)}")
    print(f"Noise SVs  (first 10): {S_noise[:10].numpy().round(1)}")
    print(f"SNR dim1: {(S_sig[0]/S_noise[0]).item():.1f}x  " f"SNR dim8: {(S_sig[7]/S_noise[7]).item():.1f}x")

    # Manual cvPCA to confirm CV structure without CVPCA dependency
    mean = r1.mean(dim=1, keepdim=True)
    _, _, Vt = torch.linalg.svd((r1 - mean).T, full_matrices=False)
    V = Vt[:20].T
    cov_manual = (V.T @ (r2 - mean) * (V.T @ (r3 - mean))).mean(dim=1).numpy()
    print(f"Manual CV cov (first 10): {cov_manual[:10].round(1)}")
    print(f"Expected: strong decay from dim1, crossing zero around dim {data['spatial_rank']}")
    print("=" * 20)


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------


def run_simulation(
    n_neurons=500,
    n_positions=100,
    n_components=80,
    n_spatial_dims=8,
    n_ns_modes=30,
    spatial_amplitude=15.0,
    ns_amplitude=4.0,
    ns_variation=0.3,
    noise_level=1.0,
    smoothing_sigma=8.0,
    seed=42,
) -> dict:
    n_comp = min(n_components, n_positions - 1, n_neurons - 1)
    data, _ = generate_population(
        n_neurons=n_neurons,
        n_positions=n_positions,
        n_spatial_dims=n_spatial_dims,
        n_ns_modes=n_ns_modes,
        spatial_amplitude=spatial_amplitude,
        ns_amplitude=ns_amplitude,
        ns_variation=ns_variation,
        noise_level=noise_level,
        smoothing_sigma=smoothing_sigma,
        seed=seed,
    )
    cov_n = run_cvpca_neuron_space(data["r1_smooth"], data["r2"], data["r3"], n_comp)
    cov_p = run_cvpca_position_space(data["r1_smooth"], data["r2"], data["r3"], n_comp)
    return {
        "cov_neuron": cov_n,
        "cov_position": cov_p,
        "frac_neg_neuron": frac_neg_cumulative(cov_n),
        "frac_neg_position": frac_neg_cumulative(cov_p),
        "spatial_rank": data["spatial_rank"],
    }


def run_simulations(n_simulations: int = 10, **kwargs) -> dict:
    """Run n_simulations independent seeds, return averaged cov and stacked frac_neg."""
    seed = kwargs.get("seed", 42)
    all_results = [run_simulation(**{**kwargs, "seed": seed + i}) for i in tqdm(range(n_simulations), desc="simulations", leave=False)]
    cov_n = np.stack([r["cov_neuron"] for r in all_results])  # (n_sims, n_dims)
    cov_p = np.stack([r["cov_position"] for r in all_results])
    return {
        "cov_neuron": cov_n.mean(axis=0),
        "cov_position": cov_p.mean(axis=0),
        "frac_neg_neuron": np.stack([r["frac_neg_neuron"] for r in all_results]),  # (n_sims, n_dims) cumulative
        "frac_neg_position": np.stack([r["frac_neg_position"] for r in all_results]),
        "frac_neg_per_dim_neuron": (cov_n < 0).mean(axis=0),  # (n_dims,) per-dim across sims
        "frac_neg_per_dim_position": (cov_p < 0).mean(axis=0),
        "spatial_rank": int(np.round(np.mean([r["spatial_rank"] for r in all_results]))),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_single(result: dict, title="") -> plt.Figure:
    has_per_dim = "frac_neg_per_dim_neuron" in result
    ncols = 3 if has_per_dim else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    ax_c, ax_f = axes[0], axes[1]
    if title:
        fig.suptitle(title, fontsize=11)
    dims = np.arange(1, len(result["cov_neuron"]) + 1)
    sr = result["spatial_rank"]

    ax_c.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax_c.axvline(sr, color="gray", lw=1, ls=":", alpha=0.6, label=f"spatial rank={sr}")
    ax_c.plot(dims, result["cov_neuron"], color="black", lw=1.5, label="neuron-space")
    ax_c.plot(dims, result["cov_position"], color="red", lw=1.5, label="position-space")
    linthresh = max(abs(result["cov_neuron"]).max() * 1e-3, 1.0)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Dimension")
    ax_c.set_ylabel("CV covariance (log)")
    ax_c.set_title("CV covariance spectrum")
    ax_c.legend(fontsize=8)

    ax_f.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5, label="50% noise floor")
    ax_f.axvline(sr, color="gray", lw=1, ls=":", alpha=0.6)
    for fn, color, label in [
        (result["frac_neg_neuron"], "black", "neuron-space"),
        (result["frac_neg_position"], "red", "position-space"),
    ]:
        if fn.ndim == 2:
            mean, std = fn.mean(axis=0), fn.std(axis=0)
            ax_f.plot(dims, mean, color=color, lw=1.5, label=label)
            ax_f.fill_between(dims, mean - std, mean + std, color=color, alpha=0.2)
        else:
            ax_f.plot(dims, fn, color=color, lw=1.5, label=label)
    ax_f.set_ylim(-0.05, 1.05)
    ax_f.set_xlabel("Dimension")
    ax_f.set_ylabel("Fraction < 0 (cumulative)")
    ax_f.set_title("Fraction negative (cumulative)")
    ax_f.legend(fontsize=8)

    if has_per_dim:
        ax_p = axes[2]
        ax_p.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5, label="50% noise floor")
        ax_p.axvline(sr, color="gray", lw=1, ls=":", alpha=0.6)
        ax_p.plot(dims, result["frac_neg_per_dim_neuron"], color="black", lw=1.5, label="neuron-space")
        ax_p.plot(dims, result["frac_neg_per_dim_position"], color="red", lw=1.5, label="position-space")
        ax_p.set_ylim(-0.05, 1.05)
        ax_p.set_xlabel("Dimension")
        ax_p.set_ylabel("Fraction < 0 across simulations")
        ax_p.set_title("Fraction negative (per dim)")
        ax_p.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_ablation(results: dict, param_name: str) -> plt.Figure:
    keys = list(results.keys())
    has_per_dim = "frac_neg_per_dim_neuron" in next(iter(results.values()))
    nrows = 3 if has_per_dim else 2
    fig, axes = plt.subplots(nrows, len(keys), figsize=(4 * len(keys), 3.5 * nrows))
    fig.suptitle(f"{param_name}", fontsize=12)

    for i, key in enumerate(keys):
        r = results[key]
        dims = np.arange(1, len(r["cov_neuron"]) + 1)
        ax_c, ax_f = axes[0, i], axes[1, i]

        ax_c.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax_c.axvline(r["spatial_rank"], color="gray", lw=1, ls=":", alpha=0.5)
        ax_c.plot(dims, r["cov_neuron"], "k-", lw=1.5, label="neuron")
        ax_c.plot(dims, r["cov_position"], "r-", lw=1.5, label="position")
        linthresh = max(abs(r["cov_neuron"]).max() * 1e-3, 1.0)
        ax_c.set_yscale("log")
        ax_c.set_title(f"{param_name}={key}", fontsize=9)
        if i == 0:
            ax_c.set_ylabel("CV covariance (log)")
        ax_c.legend(fontsize=7)

        ax_f.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax_f.axvline(r["spatial_rank"], color="gray", lw=1, ls=":", alpha=0.5)
        for fn, color in [(r["frac_neg_neuron"], "k"), (r["frac_neg_position"], "r")]:
            if fn.ndim == 2:
                mean, std = fn.mean(axis=0), fn.std(axis=0)
                ax_f.plot(dims, mean, color + "-", lw=1.5)
                ax_f.fill_between(dims, mean - std, mean + std, color=color, alpha=0.2)
            else:
                ax_f.plot(dims, fn, color + "-", lw=1.5)
        ax_f.set_ylim(-0.05, 1.05)
        ax_f.set_xlabel("Dimension")
        if i == 0:
            ax_f.set_ylabel("Fraction < 0 (cumulative)")

        if has_per_dim:
            ax_p = axes[2, i]
            ax_p.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
            ax_p.axvline(r["spatial_rank"], color="gray", lw=1, ls=":", alpha=0.5)
            ax_p.plot(dims, r["frac_neg_per_dim_neuron"], "k-", lw=1.5)
            ax_p.plot(dims, r["frac_neg_per_dim_position"], "r-", lw=1.5)
            ax_p.set_ylim(-0.05, 1.05)
            ax_p.set_xlabel("Dimension")
            if i == 0:
                ax_p.set_ylabel("Fraction < 0 (per dim)")

    plt.tight_layout()
    return fig


def plot_population(data: dict, extras: dict) -> plt.Figure:
    """Four-panel plot: spatial signal, shared nonsense, train repeat, test repeat.

    Neurons are sorted by peak position in the spatial signal so place-field
    structure appears as a diagonal. The nonsense panel reveals its non-spatial,
    global character in contrast.
    """
    spatial = extras["spatial_mean"].cpu().numpy()  # (N, P), signed
    nonsense = (extras["ns_amplitude"] * extras["ns_template"]).cpu().numpy()  # (N, P), signed
    train = data["r1"].cpu().numpy()
    test = data["r2"].cpu().numpy()

    sort_idx = np.argsort(np.argmax(spatial, axis=1))

    panels = [
        (spatial[sort_idx], "Spatial signal\n(shared place fields)", "RdBu_r", None),
        (nonsense[sort_idx], "Nonsense\n(shared ns_template)", "RdBu_r", None),
        (train[sort_idx], "Train (r1)", "RdBu_r", None),
        (test[sort_idx], "Test (r2)", "RdBu_r", None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (d, title, cmap, _) in zip(axes, panels):
        vmax = np.abs(d).max()
        im = ax.imshow(d, aspect="auto", interpolation="none", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Position bin")
        plt.colorbar(im, ax=ax, shrink=0.75)
    axes[0].set_ylabel("Neuron (sorted by spatial peak)")

    sr = data["spatial_rank"]
    fig.suptitle(
        f"spatial_rank={sr}  ns_modes={extras['n_ns_modes']}  " f"ns_amp={extras['ns_amplitude']:.1f}  noise={extras['noise_level']:.1f}",
        fontsize=10,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

BASE = dict(
    n_neurons=1000,
    n_positions=100,
    n_components=80,
    n_spatial_dims=8,
    n_ns_modes=30,
    spatial_amplitude=15.0,
    ns_amplitude=4.0,
    ns_variation=0.3,
    noise_level=1.0,
    smoothing_sigma=8.0,
    seed=42,
)

N_SIMS = 10

if __name__ == "__main__":
    # Sanity check first -- if this shows flat CV cov, CVPCA interface is wrong
    print("Running sanity check (no CVPCA needed)...")
    data, extras = generate_population(**{k: v for k, v in BASE.items() if k in generate_population.__code__.co_varnames})
    sanity_check(data, extras)
    plot_population(data, extras)
    plt.show()

    print(f"\nBase simulation ({N_SIMS} runs)...")
    r = run_simulations(n_simulations=N_SIMS, **BASE)
    fig = plot_single(r, "Neuron-space vs position-space cvPCA (synthetic)")
    plt.show()
    print(f"  neuron   frac<0 @40/79: {r['frac_neg_neuron'].mean(axis=0)[39]:.2f} / {r['frac_neg_neuron'].mean(axis=0)[78]:.2f}")
    print(f"  position frac<0 @40/79: {r['frac_neg_position'].mean(axis=0)[39]:.2f} / {r['frac_neg_position'].mean(axis=0)[78]:.2f}")

    # print("\nns_amplitude sweep...")
    # kw = {k: v for k, v in BASE.items() if k != "ns_amplitude"}
    # kw["smoothing_sigma"] = 0.5
    # abl = {amp: run_simulations(n_simulations=N_SIMS, ns_amplitude=amp, **kw) for amp in (0.0, 1.0, 2.0, 4.0, 8.0)}
    # plot_ablation(abl, "ns_amplitude")
    # plt.show()
    # plt.close()

    # print("smoothing_sigma sweep...")
    # kw = {k: v for k, v in BASE.items() if k != "smoothing_sigma"}
    # kw["ns_amplitude"] = 0.0
    # abl = {s: run_simulations(n_simulations=N_SIMS, smoothing_sigma=s, **kw) for s in (0.5, 4.0, 8.0, 15.0)}
    # plot_ablation(abl, "smoothing_sigma")
    # plt.show()
    # plt.close()

    print("Done.")
