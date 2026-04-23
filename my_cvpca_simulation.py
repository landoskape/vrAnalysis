"""
Simulation: neuron-space vs position-space cvPCA with GP place fields and nonsense correlations.

Generative model
----------------
    D_r = S_r + N_r + noise_r

    S_r : spatial signal (n_neurons, n_positions)
        Thresholded GP place fields. Each neuron has a source field drawn from
        GP(0, K_source). Optional per-repeat noise adds GP(0, K_noise) variation.

    N_r : nonsense correlations (n_neurons, n_positions)
        Shared neural modes (U) with per-repeat position patterns (V_r).
        U is fixed across repeats; V_r varies. Not localized in position space.

    noise_r : IID Gaussian, independent per neuron / position / repeat.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch
import matplotlib.pyplot as plt
import optuna
import plotly.graph_objects as go
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
class NonsenseConfig:
    """Shared neural modes with non-position-locked patterns.

    Parameters
    ----------
    n_modes : int
        Number of shared neural modes.
    amplitude : float
        Scaling factor for the nonsense signal.
    variation : float
        Per-repeat variation in position patterns relative to the base pattern.
        0 = same position pattern every repeat (fully reproducible nonsense);
        inf -> purely IID per repeat.
    """

    n_modes: int = 30
    amplitude: float = 4.0
    variation: float = 0.3


@dataclass
class SimulationConfig:
    """Top-level simulation config.

    Parameters
    ----------
    placefield : PlacefieldConfig
    nonsense : NonsenseConfig
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
    nonsense: NonsenseConfig = field(default_factory=NonsenseConfig)
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


def generate_nonsense(
    cfg: NonsenseConfig, n_neurons: int, n_positions: int, n_repeats: int, seed: int, device: str = "cpu"
) -> tuple[list[torch.Tensor], dict]:
    """Generate nonsense-correlation signal.

    Neural modes U are fixed; position patterns V_r vary per repeat.

    Returns
    -------
    repeats : list of n_repeats tensors, each (n_neurons, n_positions)
    extras : dict with keys ``U`` (n_neurons, n_modes) and ``V_base`` (n_modes, n_positions)
    """

    def rng(s):
        return torch.Generator(device=device).manual_seed(s)

    U = torch.randn(n_neurons, cfg.n_modes, generator=rng(seed), device=device)
    U = U / U.norm(dim=0, keepdim=True)  # unit columns

    V_base = torch.randn(cfg.n_modes, n_positions, generator=rng(seed + 1), device=device)

    repeats = []
    for r in range(n_repeats):
        V_noise = torch.randn(cfg.n_modes, n_positions, generator=rng(seed + 10 + r), device=device)
        V_r = V_base + cfg.variation * V_noise
        repeats.append(cfg.amplitude * (U @ V_r) / cfg.n_modes**0.5)

    return repeats, {"U": U, "V_base": V_base}


def generate_noise(noise_level: float, n_neurons: int, n_positions: int, n_repeats: int, seed: int, device: str = "cpu") -> list[torch.Tensor]:
    def rng(s):
        return torch.Generator(device=device).manual_seed(s)

    return [noise_level * torch.randn(n_neurons, n_positions, generator=rng(seed + r), device=device) for r in range(n_repeats)]


# ---------------------------------------------------------------------------
# Assembly and normalization
# ---------------------------------------------------------------------------


def assemble(spatial: list, nonsense: list, noise: list) -> list[torch.Tensor]:
    return [s + n + e for s, n, e in zip(spatial, nonsense, noise)]


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
        nonsense_repeats : list of (n_neurons, n_positions)  nonsense signal per repeat
        noise_repeats : list of (n_neurons, n_positions)  IID noise per repeat
        U : (n_neurons, n_modes)  fixed neural modes
        V_base : (n_modes, n_positions)  shared position pattern
        spatial_rank : int
    """
    N = cfg.placefield.n_neurons
    P = cfg.placefield.n_positions
    R = cfg.n_repeats

    pf_data = generate_placefields(cfg.placefield, R, cfg.seed, device)
    nonsense_repeats, ns_extras = generate_nonsense(cfg.nonsense, N, P, R, cfg.seed + 1000, device)
    noise_repeats = generate_noise(cfg.noise_level, N, P, R, cfg.seed + 2000, device)

    spatial_repeats = pf_data["repeats"]
    repeats = assemble(spatial_repeats, nonsense_repeats, noise_repeats)
    if cfg.normalize:
        repeats = normalize_by_max(repeats)

    extras = {
        "source": pf_data["source"],
        "spatial_repeats": spatial_repeats,
        "nonsense_repeats": nonsense_repeats,
        "noise_repeats": noise_repeats,
        "spatial_rank": pf_data["spatial_rank"],
        "U": ns_extras["U"],
        "V_base": ns_extras["V_base"],
    }
    return repeats, extras


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def run_analysis(repeats: list[torch.Tensor], normalize: bool, center: bool, smooth_width: float, n_components: int) -> dict:
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

    for ref in range(n_rep):
        c0 = repeats[ref]
        c1 = repeats[(ref + 1) % n_rep]
        c2 = repeats[(ref + 2) % n_rep]

        cov_n.append(CVPCA(num_components=n_comp, center=center, on_stimuli=False).fit(c0).score(c1, c2))
        cov_p.append(CVPCA(num_components=n_comp, center=center, on_stimuli=True).fit(c0).score(c1, c2))

        c0s = gaussian_filter(c0, smooth_width, axis=1)
        smooth_cov_n.append(CVPCA(num_components=n_comp, center=center, on_stimuli=False).fit(c0s).score(c1, c2))
        smooth_cov_p.append(CVPCA(num_components=n_comp, center=center, on_stimuli=True).fit(c0s).score(c1, c2))

    def _mean(lst):
        return torch.stack(lst).mean(dim=0).cpu().numpy()

    return {
        "cov_neuron": _mean(cov_n),
        "cov_position": _mean(cov_p),
        "smooth_cov_neuron": _mean(smooth_cov_n),
        "smooth_cov_position": _mean(smooth_cov_p),
    }


def run_simulation(cfg: SimulationConfig, device: str = "cpu") -> dict:
    N = cfg.placefield.n_neurons
    P = cfg.placefield.n_positions
    R = cfg.n_repeats

    pf_data = generate_placefields(cfg.placefield, R, cfg.seed, device)
    nonsense, _ = generate_nonsense(cfg.nonsense, N, P, R, cfg.seed + 1000, device)
    noise = generate_noise(cfg.noise_level, N, P, R, cfg.seed + 2000, device)

    repeats = assemble(pf_data["repeats"], nonsense, noise)
    result = run_analysis(repeats, cfg.normalize, cfg.center, cfg.smooth_width, cfg.n_components)
    result["spatial_rank"] = pf_data["spatial_rank"]
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

    for ck in cov_keys:
        stacked[f"frac_neg_{ck}"] = np.stack([frac_neg(r[ck]) for r in all_results])

    mean_spatial_rank = int(np.round(np.mean([r["spatial_rank"] for r in all_results])))
    stacked["burn_in"] = mean_spatial_rank + cfg.nonsense.n_modes

    return stacked


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def frac_neg(cov: np.ndarray) -> np.ndarray:
    return np.array([(cov[:k] < 0).mean() for k in range(1, len(cov) + 1)])


def plot_placefields(cfg: SimulationConfig, device: str = "cpu") -> plt.Figure:
    """Heatmaps of train and test placefields (with all noise), sorted by train peak position."""
    N, P, R = cfg.placefield.n_neurons, cfg.placefield.n_positions, cfg.n_repeats

    pf_data = generate_placefields(cfg.placefield, R, cfg.seed, device)
    nonsense, _ = generate_nonsense(cfg.nonsense, N, P, R, cfg.seed + 1000, device)
    noise = generate_noise(cfg.noise_level, N, P, R, cfg.seed + 2000, device)
    repeats = assemble(pf_data["repeats"], nonsense, noise)

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

    burn_in = pf_data["spatial_rank"] + cfg.nonsense.n_modes
    fig.suptitle(f"spatial_rank={pf_data['spatial_rank']}  ns_modes={cfg.nonsense.n_modes}  burn_in={burn_in}", fontsize=10)
    plt.tight_layout()
    return fig


def plot_population(repeats: list[torch.Tensor], extras: dict, cfg: SimulationConfig) -> plt.Figure:
    """Four-panel plot: spatial source, shared nonsense, train repeat, test repeat.

    Neurons are sorted by their peak position in the source place fields so
    spatial structure appears as a diagonal in the left panels.
    """
    source = extras["source"].cpu().numpy()  # (N, P), non-negative
    nonsense_shared = (cfg.nonsense.amplitude / cfg.nonsense.n_modes**0.5 * (extras["U"] @ extras["V_base"])).cpu().numpy()  # (N, P), signed
    train = repeats[0].cpu().numpy()
    test = repeats[1].cpu().numpy()

    sort_idx = np.argsort(np.argmax(source, axis=1))

    panels = [
        (source[sort_idx], "Spatial source\n(place fields)", "viridis", 0, source.max()),
        (nonsense_shared[sort_idx], "Nonsense\n(shared U·V_base)", "RdBu_r", None, None),
        (train[sort_idx], "Train (repeat 1)", "RdBu_r", None, None),
        (test[sort_idx], "Test (repeat 2)", "RdBu_r", None, None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
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
        f"spatial_rank={sr}  ns_modes={cfg.nonsense.n_modes}  "
        f"pf_amp={cfg.placefield.amplitude:.1f}  ns_amp={cfg.nonsense.amplitude:.1f}  "
        f"noise={cfg.noise_level:.1f}  normalize={cfg.normalize}",
        fontsize=10,
    )
    plt.tight_layout()
    return fig


_SERIES = [
    ("cov_neuron", "frac_neg_cov_neuron", "steelblue", "-", 1.8, "neuron"),
    ("cov_position", "frac_neg_cov_position", "tomato", "-", 1.8, "position"),
    ("smooth_cov_neuron", "frac_neg_smooth_cov_neuron", "steelblue", "--", 1.2, "smooth neuron"),
    ("smooth_cov_position", "frac_neg_smooth_cov_position", "tomato", "--", 1.2, "smooth position"),
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

        # --- Log covariance (mean; negatives naturally absent from log scale) ---
        for ck, _fk, color, ls, lw, name in _SERIES:
            if ck not in r:
                continue
            mean_cov = r[ck].mean(axis=0)
            pos = np.where(mean_cov > 0, mean_cov, np.nan)
            ax_c.plot(dims, pos, color=color, ls=ls, lw=lw, label=name)
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
            mean_fn = r[fk].mean(axis=0)
            std_fn = r[fk].std(axis=0)
            ax_f.plot(dims, mean_fn, color=color, ls=ls, lw=lw, label=name)
            ax_f.fill_between(dims, mean_fn - std_fn, mean_fn + std_fn, color=color, alpha=0.15)
        ax_f.set_ylim(-0.05, 1.05)
        ax_f.set_xlabel("Dimension")
        if col == 0:
            ax_f.set_ylabel("Fraction < 0 (cumulative)")
        if col == 0:
            ax_f.legend(fontsize=7)

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
    "ns_amplitude",
    "ns_variation",
    "noise_level",
    "smooth_width",
]
_PARAM_LABELS = ["pf amp", "pf ls", "threshold %", "repeat α", "ns amp", "ns var", "noise", "smooth"]


def _compute_asymmetry(stacked: dict) -> float:
    """Single objective: mean asymmetry after burn_in dims, averaged over raw + smooth variants.

    asymmetry = frac_neg_position[burn_in:] - frac_neg_neuron[burn_in:]

    Positive when position-space hits the noise floor faster than neuron-space.
    """
    n_comp = stacked["frac_neg_cov_neuron"].shape[1]
    burn_in = min(int(stacked["burn_in"]), n_comp - 1)
    asym_raw = stacked["frac_neg_cov_position"][:, burn_in:].mean() - stacked["frac_neg_cov_neuron"][:, burn_in:].mean()
    asym_smooth = stacked["frac_neg_smooth_cov_position"][:, burn_in:].mean() - stacked["frac_neg_smooth_cov_neuron"][:, burn_in:].mean()
    return float((asym_raw + asym_smooth) / 2)


def _suggest_config(trial: optuna.Trial, base: SimulationConfig) -> SimulationConfig:
    pf = replace(
        base.placefield,
        amplitude=trial.suggest_float("pf_amplitude", 2.0, 30.0, log=True),
        lengthscale=trial.suggest_float("pf_lengthscale", 2.0, 20.0, log=True),
        threshold_pct=trial.suggest_float("pf_threshold_pct", 20.0, 85.0),
        repeat_noise_alpha=trial.suggest_float("pf_repeat_noise_alpha", 0.0, 0.9),
    )
    ns = replace(
        base.nonsense,
        amplitude=trial.suggest_float("ns_amplitude", 0.1, 15.0, log=True),
        variation=trial.suggest_float("ns_variation", 0.0, 2.0),
    )
    return replace(
        base,
        placefield=pf,
        nonsense=ns,
        noise_level=trial.suggest_float("noise_level", 0.1, 5.0, log=True),
        smooth_width=trial.suggest_float("smooth_width", 0.5, 20.0, log=True),
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
            f"pf_alpha={t.params['pf_repeat_noise_alpha']:.2f}  ns_amp={t.params['ns_amplitude']:.2f}  ns_var={t.params['ns_variation']:.2f}<br>"
            f"noise={t.params['noise_level']:.2f}  smooth={t.params['smooth_width']:.2f}"
        )
        for t in trials
    ]

    # --- History figure ---
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

    # --- Parallel coordinates figure ---
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
            f"thr={p['pf_threshold_pct']:.1f}  alpha={p['pf_repeat_noise_alpha']:.2f}\n"
            f"    ns_amp={p['ns_amplitude']:.2f}  ns_var={p['ns_variation']:.2f}  "
            f"noise={p['noise_level']:.2f}  smooth={p['smooth_width']:.2f}"
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
    nonsense=NonsenseConfig(n_modes=30, amplitude=4.0, variation=0.3),
    noise_level=1.0,
    n_repeats=3,
    normalize=True,
    center=True,
    smooth_width=3.0,
    n_components=80,
    n_simulations=10,
    seed=42,
)

if __name__ == "__main__":
    # Base result: fraction-negative curves + placefield heatmaps
    print(f"Base simulation ({BASE.n_simulations} runs)...")
    base_result = run_simulations(BASE)
    print(f"  burn_in={base_result['burn_in']}  (spatial_rank + ns_modes)")
    for key, label in [("frac_neg_cov_neuron", "neuron  "), ("frac_neg_cov_position", "position")]:
        fn = base_result[key].mean(axis=0)
        print(f"  {label} frac<0 @dim40/dim79: {fn[39]:.2f} / {fn[78]:.2f}")
    plot_results({"base": base_result}, suptitle="Neuron vs position cvPCA (base config)")
    plt.show()

    print("\nPlacefield heatmaps...")
    plot_placefields(BASE)
    plt.show()

    print("\nRunning Optuna study...")
    study = run_optuna_study(BASE, n_trials=200, n_sims_per_trial=5)

    print_best_configs(study)

    figs = plot_optuna_results(study)
    figs["history"].show()
    figs["parallel"].show()
