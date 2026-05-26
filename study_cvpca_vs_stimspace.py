"""
Optuna study: maximize ratio of rCVPCA-fixed alpha vs stimspace alpha.

Generative model from my_precious_cvpca_simulation.py (imported, not modified).
Two spectrum methods:
  - rCVPCA fixed: CVPCA fit on smooth(r0), scored on (r2, r3)
  - stimspace pf-pf: cv_variance_squared_placefield_placefield analogue

Power-law alpha fitted on dims 8-20. Objective: alpha_cvpca / alpha_stimspace.

Fold assignment (n_repeats must be >= 4):
  r0 = train  (rCVPCA fit + stimspace eigenvector fit)
  r1 = kernel (cov_pf_test for stimspace kernel)
  r2 = cv1    (scoring for both methods)
  r3 = cv2    (scoring for both methods)
"""

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import optuna
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from dimilibi.cvpca import CVPCA
from dimilibi.helpers import fit_powerlaw_decay, gaussian_filter
from my_precious_cvpca_simulation import (
    PlacefieldConfig,
    SimulationConfig,
    assemble,
    generate_noise,
    generate_placefields,
    normalize_by_max,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_DIM = 8
END_DIM = 20

_PARAM_NAMES = [
    "pf_amplitude",
    "pf_lengthscale",
    "pf_threshold_pct",
    "pf_repeat_noise_alpha",
    "pf_repeat_noise_lengthscale",
    "peak_exponent",
    "peak_sigma_scale",
    "noise_level",
    "smooth_width",
    "normalize",
    "center",
]
_PARAM_LABELS = ["pf amp", "pf ls", "thr%", "rep a", "rep ls", "peak p", "s scale", "noise", "smooth", "norm", "center"]


# ---------------------------------------------------------------------------
# stimspace math
# ---------------------------------------------------------------------------


def make_G(pf: torch.Tensor) -> torch.Tensor:
    """Center per-neuron and scale so G @ G.T = cov(pf). Input (N, P) -> (N, P)."""
    P = pf.shape[1]
    return (pf - pf.mean(dim=1, keepdim=True)) / (P - 1) ** 0.5


def stimspace_spectrum(
    r0: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    r3: torch.Tensor,
    n_components: int,
) -> np.ndarray:
    """cv_variance_squared_placefield_placefield analogue, then sqrt.

    Returns array of shape (n_components,) - sqrt of per-component shared variance.
    """
    G_train = make_G(r0)
    cov_pf_test = torch.cov(r1)  # (N, N)
    pf_pf_kernel = G_train.T @ cov_pf_test @ G_train  # (P, P)

    n_comp = min(n_components, pf_pf_kernel.shape[0])
    eigvals, eigvecs = torch.linalg.eigh(pf_pf_kernel)
    u = eigvecs.flip(1)[:, :n_comp]  # (P, n_comp), descending

    G_cv1 = make_G(r2)
    G_cv2 = make_G(r3)
    K = G_cv1.T @ cov_pf_test @ G_cv2  # (P, P)
    cv_var = torch.sum(u * (K @ u), dim=0)  # (n_comp,)

    return torch.sqrt(cv_var.clamp(min=0)).cpu().numpy()


# ---------------------------------------------------------------------------
# Single-seed analysis
# ---------------------------------------------------------------------------


def run_pair(cfg: SimulationConfig, seed: int, device: str = "cpu") -> dict:
    """Single-seed rCVPCA-fixed vs stimspace analysis.

    Returns dict with alpha_cvpca, alpha_stim, ratio (0.0 on failure).
    """
    N = cfg.placefield.n_neurons
    P = cfg.placefield.n_positions
    R = cfg.n_repeats

    pf_data = generate_placefields(cfg.placefield, R, seed, device)
    noise = generate_noise(cfg.noise_level, N, P, R, seed + 2000, device)
    repeats = assemble(pf_data["repeats"], noise)

    if cfg.normalize:
        repeats = normalize_by_max(repeats)

    r0, r1, r2, r3 = repeats[0], repeats[1], repeats[2], repeats[3]
    n_comp = min(cfg.n_components, N - 1, P - 1)

    # rCVPCA fixed: fit on smooth(r0), score on (r2, r3)
    r0_smooth = gaussian_filter(r0, cfg.smooth_width, axis=1)
    cvpca = CVPCA(num_components=n_comp, center=cfg.center).fit(r0_smooth)
    reg_cov = cvpca.score(r2, r3).cpu().numpy()

    # stimspace pf-pf
    stim_spec = stimspace_spectrum(r0_smooth, r1, r2, r3, n_comp)

    # power law alphas on dims 8-20
    end = min(END_DIM, n_comp)
    alpha_cv, _ = fit_powerlaw_decay(reg_cov, start_idx=START_DIM, end_idx=end, ignore_nans=True, verbose=False)
    alpha_st, _ = fit_powerlaw_decay(stim_spec, start_idx=START_DIM, end_idx=end, ignore_nans=True, verbose=False)

    ratio = 0.0
    if np.isfinite(alpha_cv) and np.isfinite(alpha_st) and alpha_st > 0 and alpha_cv > 0:
        ratio = float(alpha_cv / alpha_st)

    return {"alpha_cvpca": float(alpha_cv), "alpha_stim": float(alpha_st), "ratio": ratio}


def run_pair_avg(cfg: SimulationConfig, n_sims: int, device: str = "cpu") -> dict:
    """Average run_pair over n_sims seeds. Returns mean alpha_cvpca, alpha_stim, ratio."""
    results = [run_pair(cfg, cfg.seed + i, device) for i in tqdm(range(n_sims), desc="running simulations within each trial", leave=False)]
    return {
        "alpha_cvpca": float(np.mean([r["alpha_cvpca"] for r in results])),
        "alpha_stim": float(np.mean([r["alpha_stim"] for r in results])),
        "ratio": float(np.mean([r["ratio"] for r in results])),
    }


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------


def suggest_config(trial: optuna.Trial, base: SimulationConfig) -> SimulationConfig:
    pf = replace(
        base.placefield,
        amplitude=trial.suggest_float("pf_amplitude", 2.0, 30.0, log=True),
        lengthscale=trial.suggest_float("pf_lengthscale", 2.0, 20.0, log=True),
        threshold_pct=trial.suggest_float("pf_threshold_pct", 20.0, 85.0),
        repeat_noise_alpha=trial.suggest_float("pf_repeat_noise_alpha", 0.0, 0.9),
        repeat_noise_lengthscale=trial.suggest_float("pf_repeat_noise_lengthscale", 1.0, 15.0, log=True),
        peak_exponent=trial.suggest_float("peak_exponent", 0.1, 4.0),
        peak_sigma_scale=trial.suggest_float("peak_sigma_scale", 0.1, 3.0),
    )
    return replace(
        base,
        placefield=pf,
        noise_level=trial.suggest_float("noise_level", 0.1, 5.0, log=True),
        smooth_width=trial.suggest_float("smooth_width", 0.5, 20.0, log=True),
        normalize=trial.suggest_categorical("normalize", [True, False]),
        center=trial.suggest_categorical("center", [True, False]),
    )


def config_from_params(params: dict, base: SimulationConfig) -> SimulationConfig:
    pf = replace(
        base.placefield,
        amplitude=params["pf_amplitude"],
        lengthscale=params["pf_lengthscale"],
        threshold_pct=params["pf_threshold_pct"],
        repeat_noise_alpha=params["pf_repeat_noise_alpha"],
        repeat_noise_lengthscale=params["pf_repeat_noise_lengthscale"],
        peak_exponent=params["peak_exponent"],
        peak_sigma_scale=params["peak_sigma_scale"],
    )
    return replace(
        base,
        placefield=pf,
        noise_level=params["noise_level"],
        smooth_width=params["smooth_width"],
        normalize=params["normalize"],
        center=params["center"],
    )


def run_study(
    base: SimulationConfig,
    n_trials: int = 200,
    n_sims_per_trial: int = 5,
    device: str = "cpu",
    seed: int = 0,
) -> optuna.Study:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial, base)
        result = run_pair_avg(cfg, n_sims_per_trial, device)
        trial.set_user_attr("alpha_cvpca", result["alpha_cvpca"])
        trial.set_user_attr("alpha_stim", result["alpha_stim"])
        return result["ratio"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _completed(study: optuna.Study) -> tuple[list, np.ndarray]:
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return trials, np.array([t.value for t in trials])


def print_best_configs(study: optuna.Study, top_n: int = 5) -> None:
    trials, _ = _completed(study)
    top = sorted(trials, key=lambda t: t.value, reverse=True)[:top_n]

    print(f"\n=== Top {top_n} configs by alpha_cvpca/alpha_stim ratio ===")
    for rank, t in enumerate(top, 1):
        p = t.params
        a_cv = t.user_attrs.get("alpha_cvpca", "?")
        a_st = t.user_attrs.get("alpha_stim", "?")
        print(
            f"#{rank}  ratio={t.value:.4f}  trial={t.number}"
            f"  alpha_cvpca={a_cv:.3f}  alpha_stim={a_st:.3f}\n"
            f"    pf_amp={p['pf_amplitude']:.2f}  ls={p['pf_lengthscale']:.2f}"
            f"  thr={p['pf_threshold_pct']:.1f}  rep_a={p['pf_repeat_noise_alpha']:.2f}"
            f"  rep_ls={p['pf_repeat_noise_lengthscale']:.2f}\n"
            f"    peak_p={p['peak_exponent']:.2f}  s_scale={p['peak_sigma_scale']:.2f}"
            f"  noise={p['noise_level']:.2f}  smooth={p['smooth_width']:.2f}"
            f"  norm={p['normalize']}  center={p['center']}"
        )


def plot_study(study: optuna.Study) -> dict[str, go.Figure]:
    trials, values = _completed(study)
    trial_nums = np.array([t.number for t in trials])
    running_best = np.maximum.accumulate(values)

    hover = [
        (
            f"trial={t.number}  ratio={t.value:.4f}<br>"
            f"a_cv={t.user_attrs.get('alpha_cvpca', '?'):.3f}  a_st={t.user_attrs.get('alpha_stim', '?'):.3f}<br>"
            f"pf_amp={t.params['pf_amplitude']:.2f}  ls={t.params['pf_lengthscale']:.2f}"
            f"  thr={t.params['pf_threshold_pct']:.1f}<br>"
            f"rep_a={t.params['pf_repeat_noise_alpha']:.2f}"
            f"  rep_ls={t.params['pf_repeat_noise_lengthscale']:.2f}<br>"
            f"peak_p={t.params['peak_exponent']:.2f}  s_scale={t.params['peak_sigma_scale']:.2f}<br>"
            f"noise={t.params['noise_level']:.2f}  smooth={t.params['smooth_width']:.2f}"
            f"  norm={t.params['normalize']}  center={t.params['center']}"
        )
        for t in trials
    ]

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=trial_nums,
            y=values,
            mode="markers",
            marker=dict(color=values, colorscale="Viridis", size=6, opacity=0.7, showscale=True, colorbar=dict(title="ratio")),
            text=hover,
            hoverinfo="text",
            name="trials",
        )
    )
    fig_hist.add_trace(go.Scatter(x=trial_nums, y=running_best, mode="lines", line=dict(color="red", width=2), name="running best"))
    fig_hist.update_layout(
        title="Optimization history - alpha_cvpca / alpha_stim",
        xaxis_title="Trial",
        yaxis_title="Ratio",
        height=400,
        width=800,
    )

    all_params = {name: [t.params[name] for t in trials] for name in _PARAM_NAMES}
    dimensions = [dict(label=lbl, values=all_params[name]) for name, lbl in zip(_PARAM_NAMES, _PARAM_LABELS)]
    dimensions.append(dict(label="ratio", values=values.tolist()))

    fig_par = go.Figure(
        go.Parcoords(
            line=dict(color=values, colorscale="Viridis", showscale=True, colorbar=dict(title="ratio")),
            dimensions=dimensions,
        )
    )
    fig_par.update_layout(title="Parallel coordinates - colored by ratio", height=500, width=1200)

    return {"history": fig_hist, "parallel": fig_par}


# ---------------------------------------------------------------------------
# Post-study spectrum plot
# ---------------------------------------------------------------------------


def _collect_spectra(cfg: SimulationConfig, n_sims: int, device: str) -> dict:
    """Run n_sims seeds, collect per-seed reg_cov and stim_spec arrays."""
    reg_covs, stim_specs = [], []
    N, P, R = cfg.placefield.n_neurons, cfg.placefield.n_positions, cfg.n_repeats
    n_comp = min(cfg.n_components, N - 1, P - 1)

    for i in tqdm(range(n_sims), desc="collecting spectra", leave=False):
        seed = cfg.seed + i
        pf_data = generate_placefields(cfg.placefield, R, seed, device)
        noise = generate_noise(cfg.noise_level, N, P, R, seed + 2000, device)
        repeats = assemble(pf_data["repeats"], noise)
        if cfg.normalize:
            repeats = normalize_by_max(repeats)

        r0, r1, r2, r3 = repeats[0], repeats[1], repeats[2], repeats[3]

        r0_smooth = gaussian_filter(r0, cfg.smooth_width, axis=1)
        cvpca = CVPCA(num_components=n_comp, center=cfg.center).fit(r0_smooth)
        reg_cov = cvpca.score(r2, r3).cpu().numpy()

        stim_spec = stimspace_spectrum(r0_smooth, r1, r2, r3, n_comp)

        reg_covs.append(reg_cov)
        stim_specs.append(stim_spec)

    return {
        "reg_cov": np.stack(reg_covs),  # (n_sims, n_comp)
        "stim_spec": np.stack(stim_specs),  # (n_sims, n_comp)
        "n_comp": n_comp,
    }


def plot_best_spectra(cfg: SimulationConfig, n_sims: int = 10, device: str = "cpu") -> plt.Figure:
    """Log-log plot of rCVPCA-fixed and stimspace spectra for cfg, with power-law fit lines."""
    data = _collect_spectra(cfg, n_sims, device)
    n_comp = data["n_comp"]
    dims = np.arange(1, n_comp + 1)

    mean_rc = data["reg_cov"].mean(axis=0)
    mean_ss = data["stim_spec"].mean(axis=0)
    std_rc = data["reg_cov"].std(axis=0)
    std_ss = data["stim_spec"].std(axis=0)

    end = min(END_DIM, n_comp)
    alpha_cv, amp_cv = fit_powerlaw_decay(mean_rc, start_idx=START_DIM, end_idx=end, ignore_nans=True, verbose=False)
    alpha_st, amp_st = fit_powerlaw_decay(mean_ss, start_idx=START_DIM, end_idx=end, ignore_nans=True, verbose=False)

    fit_dims = np.arange(START_DIM + 1, end + 1, dtype=float)
    fit_rc = amp_cv * fit_dims ** (-alpha_cv) if np.isfinite(alpha_cv) else None
    fit_ss = amp_st * fit_dims ** (-alpha_st) if np.isfinite(alpha_st) else None

    fig, ax = plt.subplots(figsize=(7, 5))

    # rCVPCA fixed
    pos_rc = np.where(mean_rc > 0, mean_rc, np.nan)
    ax.plot(dims, pos_rc, color="steelblue", lw=2, label=f"rCVPCA fixed (a={alpha_cv:.2f})")
    ax.fill_between(dims, np.where(mean_rc - std_rc > 0, mean_rc - std_rc, np.nan), mean_rc + std_rc, color="steelblue", alpha=0.2)
    if fit_rc is not None:
        ax.plot(fit_dims, fit_rc, color="steelblue", lw=1.2, ls="--")

    # stimspace pf-pf
    pos_ss = np.where(mean_ss > 0, mean_ss, np.nan)
    ax.plot(dims, pos_ss, color="tomato", lw=2, label=f"stimspace pf-pf (a={alpha_st:.2f})")
    ax.fill_between(dims, np.where(mean_ss - std_ss > 0, mean_ss - std_ss, np.nan), mean_ss + std_ss, color="tomato", alpha=0.2)
    if fit_ss is not None:
        ax.plot(fit_dims, fit_ss, color="tomato", lw=1.2, ls="--")

    # mark fit range
    ax.axvspan(START_DIM + 1, end, alpha=0.06, color="gray", label=f"fit range {START_DIM+1}-{end}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Spectrum")
    ratio = alpha_cv / alpha_st if (np.isfinite(alpha_cv) and np.isfinite(alpha_st) and alpha_st > 0) else float("nan")
    ax.set_title(
        f"Best config spectra  (ratio={ratio:.3f}, n_sims={n_sims})\n"
        f"smooth={cfg.smooth_width:.2f}  noise={cfg.noise_level:.2f}"
        f"  peak_p={cfg.placefield.peak_exponent}  norm={cfg.normalize}"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
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
    n_repeats=4,
    normalize=True,
    center=True,
    smooth_width=3.0,
    n_components=80,
    n_simulations=5,
    seed=42,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rCVPCA vs stimspace Optuna study.")
    parser.add_argument("--n-neurons", type=int, default=BASE.placefield.n_neurons)
    parser.add_argument("--n-positions", type=int, default=BASE.placefield.n_positions)
    parser.add_argument("--n-components", type=int, default=BASE.n_components)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--n-sims-per-trial", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    base = replace(
        BASE,
        placefield=replace(BASE.placefield, n_neurons=args.n_neurons, n_positions=args.n_positions),
        n_components=args.n_components,
        n_simulations=args.n_sims_per_trial,
        seed=args.seed,
    )

    if base.n_repeats < 4:
        raise ValueError(f"n_repeats must be >= 4 for 4-fold structure, got {base.n_repeats}")

    print(f"Running Optuna study: {args.n_trials} trials, {args.n_sims_per_trial} sims/trial, device={args.device}")
    study = run_study(base, n_trials=args.n_trials, n_sims_per_trial=args.n_sims_per_trial, device=args.device, seed=args.seed)

    print_best_configs(study, top_n=args.top_n)

    figs = plot_study(study)
    figs["history"].show()
    figs["parallel"].show()

    cfg_best = config_from_params(study.best_trial.params, base)
    print("\nPlotting best-config spectra...")
    plot_best_spectra(cfg_best, n_sims=args.n_sims_per_trial, device=args.device)
    plt.show()


if __name__ == "__main__":
    main()
