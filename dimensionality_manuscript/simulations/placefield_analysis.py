"""
Place field analysis: CVPCA workflows, stimspace comparison, Optuna, and plotting.

Two main workflows
------------------
run_neuron_position        -- 3-fold CVPCA in neuron-space vs position-space
run_cvpca_stimspace        -- rCVPCA-fixed vs stimspace alpha comparison (single seed)
run_cvpca_stimspace_avg    -- average run_cvpca_stimspace over multiple seeds
run_cvpca_stimspace_stack  -- stack full results over multiple seeds

Optuna
------
suggest_config, config_from_params, run_study

Plotting
--------
plot_placefields, plot_example_placefields, plot_neuron_position,
plot_spectra, plot_frac_neg, plot_component_alignment, plot_study,
print_best_configs
"""

from __future__ import annotations
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from tqdm import tqdm

from dimilibi.cvpca import CVPCA
from dimilibi.helpers import fit_powerlaw_decay, gaussian_filter

from .placefield_generator import (
    PlacefieldConfig,
    SimConfig,
    TilburyConfig,
    _n_neurons_positions,
    assemble,
    generate_noise,
    generate_repeats,
    normalize_by_max,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_DIM: int = 8
END_DIM: int = 20

# Series metadata for plot_neuron_position
_NP_SERIES = [
    ("cov_neuron", "frac_neg_cov_neuron", "steelblue", "-", 1.8, "neuron cvpca"),
    ("cov_position", "frac_neg_cov_position", "tomato", "-", 1.8, "position cvpca"),
    ("smooth_cov_neuron", "frac_neg_smooth_cov_neuron", "steelblue", "--", 1.2, "smooth neuron cvpca"),
    ("smooth_cov_position", "frac_neg_smooth_cov_position", "tomato", "--", 1.2, "smooth position cvpca"),
]


# ---------------------------------------------------------------------------
# Private: stimspace math
# ---------------------------------------------------------------------------


def _make_G(pf: torch.Tensor) -> torch.Tensor:
    """Center per-neuron and scale so G @ G.T = cov(pf). (N, P) → (N, P)."""
    P = pf.shape[1]
    return (pf - pf.mean(dim=1, keepdim=True)) / (P - 1) ** 0.5


def _stimspace_fit_and_score(
    r0: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    r3: torch.Tensor,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit stimspace kernel on (r0, r1) and score on (r2, r3).

    Returns
    -------
    cv_var_raw : (n_comp,) — signed cv variance before sqrt
    u : (P, n_comp) — position-space eigenvectors (descending)
    """
    G_train = _make_G(r0)
    cov_pf_test = torch.cov(r1)  # (N, N)
    pf_pf_kernel = G_train.T @ G_train  # (P, P)

    n_comp = min(n_components, pf_pf_kernel.shape[0])
    _, eigvecs = torch.linalg.eigh(pf_pf_kernel)
    u = eigvecs.flip(1)[:, :n_comp]  # (P, n_comp)

    G_cv1 = _make_G(r2)
    G_cv2 = _make_G(r3)
    K = G_cv1.T @ cov_pf_test @ G_cv2  # (P, P)
    cv_var = torch.sum(u * (K @ u), dim=0)  # (n_comp,)

    return cv_var.cpu().numpy(), u.cpu().numpy()


def _stimspace_spectrum(r0: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, r3: torch.Tensor, n_components: int) -> np.ndarray:
    """sqrt(max(cv_var_raw, 0)) for each component. Shape (n_comp,)."""
    cv_var_raw, _ = _stimspace_fit_and_score(r0, r1, r2, r3, n_components)
    return np.sqrt(np.maximum(cv_var_raw, 0))


def _fit_powerlaw_range(spectrum: np.ndarray, n_comp: int) -> tuple[float, float]:
    """Fit power-law on dims START_DIM..min(END_DIM, n_comp). Returns (alpha, amplitude)."""
    end = min(END_DIM, n_comp)
    return fit_powerlaw_decay(spectrum, start_idx=START_DIM, end_idx=end, ignore_nans=True, verbose=False)


# ---------------------------------------------------------------------------
# Private: neuron-position CVPCA fold runner
# ---------------------------------------------------------------------------


def _run_cvpca_fold(
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    n_comp: int,
    center: bool,
    smooth_width: float | None,
) -> dict:
    """Run one 3-fold CVPCA pass (raw + smoothed, neuron + position space)."""
    c0s = gaussian_filter(c0, smooth_width, axis=1) if smooth_width is not None else c0

    cvpca_n = CVPCA(num_components=n_comp, center=center, on_stimuli=False).fit(c0)
    cvpca_p = CVPCA(num_components=n_comp, center=center, on_stimuli=True).fit(c0)
    cvpca_sn = CVPCA(num_components=n_comp, center=center, on_stimuli=False).fit(c0s)
    cvpca_sp = CVPCA(num_components=n_comp, center=center, on_stimuli=True).fit(c0s)

    return {
        "cov_neuron": cvpca_n.score(c1, c2).cpu().numpy(),
        "cov_position": cvpca_p.score(c1, c2).cpu().numpy(),
        "smooth_cov_neuron": cvpca_sn.score(c1, c2).cpu().numpy(),
        "smooth_cov_position": cvpca_sp.score(c1, c2).cpu().numpy(),
        "components_n": cvpca_n.pca.get_components().cpu().numpy(),  # (N, n_comp)
        "components_p": cvpca_p.pca.get_components().cpu().numpy(),  # (P, n_comp)
        "smooth_components_n": cvpca_sn.pca.get_components().cpu().numpy(),
        "smooth_components_p": cvpca_sp.pca.get_components().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Workflow A: neuron vs position CVPCA
# ---------------------------------------------------------------------------


def run_neuron_position(cfg: SimConfig, n_sims: int, device: str = "cpu") -> dict:
    """Run neuron-space vs position-space CVPCA over n_sims independent seeds.

    Each seed: generate repeats, run 3-fold CVPCA loop (cycling train/cv1/cv2).
    Requires cfg.n_repeats >= 3.

    Parameters
    ----------
    cfg : SimConfig
    n_sims : int
        Number of independent seeds to run (seeds cfg.seed .. cfg.seed + n_sims - 1).
    device : str

    Returns
    -------
    dict with arrays stacked over n_sims seeds:
        cov_neuron, cov_position, smooth_cov_neuron, smooth_cov_position : (K, C)
        frac_neg_cov_neuron, frac_neg_cov_position,
            frac_neg_smooth_cov_neuron, frac_neg_smooth_cov_position : (C,)
        eig_neuron, eig_position, svals : (K, C)
        components_n, components_p,
            smooth_components_n, smooth_components_p : (K, n_repeats, C, space_dim)
        source_components_n : (K, N, C)
        source_components_p : (K, P, C)
        burn_in : int  (mean spatial_rank across sims)
    """
    N, P = _n_neurons_positions(cfg)
    cov_keys = ["cov_neuron", "cov_position", "smooth_cov_neuron", "smooth_cov_position"]
    comp_keys = ["components_n", "components_p", "smooth_components_n", "smooth_components_p"]

    all_covs = {k: [] for k in cov_keys}
    all_comps = {k: [] for k in comp_keys}
    eig_neuron_list, eig_position_list, svals_list = [], [], []
    src_comps_n_list, src_comps_p_list = [], []
    spatial_ranks = []

    for i in tqdm(range(n_sims), desc="neuron-position sims", leave=False):
        seed = cfg.seed + i
        pf_data = generate_repeats(cfg, seed, device)
        repeats = pf_data["repeats"]
        source = pf_data["source"]
        spatial_ranks.append(pf_data["spatial_rank"])

        n_rep = len(repeats)
        n_comp = min(cfg.n_components, N - 1, P - 1)

        fold_covs = {k: [] for k in cov_keys}
        fold_comps = {k: [] for k in comp_keys}

        for ref in range(n_rep):
            fold = _run_cvpca_fold(
                repeats[ref],
                repeats[(ref + 1) % n_rep],
                repeats[(ref + 2) % n_rep],
                n_comp,
                cfg.center,
                cfg.smooth_width,
            )
            for k in cov_keys:
                fold_covs[k].append(fold[k])
            for k in comp_keys:
                fold_comps[k].append(fold[k])

        for k in cov_keys:
            all_covs[k].append(np.stack(fold_covs[k]).mean(axis=0))  # mean over folds
        for k in comp_keys:
            all_comps[k].append(np.stack(fold_comps[k]))  # (n_rep, space_dim, n_comp)

        # Source eigenspectra
        eig_n = np.sort(torch.linalg.eigvalsh(torch.cov(source)).cpu().numpy())[::-1][:n_comp]
        eig_p = np.sort(torch.linalg.eigvalsh(torch.cov(source.T)).cpu().numpy())[::-1][:n_comp]
        sv = torch.linalg.svdvals(source).cpu().numpy()[:n_comp]
        eig_neuron_list.append(eig_n)
        eig_position_list.append(eig_p)
        svals_list.append(sv)

        u_src, _, v_src = torch.linalg.svd(source, full_matrices=False)
        src_comps_n_list.append(u_src[:, :n_comp].cpu().numpy())  # (N, n_comp)
        src_comps_p_list.append(v_src[:n_comp, :].T.cpu().numpy())  # (P, n_comp)

    stacked = {}
    for k in cov_keys:
        arr = np.stack(all_covs[k])  # (K, C)
        stacked[k] = arr
        stacked[f"frac_neg_{k}"] = np.mean(arr < 0, axis=0)
    for k in comp_keys:
        stacked[k] = np.stack(
            all_comps[k]
        )  # (K, n_rep, C, space_dim) — note: each fold stores (space_dim, n_comp), so shape is (K, n_rep, space_dim, n_comp)
    stacked["eig_neuron"] = np.stack(eig_neuron_list)
    stacked["eig_position"] = np.stack(eig_position_list)
    stacked["svals"] = np.stack(svals_list)
    stacked["source_components_n"] = np.stack(src_comps_n_list)
    stacked["source_components_p"] = np.stack(src_comps_p_list)
    stacked["burn_in"] = int(np.round(np.mean(spatial_ranks)))

    return stacked


# ---------------------------------------------------------------------------
# Workflow B: rCVPCA-fixed vs stimspace
# ---------------------------------------------------------------------------


def run_cvpca_stimspace(cfg: SimConfig, seed: int, device: str = "cpu", full: bool = False) -> dict:
    """Single-seed rCVPCA-fixed vs stimspace analysis.

    Uses a 4-fold assignment (cfg.n_repeats must be >= 4):
        r0 = train   (fit CVPCA + stimspace eigenvectors; smoothing applied here)
        r1 = kernel  (stimspace cov_pf_test)
        r2 = cv1     (scoring for both methods)
        r3 = cv2     (scoring for both methods)

    Parameters
    ----------
    cfg : SimConfig
    seed : int
    device : str
    full : bool
        False (default): lightweight, returns only scalar metrics.
        True: also returns spectral arrays and component matrices.

    Returns
    -------
    dict with:
        alpha_cvpca, alpha_stim, ratio : float
        (when full=True, additionally)
        reg_cov : (C,) rCVPCA cross-validated covariance
        cv_var_raw : (C,) stimspace cv variance before sqrt
        stim_spec : (C,) sqrt(max(cv_var_raw, 0))
        cvpca_components : (P, C) position-space rCVPCA directions
        stim_u : (P, C) stimspace eigenvectors
        source_comps_n : (N, C) source SVD left vectors
        source_comps_p : (P, C) source SVD right vectors
    """
    N, P = _n_neurons_positions(cfg)
    pf_data = generate_repeats(cfg, seed, device)
    repeats = pf_data["repeats"]
    r0, r1, r2, r3 = repeats[0], repeats[1], repeats[2], repeats[3]
    n_comp = min(cfg.n_components, N - 1, P - 1)

    r0_smooth = gaussian_filter(r0, cfg.smooth_width, axis=1) if cfg.smooth_width is not None else r0

    cvpca = CVPCA(num_components=n_comp, center=cfg.center, on_stimuli=True).fit(r0_smooth)
    reg_cov = cvpca.score(r2, r3).cpu().numpy()

    stim_spec = _stimspace_spectrum(r0_smooth, r1, r2, r3, n_comp)

    alpha_cv, _ = _fit_powerlaw_range(reg_cov, n_comp)
    alpha_st, _ = _fit_powerlaw_range(stim_spec, n_comp)

    ratio = 0.0
    if np.isfinite(alpha_cv) and np.isfinite(alpha_st) and alpha_st > 0 and alpha_cv > 0:
        ratio = float(alpha_cv / alpha_st)

    result = {"alpha_cvpca": float(alpha_cv), "alpha_stim": float(alpha_st), "ratio": ratio}

    if full:
        cv_var_raw, stim_u = _stimspace_fit_and_score(r0_smooth, r1, r2, r3, n_comp)
        cvpca_components = cvpca.pca.get_components().cpu().numpy()  # (P, n_comp)
        source = pf_data["source"]  # (N, P) — pre-noise source for component alignment
        u_src, _, v_src = torch.linalg.svd(source, full_matrices=False)
        result.update(
            {
                "reg_cov": reg_cov,
                "cv_var_raw": cv_var_raw,
                "stim_spec": stim_spec,
                "cvpca_components": cvpca_components,
                "stim_u": stim_u,
                "source_comps_n": u_src[:, :n_comp].cpu().numpy(),
                "source_comps_p": v_src[:n_comp, :].T.cpu().numpy(),
            }
        )

    return result


def run_cvpca_stimspace_avg(cfg: SimConfig, n_sims: int, device: str = "cpu") -> dict:
    """Average run_cvpca_stimspace(full=False) over n_sims seeds.

    Returns
    -------
    dict:
        alpha_cvpca : float (mean across seeds)
        alpha_stim  : float (mean across seeds)
        ratio       : float (median across seeds — robust to outliers)
    """
    results = [run_cvpca_stimspace(cfg, cfg.seed + i, device) for i in tqdm(range(n_sims), desc="averaging sims", leave=False)]
    return {
        "alpha_cvpca": float(np.mean([r["alpha_cvpca"] for r in results])),
        "alpha_stim": float(np.mean([r["alpha_stim"] for r in results])),
        "ratio": float(np.median([r["ratio"] for r in results])),
    }


def run_cvpca_stimspace_stack(cfg: SimConfig, n_sims: int, device: str = "cpu") -> dict:
    """Stack run_cvpca_stimspace(full=True) over n_sims seeds.

    Returns
    -------
    dict with stacked arrays (K = n_sims):
        reg_cov, cv_var_raw, stim_spec : (K, C)
        cvpca_components, stim_u, source_comps_p : (K, P, C)
        source_comps_n : (K, N, C)
        alpha_cvpca, alpha_stim, ratio : (K,)
        frac_neg_reg_cov, frac_neg_cv_var_raw : (C,)
    """
    results = [run_cvpca_stimspace(cfg, cfg.seed + i, device, full=True) for i in tqdm(range(n_sims), desc="full sims", leave=False)]
    array_keys = ["reg_cov", "cv_var_raw", "stim_spec", "cvpca_components", "stim_u", "source_comps_n", "source_comps_p"]
    stacked = {k: np.stack([r[k] for r in results]) for k in array_keys}
    for k in ["alpha_cvpca", "alpha_stim", "ratio"]:
        stacked[k] = np.array([r[k] for r in results])
    stacked["frac_neg_reg_cov"] = np.mean(stacked["reg_cov"] < 0, axis=0)
    stacked["frac_neg_cv_var_raw"] = np.mean(stacked["cv_var_raw"] < 0, axis=0)
    return stacked


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------


def _suggest_rbf(trial: optuna.Trial, base: SimConfig) -> SimConfig:
    gen_base = (
        base.generator
        if isinstance(base.generator, PlacefieldConfig)
        else PlacefieldConfig(
            n_neurons=base.generator.n_neurons,
            n_positions=base.generator.n_positions,
        )
    )
    smooth = trial.suggest_categorical("smooth", [True, False])
    smooth_width = None if not smooth else trial.suggest_float("smooth_width", 0.5, 20.0, log=True)
    return replace(
        base,
        generator=replace(
            gen_base,
            amplitude=trial.suggest_float("pf_amplitude", 2.0, 30.0, log=True),
            lengthscale=trial.suggest_float("pf_lengthscale", 2.0, 20.0, log=True),
            threshold_pct=trial.suggest_float("pf_threshold_pct", 20.0, 85.0),
            repeat_noise_alpha=trial.suggest_float("pf_repeat_noise_alpha", 0.0, 0.9),
            repeat_noise_lengthscale=trial.suggest_float("pf_repeat_noise_lengthscale", 1.0, 15.0, log=True),
            peak_exponent=trial.suggest_float("peak_exponent", 0.1, 2.0),
            peak_sigma_scale=trial.suggest_float("peak_sigma_scale", 0.1, 3.0),
        ),
        noise_level=trial.suggest_float("noise_level", 0.001, 5.0, log=True),
        smooth_width=smooth_width,
        normalize=trial.suggest_categorical("normalize", [True, False]),
        center=trial.suggest_categorical("center", [True, False]),
    )


def _suggest_tilbury(trial: optuna.Trial, base: SimConfig) -> SimConfig:
    gen_base = (
        base.generator
        if isinstance(base.generator, TilburyConfig)
        else TilburyConfig(
            n_neurons=base.generator.n_neurons,
            n_positions=base.generator.n_positions,
        )
    )
    smooth = trial.suggest_categorical("smooth", [True, False])
    smooth_width = None if not smooth else trial.suggest_float("smooth_width", 0.5, 20.0, log=True)
    return replace(
        base,
        generator=replace(
            gen_base,
            amplitude_mean=trial.suggest_float("tb_amplitude_mean", 2.0, 30.0, log=True),
            amplitude_spread=trial.suggest_float("tb_amplitude_spread", 0.0, 1.5),
            amplitude_ratio_beta=trial.suggest_float("tb_amplitude_ratio_beta", 1.0, 50.0, log=True),
            peak_separation_scale=trial.suggest_float("tb_peak_separation_scale", 0.0, 50.0),
            sigma_mean=trial.suggest_float("tb_sigma_mean", 2.0, 20.0, log=True),
            sigma_spread=trial.suggest_float("tb_sigma_spread", 0.0, 1.5),
            sigma_asym_std=trial.suggest_float("tb_sigma_asym_std", 0.0, 1.5),
            exponent_mean=trial.suggest_float("tb_exponent_mean", 0.5, 4.0, log=True),
            exponent_spread=trial.suggest_float("tb_exponent_spread", 0.0, 1.0),
            repeat_noise_alpha=trial.suggest_float("tb_repeat_noise_alpha", 0.0, 0.9),
            repeat_noise_lengthscale=trial.suggest_float("tb_repeat_noise_lengthscale", 1.0, 15.0, log=True),
        ),
        noise_level=trial.suggest_float("noise_level", 0.001, 5.0, log=True),
        smooth_width=smooth_width,
        normalize=trial.suggest_categorical("normalize", [True, False]),
        center=trial.suggest_categorical("center", [True, False]),
    )


def suggest_config(trial: optuna.Trial, base: SimConfig) -> SimConfig:
    """Suggest a new SimConfig for an Optuna trial, dispatching on generator type."""
    if isinstance(base.generator, TilburyConfig):
        return _suggest_tilbury(trial, base)
    return _suggest_rbf(trial, base)


def _config_from_params_rbf(params: dict, base: SimConfig) -> SimConfig:
    gen_base = (
        base.generator
        if isinstance(base.generator, PlacefieldConfig)
        else PlacefieldConfig(
            n_neurons=base.generator.n_neurons,
            n_positions=base.generator.n_positions,
        )
    )
    return replace(
        base,
        generator=replace(
            gen_base,
            amplitude=params["pf_amplitude"],
            lengthscale=params["pf_lengthscale"],
            threshold_pct=params["pf_threshold_pct"],
            repeat_noise_alpha=params["pf_repeat_noise_alpha"],
            repeat_noise_lengthscale=params["pf_repeat_noise_lengthscale"],
            peak_exponent=params["peak_exponent"],
            peak_sigma_scale=params["peak_sigma_scale"],
        ),
        noise_level=params["noise_level"],
        smooth_width=params.get("smooth_width", None),
        normalize=params["normalize"],
        center=params["center"],
    )


def _config_from_params_tilbury(params: dict, base: SimConfig) -> SimConfig:
    gen_base = (
        base.generator
        if isinstance(base.generator, TilburyConfig)
        else TilburyConfig(
            n_neurons=base.generator.n_neurons,
            n_positions=base.generator.n_positions,
        )
    )
    return replace(
        base,
        generator=replace(
            gen_base,
            amplitude_mean=params["tb_amplitude_mean"],
            amplitude_spread=params["tb_amplitude_spread"],
            amplitude_ratio_beta=params["tb_amplitude_ratio_beta"],
            peak_separation_scale=params["tb_peak_separation_scale"],
            sigma_mean=params["tb_sigma_mean"],
            sigma_spread=params["tb_sigma_spread"],
            sigma_asym_std=params["tb_sigma_asym_std"],
            exponent_mean=params["tb_exponent_mean"],
            exponent_spread=params["tb_exponent_spread"],
            repeat_noise_alpha=params["tb_repeat_noise_alpha"],
            repeat_noise_lengthscale=params["tb_repeat_noise_lengthscale"],
        ),
        noise_level=params["noise_level"],
        smooth_width=params.get("smooth_width", None),
        normalize=params["normalize"],
        center=params["center"],
    )


def config_from_params(params: dict, base: SimConfig) -> SimConfig:
    """Reconstruct a SimConfig from an Optuna trial's params dict."""
    if isinstance(base.generator, TilburyConfig):
        return _config_from_params_tilbury(params, base)
    return _config_from_params_rbf(params, base)


def run_study(
    base: SimConfig,
    n_trials: int = 200,
    n_sims_per_trial: int = 5,
    device: str = "cpu",
    seed: int = 0,
) -> optuna.Study:
    """Optuna study maximizing alpha_cvpca / alpha_stim ratio.

    Parameters
    ----------
    base : SimConfig
        Base config from which Optuna suggests variants.
    n_trials : int
        Number of Optuna trials.
    n_sims_per_trial : int
        Seeds averaged per trial (ratio is median, alphas are mean).
    device : str
    seed : int
        TPE sampler seed.
    """
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial: optuna.Trial) -> float:
        try:
            cfg = suggest_config(trial, base)
            result = run_cvpca_stimspace_avg(cfg, n_sims_per_trial, device)
            trial.set_user_attr("alpha_cvpca", result["alpha_cvpca"])
            trial.set_user_attr("alpha_stim", result["alpha_stim"])
            return result["ratio"]
        except Exception as exc:
            trial.set_user_attr("error", repr(exc))
            raise optuna.TrialPruned() from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# ---------------------------------------------------------------------------
# Private: Optuna reporting helpers
# ---------------------------------------------------------------------------


def _completed_trials(study: optuna.Study) -> tuple[list, np.ndarray]:
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return trials, np.array([t.value for t in trials])


def _study_is_tilbury(study: optuna.Study) -> bool:
    trials, _ = _completed_trials(study)
    if not trials:
        return False
    return any(k.startswith("tb_") for k in trials[0].params)


def _format_smooth_width(t: optuna.Trial) -> str:
    sw = t.params.get("smooth_width", None)
    return "None" if sw is None else f"{sw:.2f}"


def _hover_text(t: optuna.Trial, is_tilbury: bool) -> str:
    p = t.params
    a_cv = t.user_attrs.get("alpha_cvpca", float("nan"))
    a_st = t.user_attrs.get("alpha_stim", float("nan"))
    base = f"trial={t.number}  ratio={t.value:.4f}<br>a_cv={a_cv:.3f}  a_st={a_st:.3f}<br>"
    if is_tilbury:
        return (
            base + f"amp_m={p['tb_amplitude_mean']:.2f}  A2_beta={p['tb_amplitude_ratio_beta']:.2f}"
            f"  sep={p['tb_peak_separation_scale']:.2f}<br>"
            f"sig_m={p['tb_sigma_mean']:.2f}  sig_asym={p['tb_sigma_asym_std']:.2f}"
            f"  p_m={p['tb_exponent_mean']:.2f}<br>"
            f"noise={p['noise_level']:.3f}  smooth={_format_smooth_width(t)}"
            f"  norm={p['normalize']}  center={p['center']}"
        )
    return (
        base + f"pf_amp={p['pf_amplitude']:.2f}  ls={p['pf_lengthscale']:.2f}"
        f"  thr={p['pf_threshold_pct']:.1f}<br>"
        f"rep_a={p['pf_repeat_noise_alpha']:.2f}  peak_p={p['peak_exponent']:.2f}<br>"
        f"noise={p['noise_level']:.2f}  smooth={_format_smooth_width(t)}"
        f"  norm={p['normalize']}  center={p['center']}"
    )


# ---------------------------------------------------------------------------
# Private: spectrum collection for plot_spectra
# ---------------------------------------------------------------------------


def _collect_spectra(cfg: SimConfig, n_sims: int, device: str) -> dict:
    """Collect reg_cov (rescaled), stim_spec, and true_spec over n_sims seeds."""
    N, P = _n_neurons_positions(cfg)
    n_comp = min(cfg.n_components, N - 1, P - 1)

    reg_covs, stim_specs, true_specs = [], [], []

    for i in tqdm(range(n_sims), desc="collecting spectra", leave=False):
        seed = cfg.seed + i
        pf_data = generate_repeats(cfg, seed, device)
        repeats = pf_data["repeats"]
        r0, r1, r2, r3 = repeats[0], repeats[1], repeats[2], repeats[3]

        r0_smooth = gaussian_filter(r0, cfg.smooth_width, axis=1) if cfg.smooth_width is not None else r0

        cvpca = CVPCA(num_components=n_comp, center=cfg.center, on_stimuli=True).fit(r0_smooth)
        reg_cov = cvpca.score(r2, r3).cpu().numpy() * (N - 1) / (P - 1)

        stim_spec = _stimspace_spectrum(r0_smooth, r1, r2, r3, n_comp)

        # True spectrum: eigenvalues of covariance of the source. generate_repeats
        # already normalized source by the per-neuron normalizer when cfg.normalize,
        # so this matches the space the data live in.
        source = pf_data["source"]
        true_spec = np.flip(np.linalg.eigh(np.cov(source.cpu().numpy()))[0])[:n_comp]

        reg_covs.append(reg_cov)
        stim_specs.append(stim_spec)
        true_specs.append(true_spec)

    return {
        "reg_cov": np.stack(reg_covs),
        "stim_spec": np.stack(stim_specs),
        "true_spec": np.stack(true_specs),
        "n_comp": n_comp,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_placefields(cfg: SimConfig, num_show: int = 5, device: str = "cpu") -> plt.Figure:
    """Heatmaps of train/test repeats (with all noise), sorted by train peak position."""
    pf_data = generate_repeats(cfg, cfg.seed, device)
    repeats = pf_data["repeats"]
    train = repeats[0].cpu().numpy()
    test = repeats[1].cpu().numpy()
    sort_idx = np.argsort(np.argmax(train, axis=1))

    vmax = max(train.max(), test.max())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    N = train.shape[0]
    idx_show = np.random.choice(N, size=min(num_show, N), replace=False)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in np.linspace(0, 1, len(idx_show))]

    for icol, (data, title) in enumerate(zip([train, test], ["Train (repeat 1)", "Test (repeat 2)"])):
        im = axes[0, icol].imshow(data[sort_idx], aspect="auto", interpolation="none", vmin=0, vmax=vmax)
        axes[0, icol].set_title(title)
        axes[0, icol].set_xlabel("Position bin")
        axes[0, icol].set_ylabel("Neuron (sorted by train peak)")
        plt.colorbar(im, ax=axes[0, icol], label="activity")
        for ineuron, neuron in enumerate(idx_show):
            axes[1, icol].plot(data[neuron, :], color=colors[ineuron])
        axes[1, icol].set_xlabel("Position bin")
        axes[1, icol].set_ylabel("Activity")

    sr = pf_data["spatial_rank"]
    fig.suptitle(f"spatial_rank={sr}", fontsize=10)
    plt.tight_layout()
    return fig


def plot_example_placefields(
    cfg: SimConfig,
    num_examples: int = 20,
    seed: int | None = None,
    normalize: bool = True,
    center_align: bool = True,
    with_noise: bool = False,
    device: str = "cpu",
) -> plt.Figure:
    """Overlaid source place fields, peak-aligned to the position axis center.

    Parameters
    ----------
    cfg : SimConfig
    num_examples : int
    seed : int or None
        Generation seed; defaults to cfg.seed.
    normalize : bool
        Normalize each field by its source peak amplitude.
    center_align : bool
        Roll each field so its argmax sits at P//2.
    with_noise : bool
        Plot first noisy repeat instead of clean source (alignment from source).
    device : str
    """
    if seed is None:
        seed = cfg.seed

    pf_data = generate_repeats(cfg, seed, device)
    source = pf_data["source"].cpu().numpy()
    display = pf_data["repeats"][0].cpu().numpy() if with_noise else source

    active = np.where(source.max(axis=1) > 0)[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(active, size=min(num_examples, len(active)), replace=False)

    src_fields = source[idx]
    fields = display[idx]
    P = source.shape[1]

    if center_align:
        center = P // 2
        shifts = [center - int(np.argmax(f)) for f in src_fields]
        aligned = np.stack([np.roll(f, s) for f, s in zip(fields, shifts)])
        src_peak = np.stack([np.roll(f, s) for f, s in zip(src_fields, shifts)])
    else:
        center = 0
        aligned = fields
        src_peak = src_fields

    if normalize:
        peak = src_peak.max(axis=1, keepdims=True).clip(min=1e-8)
        aligned = aligned / peak

    x = np.arange(P) - center
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(aligned):
        ax.plot(x, f, alpha=1.0, lw=1.0, color=cmap(i % 10))
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Position offset from peak (bins)")
    ax.set_ylabel("Activity" + (" (normalized)" if normalize else ""))

    g = cfg.generator
    if isinstance(g, TilburyConfig):
        detail = f"p_m={g.exponent_mean:.1f}  sig_asym={g.sigma_asym_std:.1f}  A2_beta={g.amplitude_ratio_beta:.1f}"
        path = "tilbury"
    else:
        detail = f"ls={g.lengthscale:.1f}  peak_p={g.peak_exponent}"
        path = "rbf"

    noise_tag = " (repeat)" if with_noise else ""
    ax.set_title(f"Example placefields{noise_tag} — {path} — {detail}")
    plt.tight_layout()
    return fig


def plot_neuron_position(results: dict[str, dict], suptitle: str = "") -> plt.Figure:
    """Log covariance spectra and fraction-negative curves from run_neuron_position.

    Parameters
    ----------
    results : dict[str, dict]
        Keys are condition labels; values are stacked dicts from run_neuron_position().
    suptitle : str
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    for col, (label, r) in enumerate(results.items()):
        n_comp = r["cov_neuron"].shape[1]
        dims = np.arange(1, n_comp + 1)
        ax_c, ax_f = axes[0, col], axes[1, col]

        svals = (r["svals"] ** 2).mean(axis=0)
        ax_c.plot(dims, svals / svals.sum(), color="black", lw=1.0, label=r"True $\lambda$")

        for ck, fk, color, ls, lw, name in _NP_SERIES:
            if ck not in r:
                continue
            mean_cov = r[ck].mean(axis=0)
            pos = np.where(mean_cov > 0, mean_cov, np.nan)
            ax_c.plot(dims, pos / np.nansum(mean_cov), color=color, ls=ls, lw=lw, label=name)

        ax_c.set_yscale("log")
        ax_c.set_title(label, fontsize=10)
        ax_c.set_xlabel("Dimension")
        if col == 0:
            ax_c.set_ylabel("CV covariance (log)")
        ax_c.legend(fontsize=7)

        ax_f.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        for ck, fk, color, ls, lw, name in _NP_SERIES:
            if fk not in r:
                continue
            ax_f.plot(dims, r[fk], color=color, ls=ls, lw=lw, label=name)
        ax_f.set_ylim(-0.05, 1.05)
        ax_f.set_xlabel("Dimension")
        if col == 0:
            ax_f.set_ylabel("Fraction < 0")
        if col == 0:
            ax_f.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_spectra(
    cfg: SimConfig,
    n_sims: int = 10,
    device: str = "cpu",
    start_dim: int = START_DIM,
    end_dim: int = END_DIM,
) -> plt.Figure:
    """Log-log plot of rCVPCA-fixed and stimspace spectra with power-law fit lines.

    Parameters
    ----------
    cfg : SimConfig
        Config for the best/target condition.
    n_sims : int
        Number of seeds to average over.
    start_dim, end_dim : int
        Range used for power-law fitting (inclusive lower, inclusive upper).
    """
    data = _collect_spectra(cfg, n_sims, device)
    n_comp = data["n_comp"]
    dims = np.arange(1, n_comp + 1)

    mean_rc = data["reg_cov"].mean(axis=0)
    mean_ss = data["stim_spec"].mean(axis=0)
    mean_true = data["true_spec"].mean(axis=0)
    std_rc = data["reg_cov"].std(axis=0)
    std_ss = data["stim_spec"].std(axis=0)
    std_true = data["true_spec"].std(axis=0)

    end = min(end_dim, n_comp)
    alpha_cv, amp_cv = fit_powerlaw_decay(mean_rc, start_idx=start_dim, end_idx=end, ignore_nans=True, verbose=False)
    alpha_st, amp_st = fit_powerlaw_decay(mean_ss, start_idx=start_dim, end_idx=end, ignore_nans=True, verbose=False)
    alpha_true, amp_true = fit_powerlaw_decay(mean_true, start_idx=start_dim, end_idx=end, ignore_nans=True, verbose=False)

    fit_dims = np.arange(start_dim + 1, end + 1, dtype=float)

    def _fit_line(alpha, amp):
        return amp * fit_dims ** (-alpha) if np.isfinite(alpha) else None

    fig, ax = plt.subplots(figsize=(7, 5))

    for mean, std, color, label, alpha, amp in [
        (mean_rc, std_rc, "steelblue", f"rCVPCA fixed (a={alpha_cv:.2f})", alpha_cv, amp_cv),
        (mean_ss, std_ss, "tomato", f"stimspace pf-pf (a={alpha_st:.2f})", alpha_st, amp_st),
        (mean_true, std_true, "black", f"true (a={alpha_true:.2f})", alpha_true, amp_true),
    ]:
        pos = np.where(mean > 0, mean, np.nan)
        ax.plot(dims, pos, color=color, lw=2, label=label)
        ax.fill_between(dims, np.where(mean - std > 0, mean - std, np.nan), mean + std, color=color, alpha=0.2)
        fit = _fit_line(alpha, amp)
        if fit is not None:
            ax.plot(fit_dims, fit, color=color, lw=1.2, ls="--")

    ax.axvspan(start_dim + 1, end, alpha=0.06, color="gray", label=f"fit range {start_dim+1}-{end}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Spectrum")

    ratio = alpha_cv / alpha_st if (np.isfinite(alpha_cv) and np.isfinite(alpha_st) and alpha_st > 0) else float("nan")
    sw = "None" if cfg.smooth_width is None else f"{cfg.smooth_width:.2f}"
    g = cfg.generator
    if isinstance(g, TilburyConfig):
        path_info = f"path=tilbury  p_m={g.exponent_mean:.2f}  A2_beta={g.amplitude_ratio_beta:.1f}"
    else:
        path_info = f"path=rbf  peak_p={g.peak_exponent}"
    ax.set_title(f"Spectra (ratio={ratio:.3f}, n_sims={n_sims})\n" f"smooth={sw}  noise={cfg.noise_level:.2f}  {path_info}  norm={cfg.normalize}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_frac_neg(
    stacked: dict | SimConfig,
    n_sims: int = 10,
    device: str = "cpu",
) -> plt.Figure:
    """Fraction-negative curves for rCVPCA-fixed and raw stimspace cv_var.

    Parameters
    ----------
    stacked : dict or SimConfig
        Precomputed stacked dict from run_cvpca_stimspace_stack, or a SimConfig to run first.
    n_sims : int
        Number of sims when stacked is a SimConfig (ignored otherwise).
    device : str
    """
    if isinstance(stacked, SimConfig):
        stacked = run_cvpca_stimspace_stack(stacked, n_sims, device)

    n_comp = stacked["reg_cov"].shape[1]
    dims = np.arange(1, n_comp + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.plot(dims, stacked["frac_neg_reg_cov"], color="steelblue", lw=1.8, label="rCVPCA fixed")
    ax.plot(dims, stacked["frac_neg_cv_var_raw"], color="tomato", lw=1.8, label="stimspace cv_var (raw)")
    ax.axvspan(START_DIM + 1, min(END_DIM, n_comp), alpha=0.06, color="gray")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Fraction < 0")
    ax.legend(fontsize=9)
    ax.set_title("Fraction of negative values per component")
    plt.tight_layout()
    return fig


def plot_component_alignment(
    stacked: dict | SimConfig,
    n_show: int = 20,
    n_sims: int = 10,
    device: str = "cpu",
) -> plt.Figure:
    """Component comparison: rCVPCA and stimspace vs source SVD.

    Parameters
    ----------
    stacked : dict or SimConfig
        Stacked dict from run_cvpca_stimspace_stack, or a SimConfig to run first.
    n_show : int
        Number of components to display.
    n_sims : int
        Sims when stacked is a SimConfig.
    device : str
    """
    if isinstance(stacked, SimConfig):
        stacked = run_cvpca_stimspace_stack(stacked, n_sims, device)

    n_show = min(n_show, stacked["cvpca_components"].shape[2])
    xvals = np.arange(1, n_show + 1)
    extent = [0.5, n_show + 0.5, n_show + 0.5, 0.5]

    pairs = [
        ("rCVPCA (position space)", stacked["source_comps_p"][:, :, :n_show], stacked["cvpca_components"][:, :, :n_show]),
        ("stimspace (position space)", stacked["source_comps_p"][:, :, :n_show], stacked["stim_u"][:, :, :n_show]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), height_ratios=[1, 0.5], squeeze=False)

    for col, (title, src, tgt) in enumerate(pairs):
        cross0 = src[0].T @ tgt[0]  # (n_show, n_show)
        cross_all = np.einsum("sda, sdb -> sab", src, tgt)  # (K, n_show, n_show)
        subspace = np.sum(cross_all**2, axis=1)  # (K, n_show)

        axes[0, col].imshow(np.abs(cross0), aspect="auto", interpolation="none", extent=extent, cmap="gray_r", vmin=0, vmax=1)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].set_xlabel("Estimated component")
        axes[1, col].plot(xvals, np.mean(subspace, axis=0), color="black", lw=1.5)
        axes[1, col].fill_between(
            xvals,
            np.mean(subspace, axis=0) - np.std(subspace, axis=0),
            np.mean(subspace, axis=0) + np.std(subspace, axis=0),
            alpha=0.2,
            color="black",
        )
        axes[1, col].set_xlabel("Estimated component")
        axes[1, col].set_ylim(0, None)

    axes[0, 0].set_ylabel("Source component")
    axes[1, 0].set_ylabel("Subspace overlap")
    plt.tight_layout()
    return fig


def plot_study(study: optuna.Study) -> dict[str, "go.Figure"]:
    """Plotly figures for Optuna study: optimization history and parallel coordinates."""
    from plotly import graph_objects as go

    trials, values = _completed_trials(study)
    trial_nums = np.array([t.number for t in trials])
    running_best = np.maximum.accumulate(values)
    is_tilbury = _study_is_tilbury(study)

    hover = [_hover_text(t, is_tilbury) for t in trials]

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
        title="Optimization history — alpha_cvpca / alpha_stim",
        xaxis_title="Trial",
        yaxis_title="Ratio",
        height=400,
        width=800,
    )

    if is_tilbury:
        param_names = [
            "tb_amplitude_mean",
            "tb_amplitude_spread",
            "tb_amplitude_ratio_beta",
            "tb_peak_separation_scale",
            "tb_sigma_mean",
            "tb_sigma_spread",
            "tb_sigma_asym_std",
            "tb_exponent_mean",
            "tb_exponent_spread",
            "tb_repeat_noise_alpha",
            "tb_repeat_noise_lengthscale",
            "noise_level",
            "smooth_width",
            "normalize",
            "center",
        ]
        param_labels = [
            "amp mean",
            "amp std",
            "A2/A1 beta",
            "peak sep",
            "sig mean",
            "sig std",
            "sig asym",
            "p mean",
            "p std",
            "rep alpha",
            "rep ls",
            "rep thr",
            "noise",
            "smooth",
            "norm",
            "center",
        ]
    else:
        param_names = [
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
        param_labels = [
            "pf amp",
            "pf ls",
            "thr%",
            "rep a",
            "rep ls",
            "peak p",
            "s scale",
            "noise",
            "smooth",
            "norm",
            "center",
        ]

    all_params = {name: [t.params.get(name, None) for t in trials] for name in param_names}
    dimensions = [dict(label=lbl, values=all_params[name]) for name, lbl in zip(param_names, param_labels)]
    dimensions.append(dict(label="ratio", values=values.tolist()))

    fig_par = go.Figure(
        go.Parcoords(
            line=dict(color=values, colorscale="Viridis", showscale=True, colorbar=dict(title="ratio")),
            dimensions=dimensions,
        )
    )
    fig_par.update_layout(title="Parallel coordinates — colored by ratio", height=500, width=1200)

    return {"history": fig_hist, "parallel": fig_par}


def print_best_configs(study: optuna.Study, top_n: int = 5) -> None:
    """Print top_n trials sorted by alpha_cvpca / alpha_stim ratio."""
    trials, _ = _completed_trials(study)
    top = sorted(trials, key=lambda t: t.value, reverse=True)[:top_n]
    is_tilbury = _study_is_tilbury(study)

    print(f"\n=== Top {top_n} configs by alpha_cvpca/alpha_stim ratio ===")
    for rank, t in enumerate(top, 1):
        p = t.params
        a_cv = t.user_attrs.get("alpha_cvpca", float("nan"))
        a_st = t.user_attrs.get("alpha_stim", float("nan"))
        if is_tilbury:
            print(
                f"#{rank}  ratio={t.value:.4f}  trial={t.number}"
                f"  alpha_cvpca={a_cv:.3f}  alpha_stim={a_st:.3f}\n"
                f"    amp_m={p['tb_amplitude_mean']:.2f}  amp_s={p['tb_amplitude_spread']:.2f}"
                f"  A2_beta={p['tb_amplitude_ratio_beta']:.2f}  sep={p['tb_peak_separation_scale']:.2f}\n"
                f"    sig_m={p['tb_sigma_mean']:.2f}  sig_s={p['tb_sigma_spread']:.2f}"
                f"  sig_asym={p['tb_sigma_asym_std']:.2f}  p_m={p['tb_exponent_mean']:.2f}\n"
                f"    rep_a={p['tb_repeat_noise_alpha']:.2f}  rep_ls={p['tb_repeat_noise_lengthscale']:.2f}"
                f"  noise={p['noise_level']:.3f}"
                f"  smooth={_format_smooth_width(t)}  norm={p['normalize']}  center={p['center']}"
            )
        else:
            print(
                f"#{rank}  ratio={t.value:.4f}  trial={t.number}"
                f"  alpha_cvpca={a_cv:.3f}  alpha_stim={a_st:.3f}\n"
                f"    pf_amp={p['pf_amplitude']:.2f}  ls={p['pf_lengthscale']:.2f}"
                f"  thr={p['pf_threshold_pct']:.1f}  rep_a={p['pf_repeat_noise_alpha']:.2f}\n"
                f"    peak_p={p['peak_exponent']:.2f}  s_scale={p['peak_sigma_scale']:.2f}"
                f"  noise={p['noise_level']:.2f}  smooth={_format_smooth_width(t)}"
                f"  norm={p['normalize']}  center={p['center']}"
            )
