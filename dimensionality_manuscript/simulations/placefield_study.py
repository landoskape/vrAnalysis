"""
CLI entry point for place field simulation studies.

Usage
-----
python -m dimensionality_manuscript.simulations.placefield_study <subcommand> [args]

Subcommands
-----------
neuron-position     Neuron-space vs position-space CVPCA comparison
cvpca-stimspace     rCVPCA-fixed vs stimspace alpha-ratio Optuna study
"""

import argparse

import matplotlib.pyplot as plt

from .placefield_analysis import (
    config_from_params,
    plot_component_alignment,
    plot_example_placefields,
    plot_frac_neg,
    plot_neuron_position,
    plot_placefields,
    plot_spectra,
    plot_study,
    print_best_configs,
    run_cvpca_stimspace_stack,
    run_neuron_position,
    run_study,
)
from .placefield_generator import PlacefieldConfig, SimConfig, TilburyConfig


# ---------------------------------------------------------------------------
# Shared argument setup
# ---------------------------------------------------------------------------


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Add args common to both subcommands."""
    g = parser.add_argument_group("population / position")
    g.add_argument("--n-neurons", type=int, default=500)
    g.add_argument("--n-positions", type=int, default=100)
    g.add_argument("--n-components", type=int, default=80)
    g.add_argument("--n-repeats", type=int, default=4)

    g2 = parser.add_argument_group("simulation")
    g2.add_argument("--noise-level", type=float, default=1.0)
    g2.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    g2.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    g2.add_argument("--smooth-width", type=float, default=3.0, metavar="W",
                    help="Gaussian smoothing kernel width in bins (default 3.0; set to 0 to disable)")
    g2.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    g2.add_argument("--generation-path", default="rbf", choices=("rbf", "tilbury"))


def _build_sim_config(args: argparse.Namespace) -> SimConfig:
    """Construct SimConfig from parsed namespace."""
    if args.generation_path == "tilbury":
        generator: PlacefieldConfig | TilburyConfig = TilburyConfig(
            n_neurons=args.n_neurons,
            n_positions=args.n_positions,
        )
    else:
        generator = PlacefieldConfig(
            n_neurons=args.n_neurons,
            n_positions=args.n_positions,
        )
    return SimConfig(
        generator=generator,
        noise_level=args.noise_level,
        n_repeats=args.n_repeats,
        normalize=args.normalize,
        center=args.center,
        smooth_width=args.smooth_width if args.smooth_width != 0 else None,
        n_components=args.n_components,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# neuron-position subcommand
# ---------------------------------------------------------------------------


def _run_neuron_position(args: argparse.Namespace) -> None:
    cfg = _build_sim_config(args)
    device = args.device

    if not args.skip_simulations:
        print(f"Running neuron-position ({args.n_simulations} sims, device={device})...")
        results = run_neuron_position(cfg, args.n_simulations, device=device)
        print(f"  burn_in={results['burn_in']}")
        plot_neuron_position({"run": results}, suptitle="Neuron vs position cvPCA")
        plt.show()

    print("Placefield heatmaps...")
    plot_placefields(cfg, device=device)
    plt.show()

    if args.compare_peaky and isinstance(cfg.generator, PlacefieldConfig) and cfg.generator.peak_exponent is None:
        from dataclasses import replace
        peaky_cfg = replace(cfg, generator=replace(cfg.generator, peak_exponent=1.0))
        print("Placefield heatmaps (peaky, p=1)...")
        plot_placefields(peaky_cfg, device=device)
        plt.show()

    if not args.optimize:
        return

    from .placefield_analysis import suggest_config as _suggest
    print("\nRunning Optuna study (neuron-position asymmetry objective)...")
    print("  Note: neuron-position Optuna uses the cvpca-stimspace run_study.")
    print("  For asymmetry-based optimization, use the legacy my_precious_cvpca_simulation.py --optimize.")


def _build_neuron_position_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("neuron-position", help="Neuron-space vs position-space CVPCA")
    _add_shared_args(p)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-simulations", type=int, default=20)
    p.add_argument("--skip-simulations", action="store_true")
    p.add_argument("--compare-peaky", action="store_true",
                   help="Also show peaky (peak_exponent=1) placefield heatmaps for rbf path")
    p.add_argument("--optimize", action="store_true",
                   help="Run Optuna optimization after normal run (reserved for future use)")
    p.set_defaults(func=_run_neuron_position)


# ---------------------------------------------------------------------------
# cvpca-stimspace subcommand
# ---------------------------------------------------------------------------


def _run_cvpca_stimspace(args: argparse.Namespace) -> None:
    cfg = _build_sim_config(args)
    device = args.device

    if cfg.n_repeats < 4:
        raise ValueError(f"n_repeats must be >= 4 for cvpca-stimspace, got {cfg.n_repeats}")

    print(
        f"Running cvpca-stimspace Optuna study: {args.n_trials} trials, "
        f"{args.n_sims_per_trial} sims/trial, device={device}, path={args.generation_path}"
    )
    study = run_study(cfg, n_trials=args.n_trials, n_sims_per_trial=args.n_sims_per_trial, device=device, seed=args.seed)

    print_best_configs(study, top_n=args.top_n)

    figs = plot_study(study)
    figs["history"].show()
    figs["parallel"].show()

    cfg_best = config_from_params(study.best_trial.params, cfg)
    print("\nPlotting best-config spectra...")
    plot_spectra(cfg_best, n_sims=args.n_sims_per_trial, device=device)
    plt.show()

    print("Plotting fraction-negative and component alignment...")
    stacked = run_cvpca_stimspace_stack(cfg_best, n_sims=args.n_sims_per_trial, device=device)
    plot_frac_neg(stacked)
    plt.show()
    plot_component_alignment(stacked, n_show=min(20, cfg_best.n_components))
    plt.show()

    print("Placefield examples...")
    plot_example_placefields(cfg_best, device=device)
    plt.show()


def _build_cvpca_stimspace_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("cvpca-stimspace", help="rCVPCA-fixed vs stimspace alpha-ratio Optuna study")
    _add_shared_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--n-sims-per-trial", type=int, default=5)
    p.add_argument("--top-n", type=int, default=5)
    p.set_defaults(func=_run_cvpca_stimspace)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Place field simulation studies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="subcommand")
    _build_neuron_position_parser(subparsers)
    _build_cvpca_stimspace_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
