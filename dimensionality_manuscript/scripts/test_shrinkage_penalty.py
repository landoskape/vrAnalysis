"""Diagnose the per-neuron shrinkage-prior selection on a handful of sessions.

Runs :meth:`TilburyFitConfig.process` per session and reports the things that
tell you whether the shrinkage machinery is behaving:

* the distribution of per-neuron selected ``(lam_p, lam_asym)`` -- how many
  neurons want no prior at all, what the typical strength is;
* **grid clipping** -- how many neurons want the strongest penalty on offer.
  Read it against the saturation argument in ``TilburyFitConfig``'s Notes: a
  strong asymmetry penalty already clamps the widths to near-equal, so clipping
  on ``lam_asym`` costs little, whereas clipping on ``lam_p`` would mean the
  exponent range is genuinely too narrow;
* ``n_var_floored``, which should be 0 -- nonzero means some neuron's
  train-curve variance hit the normalisation floor and its effective prior is
  stronger than its ``lam`` implies;
* the ``lam = (0, 0)`` consistency check: that combination is the unregularized
  generalized model exactly, so its population validation score must match the
  generalized fit's;
* held-out (test) performance of all three models, which is the only
  informative shrinkage-vs-generalized comparison -- on *validation* the
  shrinkage model wins by construction, since ``(0, 0)`` is in its grid.

Nothing is written to the results store; this is a diagnostic only.

Usage
-----
    python -m dimensionality_manuscript.scripts.test_shrinkage_penalty \
        --mice CR_Hippocannula6 ATL022 --sessions-per-mouse 2

    # Quicker pass (fewer Adam steps), single mouse:
    python -m dimensionality_manuscript.scripts.test_shrinkage_penalty \
        --mice ATL022 --sessions-per-mouse 1 --num-steps 2000

Note the cost: the sweep is the full ``lambda_grid_p`` x ``lambda_grid_asym``
outer product, so a session costs that many fits plus the two unregularized
ones. Start with ``--num-steps`` well below the 10000 the pipeline uses.
"""

import argparse

import numpy as np

from dimensionality_manuscript import TilburyFitConfig
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.scripts.run import collect_sessions


def select_sessions(mice: list[str] | None, sessions_per_mouse: int, seed: int) -> list:
    """Draw up to ``sessions_per_mouse`` random sessions for each requested mouse.

    Parameters
    ----------
    mice : list of str or None
        Mouse names to include. ``None`` uses every mouse in the database.
    sessions_per_mouse : int
        Maximum number of sessions to draw per mouse.
    seed : int
        Seed for the random draw, so a run is reproducible.

    Returns
    -------
    list of B2Session
        The drawn sessions, grouped by mouse in the requested order.
    """
    sessions = collect_sessions()
    by_mouse: dict[str, list] = {}
    for session in sessions:
        by_mouse.setdefault(session.mouse_name, []).append(session)

    if mice is None:
        mice = sorted(by_mouse)
    unknown = [m for m in mice if m not in by_mouse]
    if unknown:
        raise ValueError(f"Unknown mice {unknown}. Available: {sorted(by_mouse)}")

    rng = np.random.default_rng(seed)
    chosen = []
    for mouse in mice:
        available = by_mouse[mouse]
        n_draw = min(sessions_per_mouse, len(available))
        idx = rng.choice(len(available), size=n_draw, replace=False)
        chosen.extend(available[i] for i in sorted(idx))
    return chosen


def summarize(result: dict, config: TilburyFitConfig) -> dict:
    """Condense one ``process`` result into the numbers this diagnostic cares about.

    Parameters
    ----------
    result : dict
        Return value of :meth:`TilburyFitConfig.process`.
    config : TilburyFitConfig
        The config that produced ``result``; supplies the grid endpoints used to
        label the clipping columns.

    Returns
    -------
    dict
        One row of the report tables.
    """
    selected = result["lambda_selected"]  # (N, 2)
    fitted = np.isfinite(selected).all(axis=1)
    lam_p, lam_asym = selected[fitted, 0], selected[fitted, 1]

    params_s = result["params_shrinkage"]
    p = params_s[:, 5]
    asym = np.abs(np.log(params_s[:, 3]) - np.log(params_s[:, 4]))

    # lam = (0, 0) is the unregularized generalized model exactly, so its sweep
    # score must match the generalized fit's population validation error.
    combos = result["lambda_combos"]
    idx_zero = int(np.flatnonzero((combos[:, 0] == 0) & (combos[:, 1] == 0))[0])
    r2_val_gen = result["r2_val"]
    score_gen = float(np.nanmean(1.0 - r2_val_gen[np.isfinite(r2_val_gen)]))

    return {
        "n_fit": int(fitted.sum()),
        "n_kept": int(selected.shape[0]),
        "frac_lam_p_zero": float(np.mean(lam_p == 0.0)) if fitted.any() else np.nan,
        "frac_lam_asym_zero": float(np.mean(lam_asym == 0.0)) if fitted.any() else np.nan,
        "median_lam_p": float(np.median(lam_p)) if fitted.any() else np.nan,
        "median_lam_asym": float(np.median(lam_asym)) if fitted.any() else np.nan,
        "clip_p": result["frac_lambda_clipped_p"],
        "clip_asym": result["frac_lambda_clipped_asym"],
        "n_var_floored": result["n_var_floored"],
        "score_lam_zero": float(result["lambda_scores"][idx_zero]),
        "score_gen": score_gen,
        "r2_test_gen": float(np.nanmedian(result["r2_test"])),
        "r2_test_gauss": float(np.nanmedian(result["r2_test_control"])),
        "r2_test_shrink": float(np.nanmedian(result["r2_test_shrinkage"])),
        "p_dev": float(np.nanmean(np.abs(p - 2.0))),
        "asym": float(np.nanmean(asym)),
        "grid_max_p": config.lambda_grid_p[-1],
        "grid_max_asym": config.lambda_grid_asym[-1],
    }


def report(rows: list[tuple[str, dict]]) -> None:
    """Print the per-session diagnostic tables."""
    print("\n" + "=" * 100)
    print("Per-neuron shrinkage prior selection")
    print("=" * 100)
    header = (
        f"{'session':<28} {'n fit':>9} {'lam_p=0':>8} {'lam_a=0':>8} {'med lam_p':>10} "
        f"{'med lam_a':>10} {'clip_p':>7} {'clip_a':>7} {'floor':>6}"
    )
    print(header)
    print("-" * len(header))
    for uid, s in rows:
        print(
            f"{uid:<28} {s['n_fit']:>5}/{s['n_kept']:<3} {s['frac_lam_p_zero']:>8.2f} {s['frac_lam_asym_zero']:>8.2f} "
            f"{s['median_lam_p']:>10.3g} {s['median_lam_asym']:>10.3g} {s['clip_p']:>7.2f} {s['clip_asym']:>7.2f} "
            f"{s['n_var_floored']:>6}"
        )

    print("\n" + "=" * 100)
    print("Held-out performance (median test R^2) and fitted shape")
    print("=" * 100)
    header2 = f"{'session':<28} {'R2 shrink':>10} {'R2 gen':>9} {'R2 gauss':>9} {'|p-2|':>7} {'asym':>7}"
    print(header2)
    print("-" * len(header2))
    for uid, s in rows:
        print(
            f"{uid:<28} {s['r2_test_shrink']:>10.4f} {s['r2_test_gen']:>9.4f} {s['r2_test_gauss']:>9.4f} "
            f"{s['p_dev']:>7.3f} {s['asym']:>7.3f}"
        )

    # --- consistency check: lam=(0,0) must reproduce the generalized fit ---
    print("\nConsistency check -- lam=(0,0) vs the unregularized generalized fit:")
    worst = 0.0
    for uid, s in rows:
        delta = abs(s["score_lam_zero"] - s["score_gen"])
        worst = max(worst, delta)
        flag = "" if delta < 1e-6 else "   <-- MISMATCH"
        print(f"  {uid:<28} lam0 {s['score_lam_zero']:.6f} vs gen {s['score_gen']:.6f}  (delta {delta:.2e}){flag}")
    print(f"  Worst delta: {worst:.2e} (should be ~0; nonzero means the two paths have diverged)")

    # --- grid adequacy ---
    clip_p = [s["clip_p"] for _, s in rows]
    clip_a = [s["clip_asym"] for _, s in rows]
    floored = sum(s["n_var_floored"] for _, s in rows)
    grid_max_p, grid_max_asym = rows[0][1]["grid_max_p"], rows[0][1]["grid_max_asym"]
    print(f"\nSessions: {len(rows)}")
    print(f"Neurons whose plain best lam_p sits at the grid max ({grid_max_p:g}):    mean {np.mean(clip_p):.2f}, max {np.max(clip_p):.2f}")
    print(f"Neurons whose plain best lam_asym sits at the grid max ({grid_max_asym:g}): mean {np.mean(clip_a):.2f}, max {np.max(clip_a):.2f}")
    print("  (More than ~0.1 means the grid is too narrow -- extend lambda_grid_p / lambda_grid_asym.)")
    print(f"Neurons hitting the variance floor: {floored} (should be 0)")
    print("\nNote: the shrinkage model contains the generalized model at lam=(0,0), so it cannot lose")
    print("on validation. Judge it only by the test-R2 column above.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mice", nargs="+", default=None, help="Mouse names to test. Default: every mouse in the database.")
    parser.add_argument("--sessions-per-mouse", type=int, default=1, help="Sessions drawn per mouse (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the session draw (default: 0)")
    parser.add_argument("--num-steps", type=int, default=2000, help="Adam steps per fit (default: 2000; pipeline uses 10000)")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Adam learning rate (default: 0.1)")
    parser.add_argument("--activity-parameters-name", default="raw", help="ActivityParameters preset (default: raw)")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available)")
    parser.add_argument("--verbose", action="store_true", help="Show a progress bar for every fit")
    args = parser.parse_args()

    config = TilburyFitConfig(activity_parameters_name=args.activity_parameters_name)
    registry = PopulationRegistry(registry_params=config.data_config.to_registry_params())
    sessions = select_sessions(args.mice, args.sessions_per_mouse, args.seed)

    n_combos = len(config.lambda_combos())
    print(f"Sessions: {len(sessions)} | Adam steps: {args.num_steps}")
    print(f"  lam_p grid:    {list(config.lambda_grid_p)}")
    print(f"  lam_asym grid: {list(config.lambda_grid_asym)}")
    print(f"Fits per session: {n_combos} shrinkage + 2 unregularized")

    process_kwargs = dict(
        verbose=args.verbose,
        device=args.device,
        gd_num_steps=args.num_steps,
        gd_learning_rate=args.learning_rate,
    )

    rows = []
    for session in sessions:
        session.params.spks_type = config.data_config.spks_type
        print(f"\n### {session.session_uid}", flush=True)
        result = config.process(session, registry, **process_kwargs)
        stats = summarize(result, config)
        rows.append((session.session_uid, stats))
        print(
            f"    n_fit={stats['n_fit']}/{stats['n_kept']} "
            f"med lam=({stats['median_lam_p']:.3g}, {stats['median_lam_asym']:.3g}) "
            f"clip=({stats['clip_p']:.2f}, {stats['clip_asym']:.2f})"
        )

    report(rows)


if __name__ == "__main__":
    main()
