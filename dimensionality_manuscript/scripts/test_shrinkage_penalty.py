"""Compare shared vs separate shrinkage penalties on a handful of sessions.

Runs :meth:`TilburyFitConfig.process` twice per session -- once with
``shared_penalty=True`` (one strength for both penalty terms, ``len(lambda_grid)``
fits) and once with ``shared_penalty=False`` (independent strengths over the full
outer product, ``len(lambda_grid) ** 2`` fits) -- and reports, per session, the
selected strengths, the validation score they achieved, and the held-out test
performance of all three models.

Nothing is written to the results store; this is a diagnostic only.

Usage
-----
    python -m dimensionality_manuscript.scripts.test_shrinkage_penalty \
        --mice CR_Hippocannula6 ATL022 --sessions-per-mouse 2

    # Quicker pass (fewer Adam steps), single mouse:
    python -m dimensionality_manuscript.scripts.test_shrinkage_penalty \
        --mice ATL022 --sessions-per-mouse 1 --num-steps 2000

Note the cost: with the default 7-point grid, ``shared_penalty=False`` is 49
fits per session against 7 for the shared sweep, so a session takes roughly 6x
longer on the separate branch. Start with ``--num-steps`` well below the 10000
the pipeline uses.
"""

import argparse
from dataclasses import replace

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


def summarize(result: dict) -> dict:
    """Condense one ``process`` result into the numbers this comparison cares about."""
    p = result["params_shrinkage"][:, 5]
    asym = np.abs(np.log(result["params_shrinkage"][:, 3]) - np.log(result["params_shrinkage"][:, 4]))
    return {
        "lam_p": result["lambda_best_p"],
        "lam_asym": result["lambda_best_asym"],
        "val_score": result["lambda_score_best"],
        "r2_test_gen": np.nanmedian(result["r2_test"]),
        "r2_test_gauss": np.nanmedian(result["r2_test_control"]),
        "r2_test_shrink": np.nanmedian(result["r2_test_shrinkage"]),
        "p_dev": np.nanmean(np.abs(p - 2.0)),
        "asym": np.nanmean(asym),
        "n_neurons": int(np.sum(~np.isnan(result["r2_test_shrinkage"]))),
    }


def report(rows: list[tuple[str, dict, dict]]) -> None:
    """Print the per-session shared-vs-separate comparison table."""
    print("\n" + "=" * 118)
    print("Selected shrinkage strengths and held-out performance (median test R^2)")
    print("=" * 118)
    header = f"{'session':<28} {'mode':<8} {'lam_p':>9} {'lam_asym':>9} {'val score':>10} {'R2 shrink':>10} {'R2 gen':>9} {'R2 gauss':>9} {'|p-2|':>7} {'asym':>7}"
    print(header)
    print("-" * len(header))
    for uid, shared, separate in rows:
        for label, stats in (("shared", shared), ("separate", separate)):
            print(
                f"{uid if label == 'shared' else '':<28} {label:<8} "
                f"{stats['lam_p']:>9.2e} {stats['lam_asym']:>9.2e} {stats['val_score']:>10.4f} "
                f"{stats['r2_test_shrink']:>10.4f} {stats['r2_test_gen']:>9.4f} {stats['r2_test_gauss']:>9.4f} "
                f"{stats['p_dev']:>7.3f} {stats['asym']:>7.3f}"
            )
        print("-" * len(header))

    # Aggregate: does the separate sweep actually buy anything?
    val_gain = [shared["val_score"] - separate["val_score"] for _, shared, separate in rows]
    r2_gain = [separate["r2_test_shrink"] - shared["r2_test_shrink"] for _, shared, separate in rows]
    ratio = [separate["lam_p"] / separate["lam_asym"] for _, _, separate in rows]
    print(f"\nSessions compared: {len(rows)}")
    print(f"Validation score improvement from separate lambdas (positive = better): mean {np.mean(val_gain):+.4f}, max {np.max(val_gain):+.4f}")
    print(f"Median test R^2 change from separate lambdas (positive = better):       mean {np.mean(r2_gain):+.4f}, max {np.max(r2_gain):+.4f}")
    print(f"Selected lam_p / lam_asym ratio on the separate sweep: {['%.3g' % r for r in ratio]}")
    print("(A ratio far from 1 means the two penalties genuinely want different strengths.)")


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

    base = TilburyFitConfig(activity_parameters_name=args.activity_parameters_name)
    registry = PopulationRegistry(registry_params=base.data_config.to_registry_params())
    sessions = select_sessions(args.mice, args.sessions_per_mouse, args.seed)

    n_shared = len(replace(base, shared_penalty=True).lambda_combos())
    n_separate = len(replace(base, shared_penalty=False).lambda_combos())
    print(f"Sessions: {len(sessions)} | Adam steps: {args.num_steps}")
    print(f"  shared grid:   {list(base.lambda_grid_shared)} -> {n_shared} combos")
    print(f"  separate grid: lam_p {list(base.lambda_grid_p)} x lam_asym {list(base.lambda_grid_asym)} -> {n_separate} combos")
    print(f"Fits per session: {2 + n_shared} (shared) + {2 + n_separate} (separate)")

    process_kwargs = dict(
        verbose=args.verbose,
        device=args.device,
        gd_num_steps=args.num_steps,
        gd_learning_rate=args.learning_rate,
    )

    rows = []
    for session in sessions:
        session.params.spks_type = base.data_config.spks_type
        print(f"\n### {session.session_uid}")
        stats = {}
        for shared in (True, False):
            config = replace(base, shared_penalty=shared)
            print(f"  fitting shared_penalty={shared} ...", flush=True)
            result = config.process(session, registry, **process_kwargs)
            stats[shared] = summarize(result)
            s = stats[shared]
            print(f"    lam_p={s['lam_p']:.3g} lam_asym={s['lam_asym']:.3g} val={s['val_score']:.4f} n={s['n_neurons']}")
        rows.append((session.session_uid, stats[True], stats[False]))

    report(rows)


if __name__ == "__main__":
    main()
