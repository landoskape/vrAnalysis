"""Which lambda-selection rule actually wins on held-out data?

A first pass on one session showed the shrinkage model losing to both of its
endpoints on test R^2 while validation asked for ever-stronger priors. A model
sitting *between* two good endpoints and scoring below both points at selection
noise rather than at a badly chosen prior strength -- each neuron picks its own
lambda off one noisy validation curve, and the mixture pays for the variance.

This script separates those two explanations. It runs the shrinkage sweep once
per session and then scores several selection rules on the **test** split from
that single sweep, so the comparison is free of refitting noise:

``plain``
    Per neuron, the lowest validation error. Pure argmin, no tolerance.
``1se``
    Per neuron, the strongest penalty within one standard error of that
    minimum -- what :meth:`TilburyFitConfig.process` ships.
``half-se`` / ``quarter-se``
    The same rule with a tightened tolerance, to see how much of any gap the
    1-SE band itself is responsible for.
``population``
    One lambda for the whole session (lowest mean validation error), applied to
    every neuron. Carries no per-neuron selection variance at all, so beating
    ``plain`` is direct evidence that per-neuron selection is too noisy here.
``oracle``
    Per neuron, the lowest *test* error. Cheating, and reported only as the
    ceiling any honest rule is chasing. If ``oracle`` barely beats the
    unregularized fit there is nothing for a better rule to recover.

Also reported per session: the two unregularized baselines, and the test score
of each fixed lambda applied uniformly (so the grid's own shape is visible).

Nothing is written to the results store; this is a diagnostic only.

Usage
-----
    python -m dimensionality_manuscript.scripts.test_lambda_selection \
        --mice ATL022 CR_Hippocannula6 --sessions-per-mouse 2 --num-steps 4000
"""

import argparse

import numpy as np

from dimensionality_manuscript import TilburyFitConfig
from dimensionality_manuscript.configs.tilbury_fit import (
    _SPEC_CONTROL,
    _SPEC_SHRINKAGE,
    _SPEC_TILBURY,
    _eval_gaussian,
    _eval_tilbury,
    _fit_all_neurons,
    _r2,
)
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.scripts.test_shrinkage_penalty import select_sessions

# Tolerance multipliers on the standard error, keyed by report label.
_SE_RULES = {"1se": 1.0, "half-se": 0.5, "quarter-se": 0.25}


def _select_with_tolerance(
    val_err: np.ndarray,
    val_se: np.ndarray,
    strengths: np.ndarray,
    n_se: float,
) -> np.ndarray:
    """Per-neuron selection with an ``n_se``-wide tolerance band.

    Parameters
    ----------
    val_err, val_se : np.ndarray
        Validation error and its standard error, shape ``(n_combos, n_cells)``.
    strengths : np.ndarray
        Penalty-strength rank per combination, shape ``(n_combos,)``.
    n_se : float
        Width of the tolerance band in standard errors. ``0`` reduces to a plain
        argmin.

    Returns
    -------
    np.ndarray
        Chosen combination index per neuron, ``-1`` where none fitted.
    """
    n_cells = val_err.shape[1]
    out = np.full(n_cells, -1, dtype=int)
    for n in range(n_cells):
        err = val_err[:, n]
        finite = np.flatnonzero(np.isfinite(err))
        if finite.size == 0:
            continue
        best = int(finite[np.argmin(err[finite])])
        se = val_se[best, n]
        tol = err[best] + n_se * (se if np.isfinite(se) else 0.0)
        admissible = finite[err[finite] <= tol]
        strongest = admissible[strengths[admissible] == strengths[admissible].max()]
        out[n] = int(strongest[np.argmin(err[strongest])])
    return out


def _score(r2_test_by_combo: np.ndarray, idx: np.ndarray, valid: np.ndarray) -> float:
    """Median test R^2 of a selection rule over the neurons it chose for."""
    rows = np.flatnonzero(valid & (idx >= 0))
    if rows.size == 0:
        return np.nan
    return float(np.nanmedian(r2_test_by_combo[idx[rows], rows]))


def run_session(session, config: TilburyFitConfig, registry: PopulationRegistry, fit_kwargs: dict) -> dict:
    """Fit one session's sweep and score every selection rule on the test split."""
    theta, curves, _, _ = config.prepare_curves(session, registry)
    n_cells = curves["train"].shape[0]

    fit_gen = _fit_all_neurons(theta, curves["train"], curves["validation"], _SPEC_TILBURY, **fit_kwargs)
    fit_gauss = _fit_all_neurons(theta, curves["train"], curves["validation"], _SPEC_CONTROL, **fit_kwargs)

    combos = config.lambda_combos()
    # Penalty-strength rank: the grid is 2-D and only partially ordered, so the
    # tolerance rules need a total order to pick "the strongest admissible
    # penalty". Summing grid indices is monotone in both lambdas.
    n_asym = len(config.lambda_grid_asym)
    strengths = np.array([i // n_asym + i % n_asym for i in range(len(combos))], dtype=int)
    n_combos = len(combos)
    val_err = np.full((n_combos, n_cells), np.nan)
    val_se = np.full((n_combos, n_cells), np.nan)
    r2_test = np.full((n_combos, n_cells), np.nan)
    for i, (lam_p, lam_asym) in enumerate(combos):
        fit_s = _fit_all_neurons(
            theta, curves["train"], curves["validation"], _SPEC_SHRINKAGE, lam=(float(lam_p), float(lam_asym)), **fit_kwargs
        )
        val_err[i], val_se[i] = fit_s.val_err, fit_s.val_se
        for n in range(n_cells):
            if not np.any(np.isnan(fit_s.params[n])):
                r2_test[i, n] = _r2(_eval_tilbury(theta, fit_s.params[n]), curves["test"][n])
        print(f"      lam=({lam_p:g}, {lam_asym:g})  median test R2 {np.nanmedian(r2_test[i]):.4f}", flush=True)

    valid = np.isfinite(val_err).any(axis=0) & np.isfinite(r2_test).any(axis=0)

    # Unregularized baselines, scored on the same neurons.
    r2_gen = np.full(n_cells, np.nan)
    r2_gauss = np.full(n_cells, np.nan)
    for n in range(n_cells):
        if not np.any(np.isnan(fit_gen.params[n])):
            r2_gen[n] = _r2(_eval_tilbury(theta, fit_gen.params[n]), curves["test"][n])
        if not np.any(np.isnan(fit_gauss.params[n])):
            r2_gauss[n] = _r2(_eval_gaussian(theta, fit_gauss.params[n]), curves["test"][n])

    rules = {"plain": _select_with_tolerance(val_err, val_se, strengths, 0.0)}
    for label, n_se in _SE_RULES.items():
        rules[label] = _select_with_tolerance(val_err, val_se, strengths, n_se)

    # Population rule: one lambda for the session, applied to every neuron.
    pop_scores = np.nanmean(val_err, axis=1)
    idx_pop = int(np.nanargmin(pop_scores))
    rules["population"] = np.where(valid, idx_pop, -1)

    # Oracle: cheats by selecting on test. Ceiling only.
    oracle = np.full(n_cells, -1, dtype=int)
    for n in np.flatnonzero(valid):
        finite = np.flatnonzero(np.isfinite(r2_test[:, n]))
        if finite.size:
            oracle[n] = int(finite[np.argmax(r2_test[finite, n])])
    rules["oracle"] = oracle

    return {
        "uid": session.session_uid,
        "n_valid": int(valid.sum()),
        "n_cells": n_cells,
        "r2_gen": float(np.nanmedian(r2_gen[valid])),
        "r2_gauss": float(np.nanmedian(r2_gauss[valid])),
        "rule_scores": {label: _score(r2_test, idx, valid) for label, idx in rules.items()},
        "pop_lambda": tuple(float(v) for v in combos[idx_pop]),
        "fixed_scores": [float(np.nanmedian(r2_test[i][valid])) for i in range(n_combos)],
        "combos": combos,
        "median_strength": {label: float(np.median(strengths[idx[idx >= 0]])) for label, idx in rules.items()},
    }


def report(rows: list[dict]) -> None:
    """Print the per-session rule comparison and the aggregate verdict."""
    labels = ["plain", "1se", "half-se", "quarter-se", "population", "oracle"]

    print("\n" + "=" * 112)
    print("Median test R^2 by lambda-selection rule (higher is better)")
    print("=" * 112)
    header = f"{'session':<28} {'n':>6} {'gen':>8} {'gauss':>8} " + " ".join(f"{lab:>11}" for lab in labels)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = " ".join(f"{r['rule_scores'][lab]:>11.4f}" for lab in labels)
        print(f"{r['uid']:<28} {r['n_valid']:>6} {r['r2_gen']:>8.4f} {r['r2_gauss']:>8.4f} {cells}")

    print("\n" + "=" * 112)
    print("Gap to the unregularized generalized fit (positive = shrinkage helps)")
    print("=" * 112)
    header2 = f"{'session':<28} " + " ".join(f"{lab:>11}" for lab in labels)
    print(header2)
    print("-" * len(header2))
    for r in rows:
        cells = " ".join(f"{r['rule_scores'][lab] - r['r2_gen']:>+11.4f}" for lab in labels)
        print(f"{r['uid']:<28} {cells}")

    print("\nMean gap to generalized across sessions:")
    for lab in labels:
        gaps = [r["rule_scores"][lab] - r["r2_gen"] for r in rows]
        print(f"  {lab:<12} {np.mean(gaps):+.4f}   (per-session: {', '.join(f'{g:+.4f}' for g in gaps)})")

    print("\nPopulation-rule lambda chosen per session:")
    for r in rows:
        print(f"  {r['uid']:<28} {r['pop_lambda']}")

    print("\nFixed-lambda test scores (same lambda for every neuron), median test R^2:")
    combos = rows[0]["combos"]
    head = f"{'session':<28} " + " ".join(f"{f'({a:g},{b:g})':>12}" for a, b in combos)
    print(head)
    for r in rows:
        print(f"{r['uid']:<28} " + " ".join(f"{s:>12.4f}" for s in r["fixed_scores"]))

    # --- verdict ---
    plain = np.mean([r["rule_scores"]["plain"] - r["r2_gen"] for r in rows])
    pop = np.mean([r["rule_scores"]["population"] - r["r2_gen"] for r in rows])
    one_se = np.mean([r["rule_scores"]["1se"] - r["r2_gen"] for r in rows])
    oracle = np.mean([r["rule_scores"]["oracle"] - r["r2_gen"] for r in rows])
    print("\n" + "-" * 112)
    print("Reading this:")
    print(f"  oracle gap {oracle:+.4f}      -- the ceiling. If ~0, no selection rule can make shrinkage pay.")
    print(f"  plain vs population: {plain:+.4f} vs {pop:+.4f}")
    print("      population ahead => per-neuron selection is too noisy; shrink lambdas toward a shared value.")
    print("      plain ahead      => per-neuron selection is sound and the tolerance/grid is the problem.")
    print(f"  1se gap {one_se:+.4f} vs plain gap {plain:+.4f}")
    print("      1se well below plain => the tolerance band is too wide; tighten or drop it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mice", nargs="+", default=None, help="Mouse names to test. Default: every mouse in the database.")
    parser.add_argument("--sessions-per-mouse", type=int, default=1, help="Sessions drawn per mouse (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the session draw (default: 0)")
    parser.add_argument("--num-steps", type=int, default=4000, help="Adam steps per fit (default: 4000)")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Adam learning rate (default: 0.1)")
    parser.add_argument("--activity-parameters-name", default="raw", help="ActivityParameters preset (default: raw)")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available)")
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = TilburyFitConfig(activity_parameters_name=args.activity_parameters_name)
    registry = PopulationRegistry(registry_params=config.data_config.to_registry_params())
    sessions = select_sessions(args.mice, args.sessions_per_mouse, args.seed)

    fit_kwargs = dict(
        sigma_min=config.sigma_min,
        device=device,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        verbose=False,
    )
    print(f"Sessions: {len(sessions)} | Adam steps: {args.num_steps} | device {device}")
    print(f"Grid: {len(config.lambda_combos())} combos from lam_p {list(config.lambda_grid_p)} x lam_asym {list(config.lambda_grid_asym)}")

    rows = []
    for session in sessions:
        session.params.spks_type = config.data_config.spks_type
        print(f"\n### {session.session_uid}", flush=True)
        rows.append(run_session(session, config, registry, fit_kwargs))
        r = rows[-1]
        print(f"    gen {r['r2_gen']:.4f} | plain {r['rule_scores']['plain']:.4f} | 1se {r['rule_scores']['1se']:.4f} | pop {r['rule_scores']['population']:.4f}")

    report(rows)


if __name__ == "__main__":
    main()
