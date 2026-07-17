"""Smoke test for `effective_dimensionality` on the regression models.

For one local session it instantiates each regression model and prints
`regressor_dimensionality()` beside `effective_dimensionality(session)` — the participation
ratio of the model's test prediction (the literal dimensionality of `Xβ`).

Sanity checks:
- effective_dimensionality > 0 and <= min(num_neurons, num_test_samples).
- RRR: effective_dimensionality <= rank (top latents dominate -> usually below rank).
- predict_latents / internal / gain variants all return a finite PR (the prediction path is
  exercised end-to-end).

Usage
-----
    conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.test_effective_parameters
    conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.test_effective_parameters --session ATL022.2023-04-12.701
"""

import argparse
import sys
import traceback

from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.regression_models.base import participation_ratio
from dimensionality_manuscript.regression_models.hyperparameters import ReducedRankRegressionHyperparameters
from dimensionality_manuscript.regression_models.models import (
    PlaceFieldModel,
    RBFPosModel,
    FullRegressorModel,
    ReducedRankRegressionModel,
)
from dimensionality_manuscript.scripts.run import build_analysis_configs, collect_sessions


def _first_loadable_session(sessions, registry, limit):
    """Return the first session whose population loads from the local cache."""
    for session in sessions[:limit]:
        try:
            registry.get_population(session)
            return session
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Smoke test effective_dimensionality")
    parser.add_argument("--session", default=None, help="session_uid to test (default: first loadable)")
    parser.add_argument("--limit", type=int, default=25, help="max sessions to probe for a loadable one")
    args = parser.parse_args()

    reg_configs = build_analysis_configs(include=["regression"])
    registry = PopulationRegistry(registry_params=reg_configs[0].data_config.to_registry_params())

    sessions = collect_sessions()
    if args.session:
        sessions = [s for s in sessions if s.session_uid == args.session]
        if not sessions:
            print(f"Session {args.session!r} not found.", file=sys.stderr)
            sys.exit(1)
        session = sessions[0]
    else:
        session = _first_loadable_session(sessions, registry, args.limit)
        if session is None:
            print(f"No loadable session found in first {args.limit} sessions.", file=sys.stderr)
            sys.exit(1)

    num_env = len(session.environments)
    print(f"Session: {session.session_uid}  (num_environments={num_env})")

    # Sanity ceiling: PR can't exceed min(num_neurons, num_test_samples).
    dummy = ReducedRankRegressionModel(registry)
    src, tgt, _ = dummy.get_session_data(session, None, "test")
    ceiling = min(tgt.shape[0], tgt.shape[1])
    print(f"num_target_neurons={tgt.shape[0]}  num_test_samples={tgt.shape[1]}  PR ceiling={ceiling}")

    # True-activity PR for reference (what neural dimensionality looks like in this space).
    print(f"true test activity PR = {participation_ratio(tgt.numpy()):.3f}")
    print("-" * 72)

    n_fail = 0

    def report(name, model, nominal, ceiling_val):
        nonlocal n_fail
        try:
            eff = model.effective_dimensionality(session)
            flag = ""
            if not (0 < eff <= ceiling_val + 1e-6):
                flag = "  !! outside (0, ceiling]"
                n_fail += 1
            print(f"  {name:<38} nominal={nominal:<7} eff_dim={eff:7.3f}{flag}")
            return eff
        except Exception:
            print(f"  {name:<38} FAILED")
            traceback.print_exc()
            n_fail += 1
            return None

    pf = PlaceFieldModel(registry)
    report("PlaceField", pf, pf.regressor_dimensionality(num_env), ceiling)

    pf_internal_gain = PlaceFieldModel(registry, internal=True, gain=True)
    report("PlaceField (internal, gain)", pf_internal_gain, pf_internal_gain.regressor_dimensionality(num_env), ceiling)

    rbf = RBFPosModel(registry)
    report("RBFPos (predict_latents)", rbf, rbf.regressor_dimensionality(num_env), ceiling)

    rbf_dec = RBFPosModel(registry, predict_latents=False)
    report("RBFPos (decoder-only)", rbf_dec, rbf_dec.regressor_dimensionality(num_env), ceiling)

    full = FullRegressorModel(registry)
    report("FullRegressor", full, full.regressor_dimensionality(num_env), ceiling)

    full_nr = FullRegressorModel(registry, no_reward=True, speed_basis=False)
    report("FullRegressor (no_reward, 1d speed)", full_nr, full_nr.regressor_dimensionality(num_env), ceiling)

    for rank in [5, 50, 200, 1000]:
        hyps = ReducedRankRegressionHyperparameters(rank=rank, alpha=100.0)
        rrr = ReducedRankRegressionModel(registry, hyperparameters=hyps)
        rank = rrr.regressor_dimensionality()
        eff_rrr = report("ReducedRank", rrr, rank, ceiling)
        if eff_rrr is not None and eff_rrr > rank + 1e-6:
            print(f"    !! RRR eff_dim {eff_rrr:.3f} exceeds rank {rank}")
            n_fail += 1

    print("-" * 72)
    print("ALL OK" if n_fail == 0 else f"{n_fail} FAILURE(S)")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
