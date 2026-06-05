"""Quick smoke test for RRR fix on a single session.

Run on the login node to verify linalg.eigh no longer crashes before
submitting a full array job.

Usage
-----
    qrsh -l h_rt=1:00:00,mem=16G
    module load python/3.11.4
    source ~/Scratch/envs/vrAnalysis/bin/activate
    cd ~/vrAnalysis

    python -m dimensionality_manuscript.scripts.test_rrr_single --sessions-file sessions.json --session ATL022.2023-03-31.703

    # Test first N sessions from file:
    python -m dimensionality_manuscript.scripts.test_rrr_single --sessions-file sessions.json --n-sessions 3
"""

import argparse
import sys
import traceback
from pathlib import Path

from dimensionality_manuscript.registry import RegistryPaths, PopulationRegistry
from dimensionality_manuscript.scripts.run import (
    build_analysis_configs,
    collect_sessions_from_file,
)

REGISTRY_PATHS = RegistryPaths()


def main():
    parser = argparse.ArgumentParser(description="Test RRR on a single session")
    parser.add_argument("--sessions-file", type=Path, required=True)
    parser.add_argument("--session", default=None, help="session_uid to test (e.g. ATL020.2023-03-27.701)")
    parser.add_argument("--n-sessions", type=int, default=1, help="Test first N sessions (default: 1)")
    parser.add_argument("--model", default=None, help="model_name filter, e.g. rrr or rrr_no_intercept (default: both)")
    args = parser.parse_args()

    sessions = collect_sessions_from_file(args.sessions_file)

    if args.session:
        sessions = [s for s in sessions if s.session_uid == args.session]
        if not sessions:
            print(f"Session {args.session!r} not found in {args.sessions_file}", file=sys.stderr)
            sys.exit(1)
    else:
        sessions = sessions[: args.n_sessions]

    param_filters = {"model_name": args.model} if args.model else None
    configs = build_analysis_configs(include=["regression"], param_filters=param_filters)
    rrr_configs = [c for c in configs if "rrr" in getattr(c, "model_name", "")]

    if not rrr_configs:
        print("No RRR configs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Sessions:  {[s.session_uid for s in sessions]}")
    print(f"Configs:   {len(rrr_configs)} RRR variants")
    print()

    registries: dict[str, PopulationRegistry] = {}
    n_ok = n_fail = 0

    for session in sessions:
        for cfg in rrr_configs:
            dk = cfg.data_config_name
            if dk not in registries:
                registries[dk] = PopulationRegistry(registry_params=cfg.to_registry_params())
            registry = registries[dk]

            label = f"{session.session_uid} / {cfg.display_name}"
            try:
                result = cfg.process(session, registry)
                print(f"  OK   {label}  r2={result.get('r2', '?'):.4f}")
                n_ok += 1
            except Exception:
                print(f"  FAIL {label}")
                traceback.print_exc()
                n_fail += 1

    print(f"\n{n_ok} passed, {n_fail} failed.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
