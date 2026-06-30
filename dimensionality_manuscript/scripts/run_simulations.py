"""Run the simulation sweep pipeline.

Usage
-----
    python -m dimensionality_manuscript.scripts.run_simulations [options]

Options
-------
--analyses          Which sweep(s) to run. Choices:
                      stim_full_sweep
                      placefield_thresholded_sweep
                      placefield_smooth_sweep
                      placefield_tilbury_sweep
                    Default: all four.
--n-jobs            Parallel workers (default: 4).
--dry-run / -n      Print jobs without executing.
--force-remake      Recompute already-stored results.
--max-jobs          Cap total jobs run.
"""

import argparse

from dimensionality_manuscript.configs.simulation_sweep import (
    SIMULATION_SESSION,
    SmoothGPSweepConfig,
    StimFullSweepConfig,
    ThresholdedGPSweepConfig,
    TilburySweepConfig,
)
from dimensionality_manuscript.pipeline.base import AnalysisConfigBase
from dimensionality_manuscript.pipeline.plan import AnalysisPlan
from dimensionality_manuscript.pipeline.store import ResultsStore
from dimensionality_manuscript.registry import RegistryPaths

_MAPPING: dict[str, type[AnalysisConfigBase]] = {
    "stim_full_sweep": StimFullSweepConfig,
    "placefield_thresholded_sweep": ThresholdedGPSweepConfig,
    "placefield_smooth_sweep": SmoothGPSweepConfig,
    "placefield_tilbury_sweep": TilburySweepConfig,
}


def build_configs(include: list[str] | None = None) -> list[AnalysisConfigBase]:
    keys = include if include is not None else list(_MAPPING.keys())
    configs: list[AnalysisConfigBase] = []
    for key in keys:
        if key not in _MAPPING:
            raise ValueError(f"Unknown sweep {key!r}. Available: {', '.join(_MAPPING)}")
        configs.extend(_MAPPING[key].generate_variations())
    return configs


def run(
    analyses: list[str] | None = None,
    n_jobs: int = 4,
    dry_run: bool = False,
    force_remake: bool = False,
    max_jobs: int | None = None,
    skip_errors: bool = False,
):
    db_path = RegistryPaths().pipeline_v2_db_path
    store = ResultsStore(db_path)
    configs = build_configs(include=analyses)

    print(f"Sweep configs: {len(configs)}")
    print(f"Store: {db_path}")
    print(f"Coverage: {store.coverage([SIMULATION_SESSION], configs):.1%}")
    print()

    plan = AnalysisPlan(analysis_configs=configs)
    plan.analyze(
        [SIMULATION_SESSION],
        store,
        n_jobs=n_jobs,
        force_remake=force_remake,
        snapshot_codebase=False,
        dry_run=dry_run,
        max_jobs=max_jobs,
        skip_errors=skip_errors,
    )


def main():
    parser = argparse.ArgumentParser(description="Run simulation sweep pipeline")
    parser.add_argument("--analyses", nargs="+", choices=list(_MAPPING), help="Which sweeps to run. Default: all.")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--force-remake", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args()
    run(
        analyses=args.analyses,
        n_jobs=args.n_jobs,
        dry_run=args.dry_run,
        force_remake=args.force_remake,
        max_jobs=args.max_jobs,
        skip_errors=args.skip_errors,
    )


if __name__ == "__main__":
    main()
