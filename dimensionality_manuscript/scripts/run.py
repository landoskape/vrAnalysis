"""Run the analysis pipeline for the dimensionality manuscript.

Usage
-----
    python -m dimensionality_manuscript.scripts.run [--dry-run] [--max-jobs N] [--n-jobs N] [--force-remake] [--no-snapshot]

Replicates the measure_cvpca.py workflow using the new pipeline architecture.
"""

import argparse
import json
from pathlib import Path
from vrAnalysis.sessions import B2Session
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript import (
    AnalysisPlan,
    AnalysisConfigBase,
    CVPCAConfig,
    RegressionConfig,
    VectorGainRankConfig,
    SubspaceConfig,
    StimSpaceConfig,
    PopulationConfig,
    ExpMaxConfig,
    LocPredConfig,
    PFPredQualityConfig,
    RRRLatentsToRBFPosConfig,
    ResultsStore,
)

REGISTRY_PATHS = RegistryPaths()


def build_analysis_configs(
    include: list[str] | None = None,
    param_filters: dict | None = None,
) -> list[AnalysisConfigBase]:
    """All valid config variations (Cartesian product minus invalid combos).

    Parameters
    ----------
    include : list of str or None
        Analysis type names to include. None = all.
    param_filters : dict or None
        Fixed param values to filter the grid, e.g. ``{"model_name": "rrr"}``.
        Applied per config class; classes that don't have the specified fields
        are skipped with a warning.
    """
    _mapping: dict[str, type[AnalysisConfigBase]] = {
        "population": PopulationConfig,
        "regression": RegressionConfig,
        "vector_gain_rank": VectorGainRankConfig,
        "cvpca": CVPCAConfig,
        "subspace": SubspaceConfig,
        "stimspace": StimSpaceConfig,
        "expmax": ExpMaxConfig,
        "locpred": LocPredConfig,
        "pfpred_quality": PFPredQualityConfig,
        "rrr_to_rbfpos": RRRLatentsToRBFPosConfig,
    }
    if include is None:
        include = list(_mapping.keys())

    configs = []
    for key in include:
        if key not in _mapping:
            raise ValueError(f"Unknown analysis config key {key!r}. Available: {', '.join(_mapping.keys())}")
        cls = _mapping[key]
        if param_filters:
            try:
                configs.extend(cls.generate_variations_matching(param_filters))
            except ValueError as e:
                print(f"  [skip {key}] {e}")
        else:
            configs.extend(cls.generate_variations())
    return configs


def collect_sessions() -> list[B2Session]:
    """All imaging sessions from the vrSessions database."""
    from vrAnalysis.database import get_database

    sessiondb = get_database("vrSessions")
    return list(sessiondb.iter_sessions(imaging=True))


def collect_sessions_from_file(path: Path) -> list[B2Session]:
    """Load sessions from a JSON file exported by export_sessions.py.

    Use this on systems where the Access database is unavailable (e.g. MYRIAD).

    Parameters
    ----------
    path : Path
        JSON file produced by ``export_sessions.py``.
    """
    records = json.loads(Path(path).read_text())
    return [B2Session.create(r["mouse_name"], r["date"], r["session_id"]) for r in records]


def run(
    analyses: list[str] | None = None,
    force_remake: bool = False,
    snapshot_codebase: bool = True,
    n_jobs: int = 1,
    dry_run: bool = False,
    max_jobs: int | None = None,
    skip_errors: bool = False,
    show_missing: bool = False,
    param_filters: dict | None = None,
):
    """Set up and execute the full analysis plan.

    Parameters
    ----------
    analyses: list of str or None
        Which analysis configs to include. Options: "cvpca", "regression", "stimspace". None = all.
    force_remake : bool
        Recompute even if results already exist in the store.
    snapshot_codebase : bool
        Save a zip snapshot of the repo before running.
    n_jobs : int
        Number of parallel workers. 1 = sequential.
    dry_run : bool
        If True, print what would be done without executing.
    max_jobs : int or None
        Maximum number of analysis jobs to run. None = no limit.
    skip_errors : bool
        Skip (session, config) pairs that already have a recorded error.
    show_missing : bool
        When dry_run is True, also print the session IDs under each config group.
    """
    db_path = REGISTRY_PATHS.pipeline_v2_db_path
    store = ResultsStore(db_path)
    sessions = collect_sessions()
    analysis_configs = build_analysis_configs(include=analyses, param_filters=param_filters)

    analysis_types = list({cfg.display_name for cfg in analysis_configs})
    n_errors = sum(len(store.get_errors(analysis_type=at)) for at in analysis_types)

    print(f"Sessions: {len(sessions)}")
    print(f"Analysis configs: {len(analysis_configs)}")
    print(f"Total combinations: {len(sessions) * len(analysis_configs)}")
    print(f"Store: {db_path}")
    print(f"Current coverage: {store.coverage(sessions, analysis_configs):.1%} | Errors recorded: {n_errors}")
    print()

    plan = AnalysisPlan(analysis_configs=analysis_configs)
    plan.analyze(
        sessions,
        store,
        n_jobs=n_jobs,
        force_remake=force_remake,
        snapshot_codebase=snapshot_codebase,
        dry_run=dry_run,
        max_jobs=max_jobs,
        skip_errors=skip_errors,
        show_sessions=show_missing,
    )

    if not dry_run:
        print(f"\nFinal coverage: {store.coverage(sessions, analysis_configs):.1%}")


def main():
    parser = argparse.ArgumentParser(description="Run dimensionality manuscript analysis pipeline")
    parser.add_argument("--analyses", nargs="+", help="Which analysis configs to include. Options: 'cvpca', 'regression', 'stimspace'. Default: all.")
    parser.add_argument("--force-remake", action="store_true", help="Recompute all results")
    parser.add_argument("--no-snapshot", action="store_true", help="Skip codebase snapshot")
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of parallel workers (default: 8)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--max-jobs", type=int, default=None, help="Maximum number of analysis jobs to run")
    parser.add_argument("--skip-errors", action="store_true", help="Skip (session, config) pairs that already have a recorded error")
    parser.add_argument("--show-missing", action="store_true", help="With --dry-run, print session IDs under each config group")
    parser.add_argument(
        "--param-filters",
        nargs="+",
        metavar="KEY=VALUE",
        help="Filter config grid by fixed param values, e.g. --param-filters model_name=rrr spks_type=oasis",
    )
    args = parser.parse_args()

    run(
        analyses=args.analyses,
        force_remake=args.force_remake,
        snapshot_codebase=not args.no_snapshot,
        n_jobs=args.n_jobs,
        dry_run=args.dry_run,
        max_jobs=args.max_jobs,
        skip_errors=args.skip_errors,
        show_missing=args.show_missing,
        param_filters=_parse_param_filters(args.param_filters),
    )


def _parse_param_filters(raw: list[str] | None) -> dict | None:
    if not raw:
        return None
    out = {}
    for token in raw:
        if "=" not in token:
            raise ValueError(f"--param-filters expects KEY=VALUE pairs, got {token!r}")
        k, _, v = token.partition("=")
        out[k.strip()] = v.strip()
    return out


if __name__ == "__main__":
    main()
