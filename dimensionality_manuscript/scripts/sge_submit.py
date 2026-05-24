"""Populate the SGE job queue and submit worker array jobs to MYRIAD.

Usage
-----
    python -m dimensionality_manuscript.scripts.sge_submit [options]

Run on the MYRIAD login node after transferring data. This script:
1. Collects pending (session, config) pairs not yet in the results store.
2. Inserts them into the job_queue table in the results database.
3. Submits an SGE array job whose workers drain the queue concurrently.

Use --dry-run to inspect the queue without submitting.
"""

import argparse
import subprocess
from pathlib import Path

from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.pipeline import AnalysisPlan, JobQueue, ResultsStore
from dimensionality_manuscript.scripts.run import (
    build_analysis_configs,
    collect_sessions,
    collect_sessions_from_file,
)

REGISTRY_PATHS = RegistryPaths()

_WORKER_SCRIPT = Path(__file__).parent / "myriad_worker.sh"


def submit(
    analyses: list[str] | None = None,
    sessions_file: Path | None = None,
    n_workers: int = 16,
    walltime: str = "8:00:00",
    mem: str = "16G",
    dry_run: bool = False,
    force_repopulate: bool = False,
):
    """Populate queue and optionally submit SGE array job.

    Parameters
    ----------
    analyses : list[str] or None
        Subset of analysis types to queue. None = all.
    sessions_file : Path or None
        JSON file from export_sessions.py. Required on MYRIAD (no Access DB).
        If None, falls back to the live vrSessions database (local use only).
    n_workers : int
        Number of parallel SGE worker slots (``-t 1-N``).
    walltime : str
        SGE wall-clock time limit (``h_rt``), e.g. ``"8:00:00"``.
    mem : str
        SGE memory per slot (``mem``), e.g. ``"16G"``.
    dry_run : bool
        If True, show what would be queued/submitted without doing it.
    force_repopulate : bool
        If True, reset all failed jobs to pending before populating.
    """
    db_path = REGISTRY_PATHS.pipeline_v2_db_path
    store = ResultsStore(db_path)
    queue = JobQueue(db_path)

    if sessions_file is not None:
        sessions = collect_sessions_from_file(sessions_file)
    else:
        sessions = collect_sessions()

    analysis_configs = build_analysis_configs(include=analyses)
    plan = AnalysisPlan(analysis_configs=analysis_configs)
    pending_jobs = plan._collect_jobs(sessions, store, force_remake=False)

    print(f"Sessions:         {len(sessions)}")
    print(f"Analysis configs: {len(analysis_configs)}")
    print(f"Pending jobs:     {len(pending_jobs)}")
    print(f"Database:         {db_path}")

    if not pending_jobs:
        print("Nothing to do — all results already computed.")
        return

    if dry_run:
        print("\n[dry-run] Would add jobs to queue and submit:")
        _print_queue_preview(pending_jobs, n_workers, walltime, mem, db_path, sessions_file)
        return

    if force_repopulate:
        n_reset = queue.reset_failed()
        if n_reset:
            print(f"Reset {n_reset} failed jobs to pending.")

    n_added = queue.populate(pending_jobs)
    summary = queue.status_summary()
    print(f"\nQueue updated: +{n_added} new | {summary}")

    qsub_cmd = _build_qsub_command(n_workers, walltime, mem, db_path, sessions_file)
    print(f"\nSubmitting: {' '.join(qsub_cmd)}")
    result = subprocess.run(qsub_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"qsub failed:\n{result.stderr}")
        raise SystemExit(1)
    print(result.stdout.strip())


def _build_qsub_command(
    n_workers: int,
    walltime: str,
    mem: str,
    db_path: Path,
    sessions_file: Path | None,
) -> list[str]:
    env_vars = f"DIM_MANUSCRIPT_DB_PATH={db_path}"
    if sessions_file is not None:
        env_vars += f",DIM_MANUSCRIPT_SESSIONS_FILE={Path(sessions_file).resolve()}"
    return [
        "qsub",
        "-t", f"1-{n_workers}",
        "-l", f"h_rt={walltime}",
        "-l", f"mem={mem}",
        "-v", env_vars,
        str(_WORKER_SCRIPT),
    ]


def _print_queue_preview(pending_jobs, n_workers, walltime, mem, db_path, sessions_file):
    from collections import Counter
    type_counts = Counter(j.analysis_config.display_name for j in pending_jobs)
    for analysis_type, count in sorted(type_counts.items()):
        print(f"  {analysis_type}: {count} jobs")
    qsub_cmd = _build_qsub_command(n_workers, walltime, mem, db_path, sessions_file)
    print(f"\nqsub command: {' '.join(qsub_cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Queue SGE jobs and submit to MYRIAD")
    parser.add_argument("--analyses", nargs="+", help="Analysis types to queue (default: all)")
    parser.add_argument("--sessions-file", type=Path, default=None, help="JSON session list from export_sessions.py (required on MYRIAD)")
    parser.add_argument("--n-workers", type=int, default=16, help="Number of parallel SGE worker slots (default: 16)")
    parser.add_argument("--walltime", default="8:00:00", help="SGE wall-clock limit, e.g. 8:00:00 (default: 8:00:00)")
    parser.add_argument("--mem", default="16G", help="SGE memory per slot, e.g. 16G (default: 16G)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be done without submitting")
    parser.add_argument("--force-repopulate", action="store_true", help="Reset failed jobs to pending before populating")
    args = parser.parse_args()

    submit(
        analyses=args.analyses,
        sessions_file=args.sessions_file,
        n_workers=args.n_workers,
        walltime=args.walltime,
        mem=args.mem,
        dry_run=args.dry_run,
        force_repopulate=args.force_repopulate,
    )


if __name__ == "__main__":
    main()
