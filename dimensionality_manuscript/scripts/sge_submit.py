"""Populate the SGE job queue and submit worker array jobs to MYRIAD.

Usage
-----
    python -m dimensionality_manuscript.scripts.sge_submit [options]

Run on the MYRIAD login node after transferring data. This script:
1. Collects pending (session, config) pairs not yet in the results store.
2. Creates a new batch and inserts those jobs into the job_queue table.
3. Submits an SGE array job whose workers drain that batch concurrently.

Each call creates an independent batch so re-submitting never disturbs active
workers.  Workers are bound to their batch via DIM_MANUSCRIPT_BATCH_ID.

Use --dry-run to populate the batch without submitting.
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
_REPO_ROOT = Path(__file__).resolve().parents[2]


def submit(
    analyses: list[str] | None = None,
    sessions_file: Path | None = None,
    n_workers: int = 16,
    walltime: str = "8:00:00",
    mem: str = "16G",
    dry_run: bool = False,
) -> str | None:
    """Create a batch, populate the queue, and optionally submit an SGE array job.

    Each call produces an independent batch.  Re-submitting while workers are
    active creates a new batch and leaves the active one untouched.

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
        If True, populate the batch but do not submit via qsub.
        Useful for inspecting what would run before committing.

    Returns
    -------
    str or None
        The batch_id created, or None if there was nothing to do.
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
    print()

    if pending_jobs:
        plan.print_job_groups(pending_jobs, label="Pending")
        print()

    if not pending_jobs:
        print("Nothing to do — all results already computed.")
        _print_batch_summary(queue)
        return None

    batch_id = queue.create_batch(analyses)
    n_added = queue.populate(pending_jobs, batch_id)
    summary = queue.status_summary(batch_id)
    print(f"\nBatch:        {batch_id}")
    print(f"Jobs added:   {n_added} | {summary}")

    if dry_run:
        print("\n[dry-run] Batch populated. Run smoke_test to validate, then re-run without --dry-run to submit.")
        _print_queue_preview(n_workers, walltime, mem, db_path, sessions_file, batch_id)
        return batch_id

    qsub_cmd = _build_qsub_command(n_workers, walltime, mem, db_path, sessions_file, batch_id)
    print(f"\nSubmitting: {' '.join(qsub_cmd)}")
    result = subprocess.run(qsub_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"qsub failed:\n{result.stderr}")
        raise SystemExit(1)
    print(result.stdout.strip())
    return batch_id


def _build_qsub_command(
    n_workers: int,
    walltime: str,
    mem: str,
    db_path: Path,
    sessions_file: Path | None,
    batch_id: str,
) -> list[str]:
    env_vars = (
        f"DIM_MANUSCRIPT_DB_PATH={db_path},"
        f"DIM_MANUSCRIPT_BATCH_ID={batch_id},"
        f"DIM_MANUSCRIPT_REPO={_REPO_ROOT}"
    )
    if sessions_file is not None:
        env_vars += f",DIM_MANUSCRIPT_SESSIONS_FILE={Path(sessions_file).resolve()}"
    return [
        "qsub",
        "-t",
        f"1-{n_workers}",
        "-l",
        f"h_rt={walltime}",
        "-l",
        f"mem={mem}",
        "-v",
        env_vars,
        str(_WORKER_SCRIPT),
    ]


def _print_queue_preview(n_workers, walltime, mem, db_path, sessions_file, batch_id):
    qsub_cmd = _build_qsub_command(n_workers, walltime, mem, db_path, sessions_file, batch_id)
    print(f"qsub command: {' '.join(qsub_cmd)}")


def _print_batch_summary(queue: JobQueue) -> None:
    batches = queue.list_batches()
    if batches:
        print("\nExisting batches (newest first):")
        for b in batches[:5]:
            print(f"  {b['batch_id']}  pending={b['pending']} running={b['running']} done={b['done']} failed={b['failed']}")


def main():
    parser = argparse.ArgumentParser(description="Queue SGE jobs and submit to MYRIAD")
    parser.add_argument("--analyses", nargs="+", help="Analysis types to queue (default: all)")
    parser.add_argument("--sessions-file", type=Path, default=None, help="JSON session list from export_sessions.py (required on MYRIAD)")
    parser.add_argument("--n-workers", type=int, default=16, help="Number of parallel SGE worker slots (default: 16)")
    parser.add_argument("--walltime", default="8:00:00", help="SGE wall-clock limit, e.g. 8:00:00 (default: 8:00:00)")
    parser.add_argument("--mem", default="16G", help="SGE memory per slot, e.g. 16G (default: 16G)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Populate batch but do not submit via qsub")
    args = parser.parse_args()

    submit(
        analyses=args.analyses,
        sessions_file=args.sessions_file,
        n_workers=args.n_workers,
        walltime=args.walltime,
        mem=args.mem,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
