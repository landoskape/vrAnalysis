"""SGE worker: drain the job queue until empty.

Each SGE task runs this script. Workers loop — claim a job, run it, mark done —
until no pending jobs remain, then exit. Multiple workers run concurrently;
SQLite BEGIN IMMEDIATE ensures each job is claimed by exactly one worker.

Usage
-----
    python -m dimensionality_manuscript.scripts.sge_worker \
        --worker-id "$JOB_ID.$SGE_TASK_ID" \
        --db-path /path/to/results.db \
        [--sessions-file /path/to/sessions.json] \
        [--claim-timeout 60] \
        [--max-jobs N]
"""

import argparse
import os
import traceback
from pathlib import Path

from dimensionality_manuscript.pipeline import JobQueue, ResultsStore
from dimensionality_manuscript.pipeline.plan import _execute_job, Job
from dimensionality_manuscript.registry import PopulationRegistry, RegistryPaths
from dimensionality_manuscript.scripts.run import (
    build_analysis_configs,
    collect_sessions,
    collect_sessions_from_file,
)


def run_worker(
    worker_id: str,
    db_path: Path,
    batch_id: str,
    sessions_file: Path | None = None,
    claim_timeout: int = 60,
    max_jobs: int | None = None,
    dry_run: bool = False,
):
    """Claim and execute jobs until the batch queue is drained.

    Parameters
    ----------
    worker_id : str
        Unique identifier for this worker (e.g. ``"12345.3"`` from SGE).
    db_path : Path
        Path to the SQLite results database.
    batch_id : str
        Batch to drain. Workers only claim jobs from their own batch.
    sessions_file : Path or None
        JSON session list from export_sessions.py. Required on MYRIAD.
        If None, falls back to the live vrSessions database.
    claim_timeout : int
        Minutes before a running job is considered stale and reclaimable.
    max_jobs : int or None
        Stop after this many jobs (useful for testing). None = no limit.
    dry_run : bool
        If True, validate session+config but skip analysis and store.put().
        Marks jobs done immediately. Used by concurrency_test.py.
    """
    db_path = Path(db_path)
    queue = JobQueue(db_path)
    store = ResultsStore(db_path)

    if sessions_file is not None:
        sessions = collect_sessions_from_file(sessions_file)
    else:
        sessions = collect_sessions()

    sessions_by_id = {s.session_uid: s for s in sessions}
    configs_by_key = {c.key(): c for c in build_analysis_configs()}

    registries: dict[str, PopulationRegistry] = {}

    def _get_registry(data_config_name: str, analysis_config) -> PopulationRegistry:
        if data_config_name not in registries:
            registries[data_config_name] = PopulationRegistry(
                registry_params=analysis_config.data_config.to_registry_params()
            )
        return registries[data_config_name]

    n_done = 0
    n_failed = 0

    print(f"[{worker_id}] Starting. Batch: {batch_id}. Queue: {queue.status_summary(batch_id)}")

    while True:
        if max_jobs is not None and (n_done + n_failed) >= max_jobs:
            print(f"[{worker_id}] Reached max_jobs={max_jobs}. Stopping.")
            break

        job_info = queue.claim_next(worker_id, batch_id, timeout_minutes=claim_timeout)
        if job_info is None:
            print(f"[{worker_id}] Batch {batch_id} empty. Done. ({n_done} succeeded, {n_failed} failed)")
            break

        uid = job_info["result_uid"]
        session_id = job_info["session_id"]
        analysis_key = job_info["analysis_key"]

        session = sessions_by_id.get(session_id)
        config = configs_by_key.get(analysis_key)

        if session is None:
            queue.mark_failed(uid, batch_id, f"Unknown session_id: {session_id!r}")
            n_failed += 1
            continue

        if config is None:
            queue.mark_failed(uid, batch_id, f"Unknown analysis_key: {analysis_key!r}")
            n_failed += 1
            continue

        print(f"[{worker_id}] {'(dry-run) ' if dry_run else ''}Running: {session_id} | {job_info['analysis_summary']}")

        if dry_run:
            queue.mark_done(uid, batch_id)
            n_done += 1
            continue

        job = Job(session=session, analysis_config=config)
        registry = _get_registry(config.data_config_name, config)

        try:
            ok = _execute_job(job, registry, store, snapshot_path=None)
            if ok:
                queue.mark_done(uid, batch_id)
                n_done += 1
            else:
                queue.mark_failed(uid, batch_id, "Job returned False (see worker log for details)")
                n_failed += 1
        except Exception:
            error_msg = traceback.format_exc()
            print(f"[{worker_id}] FAILED: {session_id} | {error_msg}")
            queue.mark_failed(uid, batch_id, error_msg)
            n_failed += 1


def main():
    parser = argparse.ArgumentParser(description="SGE worker: drain job queue batch")
    parser.add_argument("--worker-id", required=True, help="Unique worker ID (e.g. $JOB_ID.$SGE_TASK_ID)")
    parser.add_argument("--batch-id", type=str, default=None, help="Batch to drain (injected via DIM_MANUSCRIPT_BATCH_ID)")
    parser.add_argument("--db-path", type=Path, default=None, help="Path to results.db (default: RegistryPaths default)")
    parser.add_argument("--sessions-file", type=Path, default=None, help="JSON session list from export_sessions.py")
    parser.add_argument("--claim-timeout", type=int, default=60, help="Minutes before stale running job is reclaimable (default: 60)")
    parser.add_argument("--max-jobs", type=int, default=None, help="Stop after N jobs (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Validate session+config but skip analysis; marks jobs done immediately")
    args = parser.parse_args()

    db_path = args.db_path if args.db_path is not None else RegistryPaths.pipeline_v2_db_path

    # Accept env vars injected by sge_submit.py via qsub -v
    batch_id = args.batch_id or os.environ.get("DIM_MANUSCRIPT_BATCH_ID")
    if not batch_id:
        raise SystemExit("ERROR: --batch-id or DIM_MANUSCRIPT_BATCH_ID required")

    sessions_file = args.sessions_file
    if sessions_file is None:
        env_val = os.environ.get("DIM_MANUSCRIPT_SESSIONS_FILE")
        if env_val:
            sessions_file = Path(env_val)

    run_worker(
        worker_id=args.worker_id,
        db_path=db_path,
        batch_id=batch_id,
        sessions_file=sessions_file,
        claim_timeout=args.claim_timeout,
        max_jobs=args.max_jobs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
