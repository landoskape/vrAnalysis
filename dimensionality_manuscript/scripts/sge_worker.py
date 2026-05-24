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
    sessions_file: Path | None = None,
    claim_timeout: int = 60,
    max_jobs: int | None = None,
):
    """Claim and execute jobs until the queue is drained.

    Parameters
    ----------
    worker_id : str
        Unique identifier for this worker (e.g. ``"12345.3"`` from SGE).
    db_path : Path
        Path to the SQLite results database.
    sessions_file : Path or None
        JSON session list from export_sessions.py. Required on MYRIAD.
        If None, falls back to the live vrSessions database.
    claim_timeout : int
        Minutes before a running job is considered stale and reclaimable.
    max_jobs : int or None
        Stop after this many jobs (useful for testing). None = no limit.
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

    print(f"[{worker_id}] Starting. Queue: {queue.status_summary()}")

    while True:
        if max_jobs is not None and (n_done + n_failed) >= max_jobs:
            print(f"[{worker_id}] Reached max_jobs={max_jobs}. Stopping.")
            break

        job_info = queue.claim_next(worker_id, timeout_minutes=claim_timeout)
        if job_info is None:
            print(f"[{worker_id}] Queue empty. Done. ({n_done} succeeded, {n_failed} failed)")
            break

        uid = job_info["result_uid"]
        session_id = job_info["session_id"]
        analysis_key = job_info["analysis_key"]

        session = sessions_by_id.get(session_id)
        config = configs_by_key.get(analysis_key)

        if session is None:
            queue.mark_failed(uid, f"Unknown session_id: {session_id!r}")
            n_failed += 1
            continue

        if config is None:
            queue.mark_failed(uid, f"Unknown analysis_key: {analysis_key!r}")
            n_failed += 1
            continue

        print(f"[{worker_id}] Running: {session_id} | {job_info['analysis_summary']}")

        job = Job(session=session, analysis_config=config)
        registry = _get_registry(config.data_config_name, config)

        try:
            ok = _execute_job(job, registry, store, snapshot_path=None)
            if ok:
                queue.mark_done(uid)
                n_done += 1
            else:
                queue.mark_failed(uid, "Job returned False (see worker log for details)")
                n_failed += 1
        except Exception:
            error_msg = traceback.format_exc()
            print(f"[{worker_id}] FAILED: {session_id} | {error_msg}")
            queue.mark_failed(uid, error_msg)
            n_failed += 1


def main():
    parser = argparse.ArgumentParser(description="SGE worker: drain job queue")
    parser.add_argument("--worker-id", required=True, help="Unique worker ID (e.g. $JOB_ID.$SGE_TASK_ID)")
    parser.add_argument("--db-path", type=Path, default=None, help="Path to results.db (default: RegistryPaths default)")
    parser.add_argument("--sessions-file", type=Path, default=None, help="JSON session list from export_sessions.py")
    parser.add_argument("--claim-timeout", type=int, default=60, help="Minutes before stale running job is reclaimable (default: 60)")
    parser.add_argument("--max-jobs", type=int, default=None, help="Stop after N jobs (for testing)")
    args = parser.parse_args()

    db_path = args.db_path if args.db_path is not None else RegistryPaths.pipeline_v2_db_path

    # Also accept env vars injected by sge_submit.py via qsub -v
    sessions_file = args.sessions_file
    if sessions_file is None:
        env_val = os.environ.get("DIM_MANUSCRIPT_SESSIONS_FILE")
        if env_val:
            sessions_file = Path(env_val)

    run_worker(
        worker_id=args.worker_id,
        db_path=db_path,
        sessions_file=sessions_file,
        claim_timeout=args.claim_timeout,
        max_jobs=args.max_jobs,
    )


if __name__ == "__main__":
    main()
