"""Scheduler-independent planning for Myriad worker submissions."""

from pathlib import Path

from dimensionality_manuscript.pipeline import AnalysisPlan, JobQueue, ResultsStore
from dimensionality_manuscript.scripts.run import (
    build_analysis_configs,
    collect_sessions,
    collect_sessions_from_file,
)


def prepare_batch(db_path: Path, analyses=None, sessions_file=None, skip_errors=False, param_filters=None) -> str | None:
    """Populate an independent batch of missing results; return its ID, if any."""
    store = ResultsStore(db_path)
    queue = JobQueue(db_path)
    sessions = collect_sessions_from_file(sessions_file) if sessions_file is not None else collect_sessions()
    analysis_configs = build_analysis_configs(include=analyses, param_filters=param_filters)
    plan = AnalysisPlan(analysis_configs=analysis_configs)
    pending_jobs, n_skipped = plan._collect_jobs(sessions, store, force_remake=False, skip_errors=skip_errors)

    skip_suffix = f" ({n_skipped} skipped — recorded errors)" if n_skipped else ""
    print(f"Sessions:         {len(sessions)}")
    print(f"Analysis configs: {len(analysis_configs)}")
    print(f"Pending jobs:     {len(pending_jobs)}{skip_suffix}")
    print(f"Database:         {db_path}")
    print()
    if pending_jobs:
        plan.print_job_groups(pending_jobs, label="Pending")
        print()
    else:
        print("Nothing to do — all results already computed.")
        print_batch_summary(queue)
        return None

    batch_id = queue.create_batch(analyses)
    n_added = queue.populate(pending_jobs, batch_id)
    print(f"\nBatch:        {batch_id}")
    print(f"Jobs added:   {n_added} | {queue.status_summary(batch_id)}")
    return batch_id


def print_batch_summary(queue: JobQueue) -> None:
    batches = queue.list_batches()
    if batches:
        print("\nExisting batches (newest first):")
        for b in batches[:5]:
            print(f"  {b['batch_id']}  pending={b['pending']} running={b['running']} done={b['done']} failed={b['failed']}")
