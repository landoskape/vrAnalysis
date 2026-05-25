"""Concurrency test: W workers drain a shared N-job queue simultaneously.

Copies N pending jobs from the real DB into a temp SQLite DB, then spawns W
worker subprocesses in dry-run mode against the temp DB. Asserts every job is
claimed exactly once (no double-claims, no orphans).

Safe to run while real workers are active — uses a temp DB copy; the real queue
is never modified.

Usage
-----
    python -m dimensionality_manuscript.scripts.concurrency_test
    python -m dimensionality_manuscript.scripts.concurrency_test --n-jobs 10 --n-workers 4
    python -m dimensionality_manuscript.scripts.concurrency_test --sessions-file sessions.json
"""

import argparse
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from dimensionality_manuscript.pipeline import JobQueue
from dimensionality_manuscript.registry import RegistryPaths


def concurrency_test(
    n_jobs: int,
    n_workers: int,
    db_path: Path,
    sessions_file: Path | None,
) -> int:
    """Returns 0 on pass, 1 on fail."""
    queue = JobQueue(db_path)
    summary = queue.status_summary()
    pending = summary.get("pending", 0)
    print(f"Real queue: {summary}")

    if pending < n_jobs:
        print(f"Only {pending} pending jobs available; need {n_jobs}. Reduce --n-jobs.")
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "concurrency_test.db"

        # Seed temp DB with N pending jobs copied from real queue
        JobQueue(tmp_db)  # creates schema
        src = sqlite3.connect(db_path)
        src.row_factory = sqlite3.Row
        rows = src.execute(
            "SELECT result_uid, session_id, analysis_key, analysis_summary "
            "FROM job_queue WHERE status='pending' ORDER BY created_at LIMIT ?",
            (n_jobs,),
        ).fetchall()
        src.close()

        dst = sqlite3.connect(tmp_db)
        dst.executemany(
            "INSERT INTO job_queue (result_uid, session_id, analysis_key, analysis_summary) "
            "VALUES (?,?,?,?)",
            [(r["result_uid"], r["session_id"], r["analysis_key"], r["analysis_summary"]) for r in rows],
        )
        dst.commit()
        dst.close()

        print(f"Temp DB seeded with {len(rows)} jobs. Spawning {n_workers} workers...\n")

        base_cmd = [
            sys.executable, "-m", "dimensionality_manuscript.scripts.sge_worker",
            "--db-path", str(tmp_db),
            "--dry-run",
            "--max-jobs", str(n_jobs),
        ]
        if sessions_file:
            base_cmd += ["--sessions-file", str(sessions_file)]

        procs = [
            subprocess.Popen(base_cmd + ["--worker-id", f"conctest.{i}"])
            for i in range(n_workers)
        ]
        for i, p in enumerate(procs):
            p.wait()
            rc_str = "ok" if p.returncode == 0 else f"exit {p.returncode}"
            print(f"  Worker conctest.{i}: {rc_str}")

        final = JobQueue(tmp_db).status_summary()
        print(f"\nTemp queue final: {final}")

        done = final.get("done", 0)
        not_done = {k: v for k, v in final.items() if k != "done" and v > 0}

        if done == n_jobs and not not_done:
            print(f"\nPASS: all {n_jobs} jobs claimed and completed exactly once.")
            return 0
        else:
            if done != n_jobs:
                print(f"\nFAIL: expected {n_jobs} done, got {done}.")
            if not_done:
                print(f"FAIL: jobs left in unexpected states: {not_done}")
            return 1


def main():
    parser = argparse.ArgumentParser(description="Concurrency test: W workers drain N-job queue")
    parser.add_argument("--n-jobs", type=int, default=6, help="Jobs to test (default: 6)")
    parser.add_argument("--n-workers", type=int, default=3, help="Concurrent workers (default: 3)")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--sessions-file", type=Path, default=None)
    args = parser.parse_args()

    db_path = args.db_path or RegistryPaths.pipeline_v2_db_path
    sessions_file = args.sessions_file
    if sessions_file is None:
        env_val = os.environ.get("DIM_MANUSCRIPT_SESSIONS_FILE")
        if env_val:
            sessions_file = Path(env_val)

    sys.exit(concurrency_test(args.n_jobs, args.n_workers, db_path, sessions_file))


if __name__ == "__main__":
    main()
