"""Smoke test: claim N jobs as test-block, verify session+config resolve, report, release.

Workers never see test-block jobs (claim_next only claims 'pending' / stale 'running'),
so this is safe to run while the queue is live.

Usage
-----
    python -m dimensionality_manuscript.scripts.smoke_test --n-jobs 2
    python -m dimensionality_manuscript.scripts.smoke_test --n-jobs 5 --sessions-file sessions.json
"""

import argparse
import os
import sys
from pathlib import Path

from dimensionality_manuscript.pipeline import JobQueue
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.scripts.run import (
    build_analysis_configs,
    collect_sessions,
    collect_sessions_from_file,
)


def smoke_test(n: int, db_path: Path, sessions_file: Path | None) -> int:
    """Claim n jobs, verify they resolve, release. Returns number of failures."""
    queue = JobQueue(db_path)

    if sessions_file is not None:
        sessions = collect_sessions_from_file(sessions_file)
    else:
        sessions = collect_sessions()

    sessions_by_id = {s.session_uid: s for s in sessions}
    configs_by_key = {c.key(): c for c in build_analysis_configs()}

    print(f"Sessions loaded: {len(sessions_by_id)}")
    print(f"Analysis configs: {len(configs_by_key)}")
    print(f"Queue: {queue.status_summary()}\n")

    jobs = queue.claim_for_test(n)
    if not jobs:
        print("No pending jobs found — nothing to test.")
        return 0

    print(f"Claimed {len(jobs)} jobs as test-block:\n")
    n_ok = 0
    n_bad = 0
    try:
        for job in jobs:
            session = sessions_by_id.get(job["session_id"])
            config = configs_by_key.get(job["analysis_key"])
            ok = session is not None and config is not None
            tag = "OK  " if ok else "FAIL"
            print(f"  [{tag}] {job['session_id']} | {job['analysis_summary']}")
            if session is None:
                print(f"         ^ unknown session_id: {job['session_id']!r}")
            if config is None:
                print(f"         ^ unknown analysis_key: {job['analysis_key']!r}")
            if ok:
                n_ok += 1
            else:
                n_bad += 1
    finally:
        released = queue.release_test_blocks()
        print(f"\nReleased {released} test-block jobs → pending.")

    print(f"Result: {n_ok} OK, {n_bad} FAIL")
    return n_bad


def main():
    parser = argparse.ArgumentParser(description="Smoke test: verify N queued jobs resolve")
    parser.add_argument("--n-jobs", type=int, default=2, help="Number of jobs to test (default: 2)")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--sessions-file", type=Path, default=None)
    args = parser.parse_args()

    db_path = args.db_path or RegistryPaths.pipeline_v2_db_path
    sessions_file = args.sessions_file
    if sessions_file is None:
        env_val = os.environ.get("DIM_MANUSCRIPT_SESSIONS_FILE")
        if env_val:
            sessions_file = Path(env_val)

    sys.exit(smoke_test(args.n_jobs, db_path, sessions_file))


if __name__ == "__main__":
    main()
