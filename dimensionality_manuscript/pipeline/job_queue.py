"""SQLite-backed job queue with atomic claiming for SGE distributed execution."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .plan import Job
from .store import result_uid as _result_uid

_BUSY_TIMEOUT_MS = 30_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_queue (
    result_uid   TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    analysis_key TEXT NOT NULL,
    analysis_summary TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    claimed_by   TEXT,
    claimed_at   TIMESTAMP,
    completed_at TIMESTAMP,
    error        TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_STATUS_VALUES = frozenset({"pending", "running", "done", "failed", "test-block"})


class JobQueue:
    """Persistent job queue stored in a SQLite table alongside the results store.

    Multiple SGE workers can safely call :meth:`claim_next` concurrently.
    SQLite's ``BEGIN IMMEDIATE`` serialises the claim transaction so each
    pending job is claimed by exactly one worker.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database (same file as ``ResultsStore``).
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self):
        """Yield an autocommit connection. Caller manages transactions explicitly."""
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None  # autocommit — we issue BEGIN/COMMIT manually
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        try:
            yield conn
        finally:
            conn.close()

    def populate(self, jobs: list[Job]) -> int:
        """Insert pending jobs into the queue, skipping duplicates.

        Parameters
        ----------
        jobs : list[Job]
            Jobs to enqueue. Already-queued ``result_uid`` values are ignored
            (``INSERT OR IGNORE``).

        Returns
        -------
        int
            Number of new rows inserted.
        """
        rows = [
            (
                _result_uid(job.session.session_uid, job.analysis_config.key()),
                job.session.session_uid,
                job.analysis_config.key(),
                job.analysis_config.summary(),
            )
            for job in jobs
        ]
        with self._connect() as conn:
            conn.execute("BEGIN")
            before = conn.execute("SELECT COUNT(*) FROM job_queue").fetchone()[0]
            conn.executemany(
                "INSERT OR IGNORE INTO job_queue (result_uid, session_id, analysis_key, analysis_summary) VALUES (?,?,?,?)",
                rows,
            )
            after = conn.execute("SELECT COUNT(*) FROM job_queue").fetchone()[0]
            conn.execute("COMMIT")
        return after - before

    def claim_next(self, worker_id: str, timeout_minutes: int = 60) -> dict | None:
        """Atomically claim the next available job.

        A job is claimable if its ``status`` is ``'pending'``, or if it is
        ``'running'`` but its ``claimed_at`` timestamp is older than
        ``timeout_minutes`` (stale — the previous worker likely crashed).

        Parameters
        ----------
        worker_id : str
            Unique identifier for this worker (e.g. ``"$JOB_ID.$SGE_TASK_ID"``).
        timeout_minutes : int
            Minutes after which a running job is considered stale and reclaimable.

        Returns
        -------
        dict or None
            Row dict with keys ``result_uid``, ``session_id``, ``analysis_key``,
            ``analysis_summary``, or ``None`` if no claimable job exists.
        """
        stale_interval = f"-{timeout_minutes} minutes"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT result_uid, session_id, analysis_key, analysis_summary
                FROM job_queue
                WHERE status = 'pending'
                   OR (status = 'running' AND claimed_at < datetime('now', ?))
                ORDER BY created_at
                LIMIT 1
                """,
                (stale_interval,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            conn.execute(
                "UPDATE job_queue SET status='running', claimed_by=?, claimed_at=datetime('now') WHERE result_uid=?",
                (worker_id, row["result_uid"]),
            )
            conn.execute("COMMIT")
        return dict(row)

    def mark_done(self, uid: str) -> None:
        """Mark a job as successfully completed."""
        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute(
                "UPDATE job_queue SET status='done', completed_at=datetime('now') WHERE result_uid=?",
                (uid,),
            )
            conn.execute("COMMIT")

    def mark_failed(self, uid: str, error: str) -> None:
        """Mark a job as failed and store the error/traceback.

        Parameters
        ----------
        uid : str
            ``result_uid`` of the failed job.
        error : str
            Error message or formatted traceback.
        """
        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute(
                "UPDATE job_queue SET status='failed', completed_at=datetime('now'), error=? WHERE result_uid=?",
                (error[:4096], uid),
            )
            conn.execute("COMMIT")

    def reset_failed(self, result_uids: list[str] | None = None) -> int:
        """Re-queue failed jobs so workers will retry them.

        Parameters
        ----------
        result_uids : list[str] or None
            Specific UIDs to reset. If ``None``, all failed jobs are reset.

        Returns
        -------
        int
            Number of jobs reset to pending.
        """
        with self._connect() as conn:
            conn.execute("BEGIN")
            if result_uids is None:
                cur = conn.execute(
                    "UPDATE job_queue SET status='pending', claimed_by=NULL, claimed_at=NULL, error=NULL WHERE status='failed'"
                )
            else:
                placeholders = ",".join("?" for _ in result_uids)
                cur = conn.execute(
                    f"UPDATE job_queue SET status='pending', claimed_by=NULL, claimed_at=NULL, error=NULL WHERE status='failed' AND result_uid IN ({placeholders})",
                    result_uids,
                )
            n = cur.rowcount
            conn.execute("COMMIT")
        return n

    def status_summary(self) -> dict[str, int]:
        """Return job counts grouped by status.

        Returns
        -------
        dict[str, int]
            E.g. ``{"pending": 120, "running": 4, "done": 56, "failed": 2}``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM job_queue GROUP BY status"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def pending_count(self) -> int:
        """Number of pending (unclaimed) jobs."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM job_queue WHERE status='pending'"
            ).fetchone()[0]

    def claim_for_test(self, n: int) -> list[dict]:
        """Atomically mark up to n pending jobs as 'test-block' and return them.

        Real workers skip 'test-block' jobs (``claim_next`` only claims
        ``'pending'`` or stale ``'running'``). Call :meth:`release_test_blocks`
        when done to restore them to ``'pending'``.

        Parameters
        ----------
        n : int
            Maximum number of jobs to claim.

        Returns
        -------
        list[dict]
            Claimed rows with keys ``result_uid``, ``session_id``,
            ``analysis_key``, ``analysis_summary``.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT result_uid, session_id, analysis_key, analysis_summary "
                "FROM job_queue WHERE status='pending' ORDER BY created_at LIMIT ?",
                (n,),
            ).fetchall()
            if rows:
                uids = [r["result_uid"] for r in rows]
                placeholders = ",".join("?" for _ in uids)
                conn.execute(
                    f"UPDATE job_queue SET status='test-block' WHERE result_uid IN ({placeholders})",
                    uids,
                )
            conn.execute("COMMIT")
        return [dict(r) for r in rows]

    def release_test_blocks(self) -> int:
        """Reset all 'test-block' jobs back to 'pending'.

        Returns
        -------
        int
            Number of jobs released.
        """
        with self._connect() as conn:
            conn.execute("BEGIN")
            cur = conn.execute(
                "UPDATE job_queue SET status='pending' WHERE status='test-block'"
            )
            n = cur.rowcount
            conn.execute("COMMIT")
        return n
