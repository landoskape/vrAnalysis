"""SQLite-backed job queue with atomic claiming for SGE distributed execution."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .plan import Job
from .store import result_uid as _result_uid

_BUSY_TIMEOUT_MS = 30_000

_SCHEMA_BATCHES = """
CREATE TABLE IF NOT EXISTS job_batches (
    batch_id   TEXT PRIMARY KEY,
    analyses   TEXT,
    n_jobs     INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_SCHEMA_QUEUE = """
CREATE TABLE IF NOT EXISTS job_queue (
    result_uid       TEXT NOT NULL,
    batch_id         TEXT NOT NULL REFERENCES job_batches(batch_id),
    session_id       TEXT NOT NULL,
    analysis_key     TEXT NOT NULL,
    analysis_summary TEXT,
    status           TEXT NOT NULL DEFAULT 'pending',
    claimed_by       TEXT,
    claimed_at       TIMESTAMP,
    completed_at     TIMESTAMP,
    error            TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (result_uid, batch_id)
)
"""

_SCHEMA_INDEX = """
CREATE INDEX IF NOT EXISTS idx_job_queue_batch_status
ON job_queue(batch_id, status, created_at)
"""

_STATUS_VALUES = frozenset({"pending", "running", "done", "failed", "test-block"})


class JobQueue:
    """Persistent job queue stored in a SQLite table alongside the results store.

    Each :meth:`create_batch` call produces an isolated batch.  Workers are
    bound to a specific batch and only claim jobs from it.  Multiple SGE workers
    can safely call :meth:`claim_next` concurrently — SQLite ``BEGIN IMMEDIATE``
    serialises the claim transaction so each pending job is claimed by exactly
    one worker.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database (same file as ``ResultsStore``).
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        with self._connect() as conn:
            self._migrate(conn)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None  # autocommit — we issue BEGIN/COMMIT manually
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        try:
            yield conn
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        # Detect old schema (no batch_id column) and drop to recreate.
        # The queue is ephemeral — results.db is the source of truth.
        try:
            conn.execute("SELECT batch_id FROM job_queue LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("DROP TABLE IF EXISTS job_queue")
            conn.execute("DROP TABLE IF EXISTS job_batches")
        conn.execute(_SCHEMA_BATCHES)
        conn.execute(_SCHEMA_QUEUE)
        conn.execute(_SCHEMA_INDEX)

    # ── Batch management ──────────────────────────────────────────────────────

    def create_batch(self, analyses: list[str] | None = None) -> str:
        """Create a new batch and return its ID.

        Parameters
        ----------
        analyses : list[str] or None
            Analysis types included in this batch (for display only). ``None``
            means all analyses.

        Returns
        -------
        str
            Unique batch identifier (``"YYYYMMDD_HHMMSS_<8hex>"``).
        """
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        analyses_json = json.dumps(sorted(analyses)) if analyses else None
        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO job_batches (batch_id, analyses) VALUES (?, ?)",
                (batch_id, analyses_json),
            )
            conn.execute("COMMIT")
        return batch_id

    def list_batches(self) -> list[dict]:
        """Return all batches with per-status job counts, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT b.batch_id, b.analyses, b.n_jobs, b.created_at,
                       SUM(CASE WHEN q.status='pending'    THEN 1 ELSE 0 END) AS pending,
                       SUM(CASE WHEN q.status='running'    THEN 1 ELSE 0 END) AS running,
                       SUM(CASE WHEN q.status='done'       THEN 1 ELSE 0 END) AS done,
                       SUM(CASE WHEN q.status='failed'     THEN 1 ELSE 0 END) AS failed,
                       SUM(CASE WHEN q.status='test-block' THEN 1 ELSE 0 END) AS test_block
                FROM job_batches b
                LEFT JOIN job_queue q ON q.batch_id = b.batch_id
                GROUP BY b.batch_id
                ORDER BY b.created_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> tuple[int, int]:
        """Delete all rows from ``job_queue`` and ``job_batches``.

        Empties both planning tables (the tables themselves are kept). Use to
        reset the queue, which otherwise accumulates a batch on every
        ``sge_submit`` run. The results store is unaffected.

        Returns
        -------
        tuple[int, int]
            ``(n_jobs, n_batches)`` — number of rows deleted from
            ``job_queue`` and ``job_batches`` respectively.
        """
        with self._connect() as conn:
            conn.execute("BEGIN")
            (n_jobs,) = conn.execute("SELECT COUNT(*) FROM job_queue").fetchone()
            (n_batches,) = conn.execute("SELECT COUNT(*) FROM job_batches").fetchone()
            conn.execute("DELETE FROM job_queue")
            conn.execute("DELETE FROM job_batches")
            conn.execute("COMMIT")
        return n_jobs, n_batches

    # ── Queue population ──────────────────────────────────────────────────────

    def populate(self, jobs: list[Job], batch_id: str) -> int:
        """Insert pending jobs into a batch, skipping duplicates.

        Parameters
        ----------
        jobs : list[Job]
            Jobs to enqueue.
        batch_id : str
            Batch to insert into (must exist — call :meth:`create_batch` first).

        Returns
        -------
        int
            Number of new rows inserted.
        """
        rows = [
            (
                _result_uid(job.session.session_uid, job.analysis_config.key()),
                batch_id,
                job.session.session_uid,
                job.analysis_config.key(),
                job.analysis_config.summary(),
            )
            for job in jobs
        ]
        with self._connect() as conn:
            conn.execute("BEGIN")
            before = conn.execute(
                "SELECT COUNT(*) FROM job_queue WHERE batch_id=?", (batch_id,)
            ).fetchone()[0]
            conn.executemany(
                "INSERT OR IGNORE INTO job_queue "
                "(result_uid, batch_id, session_id, analysis_key, analysis_summary) "
                "VALUES (?,?,?,?,?)",
                rows,
            )
            after = conn.execute(
                "SELECT COUNT(*) FROM job_queue WHERE batch_id=?", (batch_id,)
            ).fetchone()[0]
            conn.execute(
                "UPDATE job_batches SET n_jobs=? WHERE batch_id=?", (after, batch_id)
            )
            conn.execute("COMMIT")
        return after - before

    # ── Worker interface ──────────────────────────────────────────────────────

    def claim_next(self, worker_id: str, batch_id: str, timeout_minutes: int = 60) -> dict | None:
        """Atomically claim the next available job in a batch.

        A job is claimable if its ``status`` is ``'pending'``, or if it is
        ``'running'`` but its ``claimed_at`` timestamp is older than
        ``timeout_minutes`` (stale — the previous worker likely crashed).

        Parameters
        ----------
        worker_id : str
            Unique identifier for this worker (e.g. ``"$JOB_ID.$SGE_TASK_ID"``).
        batch_id : str
            Batch to claim from.
        timeout_minutes : int
            Minutes after which a running job is considered stale.

        Returns
        -------
        dict or None
            Row dict with keys ``result_uid``, ``batch_id``, ``session_id``,
            ``analysis_key``, ``analysis_summary``, or ``None`` if no claimable
            job exists in this batch.
        """
        stale_interval = f"-{timeout_minutes} minutes"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT result_uid, batch_id, session_id, analysis_key, analysis_summary
                FROM job_queue
                WHERE batch_id = ?
                  AND (status = 'pending'
                       OR (status = 'running' AND claimed_at < datetime('now', ?)))
                ORDER BY created_at
                LIMIT 1
                """,
                (batch_id, stale_interval),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            conn.execute(
                "UPDATE job_queue SET status='running', claimed_by=?, claimed_at=datetime('now') "
                "WHERE result_uid=? AND batch_id=?",
                (worker_id, row["result_uid"], batch_id),
            )
            conn.execute("COMMIT")
        return dict(row)

    def mark_done(self, uid: str, batch_id: str) -> None:
        """Mark a job as successfully completed."""
        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute(
                "UPDATE job_queue SET status='done', completed_at=datetime('now') "
                "WHERE result_uid=? AND batch_id=?",
                (uid, batch_id),
            )
            conn.execute("COMMIT")

    def mark_failed(self, uid: str, batch_id: str, error: str) -> None:
        """Mark a job as failed and store the error/traceback."""
        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute(
                "UPDATE job_queue SET status='failed', completed_at=datetime('now'), error=? "
                "WHERE result_uid=? AND batch_id=?",
                (error[:4096], uid, batch_id),
            )
            conn.execute("COMMIT")

    # ── Status ────────────────────────────────────────────────────────────────

    def status_summary(self, batch_id: str | None = None) -> dict[str, int]:
        """Return job counts grouped by status.

        Parameters
        ----------
        batch_id : str or None
            If given, count only jobs in that batch. If ``None``, count across
            all batches.

        Returns
        -------
        dict[str, int]
            E.g. ``{"pending": 120, "running": 4, "done": 56, "failed": 2}``.
        """
        with self._connect() as conn:
            if batch_id is not None:
                rows = conn.execute(
                    "SELECT status, COUNT(*) FROM job_queue WHERE batch_id=? GROUP BY status",
                    (batch_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT status, COUNT(*) FROM job_queue GROUP BY status"
                ).fetchall()
        return {row[0]: row[1] for row in rows}

    # ── Smoke test support ────────────────────────────────────────────────────

    def claim_for_test(self, n: int) -> list[dict]:
        """Atomically mark up to n pending jobs (any batch) as 'test-block'.

        Real workers skip 'test-block' jobs (``claim_next`` only claims
        ``'pending'`` or stale ``'running'``). Call :meth:`release_test_blocks`
        to restore them to ``'pending'``.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT result_uid, batch_id, session_id, analysis_key, analysis_summary "
                "FROM job_queue WHERE status='pending' ORDER BY created_at LIMIT ?",
                (n,),
            ).fetchall()
            if rows:
                for r in rows:
                    conn.execute(
                        "UPDATE job_queue SET status='test-block' "
                        "WHERE result_uid=? AND batch_id=?",
                        (r["result_uid"], r["batch_id"]),
                    )
            conn.execute("COMMIT")
        return [dict(r) for r in rows]

    def release_test_blocks(self) -> int:
        """Reset all 'test-block' jobs back to 'pending'."""
        with self._connect() as conn:
            conn.execute("BEGIN")
            cur = conn.execute(
                "UPDATE job_queue SET status='pending' WHERE status='test-block'"
            )
            n = cur.rowcount
            conn.execute("COMMIT")
        return n
