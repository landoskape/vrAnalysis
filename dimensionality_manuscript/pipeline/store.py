"""SQLite-backed results store, content-addressed by a unified result_uid."""

from __future__ import annotations

import hashlib
import pickle
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from freezedry import freezedry
from vrAnalysis.files import repo_path
from ..registry import RegistryPaths

if TYPE_CHECKING:
    from .base import AnalysisConfigBase

# 30 second busy timeout — parallel writers wait instead of failing
_BUSY_TIMEOUT_MS = 30_000

# Single source of truth for column names and SQL types.
# Tuples of (column_name, sql_type). First entry is the primary key.
# result_blob lives in the sibling result_blobs table — touch only on store/load.
_COLUMNS = (
    ("result_uid", "TEXT PRIMARY KEY"),
    ("session_id", "TEXT NOT NULL"),
    ("analysis_key", "TEXT NOT NULL"),
    ("analysis_summary", "TEXT"),
    ("analysis_type", "TEXT"),
    ("schema_version", "TEXT"),
    ("result_stored", "INTEGER NOT NULL DEFAULT 1"),
    ("snapshot_path", "TEXT"),
    ("computed_at", "TIMESTAMP"),
)

_COLUMN_NAMES = tuple(name for name, _ in _COLUMNS)
_SCHEMA = "CREATE TABLE IF NOT EXISTS results (\n    {}\n)".format(",\n    ".join(f"{name} {sqltype}" for name, sqltype in _COLUMNS))
_BLOB_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS result_blobs (\n"
    "    result_uid TEXT PRIMARY KEY REFERENCES results(result_uid) ON DELETE CASCADE,\n"
    "    result_blob BLOB NOT NULL\n"
    ")"
)
_INSERT_SQL = "INSERT OR REPLACE INTO results ({}) VALUES ({})".format(
    ", ".join(_COLUMN_NAMES),
    ", ".join("?" for _ in _COLUMN_NAMES),
)
_INSERT_BLOB_SQL = "INSERT OR REPLACE INTO result_blobs (result_uid, result_blob) VALUES (?, ?)"
_SUMMARY_SQL = "SELECT * FROM results"
# Keep for backwards-compat with any callers that reference _SUMMARY_COLUMNS
_SUMMARY_COLUMNS = _COLUMN_NAMES


def result_uid(session_id: str, analysis_key: str) -> str:
    """Compute the unified hash for a (session, analysis) pair.

    Parameters
    ----------
    session_id : str
        Unique session identifier.
    analysis_key : str
        Content hash of the AnalysisConfigBase (which encodes data_config_name).

    Returns
    -------
    str
        SHA256 hex digest (16 chars) of the combined key.
    """
    combined = f"{session_id}:{analysis_key}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


BASE_STORE_PATH = RegistryPaths.pipeline_v2_db_path


class ResultsStore:
    """SQLite-backed store for analysis results.

    Each result is keyed by a unified hash of ``(session_id, data_key,
    analysis_key)``. Per-operation connections with WAL mode and a busy
    timeout allow safe concurrent access from joblib workers.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    """

    def __init__(self, db_path: Path = BASE_STORE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.execute(_BLOB_SCHEMA)

    @contextmanager
    def _connect(self):
        """Yield a connection that is committed and closed on exit."""
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _uid(session_id: str, analysis_cfg: AnalysisConfigBase) -> str:
        return result_uid(session_id, analysis_cfg.key())

    def has(self, session_id: str, analysis_cfg: AnalysisConfigBase) -> bool:
        """Check if a result exists for the given keys."""
        uid = self._uid(session_id, analysis_cfg)
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM results WHERE result_uid=?", (uid,)).fetchone()
        return row is not None

    def put(
        self,
        session_id: str,
        analysis_cfg: AnalysisConfigBase,
        result: dict | None,
        snapshot_path: str | None = None,
        result_stored: bool = True,
    ):
        """Store a result, overwriting any existing entry.

        Parameters
        ----------
        session_id : str
            Unique session identifier.
        analysis_cfg : AnalysisConfigBase
            Analysis configuration for this result (encodes data_config_name).
        result : dict or None
            Result dict to pickle, or None as a completion marker.
        snapshot_path : str or None
            Path to the codebase snapshot associated with this run.
        result_stored : bool
            Whether the result blob lives in this database (True) or
            is stored externally (False). Default True.
        """
        uid = self._uid(session_id, analysis_cfg)
        blob = pickle.dumps(result) if result is not None else None
        # Order must match _COLUMN_NAMES (no blob column)
        meta_values = (
            uid,
            session_id,
            analysis_cfg.key(),
            analysis_cfg.summary(),
            analysis_cfg.display_name,
            analysis_cfg.schema_version,
            int(result_stored),
            snapshot_path,
            datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as conn:
            conn.execute(_INSERT_SQL, meta_values)
            if blob is not None:
                conn.execute(_INSERT_BLOB_SQL, (uid, blob))

    def get(self, session_id: str, analysis_cfg: AnalysisConfigBase) -> dict | None:
        """Retrieve a stored result, or None if not found / completion marker."""
        uid = self._uid(session_id, analysis_cfg)
        return self.get_by_uid(uid)

    def get_by_uid(self, uid: str) -> dict | None:
        """Retrieve a stored result directly by its result_uid."""
        with self._connect() as conn:
            row = conn.execute("SELECT result_blob FROM result_blobs WHERE result_uid=?", (uid,)).fetchone()
        if row is None:
            return None
        return pickle.loads(row[0])

    def invalidate(
        self,
        *,
        schema_version: str | None = None,
        analysis_cfg: AnalysisConfigBase | None = None,
        analysis_type: str | None = None,
    ):
        """Delete results matching at least one filter.

        Parameters
        ----------
        schema_version : str, optional
            Delete results with this schema version.
        analysis_cfg : AnalysisConfigBase, optional
            Delete results with this analysis key.
        analysis_type : str, optional
            Delete results with this analysis type.
        """
        filters = {
            "schema_version": schema_version,
            "analysis_key": analysis_cfg.key() if analysis_cfg is not None else None,
            "analysis_type": analysis_type,
        }
        active = {k: v for k, v in filters.items() if v is not None}
        if not active:
            raise ValueError("At least one filter is required. Use invalidate_all() to delete everything.")
        where = " AND ".join(f"{k}=?" for k in active)
        with self._connect() as conn:
            conn.execute(f"DELETE FROM results WHERE {where}", tuple(active.values()))

    def invalidate_all(self):
        """Delete all results."""
        with self._connect() as conn:
            conn.execute("DELETE FROM results")

    def summary_table(self, as_dataframe: bool = False) -> list[dict] | pd.DataFrame:
        """Return metadata for all stored results (no blob data).

        Parameters
        ----------
        as_dataframe : bool
            If True, return a pandas DataFrame instead of a list of dicts.
        """
        with self._connect() as conn:
            rows = conn.execute(_SUMMARY_SQL).fetchall()
        records = [dict(zip(_COLUMN_NAMES, row)) for row in rows]
        if as_dataframe:
            return pd.DataFrame(records, columns=list(_COLUMN_NAMES))
        return records

    def coverage(
        self,
        sessions: list,
        analysis_configs: list[AnalysisConfigBase],
    ) -> float:
        """Fraction of (session, analysis_cfg) pairs present in the store."""
        total = len(sessions) * len(analysis_configs)
        if total == 0:
            return 1.0
        # Batch lookup with a single connection
        uids = {self._uid(ses.session_uid, acfg) for ses in sessions for acfg in analysis_configs}
        found = 0
        with self._connect() as conn:
            for uid in uids:
                row = conn.execute("SELECT 1 FROM results WHERE result_uid=?", (uid,)).fetchone()
                if row is not None:
                    found += 1
        return found / total

    def snapshot_codebase(self) -> Path:
        """Save a zip snapshot of the codebase and return its path.

        Returns
        -------
        Path
            Path to the created snapshot zip file.
        """
        snapshot_dir = self._codebase_snapshot_dir()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = snapshot_dir / f"snapshot_{timestamp}.zip"
        freezedry(repo_path(), output_path, ignore_git=True, use_gitignore=True, verbose=False)
        return output_path

    def list_snapshots(self) -> list[Path]:
        """Return paths to all codebase snapshots."""
        return sorted(self._codebase_snapshot_dir().glob("snapshot_*.zip"))

    def _codebase_snapshot_dir(self) -> Path:
        """Save codebase snapshots in a 'codebase_snapshots' subdirectory next to the database."""
        snapshot_dir = self.db_path.parent / "codebase_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        return snapshot_dir
