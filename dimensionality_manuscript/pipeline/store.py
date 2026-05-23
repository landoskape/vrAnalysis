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
# Blobs live on disk as blobs/{uid}.pkl — not in this database.
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
_INSERT_SQL = "INSERT OR REPLACE INTO results ({}) VALUES ({})".format(
    ", ".join(_COLUMN_NAMES),
    ", ".join("?" for _ in _COLUMN_NAMES),
)
_SUMMARY_SQL = "SELECT * FROM results"
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

    Each result is keyed by a unified hash of ``(session_id, analysis_key)``.
    Metadata lives in the SQLite database; result blobs are stored as
    ``blobs/{uid}.pkl`` files in the same directory as the database.

    Per-operation connections with WAL mode and a busy timeout allow safe
    concurrent access from joblib workers.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    """

    def __init__(self, db_path: Path = BASE_STORE_PATH, *, blob_cache_maxsize: int | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._blob_dir.mkdir(parents=True, exist_ok=True)
        self._blob_cache: dict[str, dict | None] = {}
        self._blob_cache_maxsize = blob_cache_maxsize
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @property
    def _blob_dir(self) -> Path:
        return self.db_path.parent / "blobs"

    def _blob_path(self, uid: str) -> Path:
        return self._blob_dir / f"{uid}.pkl"

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

        The blob file is written before the database row is committed, so a
        row will never exist that should have a blob but doesn't. If the
        database commit fails, the orphan blob file is cleaned up.

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
            Whether the result blob should be written to disk (True) or
            is stored externally (False). Default True.
        """
        uid = self._uid(session_id, analysis_cfg)
        blob_path = self._blob_path(uid)
        self._blob_cache.pop(uid, None)

        # Write blob file first — row will never be added without its blob
        if result is not None and result_stored:
            blob_path.write_bytes(pickle.dumps(result))

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
        try:
            with self._connect() as conn:
                conn.execute(_INSERT_SQL, meta_values)
        except BaseException:
            blob_path.unlink(missing_ok=True)
            raise

    def get(self, session_id: str, analysis_cfg: AnalysisConfigBase) -> dict | None:
        """Retrieve a stored result, or None if not found / completion marker."""
        uid = self._uid(session_id, analysis_cfg)
        return self.get_by_uid(uid)

    def get_by_uid(self, uid: str) -> dict | None:
        """Retrieve a stored result directly by its result_uid.

        Results are cached in-memory on this store instance until
        :meth:`clear_blob_cache` is called or the entry is invalidated.
        """
        if uid in self._blob_cache:
            return self._blob_cache[uid]
        path = self._blob_path(uid)
        if not path.exists():
            result = None
        else:
            result = pickle.loads(path.read_bytes())
        if self._blob_cache_maxsize is None or len(self._blob_cache) < self._blob_cache_maxsize:
            self._blob_cache[uid] = result
        return result

    def clear_blob_cache(self) -> None:
        """Clear the in-memory unpickled blob cache."""
        self._blob_cache.clear()

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
            rows = conn.execute(f"SELECT result_uid FROM results WHERE {where}", tuple(active.values())).fetchall()
            uids = [row[0] for row in rows]
            conn.execute(f"DELETE FROM results WHERE {where}", tuple(active.values()))
        for uid in uids:
            self._blob_path(uid).unlink(missing_ok=True)
            self._blob_cache.pop(uid, None)

    def invalidate_all(self):
        """Delete all results and their blob files."""
        with self._connect() as conn:
            conn.execute("DELETE FROM results")
        for p in self._blob_dir.glob("*.pkl"):
            p.unlink(missing_ok=True)
        self._blob_cache.clear()

    def summary_table(
        self,
        as_dataframe: bool = False,
        *,
        analysis_type: str | None = None,
        session_ids: list[str] | None = None,
    ) -> list[dict] | pd.DataFrame:
        """Return metadata for stored results (no blob data).

        Parameters
        ----------
        as_dataframe : bool
            If True, return a pandas DataFrame instead of a list of dicts.
        analysis_type : str, optional
            If given, only rows with this ``analysis_type`` are returned.
        session_ids : list of str, optional
            If given, only rows whose ``session_id`` is in this list are returned.
        """
        clauses: list[str] = []
        params: list = []
        if analysis_type is not None:
            clauses.append("analysis_type=?")
            params.append(analysis_type)
        if session_ids is not None:
            if not session_ids:
                records: list[dict] = []
                if as_dataframe:
                    return pd.DataFrame(records, columns=list(_COLUMN_NAMES))
                return records
            placeholders = ",".join("?" for _ in session_ids)
            clauses.append(f"session_id IN ({placeholders})")
            params.extend(session_ids)
        sql = _SUMMARY_SQL
        if clauses:
            sql = f"{_SUMMARY_SQL} WHERE {' AND '.join(clauses)}"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
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
