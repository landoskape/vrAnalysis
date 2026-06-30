"""SQLite-backed results store, content-addressed by a unified result_uid."""

from __future__ import annotations

import hashlib
import pickle
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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

_ERROR_COLUMNS = (
    ("result_uid", "TEXT PRIMARY KEY"),
    ("session_id", "TEXT NOT NULL"),
    ("analysis_key", "TEXT NOT NULL"),
    ("analysis_summary", "TEXT"),
    ("analysis_type", "TEXT"),
    ("schema_version", "TEXT"),
    ("error_message", "TEXT"),
    ("traceback", "TEXT"),
    ("failed_at", "TIMESTAMP"),
)
_ERROR_COLUMN_NAMES = tuple(name for name, _ in _ERROR_COLUMNS)
_ERROR_SCHEMA = "CREATE TABLE IF NOT EXISTS errors (\n    {}\n)".format(",\n    ".join(f"{name} {sqltype}" for name, sqltype in _ERROR_COLUMNS))
_ERROR_INSERT_SQL = "INSERT OR REPLACE INTO errors ({}) VALUES ({})".format(
    ", ".join(_ERROR_COLUMN_NAMES),
    ", ".join("?" for _ in _ERROR_COLUMN_NAMES),
)


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


def _analysis_config_classes() -> dict[str, type[AnalysisConfigBase]]:
    """Map ``analysis_type`` (config ``display_name``) to config class."""
    from ..configs import (
        CVPCAConfig,
        ExpMaxConfig,
        LocPredConfig,
        LocPredCrossVal,
        PFPredQualityConfig,
        PlacefieldStructureConfig,
        PopulationConfig,
        RegressionConfig,
        StimSpaceConfig,
        StimSpaceSpectraConfig,
        SubspaceConfig,
        TilburyFitConfig,
        StimFullSweepConfig,
        ThresholdedGPSweepConfig,
        SmoothGPSweepConfig,
        TilburySweepConfig,
    )

    classes = (
        CVPCAConfig,
        ExpMaxConfig,
        LocPredConfig,
        LocPredCrossVal,
        PFPredQualityConfig,
        PlacefieldStructureConfig,
        PopulationConfig,
        RegressionConfig,
        StimSpaceConfig,
        StimSpaceSpectraConfig,
        SubspaceConfig,
        TilburyFitConfig,
        StimFullSweepConfig,
        ThresholdedGPSweepConfig,
        SmoothGPSweepConfig,
        TilburySweepConfig,
    )
    return {cls.display_name: cls for cls in classes}


@dataclass(frozen=True)
class InvalidatePlan:
    """SQL plan for :meth:`ResultsStore.invalidate` (no side effects)."""

    where: str
    params: tuple[Any, ...]
    mode: Literal["equality", "param_filters"]
    analysis_keys: tuple[str, ...] = ()
    param_filters: dict[str, Any] = field(default_factory=dict)
    analysis_type: str | None = None
    schema_version: str | None = None
    config_variation_count: int = 0


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
            conn.execute(_ERROR_SCHEMA)

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

    # ── Error table ───────────────────────────────────────────────────────────

    def put_error(
        self,
        session_id: str,
        analysis_cfg: AnalysisConfigBase,
        error_message: str,
        traceback: str | None = None,
    ) -> None:
        """Record a failed result. Overwrites any previous error for this key."""
        uid = self._uid(session_id, analysis_cfg)
        values = (
            uid,
            session_id,
            analysis_cfg.key(),
            analysis_cfg.summary(),
            analysis_cfg.display_name,
            analysis_cfg.schema_version,
            error_message,
            traceback,
            datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as conn:
            conn.execute(_ERROR_INSERT_SQL, values)

    def has_error(self, session_id: str, analysis_cfg: AnalysisConfigBase) -> bool:
        """Return True if a recorded error exists for this (session, config) pair."""
        uid = self._uid(session_id, analysis_cfg)
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM errors WHERE result_uid=?", (uid,)).fetchone()
        return row is not None

    def clear_error(self, session_id: str, analysis_cfg: AnalysisConfigBase) -> bool:
        """Delete the error record for this key. Returns True if a row was deleted."""
        uid = self._uid(session_id, analysis_cfg)
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM errors WHERE result_uid=?", (uid,))
            deleted = cursor.rowcount
        return deleted > 0

    def clear_errors_bulk(
        self,
        *,
        analysis_type: str | None = None,
        schema_version: str | None = None,
    ) -> int:
        """Delete error rows matching the given filters. Returns number of rows deleted.

        Parameters
        ----------
        analysis_type : str, optional
            If given, only delete errors with this analysis_type.
        schema_version : str, optional
            If given, only delete errors with this schema_version.
        """
        clauses: list[str] = []
        params: list = []
        if analysis_type is not None:
            clauses.append("analysis_type=?")
            params.append(analysis_type)
        if schema_version is not None:
            clauses.append("schema_version=?")
            params.append(schema_version)
        sql = "DELETE FROM errors"
        if clauses:
            sql = f"{sql} WHERE {' AND '.join(clauses)}"
        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount

    def get_errors(
        self,
        *,
        analysis_type: str | None = None,
        session_ids: list[str] | None = None,
        schema_version: str | None = None,
        as_dataframe: bool = False,
    ) -> "list[dict] | pd.DataFrame":
        """Return metadata for recorded errors (no blob data).

        Parameters
        ----------
        analysis_type : str, optional
            If given, only rows with this ``analysis_type`` are returned.
        session_ids : list of str, optional
            If given, only rows whose ``session_id`` is in this list are returned.
        schema_version : str, optional
            If given, only rows with this ``schema_version`` are returned.
        as_dataframe : bool
            If True, return a pandas DataFrame instead of a list of dicts.
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
                    return pd.DataFrame(records, columns=list(_ERROR_COLUMN_NAMES))
                return records
            placeholders = ",".join("?" for _ in session_ids)
            clauses.append(f"session_id IN ({placeholders})")
            params.extend(session_ids)
        if schema_version is not None:
            clauses.append("schema_version=?")
            params.append(schema_version)
        sql = "SELECT * FROM errors"
        if clauses:
            sql = f"{sql} WHERE {' AND '.join(clauses)}"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        records = [dict(zip(_ERROR_COLUMN_NAMES, row)) for row in rows]
        if as_dataframe:
            return pd.DataFrame(records, columns=list(_ERROR_COLUMN_NAMES))
        return records

    def invalidate(
        self,
        *,
        schema_version: str | None = None,
        analysis_cfg: AnalysisConfigBase | None = None,
        analysis_type: str | None = None,
        param_filters: dict[str, Any] | None = None,
    ) -> int:
        """Delete results and matching errors for the given filters (combined with AND).

        Parameters
        ----------
        schema_version : str, optional
            Delete results with this schema version.
        analysis_cfg : AnalysisConfigBase, optional
            Delete results with this analysis key (all sessions).
        analysis_type : str, optional
            Delete results with this analysis type (``display_name``).
        param_filters : dict, optional
            Delete results whose config matches these field values. Requires
            ``analysis_type``. Expands the config param grid with fixed values
            (e.g. ``{"activity_parameters_name": "raw"}`` invalidates every
            regression combo that used raw activity parameters). Cannot be
            combined with ``analysis_cfg``.

        Returns
        -------
        int
            Number of result rows deleted.
        """
        if analysis_cfg is not None and param_filters is not None:
            raise ValueError("Cannot pass both analysis_cfg and param_filters.")

        plan = self.plan_invalidate(
            schema_version=schema_version,
            analysis_cfg=analysis_cfg,
            analysis_type=analysis_type,
            param_filters=param_filters,
        )
        return self._execute_invalidate_plan(plan)

    def plan_invalidate(
        self,
        *,
        schema_version: str | None = None,
        analysis_cfg: AnalysisConfigBase | None = None,
        analysis_type: str | None = None,
        param_filters: dict[str, Any] | None = None,
        analysis_key: str | None = None,
    ) -> InvalidatePlan:
        """Build the SQL plan used by :meth:`invalidate` without deleting anything.

        Parameters
        ----------
        schema_version : str, optional
            Match this schema version column.
        analysis_cfg : AnalysisConfigBase, optional
            Match this config's ``analysis_key``.
        analysis_type : str, optional
            Match this analysis type.
        param_filters : dict, optional
            Expand the current config param grid with fixed values; requires
            ``analysis_type``.
        analysis_key : str, optional
            Match a single 16-char analysis key (alternative to ``analysis_cfg``).

        Returns
        -------
        InvalidatePlan
        """
        if analysis_cfg is not None and param_filters is not None:
            raise ValueError("Cannot pass both analysis_cfg and param_filters.")
        if analysis_cfg is not None and analysis_key is not None:
            raise ValueError("Cannot pass both analysis_cfg and analysis_key.")
        if analysis_key is not None and param_filters is not None:
            raise ValueError("Cannot pass both analysis_key and param_filters.")

        if param_filters is not None:
            if analysis_type is None:
                raise ValueError("analysis_type is required when param_filters is set.")
            if not param_filters:
                raise ValueError("param_filters must be non-empty.")
            return self._plan_invalidate_param_filters(
                analysis_type=analysis_type,
                param_filters=param_filters,
                schema_version=schema_version,
            )

        filters: dict[str, Any] = {
            "schema_version": schema_version,
            "analysis_key": analysis_key if analysis_key is not None else (analysis_cfg.key() if analysis_cfg is not None else None),
            "analysis_type": analysis_type,
        }
        active = {k: v for k, v in filters.items() if v is not None}
        if not active:
            raise ValueError("At least one filter is required. Use invalidate_all() to delete everything.")
        where = " AND ".join(f"{k}=?" for k in active)
        return InvalidatePlan(
            where=where,
            params=tuple(active.values()),
            mode="equality",
            schema_version=schema_version,
            analysis_type=analysis_type,
        )

    def rows_matching_invalidate_plan(self, plan: InvalidatePlan) -> list[dict]:
        """Return full result rows that :meth:`invalidate` would remove."""
        sql = f"SELECT * FROM results WHERE {plan.where}"
        with self._connect() as conn:
            rows = conn.execute(sql, plan.params).fetchall()
        return [dict(zip(_COLUMN_NAMES, row)) for row in rows]

    def errors_matching_invalidate_plan(self, plan: InvalidatePlan) -> list[dict]:
        """Return error rows that :meth:`invalidate` would remove."""
        if plan.where == "0":
            return []
        sql = f"SELECT * FROM errors WHERE {plan.where}"
        with self._connect() as conn:
            rows = conn.execute(sql, plan.params).fetchall()
        return [dict(zip(_ERROR_COLUMN_NAMES, row)) for row in rows]

    def blob_paths_for_invalidate_plan(self, plan: InvalidatePlan) -> list[tuple[str, Path, bool, int | None]]:
        """Return ``(result_uid, path, exists, size_bytes)`` for each row in the plan."""
        out: list[tuple[str, Path, bool, int | None]] = []
        for row in self.rows_matching_invalidate_plan(plan):
            uid = row["result_uid"]
            path = self._blob_path(uid)
            if path.exists():
                out.append((uid, path, True, path.stat().st_size))
            else:
                out.append((uid, path, False, None))
        return out

    def _plan_invalidate_param_filters(
        self,
        *,
        analysis_type: str,
        param_filters: dict[str, Any],
        schema_version: str | None,
    ) -> InvalidatePlan:
        """Build DELETE plan for ``param_filters`` (current config class only)."""
        config_classes = _analysis_config_classes()
        if analysis_type not in config_classes:
            available = ", ".join(sorted(config_classes))
            raise ValueError(f"Unknown analysis_type {analysis_type!r}. Available: {available}")

        config_cls = config_classes[analysis_type]
        configs = config_cls.generate_variations_matching(param_filters)
        analysis_keys = tuple(cfg.key() for cfg in configs)
        if not analysis_keys:
            return InvalidatePlan(
                where="0",
                params=(),
                mode="param_filters",
                analysis_keys=(),
                param_filters=dict(param_filters),
                analysis_type=analysis_type,
                schema_version=schema_version,
                config_variation_count=0,
            )

        sql_filters: dict[str, Any] = {"analysis_type": analysis_type}
        if schema_version is not None:
            sql_filters["schema_version"] = schema_version

        placeholders = ",".join("?" for _ in analysis_keys)
        sql_filters["analysis_key__in"] = list(analysis_keys)
        where_parts: list[str] = []
        params: list[Any] = []
        for key, value in sql_filters.items():
            if key == "analysis_key__in":
                where_parts.append(f"analysis_key IN ({placeholders})")
                params.extend(value)
            else:
                where_parts.append(f"{key}=?")
                params.append(value)
        return InvalidatePlan(
            where=" AND ".join(where_parts),
            params=tuple(params),
            mode="param_filters",
            analysis_keys=analysis_keys,
            param_filters=dict(param_filters),
            analysis_type=analysis_type,
            schema_version=schema_version,
            config_variation_count=len(configs),
        )

    def _execute_invalidate_plan(self, plan: InvalidatePlan) -> int:
        """Delete result rows, matching error rows, and blobs described by ``plan``."""
        if plan.where == "0":
            return 0
        uids = [row["result_uid"] for row in self.rows_matching_invalidate_plan(plan)]
        with self._connect() as conn:
            conn.execute(f"DELETE FROM results WHERE {plan.where}", plan.params)
            conn.execute(f"DELETE FROM errors WHERE {plan.where}", plan.params)
        for uid in uids:
            self._blob_path(uid).unlink(missing_ok=True)
            self._blob_cache.pop(uid, None)
        return len(uids)

    def invalidate_all(self):
        """Delete all results, errors, and result blob files."""
        with self._connect() as conn:
            conn.execute("DELETE FROM results")
            conn.execute("DELETE FROM errors")
        for p in self._blob_dir.glob("*.pkl"):
            p.unlink(missing_ok=True)
        self._blob_cache.clear()

    def summary_table(
        self,
        as_dataframe: bool = False,
        *,
        analysis_type: str | None = None,
        session_ids: list[str] | None = None,
        schema_version: str | None = None,
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
        schema_version : str, optional
            If given, only rows with this ``schema_version`` are returned.
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
        if schema_version is not None:
            clauses.append("schema_version=?")
            params.append(schema_version)
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

    def missing_sessions_for_config(
        self,
        sessions: list,
        analysis_cfg: AnalysisConfigBase,
    ) -> list[str]:
        """Return session_ids in ``sessions`` with no result for this config."""
        uids_to_sid = {result_uid(ses.session_uid, analysis_cfg.key()): ses.session_uid for ses in sessions}
        if not uids_to_sid:
            return []
        placeholders = ",".join("?" for _ in uids_to_sid)
        with self._connect() as conn:
            found = {
                row[0]
                for row in conn.execute(
                    f"SELECT result_uid FROM results WHERE result_uid IN ({placeholders})",
                    list(uids_to_sid),
                ).fetchall()
            }
        return [sid for uid, sid in uids_to_sid.items() if uid not in found]

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
