"""Migrate ResultsStore from split (data_key, analysis_key) to unified analysis_key.

The old schema stored data_key and analysis_key as separate columns, and the
result_uid was SHA256(session_id:data_key:analysis_key)[:16].

The new schema embeds data_config_name inside AnalysisConfigBase, so the
analysis_key now encodes both. result_uid is SHA256(session_id:analysis_key)[:16].

Migration strategy
------------------
For each known config class × data_config_name combination, compute both the
*old* analysis key (dict without data_config_name) and the *new* analysis key
(dict with data_config_name). This gives a reverse-lookup table:

    (old_data_key, old_analysis_key) -> new_analysis_key

For each row in the old DB, look up the new analysis key, recompute result_uid,
and write a new row with the updated schema.

Usage
-----
    python -m dimensionality_manuscript.scripts.migrate_store [--dry-run] [--db-path PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dimensionality_manuscript.configs.cvpca import CVPCAConfig
from dimensionality_manuscript.configs.data_config import _NAMED_CONFIGS, get_data_config
from dimensionality_manuscript.configs.population import PopulationConfig
from dimensionality_manuscript.configs.regression import RegressionConfig
from dimensionality_manuscript.configs.subspace import SubspaceConfig
from dimensionality_manuscript.pipeline.store import BASE_STORE_PATH, ResultsStore

_ALL_CLASSES = [CVPCAConfig, SubspaceConfig, RegressionConfig, PopulationConfig]


def _old_analysis_key(cfg) -> str:
    """SHA256 key of cfg as if data_config_name field didn't exist (old schema)."""
    d = {k: v for k, v in asdict(cfg).items() if k != "data_config_name"}
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def build_lookup() -> dict[tuple[str, str], str]:
    """Build map (old_data_key, old_analysis_key) -> new_analysis_key."""
    lookup: dict[tuple[str, str], str] = {}
    for cls in _ALL_CLASSES:
        for data_name in _NAMED_CONFIGS:
            dcfg = get_data_config(data_name)
            old_dk = dcfg.key()
            for cfg in cls.generate_variations():
                old_ak = _old_analysis_key(cfg)
                new_ak = cfg.key()
                lookup[(old_dk, old_ak)] = new_ak
    return lookup


def migrate(db_path: Path = BASE_STORE_PATH, dry_run: bool = False) -> None:
    """Migrate the ResultsStore at db_path to the new schema.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database to migrate.
    dry_run : bool
        If True, report what would be done without modifying the database.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    print(f"Building config lookup table...")
    lookup = build_lookup()
    print(f"  {len(lookup)} (old_data_key, old_analysis_key) -> new_analysis_key entries")

    # Read all rows from the old DB
    conn_old = sqlite3.connect(db_path)
    old_cols = [d[1] for d in conn_old.execute("PRAGMA table_info(results)").fetchall()]
    rows = conn_old.execute("SELECT * FROM results").fetchall()
    conn_old.close()

    if "data_key" not in old_cols:
        print("Database already has the new schema (no data_key column). Nothing to migrate.")
        return

    row_dicts = [dict(zip(old_cols, r)) for r in rows]
    print(f"\nRows in database: {len(row_dicts)}")

    migrated = []
    skipped = []
    for row in row_dicts:
        key = (row["data_key"], row["analysis_key"])
        if key not in lookup:
            skipped.append(row["result_uid"])
            continue
        new_ak = lookup[key]
        new_uid = hashlib.sha256(f"{row['session_id']}:{new_ak}".encode()).hexdigest()[:16]
        migrated.append(
            {
                "result_uid": new_uid,
                "session_id": row["session_id"],
                "analysis_key": new_ak,
                "analysis_summary": row.get("analysis_summary"),
                "analysis_type": row.get("analysis_type"),
                "schema_version": row.get("schema_version"),
                "result_stored": row.get("result_stored", 1),
                "result_blob": row.get("result_blob"),
                "snapshot_path": row.get("snapshot_path"),
                "computed_at": row.get("computed_at"),
            }
        )

    print(f"Rows to migrate: {len(migrated)}")
    print(f"Rows skipped (unrecognized config): {len(skipped)}")
    if skipped:
        print(f"  First few skipped UIDs: {skipped[:5]}")

    if dry_run:
        print("\nDry run — no changes made.")
        return

    # Backup the existing DB
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_suffix(f".backup_{timestamp}.db")
    shutil.copy2(db_path, backup_path)
    print(f"\nBackup created: {backup_path}")

    # Write migrated rows into a new DB with the new schema
    new_db_path = db_path.with_suffix(".migrated.db")
    new_store = ResultsStore(new_db_path)

    insert_sql = (
        "INSERT OR REPLACE INTO results "
        "(result_uid, session_id, analysis_key, analysis_summary, analysis_type, "
        "schema_version, result_stored, result_blob, snapshot_path, computed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)"
    )
    with new_store._connect() as conn:
        conn.executemany(
            insert_sql,
            [
                (
                    r["result_uid"],
                    r["session_id"],
                    r["analysis_key"],
                    r["analysis_summary"],
                    r["analysis_type"],
                    r["schema_version"],
                    r["result_stored"],
                    r["result_blob"],
                    r["snapshot_path"],
                    r["computed_at"],
                )
                for r in migrated
            ],
        )

    # Replace original with migrated
    new_db_path.replace(db_path)
    print(f"Migration complete. {len(migrated)} rows written to {db_path}")
    print(f"Original backed up at {backup_path}")


def main():
    parser = argparse.ArgumentParser(description="Migrate ResultsStore to new schema")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Report what would be done without modifying the database")
    parser.add_argument("--db-path", type=Path, default=BASE_STORE_PATH, help=f"Path to the database (default: {BASE_STORE_PATH})")
    args = parser.parse_args()

    migrate(db_path=args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
