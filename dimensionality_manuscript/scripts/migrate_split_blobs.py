"""Migrate ResultsStore: move result_blob out of results into result_blobs table.

Before this migration, result_blob was stored inline in the results table,
causing full 65 GB scans even for metadata-only queries. After migration,
result_blobs is a sibling table touched only during store/load operations.

Migration is in-place with a timestamped backup created before any changes.

Usage
-----
    # Dry run — report what would happen
    python -m dimensionality_manuscript.scripts.migrate_split_blobs --dry-run

    # Apply migration
    python -m dimensionality_manuscript.scripts.migrate_split_blobs
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dimensionality_manuscript.pipeline.store import BASE_STORE_PATH, _BLOB_SCHEMA


def migrate(db_path: Path = BASE_STORE_PATH, dry_run: bool = False) -> None:
    """Split result_blob out of the results table into result_blobs.

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

    conn = sqlite3.connect(db_path)
    cols = [d[1] for d in conn.execute("PRAGMA table_info(results)").fetchall()]
    conn.close()

    if "result_blob" not in cols:
        print("result_blob column not in results table — already migrated or wrong DB.")
        return

    conn = sqlite3.connect(db_path)
    (total_rows,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
    (blob_rows,) = conn.execute("SELECT COUNT(*) FROM results WHERE result_blob IS NOT NULL").fetchone()
    conn.close()

    print(f"DB: {db_path}")
    print(f"  Total rows: {total_rows}")
    print(f"  Rows with blobs: {blob_rows}")
    print(f"  Rows without blobs (completion markers / not stored): {total_rows - blob_rows}")

    if dry_run:
        print("\nDry run — no changes made.")
        print("Would:")
        print(f"  1. Backup DB to results.backup_TIMESTAMP.db")
        print(f"  2. CREATE TABLE result_blobs (result_uid PK, result_blob BLOB NOT NULL)")
        print(f"  3. INSERT {blob_rows} rows into result_blobs from results")
        print(f"  4. ALTER TABLE results DROP COLUMN result_blob")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(db_path.stem + f".backup_{timestamp}" + db_path.suffix)
    print(f"\nBacking up to {backup_path} ...")
    shutil.copy2(db_path, backup_path)
    print("Backup done.")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    print("Creating result_blobs table ...")
    conn.execute(_BLOB_SCHEMA)

    print(f"Copying {blob_rows} blobs into result_blobs ...")
    conn.execute(
        "INSERT OR IGNORE INTO result_blobs (result_uid, result_blob) "
        "SELECT result_uid, result_blob FROM results WHERE result_blob IS NOT NULL"
    )

    print("Dropping result_blob column from results ...")
    conn.execute("ALTER TABLE results DROP COLUMN result_blob")

    conn.commit()
    conn.close()

    print(f"\nMigration complete.")
    print(f"  Backup: {backup_path}")
    print(f"  Blobs moved: {blob_rows}")


def main():
    parser = argparse.ArgumentParser(description="Split result_blob into sibling table")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Report without modifying")
    parser.add_argument("--db-path", type=Path, default=BASE_STORE_PATH, help=f"Path to DB (default: {BASE_STORE_PATH})")
    args = parser.parse_args()
    migrate(db_path=args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
