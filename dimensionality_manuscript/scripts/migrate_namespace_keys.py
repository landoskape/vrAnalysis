"""Rekey the ResultsStore onto namespaced analysis keys.

Why
---
``analysis_key`` used to be ``SHA256(dataclass fields)``. ``display_name`` is a
ClassVar, so it never entered the hash: two config classes with the same field
names and values produced the *same* key. Since ``result_uid = H(session_id :
analysis_key)`` is the results table's primary key *and* the blob filename,
colliding configs occupied the same row and the same file on disk and silently
overwrote each other.

Keys are now ``H(display_name : legacy_key)``.

Migration
---------
The new key is a pure function of two columns every row already carries::

    new_analysis_key = H(analysis_type : analysis_key)
    new_result_uid   = H(session_id : new_analysis_key)

No config object is reconstructed, so rows belonging to configs that no longer
exist in the codebase migrate correctly too. Blob files are renamed to match
their new uid; same-volume renames are metadata-only.

Usage
-----
    python -m dimensionality_manuscript.scripts.migrate_namespace_keys            # dry run
    python -m dimensionality_manuscript.scripts.migrate_namespace_keys --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dimensionality_manuscript.configs import ANALYSIS_CONFIG_CLASSES
from dimensionality_manuscript.pipeline.base import KEY_SCHEME, namespaced_key
from dimensionality_manuscript.pipeline.store import _META_SCHEMA, BASE_STORE_PATH

_TABLES = ("results", "errors")


def _result_uid(session_id: str, analysis_key: str) -> str:
    return hashlib.sha256(f"{session_id}:{analysis_key}".encode()).hexdigest()[:16]


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open the database directly, bypassing ``ResultsStore``'s scheme guard."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _already_migrated(conn: sqlite3.Connection) -> bool:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "store_meta" not in tables:
        return False
    row = conn.execute("SELECT value FROM store_meta WHERE name='key_scheme'").fetchone()
    return row is not None and row[0] == KEY_SCHEME


def _load_rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    cols = [d[1] for d in conn.execute(f"PRAGMA table_info({table})")]
    return [dict(zip(cols, r)) for r in conn.execute(f"SELECT * FROM {table}")]


def plan_migration(conn: sqlite3.Connection) -> dict:
    """Compute the new key/uid for every row and validate the result.

    Returns
    -------
    dict
        ``rows`` maps table name to a list of ``(old_uid, new_uid, new_key)``
        tuples, alongside per-type counts and any blocking problems found.
    """
    plan: dict = {"rows": {}, "per_type": defaultdict(lambda: defaultdict(int)), "problems": []}

    for table in _TABLES:
        rows = _load_rows(conn, table)
        mapped: list[tuple[str, str, str]] = []
        for row in rows:
            analysis_type = row.get("analysis_type")
            if not analysis_type:
                plan["problems"].append(f"{table}: row {row['result_uid']} has no analysis_type; cannot derive a namespace")
                continue
            new_key = namespaced_key(analysis_type, row["analysis_key"])
            new_uid = _result_uid(row["session_id"], new_key)
            mapped.append((row["result_uid"], new_uid, new_key))
            plan["per_type"][analysis_type][table] += 1
        plan["rows"][table] = mapped

    # A new uid landing on an existing uid would clobber a live row/blob mid-rename.
    for table, mapped in plan["rows"].items():
        old_uids = {old for old, _, _ in mapped}
        new_uids = [new for _, new, _ in mapped]
        if len(set(new_uids)) != len(new_uids):
            plan["problems"].append(f"{table}: new result_uids are not unique")
        overlap = old_uids & set(new_uids)
        if overlap:
            plan["problems"].append(f"{table}: {len(overlap)} new result_uid(s) collide with existing uids, e.g. {sorted(overlap)[:3]}")

    return plan


def verify_against_live_configs(conn: sqlite3.Connection) -> list[str]:
    """Confirm the script's arithmetic matches what ``cfg.key()`` now produces.

    For every registered config variation, the namespaced key derived from its
    ``legacy_key`` must equal its ``key()``, and any row currently stored under
    that legacy key must map onto it.
    """
    problems: list[str] = []
    stored = defaultdict(set)
    for atype, akey in conn.execute("SELECT DISTINCT analysis_type, analysis_key FROM results"):
        stored[atype].add(akey)

    matched = 0
    for display_name, cls in ANALYSIS_CONFIG_CLASSES.items():
        for cfg in cls.generate_variations():
            expected = namespaced_key(display_name, cfg.legacy_key())
            if expected != cfg.key():
                problems.append(f"{cls.__name__}: namespaced_key() != cfg.key() ({expected} vs {cfg.key()})")
            if cfg.legacy_key() in stored.get(display_name, ()):
                matched += 1
    print(f"  Live configs whose current rows will be carried across: {matched}")
    return problems


def _blob_dir(db_path: Path) -> Path:
    return db_path.parent / "blobs"


def summarize(plan: dict, db_path: Path) -> None:
    blob_dir = _blob_dir(db_path)
    per_type = plan["per_type"]

    print(f"\n{'analysis_type':<34}{'results':>9}{'errors':>8}")
    print("-" * 51)
    for atype in sorted(per_type):
        print(f"{atype:<34}{per_type[atype]['results']:>9}{per_type[atype]['errors']:>8}")
    print("-" * 51)
    print(f"{'TOTAL':<34}{len(plan['rows']['results']):>9}{len(plan['rows']['errors']):>8}")

    mapped = plan["rows"]["results"]
    present = sum(1 for old, _, _ in mapped if (blob_dir / f"{old}.pkl").exists())
    done = sum(1 for old, new, _ in mapped if not (blob_dir / f"{old}.pkl").exists() and (blob_dir / f"{new}.pkl").exists())
    print(f"\nBlob files to rename: {present}")
    print(f"Rows with no blob file (None-result completion markers): {len(mapped) - present - done}")
    if done:
        print(f"Blobs already at their new name (resumed run): {done}")


def migrate(db_path: Path = BASE_STORE_PATH, apply: bool = False) -> None:
    """Preview or perform the rekey migration.

    Parameters
    ----------
    db_path : Path
        Results database to migrate.
    apply : bool
        If False (default) report what would change and touch nothing.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = _connect(db_path)
    print(f"Store: {db_path}")
    if _already_migrated(conn):
        print(f"Already migrated (key_scheme={KEY_SCHEME!r}). Nothing to do.")
        conn.close()
        return

    plan = plan_migration(conn)
    summarize(plan, db_path)

    print("\nVerifying migration arithmetic against live config classes...")
    problems = plan["problems"] + verify_against_live_configs(conn)

    if problems:
        print(f"\nBLOCKED - {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  - {p}")
        conn.close()
        return
    print("  All checks passed.")

    if not apply:
        print("\nDry run - nothing modified. Re-run with --apply to migrate.")
        conn.close()
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(f"{db_path.stem}.pre_namespace_{timestamp}.db")
    shutil.copy2(db_path, backup_path)
    print(f"\nBackup: {backup_path}")

    manifest_path = db_path.parent / f"key_migration_{timestamp}.json"
    manifest = {
        "key_scheme": KEY_SCHEME,
        "db_path": str(db_path),
        "backup_path": str(backup_path),
        "rows": {table: [list(entry) for entry in mapped] for table, mapped in plan["rows"].items()},
    }
    manifest_path.write_text(json.dumps(manifest))
    print(f"Manifest: {manifest_path}")

    blob_dir = _blob_dir(db_path)
    renamed = skipped = 0
    for old, new, _ in plan["rows"]["results"]:
        old_path = blob_dir / f"{old}.pkl"
        new_path = blob_dir / f"{new}.pkl"
        if new_path.exists():
            skipped += 1
            continue
        if not old_path.exists():
            continue
        old_path.rename(new_path)
        renamed += 1
    print(f"Blobs renamed: {renamed} (skipped, already present: {skipped})")

    for table in _TABLES:
        conn.executemany(
            f"UPDATE {table} SET result_uid=?, analysis_key=? WHERE result_uid=?",
            [(new, new_key, old) for old, new, new_key in plan["rows"][table]],
        )
    conn.execute(_META_SCHEMA)
    conn.execute("INSERT OR REPLACE INTO store_meta (name, value) VALUES ('key_scheme', ?)", (KEY_SCHEME,))
    conn.commit()
    conn.close()

    print(f"\nMigration complete. {len(plan['rows']['results'])} results and {len(plan['rows']['errors'])} errors rekeyed.")
    print(f"Rollback: restore {backup_path} and rename blobs back using {manifest_path}.")


def main():
    parser = argparse.ArgumentParser(description="Rekey ResultsStore onto namespaced analysis keys")
    parser.add_argument("--apply", action="store_true", help="Perform the migration (default is a dry run)")
    parser.add_argument("--db-path", type=Path, default=BASE_STORE_PATH, help=f"Database to migrate (default: {BASE_STORE_PATH})")
    args = parser.parse_args()
    migrate(db_path=args.db_path, apply=args.apply)


if __name__ == "__main__":
    main()
