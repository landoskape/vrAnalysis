"""Migrate expmax results: convert list fields to numpy arrays.

Old format:
  step_mse, step_r2, step_rms  — Python lists of floats
  step_uniq_val_count          — list of 1-D ndarrays

New format:
  step_mse, step_r2, step_rms  — np.array(list)
  step_uniq_val_count          — np.stack(list_of_arrays)
"""

import pickle
import sqlite3
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dimensionality_manuscript.registry import RegistryPaths

DB_PATH = Path(RegistryPaths.pipeline_v2_db_path)
LIST_KEYS = ("step_mse", "step_r2", "step_rms")
STACK_KEY = "step_uniq_val_count"


def _migrate_result(result: dict) -> tuple[dict, bool]:
    changed = False
    for k in LIST_KEYS:
        if k in result and isinstance(result[k], list):
            result[k] = np.array(result[k])
            changed = True
    if STACK_KEY in result and isinstance(result[STACK_KEY], list):
        result[STACK_KEY] = np.stack(result[STACK_KEY])
        changed = True
    return result, changed


def main(dry_run: bool = False):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    rows = conn.execute(
        "SELECT result_uid, result_blob FROM results WHERE analysis_type='expmax'"
    ).fetchall()

    print(f"Found {len(rows)} expmax rows.")

    migrated = 0
    skipped = 0
    for uid, blob in tqdm(rows, desc="Migrating"):
        if blob is None:
            skipped += 1
            continue
        result = pickle.loads(blob)
        result, changed = _migrate_result(result)
        if not changed:
            skipped += 1
            continue
        if not dry_run:
            new_blob = pickle.dumps(result)
            conn.execute(
                "UPDATE results SET result_blob=? WHERE result_uid=?",
                (new_blob, uid),
            )
        migrated += 1

    if not dry_run:
        conn.commit()
    conn.close()

    print(f"{'[DRY RUN] ' if dry_run else ''}Migrated: {migrated}, already current: {skipped}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", "-n", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
