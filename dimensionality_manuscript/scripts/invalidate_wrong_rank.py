"""Invalidate VectorGainRank results whose stored score arrays have wrong length.

Usage (dry run, default):
    python -m dimensionality_manuscript.scripts.invalidate_wrong_rank

Execute deletes:
    python -m dimensionality_manuscript.scripts.invalidate_wrong_rank --execute
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from dimensionality_manuscript.pipeline.store import ResultsStore
from dimensionality_manuscript.registry import RegistryPaths

EXPECTED_RANK = 200
ANALYSIS_TYPE = "vector_gain_rank"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually delete (default: dry run)")
    parser.add_argument("--db-path", type=Path, default=None)
    args = parser.parse_args(argv)

    db_path = args.db_path or RegistryPaths().pipeline_v2_db_path
    if not Path(db_path).exists():
        print(f"No database at {db_path}", file=sys.stderr)
        return 1

    store = ResultsStore(db_path)
    plan = store.plan_invalidate(analysis_type=ANALYSIS_TYPE)
    rows = store.rows_matching_invalidate_plan(plan)

    if not rows:
        print(f"No {ANALYSIS_TYPE} rows found.")
        return 0

    print(f"Found {len(rows)} {ANALYSIS_TYPE} rows. Checking blob lengths...")

    bad_uids: list[str] = []
    for row in rows:
        uid = row["result_uid"]
        blob_path = store._blob_path(uid)
        if not blob_path.exists():
            print(f"  MISSING blob: {uid}  session={row['session_id']}")
            bad_uids.append(uid)
            continue
        with open(blob_path, "rb") as f:
            result = pickle.load(f)
        # result is a dict with keys like "mse", "r2" — each a numpy array of length max_rank
        lengths = {k: len(v) for k, v in result.items() if hasattr(v, "__len__")}
        wrong = {k: n for k, n in lengths.items() if n != EXPECTED_RANK}
        if wrong:
            print(f"  WRONG rank: {uid}  session={row['session_id']}  lengths={lengths}")
            bad_uids.append(uid)
        else:
            print(f"  ok: {uid}  session={row['session_id']}  lengths={lengths}")

    print(f"\n{len(bad_uids)} / {len(rows)} rows have wrong rank (expected {EXPECTED_RANK}).")

    if not bad_uids:
        print("Nothing to do.")
        return 0

    if not args.execute:
        print("Dry run — pass --execute to delete.")
        return 0

    print(f"Deleting {len(bad_uids)} rows + blobs...")
    with store._connect() as conn:
        placeholders = ",".join("?" for _ in bad_uids)
        conn.execute(f"DELETE FROM results WHERE result_uid IN ({placeholders})", bad_uids)

    for uid in bad_uids:
        blob_path = store._blob_path(uid)
        if blob_path.exists():
            blob_path.unlink()
            print(f"  deleted blob: {blob_path.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
