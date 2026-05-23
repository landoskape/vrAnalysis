"""Diagnose why status() is slow.

Usage
-----
    conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.profile_status
"""

import time
import sqlite3
from pathlib import Path

from dimensionality_manuscript.registry import RegistryPaths

REGISTRY_PATHS = RegistryPaths()


def _t(label: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    print(f"  {label:<40s} {elapsed*1000:.1f} ms")
    return time.perf_counter()


def main():
    db_path = REGISTRY_PATHS.pipeline_v2_db_path
    print(f"DB: {db_path}")

    if not db_path.exists():
        print("No DB found.")
        return

    db_size_mb = db_path.stat().st_size / 1024**2
    print(f"DB size on disk: {db_size_mb:.1f} MB\n")

    conn = sqlite3.connect(db_path)

    # --- row/size audit ---
    print("=== DB audit ===")
    t0 = time.perf_counter()

    (row_count,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
    t0 = _t(f"COUNT(*) -> {row_count} rows", t0)

    (blob_count,) = conn.execute("SELECT COUNT(*) FROM results WHERE result_stored=1").fetchone()
    t0 = _t(f"results with blobs: {blob_count}", t0)

    (summary_max_kb, summary_avg_kb, summary_total_kb) = conn.execute(
        "SELECT MAX(LENGTH(analysis_summary))/1024.0, AVG(LENGTH(analysis_summary))/1024.0, SUM(LENGTH(analysis_summary))/1024.0 FROM results"
    ).fetchone()
    t0 = _t(
        f"analysis_summary: total={summary_total_kb:.1f} KB, max={summary_max_kb:.2f} KB, avg={summary_avg_kb:.3f} KB",
        t0,
    )

    conn.close()

    # --- simulate status() steps ---
    print("\n=== Timing status() steps ===")
    t0 = time.perf_counter()

    from dimensionality_manuscript import ResultsStore
    store = ResultsStore(db_path)
    t0 = _t("ResultsStore.__init__", t0)

    df = store.summary_table(as_dataframe=True)
    t0 = _t(f"summary_table -> {len(df)} rows", t0)

    snaps = store.list_snapshots()
    t0 = _t(f"list_snapshots -> {len(snaps)} snapshots", t0)

    # --- per-analysis_type blob sizes ---
    print("\n=== Blob size by analysis_type ===")
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT analysis_type, COUNT(*) as n, SUM(result_stored) as has_blob
        FROM results
        GROUP BY analysis_type
        ORDER BY n DESC
        """
    ).fetchall()
    conn.close()

    if rows:
        print(f"  {'analysis_type':<35s} {'rows':>6} {'has_blob':>10}")
        print("  " + "-" * 54)
        for analysis_type, n, has_blob in rows:
            print(f"  {str(analysis_type):<35s} {n:>6} {has_blob:>10}")
    else:
        print("  No rows.")

    # --- blob dir sizes ---
    print("\n=== Blob files ===")
    blob_dir = db_path.parent / "blobs"
    if blob_dir.exists():
        blobs = list(blob_dir.glob("*.pkl"))
        total_blob_mb = sum(b.stat().st_size for b in blobs) / 1024**2
        print(f"  {len(blobs)} pkl files, {total_blob_mb:.1f} MB total")
    else:
        print("  No blob dir.")

    # --- snapshot sizes ---
    print("\n=== Snapshot files ===")
    snap_dir = db_path.parent / "codebase_snapshots"
    if snap_dir.exists():
        snaps = sorted(snap_dir.glob("snapshot_*.zip"))
        total_snap_mb = sum(s.stat().st_size for s in snaps) / 1024**2
        print(f"  {len(snaps)} snapshots, {total_snap_mb:.1f} MB total")
        for s in snaps[-5:]:
            print(f"    {s.name}  {s.stat().st_size/1024**2:.1f} MB")
    else:
        print("  No snapshot dir.")


if __name__ == "__main__":
    main()
