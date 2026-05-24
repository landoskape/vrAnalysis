"""Merge MYRIAD results into the local results store.

Run this locally after MYRIAD jobs finish. It:
1. rsyncs blob .pkl files from MYRIAD into the local blobs directory.
2. Downloads the MYRIAD results.db and merges new rows into the local DB.

After syncing, ResultsStore and ResultsAggregator work transparently — they
see the merged results exactly as if everything had run locally.

Usage
-----
    python -m dimensionality_manuscript.scripts.sync_from_myriad \\
        --host myriad \\
        --remote-db ~/Scratch/dim_manuscript/pipeline_v2/results.db \\
        --remote-blobs ~/Scratch/dim_manuscript/pipeline_v2/blobs/ \\
        [--dry-run]

The --host value is whatever SSH alias you use (myriad, ucl-myriad, etc.).
Set up ~/.ssh/config with a Host entry to avoid typing the full hostname.
"""

import argparse
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from dimensionality_manuscript.registry import RegistryPaths

REGISTRY_PATHS = RegistryPaths()


def sync(
    host: str,
    remote_db: str,
    remote_blobs: str,
    dry_run: bool = False,
) -> None:
    """Merge MYRIAD results into the local store.

    Parameters
    ----------
    host : str
        SSH host alias for MYRIAD (from ~/.ssh/config).
    remote_db : str
        Path to results.db on MYRIAD (may use ~ expansion).
    remote_blobs : str
        Path to blobs/ directory on MYRIAD (trailing slash optional).
    dry_run : bool
        If True, show what would happen without transferring or merging.
    """
    local_db = REGISTRY_PATHS.pipeline_v2_db_path
    local_blobs = local_db.parent / "blobs"
    local_blobs.mkdir(parents=True, exist_ok=True)

    print(f"Local DB:     {local_db}")
    print(f"Local blobs:  {local_blobs}")
    print(f"Remote host:  {host}")
    print(f"Remote DB:    {remote_db}")
    print(f"Remote blobs: {remote_blobs}")
    print()

    # Ensure trailing slash on remote blobs path so rsync merges contents
    remote_blobs_src = remote_blobs.rstrip("/") + "/"

    # ── Step 1: sync blobs ────────────────────────────────────────────────────
    rsync_cmd = [
        "rsync",
        "-avP",
        "--ignore-existing",   # skip blobs already present locally
        f"{host}:{remote_blobs_src}",
        str(local_blobs) + "/",
    ]
    print(f"Syncing blobs: {' '.join(rsync_cmd)}")
    if dry_run:
        rsync_cmd.insert(1, "--dry-run")
    result = subprocess.run(rsync_cmd)
    if result.returncode != 0:
        print("rsync failed.", file=sys.stderr)
        raise SystemExit(1)
    print()

    # ── Step 2: download remote results.db ───────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "myriad_results.db"
        scp_cmd = ["scp", f"{host}:{remote_db}", str(tmp_db)]
        print(f"Downloading DB: {' '.join(scp_cmd)}")
        if not dry_run:
            result = subprocess.run(scp_cmd)
            if result.returncode != 0:
                print("scp failed.", file=sys.stderr)
                raise SystemExit(1)
        else:
            print("[dry-run] skipping download")
            return

        # ── Step 3: merge results table ───────────────────────────────────────
        print(f"\nMerging results from {tmp_db} → {local_db}")
        n_before, n_added = _merge_results(local_db, tmp_db)
        print(f"Rows before: {n_before}  |  New rows added: {n_added}  |  Total: {n_before + n_added}")


def _merge_results(local_db: Path, remote_db: Path) -> tuple[int, int]:
    """Merge results table from remote_db into local_db.

    Uses SQLite ATTACH so no intermediate files are needed.

    Returns
    -------
    tuple[int, int]
        (rows_before, rows_added)
    """
    conn = sqlite3.connect(local_db, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.execute(f"ATTACH DATABASE ? AS remote", (str(remote_db),))
        (n_before,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        conn.execute(
            "INSERT OR IGNORE INTO results SELECT * FROM remote.results"
        )
        conn.commit()
        (n_after,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        conn.execute("DETACH DATABASE remote")
        conn.commit()
    finally:
        conn.close()
    return n_before, n_after - n_before


def main():
    parser = argparse.ArgumentParser(
        description="Merge MYRIAD results into local store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", required=True, help="SSH alias for MYRIAD (from ~/.ssh/config)")
    parser.add_argument(
        "--remote-db",
        default="~/Scratch/dim_manuscript/pipeline_v2/results.db",
        help="Path to results.db on MYRIAD (default: ~/Scratch/dim_manuscript/pipeline_v2/results.db)",
    )
    parser.add_argument(
        "--remote-blobs",
        default="~/Scratch/dim_manuscript/pipeline_v2/blobs/",
        help="Path to blobs/ on MYRIAD (default: ~/Scratch/dim_manuscript/pipeline_v2/blobs/)",
    )
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be done without transferring")
    args = parser.parse_args()

    sync(
        host=args.host,
        remote_db=args.remote_db,
        remote_blobs=args.remote_blobs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
