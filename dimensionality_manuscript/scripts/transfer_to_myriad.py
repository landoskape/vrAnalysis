"""Transfer session data (and optionally local results) to the MYRIAD pipeline.

Reads sessions.json and rsyncs only the subdirectories each worker needs:
  oneData/   — neural traces and behavioural variables
  roicat/    — ROI classifier results
  vrExperiment*.json — session config/values

Skips suite2p/, spkmaps/, raw timeline .npy, and .mat files — these are
either already processed into oneData or recomputed on MYRIAD.

Pass --include-results to also upload the local results.db so MYRIAD workers
skip sessions already computed locally (avoids redundant recomputation).

Usage
-----
    # Dry run (see what would be sent):
    python -m dimensionality_manuscript.scripts.transfer_to_myriad \\
        --sessions-file sessions.json \\
        --local-data D:/localData \\
        --host myriad \\
        --remote-data ~/Scratch/data \\
        --dry-run

    # Real transfer (session data only):
    python -m dimensionality_manuscript.scripts.transfer_to_myriad \\
        --sessions-file sessions.json \\
        --local-data D:/localData \\
        --host myriad \\
        --remote-data ~/Scratch/data

    # Also seed MYRIAD with local results so already-computed jobs are skipped:
    python -m dimensionality_manuscript.scripts.transfer_to_myriad \\
        --sessions-file sessions.json \\
        --local-data D:/localData \\
        --host myriad \\
        --remote-data ~/Scratch/data \\
        --include-results
"""

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from dimensionality_manuscript.registry import RegistryPaths


_INCLUDE_SUBDIRS = ["oneData", "roicat"]
_INCLUDE_GLOBS = ["vrExperiment*.json"]
_DEFAULT_REMOTE_DB = "~/Scratch/data/dimensionality-manuscript/cache/pipeline_v2/results.db"


def _posix(path: Path) -> str:
    """Convert a Windows path to MSYS2/Git Bash posix form for rsync.

    D:\\foo\\bar  →  /d/foo/bar
    """
    parts = path.parts
    if len(parts) > 0 and len(parts[0]) == 3 and parts[0][1:] == ":\\":
        drive = parts[0][0].lower()
        rest = "/".join(parts[1:])
        return f"/{drive}/{rest}"
    return path.as_posix()


def build_filter_file(sessions: list[dict], tmp_filter: Path) -> None:
    """Write an rsync filter file covering exactly the needed session paths."""
    lines = []

    # Global analysis files required by all workers (not per-session)
    lines += [
        "+ /analysis/",
        "+ /analysis/roicat_classification/",
        "+ /analysis/roicat_classification/train_classifier.joblib",
    ]

    seen_mice: set[str] = set()
    seen_dates: set[str] = set()

    for s in sessions:
        mouse = s["mouse_name"]
        date = s["date"]
        sid = str(s["session_id"])
        session_rel = f"{mouse}/{date}/{sid}"

        if mouse not in seen_mice:
            lines.append(f"+ /{mouse}/")
            seen_mice.add(mouse)

        date_key = f"{mouse}/{date}"
        if date_key not in seen_dates:
            lines.append(f"+ /{mouse}/{date}/")
            seen_dates.add(date_key)

        lines.append(f"+ /{session_rel}/")

        for subdir in _INCLUDE_SUBDIRS:
            lines.append(f"+ /{session_rel}/{subdir}/")
            lines.append(f"+ /{session_rel}/{subdir}/**")

        for glob in _INCLUDE_GLOBS:
            lines.append(f"+ /{session_rel}/{glob}")

    lines.append("- *")
    tmp_filter.write_text("\n".join(lines) + "\n")


def _find_bash() -> str | None:
    """Find Git Bash's bash.exe."""
    candidates = [
        Path("C:/Program Files/Git/bin/bash.exe"),
        Path("C:/Program Files (x86)/Git/bin/bash.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _merge_dbs(base: Path, other: Path) -> tuple[int, int]:
    """Merge rows from other into base via INSERT OR IGNORE. Returns (rows_before, rows_added)."""
    conn = sqlite3.connect(base, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.execute("ATTACH DATABASE ? AS other", (str(other),))
        (n_before,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        conn.execute("INSERT OR IGNORE INTO results SELECT * FROM other.results")
        conn.commit()
        (n_after,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        conn.execute("DETACH DATABASE other")
        conn.commit()
    finally:
        conn.close()
    return n_before, n_after - n_before


def _count_local_rows_missing_on_remote(local_db: Path, remote_db: Path) -> int:
    """Count local result rows whose ``result_uid`` is absent from remote.

    Read-only: opens ``remote_db`` and attaches ``local_db``; neither file is modified.

    Parameters
    ----------
    local_db : Path
        Local results.db (reference set).
    remote_db : Path
        Remote results.db copy to verify against.

    Returns
    -------
    int
        Number of local rows not found on remote.
    """
    conn = sqlite3.connect(remote_db, timeout=30)
    try:
        conn.execute("ATTACH DATABASE ? AS local", (str(local_db),))
        (n_missing,) = conn.execute(
            """
            SELECT COUNT(*) FROM local.results AS l
            WHERE NOT EXISTS (
                SELECT 1 FROM main.results AS r
                WHERE r.result_uid = l.result_uid
            )
            """
        ).fetchone()
        conn.execute("DETACH DATABASE local")
        return int(n_missing)
    finally:
        conn.close()


def transfer_results(host: str, remote_db: str, dry_run: bool) -> int:
    """Merge local results.db with MYRIAD's and upload the union.

    Downloads the remote DB (if it exists), merges it with local into a temp
    copy, then uploads the merged copy. Neither local nor remote loses rows.
    """
    local_db = RegistryPaths.pipeline_v2_db_path
    if not local_db.exists():
        print(f"No local results.db at {local_db} — skipping results upload.", file=sys.stderr)
        return 0

    print(f"\nResults DB: {local_db} ↔ {host}:{remote_db}")
    if dry_run:
        print("  (dry run — skipping)")
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        merged = tmp_dir / "merged_results.db"
        shutil.copy2(local_db, merged)
        print("Local results.db copied to merged_results.db")

        # Try to download MYRIAD's db and merge its rows into our copy
        tmp_remote = tmp_dir / "myriad_results.db"
        dl = subprocess.run(
            ["scp", f"{host}:{remote_db}", str(tmp_remote)],
            capture_output=True,
        )
        print(f"Remote DB downloaded to {tmp_remote}")
        if dl.returncode == 0:
            n_before, n_added = _merge_dbs(merged, tmp_remote)
            print(f"  Merged {n_added} new rows from remote (local had {n_before} rows)")
        else:
            print("  Remote DB not found — uploading local only")

        # Ensure remote directory exists
        remote_dir = PurePosixPath(remote_db).parent
        subprocess.run(["ssh", host, f"mkdir -p {remote_dir}"], check=True)
        print(f"Remote directory {remote_dir} exists (either created or already existed)")

        # Upload merged result
        ul = subprocess.run(["scp", str(merged), f"{host}:{remote_db}"])
        if ul.returncode != 0:
            print(f"scp upload exited with code {ul.returncode}", file=sys.stderr)

        else:
            tmp_remote_after = tmp_dir / "myriad_results_after.db"
            dl = subprocess.run(
                ["scp", f"{host}:{remote_db}", str(tmp_remote_after)],
                capture_output=True,
            )
            if dl.returncode != 0:
                print(
                    f"  Post-upload verify failed: scp download exited with code {dl.returncode}",
                    file=sys.stderr,
                )
                return 1
            n_missing = _count_local_rows_missing_on_remote(local_db, tmp_remote_after)
            if n_missing:
                print(
                    f"  Post-upload verify failed: {n_missing} local row(s) missing on remote",
                    file=sys.stderr,
                )
                return 1
            print("  Post-upload verify: all local rows present on remote")

        return ul.returncode


def transfer(
    sessions: list[dict],
    local_data: Path,
    host: str,
    remote_data: str,
    dry_run: bool,
) -> int:
    tmp_dir = Path(tempfile.gettempdir())
    tmp_filter = tmp_dir / "myriad_transfer_filter.txt"
    build_filter_file(sessions, tmp_filter)

    src = _posix(local_data).rstrip("/") + "/"
    dst = f"{host}:{remote_data.rstrip('/')}/"

    # rsync --filter needs a posix path to the filter file
    filter_posix = _posix(tmp_filter)

    dry_flag = "--dry-run " if dry_run else ""
    # Single-quote the filter path for bash — handles spaces in Windows temp path
    rsync_cmd = f"rsync -avP {dry_flag}" f"--filter='merge {filter_posix}' " f"{src} {dst}"

    bash = _find_bash()
    if bash:
        cmd = [bash, "-c", rsync_cmd]
    else:
        # Fallback: hope rsync is on PATH and MSYS2 DLLs are accessible
        cmd = ["bash", "-c", rsync_cmd]

    print("rsync filter:")
    print(tmp_filter.read_text())
    print("Command:", rsync_cmd)
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nrsync exited with code {result.returncode}", file=sys.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Transfer session data to MYRIAD")
    parser.add_argument("--sessions-file", type=Path, default=Path("sessions.json"))
    parser.add_argument("--local-data", type=Path, default=None, help="Local data root (e.g. D:/localData); required unless --skip-transfer")
    parser.add_argument("--host", default="myriad", help="SSH host alias (default: myriad)")
    parser.add_argument("--remote-data", default="~/Scratch/data", help="Remote data root (default: ~/Scratch/data)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true", help="Skip rsync step (data already on MYRIAD)")
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Also upload local results.db so MYRIAD workers skip already-computed jobs",
    )
    parser.add_argument(
        "--remote-db",
        default=_DEFAULT_REMOTE_DB,
        help=f"Remote path for results.db (default: {_DEFAULT_REMOTE_DB})",
    )
    args = parser.parse_args()

    if not args.sessions_file.exists():
        print(f"Error: sessions file not found: {args.sessions_file}", file=sys.stderr)
        sys.exit(1)

    sessions = json.loads(args.sessions_file.read_text())
    print(f"Sessions in manifest: {len(sessions)}")

    unique_mice = len({s["mouse_name"] for s in sessions})
    print(f"Unique mice: {unique_mice}")

    if not args.skip_transfer:
        if args.local_data is None:
            print("Error: --local-data is required unless --skip-transfer is set", file=sys.stderr)
            sys.exit(1)
        rc = transfer(sessions, args.local_data, args.host, args.remote_data, args.dry_run)
        if rc != 0:
            sys.exit(rc)

    if args.include_results:
        sys.exit(transfer_results(args.host, args.remote_db, args.dry_run))

    sys.exit(0)


if __name__ == "__main__":
    main()
