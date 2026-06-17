"""Transfer session data (and optionally local results) to the MYRIAD pipeline.

Reads sessions.json and rsyncs only the subdirectories each worker needs:
  oneData/   — neural traces and behavioural variables
  roicat/    — ROI classifier results
  vrExperiment*.json — session config/values

Skips suite2p/, spkmaps/, raw timeline .npy, and .mat files — these are
either already processed into oneData or recomputed on MYRIAD.

Pass --include-results to also upload the local results.db so MYRIAD workers
skip sessions already computed locally (avoids redundant recomputation).

Pass --include-population-cache to upload the local population-registry cache
so MYRIAD workers reuse the same train/test splits instead of regenerating them.

Pass --total-sync to make results.db identical on local and MYRIAD: remote rows
are pulled into local first (so nothing is lost), then the union is pushed back
so both sides match. This is DB-only — blob files are not touched.

Pass --clear-queue to empty the job_batches/job_queue planning tables on MYRIAD.
These grow on every sge_submit run; clearing them leaves results untouched.
When combined with --total-sync the queue is cleared as part of the same upload.

Usage
-----
    # Dry run (see what would be sent):
    python -m dimensionality_manuscript.scripts.transfer_to_myriad --sessions-file sessions.json --local-data D:/localData --host myriad --remote-data ~/Scratch/data --dry-run

    # Real transfer (session data only):
    python -m dimensionality_manuscript.scripts.transfer_to_myriad --sessions-file sessions.json --local-data D:/localData --host myriad --remote-data ~/Scratch/data

    # Also seed MYRIAD with local results so already-computed jobs are skipped:
    python -m dimensionality_manuscript.scripts.transfer_to_myriad --sessions-file sessions.json --local-data D:/localData --host myriad --remote-data ~/Scratch/data --include-results

    # Upload population cache so MYRIAD uses the same train/test splits:
    python -m dimensionality_manuscript.scripts.transfer_to_myriad --sessions-file sessions.json --local-data D:/localData --host myriad --remote-data ~/Scratch/data --include-population-cache

    # Reset: make results.db identical on both sides and clear the MYRIAD queue
    # (no session transfer; preview first with --dry-run):
    python -m dimensionality_manuscript.scripts.transfer_to_myriad --skip-transfer --host myriad --total-sync --clear-queue --dry-run
    python -m dimensionality_manuscript.scripts.transfer_to_myriad --skip-transfer --host myriad --total-sync --clear-queue

    # Just clear the MYRIAD job queue (results untouched):
    python -m dimensionality_manuscript.scripts.transfer_to_myriad --skip-transfer --host myriad --clear-queue
"""

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from dimensionality_manuscript.pipeline import JobQueue
from dimensionality_manuscript.registry import RegistryPaths


_INCLUDE_SUBDIRS = ["oneData", "roicat"]
_INCLUDE_GLOBS = ["vrExperiment*.json"]
_DEFAULT_REMOTE_DB = "~/Scratch/data/dimensionality-manuscript/cache/pipeline_v2/results.db"
_DEFAULT_REMOTE_CACHE = "~/Scratch/data/dimensionality-manuscript/cache"


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
        # Also merge errors table if other DB has one
        (has_other_errors,) = conn.execute("SELECT COUNT(*) FROM other.sqlite_master WHERE type='table' AND name='errors'").fetchone()
        if has_other_errors:
            conn.execute("INSERT OR IGNORE INTO errors SELECT * FROM other.errors")
            conn.commit()
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


def _db_result_count(db_path: Path) -> int:
    """Return the number of rows in the ``results`` table (0 if file/table absent)."""
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        (n,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        return int(n)
    finally:
        conn.close()


def _clear_queue_tables(db_path: Path) -> tuple[int, int]:
    """Empty the ``job_queue``/``job_batches`` tables of a db file.

    Returns ``(n_jobs, n_batches)`` cleared.
    """
    return JobQueue(db_path).clear()


def total_sync(host: str, remote_db: str, dry_run: bool, clear_queue: bool = False) -> int:
    """Make ``results.db`` identical on local and MYRIAD via a union of rows.

    Downloads the remote DB, merges remote rows into the local DB (so local
    becomes the union), then uploads a copy whose rows are the same union (so
    remote becomes identical to local). The remote queue tables are preserved
    unless ``clear_queue`` is True. Blob files are not touched (DB-only).

    Parameters
    ----------
    host : str
        SSH host alias for MYRIAD.
    remote_db : str
        Path to ``results.db`` on MYRIAD.
    dry_run : bool
        If True, report counts and make no changes.
    clear_queue : bool
        If True, empty the queue tables in the uploaded DB so the remote queue
        is cleared as part of the sync.

    Returns
    -------
    int
        Process-style return code (0 on success).
    """
    local_db = RegistryPaths.pipeline_v2_db_path
    if not local_db.exists():
        print(f"No local results.db at {local_db} — nothing to sync.", file=sys.stderr)
        return 0

    print(f"\nTotal sync (union): {local_db} <-> {host}:{remote_db}")
    n_local = _db_result_count(local_db)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        tmp_remote = tmp_dir / "myriad_results.db"
        dl = subprocess.run(["scp", f"{host}:{remote_db}", str(tmp_remote)], capture_output=True)
        remote_present = dl.returncode == 0

        if remote_present:
            n_remote = _db_result_count(tmp_remote)
            new_to_remote = _count_local_rows_missing_on_remote(local_db, tmp_remote)
            new_to_local = _count_local_rows_missing_on_remote(tmp_remote, local_db)
            n_union = n_local + new_to_local
            print(f"  local={n_local}  remote={n_remote}  union={n_union}")
            print(f"  new -> remote: {new_to_remote}   new -> local: {new_to_local}")
        else:
            print("  Remote DB not found — local will be uploaded as the full DB")
            n_union = n_local
            print(f"  local={n_local}  remote=0  union={n_union}")

        if clear_queue:
            print("  queue tables on remote will be cleared")

        if dry_run:
            print("  (dry run — no changes made)")
            return 0

        # ── Make local the union ──────────────────────────────────────────────
        if remote_present:
            n_before, n_added = _merge_dbs(local_db, tmp_remote)
            print(f"  Merged {n_added} remote row(s) into local (local had {n_before})")

        # ── Build the upload DB (union results, remote queue preserved) ───────
        upload_db = tmp_dir / "upload_results.db"
        if remote_present:
            shutil.copy2(tmp_remote, upload_db)
            _merge_dbs(upload_db, local_db)
        else:
            shutil.copy2(local_db, upload_db)

        if clear_queue:
            n_jobs, n_batches = _clear_queue_tables(upload_db)
            print(f"  Cleared remote queue: {n_jobs} job row(s) across {n_batches} batch(es)")

        # ── Upload and verify ─────────────────────────────────────────────────
        remote_dir = PurePosixPath(remote_db).parent
        subprocess.run(["ssh", host, f"mkdir -p {remote_dir}"], check=True)
        ul = subprocess.run(["scp", str(upload_db), f"{host}:{remote_db}"])
        if ul.returncode != 0:
            print(f"scp upload exited with code {ul.returncode}", file=sys.stderr)
            return ul.returncode

        tmp_after = tmp_dir / "myriad_results_after.db"
        dl = subprocess.run(["scp", f"{host}:{remote_db}", str(tmp_after)], capture_output=True)
        if dl.returncode != 0:
            print(f"  Post-upload verify failed: scp download exited with code {dl.returncode}", file=sys.stderr)
            return 1
        n_missing = _count_local_rows_missing_on_remote(local_db, tmp_after)
        if n_missing:
            print(f"  Post-upload verify failed: {n_missing} local row(s) missing on remote", file=sys.stderr)
            return 1
        print(f"  Post-upload verify: results identical on both sides ({_db_result_count(tmp_after)} rows)")
        return 0


def clear_remote_queue(host: str, remote_db: str, dry_run: bool) -> int:
    """Empty the ``job_batches``/``job_queue`` planning tables on MYRIAD.

    Downloads the remote ``results.db``, clears the queue tables, and uploads
    it back. Result rows are left untouched.

    Parameters
    ----------
    host : str
        SSH host alias for MYRIAD.
    remote_db : str
        Path to ``results.db`` on MYRIAD.
    dry_run : bool
        If True, report what would be cleared and make no changes.

    Returns
    -------
    int
        Process-style return code (0 on success).
    """
    print(f"\nClear remote queue: {host}:{remote_db}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_remote = Path(tmpdir) / "myriad_results.db"
        dl = subprocess.run(["scp", f"{host}:{remote_db}", str(tmp_remote)], capture_output=True)
        if dl.returncode != 0:
            print(f"  Remote DB not found at {host}:{remote_db} — nothing to clear.", file=sys.stderr)
            return 0

        queue = JobQueue(tmp_remote)
        n_batches = len(queue.list_batches())
        n_jobs = sum(queue.status_summary().values())
        print(f"  Remote queue: {n_batches} batch(es), {n_jobs} job row(s)")

        if dry_run:
            print("  (dry run — not clearing)")
            return 0

        n_jobs_cleared, n_batches_cleared = queue.clear()
        print(f"  Cleared {n_jobs_cleared} job row(s) across {n_batches_cleared} batch(es)")

        remote_dir = PurePosixPath(remote_db).parent
        subprocess.run(["ssh", host, f"mkdir -p {remote_dir}"], check=True)
        ul = subprocess.run(["scp", str(tmp_remote), f"{host}:{remote_db}"])
        if ul.returncode != 0:
            print(f"scp upload exited with code {ul.returncode}", file=sys.stderr)
        return ul.returncode


def transfer_population_cache(host: str, remote_cache: str, dry_run: bool) -> int:
    """Rsync the local population-registry cache to MYRIAD.

    The population registry holds train/test split indices (one .joblib per session).
    Uploading it ensures MYRIAD workers reuse the same splits rather than regenerating
    them randomly, which would make cross-analysis comparisons inconsistent.
    """
    local_registry = RegistryPaths().registry_path
    if not local_registry.exists():
        print(f"No local population-registry at {local_registry} — skipping.", file=sys.stderr)
        return 0

    n_files = sum(1 for _ in local_registry.glob("*.joblib"))
    print(f"\nPopulation registry: {local_registry} ({n_files} .joblib files)")

    remote_registry = remote_cache.rstrip("/") + "/population-registry/"
    src = _posix(local_registry).rstrip("/") + "/"
    dst = f"{host}:{remote_registry}"

    dry_flag = "--dry-run " if dry_run else ""
    rsync_cmd = f"rsync -avP {dry_flag}{src} {dst}"

    # Ensure remote directory exists
    if not dry_run:
        subprocess.run(["ssh", host, f"mkdir -p {remote_registry}"], check=True)

    bash = _find_bash()
    cmd = [bash, "-c", rsync_cmd] if bash else ["bash", "-c", rsync_cmd]

    print("Command:", rsync_cmd)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nrsync exited with code {result.returncode}", file=sys.stderr)
    return result.returncode


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
    parser.add_argument(
        "--include-population-cache",
        action="store_true",
        help="Upload local population-registry cache so MYRIAD workers reuse the same train/test splits",
    )
    parser.add_argument(
        "--remote-cache",
        default=_DEFAULT_REMOTE_CACHE,
        help=f"Remote path for the manuscript cache directory (default: {_DEFAULT_REMOTE_CACHE})",
    )
    parser.add_argument(
        "--total-sync",
        action="store_true",
        help="Make results.db identical on local and MYRIAD via a union of rows (DB-only; blobs untouched)",
    )
    parser.add_argument(
        "--clear-queue",
        action="store_true",
        help="Empty the job_batches/job_queue planning tables on MYRIAD (they grow every sge_submit run)",
    )
    args = parser.parse_args()

    if not args.skip_transfer:
        if not args.sessions_file.exists():
            print(f"Error: sessions file not found: {args.sessions_file}", file=sys.stderr)
            sys.exit(1)
        if args.local_data is None:
            print("Error: --local-data is required unless --skip-transfer is set", file=sys.stderr)
            sys.exit(1)
        sessions = json.loads(args.sessions_file.read_text())
        print(f"Sessions in manifest: {len(sessions)}")
        print(f"Unique mice: {len({s['mouse_name'] for s in sessions})}")
        rc = transfer(sessions, args.local_data, args.host, args.remote_data, args.dry_run)
        if rc != 0:
            sys.exit(rc)

    if args.total_sync:
        rc = total_sync(args.host, args.remote_db, args.dry_run, clear_queue=args.clear_queue)
        if rc != 0:
            sys.exit(rc)
    elif args.clear_queue:
        rc = clear_remote_queue(args.host, args.remote_db, args.dry_run)
        if rc != 0:
            sys.exit(rc)

    if args.include_results:
        rc = transfer_results(args.host, args.remote_db, args.dry_run)
        if rc != 0:
            sys.exit(rc)

    if args.include_population_cache:
        sys.exit(transfer_population_cache(args.host, args.remote_cache, args.dry_run))

    sys.exit(0)


if __name__ == "__main__":
    main()
