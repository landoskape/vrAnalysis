"""Merge MYRIAD results into the local results store.

Run this locally after MYRIAD jobs finish. It:
1. rsyncs blob .pkl files from MYRIAD into the local blobs directory.
2. Downloads the MYRIAD results.db and merges new rows into the local DB.
3. Optionally rsyncs self-cached model joblib files (regression/subspace scores) when --include-model-caches is passed.

After syncing, ResultsStore and ResultsAggregator work transparently — they
see the merged results exactly as if everything had run locally.

Usage
-----
    # Preview what would be transferred (no changes made):
    python -m dimensionality_manuscript.scripts.sync_from_myriad --host myriad --dry-run [--include-model-caches]

    # Check for local/remote collisions before syncing:
    python -m dimensionality_manuscript.scripts.sync_from_myriad --host myriad --check-overwrite [--include-model-caches]

    # Real sync (blobs + DB):
    python -m dimensionality_manuscript.scripts.sync_from_myriad --host myriad

    # Also pull back regression/subspace joblib caches:
    python -m dimensionality_manuscript.scripts.sync_from_myriad --host myriad --include-model-caches

The --host value is whatever SSH alias you use (myriad, ucl-myriad, etc.).
Set up ~/.ssh/config with a Host entry to avoid typing the full hostname.
"""

import argparse
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from dimensionality_manuscript.registry import RegistryPaths


def _find_bash() -> str | None:
    candidates = [
        Path("C:/Program Files/Git/bin/bash.exe"),
        Path("C:/Program Files (x86)/Git/bin/bash.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _posix(path: Path) -> str:
    """Convert a Windows path to MSYS2/Git Bash posix form for rsync."""
    parts = path.parts
    if len(parts) > 0 and len(parts[0]) == 3 and parts[0][1:] == ":\\":
        drive = parts[0][0].lower()
        rest = "/".join(parts[1:])
        return f"/{drive}/{rest}"
    return path.as_posix()


def _format_size(n_bytes: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def _remote_file_listing(host: str, remote_path: str, pattern: str = "*") -> list[tuple[str, int]]:
    """Return (filename, size_bytes) for files matching pattern in a remote dir via SSH find."""
    result = subprocess.run(
        ["ssh", host, f"find {remote_path} -maxdepth 1 -name '{pattern}' -printf '%f %s\\n' 2>/dev/null || true"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  SSH listing failed for {remote_path}: {result.stderr.strip()}", file=sys.stderr)
        return []
    entries = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                entries.append((parts[0], int(parts[-1])))
            except ValueError:
                pass
    return entries


def _download_remote_db(host: str, remote_db: str, tmp_dir: Path) -> Path | None:
    """Download remote results.db to tmp_dir. Returns local path or None if not found."""
    tmp_db = tmp_dir / "myriad_results.db"
    result = subprocess.run(["scp", f"{host}:{remote_db}", str(tmp_db)])
    if result.returncode != 0:
        return None
    return tmp_db


REGISTRY_PATHS = RegistryPaths()

_DEFAULT_REMOTE_CACHE = "~/Scratch/data/dimensionality-manuscript/cache"

# (subdir name on remote, accessor for local RegistryPaths attribute)
_MODEL_CACHE_SUBDIRS: list[tuple[str, str]] = [
    ("scores", "score_path"),
    ("hyperparameters", "hyperparameter_path"),
    ("subspace-scores", "subspace_score_path"),
]


# ── Preview (dry-run) ─────────────────────────────────────────────────────────


def _has_errors_table(conn, schema: str = "main") -> bool:
    """Return True if the given schema (main or remote) has an errors table."""
    (n,) = conn.execute(f"SELECT COUNT(*) FROM {schema}.sqlite_master WHERE type='table' AND name='errors'").fetchone()
    return bool(n)


def _print_db_delta(local_db: Path, remote_db: Path) -> None:
    """Print per-config summary of new rows in remote_db relative to local_db."""
    if not local_db.exists():
        conn = sqlite3.connect(str(remote_db), timeout=30)
        try:
            (n,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        finally:
            conn.close()
        print(f"  local DB absent — all {n} remote rows are new")
        return

    conn = sqlite3.connect(str(local_db), timeout=30)
    try:
        conn.execute("ATTACH DATABASE ? AS remote", (str(remote_db),))
        (n_local,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        (n_remote,) = conn.execute("SELECT COUNT(*) FROM remote.results").fetchone()
        (n_new,) = conn.execute("SELECT COUNT(*) FROM remote.results WHERE result_uid NOT IN (SELECT result_uid FROM results)").fetchone()
        print(f"  Results:  local={n_local}  remote={n_remote}  new={n_new}")
        if n_new > 0:
            rows = conn.execute("""
                SELECT analysis_type, schema_version,
                       COUNT(*) AS n, SUM(result_stored) AS stored,
                       COUNT(DISTINCT session_id) AS sessions
                FROM remote.results
                WHERE result_uid NOT IN (SELECT result_uid FROM results)
                GROUP BY analysis_type, schema_version
                ORDER BY analysis_type, schema_version
                """).fetchall()
            print()
            print(f"  {'analysis_type':<25} {'ver':<6} {'rows':>6} {'blobs':>6} {'sessions':>9}")
            print("  " + "-" * 58)
            for atype, sver, n, stored, nsess in rows:
                print(f"  {(atype or '?'):<25} {(sver or '?'):<6} {n:>6} {(stored or 0):>6} {nsess:>9}")
        if _has_errors_table(conn, "remote"):
            (n_err_remote,) = conn.execute("SELECT COUNT(*) FROM remote.errors").fetchone()
            if _has_errors_table(conn, "main"):
                (n_err_local,) = conn.execute("SELECT COUNT(*) FROM errors").fetchone()
                (n_err_new,) = conn.execute("SELECT COUNT(*) FROM remote.errors WHERE result_uid NOT IN (SELECT result_uid FROM errors)").fetchone()
            else:
                n_err_local = 0
                n_err_new = n_err_remote
            print(f"\n  Errors:   local={n_err_local}  remote={n_err_remote}  new={n_err_new}")
        conn.execute("DETACH DATABASE remote")
    finally:
        conn.close()


def _print_blob_delta(local_blobs: Path, remote_entries: list[tuple[str, int]]) -> None:
    local_names = {p.name for p in local_blobs.glob("*.pkl")} if local_blobs.exists() else set()
    new_entries = [(f, s) for f, s in remote_entries if f not in local_names]
    skipped = len({f for f, _ in remote_entries} & local_names)
    total_remote = sum(s for _, s in remote_entries)
    total_new = sum(s for _, s in new_entries)
    print(f"  remote={len(remote_entries)} files ({_format_size(total_remote)})")
    print(f"  local={len(local_names)} files")
    line = f"  new={len(new_entries)} files ({_format_size(total_new)})"
    if skipped:
        line += f"  skip={skipped} (already exist locally)"
    print(line)


def preview_sync(
    host: str,
    remote_db: str,
    remote_blobs: str,
    remote_cache: str = _DEFAULT_REMOTE_CACHE,
    include_model_caches: bool = False,
) -> None:
    """Print a summary of what sync would transfer without making any changes."""
    local_db = REGISTRY_PATHS.pipeline_v2_db_path
    local_blobs = local_db.parent / "blobs"

    print("=== Sync preview (dry run) ===")
    print(f"host={host}  local_db={local_db}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Downloading remote DB for preview...")
        tmp_db = _download_remote_db(host, remote_db, Path(tmpdir))
        print()
        print("DB delta:")
        if tmp_db is None:
            print("  Remote DB not found — nothing to add.")
        else:
            _print_db_delta(local_db, tmp_db)

        print()
        print("Blobs:")
        remote_blob_entries = _remote_file_listing(host, remote_blobs.rstrip("/"), "*.pkl")
        _print_blob_delta(local_blobs, remote_blob_entries)

        if include_model_caches:
            registry_paths = RegistryPaths()
            remote_base = remote_cache.rstrip("/")
            print()
            print("Model caches:")
            print(f"  {'dir':<38} {'new':>5}  {'size':>10}  {'skip':>5}")
            print("  " + "-" * 63)
            for name, attr in _MODEL_CACHE_SUBDIRS:
                local_path: Path = getattr(registry_paths, attr)
                remote_entries = _remote_file_listing(host, f"{remote_base}/{name}", "*.joblib")
                local_names = {p.name for p in local_path.glob("*.joblib")} if local_path.exists() else set()
                new_entries = [(f, s) for f, s in remote_entries if f not in local_names]
                skipped = len({f for f, _ in remote_entries} & local_names)
                total_new = sum(s for _, s in new_entries)
                print(f"  {name + '/':38} {len(new_entries):>5}  {_format_size(total_new):>10}  {skipped:>5}")


# ── Overwrite check ───────────────────────────────────────────────────────────


def _check_db_collisions(local_db: Path, remote_db: Path) -> None:
    """Report result_uid collisions between local and remote DB."""
    print("DB:")
    if not local_db.exists():
        conn = sqlite3.connect(str(remote_db), timeout=30)
        try:
            (n_remote,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        finally:
            conn.close()
        print(f"  Local DB absent — {n_remote} remote rows all new, no collisions")
        return

    conn = sqlite3.connect(str(local_db), timeout=30)
    try:
        conn.execute("ATTACH DATABASE ? AS remote", (str(remote_db),))
        (n_local,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        (n_remote,) = conn.execute("SELECT COUNT(*) FROM remote.results").fetchone()
        (n_shared,) = conn.execute("SELECT COUNT(*) FROM results WHERE result_uid IN (SELECT result_uid FROM remote.results)").fetchone()
        n_union = n_local + n_remote - n_shared
        print(f"  Results:  local={n_local}  remote={n_remote}  shared={n_shared}  union={n_union}")
        if n_shared == 0:
            print("  union == local+remote — no overlaps, all remote rows are new")
        else:
            print(f"  union < local+remote — {n_shared} remote row(s) share a result_uid " f"with local (INSERT OR IGNORE skips them)")
            diffs = conn.execute("""
                SELECT r.result_uid, r.analysis_type, r.result_stored, rr.result_stored
                FROM results r
                JOIN remote.results rr ON r.result_uid = rr.result_uid
                WHERE r.result_stored != rr.result_stored
                   OR coalesce(r.analysis_summary, '') != coalesce(rr.analysis_summary, '')
                LIMIT 20
                """).fetchall()
            if diffs:
                print(f"  WARNING: {len(diffs)} shared row(s) have differing content (local wins)")
                for uid, atype, ls, rs in diffs[:5]:
                    print(f"    {uid} ({atype}): result_stored local={ls} remote={rs}")
            else:
                print(f"  All {n_shared} shared rows have identical content — safe to skip")
        if _has_errors_table(conn, "remote"):
            (n_err_remote,) = conn.execute("SELECT COUNT(*) FROM remote.errors").fetchone()
            if _has_errors_table(conn, "main"):
                (n_err_local,) = conn.execute("SELECT COUNT(*) FROM errors").fetchone()
                (n_err_shared,) = conn.execute("SELECT COUNT(*) FROM errors WHERE result_uid IN (SELECT result_uid FROM remote.errors)").fetchone()
            else:
                n_err_local = 0
                n_err_shared = 0
            print(f"  Errors:   local={n_err_local}  remote={n_err_remote}  shared={n_err_shared}")
        conn.execute("DETACH DATABASE remote")
    finally:
        conn.close()


def _check_file_collisions(label: str, local_dir: Path, host: str, remote_path: str, pattern: str) -> None:
    """Report filename collisions between a local directory and a remote directory."""
    print(f"{label}:")
    remote_entries = _remote_file_listing(host, remote_path, pattern)
    local_names = {p.name for p in local_dir.glob(pattern)} if local_dir.exists() else set()
    remote_names = {f for f, _ in remote_entries}
    shared = remote_names & local_names
    new_only = remote_names - local_names
    print(f"  remote={len(remote_names)}  local={len(local_names)}  shared={len(shared)}  new={len(new_only)}")
    if not shared:
        print("  union == remote+local — no overlaps")
    else:
        print(f"  union < remote+local — {len(shared)} file(s) skipped by --ignore-existing")
        print("  Note: same filename = same result_uid = same computation, content should be identical")


def check_overwrite(
    host: str,
    remote_db: str,
    remote_blobs: str,
    remote_cache: str = _DEFAULT_REMOTE_CACHE,
    include_model_caches: bool = False,
) -> None:
    """Report local/remote collisions without making any changes."""
    local_db = REGISTRY_PATHS.pipeline_v2_db_path
    local_blobs = local_db.parent / "blobs"

    print("=== Overwrite check ===")
    print(f"host={host}  local_db={local_db}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Downloading remote DB for inspection...")
        tmp_db = _download_remote_db(host, remote_db, Path(tmpdir))
        print()
        if tmp_db is None:
            print("DB: remote DB not found")
        else:
            _check_db_collisions(local_db, tmp_db)

        print()
        _check_file_collisions("blobs", local_blobs, host, remote_blobs.rstrip("/"), "*.pkl")

        if include_model_caches:
            registry_paths = RegistryPaths()
            remote_base = remote_cache.rstrip("/")
            for name, attr in _MODEL_CACHE_SUBDIRS:
                print()
                _check_file_collisions(
                    name,
                    getattr(registry_paths, attr),
                    host,
                    f"{remote_base}/{name}",
                    "*.joblib",
                )


# ── Real sync ─────────────────────────────────────────────────────────────────


def sync(host: str, remote_db: str, remote_blobs: str) -> None:
    """Merge MYRIAD blobs and results.db into the local store."""
    local_db = REGISTRY_PATHS.pipeline_v2_db_path
    local_blobs = local_db.parent / "blobs"
    local_blobs.mkdir(parents=True, exist_ok=True)

    print(f"Local DB:     {local_db}")
    print(f"Local blobs:  {local_blobs}")
    print(f"Remote host:  {host}")
    print(f"Remote DB:    {remote_db}")
    print(f"Remote blobs: {remote_blobs}")
    print()

    # ── Step 1: sync blobs ────────────────────────────────────────────────────
    remote_blobs_src = remote_blobs.rstrip("/") + "/"
    rsync_shell_cmd = f"rsync -a --info=progress2 --partial --ignore-existing " f"{host}:{remote_blobs_src} {_posix(local_blobs)}/"
    bash = _find_bash()
    print("Syncing blobs (overall progress; per-file hex names suppressed)...")
    result = subprocess.run([bash or "bash", "-c", rsync_shell_cmd])
    if result.returncode != 0:
        print("rsync failed.", file=sys.stderr)
        raise SystemExit(1)
    print()

    # ── Step 2: download and merge results.db ─────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "myriad_results.db"
        scp_cmd = ["scp", f"{host}:{remote_db}", str(tmp_db)]
        print(f"Downloading DB: {' '.join(scp_cmd)}")
        result = subprocess.run(scp_cmd)
        if result.returncode != 0:
            print("scp failed.", file=sys.stderr)
            raise SystemExit(1)

        print(f"\nMerging results from {tmp_db} → {local_db}")
        report = _merge_results(local_db, tmp_db)
        _print_merge_report(report)


@dataclass
class MergeReport:
    """Summary of a results/errors merge.

    ``new_by_config`` / ``err_by_config`` hold per-config breakdowns of the rows
    that were actually added, each a tuple of
    ``(analysis_type, schema_version, analysis_summary, n_rows, n_blobs, n_sessions)``.
    """

    results_before: int = 0
    results_added: int = 0
    errors_before: int = 0
    errors_added: int = 0
    new_by_config: list[tuple] = field(default_factory=list)
    err_by_config: list[tuple] = field(default_factory=list)


# Group newly-added rows by config for a status.py --by-config style report.
# {blobs} is SUM(result_stored) for the results table (which has blobs) or a
# literal 0 for the errors table (which does not).
_NEW_BY_CONFIG_SQL = """
    SELECT analysis_type, schema_version, analysis_summary,
           COUNT(*) AS n, {blobs} AS blobs, COUNT(DISTINCT session_id) AS sessions
    FROM remote.{table}
    WHERE result_uid NOT IN (SELECT result_uid FROM main.{table})
    GROUP BY analysis_type, schema_version, analysis_summary
    ORDER BY analysis_type, n DESC
"""


def _merge_results(local_db: Path, remote_db: Path) -> MergeReport:
    """Merge results and errors tables from remote_db into local_db.

    Rows are added with ``INSERT OR IGNORE`` (existing local rows win). The
    returned :class:`MergeReport` records how many rows were added and a
    per-config breakdown of just the new rows.
    """
    report = MergeReport()
    conn = sqlite3.connect(local_db, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.execute("ATTACH DATABASE ? AS remote", (str(remote_db),))
        (report.results_before,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        # Capture the by-config breakdown BEFORE inserting (afterwards the rows
        # are no longer "new" relative to local).
        report.new_by_config = conn.execute(_NEW_BY_CONFIG_SQL.format(table="results", blobs="SUM(result_stored)")).fetchall()
        conn.execute("INSERT OR IGNORE INTO results SELECT * FROM remote.results")
        conn.commit()
        (n_after,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        report.results_added = n_after - report.results_before

        if _has_errors_table(conn, "remote"):
            has_local_errors = _has_errors_table(conn, "main")
            if has_local_errors:
                (report.errors_before,) = conn.execute("SELECT COUNT(*) FROM errors").fetchone()
                report.err_by_config = conn.execute(_NEW_BY_CONFIG_SQL.format(table="errors", blobs="0")).fetchall()
            conn.execute("INSERT OR IGNORE INTO errors SELECT * FROM remote.errors")
            conn.commit()
            if has_local_errors:
                (n_err_after,) = conn.execute("SELECT COUNT(*) FROM errors").fetchone()
                report.errors_added = n_err_after - report.errors_before

        conn.execute("DETACH DATABASE remote")
        conn.commit()
    finally:
        conn.close()
    return report


def _print_by_config_rows(rows: list[tuple], *, with_blobs: bool) -> None:
    """Print a status.py --by-config style table for merge breakdown rows."""
    for i, (atype, schema, summary, n, blobs, sessions) in enumerate(rows, 1):
        parts = [f"  [{i:>3}] {atype or '?'} {schema or '?'} | {summary or '?'} | {n} rows"]
        if with_blobs:
            parts.append(f"{blobs or 0} blobs")
        parts.append(f"{sessions} sessions")
        print(" | ".join(parts))


def _print_merge_report(report: MergeReport) -> None:
    """Print merge totals plus a per-config breakdown of newly-added rows."""
    total = report.results_before + report.results_added
    print(f"Results — before: {report.results_before}  new: {report.results_added}  total: {total}")
    if report.new_by_config:
        n_configs = len(report.new_by_config)
        print(f"\n  New results by config: {report.results_added} rows | {n_configs} configs")
        _print_by_config_rows(report.new_by_config, with_blobs=True)

    if report.errors_before or report.errors_added:
        err_total = report.errors_before + report.errors_added
        print(f"\nErrors  — before: {report.errors_before}  new: {report.errors_added}  total: {err_total}")
        if report.err_by_config:
            print()
            _print_by_config_rows(report.err_by_config, with_blobs=False)


def sync_model_caches(host: str, remote_cache: str) -> None:
    """Rsync regression/subspace joblib caches from MYRIAD to local.

    Pulls scores/, hyperparameters/, and subspace-scores/ from the remote cache root. Uses
    --ignore-existing so local files are never overwritten.
    """
    registry_paths = RegistryPaths()
    remote_base = remote_cache.rstrip("/")
    bash = _find_bash()
    for name, attr in _MODEL_CACHE_SUBDIRS:
        local_path: Path = getattr(registry_paths, attr)
        local_path.mkdir(parents=True, exist_ok=True)
        remote_src = f"{host}:{remote_base}/{name}/"
        rsync_shell_cmd = f"rsync -a --info=progress2 --partial --ignore-existing {remote_src} {_posix(local_path)}/"
        print(f"Syncing {name}/ (overall progress)...")
        result = subprocess.run([bash or "bash", "-c", rsync_shell_cmd])
        if result.returncode != 0:
            print(f"rsync failed for {name}/.", file=sys.stderr)
            raise SystemExit(1)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Merge MYRIAD results into local store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", required=True, help="SSH alias for MYRIAD (from ~/.ssh/config)")
    parser.add_argument(
        "--remote-db",
        default="~/Scratch/data/dimensionality-manuscript/cache/pipeline_v2/results.db",
        help="Path to results.db on MYRIAD (default: ~/Scratch/data/dimensionality-manuscript/cache/pipeline_v2/results.db)",
    )
    parser.add_argument(
        "--remote-blobs",
        default="~/Scratch/data/dimensionality-manuscript/cache/pipeline_v2/blobs/",
        help="Path to blobs/ on MYRIAD (default: ~/Scratch/data/dimensionality-manuscript/cache/pipeline_v2/blobs/)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview what would be transferred without making any changes",
    )
    parser.add_argument(
        "--check-overwrite",
        action="store_true",
        help="Report local/remote collisions without syncing (standalone safety check)",
    )
    parser.add_argument(
        "--include-model-caches",
        action="store_true",
        help="Include regression/subspace joblib caches (scores, hyperparameters)",
    )
    parser.add_argument(
        "--remote-cache",
        default=_DEFAULT_REMOTE_CACHE,
        help=f"Remote cache root for model caches (default: {_DEFAULT_REMOTE_CACHE})",
    )
    args = parser.parse_args()

    kwargs = dict(
        host=args.host,
        remote_db=args.remote_db,
        remote_blobs=args.remote_blobs,
        remote_cache=args.remote_cache,
        include_model_caches=args.include_model_caches,
    )

    if args.dry_run:
        preview_sync(**kwargs)
        return

    if args.check_overwrite:
        check_overwrite(**kwargs)
        return

    sync(host=args.host, remote_db=args.remote_db, remote_blobs=args.remote_blobs)
    if args.include_model_caches:
        sync_model_caches(args.host, args.remote_cache)


if __name__ == "__main__":
    main()
