"""Delete results from the ResultsStore (dry-run preview by default).

Uses the same planning path as :meth:`ResultsStore.invalidate` /
:meth:`ResultsStore.plan_invalidate`.

Usage
-----
    # Preview (default; same filters as invalidate_dry_run)
    python -m dimensionality_manuscript.scripts.invalidate --analysis-type regression --param-filters activity_parameters_name=raw

    python -m dimensionality_manuscript.scripts.invalidate --schema-version v2 --analysis-type stimspace

    python -m dimensionality_manuscript.scripts.invalidate --param-filters '{"activity_parameters_name": "raw"}' --analysis-type regression --schema-version v2

    # Execute delete (prompts for confirmation unless --yes)
    python -m dimensionality_manuscript.scripts.invalidate --analysis-type cvpca --execute

    python -m dimensionality_manuscript.scripts.invalidate --analysis-type regression --param-filters activity_parameters_name=raw --execute --yes

    # Delete everything in the store
    python -m dimensionality_manuscript.scripts.invalidate --all --execute --yes

    # Preview / delete results no current param grid generates (superseded schema
    # versions, dropped parameters). Analysis types absent from the config registry
    # are reported but never deleted.
    python -m dimensionality_manuscript.scripts.invalidate --prune-stale
    python -m dimensionality_manuscript.scripts.invalidate --prune-stale --analyses expmax vector_gain_rank
    python -m dimensionality_manuscript.scripts.invalidate --prune-stale --execute
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from dimensionality_manuscript.pipeline.store import InvalidatePlan, ResultsStore, StalePrunePlan
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.scripts.invalidate_dry_run import parse_param_filters, print_dry_run


def _filter_args_present(args: argparse.Namespace) -> bool:
    """Return True if any row-level filter flags were passed."""
    return any([args.schema_version, args.analysis_type, args.analysis_key, args.param_filters])


def _print_invalidate_all_preview(store: ResultsStore) -> None:
    """Print what :meth:`ResultsStore.invalidate_all` would remove."""
    rows = store.summary_table()
    errors = store.get_errors()
    blob_dir = store._blob_dir
    blobs = list(blob_dir.glob("*.pkl")) if blob_dir.exists() else []
    total_bytes = sum(p.stat().st_size for p in blobs)
    print("=== Invalidate ALL preview (no changes made) ===")
    print(f"Database: {store.db_path}")
    print(f"Rows to DELETE from results table: {len(rows)}")
    print(f"Rows to DELETE from errors table: {len(errors)}")
    print(f"Blob .pkl files on disk: {len(blobs)}")
    if total_bytes:
        print(f"Blob bytes on disk: {total_bytes:,} ({total_bytes / 1e6:.2f} MB)")


def _print_prune_stale_preview(store: ResultsStore, plan: StalePrunePlan) -> tuple[int, int]:
    """Print what a stale prune would remove. Returns ``(n_results, n_errors)``."""
    print("=== Prune stale preview (no changes made) ===")
    print(f"Database: {store.db_path}")
    print("Stale = a stored analysis_key that the analysis's current param grid no longer generates.\n")

    if plan.unknown_types:
        print("SKIPPED - analysis_type not in the config registry (never deleted):")
        for analysis_type, count in sorted(plan.unknown_types.items()):
            label = analysis_type or "<null>"
            print(f"  {label:<34} {count:>7} rows")
        print("  If any of these should still exist, add the class to ANALYSIS_CONFIG_CLASSES before pruning.\n")

    if plan.skipped_types:
        print("SKIPPED - config class generated no variations:")
        for analysis_type, reason in sorted(plan.skipped_types.items()):
            print(f"  {analysis_type:<34} {reason}")
        print()

    if not plan.plans:
        print("No stale results found.")
        return 0, 0

    total_results = total_errors = total_bytes = 0
    print(f"{'analysis_type':<34}{'live':>6}{'stale':>7}{'results':>9}{'errors':>8}{'MB':>10}")
    print("-" * 74)
    for sub_plan in plan.plans:
        analysis_type = sub_plan.analysis_type
        rows = store.rows_matching_invalidate_plan(sub_plan)
        errors = store.errors_matching_invalidate_plan(sub_plan)
        blobs = store.blob_paths_for_invalidate_plan(sub_plan)
        n_bytes = sum(size for _, _, exists, size in blobs if exists and size is not None)
        total_results += len(rows)
        total_errors += len(errors)
        total_bytes += n_bytes
        print(
            f"{analysis_type:<34}"
            f"{plan.live_key_counts.get(analysis_type, 0):>6}"
            f"{len(plan.stale_keys[analysis_type]):>7}"
            f"{len(rows):>9}{len(errors):>8}{n_bytes / 1e6:>10.2f}"
        )
    print("-" * 74)
    print(f"{'TOTAL':<34}{'':>6}{'':>7}{total_results:>9}{total_errors:>8}{total_bytes / 1e6:>10.2f}")

    print("\n--- Stale configs ---")
    for sub_plan in plan.plans:
        rows = store.rows_matching_invalidate_plan(sub_plan) or store.errors_matching_invalidate_plan(sub_plan)
        for summary, count in Counter(r.get("analysis_summary") or "" for r in rows).most_common():
            print(f"  {count:>5}  {summary}")
    return total_results, total_errors


def _confirm(message: str, *, assume_yes: bool) -> bool:
    """Prompt for confirmation unless ``assume_yes`` is True."""
    if assume_yes:
        return True
    try:
        resp = input(f"{message} [y/N] ").strip().lower()
    except EOFError:
        return False
    return resp in ("y", "yes")


def build_parser() -> argparse.ArgumentParser:
    """CLI matching :meth:`ResultsStore.invalidate` filters."""
    parser = argparse.ArgumentParser(
        description="Delete ResultsStore rows matching filters (dry run by default).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to results.db (default: RegistryPaths().pipeline_v2_db_path)",
    )
    parser.add_argument("--schema-version", default=None, help="Filter schema_version column")
    parser.add_argument("--analysis-type", default=None, help="Filter analysis_type (required with --param-filters)")
    parser.add_argument(
        "--param-filters",
        default=None,
        help='Param grid constraints, e.g. activity_parameters_name=raw or JSON {"activity_parameters_name": "raw"}',
    )
    parser.add_argument(
        "--analysis-key",
        default=None,
        help="Single 16-char analysis_key (instead of constructing analysis_cfg)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matching rows and blobs (default: preview only)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt when using --execute",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete ALL results (requires --execute; cannot combine with other filters)",
    )
    parser.add_argument(
        "--prune-stale",
        action="store_true",
        help="Delete stored results whose analysis_key no current param grid generates "
        "(i.e. superseded schema versions and dropped parameters). Types absent from the "
        "config registry are always left alone.",
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        default=None,
        metavar="ANALYSIS_TYPE",
        help="Restrict --prune-stale to these analysis types (default: all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run invalidate preview or execute deletes."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.all and args.prune_stale:
        parser.error("--all and --prune-stale are mutually exclusive")
    if args.analyses and not args.prune_stale:
        parser.error("--analyses only applies to --prune-stale; use --analysis-type otherwise")

    db_path = args.db_path or RegistryPaths().pipeline_v2_db_path
    if not Path(db_path).exists():
        print(f"No database at {db_path}", file=sys.stderr)
        return 1

    store = ResultsStore(db_path)

    if args.all:
        if _filter_args_present(args):
            parser.error("--all cannot be combined with other filter flags")
        _print_invalidate_all_preview(store)
        if not args.execute:
            print("\nDry run — pass --execute to delete everything.")
            return 0
        if not _confirm("Delete ALL results, errors, and blob files?", assume_yes=args.yes):
            print("Aborted.")
            return 1
        store.invalidate_all()
        print("Deleted all results, errors, and blobs.")
        return 0

    if args.prune_stale:
        if _filter_args_present(args):
            parser.error("--prune-stale cannot be combined with row-level filters; use --analyses to narrow it")
        plan = store.plan_prune_stale(analysis_types=args.analyses)
        n_rows, n_errors = _print_prune_stale_preview(store, plan)

        if not args.execute:
            print("\nDry run - pass --execute to delete.")
            return 0
        if n_rows == 0 and n_errors == 0:
            print("\nNothing to delete.")
            return 0
        if not _confirm(f"Delete {n_rows} stale result row(s), {n_errors} error row(s), and their blobs?", assume_yes=args.yes):
            print("Aborted.")
            return 1

        deleted = sum(_execute_plan(store, sub_plan) for sub_plan in plan.plans)
        print(f"\nDeleted {deleted} stale result row(s) and their errors/blobs.")
        return 0

    param_filters = parse_param_filters(args.param_filters) if args.param_filters else None
    if param_filters is None and not _filter_args_present(args):
        parser.error("Provide at least one of --schema-version, --analysis-type, --analysis-key, --param-filters, or --all")

    plan = store.plan_invalidate(
        schema_version=args.schema_version,
        analysis_type=args.analysis_type,
        param_filters=param_filters,
        analysis_key=args.analysis_key,
    )
    print_dry_run(store, plan)

    if not args.execute:
        print("\nDry run — pass --execute to delete.")
        return 0

    n_rows = len(store.rows_matching_invalidate_plan(plan))
    n_errors = len(store.errors_matching_invalidate_plan(plan))
    if n_rows == 0 and n_errors == 0:
        print("\nNothing to delete.")
        return 0

    parts = []
    if n_rows:
        parts.append(f"{n_rows} result row(s)")
    if n_errors:
        parts.append(f"{n_errors} error row(s)")
    if n_rows:
        parts.append("associated blobs")
    if not _confirm(f"Delete {', '.join(parts)}?", assume_yes=args.yes):
        print("Aborted.")
        return 1

    n = _execute_plan(store, plan)
    deleted = [f"{n} result row(s)"]
    if n_errors:
        deleted.append(f"{n_errors} error row(s)")
    print(f"\nDeleted {', '.join(deleted)}.")
    return 0


def _execute_plan(store: ResultsStore, plan: InvalidatePlan) -> int:
    """Execute an invalidate plan (supports analysis_key via plan_invalidate)."""
    return store._execute_invalidate_plan(plan)


if __name__ == "__main__":
    raise SystemExit(main())
