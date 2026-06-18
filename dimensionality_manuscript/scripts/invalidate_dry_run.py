"""Dry-run preview for :meth:`ResultsStore.invalidate` (no deletes).

Uses the same planning path as ``store.plan_invalidate`` / ``store.invalidate``.

Usage
-----
    python -m dimensionality_manuscript.scripts.invalidate_dry_run --analysis-type regression --param-filters activity_parameters_name=raw

    python -m dimensionality_manuscript.scripts.invalidate_dry_run --schema-version v2 --analysis-type stimspace

    python -m dimensionality_manuscript.scripts.invalidate_dry_run --param-filters '{"activity_parameters_name": "raw"}' --analysis-type regression --schema-version v2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dimensionality_manuscript.pipeline.store import InvalidatePlan, ResultsStore
from dimensionality_manuscript.registry import RegistryPaths


def parse_param_filters(raw: str) -> dict[str, Any]:
    """Parse ``param_filters`` from CLI (JSON object or comma-separated ``k=v``).

    Parameters
    ----------
    raw : str
        Either JSON like ``'{"activity_parameters_name": "raw"}'`` or
        ``activity_parameters_name=raw,method=preferred``.

    Returns
    -------
    dict
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("param_filters string is empty")
    if raw.startswith("{"):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("JSON param_filters must be an object")
        return parsed

    out: dict[str, Any] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            raise ValueError(f"Expected key=value in param_filters segment {part!r}")
        out[key.strip()] = _coerce_filter_value(value.strip())
    return out


def _coerce_filter_value(value: str) -> Any:
    """Coerce CLI string to bool / int / float when obvious."""
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def print_dry_run(
    store: ResultsStore,
    plan: InvalidatePlan,
) -> None:
    """Print what :meth:`ResultsStore.invalidate` would remove."""
    print("=== Invalidate dry run (no changes made) ===")
    print(f"Database: {store.db_path}")
    print(f"Plan mode: {plan.mode}")
    if plan.analysis_type is not None:
        print(f"analysis_type: {plan.analysis_type!r}")
    if plan.schema_version is not None:
        print(f"schema_version: {plan.schema_version!r}")
    if plan.param_filters:
        print(f"param_filters: {plan.param_filters}")

    if plan.mode == "param_filters":
        print(f"Config variations from current class: {plan.config_variation_count}")
        print(f"Distinct analysis_key values in plan: {len(plan.analysis_keys)}")
        keys_with_rows = {row["analysis_key"] for row in store.rows_matching_invalidate_plan(plan)}
        keys_without_rows = set(plan.analysis_keys) - keys_with_rows
        if keys_without_rows:
            print(f"Analysis keys in plan with no stored results: {len(keys_without_rows)}")

    rows = store.rows_matching_invalidate_plan(plan)
    error_rows = store.errors_matching_invalidate_plan(plan)
    blobs = store.blob_paths_for_invalidate_plan(plan)
    print()
    print(f"Rows to DELETE from results table: {len(rows)}")
    print(f"Rows to DELETE from errors table: {len(error_rows)}")
    blob_exists = sum(1 for _, _, exists, _ in blobs if exists)
    blob_missing = len(blobs) - blob_exists
    stored_flag = sum(1 for r in rows if r.get("result_stored"))
    total_bytes = sum(size for _, _, exists, size in blobs if exists and size is not None)
    print(f"  result_stored=1: {stored_flag}")
    print(f"  blob .pkl files present: {blob_exists}")
    print(f"  blob .pkl files missing: {blob_missing}")
    if total_bytes:
        print(f"  blob bytes on disk: {total_bytes:,} ({total_bytes / 1e6:.2f} MB)")

    if not rows and not error_rows:
        print("\nNo matching rows.")
        return

    if rows:
        print()
        print("--- By config ---")
        by_summary = Counter(r.get("analysis_summary") or "" for r in rows)
        for summary, n in by_summary.most_common():
            print(f"  {n:>5}  {summary}")
    elif error_rows:
        print()
        print("--- By config (errors only) ---")
        by_summary = Counter(r.get("analysis_summary") or "" for r in error_rows)
        for summary, n in by_summary.most_common():
            print(f"  {n:>5}  {summary}")


def build_parser() -> argparse.ArgumentParser:
    """CLI matching :meth:`ResultsStore.invalidate` filters."""
    parser = argparse.ArgumentParser(
        description="Preview ResultsStore.invalidate() without deleting anything.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run dry-run preview."""
    parser = build_parser()
    args = parser.parse_args(argv)

    db_path = args.db_path or RegistryPaths().pipeline_v2_db_path
    if not Path(db_path).exists():
        print(f"No database at {db_path}", file=sys.stderr)
        return 1

    param_filters = parse_param_filters(args.param_filters) if args.param_filters else None
    if param_filters is None and not any([args.schema_version, args.analysis_type, args.analysis_key]):
        parser.error("Provide at least one of --schema-version, --analysis-type, --analysis-key, or --param-filters")

    store = ResultsStore(db_path)
    plan = store.plan_invalidate(
        schema_version=args.schema_version,
        analysis_type=args.analysis_type,
        param_filters=param_filters,
        analysis_key=args.analysis_key,
    )
    print_dry_run(store, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
