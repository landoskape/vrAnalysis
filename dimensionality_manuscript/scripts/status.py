"""Show what's in the ResultsStore table.

Usage
-----
    python -m dimensionality_manuscript.scripts.status [--full] [--group-by COLUMN [COLUMN ...]]

Examples
--------
    # Default: summary counts by analysis_type
    python -m dimensionality_manuscript.scripts.status

    # Full table (all rows, no blob)
    python -m dimensionality_manuscript.scripts.status --full

    # Group by session
    python -m dimensionality_manuscript.scripts.status --group-by session_id

    # Group by multiple columns
    python -m dimensionality_manuscript.scripts.status --group-by analysis_type schema_version

    # Summarise recorded errors
    python -m dimensionality_manuscript.scripts.status --show-errors

    # Also show unique error messages per config group
    python -m dimensionality_manuscript.scripts.status --show-errors --include-error-types
"""

import argparse
from collections import Counter, defaultdict

import pandas as pd
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript import ResultsStore

REGISTRY_PATHS = RegistryPaths()


def print_error_summary(store: ResultsStore, include_error_types: bool = False) -> None:
    """Print recorded errors grouped by (analysis_type, analysis_summary).

    Parameters
    ----------
    store : ResultsStore
    include_error_types : bool
        If True, print unique error messages under each config group.
    """
    errors = store.get_errors()
    if not errors:
        print("No errors recorded.")
        return

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in errors:
        key = (e.get("analysis_type") or "", e.get("analysis_summary") or "")
        groups[key].append(e)

    n_sessions_total = len({e["session_id"] for e in errors})
    print(f"Errors: {len(errors)} total | {len(groups)} unique configs | {n_sessions_total} sessions\n")

    for i, ((atype, summary), rows) in enumerate(
        sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1]))), 1
    ):
        n_sessions = len({r["session_id"] for r in rows})
        schema = rows[0].get("schema_version") or ""
        print(f"  [{i:>3}] {atype} {schema} | {summary} | {n_sessions} sessions")
        if include_error_types:
            counts = Counter(r.get("error_message", "").splitlines()[0] for r in rows)
            for msg, n in counts.most_common():
                print(f"           {n}x  {msg}")


def status(
    full: bool = False,
    group_by: list[str] | None = None,
    show_errors: bool = False,
    include_error_types: bool = False,
):
    """Print a summary of the ResultsStore contents.

    Parameters
    ----------
    full : bool
        If True, print every row. Otherwise print grouped counts.
    group_by : list of str or None
        Columns to group by for the summary. Defaults to
        ``["analysis_type", "schema_version"]``.
    show_errors : bool
        If True, also print a summary of recorded errors.
    include_error_types : bool
        If True (requires show_errors), print unique error messages per config group.
    """
    db_path = REGISTRY_PATHS.pipeline_v2_db_path
    if not db_path.exists():
        print(f"No database found at {db_path}")
        return

    store = ResultsStore(db_path)
    df = store.summary_table(as_dataframe=True)

    if df.empty:
        print("Store is empty.")
        return

    print(f"Store: {db_path}")
    print(f"Total rows: {len(df)}")
    print(f"Snapshots: {len(store.list_snapshots())}")
    print()

    if full:
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(df.to_string(index=False))
    else:
        if group_by is None:
            group_by = ["analysis_type", "schema_version"]

        bad_cols = [c for c in group_by if c not in df.columns]
        if bad_cols:
            print(f"Unknown columns: {bad_cols}")
            print(f"Available: {list(df.columns)}")
            return

        grouped = df.groupby(group_by).agg(
            count=("result_uid", "size"),
            stored=("result_stored", "sum"),
            sessions=("session_id", "nunique"),
            earliest=("computed_at", "min"),
            latest=("computed_at", "max"),
        )
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(grouped.to_string())

    if show_errors:
        print()
        print_error_summary(store, include_error_types=include_error_types)


def main():
    parser = argparse.ArgumentParser(description="Show ResultsStore contents")
    parser.add_argument("--full", action="store_true", help="Print every row")
    parser.add_argument("--group-by", nargs="+", default=None, help="Columns to group by (default: analysis_type schema_version)")
    parser.add_argument("--show-errors", action="store_true", help="Summarise recorded errors grouped by config")
    parser.add_argument("--include-error-types", action="store_true", help="With --show-errors, print unique error messages per config group")
    args = parser.parse_args()

    status(
        full=args.full,
        group_by=args.group_by,
        show_errors=args.show_errors,
        include_error_types=args.include_error_types,
    )


if __name__ == "__main__":
    main()
