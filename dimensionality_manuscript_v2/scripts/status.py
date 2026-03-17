"""Show what's in the ResultsStore table.

Usage
-----
    python -m dimensionality_manuscript_v2.scripts.status [--full] [--group-by COLUMN [COLUMN ...]]

Examples
--------
    # Default: summary counts by analysis_type
    python -m dimensionality_manuscript_v2.scripts.status

    # Full table (all rows, no blob)
    python -m dimensionality_manuscript_v2.scripts.status --full

    # Group by session
    python -m dimensionality_manuscript_v2.scripts.status --group-by session_id

    # Group by multiple columns
    python -m dimensionality_manuscript_v2.scripts.status --group-by analysis_type schema_version
"""

import argparse

import pandas as pd
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript_v2 import ResultsStore

REGISTRY_PATHS = RegistryPaths()


def status(full: bool = False, group_by: list[str] | None = None):
    """Print a summary of the ResultsStore contents.

    Parameters
    ----------
    full : bool
        If True, print every row. Otherwise print grouped counts.
    group_by : list of str or None
        Columns to group by for the summary. Defaults to
        ``["analysis_type", "schema_version"]``.
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
        return

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
        data_configs=("data_key", "nunique"),
        earliest=("computed_at", "min"),
        latest=("computed_at", "max"),
    )
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(grouped.to_string())


def main():
    parser = argparse.ArgumentParser(description="Show ResultsStore contents")
    parser.add_argument("--full", action="store_true", help="Print every row")
    parser.add_argument("--group-by", nargs="+", default=None, help="Columns to group by (default: analysis_type schema_version)")
    args = parser.parse_args()

    status(full=args.full, group_by=args.group_by)


if __name__ == "__main__":
    main()
