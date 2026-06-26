"""Backfill stale ``analysis_summary`` text in the ResultsStore.

``analysis_summary`` is a display label only — identity is ``analysis_key``
(``cfg.key()``, hashed from all dataclass fields). When a config's
``summary()`` format changes, old rows keep their stale label text even
though their key/data are still correct. This script recomputes the current
``summary()`` for every config in the param grid and rewrites any row whose
stored label doesn't match, in both the ``results`` and ``errors`` tables.

Usage
-----
    python -m dimensionality_manuscript.scripts.backfill_summary --dry-run
    python -m dimensionality_manuscript.scripts.backfill_summary --analyses tilbury_fit
"""

import argparse

from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript import ResultsStore
from dimensionality_manuscript.scripts.run import build_analysis_configs

REGISTRY_PATHS = RegistryPaths()


def backfill_summary(analyses: list[str] | None = None, dry_run: bool = False) -> None:
    """Rewrite stale ``analysis_summary`` rows to match each config's current ``summary()``.

    Parameters
    ----------
    analyses : list of str or None
        Analysis config keys to include (same as ``run.py --analyses``). None = all.
    dry_run : bool
        If True, only print what would change; don't write to the database.
    """
    db_path = REGISTRY_PATHS.pipeline_v2_db_path
    store = ResultsStore(db_path)
    configs = build_analysis_configs(include=analyses)

    n_changed_results = 0
    n_changed_errors = 0
    for cfg in configs:
        key = cfg.key()
        schema = cfg.schema_version
        new_summary = cfg.summary()

        with store._connect() as conn:
            stale_results = conn.execute(
                "SELECT result_uid, analysis_summary FROM results " "WHERE analysis_key=? AND schema_version=? AND analysis_summary != ?",
                (key, schema, new_summary),
            ).fetchall()
            stale_errors = conn.execute(
                "SELECT result_uid, analysis_summary FROM errors " "WHERE analysis_key=? AND schema_version=? AND analysis_summary != ?",
                (key, schema, new_summary),
            ).fetchall()

            if stale_results or stale_errors:
                old_labels = {row[1] for row in stale_results} | {row[1] for row in stale_errors}
                print(f"{key} ({schema})")
                for old in old_labels:
                    print(f"    {old!r}")
                print(f"  -> {new_summary!r}")
                print(f"  results: {len(stale_results)} rows | errors: {len(stale_errors)} rows")

            if not dry_run:
                if stale_results:
                    conn.execute(
                        "UPDATE results SET analysis_summary=? WHERE analysis_key=? AND schema_version=? AND analysis_summary != ?",
                        (new_summary, key, schema, new_summary),
                    )
                if stale_errors:
                    conn.execute(
                        "UPDATE errors SET analysis_summary=? WHERE analysis_key=? AND schema_version=? AND analysis_summary != ?",
                        (new_summary, key, schema, new_summary),
                    )

        n_changed_results += len(stale_results)
        n_changed_errors += len(stale_errors)

    verb = "Would update" if dry_run else "Updated"
    print(f"\n{verb} {n_changed_results} result row(s) and {n_changed_errors} error row(s).")


def main():
    parser = argparse.ArgumentParser(description="Backfill stale analysis_summary labels in the ResultsStore")
    parser.add_argument("--analyses", nargs="+", help="Analysis config keys to include (default: all)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print what would change without writing")
    args = parser.parse_args()

    backfill_summary(analyses=args.analyses, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
