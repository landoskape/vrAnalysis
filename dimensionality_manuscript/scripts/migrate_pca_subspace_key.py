"""Rename variance_placefields_placefields -> variance_placefield_placefield in pca_subspace blobs."""

import pickle
from dimensionality_manuscript.pipeline.store import ResultsStore

OLD_KEY = "variance_placefields_placefields"
NEW_KEY = "variance_placefield_placefield"


def migrate(dry_run: bool = False) -> None:
    store = ResultsStore()
    df = store.summary_table(as_dataframe=True)
    mask = (df["analysis_type"] == "subspace") & df["analysis_summary"].str.contains("pca_subspace", na=False)
    rows = df[mask]
    print(f"Found {len(rows)} pca_subspace rows to scan.")

    updated = 0
    skipped = 0
    for uid in rows["result_uid"]:
        result = store.get_by_uid(uid)
        if result is None or OLD_KEY not in result:
            skipped += 1
            continue
        if dry_run:
            print(f"  [dry-run] would rename key in uid={uid}")
            updated += 1
            continue
        result[NEW_KEY] = result.pop(OLD_KEY)
        new_blob = pickle.dumps(result)
        with store._connect() as conn:
            conn.execute("UPDATE results SET result_blob=? WHERE result_uid=?", (new_blob, uid))
        updated += 1

    print(f"Updated {updated} rows, skipped {skipped} (already correct or null).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)
