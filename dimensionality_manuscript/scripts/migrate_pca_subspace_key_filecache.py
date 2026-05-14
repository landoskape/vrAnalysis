"""Rename variance_placefields_placefields -> variance_placefield_placefield in pca_subspace joblib score cache."""

from joblib import dump, load
from dimensionality_manuscript.registry import RegistryPaths

OLD_KEY = "variance_placefields_placefields"
NEW_KEY = "variance_placefield_placefield"


def migrate(dry_run: bool = False) -> None:
    score_path = RegistryPaths.subspace_score_path
    files = sorted(score_path.glob("pca_subspace_*.joblib"))
    print(f"Found {len(files)} pca_subspace score cache files.")

    updated = 0
    skipped = 0
    for path in files:
        metrics = load(path)
        if not isinstance(metrics, dict) or OLD_KEY not in metrics:
            skipped += 1
            continue
        if dry_run:
            print(f"  [dry-run] would rename key in {path.name}")
            updated += 1
            continue
        metrics[NEW_KEY] = metrics.pop(OLD_KEY)
        dump(metrics, path)
        updated += 1

    print(f"Updated {updated} files, skipped {skipped} (already correct or non-dict).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)
