"""Posthoc backfill of ``mean_notquality_filtered_*`` keys for ``regression_pf_residual``.

The new keys added to :class:`RegressionPlacefieldResidualConfig` are simple
means over already-stored per-ROI arrays, gated by the complement of the
already-stored ``quality_filtered_roi_mask``. Rather than bump the schema
version and rerun every session (expensive), this script recomputes just the
new keys from the existing result blobs (``blobs/{uid}.pkl`` next to the
results database) and adds them in place.

Idempotent: a row whose blob already has all seven new keys is skipped
untouched. Rows missing a required source key are left untouched and
reported. Each blob is rewritten atomically (write to a temp file, then
``os.replace``), so a crash mid-run cannot corrupt an existing blob.

Usage
-----
    python -m dimensionality_manuscript.scripts.backfill_notquality_filtered_means --dry-run
    python -m dimensionality_manuscript.scripts.backfill_notquality_filtered_means
"""

import argparse
import os
import pickle
import sqlite3

import numpy as np
from tqdm import tqdm

from dimensionality_manuscript.registry import RegistryPaths

ANALYSIS_TYPE = "regression_pf_residual"

# new key -> stored per-ROI array this mean is derived from
_SOURCE_KEYS = {
    "mean_notquality_filtered_within_pf_rms": "within_pf_rms",
    "mean_notquality_filtered_outside_pf_rms": "outside_pf_rms",
    "mean_notquality_filtered_variance_pf": "variance_pf",
    "mean_notquality_filtered_normalized_within_pf_rms": "normalized_within_pf_rms",
    "mean_notquality_filtered_normalized_outside_pf_rms": "normalized_outside_pf_rms",
    "mean_notquality_filtered_outside_minus_within_pf_rms": "outside_minus_within_pf_rms",
    "mean_notquality_filtered_normalized_outside_minus_within_pf_rms": "normalized_outside_minus_within_pf_rms",
}
_REQUIRED_KEYS = set(_SOURCE_KEYS.values()) | {"quality_filtered_roi_mask"}


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _backfill_result(result: dict) -> tuple[dict, bool, str | None]:
    """Add missing ``mean_notquality_filtered_*`` keys to one result dict.

    Returns
    -------
    result : dict
        The (possibly updated) result.
    changed : bool
        Whether any key was added.
    skip_reason : str or None
        Reason nothing was computed, if applicable.
    """
    if all(k in result for k in _SOURCE_KEYS):
        return result, False, "already present"
    missing_sources = _REQUIRED_KEYS - result.keys()
    if missing_sources:
        return result, False, f"missing source keys: {sorted(missing_sources)}"

    mask = np.asarray(result["quality_filtered_roi_mask"], dtype=bool)
    not_mask = ~mask
    for new_key, source_key in _SOURCE_KEYS.items():
        if new_key in result:
            continue
        source = np.asarray(result[source_key])
        if source.shape != mask.shape:
            return result, False, f"shape mismatch: {source_key} {source.shape} vs mask {mask.shape}"
        result[new_key] = _finite_mean(source[not_mask])
    return result, True, None


def _write_blob_atomic(blob_path, result: dict) -> None:
    """Write ``result`` to ``blob_path`` via a same-directory temp file + atomic rename."""
    tmp_path = blob_path.with_suffix(blob_path.suffix + ".tmp")
    tmp_path.write_bytes(pickle.dumps(result))
    os.replace(tmp_path, blob_path)


def main(dry_run: bool = False) -> None:
    db_path = RegistryPaths().pipeline_v2_db_path
    blob_dir = db_path.parent / "blobs"

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    rows = conn.execute(
        "SELECT result_uid FROM results WHERE analysis_type=? AND result_stored=1",
        (ANALYSIS_TYPE,),
    ).fetchall()
    conn.close()
    print(f"Found {len(rows)} {ANALYSIS_TYPE!r} rows with stored blobs.")

    updated = 0
    skipped = 0
    problems: list[tuple[str, str]] = []
    for (uid,) in tqdm(rows, desc="Backfilling"):
        blob_path = blob_dir / f"{uid}.pkl"
        if not blob_path.exists():
            skipped += 1
            problems.append((uid, "blob file missing"))
            continue
        result = pickle.loads(blob_path.read_bytes())
        result, changed, reason = _backfill_result(result)
        if not changed:
            skipped += 1
            if reason != "already present":
                problems.append((uid, reason))
            continue
        if not dry_run:
            _write_blob_atomic(blob_path, result)
        updated += 1

    verb = "Would update" if dry_run else "Updated"
    print(f"{verb}: {updated}, already current/skipped: {skipped}")
    if problems:
        print(f"\n{len(problems)} row(s) could not be backfilled:")
        for uid, reason in problems:
            print(f"  {uid}: {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print what would change without writing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
