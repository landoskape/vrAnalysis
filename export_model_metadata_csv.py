"""Export ``MODEL_METADATA`` from ``registry.py`` to a spreadsheet-friendly CSV.

Writes ``dimensionality_manuscript/model_metadata.csv`` (same directory as
``registry.py``). Boolean fields use ``0``/``1`` for Excel; ``reference`` is
empty when unset.

Usage
-----
From the repository root::

    python export_model_metadata_csv.py

Optional output path::

    python export_model_metadata_csv.py --output path/to/model_metadata.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from pathlib import Path

from dimensionality_manuscript.registry import MODEL_METADATA, MODEL_NAMES, ModelMetadata

REGISTRY_DIR = Path(__file__).resolve().parent / "dimensionality_manuscript"
DEFAULT_CSV_PATH = REGISTRY_DIR / "model_metadata.csv"

_BOOL_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ModelMetadata) if f.name != "reference")
_CSV_COLUMNS: tuple[str, ...] = ("idx", "model_name", *_BOOL_FIELDS, "reference")


def export_model_metadata_csv(output_path: Path) -> Path:
    """Write model metadata rows to a CSV file.

    Parameters
    ----------
    output_path : Path
        Destination CSV path.

    Returns
    -------
    Path
        ``output_path`` after writing.

    Raises
    ------
    ValueError
        If ``MODEL_NAMES`` and ``MODEL_METADATA`` keys do not match.
    """
    missing = [name for name in MODEL_NAMES if name not in MODEL_METADATA]
    extra = [name for name in MODEL_METADATA if name not in MODEL_NAMES]
    if missing or extra:
        raise ValueError(f"MODEL_NAMES / MODEL_METADATA mismatch: missing={missing!r}, extra={extra!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for idx, name in enumerate(MODEL_NAMES):
            meta = MODEL_METADATA[name]
            row = {
                "idx": idx,
                "model_name": name,
                **{field: int(getattr(meta, field)) for field in _BOOL_FIELDS},
                "reference": meta.reference or "",
            }
            writer.writerow(row)

    return output_path


def main() -> None:
    """Parse CLI arguments and export the metadata CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Output CSV path (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--i-really-want-to",
        action="store_true",
        help="I really want to export the model metadata CSV",
    )
    args = parser.parse_args()
    if args.i_really_want_to:
        path = export_model_metadata_csv(args.output)
        print(f"Wrote {len(MODEL_NAMES)} rows to {path}")
    else:
        print("Exiting without exporting model metadata CSV")
