"""Load :class:`ModelMetadata` from ``model_metadata.csv`` (no heavy registry deps)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path

MODEL_METADATA_CSV = Path(__file__).resolve().parent / "model_metadata.csv"


@dataclass(frozen=True)
class ModelMetadata:
    """Tags for grouping regression models in plots and analyses."""

    main: bool = False
    one_d_pos: bool = False
    high_d_pos: bool = False
    speed: bool = False
    reward: bool = False
    unconstrained: bool = False
    internal: bool = False
    leak: bool = False
    vector_gain: bool = False
    no_intercept: bool = False
    high_d_speed: bool = False
    reference: str | None = None

    def __post_init__(self) -> None:
        if not self.main and not self.reference:
            raise ValueError("All models must be a 'main' model or have a reference model.")
        if self.one_d_pos and self.high_d_pos:
            raise ValueError("One-D position and high-D position cannot be true at the same time.")


_METADATA_BOOL_FIELDS: tuple[str, ...] = tuple(
    field.name for field in fields(ModelMetadata) if field.name != "reference"
)


def _parse_metadata_bool(value: str) -> bool:
    """Parse a spreadsheet-style boolean cell."""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_model_metadata(
    model_names: tuple[str, ...],
    path: Path = MODEL_METADATA_CSV,
) -> dict[str, ModelMetadata]:
    """Load model metadata from a CSV file.

    Parameters
    ----------
    model_names
        Expected model names and canonical ordering.
    path
        CSV path; defaults to ``model_metadata.csv`` beside this module.

    Returns
    -------
    dict[str, ModelMetadata]
        Metadata keyed by model name.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If rows do not match ``model_names`` or ``idx`` disagrees with registry order.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model metadata CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    metadata: dict[str, ModelMetadata] = {}
    for row in rows:
        name = row["model_name"].strip()
        if name not in model_names:
            raise ValueError(f"Unknown model_name {name!r} in {path}")
        idx = int(row["idx"])
        expected_idx = model_names.index(name)
        if idx != expected_idx:
            raise ValueError(f"{name!r}: csv idx={idx} but model_names index is {expected_idx}")

        reference = (row.get("reference") or "").strip() or None
        if reference is not None and reference not in model_names:
            raise ValueError(f"Reference model {reference!r} for {name!r} not found in model_names.")
        kwargs = {field: _parse_metadata_bool(row[field]) for field in _METADATA_BOOL_FIELDS}
        kwargs["reference"] = reference
        metadata[name] = ModelMetadata(**kwargs)

    missing = [name for name in model_names if name not in metadata]
    extra = [name for name in metadata if name not in model_names]
    if missing or extra:
        raise ValueError(f"{path}: missing models {missing!r}, extra models {extra!r}")

    return metadata
