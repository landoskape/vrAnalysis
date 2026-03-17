"""Data configuration wrapping RegistryParameters fields + spks_type."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from dimensionality_manuscript.registry import RegistryParameters


@dataclass(frozen=True)
class DataConfig:
    """Frozen dataclass for data preprocessing parameters.

    Wraps ``RegistryParameters`` fields plus ``spks_type`` into a single
    content-addressed configuration object. No defaults — use
    ``get_data_config()`` for named presets.

    Parameters
    ----------
    name : str
        Human-readable name for this configuration.
    speed_threshold : float
        Minimum speed for including frames.
    time_split_groups : int
        Number of time split groups.
    time_split_relative_size : tuple of int
        Relative sizes of time split groups.
    time_split_chunks_per_group : int
        Number of chunks per time split group.
    time_split_num_buffer : int
        Number of buffer chunks between groups.
    cell_split_force_even : bool
        Whether to force even cell splits.
    spks_type : str
        Spike data type to use.
    """

    name: str
    speed_threshold: float
    time_split_groups: int
    time_split_relative_size: tuple[int, ...]
    time_split_chunks_per_group: int
    time_split_num_buffer: int
    cell_split_force_even: bool
    spks_type: str

    def key(self) -> str:
        """SHA256 of serialized config, truncated to 16 chars."""
        serialized = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def summary(self) -> str:
        """Human-readable summary string (the config name)."""
        return self.name

    def to_registry_params(self) -> RegistryParameters:
        """Construct ``RegistryParameters`` from matching fields."""
        return RegistryParameters(
            speed_threshold=self.speed_threshold,
            time_split_groups=self.time_split_groups,
            time_split_relative_size=self.time_split_relative_size,
            time_split_chunks_per_group=self.time_split_chunks_per_group,
            time_split_num_buffer=self.time_split_num_buffer,
            cell_split_force_even=self.cell_split_force_even,
        )


# -- Named presets -------------------------------------------------------------
# Each entry maps a name to a dict of DataConfig field values (excluding "name",
# which is set automatically from the key).

_NAMED_CONFIGS: dict[str, dict] = {
    "default": dict(
        speed_threshold=1.0,
        time_split_groups=4,
        time_split_relative_size=(4, 4, 1, 1),
        time_split_chunks_per_group=10,
        time_split_num_buffer=3,
        cell_split_force_even=False,
        spks_type="oasis",
    ),
}


def get_data_config(name: str = "default") -> DataConfig:
    """Look up a named DataConfig preset.

    Parameters
    ----------
    name : str
        Name of the preset. See ``list_data_configs()`` for available names.

    Returns
    -------
    DataConfig

    Raises
    ------
    KeyError
        If ``name`` is not a registered preset.
    """
    if name not in _NAMED_CONFIGS:
        available = ", ".join(sorted(_NAMED_CONFIGS))
        raise KeyError(f"Unknown DataConfig {name!r}. Available: {available}")
    return DataConfig(name=name, **_NAMED_CONFIGS[name])


def list_data_configs() -> list[str]:
    """Return the names of all registered DataConfig presets."""
    return sorted(_NAMED_CONFIGS)
