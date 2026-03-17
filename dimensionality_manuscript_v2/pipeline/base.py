"""Abstract base class for analysis configurations."""

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from dataclasses import asdict, dataclass
from itertools import product
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from dimensionality_manuscript.registry import PopulationRegistry
    from vrAnalysis.sessions import B2Session


@dataclass(frozen=True)
class AnalysisConfigBase:
    """Abstract frozen dataclass for analysis configurations.

    All analysis configs inherit from this. Provides content-addressed
    keying, parameter grid generation, and an abstract process method.
    """

    schema_version: str = "v1"
    display_name: ClassVar[str]

    def __post_init__(self):
        self.validate()

    def key(self) -> str:
        """SHA256 of serialized config, truncated to 16 chars."""
        serialized = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def summary(self) -> str:
        """Human-readable summary string."""
        return f"{self.display_name}_{self.schema_version}"

    def validate(self):
        """Override to check invariants. Called in __post_init__."""

    @staticmethod
    @abstractmethod
    def _param_grid() -> dict:
        """Return ``{field: [options]}`` for Cartesian product generation."""

    @classmethod
    def generate_variations(cls) -> list[AnalysisConfigBase]:
        """Cartesian product of ``_param_grid()``. Override to filter."""
        grid = cls._param_grid()
        fields = list(grid.keys())
        values = list(grid.values())
        return [cls(**dict(zip(fields, combo))) for combo in product(*values)]

    @classmethod
    def from_key(cls, key: str) -> AnalysisConfigBase:
        """Look up a config by its content-hash key.

        Generates all variations and returns the one whose ``key()``
        matches. Raises ``KeyError`` if no match is found.

        Parameters
        ----------
        key : str
            The 16-char hex key to look up.
        """
        for cfg in cls.generate_variations():
            if cfg.key() == key:
                return cfg
        raise KeyError(f"No {cls.__name__} variation matches key {key!r}")

    def get_result(
        self,
        store,
        row: dict,
        *,
        session: B2Session | None = None,
        registry: PopulationRegistry | None = None,
    ) -> dict | None:
        """Load the result for a summary-table row.

        Default implementation loads the blob from the store via
        ``get_by_uid``. Self-caching configs (where ``process`` returns
        None) should override this to retrieve the result from their
        own cache using information in *row* plus *session*/*registry*.

        Parameters
        ----------
        store : ResultsStore
            The results store containing this row.
        row : dict
            A single row from ``ResultsStore.summary_table()``.  Always
            contains at least ``result_uid`` and ``session_id``.
        session : B2Session, optional
            Live session object — required by self-caching overrides.
        registry : PopulationRegistry, optional
            Population registry — required by self-caching overrides.

        Returns
        -------
        dict or None
            The result dict, or None if not available.
        """
        return store.get_by_uid(row["result_uid"])

    @abstractmethod
    def process(self, session: B2Session, registry: PopulationRegistry) -> dict | None:
        """Run analysis on a session.

        Parameters
        ----------
        session : B2Session
            Session to process.
        registry : PopulationRegistry
            Population registry for getting population data.

        Returns
        -------
        dict or None
            Result dict stored by ResultsStore, or None as a completion
            marker for self-caching workflows.
        """
