"""PopulationConfig — ensures population is cached in the registry.

This is a self-caching workflow: ``PopulationRegistry.get_population`` manages
its own file-based cache.  The pipeline's ``ResultsStore`` records *that* the
computation was done (with ``result_stored=False``).

``PopulationConfig`` has no variable parameters — population caching is fully
determined by ``DataConfig``'s ``RegistryParameters``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from vrAnalysis.sessions import B2Session
from ..registry import PopulationRegistry
from ..pipeline.base import AnalysisConfigBase


@dataclass(frozen=True)
class PopulationConfig(AnalysisConfigBase):
    """Configuration for population caching.

    No variable fields — population is fully determined by the DataConfig's
    RegistryParameters and the session's spks_type.
    """

    data_config_name: str = "default"

    display_name: ClassVar[str] = "population"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "data_config_name": ["default", "even"],
        }

    def process(self, session: B2Session, registry: PopulationRegistry) -> None:
        """Cache the population for this session.

        The registry manages its own file-based cache, so we return None
        (completion marker) — no blob in ResultsStore.
        """
        registry.get_population(session)
        return None
