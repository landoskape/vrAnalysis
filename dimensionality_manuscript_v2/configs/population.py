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
from dimensionality_manuscript.registry import PopulationRegistry
from ..pipeline.base import AnalysisConfigBase


@dataclass(frozen=True)
class PopulationConfig(AnalysisConfigBase):
    """Configuration for population caching.

    No variable fields — population is fully determined by the DataConfig's
    RegistryParameters and the session's spks_type.
    """

    display_name: ClassVar[str] = "population"

    @staticmethod
    def _param_grid() -> dict:
        return {}

    def process(self, session: B2Session, registry: PopulationRegistry) -> None:
        """Cache the population for this session.

        The registry manages its own file-based cache, so we return None
        (completion marker) — no blob in ResultsStore.
        """
        registry.get_population(session)
        return None

    def get_result(
        self,
        store,
        row: dict,
        *,
        session: B2Session,
        registry: PopulationRegistry,
    ):
        """Load the population from the registry's own cache.

        Parameters
        ----------
        store : ResultsStore
            Not used — result lives in the registry's cache, not the store.
        row : dict
            Summary-table row (needs ``session_id``).
        session : B2Session
            Live session object (required).
        registry : PopulationRegistry
            Population registry (required).

        Returns
        -------
        tuple[Population, FrameBehavior]
        """
        return registry.get_population(session)
