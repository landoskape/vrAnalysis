"""SVCAConfig — wraps SVCA/covcov subspace measurements from measure_svca.py.

This is a self-caching workflow: the subspace model infrastructure manages
its own file-based cache.  The pipeline's ``ResultsStore`` records *that* the
computation was done (with ``result_stored=False``), and ``get_result`` knows
how to retrieve the score dict from the model's own cache.

Unlike ``SubspaceConfig``, this config uses ``match_dimensions=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from vrAnalysis.sessions import B2Session, SpksTypes
from dimensionality_manuscript.registry import (
    SubspaceName,
    PopulationRegistry,
    get_subspace,
)
from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldHyperparameters
from ..pipeline.base import AnalysisConfigBase

_SVCA_SUBSPACE_NAMES = ("svca_subspace", "covcov_subspace")


@dataclass(frozen=True)
class SVCAConfig(AnalysisConfigBase):
    """Configuration for SVCA/covcov subspace measurements (match_dimensions=False).

    Parameters
    ----------
    subspace_name : SubspaceName
        Name of the subspace model. Must be one of ``"svca_subspace"`` or
        ``"covcov_subspace"``.
    spks_type : SpksTypes
        Spike type to use for the population.
    num_bins : int
        Number of spatial bins for place field computation.
    smooth_width : float or None
        Gaussian smoothing width for place fields. None means no smoothing.
    """

    subspace_name: SubspaceName = "svca_subspace"
    spks_type: SpksTypes = "oasis"
    num_bins: int = 100
    smooth_width: float | None = None

    display_name: ClassVar[str] = "svca"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "subspace_name": list(_SVCA_SUBSPACE_NAMES),
            "spks_type": ["oasis"],
            "num_bins": [100],
            "smooth_width": [5.0, None],
        }

    def validate(self):
        if self.subspace_name not in _SVCA_SUBSPACE_NAMES:
            raise ValueError(f"Unknown subspace_name {self.subspace_name!r} for SVCAConfig. " f"Available: {', '.join(_SVCA_SUBSPACE_NAMES)}")

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"{self.subspace_name}",
            f"spks={self.spks_type}",
            f"bins={self.num_bins}",
            f"smooth={self.smooth_width}",
            self.schema_version,
        ]
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> None:
        """Measure SVCA subspace on this session.

        The subspace infrastructure caches results to its own file store,
        so we return None (completion marker) — no blob in ResultsStore.
        """
        model = get_subspace(self.subspace_name, registry, match_dimensions=False)
        hyps = PlaceFieldHyperparameters(num_bins=self.num_bins, smooth_width=self.smooth_width)
        model.get_score(session, spks_type=self.spks_type, hyperparameters=hyps)
        return None

    def get_result(
        self,
        store,
        row: dict,
        *,
        session: B2Session,
        registry: PopulationRegistry,
    ):
        """Load the score dict from the subspace model's own file cache.

        Parameters
        ----------
        store : ResultsStore
            Not used — result lives in the model's cache, not the store.
        row : dict
            Summary-table row (needs ``session_id``).
        session : B2Session
            Live session object (required).
        registry : PopulationRegistry
            Population registry (required).
        """
        model = get_subspace(self.subspace_name, registry, match_dimensions=False)
        hyps = PlaceFieldHyperparameters(num_bins=self.num_bins, smooth_width=self.smooth_width)
        return model.get_score(session, spks_type=self.spks_type, hyperparameters=hyps)
