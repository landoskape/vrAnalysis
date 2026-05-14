"""SubspaceConfig — wraps subspace model measurements from measure_subspaces.py.

This is a self-caching workflow: the subspace model infrastructure manages
its own file-based cache.  The pipeline's ``ResultsStore`` records *that* the
computation was done (with ``result_stored=False``), and ``get_result`` knows
how to retrieve the score dict from the model's own cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from vrAnalysis.sessions import B2Session, SpksTypes
from ..registry import (
    SUBSPACE_NAMES,
    SubspaceName,
    PopulationRegistry,
    get_subspace,
)
from ..regression_models.hyperparameters import PlaceFieldHyperparameters
from ..pipeline.base import AnalysisConfigBase


@dataclass(frozen=True)
class SubspaceConfig(AnalysisConfigBase):
    """Configuration for subspace measurements (match_dimensions=True).

    Parameters
    ----------
    subspace_name : SubspaceName
        Name of the subspace model. Must be in ``SUBSPACE_NAMES`` and must
        not be ``"cvpca_subspace"`` (marked broken).
    spks_type : SpksTypes
        Spike type to use for the population.
    num_bins : int
        Number of spatial bins for place field computation.
    smooth_width : float or None
        Gaussian smoothing width for place fields. None means no smoothing.
    """

    schema_version: str = "v3"
    data_config_name: str = "even"

    subspace_name: SubspaceName = "covcov_crossvalidated_subspace"
    spks_type: SpksTypes = "oasis"
    num_bins: int = 100
    smooth_width: float | None = None

    display_name: ClassVar[str] = "subspace"

    # Cross is (dims x dims), ordered by variance explained, but different sessions have different dims
    # if they have different numbers of environments! SO (100,100), (200,200), etc for num_bins=100.
    # Might be ragged, but easier to just handle locally since the NaNs will fill out everything I don't want!
    # _result_handling: ClassVar[dict[str, str]] = {"cross": "ragged"}

    @staticmethod
    def _param_grid() -> dict:
        return {
            "subspace_name": ["pca_subspace", "covcov_subspace", "covcov_crossvalidated_subspace"],
            "smooth_width": [5.0, None],
        }

    def validate(self):
        if self.subspace_name == "cvpca_subspace":
            raise ValueError("cvpca_subspace is marked as broken and cannot be used in SubspaceConfig.")
        if self.subspace_name not in SUBSPACE_NAMES:
            raise ValueError(f"Unknown subspace_name {self.subspace_name!r}. " f"Available: {', '.join(SUBSPACE_NAMES)}")

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
        """Measure subspace on this session.

        The subspace infrastructure caches results to its own file store,
        so we return None (completion marker) — no blob in ResultsStore.
        """
        model = get_subspace(self.subspace_name, registry)
        hyps = PlaceFieldHyperparameters(num_bins=self.num_bins, smooth_width=self.smooth_width)
        subspace = model.fit(session, spks_type=self.spks_type, hyperparameters=hyps)
        score = model.get_score(session, spks_type=self.spks_type, hyperparameters=hyps)
        cross = subspace.subspace_activity.get_components().T @ subspace.subspace_placefields.get_components()
        # Add this to the score dict so the store aggregate pipeline can pull it out and save all elements
        score["cross"] = cross
        return score
