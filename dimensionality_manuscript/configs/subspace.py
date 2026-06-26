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
    get_activity_parameters,
)
from ..regression_models.hyperparameters import PlaceFieldHyperparameters
from .regression import VALID_SPKS_TYPES
from ..pipeline.base import AnalysisConfigBase

VALID_SUBSPACE_NAMES: list[SubspaceName] = [
    "covcov_subspace",
    "covcov_crossvalidated_subspace",
]


@dataclass(frozen=True)
class SubspaceConfig(AnalysisConfigBase):
    """Configuration for subspace measurements (see shared_variance.md).

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

    schema_version: str = "v7"
    data_config_name: str = "even"

    subspace_name: SubspaceName = "covcov_subspace"
    spks_type: SpksTypes = "sigrebase"
    num_bins: int = 100
    smooth_width: float | None = None
    activity_parameters_name: str = "raw"
    display_name: ClassVar[str] = "subspace"

    # Cross is (dims x dims), ordered by variance explained, but different sessions have different dims
    # if they have different numbers of environments! SO (100,100), (200,200), etc for num_bins=100.
    # Might be ragged, but easier to just handle locally since the NaNs will fill out everything I don't want!
    # _result_handling: ClassVar[dict[str, str]] = {"cross": "ragged"}

    @staticmethod
    def _param_grid() -> dict:
        return {
            # "spks_type": list(VALID_SPKS_TYPES), # now only use sigrebase! oasis is bad bad bad
            "smooth_width": [5.0, None],
            "activity_parameters_name": ["raw", "default"],
            "subspace_name": list(VALID_SUBSPACE_NAMES),
        }

    def validate(self):
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
        if self.activity_parameters_name != "raw":
            parts.append(f"ap={self.activity_parameters_name}")
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> None:
        """Measure subspace on this session."""
        keep_components = 1000
        ap = get_activity_parameters(self.activity_parameters_name)
        model = get_subspace(self.subspace_name, registry, centered=True, activity_parameters=ap)
        hyps = PlaceFieldHyperparameters(num_bins=self.num_bins, smooth_width=self.smooth_width)
        subspace = model.fit(session, spks_type=self.spks_type, hyperparameters=hyps)
        score = model.get_score(session, spks_type=self.spks_type, hyperparameters=hyps)
        full_components = subspace.subspace_activity.get_components()
        full_components = full_components[:, :keep_components]
        full_placefields = subspace.subspace_placefields.get_components()
        full_placefields = full_placefields[:, :keep_components]
        cross = full_components.T @ full_placefields
        # Add this to the score dict so the store aggregate pipeline can pull it out and save all elements
        score["cross"] = cross
        return score
