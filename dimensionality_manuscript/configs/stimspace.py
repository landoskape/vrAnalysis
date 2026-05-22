"""StimSpaceConfig — stimulus-space cross-validated shared variance analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from vrAnalysis.sessions import B2Session, SpksTypes

from ..pipeline.base import AnalysisConfigBase
from ..registry import PopulationRegistry
from ..regression_models.hyperparameters import PlaceFieldHyperparameters


@dataclass(frozen=True)
class StimSpaceConfig(AnalysisConfigBase):
    """Configuration for stimulus-space cross-validated shared variance analysis.

    Parameters
    ----------
    center : bool
        Whether to center neural activity before analysis.
    normalize : bool
        Whether to normalize placefield matrices by per-neuron peak response.
    use_fast_sampling : bool
        Whether to use fast sampling for placefield computation.
    reliability_threshold : float or None
        Minimum leave-one-out reliability for neuron inclusion.
    fraction_active_threshold : float or None
        Minimum fraction active for neuron inclusion.
    num_bins : int
        Number of spatial bins for place field computation.
    smooth_width : float or None
        Gaussian smoothing width for place fields.
    spks_type : SpksTypes
        Spike type to use.
    """

    schema_version: str = "v1"
    data_config_name: str = "even"

    center: bool = True
    normalize: bool = False
    use_fast_sampling: bool = True
    reliability_threshold: Optional[float] = None
    fraction_active_threshold: Optional[float] = None
    num_bins: int = 100
    smooth_width: Optional[float] = None
    spks_type: SpksTypes = "oasis"

    display_name: ClassVar[str] = "stimspace"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "reliability_threshold": [None, 0.2],
            "fraction_active_threshold": [None, 0.05],
            "smooth_width": [None, 5.0],
        }

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"center={self.center}",
            f"norm={self.normalize}",
            f"fast={self.use_fast_sampling}",
            f"rel={self.reliability_threshold}",
            f"frac={self.fraction_active_threshold}",
            f"bins={self.num_bins}",
            f"smooth={self.smooth_width}",
            self.schema_version,
        ]
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Run stimulus-space shared variance analysis on a session."""
        from ..subspace_analysis.stimspace import StimSpaceSubspace

        hyps = PlaceFieldHyperparameters(num_bins=self.num_bins, smooth_width=self.smooth_width)
        model = StimSpaceSubspace(
            registry,
            centered=self.center,
            normalize=self.normalize,
            use_fast_sampling=self.use_fast_sampling,
            reliability_threshold=self.reliability_threshold,
            fraction_active_threshold=self.fraction_active_threshold,
        )
        return model.get_score(session, spks_type=self.spks_type, hyperparameters=hyps)
