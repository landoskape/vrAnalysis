"""StimSpaceConfig — stimulus-space cross-validated shared variance analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from vrAnalysis.sessions import B2Session, SpksTypes

from ..pipeline.base import AnalysisConfigBase
from ..registry import PopulationRegistry, get_activity_parameters
from ..regression_models.hyperparameters import PlaceFieldHyperparameters
from .regression import VALID_SPKS_TYPES


@dataclass(frozen=True)
class StimSpaceConfig(AnalysisConfigBase):
    """Configuration for stimulus-space cross-validated shared variance analysis.

    Parameters
    ----------
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

    schema_version: str = "v4"
    data_config_name: str = "even"

    activity_parameters_name: str = "raw"
    reliability_fraction_active_thresholds: Optional[tuple[float, float]] = (None, None)
    num_bins: int = 100
    smooth_width: Optional[float] = None
    spks_type: SpksTypes = "sigrebase"
    display_name: ClassVar[str] = "stimspace"

    @staticmethod
    def _param_grid() -> dict:
        return {
            # "spks_type": list(VALID_SPKS_TYPES), # now only use sigrebase! oasis is bad bad bad
            "activity_parameters_name": ["raw", "default"],
            # "reliability_fraction_active_thresholds": [(None, None), (0.2, 0.05)],  # Why didn't I think of this before?
            "smooth_width": [None, 5.0],
        }

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"ap={self.activity_parameters_name}",
            f"rel={self.reliability_fraction_active_thresholds[0]}",
            f"frac={self.reliability_fraction_active_thresholds[1]}",
            f"bins={self.num_bins}",
            f"smooth={self.smooth_width}",
            self.schema_version,
        ]
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Run stimulus-space shared variance analysis on a session."""
        from ..subspace_analysis.stimspace import StimSpaceSubspace

        hyps = PlaceFieldHyperparameters(num_bins=self.num_bins, smooth_width=self.smooth_width)
        ap = get_activity_parameters(self.activity_parameters_name)
        model = StimSpaceSubspace(
            registry,
            activity_parameters=ap,
            reliability_threshold=self.reliability_fraction_active_thresholds[0],
            fraction_active_threshold=self.reliability_fraction_active_thresholds[1],
        )
        metrics = model.get_score(session, spks_type=self.spks_type, hyperparameters=hyps)
        cv_variance_scale = model.compute_cv_variance_scale(session, spks_type=self.spks_type, hyperparameters=hyps)
        for key, val in cv_variance_scale.items():
            metrics[key] = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
        return metrics
