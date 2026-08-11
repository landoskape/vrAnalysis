from __future__ import annotations

from typing import TYPE_CHECKING

from .behavior_speed_env import BehaviorSpeedEnvConfig
from .cvpca import CVPCAConfig
from .data_config import DataConfig, get_data_config, list_data_configs
from .pfpred_quality import PFPredQualityConfig
from .placefield_structure import CrossValidatedPlacefieldsConfig, PlacefieldPredictionConfig, PlaceFieldStructureConfig
from .population import PopulationConfig
from .regression import (
    RegressionConfig,
    RegressionPlacefieldResidualConfig,
    RegressionResidualStructureConfig,
    VectorGainRankConfig,
    RegressionDimensionalitySweepConfig,
    StructuredAdditiveRankConfig,
    RankModelsSweepConfig,
)

# from .rrr_to_external_latents import RRRToExternalLatentsConfig # Not in use at the moment, might revive later, but import currently broken
from .subspace import SubspaceConfig
from .stimspace import StimSpaceConfig, StimSpaceSpectraConfig, StimSpaceEnvPCAConfig, StimspaceSVCAConfig
from .expmax import ExpMaxConfig
from .locprediction import LocPredConfig, LocPredCrossVal
from .tilbury_fit import TilburyFitConfig
from .simulation_sweep import (
    SimulationSession,
    SIMULATION_SESSION,
    StimFullSweepConfig,
    ThresholdedGPSweepConfig,
    SmoothGPSweepConfig,
    TilburySweepConfig,
)

if TYPE_CHECKING:
    from ..pipeline.base import AnalysisConfigBase

#: Every analysis config class in the package. Single source of truth — ``ResultsStore``
#: and ``scripts.run`` both derive their lookups from this instead of hand-written lists.
ANALYSIS_CONFIG_CLASS_LIST: tuple[type["AnalysisConfigBase"], ...] = (
    BehaviorSpeedEnvConfig,
    CVPCAConfig,
    CrossValidatedPlacefieldsConfig,
    ExpMaxConfig,
    LocPredConfig,
    LocPredCrossVal,
    PFPredQualityConfig,
    PlaceFieldStructureConfig,
    PlacefieldPredictionConfig,
    PopulationConfig,
    # RRRToExternalLatentsConfig, # see above
    RankModelsSweepConfig,
    RegressionConfig,
    RegressionPlacefieldResidualConfig,
    RegressionResidualStructureConfig,
    RegressionDimensionalitySweepConfig,
    SmoothGPSweepConfig,
    StimFullSweepConfig,
    StimSpaceConfig,
    StimSpaceEnvPCAConfig,
    StimSpaceSpectraConfig,
    StimspaceSVCAConfig,
    StructuredAdditiveRankConfig,
    SubspaceConfig,
    ThresholdedGPSweepConfig,
    TilburyFitConfig,
    TilburySweepConfig,
    VectorGainRankConfig,
)


def _build_registry() -> dict[str, type["AnalysisConfigBase"]]:
    """Map ``display_name`` to config class, rejecting duplicate names."""
    registry: dict[str, type["AnalysisConfigBase"]] = {}
    for cls in ANALYSIS_CONFIG_CLASS_LIST:
        existing = registry.get(cls.display_name)
        if existing is not None:
            raise ValueError(f"Duplicate display_name {cls.display_name!r}: {existing.__name__} and {cls.__name__}")
        registry[cls.display_name] = cls
    return registry


#: ``display_name`` -> config class, for every analysis config in the package.
ANALYSIS_CONFIG_CLASSES: dict[str, type["AnalysisConfigBase"]] = _build_registry()

__all__ = [
    "ANALYSIS_CONFIG_CLASSES",
    "ANALYSIS_CONFIG_CLASS_LIST",
    "BehaviorSpeedEnvConfig",
    "CVPCAConfig",
    "DataConfig",
    "PFPredQualityConfig",
    "CrossValidatedPlacefieldsConfig",
    "PlacefieldPredictionConfig",
    "PlaceFieldStructureConfig",
    "PopulationConfig",
    "RegressionConfig",
    "RegressionPlacefieldResidualConfig",
    "RegressionResidualStructureConfig",
    "RRRToExternalLatentsConfig",
    "VectorGainRankConfig",
    "RegressionDimensionalitySweepConfig",
    "StructuredAdditiveRankConfig",
    "RankModelsSweepConfig",
    "SubspaceConfig",
    "StimSpaceConfig",
    "StimSpaceEnvPCAConfig",
    "StimSpaceSpectraConfig",
    "StimspaceSVCAConfig",
    "ExpMaxConfig",
    "LocPredConfig",
    "LocPredCrossVal",
    "TilburyFitConfig",
    "SimulationSession",
    "SIMULATION_SESSION",
    "StimFullSweepConfig",
    "ThresholdedGPSweepConfig",
    "SmoothGPSweepConfig",
    "TilburySweepConfig",
    "get_data_config",
    "list_data_configs",
]
