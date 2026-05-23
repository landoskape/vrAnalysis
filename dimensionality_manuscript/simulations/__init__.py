"""
Simulation utilities for dimensionality analysis.

This package provides tools for generating synthetic data and performing
matrix operations useful for illustrating dimensionality analysis concepts.
"""

from .generators import (
    CovariancePairConfig,
    CovariancePairGenerator,
    CovarianceGenerator,
    PowerlawDataGenerator,
    RotatedEigenbasisGenerator,
    SharedSpaceConfig,
    SharedSpaceGenerator,
    StimFullConfig,
    StimFullGenerator,
)
from .operators import sqrtm_spd, invsqrtm_spd, geometric_mean_spd, root_sandwich
from .plotting import plot_ellipse
from .shared_variance import (
    ATLAS,
    AnalysisProvenance,
    AtlasAnalysisResult,
    AtlasBuild,
    AtlasSpec,
    EmpiricalBlock,
    EmpiricalDiagnostics,
    ModeComparison,
    PopulationBlock,
    SubspaceGeometry,
    analyze_atlas_case,
    analyze_build,
    build_atlas_case,
    energy_modes,
    get_atlas_spec,
    kappa_modes,
    list_atlas_cases,
    process,
    stimulus_space_energy_modes,
    stimulus_space_kappa_modes,
)
from .utilities import get_orthogonal_direction, generate_orthonormal, find_commute_space, find_commute_space_gated

__all__ = [
    "CovariancePairConfig",
    "CovariancePairGenerator",
    "CovarianceGenerator",
    "PowerlawDataGenerator",
    "RotatedEigenbasisGenerator",
    "SharedSpaceConfig",
    "SharedSpaceGenerator",
    "StimFullConfig",
    "StimFullGenerator",
    "sqrtm_spd",
    "invsqrtm_spd",
    "geometric_mean_spd",
    "root_sandwich",
    "plot_ellipse",
    "ATLAS",
    "AnalysisProvenance",
    "AtlasAnalysisResult",
    "AtlasBuild",
    "AtlasSpec",
    "EmpiricalBlock",
    "EmpiricalDiagnostics",
    "ModeComparison",
    "PopulationBlock",
    "SubspaceGeometry",
    "analyze_atlas_case",
    "analyze_build",
    "build_atlas_case",
    "energy_modes",
    "get_atlas_spec",
    "kappa_modes",
    "list_atlas_cases",
    "process",
    "stimulus_space_energy_modes",
    "stimulus_space_kappa_modes",
    "get_orthogonal_direction",
    "generate_orthonormal",
    "find_commute_space",
    "find_commute_space_gated",
]
