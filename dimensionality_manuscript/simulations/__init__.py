"""
Simulation utilities for dimensionality analysis.

This package provides tools for generating synthetic data and performing
matrix operations useful for illustrating dimensionality analysis concepts.
"""

from .generators import (
    CovarianceGenerator,
    PowerlawDataGenerator,
    RotatedEigenbasisGenerator,
    SharedSpaceConfig,
    SharedSpaceGenerator,
)
from .operators import sqrtm_spd, invsqrtm_spd, geometric_mean_spd, root_sandwich
from .plotting import plot_ellipse
from .utilities import get_orthogonal_direction, generate_orthonormal

__all__ = [
    "CovarianceGenerator",
    "PowerlawDataGenerator",
    "RotatedEigenbasisGenerator",
    "SharedSpaceConfig",
    "SharedSpaceGenerator",
    "sqrtm_spd",
    "invsqrtm_spd",
    "geometric_mean_spd",
    "root_sandwich",
    "plot_ellipse",
    "get_orthogonal_direction",
    "generate_orthonormal",
]
