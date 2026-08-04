"""Adaptive median-FPD power-law exponent fit: fixed configuration shared across spectrum panels."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=False)
class AdaptiveAlphaConfig:
    """Fixed configuration for the adaptive median-FPD power-law exponent fit.

    See ``_second_derivative_window`` / ``_median_fpd_alpha_session`` in
    :mod:`._spectrum_math` for the estimation procedure this configures. ``frozen=False`` so
    instances can be edited in place (e.g. from a syd viewer) if that's wired up later; not
    frozen does not by itself add any such widgets.
    """

    smooth_method: str
    """Log-space (geometric-mean) pre-smoothing kind: ``"none"``, ``"boxcar"``, or ``"gaussian"``."""
    smooth_width: float
    """Boxcar full-width in rank units; the Gaussian uses ``sigma = smooth_width / 2``."""
    fpd_window_size: int
    """Five-point-derivative stencil half-width (``deriv_width`` in the estimation functions)."""
    adaptive_buffer: int
    """Dims of margin applied on both sides of the second-derivative window."""
    minimum_window_size: int
    """Minimum finite local-exponent count inside the window; below this the fit is NaN."""


@dataclass(frozen=True)
class SpectrumSmoothingConfig:
    """The smoothing settings needed by a spectrum-only panel."""

    smooth_method: str
    smooth_width: float


class SpectrumSmoothing(Protocol):
    """Structural type shared by smoothing-only and adaptive-alpha settings."""

    smooth_method: str
    smooth_width: float


# Fixed per-side adaptive-fit configs for SpectrumAlphaFigureViewer: "placefields" governs
# source_key (and the fit_key Tilbury overlays); "full" governs full_source_key.
ADAPTIVE_ALPHA_CONFIG_REGISTRY: dict[str, AdaptiveAlphaConfig] = {
    "placefields": AdaptiveAlphaConfig(
        smooth_method="gaussian",
        smooth_width=3.0,
        fpd_window_size=1,
        adaptive_buffer=2,
        minimum_window_size=10,
    ),
    "full": AdaptiveAlphaConfig(
        smooth_method="gaussian",
        smooth_width=20.0,
        fpd_window_size=20,
        adaptive_buffer=10,
        minimum_window_size=100,
    ),
}
ADAPTIVE_ALPHA_CONFIG_NAMES: tuple[str, ...] = tuple(ADAPTIVE_ALPHA_CONFIG_REGISTRY.keys())


def get_adaptive_alpha_config(name: str) -> AdaptiveAlphaConfig:
    if name not in ADAPTIVE_ALPHA_CONFIG_REGISTRY:
        raise ValueError(f"Unknown adaptive alpha config name {name!r}. Available: {list(ADAPTIVE_ALPHA_CONFIG_REGISTRY)}")
    return ADAPTIVE_ALPHA_CONFIG_REGISTRY[name]
