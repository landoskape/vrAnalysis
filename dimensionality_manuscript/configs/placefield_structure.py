"""PlaceFieldStructureConfig — per-cell placefield feature measurement.

Measures single-cell placefield features (shape statistics, Gaussian fit,
trial-to-trial consistency, reliability) per environment experience-order
slot. No cvPCA, no downstream regression — this is measurement only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

from vrAnalysis.helpers import fit_gaussians, reliability_loo, vectorCorrelation
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.placefields import get_placefield
from vrAnalysis.sessions import B2Session, SpksTypes

from ..env_order import MAX_ENV_SLOTS, load_env_order
from ..pipeline.base import AnalysisConfigBase
from ..registry import PopulationRegistry, get_activity_parameters
from .regression import VALID_SPKS_TYPES


@dataclass(frozen=True)
class PlaceFieldStructureConfig(AnalysisConfigBase):
    """Configuration for per-cell placefield structure measurement.

    Parameters
    ----------
    activity_parameters_name : str
        Named activity-scaling preset (see ``get_activity_parameters``).
    spks_type : SpksTypes
        Spike type to use for placefield computation.
    num_bins : int
        Number of spatial bins for place field computation.
    smooth_width : float or None
        Gaussian smoothing width for place fields, or None for no smoothing.
    """

    schema_version: str = "v1"
    data_config_name: str = "default"

    activity_parameters_name: str = "raw"
    spks_type: SpksTypes = "sigrebase"
    num_bins: int = 100
    smooth_width: Optional[float] = None
    display_name: ClassVar[str] = "placefield_structure"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "activity_parameters_name": ["raw", "default"],
            "smooth_width": [None, 5.0],
        }

    def validate(self):
        if self.spks_type not in VALID_SPKS_TYPES:
            raise ValueError(f"Unknown spks_type {self.spks_type!r}. Available: {VALID_SPKS_TYPES}")

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"ap={self.activity_parameters_name}",
            f"bins={self.num_bins}",
            f"smooth={self.smooth_width}",
            self.schema_version,
        ]
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Measure per-cell placefield features for each environment experience-order slot."""
        population, frame_behavior = registry.get_population(session, spks_type=self.spks_type)
        ap = get_activity_parameters(self.activity_parameters_name)
        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)
        bin_centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])

        full_split = registry.time_split["full"]
        neuron_data = population.data[population.idx_neurons]
        data = population.apply_split(
            neuron_data,
            full_split,
            prefiltered=False,
            scale=ap.scale,
            scale_type=ap.scale_type,
            pre_split=ap.presplit,
        )
        fb = frame_behavior.filter(population.get_split_times(full_split, within_idx_samples=False))

        pf_trials = get_placefield(
            data.T.numpy(),
            fb,
            dist_edges=dist_edges,
            average=False,
            smooth_width=self.smooth_width,
            use_fast_sampling=True,
            session=session,
        )

        n_neurons = data.shape[0]
        mouse_order = load_env_order().get(session.mouse_name)

        env_slot_ids = np.full(MAX_ENV_SLOTS, np.nan)
        if mouse_order is not None:
            env_slot_ids[: len(mouse_order)] = mouse_order

        per_env: dict[str, dict[int, np.ndarray]] = {}
        envs_present = sorted(int(e) for e in np.unique(pf_trials.environment) if e >= 0)
        for env in envs_present:
            if mouse_order is None or env not in mouse_order:
                continue
            slot = mouse_order.index(env)

            pf_env = pf_trials.filter_by_environment(env)
            spkmap = np.transpose(pf_env.placefield, (2, 0, 1))  # (rois, trials, bins)
            if spkmap.shape[1] < 2:
                continue

            feats = _compute_pf_features(spkmap, bin_centers)
            for name, val in feats.items():
                per_env.setdefault(name, {})[slot] = val

        out: dict = {}
        for name, slot_map in per_env.items():
            arr = np.full((MAX_ENV_SLOTS, n_neurons), np.nan)
            for slot, val in slot_map.items():
                arr[slot] = val
            out[name] = arr
        out["env_slot_ids"] = env_slot_ids
        return out


def _compute_pf_features(spkmap: np.ndarray, bin_centers: np.ndarray) -> dict[str, np.ndarray]:
    """Per-cell placefield feature set from a single environment's trial spkmap.

    Parameters
    ----------
    spkmap : np.ndarray
        Shape (rois, trials, bins), placefield rate per trial.
    bin_centers : np.ndarray
        Shape (bins,), spatial position of each bin center.

    Returns
    -------
    dict[str, np.ndarray]
        Each value has shape (rois,).
    """
    mean_pf = np.nanmean(spkmap, axis=1)  # (rois, bins)

    pf_mean = np.nanmean(mean_pf, axis=1)
    pf_var = np.nanvar(mean_pf, axis=1)
    pf_norm = np.nansum(mean_pf**2, axis=1)
    pf_max = np.nanmax(mean_pf, axis=1)
    pf_cv = np.nanstd(mean_pf, axis=1) / pf_mean

    pf_gauss_amp, pf_gauss_center, pf_gauss_width, pf_gauss_r2 = fit_gaussians(mean_pf, x=bin_centers)

    reliability = reliability_loo(spkmap)
    fraction_active = FractionActive.compute(spkmap, activity_axis=2, fraction_axis=1, activity_method="rms", fraction_method="participation")
    spatial_participation = FractionActive.compute(spkmap, activity_axis=1, fraction_axis=1, activity_method="mean", fraction_method="participation")

    unit_pf = mean_pf / np.sqrt(np.nansum(mean_pf**2, axis=1, keepdims=True))
    trial_amplitude = np.nansum(unit_pf[:, None, :] * spkmap, axis=2)  # (rois, trials)
    pf_tdot_mean = np.nanmean(trial_amplitude, axis=1)
    pf_tdot_std = np.nanstd(trial_amplitude, axis=1)
    pf_tdot_cv = pf_tdot_std / pf_tdot_mean

    mean_pf_rep = np.repeat(mean_pf[:, None, :], spkmap.shape[1], axis=1)  # (rois, trials, bins)
    trial_corr = vectorCorrelation(spkmap, mean_pf_rep, axis=2, ignore_nan=True)  # (rois, trials)
    pf_tcorr_mean = np.nanmean(trial_corr, axis=1)
    pf_tcorr_std = np.nanstd(trial_corr, axis=1)

    return {
        "pf_mean": pf_mean,
        "pf_var": pf_var,
        "pf_norm": pf_norm,
        "pf_max": pf_max,
        "pf_cv": pf_cv,
        "pf_gauss_amp": np.asarray(pf_gauss_amp),
        "pf_gauss_center": np.asarray(pf_gauss_center),
        "pf_gauss_width": np.asarray(pf_gauss_width),
        "pf_gauss_r2": np.asarray(pf_gauss_r2),
        "reliability": reliability,
        "fraction_active": fraction_active,
        "spatial_participation": spatial_participation,
        "pf_tdot_mean": pf_tdot_mean,
        "pf_tdot_std": pf_tdot_std,
        "pf_tdot_cv": pf_tdot_cv,
        "pf_tcorr_mean": pf_tcorr_mean,
        "pf_tcorr_std": pf_tcorr_std,
    }
