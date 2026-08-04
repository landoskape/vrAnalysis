"""ScoreModelsConfig — wraps regression model scoring from score_models.py.

This is a self-caching workflow: the regression model infrastructure manages
its own file-based cache (joblib files in ``score_path``).  The pipeline's
``ResultsStore`` records *that* the computation was done (with
``result_stored=False``), and ``get_result`` knows how to retrieve the
score dict from the model's own cache.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar
from collections import defaultdict

import numpy as np
from dimilibi import measure_r2, mse
from vrAnalysis.helpers import reliability_loo
from vrAnalysis.metrics import FractionActive
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_placefield, get_placefield_prediction
from ..registry import (
    MODEL_NAMES,
    ModelName,
    PopulationRegistry,
    get_model,
    ACTIVITY_PARAMETERS_NAMES,
    get_activity_parameters,
)
from ..regression_models.models import (
    PlaceFieldModel,
    RBFPosModel,
    FullRegressorModel,
    ReducedRankRegressionModel,
)
from ..pipeline.base import AnalysisConfigBase

VALID_ACTIVITY_PARAMETERS: list[str] = ["default", "preserved", "std"]
VALID_SPKS_TYPES: list[SpksTypes] = ["oasis", "sigrebase"]

# Models used in the key regression figures and their associated analyses.
KEY_FIGURE_MODELS: list[ModelName] = [
    "external_placefield_1d",
    "internal_placefield_1d",
    "external_placefield_1d_gain",
    "internal_placefield_1d_gain",
    "rrr",
    "rrr_no_intercept",
    "fullregressor_decoder_only_1dspeed_predreward",
    "fullregressor_1dspeed_predreward",
    "fullregressor_decoder_only_1dspeed_predreward_no_intercept",
    "fullregressor_1dspeed_predreward_no_intercept",
]


def _log_int_values(start: int, stop: int, num: int = 25) -> np.ndarray:
    """Unique integer values spaced logarithmically over ``[start, stop]``."""
    raw = np.logspace(np.log10(start), np.log10(stop), num=num)
    return np.unique(np.round(raw).astype(int))


# Dimensionality sweep grids (integer, log-spaced).
NUM_BINS_VALUES: np.ndarray = _log_int_values(1, 200)  # placefield num_bins
NUM_BASIS_VALUES: np.ndarray = _log_int_values(1, 200)  # rbfpos / full num_basis
RANK_VALUES: np.ndarray = _log_int_values(1, 2000)  # rrr rank (clipped to data)


@dataclass(frozen=True)
class RegressionConfig(AnalysisConfigBase):
    """Configuration for regression model scoring.

    Parameters
    ----------
    model_name : ModelName
        Name of the regression model (must be in ``MODEL_NAMES``).
    spks_type : SpksTypes
        Spike type to use for the population.
    method : str
        Hyperparameter optimization method.
    """

    schema_version: str = "v3"
    # v3: recompute with numerically improved placefield code

    data_config_name: str = "default"
    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "sigrebase"
    method: str = "preferred"
    activity_parameters_name: str = "default"

    display_name: ClassVar[str] = "regression"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "model_name": list(KEY_FIGURE_MODELS),
            "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
            # "spks_type": list(VALID_SPKS_TYPES), # no longer analyzing anything except sigrebase
        }

    def validate(self):
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model_name {self.model_name!r}. " f"Available: {', '.join(MODEL_NAMES)}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. Available: {', '.join(list(ACTIVITY_PARAMETERS_NAMES))}"
            )

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"{self.model_name}",
            f"spks={self.spks_type}",
            f"method={self.method}",
        ]
        if self.activity_parameters_name != "default":
            parts.append(f"ap={self.activity_parameters_name}")
        parts.append(self.schema_version)
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Score the model on this session.

        The model infrastructure caches results to its own file store,
        so we return None (completion marker) — no blob in ResultsStore.
        """
        model = get_model(self.model_name, registry, activity_parameters=self.activity_parameters_name)
        score = model.get_best_score(
            session,
            spks_type=self.spks_type,
            method=self.method,
        )
        return score


def _placefield_weighted_residual_rms(
    residual: np.ndarray,
    placefield_prediction: np.ndarray,
    placefield_peak: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ROI residual RMS inside and outside a soft place field.

    The place-field prediction is converted to a membership in ``[0, 1]`` by
    dividing each ROI by the peak of its all-data place-field map. Within-field
    weights are proportional to that membership; outside-field weights are
    proportional to one minus the membership. Each set of temporal weights is
    normalized independently before calculating RMS.
    """
    residual = np.asarray(residual, dtype=float)
    placefield_prediction = np.asarray(placefield_prediction, dtype=float)
    placefield_peak = np.asarray(placefield_peak, dtype=float)
    if residual.ndim != 2:
        raise ValueError(f"residual must have shape (rois, frames), got {residual.shape}")
    if placefield_prediction.shape != residual.shape:
        raise ValueError("placefield_prediction must match residual shape " f"{residual.shape}, got {placefield_prediction.shape}")
    if placefield_peak.shape != (residual.shape[0],):
        raise ValueError(f"placefield_peak must have shape ({residual.shape[0]},), got {placefield_peak.shape}")

    valid_peak = np.isfinite(placefield_peak) & (placefield_peak > 0)
    valid = np.isfinite(residual) & np.isfinite(placefield_prediction) & valid_peak[:, None]
    membership = np.divide(
        np.clip(placefield_prediction, 0, None),
        placefield_peak[:, None],
        out=np.full_like(placefield_prediction, np.nan),
        where=valid_peak[:, None],
    )
    membership = np.clip(membership, 0, 1)

    squared_error = np.where(valid, residual**2, 0)
    within_weight = np.where(valid, membership, 0)
    outside_weight = np.where(valid, 1 - membership, 0)
    within_total = np.sum(within_weight, axis=1)
    outside_total = np.sum(outside_weight, axis=1)

    within_mse = np.divide(
        np.sum(within_weight * squared_error, axis=1),
        within_total,
        out=np.full(residual.shape[0], np.nan),
        where=within_total > 0,
    )
    outside_mse = np.divide(
        np.sum(outside_weight * squared_error, axis=1),
        outside_total,
        out=np.full(residual.shape[0], np.nan),
        where=outside_total > 0,
    )
    return np.sqrt(within_mse), np.sqrt(outside_mse)


def _finite_mean(values: np.ndarray) -> float:
    """Mean of finite values, or NaN when no finite values exist."""
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


@dataclass(frozen=True)
class RegressionPlacefieldResidualConfig(AnalysisConfigBase):
    """Compare held-out model residuals within and outside each target ROI's place field.

    The parameter grid deliberately mirrors :class:`RegressionConfig`: every
    regression result therefore has a corresponding residual-localization job.
    A common, non-cross-validated place field is estimated from the target
    population's full split in the same activity units used by the model.
    """

    schema_version: str = "v5"
    # v5: compute the ROI quality filter the same way cvpca / stimspace_spectra do
    # (fast behavioral sampling, unvisited bins at zero instead of NaN). v4 required
    # every bin to be occupied on every trial, which left zero quality ROIs in most
    # sessions.
    data_config_name: str = "default"
    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "sigrebase"
    method: str = "preferred"
    activity_parameters_name: str = "default"

    num_placefield_bins: int = 100
    placefield_smooth_width: float | None = None
    reliability_threshold: float = 0.3
    fraction_active_threshold: float = 0.1
    display_name: ClassVar[str] = "regression_pf_residual"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "model_name": list(KEY_FIGURE_MODELS),
            "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
        }

    def validate(self):
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model_name {self.model_name!r}. " f"Available: {', '.join(MODEL_NAMES)}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. Available: " f"{', '.join(list(ACTIVITY_PARAMETERS_NAMES))}"
            )
        if self.num_placefield_bins < 1:
            raise ValueError("num_placefield_bins must be at least 1")
        if not -1 <= self.reliability_threshold <= 1:
            raise ValueError("reliability_threshold must be between -1 and 1")
        if not 0 <= self.fraction_active_threshold <= 1:
            raise ValueError("fraction_active_threshold must be between 0 and 1")

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"{self.model_name}",
            f"spks={self.spks_type}",
            f"method={self.method}",
        ]
        if self.activity_parameters_name != "default":
            parts.append(f"ap={self.activity_parameters_name}")
        parts.extend([f"rel={self.reliability_threshold:g}", f"fa={self.fraction_active_threshold:g}"])
        parts.append(self.schema_version)
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Fit the cached-best model and localize its held-out residual RMS."""
        if np.unique(session.env_length).size != 1:
            raise ValueError("All trials must have the same environment length!")

        model = get_model(self.model_name, registry, activity_parameters=self.activity_parameters_name)
        hyperparameters = model.get_best_hyperparameters(
            session,
            spks_type=self.spks_type,
            method=self.method,
        )[0]
        report = model.process(
            session,
            self.spks_type,
            train_split="train",
            test_split="test",
            hyperparameters=hyperparameters,
        )

        # Estimate a common target-neuron PF from every population time split.
        # get_session_data applies the exact activity scaling used by this model.
        _, target_full, frame_behavior_full = model.get_session_data(session, self.spks_type, "full")
        dist_edges = np.linspace(0, session.env_length[0], self.num_placefield_bins + 1)
        placefield = get_placefield(
            target_full.T.numpy(),
            frame_behavior_full,
            dist_edges=dist_edges,
            speed_threshold=None,
            average=True,
            smooth_width=self.placefield_smooth_width,
            zero_to_nan=True,
        )

        # Trial-resolved maps provide per-environment quality values in the same
        # target-ROI order and activity units as the residual arrays below. This
        # mirrors the ROI-quality filters in cvpca / stimspace_spectra: behavioral
        # fast sampling, and unvisited bins left at zero rather than converted to
        # NaN, so the reliability / fraction-active thresholds select comparable
        # neurons across all three analyses.
        placefield_trials = get_placefield(
            target_full.T.numpy(),
            frame_behavior_full,
            dist_edges=dist_edges,
            speed_threshold=None,
            average=False,
            use_fast_sampling=True,
            session=session,
        )
        environments = np.sort(np.unique(placefield_trials.environment)).astype(int)
        reliability = np.full((len(environments), target_full.shape[0]), np.nan)
        fraction_active = np.full_like(reliability, np.nan)
        for env_idx, environment in enumerate(environments):
            pf_env = placefield_trials.filter_by_environment(environment)
            spkmap = np.transpose(pf_env.placefield, (2, 0, 1))
            # reliability_loo divides by (num_trials - 1); a single-trial environment
            # contributes no reliability and is left NaN (never passes the threshold).
            if spkmap.shape[1] >= 2:
                reliability[env_idx] = reliability_loo(spkmap)
            fraction_active[env_idx] = FractionActive.compute(
                spkmap,
                activity_axis=2,
                fraction_axis=1,
                activity_method="rms",
                fraction_method="participation",
            )
        # Keep a neuron reliable/active in *any* environment (NaN never passes).
        quality_filtered_roi_mask = np.any(reliability > self.reliability_threshold, axis=0) & np.any(
            fraction_active > self.fraction_active_threshold, axis=0
        )

        # Evaluate PF membership on the same held-out frames as the model report,
        # including any model-specific removal of frames with invalid predictions.
        _, _, frame_behavior_test = model.get_session_data(session, self.spks_type, "test")
        pf_prediction = get_placefield_prediction(placefield, frame_behavior_test)[0].T
        if report.extras.get("predictions_were_filtered", False):
            pf_prediction = pf_prediction[:, report.extras["idx_valid_predictions"]]

        target = np.asarray(report.target_data)
        prediction = np.asarray(report.predicted_data)
        if pf_prediction.shape != target.shape or prediction.shape != target.shape:
            raise ValueError(
                "Held-out target, model prediction, and PF prediction are misaligned: "
                f"target={target.shape}, prediction={prediction.shape}, pf={pf_prediction.shape}"
            )

        finite_pf = np.isfinite(placefield.placefield)
        placefield_peak = np.max(
            np.where(finite_pf, placefield.placefield, -np.inf),
            axis=(0, 1),
        )
        placefield_peak[~np.any(finite_pf, axis=(0, 1))] = np.nan
        within_rms, outside_rms = _placefield_weighted_residual_rms(
            target - prediction,
            pf_prediction,
            placefield_peak,
        )
        # One total held-out variance per ROI provides the scale needed to compare
        # residual magnitudes across sessions. Using ddof=0 matches the SSE / SST
        # normalization in R²: RMS / sqrt(variance) is sqrt(SSE / SST) when the
        # same frames and weights are used.
        variance_pf = np.var(target, axis=1)
        target_std = np.sqrt(variance_pf)
        normalized_within_rms = np.divide(
            within_rms,
            target_std,
            out=np.full_like(within_rms, np.nan),
            where=np.isfinite(target_std) & (target_std > 0),
        )
        normalized_outside_rms = np.divide(
            outside_rms,
            target_std,
            out=np.full_like(outside_rms, np.nan),
            where=np.isfinite(target_std) & (target_std > 0),
        )
        difference = outside_rms - within_rms
        normalized_difference = normalized_outside_rms - normalized_within_rms
        return {
            "within_pf_rms": within_rms,
            "outside_pf_rms": outside_rms,
            "variance_pf": variance_pf,
            "normalized_within_pf_rms": normalized_within_rms,
            "normalized_outside_pf_rms": normalized_outside_rms,
            "outside_minus_within_pf_rms": difference,
            "normalized_outside_minus_within_pf_rms": normalized_difference,
            "reliability": reliability,
            "fraction_active": fraction_active,
            "quality_environments": environments,
            "quality_filtered_roi_mask": quality_filtered_roi_mask,
            "num_quality_filtered_rois": int(np.sum(quality_filtered_roi_mask)),
            "mean_within_pf_rms": _finite_mean(within_rms),
            "mean_outside_pf_rms": _finite_mean(outside_rms),
            "mean_variance_pf": _finite_mean(variance_pf),
            "mean_normalized_within_pf_rms": _finite_mean(normalized_within_rms),
            "mean_normalized_outside_pf_rms": _finite_mean(normalized_outside_rms),
            "mean_outside_minus_within_pf_rms": _finite_mean(difference),
            "mean_normalized_outside_minus_within_pf_rms": _finite_mean(normalized_difference),
            "mean_quality_filtered_within_pf_rms": _finite_mean(within_rms[quality_filtered_roi_mask]),
            "mean_quality_filtered_outside_pf_rms": _finite_mean(outside_rms[quality_filtered_roi_mask]),
            "mean_quality_filtered_variance_pf": _finite_mean(variance_pf[quality_filtered_roi_mask]),
            "mean_quality_filtered_normalized_within_pf_rms": _finite_mean(normalized_within_rms[quality_filtered_roi_mask]),
            "mean_quality_filtered_normalized_outside_pf_rms": _finite_mean(normalized_outside_rms[quality_filtered_roi_mask]),
            "mean_quality_filtered_outside_minus_within_pf_rms": _finite_mean(difference[quality_filtered_roi_mask]),
            "mean_quality_filtered_normalized_outside_minus_within_pf_rms": _finite_mean(normalized_difference[quality_filtered_roi_mask]),
            "mean_notquality_filtered_within_pf_rms": _finite_mean(within_rms[~quality_filtered_roi_mask]),
            "mean_notquality_filtered_outside_pf_rms": _finite_mean(outside_rms[~quality_filtered_roi_mask]),
            "mean_notquality_filtered_variance_pf": _finite_mean(variance_pf[~quality_filtered_roi_mask]),
            "mean_notquality_filtered_normalized_within_pf_rms": _finite_mean(normalized_within_rms[~quality_filtered_roi_mask]),
            "mean_notquality_filtered_normalized_outside_pf_rms": _finite_mean(normalized_outside_rms[~quality_filtered_roi_mask]),
            "mean_notquality_filtered_outside_minus_within_pf_rms": _finite_mean(difference[~quality_filtered_roi_mask]),
            "mean_notquality_filtered_normalized_outside_minus_within_pf_rms": _finite_mean(normalized_difference[~quality_filtered_roi_mask]),
        }


@dataclass(frozen=True)
class VectorGainRankConfig(AnalysisConfigBase):
    """Score external_placefield_1d_vector_gain at each SVD rank from 1 to max_rank.

    Fits N=200 SVD components in one pass using existing best hyperparameters
    (rank-agnostic cache shared with RegressionConfig), then evaluates MSE and R²
    at each rank 1…200 on the test split.

    Parameters
    ----------
    spks_type : SpksTypes
        Spike type to use for the population.
    method : str
        Hyperparameter optimization method.
    activity_parameters_name : str
        Activity scaling method.
    """

    schema_version: str = "v1"

    data_config_name: str = "default"
    spks_type: SpksTypes = "sigrebase"
    method: str = "preferred"
    activity_parameters_name: str = "default"

    # Separate this from other parameters
    max_rank: ClassVar[int] = 200
    display_name: ClassVar[str] = "vector_gain_rank"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
            # "spks_type": list(VALID_SPKS_TYPES), # no longer analyzing anything except sigrebase
        }

    def validate(self):
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. Available: {', '.join(list(ACTIVITY_PARAMETERS_NAMES))}"
            )

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"method={self.method}",
        ]
        if self.activity_parameters_name != "default":
            parts.append(f"ap={self.activity_parameters_name}")
        parts.append(self.schema_version)
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Fit external_placefield_1d_vector_gain with N=200 SVD ranks and score at each rank."""
        activity_parameters = get_activity_parameters(self.activity_parameters_name)
        _shared_kwargs = dict(
            registry=registry,
            internal=False,
            gain=True,
            vector_gain=True,
            activity_parameters=activity_parameters,
        )
        # rank=1 for hyp lookup/optimization — same cache key as RegressionConfig, same SVD cost
        hyp_model = PlaceFieldModel(**_shared_kwargs, rank=1)
        # rank=max_rank for the actual multi-rank fit
        fit_model = PlaceFieldModel(**_shared_kwargs, rank=self.max_rank)

        hyperparameters = hyp_model.get_best_hyperparameters(
            session,
            spks_type=self.spks_type,
            method=self.method,
        )[0]

        # Train once — SVD produces U of shape (n_cells, max_rank)
        target_placefield, source_placefield, (U_target, U_source) = fit_model.train(
            session,
            spks_type=self.spks_type,
            split="train",
            hyperparameters=hyperparameters,
        )

        # Get test data
        source_data, target_data, frame_behavior = fit_model.get_session_data(session, self.spks_type, "test")
        source_data_np = source_data.numpy()

        # Source prediction and NaN filtering (mirrors predict() logic)
        source_prediction = get_placefield_prediction(source_placefield, frame_behavior)[0].T
        idx_nan = np.any(np.isnan(source_prediction), axis=0) | np.any(np.isnan(source_data_np), axis=0)
        idx_valid = ~idx_nan
        source_prediction = source_prediction[:, idx_valid]
        source_data_np = source_data_np[:, idx_valid]
        frame_behavior_filtered = frame_behavior.filter(np.where(idx_valid)[0])

        source_deviation = source_data_np - source_prediction  # (n_source, T)

        # Base target prediction on filtered frames
        target_prediction = get_placefield_prediction(target_placefield, frame_behavior_filtered)[0].T  # (n_target, T)

        # Filtered target activity for scoring
        target_data_np = target_data.numpy()[:, idx_valid]

        # Precompute latent projections for all ranks at once: (max_rank, T)
        latent = U_source.T @ source_deviation

        scores: dict = defaultdict(lambda: np.full(self.max_rank, np.nan))
        for rank in range(1, self.max_rank + 1):
            arousal_activity = U_target[:, :rank] @ latent[:rank, :]  # (n_target, T)
            prediction = target_prediction + arousal_activity
            _mse = float(mse(prediction, target_data_np, reduce="mean", dim=None))
            _r2 = float(measure_r2(prediction, target_data_np, reduce="mean", dim=None))
            scores["mse"][rank - 1] = _mse
            scores["r2"][rank - 1] = _r2

        return dict(scores)


def _pack_sweep(name: str, values: np.ndarray, dim: np.ndarray, mse_arr: np.ndarray, r2_arr: np.ndarray) -> dict:
    """Flatten one dimensionality sweep into ``{name}_{values,dim,mse,r2}`` arrays."""
    return {
        f"{name}_values": np.asarray(values, dtype=float),
        f"{name}_dim": np.asarray(dim, dtype=float),
        f"{name}_mse": np.asarray(mse_arr, dtype=float),
        f"{name}_r2": np.asarray(r2_arr, dtype=float),
    }


@dataclass(frozen=True)
class RegressionDimensionalitySweepConfig(AnalysisConfigBase):
    """Sweep test performance as a function of regressor dimensionality.

    For each figure-2 model, holds the best (cached) hyperparameters fixed and
    sweeps the model's dimensionality knob on the test split:

    - Placefield models: ``num_bins`` (log 1..200).
    - RBFPos / pos_speed / full-regressor models: ``num_basis`` (log 1..200).
    - RRR: ``rank`` (log 1..2000, clipped to the achievable rank). The model is
      fit once and re-scored at each rank, since RRR training is rank-agnostic.

    For the ``num_bins``/``num_basis`` sweeps, the Gaussian smoothing width
    (``smooth_width`` / ``basis_width``) is re-derived at every value as
    ``SMOOTH_SCALE * env_length / value`` instead of held fixed at the best-hyperparameter
    value. Fixed smoothing means resolution plateaus once bin spacing drops well below the
    smoothing width — the extra bins carry no new information because the smoothing kernel
    already blends neighbors together. Scaling smoothing to bin spacing keeps the kernel
    covering roughly one neighboring bin (adjacent-bin correlation ~= exp(-1) at
    SMOOTH_SCALE=0.5) regardless of ``num_bins``/``num_basis``, so the sweep reflects
    resolution rather than a fixed low-pass filter.

    Results are flat ``{sweep}_{values,dim,mse,r2}`` arrays, where ``dim`` is the
    nominal regressor dimensionality for the swept configuration. The activity
    preprocessing preset is swept over ``VALID_ACTIVITY_PARAMETERS``.

    Parameters
    ----------
    model_name : ModelName
        Name of the regression model (must be in ``KEY_FIGURE_MODELS``).
    spks_type : SpksTypes
        Spike type to use for the population.
    method : str
        Hyperparameter selection method used to fix the baseline hyperparameters.
    activity_parameters_name : str
        Named activity preprocessing preset.
    """

    schema_version: str = "v3"
    # v2: scale smooth_width/basis_width to bin spacing during the num_bins/num_basis
    # sweep instead of holding it fixed, so resolution doesn't plateau from oversmoothing.
    # v3: because upstream regression models changed and we need to rerun the sweep with new params.

    data_config_name: str = "default"
    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "sigrebase"
    method: str = "best"
    activity_parameters_name: str = "default"

    SMOOTH_SCALE: ClassVar[float] = 0.5
    display_name: ClassVar[str] = "regression_dim_sweep"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "model_name": list(KEY_FIGURE_MODELS),
            "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
        }

    def validate(self):
        if self.model_name not in KEY_FIGURE_MODELS:
            raise ValueError(f"Unknown model_name {self.model_name!r}. Available: {', '.join(KEY_FIGURE_MODELS)}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. " f"Available: {', '.join(ACTIVITY_PARAMETERS_NAMES)}"
            )

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"{self.model_name}",
            f"spks={self.spks_type}",
            f"method={self.method}",
        ]
        if self.activity_parameters_name != "default":
            parts.append(f"ap={self.activity_parameters_name}")
        parts.append(self.schema_version)
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Run the dimensionality sweep appropriate for this model."""
        model = get_model(self.model_name, registry, activity_parameters=self.activity_parameters_name)
        num_env = len(session.environments)
        base_hp = model.get_best_hyperparameters(session, spks_type=self.spks_type, method=self.method)[0]

        if isinstance(model, PlaceFieldModel):
            if np.unique(session.env_length).size != 1:
                raise ValueError("All trials must have the same environment length!")
            env_length = float(session.env_length[0])
            return _pack_sweep(
                "num_bins", *self._sweep_param(model, session, base_hp, num_env, "num_bins", NUM_BINS_VALUES, "smooth_width", env_length)
            )

        if isinstance(model, FullRegressorModel) or isinstance(model, RBFPosModel):
            if np.unique(session.env_length).size != 1:
                raise ValueError("All trials must have the same environment length!")
            env_length = float(session.env_length[0])
            return _pack_sweep(
                "num_basis", *self._sweep_param(model, session, base_hp, num_env, "num_basis", NUM_BASIS_VALUES, "basis_width", env_length)
            )

        if isinstance(model, ReducedRankRegressionModel):
            return self._sweep_rrr(model, session, base_hp)

        raise TypeError(f"No dimensionality sweep defined for model type {type(model).__name__}")

    def _sweep_param(
        self,
        model,
        session: B2Session,
        base_hp,
        num_env: int,
        param_name: str,
        values: np.ndarray,
        smooth_param_name: str,
        env_length: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Refit and score ``model`` at each value of a single integer hyperparameter.

        At each ``value``, ``smooth_param_name`` (``smooth_width``/``basis_width``) is
        overridden to ``SMOOTH_SCALE * env_length / value`` so the smoothing kernel tracks
        bin spacing instead of staying fixed at the best-hyperparameter width.
        """
        n = len(values)
        dim = np.full(n, np.nan)
        mse_arr = np.full(n, np.nan)
        r2_arr = np.full(n, np.nan)
        for i, value in enumerate(values):
            bin_spacing = env_length / float(value)
            smooth_width = self.SMOOTH_SCALE * bin_spacing
            hyperparameters = replace(base_hp, **{param_name: int(value), smooth_param_name: smooth_width})
            dim[i] = model.regressor_dimensionality(num_env, hyperparameters=hyperparameters)
            report = model.process(session, spks_type=self.spks_type, hyperparameters=hyperparameters)
            mse_arr[i] = float(report.metrics["mse"])
            r2_arr[i] = float(report.metrics["r2"])
        return values, dim, mse_arr, r2_arr

    def _sweep_rrr(self, model: ReducedRankRegressionModel, session: B2Session, base_hp) -> dict:
        """Fit RRR once at the best alpha, then re-score the test split at each rank."""
        source_data, target_data_train, _ = model.get_session_data(session, self.spks_type, "train")
        max_rank = int(min(source_data.shape[0], target_data_train.shape[0]))
        ranks = RANK_VALUES[RANK_VALUES <= max_rank]

        trained_model = model.train(session, spks_type=self.spks_type, split="train", hyperparameters=base_hp)
        target_test = model.get_session_data(session, self.spks_type, "test")[1]

        n = len(ranks)
        mse_arr = np.full(n, np.nan)
        r2_arr = np.full(n, np.nan)
        for i, rank in enumerate(ranks):
            hyperparameters = replace(base_hp, rank=int(rank))
            prediction, extras = model.predict(
                session,
                trained_model,
                spks_type=self.spks_type,
                split="test",
                hyperparameters=hyperparameters,
            )
            target = target_test
            if extras.get("predictions_were_filtered", False):
                target = target_test[:, extras["idx_valid_predictions"]]
            metrics = model.evaluate(prediction, target)
            mse_arr[i] = float(metrics["mse"])
            r2_arr[i] = float(metrics["r2"])

        return _pack_sweep("rank", ranks, ranks.astype(float), mse_arr, r2_arr)
