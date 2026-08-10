"""ScoreModelsConfig — wraps regression model scoring from score_models.py.

The regression model infrastructure manages its own file-based caches (joblib files in
``hyperparameter_path`` and ``score_path``), keyed independently of this module's
``schema_version``. Those caches hold the optimized hyperparameters and the whole-split
metrics; anything derived from the held-out *predictions* is not cached there and is
recomputed by the config that needs it.
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
from ..env_order import MAX_ENV_SLOTS, load_env_order
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
    make_gain_unit_index,
    least_squares_gain,
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
    "external_placefield_1d_structured_gain",
    "internal_placefield_1d_structured_gain",
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


#: Per-environment score keys, all indexed by experience-order slot rather than by environment
#: index: the two cohorts run disjoint environment sets, so slot 0 ("the first environment this
#: mouse saw") is the only axis that means the same thing for every mouse. ``env_slot_ids`` records
#: which environment index each slot actually held.
PER_ENV_SCORE_KEYS: tuple[str, ...] = (
    "mse_env",
    "r2_env",
    "mse_roi_env",
    "r2_roi_env",
    "num_frames_env",
    "env_slot_ids",
)


def _per_environment_scores(
    model,
    session: B2Session,
    spks_type: SpksTypes,
    report,
) -> dict[str, np.ndarray]:
    """Score one held-out prediction separately within each environment.

    The model is untouched: it is fit once across all environments and then scored once per
    environment, so this measures where a single global fit does well — not what a per-environment
    fit would achieve.

    The metrics mirror the whole-split ones. ``mse_env`` / ``r2_env`` reduce over ROIs and frames
    together (``dim=None``, matching :meth:`RegressionModel.evaluate`); ``mse_roi_env`` /
    ``r2_roi_env`` reduce over frames alone, matching the ``mse_roi`` / ``r2_roi`` keys.

    Parameters
    ----------
    model : RegressionModel
        The model that produced ``report``, used to recover the held-out frame behavior.
    session : B2Session
        Session being scored.
    spks_type : SpksTypes
        Spike type the model was scored with.
    report : RegressionModel.Report
        Report from ``model.process`` on the held-out split.

    Returns
    -------
    dict[str, np.ndarray]
        The keys in :data:`PER_ENV_SCORE_KEYS`, each indexed by experience-order slot. Slots the
        session never visits are NaN, except ``num_frames_env``, which is 0 there.

    Notes
    -----
    R² does not decompose across environments. Each environment's R² is measured against that
    environment's own mean, whereas the whole-split R² uses a single baseline that must also
    account for the activity difference *between* environments. Per-environment values are
    therefore usually higher than the whole-split value and must not be averaged into it.

    An environment can hold very few held-out frames — the test split is roughly a tenth of a
    session and the environments are not visited equally — so ``num_frames_env`` is reported
    alongside, and slots with few frames should be filtered rather than trusted.
    """
    _, _, frame_behavior = model.get_session_data(session, spks_type, "test")
    if report.extras.get("predictions_were_filtered", False):
        frame_behavior = frame_behavior.filter(report.extras["idx_valid_predictions"])

    target = np.asarray(report.target_data)
    prediction = np.asarray(report.predicted_data)
    if target.shape[1] != len(frame_behavior):
        raise ValueError(f"Held-out target has {target.shape[1]} frames but behavior has {len(frame_behavior)}")

    mouse_order = load_env_order().get(session.mouse_name)
    env_slot_ids = np.full(MAX_ENV_SLOTS, np.nan)
    if mouse_order is not None:
        env_slot_ids[: len(mouse_order)] = mouse_order

    scores: dict[str, np.ndarray] = {
        "mse_env": np.full(MAX_ENV_SLOTS, np.nan),
        "r2_env": np.full(MAX_ENV_SLOTS, np.nan),
        "mse_roi_env": np.full((MAX_ENV_SLOTS, target.shape[0]), np.nan),
        "r2_roi_env": np.full((MAX_ENV_SLOTS, target.shape[0]), np.nan),
        "num_frames_env": np.zeros(MAX_ENV_SLOTS),
        "env_slot_ids": env_slot_ids,
    }
    if mouse_order is None:
        return scores

    environment = np.asarray(frame_behavior.environment)
    for env in np.unique(environment):
        env = int(env)
        # Negative indices are the unlabeled sentinel, and an environment outside this mouse's
        # order (or past the last slot) has no slot to be reported in.
        if env < 0 or env not in mouse_order:
            continue
        slot = mouse_order.index(env)
        if slot >= MAX_ENV_SLOTS:
            continue
        idx = environment == env
        env_target = target[:, idx]
        env_prediction = prediction[:, idx]
        scores["mse_env"][slot] = float(mse(env_prediction, env_target, reduce="mean", dim=None))
        scores["r2_env"][slot] = float(measure_r2(env_prediction, env_target, reduce="mean", dim=None))
        scores["mse_roi_env"][slot] = np.asarray(mse(env_prediction, env_target, reduce="none", dim=1))
        scores["r2_roi_env"][slot] = np.asarray(measure_r2(env_prediction, env_target, reduce="none", dim=1))
        scores["num_frames_env"][slot] = int(np.sum(idx))
    return scores


@dataclass(frozen=True)
class RegressionConfig(AnalysisConfigBase):
    """Configuration for regression model scoring.

    Result keys
    -----------
    Whole-split metrics, straight from the model's own score cache::

        mse, r2                          scalars, reduced over ROIs and frames together
        mse_roi, r2_roi                  (rois,), reduced over frames

    Per-environment metrics, from :func:`_per_environment_scores`, indexed by experience-order
    slot (see ``..env_order``) so slot 0 is the first environment each mouse saw::

        mse_env, r2_env                  (MAX_ENV_SLOTS,)
        mse_roi_env, r2_roi_env          (MAX_ENV_SLOTS, rois)
        num_frames_env                   (MAX_ENV_SLOTS,) held-out frames scored in each slot
        env_slot_ids                     (MAX_ENV_SLOTS,) environment index behind each slot

    The model itself is unchanged — one fit across all environments, scored once per environment.
    Per-environment R² is measured against each environment's own mean and so does not decompose
    into the whole-split value; see :func:`_per_environment_scores`.

    Parameters
    ----------
    model_name : ModelName
        Name of the regression model (must be in ``MODEL_NAMES``).
    spks_type : SpksTypes
        Spike type to use for the population.
    method : str
        Hyperparameter optimization method. Must resolve to a single method, so ``"best"`` is
        rejected: it picks the cached score and the cached hyperparameters by different criteria
        (held-out MSE versus validation MSE), which would let the whole-split metrics come from
        one optimization run and the per-environment metrics from another.
    """

    schema_version: str = "v4"
    # v3: recompute with numerically improved placefield code
    # v4: add per-env scores (using global session fits)

    data_config_name: str = "default"
    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "sigrebase"
    method: str = "preferred"
    activity_parameters_name: str = "std"

    display_name: ClassVar[str] = "regression"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "model_name": list(KEY_FIGURE_MODELS),
            # "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS), # only using std!
            # "spks_type": list(VALID_SPKS_TYPES), # no longer analyzing anything except sigrebase
        }

    def validate(self):
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model_name {self.model_name!r}. " f"Available: {', '.join(MODEL_NAMES)}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. Available: {', '.join(list(ACTIVITY_PARAMETERS_NAMES))}"
            )
        if self.method == "best":
            raise ValueError(
                "method='best' would select the cached score and the cached hyperparameters by "
                "different criteria, so the whole-split and per-environment metrics could come "
                "from different optimization runs. Name a single method (e.g. 'preferred')."
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
        """Score the model on this session, over the whole held-out split and within each environment.

        The whole-split metrics come from the model's own joblib score cache, which is keyed
        independently of this config's ``schema_version`` and so is a cache hit even right after a
        schema bump. The per-environment metrics need the held-out predictions themselves, which
        that cache does not hold, so the model is refit once at its cached-best hyperparameters —
        one train plus one predict, with no hyperparameter re-optimization.
        """
        model = get_model(self.model_name, registry, activity_parameters=self.activity_parameters_name)
        score = model.get_best_score(
            session,
            spks_type=self.spks_type,
            method=self.method,
        )
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
        return {**score, **_per_environment_scores(model, session, self.spks_type, report)}


# Residual localization is reported for two membership place fields: ``xval`` estimates the place
# field on the training frames and reads it out on the held-out frames, so the membership never saw
# the residuals it weights; ``infold`` estimates it on the same held-out frames, which is the
# optimistic in-sample reference the cross-validated number should be compared against.
RESIDUAL_FOLDS: tuple[str, ...] = ("xval", "infold")
RESIDUAL_FOLD_SPLITS: dict[str, str] = {"xval": "train", "infold": "test"}
RESIDUAL_REGIONS: tuple[str, ...] = ("within", "outside")
RESIDUAL_METRICS: tuple[str, ...] = ("rms", "normalized_rms", "r2_weighted", "r2_shared")
# Scalar summaries are reported over all ROIs and over each side of the quality filter.
RESIDUAL_SUBSETS: tuple[str, ...] = ("", "quality_filtered_", "notquality_filtered_")
# Both an across-ROI mean and median are reported. The mean is the historical summary, but every
# one of these metrics is right-skewed across the population, and ``r2_weighted`` catastrophically
# so: an ROI whose within-field weighted variance is near zero produces an R² in the thousands,
# which a mean over ~2000 ROIs cannot survive (observed mean -938 against a median of -0.07 on a
# real session). Prefer the median unless the mean is specifically wanted.
RESIDUAL_STATISTICS: tuple[str, ...] = ("mean", "median")


def _validate_residual_inputs(
    residual: np.ndarray,
    placefield_prediction: np.ndarray,
    placefield_peak: np.ndarray,
) -> None:
    """Check that a residual, its place-field prediction, and the per-ROI peak line up."""
    if residual.ndim != 2:
        raise ValueError(f"residual must have shape (rois, frames), got {residual.shape}")
    if placefield_prediction.shape != residual.shape:
        raise ValueError("placefield_prediction must match residual shape " f"{residual.shape}, got {placefield_prediction.shape}")
    if placefield_peak.shape != (residual.shape[0],):
        raise ValueError(f"placefield_peak must have shape ({residual.shape[0]},), got {placefield_peak.shape}")


def _placefield_membership_weights(
    placefield_prediction: np.ndarray,
    placefield_peak: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Within- and outside-field temporal weights from a soft place-field membership.

    The place-field prediction is converted to a membership in ``[0, 1]`` by dividing each ROI by
    the peak of its place-field map. Within-field weights are proportional to that membership;
    outside-field weights are proportional to one minus the membership. Both are zeroed wherever
    ``valid`` is False, so every downstream metric ignores those frames.

    Parameters
    ----------
    placefield_prediction : np.ndarray
        Place-field prediction of shape (rois, frames).
    placefield_peak : np.ndarray
        Peak of each ROI's place-field map, shape (rois,).
    valid : np.ndarray
        Boolean mask of shape (rois, frames) marking usable samples.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Within-field and outside-field weights, both of shape (rois, frames).
    """
    valid_peak = np.isfinite(placefield_peak) & (placefield_peak > 0)
    valid = valid & np.isfinite(placefield_prediction) & valid_peak[:, None]
    membership = np.divide(
        np.clip(placefield_prediction, 0, None),
        placefield_peak[:, None],
        out=np.full_like(placefield_prediction, np.nan),
        where=valid_peak[:, None],
    )
    membership = np.clip(membership, 0, 1)
    return np.where(valid, membership, 0.0), np.where(valid, 1 - membership, 0.0)


def _weighted_mse(residual: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-ROI weighted mean squared residual, ``sum(w * r**2) / sum(w)``.

    Callers must zero the weight of every sample whose residual is not finite -- which is what
    :func:`_placefield_membership_weights` does -- otherwise the NaN propagates into the result.

    Parameters
    ----------
    residual : np.ndarray
        Residual of shape (rois, frames).
    weights : np.ndarray
        Non-negative weights of shape (rois, frames).

    Returns
    -------
    np.ndarray
        Weighted MSE per ROI, NaN where the weights sum to zero.
    """
    total = np.sum(weights, axis=1)
    error = np.where(weights > 0, residual, 0.0)
    return np.divide(
        np.sum(weights * error**2, axis=1),
        total,
        out=np.full(residual.shape[0], np.nan),
        where=total > 0,
    )


def _weighted_r2(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-ROI weighted R², measured against the weighted mean of the target.

    ``1 - sum(w * (y - yhat)**2) / sum(w * (y - ybar_w)**2)``, where ``ybar_w`` is the weighted
    mean of the target over the same weights. The baseline is therefore specific to the weight set:
    within- and outside-field R² are each well posed, but they are not on a common scale because
    place-field weighting concentrates on frames where the ROI is most active.

    Parameters
    ----------
    target : np.ndarray
        Observed activity of shape (rois, frames).
    prediction : np.ndarray
        Model prediction of shape (rois, frames).
    weights : np.ndarray
        Non-negative weights of shape (rois, frames).

    Returns
    -------
    np.ndarray
        Weighted R² per ROI, NaN where the weights sum to zero or the weighted target has no
        variance.
    """
    total = np.sum(weights, axis=1)
    weighted_mean = np.divide(
        np.sum(weights * np.where(weights > 0, target, 0.0), axis=1),
        total,
        out=np.full(target.shape[0], np.nan),
        where=total > 0,
    )
    residual = np.where(weights > 0, target - prediction, 0.0)
    deviation = np.where(weights > 0, target - weighted_mean[:, None], 0.0)
    ss_res = np.sum(weights * residual**2, axis=1)
    ss_tot = np.sum(weights * deviation**2, axis=1)
    ratio = np.full(target.shape[0], np.nan)
    np.divide(ss_res, ss_tot, out=ratio, where=ss_tot > 0)
    return 1.0 - ratio


def _placefield_weighted_residual_rms(
    residual: np.ndarray,
    placefield_prediction: np.ndarray,
    placefield_peak: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ROI residual RMS inside and outside a soft place field.

    Thin wrapper over :func:`_placefield_membership_weights` and :func:`_weighted_mse`; each set of
    temporal weights is normalized independently before calculating RMS.
    """
    residual = np.asarray(residual, dtype=float)
    placefield_prediction = np.asarray(placefield_prediction, dtype=float)
    placefield_peak = np.asarray(placefield_peak, dtype=float)
    _validate_residual_inputs(residual, placefield_prediction, placefield_peak)
    within_weight, outside_weight = _placefield_membership_weights(
        placefield_prediction,
        placefield_peak,
        np.isfinite(residual),
    )
    return np.sqrt(_weighted_mse(residual, within_weight)), np.sqrt(_weighted_mse(residual, outside_weight))


def _residual_fold_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    placefield_prediction: np.ndarray,
    placefield_peak: np.ndarray,
    variance_pf: np.ndarray,
) -> dict[str, np.ndarray]:
    """Every per-ROI residual metric for one membership place field.

    Emits ``{region}_pf_{metric}`` for each region in :data:`RESIDUAL_REGIONS` and metric in
    :data:`RESIDUAL_METRICS`, plus the ``outside_minus_within_pf_{metric}`` differences. The
    difference keys are literal: for R² a positive difference means the *outside*-field fit is
    better, the opposite reading from the RMS differences.

    ``r2_shared`` normalizes the weighted MSE by the ROI's total held-out variance rather than by
    the weighted target variance, which makes it exactly ``1 - normalized_rms**2``. It is stored as
    a convenience for readers who think in R² units; ``r2_weighted`` is the one that carries
    information the RMS keys do not.

    Parameters
    ----------
    target : np.ndarray
        Held-out target activity of shape (rois, frames).
    prediction : np.ndarray
        Model prediction of shape (rois, frames).
    placefield_prediction : np.ndarray
        Membership place-field prediction of shape (rois, frames).
    placefield_peak : np.ndarray
        Peak of each ROI's membership place field, shape (rois,).
    variance_pf : np.ndarray
        Total held-out variance per ROI (ddof=0), shape (rois,).

    Returns
    -------
    dict[str, np.ndarray]
        Per-ROI metric arrays keyed without a fold prefix.
    """
    residual = target - prediction
    _validate_residual_inputs(residual, placefield_prediction, placefield_peak)
    weights = dict(
        zip(
            RESIDUAL_REGIONS,
            _placefield_membership_weights(placefield_prediction, placefield_peak, np.isfinite(residual)),
        )
    )
    target_std = np.sqrt(variance_pf)
    scale_ok = np.isfinite(variance_pf) & (variance_pf > 0)

    metrics: dict[str, np.ndarray] = {}
    for region, weight in weights.items():
        weighted_mse = _weighted_mse(residual, weight)
        rms = np.sqrt(weighted_mse)
        metrics[f"{region}_pf_rms"] = rms
        metrics[f"{region}_pf_normalized_rms"] = np.divide(rms, target_std, out=np.full_like(rms, np.nan), where=scale_ok)
        metrics[f"{region}_pf_r2_shared"] = 1.0 - np.divide(
            weighted_mse,
            variance_pf,
            out=np.full_like(weighted_mse, np.nan),
            where=scale_ok,
        )
        metrics[f"{region}_pf_r2_weighted"] = _weighted_r2(target, prediction, weight)

    for metric in RESIDUAL_METRICS:
        metrics[f"outside_minus_within_pf_{metric}"] = metrics[f"outside_pf_{metric}"] - metrics[f"within_pf_{metric}"]
    return metrics


def residual_per_roi_keys() -> list[str]:
    """Every per-ROI array key emitted by :class:`RegressionPlacefieldResidualConfig`."""
    keys = [f"{fold}_{region}_pf_{metric}" for fold in RESIDUAL_FOLDS for region in RESIDUAL_REGIONS for metric in RESIDUAL_METRICS]
    keys += [f"{fold}_outside_minus_within_pf_{metric}" for fold in RESIDUAL_FOLDS for metric in RESIDUAL_METRICS]
    return keys + ["variance_pf"]


def residual_summary_keys() -> list[str]:
    """Every scalar summary key emitted by :class:`RegressionPlacefieldResidualConfig`."""
    return [f"{statistic}_{subset}{key}" for statistic in RESIDUAL_STATISTICS for subset in RESIDUAL_SUBSETS for key in residual_per_roi_keys()]


def _finite_mean(values: np.ndarray) -> float:
    """Mean of finite values, or NaN when no finite values exist."""
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _finite_median(values: np.ndarray) -> float:
    """Median of finite values, or NaN when no finite values exist."""
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def _residual_summary_scalars(per_roi: dict[str, np.ndarray], quality_filtered_roi_mask: np.ndarray) -> dict[str, float]:
    """Across-ROI finite mean and median of every per-ROI array, for each ROI subset."""
    subset_masks = {
        "": np.ones_like(quality_filtered_roi_mask, dtype=bool),
        "quality_filtered_": quality_filtered_roi_mask,
        "notquality_filtered_": ~quality_filtered_roi_mask,
    }
    reducers = {"mean": _finite_mean, "median": _finite_median}
    return {
        f"{statistic}_{subset}{key}": reduce(values[mask])
        for statistic, reduce in reducers.items()
        for subset, mask in subset_masks.items()
        for key, values in per_roi.items()
    }


def _membership_placefield(
    model,
    session: B2Session,
    spks_type: SpksTypes,
    split: str,
    dist_edges: np.ndarray,
    smooth_width: float | None,
):
    """Estimate the membership place field from one time split's frames.

    ``model.get_session_data`` applies the exact activity scaling the model was fit with, so the
    place field is in the same units as the residuals it will weight.
    """
    _, target, frame_behavior = model.get_session_data(session, spks_type, split)
    return get_placefield(
        target.T.numpy(),
        frame_behavior,
        dist_edges=dist_edges,
        speed_threshold=None,
        average=True,
        smooth_width=smooth_width,
        zero_to_nan=True,
    )


def _placefield_peak(placefield) -> np.ndarray:
    """Peak of each ROI's place-field map, NaN for ROIs with no finite bin."""
    finite = np.isfinite(placefield.placefield)
    peak = np.max(np.where(finite, placefield.placefield, -np.inf), axis=(0, 1))
    peak[~np.any(finite, axis=(0, 1))] = np.nan
    return peak


def _membership_prediction(placefield, frame_behavior, fold: str, session: B2Session) -> np.ndarray:
    """Read a membership place field out on a set of frames, shaped (rois, frames).

    The ``xval`` place field is built from the training frames only, so it can in principle miss an
    environment that the held-out frames visit. That would make the residual localization silently
    incomparable across sessions, so it is raised rather than dropped.
    """
    missing = np.setdiff1d(np.unique(frame_behavior.environment), placefield.environment)
    if missing.size:
        raise ValueError(
            f"The {fold!r} membership place field for {session.session_print()} has no map for "
            f"environment(s) {missing.tolist()}, which the held-out frames visit."
        )
    return get_placefield_prediction(placefield, frame_behavior)[0].T


@dataclass(frozen=True)
class RegressionPlacefieldResidualConfig(AnalysisConfigBase):
    """Compare held-out model residuals within and outside each target ROI's place field.

    The parameter grid deliberately mirrors :class:`RegressionConfig`: every
    regression result therefore has a corresponding residual-localization job.

    Result keys
    -----------
    Almost every key is assembled from the same three axes, so the whole result dict is::

        fold      = xval | infold                            which frames the membership PF saw
        region    = within | outside                         side of the soft place field
        metric    = rms | normalized_rms | r2_weighted | r2_shared
        statistic = mean | median                            across-ROI, over finite values only
        subset    = <empty> | quality_filtered_ | notquality_filtered_

    Per-ROI arrays, shape (rois,), enumerated by :func:`residual_per_roi_keys` -- 25 keys::

        {fold}_{region}_pf_{metric}                          16 keys
        {fold}_outside_minus_within_pf_{metric}               8 keys, literal outside minus within
        variance_pf                                           total held-out variance per ROI

    Scalar summaries, one per (statistic, subset, per-ROI key), enumerated by
    :func:`residual_summary_keys` -- 150 keys::

        {statistic}_{subset}{per_roi_key}

    ROI-quality diagnostics, computed once on the full split and shared by both folds::

        reliability, fraction_active                          (environments, rois)
        quality_environments                                  (environments,), env id of each row
        quality_filtered_roi_mask                             (rois,) bool
        num_quality_filtered_rois                             int

    The membership place field is estimated twice, in the same activity units used by the model.
    The ``xval`` fold estimates it on the training frames and reads it out on the held-out frames,
    so the membership never saw the residuals it weights. The ``infold`` fold estimates it on the
    held-out frames themselves, which is the in-sample reference the cross-validated number should
    be compared against. Both folds weight residuals from the *same* held-out frames -- only the
    membership changes.

    Four metrics are reported per (fold, region): ``rms``, ``normalized_rms`` (RMS over the ROI's
    held-out standard deviation), ``r2_weighted`` (weighted R² against the weighted mean of the
    target), and ``r2_shared`` (weighted MSE over the ROI's total held-out variance). Note that
    ``r2_weighted`` uses a baseline specific to each weight set, so the within- and outside-field
    values are each well posed but are not on a common scale; ``r2_shared`` shares one baseline and
    is exactly ``1 - normalized_rms**2``.

    Each per-ROI array is summarized by both an across-ROI ``mean_*`` and ``median_*``. These
    distributions are right-skewed, so the two differ substantially even for RMS; for
    ``r2_weighted`` the mean is unusable, because an ROI whose within-field weighted variance is
    near zero produces an R² in the thousands and swamps the population.

    Note that ``infold`` covers noticeably fewer ROIs than ``xval``. The held-out split is about a
    tenth of the session, so ROIs that are silent over it have an all-NaN place-field map and no
    usable membership. The two folds are therefore not measured over identical ROI sets --
    ``quality_filtered_roi_mask`` plus each fold's own finite mask is the honest comparison.

    The ROI quality filter is deliberately *not* cross-validated: it selects which neurons are
    compared, so it stays on the full split and is identical across folds.
    """

    schema_version: str = "v6"
    # v6: estimate the membership place field per fold (xval on train, infold on test) instead of
    # once on the full split, and report weighted R² alongside RMS.
    # v5: compute the ROI quality filter the same way cvpca / stimspace_spectra do
    # (fast behavioral sampling, unvisited bins at zero instead of NaN). v4 required
    # every bin to be occupied on every trial, which left zero quality ROIs in most
    # sessions.
    data_config_name: str = "default"
    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "sigrebase"
    method: str = "preferred"
    activity_parameters_name: str = "std"

    num_placefield_bins: int = 100
    placefield_smooth_width: float | None = None
    reliability_threshold: float = 0.3
    fraction_active_threshold: float = 0.1
    display_name: ClassVar[str] = "regression_pf_residual"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "model_name": list(KEY_FIGURE_MODELS),
            # "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
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

        # One membership place field per fold, in the model's own activity units.
        dist_edges = np.linspace(0, session.env_length[0], self.num_placefield_bins + 1)
        placefields = {
            fold: _membership_placefield(
                model,
                session,
                self.spks_type,
                split,
                dist_edges,
                self.placefield_smooth_width,
            )
            for fold, split in RESIDUAL_FOLD_SPLITS.items()
        }

        # The ROI quality filter is a neuron-selection criterion rather than a residual estimate,
        # so it stays on the full split and is shared by both folds.
        _, target_full, frame_behavior_full = model.get_session_data(session, self.spks_type, "full")

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
        if report.extras.get("predictions_were_filtered", False):
            frame_behavior_test = frame_behavior_test.filter(report.extras["idx_valid_predictions"])

        target = np.asarray(report.target_data)
        prediction = np.asarray(report.predicted_data)
        # One total held-out variance per ROI provides the scale needed to compare
        # residual magnitudes across sessions. Using ddof=0 matches the SSE / SST
        # normalization in R²: RMS / sqrt(variance) is sqrt(SSE / SST) when the
        # same frames and weights are used.
        variance_pf = np.var(target, axis=1)

        per_roi: dict[str, np.ndarray] = {"variance_pf": variance_pf}
        for fold, placefield in placefields.items():
            pf_prediction = _membership_prediction(placefield, frame_behavior_test, fold, session)
            if pf_prediction.shape != target.shape or prediction.shape != target.shape:
                raise ValueError(
                    f"Held-out target, model prediction, and {fold!r} PF prediction are misaligned: "
                    f"target={target.shape}, prediction={prediction.shape}, pf={pf_prediction.shape}"
                )
            fold_metrics = _residual_fold_metrics(
                target,
                prediction,
                pf_prediction,
                _placefield_peak(placefield),
                variance_pf,
            )
            per_roi.update({f"{fold}_{key}": values for key, values in fold_metrics.items()})

        return {
            **per_roi,
            **_residual_summary_scalars(per_roi, quality_filtered_roi_mask),
            "reliability": reliability,
            "fraction_active": fraction_active,
            "quality_environments": environments,
            "quality_filtered_roi_mask": quality_filtered_roi_mask,
            "num_quality_filtered_rois": int(np.sum(quality_filtered_roi_mask)),
        }


# Models whose held-out residual structure is worth storing in full. The plain place-field model
# supplies the raw place-field prediction, so the gain / structured-gain / peer-prediction variants
# can be read against it without a second denominator.
RESIDUAL_STRUCTURE_MODELS: list[ModelName] = [
    "external_placefield_1d",
    "internal_placefield_1d",
    "internal_placefield_1d_gain",
    "internal_placefield_1d_structured_gain",
    "rrr",
]

# Every key this config returns. All are excluded from ResultsAggregator stacking: they are
# per-session matrices meant to be inspected one session at a time, not padded into a grid.
RESIDUAL_STRUCTURE_KEYS: tuple[str, ...] = (
    "residual_additive",
    "residual_multiplicative",
    "model_prediction",
    "chunk_gain",
    "chunk_gain_unit_index",
    "chunk_gain_unit_frame_count",
    "chunk_gain_unit_trial",
    "chunk_gain_unit_environment",
    "frame_position",
    "frame_speed",
    "frame_trial",
    "frame_environment",
    "target_roi_index",
)


@dataclass(frozen=True)
class RegressionResidualStructureConfig(AnalysisConfigBase):
    """Store the held-out residual of a regression model in full, three ways.

    :class:`RegressionPlacefieldResidualConfig` reduces the residual to per-ROI scalars. This
    config keeps the matrices themselves so the structure of the residual *across the population*
    can be looked at directly, one session at a time:

    1. **Additive** -- ``target - prediction``, shape (targets, frames).
    2. **Multiplicative** -- ``target / (relu(prediction) + eps)``, with a per-neuron ``eps``
       proportional to that neuron's mean prediction. Every model in the grid rectifies its own
       output, ``*_structured_gain`` included now that its predicted gains are clipped at zero, so
       the ReLU here never bites and serves only to keep the denominator well posed.
    3. **Chunk gain** -- the least-squares scale factor per cell per (time chunk ∩ trial), the same
       estimator and shrinkage regularizer :class:`PlaceFieldStructuredGainModel` fits its gain
       regression on. A time chunk belongs to exactly one split, so a unit never mixes folds.

    ``min_frames_per_unit`` is deliberately not applied -- per-unit frame counts are returned so
    filtering short units stays the analyst's decision.

    Parameters
    ----------
    model_name : ModelName
        Name of the regression model (must be in ``RESIDUAL_STRUCTURE_MODELS``).
    spks_type : SpksTypes
        Spike type to use for the population.
    method : str
        Hyperparameter selection method used to fix the model's hyperparameters.
    activity_parameters_name : str
        Named activity preprocessing preset. Not a grid axis: the stored matrices are large, so
        only one preset is computed.
    multiplicative_eps_scale : float
        Denominator floor for the multiplicative residual, as a fraction of each neuron's mean
        rectified prediction. Relative rather than absolute so it means the same thing under every
        activity preset.
    chunk_gain_regularization : float
        Shrinkage of the chunk gains toward 1, passed to :func:`least_squares_gain`.
    """

    schema_version: str = "v1"

    data_config_name: str = "default"
    model_name: ModelName = "internal_placefield_1d"
    spks_type: SpksTypes = "sigrebase"
    method: str = "preferred"
    activity_parameters_name: str = "std"
    multiplicative_eps_scale: float = 0.01
    chunk_gain_regularization: float = 0.01

    display_name: ClassVar[str] = "regression_residual_structure"
    _result_handling: ClassVar[dict[str, str]] = {key: "skip" for key in RESIDUAL_STRUCTURE_KEYS}

    @staticmethod
    def _param_grid() -> dict:
        return {"model_name": list(RESIDUAL_STRUCTURE_MODELS)}

    def validate(self):
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model_name {self.model_name!r}. Available: {', '.join(MODEL_NAMES)}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. Available: " f"{', '.join(list(ACTIVITY_PARAMETERS_NAMES))}"
            )
        if self.multiplicative_eps_scale <= 0:
            raise ValueError("multiplicative_eps_scale must be positive")
        if self.chunk_gain_regularization < 0:
            raise ValueError("chunk_gain_regularization must be non-negative")

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"{self.model_name}",
            f"spks={self.spks_type}",
            f"method={self.method}",
        ]
        if self.activity_parameters_name != "default":
            parts.append(f"ap={self.activity_parameters_name}")
        parts.extend([f"eps={self.multiplicative_eps_scale:g}", f"gainreg={self.chunk_gain_regularization:g}"])
        parts.append(self.schema_version)
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Fit the cached-best model and store its held-out residual three ways."""
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
        target = np.asarray(report.target_data, dtype=float)
        prediction = np.asarray(report.predicted_data, dtype=float)

        # Chunk ids come from gaps in the split's *unfiltered* sample indices, so deriving them
        # first and subsetting afterwards preserves chunk identity. report.extras["source_data"]
        # is left unfiltered by process(), so it is never used for alignment here.
        _, _, frame_behavior = model.get_session_data(session, self.spks_type, "test")
        chunk_index = model.get_split_chunk_index(session, self.spks_type, "test")
        if report.extras.get("predictions_were_filtered", False):
            idx_valid = report.extras["idx_valid_predictions"]
            frame_behavior = frame_behavior.filter(idx_valid)
            chunk_index = chunk_index[idx_valid]
        if target.shape[1] != len(frame_behavior):
            raise ValueError(f"Held-out target has {target.shape[1]} frames but behavior has {len(frame_behavior)}")

        rectified = np.clip(prediction, 0, None)
        eps = self.multiplicative_eps_scale * np.mean(rectified, axis=1, keepdims=True)
        residual_multiplicative = np.divide(
            target,
            rectified + eps,
            out=np.full_like(target, np.nan),
            where=eps > 0,
        )

        unit_index, num_units = make_gain_unit_index(chunk_index, frame_behavior.trial)
        chunk_gain = least_squares_gain(target, prediction, unit_index, num_units, self.chunk_gain_regularization)
        # One representative frame per unit; a unit is a contiguous stretch of a single trial, so
        # its trial and environment are constant.
        unit_first_frame = np.unique(unit_index, return_index=True)[1]

        population, _ = registry.get_population(session, self.spks_type)
        target_roi_index = np.asarray(population.idx_neurons[population.cell_split_indices[1]])

        return {
            "residual_additive": (target - prediction).astype(np.float32),
            "residual_multiplicative": residual_multiplicative.astype(np.float32),
            "model_prediction": prediction.astype(np.float32),
            "chunk_gain": chunk_gain.astype(np.float32),
            "chunk_gain_unit_index": unit_index.astype(np.int32),
            "chunk_gain_unit_frame_count": np.bincount(unit_index, minlength=num_units).astype(np.int32),
            "chunk_gain_unit_trial": np.asarray(frame_behavior.trial)[unit_first_frame],
            "chunk_gain_unit_environment": np.asarray(frame_behavior.environment)[unit_first_frame],
            "frame_position": np.asarray(frame_behavior.position, dtype=np.float32),
            "frame_speed": np.asarray(frame_behavior.speed, dtype=np.float32),
            "frame_trial": np.asarray(frame_behavior.trial),
            "frame_environment": np.asarray(frame_behavior.environment),
            "target_roi_index": target_roi_index,
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
