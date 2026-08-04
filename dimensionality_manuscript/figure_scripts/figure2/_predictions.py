"""Model predictions on a session, and the raster transforms that display them.

Scoring a model means training it, which costs seconds to minutes per model per session, so
every panel that wants a full-frame prediction grid goes through :func:`get_model_predictions`
rather than recomputing. The quality features used to filter target neurons are cheaper but
still rebuild trial-resolved place fields, so they are cached beside the predictions -- both
caches are cleared together by :func:`clear_prediction_cache`.
"""

import numpy as np

from vrAnalysis.helpers import sort_by_preferred_environment
from vrAnalysis.processors import spkmaps as SMPs
from vrAnalysis.processors.support import median_zscore
from vrAnalysis.sessions import B2Session, SpksTypes
from dimilibi import measure_r2, mse
from dimensionality_manuscript.configs.placefield_structure import PlaceFieldStructureConfig
from dimensionality_manuscript.registry import ModelName, PopulationRegistry, get_model

#: Cache for :func:`get_model_predictions`, keyed on everything that changes the result.
_PREDICTION_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

#: Cache for :func:`target_prediction_quality`, keyed on the session and activity variation.
_PREDICTION_QUALITY_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def clear_prediction_cache() -> None:
    """Drop every cached prediction grid and quality feature so the next call recomputes."""
    _PREDICTION_CACHE.clear()
    _PREDICTION_QUALITY_CACHE.clear()


def _model_prediction_grid(
    model,
    session: B2Session,
    spks_type: SpksTypes,
    method: str,
    train_split: str,
    test_split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score one model on a session and return full-frame target and prediction rasters.

    The model's own optimized hyperparameters (from cache) are used. Predictions are
    reconstructed onto the full test-frame grid, with NaN in frames that the model
    filtered out, so columns stay aligned across models.

    Returns
    -------
    target : np.ndarray, shape (n_target, n_frames)
    prediction : np.ndarray, shape (n_target, n_frames)
    """
    hyperparameters = model.get_best_hyperparameters(
        session,
        spks_type=spks_type,
        method=method,
    )[0]
    report = model.process(
        session,
        spks_type,
        train_split=train_split,
        test_split=test_split,
        hyperparameters=hyperparameters,
    )
    target = model.get_session_data(session, spks_type, test_split)[1].numpy()
    n_target, n_frames = target.shape

    prediction = np.full((n_target, n_frames), np.nan)
    if report.extras.get("predictions_were_filtered", False):
        idx_valid = report.extras["idx_valid_predictions"]
        prediction[:, idx_valid] = report.predicted_data
    else:
        prediction[:] = report.predicted_data
    return target, prediction


def get_model_predictions(
    model_name: ModelName,
    session: B2Session,
    registry: PopulationRegistry,
    spks_type: SpksTypes,
    activity_parameters_name: str = "default",
    method: str = "preferred",
    train_split: str = "train",
    test_split: str = "test",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Target and prediction rasters for one model on one session, cached in process.

    Wraps :func:`_model_prediction_grid` in a memo keyed on everything that changes the
    result, so a viewer can switch back and forth between models and neurons without
    retraining. The cache is not invalidated by edits to the model code -- call
    :func:`clear_prediction_cache` after changing anything upstream of the prediction.

    Parameters
    ----------
    model_name : ModelName
    session : B2Session
    registry : PopulationRegistry
    spks_type : SpksTypes
    activity_parameters_name : str
        Activity scaling registry name passed to ``get_model``.
    method : str
        Hyperparameter optimization method used to look up the best hyperparameters.
    train_split, test_split : str

    Returns
    -------
    target : np.ndarray, shape (n_target, n_frames)
    prediction : np.ndarray, shape (n_target, n_frames)
        Shared read-only arrays -- NaN on frames the model filtered out.
    """
    key = (session.session_name, spks_type, model_name, activity_parameters_name, method, train_split, test_split)
    if key not in _PREDICTION_CACHE:
        model = get_model(model_name, registry, activity_parameters=activity_parameters_name)
        _PREDICTION_CACHE[key] = _model_prediction_grid(model, session, spks_type, method, train_split, test_split)
    return _PREDICTION_CACHE[key]


def target_prediction_quality(
    session: B2Session,
    registry: PopulationRegistry,
    spks_type: SpksTypes,
    activity_parameters_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean spatial reliability and fraction-active for the model's target neurons."""
    key = (session.session_name, spks_type, activity_parameters_name)
    if key not in _PREDICTION_QUALITY_CACHE:
        config = PlaceFieldStructureConfig(
            spks_type=spks_type,
            activity_parameters_name=activity_parameters_name,
        )
        quality = config.process(session, registry)
        population = registry.get_population(session, spks_type)[0]
        target_idx = population.cell_split_indices[1]
        with np.errstate(invalid="ignore"):
            reliability = np.nanmean(np.asarray(quality["reliability"])[:, target_idx], axis=0)
            fraction_active = np.nanmean(np.asarray(quality["fraction_active"])[:, target_idx], axis=0)
        _PREDICTION_QUALITY_CACHE[key] = reliability, fraction_active
    return _PREDICTION_QUALITY_CACHE[key]


def target_environment_sort(
    session: B2Session,
    registry: PopulationRegistry,
    spks_type: SpksTypes,
) -> np.ndarray:
    """
    Sort target neurons by preferred environment then place-field position (figure1 style).

    The target neurons are the second cell split of the session's population. Their
    absolute ROI indices are mapped onto the session-filtered ROI ordering that
    ``get_env_maps`` uses, then sorted with ``sort_by_preferred_environment``.

    Returns
    -------
    idx_sort : np.ndarray
        Permutation of the target rows.
    """
    population = registry.get_population(session, spks_type)[0]
    target_roi_abs = population.idx_neurons[population.cell_split_indices[1]]

    session_rois = np.where(session.idx_rois)[0]
    abs_to_row = np.full(session.idx_rois.shape[0], -1)
    abs_to_row[session_rois] = np.arange(len(session_rois))
    env_rows = abs_to_row[target_roi_abs]
    if np.any(env_rows < 0):
        raise ValueError("Target neuron missing from the session's ROI filter — cannot map to env maps.")

    smp = SMPs.SpkmapProcessor(session, params=SMPs.SpkmapParams())
    return sort_by_preferred_environment(smp, idx_rois=env_rows)


def _median_zscore_rows(x: np.ndarray, median_subtract: bool) -> np.ndarray:
    """
    Median z-score each neuron (row) across frames using ``median_zscore``.

    ``median_zscore`` expects (frames, neurons) and is not NaN-aware, so stats are
    computed on frames with no NaN across neurons; NaN frames are preserved.
    """
    idx_valid = ~np.any(np.isnan(x), axis=0)
    out = np.full_like(x, np.nan)
    zscored = median_zscore(np.ascontiguousarray(x[:, idx_valid].T), median_subtract=median_subtract).T
    out[:, idx_valid] = zscored
    return out


def transform_raster_rows(x: np.ndarray, zscore: bool, subtract_median: bool) -> np.ndarray:
    """Apply the selected per-ROI display transform while preserving NaN frames."""
    if zscore:
        return _median_zscore_rows(x, median_subtract=subtract_median)
    if not subtract_median:
        return x

    idx_valid = ~np.any(np.isnan(x), axis=0)
    out = np.full_like(x, np.nan)
    valid = x[:, idx_valid]
    out[:, idx_valid] = valid - np.median(valid, axis=1, keepdims=True)
    return out


SAMPLE_FIT_METRICS = ["r2", "mse", "rms"]
SAMPLE_FIT_LABELS = {"r2": r"$R^2$", "mse": "MSE", "rms": "RMS Error"}


def per_sample_fit(prediction: np.ndarray, target: np.ndarray, metric: str) -> np.ndarray:
    """Compute a per-frame fit metric (over ROIs). NaN frames pass through as NaN."""
    if metric == "r2":
        return np.asarray(measure_r2(prediction, target, reduce="none", dim=0))
    if metric == "mse":
        return np.asarray(mse(prediction, target, reduce="none", dim=0))
    if metric == "rms":
        return np.sqrt(np.asarray(mse(prediction, target, reduce="none", dim=0)))
    raise ValueError(f"sample_fit_metric must be one of {SAMPLE_FIT_METRICS}, got {metric!r}")
