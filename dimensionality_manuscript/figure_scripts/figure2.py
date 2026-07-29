from dataclasses import dataclass, replace

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from syd import Viewer
from rastermap import Rastermap

from vrAnalysis.helpers import sort_by_preferred_environment
from vrAnalysis.helpers.plotting import beeswarm, errorPlot, format_spines
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors import spkmaps as SMPs
from vrAnalysis.processors.support import median_zscore
from dimilibi import measure_r2, mse
from dimensionality_manuscript.registry import (
    ACTIVITY_PARAMETERS_NAMES,
    ModelName,
    PopulationRegistry,
    get_model,
    short_model_name,
)
from dimensionality_manuscript.configs.regression import KEY_FIGURE_MODELS
from dimensionality_manuscript.configs.placefield_structure import PlaceFieldStructureConfig
from dimensionality_manuscript.configs.rrr_to_external_latents import (
    DEFAULT_MODEL_EXT_INT_PAIR,
    VALID_MODEL_EXT_INT_PAIRS,
    VALID_RRR_VARIANCE,
    VALID_SPKS_TYPES,
    pair_has_omission_response,
)
from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, apply_legend, update_legend_widgets
from dimensionality_manuscript.pipeline import ResultsAggregator

SORT_METHODS = ["environment", "rastermap", "activity"]

plt.rcParams["font.size"] = 18

_RASTER_HEIGHT = 6.0
_ERROR_TRACE_HEIGHT = 3.0


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


#: In-process store for :func:`get_model_predictions`. Scoring a model means training it, which
#: costs seconds to minutes per model per session, so every figure function that wants a
#: full-frame prediction grid goes through the cache rather than recomputing.
_PREDICTION_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


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
    retraining. The cache is not invalidated by edits to the model code — call
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
        Shared read-only arrays — NaN on frames the model filtered out.
    """
    key = (session.session_name, spks_type, model_name, activity_parameters_name, method, train_split, test_split)
    if key not in _PREDICTION_CACHE:
        model = get_model(model_name, registry, activity_parameters=activity_parameters_name)
        _PREDICTION_CACHE[key] = _model_prediction_grid(model, session, spks_type, method, train_split, test_split)
    return _PREDICTION_CACHE[key]


def clear_prediction_cache() -> None:
    """Drop every cached prediction grid so the next call retrains."""
    _PREDICTION_CACHE.clear()
    _PREDICTION_QUALITY_CACHE.clear()


#: Quality features are considerably cheaper than fitting a regression model, but still require
#: rebuilding trial-resolved place fields.  Keep them beside the prediction cache so switching
#: between single-ROI and quality-filtered views remains instantaneous after the first visit.
_PREDICTION_QUALITY_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _target_prediction_quality(
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


def _target_environment_sort(
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


def _transform_raster_rows(x: np.ndarray, zscore: bool, subtract_median: bool) -> np.ndarray:
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
_SAMPLE_FIT_LABELS = {"r2": r"$R^2$", "mse": "MSE", "rms": "RMS Error"}


def _per_sample_fit(prediction: np.ndarray, target: np.ndarray, metric: str) -> np.ndarray:
    """Compute a per-frame fit metric (over ROIs). NaN frames pass through as NaN."""
    if metric == "r2":
        return np.asarray(measure_r2(prediction, target, reduce="none", dim=0))
    if metric == "mse":
        return np.asarray(mse(prediction, target, reduce="none", dim=0))
    if metric == "rms":
        return np.sqrt(np.asarray(mse(prediction, target, reduce="none", dim=0)))
    raise ValueError(f"sample_fit_metric must be one of {SAMPLE_FIT_METRICS}, got {metric!r}")


class ModelRasterFocus(Viewer):
    """Interactive stacked raster of regression-model predictions for one session.

    One shared target-activity raster (optional) sits atop one prediction raster per
    model. An optional second column shows residuals (target - prediction) with a bwr
    colormap, and an optional single panel under all rasters superimposes the per-frame
    fit of every model as colored lines with a legend.
    """

    def __init__(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        model_names: list[ModelName],
        target: np.ndarray,
        predictions: dict[str, np.ndarray],
        idx_sort_environment: np.ndarray,
        figsize: tuple[float, float],
        xslice_start: int,
        xslice_stop: int,
        fontsize: float = 7,
    ):
        self.session = session
        self.spks_type = spks_type
        self.model_names = list(model_names)
        self.target = target
        self.predictions = predictions
        self.figsize = figsize
        self.fontsize = fontsize

        # Frames are the x-axis unit of every raster, so the scale bar's length in frames
        # is its requested duration divided by the imaging frame period.
        mpci_times = self.session.loadone("mpci.times")
        self.frame_duration = float(np.median(np.diff(mpci_times)))

        # Row orderings shared across every panel. Environment sort comes precomputed;
        # activity and rastermap sorts are stable (computed over all frames) and cached.
        self._sort_cache: dict[str, np.ndarray] = {
            "environment": idx_sort_environment,
            "activity": np.argsort(-np.nansum(target, axis=1)),
        }

        num_frames = target.shape[1]
        self.add_boolean("include_data_raster", value=True)
        self.add_boolean("include_error_column", value=True)
        self.add_boolean("include_sample_fit_curve", value=True)
        self.add_boolean("zscore", value=False)
        self.add_boolean("subtract_median", value=False)
        self.add_selection("sort_method", value="environment", options=SORT_METHODS)
        self.add_selection("sample_fit_metric", value="r2", options=SAMPLE_FIT_METRICS)
        self.add_float("vmax", value=1.0, min=0.01, max=20.0)
        self.add_float("scale_bar_seconds", value=10.0, min=0.0, max=120.0)
        self.add_integer("xslice_start", value=xslice_start, min=0, max=num_frames - 1)
        self.add_integer("xslice_stop", value=xslice_stop, min=1, max=num_frames)

    def _get_sort(self, method: str) -> np.ndarray:
        """Return (and cache) the target-row ordering for a sort method."""
        if method not in self._sort_cache:
            if method == "rastermap":
                self._sort_cache[method] = Rastermap().fit(self.target).isort
            else:
                raise ValueError(f"Unknown sort_method {method!r}. Options: {SORT_METHODS}")
        return self._sort_cache[method]

    def plot(self, state):
        include_data = state["include_data_raster"]
        include_error = state["include_error_column"]
        include_curve = state["include_sample_fit_curve"]
        metric = state["sample_fit_metric"]
        vmax = state["vmax"]
        scale_bar_seconds = state["scale_bar_seconds"]
        zscore = state["zscore"]
        subtract_median = state["subtract_median"]
        xslice = slice(state["xslice_start"], state["xslice_stop"])
        idx_sort = self._get_sort(state["sort_method"])

        # Display transforms are computed per neuron over all frames, so their statistics
        # remain stable while the interactive x slice changes.
        disp_cache: dict[str, np.ndarray] = {}

        def disp(name: str) -> np.ndarray:
            if name not in disp_cache:
                raw = self.target if name == "__target__" else self.predictions[name]
                disp_cache[name] = _transform_raster_rows(raw, zscore=zscore, subtract_median=subtract_median)
            return disp_cache[name]

        # Build the row layout: an optional shared data raster, one prediction raster per
        # model, then an optional single error-trace panel spanning below all rasters.
        rows: list[tuple[str, str | None]] = []
        if include_data:
            rows.append(("data", None))
        for model_name in self.model_names:
            rows.append(("raster", model_name))
        if include_curve:
            rows.append(("error_trace", None))

        height_ratios = [_ERROR_TRACE_HEIGHT if kind == "error_trace" else _RASTER_HEIGHT for kind, _ in rows]
        ncols = 2 if include_error else 1

        plt.close("all")
        fig = plt.figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(len(rows), ncols, height_ratios=height_ratios)

        raster_kwargs = dict(aspect="auto", cmap="gray_r", vmin=0, vmax=vmax)
        error_kwargs = dict(aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)

        cmap_models = plt.get_cmap("tab10")
        model_colors = {name: cmap_models(i % 10) for i, name in enumerate(self.model_names)}

        last_raster_row = max(i for i, (kind, _) in enumerate(rows) if kind in ("data", "raster"))
        xvals = np.arange(xslice.start, xslice.stop)

        # Scale bar length as a fraction of the displayed window (rasters share this x-axis).
        num_shown_frames = xslice.stop - xslice.start
        scale_bar_frac = (scale_bar_seconds / self.frame_duration) / num_shown_frames

        _inset_built = False
        for irow, (kind, model_name) in enumerate(rows):
            if kind == "error_trace":
                ax = fig.add_subplot(gs[irow, :])
                fit_min, fit_max = np.inf, -np.inf
                for name in self.model_names:
                    fit = _per_sample_fit(self.predictions[name][:, xslice], self.target[:, xslice], metric)
                    ax.plot(xvals, fit, color=model_colors[name], linewidth=1, label=short_model_name(name))
                    finite = fit[np.isfinite(fit)]
                    if finite.size:
                        fit_min = min(fit_min, float(np.min(finite)))
                        fit_max = max(fit_max, float(np.max(finite)))
                ax.set_xlim(xslice.start, xslice.stop)
                ybounds = [fit_min, fit_max] if np.isfinite(fit_min) else [0, 1]
                format_spines(
                    ax,
                    x_pos=-0.02,
                    y_pos=-0.02,
                    xbounds=[xslice.start, xslice.stop],
                    ybounds=ybounds,
                    xticks=[xslice.start, xslice.stop],
                    yticks=[round(b, 2) for b in ybounds],
                    tick_length=4,
                    spines_visible=["left", "bottom"],
                    tick_fontsize=self.fontsize,
                )
                ax.set_ylabel(_SAMPLE_FIT_LABELS[metric], fontsize=self.fontsize)
                ax.legend(loc="upper right", fontsize=self.fontsize, ncol=max(1, len(self.model_names) // 2), frameon=False)
                continue

            if kind == "data":
                data = disp("__target__")[:, xslice][idx_sort]
                title = "Data"
            else:
                data = disp(model_name)[:, xslice][idx_sort]
                title = short_model_name(model_name)

            ax = fig.add_subplot(gs[irow, 0])
            ax.imshow(data, **raster_kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("ROIs", fontsize=self.fontsize)
            ax.text(1.0, 1.0, title, transform=ax.transAxes, ha="right", va="top", color="black", fontsize=self.fontsize)
            for spine in ["top", "right", "bottom", "left"]:
                ax.spines[spine].set_visible(False)
            scale_bar_y = 0.1
            scale_text_y_offset = 0.025
            if irow == last_raster_row and scale_bar_seconds > 0:
                ax.plot(
                    [1.0 - scale_bar_frac, 1.0],
                    [scale_bar_y, scale_bar_y],
                    transform=ax.transAxes,
                    color="black",
                    linewidth=1.5,
                    solid_capstyle="butt",
                    clip_on=False,
                )
                ax.text(
                    1.0 - scale_bar_frac / 2,
                    scale_bar_y + scale_text_y_offset,
                    f"{scale_bar_seconds:g}s",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=self.fontsize,
                    clip_on=False,
                )

            if not _inset_built:
                gray_values = np.linspace(0, vmax, 255) / vmax if vmax > 0 else np.zeros(255)
                cmap = plt.get_cmap("gray_r")
                colors = cmap(gray_values)
                cmap_data = colors[np.newaxis, :, :]
                inset = ax.inset_axes([0.2, 0.05, 0.6, 0.1])
                inset.imshow(cmap_data, aspect="auto")
                inset.set_xticks([])
                inset.set_yticks([])
                inset.text(-0.03, 0.5, "0", transform=inset.transAxes, ha="right", va="center", clip_on=False, fontsize=self.fontsize)
                inset.text(1.03, 0.5, f"{vmax}", transform=inset.transAxes, ha="left", va="center", clip_on=False, fontsize=self.fontsize)
                inset.text(0.5, 1.25, "zscore activity", transform=inset.transAxes, ha="center", va="bottom", clip_on=False, fontsize=self.fontsize)
                _inset_built = True

            if include_error and kind == "raster":
                residual = (disp("__target__")[:, xslice] - disp(model_name)[:, xslice])[idx_sort]
                ax_err = fig.add_subplot(gs[irow, 1])
                ax_err.imshow(residual, **error_kwargs)
                ax_err.set_xticks([])
                ax_err.set_yticks([])
                ax_err.text(1.0, 1.0, "Residual", transform=ax_err.transAxes, ha="right", va="top", color="black", fontsize=self.fontsize)
                for spine in ["top", "right", "bottom", "left"]:
                    ax_err.spines[spine].set_visible(False)

        return fig


def stacked_model_rasters(
    session: B2Session,
    spks_type: SpksTypes,
    model_names: list[ModelName],
    activity_parameters_name: str = "default",
    method: str = "preferred",
    registry: PopulationRegistry | None = None,
    train_split: str = "train",
    test_split: str = "test",
    include_data_raster: bool = True,
    include_error_column: bool = True,
    include_sample_fit_curve: bool = True,
    sample_fit_metric: str = "r2",
    sort_method: str = "environment",
    zscore: bool = False,
    subtract_median: bool = False,
    xslice: slice | None = None,
    vmax: float = 6,
    scale_bar_seconds: float = 10.0,
    figsize: tuple[float, float] | None = None,
    fontsize: float = 7,
    return_syd_viewer: bool = False,
):
    """
    Stack prediction rasters for several regression models scored on one session.

    Each model is scored with its own optimized hyperparameters (from cache). The
    target activity is shared across models, so a single data raster and a single row
    ordering apply to every panel.

    Parameters
    ----------
    session : B2Session
    spks_type : SpksTypes
    model_names : list[ModelName]
        Regression models to score and stack (from the registry).
    activity_parameters_name : str
        Activity scaling registry name passed to ``get_model``.
    method : str
        Hyperparameter optimization method used to look up the best hyperparameters.
    registry : PopulationRegistry or None
        Population registry. A default one is created when None.
    train_split, test_split : str
        Splits used to train and evaluate each model.
    include_data_raster : bool
        Draw the shared target-activity raster at the top.
    include_error_column : bool
        Draw a second column of residual (target - prediction) rasters.
    include_sample_fit_curve : bool
        Draw a single panel below all rasters superimposing every model's per-frame fit.
    sample_fit_metric : {"r2", "mse", "rms"}
        Metric for the per-frame fit trace.
    sort_method : {"environment", "rastermap", "activity"}
        Row ordering shared across every panel. ``environment`` sorts by preferred
        environment then place-field position (figure1 style); ``rastermap`` sorts by a
        Rastermap embedding of the target activity; ``activity`` sorts by total activity.
    zscore : bool
        Divide each neuron by its standard deviation over all frames before display.
    subtract_median : bool
        Subtract each neuron's all-frame median before display. When combined with
        ``zscore=True``, display ``(activity - median) / std``.
    xslice : slice | None
    vmax : float
    scale_bar_seconds : float
        Duration of the time scale bar drawn above the top raster (0 hides it).
    figsize : tuple[float, float] or None
        Sized from the layout when None.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    """
    if sample_fit_metric not in SAMPLE_FIT_METRICS:
        raise ValueError(f"sample_fit_metric must be one of {SAMPLE_FIT_METRICS}, got {sample_fit_metric!r}")
    if sort_method not in SORT_METHODS:
        raise ValueError(f"sort_method must be one of {SORT_METHODS}, got {sort_method!r}")

    registry = registry or PopulationRegistry()

    target = None
    predictions: dict[str, np.ndarray] = {}
    for model_name in model_names:
        model_target, prediction = get_model_predictions(
            model_name,
            session,
            registry,
            spks_type,
            activity_parameters_name=activity_parameters_name,
            method=method,
            train_split=train_split,
            test_split=test_split,
        )
        if target is None:
            target = model_target
        predictions[model_name] = prediction

    idx_sort_environment = _target_environment_sort(session, registry, spks_type)

    num_frames = target.shape[1]
    if xslice is None:
        xstart = 0
        xstop = num_frames
    else:
        xstart = xslice.start if xslice.start is not None else 0
        xstop = xslice.stop if xslice.stop is not None else num_frames

    if figsize is None:
        width = 16 if include_error_column else 10
        height = 2.2 * len(model_names) + (1.5 if include_sample_fit_curve else 0)
        if include_data_raster:
            height += 2.2
        figsize = (width, max(height, 3.0))

    viewer = ModelRasterFocus(session, spks_type, model_names, target, predictions, idx_sort_environment, figsize, xstart, xstop, fontsize)
    viewer.update_boolean("include_data_raster", value=include_data_raster)
    viewer.update_boolean("include_error_column", value=include_error_column)
    viewer.update_boolean("include_sample_fit_curve", value=include_sample_fit_curve)
    viewer.update_selection("sort_method", value=sort_method)
    viewer.update_boolean("zscore", value=zscore)
    viewer.update_boolean("subtract_median", value=subtract_median)
    viewer.update_selection("sample_fit_metric", value=sample_fit_metric)
    viewer.update_float("vmax", value=vmax)
    viewer.update_float("scale_bar_seconds", value=scale_bar_seconds)
    viewer.update_integer("xslice_start", value=xstart)
    viewer.update_integer("xslice_stop", value=xstop)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ======================================================================================
# Model-zoo schematic
# ======================================================================================


@dataclass
class ModelZooSchematicConfig:
    """Fully tunable layout/style for :func:`model_zoo_schematic`.

    Coordinates are in abstract "data units"; the axis uses an equal aspect ratio, so a
    unit is the same horizontally and vertically. Every box shares ``box_width`` (except the
    gain box, which uses ``gain_width``); the source/target rows share ``box_height`` while
    the latent row is scaled by ``latent_height_scale``.

    Corner radii behave like CSS ``border-radius`` (in data units) and are clamped so they
    never exceed half of a box's smaller side.
    """

    # --- Shared box geometry -----------------------------------------------------------
    box_width: float = 2.0  # shared width of every standard box
    box_height: float = 1.0  # shared height of the source and target rows
    latent_height_scale: float = 1.65  # latent (middle) row height as a multiple of box_height
    gain_width: float = 1.2  # the single box with a unique width

    # --- Spacing (data units) ----------------------------------------------------------
    col_gap: float = 0.25  # horizontal gap between columns
    row_gap: float = 0.5  # vertical gap between rows (leaves room for arrows)
    panel_gap: float = 0.3  # gap between the three containers (defaults to col_gap)
    container_pad: float = 0.3  # padding between boxes and their container edge
    label_pad: float = 0.75  # extra room on the labelled side for the container title (sets container height)
    label_gap: float = 0.45  # gap between the boxes and the container label (the pad beneath the label)
    gain_separator_gap: float = 0.2  # extra horizontal space left of the gain model for its dotted divider
    neural_label_offset: float = 0.18  # horizontal nudge of each neural label off the arrow, as a fraction of box_width

    # --- Rounded corners (data units, like CSS border-radius) --------------------------
    box_corner_radius: float = 0.18
    container_corner_radius: float = 0.4

    # --- Container fill colors ---------------------------------------------------------
    external_container_color: str = "#d9d9d9"
    internal_container_color: str = "#f4b0ab"
    neural_container_color: str = "#b8bbe8"

    # --- Box fill colors ---------------------------------------------------------------
    black_box_color: str = "#000000"
    red_box_color: str = "#c00000"
    blue_box_color: str = "#0000cd"
    box_text_color: str = "#ffffff"
    container_label_color: str = "#000000"

    # --- Arrow colors (one per panel) --------------------------------------------------
    external_arrow_color: str = "#000000"
    internal_arrow_color: str = "#c00000"
    neural_arrow_color: str = "#0000cd"
    gain_second_arrow_color: str = "#000000"  # the "other" model in the two-model gain column

    # --- Line / arrow styling ----------------------------------------------------------
    arrow_linewidth: float = 2.2
    arrow_mutation_scale: float = 10.0  # arrowhead size
    junction_dot_size: float = 4.0  # multiplicative-gain junction marker
    junction_dot_offset: float = 0.15  # rightward shift of the red junction dot off the arrow, as a fraction of box_width
    gain_two_model_offset: float = 0.1  # half-separation of the paired gain arrows, as a fraction of box_width
    gain_route_hoffset: float = 0.10  # horizontal separation of the black/red gain routes, as a fraction of box_width
    gain_route_voffset: float = 0.10  # vertical separation of the black/red gain routes, as a fraction of box_height
    gain_separator_linewidth: float = 1.6  # dotted divider left of the gain model

    # --- Fonts -------------------------------------------------------------------------
    box_fontsize: float = 13.0
    container_label_fontsize: float = 15.0
    arrow_label_fontsize: float = 13.0

    # --- Figure ------------------------------------------------------------------------
    figsize: tuple[float, float] = (18.0, 5.0)
    background_color: str = "#ffffff"


# Text as it appears on the source slide. The four model boxes are shared verbatim by the
# external and internal panels; explicit line breaks reproduce the slide's wrapping.
_ZOO_MODEL_LABELS = ["placefield", "high-d\nposition", "high-d pos\n+speed", "high-d pos\n+speed\n+reward"]

# ModelZooSchematicConfig fields exposed as live Syd controls.
_ZOO_TUNABLES = [
    "box_width",
    "box_height",
    "latent_height_scale",
    "gain_width",
    "col_gap",
    "row_gap",
    "panel_gap",
    "container_pad",
    "label_pad",
    "label_gap",
    "gain_separator_gap",
    "neural_label_offset",
    "box_corner_radius",
    "container_corner_radius",
    "arrow_linewidth",
    "arrow_mutation_scale",
    "junction_dot_size",
    "gain_two_model_offset",
    "junction_dot_offset",
    "gain_route_hoffset",
    "gain_route_voffset",
    "box_fontsize",
    "container_label_fontsize",
]

# Slider ranges for every tunable ModelZooSchematicConfig field, shared by the schematic viewers.
_ZOO_TUNABLE_LIMITS = {
    "box_width": (0.5, 4.0),
    "box_height": (0.5, 3.0),
    "latent_height_scale": (1.0, 2.5),
    "gain_width": (0.3, 3.0),
    "col_gap": (0.0, 2.0),
    "row_gap": (0.1, 3.0),
    "panel_gap": (0.0, 4.0),
    "container_pad": (0.0, 2.0),
    "label_pad": (0.0, 2.5),
    "label_gap": (0.0, 2.0),
    "gain_separator_gap": (0.0, 2.0),
    "neural_label_offset": (0.0, 0.5),
    "box_corner_radius": (0.0, 0.5),
    "container_corner_radius": (0.0, 1.0),
    "arrow_linewidth": (0.5, 6.0),
    "arrow_mutation_scale": (5.0, 40.0),
    "junction_dot_size": (2.0, 24.0),
    "gain_two_model_offset": (0.0, 0.4),
    "junction_dot_offset": (0.0, 0.5),
    "gain_route_hoffset": (0.0, 0.3),
    "gain_route_voffset": (0.0, 0.5),
    "box_fontsize": (6.0, 28.0),
    "container_label_fontsize": (6.0, 30.0),
}


def _zoo_box(ax, cx, cy, w, h, text, facecolor, cfg: ModelZooSchematicConfig):
    """Draw a rounded box of total size ``w`` x ``h`` centered at ``(cx, cy)`` with centered text."""
    r = min(cfg.box_corner_radius, 0.5 * min(w, h) - 1e-6)
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2 + r, cy - h / 2 + r),
            w - 2 * r,
            h - 2 * r,
            boxstyle=f"round,pad={r},rounding_size={r}",
            mutation_scale=1.0,
            facecolor=facecolor,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", color=cfg.box_text_color, fontsize=cfg.box_fontsize, zorder=3)


def _zoo_container(ax, x0, y0, x1, y1, color, cfg: ModelZooSchematicConfig):
    """Draw a rounded container spanning ``[x0, x1] x [y0, y1]`` behind the boxes."""
    r = min(cfg.container_corner_radius, 0.5 * min(x1 - x0, y1 - y0) - 1e-6)
    ax.add_patch(
        FancyBboxPatch(
            (x0 + r, y0 + r),
            (x1 - x0) - 2 * r,
            (y1 - y0) - 2 * r,
            boxstyle=f"round,pad={r},rounding_size={r}",
            mutation_scale=1.0,
            facecolor=color,
            edgecolor="none",
            zorder=0,
        )
    )


def _zoo_arrow(ax, x0, y0, x1, y1, color, cfg: ModelZooSchematicConfig):
    """Draw a straight arrow from ``(x0, y0)`` to ``(x1, y1)``."""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=cfg.arrow_mutation_scale,
            color=color,
            lw=cfg.arrow_linewidth,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def _zoo_line(ax, xs, ys, color, cfg: ModelZooSchematicConfig):
    """Draw a (possibly elbowed) plain connector through the given points."""
    ax.plot(xs, ys, color=color, lw=cfg.arrow_linewidth, solid_capstyle="round", solid_joinstyle="round", zorder=1)


class ModelZooSchematic(Viewer):
    """Interactive rounded-box schematic of the external / internal / neural model zoo.

    The three panels reproduce the source slide: an external panel (model -> target), an
    internal panel adding a source row and a multiplicatively-wired gain box, and a neural
    reduced-rank-regression panel. All geometry, colors, corner radii, and fonts come from
    a :class:`ModelZooSchematicConfig`; the numeric fields in ``_ZOO_TUNABLES`` are exposed
    as live sliders.
    """

    def __init__(self, config: ModelZooSchematicConfig):
        self.cfg = config
        for name in _ZOO_TUNABLES:
            lo, hi = _ZOO_TUNABLE_LIMITS[name]
            self.add_float(name, value=float(getattr(config, name)), min=lo, max=hi)

    def plot(self, state):
        cfg = replace(self.cfg, **{name: state[name] for name in _ZOO_TUNABLES})
        bw = cfg.box_width
        h = cfg.box_height  # source / target rows
        hm = cfg.box_height * cfg.latent_height_scale  # taller latent (middle) row
        cg, rg, pad = cfg.col_gap, cfg.row_gap, cfg.container_pad

        # Row centers (target row bottom sits at y = 0). The latent row is taller, so
        # neighbours are spaced by the appropriate half-heights. External uses only the
        # latent + target rows; internal/neural add a source row on top, so all panels align.
        y_target = h / 2
        y_model = y_target + h / 2 + rg + hm / 2
        y_source = y_model + hm / 2 + rg + h / 2

        # Row edges reused throughout.
        src_top, src_bot = y_source + h / 2, y_source - h / 2
        mid_top, mid_bot = y_model + hm / 2, y_model - hm / 2
        tgt_top, tgt_bot = y_target + h / 2, y_target - h / 2

        fig, ax = plt.subplots(figsize=cfg.figsize, layout="constrained")
        fig.patch.set_facecolor(cfg.background_color)
        ax.set_aspect("equal")
        ax.axis("off")

        black, red, blue = cfg.black_box_color, cfg.red_box_color, cfg.blue_box_color

        # ---- External panel (4 columns, latent -> target) ----------------------------
        ext_left = 0.0
        ext_cx = [ext_left + bw / 2 + i * (bw + cg) for i in range(4)]
        ext_right = ext_cx[-1] + bw / 2
        for cx, label in zip(ext_cx, _ZOO_MODEL_LABELS):
            _zoo_box(ax, cx, y_model, bw, hm, label, black, cfg)
            _zoo_box(ax, cx, y_target, bw, h, "target", black, cfg)
            _zoo_arrow(ax, cx, mid_bot, cx, tgt_top, cfg.external_arrow_color, cfg)
        ext_outer = (ext_left - pad, tgt_bot - pad, ext_right + pad, mid_top + pad + cfg.label_pad)
        _zoo_container(ax, *ext_outer, cfg.external_container_color, cfg)
        ax.text(
            (ext_outer[0] + ext_outer[2]) / 2,
            mid_top + cfg.label_gap,
            "external models",
            ha="center",
            va="center",
            color=cfg.container_label_color,
            fontsize=cfg.container_label_fontsize,
            zorder=3,
        )

        # ---- Internal panel (5 columns, source -> latent -> target, + gain) ----------
        int_content_left = ext_outer[2] + cfg.panel_gap + pad
        # Columns 0-3 are evenly spaced; an extra gap before column 4 sets off the gain model.
        int_cx = []
        x = int_content_left + bw / 2
        for i in range(5):
            int_cx.append(x)
            x += bw + cg + (cfg.gain_separator_gap if i == 3 else 0.0)
        gain_cx = int_cx[-1] + bw / 2 + cg + cfg.gain_width / 2
        int_right = gain_cx + cfg.gain_width / 2
        ai = cfg.internal_arrow_color
        cx5 = int_cx[-1]
        off = cfg.gain_two_model_offset * bw
        for i, cx in enumerate(int_cx):
            is_last = i == 4
            model_color = black if is_last else red  # last column is the neural/gain model (black boxes)
            model_label = "placefield" if is_last else _ZOO_MODEL_LABELS[i]
            _zoo_box(ax, cx, y_source, bw, h, "source", red, cfg)  # source is always red
            _zoo_box(ax, cx, y_model, bw, hm, model_label, model_color, cfg)
            _zoo_box(ax, cx, y_target, bw, h, "target", model_color, cfg)
            # In the gain column the source -> latent arrow shares the vertical slice of the
            # (offset) red latent -> target arrow below it.
            src_x = cx + off if is_last else cx
            _zoo_arrow(ax, src_x, src_bot, src_x, mid_top, ai, cfg)
            if not is_last:
                _zoo_arrow(ax, cx, mid_bot, cx, tgt_top, ai, cfg)

        # Dotted divider setting the gain model off as its own sub-group (spans source..target).
        x_sep = (int_cx[3] + bw / 2 + cx5 - bw / 2) / 2
        ax.plot([x_sep, x_sep], [tgt_bot, src_top], linestyle=":", color=black, lw=cfg.gain_separator_linewidth, zorder=1)

        # Gain box wiring. Two paired latent -> target arrows carry two models: a plain
        # placefield model (black, left) and a gain-modulated one (red, right). The gain is
        # fed from the source and modulates BOTH: each colored route runs source -> gain ->
        # its own arrow. Horizontal/vertical route separation are independent knobs.
        black_arrow = cfg.gain_second_arrow_color
        hgr = cfg.gain_route_hoffset * bw
        vgr = cfg.gain_route_voffset * h
        _zoo_box(ax, gain_cx, y_model, cfg.gain_width, hm, "gain", red, cfg)  # gain is always red
        _zoo_arrow(ax, cx5 - off, mid_bot, cx5 - off, tgt_top, black_arrow, cfg)
        _zoo_arrow(ax, cx5 + off, mid_bot, cx5 + off, tgt_top, ai, cfg)

        # Source -> gain, one route per model (red high/right, black low/left).
        _zoo_line(ax, [cx5 + bw / 2, gain_cx + hgr], [y_source + vgr, y_source + vgr], ai, cfg)
        _zoo_arrow(ax, gain_cx + hgr, y_source + vgr, gain_cx + hgr, mid_top, ai, cfg)
        _zoo_line(ax, [cx5 + bw / 2, gain_cx - hgr], [y_source - vgr, y_source - vgr], black_arrow, cfg)
        _zoo_arrow(ax, gain_cx - hgr, y_source - vgr, gain_cx - hgr, mid_top, black_arrow, cfg)

        # Gain -> junction dots. Both dots share the same leftmost x; black sits upper, red
        # lower. To nest without crossing, the upper (black) route is the inner one (exits the
        # gain's left, turns higher) and the lower (red) route is the outer one (exits the
        # right, turns lower) -- also keeping each color on one side of the gain throughout.
        y_junction = (mid_bot + tgt_top) / 2
        dot_x = cx5 + off + cfg.junction_dot_offset * bw
        y_black, y_red = y_junction + vgr, y_junction - vgr
        _zoo_line(ax, [gain_cx - hgr, gain_cx - hgr, dot_x], [mid_bot, y_black, y_black], black_arrow, cfg)
        ax.plot(dot_x, y_black, marker="o", markersize=cfg.junction_dot_size, color=black_arrow, zorder=4)
        _zoo_line(ax, [gain_cx + hgr, gain_cx + hgr, dot_x], [mid_bot, y_red, y_red], ai, cfg)
        ax.plot(dot_x, y_red, marker="o", markersize=cfg.junction_dot_size, color=ai, zorder=4)

        int_outer = (int_content_left - pad, tgt_bot - pad, int_right + pad, src_top + pad + cfg.label_pad)
        _zoo_container(ax, *int_outer, cfg.internal_container_color, cfg)
        ax.text(
            (int_outer[0] + int_outer[2]) / 2,
            src_top + cfg.label_gap,
            "internal models",
            ha="center",
            va="center",
            color=cfg.container_label_color,
            fontsize=cfg.container_label_fontsize,
            zorder=3,
        )

        # ---- Neural panel (source -> target; label superimposed inside the column) ----
        # No extra horizontal gutter: the box column uses the same padding as the others,
        # and the rotated label is drawn over the source->target gap, nudged right of the arrow.
        neu_content_left = int_outer[2] + cfg.panel_gap + pad
        neu_cx = neu_content_left + bw / 2
        an = cfg.neural_arrow_color
        _zoo_box(ax, neu_cx, y_source, bw, h, "source", blue, cfg)
        _zoo_box(ax, neu_cx, y_target, bw, h, "target", black, cfg)
        _zoo_arrow(ax, neu_cx, src_bot, neu_cx, tgt_top, an, cfg)
        # Two independent labels flanking the arrow: "reduced rank" left, "regression" right.
        y_mid = (y_source + y_target) / 2
        label_kwargs = dict(
            ha="center",
            va="center",
            rotation=90,
            color=cfg.container_label_color,
            fontsize=cfg.arrow_label_fontsize,
            zorder=3,
        )
        ax.text(neu_cx - cfg.neural_label_offset * bw, y_mid, "reduced rank", **label_kwargs)
        ax.text(neu_cx + cfg.neural_label_offset * bw, y_mid, "regression", **label_kwargs)
        neu_outer = (neu_content_left - pad, tgt_bot - pad, neu_cx + bw / 2 + pad, src_top + pad + cfg.label_pad)
        _zoo_container(ax, *neu_outer, cfg.neural_container_color, cfg)
        ax.text(
            (neu_outer[0] + neu_outer[2]) / 2,
            src_top + cfg.label_gap,
            "neural model",
            ha="center",
            va="center",
            color=cfg.container_label_color,
            fontsize=cfg.container_label_fontsize,
            zorder=3,
        )

        margin = 0.3
        ax.set_xlim(ext_outer[0] - margin, neu_outer[2] + margin)
        ax.set_ylim(
            min(ext_outer[1], int_outer[1], neu_outer[1]) - margin,
            max(int_outer[3], neu_outer[3]) + margin,
        )
        return fig


def model_zoo_schematic(
    config: ModelZooSchematicConfig | None = None,
    return_syd_viewer: bool = False,
):
    """
    Rounded-box schematic of the external / internal / neural model zoo.

    Reproduces the source slide with three rounded containers: an external panel (each
    model box feeds a target), an internal panel that adds a source row and a
    multiplicatively-wired gain box on its last column, and a neural reduced-rank-regression
    panel. Every dimension, color, corner radius, and font is controlled by ``config``; the
    numeric fields are also exposed as live Syd sliders.

    Parameters
    ----------
    config : ModelZooSchematicConfig or None
        Full style/layout config. A default one is created when None.
    return_syd_viewer : bool
        If True, return the Syd viewer instead of a rendered figure.
    """
    viewer = ModelZooSchematic(config or ModelZooSchematicConfig())
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ======================================================================================
# Condensed model-zoo schematic (one column per model type)
# ======================================================================================

# One column per model type rather than one panel per external/internal/neural group. The
# external/internal distinction is carried by the vertical stack instead: the black latent ->
# target pair is the external model, and the red source row above it is what the internal
# version adds.
_ZOO_CONDENSED_MODEL_LABELS = ["PF", "rbf-pos\nspeed\nreward", "PF", "neural"]

# The column whose latent is multiplicatively modulated by a gain box drawn to its right.
_ZOO_CONDENSED_GAIN_COLUMN = 2

# The reduced-rank-regression column, drawn entirely in blue instead of red-over-black: it has
# no external counterpart, so the vertical stack carries no external/internal distinction.
_ZOO_CONDENSED_NEURAL_COLUMN = 3

# ModelZooSchematicConfig fields exposed as live Syd controls. The condensed layout has no
# containers and a single gain route, so the container/panel and paired-route fields drop out.
_ZOO_CONDENSED_TUNABLES = [
    "box_width",
    "box_height",
    "latent_height_scale",
    "gain_width",
    "col_gap",
    "row_gap",
    "box_corner_radius",
    "arrow_linewidth",
    "arrow_mutation_scale",
    "junction_dot_size",
    "junction_dot_offset",
    "box_fontsize",
]


class ModelZooCondensed(Viewer):
    """Condensed rounded-box schematic with one column per model type.

    Every column is the same source -> latent -> target stack: the two lower boxes (latent and
    target) are black and joined by a black arrow, which is the external model, and the red
    source box and arrow on top are what the internal version adds. The gain column carries an
    extra red gain box, fed from the source and multiplicatively joined onto that column's
    latent -> target arrow, and the neural column is drawn entirely in blue since reduced-rank
    regression has no external counterpart. Style comes from the same :class:`ModelZooSchematicConfig` as
    :class:`ModelZooSchematic`; ``_ZOO_CONDENSED_TUNABLES`` are exposed as live sliders.
    """

    def __init__(self, config: ModelZooSchematicConfig):
        self.cfg = config
        for name in _ZOO_CONDENSED_TUNABLES:
            lo, hi = _ZOO_TUNABLE_LIMITS[name]
            self.add_float(name, value=float(getattr(config, name)), min=lo, max=hi)

    def plot(self, state):
        cfg = replace(self.cfg, **{name: state[name] for name in _ZOO_CONDENSED_TUNABLES})
        bw = cfg.box_width
        h = cfg.box_height  # source / target rows
        hm = cfg.box_height * cfg.latent_height_scale  # taller latent (middle) row
        cg, rg = cfg.col_gap, cfg.row_gap

        # Row centers (target row bottom sits at y = 0), matching ModelZooSchematic.
        y_target = h / 2
        y_model = y_target + h / 2 + rg + hm / 2
        y_source = y_model + hm / 2 + rg + h / 2
        src_top, src_bot = y_source + h / 2, y_source - h / 2
        mid_top, mid_bot = y_model + hm / 2, y_model - hm / 2
        tgt_top = y_target + h / 2
        tgt_bot = y_target - h / 2

        fig, ax = plt.subplots(figsize=cfg.figsize, layout="constrained")
        fig.patch.set_facecolor(cfg.background_color)
        ax.set_aspect("equal")
        ax.axis("off")

        black, red, blue = cfg.black_box_color, cfg.red_box_color, cfg.blue_box_color
        ae, ai, an = cfg.external_arrow_color, cfg.internal_arrow_color, cfg.neural_arrow_color

        # Column centers; the gain column reserves room for its gain box before the next column.
        cxs = []
        x = bw / 2
        for i in range(len(_ZOO_CONDENSED_MODEL_LABELS)):
            cxs.append(x)
            x += bw + cg + (cfg.gain_width + cg if i == _ZOO_CONDENSED_GAIN_COLUMN else 0.0)

        for i, (cx, label) in enumerate(zip(cxs, _ZOO_CONDENSED_MODEL_LABELS)):
            is_neural = i == _ZOO_CONDENSED_NEURAL_COLUMN
            source_color, model_color = (blue, blue) if is_neural else (red, black)
            down_color, across_color = (an, an) if is_neural else (ai, ae)
            _zoo_box(ax, cx, y_source, bw, h, "source", source_color, cfg)
            _zoo_box(ax, cx, y_model, bw, hm, label, model_color, cfg)
            _zoo_box(ax, cx, y_target, bw, h, "target", model_color, cfg)
            _zoo_arrow(ax, cx, src_bot, cx, mid_top, down_color, cfg)  # internal: source -> latent
            _zoo_arrow(ax, cx, mid_bot, cx, tgt_top, across_color, cfg)  # external: latent -> target

        # Gain wiring: source -> gain along the source row, then gain down and back left onto
        # its column's latent -> target arrow, where the dot marks the multiplicative junction.
        # junction_dot_offset slides that dot rightward off the arrow so both stay visible.
        gain_col_cx = cxs[_ZOO_CONDENSED_GAIN_COLUMN]
        gain_cx = gain_col_cx + bw / 2 + cg + cfg.gain_width / 2
        y_junction = (mid_bot + tgt_top) / 2
        dot_x = gain_col_cx + cfg.junction_dot_offset * bw
        _zoo_box(ax, gain_cx, y_model, cfg.gain_width, hm, "gain", red, cfg)
        _zoo_line(ax, [gain_col_cx + bw / 2, gain_cx], [y_source, y_source], ai, cfg)
        _zoo_arrow(ax, gain_cx, y_source, gain_cx, mid_top, ai, cfg)
        _zoo_line(ax, [gain_cx, gain_cx, dot_x], [mid_bot, y_junction, y_junction], ai, cfg)
        ax.plot(dot_x, y_junction, marker="o", markersize=cfg.junction_dot_size, color=ai, zorder=4)

        margin = 0.3
        ax.set_xlim(cxs[0] - bw / 2 - margin, cxs[-1] + bw / 2 + margin)
        ax.set_ylim(tgt_bot - margin, src_top + margin)
        return fig


def model_zoo_condensed(
    config: ModelZooSchematicConfig | None = None,
    return_syd_viewer: bool = False,
):
    """
    Condensed model-zoo schematic laid out by model type instead of by model group.

    One column per model type, each a source -> latent -> target stack. The black latent and
    target boxes joined by a black arrow are the external model; the red source box and arrow
    above them are what the internal version adds, so the external/internal pairing is implied
    by the vertical stack rather than by separate panels. The gain column carries an extra red
    gain box wired from the source onto that column's latent -> target arrow, and the neural
    (reduced-rank regression) column is drawn entirely in blue.

    Parameters
    ----------
    config : ModelZooSchematicConfig or None
        Full style/layout config, shared with :func:`model_zoo_schematic`. The container and
        panel fields are unused here. A default one is created when None.
    return_syd_viewer : bool
        If True, return the Syd viewer instead of a rendered figure.
    """
    viewer = ModelZooCondensed(config or ModelZooSchematicConfig(figsize=(10.0, 5.0)))
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ======================================================================================
# Regression dimensionality sweep (mean +/- SE across sessions, one curve per model)
# ======================================================================================

# RegressionDimensionalitySweepConfig.process() (configs/regression.py) sweeps a different
# hyperparameter per model type: placefield models vary num_bins, RBFPos/pos_speed/full-regressor
# models vary num_basis, and RRR varies rank. Each produces its own {prefix}_{values,dim,mse,r2}
# result keys, so a model's sweep data must be read from its own prefix.
_SWEEP_KEY_BY_MODEL: dict[ModelName, str] = {
    "external_placefield_1d": "num_bins",
    "internal_placefield_1d": "num_bins",
    "external_placefield_1d_gain": "num_bins",
    "internal_placefield_1d_gain": "num_bins",
    "rbfpos_decoder_only": "num_basis",
    "rbfpos": "num_basis",
    "pos_speed_decoder_only_1dspeed": "num_basis",
    "pos_speed_1dspeed": "num_basis",
    "fullregressor_decoder_only_1dspeed": "num_basis",
    "fullregressor_1dspeed": "num_basis",
    "rrr": "rank",
    "rrr_no_intercept": "rank",
    "fullregressor_decoder_only_1dspeed_predreward": "num_basis",
    "fullregressor_1dspeed_predreward": "num_basis",
    "fullregressor_decoder_only_1dspeed_predreward_no_intercept": "num_basis",
    "fullregressor_1dspeed_predreward_no_intercept": "num_basis",
}

_DIM_SWEEP_METRIC_LABELS = {"r2": r"$R^2$", "mse": "MSE"}

# Match the model-zoo schematic: color encodes where the model lives, while line style
# encodes architecture. Matching external/internal models share a dash pattern and differ
# only in black/red; RRR retains the schematic's blue.
_DIM_SWEEP_ROLE_COLOR = {
    "external": "#000000",
    "internal": "#c00000",
    "neural": "#0000cd",
}
_DIM_SWEEP_ARCH_LINESTYLE = {
    "placefield": "-",
    "highd_pos": (0, (6, 2)),
    "highd_pos_speed": (0, (1, 1.7)),
    "highd_pos_speed_reward": "-.",
    "placefield_gain": (0, (3, 1, 1, 1, 1, 1)),
}
# Legend text for each architecture, in the order the entries are listed.
_DIM_SWEEP_ARCH_LABELS = {
    "placefield": "placefield",
    "highd_pos": "rbf-pos",
    "highd_pos_speed": "+speed",
    "highd_pos_speed_reward": "+reward",
    "placefield_gain": "+gain",
}
_DIM_SWEEP_MODEL_STYLE: dict[ModelName, dict] = {
    "external_placefield_1d": dict(role="external", arch="placefield"),
    "rbfpos_decoder_only": dict(role="external", arch="highd_pos"),
    "pos_speed_decoder_only_1dspeed": dict(role="external", arch="highd_pos_speed"),
    "fullregressor_decoder_only_1dspeed": dict(role="external", arch="highd_pos_speed_reward"),
    "internal_placefield_1d": dict(role="internal", arch="placefield"),
    "rbfpos": dict(role="internal", arch="highd_pos"),
    "pos_speed_1dspeed": dict(role="internal", arch="highd_pos_speed"),
    "fullregressor_1dspeed": dict(role="internal", arch="highd_pos_speed_reward"),
    "external_placefield_1d_gain": dict(role="external", arch="placefield_gain"),
    "internal_placefield_1d_gain": dict(role="internal", arch="placefield_gain"),
    "rrr": dict(role="neural", arch="placefield"),
    "rrr_no_intercept": dict(role="neural", arch="placefield"),
    "fullregressor_decoder_only_1dspeed_predreward": dict(role="external", arch="highd_pos_speed_reward"),
    "fullregressor_1dspeed_predreward": dict(role="internal", arch="highd_pos_speed_reward"),
    "fullregressor_decoder_only_1dspeed_predreward_no_intercept": dict(role="external", arch="highd_pos_speed_reward"),
    "fullregressor_1dspeed_predreward_no_intercept": dict(role="internal", arch="highd_pos_speed_reward"),
}

# Named model subsets for the sweep panel. "reduced" is the model set of
# :func:`model_performance_pairs` (see ``_paired_model_groups``): the placefield, full-regressor
# (position + speed + reward), and gain external/internal pairs, plus RRR. It drops the
# intermediate rbf-position and +speed architectures, so the sweep carries the same model types
# that panel condenses the zoo down to. Listed in ``KEY_FIGURE_MODELS`` order, which is the order
# the curves are drawn in.
_DIM_SWEEP_MODEL_SETS: dict[str, list[ModelName]] = {
    "all": list(KEY_FIGURE_MODELS),
    "reduced": [
        "external_placefield_1d",
        "fullregressor_decoder_only_1dspeed_predreward",
        "internal_placefield_1d",
        "fullregressor_1dspeed_predreward",
        "external_placefield_1d_gain",
        "internal_placefield_1d_gain",
        "rrr",
    ],
}


def _dim_sweep_curve(results: ResultsAggregator, model_name: ModelName, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Across-session x (mean dim, sorted) and y (metric) arrays: shapes (n_pts,), (n_sess, n_pts)."""
    prefix = _SWEEP_KEY_BY_MODEL[model_name]
    sel = results.sel(model_name=model_name, squeeze_ones=False)
    dim = np.atleast_2d(sel[f"{prefix}_dim"])
    y = np.atleast_2d(sel[f"{prefix}_{metric}"])
    x = np.nanmean(dim, axis=0)
    idx_sort = np.argsort(x)
    return x[idx_sort], y[:, idx_sort]


def _draw_dim_sweep_panel(
    ax,
    results: ResultsAggregator,
    model_names,
    metric: str,
    se: bool,
    xlog: bool,
    linewidth: float,
    fill_alpha: float,
    fontsize: float,
) -> None:
    """Draw the performance-vs-dimensionality curves (one per model) onto ``ax``."""
    for model_name in model_names:
        x, y = _dim_sweep_curve(results, model_name, metric)
        style = _DIM_SWEEP_MODEL_STYLE[model_name]
        color = _DIM_SWEEP_ROLE_COLOR[style["role"]]
        linestyle = _DIM_SWEEP_ARCH_LINESTYLE[style["arch"]]
        curve_width = linewidth * (1.35 if style["role"] == "neural" else 1.0)
        errorPlot(
            x,
            y,
            axis=0,
            se=se,
            ax=ax,
            color=color,
            linestyle=linestyle,
            linewidth=curve_width,
            alpha=fill_alpha,
        )

    if xlog:
        ax.set_xscale("log")
    xlim = ax.get_xlim()
    xticks = [1, 10, 100, 1000, 10000]
    xticks = [t for t in xticks if t <= xlim[1]]
    ax.set_xlabel("Regressor Dimensionality", fontsize=fontsize)
    ax.set_ylabel(_DIM_SWEEP_METRIC_LABELS[metric], fontsize=fontsize)
    # Only the architectures actually drawn get an entry, so a reduced model selection doesn't
    # advertise line styles that aren't on the axis. Ordered by _DIM_SWEEP_ARCH_LABELS, not by
    # the model selection, so the entries keep the same order whatever is selected.
    drawn_archs = {_DIM_SWEEP_MODEL_STYLE[model_name]["arch"] for model_name in model_names}
    architecture_handles = [
        Line2D([], [], color="0.25", linestyle=_DIM_SWEEP_ARCH_LINESTYLE[arch], linewidth=2.0, label=label)
        for arch, label in _DIM_SWEEP_ARCH_LABELS.items()
        if arch in drawn_archs
    ]
    ax.legend(
        handles=architecture_handles,
        loc="upper left",
        fontsize=fontsize,
        frameon=False,
        markerfirst=True,
        handlelength=3.0,
        handletextpad=0.8,
    )
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        xbounds=xlim,
        xticks=xticks,
        tick_fontsize=fontsize,
        spines_visible=["left", "bottom"],
    )
    # Apply this after format_spines, which calls tick_params itself. Include minor ticks
    # so logarithmic-axis labels follow the viewer fontsize as well.
    ax.tick_params(axis="both", which="both", labelsize=fontsize)


class RegressionDimSweepViewer(Viewer):
    """Mean +/- SE performance-vs-dimensionality curves, one per selected figure-2 model.

    Each model sweeps a different hyperparameter (num_bins / num_basis / rank; see
    ``_SWEEP_KEY_BY_MODEL``), and the resulting regressor dimensionality also depends on a
    session's number of environments, so sessions don't share an exact dimensionality grid.
    The x position at sweep index ``i`` is therefore the across-session mean dimensionality at
    that index, and the curve is the across-session mean +/- SE of the metric at that index,
    drawn with :func:`errorPlot`.
    """

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (6.0, 4.5)):
        self.results = results
        self.figsize = figsize
        self.add_multiple_selection("model_names", value=list(KEY_FIGURE_MODELS), options=list(KEY_FIGURE_MODELS))
        self.add_selection("metric", value="r2", options=list(_DIM_SWEEP_METRIC_LABELS))
        self.add_boolean("se", value=True)
        self.add_boolean("xlog", value=True)
        self.add_float("linewidth", value=2.0, min=0.5, max=6.0)
        self.add_float("fill_alpha", value=0.12, min=0.0, max=1.0)
        self.add_float("fontsize", value=9.0, min=4.0, max=24.0)

    def plot(self, state):
        fig, ax = plt.subplots(figsize=self.figsize, layout="constrained")
        _draw_dim_sweep_panel(
            ax,
            self.results,
            state["model_names"],
            state["metric"],
            se=state["se"],
            xlog=state["xlog"],
            linewidth=state["linewidth"],
            fill_alpha=state["fill_alpha"],
            fontsize=state["fontsize"],
        )
        return fig


def regression_dim_sweep(
    results: ResultsAggregator,
    model_names: list[ModelName] | None = None,
    metric: str = "r2",
    se: bool = True,
    xlog: bool = True,
    linewidth: float = 2.0,
    fill_alpha: float = 0.12,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (6.0, 4.5),
    return_syd_viewer: bool = False,
):
    """
    Mean +/- SE regression performance vs. dimensionality, one curve per model.

    For each selected figure-2 model, reads aggregated ``RegressionDimensionalitySweepConfig``
    results and draws the across-session mean +/- SE curve (via
    :func:`vrAnalysis.helpers.plotting.errorPlot`) of a fit metric against regressor
    dimensionality. Color follows the model-zoo schematic (black external, red internal, blue
    neural), while five line styles identify placefield, high-D position, +speed, +reward, and
    gain architectures. Matching external/internal architectures therefore share a line style.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionDimensionalitySweepConfig`` results, with ``model_name`` as a
        param axis over ``KEY_FIGURE_MODELS``.
    model_names : list[ModelName] or None
        Models to plot. Defaults to all of ``KEY_FIGURE_MODELS``.
    metric : {"r2", "mse"}
        Fit metric to plot on the y-axis.
    se : bool
        Standard error (True) or standard deviation (False) shading.
    xlog : bool
        Log-scale the dimensionality axis.
    linewidth : float
        Mean-curve line width.
    fill_alpha : float
        Opacity of the +/- SE/SD fill band.
    fontsize : float
        Font size for axis labels, tick labels, and both legends.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    """
    if metric not in _DIM_SWEEP_METRIC_LABELS:
        raise ValueError(f"metric must be one of {list(_DIM_SWEEP_METRIC_LABELS)}, got {metric!r}")

    viewer = RegressionDimSweepViewer(results, figsize=figsize)
    viewer.update_multiple_selection("model_names", value=list(model_names) if model_names is not None else list(KEY_FIGURE_MODELS))
    viewer.update_selection("metric", value=metric)
    viewer.update_boolean("se", value=se)
    viewer.update_boolean("xlog", value=xlog)
    viewer.update_float("linewidth", value=linewidth)
    viewer.update_float("fill_alpha", value=fill_alpha)
    viewer.update_float("fontsize", value=fontsize)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ======================================================================================
# RRR <-> external-latents predictability (rrr_to_external_latents results)
# ======================================================================================

# Colors mirror the model-zoo schematic: external models are black, internal (leak) models red.
_LATENTS_EXTERNAL_COLOR = "#000000"
_LATENTS_INTERNAL_COLOR = "#c00000"

_LATENTS_EACH_KEYS = [
    "test_score_each_rrr_to_true",
    "test_score_each_true_to_rrr",
    "test_score_each_rrr_to_pred",
    "test_score_each_pred_to_rrr",
]
# The external and internal models optimize hyperparameters independently, so each set of
# per-dimension scores must be sliced with its own model's column counts. The unsuffixed keys
# describe the external model (the ``*_true`` scores); ``*_pred`` describe the internal model.
_LATENTS_DIM_KEYS = [
    "num_pos_params_true",
    "num_speed_params_true",
    "num_reward_params_true",
    "num_pos_params_pred",
    "num_speed_params_pred",
    "num_reward_params_pred",
]

# The three regressor groups that make up the external model's basis (see
# RRRToExternalLatentsConfig / _pos_speed_reward_dims), in column order.
_LATENTS_GROUP_NAMES = ["Position", "Speed", "Reward"]


def _model_pair_label(model_ext_int_pair: tuple[ModelName, ModelName]) -> str:
    """Render an (external, internal) model pair as a widget-safe string label.

    Syd selections take scalar values, so the tuple-valued ``model_ext_int_pair`` param axis is
    shown as a label and mapped back to its tuple before slicing (same idea as ``_tuple_label``
    in figure3/figure4).
    """
    return " / ".join(short_model_name(name) for name in model_ext_int_pair)


# Widget label -> raw (external, internal) tuple, for decoding the selection before results.sel.
_MODEL_PAIR_LABELS: dict[str, tuple[ModelName, ModelName]] = {_model_pair_label(pair): pair for pair in VALID_MODEL_EXT_INT_PAIRS}
_DEFAULT_MODEL_PAIR_LABEL: str = _model_pair_label(DEFAULT_MODEL_EXT_INT_PAIR)


def _trimmed_rank_length(true_arr: np.ndarray, pred_arr: np.ndarray) -> int:
    """Last column index (+1) where at least one mouse has non-NaN data in BOTH arrays.

    Mice differ in how many RRR ranks they have, so high-rank columns can be all-NaN
    padding; an all-NaN column feeds NaN into errorPlot's fill polygon and crashes
    legend(loc="best").
    """
    n = min(true_arr.shape[1], pred_arr.shape[1])
    has_true = np.any(~np.isnan(true_arr[:, :n]), axis=0)
    has_pred = np.any(~np.isnan(pred_arr[:, :n]), axis=0)
    valid_cols = np.where(has_true & has_pred)[0]
    return int(valid_cols[-1]) + 1 if valid_cols.size else 0


def _valid_agg(values: np.ndarray, agg) -> float:
    """Aggregate one mouse's per-dimension scores within a group, dropping blown-up R^2 outliers.

    R^2 on a near-zero-variance target dim can blow up to a finite-but-astronomical value
    (passes isfinite, e.g. -3e29) instead of -inf, which would corrupt a fixed-ylim figure.
    """
    finite = values[np.isfinite(values) & (np.abs(values) < 1e3)]
    return float(agg(finite)) if finite.size else np.nan


def _omission_param_count(num_reward: int) -> int | None:
    """Width of the omission-response block at the tail of a session's reward columns.

    ``FullRegressorModel.build_regressors`` appends the reward blocks in the order
    (expectation, delivered response, omission response), so omission is the trailing slice of
    the reward group. With ``expectation_symmetric=True`` and ``reward_num_basis_lags = L`` the
    widths are ``(2L + 1, L + 1, L + 1)``, hence ``num_reward = 4L + 3`` and the omission width
    is ``L + 1 = (num_reward + 1) / 4``. Only call this for model pairs that have an omission
    block (see ``pair_has_omission_response``) -- a predictive-reward pair's narrower reward
    width never satisfies ``4L + 3``.

    Parameters
    ----------
    num_reward : int
        Total number of reward regressor columns for one session (``num_reward_params``).

    Returns
    -------
    int or None
        The omission block width, or None when ``num_reward`` is not of the form ``4L + 3``.
        A mismatch means ``make_temporal_basis(remove_empty=True)`` dropped columns, so the
        block boundaries can no longer be inferred from the stored dimension counts alone.
    """
    if num_reward <= 0 or (num_reward + 1) % 4 != 0:
        return None
    return (num_reward + 1) // 4


def _group_agg_scores(
    each_scores: np.ndarray,
    num_pos: np.ndarray,
    num_speed: np.ndarray,
    num_reward: np.ndarray,
    agg,
    exclude_omission: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-session (position, speed, reward) aggregate of a per-dimension score array.

    Each session's own ``(num_pos, num_speed, num_reward)`` boundaries slice its row of
    ``each_scores`` before aggregating, since a session's basis dimensionality (num_basis,
    num_environments, reward_num_basis_lags) differs from every other session's, so raw
    column index has no shared meaning across sessions.

    Parameters
    ----------
    each_scores : np.ndarray
        Per-dimension scores, shape (n_sessions, n_columns).
    num_pos, num_speed, num_reward : np.ndarray
        Per-session column counts for each group, shape (n_sessions,).
    agg : callable
        Reduction applied within a group (``np.nanmean`` or ``np.nanmedian``).
    exclude_omission : bool
        If True, drop the trailing omission-response columns from the reward group. On sessions
        with very few omission trials those columns are near-constant, so their R^2 is a
        measure of an unestimable regressor rather than of reward coding. Sessions whose reward
        width does not decompose (see :func:`_omission_param_count`) yield NaN and drop out.

    Returns
    -------
    tuple of np.ndarray
        Per-session (position, speed, reward) aggregates, each shape (n_sessions,).
    """
    n_sessions = each_scores.shape[0]
    pos_out = np.full(n_sessions, np.nan)
    speed_out = np.full(n_sessions, np.nan)
    reward_out = np.full(n_sessions, np.nan)
    for i in range(n_sessions):
        npos, nspeed, nreward = int(num_pos[i]), int(num_speed[i]), int(num_reward[i])
        if exclude_omission:
            num_omission = _omission_param_count(nreward)
            # An undecomposable width leaves an empty reward slice, so _valid_agg returns NaN
            # and the session drops out rather than being silently mis-sliced.
            nreward = nreward - num_omission if num_omission is not None else 0
        row = each_scores[i]
        pos_out[i] = _valid_agg(row[:npos], agg)
        speed_out[i] = _valid_agg(row[npos : npos + nspeed], agg)
        reward_out[i] = _valid_agg(row[npos + nspeed : npos + nspeed + nreward], agg)
    return pos_out, speed_out, reward_out


def _center_spread_plot(ax, x, data, axis, agg_method: str, **kwargs):
    """Draw a mean+/-SE (errorPlot) or median+/-IQR curve, sharing errorPlot's kwarg contract."""
    if agg_method == "median":
        center = np.nanmedian(data, axis=axis)
        lo = np.nanpercentile(data, 25, axis=axis)
        hi = np.nanpercentile(data, 75, axis=axis)
        fill_kwargs = kwargs.copy()
        fill_kwargs.pop("label", None)
        fill_kwargs["linewidth"] = 0
        ax.fill_between(x, hi, lo, **fill_kwargs)
        kwargs.pop("alpha", None)
        ax.plot(x, center, **kwargs)
    else:
        errorPlot(x, data, axis=axis, se=True, ax=ax, **kwargs)


def _latents_show_style(latents_show: str) -> tuple[bool, bool, bool, str, str]:
    """External/Internal/both selection -> (show_true, show_pred, both, color_true, color_pred).

    A single selection is drawn in black; "both" colors External black and Internal red.
    """
    show_true = latents_show in ("both", "external")
    show_pred = latents_show in ("both", "internal")
    both = latents_show == "both"
    color_true = _LATENTS_EXTERNAL_COLOR
    color_pred = _LATENTS_INTERNAL_COLOR if both else _LATENTS_EXTERNAL_COLOR
    return show_true, show_pred, both, color_true, color_pred


def _draw_latents_group_panel(
    ax,
    data: dict[str, np.ndarray],
    agg_method: str,
    latents_show: str,
    exclude_omission: bool,
    beewidth: float,
    dodge: float,
    alpha: float,
    fontsize: float,
) -> None:
    """Draw the dodged position/speed/reward beeswarms (RRR -> external/internal) onto ``ax``."""
    agg = np.nanmean if agg_method == "mean" else np.nanmedian
    show_true, show_pred, both, color_true, color_pred = _latents_show_style(latents_show)

    pos_true, speed_true, reward_true = _group_agg_scores(
        data["test_score_each_rrr_to_true"],
        data["num_pos_params_true"],
        data["num_speed_params_true"],
        data["num_reward_params_true"],
        agg,
        exclude_omission=exclude_omission,
    )
    pos_pred, speed_pred, reward_pred = _group_agg_scores(
        data["test_score_each_rrr_to_pred"],
        data["num_pos_params_pred"],
        data["num_speed_params_pred"],
        data["num_reward_params_pred"],
        agg,
        exclude_omission=exclude_omission,
    )
    group_scores = [(pos_true, pos_pred), (speed_true, speed_pred), (reward_true, reward_pred)]
    group_names = list(_LATENTS_GROUP_NAMES)
    if exclude_omission:
        group_names[-1] = "Reward (no omission)"
    group_centers = np.arange(len(group_names), dtype=float)

    for center, (true_vals, pred_vals) in zip(group_centers, group_scores):
        entries = []
        if show_true:
            entries.append((true_vals, center - dodge if both else center, color_true))
        if show_pred:
            entries.append((pred_vals, center + dodge if both else center, color_pred))
        for vals, x0, color in entries:
            finite_vals = vals[np.isfinite(vals)]
            offsets = beeswarm(finite_vals) if finite_vals.size > 1 else np.zeros_like(finite_vals)
            ax.plot(x0 + beewidth * offsets, finite_vals, linestyle="none", color=color, marker="o", markersize=4, alpha=alpha)

    spacing_required = 1.2 * (dodge + beewidth)
    ax.set_xlim(group_centers[0] - spacing_required, group_centers[-1] + spacing_required)
    # Fix ylim before format_spines so the bottom-spine position is data-independent (and, via
    # sharey, identical on the rank panel -> continuous x-spine across both panels).
    ax.set_ylim(-0.1, 1.1)
    format_spines(
        ax,
        x_pos=-0.01,
        y_pos=-0.01,
        spines_visible=["left", "bottom"],
        xbounds=[group_centers[0], group_centers[-1]],
        ybounds=[0, 1],
    )
    ax.set_xticks(group_centers, labels=group_names, rotation=0, rotation_mode="anchor", ha="center", fontsize=fontsize)
    ax.set_yticks([0, 1])
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_ylabel(r"$R^2$", labelpad=-10, fontsize=fontsize)
    ax.set_xlabel("from neural latents", fontsize=fontsize)


def _draw_latents_rank_panel(
    ax,
    data: dict[str, np.ndarray],
    agg_method: str,
    latents_show: str,
    rank_axis_log: bool,
    inset_num_dims: int,
    fontsize: float,
) -> None:
    """Draw the rank-ordered curve (external/internal -> RRR) onto ``ax``, with optional inset."""
    show_true, show_pred, both, color_true, color_pred = _latents_show_style(latents_show)

    n_ranks = max(_trimmed_rank_length(data["test_score_each_true_to_rrr"], data["test_score_each_pred_to_rrr"]), 1)
    curve_true = data["test_score_each_true_to_rrr"][:, :n_ranks]
    curve_pred = data["test_score_each_pred_to_rrr"][:, :n_ranks]
    rank_positions = np.arange(1, n_ranks + 1)  # 1-indexed: rank 0 is undefined on a log axis

    if rank_axis_log:
        ax.set_xscale("log")
    if show_true:
        _center_spread_plot(
            ax,
            rank_positions,
            curve_true,
            axis=0,
            agg_method=agg_method,
            color=color_true,
            linewidth=1.5,
            alpha=0.2,
            label="External",
        )
    if show_pred:
        _center_spread_plot(
            ax,
            rank_positions,
            curve_pred,
            axis=0,
            agg_method=agg_method,
            color=color_pred,
            linewidth=1.5,
            alpha=0.2,
            label="Internal",
        )

    # Inset highlighting the initial-dimension dropoff (linear x only).
    if not rank_axis_log:
        n_inset = min(inset_num_dims, n_ranks)
        inset_ypush = 0 if both else 0.05
        axins = ax.inset_axes([0.35, 0.35 + inset_ypush, 0.60, 0.45])
        if show_true:
            _center_spread_plot(
                axins,
                rank_positions[:n_inset],
                curve_true[:, :n_inset],
                axis=0,
                agg_method=agg_method,
                color=color_true,
                linewidth=1.5,
                alpha=0.2,
            )
        if show_pred:
            _center_spread_plot(
                axins,
                rank_positions[:n_inset],
                curve_pred[:, :n_inset],
                axis=0,
                agg_method=agg_method,
                color=color_pred,
                linewidth=1.5,
                alpha=0.2,
            )
        axins.set_xlim(0, max(n_inset, 2))
        axins.set_ylim(-0.1, 1.1)
        format_spines(
            axins,
            x_pos=-0.02,
            y_pos=-0.01,
            spines_visible=["left", "bottom"],
            xbounds=[0, n_inset],
            ybounds=[0, 1],
            xticks=[0, n_inset],
            yticks=[0, 1],
            tick_fontsize=fontsize * 0.95,
        )

    format_spines(
        ax,
        x_pos=-0.01,
        y_pos=-0.01,
        spines_visible=["bottom"],
        xbounds=[1, max(n_ranks, 2)],
        ybounds=[0, 1],
    )
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.set_xlabel("rank\nto neural latents", fontsize=fontsize)
    if both:
        ax.legend(loc="upper right", fontsize=fontsize, frameon=False)


def _select_latents_data(results: ResultsAggregator, state) -> dict[str, np.ndarray]:
    """Select latents results for the current (spks_type, activity_parameters_name, model_ext_int_pair, rrr_variance, normalize)."""
    return results.sel(
        keys=_LATENTS_EACH_KEYS + _LATENTS_DIM_KEYS,
        avg_by_mouse=False,
        squeeze_ones=False,
        spks_type=state["spks_type"],
        activity_parameters_name=state["activity_parameters_name"],
        model_ext_int_pair=_MODEL_PAIR_LABELS[state["model_ext_int_pair"]],
        rrr_variance=state["rrr_variance"],
        normalize=state["normalize"],
    )


class RRRExternalLatentsViewer(Viewer):
    """RRR <-> external-latents predictability, per-dimension breakdown as two linked x-axes.

    ax[0] and ax[1] share one y-axis (R^2) but have independent, differently-scaled x-axes:
    - ax[0] (Position/Speed/Reward): ``rrr_to_*`` (RRR -> external/internal) is scored per
      external-basis dimension. Those dimensions aren't a single homogeneous group -- they're
      the concatenation of position, speed, and reward regressors (see
      ``RRRToExternalLatentsConfig`` / ``_pos_speed_reward_dims``), and sessions differ in how
      many columns each sub-group has. Each session's per-dimension scores are therefore split
      into (position, speed, reward) using that session's own boundaries, aggregated (mean or
      median, per ``agg_method``) within each group, and drawn as three beeswarms, each dodged
      into External (black) / Internal (red) halves. One point is one session (results are
      selected with ``avg_by_mouse=False``), not one mouse. ``exclude_reward_omission`` drops
      the omission-response columns from the reward group. External and Internal latents come
      from two different models with independently optimized hyperparameters, so each half is
      sliced with its own column counts (``num_*_params`` vs ``num_*_params_pred``).
    - ax[1] (Rank, log-scale): ``*_to_rrr`` (external/internal -> RRR) is scored per RRR-latent
      (dim=0 of the ridge fit); RRR latents are rank-ordered (1-indexed for the log axis), so
      those scores are drawn as a mean+/-SE (or median+/-IQR) curve over every available rank.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        figsize: tuple[float, float] = (7.0, 4.0),
        fontsize: float = 10,
    ):
        self.results = results
        self.figsize = figsize
        self.fontsize = fontsize
        self._data: dict[str, np.ndarray] = {}

        self.add_selection("spks_type", value="sigrebase", options=list(VALID_SPKS_TYPES))
        self.add_selection("activity_parameters_name", value="default", options=list(ACTIVITY_PARAMETERS_NAMES))
        self.add_selection("model_ext_int_pair", value=_DEFAULT_MODEL_PAIR_LABEL, options=list(_MODEL_PAIR_LABELS))
        self.add_selection("rrr_variance", value=0.95, options=list(VALID_RRR_VARIANCE))
        self.add_boolean("normalize", value=False)
        self.add_selection("agg_method", value="mean", options=["mean", "median"])
        self.add_selection("latents_show", value="both", options=["both", "external", "internal"])
        self.add_boolean("exclude_reward_omission", value=False)
        self.add_float("beewidth", value=0.3, min=0.0, max=1.0)
        self.add_float("dodge_offset", value=0.3, min=0.0, max=1.0)
        self.add_float("alpha", value=0.5, min=0.0, max=1.0)
        self.add_float("width_rank_axis", value=0.6, min=0.25, max=2.0)
        self.add_boolean("rank_axis_log", value=True)
        self.add_integer("inset_num_dims", value=10, min=2, max=50)

        for name in ("spks_type", "activity_parameters_name", "model_ext_int_pair", "rrr_variance", "normalize"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select data for the current (spks_type, activity_parameters_name, model_ext_int_pair, rrr_variance, normalize)."""
        self._data = _select_latents_data(self.results, state)

    def plot(self, state):
        plt.close("all")
        fig = plt.figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, state["width_rank_axis"]])
        ax_group = fig.add_subplot(gs[0])
        ax_curve = fig.add_subplot(gs[1], sharey=ax_group)

        _draw_latents_group_panel(
            ax_group,
            self._data,
            agg_method=state["agg_method"],
            latents_show=state["latents_show"],
            # The predictive-reward pair has no omission columns to drop, so the toggle is inert
            # there rather than slicing a block that doesn't exist.
            exclude_omission=state["exclude_reward_omission"] and pair_has_omission_response(_MODEL_PAIR_LABELS[state["model_ext_int_pair"]]),
            beewidth=state["beewidth"],
            dodge=state["dodge_offset"],
            alpha=state["alpha"],
            fontsize=self.fontsize,
        )
        _draw_latents_rank_panel(
            ax_curve,
            self._data,
            agg_method=state["agg_method"],
            latents_show=state["latents_show"],
            rank_axis_log=state["rank_axis_log"],
            inset_num_dims=state["inset_num_dims"],
            fontsize=self.fontsize,
        )
        return fig


def rrr_external_latents_score(
    results: ResultsAggregator,
    spks_type: SpksTypes = "sigrebase",
    activity_parameters_name: str = "default",
    model_ext_int_pair: tuple[ModelName, ModelName] = DEFAULT_MODEL_EXT_INT_PAIR,
    rrr_variance: float | str = 0.95,
    normalize: bool = False,
    agg_method: str = "mean",
    latents_show: str = "both",
    exclude_reward_omission: bool = False,
    beewidth: float = 0.3,
    dodge_offset: float = 0.3,
    alpha: float = 0.5,
    width_rank_axis: float = 0.6,
    rank_axis_log: bool = True,
    inset_num_dims: int = 10,
    fontsize: float = 8,
    figsize: tuple[float, float] = (5.0, 3.0),
    return_syd_viewer: bool = False,
):
    """
    RRR <-> external-latents predictability from ``RRRToExternalLatentsConfig`` results.

    Two panels sharing one y-axis (R^2) with independent x-axes. ax[0] (Position/Speed/Reward)
    splits ``rrr_to_*`` (RRR -> external/internal) per session into position/speed/reward sub-groups
    (their sizes vary by session -- see ``RRRToExternalLatentsConfig``), aggregates within each
    group, and draws three beeswarms (one point per session), each dodged into External/Internal
    halves. ax[1] (Rank,
    log-scale) draws ``*_to_rrr`` (external/internal -> RRR), scored per RRR-latent and rank-
    ordered, as a mean+/-SE (or median+/-IQR) curve over every available rank.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RRRToExternalLatentsConfig`` results.
    spks_type, activity_parameters_name, model_ext_int_pair, rrr_variance, normalize
        Selects which stored variation to read (see ``RRRToExternalLatentsConfig``).
        ``model_ext_int_pair`` picks the (external, internal) regressor pair and is passed as a
        native tuple (it is encoded to the widget's string label internally); the
        predictive-reward pair has no omission-response columns, so
        ``exclude_reward_omission`` is a no-op for it.
    agg_method : {"mean", "median"}
        Aggregation used both for the per-session position/speed/reward group scores and for the
        rank-ordered curve's center line (mean+/-SE or median+/-IQR).
    latents_show : {"both", "external", "internal"}
        Which latents to draw. "both" colors External black and Internal red; a single selection
        is drawn in black.
    exclude_reward_omission : bool
        If True, drop the omission-response columns from the reward group. Sessions with few
        omission trials fit many omission regressors to almost no events, making those columns
        near-constant; their R^2 then reports an unestimable regressor rather than reward coding.
    beewidth : float
        Horizontal spread of points within one beeswarm (External or Internal) half.
    dodge_offset : float
        Horizontal separation between the External and Internal beeswarm halves within a group.
    alpha : float
        Marker opacity for the beeswarm points.
    width_rank_axis : float
        Width ratio for the rank-axis panel relative to the position/speed/reward panel.
    rank_axis_log : bool
        If True, log-scale the rank-axis panel's x-axis.
    inset_num_dims : int
        Number of initial ranks shown in the ax[1] inset. The inset only appears when
        ``rank_axis_log`` is False.
    fontsize : float
        Tick-label fontsize for both panels.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    """
    viewer = RRRExternalLatentsViewer(results, figsize=figsize, fontsize=fontsize)
    viewer.update_selection("spks_type", value=spks_type)
    viewer.update_selection("activity_parameters_name", value=activity_parameters_name)
    viewer.update_selection("model_ext_int_pair", value=_model_pair_label(tuple(model_ext_int_pair)))
    viewer.update_selection("rrr_variance", value=rrr_variance)
    viewer.update_boolean("normalize", value=normalize)
    viewer.refresh_data(viewer.state)  # pre-deploy update_* may not fire on_change

    viewer.update_selection("agg_method", value=agg_method)
    viewer.update_selection("latents_show", value=latents_show)
    viewer.update_boolean("exclude_reward_omission", value=exclude_reward_omission)
    viewer.update_float("beewidth", value=beewidth)
    viewer.update_float("dodge_offset", value=dodge_offset)
    viewer.update_float("alpha", value=alpha)
    viewer.update_float("width_rank_axis", value=width_rank_axis)
    viewer.update_boolean("rank_axis_log", value=rank_axis_log)
    viewer.update_integer("inset_num_dims", value=inset_num_dims)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ======================================================================================
# Combined: dimensionality sweep + RRR <-> external-latents predictability
# ======================================================================================


class DimSweepLatentsViewer(Viewer):
    """Three panels combining latents predictability with the regression dimensionality sweep.

    The panels answer one question in two halves: how well the neural latents and the
    external/internal regressor latents predict each other (ax[0], ax[1]), and how much of a
    session's activity each model explains as a function of how many regressor dimensions it
    spends (ax[2]).

    - ax[0] (Position/Speed/Reward) and ax[1] (Rank): the two panels of
      :class:`RRRExternalLatentsViewer`, sharing one y-axis (R^2) with each other but not with
      ax[2], whose y-axis is the sweep metric over its own range.
    - ax[2] (Regressor Dimensionality): across-session mean +/- SE of a fit metric versus
      regressor dimensionality, one curve per model. See :class:`RegressionDimSweepViewer`.

    The two halves read different aggregated results (``RRRToExternalLatentsConfig`` and
    ``RegressionDimensionalitySweepConfig``), so both aggregators are passed in. ``fontsize`` is
    shared by all three panels; every other knob keeps the per-panel meaning it has in the
    single-figure viewers.
    """

    def __init__(
        self,
        results_latents: ResultsAggregator,
        results_sweep: ResultsAggregator,
        figsize: tuple[float, float] = (9.0, 3.0),
    ):
        self.results_latents = results_latents
        self.results_sweep = results_sweep
        self.figsize = figsize
        self._data: dict[str, np.ndarray] = {}

        # --- shared ---
        self.add_float("fontsize", value=9.0, min=4.0, max=24.0)

        # --- ax[0] / ax[1]: RRR <-> external latents ---
        self.add_selection("spks_type", value="sigrebase", options=list(VALID_SPKS_TYPES))
        self.add_selection("activity_parameters_name", value="default", options=list(ACTIVITY_PARAMETERS_NAMES))
        self.add_selection("model_ext_int_pair", value=_DEFAULT_MODEL_PAIR_LABEL, options=list(_MODEL_PAIR_LABELS))
        self.add_selection("rrr_variance", value=0.95, options=list(VALID_RRR_VARIANCE))
        self.add_boolean("normalize", value=False)
        self.add_selection("agg_method", value="mean", options=["mean", "median"])
        self.add_selection("latents_show", value="both", options=["both", "external", "internal"])
        self.add_boolean("exclude_reward_omission", value=False)
        self.add_float("beewidth", value=0.3, min=0.0, max=1.0)
        self.add_float("dodge_offset", value=0.3, min=0.0, max=1.0)
        self.add_float("alpha", value=0.5, min=0.0, max=1.0)
        self.add_boolean("rank_axis_log", value=True)
        self.add_integer("inset_num_dims", value=10, min=2, max=50)

        # --- ax[2]: dimensionality sweep ---
        self.add_selection("model_set", value="all", options=list(_DIM_SWEEP_MODEL_SETS))
        self.add_multiple_selection("model_names", value=list(KEY_FIGURE_MODELS), options=list(KEY_FIGURE_MODELS))
        self.add_selection("metric", value="r2", options=list(_DIM_SWEEP_METRIC_LABELS))
        self.add_boolean("se", value=True)
        self.add_boolean("xlog", value=True)
        self.add_float("linewidth", value=2.0, min=0.5, max=6.0)
        self.add_float("fill_alpha", value=0.12, min=0.0, max=1.0)

        # --- layout ---
        self.add_float("width_sweep_axis", value=1.0, min=0.25, max=3.0)
        self.add_float("width_rank_axis", value=0.6, min=0.25, max=2.0)

        for name in ("spks_type", "activity_parameters_name", "model_ext_int_pair", "rrr_variance", "normalize"):
            self.on_change(name, self.refresh_data)
        self.on_change("model_set", self.apply_model_set)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select latents data for the current (spks_type, activity_parameters_name, model_ext_int_pair, rrr_variance, normalize)."""
        self._data = _select_latents_data(self.results_latents, state)

    def apply_model_set(self, state):
        """Reset ``model_names`` to the chosen set. Individual models stay selectable afterwards."""
        self.update_multiple_selection("model_names", value=list(_DIM_SWEEP_MODEL_SETS[state["model_set"]]))

    def plot(self, state):
        plt.close("all")
        fig = plt.figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, state["width_rank_axis"], state["width_sweep_axis"]])
        ax_group = fig.add_subplot(gs[0])
        # sharey only within the latents pair: ax_sweep's y-axis is the sweep metric, not R^2.
        ax_curve = fig.add_subplot(gs[1], sharey=ax_group)
        ax_sweep = fig.add_subplot(gs[2])

        fontsize = state["fontsize"]
        _draw_latents_group_panel(
            ax_group,
            self._data,
            agg_method=state["agg_method"],
            latents_show=state["latents_show"],
            # See RRRExternalLatentsViewer.plot: no omission block on the predictive-reward pair.
            exclude_omission=state["exclude_reward_omission"] and pair_has_omission_response(_MODEL_PAIR_LABELS[state["model_ext_int_pair"]]),
            beewidth=state["beewidth"],
            dodge=state["dodge_offset"],
            alpha=state["alpha"],
            fontsize=fontsize,
        )
        _draw_latents_rank_panel(
            ax_curve,
            self._data,
            agg_method=state["agg_method"],
            latents_show=state["latents_show"],
            rank_axis_log=state["rank_axis_log"],
            inset_num_dims=state["inset_num_dims"],
            fontsize=fontsize,
        )
        _draw_dim_sweep_panel(
            ax_sweep,
            self.results_sweep,
            state["model_names"],
            state["metric"],
            se=state["se"],
            xlog=state["xlog"],
            linewidth=state["linewidth"],
            fill_alpha=state["fill_alpha"],
            fontsize=fontsize,
        )
        return fig


def dim_sweep_latents(
    results_latents: ResultsAggregator,
    results_sweep: ResultsAggregator,
    spks_type: SpksTypes = "sigrebase",
    activity_parameters_name: str = "default",
    model_ext_int_pair: tuple[ModelName, ModelName] = DEFAULT_MODEL_EXT_INT_PAIR,
    rrr_variance: float | str = 0.95,
    normalize: bool = False,
    agg_method: str = "mean",
    latents_show: str = "both",
    exclude_reward_omission: bool = False,
    beewidth: float = 0.3,
    dodge_offset: float = 0.3,
    alpha: float = 0.5,
    rank_axis_log: bool = True,
    inset_num_dims: int = 10,
    model_set: str = "all",
    model_names: list[ModelName] | None = None,
    metric: str = "r2",
    se: bool = True,
    xlog: bool = True,
    linewidth: float = 2.0,
    fill_alpha: float = 0.12,
    width_sweep_axis: float = 1.0,
    width_rank_axis: float = 0.6,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (9.0, 3.0),
    return_syd_viewer: bool = False,
):
    """
    RRR <-> external-latents predictability and the regression dimensionality sweep, in one figure.

    Merges :func:`rrr_external_latents_score` (ax[0], ax[1]) and :func:`regression_dim_sweep`
    (ax[2]) into a single three-panel figure. ax[0] and ax[1] share one y-axis (R^2) and show,
    respectively, the per-session position/speed/reward beeswarms of RRR -> external/internal
    predictability and the rank-ordered external/internal -> RRR curve. ax[2] draws the
    across-session mean +/- SE fit metric versus regressor dimensionality, one curve per model
    (black external, red internal, blue neural; line style encodes architecture).

    The two halves come from different aggregated results, so both aggregators are required.
    ``fontsize`` and ``figsize`` are shared across panels; all other arguments keep the meaning
    they have in the single-figure functions.

    Parameters
    ----------
    results_latents : ResultsAggregator
        Aggregated ``RRRToExternalLatentsConfig`` results. Drives ax[0] and ax[1].
    results_sweep : ResultsAggregator
        Aggregated ``RegressionDimensionalitySweepConfig`` results, with ``model_name`` as a
        param axis over ``KEY_FIGURE_MODELS``. Drives ax[2].
    spks_type, activity_parameters_name, model_ext_int_pair, rrr_variance, normalize
        Selects which stored latents variation to read (see ``RRRToExternalLatentsConfig``).
        ``model_ext_int_pair`` picks the (external, internal) regressor pair and is passed as a
        native tuple (it is encoded to the widget's string label internally); the
        predictive-reward pair has no omission-response columns, so ``exclude_reward_omission``
        is a no-op for it.
    agg_method : {"mean", "median"}
        Aggregation used both for the per-session position/speed/reward group scores in ax[0]
        and for the ax[1] curve's center line (mean+/-SE or median+/-IQR).
    latents_show : {"both", "external", "internal"}
        Which latents to draw. "both" colors External black and Internal red; a single selection
        is drawn in black.
    exclude_reward_omission : bool
        If True, drop the omission-response columns from the ax[0] reward group. Sessions with
        few omission trials fit many omission regressors to almost no events, making those
        columns near-constant; their R^2 then reports an unestimable regressor rather than
        reward coding.
    beewidth : float
        Horizontal spread of points within one ax[0] beeswarm (External or Internal) half.
    dodge_offset : float
        Horizontal separation between the External and Internal beeswarm halves within a group.
    alpha : float
        Marker opacity for the ax[0] beeswarm points.
    rank_axis_log : bool
        If True, log-scale the ax[1] x-axis.
    inset_num_dims : int
        Number of initial ranks shown in the ax[1] inset. The inset only appears when
        ``rank_axis_log`` is False.
    model_set : {"all", "reduced"}
        Which models ax[2] draws. "all" is every model in ``KEY_FIGURE_MODELS``; "reduced" is
        the model set of :func:`model_performance_pairs` -- the placefield, full-regressor, and
        gain external/internal pairs plus RRR, without the intermediate rbf-position and +speed
        architectures. Ignored when ``model_names`` is given.
    model_names : list[ModelName] or None
        Explicit model list for ax[2], overriding ``model_set``.
    metric : {"r2", "mse"}
        Fit metric on the ax[2] y-axis.
    se : bool
        Standard error (True) or standard deviation (False) shading in ax[2].
    xlog : bool
        Log-scale the ax[2] dimensionality axis.
    linewidth : float
        Mean-curve line width in ax[2].
    fill_alpha : float
        Opacity of the ax[2] +/- SE/SD fill band.
    width_sweep_axis, width_rank_axis : float
        Width ratios for the sweep and rank panels relative to the position/speed/reward panel.
    fontsize : float
        Font size for axis labels, tick labels, and legends in all three panels.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    """
    if metric not in _DIM_SWEEP_METRIC_LABELS:
        raise ValueError(f"metric must be one of {list(_DIM_SWEEP_METRIC_LABELS)}, got {metric!r}")

    viewer = DimSweepLatentsViewer(results_latents, results_sweep, figsize=figsize)

    # Seed the params the latents selection depends on first, then refresh explicitly:
    # pre-deploy update_* may not fire on_change.
    viewer.update_selection("spks_type", value=spks_type)
    viewer.update_selection("activity_parameters_name", value=activity_parameters_name)
    viewer.update_selection("model_ext_int_pair", value=_model_pair_label(tuple(model_ext_int_pair)))
    viewer.update_selection("rrr_variance", value=rrr_variance)
    viewer.update_boolean("normalize", value=normalize)
    viewer.refresh_data(viewer.state)

    viewer.update_float("fontsize", value=fontsize)

    viewer.update_selection("agg_method", value=agg_method)
    viewer.update_selection("latents_show", value=latents_show)
    viewer.update_boolean("exclude_reward_omission", value=exclude_reward_omission)
    viewer.update_float("beewidth", value=beewidth)
    viewer.update_float("dodge_offset", value=dodge_offset)
    viewer.update_float("alpha", value=alpha)
    viewer.update_boolean("rank_axis_log", value=rank_axis_log)
    viewer.update_integer("inset_num_dims", value=inset_num_dims)

    # model_set drives model_names through apply_model_set once deployed, but pre-deploy update_*
    # may not fire on_change, so the selection is set here explicitly.
    viewer.update_selection("model_set", value=model_set)
    viewer.update_multiple_selection("model_names", value=list(model_names) if model_names is not None else list(_DIM_SWEEP_MODEL_SETS[model_set]))
    viewer.update_selection("metric", value=metric)
    viewer.update_boolean("se", value=se)
    viewer.update_boolean("xlog", value=xlog)
    viewer.update_float("linewidth", value=linewidth)
    viewer.update_float("fill_alpha", value=fill_alpha)

    viewer.update_float("width_sweep_axis", value=width_sweep_axis)
    viewer.update_float("width_rank_axis", value=width_rank_axis)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ======================================================================================
# Model performance summary (one score per model, relative to a reference model)
# ======================================================================================

# This panel's model set is fixed: it is the whole external / internal / neural zoo laid out
# along the x-axis in a deliberate order (externals, then their internal counterparts, then
# the gain pair, then RRR), so it is not a selectable knob the way it is in the sweep viewer.
_PERFORMANCE_MODEL_NAMES: list[ModelName] = list(KEY_FIGURE_MODELS)

# Axis label for each selectable metric. Both are plotted in their stored units -- R^2 as a
# fraction (<= 1), not a percentage -- so there is no display rescaling anywhere in this panel.
_PERFORMANCE_METRICS: dict[str, str] = {
    "r2": r"R$^2$",
    "mse": "MSE",
}

# Role (and therefore dot color) of each model, reusing the sweep panel's schematic-matched
# assignment: black external, red internal, blue neural.
_PERFORMANCE_ROLES = np.array([_DIM_SWEEP_MODEL_STYLE[name]["role"] for name in _PERFORMANCE_MODEL_NAMES])

_PERFORMANCE_ROLE_LABELS = {
    "external": "External Latent",
    "internal": "Internal Latent",
    "neural": "Neural Latent",
}


def _performance_scores(
    results: ResultsAggregator,
    model_names: list[ModelName],
    metric: str,
    spks_type: SpksTypes,
    activity_parameters_name: str,
    avg_by_mouse: bool,
) -> np.ndarray:
    """Metric for every model in ``model_names``, shape (n_models, n_subjects).

    Subjects are mice when ``avg_by_mouse`` is True and sessions otherwise. Values are returned in
    their stored units, R^2 as a fraction.
    """
    scores = [
        results.sel(
            model_name=model_name,
            spks_type=spks_type,
            activity_parameters_name=activity_parameters_name,
            avg_by_mouse=avg_by_mouse,
        )[metric]
        for model_name in model_names
    ]
    return np.stack(scores, axis=0)


def _draw_performance_traces(
    ax,
    xvals: np.ndarray,
    values: np.ndarray,
    roles: np.ndarray,
    markersize: float,
    mean_linewidth: float,
    subject_linewidth: float,
    subject_alpha: float,
) -> None:
    """Draw one faint line per subject, the across-subject mean, and role-colored mean dots."""
    ax.plot(xvals, values, color=("k", subject_alpha), linewidth=subject_linewidth)
    mean_values = np.nanmean(values, axis=1)
    ax.plot(xvals, mean_values, color="k", linewidth=mean_linewidth)
    for role, color in _DIM_SWEEP_ROLE_COLOR.items():
        idx_role = roles == role
        ax.plot(xvals[idx_role], mean_values[idx_role], color=color, marker="o", markersize=markersize, linestyle="none")


class ModelPerformanceViewer(Viewer):
    """One score per model for the whole model zoo, relative to a reference model.

    Each model contributes a single number per subject (a mouse when ``avg_by_mouse``, else a
    session): its optimized-hyperparameter test score from ``RegressionConfig``. The main axis
    plots that score *minus the reference model's* score, so the panel reads as "what each extra
    ingredient buys over a plain placefield model", with one faint line per subject behind the
    across-subject mean. Mean dots are colored by where the model's latent lives (black external,
    red internal, blue neural), matching the model-zoo schematic and the dimensionality sweep.

    An inset repeats the same traces without the reference subtraction, so the absolute scale of
    the scores stays visible next to the differences.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        figsize: tuple[float, float] = (7.0, 5.0),
    ):
        self.results = results
        self.figsize = figsize
        self.short_names = [short_model_name(name) for name in _PERFORMANCE_MODEL_NAMES]
        self._scores = np.empty((len(_PERFORMANCE_MODEL_NAMES), 0))

        # --- data selection ---
        self.add_selection("spks_type", value="sigrebase", options=list(VALID_SPKS_TYPES))
        self.add_selection("activity_parameters_name", value="default", options=list(ACTIVITY_PARAMETERS_NAMES))
        self.add_selection("metric", value="r2", options=list(_PERFORMANCE_METRICS))
        self.add_boolean("avg_by_mouse", value=True)
        self.add_selection("reference_model", value=_PERFORMANCE_MODEL_NAMES[0], options=list(_PERFORMANCE_MODEL_NAMES))

        # --- main axis style ---
        self.add_float("fontsize", value=12.0, min=4.0, max=24.0)
        self.add_float("markersize", value=5.0, min=1.0, max=15.0)
        self.add_float("mean_linewidth", value=1.5, min=0.25, max=5.0)
        self.add_float("subject_linewidth", value=0.5, min=0.0, max=3.0)
        self.add_float("subject_alpha", value=0.3, min=0.0, max=1.0)
        self.add_float("xtick_rotation", value=45.0, min=0.0, max=90.0)
        # y knobs are in metric units (R^2 as a fraction), with explicit steps: syd rounds a value
        # to its parameter's step, and the default 0.01 is too coarse for an axis spanning ~0.1.
        self.add_float("ytick_max", value=0.1, min=0.001, max=2.0, step=0.001)
        self.add_float_range("ylim", value=(-0.03, 0.12), min=-1.0, max=2.0, step=0.001)
        self.add_boolean("show_zero_line", value=True)
        self.add_float("zero_line_pad", value=0.5, min=0.0, max=2.0)

        # --- legend (data coordinates of the main axis: x in model units, y in metric units) ---
        self.add_boolean("show_legend", value=True)
        self.add_float("legend_x", value=5.0, min=-1.0, max=12.0)
        self.add_float("legend_y", value=0.085, min=-0.5, max=1.0, step=0.005)
        self.add_float("legend_dy", value=-0.01, min=-0.1, max=0.1, step=0.005)

        # --- inset (absolute scores) ---
        self.add_boolean("show_inset", value=True)
        self.add_float("inset_x", value=0.1, min=0.0, max=1.0)
        self.add_float("inset_y", value=0.5, min=0.0, max=1.0)
        self.add_float("inset_width", value=0.35, min=0.05, max=1.0)
        self.add_float("inset_height", value=0.45, min=0.05, max=1.0)
        self.add_float("inset_markersize", value=3.0, min=1.0, max=12.0)
        self.add_float("inset_fontsize", value=10.0, min=4.0, max=24.0)
        self.add_float("inset_ytick_max", value=0.2, min=0.001, max=2.0, step=0.001)
        self.add_float_range("inset_ylim", value=(-0.02, 0.25), min=-1.0, max=2.0, step=0.001)

        for name in ("spks_type", "activity_parameters_name", "metric", "avg_by_mouse"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select scores for the current (spks_type, activity_parameters_name, metric, avg_by_mouse)."""
        self._scores = _performance_scores(
            self.results,
            _PERFORMANCE_MODEL_NAMES,
            state["metric"],
            state["spks_type"],
            state["activity_parameters_name"],
            state["avg_by_mouse"],
        )

    def plot(self, state):
        fontsize = state["fontsize"]
        markersize = state["markersize"]
        mean_linewidth = state["mean_linewidth"]
        subject_linewidth = state["subject_linewidth"]
        subject_alpha = state["subject_alpha"]
        metric_label = _PERFORMANCE_METRICS[state["metric"]]

        xvals = np.arange(len(_PERFORMANCE_MODEL_NAMES), dtype=float)
        scores = self._scores
        idx_reference = _PERFORMANCE_MODEL_NAMES.index(state["reference_model"])
        relative_scores = scores - scores[idx_reference]

        plt.close("all")
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, layout="constrained")

        if state["show_zero_line"]:
            pad = state["zero_line_pad"]
            ax.plot([xvals[0] - pad, xvals[-1] + pad], [0, 0], color="k", linewidth=0.5, linestyle="--")
        _draw_performance_traces(ax, xvals, relative_scores, _PERFORMANCE_ROLES, markersize, mean_linewidth, subject_linewidth, subject_alpha)

        # Dot-colored legend drawn by hand in data coordinates: the entries explain the marker
        # colors, so they are the markers themselves rather than proxy handles.
        if state["show_legend"]:
            legend_x, legend_y, legend_dy = state["legend_x"], state["legend_y"], state["legend_dy"]
            for i, (role, color) in enumerate(_DIM_SWEEP_ROLE_COLOR.items()):
                y = legend_y + i * legend_dy
                ax.plot(legend_x, y, color=color, marker="o", markersize=markersize, linestyle="none")
                ax.text(
                    legend_x + 0.25,
                    y,
                    _PERFORMANCE_ROLE_LABELS[role],
                    color=color,
                    ha="left",
                    va="center",
                    fontsize=fontsize,
                )

        # Fix ylim before format_spines, which positions the offset spines relative to the
        # current limits. The x limits are left to matplotlib (the dashed zero line and the
        # legend text set the padding), while the bottom spine spans the models themselves.
        ax.set_ylim(state["ylim"])
        format_spines(
            ax,
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[xvals[0], xvals[-1]],
            ybounds=state["ylim"],
            yticks=[0, state["ytick_max"]],
            tick_fontsize=fontsize,
        )
        ax.set_xticks(
            xvals,
            labels=self.short_names,
            rotation=state["xtick_rotation"],
            rotation_mode="anchor",
            ha="right",
            fontsize=fontsize,
        )
        # Applied after format_spines, which calls tick_params itself.
        ax.tick_params(axis="both", which="both", labelsize=fontsize)
        ax.set_ylabel(rf"$\Delta$ {metric_label}", labelpad=-10, fontsize=fontsize)

        if state["show_inset"]:
            inset_fontsize = state["inset_fontsize"]
            inset = ax.inset_axes([state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]])
            _draw_performance_traces(
                inset,
                xvals,
                scores,
                _PERFORMANCE_ROLES,
                state["inset_markersize"],
                mean_linewidth,
                subject_linewidth,
                subject_alpha,
            )
            inset.set_ylim(state["inset_ylim"])
            format_spines(
                inset,
                x_pos=-0.02,
                y_pos=-0.02,
                spines_visible=["left"],
                ybounds=state["inset_ylim"],
                yticks=[0, state["inset_ytick_max"]],
                tick_fontsize=inset_fontsize,
            )
            inset.set_xticks([])
            inset.tick_params(axis="y", which="both", labelsize=inset_fontsize)
            inset.set_ylabel(metric_label, labelpad=-10, fontsize=inset_fontsize)

        return fig


def model_performance_summary(
    results: ResultsAggregator,
    spks_type: SpksTypes = "sigrebase",
    activity_parameters_name: str = "default",
    metric: str = "r2",
    avg_by_mouse: bool = True,
    reference_model: ModelName = _PERFORMANCE_MODEL_NAMES[0],
    markersize: float = 5.0,
    mean_linewidth: float = 1.5,
    subject_linewidth: float = 0.5,
    subject_alpha: float = 0.3,
    xtick_rotation: float = 45.0,
    ytick_max: float = 0.1,
    ylim: tuple[float, float] = (-0.03, 0.12),
    show_zero_line: bool = True,
    zero_line_pad: float = 0.5,
    show_legend: bool = True,
    legend_x: float = 5.0,
    legend_y: float = 0.085,
    legend_dy: float = -0.01,
    show_inset: bool = True,
    inset_x: float = 0.1,
    inset_y: float = 0.5,
    inset_width: float = 0.35,
    inset_height: float = 0.45,
    inset_markersize: float = 3.0,
    inset_fontsize: float = 10.0,
    inset_ytick_max: float = 0.2,
    inset_ylim: tuple[float, float] = (-0.02, 0.25),
    fontsize: float = 12.0,
    figsize: tuple[float, float] = (7.0, 5.0),
    return_syd_viewer: bool = False,
):
    """
    Per-model regression performance across the model zoo, relative to a reference model.

    Reads aggregated ``RegressionConfig`` results (one optimized-hyperparameter test score per
    model per session) and draws, for the fixed model list ``KEY_FIGURE_MODELS``, each model's
    score minus the reference model's score. One faint line per subject (mouse when
    ``avg_by_mouse``, else session) runs behind the across-subject mean, and the mean dots are
    colored by where the model's latent lives: black external, red internal, blue neural, matching
    the model-zoo schematic. An inset repeats the same traces without the reference subtraction so
    the absolute scale stays visible. The model list is fixed, so it is not an argument.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionConfig`` results, with ``model_name`` as a param axis covering
        every model in ``KEY_FIGURE_MODELS``.
    spks_type, activity_parameters_name
        Selects which stored variation to read (see ``RegressionConfig``).
    metric : {"r2", "mse"}
        Fit metric, plotted in its stored units: R^2 as a fraction (<= 1), MSE in its own units.
        The y-axis arguments (``ylim``, ``ytick_max``, the inset's, and the legend's y position)
        are in those units too.
    avg_by_mouse : bool
        Average sessions within a mouse before plotting, so one trace is one mouse.
    reference_model : ModelName
        Model subtracted from every score on the main axis.
    markersize : float
        Size of the role-colored mean dots on the main axis.
    mean_linewidth : float
        Line width of the across-subject mean trace.
    subject_linewidth, subject_alpha : float
        Line width and opacity of the individual subject traces.
    xtick_rotation : float
        Rotation of the model-name tick labels, in degrees.
    ytick_max : float
        Upper y tick on the main axis (the lower one is 0).
    ylim : tuple[float, float]
        Main-axis y limits. The offset left spine spans exactly this range, so it also sets how
        far the spine extends past the ticks.
    show_zero_line : bool
        Draw the dashed y = 0 reference line on the main axis.
    zero_line_pad : float
        How far the zero line extends past the outermost ticks, in tick units, on both sides.
    show_legend : bool
        Draw the hand-placed external/internal/neural dot legend.
    legend_x, legend_y, legend_dy : float
        Position of the first legend entry and the vertical step between entries, in the main
        axis's data coordinates.
    show_inset : bool
        Draw the absolute-score inset.
    inset_x, inset_y, inset_width, inset_height : float
        Inset bounds as a fraction of the main axis.
    inset_markersize, inset_fontsize, inset_ytick_max : float
        Inset marker size, tick/label font size, and upper y tick.
    inset_ylim : tuple[float, float]
        Inset y limits, spanned by its offset left spine.
    fontsize : float
        Font size for the main axis's tick labels, y label, and legend text.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    """
    if metric not in _PERFORMANCE_METRICS:
        raise ValueError(f"metric must be one of {list(_PERFORMANCE_METRICS)}, got {metric!r}")

    viewer = ModelPerformanceViewer(results, figsize=figsize)

    # Seed the params the score selection depends on first, then refresh explicitly:
    # pre-deploy update_* may not fire on_change.
    viewer.update_selection("spks_type", value=spks_type)
    viewer.update_selection("activity_parameters_name", value=activity_parameters_name)
    viewer.update_selection("metric", value=metric)
    viewer.update_boolean("avg_by_mouse", value=avg_by_mouse)
    viewer.refresh_data(viewer.state)

    viewer.update_selection("reference_model", value=reference_model)

    viewer.update_float("fontsize", value=fontsize)
    viewer.update_float("markersize", value=markersize)
    viewer.update_float("mean_linewidth", value=mean_linewidth)
    viewer.update_float("subject_linewidth", value=subject_linewidth)
    viewer.update_float("subject_alpha", value=subject_alpha)
    viewer.update_float("xtick_rotation", value=xtick_rotation)
    viewer.update_float("ytick_max", value=ytick_max)
    viewer.update_float_range("ylim", value=tuple(ylim))
    viewer.update_boolean("show_zero_line", value=show_zero_line)
    viewer.update_float("zero_line_pad", value=zero_line_pad)

    viewer.update_boolean("show_legend", value=show_legend)
    viewer.update_float("legend_x", value=legend_x)
    viewer.update_float("legend_y", value=legend_y)
    viewer.update_float("legend_dy", value=legend_dy)

    viewer.update_boolean("show_inset", value=show_inset)
    viewer.update_float("inset_x", value=inset_x)
    viewer.update_float("inset_y", value=inset_y)
    viewer.update_float("inset_width", value=inset_width)
    viewer.update_float("inset_height", value=inset_height)
    viewer.update_float("inset_markersize", value=inset_markersize)
    viewer.update_float("inset_fontsize", value=inset_fontsize)
    viewer.update_float("inset_ytick_max", value=inset_ytick_max)
    viewer.update_float_range("inset_ylim", value=tuple(inset_ylim))

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ======================================================================================
# Model performance by model type (external / internal pairs around one tick per type)
# ======================================================================================

# The Full type comes in two parallel flavors, differing only in which reward regressor the models
# were fit with: the delivered reward ("reward") or the predictive reward signal ("predreward").
# Which flavor the panel shows is a knob, so the Full group's members are chosen at plot time while
# everything else about the model set stays fixed.
_PAIRED_FULL_VARIANTS: dict[str, dict[str, ModelName]] = {
    "predreward": {"external": "fullregressor_decoder_only_1dspeed_predreward", "internal": "fullregressor_1dspeed_predreward"},
    "reward": {"external": "fullregressor_decoder_only_1dspeed", "internal": "fullregressor_1dspeed"},
}
_PAIRED_DEFAULT_FULL_VARIANT = "predreward"


def _paired_model_groups(full_variant: str) -> list[tuple[str, dict[str, ModelName]]]:
    """The condensed performance panel's model types, for one Full-regressor variant.

    Four model *types* instead of eleven models, each drawn as its external and internal members
    flanking a single tick. RRR is a type with only a neural member, since reduced-rank regression
    has no external/internal counterpart. Each entry is (tick label, members); the label is written
    out here rather than derived from short_model_name, which is too long once a whole type shares
    one tick. Ordering within a group is the x-order its members are drawn in. Fixed by design,
    exactly like _PERFORMANCE_MODEL_NAMES, apart from the Full group's variant.

    Parameters
    ----------
    full_variant : {"predreward", "reward"}
        Which reward regressor the Full group's models were fit with (``_PAIRED_FULL_VARIANTS``).

    Returns
    -------
    list[tuple[str, dict[str, ModelName]]]
        (tick label, {role: model name}) per type, in x order.
    """
    return [
        ("PF", {"external": "external_placefield_1d", "internal": "internal_placefield_1d"}),
        ("Full", _PAIRED_FULL_VARIANTS[full_variant]),
        ("+Gain", {"external": "external_placefield_1d_gain", "internal": "internal_placefield_1d_gain"}),
        ("Neural", {"neural": "rrr"}),
    ]


def _paired_model_names(full_variant: str) -> list[ModelName]:
    """Flattened model list for one Full variant, in x order (group by group, member by member)."""
    return [name for _, group in _paired_model_groups(full_variant) for name in group.values()]


def _paired_model_index(model_name: ModelName) -> int:
    """Position of a model in the flattened paired set, whichever Full variant it belongs to.

    Lets a reference model chosen under one Full variant carry over to the other: the variants share
    a layout, so the same position holds the counterpart model.

    Parameters
    ----------
    model_name : ModelName
        A model belonging to some variant of the paired set.

    Returns
    -------
    int
        Its index in ``_paired_model_names``.
    """
    for variant in _PAIRED_FULL_VARIANTS:
        names = _paired_model_names(variant)
        if model_name in names:
            return names.index(model_name)
    raise ValueError(f"{model_name!r} is not part of the paired model set")


# The layout of the groups -- roles, tick assignment, labels -- is the same for every Full variant;
# only the two Full model names change. So these flattened views are built once, from the default.
_PAIRED_ROLES = np.array([role for _, group in _paired_model_groups(_PAIRED_DEFAULT_FULL_VARIANT) for role in group])
_PAIRED_GROUP_INDEX = np.array(
    [igroup for igroup, (_, group) in enumerate(_paired_model_groups(_PAIRED_DEFAULT_FULL_VARIANT)) for _ in group], dtype=float
)
_PAIRED_GROUP_LABELS = [label for label, _ in _paired_model_groups(_PAIRED_DEFAULT_FULL_VARIANT)]

# The legend here explains the pair flanking each tick, not the whole zoo. The neural point is
# alone on its own tick and already named by that tick's label, so it gets no entry.
_PAIRED_ROLE_LABELS = {"external": "Ext", "internal": "Int"}

# Fractional spine offset handed to format_spines on this panel. The segmented x spine is drawn by
# hand at the same offset (and format_spines' default spine width) so its segments land exactly
# where the continuous bottom spine would have.
_PAIRED_SPINE_OFFSET = -0.02
_PAIRED_SPINE_LINEWIDTH = 1.0


class ModelPerformancePairsViewer(Viewer):
    """Per-model-type regression performance, external and internal members flanking one tick.

    The condensed counterpart to :class:`ModelPerformanceViewer`: the same relative-score main
    axis and absolute-score inset, but only four model types (see ``_paired_model_groups``), one
    x tick each. A type's external (black) and internal (red) members are shifted off their shared
    tick by -/+ ``pair_offset``, so the pair reads as one comparison instead of two unrelated
    positions; RRR has only a neural (blue) member and sits on its tick. The legend names just the
    flanking pair (Ext / Int) -- the neural point is already named by its tick label.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        figsize: tuple[float, float] = (5.0, 4.0),
    ):
        self.results = results
        self.figsize = figsize
        default_names = _paired_model_names(_PAIRED_DEFAULT_FULL_VARIANT)
        self._scores = np.empty((len(default_names), 0))

        # --- data selection ---
        self.add_selection("spks_type", value="sigrebase", options=list(VALID_SPKS_TYPES))
        self.add_selection("activity_parameters_name", value="default", options=list(ACTIVITY_PARAMETERS_NAMES))
        self.add_selection("metric", value="r2", options=list(_PERFORMANCE_METRICS))
        self.add_boolean("avg_by_mouse", value=True)
        self.add_selection("full_variant", value=_PAIRED_DEFAULT_FULL_VARIANT, options=list(_PAIRED_FULL_VARIANTS))
        self.add_selection("reference_model", value=default_names[0], options=list(default_names))

        # --- pair layout ---
        self.add_float("pair_offset", value=0.15, min=0.0, max=0.5)

        # --- main axis style ---
        self.add_float("fontsize", value=12.0, min=4.0, max=24.0)
        self.add_float("markersize", value=5.0, min=1.0, max=15.0)
        self.add_float("mean_linewidth", value=1.5, min=0.25, max=5.0)
        self.add_float("subject_linewidth", value=0.5, min=0.0, max=3.0)
        self.add_float("subject_alpha", value=0.3, min=0.0, max=1.0)
        self.add_float("xtick_rotation", value=45.0, min=0.0, max=90.0)
        # Every y knob below is in metric units (R^2 as a fraction). Steps are given explicitly:
        # syd rounds a value to its parameter's step, and the default step of 0.01 is a coarse grid
        # once the axis spans ~0.1 rather than ~10. The limits move on a deliberately coarse 0.02
        # grid, so dragging them lands on round numbers instead of arbitrary decimals.
        self.add_float("ytick_max", value=0.1, min=0.01, max=2.0, step=0.01)
        self.add_float_range("ylim", value=(-0.04, 0.12), min=-1.0, max=2.0, step=0.02)
        self.add_boolean("show_zero_line", value=True)
        self.add_float("zero_line_pad", value=0.25, min=0.0, max=2.0)
        self.add_boolean("segmented_xspine", value=False)

        # --- legend (data coordinates of the main axis: x in tick units, y in metric units) ---
        self.add_boolean("show_legend", value=True)
        self.add_float("legend_x", value=1.5, min=-1.0, max=5.0)
        self.add_float("legend_y", value=0.085, min=-0.5, max=1.0, step=0.005)
        self.add_float("legend_dy", value=-0.01, min=-0.1, max=0.1, step=0.005)
        self.add_float("legend_markersize", value=5.0, min=1.0, max=15.0)
        self.add_float("legend_text_dx", value=0.1, min=0.0, max=1.0)

        # --- inset (absolute scores) ---
        self.add_boolean("show_inset", value=True)
        self.add_float("inset_x", value=0.1, min=0.0, max=1.0)
        self.add_float("inset_y", value=0.5, min=0.0, max=1.0)
        self.add_float("inset_width", value=0.35, min=0.05, max=1.0)
        self.add_float("inset_height", value=0.45, min=0.05, max=1.0)
        self.add_float("inset_markersize", value=3.0, min=1.0, max=12.0)
        self.add_float("inset_fontsize", value=10.0, min=4.0, max=24.0)
        self.add_float("inset_ytick_max", value=0.2, min=0.01, max=2.0, step=0.01)
        self.add_float_range("inset_ylim", value=(-0.02, 0.24), min=-1.0, max=2.0, step=0.02)

        for name in ("spks_type", "activity_parameters_name", "metric", "avg_by_mouse"):
            self.on_change(name, self.refresh_data)
        self.on_change("full_variant", self.switch_full_variant)
        self.refresh_data(self.state)

    def switch_full_variant(self, state):
        """Re-point ``reference_model`` and the scores at the newly selected Full variant.

        The reference keeps its position in the model set rather than its name, so a reference on a
        Full member follows the switch to that variant's counterpart.
        """
        names = _paired_model_names(state["full_variant"])
        self.update_selection("reference_model", value=names[_paired_model_index(state["reference_model"])], options=list(names))
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select scores for the current (spks_type, activity_parameters_name, metric, avg_by_mouse, full_variant)."""
        self._scores = _performance_scores(
            self.results,
            _paired_model_names(state["full_variant"]),
            state["metric"],
            state["spks_type"],
            state["activity_parameters_name"],
            state["avg_by_mouse"],
        )

    def plot(self, state):
        fontsize = state["fontsize"]
        markersize = state["markersize"]
        mean_linewidth = state["mean_linewidth"]
        subject_linewidth = state["subject_linewidth"]
        subject_alpha = state["subject_alpha"]
        metric_label = _PERFORMANCE_METRICS[state["metric"]]

        # Each model sits at its group's tick, external and internal shifted symmetrically off it;
        # the lone neural member has no counterpart to be separated from, so it stays on the tick.
        pair_offset = state["pair_offset"]
        role_offsets = {"external": -pair_offset, "internal": pair_offset, "neural": 0.0}
        xvals = _PAIRED_GROUP_INDEX + np.array([role_offsets[role] for role in _PAIRED_ROLES])
        group_ticks = np.arange(len(_PAIRED_GROUP_LABELS), dtype=float)

        scores = self._scores
        idx_reference = _paired_model_names(state["full_variant"]).index(state["reference_model"])
        relative_scores = scores - scores[idx_reference]

        plt.close("all")
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, layout="constrained")

        if state["show_zero_line"]:
            # Padded from the outermost drawn points, not the ticks: the left end has to clear the
            # first group's offset external member, so it sits pair_offset further left than the
            # right end (whose neural point is on its tick) sits right.
            pad = state["zero_line_pad"]
            ax.plot([xvals.min() - pad, xvals.max() + pad], [0, 0], color="k", linewidth=0.5, linestyle="--")
        _draw_performance_traces(ax, xvals, relative_scores, _PAIRED_ROLES, markersize, mean_linewidth, subject_linewidth, subject_alpha)

        if state["show_legend"]:
            legend_x, legend_y, legend_dy = state["legend_x"], state["legend_y"], state["legend_dy"]
            legend_markersize = state["legend_markersize"]
            legend_text_dx = state["legend_text_dx"]
            for i, (role, label) in enumerate(_PAIRED_ROLE_LABELS.items()):
                y = legend_y + i * legend_dy
                color = _DIM_SWEEP_ROLE_COLOR[role]
                ax.plot(legend_x, y, color=color, marker="o", markersize=legend_markersize, linestyle="none")
                # center_baseline, not center: "center" centers the text's full bbox, which sits
                # visibly high next to a dot for labels with no descender ("Ext", "Int").
                ax.text(legend_x + legend_text_dx, y, label, color=color, ha="left", va="center_baseline", fontsize=fontsize)

        # Ticks mark the group centers, but the continuous spine spans the drawn points, which the
        # offsets push outside the first and last tick.
        segmented = state["segmented_xspine"]
        ylim = state["ylim"]
        ax.set_ylim(ylim)
        format_spines(
            ax,
            x_pos=_PAIRED_SPINE_OFFSET,
            y_pos=_PAIRED_SPINE_OFFSET,
            spines_visible=["left"] if segmented else ["left", "bottom"],
            xbounds=[min(xvals.min(), group_ticks[0]), max(xvals.max(), group_ticks[-1])],
            ybounds=ylim,
            yticks=[0, state["ytick_max"]],
            tick_fontsize=fontsize,
        )
        ax.set_xticks(
            group_ticks,
            labels=_PAIRED_GROUP_LABELS,
            rotation=state["xtick_rotation"],
            rotation_mode="anchor",
            ha="right",
            fontsize=fontsize,
        )
        # Applied after format_spines, which calls tick_params itself.
        ax.tick_params(axis="both", which="both", labelsize=fontsize)

        if segmented:
            # One spine segment per group instead of one spine across all of them, each spanning
            # the pair it holds. The neural group gets the same width even though only one point
            # sits in it, so every group reads as the same kind of bracket. Drawn by hand at the
            # y the hidden bottom spine would have taken (format_spines' y_pos, in data units),
            # which is outside ylim -- hence clip_on=False. The tick marks would then sit under the
            # middle of each segment, so they are dropped and only the labels are kept.
            y_spine = ylim[0] + _PAIRED_SPINE_OFFSET * (ylim[1] - ylim[0])
            for tick in group_ticks:
                ax.plot(
                    [tick - pair_offset, tick + pair_offset],
                    [y_spine, y_spine],
                    color="k",
                    linewidth=_PAIRED_SPINE_LINEWIDTH,
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=3,
                )
            ax.tick_params(axis="x", length=0)

        ax.set_ylabel(rf"$\Delta$ {metric_label}", labelpad=-10, fontsize=fontsize)

        if state["show_inset"]:
            inset_fontsize = state["inset_fontsize"]
            inset = ax.inset_axes([state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]])
            _draw_performance_traces(
                inset,
                xvals,
                scores,
                _PAIRED_ROLES,
                state["inset_markersize"],
                mean_linewidth,
                subject_linewidth,
                subject_alpha,
            )
            inset.set_ylim(state["inset_ylim"])
            format_spines(
                inset,
                x_pos=-0.02,
                y_pos=-0.02,
                spines_visible=["left"],
                ybounds=state["inset_ylim"],
                yticks=[0, state["inset_ytick_max"]],
                tick_fontsize=inset_fontsize,
            )
            inset.set_xticks([])
            inset.tick_params(axis="y", which="both", labelsize=inset_fontsize)
            inset.set_ylabel(metric_label, labelpad=-10, fontsize=inset_fontsize)

        return fig


def model_performance_pairs(
    results: ResultsAggregator,
    spks_type: SpksTypes = "sigrebase",
    activity_parameters_name: str = "default",
    metric: str = "r2",
    avg_by_mouse: bool = True,
    full_variant: str = _PAIRED_DEFAULT_FULL_VARIANT,
    reference_model: ModelName = _paired_model_names(_PAIRED_DEFAULT_FULL_VARIANT)[0],
    pair_offset: float = 0.15,
    markersize: float = 5.0,
    mean_linewidth: float = 1.5,
    subject_linewidth: float = 0.5,
    subject_alpha: float = 0.3,
    xtick_rotation: float = 45.0,
    ytick_max: float = 0.1,
    ylim: tuple[float, float] = (-0.04, 0.12),
    show_zero_line: bool = True,
    zero_line_pad: float = 0.25,
    segmented_xspine: bool = False,
    show_legend: bool = True,
    legend_x: float = 1.5,
    legend_y: float = 0.085,
    legend_dy: float = -0.01,
    legend_markersize: float = 5.0,
    legend_text_dx: float = 0.1,
    show_inset: bool = True,
    inset_x: float = 0.1,
    inset_y: float = 0.5,
    inset_width: float = 0.35,
    inset_height: float = 0.45,
    inset_markersize: float = 3.0,
    inset_fontsize: float = 10.0,
    inset_ytick_max: float = 0.2,
    inset_ylim: tuple[float, float] = (-0.02, 0.24),
    fontsize: float = 12.0,
    figsize: tuple[float, float] = (5.0, 4.0),
    return_syd_viewer: bool = False,
):
    """
    Regression performance for four model types, external/internal members flanking one tick.

    The condensed counterpart to :func:`model_performance_summary`: same relative-score main axis
    and absolute-score inset, but only the placefield, full-regressor (position + speed + reward),
    gain, and RRR types -- one x tick each. A type's external (black) and internal (red) members
    are shifted symmetrically off their shared tick by ``pair_offset``; RRR has only a neural
    (blue) member and sits on its tick. The model set and the tick labels are both fixed in
    ``_paired_model_groups``, so neither is an argument -- apart from which Full variant it uses.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionConfig`` results, with ``model_name`` as a param axis covering every
        model in ``_paired_model_groups(full_variant)``.
    spks_type, activity_parameters_name
        Selects which stored variation to read (see ``RegressionConfig``).
    metric : {"r2", "mse"}
        Fit metric, plotted in its stored units: R^2 as a fraction (<= 1), MSE in its own units.
        The y-axis arguments (``ylim``, ``ytick_max``, the inset's, and the legend's y position)
        are in those units too.
    avg_by_mouse : bool
        Average sessions within a mouse before plotting, so one trace is one mouse.
    full_variant : {"predreward", "reward"}
        Which reward regressor the Full pair was fit with: the predictive reward signal
        (``fullregressor[_decoder_only]_1dspeed_predreward``) or the delivered reward
        (``fullregressor[_decoder_only]_1dspeed``). Nothing else about the panel changes.
    reference_model : ModelName
        Model subtracted from every score on the main axis. Must belong to the selected
        ``full_variant`` if it is one of the Full pair.
    pair_offset : float
        Half-separation of a type's external and internal members, in tick units: external is
        drawn at ``-pair_offset`` from the tick and internal at ``+pair_offset``.
    markersize : float
        Size of the role-colored mean dots on the main axis.
    mean_linewidth : float
        Line width of the across-subject mean trace.
    subject_linewidth, subject_alpha : float
        Line width and opacity of the individual subject traces.
    xtick_rotation : float
        Rotation of the type tick labels, in degrees.
    ytick_max : float
        Upper y tick on the main axis (the lower one is 0).
    ylim : tuple[float, float]
        Main-axis y limits, spanned by its offset left spine. Rounded to the nearest 0.02, the
        grid both limit knobs move on.
    show_zero_line : bool
        Draw the dashed y = 0 reference line on the main axis.
    zero_line_pad : float
        How far the zero line extends past the outermost drawn points, in tick units, on both
        sides. Measured from the points rather than the ticks, so the left end clears the first
        group's offset external member by the same distance the right end clears the neural point.
    segmented_xspine : bool
        Replace the continuous bottom spine with one segment per group, each spanning
        ``-pair_offset`` to ``+pair_offset`` around its tick -- including the neural group, whose
        segment matches the others despite holding a single point. The x tick marks are dropped
        with it (the labels stay), since they would sit under the middle of each segment.
    show_legend : bool
        Draw the hand-placed Ext/Int dot legend. It covers only the flanking pair; the neural point
        is named by its own tick label.
    legend_x, legend_y, legend_dy : float
        Position of the first legend entry and the vertical step between entries, in the main
        axis's data coordinates.
    legend_markersize : float
        Size of the legend dots, independent of the mean dots on the data.
    legend_text_dx : float
        Horizontal gap between a legend dot and its label, in data coordinates.
    show_inset : bool
        Draw the absolute-score inset.
    inset_x, inset_y, inset_width, inset_height : float
        Inset bounds as a fraction of the main axis.
    inset_markersize, inset_fontsize, inset_ytick_max : float
        Inset marker size, tick/label font size, and upper y tick.
    inset_ylim : tuple[float, float]
        Inset y limits, spanned by its offset left spine. Rounded to the nearest 0.02, like
        ``ylim``.
    fontsize : float
        Font size for the main axis's tick labels, y label, and legend text.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    """
    if metric not in _PERFORMANCE_METRICS:
        raise ValueError(f"metric must be one of {list(_PERFORMANCE_METRICS)}, got {metric!r}")
    if full_variant not in _PAIRED_FULL_VARIANTS:
        raise ValueError(f"full_variant must be one of {list(_PAIRED_FULL_VARIANTS)}, got {full_variant!r}")

    viewer = ModelPerformancePairsViewer(results, figsize=figsize)

    # Seed the params the score selection depends on first, then refresh explicitly:
    # pre-deploy update_* may not fire on_change.
    viewer.update_selection("spks_type", value=spks_type)
    viewer.update_selection("activity_parameters_name", value=activity_parameters_name)
    viewer.update_selection("metric", value=metric)
    viewer.update_boolean("avg_by_mouse", value=avg_by_mouse)
    viewer.update_selection("full_variant", value=full_variant)
    viewer.refresh_data(viewer.state)

    # The reference options are the selected variant's models, so they are set alongside the value
    # rather than relying on the (possibly unfired) full_variant callback to have widened them.
    viewer.update_selection("reference_model", value=reference_model, options=list(_paired_model_names(full_variant)))

    viewer.update_float("pair_offset", value=pair_offset)
    viewer.update_boolean("segmented_xspine", value=segmented_xspine)

    viewer.update_float("fontsize", value=fontsize)
    viewer.update_float("markersize", value=markersize)
    viewer.update_float("mean_linewidth", value=mean_linewidth)
    viewer.update_float("subject_linewidth", value=subject_linewidth)
    viewer.update_float("subject_alpha", value=subject_alpha)
    viewer.update_float("xtick_rotation", value=xtick_rotation)
    viewer.update_float("ytick_max", value=ytick_max)
    viewer.update_float_range("ylim", value=tuple(ylim))
    viewer.update_boolean("show_zero_line", value=show_zero_line)
    viewer.update_float("zero_line_pad", value=zero_line_pad)

    viewer.update_boolean("show_legend", value=show_legend)
    viewer.update_float("legend_x", value=legend_x)
    viewer.update_float("legend_y", value=legend_y)
    viewer.update_float("legend_dy", value=legend_dy)
    viewer.update_float("legend_markersize", value=legend_markersize)
    viewer.update_float("legend_text_dx", value=legend_text_dx)

    viewer.update_boolean("show_inset", value=show_inset)
    viewer.update_float("inset_x", value=inset_x)
    viewer.update_float("inset_y", value=inset_y)
    viewer.update_float("inset_width", value=inset_width)
    viewer.update_float("inset_height", value=inset_height)
    viewer.update_float("inset_markersize", value=inset_markersize)
    viewer.update_float("inset_fontsize", value=inset_fontsize)
    viewer.update_float("inset_ytick_max", value=inset_ytick_max)
    viewer.update_float_range("inset_ylim", value=tuple(inset_ylim))

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ======================================================================================
# Regression performance over familiarity (one curve per model, x = session #)
# ======================================================================================

# A RegressionConfig score is a whole-session number, so the x-axis is a mouse's own
# chronological session index -- there is no per-environment breakdown to align to. Curves are
# stacked across mice and averaged, the same shape as figure3's familiarity panel.
_FAMILIARITY_XLABEL = "Session #"

# The neural model is the point of the comparison, so it is fixed rather than selectable; the
# control is any other model the pipeline swept.
_FAMILIARITY_NEURAL_MODEL: ModelName = "rrr"
_FAMILIARITY_CONTROL_OPTIONS: list[ModelName] = [name for name in KEY_FIGURE_MODELS if name != _FAMILIARITY_NEURAL_MODEL]
_FAMILIARITY_DEFAULT_CONTROL: ModelName = "external_placefield_1d"

# The control is drawn black and the neural model blue whatever role the control belongs to, so
# the pair reads as "control vs neural" rather than as two entries in the external/internal zoo.
_FAMILIARITY_CONTROL_COLOR = _DIM_SWEEP_ROLE_COLOR["external"]
_FAMILIARITY_NEURAL_COLOR = _DIM_SWEEP_ROLE_COLOR["neural"]

# y-axis label per value mode. "improvement" subtracts the control's own score session by
# session, matching the Delta R^2 convention of the other figure-2 performance panels: the
# control then sits flat at zero and the neural curve is the gap between them.
_FAMILIARITY_VALUE_MODES: dict[str, str] = {
    "absolute": r"R$^2$",
    "improvement": r"$\Delta$ R$^2$",
}


def _num_environments(session) -> int:
    """How many real environments a session ran, ignoring the invalid (negative) sentinel."""
    environments = np.asarray(session.environments)
    return int(np.unique(environments[environments >= 0]).size)


def _chronological_mouse_sessions(results: ResultsAggregator, mouse: str, single_env_only: bool = False) -> np.ndarray:
    """Session indices for one mouse, sorted chronologically by date.

    Parameters
    ----------
    results : ResultsAggregator
        Provides ``sessions`` and ``mouse_names``.
    mouse : str
        Mouse to select.
    single_env_only : bool
        Keep only sessions that ran exactly one environment (see :func:`_num_environments`).

    Returns
    -------
    np.ndarray
        Indices into ``results.sessions``, in date order.
    """
    idx_mouse = np.where(results.mouse_names == mouse)[0]
    if single_env_only:
        idx_mouse = idx_mouse[[_num_environments(results.sessions[i]) == 1 for i in idx_mouse]]
    dates = np.array([results.sessions[i].date for i in idx_mouse])
    return idx_mouse[np.argsort(dates)]


def _mouse_session_curves(results: ResultsAggregator, values: np.ndarray, single_env_only: bool) -> dict[str, np.ndarray]:
    """Per-mouse ``values`` in chronological session order, renumbered from 0.

    Filtered sessions are dropped rather than left as gaps, so index 0 is always the first
    *plotted* session of a mouse. With ``single_env_only`` that is the mouse's first session
    outright, since a second environment is only ever introduced later.

    Parameters
    ----------
    results : ResultsAggregator
        Provides ``sessions``, ``mouse_names`` and ``unique_mice``.
    values : np.ndarray
        One value per session, in ``results.sessions`` order, shape ``(n_sessions,)``.
    single_env_only : bool
        Keep only single-environment sessions.

    Returns
    -------
    dict[str, np.ndarray]
        ``{mouse: 1D array}``, ragged across mice. Mice left with no sessions are omitted.
    """
    curves: dict[str, np.ndarray] = {}
    for mouse in results.unique_mice:
        idx_sorted = _chronological_mouse_sessions(results, mouse, single_env_only=single_env_only)
        if idx_sorted.size:
            curves[mouse] = values[idx_sorted]
    return curves


def _familiarity_pad_stack(curves: dict[str, np.ndarray]) -> np.ndarray:
    """Stack ragged per-mouse 1D curves into a NaN-padded ``(n_mice, max_len)`` array."""
    max_len = max((len(v) for v in curves.values()), default=0)
    stack = np.full((len(curves), max_len), np.nan)
    for i, values in enumerate(curves.values()):
        stack[i, : len(values)] = values
    return stack


def _familiarity_supported_length(stack: np.ndarray, min_mice: int) -> int:
    """Columns up to the last one where at least ``min_mice`` mice have finite data."""
    support = np.sum(np.isfinite(stack), axis=0)
    valid = np.where(support >= min_mice)[0]
    return int(valid[-1]) + 1 if valid.size else 0


def _draw_familiarity_series(
    ax,
    curves: dict[str, np.ndarray],
    color: str,
    label: str,
    style: str,
    se: bool,
    min_mice: int,
    linewidth: float,
    subject_linewidth: float,
    subject_alpha: float,
    fill_alpha: float,
) -> tuple[int, np.ndarray]:
    """Draw one model's per-mouse session curves onto ``ax``.

    ``style == "all"`` draws every mouse thin plus a solid across-mouse mean; ``"errorPlot"``
    draws a mean +/- SE (or SD) band instead. Either way the mean is truncated where fewer than
    ``min_mice`` mice remain, since the tail of the x-axis is carried by whichever mouse ran
    longest.

    Returns
    -------
    tuple[int, np.ndarray]
        How many x positions carry a drawn artist (the mean's length, or the longest mouse when
        the individual traces are shown), and the values visible on the axis (for the caller's
        y autoscaling).
    """
    stack = _familiarity_pad_stack(curves)
    length = _familiarity_supported_length(stack, min_mice)
    if style == "all":
        for values in curves.values():
            ax.plot(np.arange(len(values)), values, color=(color, subject_alpha), linewidth=subject_linewidth)
        if length:
            ax.plot(np.arange(length), np.nanmean(stack[:, :length], axis=0), color=color, linewidth=linewidth, label=label)
        return stack.shape[1], stack
    if not length:
        return 0, stack
    supported = stack[:, :length]
    errorPlot(np.arange(length), supported, axis=0, se=se, ax=ax, color=color, linewidth=linewidth, alpha=fill_alpha, label=label)
    # Only the band is drawn, not the individual mice, so the autoscale should track its edges
    # rather than single-mouse outliers well outside it.
    num_valid = np.sum(np.isfinite(supported), axis=0)
    spread = np.nanstd(supported, axis=0) / (np.sqrt(num_valid) if se else 1.0)
    mean = np.nanmean(supported, axis=0)
    return length, np.concatenate([mean - spread, mean + spread])


class RegressionFamiliarityViewer(Viewer):
    """R^2 over sessions for a control model (black) and the neural RRR model (blue).

    The counterpart of figure3's familiarity panel for regression scores: each mouse's sessions
    are ordered chronologically and renumbered from 0, then the per-mouse curves are averaged
    across mice. A ``RegressionConfig`` score is a whole-session number, so the x-axis is the
    mouse's own session index -- there is nothing per-environment to align to.

    ``single_env_only`` restricts the panel to sessions that ran exactly one environment, which
    in practice is each mouse's run of early sessions before a second environment was
    introduced. ``value_mode`` switches between the models' absolute R^2 and their improvement
    over the control, which is the control's score subtracted session by session (so the
    control sits flat at zero and the neural curve reads as the gap between them), matching the
    Delta R^2 convention of the other figure-2 performance panels.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        figsize: tuple[float, float] = (3.5, 2.5),
    ):
        self.results = results
        self.figsize = figsize
        self._scores: dict[str, np.ndarray] = {}

        # --- data selection ---
        self.add_selection("spks_type", value="sigrebase", options=list(VALID_SPKS_TYPES))
        self.add_selection("activity_parameters_name", value="default", options=list(ACTIVITY_PARAMETERS_NAMES))
        self.add_selection("control_model", value=_FAMILIARITY_DEFAULT_CONTROL, options=list(_FAMILIARITY_CONTROL_OPTIONS))
        self.add_selection("value_mode", value="absolute", options=list(_FAMILIARITY_VALUE_MODES))
        self.add_boolean("single_env_only", value=False)

        # --- curve rendering ---
        self.add_selection("style", value="errorPlot", options=["errorPlot", "all"])
        self.add_boolean("se", value=True)
        self.add_integer("min_mice", value=2, min=1, max=max(len(results.unique_mice), 1))
        self.add_float("linewidth", value=1.5, min=0.25, max=5.0)
        self.add_float("subject_linewidth", value=0.5, min=0.0, max=3.0)
        self.add_float("subject_alpha", value=0.3, min=0.0, max=1.0)
        self.add_float("fill_alpha", value=0.2, min=0.0, max=1.0)

        # --- axes ---
        self.add_float("fontsize", value=9.0, min=4.0, max=24.0)
        self.add_boolean("auto_ylim", value=True)
        self.add_float_range("ylim", value=(0.0, 0.3), min=-1.0, max=2.0, step=0.01)
        self.add_float("ytick_max", value=0.3, min=0.01, max=2.0, step=0.01)
        self.add_boolean("show_zero_line", value=True)

        add_legend_widgets(self)

        for name in ("spks_type", "activity_parameters_name", "control_model"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select per-session R^2 for the control and neural models."""
        scores = _performance_scores(
            self.results,
            [state["control_model"], _FAMILIARITY_NEURAL_MODEL],
            "r2",
            state["spks_type"],
            state["activity_parameters_name"],
            avg_by_mouse=False,
        )
        self._scores = {"control": scores[0], "neural": scores[1]}

    def plot(self, state):
        fontsize = state["fontsize"]
        control = self._scores["control"]
        neural = self._scores["neural"]
        improvement = state["value_mode"] == "improvement"
        if improvement:
            neural = neural - control
            control = control - control
        series = [
            (short_model_name(state["control_model"]), _FAMILIARITY_CONTROL_COLOR, control),
            (short_model_name(_FAMILIARITY_NEURAL_MODEL), _FAMILIARITY_NEURAL_COLOR, neural),
        ]

        plt.close("all")
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, layout="constrained")

        if improvement and state["show_zero_line"]:
            ax.axhline(0.0, color="k", linewidth=0.5, linestyle="--")

        extents = []
        visible_values = []
        for label, color, values in series:
            extent, visible = _draw_familiarity_series(
                ax,
                _mouse_session_curves(self.results, values, state["single_env_only"]),
                color,
                label,
                state["style"],
                state["se"],
                state["min_mice"],
                state["linewidth"],
                state["subject_linewidth"],
                state["subject_alpha"],
                state["fill_alpha"],
            )
            extents.append(extent)
            visible_values.append(visible[np.isfinite(visible)])

        finite = np.concatenate(visible_values) if visible_values else np.array([])
        if state["auto_ylim"] and finite.size:
            low, high = min(float(finite.min()), 0.0), float(finite.max())
            pad = 0.05 * (high - low) if high > low else 0.05
            ylim = (low - pad, high + pad)
            ytick_max = round(high, 2)
        else:
            ylim = state["ylim"]
            ytick_max = state["ytick_max"]

        ax.set_ylim(ylim)
        format_spines(
            ax,
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[0, max(max(extents) - 1, 1)],
            ybounds=ylim,
            yticks=[0, ytick_max],
            tick_fontsize=fontsize,
        )
        ax.tick_params(axis="both", which="both", labelsize=fontsize)
        ax.set_xlabel(_FAMILIARITY_XLABEL, fontsize=fontsize)
        ax.set_ylabel(_FAMILIARITY_VALUE_MODES[state["value_mode"]], labelpad=-10, fontsize=fontsize)
        apply_legend(ax, state, fontsize, auto_loc="lower right")
        return fig


def regression_familiarity(
    results: ResultsAggregator,
    spks_type: SpksTypes = "sigrebase",
    activity_parameters_name: str = "default",
    control_model: ModelName = _FAMILIARITY_DEFAULT_CONTROL,
    value_mode: str = "absolute",
    single_env_only: bool = False,
    style: str = "errorPlot",
    se: bool = True,
    min_mice: int = 2,
    linewidth: float = 1.5,
    subject_linewidth: float = 0.5,
    subject_alpha: float = 0.3,
    fill_alpha: float = 0.2,
    auto_ylim: bool = True,
    ylim: tuple[float, float] = (0.0, 0.3),
    ytick_max: float = 0.3,
    show_zero_line: bool = True,
    legend_options: dict | None = None,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (3.5, 2.5),
    return_syd_viewer: bool = False,
):
    """
    Regression R^2 over sessions, for a control model and the neural RRR model.

    The regression counterpart of figure3's familiarity panel: aggregated ``RegressionConfig``
    results (one optimized-hyperparameter test score per model per session) are re-indexed onto
    each mouse's chronological session order, renumbered from 0, and averaged across mice. A
    score is a whole-session number, so the x-axis is the session index itself -- unlike the
    stimulus-space spectra there is no per-environment breakdown to align to.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionConfig`` results, with ``model_name`` as a param axis covering
        ``control_model`` and ``"rrr"``.
    spks_type, activity_parameters_name
        Selects which stored variation to read (see ``RegressionConfig``).
    control_model : ModelName
        The black curve: any swept model other than ``"rrr"``, which is always the blue curve.
    value_mode : {"absolute", "improvement"}
        ``"absolute"`` plots each model's R^2. ``"improvement"`` subtracts the control's score
        session by session, so the control sits flat at zero and the neural curve is the gap
        between them -- the Delta R^2 convention of the other figure-2 performance panels.
    single_env_only : bool
        Keep only sessions that ran exactly one environment (the invalid negative environment
        sentinel is not counted). In practice that is each mouse's run of early sessions, before
        a second environment was introduced, so the curves stay within one familiarity regime.
        Dropped sessions are removed rather than left as gaps, so index 0 is still a mouse's
        first plotted session.
    style : {"errorPlot", "all"}
        ``"errorPlot"`` draws a mean +/- SE (or SD) band per model; ``"all"`` draws every mouse
        as a thin line plus a solid across-mouse mean.
    se : bool
        Standard error (True) or standard deviation (False) shading, for ``style="errorPlot"``.
    min_mice : int
        Truncate each mean where fewer than this many mice still have data, so the tail isn't
        carried by whichever mouse ran longest.
    linewidth : float
        Line width of the across-mouse mean curves.
    subject_linewidth, subject_alpha : float
        Line width and opacity of the individual mouse traces (``style="all"`` only).
    fill_alpha : float
        Opacity of the +/- SE/SD band (``style="errorPlot"`` only).
    auto_ylim : bool
        Fit the y limits and the upper y tick to the drawn data, ignoring ``ylim`` and
        ``ytick_max``.
    ylim : tuple[float, float]
        Y limits when ``auto_ylim`` is False, spanned by the offset left spine.
    ytick_max : float
        Upper y tick when ``auto_ylim`` is False (the lower one is 0).
    show_zero_line : bool
        Draw the dashed y = 0 reference line. Only applies to ``value_mode="improvement"``.
    legend_options : dict or None
        Legend knobs forwarded to ``figure_scripts.legends`` (``{"loc": ..., "ncols": ...}``);
        ``{"loc": "none"}`` hides it.
    fontsize : float
        Font size for axis labels, tick labels, and the legend.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.

    Returns
    -------
    matplotlib.figure.Figure or RegressionFamiliarityViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    if value_mode not in _FAMILIARITY_VALUE_MODES:
        raise ValueError(f"value_mode must be one of {list(_FAMILIARITY_VALUE_MODES)}, got {value_mode!r}")
    if control_model == _FAMILIARITY_NEURAL_MODEL:
        raise ValueError(f"control_model must not be {_FAMILIARITY_NEURAL_MODEL!r} -- it is always the neural curve")

    viewer = RegressionFamiliarityViewer(results, figsize=figsize)

    # Seed the params the score selection depends on first, then refresh explicitly:
    # pre-deploy update_* may not fire on_change.
    viewer.update_selection("spks_type", value=spks_type)
    viewer.update_selection("activity_parameters_name", value=activity_parameters_name)
    viewer.update_selection("control_model", value=control_model)
    viewer.refresh_data(viewer.state)

    viewer.update_selection("value_mode", value=value_mode)
    viewer.update_boolean("single_env_only", value=single_env_only)

    viewer.update_selection("style", value=style)
    viewer.update_boolean("se", value=se)
    viewer.update_integer("min_mice", value=min_mice)
    viewer.update_float("linewidth", value=linewidth)
    viewer.update_float("subject_linewidth", value=subject_linewidth)
    viewer.update_float("subject_alpha", value=subject_alpha)
    viewer.update_float("fill_alpha", value=fill_alpha)

    viewer.update_float("fontsize", value=fontsize)
    viewer.update_boolean("auto_ylim", value=auto_ylim)
    viewer.update_float_range("ylim", value=tuple(ylim))
    viewer.update_float("ytick_max", value=ytick_max)
    viewer.update_boolean("show_zero_line", value=show_zero_line)
    update_legend_widgets(viewer, legend_options or {})

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ======================================================================================
# Model prediction versus held-out neural activity
# ======================================================================================


class RegressionPredictionViewer(Viewer):
    """Inspect held-out model predictions for one ROI or a quality-filtered population."""

    def __init__(
        self,
        results: ResultsAggregator,
        registry: PopulationRegistry,
        figsize: tuple[float, float] = (5.0, 5.0),
    ):
        self.results = results
        self.registry = registry
        self.figsize = figsize
        self._rows_by_mouse = {mouse: np.flatnonzero(results.mouse_names == mouse).tolist() for mouse in results.unique_mice}
        if not self._rows_by_mouse:
            raise ValueError("RegressionPredictionViewer requires at least one session.")
        self._session_rows: dict[str, int] = {}

        first_mouse = results.unique_mice[0]
        first_sessions = self._session_options(first_mouse)
        self.add_selection("mouse", value=first_mouse, options=list(results.unique_mice))
        self.add_selection("session", value=first_sessions[0], options=first_sessions)
        self.add_selection("model_name", value=KEY_FIGURE_MODELS[0], options=list(KEY_FIGURE_MODELS))
        self.add_selection("spks_type", value="sigrebase", options=["sigrebase", "oasis"])
        self.add_selection("activity_parameters_name", value="default", options=list(ACTIVITY_PARAMETERS_NAMES))
        self.add_selection("method", value="preferred", options=["preferred", "best"])

        self.add_selection("roi_mode", value="single", options=["single", "quality_filtered"])
        self.add_integer("roi", value=0, min=0, max=0)
        self.add_float_range("reliability_range", value=(-1.0, 1.0), min=-1.0, max=1.0, step=0.05)
        self.add_float_range("fraction_active_range", value=(0.0, 1.0), min=0.0, max=1.0, step=0.025)

        self.add_selection("plot_style", value="scatter", options=["scatter", "hexbin", "kde"])
        self.add_selection("axis_scale", value="linear", options=["linear", "log", "symlog"])
        self.add_float("alpha", value=0.08, min=0.01, max=1.0, step=0.01)
        self.add_float("markersize", value=5.0, min=0.1, max=100.0, step=0.5)
        self.add_integer("density_levels", value=12, min=3, max=50)
        self.add_integer("hex_gridsize", value=60, min=10, max=200)
        self.add_selection("density_norm", value="log", options=["linear", "log"])
        self.add_boolean("show_identity", value=True)
        self.add_boolean("show_title", value=True)

        self.on_change("mouse", self.update_sessions)
        for name in ("mouse", "session", "spks_type"):
            self.on_change(name, self.update_roi_bounds)
        self.update_sessions(self.state)
        self.update_roi_bounds(self.state)

    def _session_options(self, mouse: str) -> list[str]:
        labels = []
        self._session_rows = {}
        for row in self._rows_by_mouse[mouse]:
            label = self.results.sessions[row].session_print()
            labels.append(label)
            self._session_rows[label] = row
        return labels

    def update_sessions(self, state):
        """Update session choices when the selected mouse changes."""
        labels = self._session_options(state["mouse"])
        current = state.get("session")
        self.update_selection("session", value=current if current in labels else labels[0], options=labels)

    def _session(self, state) -> B2Session:
        return self.results.sessions[self._session_rows[state["session"]]]

    def update_roi_bounds(self, state):
        """Keep the ROI index within the target split of the current session."""
        session = self._session(state)
        population = self.registry.get_population(session, state["spks_type"])[0]
        self.update_integer("roi", max=max(len(population.cell_split_indices[1]) - 1, 0))

    def _selected_values(self, state) -> tuple[np.ndarray, np.ndarray, int, int]:
        session = self._session(state)
        target, prediction = get_model_predictions(
            state["model_name"],
            session,
            self.registry,
            state["spks_type"],
            activity_parameters_name=state["activity_parameters_name"],
            method=state["method"],
        )
        if state["roi_mode"] == "single":
            roi_idx = np.array([state["roi"]], dtype=int)
        else:
            reliability, fraction_active = _target_prediction_quality(
                session,
                self.registry,
                state["spks_type"],
                state["activity_parameters_name"],
            )
            rel_low, rel_high = state["reliability_range"]
            fa_low, fa_high = state["fraction_active_range"]
            roi_idx = np.flatnonzero(
                np.isfinite(reliability)
                & np.isfinite(fraction_active)
                & (reliability >= rel_low)
                & (reliability <= rel_high)
                & (fraction_active >= fa_low)
                & (fraction_active <= fa_high)
            )

        data = target[roi_idx].ravel()
        pred = prediction[roi_idx].ravel()
        keep = np.isfinite(data) & np.isfinite(pred)
        if state["axis_scale"] == "log":
            keep &= (data > 0) & (pred > 0)
        return data[keep], pred[keep], len(roi_idx), int(np.sum(keep))

    @staticmethod
    def _equal_limits(data: np.ndarray, prediction: np.ndarray, scale: str) -> tuple[float, float]:
        values = np.concatenate((data, prediction))
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        if low == high:
            pad = max(abs(low) * 0.05, 1e-3)
            return low - pad, high + pad
        if scale == "log":
            return low / 1.08, high * 1.08
        pad = 0.04 * (high - low)
        return low - pad, high + pad

    def plot(self, state):
        plt.close("all")
        fig, ax = plt.subplots(figsize=self.figsize, layout="constrained")
        data, prediction, num_rois, num_points = self._selected_values(state)

        if num_points:
            style = state["plot_style"]
            if style == "scatter":
                ax.scatter(data, prediction, s=state["markersize"], alpha=state["alpha"], edgecolors="none")
            elif style == "hexbin":
                norm = LogNorm() if state["density_norm"] == "log" else None
                ax.hexbin(data, prediction, gridsize=state["hex_gridsize"], mincnt=1, norm=norm, cmap="viridis")
            elif style == "kde":
                if num_points > 2:
                    sns.kdeplot(
                        x=data,
                        y=prediction,
                        ax=ax,
                        fill=True,
                        levels=state["density_levels"],
                        thresh=0.01,
                        cmap="viridis",
                        log_scale=state["axis_scale"] == "log",
                    )
                else:
                    ax.scatter(data, prediction, s=state["markersize"], alpha=state["alpha"])
            else:
                raise ValueError(f"Unknown plot_style {style!r}")

            limits = self._equal_limits(data, prediction, state["axis_scale"])
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            if state["show_identity"]:
                ax.plot(limits, limits, color="0.25", linestyle="--", linewidth=1.0, zorder=10)
        else:
            ax.text(0.5, 0.5, "No finite samples in selection", ha="center", va="center", transform=ax.transAxes)

        scale = state["axis_scale"]
        if scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")
        elif scale == "symlog":
            ax.set_xscale("symlog", linthresh=0.01)
            ax.set_yscale("symlog", linthresh=0.01)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Held-out neural activity")
        ax.set_ylabel("Model prediction")
        if state["show_title"]:
            ax.set_title(f"{state['model_name']} · {num_rois} ROI(s), {num_points:,} samples")
        format_spines(ax, x_pos=-0.03, y_pos=-0.03)
        return fig


def regression_prediction_vs_data(
    results: ResultsAggregator,
    registry: PopulationRegistry,
    mouse: str | None = None,
    session: str | None = None,
    roi: int = 0,
    model_name: ModelName = "external_placefield_1d",
    spks_type: SpksTypes = "sigrebase",
    activity_parameters_name: str = "default",
    method: str = "preferred",
    roi_mode: str = "single",
    reliability_range: tuple[float, float] = (-1.0, 1.0),
    fraction_active_range: tuple[float, float] = (0.0, 1.0),
    plot_style: str = "scatter",
    axis_scale: str = "linear",
    alpha: float = 0.08,
    markersize: float = 5.0,
    density_levels: int = 12,
    hex_gridsize: int = 60,
    density_norm: str = "log",
    show_identity: bool = True,
    show_title: bool = True,
    figsize: tuple[float, float] = (5.0, 5.0),
    return_syd_viewer: bool = False,
):
    """Plot a model prediction against held-out data for one ROI or a quality-selected pool."""
    viewer = RegressionPredictionViewer(results, registry, figsize=figsize)
    if mouse is not None:
        viewer.update_selection("mouse", value=mouse)
        viewer.update_sessions(viewer.state)
    if session is not None:
        viewer.update_selection("session", value=session)
    viewer.update_selection("spks_type", value=spks_type)
    viewer.update_roi_bounds(viewer.state)
    viewer.update_integer("roi", value=roi)
    viewer.update_selection("model_name", value=model_name)
    viewer.update_selection("activity_parameters_name", value=activity_parameters_name)
    viewer.update_selection("method", value=method)
    viewer.update_selection("roi_mode", value=roi_mode)
    viewer.update_float_range("reliability_range", value=tuple(reliability_range))
    viewer.update_float_range("fraction_active_range", value=tuple(fraction_active_range))
    viewer.update_selection("plot_style", value=plot_style)
    viewer.update_selection("axis_scale", value=axis_scale)
    viewer.update_float("alpha", value=alpha)
    viewer.update_float("markersize", value=markersize)
    viewer.update_integer("density_levels", value=density_levels)
    viewer.update_integer("hex_gridsize", value=hex_gridsize)
    viewer.update_selection("density_norm", value=density_norm)
    viewer.update_boolean("show_identity", value=show_identity)
    viewer.update_boolean("show_title", value=show_title)
    if return_syd_viewer:
        return viewer
    fig = viewer.plot(viewer.state)
    plt.show()
    return fig
