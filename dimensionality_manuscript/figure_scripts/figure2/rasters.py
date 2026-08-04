"""Prediction rasters: many models stacked on one session, and the two-model figure panel."""

import numpy as np
from matplotlib import pyplot as plt
from rastermap import Rastermap

from vrAnalysis.helpers.plotting import format_spines
from vrAnalysis.sessions import B2Session, SpksTypes
from dimensionality_manuscript.configs.rrr_to_external_latents import VALID_SPKS_TYPES
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import ACTIVITY_PARAMETERS_NAMES, ModelName, PopulationRegistry, short_model_name
from dimensionality_manuscript.figure_scripts.panels import FigureViewer, add_data_selection_widgets

from ._predictions import (
    SAMPLE_FIT_LABELS,
    SAMPLE_FIT_METRICS,
    get_model_predictions,
    per_sample_fit,
    target_environment_sort,
    transform_raster_rows,
)
from .zoo import ModelZooCondensed, ModelZooSchematicConfig

SORT_METHODS = ["environment", "rastermap", "activity"]

_RASTER_HEIGHT = 6.0
_ERROR_TRACE_HEIGHT = 3.0

#: Row key for the shared target-activity raster, which has no model name of its own.
_TARGET = "__target__"


def compute_row_sort(method: str, target: np.ndarray, session: B2Session, registry: PopulationRegistry, spks_type: SpksTypes) -> np.ndarray:
    """Row ordering for a target raster under one of :data:`SORT_METHODS`."""
    if method == "rastermap":
        return Rastermap().fit(target).isort
    if method == "activity":
        return np.argsort(-np.nansum(target, axis=1))
    if method == "environment":
        return target_environment_sort(session, registry, spks_type)
    raise ValueError(f"Unknown sort_method {method!r}. Options: {SORT_METHODS}")


def _hide_raster_frame(ax, title: str | None, fontsize: float) -> None:
    """Strip ticks and spines from a raster axis, with an optional top-right title."""
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.text(1.0, 1.0, title, transform=ax.transAxes, ha="right", va="top", color="black", fontsize=fontsize)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_time_scale_bar(ax, seconds: float, frame_duration: float, num_shown_frames: int, fontsize: float, y: float = 0.1) -> None:
    """Draw a horizontal time scale bar in the axis's lower right, in axes coordinates."""
    if seconds <= 0:
        return
    # The rasters' x unit is frames, so the bar's length is its duration over the frame period,
    # as a fraction of the displayed window.
    bar_fraction = (seconds / frame_duration) / num_shown_frames
    ax.plot(
        [1.0 - bar_fraction, 1.0],
        [y, y],
        transform=ax.transAxes,
        color="black",
        linewidth=1.5,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        1.0 - bar_fraction / 2,
        y + 0.025,
        f"{seconds:g}s",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        clip_on=False,
    )


def _draw_joint_scale_bar(
    ax,
    *,
    seconds: float,
    num_rois: int,
    frame_duration: float,
    num_shown_frames: int,
    total_rois: int,
    x: float,
    y: float,
    fontsize: float,
) -> None:
    """Draw connected horizontal-time and vertical-ROI scale bars in axes coordinates."""
    if seconds <= 0 or num_rois <= 0 or total_rois <= 0:
        return
    time_fraction = (seconds / frame_duration) / num_shown_frames
    roi_fraction = min(num_rois, total_rois) / total_rois
    ax.plot(
        [x, x, x + time_fraction],
        [y + roi_fraction, y, y],
        transform=ax.transAxes,
        color="black",
        linewidth=1.5,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        x + time_fraction / 2,
        y + 0.025,
        f"{seconds:g}s",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        clip_on=False,
    )
    ax.text(
        x - 0.015,
        y,  # + roi_fraction / 2,
        f"{num_rois:g} ROIs",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        rotation=90,
        fontsize=fontsize,
        clip_on=False,
    )


_PREDICTION_COLOR_INSET_RECT = (0.72, 0.10, 0.25, 0.20)


def _colorscale_inset_rect(state) -> list[float]:
    return [
        state["colorscale_inset_x"],
        state["colorscale_inset_y"],
        state["colorscale_inset_width"],
        state["colorscale_inset_height"],
    ]


def _draw_colorscale_inset(
    ax,
    cmap_name: str,
    *,
    left_label: str | None,
    right_label: str,
    left_color: str,
    right_color: str,
    fontsize: float,
    rect,
) -> None:
    """Draw a labeled horizontal colormap strip inside ``ax``."""
    colors = plt.get_cmap(cmap_name)(np.linspace(0, 1, 255))[None, :, :]
    inset = ax.inset_axes(rect)
    inset.imshow(colors, aspect="auto")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.text(0.97, 0.5, right_label, transform=inset.transAxes, ha="right", va="center", color=right_color, fontsize=fontsize)
    if left_label is not None:
        inset.text(0.03, 0.5, left_label, transform=inset.transAxes, ha="left", va="center", color=left_color, fontsize=fontsize)


class ModelRasterFocus(FigureViewer):
    """Interactive stacked raster of regression-model predictions for one session.

    One shared target-activity raster (optional) sits atop one prediction raster per
    model. An optional second column shows residuals (target - prediction) with a bwr
    colormap, and an optional single panel under all rasters superimposes the per-frame
    fit of every model as colored lines with a legend.

    Each model is scored with its own optimized hyperparameters (from cache). The target
    activity is shared across models, so a single data raster and a single row ordering
    apply to every panel.

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
    xslice : slice or None
        Initial frame window. Defaults to every frame.
    vmax : float
        Upper limit of the raster color scale.
    scale_bar_seconds : float
        Duration of the time scale bar drawn on the bottom raster (0 hides it).
    figsize : tuple[float, float] or None
        Sized from the layout when None.
    fontsize : float
        Font size for panel titles, axis labels, tick labels, and the fit-curve legend.
    """

    def __init__(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        model_names: list[ModelName],
        *,
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
    ):
        if not model_names:
            raise ValueError("model_names must contain at least one model")
        if sample_fit_metric not in SAMPLE_FIT_METRICS:
            raise ValueError(f"sample_fit_metric must be one of {SAMPLE_FIT_METRICS}, got {sample_fit_metric!r}")
        if sort_method not in SORT_METHODS:
            raise ValueError(f"sort_method must be one of {SORT_METHODS}, got {sort_method!r}")

        self.session = session
        self.registry = registry or PopulationRegistry()
        self._sort_cache: dict[tuple, np.ndarray] = {}
        self._display_cache: dict[tuple, np.ndarray] = {}

        # Frames are the x-axis unit of every raster, so the scale bar's length in frames
        # is its requested duration divided by the imaging frame period.
        mpci_times = self.session.loadone("mpci.times")
        self.frame_duration = float(np.median(np.diff(mpci_times)))

        # Fetch one cached grid to establish frame bounds. Remaining models are loaded
        # lazily by plot(), and get_model_predictions caches each expensive fit separately.
        target, _ = get_model_predictions(
            model_names[0],
            self.session,
            self.registry,
            spks_type,
            activity_parameters_name=activity_parameters_name,
            method=method,
            train_split=train_split,
            test_split=test_split,
        )
        num_frames = target.shape[1]
        xstart = 0 if xslice is None or xslice.start is None else xslice.start
        xstop = num_frames if xslice is None or xslice.stop is None else xslice.stop
        if figsize is None:
            width = 16 if include_error_column else 10
            height = 2.2 * len(model_names) + (1.5 if include_sample_fit_curve else 0)
            if include_data_raster:
                height += 2.2
            figsize = (width, max(height, 3.0))

        split_options = list(dict.fromkeys([train_split, test_split, "train", "test"]))
        self.add_selection("spks_type", value=spks_type, options=list(VALID_SPKS_TYPES))
        self.add_multiple_selection("model_names", value=list(model_names), options=list(model_names))
        self.add_selection(
            "activity_parameters_name",
            value=activity_parameters_name,
            options=list(dict.fromkeys([activity_parameters_name, *ACTIVITY_PARAMETERS_NAMES])),
        )
        self.add_selection("method", value=method, options=["preferred", "best", "grid", "optuna", "golden"])
        self.add_selection("train_split", value=train_split, options=split_options)
        self.add_selection("test_split", value=test_split, options=split_options)
        self.add_boolean("include_data_raster", value=include_data_raster)
        self.add_boolean("include_error_column", value=include_error_column)
        self.add_boolean("include_sample_fit_curve", value=include_sample_fit_curve)
        self.add_boolean("zscore", value=zscore)
        self.add_boolean("subtract_median", value=subtract_median)
        self.add_selection("sort_method", value=sort_method, options=SORT_METHODS)
        self.add_selection("sample_fit_metric", value=sample_fit_metric, options=SAMPLE_FIT_METRICS)
        self.add_float("vmax", value=vmax, min=0.01, max=20.0)
        self.add_float("scale_bar_seconds", value=scale_bar_seconds, min=0.0, max=120.0)
        self.add_integer("xslice_start", value=xstart, min=0, max=num_frames - 1)
        self.add_integer("xslice_stop", value=xstop, min=1, max=num_frames)
        self.add_float("figsize_width", value=figsize[0], min=1.0, max=30.0)
        self.add_float("figsize_height", value=figsize[1], min=1.0, max=40.0)
        self.add_float("fontsize", value=fontsize, min=1.0, max=30.0)
        for name in (
            "spks_type",
            "model_names",
            "activity_parameters_name",
            "method",
            "train_split",
            "test_split",
        ):
            self.on_change(name, self._update_frame_bounds)

    @staticmethod
    def _data_key(state) -> tuple:
        return (
            state["spks_type"],
            state["activity_parameters_name"],
            state["method"],
            state["train_split"],
            state["test_split"],
        )

    def _get_data(self, state) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load selected grids through the per-model prediction cache."""
        target = None
        predictions = {}
        for model_name in state["model_names"]:
            model_target, prediction = get_model_predictions(
                model_name,
                self.session,
                self.registry,
                state["spks_type"],
                activity_parameters_name=state["activity_parameters_name"],
                method=state["method"],
                train_split=state["train_split"],
                test_split=state["test_split"],
            )
            if target is None:
                target = model_target
            elif target.shape != model_target.shape:
                raise ValueError("Selected models returned incompatible target raster shapes")
            predictions[model_name] = prediction
        if target is None:
            raise ValueError("Select at least one model")
        return target, predictions

    def _update_frame_bounds(self, state) -> None:
        """Refresh slice limits when a data-selection knob changes."""
        if not state["model_names"]:
            return
        target, _ = get_model_predictions(
            state["model_names"][0],
            self.session,
            self.registry,
            state["spks_type"],
            activity_parameters_name=state["activity_parameters_name"],
            method=state["method"],
            train_split=state["train_split"],
            test_split=state["test_split"],
        )
        num_frames = target.shape[1]
        slice_start = min(state["xslice_start"], max(num_frames - 1, 0))
        self.update_integer(
            "xslice_start",
            value=slice_start,
            max=max(num_frames - 1, 0),
        )
        self.update_integer(
            "xslice_stop",
            value=max(slice_start + 1, min(state["xslice_stop"], num_frames)),
            max=num_frames,
        )

    def _get_sort(self, state, target: np.ndarray) -> np.ndarray:
        """Return (and cache) the target-row ordering for a sort method.

        The data key already identifies which target raster is in play, so it alone (plus the
        method) keys the cache -- the array's identity adds nothing.
        """
        method = state["sort_method"]
        key = (*self._data_key(state), method)
        if key not in self._sort_cache:
            self._sort_cache[key] = compute_row_sort(method, target, self.session, self.registry, state["spks_type"])
        return self._sort_cache[key]

    def _display(self, name: str, raw: np.ndarray, state) -> np.ndarray:
        """Cache all-frame display transforms independently of layout/x-slice knobs."""
        key = (*self._data_key(state), name, state["zscore"], state["subtract_median"])
        if key not in self._display_cache:
            self._display_cache[key] = transform_raster_rows(
                raw,
                zscore=state["zscore"],
                subtract_median=state["subtract_median"],
            )
        return self._display_cache[key]

    def plot(self, state):
        target, predictions = self._get_data(state)
        model_names = list(state["model_names"])
        include_error = state["include_error_column"]
        metric = state["sample_fit_metric"]
        fontsize = state["fontsize"]
        vmax = state["vmax"]
        scale_bar_seconds = state["scale_bar_seconds"]
        xslice = slice(state["xslice_start"], state["xslice_stop"])
        idx_sort = self._get_sort(state, target)

        def disp(name: str) -> np.ndarray:
            raw = target if name == _TARGET else predictions[name]
            return self._display(name, raw, state)

        # Build the row layout: an optional shared data raster, one prediction raster per
        # model, then an optional single error-trace panel spanning below all rasters.
        rows: list[tuple[str, str | None]] = []
        if state["include_data_raster"]:
            rows.append(("data", None))
        for model_name in model_names:
            rows.append(("raster", model_name))
        if state["include_sample_fit_curve"]:
            rows.append(("error_trace", None))

        height_ratios = [_ERROR_TRACE_HEIGHT if kind == "error_trace" else _RASTER_HEIGHT for kind, _ in rows]
        ncols = 2 if include_error else 1

        fig = self.new_figure(figsize=(state["figsize_width"], state["figsize_height"]), layout="constrained")
        gs = fig.add_gridspec(len(rows), ncols, height_ratios=height_ratios)

        raster_kwargs = dict(aspect="auto", cmap="gray_r", vmin=0, vmax=vmax)
        error_kwargs = dict(aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)

        cmap_models = plt.get_cmap("tab10")
        model_colors = {name: cmap_models(i % 10) for i, name in enumerate(model_names)}

        last_raster_row = max(i for i, (kind, _) in enumerate(rows) if kind in ("data", "raster"))
        xvals = np.arange(xslice.start, xslice.stop)
        num_shown_frames = xslice.stop - xslice.start

        inset_built = False
        for irow, (kind, model_name) in enumerate(rows):
            if kind == "error_trace":
                ax = fig.add_subplot(gs[irow, :])
                fit_min, fit_max = np.inf, -np.inf
                for name in model_names:
                    fit = per_sample_fit(predictions[name][:, xslice], target[:, xslice], metric)
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
                    tick_fontsize=fontsize,
                )
                ax.set_ylabel(SAMPLE_FIT_LABELS[metric], fontsize=fontsize)
                ax.legend(loc="upper right", fontsize=fontsize, ncol=max(1, len(model_names) // 2), frameon=False)
                continue

            if kind == "data":
                data = disp(_TARGET)[:, xslice][idx_sort]
                title = "Data"
            else:
                data = disp(model_name)[:, xslice][idx_sort]
                title = short_model_name(model_name)

            ax = fig.add_subplot(gs[irow, 0])
            ax.imshow(data, **raster_kwargs)
            _hide_raster_frame(ax, title, fontsize)
            ax.set_ylabel("ROIs", fontsize=fontsize)
            if irow == last_raster_row:
                _draw_time_scale_bar(ax, scale_bar_seconds, self.frame_duration, num_shown_frames, fontsize)

            if not inset_built:
                gray_values = np.linspace(0, vmax, 255) / vmax if vmax > 0 else np.zeros(255)
                colors = plt.get_cmap("gray_r")(gray_values)
                inset = ax.inset_axes([0.2, 0.05, 0.6, 0.1])
                inset.imshow(colors[np.newaxis, :, :], aspect="auto")
                inset.set_xticks([])
                inset.set_yticks([])
                inset.text(-0.03, 0.5, "0", transform=inset.transAxes, ha="right", va="center", clip_on=False, fontsize=fontsize)
                inset.text(1.03, 0.5, f"{vmax}", transform=inset.transAxes, ha="left", va="center", clip_on=False, fontsize=fontsize)
                inset.text(0.5, 1.25, "zscore activity", transform=inset.transAxes, ha="center", va="bottom", clip_on=False, fontsize=fontsize)
                inset_built = True

            if include_error and kind == "raster":
                residual = (disp(_TARGET)[:, xslice] - disp(model_name)[:, xslice])[idx_sort]
                ax_err = fig.add_subplot(gs[irow, 1])
                ax_err.imshow(residual, **error_kwargs)
                _hide_raster_frame(ax_err, "Residual", fontsize)

        return fig


# ======================================================================================
# Data, internal place-field, and RRR prediction rasters
# ======================================================================================

PREDICTION_FIGURE_MODELS: tuple[ModelName, ModelName] = (
    "internal_placefield_1d_gain",
    "rrr",
)


class ModelPredictionsFigureViewer(FigureViewer):
    """Two-row comparison of held-out data with place-field and RRR predictions.

    The top row contains data and the two fixed model predictions. The bottom row contains the
    three-model schematic followed by the corresponding residuals (data - prediction).
    ``model_name`` is therefore consumed as a fixed comparison axis; the other axes of a
    ``RegressionConfig`` aggregator remain available as Syd selections.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionConfig`` results. Supplies the available mice and sessions, and
        the data-selection options (every param axis except ``model_name``).
    registry : PopulationRegistry or None
        Population registry. A default one is created when None.
    mouse, session : str or None
        Initial selection. Each falls back to the first available option.
    sort_method : {"environment", "rastermap", "activity"}
        Row ordering shared across every panel.
    zscore, subtract_median : bool
        Per-ROI display transforms, as in :class:`ModelRasterFocus`.
    xslice : slice or None
        Initial frame window. Defaults to every frame.
    vmax : float
        Upper limit of the raster color scale (the residuals use +/- this on a bwr map).
    scale_bar_seconds : float
        Duration of the horizontal time scale bar on the data raster (0 hides the joint bar).
    scale_bar_x, scale_bar_y : float
        Axes-fraction coordinates of the joint scale bar's bottom-left corner.
    scale_bar_num_rois : int
        Number of ROIs represented by the joint scale bar's vertical arm.
    figsize : tuple[float, float]
        Figure size in inches.
    fontsize : float
        Font size for the panel titles, axis labels, and scale-bar label.
    zoo_config : ModelZooSchematicConfig or None
        Initial style and layout for the embedded model-zoo schematic. All condensed-zoo
        geometry and typography fields are also exposed as controls on this viewer.
    colorscale_inset_rect : tuple[float, float, float, float]
        Shared ``(x, y, width, height)`` for the prediction and residual colormap insets,
        expressed in their parent axes' coordinates.
    **selection_defaults
        Starting values for the data-selection widgets, overriding the config's own. The widgets
        are built from the aggregator's param axes (minus ``model_name``, fixed by the panel);
        unswept ones are pinned to the value its config fixes.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        registry: PopulationRegistry | None = None,
        *,
        mouse: str | None = None,
        session: str | None = None,
        sort_method: str = "environment",
        zscore: bool = False,
        subtract_median: bool = False,
        xslice: slice | None = None,
        vmax: float = 6,
        scale_bar_seconds: float = 10.0,
        scale_bar_x: float = 0.05,
        scale_bar_y: float = 0.08,
        scale_bar_num_rois: int = 50,
        figsize: tuple[float, float] = (15.0, 7.0),
        fontsize: float = 7,
        zoo_config: ModelZooSchematicConfig | None = None,
        colorscale_inset_rect: tuple[float, float, float, float] = _PREDICTION_COLOR_INSET_RECT,
        **selection_defaults,
    ):
        if not results.sessions:
            raise ValueError("ModelPredictionsFigureViewer requires at least one session")
        if sort_method not in SORT_METHODS:
            raise ValueError(f"sort_method must be one of {SORT_METHODS}, got {sort_method!r}")

        # The two compared models are fixed by the panel design, so they are checked here rather
        # than offered as a widget (syd validates the selectable axes against their own options).
        model_options = results.param_axes.get("model_name", [])
        missing_models = [name for name in PREDICTION_FIGURE_MODELS if name not in model_options]
        if model_options and missing_models:
            raise ValueError("Regression results are missing required model_name values: " + ", ".join(missing_models))

        self.results = results
        self.registry = registry or PopulationRegistry()
        self._sort_cache: dict[tuple, np.ndarray] = {}
        self._display_cache: dict[tuple, np.ndarray] = {}

        mouse_names = np.asarray(results.mouse_names)
        self._rows_by_mouse = {name: np.flatnonzero(mouse_names == name).tolist() for name in results.unique_mice}
        self._session_rows = {name: {results.sessions[row].session_print(): row for row in rows} for name, rows in self._rows_by_mouse.items()}
        initial_mouse = mouse if mouse in self._session_rows else results.unique_mice[0]
        session_options = list(self._session_rows[initial_mouse])
        initial_session = session if session in session_options else session_options[0]

        self.add_selection("mouse", value=initial_mouse, options=list(self._session_rows))
        self.add_selection("session", value=initial_session, options=session_options)
        # These two go to get_model_predictions, not to results.sel, so the panel needs a value
        # for each even when the config no longer sweeps it.
        self.selection_names = add_data_selection_widgets(
            self,
            results,
            skip=("model_name",),
            defaults=selection_defaults,
            require=("spks_type", "activity_parameters_name"),
        )

        selected_session = results.sessions[self._session_rows[initial_mouse][initial_session]]
        target, _ = get_model_predictions(
            PREDICTION_FIGURE_MODELS[0],
            selected_session,
            self.registry,
            self.state["spks_type"],
            activity_parameters_name=self.state["activity_parameters_name"],
        )
        num_frames = target.shape[1]
        xstart = 0 if xslice is None or xslice.start is None else xslice.start
        xstop = num_frames if xslice is None or xslice.stop is None else xslice.stop
        xstart = min(max(int(xstart), 0), max(num_frames - 1, 0))
        xstop = min(max(int(xstop), xstart + 1), num_frames)

        self.add_selection("sort_method", value=sort_method, options=SORT_METHODS)
        self.add_boolean("zscore", value=zscore)
        self.add_boolean("subtract_median", value=subtract_median)
        self.add_float("vmax", value=vmax, min=0.01, max=20.0)
        self.add_float("scale_bar_seconds", value=scale_bar_seconds, min=0.0, max=120.0)
        self.add_float("scale_bar_x", value=scale_bar_x, min=0.0, max=1.0, step=0.001)
        self.add_float("scale_bar_y", value=scale_bar_y, min=0.0, max=1.0, step=0.001)
        self.add_integer(
            "scale_bar_num_rois",
            value=min(max(int(scale_bar_num_rois), 1), target.shape[0]),
            min=1,
            max=target.shape[0],
        )
        self.add_integer("xslice_start", value=xstart, min=0, max=max(num_frames - 1, 0))
        self.add_integer("xslice_stop", value=xstop, min=1, max=num_frames)
        self.add_float("figsize_width", value=figsize[0], min=1.0, max=30.0)
        self.add_float("figsize_height", value=figsize[1], min=1.0, max=30.0)
        self.add_float("fontsize", value=fontsize, min=1.0, max=30.0)
        self.add_float("colorscale_inset_x", value=colorscale_inset_rect[0], min=0.0, max=1.0, step=0.001)
        self.add_float("colorscale_inset_y", value=colorscale_inset_rect[1], min=0.0, max=1.0, step=0.001)
        self.add_float("colorscale_inset_width", value=colorscale_inset_rect[2], min=0.001, max=1.0, step=0.001)
        self.add_float("colorscale_inset_height", value=colorscale_inset_rect[3], min=0.001, max=1.0, step=0.001)
        self.zoo_config = ModelZooCondensed.add_controls(self, zoo_config)

        self.on_change("mouse", self._update_sessions)
        for name in ("session", *self.selection_names):
            self.on_change(name, self._update_frame_bounds)

    def _session(self, state) -> B2Session:
        return self.results.sessions[self._session_rows[state["mouse"]][state["session"]]]

    def _update_sessions(self, state) -> None:
        options = list(self._session_rows[state["mouse"]])
        selected = state["session"] if state["session"] in options else options[0]
        self.update_selection("session", value=selected, options=options)
        self._update_frame_bounds({**state, "session": selected})

    def _update_frame_bounds(self, state) -> None:
        target, _ = get_model_predictions(
            PREDICTION_FIGURE_MODELS[0],
            self._session(state),
            self.registry,
            state["spks_type"],
            activity_parameters_name=state["activity_parameters_name"],
        )
        num_frames = target.shape[1]
        start = min(state["xslice_start"], max(num_frames - 1, 0))
        self.update_integer(
            "scale_bar_num_rois",
            value=min(state["scale_bar_num_rois"], target.shape[0]),
            max=target.shape[0],
        )
        self.update_integer("xslice_start", value=start, max=max(num_frames - 1, 0))
        self.update_integer(
            "xslice_stop",
            value=max(start + 1, min(state["xslice_stop"], num_frames)),
            max=num_frames,
        )

    def _get_data(self, state) -> tuple[np.ndarray, dict[ModelName, np.ndarray]]:
        target = None
        predictions = {}
        for model_name in PREDICTION_FIGURE_MODELS:
            model_target, prediction = get_model_predictions(
                model_name,
                self._session(state),
                self.registry,
                state["spks_type"],
                activity_parameters_name=state["activity_parameters_name"],
            )
            if target is None:
                target = model_target
            elif target.shape != model_target.shape:
                raise ValueError("Prediction models returned incompatible target raster shapes")
            predictions[model_name] = prediction
        return target, predictions

    def _data_key(self, state) -> tuple:
        return (self._session(state).session_name, state["spks_type"], state["activity_parameters_name"])

    def _get_sort(self, state, target: np.ndarray) -> np.ndarray:
        method = state["sort_method"]
        key = (*self._data_key(state), method)
        if key not in self._sort_cache:
            self._sort_cache[key] = compute_row_sort(method, target, self._session(state), self.registry, state["spks_type"])
        return self._sort_cache[key]

    def _display(self, name: str, raw: np.ndarray, state) -> np.ndarray:
        key = (*self._data_key(state), name, state["zscore"], state["subtract_median"])
        if key not in self._display_cache:
            self._display_cache[key] = transform_raster_rows(raw, zscore=state["zscore"], subtract_median=state["subtract_median"])
        return self._display_cache[key]

    def plot(self, state):
        target, predictions = self._get_data(state)
        idx_sort = self._get_sort(state, target)
        xslice = slice(state["xslice_start"], state["xslice_stop"])
        vmax = state["vmax"]
        fontsize = state["fontsize"]

        data_display = self._display("data", target, state)
        prediction_display = {name: self._display(name, prediction, state) for name, prediction in predictions.items()}

        fig, ax = self.new_subplots(2, 3, figsize=(state["figsize_width"], state["figsize_height"]), layout="constrained")
        raster_kwargs = dict(aspect="auto", cmap="gray_r", vmin=0, vmax=vmax)
        residual_kwargs = dict(aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)

        top_data = [data_display, *(prediction_display[name] for name in PREDICTION_FIGURE_MODELS)]
        top_titles = ["Target Data", "PF+Gain", "Peer Prediction"]
        for axis, raster, title in zip(ax[0], top_data, top_titles):
            axis.imshow(raster[:, xslice][idx_sort], **raster_kwargs)
            _hide_raster_frame(axis, title, fontsize)
        ax[0, 1].set_ylabel("Prediction", fontsize=fontsize)

        zoo_config = ModelZooCondensed.config_from_state(self.zoo_config, state)
        ModelZooCondensed.draw(ax[1, 0], zoo_config)
        ax[1, 1].set_ylabel("Residual", fontsize=fontsize)
        for column, model_name in enumerate(PREDICTION_FIGURE_MODELS, start=1):
            residual = (data_display[:, xslice] - prediction_display[model_name][:, xslice])[idx_sort]
            ax[1, column].imshow(residual, **residual_kwargs)
            _hide_raster_frame(ax[1, column], None, fontsize)

        inset_rect = _colorscale_inset_rect(state)
        _draw_colorscale_inset(
            ax[0, 1],
            "gray_r",
            left_label=None,
            right_label=rf"${vmax:g}\,\sigma$",
            left_color="black",
            right_color="white",
            fontsize=fontsize,
            rect=inset_rect,
        )
        _draw_colorscale_inset(
            ax[1, 1],
            "bwr",
            left_label=rf"$-{vmax:g}\,\sigma$",
            right_label=rf"$+{vmax:g}\,\sigma$",
            left_color="white",
            right_color="white",
            fontsize=fontsize,
            rect=inset_rect,
        )

        mpci_times = self._session(state).loadone("mpci.times")
        _draw_joint_scale_bar(
            ax[0, 0],
            seconds=state["scale_bar_seconds"],
            num_rois=state["scale_bar_num_rois"],
            frame_duration=float(np.median(np.diff(mpci_times))),
            num_shown_frames=xslice.stop - xslice.start,
            total_rois=target.shape[0],
            x=state["scale_bar_x"],
            y=state["scale_bar_y"],
            fontsize=fontsize,
        )
        return fig
