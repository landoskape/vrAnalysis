from dataclasses import dataclass, replace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
from dimensionality_manuscript.configs.regression import FIGURE_MODEL_NAMES
from dimensionality_manuscript.configs.rrr_to_external_latents import VALID_RRR_VARIANCE, VALID_SPKS_TYPES
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
    ):
        self.session = session
        self.spks_type = spks_type
        self.model_names = list(model_names)
        self.target = target
        self.predictions = predictions
        self.figsize = figsize

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
        self.add_selection("sort_method", value="environment", options=SORT_METHODS)
        self.add_selection("sample_fit_metric", value="r2", options=SAMPLE_FIT_METRICS)
        self.add_float("vmax", value=1.0, min=0.01, max=20.0)
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
        zscore = state["zscore"]
        xslice = slice(state["xslice_start"], state["xslice_stop"])
        idx_sort = self._get_sort(state["sort_method"])

        # Display arrays: median z-scored per neuron over all frames (stats stable across xslice) or raw.
        median_subtract = not self.session.zero_baseline_spks
        disp_cache: dict[str, np.ndarray] = {}

        def disp(name: str) -> np.ndarray:
            if name not in disp_cache:
                raw = self.target if name == "__target__" else self.predictions[name]
                disp_cache[name] = _median_zscore_rows(raw, median_subtract) if zscore else raw
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
                )
                ax.set_ylabel(_SAMPLE_FIT_LABELS[metric])
                ax.set_xlabel("Imaging Frames")
                ax.legend(loc="upper right", fontsize=10, ncol=max(1, len(self.model_names) // 2), frameon=False)
                continue

            if kind == "data":
                data = disp("__target__")[:, xslice][idx_sort]
                title = "True Activity"
            else:
                data = disp(model_name)[:, xslice][idx_sort]
                title = short_model_name(model_name)

            ax = fig.add_subplot(gs[irow, 0])
            ax.imshow(data, **raster_kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("ROIs")
            ax.text(1.0, 1.0, title, transform=ax.transAxes, ha="right", va="top", color="black")
            for spine in ["top", "right", "bottom", "left"]:
                ax.spines[spine].set_visible(False)
            if irow == last_raster_row and not include_curve:
                ax.set_xlabel("Imaging Frames")

            if not _inset_built:
                gray_values = np.linspace(0, vmax, 255) / vmax if vmax > 0 else np.zeros(255)
                cmap = plt.get_cmap("gray_r")
                colors = cmap(gray_values)
                cmap_data = colors[np.newaxis, :, :]
                inset = ax.inset_axes([0.2, 0.05, 0.6, 0.1])
                inset.imshow(cmap_data, aspect="auto")
                inset.set_xticks([])
                inset.set_yticks([])
                inset.text(-0.03, 0.5, "0", transform=inset.transAxes, ha="right", va="center", clip_on=False)
                inset.text(1.03, 0.5, f"{vmax}", transform=inset.transAxes, ha="left", va="center", clip_on=False)
                inset.text(0.5, 1.25, "zscore activity", transform=inset.transAxes, ha="center", va="bottom", clip_on=False)
                _inset_built = True

            if include_error and kind == "raster":
                residual = (disp("__target__")[:, xslice] - disp(model_name)[:, xslice])[idx_sort]
                ax_err = fig.add_subplot(gs[irow, 1])
                ax_err.imshow(residual, **error_kwargs)
                ax_err.set_xticks([])
                ax_err.set_yticks([])
                ax_err.text(1.0, 1.0, "Residual", transform=ax_err.transAxes, ha="right", va="top", color="black")
                for spine in ["top", "right", "bottom", "left"]:
                    ax_err.spines[spine].set_visible(False)
                if irow == last_raster_row:
                    ax_err.set_xlabel("Imaging Frames")

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
    xslice: slice | None = None,
    vmax: float = 6,
    figsize: tuple[float, float] | None = None,
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
        Median z-score each neuron (over all frames) before display via ``median_zscore``.
    xslice : slice | None
    vmax : float
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
        model = get_model(model_name, registry, activity_parameters=activity_parameters_name)
        model_target, prediction = _model_prediction_grid(model, session, spks_type, method, train_split, test_split)
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

    viewer = ModelRasterFocus(session, spks_type, model_names, target, predictions, idx_sort_environment, figsize, xstart, xstop)
    viewer.update_boolean("include_data_raster", value=include_data_raster)
    viewer.update_boolean("include_error_column", value=include_error_column)
    viewer.update_boolean("include_sample_fit_curve", value=include_sample_fit_curve)
    viewer.update_selection("sort_method", value=sort_method)
    viewer.update_boolean("zscore", value=zscore)
    viewer.update_selection("sample_fit_metric", value=sample_fit_metric)
    viewer.update_float("vmax", value=vmax)
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
        limits = {
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
        for name in _ZOO_TUNABLES:
            lo, hi = limits[name]
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
}


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
        self.add_multiple_selection("model_names", value=list(FIGURE_MODEL_NAMES), options=list(FIGURE_MODEL_NAMES))
        self.add_selection("metric", value="r2", options=list(_DIM_SWEEP_METRIC_LABELS))
        self.add_boolean("se", value=True)
        self.add_boolean("xlog", value=True)
        self.add_float("linewidth", value=2.0, min=0.5, max=6.0)
        self.add_float("fill_alpha", value=0.12, min=0.0, max=1.0)
        self.add_float("fontsize", value=9.0, min=4.0, max=24.0)

    def _model_curve(self, model_name: ModelName, metric: str) -> tuple[np.ndarray, np.ndarray]:
        """Across-session x (mean dim, sorted) and y (metric) arrays: shapes (n_pts,), (n_sess, n_pts)."""
        prefix = _SWEEP_KEY_BY_MODEL[model_name]
        sel = self.results.sel(model_name=model_name, squeeze_ones=False)
        dim = np.atleast_2d(sel[f"{prefix}_dim"])
        y = np.atleast_2d(sel[f"{prefix}_{metric}"])
        x = np.nanmean(dim, axis=0)
        idx_sort = np.argsort(x)
        return x[idx_sort], y[:, idx_sort]

    def plot(self, state):
        fig, ax = plt.subplots(figsize=self.figsize, layout="constrained")
        metric = state["metric"]
        for model_name in state["model_names"]:
            x, y = self._model_curve(model_name, metric)
            style = _DIM_SWEEP_MODEL_STYLE[model_name]
            color = _DIM_SWEEP_ROLE_COLOR[style["role"]]
            linestyle = _DIM_SWEEP_ARCH_LINESTYLE[style["arch"]]
            curve_width = state["linewidth"] * (1.35 if style["role"] == "neural" else 1.0)
            errorPlot(
                x,
                y,
                axis=0,
                se=state["se"],
                ax=ax,
                color=color,
                linestyle=linestyle,
                linewidth=curve_width,
                alpha=state["fill_alpha"],
            )

        if state["xlog"]:
            ax.set_xscale("log")
        xlim = ax.get_xlim()
        xticks = [1, 10, 100, 1000, 10000]
        xticks = [t for t in xticks if t <= xlim[1]]
        ax.set_xlabel("Regressor Dimensionality", fontsize=state["fontsize"])
        ax.set_ylabel(_DIM_SWEEP_METRIC_LABELS[metric], fontsize=state["fontsize"])
        architecture_handles = [
            Line2D([], [], color="0.25", linestyle=_DIM_SWEEP_ARCH_LINESTYLE[arch], linewidth=2.0, label=label)
            for arch, label in (
                ("placefield", "placefield"),
                ("highd_pos", "rbf-pos"),
                ("highd_pos_speed", "+speed"),
                ("highd_pos_speed_reward", "+reward"),
                ("placefield_gain", "+gain"),
            )
        ]
        ax.legend(
            handles=architecture_handles,
            loc="upper left",
            fontsize=state["fontsize"],
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
            tick_fontsize=state["fontsize"],
            spines_visible=["left", "bottom"],
        )
        # Apply this after format_spines, which calls tick_params itself. Include minor ticks
        # so logarithmic-axis labels follow the viewer fontsize as well.
        ax.tick_params(axis="both", which="both", labelsize=state["fontsize"])
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
        param axis over ``FIGURE_MODEL_NAMES``.
    model_names : list[ModelName] or None
        Models to plot. Defaults to all of ``FIGURE_MODEL_NAMES``.
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
    viewer.update_multiple_selection("model_names", value=list(model_names) if model_names is not None else list(FIGURE_MODEL_NAMES))
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
_LATENTS_DIM_KEYS = ["num_pos_params", "num_speed_params", "num_reward_params"]

# The three regressor groups that make up the external model's basis (see
# RRRToExternalLatentsConfig / _pos_speed_reward_dims), in column order.
_LATENTS_GROUP_NAMES = ["Position", "Speed", "Reward"]


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


def _group_agg_scores(
    each_scores: np.ndarray,
    num_pos: np.ndarray,
    num_speed: np.ndarray,
    num_reward: np.ndarray,
    agg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-mouse (position, speed, reward) aggregate of a per-dimension score array.

    Each mouse's own ``(num_pos, num_speed, num_reward)`` boundaries slice its row of
    ``each_scores`` before aggregating, since a mouse's basis dimensionality (num_basis,
    num_environments, reward_num_basis_lags) differs from every other mouse's, so raw
    column index has no shared meaning across mice.
    """
    n_mice = each_scores.shape[0]
    pos_out = np.full(n_mice, np.nan)
    speed_out = np.full(n_mice, np.nan)
    reward_out = np.full(n_mice, np.nan)
    for i in range(n_mice):
        npos, nspeed, nreward = int(num_pos[i]), int(num_speed[i]), int(num_reward[i])
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


class RRRExternalLatentsViewer(Viewer):
    """RRR <-> external-latents predictability, per-dimension breakdown as two linked x-axes.

    ax[0] and ax[1] share one y-axis (R^2) but have independent, differently-scaled x-axes:
    - ax[0] (Position/Speed/Reward): ``rrr_to_*`` (RRR -> external/internal) is scored per
      external-basis dimension. Those dimensions aren't a single homogeneous group -- they're
      the concatenation of position, speed, and reward regressors (see
      ``RRRToExternalLatentsConfig`` / ``_pos_speed_reward_dims``), and mice differ in how many
      columns each sub-group has. Each mouse's per-dimension scores are therefore split into
      (position, speed, reward) using that mouse's own boundaries, aggregated (mean or median,
      per ``agg_method``) within each group, and drawn as three beeswarms, each dodged into
      External (black) / Internal (red) halves.
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
        self.add_selection("rrr_variance", value=0.95, options=list(VALID_RRR_VARIANCE))
        self.add_boolean("normalize", value=False)
        self.add_selection("agg_method", value="mean", options=["mean", "median"])
        self.add_float("beewidth", value=0.3, min=0.0, max=1.0)
        self.add_float("dodge_offset", value=0.3, min=0.0, max=1.0)
        self.add_float("alpha", value=0.5, min=0.0, max=1.0)
        self.add_float("width_rank_axis", value=0.6, min=0.25, max=2.0)

        for name in ("spks_type", "activity_parameters_name", "rrr_variance", "normalize"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select data for the current (spks_type, activity_parameters_name, rrr_variance, normalize)."""
        self._data = self.results.sel(
            keys=_LATENTS_EACH_KEYS + _LATENTS_DIM_KEYS,
            avg_by_mouse=False,
            squeeze_ones=False,
            spks_type=state["spks_type"],
            activity_parameters_name=state["activity_parameters_name"],
            rrr_variance=state["rrr_variance"],
            normalize=state["normalize"],
        )

    def plot(self, state):
        out = self._data
        agg_method = state["agg_method"]
        agg = np.nanmean if agg_method == "mean" else np.nanmedian
        beewidth = state["beewidth"]
        dodge = state["dodge_offset"]
        alpha = state["alpha"]

        n_ranks = max(_trimmed_rank_length(out["test_score_each_true_to_rrr"], out["test_score_each_pred_to_rrr"]), 1)
        curve_true = out["test_score_each_true_to_rrr"][:, :n_ranks]
        curve_pred = out["test_score_each_pred_to_rrr"][:, :n_ranks]
        rank_positions = np.arange(1, n_ranks + 1)  # 1-indexed: rank 0 is undefined on a log axis

        pos_true, speed_true, reward_true = _group_agg_scores(
            out["test_score_each_rrr_to_true"], out["num_pos_params"], out["num_speed_params"], out["num_reward_params"], agg
        )
        pos_pred, speed_pred, reward_pred = _group_agg_scores(
            out["test_score_each_rrr_to_pred"], out["num_pos_params"], out["num_speed_params"], out["num_reward_params"], agg
        )
        group_scores = [(pos_true, pos_pred), (speed_true, speed_pred), (reward_true, reward_pred)]
        group_centers = np.arange(len(_LATENTS_GROUP_NAMES), dtype=float)

        plt.close("all")
        fig = plt.figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, state["width_rank_axis"]])
        ax_group = fig.add_subplot(gs[0])
        ax_curve = fig.add_subplot(gs[1], sharey=ax_group)

        # --- ax[0]: three dodged pos/speed/reward beeswarms (RRR -> external/internal) ---
        for center, (true_vals, pred_vals) in zip(group_centers, group_scores):
            for vals, x0, color in [(true_vals, center - dodge, _LATENTS_EXTERNAL_COLOR), (pred_vals, center + dodge, _LATENTS_INTERNAL_COLOR)]:
                finite_vals = vals[np.isfinite(vals)]
                offsets = beeswarm(finite_vals) if finite_vals.size > 1 else np.zeros_like(finite_vals)
                ax_group.plot(x0 + beewidth * offsets, finite_vals, linestyle="none", color=color, marker="o", markersize=4, alpha=alpha)

        spacing_required = 1.2 * (dodge + beewidth)
        ax_group.set_xlim(group_centers[0] - spacing_required, group_centers[-1] + spacing_required)
        format_spines(
            ax_group,
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[group_centers[0], group_centers[-1]],
            ybounds=[0, 1],
        )
        ax_group.set_xticks(group_centers, labels=_LATENTS_GROUP_NAMES, rotation=45, rotation_mode="anchor", ha="right", fontsize=self.fontsize)
        ax_group.set_ylim(-0.1, 1.1)
        ax_group.set_yticks([0, 1])
        ax_group.tick_params(axis="y", labelsize=self.fontsize)
        ax_group.set_ylabel(r"$R^2$", labelpad=-10, fontsize=self.fontsize)
        ax_group.set_xlabel("from neural", fontsize=self.fontsize)

        # --- ax[1]: rank-ordered curve (external/internal -> RRR), log-scale x, shares y with ax[0] ---
        ax_curve.set_xscale("log")
        _center_spread_plot(
            ax_curve,
            rank_positions,
            curve_true,
            axis=0,
            agg_method=agg_method,
            color=_LATENTS_EXTERNAL_COLOR,
            linewidth=1.5,
            alpha=0.2,
            label="External",
        )
        _center_spread_plot(
            ax_curve,
            rank_positions,
            curve_pred,
            axis=0,
            agg_method=agg_method,
            color=_LATENTS_INTERNAL_COLOR,
            linewidth=1.5,
            alpha=0.2,
            label="Internal",
        )
        format_spines(
            ax_curve,
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["bottom"],
            xbounds=[1, max(n_ranks, 2)],
            ybounds=[0, 1],
        )
        ax_curve.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_curve.tick_params(axis="x", labelsize=self.fontsize)
        ax_curve.set_xlabel("to neural", fontsize=self.fontsize)
        ax_curve.legend(loc="upper right", fontsize=self.fontsize, frameon=False)

        return fig


def rrr_external_latents_score(
    results: ResultsAggregator,
    spks_type: SpksTypes = "sigrebase",
    activity_parameters_name: str = "default",
    rrr_variance: float | str = 0.95,
    normalize: bool = False,
    agg_method: str = "mean",
    beewidth: float = 0.3,
    dodge_offset: float = 0.3,
    alpha: float = 0.5,
    width_rank_axis: float = 0.6,
    fontsize: float = 8,
    figsize: tuple[float, float] = (5.0, 3.0),
    return_syd_viewer: bool = False,
):
    """
    RRR <-> external-latents predictability from ``RRRToExternalLatentsConfig`` results.

    Two panels sharing one y-axis (R^2) with independent x-axes. ax[0] (Position/Speed/Reward)
    splits ``rrr_to_*`` (RRR -> external/internal) per mouse into position/speed/reward sub-groups
    (their sizes vary by mouse -- see ``RRRToExternalLatentsConfig``), aggregates within each
    group, and draws three beeswarms, each dodged into External/Internal halves. ax[1] (Rank,
    log-scale) draws ``*_to_rrr`` (external/internal -> RRR), scored per RRR-latent and rank-
    ordered, as a mean+/-SE (or median+/-IQR) curve over every available rank.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RRRToExternalLatentsConfig`` results.
    spks_type, activity_parameters_name, rrr_variance, normalize
        Selects which stored variation to read (see ``RRRToExternalLatentsConfig``).
    agg_method : {"mean", "median"}
        Aggregation used both for the per-mouse position/speed/reward group scores and for the
        rank-ordered curve's center line (mean+/-SE or median+/-IQR).
    beewidth : float
        Horizontal spread of points within one beeswarm (External or Internal) half.
    dodge_offset : float
        Horizontal separation between the External and Internal beeswarm halves within a group.
    alpha : float
        Marker opacity for the beeswarm points.
    width_rank_axis : float
        Width ratio for the rank-axis panel relative to the position/speed/reward panel.
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
    viewer.update_selection("rrr_variance", value=rrr_variance)
    viewer.update_boolean("normalize", value=normalize)
    viewer.refresh_data(viewer.state)  # pre-deploy update_* may not fire on_change

    viewer.update_selection("agg_method", value=agg_method)
    viewer.update_float("beewidth", value=beewidth)
    viewer.update_float("dodge_offset", value=dodge_offset)
    viewer.update_float("alpha", value=alpha)
    viewer.update_float("width_rank_axis", value=width_rank_axis)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig
