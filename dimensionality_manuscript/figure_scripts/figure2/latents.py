"""RRR <-> external-latents predictability, alone and beside the dimensionality sweep."""

import numpy as np

from vrAnalysis.helpers.plotting import beeswarm, errorPlot, format_spines
from dimensionality_manuscript.configs.rrr_to_external_latents import (
    DEFAULT_MODEL_EXT_INT_PAIR,
    VALID_MODEL_EXT_INT_PAIRS,
    pair_has_omission_response,
)
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import ModelName, short_model_name
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)

from .dim_sweep import add_dim_sweep_widgets, draw_dim_sweep_panel
from .model_style import ROLE_COLOR

# Colors mirror the model-zoo schematic: external models are black, internal (leak) models red.
LATENTS_EXTERNAL_COLOR = ROLE_COLOR["external"]
LATENTS_INTERNAL_COLOR = ROLE_COLOR["internal"]

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
LATENTS_GROUP_NAMES = ["Position", "Speed", "Reward"]


def model_pair_label(model_ext_int_pair: tuple[ModelName, ModelName]) -> str:
    """Render an (external, internal) model pair as a widget-safe string label.

    Syd selections take scalar values, so the tuple-valued ``model_ext_int_pair`` param axis is
    shown as a label and mapped back to its tuple before slicing (same idea as ``_tuple_label``
    in figure3/figure4).
    """
    return " / ".join(short_model_name(name) for name in model_ext_int_pair)


# Widget label -> raw (external, internal) tuple, for decoding the selection before results.sel.
MODEL_PAIR_LABELS: dict[str, tuple[ModelName, ModelName]] = {model_pair_label(pair): pair for pair in VALID_MODEL_EXT_INT_PAIRS}
DEFAULT_MODEL_PAIR_LABEL: str = model_pair_label(DEFAULT_MODEL_EXT_INT_PAIR)


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
    color_true = LATENTS_EXTERNAL_COLOR
    color_pred = LATENTS_INTERNAL_COLOR if both else LATENTS_EXTERNAL_COLOR
    return show_true, show_pred, both, color_true, color_pred


def add_latents_widgets(
    viewer,
    *,
    model_ext_int_pair: tuple[ModelName, ModelName] = DEFAULT_MODEL_EXT_INT_PAIR,
    agg_method: str = "mean",
    latents_show: str = "both",
    exclude_reward_omission: bool = False,
    beewidth: float = 0.3,
    dodge_offset: float = 0.3,
    alpha: float = 0.5,
    rank_axis_log: bool = True,
    inset_num_dims: int = 10,
) -> None:
    """Add the latents panels' knobs, including the label-encoded model-pair selection."""
    viewer.add_selection("model_ext_int_pair", value=model_pair_label(tuple(model_ext_int_pair)), options=list(MODEL_PAIR_LABELS))
    viewer.add_selection("agg_method", value=agg_method, options=["mean", "median"])
    viewer.add_selection("latents_show", value=latents_show, options=["both", "external", "internal"])
    viewer.add_boolean("exclude_reward_omission", value=exclude_reward_omission)
    viewer.add_float("beewidth", value=beewidth, min=0.0, max=1.0)
    viewer.add_float("dodge_offset", value=dodge_offset, min=0.0, max=1.0)
    viewer.add_float("alpha", value=alpha, min=0.0, max=1.0)
    viewer.add_boolean("rank_axis_log", value=rank_axis_log)
    viewer.add_integer("inset_num_dims", value=inset_num_dims, min=2, max=50)


def select_latents_data(results: ResultsAggregator, state, selection_names) -> dict[str, np.ndarray]:
    """Select the latents results for the current data-selection and model pair."""
    return results.sel(
        keys=_LATENTS_EACH_KEYS + _LATENTS_DIM_KEYS,
        avg_by_mouse=False,
        squeeze_ones=False,
        **data_selection(state, results, selection_names, model_ext_int_pair=MODEL_PAIR_LABELS[state["model_ext_int_pair"]]),
    )


def _exclude_omission(state) -> bool:
    """Whether to drop the reward group's omission columns for the selected pair.

    The predictive-reward pair has no omission columns, so the toggle is inert there rather
    than slicing a block that doesn't exist.
    """
    return state["exclude_reward_omission"] and pair_has_omission_response(MODEL_PAIR_LABELS[state["model_ext_int_pair"]])


def draw_latents_group_panel(ax, data: dict[str, np.ndarray], state, fontsize: float) -> None:
    """Draw the dodged position/speed/reward beeswarms (RRR -> external/internal) onto ``ax``."""
    agg = np.nanmean if state["agg_method"] == "mean" else np.nanmedian
    show_true, show_pred, both, color_true, color_pred = _latents_show_style(state["latents_show"])
    exclude_omission = _exclude_omission(state)
    dodge = state["dodge_offset"]
    beewidth = state["beewidth"]

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
    group_names = list(LATENTS_GROUP_NAMES)
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
            ax.plot(x0 + beewidth * offsets, finite_vals, linestyle="none", color=color, marker="o", markersize=4, alpha=state["alpha"])

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
        tick_fontsize=fontsize,
    )
    ax.set_xticks(group_centers, labels=group_names, rotation=0, rotation_mode="anchor", ha="center", fontsize=fontsize)
    ax.set_yticks([0, 1])
    ax.tick_params(axis="both", which="both", labelsize=fontsize)
    ax.set_ylabel(r"$R^2$", labelpad=-10, fontsize=fontsize)
    ax.set_xlabel("from neural latents", fontsize=fontsize)


def draw_latents_rank_panel(ax, data: dict[str, np.ndarray], state, fontsize: float) -> None:
    """Draw the rank-ordered curve (external/internal -> RRR) onto ``ax``, with optional inset."""
    show_true, show_pred, both, color_true, color_pred = _latents_show_style(state["latents_show"])
    agg_method = state["agg_method"]

    n_ranks = max(_trimmed_rank_length(data["test_score_each_true_to_rrr"], data["test_score_each_pred_to_rrr"]), 1)
    curve_true = data["test_score_each_true_to_rrr"][:, :n_ranks]
    curve_pred = data["test_score_each_pred_to_rrr"][:, :n_ranks]
    rank_positions = np.arange(1, n_ranks + 1)  # 1-indexed: rank 0 is undefined on a log axis

    if state["rank_axis_log"]:
        ax.set_xscale("log")
    if show_true:
        _center_spread_plot(
            ax, rank_positions, curve_true, axis=0, agg_method=agg_method, color=color_true, linewidth=1.5, alpha=0.2, label="External"
        )
    if show_pred:
        _center_spread_plot(
            ax, rank_positions, curve_pred, axis=0, agg_method=agg_method, color=color_pred, linewidth=1.5, alpha=0.2, label="Internal"
        )

    # Inset highlighting the initial-dimension dropoff (linear x only).
    if not state["rank_axis_log"]:
        n_inset = min(state["inset_num_dims"], n_ranks)
        inset_ypush = 0 if both else 0.05
        axins = ax.inset_axes([0.35, 0.35 + inset_ypush, 0.60, 0.45])
        if show_true:
            _center_spread_plot(
                axins, rank_positions[:n_inset], curve_true[:, :n_inset], axis=0, agg_method=agg_method, color=color_true, linewidth=1.5, alpha=0.2
            )
        if show_pred:
            _center_spread_plot(
                axins, rank_positions[:n_inset], curve_pred[:, :n_inset], axis=0, agg_method=agg_method, color=color_pred, linewidth=1.5, alpha=0.2
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
        tick_fontsize=fontsize,
    )
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.tick_params(axis="x", which="both", labelsize=fontsize)
    ax.set_xlabel("rank\nto neural latents", fontsize=fontsize)
    if both:
        ax.legend(loc="upper right", fontsize=fontsize, frameon=False)


class RRRExternalLatentsViewer(FigureViewer):
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

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RRRToExternalLatentsConfig`` results.
    model_ext_int_pair : tuple[ModelName, ModelName]
        The (external, internal) regressor pair, passed as a native tuple (it is encoded to the
        widget's string label internally). The predictive-reward pair has no omission-response
        columns, so ``exclude_reward_omission`` is a no-op for it.
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
    rank_axis_log : bool
        If True, log-scale the rank-axis panel's x-axis.
    inset_num_dims : int
        Number of initial ranks shown in the ax[1] inset. The inset only appears when
        ``rank_axis_log`` is False.
    width_rank_axis : float
        Width ratio for the rank-axis panel relative to the position/speed/reward panel.
    fontsize : float
        Tick-label and axis-label font size for both panels.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the data-selection widgets built from the aggregator's param axes.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        model_ext_int_pair: tuple[ModelName, ModelName] = DEFAULT_MODEL_EXT_INT_PAIR,
        agg_method: str = "mean",
        latents_show: str = "both",
        exclude_reward_omission: bool = False,
        beewidth: float = 0.3,
        dodge_offset: float = 0.3,
        alpha: float = 0.5,
        rank_axis_log: bool = True,
        inset_num_dims: int = 10,
        width_rank_axis: float = 0.6,
        fontsize: float = 10.0,
        figsize: tuple[float, float] = (7.0, 4.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self._data: dict[str, np.ndarray] = {}

        self.selection_names = add_data_selection_widgets(
            self, results, skip=("model_ext_int_pair",), defaults=selection_defaults
        )
        add_latents_widgets(
            self,
            model_ext_int_pair=model_ext_int_pair,
            agg_method=agg_method,
            latents_show=latents_show,
            exclude_reward_omission=exclude_reward_omission,
            beewidth=beewidth,
            dodge_offset=dodge_offset,
            alpha=alpha,
            rank_axis_log=rank_axis_log,
            inset_num_dims=inset_num_dims,
        )
        self.add_float("width_rank_axis", value=width_rank_axis, min=0.25, max=2.0)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        for name in (*self.selection_names, "model_ext_int_pair"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select data for the current data-selection and model pair."""
        self._data = select_latents_data(self.results, state, self.selection_names)

    def plot(self, state):
        fontsize = state["fontsize"]
        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, state["width_rank_axis"]])
        ax_group = fig.add_subplot(gs[0])
        ax_curve = fig.add_subplot(gs[1], sharey=ax_group)

        draw_latents_group_panel(ax_group, self._data, state, fontsize)
        draw_latents_rank_panel(ax_curve, self._data, state, fontsize)
        return fig


class DimSweepLatentsViewer(FigureViewer):
    """Three panels combining latents predictability with the regression dimensionality sweep.

    The panels answer one question in two halves: how well the neural latents and the
    external/internal regressor latents predict each other (ax[0], ax[1]), and how much of a
    session's activity each model explains as a function of how many regressor dimensions it
    spends (ax[2]).

    - ax[0] (Position/Speed/Reward) and ax[1] (Rank): the two panels of
      :class:`RRRExternalLatentsViewer`, sharing one y-axis (R^2) with each other but not with
      ax[2], whose y-axis is the sweep metric over its own range.
    - ax[2] (Regressor Dimensionality): across-session mean +/- SE of a fit metric versus
      regressor dimensionality, one curve per model. See
      :class:`~dimensionality_manuscript.figure_scripts.figure2.dim_sweep.RegressionDimSweepViewer`.

    The two halves read different aggregated results (``RRRToExternalLatentsConfig`` and
    ``RegressionDimensionalitySweepConfig``), so both aggregators are passed in. The
    data-selection widgets cover the union of the two aggregators' param axes, and each half is
    only handed the axes its own aggregator declares. ``fontsize`` is shared by all three
    panels; every other knob keeps the per-panel meaning it has in the single-figure viewers.

    Parameters
    ----------
    results_latents : ResultsAggregator
        Aggregated ``RRRToExternalLatentsConfig`` results. Drives ax[0] and ax[1].
    results_sweep : ResultsAggregator
        Aggregated ``RegressionDimensionalitySweepConfig`` results, with ``model_name`` as a
        param axis over ``KEY_FIGURE_MODELS``. Drives ax[2].
    width_sweep_axis, width_rank_axis : float
        Width ratios for the sweep and rank panels relative to the position/speed/reward panel.
    fontsize : float
        Font size for axis labels, tick labels, and legends in all three panels.
    figsize : tuple[float, float]
        Figure size in inches.
    **panel_knobs
        The latents knobs (see :class:`RRRExternalLatentsViewer`), the sweep knobs
        (``metric``, ``se``, ``xlog``, ``linewidth``, ``fill_alpha``, ``legend_options``),
        and starting values for the data-selection widgets.
    """

    _LATENTS_KNOBS = (
        "model_ext_int_pair",
        "agg_method",
        "latents_show",
        "exclude_reward_omission",
        "beewidth",
        "dodge_offset",
        "alpha",
        "rank_axis_log",
        "inset_num_dims",
    )
    _SWEEP_KNOBS = ("metric", "se", "xlog", "linewidth", "fill_alpha", "legend_options")

    def __init__(
        self,
        results_latents: ResultsAggregator,
        results_sweep: ResultsAggregator,
        *,
        width_sweep_axis: float = 1.0,
        width_rank_axis: float = 0.6,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (9.0, 3.0),
        **panel_knobs,
    ):
        self.results_latents = results_latents
        self.results_sweep = results_sweep
        self.figsize = figsize
        self._data: dict[str, np.ndarray] = {}

        latents_kwargs = {name: panel_knobs.pop(name) for name in self._LATENTS_KNOBS if name in panel_knobs}
        sweep_kwargs = {name: panel_knobs.pop(name) for name in self._SWEEP_KNOBS if name in panel_knobs}

        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        self.selection_names = add_data_selection_widgets(
            self, results_latents, results_sweep, skip=("model_name", "model_ext_int_pair"), defaults=panel_knobs
        )
        add_latents_widgets(self, **latents_kwargs)
        add_dim_sweep_widgets(self, **sweep_kwargs)

        self.add_float("width_sweep_axis", value=width_sweep_axis, min=0.25, max=3.0)
        self.add_float("width_rank_axis", value=width_rank_axis, min=0.25, max=2.0)

        for name in (*self.selection_names, "model_ext_int_pair"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select latents data for the current data-selection and model pair."""
        self._data = select_latents_data(self.results_latents, state, self.selection_names)

    def plot(self, state):
        fontsize = state["fontsize"]
        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, state["width_rank_axis"], state["width_sweep_axis"]])
        ax_group = fig.add_subplot(gs[0])
        # sharey only within the latents pair: ax_sweep's y-axis is the sweep metric, not R^2.
        ax_curve = fig.add_subplot(gs[1], sharey=ax_group)
        ax_sweep = fig.add_subplot(gs[2])

        draw_latents_group_panel(ax_group, self._data, state, fontsize)
        draw_latents_rank_panel(ax_curve, self._data, state, fontsize)
        draw_dim_sweep_panel(
            ax_sweep,
            self.results_sweep,
            state["metric"],
            data_selection(state, self.results_sweep, self.selection_names),
            se=state["se"],
            xlog=state["xlog"],
            linewidth=state["linewidth"],
            fill_alpha=state["fill_alpha"],
            fontsize=fontsize,
            legend_state=state,
        )
        return fig
