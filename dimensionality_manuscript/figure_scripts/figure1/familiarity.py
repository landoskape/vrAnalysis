"""R² as a function of familiarity: one ROI, one mouse, the population, the environment slots."""

import matplotlib as mpl
import numpy as np
from scipy.stats import rankdata

from vrAnalysis.helpers import edge2center

from dimensionality_manuscript.configs.pfpred_quality import _kde_r2
from dimensionality_manuscript.configs.placefield_structure import PlacefieldPredictionConfig
from dimensionality_manuscript.env_order import MAX_ENV_SLOTS, _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    curve_group_bounds,
    data_selection,
    render_curve_group,
)
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import (
    add_session_curve_widgets,
    by_session_curves,
    decimal_yticks,
    draw_example_roi_panel,
    draw_session_curve_groups,
    env_slot_color,
    fit_square_panels,
    ordinal,
    pad_stack,
    style_axis,
    support_length,
)

#: Grid resolution of an ECDF curve. The ECDF is a step function, so this only has to be fine
#: enough that the steps are not visible at figure scale.
_ECDF_POINTS = 201

#: How close two of :class:`R2Familiarity`'s y ranges have to be, as a fraction of the larger, for
#: the panels to be put on one shared axis instead of two nearly-identical ones.
_SHARE_Y_TOLERANCE = 0.15


class R2Familiarity(FigureViewer):
    """R² as a function of familiarity: one ROI, one mouse, the population, the slots.

    ``panel_mode`` picks what ``ax[1]``/``ax[2]`` draw, and the four options answer different
    questions. ``"r2_rel"`` and ``"r2_rel_prct"`` show R² *conditioned* on spatial reliability, so
    they ask whether a cell of given quality is better predicted with experience; ``"histogram"``
    and ``"ecdf"`` show the marginal distribution of R², which folds in any drift of the
    reliability distribution itself.

    - ``"r2_rel"``: the running average E[R² | reliability] -- the same kernel regression drawn on
      ``ax[1]`` of :class:`~.r2_placefield.R2PlacefieldFocus`, but resolved per session rather than
      pooled, read straight from the aggregator's ``r2_kde_slot`` / ``r2_kde_pooled``.
    - ``"r2_rel_prct"``: the same regression against reliability *rank* within the session. The
      reliability distribution is heavily skewed, so uniform bins of reliability are wildly
      unequally populated and the interesting range is squeezed into a sliver of the axis; ranks
      put the same number of ROIs behind every x.
    - ``"histogram"``: the fraction of ROIs per R² bin, with out-of-range R² clipped into the edge
      bins rather than dropped. Fractions, not counts, since sessions differ in ROI count.
    - ``"ecdf"``: the same distribution as a cumulative fraction -- no bin-edge choice, and curves
      of many sessions overlay far more legibly than histograms do.

    The panels themselves:

    - ``ax[0]``: one ROI's activity against its place-field prediction. Unlike everything to its
      right this panel is not split by environment: it uses every predictable frame of the
      session. The prediction and per-neuron standard deviation are loaded from
      ``PlacefieldPredictionConfig`` rather than rebuilt here.
    - ``ax[1]``: every session of that mouse, one curve per session colored by session number.
    - ``ax[2]``: the same curves for the whole population -- mouse-averaged overall
      (``curve_mode="average"``) or grouped by session number (``"by_session"``). Shares
      ``ax[1]``'s y-axis.
    - ``ax[3]``: R² collapsed to one number per session (``summary_stat`` over ROIs), against how
      many sessions of that environment the mouse has had, one curve per experience slot. It joins
      the shared y-axis too when its range comes out close enough to the curve panels'.

    Environments are indexed by experience-order slot, so slot 0 is the first environment that
    mouse ever saw; ``env_slot_ids`` carries the underlying environment index, which differs
    between the two cohorts.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``PFPredQualityConfig`` results (schema v5 or later).
    mouse : str or None
        Example mouse for ``ax[0]`` and ``ax[1]``. None takes the first alphabetically.
    example_session : str or None
        Which of that mouse's sessions ``ax[0]``'s ROI comes from, as ``session_print()`` prints it
        (``"mouse/date/id"``). None takes the mouse's first session.
    roi : int
        ROI index within ``example_session``, in session-filtered ROI coordinates.
    example_style : {"points", "density"}
        How ``ax[0]`` draws the frames.
    example_alpha : float
        Opacity of ``ax[0]``'s point cloud. Ignored by ``"density"``.
    env_mode : {"pooled", "best", "slot"}
        Which environments contribute to ``ax[1]`` and ``ax[2]``. ``ax[0]`` always uses every
        environment and ``ax[3]`` always splits by slot, so neither is affected.
    env_slot : int
        Experience-order slot used when ``env_mode="slot"``.
    panel_mode : {"r2_rel", "r2_rel_prct", "histogram", "ecdf"}
        What ``ax[1]``/``ax[2]`` draw (see above).
    xrange : tuple[float, float]
        Reliability limits of ``ax[1]``/``ax[2]`` under ``panel_mode="r2_rel"``. The rank axis is
        always the full 0-100.
    r2_range : tuple[float, float]
        R² limits of the ``"histogram"`` and ``"ecdf"`` modes, and the range binning and clipping
        use: ROIs outside it land in the edge bin rather than being dropped.
    r2_bins : int
        Number of equal-width R² bins inside ``r2_range`` for ``panel_mode="histogram"``.
    prct_points : int
        Grid resolution of the rank axis for ``panel_mode="r2_rel_prct"``.
    prct_bandwidth : float
        Kernel width of the rank regression, in percentile units.
    auto_ylim : bool
        Fit the top of the shared R² axis to the curves inside ``xrange``; the bottom is always 0.
        False uses ``r2_ylim``. Ignored by ``"histogram"`` (always fit) and ``"ecdf"`` (always 0-1).
    r2_ylim : tuple[float, float]
        Explicit R² limits, used when ``auto_ylim`` is False.
    summary_stat : {"mean", "median", "percentile"}
        How ``ax[3]`` collapses a session's per-ROI R² to one number. A high percentile tracks the
        best-predicted cells rather than the bulk, which the mean is dominated by.
    summary_percentile : float
        Percentile in ``[0, 100]`` used when ``summary_stat="percentile"``, in steps of 0.5.
    slot_style : {"each", "errorPlot"}
        ``ax[3]``'s rendering.
    show_slot_legend : bool
        Draw the env-slot legend on ``ax[3]``.
    cmap : str
        Colormap sampled across sessions, first session at 0 and last at 1.
    linewidth, alpha : float
        Line width and opacity of ``ax[1]``'s curves.
    fontsize : float
        Base font size in points.
    figsize : tuple[float, float]
        Figure size in inches. The height is ignored when ``square_panels`` is set.
    square_panels : bool
        Give all four panels a square aspect: the panels are drawn one width (the panel-specific
        width ratios are dropped, since a row of axes shares one height and only equal widths can
        all be square) and the figure height is solved for from ``figsize[0]`` and however much
        room the decorations turn out to need.
    curve_mode, plot_style, hide_error, skip_sessions, inset_position
        ``ax[2]``'s knobs, forwarded to :func:`~._shared.add_session_curve_widgets`.
    **selection_defaults
        Starting values for the aggregator's own param axes, by name.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        example_session: str | None = None,
        roi: int = 0,
        example_style: str = "points",
        example_alpha: float = 0.1,
        env_mode: str = "best",
        env_slot: int = 0,
        panel_mode: str = "r2_rel",
        xrange: tuple[float, float] = (-1.0, 1.0),
        r2_range: tuple[float, float] = (-1.0, 1.0),
        r2_bins: int = 20,
        prct_points: int = 101,
        prct_bandwidth: float = 5.0,
        auto_ylim: bool = True,
        r2_ylim: tuple[float, float] = (-0.05, 0.4),
        curve_mode: str = "by_session",
        plot_style: str = "each",
        hide_error: bool = False,
        skip_sessions: int = 0,
        inset_position: tuple[float, float, float, float] = (0.35, 0.85, 0.6, 0.075),
        summary_stat: str = "mean",
        summary_percentile: float = 90.0,
        slot_style: str = "errorPlot",
        show_slot_legend: bool = True,
        cmap: str = "coolwarm",
        linewidth: float = 1.0,
        alpha: float = 0.9,
        fontsize: float = 8.0,
        figsize: tuple[float, float] = (8.0, 2.0),
        square_panels: bool = False,
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize

        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self._load_arrays(self.state)

        # ax[0] needs one large frame x ROI prediction at a time. Index the store metadata now,
        # but defer unpickling until that session is actually selected; a companion aggregator
        # would pad and materialize this array over every session.
        self._example_config = PlacefieldPredictionConfig()
        example_rows = results.store.summary_table(
            analysis_type=self._example_config.display_name,
            session_ids=list(results.session_ids),
            schema_version=self._example_config.schema_version,
        )
        self._example_store_rows = {row["session_id"]: row for row in example_rows if row["analysis_key"] == self._example_config.key()}
        self._example_result_cache: dict[int, dict] = {}
        self._example_activity_cache: dict[int, np.ndarray] = {}
        self._example_roi_indices_cache: dict[int, np.ndarray] = {}

        # Rows of every array above are aggregator sessions; a mouse's rows are ordered
        # chronologically here so "session number" means what it says on the color axis.
        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(name) for name in mouse_names})
        self._rows_by_mouse = {
            name: np.array(sorted(np.flatnonzero(mouse_names == name), key=lambda row: _session_sort_key(results.sessions[row])))
            for name in self.mice
        }
        # ax[0]'s session picker works in printable labels rather than aggregator rows.
        self._example_rows_by_mouse = {
            name: {
                results.sessions[row].session_print(): int(row) for row in rows if results.sessions[row].session_uid in self._example_store_rows
            }
            for name, rows in self._rows_by_mouse.items()
        }

        # --- what goes into a curve ---
        self.add_selection("mouse", value=mouse if mouse is not None else self.mice[0], options=self.mice)
        self.add_selection("env_mode", value=env_mode, options=["pooled", "best", "slot"])
        self.add_integer("env_slot", value=env_slot, min=0, max=MAX_ENV_SLOTS - 1)
        self.add_selection("panel_mode", value=panel_mode, options=["r2_rel", "r2_rel_prct", "histogram", "ecdf"])
        # Each panel mode has its own x quantity, so each carries its own range: reliability for
        # r2_rel, rank for r2_rel_prct (always the full 0-100), R² for the distribution modes.
        self.add_float_range("xrange", value=tuple(xrange), min=-1.0, max=1.0, step=0.05)
        self.add_float_range("r2_range", value=tuple(r2_range), min=-2.0, max=2.0, step=0.05)
        self.add_integer("r2_bins", value=r2_bins, min=4, max=100)
        self.add_integer("prct_points", value=prct_points, min=11, max=201)
        # Kernel width in percentile units: 5 means E[R² | rank] is smoothed over a 5-point-wide
        # slice of the session's ROIs.
        self.add_float("prct_bandwidth", value=prct_bandwidth, min=0.5, max=25.0, step=0.5)
        self.add_boolean("auto_ylim", value=auto_ylim)
        self.add_float_range("r2_ylim", value=tuple(r2_ylim), min=-1.0, max=1.0, step=0.05)

        # --- ax[0]: one ROI of one session of the selected mouse, data vs PF prediction ---
        # Options are filled in by update_example_session, which the mouse selector drives.
        self.add_selection("example_session", value=example_session, options=[example_session])
        self.add_integer("roi", value=roi, min=0, max=max(roi, 0))
        self.add_selection("example_style", value=example_style, options=["points", "density"])
        self.add_float("example_alpha", value=example_alpha, min=0.01, max=1.0, step=0.01)

        # --- ax[2]: population curves ---
        add_session_curve_widgets(
            self,
            num_sessions=len(results.sessions),
            curve_mode=curve_mode,
            plot_style=plot_style,
            hide_error=hide_error,
            skip_sessions=skip_sessions,
            inset_position=inset_position,
        )

        # --- ax[3]: summary stat against experience, one curve per environment slot ---
        self.add_selection("summary_stat", value=summary_stat, options=["mean", "median", "percentile"])
        # Half-percentile steps so the conventional tail levels (2.5, 97.5) are reachable.
        self.add_float("summary_percentile", value=summary_percentile, min=0.0, max=100.0, step=0.5)
        self.add_selection("slot_style", value=slot_style, options=["each", "errorPlot"])
        self.add_boolean("show_slot_legend", value=show_slot_legend)

        # --- style ---
        self.add_text("cmap", value=cmap)
        self.add_float("linewidth", value=linewidth, min=0.25, max=4.0)
        self.add_float("alpha", value=alpha, min=0.05, max=1.0)
        self.add_float("fontsize", value=fontsize, min=3.0, max=30.0)
        # Square panels are all one width, so they cost the panel-specific width ratios; the
        # figure height stops being an input and becomes whatever squareness demands.
        self.add_boolean("square_panels", value=square_panels)

        self.on_change(list(self.selection_names), self.reload_arrays)
        self.on_change("mouse", self.update_slot_bounds)
        self.on_change("mouse", self.update_example_session)
        self.on_change("example_session", self.update_roi_bounds)
        self.on_change(
            ["mouse", "env_mode", "env_slot", "panel_mode", "r2_range", "r2_bins", "prct_points", "prct_bandwidth", "summary_stat", "summary_percentile"],
            self.refresh_data,
        )
        self.update_slot_bounds(self.state)
        self.update_example_session(self.state)
        self.refresh_data(self.state)

    # ---------------------------------------------------------------- data selection --

    def _load_arrays(self, state) -> None:
        """Pull the per-slot R² and reliability keys for the current data selection."""
        keys = ["r2_kde_grid", "r2_kde_slot", "r2_kde_pooled", "r2_slot", "reliability_slot", "best_env_slot", "env_slot_ids"]
        out = self.results.sel(keys=keys, squeeze_ones=False, **data_selection(state, self.results, self.selection_names))
        # The per-slot R² keys arrived in schema v5; against an older store they are simply absent,
        # which would otherwise surface as a bare KeyError on an arbitrary one of them.
        missing = [key for key in keys if key not in out]
        if missing:
            raise KeyError(f"Aggregator is missing {missing} -- rerun the pfpred_quality sweep (these keys need schema v5).")
        self.r2_kde_slot = np.asarray(out["r2_kde_slot"], dtype=float)  # (sessions, slots, grid)
        self.r2_kde_pooled = np.asarray(out["r2_kde_pooled"], dtype=float)  # (sessions, grid)
        self.r2_slot = np.asarray(out["r2_slot"], dtype=float)  # (sessions, slots, max_rois)
        self.reliability_slot = np.asarray(out["reliability_slot"], dtype=float)  # (sessions, slots, max_rois)
        self.env_slot_ids = np.asarray(out["env_slot_ids"], dtype=float)  # (sessions, slots)
        # Stored as one number per session; how the aggregator pads a scalar key is not worth
        # depending on, so it is flattened back to one column.
        self.best_env_slot = np.asarray(out["best_env_slot"], dtype=float).reshape(len(self.results.sessions), -1)[:, 0]
        # Every session stores the same grid (it comes from the config), so take the first session
        # that actually has it.
        grid = np.asarray(out["r2_kde_grid"], dtype=float)
        idx_grid = np.flatnonzero(np.all(np.isfinite(grid), axis=1))
        if idx_grid.size == 0:
            raise ValueError("No session in the aggregator has r2_kde_grid -- rerun the pfpred_quality sweep.")
        self.kde_grid = grid[idx_grid[0]]
        self.num_slots = self.r2_kde_slot.shape[1]

    def reload_arrays(self, state):
        """Re-pull the arrays after a data-selection change, then re-derive every curve."""
        self._load_arrays(state)
        self.update_slot_bounds(state)
        self.refresh_data(self.state)

    def update_slot_bounds(self, state):
        """Limit the slot selector to the environments this mouse actually ran."""
        rows = self._rows_by_mouse[state["mouse"]]
        num_slots = int(np.sum(np.any(np.isfinite(self.r2_kde_slot[rows]), axis=(0, 2))))
        self.update_integer("env_slot", max=max(num_slots - 1, 0))

    def update_example_session(self, state):
        """Offer ax[0] the sessions of the selected mouse, keeping the current one if it survives."""
        labels = list(self._example_rows_by_mouse[state["mouse"]])
        if not labels:
            raise ValueError(f"No PlacefieldPredictionConfig schema-v1 result is stored for {state['mouse']}.")
        current = state["example_session"] if state["example_session"] in labels else labels[0]
        self.update_selection("example_session", value=current, options=labels)
        # Seeding a selection does not always fire its callbacks, and the ROI count is a property
        # of the session that was just chosen, so the bound is refreshed here directly.
        self.update_roi_bounds({**state, "example_session": current})

    def update_roi_bounds(self, state):
        """Limit the ROI selector to the ROIs the example session actually has.

        The selector works in session-filtered ROI coordinates even though the stored prediction
        contains every ROI.
        """
        row = self._example_rows_by_mouse[state["mouse"]][state["example_session"]]
        self.update_integer("roi", max=max(len(self._example_roi_indices(row)) - 1, 0))

    # --------------------------------------------------------------- the example ROI --

    def _example_roi_indices(self, row: int) -> np.ndarray:
        """Raw ROI indices corresponding to the example panel's filtered ROI coordinates."""
        if row not in self._example_roi_indices_cache:
            session = self.results.sessions[row]
            previous_spks_type = session.params.spks_type
            session.params.spks_type = self._example_config.spks_type
            try:
                self._example_roi_indices_cache[row] = np.flatnonzero(session.idx_rois)
            finally:
                session.params.spks_type = previous_spks_type
        return self._example_roi_indices_cache[row]

    def _example_roi_traces(self, row: int, roi: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw activity and stored all-trial PF prediction, normalized by the saved std."""
        session = self.results.sessions[row]
        raw_roi = self._example_roi_indices(row)[roi]
        if row not in self._example_result_cache:
            store_row = self._example_store_rows[session.session_uid]
            result = self._example_config.get_result(self.results.store, store_row)
            if result is None or "placefield_prediction" not in result or "spks_std" not in result:
                raise KeyError(
                    f"Stored cross_validated_placefields result for {session.session_print()} "
                    "does not contain placefield_prediction and spks_std (PlacefieldPredictionConfig schema v1 required)."
                )
            self._example_result_cache[row] = result
        if row not in self._example_activity_cache:
            previous_spks_type = session.params.spks_type
            session.params.spks_type = self._example_config.spks_type
            try:
                self._example_activity_cache[row] = np.asarray(session.spks)
            finally:
                session.params.spks_type = previous_spks_type

        result = self._example_result_cache[row]
        data = self._example_activity_cache[row][:, raw_roi]
        prediction = np.asarray(result["placefield_prediction"])[: data.shape[0], raw_roi]
        scale = np.asarray(result["spks_std"])[raw_roi]
        idx_keep = np.isfinite(data) & np.isfinite(prediction)
        if np.isfinite(scale) and scale > 0:
            data = data / scale
            prediction = prediction / scale
        return data[idx_keep], prediction[idx_keep]

    # --------------------------------------------------------------- panel-mode maths --

    def _grid(self, state) -> np.ndarray:
        """The x values the current panel mode's curves are evaluated on."""
        panel_mode = state["panel_mode"]
        if panel_mode == "r2_rel":
            return self.kde_grid
        if panel_mode == "r2_rel_prct":
            return np.linspace(0.0, 100.0, state["prct_points"])
        low, high = tuple(state["r2_range"])
        if panel_mode == "histogram":
            return edge2center(np.linspace(low, high, state["r2_bins"] + 1))
        if panel_mode == "ecdf":
            return np.linspace(low, high, _ECDF_POINTS)
        raise ValueError(f"Invalid panel_mode: {panel_mode!r}")

    def _xlimits(self, state) -> tuple[float, float]:
        """Visible x range of ``ax[1]``/``ax[2]``, in the units the current panel mode plots."""
        panel_mode = state["panel_mode"]
        if panel_mode == "r2_rel":
            return tuple(state["xrange"])
        if panel_mode == "r2_rel_prct":
            return (0.0, 100.0)
        return tuple(state["r2_range"])

    def _xlabel(self, state) -> str:
        panel_mode = state["panel_mode"]
        if panel_mode == "r2_rel":
            return "Spatial reliability"
        if panel_mode == "r2_rel_prct":
            return "Reliability percentile"
        return r"$R^2$"

    def _ylabel(self, state) -> str:
        panel_mode = state["panel_mode"]
        if panel_mode == "histogram":
            return "Fraction of ROIs"
        if panel_mode == "ecdf":
            return "Cumulative fraction"
        return r"$R^2$"

    def _row_values(self, row: int, state) -> tuple[np.ndarray, np.ndarray] | None:
        """One session's per-ROI ``(R², reliability)`` under the current env mode, None if empty.

        ROIs missing either quantity are dropped from both, so every panel mode describes the same
        set of ROIs regardless of whether it uses reliability.
        """
        env_mode = state["env_mode"]
        if env_mode == "pooled":
            # Every (ROI, environment) pair, the same population r2_kde_pooled is built from.
            r2 = self.r2_slot[row].reshape(-1)
            reliability = self.reliability_slot[row].reshape(-1)
        elif env_mode == "best":
            slot = self.best_env_slot[row]
            if not np.isfinite(slot):
                return None
            r2 = self.r2_slot[row, int(slot)]
            reliability = self.reliability_slot[row, int(slot)]
        elif env_mode == "slot":
            r2 = self.r2_slot[row, state["env_slot"]]
            reliability = self.reliability_slot[row, state["env_slot"]]
        else:
            raise ValueError(f"Invalid env_mode: {env_mode!r}")
        valid = np.isfinite(r2) & np.isfinite(reliability)
        return (r2[valid], reliability[valid]) if np.any(valid) else None

    def _row_kde_curve(self, row: int, state) -> np.ndarray | None:
        """One session's stored E[R² | reliability] under the current env mode, None if it has none."""
        env_mode = state["env_mode"]
        if env_mode == "pooled":
            curve = self.r2_kde_pooled[row]
        elif env_mode == "best":
            slot = self.best_env_slot[row]
            # A session whose best environment is outside the mouse's slot list has no curve.
            if not np.isfinite(slot):
                return None
            curve = self.r2_kde_slot[row, int(slot)]
        elif env_mode == "slot":
            curve = self.r2_kde_slot[row, state["env_slot"]]
        else:
            raise ValueError(f"Invalid env_mode: {env_mode!r}")
        return curve if np.any(np.isfinite(curve)) else None

    def _row_curve(self, row: int, state) -> np.ndarray | None:
        """One session's curve on ``_grid(state)`` under the current panel mode, None if it has none."""
        panel_mode = state["panel_mode"]
        if panel_mode == "r2_rel":
            return self._row_kde_curve(row, state)

        values = self._row_values(row, state)
        if values is None:
            return None
        r2, reliability = values
        grid = self._grid(state)

        if panel_mode == "r2_rel_prct":
            # Rank, mapped to (0, 100) at bin midpoints, is uniform by construction: every x of the
            # regression is backed by the same number of ROIs, which uniform reliability bins are
            # very far from being.
            percentile = 100.0 * (rankdata(reliability) - 0.5) / reliability.size
            return _kde_r2(r2, percentile, grid, bw=state["prct_bandwidth"])["r2_kde_mean"]

        low, high = tuple(state["r2_range"])
        # Clipped rather than masked: badly predicted ROIs belong in the edge bin, and silently
        # dropping them would renormalize the fractions over a session-dependent population.
        clipped = np.clip(r2, low, high)
        if panel_mode == "histogram":
            counts, _ = np.histogram(clipped, bins=np.linspace(low, high, state["r2_bins"] + 1))
            return counts / r2.size
        if panel_mode == "ecdf":
            return np.searchsorted(np.sort(clipped), grid, side="right") / r2.size
        raise ValueError(f"Invalid panel_mode: {panel_mode!r}")

    def _summarizer(self, state):
        """The ROI-collapsing statistic ax[3] uses, as a callable on a 1-D array of R² values."""
        summary_stat = state["summary_stat"]
        if summary_stat == "mean":
            return np.mean
        if summary_stat == "median":
            return np.median
        if summary_stat == "percentile":
            percentile = state["summary_percentile"]
            return lambda values: np.percentile(values, percentile)
        raise ValueError(f"Invalid summary_stat: {summary_stat!r}")

    def _summary_label(self, state) -> str:
        """Y-axis label of ax[3], naming the statistic (percentiles carry their level)."""
        if state["summary_stat"] == "percentile":
            return f"{ordinal(state['summary_percentile'])} pct " + r"$R^2$"
        return f"{state['summary_stat'].capitalize()} " + r"$R^2$"

    def refresh_data(self, state):
        """Rebuild every curve on the figure: the example mouse's, the population's, and the slots'.

        Syd re-runs ``plot`` on every widget change, so the kernel regressions and per-slot
        summaries are computed here -- once per data change -- rather than on every slider drag.
        """
        self.grid = self._grid(state)
        self.mouse_curves = [self._row_curve(row, state) for row in self._rows_by_mouse[state["mouse"]]]
        self.by_session = by_session_curves(
            [[self._row_curve(row, state) for row in self._rows_by_mouse[mouse]] for mouse in self.mice],
            len(self.grid),
        )

        # Per-slot (mice, sessions) summary: a mouse's curve for slot j is the summary stat over
        # ROIs of every session in which it ran that environment, in chronological order -- so x is
        # "how many sessions of this environment the mouse has had", not the session number.
        summarize = self._summarizer(state)
        self.slot_stacks: dict[int, np.ndarray] = {}
        for slot in range(self.num_slots):
            per_mouse = []
            for mouse in self.mice:
                values = []
                for row in self._rows_by_mouse[mouse]:
                    r2 = self.r2_slot[row, slot]
                    r2 = r2[np.isfinite(r2)]
                    if r2.size:
                        values.append(float(summarize(r2)))
                per_mouse.append(np.array(values))
            self.slot_stacks[slot] = pad_stack(per_mouse)

    # -------------------------------------------------------------------- drawing --

    def _draw_slot_panel(self, ax, state, fontsize) -> tuple[list[float] | None, int]:
        """ax[3]: the summary stat against environment experience, one curve per slot."""
        slot_ymax, slot_xmax = -np.inf, 0
        for slot, stack in self.slot_stacks.items():
            length = support_length(stack)
            if length == 0:
                continue
            stack = stack[:, :length]
            render_curve_group(
                ax,
                np.arange(1, length + 1),
                stack,
                env_slot_color(slot),
                state["slot_style"],
                label=ordinal(slot + 1),
                hide_error=state["hide_error"],
                linewidth=1.5,
            )
            bounds = curve_group_bounds(stack, state["slot_style"])
            if bounds is not None:
                slot_ymax = max(slot_ymax, bounds[1])
            slot_xmax = max(slot_xmax, length)

        ax.set_xlabel("Env session #", fontsize=fontsize)
        ax.set_ylabel(self._summary_label(state), fontsize=fontsize)
        if state["show_slot_legend"]:
            legend = ax.legend(fontsize=fontsize, frameon=False, handlelength=0.8, handletextpad=0.5, title="Env")
            legend.get_title().set_fontsize(fontsize)
        # Anchored at zero like the curve panels, so the two ranges are comparable at a glance.
        return ([0.0, slot_ymax] if np.isfinite(slot_ymax) else None), slot_xmax

    def plot(self, state):
        fontsize = state["fontsize"]
        cmap = mpl.colormaps[state["cmap"]]
        panel_mode = state["panel_mode"]
        grid = self.grid
        xlow, xhigh = self._xlimits(state)
        in_range = (grid >= xlow) & (grid <= xhigh)

        # Every panel mode's quantity is anchored at zero, so only the top of the shared y-axis is
        # fit to the curves.
        ymax = -np.inf

        def track(curve) -> None:
            """Grow the shared y limit to cover a curve's max inside the visible x range."""
            nonlocal ymax
            visible = np.asarray(curve)[in_range]
            visible = visible[np.isfinite(visible)]
            if visible.size:
                ymax = max(ymax, float(np.max(visible)))

        width_ratios = [1.0, 1.0, 1.0, 1.0] if state["square_panels"] else [0.8, 1.2, 1.2, 1.0]
        fig, ax = self.new_subplots(1, 4, figsize=self.figsize, layout="constrained", width_ratios=width_ratios)
        # ax[1] and ax[2] draw the same quantity, so the y-axis is drawn once, on ax[1].
        ax[2].sharey(ax[1])

        # ---- ax[0]: one ROI of one session, activity vs place-field prediction ----
        # Every predictable frame of the session, not just one environment's: this panel is about
        # how well a place field describes a cell at all, which is not a per-environment question.
        example_row = self._example_rows_by_mouse[state["mouse"]][state["example_session"]]
        example_data, example_prediction = self._example_roi_traces(example_row, state["roi"])
        draw_example_roi_panel(
            ax[0],
            example_data,
            example_prediction,
            style=state["example_style"],
            alpha=state["example_alpha"],
            fontsize=fontsize,
            ylabel="PF Prediction",
        )

        # ---- ax[1]: every session of the example mouse, colored by session number ----
        num_sessions = len(self.mouse_curves)
        for isession, curve in enumerate(self.mouse_curves):
            if curve is None:
                continue
            color = cmap(isession / max(num_sessions - 1, 1))
            ax[1].plot(grid, curve, color=color, linewidth=state["linewidth"], alpha=state["alpha"])
            track(curve)
        ax[1].set_xlabel(self._xlabel(state), fontsize=fontsize)
        ax[1].set_ylabel(self._ylabel(state), fontsize=fontsize)

        # ---- ax[2]: the population, mouse-averaged overall or grouped by session number ----
        for mean_curve in draw_session_curve_groups(ax[2], grid, self.by_session, state, fontsize):
            track(mean_curve)
        ax[2].set_xlabel(self._xlabel(state), fontsize=fontsize)

        for axis in (ax[1], ax[2]):
            axis.set_xlim(xlow, xhigh)
        # ymax stays infinite when nothing was plotted -- an env slot no session of this mouse ran,
        # with a curve_mode that also produced nothing.
        if panel_mode == "ecdf":
            # A cumulative fraction spans its own axis; r2_ylim is about R².
            ybounds = [0.0, 1.0]
        elif panel_mode == "histogram" or state["auto_ylim"]:
            ybounds = [0.0, ymax] if np.isfinite(ymax) else None
        else:
            ybounds = list(state["r2_ylim"])

        # ---- ax[3]: summary R² against experience, one curve per environment slot ----
        slot_ybounds, slot_xmax = self._draw_slot_panel(ax[3], state, fontsize)

        # Two nearly-equal y ranges side by side read as one axis but aren't, which is exactly the
        # comparison the eye gets wrong. Within the tolerance they are made into one for real.
        if ybounds is not None and slot_ybounds is not None:
            higher = max(ybounds[1], slot_ybounds[1])
            if higher > 0 and min(ybounds[1], slot_ybounds[1]) >= (1.0 - _SHARE_Y_TOLERANCE) * higher:
                ybounds = [0.0, higher]
                slot_ybounds = [0.0, higher]
                ax[3].sharey(ax[1])

        if ybounds is not None:
            ax[1].set_ylim(ybounds)
        style_axis(
            ax[1],
            fontsize=fontsize,
            xbounds=[xlow, xhigh],
            ybounds=ybounds,
            yticks=decimal_yticks(*ybounds) if ybounds is not None else None,
        )
        # ax[2] borrows ax[1]'s y-axis, so it only gets a bottom spine. Its ticks are hidden with
        # tick_params rather than set_yticks, which would propagate through the shared axis and
        # strip ax[1]'s y ticks too.
        style_axis(ax[2], fontsize=fontsize, xbounds=[xlow, xhigh], spines_visible=["bottom"])
        ax[2].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        if slot_xmax:
            ax[3].set_xlim(1, slot_xmax)
        if slot_ybounds is not None:
            ax[3].set_ylim(slot_ybounds)
        style_axis(
            ax[3],
            fontsize=fontsize,
            xbounds=[1, max(slot_xmax, 1)],
            ybounds=slot_ybounds,
            xticks=[1, max(slot_xmax, 1)],
            yticks=decimal_yticks(*slot_ybounds) if slot_ybounds is not None else None,
        )

        # Last, because it measures the drawn figure: everything that takes up room must be on it.
        if state["square_panels"]:
            fit_square_panels(fig, ax)
        return fig
