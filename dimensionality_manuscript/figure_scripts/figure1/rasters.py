"""Stacked rasters of one session: activity, its place-field prediction, and the residual."""

import numpy as np
from rastermap import Rastermap

from dimensionality_manuscript.configs.placefield_structure import PlacefieldPredictionConfig
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts import session_cache
from dimensionality_manuscript.figure_scripts.panels import FigureViewer, add_data_selection_widgets
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import (
    COLORSCALE_INSET_RECT,
    add_colorscale_widgets,
    colorscale_inset_rect,
    draw_colorscale_inset,
    env_slot_color,
)


class StackedRasterFocus(FigureViewer):
    """Three stacked rasters of one session: activity, PF prediction, and their residual.

    Every ROI that passes the reliability filter contributes one row, in a shared sort order, to
    all three rasters -- so a structure visible in the top panel and absent from the middle one
    is exactly what the bottom panel shows. Activity and prediction are in units of each ROI's
    standard deviation in time (the ``spks_std`` stored alongside the prediction), which is what
    makes one ``vmax`` meaningful across ROIs.

    An optional top panel draws the mouse's position over the plotted frames, one colored
    trace per environment (:data:`~dimensionality_manuscript.env_order.ENV_SLOT_COLORS`, shared
    with the VR schematic), so the position structure behind the rasters is visible.

    Frames without a prediction are dropped before anything is plotted, so the x axis is
    plotted-frame index, not time: the optional scalebar measures imaging time, not the
    wall-clock span of the slice.

    Parameters
    ----------
    results : ResultsAggregator
        Results collection whose store contains ``PlacefieldPredictionConfig`` schema-v1 results.
    mouse : str or None
        Initial mouse selection. None uses the first mouse with a stored prediction.
    session : str or None
        Initial printable session label (``mouse/date/id``). None uses the selected mouse's first
        session with a stored prediction.
    sort_method : {"environment", "rastermap"}
        ROI ordering of every raster. ``environment`` sorts by preferred environment then
        place-field position; ``rastermap`` sorts by a Rastermap embedding of the activity, which
        is position-agnostic and takes tens of seconds the first time a given ROI subset is fit.
        Either way the sort is computed on the activity, not the prediction.
    use_reliable : bool
        Keep only ROIs reliable in at least one environment.
    reliability_threshold : float
        Reliability an ROI must exceed in some environment to be kept.
    xslice : slice
        Range of plotted frames. Clamped to the session's frame count on load.
    vmax : float
        Upper limit (in sigma) of the gray_r rasters; the residual raster uses ``+/- vmax``.
    show_position : bool
        Add the position panel above the rasters.
    position_height : float
        Height ratio of the position panel relative to each raster.
    env_gap : float
        Extra vertical gap between environment bands in the position panel, as a fraction of the
        position-bin count.
    show_zero_sigma : bool
        Label the low end of the gray_r colorscale with ``0 sigma``.
    show_scalebar : bool
        Draw a time scalebar on the residual raster, mirroring the colorscale inset's placement.
    scalebar_seconds : float
        Duration of the scalebar, in seconds of imaging time.
    fontsize : float
        Font size of every text element: panel titles, axis labels, colorscale end labels, and
        the scalebar label.
    figsize : tuple[float, float]
        Figure size in inches.
    colorscale_text_y : float
        Vertical position of the colorscale end labels, in inset-axes coordinates.
    colorscale_inset_rect : tuple of float
        ``(x, y, width, height)`` of the colorscale insets, in parent-axes coordinates.
    **selection_defaults
        Starting values for the aggregator's own param axes, by name.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        session: str | None = None,
        sort_method: str = "environment",
        use_reliable: bool = True,
        reliability_threshold: float = 0.7,
        xslice: slice = slice(0, 2000),
        vmax: float = 6.0,
        show_position: bool = False,
        position_height: float = 0.5,
        env_gap: float = 0.2,
        show_zero_sigma: bool = False,
        show_scalebar: bool = False,
        scalebar_seconds: float = 60.0,
        fontsize: float = 8.0,
        figsize: tuple[float, float] = (12.0, 6.0),
        colorscale_text_y: float = 0.5,
        colorscale_inset_rect: tuple[float, float, float, float] = COLORSCALE_INSET_RECT,
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self._config = PlacefieldPredictionConfig()

        store_rows = results.store.summary_table(
            analysis_type=self._config.display_name,
            session_ids=list(results.session_ids),
            schema_version=self._config.schema_version,
        )
        self._store_rows = {row["session_id"]: row for row in store_rows if row["analysis_key"] == self._config.key()}
        mouse_names = np.asarray(results.mouse_names)
        rows_by_mouse = {
            str(mouse_name): [
                int(row)
                for row in sorted(np.flatnonzero(mouse_names == mouse_name), key=lambda row: _session_sort_key(results.sessions[row]))
                if results.sessions[row].session_uid in self._store_rows
            ]
            for mouse_name in sorted(set(mouse_names))
        }
        self._rows_by_mouse = {name: rows for name, rows in rows_by_mouse.items() if rows}
        if not self._rows_by_mouse:
            raise ValueError("No PlacefieldPredictionConfig schema-v1 results are stored for these sessions.")
        self._session_rows = {name: {results.sessions[row].session_print(): row for row in rows} for name, rows in self._rows_by_mouse.items()}
        initial_mouse = mouse if mouse in self._rows_by_mouse else next(iter(self._rows_by_mouse))
        session_options = list(self._session_rows[initial_mouse])
        initial_session = session if session in session_options else session_options[0]

        self._result_cache: dict[int, dict] = {}
        self._sort_cache: dict[tuple, np.ndarray] = {}

        # --- data selection ---
        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("mouse", value=initial_mouse, options=list(self._rows_by_mouse))
        self.add_selection("session", value=initial_session, options=session_options)
        self.add_selection("sort_method", value=sort_method, options=list(session_cache.SORT_METHODS))
        self.add_boolean("use_reliable", value=use_reliable)
        self.add_float("reliability_threshold", value=reliability_threshold, min=0.0, max=1.0)

        # --- what is drawn ---
        start = max(0, xslice.start if xslice.start is not None else 0)
        stop = xslice.stop if xslice.stop is not None else 2000
        self.add_integer("xslice_start", value=start, min=0, max=start)
        self.add_integer("xslice_stop", value=max(start + 1, stop), min=1, max=max(start + 1, stop))
        self.add_float("vmax", value=vmax, min=1.0, max=20.0)
        self.add_boolean("show_position", value=show_position)
        self.add_float("position_height", value=position_height, min=0.1, max=3.0)
        self.add_float("env_gap", value=env_gap, min=0.0, max=3.0)
        self.add_boolean("show_zero_sigma", value=show_zero_sigma)
        self.add_boolean("show_scalebar", value=show_scalebar)
        self.add_float("scalebar_seconds", value=scalebar_seconds, min=1.0, max=600.0)

        # --- style ---
        self.add_float("fontsize", value=fontsize, min=1.0, max=30.0)
        add_colorscale_widgets(self, colorscale_text_y=colorscale_text_y, colorscale_inset_rect=colorscale_inset_rect)

        self.on_change("mouse", self.update_session)
        self.on_change([*self.selection_names, "session"], self.load_session)
        self.on_change(["sort_method", "use_reliable", "reliability_threshold"], self.refresh_data)
        self.on_change("xslice_start", self.update_xslice_bounds)
        self.load_session(self.state)

    # ---------------------------------------------------------------- data selection --

    def update_session(self, state):
        """Offer only sessions of the selected mouse with a stored PF prediction."""
        options = list(self._session_rows[state["mouse"]])
        current = state["session"] if state["session"] in options else options[0]
        self.update_selection("session", value=current, options=options)
        self.load_session({**state, "session": current})

    def update_xslice_bounds(self, state):
        """Keep the slice stop strictly after the slice start."""
        self.update_integer("xslice_stop", min=state["xslice_start"] + 1)

    def load_session(self, state):
        """Load activity and the stored PlacefieldPredictionConfig result for one session."""
        row = self._session_rows[state["mouse"]][state["session"]]
        self.session = self.results.sessions[row]
        if row not in self._result_cache:
            result = self._config.get_result(self.results.store, self._store_rows[self.session.session_uid])
            if result is None or "placefield_prediction" not in result or "spks_std" not in result:
                raise KeyError(
                    f"{self.session.session_print()} lacks placefield_prediction or spks_std " "(PlacefieldPredictionConfig schema v1 required)."
                )
            self._result_cache[row] = result
        result = self._result_cache[row]

        previous_spks_type = self.session.params.spks_type
        self.session.params.spks_type = self._config.spks_type
        try:
            raw_roi_indices = np.flatnonzero(self.session.idx_rois)
            raw_spks = np.asarray(self.session.spks)[:, raw_roi_indices]
            scale = np.asarray(result["spks_std"])[raw_roi_indices]
            self.spks = np.divide(
                raw_spks,
                scale[None, :],
                out=np.full(raw_spks.shape, np.nan, dtype=float),
                where=np.isfinite(scale[None, :]) & (scale[None, :] > 0),
            )
            prediction = np.asarray(result["placefield_prediction"])[:, raw_roi_indices]
            num_frames = min(self.spks.shape[0], prediction.shape[0])
            self.spks = self.spks[:num_frames]
            prediction = prediction[:num_frames]
            prediction = np.divide(
                prediction,
                scale[None, :],
                out=np.full(prediction.shape, np.nan, dtype=float),
                where=np.isfinite(scale[None, :]) & (scale[None, :] > 0),
            )
            self.smp = session_cache.get_smp(self.session)
            self._reliability = session_cache.get_reliability(self.session)
            frame_position, _, frame_environment, _ = self.smp.get_frame_behavior()
            frame_position = frame_position[:num_frames]
            frame_environment = frame_environment[:num_frames]
        finally:
            self.session.params.spks_type = previous_spks_type

        idx_valid = np.any(np.isfinite(prediction), axis=1)
        self.spks_valid = self.spks[idx_valid]
        self._stored_prediction_valid = prediction[idx_valid]
        self.num_valid = int(np.sum(idx_valid))
        num_bins = len(self.smp.dist_edges) - 1
        env_length = float(self.session.env_length[0])
        self.frame_position = frame_position[idx_valid] / env_length * num_bins
        env_values = sorted(np.unique(frame_environment[idx_valid][np.isfinite(frame_environment[idx_valid])]))
        env_to_slot = {env: slot for slot, env in enumerate(env_values)}
        self.frame_environment = np.array([env_to_slot.get(env, -1) for env in frame_environment[idx_valid]])
        self.frame_period = float(np.median(np.diff(self.session.timestamps)))

        start = int(np.clip(state["xslice_start"], 0, max(self.num_valid - 1, 0)))
        stop = int(np.clip(state["xslice_stop"], start + 1, max(self.num_valid, start + 1)))
        self.update_integer("xslice_start", value=start, min=0, max=max(self.num_valid - 1, 0))
        self.update_integer("xslice_stop", value=stop, min=start + 1, max=max(self.num_valid, start + 1))
        self.refresh_data({**state, "xslice_start": start, "xslice_stop": stop})

    def refresh_data(self, state):
        """Rebuild the ROI-filtered, ROI-sorted (rois, frames) rasters. Slicing them is then a view."""
        if state["use_reliable"]:
            threshold = state["reliability_threshold"]
            idx_reliable = np.where(np.any(np.stack([rval > threshold for rval in self._reliability.values], axis=0), axis=0))[0]
        else:
            idx_reliable = np.arange(self.spks.shape[1])

        sort_key = (self.session.session_uid, state["sort_method"], idx_reliable.tobytes())
        if sort_key not in self._sort_cache:
            if state["sort_method"] == "environment":
                previous_spks_type = self.session.params.spks_type
                self.session.params.spks_type = self._config.spks_type
                try:
                    self._sort_cache[sort_key] = session_cache.get_roi_sort(self.session, "environment", idx_reliable)
                finally:
                    self.session.params.spks_type = previous_spks_type
            else:
                self._sort_cache[sort_key] = Rastermap().fit(self.spks_valid[:, idx_reliable].T).isort
        idx_sort = self._sort_cache[sort_key]

        self._activity = self.spks_valid[:, idx_reliable].T[idx_sort]
        self._prediction = self._stored_prediction_valid[:, idx_reliable].T[idx_sort]

    # -------------------------------------------------------------------- drawing --

    def _draw_position_panel(self, ax, state, xslice, num_frames):
        """One colored position trace per environment, broken at lap resets."""
        num_bins = len(self.smp.dist_edges) - 1
        fontsize = state["fontsize"]
        frame_position = self.frame_position[xslice]
        frame_environment = self.frame_environment[xslice]
        band = num_bins + state["env_gap"] * num_bins
        xpos_base = np.arange(num_frames, dtype=float)
        # Break the line at lap resets (position wraps back) so vertical jumps disappear.
        lap_resets = np.diff(frame_position) < -num_bins / 2
        breaks = np.where(lap_resets)[0] + 1
        # Plot each environment as its own colored line, offset vertically by a gapped band.
        # NaN-masking the other environments' frames breaks the line between environments.
        for env in np.unique(frame_environment[frame_environment >= 0]):
            y = frame_position.astype(float).copy()
            y[frame_environment != env] = np.nan
            y = y + env * band
            xins = np.insert(xpos_base, breaks, np.nan)
            yins = np.insert(y, breaks, np.nan)
            ax.plot(xins, yins, color=env_slot_color(env), linewidth=1)
        ax.set_ylabel("Pos.", fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yinverted(True)

    def _draw_scalebar(self, ax, state, num_frames):
        """Time scalebar, mirroring the colorscale inset to the other side of the axes.

        Frames are non-contiguous (invalid frames are dropped), so this measures plotted imaging
        time, not the wall-clock time the slice spans.
        """
        seconds = state["scalebar_seconds"]
        rect = colorscale_inset_rect(state)
        bar_width = (seconds / self.frame_period) / num_frames  # axes fraction
        x0 = 1.0 - (rect[0] + rect[2])  # same inset from the edge, mirrored to the left
        ycenter = rect[1] + rect[3] / 2
        label = f"{seconds / 60:g} min" if seconds >= 60 else f"{seconds:g} s"
        ax.plot([x0, x0 + bar_width], [ycenter, ycenter], transform=ax.transAxes, color="k", linewidth=2.5)
        ax.text(
            x0 + bar_width / 2,
            ycenter + 0.05,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=state["fontsize"],
        )

    def plot(self, state):
        xslice = slice(state["xslice_start"], min(state["xslice_stop"], self.num_valid))
        activity = self._activity[:, xslice]
        prediction = self._prediction[:, xslice]
        num_frames = activity.shape[1]
        vmax = state["vmax"]
        fontsize = state["fontsize"]
        show_position = state["show_position"]

        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        if show_position:
            gs = fig.add_gridspec(4, 1, height_ratios=[state["position_height"], 1, 1, 1])
            raster_row = 1
        else:
            gs = fig.add_gridspec(3, 1)
            raster_row = 0
        ax = [fig.add_subplot(gs[raster_row, 0])]
        ax.append(fig.add_subplot(gs[raster_row + 1, 0], sharex=ax[0], sharey=ax[0]))
        ax.append(fig.add_subplot(gs[raster_row + 2, 0], sharex=ax[0], sharey=ax[0]))

        ax[0].imshow(activity, aspect="auto", cmap="gray_r", vmin=0, vmax=vmax)
        ax[1].imshow(prediction, aspect="auto", cmap="gray_r", vmin=0, vmax=vmax)
        ax[2].imshow(activity - prediction, aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)

        panel_titles = (
            "Deconvolved Calcium Activity" if self._config.spks_type == "oasis" else "Fluorescence",
            "Prediction From Place Field",
            "Residuals",
        )
        for a, title in zip(ax, panel_titles):
            a.set_xticks([])
            a.set_yticks([])
            a.set_ylabel("ROIs", fontsize=fontsize)
            a.text(1.0, 1.0, title, transform=a.transAxes, ha="right", va="top", color="black", fontsize=fontsize)
            for spine in a.spines.values():
                spine.set_visible(False)

        # Colorscale insets: gray_r on the first raster, bwr on the last raster.
        rect = colorscale_inset_rect(state)
        draw_colorscale_inset(
            ax[0],
            "gray_r",
            left_label=r"$0\,\sigma$" if state["show_zero_sigma"] else None,
            right_label=rf"${int(vmax)}\,\sigma$",
            left_color="k",
            right_color="w",
            fontsize=fontsize,
            text_y=state["colorscale_text_y"],
            inset_rect=rect,
        )
        draw_colorscale_inset(
            ax[2],
            "bwr",
            left_label=rf"$-{int(vmax)}\,\sigma$",
            right_label=rf"$+{int(vmax)}\,\sigma$",
            left_color="w",
            right_color="w",
            fontsize=fontsize,
            text_y=state["colorscale_text_y"],
            inset_rect=rect,
        )

        if show_position:
            self._draw_position_panel(fig.add_subplot(gs[0, 0], sharex=ax[0]), state, xslice, num_frames)
        if state["show_scalebar"]:
            self._draw_scalebar(ax[2], state, num_frames)

        return fig
