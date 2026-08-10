"""One ROI's place field: the trial-by-position map, its reliability, prediction, and error."""

import numpy as np

from vrAnalysis.helpers.vrsupport import _jit_reliability_loo
from vrAnalysis.processors.spkmaps import Maps

from dimensionality_manuscript.configs.placefield_structure import CrossValidatedPlacefieldsConfig
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts import session_cache
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
)
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import draw_vertical_colorscale, style_axis


def _trial_consistency(spkmap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-trial leave-one-out consistency of one ROI's spike map, and the reliability it averages to.

    Trials are weighted by their RMS activity (normalized to the strongest trial), so a silent
    trial -- whose correlation with the others is noise -- contributes nothing. Trials with zero
    weight are dropped outright.

    Parameters
    ----------
    spkmap : np.ndarray
        ``(trials, positions)`` activity of one ROI in one environment.

    Returns
    -------
    trial_numbers, trial_weights, trial_consistency, reliability
    """
    trial_weights = np.sqrt(np.mean(spkmap**2, axis=1))
    trial_consistency = _jit_reliability_loo(spkmap[None, ...])[0]
    trial_weights = trial_weights / np.max(trial_weights)

    idx_include = trial_weights > 0
    trial_numbers = np.arange(spkmap.shape[0])[idx_include]
    trial_weights = trial_weights[idx_include] / np.max(trial_weights[idx_include])
    trial_consistency = trial_consistency[idx_include]
    reliability = float(np.sum(trial_weights * trial_consistency) / np.sum(trial_weights))
    return trial_numbers, trial_weights, trial_consistency, reliability


class PlaceFieldFocus(FigureViewer):
    """One ROI's spike map with its place field below and its trial consistency beside it.

    The two-column ancestor of :class:`PlaceFieldPredictionFocus`, and the only panel here that
    is driven by a :class:`~vrAnalysis.sessions.B2Session` directly rather than by stored
    results -- so it can be pointed at any session, computed or not.

    The ROI selector offers only ROIs that clear both filters in the selected environment, so
    changing ``env`` or either threshold can move the selection.

    Parameters
    ----------
    session : B2Session
        Session the maps, reliability, and fraction-active are computed from (cached).
    roi : int
        Initial ROI, in session-filtered ROI coordinates. Ignored if it fails the filters.
    env : int
        Index into ``env_maps.environments``.
    reliability_threshold, fraction_active_threshold : float
        An ROI is offered only if it exceeds both in the selected environment.
    vmax : float
        Upper limit (in sigma) of the gray_r colorscale.
    fontsize : float
        Font size of every text element.
    figsize : tuple[float, float]
        Figure size in inches.
    """

    def __init__(
        self,
        session,
        *,
        roi: int = 0,
        env: int = 0,
        reliability_threshold: float = 0.7,
        fraction_active_threshold: float = 0.5,
        vmax: float = 5.0,
        fontsize: float = 12.0,
        figsize: tuple[float, float] = (5.0, 6.0),
    ):
        self.session = session
        self.figsize = figsize
        self.env_maps = session_cache.get_env_maps(session)
        self.reliability = session_cache.get_reliability(session)
        self.fraction_active = session_cache.get_fraction_active(session)
        self.dist_edges = session_cache.get_smp(session).dist_edges

        self.add_integer("env", value=env, min=0, max=len(self.env_maps.environments) - 1)
        self.add_float("reliability_threshold", value=reliability_threshold, min=0.0, max=1.0)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0)
        # Options are narrowed to the ROIs passing the filters by update_filters, below.
        self.add_selection("roi", value=roi, options=[roi])
        self.add_float("vmax", value=vmax, min=1.0, max=20.0)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)

        self.on_change(["env", "reliability_threshold", "fraction_active_threshold"], self.update_filters)
        self.on_change(["env", "roi"], self.refresh_data)
        self.update_filters(self.state)

    def update_filters(self, state):
        """Offer only ROIs that clear both filters in the selected environment."""
        env = state["env"]
        idx_reliable = self.reliability.values[env] > state["reliability_threshold"]
        idx_active = self.fraction_active[env] > state["fraction_active_threshold"]
        options = [int(roi) for roi in np.where(idx_reliable & idx_active)[0]]
        if not options:
            raise ValueError(f"No ROI of environment {env} passes both thresholds.")
        current = state["roi"] if state["roi"] in options else options[0]
        self.update_selection("roi", value=current, options=options)
        self.refresh_data({**state, "roi": current})

    def refresh_data(self, state):
        """Select one ROI's spike map and derive its place field and trial consistency."""
        self.spkmap = self.env_maps.spkmap[state["env"]][state["roi"]]
        self.placefield = np.nanmean(self.spkmap, axis=0)
        self.consistency = _trial_consistency(self.spkmap)

    def plot(self, state):
        fontsize = state["fontsize"]
        vmax = state["vmax"]
        spkmap = self.spkmap
        trial_numbers, trial_weights, trial_consistency, reliability = self.consistency

        xlims = [self.dist_edges[0], self.dist_edges[-1]]
        ylims = (spkmap.shape[0] + 0.5, -0.5)
        extent = (0, spkmap.shape[1], spkmap.shape[0], 0)
        ymax_pf = np.nanmax(self.placefield) * 1.2

        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[6, 1])
        ax_spkmap = fig.add_subplot(gs[0, 0])
        ax_placefield = fig.add_subplot(gs[1, 0])
        ax_consistency = fig.add_subplot(gs[0, 1])
        ax_reliability = fig.add_subplot(gs[1, 1])

        ax_spkmap.imshow(spkmap, interpolation="none", aspect="auto", cmap="gray_r", vmin=0, vmax=vmax, extent=extent)
        ax_spkmap.set_ylabel("Trials", fontsize=fontsize)
        ax_spkmap.set_xlim(xlims[0], xlims[1])
        ax_spkmap.set_ylim(*ylims)
        style_axis(ax_spkmap, fontsize=fontsize, xbounds=xlims, xticks=[], yticks=[], spines_visible=["left"])

        ax_placefield.plot(self.placefield, color="k", linewidth=1.5)
        ax_placefield.set_facecolor(("black", 0.04))
        ax_placefield.set_xlabel("VR Position", labelpad=-10, fontsize=fontsize)
        ax_placefield.set_xlim(xlims[0], xlims[1])
        ax_placefield.set_ylim(-0.05, ymax_pf)
        ax_placefield.text(xlims[0], ymax_pf, "Place Field", ha="left", va="top", color="k", fontsize=fontsize)
        style_axis(
            ax_placefield,
            fontsize=fontsize,
            y_pos=-0.15,
            xbounds=xlims,
            xticks=xlims,
            yticks=[],
            spines_visible=["bottom"],
        )

        ax_consistency.scatter(trial_consistency, trial_numbers, color="k", s=5, alpha=trial_weights)
        ax_consistency.set_facecolor(("black", 0.04))
        ax_consistency.set_xlim(-1.05, 1.05)
        ax_consistency.set_ylim(*ylims)
        ax_consistency.set_xlabel(r"$\sigma$", fontsize=fontsize)
        ax_consistency.text(
            -0.5,
            max(trial_numbers) / 2,
            r"$\sigma = \mathrm{corr}(\langle\mathrm{other\ trials}\rangle)$",
            ha="center",
            va="center",
            rotation=90,
            fontsize=fontsize,
        )
        style_axis(ax_consistency, fontsize=fontsize, xbounds=(-1, 1), xticks=[-1, 0, 1], yticks=[], spines_visible=["bottom"])

        ax_reliability.plot([-1, 1], [0, 0], color="black", linewidth=1.5)
        ax_reliability.plot([reliability], [0], color="black", marker="o", markersize=8)
        ax_reliability.set_xlim(-1, 1)
        ax_reliability.set_ylim(-0.05, 0.05)
        ax_reliability.set_xlabel("Reliability", fontsize=fontsize)
        style_axis(ax_reliability, fontsize=fontsize, xbounds=(-1, 1), xticks=[-1, 0, 1], yticks=[], spines_visible=["bottom"])

        draw_vertical_colorscale(
            ax_spkmap.inset_axes([0.1, 0.15, 0.075, 0.7]),
            "gray_r",
            low_label="0",
            high_label=f"{int(vmax)}",
            fontsize=fontsize,
            ylabel=r"Fluorescence ($\sigma$)",
        )
        return fig


class PlaceFieldPredictionFocus(FigureViewer):
    """Place field, trial consistency, PF prediction, and prediction error for one ROI/environment.

    Up to four columns, all sharing the VR-position axis (columns 1, 3, 4) and the trial axis (row 1):

    1. Trial-by-position spike map with the place field (trial average) below.
    2. Per-trial consistency with the other trials, with the weighted average (reliability) below.
       Dropped when ``show_consistency`` is False.
    3. The place-field prediction of each trial, with its trial average below. Dropped when
       ``show_prediction`` is False.
    4. Prediction error (prediction - activity), with the per-position RMS error below.

    Turning both toggles off leaves the two-column version: the data and the error.

    The place-field prediction of a trial is the place field itself: ``get_placefield_prediction``
    predicts each frame from the trial-averaged map, so in position units every trial's prediction
    is the same row. Column 3 is therefore the place field tiled across trials (masked where the
    trial has no data), which is what makes column 4 a picture of trial-to-trial variability.

    Parameters
    ----------
    results : ResultsAggregator
        Results whose sessions are available to the mouse/session selectors.
    mouse : str or None
        Example mouse. None takes the first alphabetically.
    example_session : str or None
        Which of that mouse's sessions to use, as ``session_print()`` prints it
        (``"mouse/date/id"``). None takes the mouse's first session.
    roi : int
        ROI index within the session, in session-filtered ROI coordinates -- the same coordinate
        system :class:`~.familiarity.R2Familiarity` uses.
    env : int
        Index into ``env_maps.environments``.
    vmax : float
        Upper limit (in sigma) of the gray_r colorscale for the activity and prediction maps.
    vmax_error : float
        Saturation (in sigma) of the symmetric bwr colorscale for the error map.
    fontsize : float
        Single font size for the whole panel: axis labels, tick labels, in-axes annotations, and
        colorscale labels.
    show_consistency : bool
        Draw the trial-consistency / reliability column (column 2).
    show_prediction : bool
        Draw the place-field prediction column (column 3).
    use_trial_weights : bool
        Weight the trial-consistency scatter by each trial's RMS activity, so silent trials
        don't show up in the plot.
    figsize : tuple[float, float]
        Figure size in inches. Not rescaled when columns are dropped, so the remaining columns get
        wider; pass a narrower width for the two-column (activity + error) version.
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
        env: int = 0,
        vmax: float = 5.0,
        vmax_error: float = 5.0,
        fontsize: float = 12.0,
        show_consistency: bool = True,
        show_prediction: bool = True,
        use_trial_weights: bool = True,
        figsize: tuple[float, float] = (12.0, 6.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self._example_config = CrossValidatedPlacefieldsConfig()

        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(mouse_name) for mouse_name in mouse_names})
        self._sessions_by_mouse = {
            mouse_name: {
                results.sessions[row].session_print(): results.sessions[row]
                for row in sorted(
                    np.flatnonzero(mouse_names == mouse_name),
                    key=lambda row: _session_sort_key(results.sessions[row]),
                )
            }
            for mouse_name in self.mice
        }
        initial_mouse = mouse if mouse is not None else self.mice[0]
        initial_sessions = list(self._sessions_by_mouse[initial_mouse])
        initial_session = example_session if example_session is not None else initial_sessions[0]

        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("mouse", value=initial_mouse, options=self.mice)
        self.add_selection("example_session", value=initial_session, options=initial_sessions)
        # Bounds are set by update_example_bounds once the session's maps are known.
        self.add_integer("roi", value=roi, min=0, max=max(roi, 0))
        self.add_integer("env", value=env, min=0, max=max(env, 0))

        self.add_float("vmax", value=vmax, min=1.0, max=20.0)
        self.add_float("vmax_error", value=vmax_error, min=1.0, max=20.0)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.add_boolean("show_consistency", value=show_consistency)
        self.add_boolean("show_prediction", value=show_prediction)
        self.add_boolean("use_trial_weights", value=use_trial_weights)

        self.on_change("mouse", self.update_example_session)
        self.on_change("example_session", self.update_example_bounds)
        self.on_change(["roi", "env"], self.refresh_data)
        self.update_example_session(self.state)

    def update_example_session(self, state):
        """Offer the selected mouse's sessions, in chronological order."""
        labels = list(self._sessions_by_mouse[state["mouse"]])
        current = state["example_session"] if state["example_session"] in labels else labels[0]
        self.update_selection("example_session", value=current, options=labels)
        self.update_example_bounds({**state, "example_session": current})

    def update_example_bounds(self, state):
        """Set ROI/environment bounds for the selected session, then reload its maps."""
        env_maps = self._get_env_maps(state)
        self.update_integer("roi", max=max(env_maps.spkmap[0].shape[0] - 1, 0))
        self.update_integer("env", max=max(len(env_maps.environments) - 1, 0))
        self.refresh_data(self.state)

    def _get_env_maps(self, state) -> Maps:
        """Maps in the same session-filtered ROI coordinates :class:`R2Familiarity` uses."""
        session = self._sessions_by_mouse[state["mouse"]][state["example_session"]]
        previous_spks_type = session.params.spks_type
        session.params.spks_type = self._example_config.spks_type
        try:
            return session_cache.get_env_maps(session)
        finally:
            session.params.spks_type = previous_spks_type

    def refresh_data(self, state):
        """Select one ROI's spike map and derive the place field, its prediction, and the error."""
        env_maps = self._get_env_maps(state)
        self.distcenters = env_maps.distcenters
        self.spkmap = env_maps.spkmap[state["env"]][state["roi"]]
        self.placefield = np.nanmean(self.spkmap, axis=0)

        # Every trial is predicted by the same place field; mask where the trial has no data.
        self.pred_spkmap = np.broadcast_to(self.placefield, self.spkmap.shape).copy()
        self.pred_spkmap[np.isnan(self.spkmap)] = np.nan
        self.error = self.spkmap - self.pred_spkmap
        self.avg_prediction = np.nanmean(self.pred_spkmap, axis=0)
        self.rms_error = np.sqrt(np.nanmean(self.error**2, axis=0))
        self.consistency = _trial_consistency(self.spkmap)

    def plot(self, state):
        vmax = state["vmax"]
        vmax_error = state["vmax_error"]
        fontsize = state["fontsize"]
        show_consistency = state["show_consistency"]
        show_prediction = state["show_prediction"]

        distcenters = self.distcenters
        spkmap = self.spkmap

        xlims = [distcenters[0], distcenters[-1]]
        ylims = [spkmap.shape[0] + 0.5, -0.5]
        extent = (xlims[0], xlims[1], spkmap.shape[0], 0)
        xlims_clean = (np.round(xlims[0] / 50) * 50, np.round(xlims[1] / 50) * 50)
        xlabels = [f"{int(round(x))}" for x in xlims_clean]
        # One y range for the three line panels so place field, prediction, and error are comparable.
        ymax_pf = np.nanmax([np.nanmax(self.placefield), np.nanmax(self.avg_prediction), np.nanmax(self.rms_error)]) * 1.2

        # Columns are added left to right, so the optional ones only shift what follows them.
        width_ratios = [1]
        icol_consistency = None
        icol_prediction = None
        if show_consistency:
            icol_consistency = len(width_ratios)
            width_ratios.append(0.25)
        if show_prediction:
            icol_prediction = len(width_ratios)
            width_ratios.append(1)
        icol_error = len(width_ratios)
        width_ratios.append(1)

        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(2, len(width_ratios), width_ratios=width_ratios, height_ratios=[6, 1])
        ax_spkmap = fig.add_subplot(gs[0, 0])
        ax_placefield = fig.add_subplot(gs[1, 0])
        ax_error = fig.add_subplot(gs[0, icol_error])
        ax_rms_error = fig.add_subplot(gs[1, icol_error])

        def draw_map(ax, values, cmap, vlow, vhigh, spines_visible):
            """One trial-by-position image, sharing the trial axis and the cleaned position axis."""
            ax.imshow(values, interpolation="none", aspect="auto", cmap=cmap, vmin=vlow, vmax=vhigh, extent=extent)
            ax.set_xlim(xlims_clean)
            ax.set_ylim(ylims[0], ylims[1])
            style_axis(ax, fontsize=fontsize, xbounds=xlims_clean, xticks=[], yticks=[], spines_visible=spines_visible)

        def draw_curve(ax, values, label, ylabel=None, include_left_spine=False):
            """One position-axis curve below a map, on the shared y range, labeled in-axes."""
            ax.plot(distcenters, values, color="k", linewidth=1.5)
            ax.set_facecolor(("black", 0.04))
            ax.set_xlabel("VR Position", labelpad=-10, fontsize=fontsize)
            ax.set_xlim(xlims_clean)
            ax.set_ylim(-0.05, ymax_pf)
            ax.text(xlims[0], ymax_pf, label, ha="left", va="top", color="k", fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=-2)

            if include_left_spine:
                spines_visible = ["bottom", "left"]
                yticks = [0, int(np.fix(ymax_pf))]
                ybounds = (0, ymax_pf)
            else:
                spines_visible = ["bottom"]
                yticks = []
                ybounds = (-0.05, ymax_pf)

            style_axis(
                ax,
                fontsize=fontsize,
                y_pos=-0.15,
                xbounds=xlims_clean,
                xticks=xlims_clean,
                xlabels=xlabels,
                ybounds=ybounds,
                yticks=yticks,
                spines_visible=spines_visible,
            )

        # ------------------------------------------------------------- col 1: activity --
        draw_map(ax_spkmap, spkmap, "gray_r", 0, vmax, ["left"])
        ax_spkmap.set_ylabel("Trials", fontsize=fontsize)
        draw_curve(ax_placefield, self.placefield, "Placefield", r"$\sigma$", include_left_spine=True)

        # ---------------------------------------------------------- col 2: consistency --
        if show_consistency:
            self._draw_consistency(fig, gs, icol_consistency, state, ylims, fontsize)

        # ----------------------------------------------------------- col 3: prediction --
        if show_prediction:
            draw_map(fig.add_subplot(gs[0, icol_prediction]), self.pred_spkmap, "gray_r", 0, vmax, [])
            draw_curve(fig.add_subplot(gs[1, icol_prediction]), self.avg_prediction, "PF Prediction")

        # ---------------------------------------------------------------- col 4: error --
        draw_map(ax_error, self.error, "bwr", -vmax_error, vmax_error, [])
        draw_curve(ax_rms_error, self.rms_error, "RMS Error")

        # --------------------------------------------------------- inset colorscales --
        draw_vertical_colorscale(
            ax_spkmap.inset_axes([0.225, 0.15, 0.125, 0.7]),
            "gray_r",
            low_label="0",
            high_label=f"{int(vmax)}",
            fontsize=fontsize,
            ylabel=r"Fluorescence ($\sigma$)",
        )
        draw_vertical_colorscale(
            ax_error.inset_axes([0.225, 0.15, 0.125, 0.7]),
            "bwr",
            low_label=f"-{int(vmax_error)}",
            high_label=f"{int(vmax_error)}",
            low_color="w",
            fontsize=fontsize,
            ylabel=r"Error ($\sigma$)",
        )
        return fig

    def _draw_consistency(self, fig, gs, icol, state, ylims, fontsize):
        """Column 2: per-trial consistency with the other trials, and the reliability it averages to."""
        ax_consistency = fig.add_subplot(gs[0, icol])
        ax_reliability = fig.add_subplot(gs[1, icol])
        trial_numbers, trial_weights, trial_consistency, reliability = self.consistency

        alpha = trial_weights if state["use_trial_weights"] else 1.0
        ax_consistency.scatter(trial_consistency, trial_numbers, color="k", s=1.75, alpha=alpha)
        ax_consistency.set_facecolor(("black", 0.04))
        ax_consistency.set_xlim(-1.1, 1.1)
        ax_consistency.set_ylim(ylims[0], ylims[1])
        ax_consistency.text(
            -0.5,
            max(trial_numbers) / 2,
            r"$r = \mathrm{corr}(\langle\mathrm{other}\rangle)$",
            ha="center",
            va="center",
            rotation=90,
            fontsize=fontsize,
        )
        style_axis(ax_consistency, fontsize=fontsize, xbounds=(-1, 1), xticks=[], yticks=[], spines_visible=[])

        ax_reliability.plot([-1, 1], [0, 0], color="black", linewidth=1.5)
        ax_reliability.plot([reliability], [0], color="black", marker="o", markersize=2.5)
        ax_reliability.set_xlim(-1.1, 1.1)
        ax_reliability.set_ylim(-0.05, 0.15)
        ax_reliability.set_xlabel(r"$r$", fontsize=fontsize, labelpad=-2)
        ax_reliability.text(-1.05, 0.15, "Avg.", ha="left", va="top", color="k", fontsize=fontsize)
        # ax_reliability.set_xlabel("Spatial\nReliability", fontsize=fontsize)
        style_axis(ax_reliability, fontsize=fontsize, xbounds=(-1, 1), xticks=[-1, 1], yticks=[], spines_visible=["bottom"])
