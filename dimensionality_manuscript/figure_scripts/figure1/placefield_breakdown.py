"""Single-row activity, multiplicative placefield prediction, and additive residual."""

import numpy as np
from matplotlib import colormaps

from dimensionality_manuscript.configs.gain_regression import least_squares_gain_matrix
from dimensionality_manuscript.configs.pfpred_quality import _trial_unexplained_fraction
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import add_data_selection_widgets
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import draw_vertical_colorscale, style_axis
from .placefield import PlaceFieldPredictionFocus, _trial_consistency

# Match figure2.model_style.COMPARISON_MODEL_COLORS without loading figure 2's viewers.
_PANEL_LEGENDS = {
    "Placefield map": ("Placefield", "#666666"),
    "Gain × PF": ("Gain x PF\nModel", "#c00000"),
    "Remaining additive component": ("Gain x PF\nResidual", "#c06000"),
}


def _add_breakdown_widgets(
    viewer,
    *,
    include_spatial_reliability,
    include_trial_gain,
    include_residual_rms,
    residual_metric,
    summary_width,
    markersize,
    colorbar_x,
    colorbar_y,
    colorbar_width,
    colorbar_height,
    legend_x,
    legend_y,
    legend_ha,
    legend_va,
):
    """Register the display controls shared by both breakdown viewers."""
    viewer.add_boolean("include_spatial_reliability", value=include_spatial_reliability)
    viewer.add_boolean("include_trial_gain", value=include_trial_gain)
    viewer.add_boolean("include_residual_rms", value=include_residual_rms)
    viewer.add_selection("residual_metric", value=residual_metric, options=["rms", "unexplained_fraction"])
    viewer.add_float("summary_width", value=summary_width, min=0.0, max=1.0, step=0.01)
    viewer.add_float("markersize", value=markersize, min=0.0, max=100.0, step=0.25)
    for name, value in (("x", colorbar_x), ("y", colorbar_y), ("width", colorbar_width), ("height", colorbar_height)):
        viewer.add_float(f"colorbar_{name}", value=value, min=0.001 if name in ("width", "height") else -1.0, max=2.0, step=0.001)
    viewer.add_float("legend_x", value=legend_x, min=-1.0, max=2.0, step=0.01)
    viewer.add_float("legend_y", value=legend_y, min=-1.0, max=2.0, step=0.01)
    viewer.add_selection("legend_ha", value=legend_ha, options=["left", "center", "right"])
    viewer.add_selection("legend_va", value=legend_va, options=["top", "center", "bottom", "baseline", "center_baseline"])


class PlacefieldBreakdownFocus(PlaceFieldPredictionFocus):
    """One ROI/environment, decomposed as activity = gain × PF + residual.

    The mandatory columns are the observed trial-by-position map, the all-trial
    mean placefield multiplied by each trial's least-squares gain (no intercept),
    and their signed difference. This is a descriptive fit, not cross-validation.
    Missing activity bins remain missing in the prediction and residual.

    Optional summary columns follow activity (``include_spatial_reliability``),
    prediction (``include_trial_gain``), and residual (``include_residual_rms``).
    They show leave-one-out spatial similarity, the gain used in the prediction,
    and RMS across finite position bins of the displayed residual, respectively.
    Set ``residual_metric="unexplained_fraction"`` to show per-trial 1 − R² in
    the residual column's scatter, with x limits of −0.05 and 1.05 and ticks at
    0 and 1. The denominator is the trial's centered activity sum of squares over
    those same bins. Constant trials are omitted.
    ``use_trial_weights`` weights spatial-similarity points by RMS activity.
    There is no second row. Activity and prediction share ``vmax``; the residual
    uses a diverging colorscale with limits ±``vmax``.

    ``summary_width`` sets summary-column width relative to each map, from 0 to 1;
    zero hides the summary columns. ``markersize`` is scatter marker area in points².
    ``colorbar_x``, ``colorbar_y``, ``colorbar_width``, and ``colorbar_height``
    set both inset colorscales in their parent axes' coordinates. Map position axes
    run from zero to the support's upper bound rounded up to a multiple of 50 cm.
    ``legend_x``/``legend_y`` place all three colored map labels in axes coordinates;
    ``legend_ha``/``legend_va`` set their horizontal/vertical alignment. The residual
    colorscale is centered on white, and its axes background is white.

    Session, ROI, environment, and data selection follow PlaceFieldPredictionFocus.
    Only local spike maps are needed; stored quality summaries are not read.

    Example::

        viewer = PlacefieldBreakdownFocus(results, include_trial_gain=True)
        viewer.show()
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
        fontsize: float = 12.0,
        include_spatial_reliability: bool = False,
        include_trial_gain: bool = False,
        include_residual_rms: bool = False,
        residual_metric: str = "rms",
        use_trial_weights: bool = True,
        summary_width: float = 0.5,
        markersize: float = 1.75,
        colorbar_x: float = 0.82,
        colorbar_y: float = 0.12,
        colorbar_width: float = 0.045,
        colorbar_height: float = 0.32,
        legend_x: float = 0.02,
        legend_y: float = 0.98,
        legend_ha: str = "left",
        legend_va: str = "top",
        figsize: tuple[float, float] = (12.0, 4.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(name) for name in mouse_names})
        self._sessions_by_mouse = {
            name: {
                results.sessions[row].session_print(): results.sessions[row]
                for row in sorted(
                    np.flatnonzero(mouse_names == name),
                    key=lambda row: _session_sort_key(results.sessions[row]),
                )
            }
            for name in self.mice
        }
        initial_mouse = mouse if mouse is not None else self.mice[0]
        sessions = list(self._sessions_by_mouse[initial_mouse])
        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("mouse", value=initial_mouse, options=self.mice)
        self.add_selection("example_session", value=example_session or sessions[0], options=sessions)
        self.add_integer("roi", value=roi, min=0, max=max(roi, 0))
        self.add_integer("env", value=env, min=0, max=max(env, 0))
        self.add_float("vmax", value=vmax, min=1.0, max=20.0)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.add_boolean("use_trial_weights", value=use_trial_weights)
        _add_breakdown_widgets(
            self,
            include_spatial_reliability=include_spatial_reliability,
            include_trial_gain=include_trial_gain,
            include_residual_rms=include_residual_rms,
            residual_metric=residual_metric,
            summary_width=summary_width,
            markersize=markersize,
            colorbar_x=colorbar_x,
            colorbar_y=colorbar_y,
            colorbar_width=colorbar_width,
            colorbar_height=colorbar_height,
            legend_x=legend_x,
            legend_y=legend_y,
            legend_ha=legend_ha,
            legend_va=legend_va,
        )
        self.on_change("mouse", self.update_example_session)
        self.on_change("example_session", self.update_example_bounds)
        self.on_change(["roi", "env"], self.refresh_data)
        self.on_change(list(self.selection_names), self.refresh_data)
        self.update_example_session(self.state)

    def refresh_data(self, state):
        """Fit gains to the all-trial template and retain the exact additive remainder."""
        maps = self._get_env_maps(state)
        self.distcenters = maps.distcenters
        self.spkmap = np.asarray(maps.spkmap[state["env"]][state["roi"]], dtype=float)
        gains, templates, _ = least_squares_gain_matrix(self.spkmap[None, ...])
        self.placefield = templates[0]
        self.trial_gain = gains[0]
        self.pred_spkmap = self.trial_gain[:, None] * self.placefield[None, :]
        self.pred_spkmap[~np.isfinite(self.spkmap)] = np.nan
        self.residual_spkmap = self.spkmap - self.pred_spkmap
        valid = np.isfinite(self.residual_spkmap)
        count = valid.sum(axis=1)
        self.residual_rms = np.sqrt(
            np.divide(
                np.sum(np.where(valid, self.residual_spkmap**2, 0.0), axis=1),
                count,
                out=np.full(count.shape, np.nan),
                where=count > 0,
            )
        )

        # Match the original viewer on complete maps. Missing positions are excluded
        # across trials because the shared leave-one-out estimator expects finite maps.
        complete = self.spkmap[:, np.all(np.isfinite(self.spkmap), axis=0)]
        if complete.shape[0] > 1 and complete.shape[1] > 1 and np.any(complete):
            self.consistency = _trial_consistency(complete)
        else:
            self.consistency = (np.array([], dtype=int), np.array([]), np.array([]), np.nan)

    def draw(self, state, fig, subplot_spec=None):
        """Draw a single row, optionally inside a nested subplot."""
        summary_width = state["summary_width"]
        columns = [("Placefield map", self.spkmap, "map")]
        if state["include_spatial_reliability"] and summary_width > 0:
            columns.append(("Spatial reliability", self.consistency[2], "reliability"))
        columns.append(("Gain × PF", self.pred_spkmap, "map"))
        if state["include_trial_gain"] and summary_width > 0:
            columns.append(("Trial Gain", self.trial_gain, "scatter"))
        columns.append(("Remaining additive component", self.residual_spkmap, "residual"))
        if state["include_residual_rms"] and summary_width > 0:
            if state["residual_metric"] == "unexplained_fraction":
                values = _trial_unexplained_fraction(self.spkmap, self.residual_spkmap)
                columns.append((r"$1-R^2$", values, "unexplained_fraction"))
            else:
                columns.append(("Resid.\nRMS", self.residual_rms, "scatter"))
        grid = fig.add_gridspec if subplot_spec is None else subplot_spec.subgridspec
        gs = grid(1, len(columns), width_ratios=[1 if kind in ("map", "residual") else summary_width for _, _, kind in columns])
        fontsize, vmax = state["fontsize"], state["vmax"]
        position_bounds = (0.0, float(max(50, np.ceil(np.max(self.distcenters) / 50) * 50)))
        ntrials = self.spkmap.shape[0]
        trial_axis = None
        for index, (label, values, kind) in enumerate(columns):
            ax = fig.add_subplot(gs[0, index], sharey=trial_axis)
            if trial_axis is None:
                trial_axis = ax
            ax.set_label(label)
            ax.set_ylim(ntrials + 0.5, -0.5)
            if kind in ("map", "residual"):
                residual = kind == "residual"
                cmap = colormaps["bwr"].resampled(257) if residual else "gray_r"
                if residual:
                    ax.set_facecolor("white")
                ax.imshow(
                    values,
                    interpolation="none",
                    aspect="auto",
                    extent=(self.distcenters[0], self.distcenters[-1], ntrials, 0),
                    cmap=cmap,
                    vmin=-vmax if residual else 0,
                    vmax=vmax,
                )
                legend, color = _PANEL_LEGENDS[label]
                ax.text(
                    state["legend_x"],
                    state["legend_y"],
                    legend,
                    transform=ax.transAxes,
                    ha=state["legend_ha"],
                    va=state["legend_va"],
                    fontsize=fontsize,
                    color=color,
                )
                bounds = position_bounds
                ax.set_xlabel("Position (cm)", fontsize=fontsize)
                if index == 0 or residual:
                    draw_vertical_colorscale(
                        ax.inset_axes([state[f"colorbar_{name}"] for name in ("x", "y", "width", "height")]),
                        cmap,
                        low_label=f"{-vmax:g}" if residual else "0",
                        high_label=f"{vmax:g}",
                        fontsize=fontsize,
                        ylabel="Norm. Fluor.",
                    )
            else:
                numbers = np.arange(ntrials)
                alpha = 1.0
                if kind == "reliability":
                    numbers, weights, values, _ = self.consistency
                    alpha = weights if state["use_trial_weights"] else 1.0
                finite = np.isfinite(values)
                point_alpha = alpha[finite] if isinstance(alpha, np.ndarray) else alpha
                if not np.any(finite):
                    point_alpha = 1.0
                ax.scatter(values[finite], numbers[finite], color="k", s=state["markersize"], alpha=point_alpha)
                ax.set_facecolor(("black", 0.04))
                if kind == "reliability":
                    bounds = (-1.0, 1.0)
                elif kind == "unexplained_fraction":
                    bounds = (0.0, 1.0)
                else:
                    bounds = self._zero_to_ceiling_bounds(values)
                    if np.any(finite):
                        bounds = (min(0.0, float(np.floor(np.min(values[finite])))), bounds[1])
                ax.set_xlabel("Trial\nSim." if kind == "reliability" else label, fontsize=fontsize)
            if kind == "reliability":
                ax.set_xlim(-1.1, 1.1)
            elif kind == "unexplained_fraction":
                ax.set_xlim(-0.05, 1.05)
            else:
                ax.set_xlim(bounds)
            style_axis(
                ax,
                fontsize=fontsize,
                xbounds=bounds,
                xticks=list(bounds),
                yticks=[0, ntrials] if index == 0 else [],
                spines_visible=["left", "bottom"] if index == 0 else ["bottom"],
            )
            # Shared trial limits must not share tick visibility.
            ax.tick_params(axis="y", left=index == 0, labelleft=index == 0)
        trial_axis.set_yticks([0, ntrials])
        trial_axis.set_ylabel("Trials", fontsize=fontsize)
        return fig
