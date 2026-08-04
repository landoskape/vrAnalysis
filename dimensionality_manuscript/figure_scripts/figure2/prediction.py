"""Model prediction versus held-out neural activity, for one ROI or a quality-filtered pool."""

import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

from vrAnalysis.helpers.plotting import format_spines
from vrAnalysis.sessions import B2Session
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.figure_scripts.panels import FigureViewer, add_data_selection_widgets

from ._predictions import get_model_predictions, target_prediction_quality


class RegressionPredictionViewer(FigureViewer):
    """Inspect held-out model predictions for one ROI or a quality-filtered population.

    Predictions come from :func:`get_model_predictions`, so switching models or ROIs after the
    first visit is instant. ``roi_mode="quality_filtered"`` pools every target ROI whose mean
    spatial reliability and fraction-active fall inside the selected ranges.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionConfig`` results. Supplies the session list and the
        data-selection options.
    registry : PopulationRegistry
        Population registry used to fetch each session's cell splits.
    method : str
        Hyperparameter optimization method used to look up the best hyperparameters.
    roi_mode : {"single", "quality_filtered"}
        Draw one ROI, or every ROI passing the reliability / fraction-active ranges.
    roi : int
        ROI index within the session's target split, for ``roi_mode="single"``.
    reliability_range, fraction_active_range : tuple[float, float]
        Inclusive quality bounds, for ``roi_mode="quality_filtered"``.
    plot_style : {"scatter", "hexbin", "kde"}
        How the joint distribution is drawn.
    axis_scale : {"linear", "log", "symlog"}
        Scale shared by both axes. ``"log"`` also drops non-positive samples.
    alpha, markersize : float
        Scatter opacity and marker size.
    density_levels, hex_gridsize : int
        Contour count for ``"kde"`` and hex resolution for ``"hexbin"``.
    density_norm : {"linear", "log"}
        Color normalization for ``"hexbin"``.
    show_identity : bool
        Draw the dashed y = x line.
    show_title : bool
        Draw the model name and sample counts above the axis.
    fontsize : float
        Font size for the axis labels, tick labels, and title.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the data-selection widgets, overriding the config's own. The widgets
        (``model_name``, ``spks_type``, ``activity_parameters_name``, ...) are built from the
        aggregator's param axes; unswept ones are pinned to the value its config fixes.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        registry: PopulationRegistry,
        *,
        method: str = "preferred",
        roi_mode: str = "single",
        roi: int = 0,
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
        fontsize: float = 10.0,
        figsize: tuple[float, float] = (5.0, 5.0),
        **selection_defaults,
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
        # These three go to get_model_predictions / get_population, not to results.sel, so the
        # panel needs a value for each even when the config no longer sweeps it.
        self.selection_names = add_data_selection_widgets(
            self,
            results,
            defaults=selection_defaults,
            require=("model_name", "spks_type", "activity_parameters_name"),
        )
        self.add_selection("method", value=method, options=["preferred", "best"])

        self.add_selection("roi_mode", value=roi_mode, options=["single", "quality_filtered"])
        self.add_integer("roi", value=roi, min=0, max=max(roi, 0))
        self.add_float_range("reliability_range", value=reliability_range, min=-1.0, max=1.0, step=0.05)
        self.add_float_range("fraction_active_range", value=fraction_active_range, min=0.0, max=1.0, step=0.025)

        self.add_selection("plot_style", value=plot_style, options=["scatter", "hexbin", "kde"])
        self.add_selection("axis_scale", value=axis_scale, options=["linear", "log", "symlog"])
        self.add_float("alpha", value=alpha, min=0.01, max=1.0, step=0.01)
        self.add_float("markersize", value=markersize, min=0.1, max=100.0, step=0.5)
        self.add_integer("density_levels", value=density_levels, min=3, max=50)
        self.add_integer("hex_gridsize", value=hex_gridsize, min=10, max=200)
        self.add_selection("density_norm", value=density_norm, options=["linear", "log"])
        self.add_boolean("show_identity", value=show_identity)
        self.add_boolean("show_title", value=show_title)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

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
            reliability, fraction_active = target_prediction_quality(
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
        fontsize = state["fontsize"]
        fig, ax = self.new_subplots(figsize=self.figsize, layout="constrained")
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
            ax.text(0.5, 0.5, "No finite samples in selection", ha="center", va="center", transform=ax.transAxes, fontsize=fontsize)

        scale = state["axis_scale"]
        if scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")
        elif scale == "symlog":
            ax.set_xscale("symlog", linthresh=0.01)
            ax.set_yscale("symlog", linthresh=0.01)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Held-out neural activity", fontsize=fontsize)
        ax.set_ylabel("Model prediction", fontsize=fontsize)
        if state["show_title"]:
            ax.set_title(f"{state['model_name']} · {num_rois} ROI(s), {num_points:,} samples", fontsize=fontsize)
        format_spines(ax, x_pos=-0.03, y_pos=-0.03, tick_fontsize=fontsize)
        # Applied after format_spines, which calls tick_params itself; minor ticks matter on log axes.
        ax.tick_params(axis="both", which="both", labelsize=fontsize)
        return fig
