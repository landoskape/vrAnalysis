"""Mouse-level shuffle and trial-lag comparisons for stored gain regressions."""

from __future__ import annotations

import numpy as np

from dimensionality_manuscript.configs.gain_regression import TRIAL_SHIFTS
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    add_trace_style_widgets,
    data_selection,
    style_model_axis,
)
from dimensionality_manuscript.pipeline import ResultsAggregator
from vrAnalysis.helpers.plotting import beeswarm, errorPlot

from .placefield_gain import _finite_mean


class GainRegressionViewer(FigureViewer):
    """Data versus shuffle above, held-out prediction versus trial lag below.

    Reads ``GainRegressionConfig`` results without recomputing fits. Computation
    controls come from the aggregator's parameter axes, including PF train/all,
    Gaussian/least-squares gain, and raw/sqrt fitting coordinates. ``score_scale``
    independently selects raw/sqrt scoring coordinates; ``metric`` is R² or MSE.
    All other stored scalar parameter axes are selectable too.

    Each dot/curve represents a mouse: environments are averaged within sessions,
    then sessions within mice. The top row pairs finite same-trial and shuffle
    observations; the bottom row uses observations finite at *every* lag, so its
    trajectory compares the same sessions/environments throughout. The two rows
    are matched independently. ``min_trial`` screens environment trial counts.
    Shuffle is the stored mean of the zero-lag independent neuron-roll controls.
    Positive lag predicts target T+k from source T.

    ``display="each"`` draws every mouse trajectory plus the mean;
    ``display="errorPlot"`` draws the mean ± SE across mice. The top row always
    shows the mouse swarm and mean bars, with horizontal labels and black points.
    ``plot_structure="split"`` keeps these two rows; ``"condensed"`` overlays
    the shuffle controls as dotted horizontal lines (or mean ± SE) on one lag panel.
    ``value="relative"`` reports each mouse's shuffle minus its prediction score,
    after mouse averaging and before computing the across-mouse mean or SE.
    Shuffle is then zero. ``"absolute"`` preserves the stored scores.

    Example::

        viewer = GainRegressionViewer(
            results, placefield_split="train", gain_estimator="least_squares",
            gain_transform="sqrt", score_scale="raw", display="errorPlot",
        )
        viewer.show()
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        score_scale: str = "raw",
        metric: str = "r2",
        display: str = "each",
        plot_structure: str = "split",
        value: str = "absolute",
        mouse: str = "All",
        min_trial: int = 10,
        fontsize: float = 10.0,
        tick_fontsize: float = 8.0,
        tick_length: float = 2.0,
        tick_width: float = 0.6,
        beewidth: float = 0.2,
        markersize: float = 4.0,
        mean_linewidth: float = 2.0,
        subject_linewidth: float = 0.7,
        subject_alpha: float = 0.3,
        figsize: tuple[float, float] = (4.0, 7.0),
        **selection_defaults,
    ):
        if "trial_shift" in selection_defaults:
            raise ValueError("trial_shift is fixed by the plotted lag trajectory")
        if not set(TRIAL_SHIFTS) <= set(results.param_axes.get("trial_shift", ())):
            raise ValueError("GainRegressionViewer requires trial_shift values -3 through +3")
        if min_trial < 0:
            raise ValueError("min_trial must be non-negative")
        self.results = results
        self.figsize = figsize
        self.selection_names = add_data_selection_widgets(
            self,
            results,
            skip=("trial_shift",),
            defaults=selection_defaults,
            require=("placefield_split", "gain_estimator", "gain_transform"),
        )
        unknown = set(selection_defaults) - set(self.selection_names)
        if unknown:
            raise ValueError(f"Unknown computation selections: {sorted(unknown)}")
        self.add_selection("score_scale", value=score_scale, options=["raw", "sqrt"])
        self.add_selection("metric", value=metric, options=["r2", "mse"])
        self.add_selection("display", value=display, options=["each", "errorPlot"])
        self.add_selection("plot_structure", value=plot_structure, options=["split", "condensed"])
        self.add_selection("value", value=value, options=["absolute", "relative"])
        mice = list(dict.fromkeys(np.asarray(results.mouse_names).tolist()))
        self.add_selection("mouse", value=mouse, options=["All", *mice])
        self.add_integer("min_trial", value=min_trial, min=0, max=max(1000, min_trial))
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.add_float("tick_fontsize", value=tick_fontsize, min=4.0, max=30.0)
        self.add_float("tick_length", value=tick_length, min=0.0, max=10.0)
        self.add_float("tick_width", value=tick_width, min=0.1, max=3.0)
        self.add_float("beewidth", value=beewidth, min=0.0, max=0.5, step=0.01)
        add_trace_style_widgets(
            self,
            markersize=markersize,
            mean_linewidth=mean_linewidth,
            subject_linewidth=subject_linewidth,
            subject_alpha=subject_alpha,
        )
        self.on_change([*self.selection_names, "score_scale", "metric", "min_trial"], self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Load explicit-scale scores and match observations separately in each row."""
        key = f"{state['metric']}_test_{state['score_scale']}"
        selection = data_selection(state, self.results, self.selection_names)
        lag_scores = []
        for lag in TRIAL_SHIFTS:
            keys = [key, "n_trials_env"]
            if lag == 0:
                keys.append(f"{key}_null")
            selected = self.results.sel(
                keys=keys,
                squeeze_ones=False,
                trial_shift=lag,
                **selection,
            )
            score = np.asarray(selected[key], dtype=float)
            if score.ndim != 2 or score.shape[0] != len(self.results.mouse_names):
                raise ValueError("Expected session × environment scores; select all extra parameter axes")
            enough = np.isfinite(selected["n_trials_env"]) & (selected["n_trials_env"] >= state["min_trial"])
            lag_scores.append(np.where(enough & np.isfinite(score), score, np.nan))
            if lag == 0:
                comparison = np.stack((score, selected[f"{key}_null"]), axis=-1)
                paired = enough & np.all(np.isfinite(comparison), axis=-1)
                self.comparison_scores = np.where(paired[..., None], comparison, np.nan)
        trajectory = np.stack(lag_scores, axis=-1)
        matched = np.all(np.isfinite(trajectory), axis=-1)
        self.trajectory_scores = np.where(matched[..., None], trajectory, np.nan)

    def mouse_scores(self, state):
        """Return mouse names, paired comparison scores, and matched lag curves."""
        names = np.asarray(self.results.mouse_names)
        keep = np.ones(names.size, dtype=bool) if state["mouse"] == "All" else names == state["mouse"]
        mice = list(dict.fromkeys(names[keep].tolist()))

        def aggregate(values):
            sessions = _finite_mean(values[keep], axis=1)
            return np.asarray([_finite_mean(sessions[names[keep] == mouse], axis=0) for mouse in mice]).reshape(len(mice), values.shape[-1])

        comparison = aggregate(self.comparison_scores)
        trajectory = aggregate(self.trajectory_scores)
        if state["value"] == "relative":
            shuffle = comparison[:, 1:2]
            trajectory = trajectory - shuffle
            comparison = comparison - shuffle
        return mice, comparison, trajectory

    def plot(self, state):
        mice, comparison, trajectory = self.mouse_scores(state)
        self.mouse_names = mice
        split = state["plot_structure"] == "split"
        if split:
            fig, (top, bottom) = self.new_subplots(2, 1, figsize=self.figsize, layout="constrained")
            for x in range(2):
                values = comparison[:, x]
                values = values[np.isfinite(values)]
                if values.size:
                    top.plot(x + state["beewidth"] * beeswarm(values), values, "o", color="k", markersize=state["markersize"], alpha=0.8)
                    top.plot(x + np.array([-0.25, 0.25]), [np.mean(values)] * 2, color="k", linewidth=state["mean_linewidth"])
            top.set_xlim(-0.5, 1.5)
        else:
            fig, bottom = self.new_subplots(figsize=self.figsize, layout="constrained")
        lags = np.asarray(TRIAL_SHIFTS)
        valid = np.any(np.isfinite(trajectory), axis=0)
        if np.any(valid):
            if state["display"] == "each":
                bottom.plot(lags, trajectory.T, color="k", alpha=state["subject_alpha"], linewidth=state["subject_linewidth"])
                bottom.plot(lags[valid], _finite_mean(trajectory, axis=0)[valid], color="k", linewidth=state["mean_linewidth"])
            else:
                errorPlot(lags[valid], trajectory[:, valid], axis=0, se=True, ax=bottom, color="k", alpha=0.25, linewidth=state["mean_linewidth"])
        bottom.set_xlim(-3.25, 3.25)
        if not split:
            self._draw_shuffle(bottom, comparison[:, 1], lags, state)
        metric_label = r"$R^2$" if state["metric"] == "r2" else "MSE"
        delta = r"$\Delta$ " if state["value"] == "relative" else ""
        label = f"Gain Pred. {delta}{metric_label}"
        panels = [(bottom, lags, [str(lag) if lag in (-3, 0, 3) else "" for lag in lags])]
        if split:
            panels.insert(0, (top, [0, 1], ["Data", "Shuffle"]))
        for ax, xvals, labels in panels:
            ax.set_ylabel(label, fontsize=state["fontsize"])
            style_model_axis(
                ax,
                fontsize=state["fontsize"],
                xvals=xvals,
                labels=labels,
                xbounds=[xvals[0], xvals[-1]],
                xtick_rotation=0,
                xha="center",
                y_pos=-0.1,
            )
            ax.tick_params(axis="both", which="both", labelsize=state["tick_fontsize"], length=state["tick_length"], width=state["tick_width"])
            if not ax.has_data():
                ax.text(0.5, 0.5, "No matched results", transform=ax.transAxes, ha="center")
        bottom.set_xlabel("Trial Lag", fontsize=state["fontsize"])
        return fig

    def _draw_shuffle(self, ax, shuffle, lags, state):
        """Overlay the mouse nulls and label below their lowest visible extent."""
        shuffle = shuffle[np.isfinite(shuffle)]
        if not shuffle.size:
            return
        if state["display"] == "each":
            for value in shuffle:
                ax.axhline(value, color="k", linestyle=":", alpha=state["subject_alpha"], linewidth=state["subject_linewidth"])
            lower = np.min(shuffle)
        else:
            errorPlot(
                lags,
                np.repeat(shuffle[:, None], len(lags), axis=1),
                axis=0,
                se=True,
                ax=ax,
                color="k",
                linestyle=":",
                alpha=0.25,
                linewidth=state["mean_linewidth"],
            )
            lower = np.mean(shuffle) - np.std(shuffle) / np.sqrt(shuffle.size)
        ax.annotate(
            "shuffle baseline",
            xy=(0, lower),
            xycoords=ax.get_yaxis_transform(),
            xytext=(3, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=state["fontsize"],
        )
