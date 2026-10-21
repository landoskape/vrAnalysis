"""Environment decoding from running speed, grouped by session environment count."""

import numpy as np

from dimensionality_manuscript.figure_scripts.panels import FigureViewer, add_data_selection_widgets, data_selection
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import style_axis

SPEED_DECODING_SPLITS = {
    "random": "Random trials",
    "block": "First of block",
}
_SPLIT_COLORS = {"random": "black", "block": "0.6"}


class SpeedEnvironmentDecoding(FigureViewer):
    """Standalone summary of logistic regression predicting environment from running speed.

    Reads precomputed ``BehaviorSpeedEnvConfig`` results; no models are fitted here.
    ``random`` is the stratified 50/50 trial split. ``block`` trains on second and later
    trials of each block and tests on first trials, using the original block identities.

    Sessions are grouped by their number of environments with valid trials
    (``n_trials_per_env``), not the length of the padded canonical environment axis.
    Sessions where the pre-reward window loses an environment are excluded so that a
    three-environment session cannot contribute a two-class decoding result.
    Within each group/split, finite session scores are averaged within mouse before
    drawing mouse points and the equally weighted mouse mean +/- SEM (sample SD).
    A single mouse has no SEM. Missing split results are omitted independently.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``BehaviorSpeedEnvConfig`` results.
    environment_counts : list[int] or None
        Session groups to show: 2 and/or 3. None shows both.
    splits : list[str] or None
        ``"random"`` and/or ``"block"``. None shows both side by side.
    metric : {"accuracy", "balanced_accuracy"}
        Held-out performance. Ordinary accuracy uses the stored majority-training-class
        test accuracy as its chance baseline; balanced accuracy uses 1 / environment count.
    show_mice, show_sessions, show_chance, show_legend : bool
        Display mouse means, faint session points, dashed chance segments, and the split
        legend. Chance is averaged over the same sessions and mice as performance.
    markersize, alpha, fontsize : float
        Point size, mouse/session opacity, and font size in points.
    ylim : tuple[float, float]
        Performance axis limits, in fractions rather than percentages.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Initial values for the aggregator's parameter selection widgets.

    Example::

        viewer = SpeedEnvironmentDecoding(bse_results, splits=["block"])
        viewer.show()
        fig = viewer.plot(viewer.state)
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        environment_counts: list[int] | None = None,
        splits: list[str] | None = None,
        metric: str = "accuracy",
        show_mice: bool = True,
        show_sessions: bool = False,
        show_chance: bool = True,
        show_legend: bool = True,
        markersize: float = 4.0,
        alpha: float = 0.5,
        fontsize: float = 9.0,
        ylim: tuple[float, float] = (0.0, 1.05),
        figsize: tuple[float, float] = (4.0, 3.5),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_multiple_selection("environment_counts", options=[2, 3], value=[2, 3] if environment_counts is None else environment_counts)
        self.add_multiple_selection("splits", options=list(SPEED_DECODING_SPLITS), value=list(SPEED_DECODING_SPLITS) if splits is None else splits)
        self.add_selection("metric", options=["accuracy", "balanced_accuracy"], value=metric)
        for name, value in (
            ("show_mice", show_mice),
            ("show_sessions", show_sessions),
            ("show_chance", show_chance),
            ("show_legend", show_legend),
        ):
            self.add_boolean(name, value=value)
        self.add_float("markersize", value=markersize, min=1.0, max=12.0)
        self.add_float("alpha", value=alpha, min=0.0, max=1.0)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        self.add_float_range("ylim", value=ylim, min=0.0, max=1.2, step=0.01)
        if self.selection_names:
            self.on_change(list(self.selection_names), self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Load one parameter combination, preserving the session axis even for one row."""
        keys = ["n_trials_per_env", "env_used"]
        keys += [f"{metric}_{split}" for split in SPEED_DECODING_SPLITS for metric in ("acc_test", "bal_acc_test", "chance")]
        self._arrays = self.results.sel(
            keys=keys,
            squeeze_ones=False,
            **data_selection(state, self.results, self.selection_names),
        )
        self._mouse_names = np.asarray(self.results.mouse_names)
        self._environment_counts = np.sum(np.asarray(self._arrays["n_trials_per_env"]) > 0, axis=1)
        self._decoded_counts = np.sum(np.asarray(self._arrays["env_used"]) > 0, axis=1)

    def summarize(self, state):
        """Return session values and aligned per-mouse performance/chance for each category."""
        metric = "acc_test" if state["metric"] == "accuracy" else "bal_acc_test"
        groups = []
        for count in sorted(state["environment_counts"]):
            for split in state["splits"]:
                scores = np.asarray(self._arrays[f"{metric}_{split}"], dtype=float).reshape(-1)
                chance = (
                    np.asarray(self._arrays[f"chance_{split}"], dtype=float).reshape(-1)
                    if state["metric"] == "accuracy"
                    else np.full(scores.shape, 1.0 / count)
                )
                keep = (self._environment_counts == count) & (self._decoded_counts == count) & np.isfinite(scores) & np.isfinite(chance)
                mice = sorted(set(self._mouse_names[keep]))
                mouse_scores = np.array([np.mean(scores[keep & (self._mouse_names == mouse)]) for mouse in mice])
                mouse_chance = np.array([np.mean(chance[keep & (self._mouse_names == mouse)]) for mouse in mice])
                groups.append(
                    dict(
                        environment_count=count,
                        split=split,
                        mice=mice,
                        scores=mouse_scores,
                        chance=mouse_chance,
                        session_scores=scores[keep],
                    )
                )
        return groups

    def plot(self, state):
        fig, ax = self.new_subplots(figsize=self.figsize)
        counts = sorted(state["environment_counts"])
        splits = list(state["splits"])
        width = 0.7 / max(len(splits), 1)
        groups = self.summarize(state)
        for group in groups:
            split = group["split"]
            x = counts.index(group["environment_count"]) + (splits.index(split) - (len(splits) - 1) / 2) * width
            color = _SPLIT_COLORS[split]
            scores = group["scores"]
            n = len(scores)
            if state["show_sessions"]:
                values = group["session_scores"]
                offsets = np.linspace(-width * 0.3, width * 0.3, len(values))
                ax.scatter(x + offsets, values, s=state["markersize"] ** 2 / 2, color=color, alpha=state["alpha"] / 2, edgecolors="none")
            if state["show_mice"]:
                offsets = np.linspace(-width * 0.2, width * 0.2, n) if n > 1 else np.zeros(n)
                ax.plot(x + offsets, scores, "o", color=color, markersize=state["markersize"], alpha=state["alpha"], markeredgewidth=0)
            if n:
                sem = np.std(scores, ddof=1) / np.sqrt(n) if n > 1 else None
                ax.errorbar(x, np.mean(scores), yerr=sem, fmt="o", color=color, markersize=state["markersize"] + 1, capsize=3, zorder=4)
                if state["show_chance"]:
                    ax.plot([x - width * 0.4, x + width * 0.4], [np.mean(group["chance"])] * 2, "--", color=color, linewidth=1)
        ax.set_xticks(range(len(counts)), [f"#Env={count}" for count in counts])
        ax.set_xlim(-0.6, max(len(counts) - 0.4, 0.6))
        ax.set_ylim(*state["ylim"])
        ax.set_ylabel("Test accuracy", fontsize=state["fontsize"])
        if not any(len(group["scores"]) for group in groups):
            ax.text(0.5, 0.5, "No valid decoding results", transform=ax.transAxes, ha="center", fontsize=state["fontsize"])
        if state["show_legend"] and splits:
            from matplotlib.lines import Line2D

            handles = [
                Line2D(
                    [],
                    [],
                    color=_SPLIT_COLORS[split],
                    marker="o",
                    linestyle="none",
                    markersize=state["markersize"],
                    label=SPEED_DECODING_SPLITS[split],
                )
                for split in splits
            ]
            ax.legend(handles=handles, fontsize=state["fontsize"], frameon=False, loc="lower left")
        # style_axis forwards these bounds to format_spines.
        style_axis(ax, fontsize=state["fontsize"], xbounds=(0, len(counts) - 1) if counts else None)
        fig.tight_layout()
        return fig
