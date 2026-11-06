"""Paired absolute-session and environment-session summaries of place-field familiarity."""

from collections.abc import Mapping

import numpy as np

from dimensionality_manuscript.configs.pfpred_quality import (
    TRIAL_ACTIVITY_METRICS,
    TRIAL_ACTIVITY_STATISTICS,
    pf_rms_correlation_keys,
)
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
    render_curve_group,
)
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import env_slot_color, ordinal, pad_stack, style_axis
from .placefield import (
    _AMPLITUDE_TRANSFORMS,
    _transform_amplitude,
)
from .timeline import ExperimentTimeline


class HolisticPlacefieldFamiliarityViewer(FigureViewer):
    """Timeline, PF-Var Alignment, spatial reliability, and amplitude in two session coordinates.

    Rows are schematic / position mean-STD Pearson r / spatial reliability / amplitude.
    Columns use absolute imaging-session order and session within environment, respectively.
    Each curve summarizes cells within a session, then weights mice equally (mean +/- STD,
    or individual mouse curves). ``min_mice`` controls the required finite mouse count
    at each point (default two, or one when only one mouse is available).
    ``mouse`` selects only the illustrative schematic.
    ``include_mean_std=False`` omits the PF-Var Alignment row from both columns.

    ``relative_values=True`` divides each mouse's session summaries by its own first-
    environment baseline, separately for each metric and cell group. The baseline is
    the mean of finite summaries in the first ``baseline_sessions`` exposures (or all
    available exposures if fewer exist); missing sessions never shift the window forward.
    Zero/missing baselines produce missing curves. Ratios autoscale and omit the absolute
    placecell cutoff, regardless of the full-limit toggles.

    ``ylim_full_correlation`` and ``ylim_full_reliability`` fix their row limits to
    [-1, 1] and [0, 1]. ``ylim_full_amplitude`` uses [0, 1] only for the bounded
    transform; raw and inverse always autoscale. Disable a toggle to autoscale its row.

    ``schematic_config`` accepts an ExperimentTimeline instance or its state/constructor
    mapping. Its settings are copied at construction; the source viewer is never mutated.
    The two column modes are fixed and training is omitted. Phase labels are Imaging
    Timeline and Relative Timeline; other schematic settings are retained.

    ``pf_rms_comparison`` offers the pairings used by PlaceFieldPredictionFocus, plus
    ``cv`` and ``independent_cv`` (averages of the two fold directions). Stored per-cell
    amplitude statistics are transformed BEFORE the across-cell mean/median.

    ``cells`` is all / quality / not quality / split. Quality requires spatial reliability
    strictly above ``reliability_threshold`` AND fraction active strictly above
    ``fraction_active_threshold``. Not quality is the complement among cells with finite
    classification measurements. Split overlays solid quality and dashed not-quality curves.
    Each metric uses its own finite values within that selection. Empty selections retain
    their session position, including in relative mode. Cutoffs need no trial-map recompute.

    Requires per-ROI pfpred_quality schema v11 results. Result-parameter axes are exposed
    through ``**selection_defaults``, as in the other Figure 1 viewers.

    Example::

        viewer = HolisticPlacefieldFamiliarityViewer(
            results, schematic_config=timeline, cells="split",
            amplitude_statistic="cv", amplitude_transform="bounded",
        )
        viewer.show()
    """

    _METRICS = ("correlation", "reliability", "amplitude")

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        schematic_config: ExperimentTimeline | Mapping | None = None,
        mouse: str | None = None,
        pf_rms_comparison: str = "f2_from_f1",
        amplitude_metric: str = "trial_rms",
        amplitude_statistic: str = "cv",
        amplitude_transform: str = "bounded",
        cells: str = "all",
        reliability_threshold: float = 0.7,
        fraction_active_threshold: float = 0.2,
        summary_stat: str = "mean",
        plot_style: str = "errorPlot",
        hide_error: bool = False,
        show_legend: bool = True,
        include_mean_std: bool = True,
        relative_values: bool = False,
        baseline_sessions: int = 1,
        min_mice: int = 2,
        ylim_full_correlation: bool = True,
        ylim_full_reliability: bool = True,
        ylim_full_amplitude: bool = True,
        fontsize: float = 9.0,
        schema_height_ratio: float = 0.8,
        figsize: tuple[float, float] = (8.0, 10.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(name) for name in mouse_names})
        if not self.mice:
            raise ValueError("The aggregator contains no mice.")
        self._rows_by_mouse = {
            name: sorted(np.flatnonzero(mouse_names == name), key=lambda row: _session_sort_key(results.sessions[row])) for name in self.mice
        }
        config = dict(schematic_config.state if isinstance(schematic_config, ExperimentTimeline) else schematic_config or {})
        if "fig_width" in config or "fig_height" in config:
            config["figsize"] = (config.pop("fig_width", 3.6), config.pop("fig_height", 2.0))
        initial_mouse = mouse if mouse is not None else config.get("mouse", self.mice[0])
        config["mouse"] = initial_mouse
        self.timeline = ExperimentTimeline(results.sessions, **config)

        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("mouse", value=initial_mouse, options=self.mice)
        comparisons = [key.removeprefix("pf_rms_r_") for key in pf_rms_correlation_keys()]
        self.add_selection("pf_rms_comparison", value=pf_rms_comparison, options=comparisons)
        self.add_selection("amplitude_metric", value=amplitude_metric, options=list(TRIAL_ACTIVITY_METRICS))
        self.add_selection("amplitude_statistic", value=amplitude_statistic, options=list(TRIAL_ACTIVITY_STATISTICS))
        self.add_selection("amplitude_transform", value=amplitude_transform, options=list(_AMPLITUDE_TRANSFORMS))
        self.add_selection("cells", value=cells, options=["all", "quality", "not quality", "split"])
        self.add_float("reliability_threshold", value=reliability_threshold, min=-1.0, max=1.0, step=0.01)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0, step=0.01)
        self.add_selection("summary_stat", value=summary_stat, options=["mean", "median"])
        self.add_selection("plot_style", value=plot_style, options=["each", "errorPlot"])
        self.add_boolean("hide_error", value=hide_error)
        self.add_boolean("show_legend", value=show_legend)
        self.add_boolean("include_mean_std", value=include_mean_std)
        self.add_boolean("relative_values", value=relative_values)
        self.add_integer("baseline_sessions", value=baseline_sessions, min=1, max=5)
        self.add_integer("min_mice", value=min(min_mice, len(self.mice)), min=1, max=len(self.mice))
        self.add_boolean("ylim_full_correlation", value=ylim_full_correlation)
        self.add_boolean("ylim_full_reliability", value=ylim_full_reliability)
        self.add_boolean("ylim_full_amplitude", value=ylim_full_amplitude)
        self.add_float("fontsize", value=fontsize, min=3.0, max=30.0)
        self.add_float("schema_height_ratio", value=schema_height_ratio, min=0.1, max=5.0)
        self.on_change(
            [*self.selection_names, "pf_rms_comparison", "amplitude_metric", "amplitude_statistic"],
            self.reload_arrays,
        )
        self.on_change(
            ["cells", "reliability_threshold", "fraction_active_threshold", "summary_stat", "amplitude_transform",
             "relative_values", "baseline_sessions"],
            self.refresh_summary_data,
        )
        self.reload_arrays(self.state)

    def reload_arrays(self, state):
        correlation_key = f"pf_rms_r_{state['pf_rms_comparison']}_slot"
        amplitude_key = f"{state['amplitude_metric']}_{state['amplitude_statistic']}"
        keys = [correlation_key, "reliability_slot", amplitude_key, "fraction_active_slot", "env_slot_ids"]
        out = self.results.sel(keys=keys, squeeze_ones=False, **data_selection(state, self.results, self.selection_names))
        missing = [key for key in keys if key not in out]
        if missing:
            raise KeyError(f"Missing {missing}; rerun the pfpred_quality schema v11 sweep.")
        arrays = [np.asarray(out[key], dtype=float) for key in keys[:-1]]
        if any(array.ndim != 3 for array in arrays) or len({array.shape[:2] for array in arrays}) != 1:
            raise ValueError("Per-cell arrays must have matching session and environment-slot dimensions.")
        # Lazy aggregation can retain different amounts of trailing ROI padding per key.
        width = min(array.shape[-1] for array in arrays)
        if any(not np.isnan(array[..., width:]).all() for array in arrays):
            raise ValueError("Per-cell arrays have incompatible ROI dimensions beyond NaN padding.")
        self.arrays = dict(zip((*self._METRICS, "fraction_active"), (array[..., :width] for array in arrays)))
        self.env_slot_ids = np.asarray(out["env_slot_ids"], dtype=float)
        if self.env_slot_ids.shape != arrays[0].shape[:2] or arrays[0].shape[0] != len(self.results.sessions):
            raise ValueError("env_slot_ids and per-cell arrays must match the aggregator's sessions and slots.")
        # Slot IDs describe the mouse's complete environment order, even in sessions
        # before an environment was introduced. Only session metadata records exposure.
        self.env_present = np.array(
            [
                [np.isfinite(env) and env in session.environments for env in slot_ids]
                for session, slot_ids in zip(self.results.sessions, self.env_slot_ids)
            ],
            dtype=bool,
        )
        self.refresh_summary_data(state)

    def refresh_summary_data(self, state):
        """Summarize once per session, then place the same values in both coordinates."""
        groups = ("quality", "not quality") if state["cells"] == "split" else (state["cells"],)
        self.metric_stacks = {mode: {group: {metric: {} for metric in self._METRICS} for group in groups} for mode in ("absolute", "relative")}
        self.baselines = {group: {} for group in groups}
        reliability = self.arrays["reliability"]
        active = self.arrays["fraction_active"]
        classified = np.isfinite(reliability) & np.isfinite(active)
        quality = (reliability > state["reliability_threshold"]) & (active > state["fraction_active_threshold"])
        transformed = {**self.arrays, "amplitude": _transform_amplitude(self.arrays["amplitude"], state["amplitude_transform"])}
        reduce = np.mean if state["summary_stat"] == "mean" else np.median
        for group in groups:
            mask = np.ones_like(quality) if group == "all" else classified & (quality if group == "quality" else ~quality)
            summaries = {}
            for metric in self._METRICS:
                values = transformed[metric]
                summary = np.full(values.shape[:2], np.nan)
                for row, slot in np.ndindex(summary.shape):
                    keep = mask[row, slot] & np.isfinite(values[row, slot])
                    if np.any(keep) and self.env_present[row, slot]:
                        summary[row, slot] = reduce(values[row, slot, keep])
                summaries[metric] = summary
            if state["relative_values"]:
                for metric, summary in summaries.items():
                    baselines = []
                    for mouse in self.mice:
                        rows = np.asarray(self._rows_by_mouse[mouse], dtype=int)
                        first_env_rows = rows[self.env_present[rows, 0]][:state["baseline_sessions"]]
                        values = summary[first_env_rows, 0]
                        finite = values[np.isfinite(values)]
                        baseline = float(np.mean(finite)) if finite.size else np.nan
                        baselines.append(baseline)
                        if np.isfinite(baseline) and baseline != 0:
                            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                normalized = summary[rows] / baseline
                            summary[rows] = np.where(np.isfinite(normalized), normalized, np.nan)
                        else:
                            summary[rows] = np.nan
                    self.baselines[group][metric] = np.asarray(baselines)
            for mode in self.metric_stacks:
                for metric, summary in summaries.items():
                    for slot in range(summary.shape[1]):
                        curves = []
                        for mouse in self.mice:
                            rows = np.asarray(self._rows_by_mouse[mouse], dtype=int)
                            if mode == "relative":
                                # Count exposure even if no selected cell has a finite metric.
                                rows = rows[self.env_present[rows, slot]]
                            curves.append(summary[rows, slot])
                        self.metric_stacks[mode][group][metric][slot] = pad_stack(curves)

    def _draw_summary_panel(self, ax, mode, metric, state):
        if metric == "reliability" and state["cells"] == "split" and not state["relative_values"]:
            ax.axhline(state["reliability_threshold"], color="0.5", linestyle=":", linewidth=1.0)
            ax.annotate(
                "Placecell Cutoff",
                xy=(0.5, state["reliability_threshold"]),
                xycoords=ax.get_yaxis_transform(),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=state["fontsize"],
                color="0.25",
            )
        for group, metrics in self.metric_stacks[mode].items():
            for slot, stack in metrics[metric].items():
                # Apply support to every plotted series, including plot_style="each".
                # Keep unsupported columns as gaps so later exposures stay aligned.
                stack = np.where((np.isfinite(stack).sum(axis=0) >= state["min_mice"])[None, :], stack, np.nan)
                if not np.isfinite(stack).any():
                    continue
                label = f"{ordinal(slot + 1)} Env"
                if state["cells"] != "all":
                    label += f" ({group})"
                previous_lines = len(ax.lines)
                render_curve_group(
                    ax,
                    np.arange(1, stack.shape[1] + 1),
                    stack,
                    env_slot_color(slot),
                    state["plot_style"],
                    label=label,
                    hide_error=state["hide_error"],
                    min_support=state["min_mice"],
                    linewidth=1.5,
                )
                for line in ax.lines[previous_lines:]:
                    line.set_linestyle("--" if group == "not quality" else "-")
                    if np.isfinite(line.get_ydata()).sum() == 1:
                        line.set_marker(".")
        bottom = metric == "amplitude"
        if bottom:
            ax.set_xlabel("Overall session #" if mode == "absolute" else "Env session #", fontsize=state["fontsize"])
        ax.set_xlim(self._column_xlim(mode, state))
        if state[f"ylim_full_{metric}"] and not state["relative_values"]:
            if metric == "correlation":
                ax.set_ylim(-1.0, 1.0)
            elif metric == "reliability" or state["amplitude_transform"] == "bounded":
                ax.set_ylim(0.0, 1.0)
        style_axis(ax, fontsize=state["fontsize"], spines_visible=("left", "bottom") if bottom else ("left",))
        if not bottom:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

    def _column_xlim(self, mode, state):
        """Shared extent of supported points across visible rows; ignore padded tails."""
        positions = [
            np.flatnonzero(np.isfinite(stack).sum(axis=0) >= state["min_mice"]) + 1
            for metrics in self.metric_stacks[mode].values()
            for metric, stacks in metrics.items()
            if metric != "correlation" or state["include_mean_std"]
            for stack in stacks.values()
        ]
        positions = [values for values in positions if values.size]
        if not positions:
            return (0.5, 1.5)
        low = min(values[0] for values in positions)
        high = max(values[-1] for values in positions)
        return (low, high) if low < high else (low - 0.5, high + 0.5)

    def plot(self, state):
        metrics = self._METRICS if state["include_mean_std"] else self._METRICS[1:]
        fig, axes = self.new_subplots(
            len(metrics) + 1,
            2,
            figsize=self.figsize,
            layout="constrained",
            height_ratios=[state["schema_height_ratio"], *[1] * len(metrics)],
            sharey="row",
        )
        labels = {"correlation": "PF-Var Alignment", "reliability": "Spatial Reliability", "amplitude": "Amplitude Reliability"}
        timeline_state = {**self.timeline.state, "mouse": state["mouse"], "num_training": 0}
        self.timeline.refresh_data(timeline_state)
        for column, mode in enumerate(("absolute", "relative")):
            self.timeline.draw(
                axes[0, column],
                {**timeline_state, "mode": mode, "imaging_label": "Imaging Timeline" if mode == "absolute" else "Relative Timeline"},
                session_xlim=self._column_xlim(mode, state),
            )
            for row, metric in enumerate(metrics, start=1):
                self._draw_summary_panel(axes[row, column], mode, metric, state)
                if column == 1:
                    axes[row, column].spines["left"].set_visible(False)
                    axes[row, column].tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
                if column == 0:
                    axes[row, column].set_ylabel(labels[metric], fontsize=state["fontsize"])
            if state["show_legend"] and axes[1, column].get_legend_handles_labels()[0]:
                axes[1, column].legend(frameon=False, fontsize=state["fontsize"])
        return fig
