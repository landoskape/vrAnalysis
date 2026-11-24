"""Single-mouse session summaries alongside across-mouse relative-time familiarity."""

from collections.abc import Mapping

import numpy as np

from dimensionality_manuscript.configs.pfpred_quality import (
    TRIAL_ACTIVITY_METRICS,
    TRIAL_ACTIVITY_STATISTICS,
    PF_VARIANCE_COMPARISONS,
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

from ._shared import env_slot_color, load_placefield_cell_arrays, ordinal, pad_stack, style_axis
from .placefield import (
    _AMPLITUDE_TRANSFORMS,
    _transform_amplitude,
)
from .timeline import ExperimentTimeline


class HolisticPlacefieldFamiliarityViewer(FigureViewer):
    """Timeline, PF-Var Alignment, spatial reliability, and amplitude in two session coordinates.

    Rows are schematic / position mean-STD Pearson r / spatial reliability / amplitude.
    The first two columns show ``mouse`` in chronological and relative (within-environment)
    time. The third shows relative time across mice, weighted equally (mean +/- STD,
    or individual mouse curves). Each curve first summarizes cells within a session.
    ``min_mice`` controls the required finite mouse count in the third column only
    (default two, or one when only one mouse is available).
    ``include_mean_std=False`` omits the PF-Var Alignment row from all columns.
    ``show_schematic=False`` omits the timeline row; data rows fill the requested figsize.

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
    Column modes are fixed and training and phase bars are omitted. Phase labels are
    Chronological Time and Relative Time; other schematic settings are retained.

    ``pf_rms_comparison`` offers the pairings used by PlaceFieldPredictionFocus, plus
    ``cv`` and ``independent_cv`` (averages of the two fold directions). Stored per-cell
    amplitude statistics are transformed BEFORE the across-cell mean/median.

    ``include_residual_rms=True`` appends a bottom row of gain-corrected trial RMS,
    matching PlacefieldBreakdownFocus when ``residual_rms_fold="all_from_all"``.
    ``residual_rms_statistic`` selects mean / variance / cv across trials;
    ``residual_rms_transform`` selects raw / inverse (also accepts inv) / bounded,
    independently of the amplitude row. Transformations precede cell summaries.
    ``residual_rms_fold`` selects the five directional PF folds, cv, or within_cv.
    Only templates are held out; gains are fitted on each evaluation trial. CV
    choices average directional per-cell statistics before transforms/aggregation.
    Bounded values use [0, 1] limits; raw, inverse, and relative ratios autoscale.
    Enabling this row requires pfpred_quality schema v13 results.

    Set ``residual_metric="unexplained_fraction"`` to show temporal 1 − R² for
    PF × gain in that bottom row. ``residual_rms_fold`` selects its split, and
    cv/within_cv average the two fold scores. This mode requires schema v14,
    uses the stored per-cell score directly, and autoscales above one.

    ``cells`` is all / quality / not quality / split. Quality requires spatial reliability
    strictly above ``reliability_threshold`` AND fraction active strictly above
    ``fraction_active_threshold``. Not quality is the complement among cells with finite
    classification measurements. Split overlays solid quality and dashed not-quality curves.
    Each metric uses its own finite values within that selection. Empty selections retain
    their session position, including in relative mode. Cutoffs need no trial-map recompute.

    ``include_gamma_control=True`` overlays dotted Gamma null means in the same
    environment colors for PF-Var Alignment, amplitude, and residual metrics.
    Spatial reliability and cell selection use observed data. The overlay is
    suppressed for ``cells="split"``. Gamma controls require schema v15.
    Relative values normalize each control by its own first-environment baseline.

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
    _METRIC_LABELS = {"correlation": "PF-Var Alignment", "reliability": "Spatial Reliability", "amplitude": "Amplitude Reliability",
                      "unexplained_fraction": r"PF × Gain $1-R^2$"}
    _RELOAD_CONTROLS = ("pf_rms_comparison", "amplitude_metric", "amplitude_statistic",
                        "include_residual_rms", "residual_metric", "residual_rms_fold", "residual_rms_statistic")
    _SUMMARY_CONTROLS = ("amplitude_transform", "residual_rms_transform")

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
        include_residual_rms: bool = False,
        residual_metric: str = "rms",
        residual_rms_fold: str = "all_from_all",
        residual_rms_statistic: str = "mean",
        residual_rms_transform: str = "raw",
        cells: str = "all",
        reliability_threshold: float = 0.7,
        fraction_active_threshold: float = 0.2,
        summary_stat: str = "mean",
        plot_style: str = "errorPlot",
        hide_error: bool = False,
        show_legend: bool = True,
        show_schematic: bool = True,
        include_mean_std: bool = True,
        include_gamma_control: bool = False,
        relative_values: bool = False,
        baseline_sessions: int = 1,
        min_mice: int = 2,
        ylim_full_correlation: bool = True,
        ylim_full_reliability: bool = True,
        ylim_full_amplitude: bool = True,
        fontsize: float = 9.0,
        schema_height_ratio: float = 0.8,
        figsize: tuple[float, float] = (12.0, 10.0),
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
        self._add_metric_widgets(dict(
            pf_rms_comparison=pf_rms_comparison, amplitude_metric=amplitude_metric,
            amplitude_statistic=amplitude_statistic, amplitude_transform=amplitude_transform,
            include_mean_std=include_mean_std, ylim_full_correlation=ylim_full_correlation,
            ylim_full_reliability=ylim_full_reliability, ylim_full_amplitude=ylim_full_amplitude,
            include_residual_rms=include_residual_rms, residual_metric=residual_metric, residual_rms_fold=residual_rms_fold,
            residual_rms_statistic=residual_rms_statistic,
            residual_rms_transform="inverse" if residual_rms_transform == "inv" else residual_rms_transform,
        ))
        self.add_selection("cells", value=cells, options=["all", "quality", "not quality", "split"])
        self.add_float("reliability_threshold", value=reliability_threshold, min=-1.0, max=1.0, step=0.01)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0, step=0.01)
        self.add_selection("summary_stat", value=summary_stat, options=["mean", "median"])
        self.add_selection("plot_style", value=plot_style, options=["each", "errorPlot"])
        self.add_boolean("hide_error", value=hide_error)
        self.add_boolean("show_legend", value=show_legend)
        self.add_boolean("show_schematic", value=show_schematic)
        self.add_boolean("include_gamma_control", value=include_gamma_control)
        self.add_boolean("relative_values", value=relative_values)
        self.add_integer("baseline_sessions", value=baseline_sessions, min=1, max=5)
        self.add_integer("min_mice", value=min(min_mice, len(self.mice)), min=1, max=len(self.mice))
        self.add_float("fontsize", value=fontsize, min=3.0, max=30.0)
        self.add_float("schema_height_ratio", value=schema_height_ratio, min=0.1, max=5.0)
        self.on_change(
            [*self.selection_names, *self._RELOAD_CONTROLS, "include_gamma_control", "cells"],
            self.reload_arrays,
        )
        self.on_change(
            ["reliability_threshold", "fraction_active_threshold", "summary_stat",
             "relative_values", "baseline_sessions", *self._SUMMARY_CONTROLS],
            self.refresh_summary_data,
        )
        self.reload_arrays(self.state)

    def _add_metric_widgets(self, defaults):
        options = {
            "pf_rms_comparison": [key.removeprefix("pf_rms_r_") for key in pf_rms_correlation_keys()],
            "amplitude_metric": list(TRIAL_ACTIVITY_METRICS),
            "amplitude_statistic": list(TRIAL_ACTIVITY_STATISTICS),
            "amplitude_transform": list(_AMPLITUDE_TRANSFORMS),
            "residual_rms_fold": list(PF_VARIANCE_COMPARISONS),
            "residual_rms_statistic": list(TRIAL_ACTIVITY_STATISTICS),
            "residual_rms_transform": list(_AMPLITUDE_TRANSFORMS),
            "residual_metric": ["rms", "unexplained_fraction"],
        }
        for name, values in options.items():
            self.add_selection(name, value=defaults[name], options=values)
        for name in ("include_mean_std", "ylim_full_correlation", "ylim_full_reliability", "ylim_full_amplitude",
                     "include_residual_rms"):
            self.add_boolean(name, value=defaults[name])

    def reload_arrays(self, state):
        keys = {
            "correlation": f"pf_rms_r_{state['pf_rms_comparison']}_slot",
            "reliability": "reliability_slot",
            "amplitude": f"{state['amplitude_metric']}_{state['amplitude_statistic']}",
        }
        schema_version = "v11"
        if state["include_residual_rms"]:
            fold = state["residual_rms_fold"]
            if state["residual_metric"] == "unexplained_fraction":
                keys["unexplained_fraction"] = f"pf_frac_unexplained_{fold}_slot"
                schema_version = "v14"
            else:
                keys["residual_rms"] = f"trial_residual_rms_{state['residual_rms_statistic']}_{fold}"
                schema_version = "v13"
        self._load_metric_arrays(state, keys, schema_version=schema_version)

    def _load_metric_arrays(self, state, metric_keys, *, schema_version):
        """Align plotted metrics and classification arrays on the same session/slot/ROI axes."""
        if state["include_gamma_control"] and state["cells"] != "split":
            gamma_keys = {}
            for metric, key in metric_keys.items():
                if metric == "reliability":
                    continue
                slot_key = key if key.endswith("_slot") else f"{key}_slot"
                gamma_keys[f"gamma_{metric}"] = f"gamma_mean_{slot_key}"
            metric_keys = {**metric_keys, **gamma_keys}
            schema_version = "v15 with Gamma simulations enabled"
        selection = data_selection(state, self.results, self.selection_names)
        self.arrays, self.env_slot_ids, self.env_present = load_placefield_cell_arrays(
            self.results, selection, metric_keys, schema_version=schema_version,
        )
        self.refresh_summary_data(state)

    def _summary_values(self, state):
        values = {**self.arrays, "amplitude": _transform_amplitude(self.arrays["amplitude"], state["amplitude_transform"])}
        if "residual_rms" in self.arrays:
            transform = state["residual_rms_transform"]
            values["residual_rms"] = _transform_amplitude(self.arrays["residual_rms"], "inverse" if transform == "inv" else transform)
        for metric, transform in (("amplitude", state["amplitude_transform"]),
                                  ("residual_rms", state["residual_rms_transform"])):
            key = f"gamma_{metric}"
            if key in self.arrays:
                values[key] = _transform_amplitude(self.arrays[key], "inverse" if transform == "inv" else transform)
        return values

    def refresh_summary_data(self, state):
        """Summarize once per session, then place the same values in both coordinates."""
        groups = ("quality", "not quality") if state["cells"] == "split" else (state["cells"],)
        metrics = self._METRICS + tuple(metric for metric in ("residual_rms", "unexplained_fraction") if metric in self.arrays)
        if state["include_gamma_control"] and state["cells"] != "split":
            metrics += tuple(f"gamma_{metric}" for metric in metrics if f"gamma_{metric}" in self.arrays)
        self.metric_stacks = {mode: {group: {metric: {} for metric in metrics} for group in groups} for mode in ("absolute", "relative")}
        self.baselines = {group: {} for group in groups}
        reliability = self.arrays["reliability"]
        active = self.arrays["fraction_active"]
        classified = np.isfinite(reliability) & np.isfinite(active)
        quality = (reliability > state["reliability_threshold"]) & (active > state["fraction_active_threshold"])
        transformed = self._summary_values(state)
        reduce = np.mean if state["summary_stat"] == "mean" else np.median
        for group in groups:
            mask = np.ones_like(quality) if group == "all" else classified & (quality if group == "quality" else ~quality)
            summaries = {}
            for metric in metrics:
                values = transformed[metric]
                summary = np.full(values.shape[:2], np.nan)
                for row, slot in np.ndindex(summary.shape):
                    keep = mask[row, slot] & np.isfinite(values[row, slot])
                    if metric.startswith("gamma_"):
                        keep &= np.isfinite(transformed[metric.removeprefix("gamma_")][row, slot])
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

    def _panel_stack(self, stack, state, mouse=None):
        """Select one mouse, or mask points lacking across-mouse support."""
        if mouse is not None:
            index = self.mice.index(mouse)
            return stack[index:index + 1]
        return np.where((np.isfinite(stack).sum(axis=0) >= state["min_mice"])[None, :], stack, np.nan)

    def _draw_summary_panel(self, ax, mode, metric, state, *, mouse=None):
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
            series = [(metric, False)]
            if state["include_gamma_control"] and state["cells"] != "split" and f"gamma_{metric}" in metrics:
                series.append((f"gamma_{metric}", True))
            curves = [(slot, stack, gamma) for key, gamma in series for slot, stack in metrics[key].items()]
            for slot, stack, gamma in curves:
                # Apply support to every plotted series, including plot_style="each".
                # Keep unsupported columns as gaps so later exposures stay aligned.
                stack = self._panel_stack(stack, state, mouse)
                if not np.isfinite(stack).any():
                    continue
                label = f"{ordinal(slot + 1)} Env"
                if state["cells"] != "all":
                    label += f" ({group})"
                if gamma:
                    label += " (Gamma)"
                previous_lines = len(ax.lines)
                render_curve_group(
                    ax,
                    np.arange(1, stack.shape[1] + 1),
                    stack,
                    env_slot_color(slot),
                    state["plot_style"] if mouse is None else "errorPlot",
                    label=label,
                    hide_error=state["hide_error"] or mouse is not None,
                    min_support=state["min_mice"] if mouse is None else 1,
                    linewidth=1.5,
                )
                for line in ax.lines[previous_lines:]:
                    line.set_linestyle(":" if gamma else "--" if group == "not quality" else "-")
                    if np.isfinite(line.get_ydata()).sum() == 1:
                        line.set_marker(".")
        bottom = metric == self._visible_metrics(state)[-1]
        if bottom:
            ax.set_xlabel("Overall session #" if mode == "absolute" else "Env session #", fontsize=state["fontsize"])
        ax.set_xlim(self._column_xlim(mode, state, mouse=mouse))
        self._set_metric_ylim(ax, metric, state)
        style_axis(ax, fontsize=state["fontsize"], spines_visible=("left", "bottom") if bottom else ("left",))
        # Later columns can expand the shared y limits. Keep the spine below the
        # panel rather than at a data coordinate computed from an earlier range.
        ax.spines["bottom"].set_position(("axes", -0.02))
        if not bottom:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

    def _set_metric_ylim(self, ax, metric, state):
        if metric == "unexplained_fraction":
            return
        if metric == "residual_rms":
            if state["residual_rms_transform"] == "bounded" and not state["relative_values"]:
                ax.set_ylim(0.0, 1.0)
            return
        if state[f"ylim_full_{metric}"] and not state["relative_values"]:
            if metric == "correlation":
                ax.set_ylim(-1.0, 1.0)
            elif metric == "reliability" or state["amplitude_transform"] == "bounded":
                ax.set_ylim(0.0, 1.0)

    def _visible_metrics(self, state):
        metrics = self._METRICS if state["include_mean_std"] else self._METRICS[1:]
        if state["include_residual_rms"]:
            metric = "unexplained_fraction" if state["residual_metric"] == "unexplained_fraction" else "residual_rms"
            metrics += (metric,)
        return metrics

    def _column_xlim(self, mode, state, *, mouse=None):
        """Shared extent of supported points across visible rows; ignore padded tails."""
        positions = [
            np.flatnonzero(np.isfinite(self._panel_stack(stack, state, mouse)).any(axis=0)) + 1
            for metrics in self.metric_stacks[mode].values()
            for metric, stacks in metrics.items()
            if metric.removeprefix("gamma_") in self._visible_metrics(state)
            for stack in stacks.values()
        ]
        positions = [values for values in positions if values.size]
        if not positions:
            return (0.5, 1.5)
        low = min(values[0] for values in positions)
        high = max(values[-1] for values in positions)
        return (low, high) if low < high else (low - 0.5, high + 0.5)

    def _summary_timeline_ranges(self, state):
        """Exposure endpoints supported by any visible metric/cell-group curve."""
        ranges = {}
        for metrics in self.metric_stacks["relative"].values():
            for metric, stacks in metrics.items():
                if metric.removeprefix("gamma_") not in self._visible_metrics(state):
                    continue
                for slot, stack in stacks.items():
                    positions = np.flatnonzero(np.isfinite(self._panel_stack(stack, state)).any(axis=0)) + 1
                    if positions.size:
                        low, high = ranges.get(slot, (positions[0], positions[-1]))
                        ranges[slot] = (min(low, positions[0]), max(high, positions[-1]))
        return dict(sorted(ranges.items()))

    def plot(self, state):
        metrics = self._visible_metrics(state)
        data_start = int(state["show_schematic"])
        fig, axes = self.new_subplots(
            len(metrics) + data_start,
            3,
            figsize=self.figsize,
            layout="constrained",
            height_ratios=([state["schema_height_ratio"]] if data_start else []) + [1] * len(metrics),
            sharey="row",
        )
        timeline_state = {**self.timeline.state, "mouse": state["mouse"], "num_training": 0, "show_phase_bar": False}
        if data_start:
            self.timeline.refresh_data(timeline_state)
        for column, mode in enumerate(("absolute", "relative", "relative")):
            mouse = state["mouse"] if column < 2 else None
            if data_start:
                self.timeline.draw(
                    axes[0, column],
                    {**timeline_state, "mode": mode, "imaging_label": "Chronological Time" if mode == "absolute" else "Relative Time"},
                    session_xlim=self._column_xlim(mode, state, mouse=mouse),
                    relative_env_ranges=self._summary_timeline_ranges(state) if column == 2 else None,
                )
            for row, metric in enumerate(metrics, start=data_start):
                self._draw_summary_panel(axes[row, column], mode, metric, state, mouse=mouse)
                if column > 0:
                    axes[row, column].spines["left"].set_visible(False)
                    axes[row, column].tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
                if column == 0:
                    if metric == "residual_rms":
                        statistic = {"mean": "Mean", "variance": "Variance", "cv": "CoV"}[state["residual_rms_statistic"]]
                        transform = state["residual_rms_transform"]
                        prefix = {"raw": "", "inverse": "Inverse ", "inv": "Inverse ", "bounded": "Bounded "}[transform]
                        label = f"{prefix}Residual RMS {statistic}"
                    else:
                        label = self._METRIC_LABELS[metric]
                    if metric == "amplitude" and state["amplitude_metric"] in ("trial_gain", "trial_gain_least_squares"):
                        if state["amplitude_transform"] == "raw":
                            statistic = {"mean": "Mean", "variance": "Variance", "cv": "CoV"}[state["amplitude_statistic"]]
                            label = f"Gain {statistic}"
                        else:
                            label = "Gain Reliability"
                    axes[row, column].set_ylabel(label, fontsize=state["fontsize"])
            if state["show_legend"] and axes[data_start, column].get_legend_handles_labels()[0]:
                axes[data_start, column].legend(frameon=False, fontsize=state["fontsize"])
        return fig
