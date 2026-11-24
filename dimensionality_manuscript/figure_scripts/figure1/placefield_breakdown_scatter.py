"""Single-session scatter plots of spatial reliability, gain, and residual activity."""

import numpy as np
from matplotlib.ticker import MaxNLocator

from dimensionality_manuscript.configs.pfpred_quality import (
    PF_VARIANCE_COMPARISONS,
    TRIAL_ACTIVITY_METRICS,
    TRIAL_ACTIVITY_STATISTICS,
)
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import FigureViewer, add_data_selection_widgets, data_selection
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import style_axis
from .placefield import _AMPLITUDE_METRIC_LABELS, _AMPLITUDE_STATISTIC_LABELS, _AMPLITUDE_TRANSFORMS, _transform_amplitude


class PlacefieldBreakdownScatter(FigureViewer):
    """Compare spatial reliability with trial gain and the residual for one session/environment.

    Environment and ROI indices follow PlacefieldBreakdownFocus. Use -1 to disable
    either ROI highlight. Plotted ROIs, including highlights, must have finite
    coordinates and exceed both the spatial reliability and fraction active
    thresholds.

    The amplitude options match HolisticPlacefieldFamiliarityViewer and default to
    the raw variance of least-squares trial gain. Residual RMS uses the selected
    fold, trial statistic, and transform. The unexplained fraction uses the stored
    temporal PF × Gain score (1 − R²) for the selected fold, which can exceed one.
    Observed mode needs the trial RMS keys from v13 or the fraction keys from v14.

    Set ``metric_source="gamma"`` to plot each ROI's stored Gamma null mean for the
    selected amplitude and residual statistics. Reliability and fraction-active
    filtering always use the observed data. Gamma mode requires v15 results with
    the null enabled. Use ``metric_source="both"`` to overlay observed points in
    black and Gamma points in red. Selected Gamma ROIs have larger black backing
    markers beneath their selected colors.

    Vertical panels share a spatial reliability x-axis. Horizontal panels share a
    spatial reliability y-axis. Pass figsize to override the size for either layout.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        example_session: str | None = None,
        env: int = 0,
        orientation: str = "vertical",
        reliability_threshold: float = -1.0,
        fraction_active_threshold: float = 0.01,
        metric_source: str = "observed",
        amplitude_metric: str = "trial_gain_least_squares",
        amplitude_statistic: str = "variance",
        amplitude_transform: str = "raw",
        residual_metric: str = "rms",
        residual_rms_fold: str = "all_from_all",
        residual_rms_statistic: str = "mean",
        residual_rms_transform: str = "raw",
        roi1: int = -1,
        roi2: int = -1,
        roi1_color: str = "darkviolet",
        roi2_color: str = "magenta",
        fontsize: float = 8.0,
        figsize: tuple[float, float] | None = None,
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self._session_result_cache = None
        self._sessions_by_mouse = {}
        for session in sorted(results.sessions, key=lambda session: (session.mouse_name, _session_sort_key(session))):
            self._sessions_by_mouse.setdefault(session.mouse_name, {})[session.session_print()] = session
        if not self._sessions_by_mouse:
            raise ValueError("The aggregator contains no sessions.")
        mice = list(self._sessions_by_mouse)
        if mouse is None:
            mouse = example_session.split("/", 1)[0] if example_session is not None else mice[0]
        sessions = list(self._sessions_by_mouse[mouse])

        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("mouse", value=mouse, options=mice)
        self.add_selection("example_session", value=example_session or sessions[0], options=sessions)
        self.add_integer("env", value=env, min=0, max=max(env, 0))
        self.add_selection("orientation", value=orientation, options=["vertical", "horizontal"])
        self.add_float("reliability_threshold", value=reliability_threshold, min=-1.0, max=1.0, step=0.01)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0, step=0.01)
        self._add_metric_source_widget(metric_source)
        self.add_selection("amplitude_metric", value=amplitude_metric, options=list(TRIAL_ACTIVITY_METRICS))
        self.add_selection("amplitude_statistic", value=amplitude_statistic, options=list(TRIAL_ACTIVITY_STATISTICS))
        self.add_selection("amplitude_transform", value=amplitude_transform, options=list(_AMPLITUDE_TRANSFORMS))
        self.add_selection("residual_metric", value=residual_metric, options=["rms", "unexplained_fraction"])
        self.add_selection("residual_rms_fold", value=residual_rms_fold, options=list(PF_VARIANCE_COMPARISONS))
        self.add_selection("residual_rms_statistic", value=residual_rms_statistic, options=list(TRIAL_ACTIVITY_STATISTICS))
        self.add_selection(
            "residual_rms_transform",
            value="inverse" if residual_rms_transform == "inv" else residual_rms_transform,
            options=list(_AMPLITUDE_TRANSFORMS),
        )
        self.add_integer("roi1", value=roi1, min=-1, max=max(roi1, -1))
        self.add_integer("roi2", value=roi2, min=-1, max=max(roi2, -1))
        self.add_selection("roi1_color", value=roi1_color, options=["darkviolet", "purple", "indigo", "blueviolet", "rebeccapurple"])
        self.add_selection("roi2_color", value=roi2_color, options=["magenta", "darkmagenta", "deeppink", "mediumvioletred", "crimson"])
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.on_change("mouse", self.update_example_session)
        self.on_change(["example_session", *self.selection_names], self.update_example_bounds)
        self.update_example_bounds(self.state)

    def update_example_session(self, state):
        """Offer the selected mouse's sessions in chronological order."""
        sessions = list(self._sessions_by_mouse[state["mouse"]])
        current = state["example_session"] if state["example_session"] in sessions else sessions[0]
        self.update_selection("example_session", value=current, options=sessions)
        self.update_example_bounds(self.state)

    def _session_result(self, state):
        session = self._sessions_by_mouse[state["mouse"]][state["example_session"]]
        selection = data_selection(state, self.results, self.selection_names)
        schema_version = getattr(self.results, "_schema_version", None)
        if schema_version is not None:
            selection["schema_version"] = schema_version
        cache_key = (session.session_uid, tuple(selection.items()))
        if self._session_result_cache is None or self._session_result_cache[0] != cache_key:
            config = self.results.config_class(**selection)
            selected = self.results.store.get(session.session_uid, config)
            if selected is None:
                raise KeyError(f"No pfpred_quality result for {session.session_print()} and the selected parameters.")
            self._session_result_cache = (cache_key, selected)
        return session, self._session_result_cache[1]

    def update_example_bounds(self, state):
        """Keep environment and ROI sliders within the selected session's bounds."""
        session, selected = self._session_result(state)
        self.update_integer("env", max=max(len(session.environments) - 1, 0))
        last_roi = np.asarray(selected["reliability_slot"]).shape[-1] - 1
        self.update_integer("roi1", max=last_roi)
        self.update_integer("roi2", max=last_roi)

    @staticmethod
    def _metric_label(label, statistic, transform):
        label = f"{label} {_AMPLITUDE_STATISTIC_LABELS[statistic]}"
        prefix = {"raw": "", "inverse": "Inverse ", "bounded": "Bounded "}[transform]
        return prefix + label

    def _add_metric_source_widget(self, metric_source):
        self.add_selection("metric_source", value=metric_source, options=["observed", "gamma", "both"])

    def _metric_sources(self, state):
        return ("observed", "gamma") if state["metric_source"] == "both" else (state["metric_source"],)

    def _scatter_data(self, state):
        """Load aligned per-ROI metrics and the observed-data selection mask."""
        session, selected = self._session_result(state)
        amplitude_metric = state["amplitude_metric"]
        amplitude_statistic = state["amplitude_statistic"]
        amplitude_key = f"{amplitude_metric}_{amplitude_statistic}"
        fold = state["residual_rms_fold"]
        amplitude_transform = state["amplitude_transform"]
        metric_label = _AMPLITUDE_METRIC_LABELS[amplitude_metric]
        if amplitude_metric in ("trial_gain", "trial_gain_least_squares"):
            metric_label = "Gain"
        amplitude_label = self._metric_label(metric_label, amplitude_statistic, amplitude_transform)
        if state["residual_metric"] == "unexplained_fraction":
            residual_key = f"pf_frac_unexplained_{fold}_slot"
            residual_label = r"PF × Gain $1-R^2$"
            residual_transform, required_results = "raw", "schema v14 or later"
        else:
            statistic = state["residual_rms_statistic"]
            residual_key = f"trial_residual_rms_{statistic}_{fold}"
            residual_transform, required_results = state["residual_rms_transform"], "schema v13 or later"
            residual_label = self._metric_label("Residual RMS", statistic, residual_transform)

        sources = self._metric_sources(state)
        metric_keys = {}
        for source in sources:
            if source == "gamma":
                gamma_residual_key = residual_key if residual_key.endswith("_slot") else f"{residual_key}_slot"
                metric_keys[source] = (f"gamma_mean_{amplitude_key}_slot", f"gamma_mean_{gamma_residual_key}")
            else:
                metric_keys[source] = (amplitude_key, residual_key)
        if sources == ("gamma",):
            amplitude_label += " (Gamma)"
            residual_label += " (Gamma)"
        if "gamma" in sources:
            required_results = "schema v15 or later with Gamma simulations enabled"

        keys = ["reliability_slot", "fraction_active_slot", "env_slot_ids"]
        keys += [key for source_keys in metric_keys.values() for key in source_keys]
        missing = [key for key in keys if key not in selected]
        if missing:
            raise KeyError(f"Missing {missing} in pfpred_quality results. Select {required_results}.")
        environment = session.environments[state["env"]]
        slots = np.flatnonzero(np.asarray(selected["env_slot_ids"]) == environment)
        if slots.size != 1:
            raise ValueError(f"Environment {environment} has no unique stored slot for {session.session_print()}.")
        slot = slots[0]
        reliability = np.asarray(selected["reliability_slot"], dtype=float)[slot]
        active = np.asarray(selected["fraction_active_slot"], dtype=float)[slot]
        metrics = {
            source: [np.asarray(selected[key], dtype=float)[slot] for key in source_keys]
            for source, source_keys in metric_keys.items()
        }
        values_to_check = [active] + [values for source_values in metrics.values() for values in source_values]
        if reliability.ndim != 1 or any(values.shape != reliability.shape for values in values_to_check):
            raise ValueError("Stored spatial reliability, amplitude, residual, and fraction active must share the same ROI axis.")
        eligible = (reliability > state["reliability_threshold"]) & np.isfinite(active) & (active > state["fraction_active_threshold"])
        panels = [(amplitude_label, amplitude_transform), (residual_label, residual_transform)]
        return reliability, eligible, panels, metrics

    def plot(self, state):
        reliability, eligible, panels, metrics = self._scatter_data(state)
        sources = self._metric_sources(state)

        vertical = state["orientation"] == "vertical"
        layout = (2, 1) if vertical else (1, 2)
        figsize = self.figsize or ((3.5, 4.4) if vertical else (6.0, 2.6))
        fig, axes = self.new_subplots(*layout, sharex=vertical, sharey=not vertical, figsize=figsize, layout="constrained")
        fontsize = state["fontsize"]
        for index, (ax, (label, transform)) in enumerate(zip(axes, panels)):
            for source in sources:
                values = _transform_amplitude(metrics[source][index], transform)
                x, y = (reliability, values) if vertical else (values, reliability)
                visible = eligible & np.isfinite(x) & np.isfinite(y)
                ax.scatter(
                    x[visible], y[visible], color="red" if source == "gamma" else "k",
                    s=4, alpha=0.3, linewidths=0,
                )
                for roi_name, marker, size in (("roi1", "H", 50), ("roi2", "D", 35)):
                    roi = state[roi_name]
                    if 0 <= roi < visible.size and visible[roi]:
                        if source == "gamma":
                            ax.scatter(
                                x[roi], y[roi], color="black", s=size * 1.8,
                                marker=marker, linewidths=0, zorder=3,
                            )
                        ax.scatter(
                            x[roi],
                            y[roi],
                            color=state[f"{roi_name}_color"],
                            s=size,
                            marker=marker,
                            edgecolors="white" if source == "observed" else "none",
                            linewidths=0.5 if source == "observed" else 0,
                            zorder=4,
                        )

            metric_axis = ax.yaxis if vertical else ax.xaxis
            metric_axis.set_major_locator(MaxNLocator(nbins=4))
            if transform == "bounded":
                set_limits = ax.set_ylim if vertical else ax.set_xlim
                set_limits(-0.035, 1.035)
                metric_axis.set_ticks([0, 0.5, 1])

            show_shared = index == (1 if vertical else 0)
            if vertical:
                ax.set_ylabel(label, fontsize=fontsize)
                if show_shared:
                    ax.set_xlabel("Spatial Reliability", fontsize=fontsize)
            else:
                ax.set_xlabel(label, fontsize=fontsize)
                if show_shared:
                    ax.set_ylabel("Spatial Reliability", fontsize=fontsize)
            spines = ("left", "bottom") if show_shared else ("left",) if vertical else ("bottom",)
            style_axis(ax, fontsize=fontsize, spines_visible=spines)
            if not show_shared:
                ax.tick_params(axis="x" if vertical else "y", bottom=False, left=False, labelbottom=False, labelleft=False)

        shared_axis = axes[-1].xaxis if vertical else axes[0].yaxis
        shared_axis.set_major_locator(MaxNLocator(nbins=4))
        return fig


class PlacefieldBreakdownGammaScatter(PlacefieldBreakdownScatter):
    """Compare each ROI's Gamma null mean with its observed amplitude and residual.

    The two panels use the same controls, observed reliability/fraction-active
    filters, and ROI highlights as PlacefieldBreakdownScatter. Both axes receive
    the selected metric transform. Gamma values are on x and observed values on
    y. Orientation controls whether the two panels are stacked or side by side.
    Each panel has its own axes because amplitude and residual use different
    metrics, with matching x/y limits and equal scaling. Unexplained fractions
    use [0, 1] on both axes. Set ``unity_line=True`` to add dotted unity lines
    to both panels. Requires v15 results with Gamma simulations enabled.
    """

    def __init__(self, results: ResultsAggregator, *, unity_line: bool = False, **kwargs):
        super().__init__(results, **kwargs)
        self.add_boolean("unity_line", value=unity_line)

    def _add_metric_source_widget(self, metric_source):
        """Both sources supply the coordinates, so no source selector is needed."""

    def _metric_sources(self, state):
        return ("observed", "gamma")

    def plot(self, state):
        _, eligible, panels, metrics = self._scatter_data(state)
        vertical = state["orientation"] == "vertical"
        layout = (2, 1) if vertical else (1, 2)
        figsize = self.figsize or ((3.5, 4.4) if vertical else (6.0, 2.6))
        fig, axes = self.new_subplots(*layout, figsize=figsize, layout="constrained")
        fontsize = state["fontsize"]
        for index, (ax, (label, transform)) in enumerate(zip(axes, panels)):
            gamma = _transform_amplitude(metrics["gamma"][index], transform)
            observed = _transform_amplitude(metrics["observed"][index], transform)
            visible = eligible & np.isfinite(gamma) & np.isfinite(observed)
            ax.scatter(gamma[visible], observed[visible], color="k", s=4, alpha=0.3, linewidths=0)
            for roi_name, marker, size in (("roi1", "H", 50), ("roi2", "D", 35)):
                roi = state[roi_name]
                if 0 <= roi < visible.size and visible[roi]:
                    ax.scatter(
                        gamma[roi], observed[roi], color="black", s=size * 1.8,
                        marker=marker, linewidths=0, zorder=3,
                    )
                    ax.scatter(
                        gamma[roi], observed[roi], color=state[f"{roi_name}_color"],
                        s=size, marker=marker, linewidths=0, zorder=4,
                    )
            if index == 1 and state["residual_metric"] == "unexplained_fraction":
                limits = (0, 1)
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([0, 0.5, 1])
            elif transform == "bounded":
                limits = (-0.035, 1.035)
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([0, 0.5, 1])
            else:
                x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
                limits = (min(x_limits[0], y_limits[0]), max(x_limits[1], y_limits[1]))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect("equal", adjustable="box")
            if state["unity_line"]:
                ax.plot(limits, limits, color="black", linestyle=":", linewidth=0.5, zorder=1)
            ax.set_xlabel(f"{label} (Gamma)", fontsize=fontsize)
            ax.set_ylabel(f"{label} (Observed)", fontsize=fontsize)
            style_axis(ax, fontsize=fontsize, spines_visible=("left", "bottom"))
        return fig
