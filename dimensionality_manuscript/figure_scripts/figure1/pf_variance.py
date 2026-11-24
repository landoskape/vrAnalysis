"""Activity decomposition examples and compact cross-validated population summaries.

    focus = PlacefieldVarianceFocus(results, mouse="ATL027", roi=0, xslice=slice(0, 1500))
    summary = PlacefieldVarianceSummary(results, mouse="ATL027", cells="split")
    familiarity = PlacefieldVarianceFamiliarity(results, cells="split")
    focus.show()  # or focus.plot(focus.state)

Focus uses the stored all-trial average PF, and fits gain over complete moving VR trials
before zooming. The stem plot measures variances over the displayed slice (or all retained
frames with ``variance_scope="session"``). Time is true elapsed session time; removed frames
are not compressed and lines break across gaps. Non-VR, slow, and undefined frames are excluded.

Summary and familiarity panels use ``pf_frac_var_*_cv_slot``: arithmetic averages of f2_from_f1
and f1_from_f2. PlacefieldVarianceDistributions has a fold selector defaulting to ``all_from_all``. All retain
values above one and signed covariance. Summaries first reduce ROIs
within each session/environment. Familiarity aligns actual environment exposures, preserving
missing metric values, and weights mice equally. Tail trimming affects display only
and reports excluded values. No data are pooled across cells to estimate a population variance.
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from dimensionality_manuscript.configs.pfpred_quality import _trial_gain_component, PF_VARIANCE_COMPARISONS
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts import session_cache
from dimensionality_manuscript.figure_scripts.panels import FigureViewer, add_data_selection_widgets, data_selection, render_curve_group
from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, update_legend_widgets, apply_legend
from vrAnalysis.processors.placefields import get_frame_behavior

from ._shared import env_slot_color, ordinal, pad_stack, style_axis


COMPONENTS = ("template", "gain", "additive")
METRICS = (*COMPONENTS, "covariance")
COLORS = {"template": "0.55", "gain": "tab:red", "additive": "tab:orange", "covariance": "0.3"}
LABELS = {"template": "PF component", "gain": "Gain component", "additive": "Residual", "covariance": "Covariance"}


def _session_rows(results):
    return {
        mouse: {results.sessions[row].session_print(): row for row in sorted(
            np.flatnonzero(np.asarray(results.mouse_names) == mouse),
            key=lambda row: _session_sort_key(results.sessions[row]),
        )}
        for mouse in sorted(set(str(mouse) for mouse in results.mouse_names))
    }


def _config(results, state, selection_names):
    params = data_selection(state, results, selection_names)
    if getattr(results, "_schema_version", None) is not None:
        params["schema_version"] = results._schema_version
    return results.config_class(**params)


def _broken_trace(time, values, breaks):
    """Insert gaps without dropping any retained observation."""
    return np.insert(time, breaks, np.nan), np.insert(values, breaks, np.nan)


def _add_variance_legend_widgets(viewer, options):
    options = dict(options or {})
    for axis in ("x", "y"):
        viewer.add_float(f"legend_{axis}_offset", value=float(options.pop(f"{axis}_offset", 0.)),
                         min=-2., max=2., step=0.01)
    add_legend_widgets(viewer)
    update_legend_widgets(viewer, options)


def _apply_variance_legend(ax, state, fontsize, auto_loc):
    apply_legend(ax, state, fontsize, auto_loc=auto_loc,
                 offset=(state["legend_x_offset"], state["legend_y_offset"]))


class PlacefieldVarianceFocus(FigureViewer):
    """One ROI's data, three components, position, and a narrow full-height variance stem.

    ROI indices refer to session-filtered ROIs, as in the existing PF viewers. By default all
    VR environments are included, using each environment's own stored average PF. Set
    ``all_environments=False`` to select one experience-order ``env_slot``. Variances are
    descriptive all-template measurements, distinct from the held-out population panels.
    """

    def __init__(self, results, *, mouse=None, session=None, roi=0, xslice=slice(0, 1500),
                 all_environments=True, env_slot=0, variance_scope="slice", fontsize=9.,
                 figsize=(10., 5.), **selection_defaults):
        self.results, self.figsize = results, figsize
        self._session_rows = _session_rows(results)
        if not self._session_rows:
            raise ValueError("No sessions available.")
        mouse = mouse or next(iter(self._session_rows))
        options = list(self._session_rows[mouse])
        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("mouse", value=mouse, options=list(self._session_rows))
        self.add_selection("session", value=session or options[0], options=options)
        self.add_integer("roi", value=roi, min=0, max=max(roi, 0))
        self.add_boolean("all_environments", value=all_environments)
        self.add_integer("env_slot", value=env_slot, min=0, max=max(env_slot, 0))
        start, stop = xslice.start or 0, xslice.stop or 1500
        self.add_integer("xslice_start", value=start, min=0, max=max(start, 0))
        self.add_integer("xslice_stop", value=max(start + 1, stop), min=1, max=max(start + 1, stop))
        self.add_selection("variance_scope", value=variance_scope, options=["slice", "session"])
        self.add_float("fontsize", value=fontsize, min=5., max=30.)
        self.on_change("mouse", self.update_session)
        self.on_change([*self.selection_names, "session"], self.load_session)
        self.on_change(["roi", "all_environments", "env_slot"], self.refresh_data)
        self.on_change("xslice_start", self.update_xslice_bounds)
        self.load_session(self.state)

    def update_session(self, state):
        options = list(self._session_rows[state["mouse"]])
        current = state["session"] if state["session"] in options else options[0]
        self.update_selection("session", value=current, options=options)
        self.load_session({**state, "session": current})

    def update_xslice_bounds(self, state):
        self.update_integer("xslice_stop", min=state["xslice_start"] + 1)

    def load_session(self, state):
        self.session = self.results.sessions[self._session_rows[state["mouse"]][state["session"]]]
        config = _config(self.results, state, self.selection_names)
        self.stored = self.results.store.get(self.session.session_uid, config)
        required = ("pf_all_slot", "pf_rms_distcenters", "env_slot_ids")
        if self.stored is None or any(key not in self.stored for key in required):
            raise KeyError("Missing average placefields; rerun the pfpred_quality sweep.")
        previous = self.session.params.spks_type
        self.session.params.spks_type = config.spks_type
        try:
            self.activity = session_cache.get_spks(self.session)
            self.behavior = get_frame_behavior(self.session)
            self.dist_edges = np.asarray(session_cache.get_smp(self.session).dist_edges)
            self.speed_threshold = session_cache.get_smp(self.session).params.speed_threshold
        finally:
            self.session.params.spks_type = previous
        self.timestamps = np.asarray(self.session.timestamps, dtype=float)
        self.timestamps = self.timestamps - self.timestamps[0]
        slot = min(state["env_slot"], len(self.stored["env_slot_ids"]) - 1)
        roi = self._configure_roi({**state, "env_slot": slot})
        self.update_integer("env_slot", value=slot, max=len(self.stored["env_slot_ids"]) - 1)
        self.refresh_data({**state, "roi": roi, "env_slot": slot})

    def _configure_roi(self, state):
        roi = min(state["roi"], self.activity.shape[1] - 1)
        self.update_integer("roi", value=roi, max=self.activity.shape[1] - 1)
        return roi

    def refresh_data(self, state):
        b = self.behavior
        bins = np.searchsorted(self.dist_edges, b.position, side="right") - 1
        valid = np.isfinite(b.position) & np.isfinite(b.trial) & np.isfinite(b.environment)
        valid &= (b.environment >= 0) & np.isfinite(b.speed) & (b.speed >= self.speed_threshold)
        valid &= (bins >= 0) & (bins < len(self.dist_edges) - 1) & np.isfinite(self.timestamps)
        prediction = np.full(len(b.position), np.nan)
        slots = range(len(self.stored["env_slot_ids"])) if state["all_environments"] else [state["env_slot"]]
        for slot in slots:
            env = self.stored["env_slot_ids"][slot]
            rows = np.flatnonzero(valid & (b.environment == env))
            prediction[rows] = self.stored["pf_all_slot"][slot, state["roi"], bins[rows]]
        y = self.activity[:, state["roi"]]
        valid &= np.isfinite(prediction) & np.isfinite(y)
        self.frames = np.flatnonzero(valid)
        if not len(self.frames):
            raise ValueError("No finite moving VR frames for this ROI/environment.")
        self.data = y[valid]
        self.components = {"template": prediction[valid]}
        # Trial IDs are session-global. Fit all retained frames of each trial before slicing.
        self.components["gain"] = _trial_gain_component(
            self.data[:, None], prediction[valid, None], b.trial[valid],
        )[:, 0]
        self.components["additive"] = self.data - self.components["template"] - self.components["gain"]
        self._update_slice_bounds(state)

    def _update_slice_bounds(self, state):
        n = len(self.frames)
        start = int(np.clip(state["xslice_start"], 0, n - 1))
        stop = int(np.clip(state["xslice_stop"], start + 1, n))
        self.update_integer("xslice_start", value=start, max=n - 1)
        self.update_integer("xslice_stop", value=stop, min=start + 1, max=n)

    def plot(self, state):
        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        grid = fig.add_gridspec(3, 2, width_ratios=[6, 1], height_ratios=[1, 1.5, 0.65])
        axes = [fig.add_subplot(grid[0, 0])]
        axes.extend(fig.add_subplot(grid[row, 0], sharex=axes[0]) for row in (1, 2))
        stem = fig.add_subplot(grid[:, 1])
        selected = slice(state["xslice_start"], state["xslice_stop"])
        frames = self.frames[selected]
        time = self.timestamps[frames]
        breaks = np.flatnonzero(np.diff(frames) != 1) + 1
        fontsize = state["fontsize"]
        axes[0].plot(*_broken_trace(time, self.data[selected], breaks), color="k", linewidth=0.8)
        axes[0].set_ylabel("Data", fontsize=fontsize)
        for component in COMPONENTS:
            axes[1].plot(*_broken_trace(time, self.components[component][selected], breaks),
                         color=COLORS[component], label=LABELS[component], linewidth=0.9)
        axes[1].axhline(0, color="0.8", linewidth=0.5, zorder=0)
        axes[1].set_ylabel("Components", fontsize=fontsize)
        axes[1].legend(frameon=False, fontsize=fontsize, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=3)
        position_breaks = np.union1d(breaks, np.flatnonzero(
            np.diff(self.behavior.trial[frames]) != 0) + 1).astype(int)
        axes[2].plot(*_broken_trace(time, self.behavior.position[frames], position_breaks), color="k", linewidth=0.8)
        axes[2].set_ylabel("Position (cm)", fontsize=fontsize)
        axes[2].set_xlabel("Time (s)", fontsize=fontsize)
        for row, ax in enumerate(axes):
            if len(time) > 1:
                ax.set_xlim(time[0], time[-1])
            style_axis(ax, fontsize=fontsize, spines_visible=("left", "bottom") if row == 2 else ("left",))
            ax.tick_params(axis="x", bottom=row == 2, labelbottom=row == 2)
        scope = selected if state["variance_scope"] == "slice" else slice(None)
        self.display_variances = {name: float(np.var(values[scope])) for name, values in self.components.items()}
        for index, name in enumerate(COMPONENTS):
            value = self.display_variances[name]
            stem.vlines(index, 0, value, color=COLORS[name], linewidth=1.5)
            stem.plot(index, value, "o", color=COLORS[name], markersize=4)
        stem.set_xlim(-0.5, 2.5)
        stem.set_ylim(bottom=0)
        stem.set_ylabel("Variance", fontsize=fontsize)
        stem.set_title("Shown interval" if state["variance_scope"] == "slice" else "Session", fontsize=fontsize)
        style_axis(stem, fontsize=fontsize)
        stem.set_xticks(range(3), ["PF", "Gain", "Residual"], rotation=60, ha="right")
        fig.suptitle(f"{self.session.session_print()} · ROI {state['roi']}", fontsize=fontsize)
        return fig


class PlacefieldVarianceTrialFocus(PlacefieldVarianceFocus):
    """Position-scaled trial traces with a quality-filtered, variance-sorted ROI selector.

    ``xslice`` selects chronological trials *within the selected environment*, not frames.
    At each position bin the curves show the mean of the original moving imaging samples;
    gains are fitted on complete temporal trials before binning or zooming. Stem variances
    retain temporal frame weighting (over shown trials or the whole selected environment).
    Actual session trial IDs label trial centers. Curves join observed bin means across
    unsampled positions within each trial; NaNs separate trials without shifting samples.
    Unsampled bins remain NaN in ``trial_maps`` and do not enter variances.

    Reliability and fraction active must be strictly above their thresholds in ``env_slot``.
    Sort keys use stored held-out mean fractions: gain / residual / PF. The dropdown values
    are actual session-filtered ROI indices, ordered largest-first by default; NaNs sort last.
    ``roi=None`` initially selects the first eligible ROI. Empty selections display a message.
    ``column_ratio`` is trace-column width / stem-column width. ``legend_options`` seeds the
    standard legend widgets; their one-column default gives a vertical component legend.
    """

    def __init__(self, results, *, mouse=None, session=None, env_slot=0, roi=None,
                 reliability_threshold=0.7, fraction_active_threshold=0.2,
                 sort_by="gain", sort_descending=True, xslice=slice(0, 5),
                 variance_scope="slice", column_ratio=6.0, legend_options=None, component_legend_horizontal=False,
                 fontsize=8.0, figsize=(7.5, 3.5), **selection_defaults):
        self.results, self.figsize = results, figsize
        self._session_rows = _session_rows(results)
        if not self._session_rows:
            raise ValueError("No sessions available.")
        mouse = mouse or next(iter(self._session_rows))
        options = list(self._session_rows[mouse])
        self._initial_roi = roi
        self._initialized_roi = False
        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("mouse", value=mouse, options=list(self._session_rows))
        self.add_selection("session", value=session or options[0], options=options)
        self.add_integer("env_slot", value=env_slot, min=0, max=max(env_slot, 0))
        self.add_selection("roi", value=roi, options=[roi])
        self.add_float("reliability_threshold", value=reliability_threshold, min=-1., max=1.)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0., max=1.)
        self.add_selection("sort_by", value=sort_by, options=["gain", "residual", "pf"])
        self.add_boolean("sort_descending", value=sort_descending)
        start, stop = xslice.start or 0, xslice.stop if xslice.stop is not None else 5
        self.add_integer("xslice_start", value=start, min=0, max=max(start, 0))
        self.add_integer("xslice_stop", value=max(start + 1, stop), min=1, max=max(start + 1, stop))
        self.add_selection("variance_scope", value=variance_scope, options=["slice", "session"])
        self.add_float("column_ratio", value=column_ratio, min=0.5, max=30.)
        self.add_boolean("component_legend_horizontal", value=component_legend_horizontal)
        self.add_float("fontsize", value=fontsize, min=5., max=30.)
        _add_variance_legend_widgets(self, legend_options)
        self.on_change("mouse", self.update_session)
        self.on_change([*self.selection_names, "session"], self.load_session)
        self.on_change(["env_slot", "reliability_threshold", "fraction_active_threshold", "sort_by", "sort_descending"],
                       self.update_roi_selection)
        self.on_change("roi", self.refresh_data)
        self.on_change("xslice_start", self.update_xslice_bounds)
        self.load_session(self.state)

    def _configure_roi(self, state):
        slot = state["env_slot"]
        metric = {"gain": "gain", "residual": "additive", "pf": "template"}[state["sort_by"]]
        key = f"pf_frac_var_{metric}_cv_slot"
        required = ("reliability_slot", "fraction_active_slot", key)
        if any(name not in self.stored for name in required):
            raise KeyError("Missing ROI quality/sort metrics; rerun the pfpred_quality v12 sweep.")
        rel, active = (np.asarray(self.stored[name][slot]) for name in required[:2])
        keep = np.isfinite(rel) & np.isfinite(active)
        keep &= (rel > state["reliability_threshold"]) & (active > state["fraction_active_threshold"])
        indices = np.flatnonzero(keep)
        values = np.asarray(self.stored[key][slot])[indices]
        scores = np.where(np.isfinite(values), -values if state["sort_descending"] else values, np.inf)
        self.roi_options = indices[np.argsort(scores, kind="stable")].tolist()
        preferred = state["roi"] if self._initialized_roi else self._initial_roi
        roi = preferred if preferred in self.roi_options else (self.roi_options[0] if self.roi_options else None)
        self._initialized_roi = True
        self.update_selection("roi", value=roi, options=self.roi_options or [None])
        return roi

    def update_roi_selection(self, state):
        roi = self._configure_roi(state)
        self.refresh_data({**state, "roi": roi})

    def _update_slice_bounds(self, state):
        self.trial_numbers = np.unique(self.behavior.trial[self.frames])
        n = len(self.trial_numbers)
        start = int(np.clip(state["xslice_start"], 0, n - 1))
        stop = int(np.clip(state["xslice_stop"], start + 1, n))
        self.update_integer("xslice_start", value=start, max=n - 1)
        self.update_integer("xslice_stop", value=stop, min=start + 1, max=n)

    def refresh_data(self, state):
        if state["roi"] is None:
            self.frames = np.array([], dtype=int)
            self.trial_numbers = np.array([])
            return
        super().refresh_data({**state, "all_environments": False})
        bins = np.searchsorted(self.dist_edges, self.behavior.position[self.frames], side="right") - 1
        trial_indices = np.searchsorted(self.trial_numbers, self.behavior.trial[self.frames])
        shape = (len(self.trial_numbers), len(self.dist_edges) - 1)
        linear_bins = trial_indices * shape[1] + bins
        count = np.bincount(linear_bins, minlength=np.prod(shape)).reshape(shape)
        self.trial_maps = {}
        for name, values in {"data": self.data, **self.components}.items():
            total = np.bincount(linear_bins, weights=values, minlength=np.prod(shape)).reshape(shape)
            self.trial_maps[name] = np.divide(total, count, out=np.full(shape, np.nan), where=count > 0)

    def plot(self, state):
        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        grid = fig.add_gridspec(3, 2, width_ratios=[state["column_ratio"], 1], height_ratios=[1, 1.4, 0.65])
        axes = [fig.add_subplot(grid[0, 0])]
        axes.extend(fig.add_subplot(grid[row, 0], sharex=axes[0]) for row in (1, 2))
        stem = fig.add_subplot(grid[:, 1])
        fontsize = state["fontsize"]
        if state["roi"] is None or not len(self.frames):
            axes[0].text(0.5, 0.5, "No ROIs pass the selected thresholds", ha="center", transform=axes[0].transAxes,
                         fontsize=fontsize)
            for ax in [*axes, stem]:
                ax.set_axis_off()
            return fig
        indices = np.arange(len(self.trial_numbers))[state["xslice_start"]:state["xslice_stop"]]
        centers = (self.dist_edges[:-1] + self.dist_edges[1:]) / 2
        position_x = (centers - self.dist_edges[0]) / (self.dist_edges[-1] - self.dist_edges[0])
        x = (np.arange(len(indices))[:, None] + position_x).ravel()
        def trial_line(values):
            finite = np.isfinite(values)
            xx, yy = x[finite], values[finite]
            breaks = np.flatnonzero(np.diff(np.floor(xx)) != 0) + 1
            return np.insert(xx, breaks, np.nan), np.insert(yy, breaks, np.nan)
        values = self.trial_maps["data"][indices].ravel()
        axes[0].plot(*trial_line(values), color="k", linewidth=0.8, label="data")
        for name in COMPONENTS:
            values = self.trial_maps[name][indices].ravel()
            axes[1].plot(*trial_line(values), color=COLORS[name], linewidth=0.9, label=LABELS[name])
        axes[2].plot(*trial_line(np.tile(centers, len(indices))), color="k", linewidth=0.7)
        axes[0].set_ylabel("Norm. Fluor.", fontsize=fontsize)
        axes[1].set_ylabel("Norm. Fluor.", fontsize=fontsize)
        axes[2].set_ylabel("Position (cm)", fontsize=fontsize)
        axes[2].set_xlabel("Trials", fontsize=fontsize)
        for row, ax in enumerate(axes):
            ax.set_xlim(0, len(indices))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
            style_axis(ax, fontsize=fontsize, spines_visible=("left", "bottom") if row == 2 else ("left",))
            ax.tick_params(axis="x", bottom=row == 2, labelbottom=row == 2)
        axes[2].set_xticks(np.arange(len(indices)) + 0.5, [f"{value:g}" for value in self.trial_numbers[indices]])
        _apply_variance_legend(axes[0], state, fontsize, "upper right")
        component_state = {**state, "legend_ncols": 3} if state["component_legend_horizontal"] else state
        _apply_variance_legend(axes[1], component_state, fontsize, "upper right")
        scope = (np.isin(self.behavior.trial[self.frames], self.trial_numbers[indices])
                 if state["variance_scope"] == "slice" else slice(None))
        self.display_variances = {name: float(np.var(values[scope])) for name, values in self.components.items()}
        for index, name in enumerate(COMPONENTS):
            value = self.display_variances[name]
            stem.vlines(index, 0, value, color=COLORS[name], linewidth=1.3)
            stem.plot(index, value, "o", color=COLORS[name], markersize=3)
        stem.set_xlim(-0.5, 2.5)
        stem.set_ylim(bottom=0)
        stem.set_ylabel("Variance", fontsize=fontsize)
        stem.yaxis.set_major_locator(MaxNLocator(nbins=2))
        style_axis(stem, fontsize=fontsize)
        stem.set_xticks(range(3), ["PF", "Gain", "Residual"], rotation=60, ha="right")
        return fig


class _VariancePopulationViewer(FigureViewer):
    """Shared CV arrays, quality selection, and per-session reductions."""

    def _initialize(self, results, mouse, session, env_slot, cells, reliability_threshold,
                    fraction_active_threshold, summary_stat, fontsize, selection_defaults):
        self.results = results
        self._session_rows = _session_rows(results)
        if not self._session_rows:
            raise ValueError("No sessions available.")
        self.mice = list(self._session_rows)
        mouse = mouse or self.mice[0]
        options = list(self._session_rows[mouse])
        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("mouse", value=mouse, options=self.mice)
        self.add_selection("session", value=session or options[0], options=options)
        self.add_integer("env_slot", value=env_slot, min=0, max=max(env_slot, 0))
        self._register_cells(cells)
        self.add_float("reliability_threshold", value=reliability_threshold, min=-1., max=1.)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0., max=1.)
        self.add_selection("summary_stat", value=summary_stat, options=["mean", "median"])
        self.add_float("fontsize", value=fontsize, min=5., max=30.)
        self.on_change("mouse", self.update_session)
        self.on_change(list(self.selection_names), self.reload_arrays)
        self.reload_arrays(self.state)

    def _register_cells(self, cells):
        self.add_selection("cells", value=cells, options=["all", "quality", "not quality", "split"])

    def update_session(self, state):
        options = list(self._session_rows[state["mouse"]])
        current = state["session"] if state["session"] in options else options[0]
        self.update_selection("session", value=current, options=options)

    def _variance_metric_keys(self):
        return [f"pf_frac_var_{metric}_cv_slot" for metric in METRICS]

    def reload_arrays(self, state):
        metric_keys = self._variance_metric_keys()
        keys = [*metric_keys, "reliability_slot", "fraction_active_slot", "env_slot_ids"]
        out = self.results.sel(keys=keys, squeeze_ones=False,
                               **data_selection(state, self.results, self.selection_names))
        missing = [key for key in keys if key not in out]
        if missing:
            raise KeyError(f"Missing {missing}; rerun the pfpred_quality schema v12 sweep.")
        arrays = [np.asarray(out[key], dtype=float) for key in keys[:-1]]
        if any(array.ndim != 3 for array in arrays) or len({array.shape[:2] for array in arrays}) != 1:
            raise ValueError("Select one configuration; arrays must be (sessions, environment slots, ROIs).")
        width = min(array.shape[-1] for array in arrays)
        if any(np.any(np.isfinite(array[..., width:])) for array in arrays):
            raise ValueError("Incompatible ROI axes beyond NaN padding.")
        self.arrays = dict(zip((*METRICS, "reliability", "active"), (array[..., :width] for array in arrays)))
        self.env_slot_ids = np.asarray(out["env_slot_ids"], dtype=float)
        shape = arrays[0].shape[:2]
        if self.env_slot_ids.shape != shape or shape[0] != len(self.results.sessions):
            raise ValueError("Environment slots do not match session axes.")
        self.env_present = np.array([[np.isfinite(env) and env >= 0 and env in session.environments for env in slots]
                                    for session, slots in zip(self.results.sessions, self.env_slot_ids)])
        slot = min(state["env_slot"], shape[1] - 1)
        self.update_integer("env_slot", value=slot, max=shape[1] - 1)

    def groups(self, state):
        return ("quality", "not quality") if state["cells"] == "split" else (state["cells"],)

    def mask(self, row, slot, group, state):
        rel, active = (self.arrays[name][row, slot] for name in ("reliability", "active"))
        if not self.env_present[row, slot]:
            return np.zeros(rel.shape, dtype=bool)
        if group == "all":
            return np.ones(rel.shape, dtype=bool)
        classified = np.isfinite(rel) & np.isfinite(active)
        quality = (rel > state["reliability_threshold"]) & (active > state["fraction_active_threshold"])
        return classified & (quality if group == "quality" else ~quality)

    def session_statistics(self, state=None):
        """Tidy session/environment/group metrics, including paired spatial Spearman r.

        All finite values enter summaries, including histogram tails and ratios above one.
        """
        state = self.state if state is None else state
        records = []
        for mouse, sessions in self._session_rows.items():
            exposures = np.zeros(self.env_present.shape[1], dtype=int)
            for chronological, (label, row) in enumerate(sessions.items(), 1):
                for slot in np.flatnonzero(self.env_present[row]):
                    exposures[slot] += 1
                    for group in self.groups(state):
                        mask = self.mask(row, slot, group, state)
                        for metric in METRICS:
                            values = self.arrays[metric][row, slot]
                            finite = mask & np.isfinite(values)
                            rel = self.arrays["reliability"][row, slot]
                            paired = finite & np.isfinite(rel)
                            rho = np.nan
                            if paired.sum() >= 3 and np.ptp(values[paired]) > 0 and np.ptp(rel[paired]) > 0:
                                rho = float(spearmanr(rel[paired], values[paired]).statistic)
                            records.append(dict(mouse=mouse, session=label, row=row, env_slot=slot,
                                                environment=self.env_slot_ids[row, slot], exposure=exposures[slot],
                                                chronological=chronological, group=group, component=metric,
                                                n_rois=int(finite.sum()),
                                                mean=float(np.mean(values[finite])) if finite.any() else np.nan,
                                                median=float(np.median(values[finite])) if finite.any() else np.nan,
                                                spatial_spearman_r=rho, n_paired=int(paired.sum())))
        return pd.DataFrame.from_records(records)


class PlacefieldVarianceSummary(_VariancePopulationViewer):
    """Selected-session histograms/reliability scatters and per-mouse session summaries.

    Columns are PF, gain, additive residual, and signed covariance. The bottom row follows
    the selected mouse in chronological time, colored black/blue/green by environment slot.
    ``display_percentile`` controls histogram/scatter limits (absolute tail for covariance);
    out-of-range counts are printed and all values still enter statistics.
    """

    def __init__(self, results, *, mouse=None, session=None, env_slot=0, cells="all",
                 reliability_threshold=0.7, fraction_active_threshold=0.2, summary_stat="median",
                 nbins=35, display_percentile=99., fontsize=9., figsize=(12., 8.), **selection_defaults):
        self.figsize = figsize
        self.add_integer("nbins", value=nbins, min=5, max=100)
        self.add_float("display_percentile", value=display_percentile, min=50., max=100.)
        self._initialize(results, mouse, session, env_slot, cells, reliability_threshold,
                         fraction_active_threshold, summary_stat, fontsize, selection_defaults)

    def plot(self, state):
        fig, axes = self.new_subplots(3, 4, figsize=self.figsize, layout="constrained")
        row = self._session_rows[state["mouse"]][state["session"]]
        slot, fontsize = state["env_slot"], state["fontsize"]
        stats = self.session_statistics(state)
        for column, metric in enumerate(METRICS):
            hax, sax, tax = axes[:, column]
            values = self.arrays[metric][row, slot]
            masks = {group: self.mask(row, slot, group, state) & np.isfinite(values) for group in self.groups(state)}
            combined = np.concatenate([values[mask] for mask in masks.values()])
            limit = float(np.percentile(np.abs(combined), state["display_percentile"])) if combined.size else 1.
            limit = max(limit, 1e-6)
            edges = np.linspace(-limit if metric == "covariance" else 0, limit, state["nbins"] + 1)
            outside = scatter_outside = 0
            for group, mask in masks.items():
                selected = values[mask]
                outside += int(np.sum((selected < edges[0]) | (selected > edges[-1])))
                counts, _ = np.histogram(selected, bins=edges)
                if selected.size:
                    hax.stairs(counts / selected.size / np.diff(edges), edges, color=COLORS[metric],
                               linestyle="--" if group == "not quality" else "-", label=group)
                rel = self.arrays["reliability"][row, slot]
                paired = mask & np.isfinite(rel)
                scatter_outside += int(np.sum(paired & ((values < edges[0]) | (values > edges[-1]))))
                sax.scatter(rel[paired], values[paired], s=4, alpha=0.25, color=COLORS[metric],
                            marker="x" if group == "not quality" else ".", rasterized=True)
            hax.set_title(LABELS[metric], fontsize=fontsize)
            hax.text(0.98, 0.97, f"n={combined.size}\noutside: {outside}", ha="right", va="top",
                     fontsize=fontsize - 1, transform=hax.transAxes)
            hax.set_xlabel("Variance / data variance", fontsize=fontsize)
            hax.set_ylabel("Density", fontsize=fontsize)
            sax.set_xlabel("Spatial reliability", fontsize=fontsize)
            sax.set_ylabel("Variance / data variance", fontsize=fontsize)
            sax.set_xlim(-1, 1)
            sax.set_ylim(edges[0], edges[-1])
            sax.text(0.02, 0.98, f"outside: {scatter_outside}", ha="left", va="top",
                     fontsize=fontsize - 1, transform=sax.transAxes)
            if metric == "covariance":
                sax.axhline(0, color="0.7", linewidth=0.5)
            for group in self.groups(state):
                for env_slot in range(self.env_present.shape[1]):
                    table = stats[(stats.mouse == state["mouse"]) & (stats.component == metric)
                                  & (stats.group == group) & (stats.env_slot == env_slot)]
                    if not table.empty:
                        tax.plot(table.chronological, table[state["summary_stat"]], color=env_slot_color(env_slot),
                                 linestyle="--" if group == "not quality" else "-", marker=".", markersize=3,
                                 label=f"{ordinal(env_slot + 1)} env · {group}")
            tax.set_xlabel("Overall session #", fontsize=fontsize)
            tax.set_ylabel(f"ROI {state['summary_stat']} ratio", fontsize=fontsize)
            for ax in (hax, sax, tax):
                style_axis(ax, fontsize=fontsize)
        if state["cells"] == "split":
            axes[0, 0].legend(frameon=False, fontsize=state["fontsize"] - 1)
        handles, labels = axes[2, -1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=fontsize - 1)
        env = self.env_slot_ids[row, slot]
        fig.suptitle(f"{state['session']} · environment {env:g} · held-out PF templates", fontsize=fontsize)
        return fig


class PlacefieldVarianceDistributions(_VariancePopulationViewer):
    """One session's histograms alongside session-balanced, equally weighted mouse averages.

    ``comparison`` selects the stored variance fractions for both columns, defaulting to
    ``all_from_all``. ``cv`` averages the two held-out directions. ``within_cv`` averages
    f1_from_f1 and f2_from_f2: these are within-fold fits, not held-out scores. ``session_cells`` and
    ``overview_cells`` independently select all / quality / non-quality / split; quality is
    assessed within each session/environment. Non-quality uses dotted curves.

    ``overview_environment='best'`` uses each session's stored best environment, as in
    PlaceFieldPopulationDistributions; 'selected' uses ``env_slot`` and 'all' first averages
    environment densities within each session. Sessions are then averaged within mouse and
    mice equally. Empty selections remain undefined, rather than zero-density observations.
    ``plot_style`` selects mouse-mean +/- STD bands or individual curves and their mean.

    Session and overview occupy columns; components occupy rows. All x axes span [0, 1].
    This range excludes negative covariance as well as fractions above one, without changing
    the stored values or renormalizing away tails. Counts
    outside the range are available in ``distribution_diagnostics``, not printed on the axes.
    Both top-panel legends share ``legend_options``: cell groups at left, component colors at
    right. Only the bottom row displays x tick labels and the common Variance Fraction label.
    """

    def __init__(self, results, *, mouse=None, session=None, env_slot=0,
                 include_covariance=False, session_cells="all", overview_cells="all",
                 reliability_threshold=0.7, fraction_active_threshold=0.2,
                 overview_environment="best", plot_style="errorPlot", comparison="all_from_all",
                 nbins=40, legend_options=None,
                 fontsize=8.0, figsize=(6.0, 6.0), **selection_defaults):
        self.figsize = figsize
        self._initial_overview_cells = overview_cells
        self.add_boolean("include_covariance", value=include_covariance)
        self.add_selection("overview_environment", value=overview_environment, options=["best", "selected", "all"])
        self.add_selection("plot_style", value=plot_style, options=["errorPlot", "each"])
        self.add_selection("comparison", value=comparison, options=list(PF_VARIANCE_COMPARISONS))
        self.add_integer("nbins", value=nbins, min=5, max=100)
        _add_variance_legend_widgets(self, legend_options)
        self._initialize(results, mouse, session, env_slot, session_cells, reliability_threshold,
                         fraction_active_threshold, "median", fontsize, selection_defaults)
        self.on_change("comparison", self.reload_arrays)

    def _register_cells(self, cells):
        options = ["all", "Placecells", "Non-Placecells", "split"]
        aliases = {"quality": "Placecells", "non-quality": "Non-Placecells"}
        self.add_selection("session_cells", value=aliases.get(cells, cells), options=options)
        value = self._initial_overview_cells
        self.add_selection("overview_cells", value=aliases.get(value, value), options=options)

    def _variance_metric_keys(self):
        return [f"pf_frac_var_{metric}_{self.state['comparison']}_slot" for metric in METRICS]

    @staticmethod
    def _groups(cells):
        cells = {"Placecells": "quality", "Non-Placecells": "non-quality"}.get(cells, cells)
        return ("quality", "non-quality") if cells == "split" else (cells,)

    def groups(self, state):
        return self._groups(state["session_cells"])

    def reload_arrays(self, state):
        super().reload_arrays(state)
        out = self.results.sel(keys=["best_env_slot"], squeeze_ones=False,
                               **data_selection(state, self.results, self.selection_names))
        if "best_env_slot" not in out:
            raise KeyError("Missing best_env_slot; rerun the pfpred_quality sweep.")
        best = np.asarray(out["best_env_slot"], dtype=float).reshape(len(self.results.sessions), -1)
        if best.shape[1] != 1:
            raise ValueError("Select a single configuration for best_env_slot.")
        self.best_env_slots = best[:, 0]

    def _overview_slots(self, row, state):
        mode = state["overview_environment"]
        if mode == "all":
            return np.flatnonzero(self.env_present[row]).tolist()
        slot = self.best_env_slots[row] if mode == "best" else state["env_slot"]
        if not np.isfinite(slot) or slot != int(slot) or not 0 <= slot < self.env_present.shape[1]:
            return []
        return [int(slot)] if self.env_present[row, int(slot)] else []

    def distributions(self, state=None):
        """Shared bin edges, selected-session densities, and mouse-by-bin overview arrays."""
        state = self.state if state is None else state
        metrics = METRICS if state["include_covariance"] else COMPONENTS
        example_row = self._session_rows[state["mouse"]][state["session"]]
        output, self.distribution_diagnostics = {}, {}
        for metric in metrics:
            def values(row, slot, group):
                data = self.arrays[metric][row, slot]
                return data[self.mask(row, slot, group, state) & np.isfinite(data)]

            session = {group: values(example_row, state["env_slot"], group)
                       for group in self._groups(state["session_cells"])}
            rows = {group: [(row, [values(row, slot, group) for slot in self._overview_slots(row, state)])
                            for row in range(len(self.results.sessions))]
                    for group in self._groups(state["overview_cells"])}
            edges = np.linspace(0, 1, state["nbins"] + 1)

            def density(data):
                return np.histogram(data, edges)[0] / len(data) / np.diff(edges) if len(data) else np.full(state["nbins"], np.nan)

            overview = {}
            for group, sessions in rows.items():
                by_row = {}
                for row, environments in sessions:
                    finite = [density(v) for v in environments if len(v)]
                    if finite:
                        by_row[row] = np.mean(finite, axis=0)
                curves = np.full((len(self.mice), state["nbins"]), np.nan)
                for index, mouse in enumerate(self.mice):
                    samples = [by_row[row] for row in self._session_rows[mouse].values() if row in by_row]
                    if samples:
                        curves[index] = np.mean(samples, axis=0)
                overview[group] = curves
            self.distribution_diagnostics[metric] = {
                "range": (edges[0], edges[-1]),
                "session": {group: dict(n=len(data), outside=int(np.sum((data < edges[0]) | (data > edges[-1]))))
                            for group, data in session.items()},
            }
            output[metric] = dict(edges=edges, session={group: density(data) for group, data in session.items()},
                                  overview=overview)
        return output

    def plot(self, state):
        distributions = self.distributions(state)
        fig, axes = self.new_subplots(len(distributions), 2, figsize=self.figsize, layout="constrained", sharex=True)
        fontsize = state["fontsize"]
        for column, (metric, data) in enumerate(distributions.items()):
            edges = data["edges"]
            centers = (edges[:-1] + edges[1:]) / 2
            for group, density in data["session"].items():
                axes[column, 0].stairs(density, edges, color=COLORS[metric], label=group,
                                       linestyle=":" if group == "non-quality" else "-", linewidth=1.2)
            for group, curves in data["overview"].items():
                if not np.isfinite(curves).any():
                    continue
                ax = axes[column, 1]
                first_line = len(ax.lines)
                render_curve_group(ax, centers, curves, COLORS[metric], state["plot_style"], label=group,
                                   hide_error=state["plot_style"] == "each", linewidth=1.4)
                for line in ax.lines[first_line:]:
                    line.set_linestyle(":" if group == "non-quality" else "-")
            for row in range(2):
                ax = axes[column, row]
                ax.set_xlim(edges[0], edges[-1])
                ax.set_ylim(bottom=0)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
                style_axis(ax, fontsize=fontsize)
                ax.tick_params(axis="x", bottom=True, labelbottom=column == len(distributions) - 1)
                ax.set_ylabel("Density", fontsize=fontsize)
            for ax in axes[column]:
                ax.set_xlabel("Variance Fraction" if column == len(distributions) - 1 else "", fontsize=fontsize)
                ax.set_xticks([0, 0.5, 1])
        axes[0, 0].set_title("Session", fontsize=fontsize)
        axes[0, 1].set_title("Overview", fontsize=fontsize)
        groups = list(dict.fromkeys([*self._groups(state["session_cells"]), *self._groups(state["overview_cells"])]))
        handles = [Line2D([], [], color="k", linestyle=":" if group == "non-quality" else "-") for group in groups]
        labels = [{"quality": "Placecells", "non-quality": "Non-Placecells", "all": "All cells"}[group] for group in groups]
        apply_legend(axes[0, 0], state, fontsize, handles=handles, labels=labels, auto_loc="upper right",
                     offset=(state["legend_x_offset"], state["legend_y_offset"]))
        handles = [Line2D([], [], color=COLORS[name]) for name in COMPONENTS]
        apply_legend(axes[0, 1], state, fontsize, handles=handles,
                     labels=["Placefield", "Gain", "Residual"], auto_loc="upper right",
                     offset=(state["legend_x_offset"], state["legend_y_offset"]))
        legend = axes[0, 1].get_legend()
        if legend is not None:
            legend.set_title("Component", prop={"size": fontsize * state["legend_fontsize_scale"]})
            legend.get_title().set_color("black")
        return fig


class PlacefieldVarianceFamiliarity(_VariancePopulationViewer):
    """Environment-specific exposure curves for one mouse and an equally weighted mouse mean.

    Rows are selected mouse / across mice; columns are PF / gain / additive / covariance.
    Solid/dashed lines distinguish quality and not-quality when ``cells='split'``. Missing
    sessions retain their exposure number. Shading is between-mouse STD, not cell uncertainty.
    """

    def __init__(self, results, *, mouse=None, cells="all", reliability_threshold=0.7,
                 fraction_active_threshold=0.2, summary_stat="median", min_mice=2,
                 fontsize=9., figsize=(12., 5.5), **selection_defaults):
        self.figsize = figsize
        self._initialize(results, mouse, None, 0, cells, reliability_threshold,
                         fraction_active_threshold, summary_stat, fontsize, selection_defaults)
        self.add_integer("min_mice", value=min(min_mice, len(self.mice)), min=1, max=len(self.mice))

    def exposure_curves(self, state=None):
        state = self.state if state is None else state
        table = self.session_statistics(state)
        curves = {}
        for group in self.groups(state):
            for metric in METRICS:
                for slot in range(self.env_present.shape[1]):
                    rows = []
                    for mouse in self.mice:
                        selected = table[(table.mouse == mouse) & (table.group == group)
                                         & (table.component == metric) & (table.env_slot == slot)]
                        rows.append(selected[state["summary_stat"]].to_numpy())
                    curves[group, metric, slot] = pad_stack(rows)
        return curves

    def plot(self, state):
        fig, axes = self.new_subplots(2, 4, figsize=self.figsize, layout="constrained", sharex="col")
        self.curves = self.exposure_curves(state)
        fontsize = state["fontsize"]
        for column, metric in enumerate(METRICS):
            for group in self.groups(state):
                for slot in range(self.env_present.shape[1]):
                    stack = self.curves[group, metric, slot]
                    if not stack.shape[1]:
                        continue
                    x = np.arange(1, stack.shape[1] + 1)
                    label = f"{ordinal(slot + 1)} env" + (f" · {group}" if state["cells"] != "all" else "")
                    kwargs = dict(color=env_slot_color(slot), linestyle="--" if group == "not quality" else "-",
                                  linewidth=1.2, marker=".", markersize=3, label=label)
                    axes[0, column].plot(x, stack[self.mice.index(state["mouse"])], **kwargs)
                    count = np.isfinite(stack).sum(axis=0)
                    mean = np.divide(np.nansum(stack, axis=0), count,
                                     out=np.full(len(x), np.nan), where=count > 0)
                    variance = np.divide(np.nansum((stack - mean)**2, axis=0), count - 1,
                                         out=np.zeros(len(x)), where=count > 1)
                    mean[count < state["min_mice"]] = np.nan
                    std = np.sqrt(variance)
                    axes[1, column].plot(x, mean, **kwargs)
                    axes[1, column].fill_between(x, mean - std, mean + std, color=env_slot_color(slot),
                                                alpha=0.1, linewidth=0)
            axes[0, column].set_title(LABELS[metric], fontsize=fontsize)
            axes[0, column].set_ylabel(f"{state['mouse']}\nROI {state['summary_stat']} ratio", fontsize=fontsize)
            axes[1, column].set_ylabel("Mouse mean ± STD", fontsize=fontsize)
            axes[1, column].set_xlabel("Environment session #", fontsize=fontsize)
            for ax in axes[:, column]:
                if metric == "covariance":
                    ax.axhline(0, color="0.7", linewidth=0.5)
                ax.set_xlim(left=0.8)
                style_axis(ax, fontsize=fontsize)
        handles, labels = axes[0, -1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=fontsize - 1)
        fig.suptitle("Variance / data variance · held-out PF templates", fontsize=fontsize)
        return fig
