"""Place-field examples and mouse-balanced population distributions and reliability relationships."""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.transforms import Bbox, IdentityTransform, ScaledTranslation
from scipy.stats import pearsonr, spearmanr

from vrAnalysis.helpers.vrsupport import _jit_reliability_loo
from vrAnalysis.processors.spkmaps import Maps

from dimensionality_manuscript.configs.pfpred_quality import (
    PF_RMS_COMPARISONS,
    TRIAL_ACTIVITY_METRICS,
    TRIAL_ACTIVITY_STATISTICS,
    _summarize_trial_activity,
    _trial_activity_measurements,
    pf_rms_map_keys,
    pf_rms_correlation_keys,
    trial_activity_summary_keys,
)
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts import session_cache
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import draw_vertical_colorscale, style_axis

_AMPLITUDE_METRIC_LABELS = {
    "trial_rms": "Trial RMS",
    "trial_mean": "Trial Mean",
    "trial_pf_dot": "PF Projection",
    "trial_gain": "Gaussian Gain",
    "trial_gain_least_squares": "Least-Squares Gain",
}
_AMPLITUDE_STATISTIC_LABELS = {"mean": "Mean", "variance": "Variance", "cv": "CV"}
_AMPLITUDE_TRANSFORMS = ("raw", "inverse", "bounded")

_POPULATION_COMPARISONS = {"shared": "all_from_all", "heldout": "cv", "independent": "independent_cv"}


def _transform_amplitude(values, transform):
    """Transform scalar or array summaries, preserving missing/undefined values."""
    values = np.asarray(values, dtype=float)
    if transform == "raw":
        return values
    transformed = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    if transform == "inverse":
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(1.0, values, out=transformed, where=finite & (values != 0))
    elif transform == "bounded":
        # Scale before squaring to avoid overflow while keeping 0 -> 1 and 1 -> .5 exact.
        with np.errstate(under="ignore"):
            scale = np.maximum(1.0, np.abs(values[finite]))
            numerator = (1.0 / scale) ** 2
            transformed[finite] = numerator / (numerator + (values[finite] / scale) ** 2)
    else:
        raise ValueError(f"Unknown amplitude transform: {transform!r}")
    return transformed


def _finite_column_summary(values):
    """Equal-mouse mean and sample STD, retaining gaps with no contributing mice."""
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=0)
    mean = np.divide(np.nansum(values, axis=0), count, out=np.full(values.shape[1], np.nan), where=count > 0)
    variance = np.divide(np.nansum((values - mean) ** 2, axis=0), count - 1, out=np.zeros_like(mean), where=count > 1)
    return mean, np.sqrt(variance), count


class _UnderlinedTextArea(TextArea):
    """Text with a matching-width underline, without requiring a TeX installation."""

    def get_bbox(self, renderer):
        bbox = super().get_bbox(renderer)
        return Bbox.from_extents(bbox.x0, bbox.y0 - renderer.points_to_pixels(2), bbox.x1, bbox.y1)

    def draw(self, renderer):
        super().draw(renderer)
        bbox = self.get_window_extent(renderer)
        y = bbox.y0 + renderer.points_to_pixels(0.75)
        line = Line2D(
            [bbox.x0, bbox.x1], [y, y], linewidth=0.6,
            color=self._text.get_color(), transform=IdentityTransform(),
        )
        line.set_figure(self.get_figure())
        line.draw(renderer)


def _add_summary_title(ax, label, value, fontsize):
    """Add an underlined label above a centered '=value'."""
    title = AnnotationBbox(
        VPacker(
            children=[
                _UnderlinedTextArea(label, textprops={"fontsize": fontsize}),
                TextArea(f"=${value:.2f}$", textprops={"fontsize": fontsize}),
            ],
            align="center", pad=0, sep=1,
        ),
        (0.5, 1), xycoords="axes fraction", xybox=(0, 3), boxcoords="offset points",
        box_alignment=(0.5, 0), frameon=False, pad=0, annotation_clip=False,
    )
    title.set_label(f"{label}\n={value:.2f}")
    ax.add_artist(title)


class _PlaceFieldPopulationViewer(FigureViewer):
    """Common stored-data selection and mouse-balanced summaries for population panels."""

    def __init__(
        self,
        results,
        *,
        comparison,
        amplitude_metric,
        amplitude_statistic,
        amplitude_transform,
        fraction_active_threshold,
        nbins,
        fontsize,
        figsize,
        selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.mice = sorted(set(str(mouse) for mouse in results.mouse_names))
        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("comparison", value=comparison, options=list(_POPULATION_COMPARISONS))
        self.add_selection("amplitude_metric", value=amplitude_metric, options=list(TRIAL_ACTIVITY_METRICS))
        self.add_selection("amplitude_statistic", value=amplitude_statistic, options=list(TRIAL_ACTIVITY_STATISTICS))
        self.add_selection("amplitude_transform", value=amplitude_transform, options=list(_AMPLITUDE_TRANSFORMS))
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0)
        self.add_integer("nbins", value=nbins, min=2, max=100)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.add_boolean("show_mouse_curves", value=True)
        self._population_cache = None

    def _population_sessions(self, state):
        """Read only summary arrays, preserving ROI alignment and using each session's best slot."""
        selection = data_selection(state, self.results, self.selection_names)
        correlation_key = f"pf_rms_r_{_POPULATION_COMPARISONS[state['comparison']]}_slot"
        amplitude_key = f"{state['amplitude_metric']}_{state['amplitude_statistic']}"
        cache_key = (tuple(selection.items()), correlation_key, amplitude_key)
        if self._population_cache is not None and self._population_cache[0] == cache_key:
            return self._population_cache[1]
        keys = [correlation_key, "reliability_slot", amplitude_key, "fraction_active_slot", "best_env_slot"]
        selected = self.results.sel(keys=keys, squeeze_ones=False, **selection)
        missing = [key for key in keys if key not in selected]
        if missing:
            raise KeyError(f"Missing {missing}; rerun the pfpred_quality schema v11 sweep.")
        n_sessions = len(self.results.mouse_names)
        best = np.asarray(selected["best_env_slot"], dtype=float).reshape(n_sessions, -1)
        if best.shape[1] != 1:
            raise ValueError("Select a single result configuration for the population viewer.")
        arrays = [np.asarray(selected[key], dtype=float) for key in keys[:-1]]
        if any(array.ndim != 3 or array.shape[0] != n_sessions for array in arrays):
            raise ValueError("Population summaries must have shape (sessions, environment slots, ROIs).")
        if len({array.shape[:2] for array in arrays}) != 1:
            raise ValueError("Population summaries must have matching session and environment axes.")
        width = min(array.shape[2] for array in arrays)
        if any(not np.isnan(array[..., width:]).all() for array in arrays):
            raise ValueError("Population summaries have mismatched ROI widths containing non-padding values.")
        sessions = []
        for row, mouse in enumerate(self.results.mouse_names):
            slot = best[row, 0]
            if not np.isfinite(slot):
                continue
            if slot != int(slot) or not 0 <= slot < arrays[0].shape[1]:
                raise ValueError(f"Invalid best environment slot {slot} in session {row}.")
            vectors = [array[row, int(slot), :width] for array in arrays]
            sessions.append((str(mouse), dict(zip(("correlation", "spatial", "amplitude", "active"), vectors))))
        self._population_cache = (cache_key, sessions)
        return sessions

    def _selected_sessions(self, state):
        sessions = []
        for mouse, data in self._population_sessions(state):
            selected = {key: values[data["active"] > state["fraction_active_threshold"]] for key, values in data.items()}
            selected["amplitude"] = _transform_amplitude(selected["amplitude"], state["amplitude_transform"])
            sessions.append((mouse, selected))
        return sessions

    @staticmethod
    def _amplitude_label(state):
        statistic = "CoV" if state["amplitude_statistic"] == "cv" else _AMPLITUDE_STATISTIC_LABELS[state["amplitude_statistic"]]
        label = f"{_AMPLITUDE_METRIC_LABELS[state['amplitude_metric']]} {statistic}"
        if state["amplitude_transform"] == "inverse":
            return f"Amp Rel.\n1 / ({label})"
        if state["amplitude_transform"] == "bounded":
            return f"Amp Rel.\n1 / (1 + ({label})²)"
        return label

    def _mouse_average(self, rows, width):
        """Average eligible sessions equally within each mouse, independently per bin."""
        result = np.full((len(self.mice), width), np.nan)
        for index, mouse in enumerate(self.mice):
            values = [values for selected_mouse, values in rows if selected_mouse == mouse]
            if values:
                result[index] = _finite_column_summary(np.asarray(values))[0]
        return result

    def _draw_mouse_curves(self, ax, centers, curves, state, *, min_mice=1):
        if state["show_mouse_curves"]:
            for values in curves:
                ax.plot(centers, values, color="0.55", alpha=0.4, linewidth=0.6)
        mean, std, count = _finite_column_summary(curves)
        mean[count < min_mice] = np.nan
        ax.fill_between(centers, mean - std, mean + std, color="k", alpha=0.12, linewidth=0)
        ax.plot(centers, mean, color="k", linewidth=1.8)

    @staticmethod
    def _style_population_axis(ax, state, *, xbounds, xlabel, ylabel, ybounds=None, xticks=None, xlabels=None):
        """Offset, bounded spines and sparse ticks with room around the data."""
        xspan = xbounds[1] - xbounds[0]
        ax.set_xlim(xbounds[0] - 0.035 * xspan, xbounds[1] + 0.035 * xspan)
        if ybounds is None:
            upper = ax.get_ylim()[1]
            ybounds = (0, float(max(0.5, np.ceil(upper * 2) / 2)))
        yspan = ybounds[1] - ybounds[0]
        ax.set_ylim(ybounds[0] - 0.025 * yspan, ybounds[1] + 0.06 * yspan)
        ax.set_xlabel(xlabel, fontsize=state["fontsize"])
        ax.set_ylabel(ylabel, fontsize=state["fontsize"])
        style_axis(
            ax,
            fontsize=state["fontsize"],
            x_pos=-0.035,
            y_pos=-0.035,
            xbounds=xbounds,
            ybounds=ybounds,
            xticks=xticks if xticks is not None else [xbounds[0], xbounds[1]],
            yticks=[ybounds[0], ybounds[1]],
            xlabels=xlabels,
        )


class PlaceFieldPopulationDistributions(_PlaceFieldPopulationViewer):
    """Mouse-balanced distributions of position Mean–STD r, spatial reliability, and amplitude.

    Reads schema v11 PFPredQuality results, using the most-sampled environment per session.
    Finite ROI values are histogrammed separately per session and normalized to density;
    sessions are averaged equally within mouse, then mice equally. Thin gray curves show
    mice; black curves and bands show the mouse mean ± sample STD (not SEM).
    ``comparison`` selects shared, heldout (mean of fold directions), or independent
    (mean of independent fold directions). ``amplitude_metric``/``amplitude_statistic``
    select the stored amplitude summary. The activity filter is strictly greater than
    ``fraction_active_threshold``. Each distribution uses its own finite denominator.
    Amplitude bounds span all finite selected values; no tails are dropped or clipped.
    ``amplitude_transform`` selects raw values, inverse (1 / value), or bounded
    (1 / (1 + value²)). Transformations precede histogramming; inverse excludes zeros,
    bounded maps zero to one, and both exclude nonfinite inputs from amplitude calculations.
    Bounded distributions always span [0, 1].
    ``nbins`` controls all three histograms; ``show_mouse_curves`` toggles thin mouse lines.
    Additional keyword arguments seed the aggregator's data-selection controls.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        comparison: str = "heldout",
        amplitude_metric: str = "trial_rms",
        amplitude_statistic: str = "cv",
        amplitude_transform: str = "raw",
        fraction_active_threshold: float = 0.1,
        nbins: int = 40,
        fontsize: float = 8.0,
        figsize: tuple[float, float] = (7.5, 2.5),
        **selection_defaults,
    ):
        super().__init__(
            results,
            comparison=comparison,
            amplitude_metric=amplitude_metric,
            amplitude_transform=amplitude_transform,
            amplitude_statistic=amplitude_statistic,
            fraction_active_threshold=fraction_active_threshold,
            nbins=nbins,
            fontsize=fontsize,
            figsize=figsize,
            selection_defaults=selection_defaults,
        )

    def distributions(self, state):
        """Return (edges, mouse-by-bin densities) for the three selected measurements."""
        sessions = self._selected_sessions(state)
        amplitude = np.concatenate([data["amplitude"] for _, data in sessions]) if sessions else np.array([])
        finite = amplitude[np.isfinite(amplitude)]
        low = min(0.0, float(np.min(finite))) if finite.size else 0.0
        high = max(low + 1.0, float(np.max(finite))) if finite.size else 1.0
        if state["amplitude_transform"] == "bounded":
            low, high = 0.0, 1.0
        bounds = {"correlation": (-1, 1), "spatial": (-1, 1), "amplitude": (low, high)}
        output = {}
        for key, (low, high) in bounds.items():
            edges = np.linspace(low, high, state["nbins"] + 1)
            rows = []
            for mouse, data in sessions:
                values = data[key][np.isfinite(data[key])]
                if values.size:
                    counts = np.histogram(values, edges)[0]
                    rows.append((mouse, counts / values.size / np.diff(edges)))
            output[key] = (edges, self._mouse_average(rows, state["nbins"]))
        return output

    def plot(self, state):
        fig, axes = self.new_subplots(1, 3, figsize=self.figsize, layout="constrained")
        labels = ["PF-Var\nAlignment", "Spatial\nReliability", "Amplitude\nReliability"]
        for ax, (key, (edges, curves)), xlabel in zip(axes, self.distributions(state).items(), labels):
            self._draw_mouse_curves(ax, (edges[:-1] + edges[1:]) / 2, curves, state)
            if key != "amplitude":
                ax.axvline(0, color="0.65", linestyle=":", linewidth=0.7, zorder=0)
            ybounds = (0, float(max(1, np.ceil(ax.get_ylim()[1]))))
            self._style_population_axis(
                ax,
                state,
                xbounds=(edges[0], edges[-1]),
                xlabel=xlabel,
                ylabel="prob. density" if ax is axes[0] else "",
                ybounds=ybounds,
                xticks=[-1, 0, 1] if key != "amplitude" else None,
            )
        return fig


class PlaceFieldReliabilityRelationships(_PlaceFieldPopulationViewer):
    """Relate position Mean–STD r to spatial and amplitude reliability, one comparison at a time.

    Uses the same most-sampled environments and activity filter as
    PlaceFieldPopulationDistributions. Each session-bin must have at least ``min_cells``
    finite neuron pairs. Session-bin means are averaged equally within mouse; the black
    curve and band show the equal-mouse mean ± sample STD where at least ``min_mice``
    contribute. Thin curves show mouse means. Amplitude bins start at zero and end at
    ``amplitude_max``, followed by an open-ended overflow bin (labeled ≥); negative
    amplitudes enter association correlations but not these nonnegative-bin curves.

    ``display`` selects a session scatter ("example"), population curves ("summary"),
    or both, in that order. ``mouse``, ``example_session``, and ``env`` select the example
    as in PlaceFieldPredictionFocus; no ROI selection is needed.
    ``include_association_strength`` adds a final row: histograms of mouse-average
    within-session correlations across neurons between position Mean–STD r
    and the selected reliability. These use all finite pairs, require ``min_cells``
    per session, and exclude constant inputs. Each eligible mouse contributes once.
    ``association_level`` selects "mouse_averages" (default) or "all_sessions" for
    the histograms only; summary curves always retain equal-mouse averaging.
    ``association_method`` selects "spearman" (default) or "pearson" for these correlations.
    In "both" display mode, only the summary row shows the reliability x-axes.
    ``row_ratio`` (1 to 5) gives each example/summary row this height relative to
    the histogram row. ``figsize`` always specifies the total figure size.
    ``nbins`` controls curve bins and association histogram bins; amplitude curves have
    one additional overflow bin. Additional keywords seed aggregator selections.
    ``amplitude_transform`` selects raw, inverse (1 / value), or bounded
    (1 / (1 + value²)) before binning and before calculating session correlations.
    Inverse excludes zeros; bounded maps zero to one. Nonfinite inputs are excluded.
    ``amplitude_max`` is in the selected units for raw/inverse. Bounded curves use
    [0, 1] with no overflow bin. Cached measurements and spatial summaries are unchanged.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        comparison: str = "heldout",
        amplitude_metric: str = "trial_rms",
        amplitude_statistic: str = "cv",
        amplitude_transform: str = "raw",
        fraction_active_threshold: float = 0.1,
        nbins: int = 20,
        amplitude_max: float = 6.0,
        min_cells: int = 20,
        min_mice: int = 3,
        include_association_strength: bool = False,
        row_ratio: float = 1.0,
        association_level: str = "mouse_averages",
        association_method: str = "spearman",
        display: str = "summary",
        mouse: str | None = None,
        example_session: str | None = None,
        env: int = 0,
        fontsize: float = 8.0,
        figsize: tuple[float, float] = (5.2, 2.5),
        **selection_defaults,
    ):
        super().__init__(
            results,
            comparison=comparison,
            amplitude_metric=amplitude_metric,
            amplitude_transform=amplitude_transform,
            amplitude_statistic=amplitude_statistic,
            fraction_active_threshold=fraction_active_threshold,
            nbins=nbins,
            fontsize=fontsize,
            figsize=figsize,
            selection_defaults=selection_defaults,
        )
        self.add_float("amplitude_max", value=amplitude_max, min=0.1, max=1000.0)
        self.add_integer("min_cells", value=min_cells, min=2, max=1000)
        self.add_integer("min_mice", value=min_mice, min=1, max=max(min_mice, len(self.mice)))
        self.add_boolean("include_association_strength", value=include_association_strength)
        self.add_float("row_ratio", value=row_ratio, min=1.0, max=5.0)
        self.add_selection("association_level", value=association_level, options=["all_sessions", "mouse_averages"])
        self.add_selection("association_method", value=association_method, options=["spearman", "pearson"])
        self.add_selection("display", value=display, options=["example", "summary", "both"])
        self._sessions_by_mouse = {
            name: {
                results.sessions[row].session_print(): row
                for row in sorted(np.flatnonzero(np.asarray(results.mouse_names) == name), key=lambda row: _session_sort_key(results.sessions[row]))
            }
            for name in self.mice
        }
        initial_mouse = mouse if mouse is not None else self.mice[0]
        sessions = list(self._sessions_by_mouse[initial_mouse])
        self.add_selection("mouse", value=initial_mouse, options=self.mice)
        self.add_selection("example_session", value=example_session if example_session is not None else sessions[0], options=sessions)
        self.add_integer("env", value=env, min=0, max=max(env, 0))
        self._example_cache = None
        self.on_change("mouse", self.update_example_session)
        self.on_change(["example_session", "display"], self.update_example_bounds)
        self.update_example_bounds(self.state)

    def update_example_session(self, state):
        """Offer the selected mouse's sessions in chronological order."""
        sessions = list(self._sessions_by_mouse[state["mouse"]])
        current = state["example_session"] if state["example_session"] in sessions else sessions[0]
        self.update_selection("example_session", value=current, options=sessions)
        self.update_example_bounds({**state, "example_session": current})

    def _example_environments(self, state):
        """Use the same environment indexing as PlaceFieldPredictionFocus."""
        row = self._sessions_by_mouse[state["mouse"]][state["example_session"]]
        # get_env_maps uses this same list; no activity maps are needed here.
        return self.results.sessions[row].environments

    def update_example_bounds(self, state):
        if state["display"] != "summary":
            self.update_integer("env", min=0, max=max(len(self._example_environments(state)) - 1, 0))

    def example_data(self, state):
        """Filtered ROI vectors for the exact selected session and environment."""
        selection = data_selection(state, self.results, self.selection_names)
        correlation_key = f"pf_rms_r_{_POPULATION_COMPARISONS[state['comparison']]}_slot"
        amplitude_key = f"{state['amplitude_metric']}_{state['amplitude_statistic']}"
        row = self._sessions_by_mouse[state["mouse"]][state["example_session"]]
        session = self.results.sessions[row]
        schema_version = getattr(self.results, "_schema_version", None)
        if schema_version is not None:
            selection["schema_version"] = schema_version
        config = self.results.config_class(**selection)
        cache_key = (session.session_uid, config.key())
        if self._example_cache is None or self._example_cache[0] != cache_key:
            selected = self.results.store.get(session.session_uid, config)
            if selected is None:
                raise KeyError(f"No pfpred_quality result for {session.session_print()}; rerun the schema v11 sweep.")
            self._example_cache = (cache_key, selected)
        selected = self._example_cache[1]
        keys = [correlation_key, "reliability_slot", amplitude_key, "fraction_active_slot", "env_slot_ids"]
        missing = [key for key in keys if key not in selected]
        if missing:
            raise KeyError(f"Missing {missing}; rerun the pfpred_quality schema v11 sweep.")
        environment = self._example_environments(state)[state["env"]]
        slots = np.flatnonzero(np.asarray(selected["env_slot_ids"]) == environment)
        if slots.size != 1:
            raise ValueError(f"Environment {environment} has no unique stored slot for {state['example_session']}.")
        vectors = [np.asarray(selected[key], dtype=float)[slots[0]] for key in keys[:-1]]
        width = min(vector.size for vector in vectors)
        if any(not np.isnan(vector[width:]).all() for vector in vectors):
            raise ValueError("Example summaries have mismatched ROI widths containing non-padding values.")
        active = vectors[-1][:width] > state["fraction_active_threshold"]
        data = {name: vector[:width][active] for name, vector in zip(("correlation", "spatial", "amplitude", "active"), vectors)}
        data["amplitude"] = _transform_amplitude(data["amplitude"], state["amplitude_transform"])
        return data

    def relationships(self, state):
        """Return predictor edges, mouse-bin means, and selected session/mouse correlations."""
        sessions = self._selected_sessions(state)
        correlation = spearmanr if state["association_method"] == "spearman" else pearsonr
        output = {}
        for predictor in ("spatial", "amplitude"):
            edges = (
                np.linspace(-1, 1, state["nbins"] + 1)
                if predictor == "spatial"
                else (
                    np.linspace(0, 1, state["nbins"] + 1)
                    if state["amplitude_transform"] == "bounded"
                    else np.r_[np.linspace(0, state["amplitude_max"], state["nbins"] + 1), np.inf]
                )
            )
            rows, associations = [], []
            for mouse, data in sessions:
                x, y = data[predictor], data["correlation"]
                finite = np.isfinite(x) & np.isfinite(y)
                x, y = x[finite], y[finite]
                means = np.full(len(edges) - 1, np.nan)
                for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                    inside = (x >= low) & ((x < high) | ((index == len(means) - 1) & (x == high)))
                    if np.sum(inside) >= state["min_cells"]:
                        means[index] = np.mean(y[inside])
                rows.append((mouse, means))
                if x.size >= state["min_cells"] and np.ptp(x) > 0 and np.ptp(y) > 0:
                    associations.append((mouse, np.array([correlation(x, y).statistic])))
            output[predictor] = (
                edges,
                self._mouse_average(rows, len(edges) - 1),
                (
                    np.array([value[0] for _, value in associations])
                    if state["association_level"] == "all_sessions"
                    else self._mouse_average(associations, 1)[:, 0]
                ),
            )
        return output

    def plot(self, state):
        displays = ["example", "summary"] if state["display"] == "both" else [state["display"]]
        histogram = state["include_association_strength"]
        heights = [state["row_ratio"]] * len(displays) + ([1] if histogram else [])
        fig, axes = self.new_subplots(
            len(heights),
            2,
            figsize=self.figsize,
            squeeze=False,
            layout="constrained",
            height_ratios=heights,
        )
        example = self.example_data(state) if "example" in displays else None
        relationships = self.relationships(state) if "summary" in displays or histogram else {}
        labels = {"spatial": "Spatial Reliability", "amplitude": "Amplitude Reliability"}
        for column, predictor in enumerate(labels):
            bounded = predictor == "amplitude" and state["amplitude_transform"] == "bounded"
            for row, display in enumerate(displays):
                ax = axes[row, column]
                if display == "summary":
                    edges, curves, _ = relationships[predictor]
                    display_edges = edges.copy()
                    if predictor == "amplitude" and not bounded:
                        display_edges[-1] = edges[-2] + edges[-2] / state["nbins"]
                    centers = (display_edges[:-1] + display_edges[1:]) / 2
                    self._draw_mouse_curves(ax, centers, curves, state, min_mice=state["min_mice"])
                    xbounds = (display_edges[0], display_edges[-1])
                    ticks = [-1, 0, 1] if predictor == "spatial" else [0, 1] if bounded else [0, state["amplitude_max"] / 2, centers[-1]]
                    ticklabels = None if predictor == "spatial" or bounded else ["0", f"{ticks[1]:g}", f"≥{state['amplitude_max']:g}"]
                else:
                    x, y = example[predictor], example["correlation"]
                    finite = np.isfinite(x) & np.isfinite(y)
                    ax.scatter(x[finite], y[finite], s=5, color="k", alpha=0.05, linewidths=0)
                    if predictor == "spatial":
                        xbounds, ticks = (-1, 1), [-1, 0, 1]
                    elif bounded:
                        xbounds, ticks = (0, 1), [0, 1]
                    else:
                        values = x[finite]
                        xbounds = (min(0.0, float(values.min())) if values.size else 0.0, max(1.0, float(values.max())) if values.size else 1.0)
                        ticks = list(xbounds)
                    ticklabels = None
                ax.axhline(0, color="0.65", linestyle=":", linewidth=0.7, zorder=0)
                self._style_population_axis(
                    ax, state, xbounds=xbounds, xlabel=labels[predictor], ylabel="", ybounds=(-1, 1), xticks=ticks, xlabels=ticklabels
                )
                if state["display"] == "both" and row == 0:
                    ax.spines["bottom"].set_visible(False)
                    ax.xaxis.set_visible(False)
                if column == 0:
                    ax.set_ylabel("PF-Var Alignment", fontsize=state["fontsize"])
            if histogram:
                ax = axes[-1, column]
                associations = relationships[predictor][2]
                finite = associations[np.isfinite(associations)]
                ax.hist(finite, bins=state["nbins"], range=(-1, 1), color="0.75", edgecolor="none")
                ax.axvline(0, color="0.65", linestyle=":", linewidth=0.7, zorder=0)
                self._style_population_axis(
                    ax,
                    state,
                    xbounds=(-1, 1),
                    xticks=[-1, 0, 1],
                    xlabel="Spearman ρ" if state["association_method"] == "spearman" else "Pearson r",
                    ylabel=("Sessions" if state["association_level"] == "all_sessions" else "Mice") if column == 0 else "",
                    ybounds=(0, max(1, float(np.ceil(ax.get_ylim()[1])))),
                )
        return fig


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


class SpatialAmplitudeReliability(FigureViewer):
    """Compare spatial reliability with a trial-amplitude summary for every ROI in one session.

    All plotted values are read from the per-environment-slot arrays already emitted by
    :class:`~dimensionality_manuscript.configs.pfpred_quality.PFPredQualityConfig`; changing
    sessions or metrics does not recompute maps or Gaussian fits.

    Parameters
    ----------
    results : ResultsAggregator
        Results whose sessions are available to the mouse/session selectors.
    mouse : str or None
        Initial mouse. None takes the first alphabetically.
    example_session : str or None
        Initial session label, in ``"mouse/date/id"`` form. None takes the mouse's first.
    env : int
        Index into ``env_maps.environments``.
    amplitude_metric : {"trial_rms", "trial_mean", "trial_pf_dot", "trial_gain", "trial_gain_least_squares"}
        Trial-wise activity-amplitude measurement.
    amplitude_statistic : {"mean", "variance", "cv"}
        Statistic reducing that measurement across trials for each ROI.
    fraction_active_threshold : float
        Include only ROIs whose fraction active is strictly above this threshold in the
        selected environment.
    markersize, alpha, fontsize : float
        Scatter and text styling.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the aggregator's own parameter axes, by name.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        example_session: str | None = None,
        env: int = 0,
        amplitude_metric: str = "trial_rms",
        amplitude_statistic: str = "cv",
        fraction_active_threshold: float = 0.1,
        markersize: float = 8.0,
        alpha: float = 0.5,
        fontsize: float = 12.0,
        figsize: tuple[float, float] = (5.0, 4.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize

        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(mouse_name) for mouse_name in mouse_names})
        rows_by_mouse = {mouse_name: np.flatnonzero(mouse_names == mouse_name) for mouse_name in self.mice}
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
        self._local_rows_by_mouse = {
            mouse_name: {results.sessions[row].session_print(): local_row for local_row, row in enumerate(rows_by_mouse[mouse_name])}
            for mouse_name in self.mice
        }
        initial_mouse = mouse if mouse is not None else self.mice[0]
        initial_sessions = list(self._sessions_by_mouse[initial_mouse])
        initial_session = example_session if example_session is not None else initial_sessions[0]

        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("mouse", value=initial_mouse, options=self.mice)
        self.add_selection("example_session", value=initial_session, options=initial_sessions)
        self.add_integer("env", value=env, min=0, max=max(env, 0))
        self.add_selection("amplitude_metric", value=amplitude_metric, options=list(TRIAL_ACTIVITY_METRICS))
        self.add_selection(
            "amplitude_statistic",
            value=amplitude_statistic,
            options=list(TRIAL_ACTIVITY_STATISTICS),
        )
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0)
        self.add_float("markersize", value=markersize, min=1.0, max=50.0)
        self.add_float("alpha", value=alpha, min=0.05, max=1.0)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)

        self.on_change(list(self.selection_names), self.reload_arrays)
        self.on_change("mouse", self.update_mouse)
        self.on_change("example_session", self.update_example_bounds)
        self.on_change("env", self.refresh_data)
        self.on_change(["amplitude_metric", "amplitude_statistic"], self.reload_arrays)
        self._load_arrays(self.state)
        self.update_example_session(self.state)

    def _load_arrays(self, state):
        """Load the selected mouse and amplitude combination from the result aggregator."""
        amplitude_key = f"{state['amplitude_metric']}_{state['amplitude_statistic']}"
        keys = [amplitude_key, "reliability_slot", "fraction_active_slot", "env_slot_ids"]
        out = self.results.sel(
            mouse=state["mouse"],
            keys=keys,
            squeeze_ones=False,
            **data_selection(state, self.results, self.selection_names),
        )
        missing = [key for key in keys if key not in out]
        if missing:
            raise KeyError(f"Aggregator is missing {missing} -- rerun the pfpred_quality sweep (schema v9 required).")
        self.amplitude_key = amplitude_key
        self.amplitude_slot = np.asarray(out[amplitude_key], dtype=float)
        self.reliability_slot = np.asarray(out["reliability_slot"], dtype=float)
        self.fraction_active_slot = np.asarray(out["fraction_active_slot"], dtype=float)
        self.env_slot_ids = np.asarray(out["env_slot_ids"], dtype=float)
        slot_arrays = (self.amplitude_slot, self.reliability_slot, self.fraction_active_slot)
        if all(array.ndim == 3 for array in slot_arrays) and len({array.shape[:2] for array in slot_arrays}) == 1:
            # Lazy aggregation retains each key's largest previously loaded ROI width.
            # Remove only surplus NaN padding; keep internal NaNs and ROI positions.
            n_rois = min(array.shape[2] for array in slot_arrays)
            if all(np.isnan(array[..., n_rois:]).all() for array in slot_arrays):
                self.amplitude_slot, self.reliability_slot, self.fraction_active_slot = (array[..., :n_rois] for array in slot_arrays)
        if not (self.amplitude_slot.shape == self.reliability_slot.shape == self.fraction_active_slot.shape):
            raise ValueError("Amplitude, reliability, and fraction-active slot arrays must have matching shapes.")

    def reload_arrays(self, state):
        """Reload stored arrays after a result-parameter or amplitude-key change."""
        self._load_arrays(state)
        self.update_example_bounds(state)

    def update_mouse(self, state):
        """Load the selected mouse's arrays and update its session selector."""
        self._load_arrays(state)
        self.update_example_session(state)

    def update_example_session(self, state):
        """Offer the selected mouse's sessions in chronological order."""
        labels = list(self._sessions_by_mouse[state["mouse"]])
        current = state["example_session"] if state["example_session"] in labels else labels[0]
        self.update_selection("example_session", value=current, options=labels)
        self.update_example_bounds({**state, "example_session": current})

    def update_example_bounds(self, state):
        """Offer only environments with stored values for the selected session."""
        row = self._local_rows_by_mouse[state["mouse"]][state["example_session"]]
        self.slots = [
            slot
            for slot in range(self.amplitude_slot.shape[1])
            if np.isfinite(self.env_slot_ids[row, slot])
            and np.any(
                np.isfinite(self.amplitude_slot[row, slot])
                & np.isfinite(self.reliability_slot[row, slot])
                & np.isfinite(self.fraction_active_slot[row, slot])
            )
        ]
        if not self.slots:
            raise ValueError("The selected session has no matched amplitude, reliability, and fraction-active values.")
        self.update_integer("env", min=0, max=len(self.slots) - 1)
        selected_env = min(state["env"], len(self.slots) - 1)
        self.refresh_data({**state, "env": selected_env})

    def refresh_data(self, state):
        """Select one stored session/environment vector without recomputing session data."""
        row = self._local_rows_by_mouse[state["mouse"]][state["example_session"]]
        slot = self.slots[state["env"]]
        self.environment = int(self.env_slot_ids[row, slot])
        self.spatial_reliability = self.reliability_slot[row, slot]
        self.fraction_active = self.fraction_active_slot[row, slot]
        self.amplitude_summaries = {self.amplitude_key: self.amplitude_slot[row, slot]}

    def plot(self, state):
        metric = state["amplitude_metric"]
        statistic = state["amplitude_statistic"]
        amplitude = self.amplitude_summaries[f"{metric}_{statistic}"]
        included = np.isfinite(self.spatial_reliability) & np.isfinite(amplitude) & np.isfinite(self.fraction_active)
        included &= self.fraction_active > state["fraction_active_threshold"]

        fig, ax = self.new_subplots(figsize=self.figsize, layout="constrained")
        ax.scatter(
            self.spatial_reliability[included],
            amplitude[included],
            color="k",
            s=state["markersize"],
            alpha=state["alpha"],
            linewidths=0,
        )
        ax.set_xlabel("Spatial Reliability", fontsize=state["fontsize"])
        ax.set_ylabel(
            f"Amplitude Reliability\n{_AMPLITUDE_METRIC_LABELS[metric]} {_AMPLITUDE_STATISTIC_LABELS[statistic]}",
            fontsize=state["fontsize"],
        )
        ax.set_title(
            f"{state['example_session']} | environment {self.environment} | n={np.sum(included)}",
            fontsize=state["fontsize"],
        )
        style_axis(ax, fontsize=state["fontsize"], spines_visible=["bottom", "left"])
        return fig


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
    """Position mean/STD, spatial reliability, and amplitude reliability for one ROI/environment.

    Up to four columns. The map columns share the VR-position axis; trial-wise panels share
    the trial axis:

    1. Trial-by-position spike map with the selected PF and RMS error placefield below.
    2. Per-trial consistency with the other trials, with this cell's correlation histogram below.
       Dropped when ``show_consistency`` is False.
    3. A selectable trial-wise activity-amplitude measurement, with its histogram and CoV
       below. Dropped when ``include_amplitude_summary`` is False.
    4. The place-field prediction of each trial, with its trial average below. Dropped when
       ``show_prediction`` is False.
    Turning the optional columns off leaves only the activity map and position curves.

    The place-field prediction of a trial is the place field itself: ``get_placefield_prediction``
    predicts each frame from the trial-averaged map, so in position units every trial's prediction
    is the same row. The optional prediction column shows that all-trial template.
    PF/RMS maps and Pearson r are read from schema v11 quality results. Folds alternate trials
    chronologically within each environment. ``f2_from_f1`` means RMS on f2 relative to PF f1;
    ``independent_f1`` compares PF f1 with RMS on f2 relative to PF f2 (and vice versa).

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
        Upper limit (in normalized fluorescence) of the activity and prediction colorscale.
    pf_rms_comparison : str
        Which all-trial, within-fold, held-out, or independent-fold PF/RMS pairing to show.
    show_correlation : bool
        Show PF-Var Alignment (position mean/STD Pearson r) above the position curves.
    fontsize : float
        Single font size for the whole panel: axis labels, tick labels, in-axes annotations, and
        colorscale labels.
    show_consistency : bool
        Draw the trial-consistency / reliability column (column 2).
    show_prediction : bool
        Draw the place-field prediction column.
    include_amplitude_summary : bool
        Draw the trial-amplitude column immediately after spatial reliability.
    amplitude_metric : {"trial_rms", "trial_mean", "trial_pf_dot", "trial_gain", "trial_gain_least_squares"}
        Trial-wise measurement drawn in the top row.
    amplitude_statistic : str
        Accepted for compatibility with existing notebooks; the summary uses CoV before transformation.
    amplitude_transform : {"raw", "inverse", "bounded"}
        Transform the CoV summary: raw shows CoV, inverse (1 / CoV) and bounded
        (1 / (1 + CoV²)) show AmpRel. Trial amplitudes and their histogram retain their units.
    nbins : int
        Number of bins in the example cell's trial-correlation and amplitude histograms.
    row_ratio : float
        Top-row height divided by bottom-row height; larger values give the top row more room.
    use_trial_weights : bool
        Weight the trial-consistency scatter by each trial's RMS activity, so silent trials
        don't show up in the plot.
    figsize : tuple[float, float]
        Figure size in inches. Not rescaled when columns are dropped, so the remaining columns get
        wider; pass a narrower width for the two-column version.
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
        pf_rms_comparison: str = "f2_from_f1",
        show_correlation: bool = False,
        fontsize: float = 12.0,
        show_consistency: bool = True,
        show_prediction: bool = True,
        include_amplitude_summary: bool = False,
        amplitude_metric: str = "trial_rms",
        amplitude_statistic: str = "cv",
        amplitude_transform: str = "raw",
        nbins: int = 10,
        row_ratio: float = 2.0,
        use_trial_weights: bool = True,
        figsize: tuple[float, float] = (12.0, 6.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self._example_config = results.config_class()
        self._quality_result_cache = None

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
        self.add_selection("pf_rms_comparison", value=pf_rms_comparison, options=list(PF_RMS_COMPARISONS))
        self.add_boolean("show_correlation", value=show_correlation)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.add_boolean("show_consistency", value=show_consistency)
        self.add_boolean("show_prediction", value=show_prediction)
        self.add_boolean("include_amplitude_summary", value=include_amplitude_summary)
        self.add_selection("amplitude_metric", value=amplitude_metric, options=list(TRIAL_ACTIVITY_METRICS))
        # Retain amplitude_statistic for existing notebooks; the summary is derived from CoV.
        self.add_selection("amplitude_transform", value=amplitude_transform, options=list(_AMPLITUDE_TRANSFORMS))
        self.add_integer("nbins", value=nbins, min=1, max=100)
        self.add_float("row_ratio", value=row_ratio, min=0.25, max=10.0)
        self.add_boolean("use_trial_weights", value=use_trial_weights)

        self.on_change("mouse", self.update_example_session)
        self.on_change("example_session", self.update_example_bounds)
        self.on_change(["roi", "env"], self.refresh_data)
        self.on_change(list(self.selection_names), self.refresh_data)
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
        config = self.results.config_class(**data_selection(state, self.results, self.selection_names))
        session.params.spks_type = config.spks_type
        try:
            return session_cache.get_env_maps(session)
        finally:
            session.params.spks_type = previous_spks_type

    def refresh_data(self, state):
        """Select one ROI's spike map and derive the place field, its prediction, and the error."""
        env_maps = self._get_env_maps(state)
        self.distcenters = env_maps.distcenters
        session_spkmap = env_maps.spkmap[state["env"]]
        self.spkmap = session_spkmap[state["roi"]]
        self.placefield = np.nanmean(self.spkmap, axis=0)

        # Every trial is predicted by the same place field; mask where the trial has no data.
        self.pred_spkmap = np.broadcast_to(self.placefield, self.spkmap.shape).copy()
        self.pred_spkmap[np.isnan(self.spkmap)] = np.nan
        self.avg_prediction = np.nanmean(self.pred_spkmap, axis=0)
        measured = _trial_activity_measurements(self.spkmap[None, ...], self.distcenters)
        self.trial_activity = {key: values[0] for key, values in measured.items()}

        spatial_keys = pf_rms_map_keys() + pf_rms_correlation_keys()
        keys = ["env_slot_ids", "reliability_slot", *trial_activity_summary_keys(), "pf_rms_distcenters", *[f"{key}_slot" for key in spatial_keys]]
        session = self._sessions_by_mouse[state["mouse"]][state["example_session"]]
        params = data_selection(state, self.results, self.selection_names)
        schema_version = getattr(self.results, "_schema_version", None)
        if schema_version is not None:
            params["schema_version"] = schema_version
        config = self.results.config_class(**params)
        cache_key = (session.session_uid, config.key())
        # These maps are large: retain only this session instead of padding every session's
        # ROI-by-position arrays through the aggregator.
        if self._quality_result_cache is None or self._quality_result_cache[0] != cache_key:
            selected = self.results.store.get(session.session_uid, config)
            if selected is None:
                raise KeyError(f"No pfpred_quality result for {session.session_print()} -- rerun the schema v11 sweep.")
            self._quality_result_cache = (cache_key, selected)
        selected = self._quality_result_cache[1]
        missing = [key for key in keys if key not in selected]
        if missing:
            raise KeyError(f"Missing {missing} -- rerun the pfpred_quality sweep (schema v11 required).")
        env_slot_ids = np.asarray(selected["env_slot_ids"], dtype=float)
        environment = int(env_maps.environments[state["env"]])
        matching_slots = np.flatnonzero(env_slot_ids == environment)
        if matching_slots.size == 0:
            raise ValueError(f"Environment {environment} is absent from the precomputed pfpred_quality results " f"for {state['example_session']}.")
        slot = int(matching_slots[0])
        self.pf_rms_distcenters = np.asarray(selected["pf_rms_distcenters"], dtype=float)
        self.session_pf_rms = {key: np.asarray(selected[f"{key}_slot"][slot], dtype=float) for key in spatial_keys}
        self.session_activity_summary = {key: np.asarray(selected[key][slot], dtype=float) for key in trial_activity_summary_keys()}
        self.trial_activity_summary = {key: values[state["roi"]] for key, values in self.session_activity_summary.items()}
        self.session_reliability = np.asarray(selected["reliability_slot"][slot], dtype=float)
        trial_numbers, trial_weights, trial_consistency, _ = _trial_consistency(self.spkmap)
        self.consistency = (
            trial_numbers,
            trial_weights,
            trial_consistency,
            float(self.session_reliability[state["roi"]]),
        )

    def plot(self, state):
        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        return self.draw(state, fig)

    def draw(self, state, fig, subplot_spec=None):
        """Draw into a figure or a nested subplot, for tracked-session comparisons."""
        first_axis = len(fig.axes)
        vmax = state["vmax"]
        fontsize = state["fontsize"]
        show_consistency = state["show_consistency"]
        show_prediction = state["show_prediction"]
        include_amplitude_summary = state["include_amplitude_summary"]
        comparison = state["pf_rms_comparison"]
        fold, rms_name = PF_RMS_COMPARISONS[comparison]
        pf = self.session_pf_rms[f"pf_{fold}"][state["roi"]]
        rms = self.session_pf_rms[f"rms_{rms_name}"][state["roi"]]
        r = self.session_pf_rms[f"pf_rms_r_{comparison}"][state["roi"]]

        distcenters = self.distcenters
        spkmap = self.spkmap

        xlims = [distcenters[0], distcenters[-1]]
        ylims = [spkmap.shape[0] + 0.5, -0.5]
        extent = (xlims[0], xlims[1], spkmap.shape[0], 0)
        xlims_clean = (np.round(xlims[0] / 50) * 50, np.round(xlims[1] / 50) * 50)
        xlabels = [f"{int(round(x))}" for x in xlims_clean]
        # Keep at least 25% headroom and a whole-number upper tick.
        finite_amplitudes = np.concatenate([pf, rms])
        finite_amplitudes = finite_amplitudes[np.isfinite(finite_amplitudes)]
        curve_max = np.max(finite_amplitudes) if finite_amplitudes.size else 0.0
        ymax_pf = float(max(1, np.ceil(curve_max * 1.25)))

        # Columns are added left to right, so the optional ones only shift what follows them.
        width_ratios = [1]
        icol_consistency = None
        icol_amplitude = None
        icol_prediction = None
        if show_consistency:
            icol_consistency = len(width_ratios)
            width_ratios.append(0.5)
        if include_amplitude_summary:
            icol_amplitude = len(width_ratios)
            width_ratios.append(0.5)
        if show_prediction:
            icol_prediction = len(width_ratios)
            width_ratios.append(1)
        grid = fig.add_gridspec if subplot_spec is None else subplot_spec.subgridspec
        gs = grid(2, len(width_ratios), width_ratios=width_ratios, height_ratios=[state["row_ratio"], 1])
        ax_spkmap = fig.add_subplot(gs[0, 0])
        ax_placefield = fig.add_subplot(gs[1, 0])

        def draw_map(ax, values, cmap, vlow, vhigh, spines_visible):
            """One trial-by-position image, sharing the trial axis and the cleaned position axis."""
            ax.imshow(values, interpolation="none", aspect="auto", cmap=cmap, vmin=vlow, vmax=vhigh, extent=extent)
            ax.set_xlim(xlims_clean)
            ax.set_ylim(ylims[0], ylims[1])
            style_axis(ax, fontsize=fontsize, xbounds=xlims_clean, xticks=[], yticks=[], spines_visible=spines_visible)

        def draw_curve(ax, values, label, ylabel=None, include_left_spine=False, positions=None, ymax=ymax_pf):
            """One position-axis curve below a map, labeled in-axes."""
            ax.plot(distcenters if positions is None else positions, values, color="k", linewidth=1.5)
            ax.set_facecolor(("black", 0.04))
            ax.set_xlabel("Position (cm)", labelpad=-10, fontsize=fontsize)
            ax.set_xlim(xlims_clean)
            ax.set_ylim(0, ymax)
            if label:
                ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top", color="k", fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=-2)

            if include_left_spine:
                spines_visible = ["bottom", "left"]
                yticks = [0, ymax]
                ybounds = (0, ymax)
            else:
                spines_visible = ["bottom"]
                yticks = []
                ybounds = (0, ymax)

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
        ax_spkmap.set_title("Placefield\nStructure", ha="center", fontsize=fontsize)
        ax_spkmap.set_ylabel("Trials", fontsize=fontsize)
        ax_spkmap.set_yticks([0, spkmap.shape[0]])
        ax_spkmap.spines["left"].set_bounds(0, spkmap.shape[0])
        draw_curve(ax_placefield, pf, None, "N.Fluor", include_left_spine=True, positions=self.pf_rms_distcenters)
        ax_placefield.plot(self.pf_rms_distcenters, rms, color="tab:red", linewidth=1.5)
        label = ax_placefield.text(0.02, 0.98, "Pos Mean", transform=ax_placefield.transAxes, ha="left", va="top", color="k", fontsize=fontsize)
        label = ax_placefield.annotate(
            "Pos STD", xy=(0, 0), xycoords=label, xytext=(0, -2), textcoords="offset points", ha="left", va="top", color="tab:red", fontsize=fontsize
        )
        if state["show_correlation"]:
            _add_summary_title(ax_placefield, "PF-Var Alignment", r, fontsize)

        # ---------------------------------------------------------- col 2: consistency --
        if show_consistency:
            self._draw_consistency(fig, gs, icol_consistency, state, ylims, fontsize)

        # ------------------------------------------------------------ col 3: amplitude --
        if include_amplitude_summary:
            self._draw_amplitude_summary(fig, gs, icol_amplitude, state, ylims, fontsize)

        # ----------------------------------------------------------- col 4: prediction --
        if show_prediction:
            draw_map(fig.add_subplot(gs[0, icol_prediction]), self.pred_spkmap, "gray_r", 0, vmax, [])
            finite_prediction = self.avg_prediction[np.isfinite(self.avg_prediction)]
            ymax_prediction = max(ymax_pf, float(np.max(finite_prediction)) * 1.25) if finite_prediction.size else ymax_pf
            draw_curve(fig.add_subplot(gs[1, icol_prediction]), self.avg_prediction, "PF Prediction", ymax=ymax_prediction)

        # --------------------------------------------------------- inset colorscales --
        draw_vertical_colorscale(
            ax_spkmap.inset_axes([0.225, 0.15, 0.125, 0.7]),
            "gray_r",
            low_label="0",
            high_label=f"{int(vmax)}",
            fontsize=fontsize,
            ylabel="Norm. Fluor.",
        )
        # Use one spine position and a fixed point offset for top-aligned x-labels,
        # independent of each panel's data range or number of label lines.
        for ax in fig.axes[first_axis:]:
            if ax.get_subplotspec().rowspan.start == 1:
                ax.spines["bottom"].set_position(("axes", -0.02))
                ax.xaxis.set_label_coords(
                    0.5,
                    -0.02,
                    transform=ax.transAxes + ScaledTranslation(0, -(fontsize + 8) / 72, fig.dpi_scale_trans),
                )
                ax.xaxis.label.set_verticalalignment("top")
        # Keep both left labels close to the spine, without moving them out for
        # the wider trial-count tick labels at the ends of the upper panel.
        for ax in (ax_spkmap, ax_placefield):
            ax.yaxis.set_label_coords(
                0,
                0.5,
                transform=ax.transAxes + ScaledTranslation(-(fontsize + 7) / 72, 0, fig.dpi_scale_trans),
            )
            ax.yaxis.label.set_verticalalignment("center")
        return fig

    @staticmethod
    def _zero_to_ceiling_bounds(values):
        """Bounds from zero through the integer ceiling of the finite natural maximum."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return (0.0, 1.0)
        return (0.0, float(max(1, np.ceil(np.max(finite)))))

    def _draw_amplitude_summary(self, fig, gs, icol, state, ylims, fontsize):
        """Selected per-trial amplitude, its histogram, and optionally transformed CoV."""
        metric = state["amplitude_metric"]
        trial_values = self.trial_activity[metric]
        summary_value = float(_transform_amplitude(self.trial_activity_summary[f"{metric}_cv"], state["amplitude_transform"]))
        summary_label = "CoV" if state["amplitude_transform"] == "raw" else "AmpRel"
        trial_numbers = np.arange(trial_values.size)

        ax_trials = fig.add_subplot(gs[0, icol])
        ax_trials.set_title("Amplitude\nReliability", ha="center", rotation=0, fontsize=fontsize)
        finite = np.isfinite(trial_values)
        ax_trials.scatter(trial_values[finite], trial_numbers[finite], color="k", s=1.75)
        ax_trials.set_facecolor(("black", 0.04))
        trial_bounds = self._zero_to_ceiling_bounds(trial_values)
        ax_trials.set_xlim(trial_bounds)
        ax_trials.set_ylim(ylims[0], ylims[1])
        style_axis(
            ax_trials,
            fontsize=fontsize,
            xbounds=trial_bounds,
            xticks=[],
            yticks=[],
            spines_visible=[],
        )

        ax_summary = fig.add_subplot(gs[1, icol])
        metric_label = _AMPLITUDE_METRIC_LABELS[metric].removeprefix("Trial ")
        ax_summary.set_xlabel(f"Trial\n{metric_label}", fontsize=fontsize)
        self._draw_trial_histogram(
            ax_summary,
            trial_values,
            state["nbins"],
            trial_bounds,
            summary_label,
            summary_value,
            fontsize,
        )

    def _draw_consistency(self, fig, gs, icol, state, ylims, fontsize):
        """Column 2: per-trial consistency and this cell's distribution of correlations."""
        ax_consistency = fig.add_subplot(gs[0, icol])
        ax_consistency.set_title("Spatial\nReliability", ha="center", rotation=0, fontsize=fontsize)
        ax_reliability = fig.add_subplot(gs[1, icol])
        trial_numbers, trial_weights, trial_consistency, reliability = self.consistency

        alpha = trial_weights if state["use_trial_weights"] else 1.0
        ax_consistency.scatter(trial_consistency, trial_numbers, color="k", s=1.75, alpha=alpha)
        ax_consistency.set_facecolor(("black", 0.04))
        ax_consistency.set_xlim(-1.1, 1.1)
        ax_consistency.set_ylim(ylims[0], ylims[1])
        style_axis(ax_consistency, fontsize=fontsize, xbounds=(-1, 1), xticks=[], yticks=[], spines_visible=[])

        ax_reliability.set_xlabel("Trial\nSimilarity", fontsize=fontsize)
        self._draw_trial_histogram(
            ax_reliability,
            trial_consistency,
            state["nbins"],
            (-1, 1),
            "SpRel",
            reliability,
            fontsize,
            xlim=(-1.1, 1.1),
        )

    @staticmethod
    def _draw_trial_histogram(ax, values, nbins, bounds, summary_label, summary_value, fontsize, *, xlim=None):
        """Finite trial values with an underlined label above a centered '=value'."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        counts, _, _ = ax.hist(finite, bins=nbins, range=bounds, color="0.75", edgecolor="none")
        tallest = max(1.0, float(np.max(counts)))
        ax.set_ylim(0, tallest * 1.1)
        ax.set_xlim(bounds if xlim is None else xlim)
        _add_summary_title(ax, summary_label, summary_value, fontsize)
        style_axis(ax, fontsize=fontsize, xbounds=bounds, xticks=list(bounds), yticks=[], spines_visible=["bottom"])
