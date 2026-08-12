"""Per-ROI reliability versus prediction quality and their change with familiarity."""

import numpy as np

from vrAnalysis.helpers import vectorRSquared

from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
    render_curve_group,
)
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import env_slot_color, ordinal, pad_stack, style_axis, support_length

# Loading session_cache imports Rastermap and its numerical stack. Keep the existing
# ReliabilityPredictionFamiliarity import lightweight; the focus viewer resolves it on demand.
session_cache = None


def _get_session_cache():
    global session_cache
    if session_cache is None:
        from dimensionality_manuscript.figure_scripts import session_cache as cache

        session_cache = cache
    return session_cache


def _soft_placefield_weights(prediction: np.ndarray, peak: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RegressionPlacefieldResidualConfig's soft PF membership, kept local to avoid its heavy model imports."""
    valid_peak = np.isfinite(peak) & (peak > 0)
    valid = valid & np.isfinite(prediction) & valid_peak[:, None]
    membership = np.divide(
        np.clip(prediction, 0, None),
        peak[:, None],
        out=np.full_like(prediction, np.nan),
        where=valid_peak[:, None],
    )
    membership = np.clip(membership, 0, 1)
    return np.where(valid, membership, 0.0), np.where(valid, 1.0 - membership, 0.0)


def _weighted_mse(residual: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = np.sum(weights, axis=1)
    error = np.where(weights > 0, residual, 0.0)
    return np.divide(
        np.sum(weights * error**2, axis=1),
        total,
        out=np.full(residual.shape[0], np.nan),
        where=total > 0,
    )


def _weighted_r2(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = np.sum(weights, axis=1)
    weighted_mean = np.divide(
        np.sum(weights * np.where(weights > 0, target, 0.0), axis=1),
        total,
        out=np.full(target.shape[0], np.nan),
        where=total > 0,
    )
    residual = np.where(weights > 0, target - prediction, 0.0)
    deviation = np.where(weights > 0, target - weighted_mean[:, None], 0.0)
    ss_res = np.sum(weights * residual**2, axis=1)
    ss_tot = np.sum(weights * deviation**2, axis=1)
    ratio = np.divide(ss_res, ss_tot, out=np.full(target.shape[0], np.nan), where=ss_tot > 0)
    return 1.0 - ratio


def _prediction_quality_by_region(activity: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Per-ROI R² and RMS for all frames and soft within/outside-PF regions.

    ``activity`` and ``prediction`` are ``(frames, rois)`` arrays from the same environment.
    Overall values exactly follow :class:`PFPredQualityConfig`. Regional values use the soft
    place-field membership definition from :class:`RegressionPlacefieldResidualConfig`, but on
    these all-trial/in-fold predictions rather than on a regression model's held-out frames.
    """
    activity = np.asarray(activity, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if activity.shape != prediction.shape or activity.ndim != 2:
        raise ValueError("activity and prediction must be matching (frames, rois) arrays")

    residual = (activity - prediction).T
    target = activity.T
    predicted = prediction.T
    peak = np.nanmax(predicted, axis=1)
    within_weight, outside_weight = _soft_placefield_weights(predicted, peak, np.isfinite(residual))

    overall_r2 = vectorRSquared(prediction, activity, axis=0)
    overall_r2[overall_r2 < -1] = np.nan
    metrics = {
        "overall": {
            "r2": overall_r2,
            "rms": np.sqrt(np.mean((prediction - activity) ** 2, axis=0)),
        }
    }
    for region, weight in (("within", within_weight), ("outside", outside_weight)):
        r2 = _weighted_r2(target, predicted, weight)
        r2[r2 < -1] = np.nan
        metrics[region] = {"r2": r2, "rms": np.sqrt(_weighted_mse(residual, weight))}
    return metrics


class ReliabilityPredictionFocus(FigureViewer):
    """Inspect one sorted ROI alongside session-wide reliability and PF-quality distributions.

    The activity map and all metrics use the same median-z-scored activity, spike-map parameters,
    all-trial place field, and valid frame mask as :class:`PFPredQualityConfig`. ``within`` and
    ``outside`` use the residual analysis' soft membership weights. They are still all-trial
    (in-fold) measurements; no regression model or held-out split is introduced.

    ``roi`` is a rank, not the session's underlying ROI index. Rank zero is the best value under
    ``sort_by``: largest reliability/R² or smallest RMS. R² and RMS sorting follows
    ``quality_region``. Metrics are computed for every ROI before ``fraction_active_threshold``
    restricts the histogram populations and the selectable ROI ranks.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        example_session: str | None = None,
        env: int = 0,
        roi: int = 0,
        quality_region: str = "overall",
        sort_by: str = "reliability",
        fraction_active_threshold: float = 0.0,
        vmax: float = 5.0,
        hist_bins: int = 30,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (13.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.selection_names = add_data_selection_widgets(
            self, results, defaults=selection_defaults, require=("spks_type",)
        )

        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(name) for name in mouse_names})
        if not self.mice:
            raise ValueError("The aggregator contains no mice.")
        self._sessions_by_mouse = {
            name: {
                results.sessions[row].session_print(): results.sessions[row]
                for row in sorted(np.flatnonzero(mouse_names == name), key=lambda row: _session_sort_key(results.sessions[row]))
            }
            for name in self.mice
        }
        initial_mouse = mouse if mouse in self.mice else self.mice[0]
        session_options = list(self._sessions_by_mouse[initial_mouse])
        initial_session = example_session if example_session in session_options else session_options[0]

        self.add_selection("mouse", value=initial_mouse, options=self.mice)
        self.add_selection("example_session", value=initial_session, options=session_options)
        self.add_integer("env", value=max(0, int(env)), min=0, max=max(0, int(env)))
        self.add_integer("roi", value=max(0, int(roi)), min=0, max=max(0, int(roi)))
        self.add_selection("quality_region", value=quality_region, options=["overall", "within", "outside"])
        self.add_selection("sort_by", value=sort_by, options=["reliability", "rms", "r2"])
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0, step=0.01)
        self.add_float("vmax", value=vmax, min=0.1, max=20.0, step=0.1)
        self.add_integer("hist_bins", value=hist_bins, min=5, max=100)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)

        self.on_change(list(self.selection_names), self.update_example_bounds)
        self.on_change("mouse", self.update_example_session)
        self.on_change("example_session", self.update_example_bounds)
        self.on_change(["env", "quality_region", "sort_by", "fraction_active_threshold"], self.refresh_metrics)
        self.on_change("roi", self.select_roi)
        self.update_example_session(self.state)

    def update_example_session(self, state) -> None:
        options = list(self._sessions_by_mouse[state["mouse"]])
        current = state["example_session"] if state["example_session"] in options else options[0]
        self.update_selection("example_session", value=current, options=options)
        self.update_example_bounds({**state, "example_session": current})

    def _session(self, state):
        return self._sessions_by_mouse[state["mouse"]][state["example_session"]]

    def _with_spks_type(self, state, callback):
        session = self._session(state)
        previous = session.params.spks_type
        session.params.spks_type = state["spks_type"]
        try:
            return callback(session)
        finally:
            session.params.spks_type = previous

    def update_example_bounds(self, state) -> None:
        env_maps = self._with_spks_type(state, _get_session_cache().get_env_maps)
        if not env_maps.environments:
            raise ValueError("The selected session contains no environments.")
        current = min(max(int(state["env"]), 0), len(env_maps.environments) - 1)
        self.update_integer("env", value=current, min=0, max=len(env_maps.environments) - 1)
        self.refresh_metrics({**state, "env": current})

    def refresh_metrics(self, state) -> None:
        cache = _get_session_cache()

        def load(session):
            env_maps = cache.get_env_maps(session)
            reliability = cache.get_reliability(session).values[state["env"]]
            fraction_active = cache.get_fraction_active(session)[state["env"]]
            prediction = cache.get_prediction(session, "spkmap")
            extras = cache.get_prediction_extras(session)
            activity = cache.get_spks(session)
            return env_maps, reliability, fraction_active, prediction, extras, activity

        env_maps, self.reliability, self.fraction_active, prediction, extras, activity = self._with_spks_type(state, load)
        self.environments = [int(value) for value in env_maps.environments]
        idx_keep = extras["idx_valid"] & (extras["frame_environment_index"] == state["env"])
        if not np.any(idx_keep):
            raise ValueError(f"Environment {state['env']} contains no valid place-field prediction frames.")

        self.env_spkmap = env_maps.spkmap[state["env"]]
        self.distcenters = np.asarray(env_maps.distcenters)
        self.quality = _prediction_quality_by_region(activity[idx_keep], prediction[idx_keep])

        sort_values = self.reliability if state["sort_by"] == "reliability" else self.quality[state["quality_region"]][state["sort_by"]]
        self.roi_mask = np.isfinite(self.fraction_active) & (self.fraction_active > state["fraction_active_threshold"])
        eligible_rois = np.flatnonzero(self.roi_mask)
        if not eligible_rois.size:
            raise ValueError(
                f"No ROI has fraction active above {state['fraction_active_threshold']:.2f} "
                f"in environment {self.environments[state['env']]}.")
        eligible_values = sort_values[eligible_rois]
        finite = np.isfinite(eligible_values)
        primary = -sort_values if state["sort_by"] in ("reliability", "r2") else sort_values
        # Stable sorting preserves underlying ROI order for ties; non-finite values always trail.
        order = np.argsort(np.where(finite, primary[eligible_rois], np.inf), kind="stable")
        self.roi_indices = eligible_rois[order]

        current = min(max(int(state["roi"]), 0), len(self.roi_indices) - 1)
        self.update_integer("roi", value=current, min=0, max=max(len(self.roi_indices) - 1, 0))
        self.select_roi({**state, "roi": current})

    def select_roi(self, state) -> None:
        self.roi_index = int(self.roi_indices[state["roi"]])
        self.spkmap = np.asarray(self.env_spkmap[self.roi_index], dtype=float)
        self.placefield = np.nanmean(self.spkmap, axis=0)
        self.rms_by_position = np.sqrt(np.nanmean((self.spkmap - self.placefield) ** 2, axis=0))

    def plot(self, state):
        fig, ax = self.new_subplots(1, 5, figsize=self.figsize, layout="constrained")
        fontsize = state["fontsize"]
        xlims = (float(self.distcenters[0]), float(self.distcenters[-1]))

        ax[0].imshow(
            self.spkmap,
            interpolation="none",
            aspect="auto",
            cmap="gray_r",
            vmin=0,
            vmax=state["vmax"],
            extent=(xlims[0], xlims[1], self.spkmap.shape[0], 0),
        )
        ax[0].set_title(f"ROI {self.roi_index} · env {self.environments[state['env']]}", fontsize=fontsize)
        ax[0].set_xlabel("VR position", fontsize=fontsize)
        ax[0].set_ylabel("Trials", fontsize=fontsize)
        style_axis(ax[0], fontsize=fontsize, spines_visible=["bottom", "left"])

        ax[1].plot(self.distcenters, self.placefield, color="black", linewidth=1.5, label="Place field")
        ax[1].plot(self.distcenters, self.rms_by_position, color="red", linewidth=1.5, label="RMS error")
        ax[1].set_xlim(xlims)
        ax[1].set_xlabel("VR position", fontsize=fontsize)
        ax[1].set_ylabel(r"Activity ($\sigma$)", fontsize=fontsize)
        ax[1].legend(frameon=False, fontsize=fontsize)
        style_axis(ax[1], fontsize=fontsize)

        region = state["quality_region"]
        histogram_specs = (
            (self.reliability[self.roi_mask], self.reliability[self.roi_index], "Reliability", "black"),
            (self.quality[region]["r2"][self.roi_mask], self.quality[region]["r2"][self.roi_index], rf"$R^2$ ({region})", "black"),
            (self.quality[region]["rms"][self.roi_mask], self.quality[region]["rms"][self.roi_index], f"RMS ({region})", "red"),
        )
        for axis, (values, selected, xlabel, color) in zip(ax[2:], histogram_specs):
            finite_values = values[np.isfinite(values)]
            axis.hist(finite_values, bins=state["hist_bins"], color="0.65", edgecolor="none")
            if np.isfinite(selected):
                axis.axvline(selected, color=color, linewidth=1.5)
            axis.set_xlabel(xlabel, fontsize=fontsize)
            axis.set_ylabel("ROIs", fontsize=fontsize)
            style_axis(axis, fontsize=fontsize)
        return fig


class ReliabilityPredictionFamiliarity(FigureViewer):
    """Compare spatial reliability with R² or RMS error, then follow both over familiarity.

    ``ax[0]`` is a per-ROI scatter from one selected session and environment. ``ax[1]`` and
    ``ax[2]`` use the experience-slot presentation of :class:`R2Familiarity`'s final panel:
    each color is an environment in acquisition order, x is the number of sessions the mouse
    has experienced that environment, and curves are aggregated across mice. The same ROI mask
    is used in all panels, so enabling either filter keeps the reliability and prediction-quality
    summaries paired.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``PFPredQualityConfig`` schema-v8 results.
    mouse, example_session, env
        Initial example selection, following ``CrossValidatedPlacefieldFocus``. ``env`` is the
        zero-based index into the selected session's stored environments.
    metric : {"rms", "norm_rms", "r2", "pf_peak", "fraction_variance"}
        Prediction-quality value shown on the scatter y-axis and in ``ax[2]``.
        ``pf_peak`` is the peak place-field amplitude in activity-standard-deviation units.
        ``fraction_variance`` is prediction variance divided by activity variance.
    filter_by_reliability, reliability_threshold
        Optionally retain only ROIs above the spatial-reliability threshold.
    filter_by_metric, r2_filter_range, rms_filter_range, norm_rms_filter_range,
    pf_peak_filter_range, fraction_variance_filter_range
        Optionally retain only ROIs in the selected metric's inclusive range.
    summary_stat : {"mean", "median"}
        How each session's surviving ROIs are reduced to one point in ``ax[1]``/``ax[2]``.
    plot_style : {"each", "errorPlot"}
        Draw individual-mouse curves plus their mean, or the across-mouse mean and error band.
    hide_error : bool
        Suppress the error band in ``plot_style="errorPlot"``.
    show_legend : bool
        Show the environment-slot legend on ``ax[2]``.
    scatter_size, scatter_alpha, fontsize, figsize
        Figure styling controls.
    **selection_defaults
        Starting values for the aggregator's own parameter axes.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        example_session: str | None = None,
        env: int = 0,
        metric: str = "rms",
        filter_by_reliability: bool = False,
        reliability_threshold: float = 0.7,
        filter_by_metric: bool = False,
        r2_filter_range: tuple[float, float] = (-1.0, 1.0),
        rms_filter_range: tuple[float, float] = (0.0, 5.0),
        norm_rms_filter_range: tuple[float, float] = (0.0, 5.0),
        pf_peak_filter_range: tuple[float, float] = (0.0, 10.0),
        fraction_variance_filter_range: tuple[float, float] = (0.0, 1.0),
        summary_stat: str = "mean",
        plot_style: str = "errorPlot",
        hide_error: bool = False,
        show_legend: bool = True,
        scatter_size: float = 6.0,
        scatter_alpha: float = 0.35,
        fontsize: float = 8.0,
        figsize: tuple[float, float] = (7.0, 2.2),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self._load_arrays(self.state)

        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(name) for name in mouse_names})
        if not self.mice:
            raise ValueError("The aggregator contains no mice.")
        self._rows_by_mouse = {
            name: np.array(
                sorted(np.flatnonzero(mouse_names == name), key=lambda row: _session_sort_key(results.sessions[row])),
                dtype=int,
            )
            for name in self.mice
        }
        self._example_rows_by_mouse = {
            name: {results.sessions[row].session_print(): int(row) for row in rows if self._row_slots(int(row))}
            for name, rows in self._rows_by_mouse.items()
        }
        self.mice = [name for name in self.mice if self._example_rows_by_mouse[name]]
        if not self.mice:
            raise ValueError("No session contains matched reliability and prediction-quality values.")

        initial_mouse = mouse if mouse in self.mice else self.mice[0]
        session_options = list(self._example_rows_by_mouse[initial_mouse])
        initial_session = example_session if example_session in session_options else session_options[0]

        self.add_selection("mouse", value=initial_mouse, options=self.mice)
        self.add_selection("example_session", value=initial_session, options=session_options)
        self.environments = self._row_environments(self._example_rows_by_mouse[initial_mouse][initial_session])
        self.add_integer("env", value=min(max(int(env), 0), len(self.environments) - 1), min=0, max=len(self.environments) - 1)
        self.add_selection("metric", value=metric, options=["rms", "norm_rms", "r2", "pf_peak", "fraction_variance"])

        self.add_boolean("filter_by_reliability", value=filter_by_reliability)
        self.add_float("reliability_threshold", value=reliability_threshold, min=-1.0, max=1.0, step=0.05)
        self.add_boolean("filter_by_metric", value=filter_by_metric)
        self.add_float_range("r2_filter_range", value=tuple(r2_filter_range), min=-1.0, max=1.0, step=0.05)
        self.add_float_range("rms_filter_range", value=tuple(rms_filter_range), min=0.0, max=20.0, step=0.1)
        self.add_float_range("norm_rms_filter_range", value=tuple(norm_rms_filter_range), min=0.0, max=20.0, step=0.1)
        self.add_float_range("pf_peak_filter_range", value=tuple(pf_peak_filter_range), min=0.0, max=20.0, step=0.1)
        self.add_float_range(
            "fraction_variance_filter_range", value=tuple(fraction_variance_filter_range), min=0.0, max=10.0, step=0.05
        )
        self.add_selection("summary_stat", value=summary_stat, options=["mean", "median"])

        self.add_selection("plot_style", value=plot_style, options=["each", "errorPlot"])
        self.add_boolean("hide_error", value=hide_error)
        self.add_boolean("show_legend", value=show_legend)
        self.add_float("scatter_size", value=scatter_size, min=0.5, max=30.0)
        self.add_float("scatter_alpha", value=scatter_alpha, min=0.01, max=1.0, step=0.01)
        self.add_float("fontsize", value=fontsize, min=3.0, max=30.0)

        self.on_change(list(self.selection_names), self.reload_arrays)
        self.on_change("mouse", self.update_example_session)
        self.on_change("example_session", self.update_example_environment)
        self.on_change(
            [
                "env",
                "metric",
                "filter_by_reliability",
                "reliability_threshold",
                "filter_by_metric",
                "r2_filter_range",
                "rms_filter_range",
                "norm_rms_filter_range",
                "pf_peak_filter_range",
                "fraction_variance_filter_range",
                "summary_stat",
            ],
            self.refresh_data,
        )
        self.update_example_session(self.state)

    def _load_arrays(self, state) -> None:
        keys = ["r2_slot", "rms_slot", "norm_rms_slot", "pf_peak", "frac_var_pred_slot", "reliability_slot", "env_slot_ids"]
        out = self.results.sel(keys=keys, squeeze_ones=False, **data_selection(state, self.results, self.selection_names))
        missing = [key for key in keys if key not in out]
        if missing:
            raise KeyError(f"Aggregator is missing {missing} -- rerun the pfpred_quality sweep (schema v8 required).")
        self.r2_slot = np.asarray(out["r2_slot"], dtype=float)
        self.rms_slot = np.asarray(out["rms_slot"], dtype=float)
        self.norm_rms_slot = np.asarray(out["norm_rms_slot"], dtype=float)
        self.pf_peak_slot = np.asarray(out["pf_peak"], dtype=float)
        self.fraction_variance_slot = np.asarray(out["frac_var_pred_slot"], dtype=float)
        self.reliability_slot = np.asarray(out["reliability_slot"], dtype=float)
        self.env_slot_ids = np.asarray(out["env_slot_ids"], dtype=float)
        if not (
            self.r2_slot.shape
            == self.rms_slot.shape
            == self.norm_rms_slot.shape
            == self.pf_peak_slot.shape
            == self.fraction_variance_slot.shape
            == self.reliability_slot.shape
        ):
            raise ValueError(
                "r2_slot, rms_slot, norm_rms_slot, pf_peak, frac_var_pred_slot, and reliability_slot must have matching shapes."
            )
        self.num_slots = self.r2_slot.shape[1]

    def reload_arrays(self, state) -> None:
        self._load_arrays(state)
        self.update_example_environment(state)
        self.refresh_data(self.state)

    def _row_slots(self, row: int) -> list[int]:
        return [
            slot
            for slot in range(self.num_slots)
            if np.isfinite(self.env_slot_ids[row, slot])
            and np.any(
                np.isfinite(self.reliability_slot[row, slot])
                & np.isfinite(self.r2_slot[row, slot])
                & np.isfinite(self.rms_slot[row, slot])
                & np.isfinite(self.norm_rms_slot[row, slot])
                & np.isfinite(self.pf_peak_slot[row, slot])
            )
        ]

    def _row_environments(self, row: int) -> list[int]:
        environments = [int(self.env_slot_ids[row, slot]) for slot in self._row_slots(row)]
        if not environments:
            raise ValueError("The selected session has no matched reliability and prediction-quality values.")
        return environments

    def update_example_session(self, state) -> None:
        options = list(self._example_rows_by_mouse[state["mouse"]])
        current = state["example_session"] if state["example_session"] in options else options[0]
        self.update_selection("example_session", value=current, options=options)
        self.update_example_environment({**state, "example_session": current})

    def update_example_environment(self, state) -> None:
        row = self._example_rows_by_mouse[state["mouse"]][state["example_session"]]
        self.environments = self._row_environments(row)
        current = min(max(int(state["env"]), 0), len(self.environments) - 1)
        self.update_integer("env", value=current, min=0, max=len(self.environments) - 1)
        self.refresh_data({**state, "env": current})

    def _example_slot(self, state) -> tuple[int, int]:
        row = self._example_rows_by_mouse[state["mouse"]][state["example_session"]]
        environment = self.environments[state["env"]]
        matches = np.flatnonzero(self.env_slot_ids[row] == environment)
        if matches.size != 1:
            raise ValueError(f"Expected one slot for environment {environment}, found {matches.size}.")
        return row, int(matches[0])

    def _paired_values(self, row: int, slot: int, state) -> tuple[np.ndarray, np.ndarray]:
        reliability = self.reliability_slot[row, slot]
        metric = {
            "rms": self.rms_slot,
            "norm_rms": self.norm_rms_slot,
            "r2": self.r2_slot,
            "pf_peak": self.pf_peak_slot,
            "fraction_variance": self.fraction_variance_slot,
        }[state["metric"]][row, slot]
        keep = np.isfinite(reliability) & np.isfinite(metric)
        if state["filter_by_reliability"]:
            keep &= reliability >= state["reliability_threshold"]
        if state["filter_by_metric"]:
            low, high = state[f"{state['metric']}_filter_range"]
            keep &= (metric >= low) & (metric <= high)
        return reliability[keep], metric[keep]

    @staticmethod
    def _summarize(values: np.ndarray, stat: str) -> float:
        return float(np.nanmean(values) if stat == "mean" else np.nanmedian(values))

    def refresh_data(self, state) -> None:
        row, slot = self._example_slot(state)
        self.example_reliability, self.example_metric = self._paired_values(row, slot, state)

        self.reliability_stacks: dict[int, np.ndarray] = {}
        self.metric_stacks: dict[int, np.ndarray] = {}
        for slot in range(self.num_slots):
            reliability_by_mouse = []
            metric_by_mouse = []
            for mouse in self.mice:
                reliability_curve = []
                metric_curve = []
                for row in self._rows_by_mouse[mouse]:
                    reliability, metric = self._paired_values(int(row), slot, state)
                    if reliability.size:
                        reliability_curve.append(self._summarize(reliability, state["summary_stat"]))
                        metric_curve.append(self._summarize(metric, state["summary_stat"]))
                reliability_by_mouse.append(np.asarray(reliability_curve))
                metric_by_mouse.append(np.asarray(metric_curve))
            self.reliability_stacks[slot] = pad_stack(reliability_by_mouse)
            self.metric_stacks[slot] = pad_stack(metric_by_mouse)

    def _draw_by_env(self, ax, stacks, state, ylabel: str, *, legend: bool = False) -> None:
        xmax = 0
        for slot, stack in stacks.items():
            length = support_length(stack)
            if length == 0:
                continue
            render_curve_group(
                ax,
                np.arange(1, length + 1),
                stack[:, :length],
                env_slot_color(slot),
                state["plot_style"],
                label=ordinal(slot + 1),
                hide_error=state["hide_error"],
                linewidth=1.5,
            )
            xmax = max(xmax, length)
        ax.set_xlabel("Env session #", fontsize=state["fontsize"])
        ax.set_ylabel(ylabel, fontsize=state["fontsize"])
        if xmax:
            ax.set_xlim(1, xmax)
        style_axis(ax, fontsize=state["fontsize"])
        if legend:
            handle = ax.legend(fontsize=state["fontsize"], frameon=False, handlelength=0.8, handletextpad=0.5, title="Env")
            if handle is not None:
                handle.get_title().set_fontsize(state["fontsize"])

    def plot(self, state):
        fig, ax = self.new_subplots(1, 3, figsize=self.figsize, layout="constrained")
        metric_label = {
            "rms": "RMS error",
            "norm_rms": "Normalized RMS error",
            "r2": r"$R^2$",
            "pf_peak": r"Peak place-field amplitude ($\sigma$)",
            "fraction_variance": "Fraction of variance in prediction",
        }[state["metric"]]

        ax[0].scatter(
            self.example_reliability,
            self.example_metric,
            s=state["scatter_size"],
            alpha=state["scatter_alpha"],
            color="black",
            linewidths=0,
        )
        ax[0].set_xlabel("Spatial reliability", fontsize=state["fontsize"])
        ax[0].set_ylabel(metric_label, fontsize=state["fontsize"])
        style_axis(ax[0], fontsize=state["fontsize"])

        stat_label = state["summary_stat"].capitalize()
        self._draw_by_env(ax[1], self.reliability_stacks, state, f"{stat_label} reliability")
        self._draw_by_env(ax[2], self.metric_stacks, state, f"{stat_label} {metric_label}", legend=state["show_legend"])
        return fig
