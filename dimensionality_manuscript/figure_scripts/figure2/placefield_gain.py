"""Single-session gain matrices, covariance, and cross-session regression summary."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

from dimensionality_manuscript.configs.gain_regression import GAIN_TRANSFORMS, PLACEFIELD_SPLITS, GainRegressionConfig, apply_gain_transform
from dimensionality_manuscript.env_order import ENV_SLOT_COLORS, MAX_ENV_SLOTS, _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import FigureViewer
from dimensionality_manuscript.pipeline import ResultsAggregator
from vrAnalysis.helpers import reliability_loo
from vrAnalysis.helpers.plotting import beeswarm, format_spines
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.placefields import get_frame_behavior
from vrAnalysis.processors.spkmaps import SpkmapParams, SpkmapProcessor
from vrAnalysis.sessions import B2Session, SpksTypes

GAIN_SORT_METHODS = ("environment", "rastermap_activity", "rastermap_gain")


def _values_by_mouse(values: np.ndarray, mouse_names: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    """Flatten finite session/environment scores into one vector per mouse."""
    values = np.asarray(values, dtype=float)
    mouse_names = np.asarray(mouse_names)
    if values.shape[0] != mouse_names.size:
        raise ValueError("values and mouse_names must have the same session axis")
    mice = list(dict.fromkeys(mouse_names.tolist()))
    grouped = []
    for mouse in mice:
        mouse_values = values[mouse_names == mouse]
        grouped.append(mouse_values[np.isfinite(mouse_values)])
    return mice, grouped


def _finite_mean(values: np.ndarray, axis=None) -> np.ndarray | float:
    """NaN-aware mean without warnings for empty slices."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, values, 0.0), axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan, dtype=float), where=count > 0)


def _pad_curves(curves: dict[str, np.ndarray]) -> np.ndarray:
    """NaN-pad ragged per-mouse curves along their session axis."""
    width = max((len(curve) for curve in curves.values()), default=0)
    padded = np.full((len(curves), width), np.nan)
    for row, curve in enumerate(curves.values()):
        padded[row, : len(curve)] = curve
    return padded


def _full_trial_indices(occupancy: np.ndarray, required_bins: np.ndarray) -> np.ndarray:
    """Return trials covering every required bin under Spkmap's NaN convention."""
    occupancy = np.asarray(occupancy)
    required_bins = np.asarray(required_bins, dtype=int)
    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a (trials, position bins) matrix")
    return np.flatnonzero(np.all(~np.isnan(occupancy[:, required_bins]), axis=1))


def _pairwise_nan_covariance(data: np.ndarray) -> np.ndarray:
    """Neuron covariance using every jointly finite trial for each pair."""
    data = np.asarray(data, dtype=float)
    if np.all(np.isfinite(data)):
        return np.atleast_2d(np.cov(data))
    valid = np.isfinite(data)
    safe = np.where(valid, data, 0.0)
    valid_float = valid.astype(float)
    count = valid_float @ valid_float.T
    pair_sum = safe @ valid_float.T
    cross_product = safe @ safe.T
    correction = np.divide(pair_sum * pair_sum.T, count, out=np.zeros_like(cross_product), where=count > 0)
    return np.divide(
        cross_product - correction,
        count - 1,
        out=np.full_like(cross_product, np.nan),
        where=count > 1,
    )


def _rastermap_sort(data: np.ndarray) -> np.ndarray:
    """Rastermap-sort informative rows while tolerating NaNs and constant rows."""
    from rastermap import Rastermap

    data = np.asarray(data, dtype=float)
    finite = np.isfinite(data)
    counts = np.sum(finite, axis=1, keepdims=True)
    means = np.divide(
        np.sum(np.where(finite, data, 0.0), axis=1, keepdims=True),
        counts,
        out=np.zeros((data.shape[0], 1)),
        where=counts > 0,
    )
    filled = np.where(finite, data, means)
    variance = np.divide(
        np.sum(np.where(finite, data - means, 0.0) ** 2, axis=1),
        counts[:, 0],
        out=np.zeros(data.shape[0]),
        where=counts[:, 0] > 0,
    )
    informative = (counts[:, 0] >= 2) & (variance > np.finfo(float).eps)
    included, excluded = np.flatnonzero(informative), np.flatnonzero(~informative)
    if len(included) < 2:
        return np.concatenate((included, excluded))
    order = np.asarray(Rastermap().fit(filled[included]).isort, dtype=int)
    return np.concatenate((included[order], excluded))


def standardize_by_std(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide every activity column by its full-session standard deviation."""
    activity = np.asarray(activity, dtype=float)
    if activity.ndim != 2:
        raise ValueError("activity must be a (frames, rois) matrix")
    scale = np.nanstd(activity, axis=0)
    valid = np.isfinite(scale) & (scale > 0)
    scaled = np.divide(
        activity,
        scale[None, :],
        out=np.full(activity.shape, np.nan, dtype=float),
        where=valid[None, :],
    )
    return scaled, valid


def threshold_gain_matrix(
    trial_maps: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Trial activity divided by prediction within bins above ``threshold``.

    Parameters use raster orientation: ``trial_maps`` is ``(rois, trials, bins)``
    and ``prediction`` is the trial-average place field, ``(rois, bins)``.
    """
    trial_maps = np.asarray(trial_maps, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if trial_maps.ndim != 3 or prediction.shape != (trial_maps.shape[0], trial_maps.shape[2]):
        raise ValueError("trial_maps and prediction must have shapes (rois, trials, bins) and (rois, bins)")
    zones = np.isfinite(prediction) & (prediction > threshold)
    numerator = _masked_row_mean(trial_maps, zones)
    denominator = _masked_row_mean(prediction[:, None, :], zones)[:, 0]
    return _safe_gain(numerator, denominator)


def gaussian_gain_matrix(
    trial_maps: np.ndarray,
    prediction: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gain from locally fitted Gaussian place fields.

    The fitted Gaussian bump (baseline excluded) supplies the spatial weights. The
    numerator is the weighted trial map and the denominator is the identically weighted
    fitted curve. The returned fitted curves have shape ``(rois, bins)``.
    """
    trial_maps = np.asarray(trial_maps, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if trial_maps.ndim != 3 or prediction.shape != (trial_maps.shape[0], trial_maps.shape[2]):
        raise ValueError("trial_maps and prediction must have shapes (rois, trials, bins) and (rois, bins)")
    if positions.shape != (trial_maps.shape[2],):
        raise ValueError("positions must contain one value per place-field bin")

    fitted = np.full_like(prediction, np.nan, dtype=float)
    weights = np.full_like(prediction, np.nan, dtype=float)
    for roi, curve in enumerate(prediction):
        fit, bump = _fit_gaussian(positions, curve)
        fitted[roi] = fit
        weights[roi] = bump
    numerator = _weighted_row_mean(trial_maps, weights)
    denominator = _weighted_row_mean(fitted[:, None, :], weights)[:, 0]
    return _safe_gain(numerator, denominator), fitted


def _masked_row_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & mask[:, None, :]
    return np.divide(
        np.sum(np.where(valid, values, 0.0), axis=2),
        np.sum(valid, axis=2),
        out=np.full(values.shape[:2], np.nan, dtype=float),
        where=np.sum(valid, axis=2) > 0,
    )


def _weighted_row_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(weights[:, None, :]) & (weights[:, None, :] > 0)
    effective_weights = np.where(valid, weights[:, None, :], 0.0)
    weight_sum = np.sum(effective_weights, axis=2)
    return np.divide(
        np.sum(np.where(valid, values, 0.0) * effective_weights, axis=2),
        weight_sum,
        out=np.full(values.shape[:2], np.nan, dtype=float),
        where=weight_sum > 0,
    )


def _safe_gain(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    valid = np.isfinite(denominator) & (denominator > np.finfo(float).eps)
    return np.divide(
        numerator,
        denominator[:, None],
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=valid[:, None],
    )


def _gaussian(position: np.ndarray, baseline: float, amplitude: float, center: float, sigma: float) -> np.ndarray:
    return baseline + amplitude * np.exp(-0.5 * ((position - center) / sigma) ** 2)


def _fit_gaussian(position: np.ndarray, curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(position) & np.isfinite(curve)
    if np.sum(valid) < 4 or np.ptp(position[valid]) <= 0:
        return np.full(curve.shape, np.nan), np.full(curve.shape, np.nan)
    x, y = position[valid], curve[valid]
    span = float(np.ptp(x))
    bin_width = float(np.median(np.diff(np.unique(x))))
    baseline = float(np.nanpercentile(y, 10))
    amplitude = max(float(np.nanmax(y) - baseline), 1e-6)
    initial = (baseline, amplitude, float(x[np.nanargmax(y)]), max(span / 10, bin_width))
    try:
        params, _ = curve_fit(
            _gaussian,
            x,
            y,
            p0=initial,
            bounds=([-np.inf, 0.0, x.min(), max(bin_width / 2, 1e-6)], [np.inf, np.inf, x.max(), span]),
            maxfev=5000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return np.full(curve.shape, np.nan), np.full(curve.shape, np.nan)
    fitted = _gaussian(position, *params)
    bump = np.exp(-0.5 * ((position - params[2]) / params[3]) ** 2)
    return fitted, bump


class PlacefieldGainViewer(FigureViewer):
    """View place-field gain, its covariance, and held-out gain-regression scores.

    The first two panels consume the selected session directly. Activity is loaded as
    ``spks_type`` (``sigrebase`` by default), restricted to the session's filtered ROIs, and
    divided by each ROI's full-session standard deviation before any place fields or selection
    metrics are computed. The third panel reads ``r2_test`` from a ``GainRegressionConfig``
    results aggregator. Pooled mode compares mouse-level data and roll-null scores in adjacent
    columns; by-mouse mode shows the observed session/environment swarm and marks each mouse's
    roll-null mean with a red horizontal line.

    Gain always uses the four-parameter Gaussian control curve used by ``GainRegressionConfig``;
    its baseline-free bump supplies weights for both the trial numerator and fitted-curve
    denominator. The reliability and fraction-active cutoffs likewise come directly from that
    config and are not viewer controls. ``gain_transform`` is shared by every panel, while
    ``placefield_split`` applies only to stored regression results.
    """

    def __init__(
        self,
        sessions: list[B2Session],
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        session: str | None = None,
        environment: int | None = None,
        gain_transform: str = "raw",
        placefield_split: str = "train",
        swarm_mode: str = "pooled",
        include_covariance: bool = True,
        include_familiarity: bool = True,
        by_env: bool = False,
        min_trial: int = 10,
        sort_method: str = "environment",
        spks_type: SpksTypes = "sigrebase",
        num_bins: int = 100,
        speed_threshold: float = 1.0,
        smooth_width: float | None = 1.0,
        full_trial_flexibility: float | None = 3.0,
        gain_vmax: float = 2.0,
        covariance_vmax: float = 0.5,
        beewidth: float = 0.2,
        interpolation: str = "none",
        fontsize: float = 8.0,
        figsize: tuple[float, float] = (16.0, 5.0),
    ):
        if not sessions:
            raise ValueError("sessions must contain at least one B2Session")
        if sort_method not in GAIN_SORT_METHODS:
            raise ValueError(f"sort_method must be one of {GAIN_SORT_METHODS}")
        if gain_transform not in GAIN_TRANSFORMS:
            raise ValueError(f"gain_transform must be one of {GAIN_TRANSFORMS}")
        if placefield_split not in PLACEFIELD_SPLITS:
            raise ValueError(f"placefield_split must be one of {PLACEFIELD_SPLITS}")
        if swarm_mode not in ("pooled", "by_mouse"):
            raise ValueError("swarm_mode must be 'pooled' or 'by_mouse'")
        if min_trial < 0:
            raise ValueError("min_trial must be non-negative")
        if num_bins < 4:
            raise ValueError("num_bins must be at least 4")

        self.sessions = list(sessions)
        self.results = results
        self.spks_type = spks_type
        self.num_bins = int(num_bins)
        self.speed_threshold = speed_threshold
        self.smooth_width = smooth_width
        self.full_trial_flexibility = full_trial_flexibility
        self.figsize = figsize
        self.interpolation = interpolation
        config = results.config_class if isinstance(results.config_class, GainRegressionConfig) else GainRegressionConfig()
        self.reliability_threshold = config.reliability_threshold
        self.fraction_active_threshold = config.fraction_active_threshold
        self._loaded: dict[str, dict] = {}
        self._gain_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._sort_cache: dict[tuple, np.ndarray] = {}

        by_mouse: dict[str, list[B2Session]] = defaultdict(list)
        for candidate in self.sessions:
            by_mouse[candidate.mouse_name].append(candidate)
        self._sessions_by_mouse = {name: sorted(candidates, key=_session_sort_key) for name, candidates in sorted(by_mouse.items())}
        self._session_lookup = {
            name: {candidate.session_print(): candidate for candidate in candidates} for name, candidates in self._sessions_by_mouse.items()
        }
        initial_mouse = mouse if mouse in self._sessions_by_mouse else next(iter(self._sessions_by_mouse))
        session_options = list(self._session_lookup[initial_mouse])
        initial_session = session if session in session_options else session_options[0]

        # Discover the initial session's environment values before constructing its widget.
        initial_data = self._load(self._session_lookup[initial_mouse][initial_session])
        env_options = list(initial_data["environments"])
        initial_environment = environment if environment in env_options else env_options[0]

        self.add_selection("mouse", value=initial_mouse, options=list(self._sessions_by_mouse))
        self.add_selection("session", value=initial_session, options=session_options)
        self.add_selection("environment", value=initial_environment, options=env_options)
        # One transform controls both direct-session images and the matching stored regression.
        self.add_selection("gain_transform", value=gain_transform, options=list(GAIN_TRANSFORMS))
        self.add_selection("placefield_split", value=placefield_split, options=list(PLACEFIELD_SPLITS))
        self.add_selection("swarm_mode", value=swarm_mode, options=["pooled", "by_mouse"])
        self.add_boolean("include_covariance", value=include_covariance)
        self.add_boolean("include_familiarity", value=include_familiarity)
        self.add_boolean("by_env", value=by_env)
        self.add_integer("min_trial", value=min_trial, min=0, max=100)
        self.add_selection("sort_method", value=sort_method, options=list(GAIN_SORT_METHODS))
        self.add_float("gain_vmax", value=gain_vmax, min=0.01, max=20.0)
        self.add_float("covariance_vmax", value=covariance_vmax, min=0.001, max=20.0)
        self.add_float("beewidth", value=beewidth, min=0.0, max=0.5, step=0.01)
        self.add_selection("interpolation", value=interpolation, options=["none", "nearest", "bilinear", "bicubic", "auto"])
        self.add_float("fontsize", value=fontsize, min=1.0, max=30.0)

        self.on_change("mouse", self._update_mouse)
        self.on_change("session", self._update_session)
        self.on_change(
            [
                "environment",
                "gain_transform",
                "sort_method",
            ],
            self.refresh_data,
        )
        self.on_change(["placefield_split", "min_trial"], self.refresh_summary)
        self._select_session(initial_mouse, initial_session, initial_environment)
        self.refresh_data(self.state)

    def _load(self, session: B2Session) -> dict:
        if session.session_uid in self._loaded:
            return self._loaded[session.session_uid]
        previous_spks_type = session.params.spks_type
        session.params.spks_type = self.spks_type
        try:
            raw = np.asarray(session.spks)[:, np.asarray(session.idx_rois, dtype=bool)]
            activity, valid_rois = standardize_by_std(raw)
            activity = activity[:, valid_rois]
            frame_behavior = get_frame_behavior(session)
            env_length = float(np.asarray(session.env_length).flat[0])
            smp = SpkmapProcessor(
                session,
                params=SpkmapParams(
                    dist_step=env_length / self.num_bins,
                    speed_threshold=self.speed_threshold,
                    standardize_spks=False,
                    smooth_width=self.smooth_width,
                    full_trial_flexibility=self.full_trial_flexibility,
                    autosave=False,
                ),
            )
            # Reproduce get_env_maps' trial bookkeeping before its processing mutates maps:
            # a trial is full when every bin outside the allowed edge flexibility lies
            # inside that trial's observed positional range. Internal zero-occupancy bins
            # do not cause an otherwise full traversal to be rejected.
            raw_maps = smp.get_raw_maps()
            required_bins = smp._idx_required_position_bins()
            full_trials = _full_trial_indices(raw_maps.occmap, required_bins)
            trials_by_environment = {env: full_trials[np.asarray(session.trial_environment)[full_trials] == env] for env in session.environments}
            env_maps = smp.get_env_maps()
            env_maps.distcenters = smp.dist_centers
            # This is the same post-full-trial cleanup used by figure_scripts.session_cache:
            # remove only bins that remain NaN in at least one retained traversal.
            env_maps.pop_nan_positions()
            env_maps.spkmap = [maps[valid_rois] / np.nanstd(raw, axis=0)[valid_rois, None, None] for maps in env_maps.spkmap]
        finally:
            session.params.spks_type = previous_spks_type
        environments = np.asarray(env_maps.environments)
        if not len(environments):
            raise ValueError(f"{session.session_print()} contains no environments with place-field data")
        data = {
            "session": session,
            "activity": activity,
            "frame_behavior": frame_behavior,
            "env_maps": env_maps,
            "trials_by_environment": trials_by_environment,
            "positions": np.asarray(env_maps.distcenters),
            "environments": environments,
        }
        self._loaded[session.session_uid] = data
        return data

    def _update_mouse(self, state) -> None:
        options = list(self._session_lookup[state["mouse"]])
        selected = state["session"] if state["session"] in options else options[0]
        self.update_selection("session", value=selected, options=options)
        self._update_session({**state, "session": selected})

    def _update_session(self, state) -> None:
        data = self._load(self._session_lookup[state["mouse"]][state["session"]])
        options = list(data["environments"])
        selected = state["environment"] if state["environment"] in options else options[0]
        self._select_session(state["mouse"], state["session"], selected)
        self.update_selection("environment", value=selected, options=options)
        self.refresh_data({**state, "environment": selected})

    def _select_session(self, mouse: str, session: str, environment: int) -> None:
        self.data = self._load(self._session_lookup[mouse][session])
        self.session = self.data["session"]
        self.environment = environment

    def _environment_maps(self, environment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        env_maps = self.data["env_maps"]
        env_index = list(env_maps.environments).index(environment)
        maps = np.asarray(env_maps.spkmap[env_index], dtype=float)
        if maps.shape[1] < 2:
            raise ValueError(f"Environment {environment} has fewer than two full trials")
        if maps.shape[2] < 4:
            raise ValueError(f"Environment {environment} has fewer than four common position bins")
        return maps, self.data["positions"], self.data["trials_by_environment"][environment]

    def refresh_data(self, state) -> None:
        self._select_session(state["mouse"], state["session"], state["environment"])
        trial_maps, positions, trials = self._environment_maps(state["environment"])
        reliability = reliability_loo(trial_maps)
        fraction_active = FractionActive.compute(
            trial_maps,
            activity_axis=2,
            fraction_axis=1,
            activity_method="rms",
            fraction_method="participation",
        )
        idx_keep = np.isfinite(reliability) & np.isfinite(fraction_active)
        idx_keep &= reliability >= self.reliability_threshold
        idx_keep &= fraction_active >= self.fraction_active_threshold
        if not np.any(idx_keep):
            raise ValueError("No ROI passes the reliability and fraction-active thresholds")

        trial_maps = trial_maps[idx_keep]
        prediction = np.nanmean(trial_maps, axis=1)
        gain_key = (
            self.session.session_uid,
            state["environment"],
            idx_keep.tobytes(),
        )
        if gain_key not in self._gain_cache:
            gain, fitted = gaussian_gain_matrix(trial_maps, prediction, positions)
            self._gain_cache[gain_key] = gain, fitted
        gain, fitted = self._gain_cache[gain_key]

        informative = np.any(np.isfinite(gain), axis=1)
        if not np.any(informative):
            raise ValueError("No selected ROI has a valid gain denominator")
        gain = gain[informative]
        fitted = fitted[informative]
        kept_indices = np.flatnonzero(idx_keep)[informative]
        sort_key = gain_key + (state["sort_method"], informative.tobytes())
        if sort_key not in self._sort_cache:
            if state["sort_method"] == "environment":
                peak = np.argmax(np.where(np.isfinite(fitted), fitted, -np.inf), axis=1)
                order = np.lexsort((kept_indices, positions[peak]))
            elif state["sort_method"] == "rastermap_activity":
                fb = self.data["frame_behavior"]
                idx_frames = (fb.environment == state["environment"]) & fb.valid_frames()
                activity = self.data["activity"][idx_frames][:, kept_indices].T
                order = _rastermap_sort(activity)
            else:
                order = _rastermap_sort(gain)
            self._sort_cache[sort_key] = np.asarray(order, dtype=int)
        order = self._sort_cache[sort_key]
        self.gain = apply_gain_transform(gain[order], state["gain_transform"])
        self.covariance = _pairwise_nan_covariance(self.gain)
        self.roi_indices = kept_indices[order]
        self.trials = trials
        self.refresh_summary(state)

    def refresh_summary(self, state) -> None:
        """Select and trial-filter held-out gain-prediction scores for ax[2:4]."""
        selection = {
            "gain_transform": state["gain_transform"],
            "placefield_split": state["placefield_split"],
        }
        selected = self.results.sel(
            keys=["r2_test", "r2_test_null", "n_trials_env", "env_slot_ids"],
            squeeze_ones=False,
            **selection,
        )
        self.summary_trials = np.asarray(selected["n_trials_env"], dtype=float)
        self.summary_env_ids = np.asarray(selected["env_slot_ids"], dtype=float)
        self.summary_scores = np.asarray(selected["r2_test"], dtype=float).copy()
        self.summary_null_scores = np.asarray(selected["r2_test_null"], dtype=float).copy()
        enough_trials = np.isfinite(self.summary_trials) & (self.summary_trials >= state["min_trial"])
        self.summary_scores[~enough_trials] = np.nan
        self.summary_null_scores[~enough_trials] = np.nan
        self.summary_mice, self.summary_values = _values_by_mouse(self.summary_scores, self.results.mouse_names)
        null_mice, self.summary_null_values = _values_by_mouse(self.summary_null_scores, self.results.mouse_names)
        if null_mice != self.summary_mice:
            raise ValueError("real and null results have inconsistent mouse axes")

        self.selected_result = None
        result_rows = np.flatnonzero(np.asarray(self.results.session_ids) == self.session.session_uid)
        if result_rows.size:
            row = int(result_rows[0])
            slots = np.flatnonzero(self.summary_env_ids[row] == state["environment"])
            if slots.size:
                slot = int(slots[0])
                self.selected_result = (row, slot, float(self.summary_scores[row, slot]))

    @staticmethod
    def _style_image_axis(ax, title: str, fontsize: float) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(1.0, 1.0, title, transform=ax.transAxes, ha="right", va="top", fontsize=fontsize)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_summary(self, ax, state) -> None:
        """Draw pooled mouse averages or per-mouse session/environment swarms."""
        line_extent = np.array([-0.25, 0.25])
        mouse_names = np.asarray(self.results.mouse_names)
        selected_x = None

        if state["swarm_mode"] == "pooled":
            session_scores = _finite_mean(self.summary_scores, axis=1)
            values = np.asarray([_finite_mean(session_scores[mouse_names == mouse]) for mouse in self.summary_mice])
            null_session_scores = _finite_mean(self.summary_null_scores, axis=1)
            null_values = np.asarray([_finite_mean(null_session_scores[mouse_names == mouse]) for mouse in self.summary_mice])
            for x, column_values, color in ((0, values, "k"), (1, null_values, "k")):
                finite = np.isfinite(column_values)
                offsets = np.zeros_like(column_values)
                if finite.any():
                    offsets[finite] = beeswarm(column_values[finite])
                ax.plot(
                    x + state["beewidth"] * offsets,
                    column_values,
                    color=color,
                    linestyle="none",
                    marker="o",
                    markersize=3,
                    alpha=0.8,
                )
                if finite.any():
                    ax.plot(x + line_extent, [_finite_mean(column_values)] * 2, color=color, linewidth=2.0)
            ax.set_xlim(-0.5, 1.5)
            xticks = [0, 1]
            xlabels = ["Data", "Shuffle"]
            xbounds = [0, 1]
            selected_x = 0.0
        else:
            for x, (values, null_values) in enumerate(zip(self.summary_values, self.summary_null_values)):
                offsets = beeswarm(values) if values.size else np.empty(0)
                ax.plot(
                    x + state["beewidth"] * offsets,
                    values,
                    color="k",
                    linestyle="none",
                    marker="o",
                    markersize=3,
                    alpha=0.3,
                )
                if values.size:
                    ax.plot(x + line_extent, [np.mean(values)] * 2, color="k", linewidth=2.0)
                if null_values.size:
                    ax.plot(x + line_extent, [np.mean(null_values)] * 2, color="red", linewidth=2.0)
            ax.set_xlim(-0.5, len(self.summary_mice) - 0.5)
            xticks = range(len(self.summary_mice))
            xlabels = self.summary_mice
            xbounds = [0, len(self.summary_mice) - 1]
            if self.selected_result is not None:
                selected_mouse = mouse_names[self.selected_result[0]]
                selected_x = float(self.summary_mice.index(selected_mouse))

        if self.selected_result is not None and selected_x is not None and np.isfinite(self.selected_result[2]):
            ax.plot(selected_x, self.selected_result[2], color="red", marker="o", markersize=2, linestyle="none", zorder=5)

        ax.set_ylabel(r"Gain Prediction $R^2$", fontsize=state["fontsize"])
        ylim = ax.get_ylim()
        tick_start = int(np.floor(10 * ylim[0]))
        tick_stop = int(np.ceil(10 * ylim[1]))
        yticks = np.arange(tick_start, tick_stop + 1, dtype=float) / 10
        ax.set_ylim(yticks[0], yticks[-1])
        format_spines(
            ax,
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=xbounds,
            xticks=xticks,
            ybounds=(yticks[0], yticks[-1]),
            tick_length=2,
            tick_fontsize=state["fontsize"],
            yticks=yticks,
        )
        ax.set_xticks(xticks, xlabels, rotation=45, ha="right", fontsize=state["fontsize"])

    def _familiarity_curves(self, by_env: bool) -> list[tuple[str, str, dict[str, np.ndarray]]]:
        """Build per-mouse chronological score curves for the familiarity panel."""
        mice = list(dict.fromkeys(np.asarray(self.results.mouse_names).tolist()))
        if by_env:
            output = []
            for slot in range(MAX_ENV_SLOTS):
                curves = {}
                for mouse in mice:
                    indices = np.flatnonzero(np.asarray(self.results.mouse_names) == mouse)
                    indices = np.asarray(sorted(indices, key=lambda index: _session_sort_key(self.results.sessions[index])))
                    present = np.isfinite(self.summary_trials[indices, slot])
                    curves[mouse] = self.summary_scores[indices[present], slot]
                output.append((f"Env #{slot + 1}", ENV_SLOT_COLORS[slot], curves))
            return output

        curves = {}
        session_scores = _finite_mean(self.summary_scores, axis=1)
        for mouse in mice:
            indices = np.flatnonzero(np.asarray(self.results.mouse_names) == mouse)
            indices = np.asarray(sorted(indices, key=lambda index: _session_sort_key(self.results.sessions[index])))
            curves[mouse] = session_scores[indices]
        return [("All Environments", "k", curves)]

    def _draw_familiarity(self, ax, state) -> None:
        """Draw mean gain-prediction performance over familiarity, with mice faintly behind it."""
        max_width = 0
        for label, color, curves in self._familiarity_curves(state["by_env"]):
            padded = _pad_curves(curves)
            max_width = max(max_width, padded.shape[1])
            for values in curves.values():
                ax.plot(np.arange(len(values)), values, color=(color, 0.2), linewidth=0.5)
            if padded.shape[1]:
                ax.plot(
                    np.arange(padded.shape[1]),
                    _finite_mean(padded, axis=0),
                    color=color,
                    linewidth=2.0,
                    label=label,
                )

        max_x = max(max_width - 1, 0)
        ax.set_xlim(-0.5, max_x + 0.5)
        ax.set_xlabel("Session # Within Environment" if state["by_env"] else "Overall Session #", fontsize=state["fontsize"])
        ax.set_ylabel(r"Gain Prediction $R^2$", fontsize=state["fontsize"])
        ylim = ax.get_ylim()
        format_spines(
            ax,
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[0, max_x],
            ybounds=ylim,
            tick_length=2,
            tick_fontsize=state["fontsize"],
        )
        if state["by_env"]:
            ax.legend(frameon=False, fontsize=state["fontsize"])

    def plot(self, state):
        summary_width = 0.8 if state["swarm_mode"] == "pooled" else 1.5
        width_ratios = [2.0]
        if state["include_covariance"]:
            width_ratios.append(1.0)
        width_ratios.append(summary_width)
        if state["include_familiarity"]:
            width_ratios.append(1.5)
        fig, ax = self.new_subplots(1, len(width_ratios), figsize=self.figsize, layout="constrained", width_ratios=width_ratios)
        # The panels are allocated at a 2:1 width ratio. Matching that with 1:2 and
        # 1:1 box aspects gives them the same physical height while keeping the
        # covariance panel square (and therefore preserving its equal pixel aspect).

        if state["include_covariance"]:
            ax[0].set_box_aspect(0.5)
        ax[0].imshow(self.gain, aspect="auto", cmap="coolwarm", interpolation=state["interpolation"], vmin=0.0, vmax=state["gain_vmax"])
        self._style_image_axis(ax[0], "", state["fontsize"])
        ax[0].set_xlabel("Trials", fontsize=state["fontsize"])
        ax[0].set_ylabel("ROIs", fontsize=state["fontsize"])

        next_axis = 1
        if state["include_covariance"]:
            covariance_ax = ax[next_axis]
            covariance_ax.set_box_aspect(1.0)
            covariance_ax.imshow(
                self.covariance,
                aspect="equal",
                cmap="coolwarm",
                interpolation=state["interpolation"],
                vmin=-state["covariance_vmax"],
                vmax=state["covariance_vmax"],
            )
            self._style_image_axis(covariance_ax, "Gain Covariance", state["fontsize"])
            covariance_ax.set_xlabel("ROIs", fontsize=state["fontsize"])
            covariance_ax.set_ylabel("ROIs", fontsize=state["fontsize"])
            next_axis += 1

        self._draw_summary(ax[next_axis], state)
        next_axis += 1
        if state["include_familiarity"]:
            self._draw_familiarity(ax[next_axis], state)
        return fig
