"""Single-session rasters from :class:`RegressionResidualStructureConfig`."""

from __future__ import annotations

import numpy as np
from rastermap import Rastermap

from dimensionality_manuscript.configs.regression import RegressionResidualStructureConfig
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts.figure1._shared import env_slot_color
from dimensionality_manuscript.figure_scripts.panels import FigureViewer
from dimensionality_manuscript.pipeline import ResultsAggregator

RESIDUAL_MODES = ("additive", "multiplicative", "chunk_gain")
SORT_METHODS = ("environment", "rastermap", "rastermap_residual")


def logistic_multiplicative_residual(gain: np.ndarray) -> np.ndarray:
    """Map a positive multiplicative gain to ``[-1, 1]``, with gain 1 at zero.

    This is ``2 * logistic(log(gain)) - 1``.  The algebraically equivalent expression
    below is more stable at very large gains and makes the neutral value explicit.
    Non-positive gains do not have a real log and are left as NaN.
    """
    gain = np.asarray(gain, dtype=float)
    out = np.full(gain.shape, np.nan, dtype=float)
    valid = np.isfinite(gain) & (gain > 0)
    out[valid] = (gain[valid] - 1.0) / (gain[valid] + 1.0)
    return out


def mean_by_unit(data: np.ndarray, unit_index: np.ndarray, num_units: int) -> np.ndarray:
    """Average a ``(rois, frames)`` matrix within integer-valued frame units."""
    data = np.asarray(data, dtype=float)
    unit_index = np.asarray(unit_index, dtype=int)
    if data.ndim != 2 or data.shape[1] != len(unit_index):
        raise ValueError("data must be (rois, frames) and align with unit_index")
    valid = np.isfinite(data)
    sums = np.zeros((data.shape[0], num_units), dtype=float)
    counts = np.zeros((data.shape[0], num_units), dtype=float)
    for unit in range(num_units):
        idx = unit_index == unit
        if np.any(idx):
            sums[:, unit] = np.nansum(data[:, idx], axis=1)
            counts[:, unit] = np.sum(valid[:, idx], axis=1)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def pairwise_nan_covariance(data: np.ndarray) -> np.ndarray:
    """ROI-by-ROI sample covariance, using all finite samples for each pair."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a (rois, samples) matrix")
    if np.all(np.isfinite(data)):
        return np.cov(data)

    valid = np.isfinite(data)
    valid_float = valid.astype(float)
    safe = np.where(valid, data, 0.0)
    count = valid_float @ valid_float.T
    pair_sum = safe @ valid_float.T
    cross_product = safe @ safe.T
    # For each pair (i, j): sum(x_i*x_j) - sum(x_i)*sum(x_j)/n, where every
    # sum is restricted to samples finite in both rows.
    correction = np.divide(
        pair_sum * pair_sum.T,
        count,
        out=np.zeros_like(cross_product),
        where=count > 0,
    )
    return np.divide(
        cross_product - correction,
        count - 1,
        out=np.full_like(cross_product, np.nan),
        where=count > 1,
    )


def prediction_environment_sort(
    prediction: np.ndarray,
    position: np.ndarray,
    environment: np.ndarray,
    num_position_bins: int = 100,
) -> np.ndarray:
    """Sort prediction rows by strongest environment, then predicted peak position."""
    prediction = np.asarray(prediction, dtype=float)
    position = np.asarray(position, dtype=float)
    environment = np.asarray(environment)
    if prediction.ndim != 2 or prediction.shape[1] != len(position) or len(position) != len(environment):
        raise ValueError("prediction, position, and environment must have aligned samples")

    env_values = sorted(np.unique(environment[np.isfinite(environment)]))
    if not env_values:
        return np.arange(prediction.shape[0])
    env_peak = np.full((len(env_values), prediction.shape[0]), -np.inf)
    env_peak_position = np.full((len(env_values), prediction.shape[0]), np.inf)
    for env_slot, env in enumerate(env_values):
        idx_env = (environment == env) & np.isfinite(position)
        if not np.any(idx_env):
            continue
        env_position = position[idx_env]
        low, high = np.nanmin(env_position), np.nanmax(env_position)
        if not np.isfinite(low) or not np.isfinite(high):
            continue
        if high == low:
            position_bin = np.zeros(len(env_position), dtype=int)
        else:
            edges = np.linspace(low, high, num_position_bins + 1)
            position_bin = np.clip(np.searchsorted(edges, env_position, side="right") - 1, 0, num_position_bins - 1)
        binned = np.full((prediction.shape[0], num_position_bins), np.nan)
        env_prediction = prediction[:, idx_env]
        for bin_idx in np.unique(position_bin):
            with np.errstate(invalid="ignore"):
                binned[:, bin_idx] = np.nanmean(env_prediction[:, position_bin == bin_idx], axis=1)
        finite = np.any(np.isfinite(binned), axis=1)
        if np.any(finite):
            safe_binned = np.where(np.isfinite(binned), binned, -np.inf)
            peak_bin = np.argmax(safe_binned, axis=1)
            rows = np.flatnonzero(finite)
            env_peak[env_slot, rows] = safe_binned[rows, peak_bin[rows]]
            env_peak_position[env_slot, rows] = peak_bin[rows]

    strongest_env = np.argmax(env_peak, axis=0)
    rows = np.arange(prediction.shape[0])
    preferred_position = env_peak_position[strongest_env, rows]
    no_prediction = ~np.isfinite(env_peak[strongest_env, rows])
    strongest_env[no_prediction] = len(env_values)
    preferred_position[no_prediction] = np.inf
    return np.lexsort((rows, preferred_position, strongest_env))


def rastermap_sort(data: np.ndarray) -> np.ndarray:
    """Fit Rastermap to informative rows, appending invalid/constant rows afterwards."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a (rois, samples) matrix")
    finite = np.isfinite(data)
    counts = np.sum(finite, axis=1, keepdims=True)
    safe = np.where(finite, data, 0.0)
    row_mean = np.divide(
        np.sum(safe, axis=1, keepdims=True),
        counts,
        out=np.zeros((data.shape[0], 1), dtype=float),
        where=counts > 0,
    )
    filled = np.where(finite, data, row_mean)
    centered = np.where(finite, data - row_mean, 0.0)
    variance = np.divide(
        np.sum(centered**2, axis=1),
        counts[:, 0],
        out=np.zeros(data.shape[0], dtype=float),
        where=counts[:, 0] > 0,
    )
    informative = (counts[:, 0] >= 2) & np.isfinite(variance) & (variance > np.finfo(float).eps)
    informative_rows = np.flatnonzero(informative)
    excluded_rows = np.flatnonzero(~informative)
    if len(informative_rows) < 2:
        return np.concatenate((informative_rows, excluded_rows))
    fitted_order = np.asarray(Rastermap().fit(filled[informative_rows]).isort, dtype=int)
    return np.concatenate((informative_rows[fitted_order], excluded_rows))


class RegressionResidualStructureViewer(FigureViewer):
    """Inspect prediction and residual population structure for one stored session.

    The left column contains position, data, prediction, and residual rasters.  A covariance
    matrix sits immediately to the right of each neural raster, with the upper-right grid cell
    intentionally empty.  Data uses ``gray_r`` from zero to ``vmax``; prediction and residual
    use symmetric ``bwr`` limits.  Covariances have their own symmetric ``bwr`` limit.

    ``chunk_gain`` changes the horizontal unit from held-out frames to the stored
    cell-by-(chunk x trial) gain units.  Prediction is averaged within each unit, gain is
    shown relative to its neutral value of one, and position/environment are reduced to one
    sample per unit.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        session: str | None = None,
        model_name: str | None = None,
        residual_mode: str = "additive",
        sort_method: str = "environment",
        xslice: slice = slice(0, 2000),
        vmax: float = 3.0,
        covariance_vmax: float = 1.0,
        position_height: float = 0.35,
        env_gap: float = 0.2,
        covariance_width: float = 1.0,
        fontsize: float = 8.0,
        figsize: tuple[float, float] = (12.0, 6.0),
    ):
        self.results = results
        self.figsize = figsize
        # Keep only the currently selected large per-session blob.  The aggregator is used for
        # metadata only; this viewer must never materialize its session x model object grid.
        self._result_uid: str | None = None
        self.result: dict | None = None
        self._sort_cache: dict[tuple, np.ndarray] = {}
        self._covariance_key: tuple | None = None
        self._covariances: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._rows = self._stored_rows(results)
        if not self._rows:
            raise ValueError("No RegressionResidualStructureConfig results are stored for these sessions.")

        models = list(dict.fromkeys(model for model, _ in self._rows))
        initial_model = model_name if model_name in models else models[0]
        mice = self._mice_for_model(initial_model)
        initial_mouse = mouse if mouse in mice else mice[0]
        sessions = self._sessions_for(initial_model, initial_mouse)
        initial_session = session if session in sessions else sessions[0]

        self.add_selection("model_name", value=initial_model, options=models)
        self.add_selection("mouse", value=initial_mouse, options=mice)
        self.add_selection("session", value=initial_session, options=sessions)
        self.add_selection("residual_mode", value=residual_mode, options=list(RESIDUAL_MODES))
        self.add_selection("sort_method", value=sort_method, options=list(SORT_METHODS))
        start = max(0, 0 if xslice.start is None else int(xslice.start))
        stop = max(start + 1, 2000 if xslice.stop is None else int(xslice.stop))
        self.add_integer("xslice_start", value=start, min=0, max=start)
        self.add_integer("xslice_stop", value=stop, min=start + 1, max=stop)
        self.add_float("vmax", value=vmax, min=0.01, max=50.0)
        self.add_float("covariance_vmax", value=covariance_vmax, min=0.001, max=50.0)
        self.add_float("position_height", value=position_height, min=0.05, max=2.0)
        self.add_float("env_gap", value=env_gap, min=0.0, max=3.0)
        self.add_float("covariance_width", value=covariance_width, min=0.1, max=4.0)
        self.add_float("fontsize", value=fontsize, min=1.0, max=30.0)

        self.on_change("model_name", self._update_model)
        self.on_change("mouse", self._update_mouse)
        self.on_change("session", self.load_session)
        self.on_change("residual_mode", self._update_mode)
        self.on_change("xslice_start", self._update_stop_bound)
        self.load_session(self.state)

    @staticmethod
    def _stored_rows(results: ResultsAggregator) -> dict[tuple[str, str], tuple[object, dict, int]]:
        """Map ``(model_name, session_uid)`` to config, store row, and result row."""
        rows = results.store.summary_table(
            analysis_type=RegressionResidualStructureConfig.display_name,
            session_ids=list(results.session_ids),
            schema_version=RegressionResidualStructureConfig.schema_version,
        )
        configs = getattr(results, "_key_to_config", {})
        session_row = {session.session_uid: i for i, session in enumerate(results.sessions)}
        out = {}
        for row in rows:
            config = configs.get(row["analysis_key"])
            if config is None or not isinstance(config, RegressionResidualStructureConfig):
                continue
            uid = row["session_id"]
            if uid in session_row:
                out[(config.model_name, uid)] = (config, row, session_row[uid])
        return out

    def _mice_for_model(self, model_name: str) -> list[str]:
        return sorted({self.results.sessions[result_row].mouse_name for (model, _), (_, _, result_row) in self._rows.items() if model == model_name})

    def _sessions_for(self, model_name: str, mouse: str) -> list[str]:
        rows = [
            result_row
            for (model, _), (_, _, result_row) in self._rows.items()
            if model == model_name and self.results.sessions[result_row].mouse_name == mouse
        ]
        rows.sort(key=lambda row: _session_sort_key(self.results.sessions[row]))
        return [self.results.sessions[row].session_print() for row in rows]

    def _update_model(self, state) -> None:
        mice = self._mice_for_model(state["model_name"])
        mouse = state["mouse"] if state["mouse"] in mice else mice[0]
        self.update_selection("mouse", value=mouse, options=mice)
        self._set_sessions_and_load({**state, "mouse": mouse})

    def _update_mouse(self, state) -> None:
        self._set_sessions_and_load(state)

    def _set_sessions_and_load(self, state) -> None:
        sessions = self._sessions_for(state["model_name"], state["mouse"])
        session = state["session"] if state["session"] in sessions else sessions[0]
        self.update_selection("session", value=session, options=sessions)
        self.load_session({**state, "session": session})

    def _update_stop_bound(self, state) -> None:
        self.update_integer("xslice_stop", min=state["xslice_start"] + 1)

    def _update_mode(self, state) -> None:
        self._set_xslice_bounds(state)

    def load_session(self, state) -> None:
        result_row = next(
            row
            for row, candidate in enumerate(self.results.sessions)
            if candidate.session_print() == state["session"] and candidate.mouse_name == state["mouse"]
        )
        self.session = self.results.sessions[result_row]
        config, store_row, _ = self._rows[(state["model_name"], self.session.session_uid)]
        uid = store_row["result_uid"]
        if uid != self._result_uid:
            result = config.get_result(self.results.store, store_row)
            if result is None:
                raise KeyError(f"Stored residual structure result {uid!r} could not be loaded")
            self.result = result
            self._result_uid = uid
        self._validate_result()
        self._set_xslice_bounds(state)

    def _validate_result(self) -> None:
        required = {
            "residual_additive",
            "residual_multiplicative",
            "model_prediction",
            "chunk_gain",
            "chunk_gain_unit_index",
            "chunk_gain_unit_environment",
            "frame_position",
            "frame_environment",
        }
        missing = required - self.result.keys()
        if missing:
            raise KeyError("Residual structure result is missing: " + ", ".join(sorted(missing)))

    def _mode_length(self, mode: str) -> int:
        if mode == "chunk_gain":
            return np.asarray(self.result["chunk_gain"]).shape[1]
        return np.asarray(self.result["model_prediction"]).shape[1]

    def _set_xslice_bounds(self, state) -> None:
        length = self._mode_length(state["residual_mode"])
        start = min(int(state["xslice_start"]), max(length - 1, 0))
        stop = min(max(int(state["xslice_stop"]), start + 1), length)
        self.update_integer("xslice_start", value=start, min=0, max=max(length - 1, 0))
        self.update_integer("xslice_stop", value=stop, min=start + 1, max=max(length, start + 1))

    def _display_data(self, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        prediction = np.asarray(self.result["model_prediction"], dtype=float)
        data = prediction + np.asarray(self.result["residual_additive"], dtype=float)
        position = np.asarray(self.result["frame_position"], dtype=float)
        environment = np.asarray(self.result["frame_environment"])
        if mode == "additive":
            residual = np.asarray(self.result["residual_additive"], dtype=float)
        elif mode == "multiplicative":
            residual = logistic_multiplicative_residual(self.result["residual_multiplicative"])
        elif mode == "chunk_gain":
            gain = np.asarray(self.result["chunk_gain"], dtype=float)
            unit_index = np.asarray(self.result["chunk_gain_unit_index"], dtype=int)
            data = mean_by_unit(data, unit_index, gain.shape[1])
            prediction = mean_by_unit(prediction, unit_index, gain.shape[1])
            position = mean_by_unit(position[None, :], unit_index, gain.shape[1])[0]
            environment = np.asarray(self.result["chunk_gain_unit_environment"])
            residual = gain - 1.0
        else:
            raise ValueError(f"Unknown residual mode {mode!r}")
        return data, prediction, residual, position, environment

    def _sort(
        self,
        state,
        prediction: np.ndarray,
        residual: np.ndarray,
        position: np.ndarray,
        environment: np.ndarray,
    ) -> np.ndarray:
        key = (
            self.session.session_uid,
            state["model_name"],
            state["residual_mode"],
            state["sort_method"],
        )
        if key not in self._sort_cache:
            if state["sort_method"] == "environment":
                idx = prediction_environment_sort(prediction, position, environment)
            elif state["sort_method"] == "rastermap":
                idx = rastermap_sort(prediction)
            elif state["sort_method"] == "rastermap_residual":
                idx = rastermap_sort(residual)
            else:
                raise ValueError(f"Unknown sort method {state['sort_method']!r}")
            self._sort_cache[key] = np.asarray(idx, dtype=int)
        return self._sort_cache[key]

    @staticmethod
    def _hide_image_axis(ax, title: str, fontsize: float) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(1.0, 1.0, title, transform=ax.transAxes, ha="right", va="top", fontsize=fontsize)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_position(self, ax, position: np.ndarray, environment: np.ndarray, state) -> None:
        env_values = sorted(np.unique(environment[np.isfinite(environment)]))
        env_to_slot = {env: slot for slot, env in enumerate(env_values)}
        finite_position = position[np.isfinite(position)]
        span = float(np.ptp(finite_position)) if len(finite_position) else 1.0
        band = span * (1.0 + state["env_gap"])
        x = np.arange(len(position))
        for env in env_values:
            y = position.astype(float).copy()
            y[environment != env] = np.nan
            y += env_to_slot[env] * band
            ax.plot(x, y, color=env_slot_color(env_to_slot[env]), linewidth=1.0)
        ax.set_ylabel("Position", fontsize=state["fontsize"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)

    def plot(self, state):
        data, prediction, residual, position, environment = self._display_data(state["residual_mode"])
        idx_sort = self._sort(state, prediction, residual, position, environment)
        xslice = slice(state["xslice_start"], min(state["xslice_stop"], prediction.shape[1]))
        data = data[idx_sort, xslice]
        prediction = prediction[idx_sort, xslice]
        residual = residual[idx_sort, xslice]
        position = position[xslice]
        environment = environment[xslice]
        covariance_key = (
            self._result_uid,
            state["residual_mode"],
            state["sort_method"],
            xslice.start,
            xslice.stop,
        )
        if covariance_key != self._covariance_key:
            self._covariances = tuple(pairwise_nan_covariance(matrix) for matrix in (data, prediction, residual))
            self._covariance_key = covariance_key
        data_cov, prediction_cov, residual_cov = self._covariances

        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(
            4,
            2,
            height_ratios=[state["position_height"], 1.0, 1.0, 1.0],
            width_ratios=[3.0, state["covariance_width"]],
        )
        ax_position = fig.add_subplot(gs[0, 0])
        ax_data = fig.add_subplot(gs[1, 0])
        ax_prediction = fig.add_subplot(gs[2, 0], sharex=ax_data, sharey=ax_data)
        ax_residual = fig.add_subplot(gs[3, 0], sharex=ax_data, sharey=ax_data)
        ax_data_cov = fig.add_subplot(gs[1, 1])
        ax_prediction_cov = fig.add_subplot(gs[2, 1], sharex=ax_data_cov, sharey=ax_data_cov)
        ax_residual_cov = fig.add_subplot(gs[3, 1], sharex=ax_data_cov, sharey=ax_data_cov)

        image_kwargs = dict(cmap="bwr", vmin=-state["vmax"], vmax=state["vmax"])
        covariance_kwargs = dict(
            cmap="bwr",
            vmin=-state["covariance_vmax"],
            vmax=state["covariance_vmax"],
        )
        ax_data.imshow(data, aspect="auto", cmap="gray_r", vmin=0, vmax=state["vmax"])
        ax_prediction.imshow(prediction, aspect="auto", cmap="gray_r", vmin=0, vmax=state["vmax"])
        ax_residual.imshow(residual, aspect="auto", **image_kwargs)
        ax_data_cov.imshow(data_cov, aspect="equal", **covariance_kwargs)
        ax_prediction_cov.imshow(prediction_cov, aspect="equal", **covariance_kwargs)
        ax_residual_cov.imshow(residual_cov, aspect="equal", **covariance_kwargs)
        self._draw_position(ax_position, position, environment, state)
        mode_title = {
            "additive": "Additive Residual",
            "multiplicative": "Multiplicative Residual (logistic)",
            "chunk_gain": "Chunk Gain - 1",
        }
        self._hide_image_axis(ax_data, "Data", state["fontsize"])
        self._hide_image_axis(ax_prediction, "Prediction", state["fontsize"])
        self._hide_image_axis(ax_residual, mode_title[state["residual_mode"]], state["fontsize"])
        self._hide_image_axis(ax_data_cov, "Data Covariance", state["fontsize"])
        self._hide_image_axis(ax_prediction_cov, "Prediction Covariance", state["fontsize"])
        self._hide_image_axis(ax_residual_cov, "Residual Covariance", state["fontsize"])
        ax_data.set_ylabel("ROIs", fontsize=state["fontsize"])
        ax_prediction.set_ylabel("ROIs", fontsize=state["fontsize"])
        ax_residual.set_ylabel("ROIs", fontsize=state["fontsize"])
        return fig
