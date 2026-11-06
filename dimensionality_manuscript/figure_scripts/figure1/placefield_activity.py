"""Mean place fields, full-session activity, and their population cross-covariance."""

import numpy as np
from rastermap import Rastermap

from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts import session_cache

from ._shared import (
    COLORSCALE_INSET_RECT, add_colorscale_widgets, colorscale_inset_rect,
    draw_colorscale_inset, env_slot_color,
)
from .rasters import StackedRasterFocus


def _population_covariance(placefield, activity):
    """Covariance across ROIs, with shape (positions, samples), using N - 1.

    Columns containing missing values remain undefined; no missing activity is
    replaced with zero. Fewer than two ROIs also yields undefined covariance.
    """
    if placefield.shape[0] < 2:
        return np.full((placefield.shape[1], activity.shape[1]), np.nan)
    pf = placefield - placefield.mean(axis=0, keepdims=True)
    act = activity - activity.mean(axis=0, keepdims=True)
    return pf.T @ act / (placefield.shape[0] - 1)


def _rastermap_order(data):
    """Order informative rows, leaving constant or missing rows at the end."""
    finite = np.isfinite(data)
    counts = finite.sum(axis=1, keepdims=True)
    means = np.divide(np.where(finite, data, 0).sum(axis=1, keepdims=True), counts,
                      out=np.zeros((len(data), 1)), where=counts > 0)
    filled = np.where(finite, data, means)
    good = (counts[:, 0] > 1) & (np.std(filled, axis=1) > np.finfo(float).eps)
    rows, rest = np.flatnonzero(good), np.flatnonzero(~good)
    if len(rows) < 3 or data.shape[1] < 2:
        return np.concatenate((rows, rest))
    return np.concatenate((rows[Rastermap().fit(filled[rows]).isort], rest))


class PlacefieldActivityFocus(StackedRasterFocus):
    """Three panels: ROI x position, ROI x sample, and position x sample.

    ``results`` supplies mouse/session metadata; stored predictions are not required.
    Maps and activity use the shared session cache and the session's current spks_type,
    in units of each ROI's temporal standard deviation (median-subtracted for
    fluorescence). All frames are retained, even outside VR or below running speed.

    ``all_environments=False`` selects the zero-based ``env_slot`` in the session's
    environment maps. Otherwise maps are concatenated in that same environment order.
    Reliability is required in at least one displayed environment. ``roi_sort`` accepts
    ``environment`` (or ``env_pos``), sorting by strongest mean PF then peak bin,
    or ``rastermap`` fitted to full-session activity. ``sample_sort`` is either
    ``chronological`` or ``rastermap``, fitted to sample population vectors from the
    full session. ``xslice`` selects from this ordered sample axis in both panels.

    Cross-covariance centers each position/sample population vector across ROIs and
    divides their dot product by N - 1; units are sigma squared. It is computed only
    for the displayed slice to avoid materializing a full-session covariance matrix.
    ``transpose_pf`` changes only the first panel's orientation.

    Position traces follow the sample order; Rastermap samples are drawn as points.
    The optional time scalebar appears only for chronological samples and when its
    duration fits inside the displayed window.
    """

    def __init__(
        self, results, *, mouse=None, session=None, all_environments=True, env_slot=0,
        roi_sort="environment", sample_sort="chronological", transpose_pf=False,
        use_reliable=True, reliability_threshold=0.7, xslice=slice(0, 2000),
        vmax_pf=6.0, vmax_activity=6.0, vmax_covariance=6.0,
        show_position=False, position_height=0.5, env_gap=0.2,
        show_zero_sigma=False, show_scalebar=False, scalebar_seconds=60.0,
        scalebar_xy=(0.03, 0.2), scalebar_text_yoffset=0.05,
        fontsize=8.0, figsize=(14.0, 7.0), colorscale_text_y=0.5,
        colorscale_inset_rect=COLORSCALE_INSET_RECT,
    ):
        self.results, self.figsize = results, figsize
        names = np.asarray(results.mouse_names)
        self._session_rows = {
            str(name): {results.sessions[row].session_print(): int(row)
                        for row in sorted(np.flatnonzero(names == name),
                                          key=lambda r: _session_sort_key(results.sessions[r]))}
            for name in sorted(set(names))
        }
        if not self._session_rows:
            raise ValueError("No sessions available.")
        mouse = mouse if mouse in self._session_rows else next(iter(self._session_rows))
        options = list(self._session_rows[mouse])
        session = session if session in options else options[0]
        self._sort_cache = {}
        self.add_selection("mouse", value=mouse, options=list(self._session_rows))
        self.add_selection("session", value=session, options=options)
        self.add_boolean("all_environments", value=all_environments)
        self.add_integer("env_slot", value=max(0, env_slot), min=0, max=max(0, env_slot))
        self.add_selection("roi_sort", value=roi_sort, options=["environment", "env_pos", "rastermap"])
        self.add_selection("sample_sort", value=sample_sort, options=["chronological", "rastermap"])
        self.add_boolean("transpose_pf", value=transpose_pf)
        self.add_boolean("use_reliable", value=use_reliable)
        self.add_float("reliability_threshold", value=reliability_threshold, min=0.0, max=1.0)
        start = max(0, xslice.start or 0)
        stop = max(start + 1, xslice.stop if xslice.stop is not None else 2000)
        self.add_integer("xslice_start", value=start, min=0, max=start)
        self.add_integer("xslice_stop", value=stop, min=1, max=stop)
        for name, value in (("vmax_pf", vmax_pf), ("vmax_activity", vmax_activity),
                            ("vmax_covariance", vmax_covariance)):
            self.add_float(name, value=value, min=0.01, max=max(20.0, value))
        self.add_boolean("show_position", value=show_position)
        self.add_float("position_height", value=position_height, min=0.1, max=3.0)
        self.add_float("env_gap", value=env_gap, min=0.0, max=3.0)
        self.add_boolean("show_zero_sigma", value=show_zero_sigma)
        self.add_boolean("show_scalebar", value=show_scalebar)
        self.add_float("scalebar_seconds", value=scalebar_seconds, min=1.0, max=600.0)
        for name, value in zip(("scalebar_x", "scalebar_y"), scalebar_xy):
            self.add_float(name, value=value, min=0.0, max=1.0, step=0.01)
        self.add_float("scalebar_text_yoffset", value=scalebar_text_yoffset, min=-1.0, max=1.0, step=0.01)
        self.add_float("fontsize", value=fontsize, min=1.0, max=30.0)
        add_colorscale_widgets(self, colorscale_text_y=colorscale_text_y,
                              colorscale_inset_rect=colorscale_inset_rect)
        self.on_change("mouse", self.update_session)
        self.on_change("session", self.load_session)
        self.on_change(["all_environments", "env_slot", "roi_sort", "sample_sort",
                        "use_reliable", "reliability_threshold"], self.refresh_data)
        self.on_change("xslice_start", self.update_xslice_bounds)
        self.load_session(self.state)

    def load_session(self, state):
        self.session = self.results.sessions[self._session_rows[state["mouse"]][state["session"]]]
        self.smp = session_cache.get_smp(self.session)
        maps = session_cache.get_env_maps(self.session)
        maps.average_trials()
        self._mean_maps = maps.spkmap
        self.environments = list(maps.environments)
        if not self.environments:
            raise ValueError("Session has no environment maps.")
        self._full_activity = session_cache.get_spks(self.session).T
        self._reliability = session_cache.get_reliability(self.session)
        position, _, environment, _ = self.smp.get_frame_behavior()
        self.frame_position = np.asarray(position) / float(self.session.env_length[0]) * (len(self.smp.dist_edges) - 1)
        self.frame_environment = np.array([self.environments.index(env) if env in self.environments else -1
                                           for env in environment])
        self.frame_period = float(np.median(np.diff(self.session.timestamps)))
        self.num_valid = self._full_activity.shape[1]
        slot = int(np.clip(state["env_slot"], 0, len(self.environments) - 1))
        start = int(np.clip(state["xslice_start"], 0, max(0, self.num_valid - 1)))
        stop = int(np.clip(state["xslice_stop"], start + 1, max(start + 1, self.num_valid)))
        self.update_integer("env_slot", value=slot, min=0, max=len(self.environments) - 1)
        self.update_integer("xslice_start", value=start, min=0, max=max(0, self.num_valid - 1))
        self.update_integer("xslice_stop", value=stop, min=start + 1, max=max(start + 1, self.num_valid))
        self.refresh_data({**state, "env_slot": slot})

    def refresh_data(self, state):
        slots = list(range(len(self.environments))) if state["all_environments"] else [state["env_slot"]]
        pf = np.concatenate([self._mean_maps[slot] for slot in slots], axis=1)
        keep = np.ones(pf.shape[0], dtype=bool)
        if state["use_reliable"]:
            reliability = {env: val for env, val in zip(self._reliability.environments, self._reliability.values)}
            keep &= np.any([reliability[self.environments[slot]] > state["reliability_threshold"] for slot in slots], axis=0)
        self._roi_indices = np.flatnonzero(keep)
        pf, activity = pf[keep], self._full_activity[keep]
        key = (self.session.session_uid, self.session.params.spks_type, tuple(slots),
               self._roi_indices.tobytes(), state["roi_sort"], state["sample_sort"])
        if key not in self._sort_cache:
            if state["roi_sort"] == "rastermap":
                roi_order = _rastermap_order(activity)
            else:
                # Concatenated bins encode environment first, position second.
                peaks = np.argmax(np.where(np.isfinite(pf), pf, -np.inf), axis=1)
                roi_order = np.argsort(peaks, kind="stable")
            sample_order = (_rastermap_order(activity.T) if state["sample_sort"] == "rastermap"
                            else np.arange(activity.shape[1]))
            self._sort_cache[key] = roi_order, sample_order
        roi_order, self._sample_order = self._sort_cache[key]
        self._placefield = pf[roi_order]
        self._activity = activity[roi_order]
        self._position_boundaries = np.cumsum([self._mean_maps[s].shape[1] for s in slots])[:-1] - 0.5

    def plot(self, state):
        frames = self._sample_order[state["xslice_start"]:state["xslice_stop"]]
        activity = self._activity[:, frames]
        pf = self._placefield
        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(3 if state["show_position"] else 2, 2,
                             width_ratios=[1, 3],
                             height_ratios=([state["position_height"], 1, 1]
                                            if state["show_position"] else [1, 1]))
        row = int(state["show_position"])
        ax_pf = fig.add_subplot(gs[row, 0])
        ax_act = fig.add_subplot(gs[row, 1], sharey=None if state["transpose_pf"] else ax_pf)
        ax_cov = fig.add_subplot(gs[row + 1, 1], sharex=ax_act,
                                sharey=ax_pf if state["transpose_pf"] else None)
        fontsize = state["fontsize"]
        if not pf.shape[0] or not len(frames) or not pf.shape[1]:
            ax_pf.text(0.5, 0.5, "No ROIs, positions, or samples selected", ha="center", transform=ax_pf.transAxes)
            return fig
        covariance = _population_covariance(pf, activity)
        panels = ((ax_pf, pf.T if state["transpose_pf"] else pf, "Mean placefield", "vmax_pf", False),
                  (ax_act, activity, "Activity", "vmax_activity", False),
                  (ax_cov, covariance, "PF–activity cross-covariance", "vmax_covariance", True))
        for ax, data, title, limit, symmetric in panels:
            vmax = state[limit]
            cmap = "bwr" if symmetric else "gray_r"
            ax.imshow(data, aspect="auto", interpolation="none", cmap=cmap,
                      vmin=-vmax if symmetric else 0, vmax=vmax)
            ax.set_title(title, fontsize=fontsize)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            unit = r"\sigma^2" if symmetric else r"\sigma"
            draw_colorscale_inset(ax, cmap,
                left_label=rf"$-{vmax:g}\,{unit}$" if symmetric else (r"$0\,\sigma$" if state["show_zero_sigma"] else None),
                right_label=rf"${vmax:g}\,{unit}$", left_color="w" if symmetric else "k",
                right_color="w", fontsize=fontsize, text_y=state["colorscale_text_y"],
                inset_rect=colorscale_inset_rect(state))
        ax_pf.set_xlabel("ROIs" if state["transpose_pf"] else "Position bins", fontsize=fontsize)
        ax_pf.set_ylabel("Position bins" if state["transpose_pf"] else "ROIs", fontsize=fontsize)
        ax_act.set_ylabel("ROIs", fontsize=fontsize)
        ax_cov.set_ylabel("Position bins", fontsize=fontsize)
        ax_cov.set_xlabel("Samples" if state["sample_sort"] == "chronological" else "Samples (Rastermap order)", fontsize=fontsize)
        for boundary in self._position_boundaries:
            (ax_pf.axhline if state["transpose_pf"] else ax_pf.axvline)(boundary, color="0.5", linewidth=0.5)
            ax_cov.axhline(boundary, color="0.5", linewidth=0.5)
        if state["show_position"]:
            ax_pos = fig.add_subplot(gs[0, 1], sharex=ax_act)
            for slot in range(len(self.environments)):
                y = np.where(self.frame_environment[frames] == slot, self.frame_position[frames], np.nan)
                if state["sample_sort"] == "chronological":
                    y = y.copy()
                    y[np.flatnonzero(np.diff(y) < -(len(self.smp.dist_edges) - 1) / 2) + 1] = np.nan
                y = y + slot * (len(self.smp.dist_edges) - 1) * (1 + state["env_gap"])
                ax_pos.plot(np.arange(len(frames)), y, color=env_slot_color(slot),
                            linestyle="-" if state["sample_sort"] == "chronological" else "none",
                            marker=None if state["sample_sort"] == "chronological" else ".", markersize=1)
            ax_pos.set_ylabel("Pos.", fontsize=fontsize)
            ax_pos.set_xticks([])
            ax_pos.set_yticks([])
            ax_pos.set_yinverted(True)
            for spine in ax_pos.spines.values():
                spine.set_visible(False)
        if (state["show_scalebar"] and state["sample_sort"] == "chronological"
                and state["scalebar_x"] + state["scalebar_seconds"] / self.frame_period / len(frames) <= 1):
            self._draw_scalebar(ax_cov, state, len(frames))
        return fig
