"""Odd-trial place fields across a mouse's sessions, each sorted independently on even trials."""

import numpy as np

from dimensionality_manuscript.configs.behavior_speed_env import REFERENCE_ENV_LENGTH_CM
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import draw_vertical_colorscale, hide_spines


class CrossValidatedPlacefields(FigureViewer):
    """Precomputed odd-trial place fields, sorted by the *even* trials, one panel per session.

    Sorting on held-out trials is the whole point: a map sorted on the trials it displays shows a
    diagonal whether or not the cells have place fields, so the structure visible here is the
    structure that generalizes. Each panel is one session of one mouse in one environment, ROIs
    filtered by reliability and fraction-active and normalized by each ROI's standard deviation in
    time, so a single ``vmax`` is meaningful across ROIs and sessions.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``CrossValidatedPlacefieldsConfig`` results.
    mouse : str or None
        Mouse to draw. None takes the first with computed place fields.
    environment : int or None
        Environment id. None takes the first this mouse ran.
    session_range : tuple[int, int] or None
        Inclusive 1-based range of the mouse's sessions to draw. None uses the first and last
        session containing the environment.
    skip_session : int
        Draw every ``skip_session + 1``-th session in the range, thinning a long history.
    reliability_threshold, fraction_active_threshold : float
        An ROI is drawn only if it exceeds both in the session and environment shown.
    vmax : float
        Upper limit (in sigma) of the gray_r maps.
    verbose : bool
        Title each panel with its session number and date, and spell out that the ROI order comes
        from the even trials. False leaves a bare strip of rasters.
    scalebar_xy : tuple[float, float]
        Lower-left corner of the scale bar on the first panel, in axes fractions.
    scalebar_cm : float
        Length of the scale bar, converted to a fraction of the reference track length.
    scalebar_fontsize : float
        Font size of the scale-bar label; separate from ``fontsize`` since it usually sits inside
        a panel rather than beside it.
    colorbar_width_ratio : float
        Width of the leading activity color strip, relative to one session panel.
    fontsize : float
        Font size of every other text element.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the aggregator's own param axes, by name.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        environment: int | None = None,
        session_range: tuple[int, int] | None = None,
        skip_session: int = 0,
        reliability_threshold: float = 0.7,
        fraction_active_threshold: float = 0.2,
        vmax: float = 5.0,
        verbose: bool = True,
        scalebar_xy: tuple[float, float] = (0.05, 0.08),
        scalebar_cm: float = 50.0,
        scalebar_fontsize: float = 9.0,
        colorbar_width_ratio: float = 0.12,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (12.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self._keys = ["even_placefield", "odd_placefield", "reliability", "fraction_active", "env_slot_ids", "spks_std"]

        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self._load_arrays(self.state)

        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(name) for name in mouse_names if np.any(np.isfinite(self.even_placefield[mouse_names == name]))})
        if not self.mice:
            raise ValueError("The aggregator contains no computed cross-validated place fields.")
        self._rows_by_mouse = {
            name: np.array(sorted(np.flatnonzero(mouse_names == name), key=lambda row: _session_sort_key(results.sessions[row])))
            for name in self.mice
        }

        # --- data selection; environment options and the session range follow the mouse ---
        self.add_selection("mouse", value=mouse if mouse in self.mice else self.mice[0], options=self.mice)
        self.add_selection("environment", value=environment if environment is not None else 0, options=[environment if environment is not None else 0])
        self.add_integer_range("session_range", value=(1, 1), min=1, max=1)
        self.add_integer("skip_session", value=skip_session, min=0, max=20)
        self.add_float("reliability_threshold", value=reliability_threshold, min=-1.0, max=1.0)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0)

        # --- style ---
        self.add_float("vmax", value=vmax, min=0.1, max=20.0)
        self.add_float("fontsize", value=fontsize, min=1.0, max=30.0)
        self.add_boolean("verbose", value=verbose)
        self.add_float("scalebar_x", value=scalebar_xy[0], min=0.0, max=1.0)
        self.add_float("scalebar_y", value=scalebar_xy[1], min=0.0, max=1.0)
        self.add_float("scalebar_cm", value=scalebar_cm, min=0.1, max=REFERENCE_ENV_LENGTH_CM)
        self.add_float("scalebar_fontsize", value=scalebar_fontsize, min=1.0, max=30.0)
        self.add_float("colorbar_width_ratio", value=colorbar_width_ratio, min=0.01, max=1.0)

        self.on_change(list(self.selection_names), self.reload_arrays)
        self.on_change("mouse", self.update_mouse)
        self.on_change("environment", self.update_session_range)
        self.on_change(["session_range", "skip_session", "reliability_threshold", "fraction_active_threshold"], self.refresh_data)
        self.update_mouse(self.state)
        if session_range is not None:
            self.update_integer_range("session_range", value=tuple(session_range))
            self.refresh_data(self.state)

    # ---------------------------------------------------------------- data selection --

    def _load_arrays(self, state) -> None:
        """Pull every key this panel draws for the current data selection."""
        out = self.results.sel(keys=self._keys, squeeze_ones=False, **data_selection(state, self.results, self.selection_names))
        missing = [key for key in self._keys if key not in out]
        if missing:
            raise KeyError(f"Aggregator is missing {missing} -- run CrossValidatedPlacefieldsConfig first.")
        self.even_placefield = np.asarray(out["even_placefield"], dtype=float)
        self.odd_placefield = np.asarray(out["odd_placefield"], dtype=float)
        self.reliability = np.asarray(out["reliability"], dtype=float)
        self.fraction_active = np.asarray(out["fraction_active"], dtype=float)
        self.env_slot_ids = np.asarray(out["env_slot_ids"], dtype=float)
        self.spks_std = np.asarray(out["spks_std"], dtype=float)

    def reload_arrays(self, state):
        """Re-pull the arrays after a data-selection change, then re-derive what is shown."""
        self._load_arrays(state)
        self.update_mouse(state)

    def _slot_for_row(self, row: int, environment: int) -> int | None:
        slots = np.flatnonzero(self.env_slot_ids[row] == environment)
        return int(slots[0]) if slots.size else None

    def _row_has_environment(self, row: int, environment: int) -> bool:
        slot = self._slot_for_row(row, environment)
        return slot is not None and np.any(np.isfinite(self.even_placefield[row, slot]))

    def update_mouse(self, state):
        """Populate the environment selector, then constrain the session-number range."""
        rows = self._rows_by_mouse[state["mouse"]]
        candidates = {int(env) for env in self.env_slot_ids[rows].ravel() if np.isfinite(env) and env >= 0}
        environments = sorted(env for env in candidates if any(self._row_has_environment(int(row), env) for row in rows))
        if not environments:
            raise ValueError(f"No valid environments found for {state['mouse']}")
        value = state["environment"] if state["environment"] in environments else environments[0]
        self.update_selection("environment", value=value, options=environments)
        self.update_session_range(self.state)

    def update_session_range(self, state):
        """Move the integer-range bounds to the first/last session containing the environment."""
        rows = self._rows_by_mouse[state["mouse"]]
        session_numbers = [i + 1 for i, row in enumerate(rows) if self._row_has_environment(int(row), state["environment"])]
        if not session_numbers:
            return
        bounds = (session_numbers[0], session_numbers[-1])
        self.update_integer_range("session_range", value=bounds, min=bounds[0], max=bounds[1])
        self.update_integer("skip_session", max=max(0, len(session_numbers) - 1))
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Filter, normalize, and even-trial sort the maps of every session that will be drawn."""
        start, stop = state["session_range"]
        env = state["environment"]
        rows = self._rows_by_mouse[state["mouse"]]
        available = [
            (session_number, int(rows[session_number - 1]))
            for session_number in range(start, stop + 1)
            if self._row_has_environment(int(rows[session_number - 1]), env)
        ]
        shown = available[:: state["skip_session"] + 1]
        if not shown:
            raise ValueError("The selected range contains no sessions with this environment")

        self.panels: list[tuple[int, int, np.ndarray]] = []
        for session_number, row in shown:
            slot = self._slot_for_row(row, env)
            even = self.even_placefield[row, slot]
            odd = self.odd_placefield[row, slot]
            scale = self.spks_std[row]
            reliability = self.reliability[row, slot]
            fraction_active = self.fraction_active[row, slot]
            idx_keep = (
                np.isfinite(reliability)
                & np.isfinite(fraction_active)
                & np.isfinite(scale)
                & (scale > 0)
                & (reliability > state["reliability_threshold"])
                & (fraction_active > state["fraction_active_threshold"])
                & np.any(np.isfinite(even), axis=1)
                & np.any(np.isfinite(odd), axis=1)
            )
            even = even[idx_keep] / scale[idx_keep, None]
            odd = odd[idx_keep] / scale[idx_keep, None]
            # Sorted by where each ROI peaks on the *even* trials, then displayed on the odd ones.
            order = np.argsort(np.nanargmax(even, axis=1), kind="stable") if even.shape[0] else np.empty(0, dtype=int)
            self.panels.append((session_number, row, odd[order]))

    # -------------------------------------------------------------------- drawing --

    def _draw_scalebar(self, ax, state):
        """A track-distance scale bar, in axes fractions so it survives a changing panel count.

        The maps span the reference 200 cm track, so the requested distance converts to an axes
        fraction directly.
        """
        x0, y0 = state["scalebar_x"], state["scalebar_y"]
        bar_width = state["scalebar_cm"] / REFERENCE_ENV_LENGTH_CM
        ax.plot(
            [x0, x0 + bar_width],
            [y0, y0],
            transform=ax.transAxes,
            color="black",
            linewidth=1.5,
            solid_capstyle="butt",
            clip_on=False,
        )
        ax.text(
            x0 + bar_width / 2,
            y0,
            f"{state['scalebar_cm']:g}cm",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=state["scalebar_fontsize"],
        )

    def plot(self, state):
        fontsize = state["fontsize"]
        width_ratios = [state["colorbar_width_ratio"]] + [1.0] * len(self.panels)
        fig, axes = self.new_subplots(
            1,
            len(width_ratios),
            figsize=self.figsize,
            layout="constrained",
            squeeze=False,
            width_ratios=width_ratios,
        )
        placefield_axes = axes[0, 1:]

        draw_vertical_colorscale(
            axes[0, 0],
            "gray_r",
            low_label="0",
            high_label=f"{state['vmax']:g}",
            fontsize=fontsize,
            center_label=r"Activity ($\sigma$)",
        )

        for ax, (session_number, row, odd) in zip(placefield_axes, self.panels):
            if odd.shape[0]:
                ax.imshow(odd, aspect="auto", interpolation="none", cmap="gray_r", vmin=0, vmax=state["vmax"])
            else:
                ax.text(0.5, 0.5, "No ROIs pass\nthresholds", transform=ax.transAxes, ha="center", va="center", fontsize=fontsize)
            if state["verbose"]:
                session = self.results.sessions[row]
                ax.set_title(f"Session {session_number}\n{session.date}", fontsize=fontsize)
            ax.set_xlabel(f"{session_number}", fontsize=fontsize)
            hide_spines(ax)
            ax.tick_params(axis="both", labelsize=fontsize)

        placefield_axes[0].set_ylabel("ROIs (even-trial PF order)" if state["verbose"] else "ROIs", fontsize=fontsize)
        self._draw_scalebar(placefield_axes[0], state)
        fig.supxlabel("Session #", fontsize=fontsize)
        return fig
