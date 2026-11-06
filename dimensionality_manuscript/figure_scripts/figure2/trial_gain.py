"""A single real trial's place field and least-squares gain fit."""

from matplotlib.transforms import ScaledTranslation
import numpy as np

from dimensionality_manuscript.configs.placefield_structure import CrossValidatedPlacefieldsConfig
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts import session_cache
from dimensionality_manuscript.figure_scripts.panels import FigureViewer
from dimensionality_manuscript.pipeline import ResultsAggregator
from vrAnalysis.helpers.plotting import format_spines

from .model_style import ROLE_COLOR
from .placefield_gain import least_squares_gain_matrix


class TrialGainFocus(FigureViewer):
    """Average place field and one trial, beside the zero-intercept gain fit.

    Mouse/session/ROI/environment selection and cached maps follow Figure 1's
    ``PlaceFieldPredictionFocus``. ``roi`` uses session-filtered coordinates;
    ``env`` indexes ``env_maps.environments`` and ``trial`` indexes that
    environment's retained full trials (all zero-based). The template is the
    average over all those trials, including the selected trial. The scatter
    contains one point per finite position bin, with the unregularized
    least-squares slope ``sum(placefield * trial) / sum(placefield ** 2)``.

    ``show_gain_fit`` toggles the second (gain-fit scatter) panel without changing
    ``figsize``. The position panel spans 0-200 cm and uses a whole-number upper
    fluorescence limit rounded up from the maximum of both curves. A small
    margin below zero separates the curves from the bottom spine.

    In a notebook::

        viewer = TrialGainFocus(results, mouse="ATL012", roi=0, env=0, trial=0)
        viewer.show()
        fig = viewer.plot(viewer.state)
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        example_session: str | None = None,
        roi: int = 0,
        env: int = 0,
        trial: int = 0,
        show_gain_fit: bool = True,
        fontsize: float = 12.0,
        figsize: tuple[float, float] = (8.0, 3.0),
    ):
        self.results = results
        self.figsize = figsize
        self._example_config = CrossValidatedPlacefieldsConfig()
        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(name) for name in mouse_names})
        if not self.mice:
            raise ValueError("results must contain at least one session")
        self._sessions_by_mouse = {
            name: {
                results.sessions[row].session_print(): results.sessions[row]
                for row in sorted(
                    np.flatnonzero(mouse_names == name),
                    key=lambda row: _session_sort_key(results.sessions[row]),
                )
            }
            for name in self.mice
        }
        initial_mouse = mouse if mouse is not None else self.mice[0]
        labels = list(self._sessions_by_mouse[initial_mouse])
        initial_session = example_session if example_session is not None else labels[0]
        self.add_selection("mouse", value=initial_mouse, options=self.mice)
        self.add_selection("example_session", value=initial_session, options=labels)
        for name, value in (("roi", roi), ("env", env), ("trial", trial)):
            self.add_integer(name, value=value, min=0, max=max(value, 0))
        self.add_boolean("show_gain_fit", value=show_gain_fit)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.on_change("mouse", self.update_example_session)
        self.on_change(["example_session", "env"], self.update_example_bounds)
        self.on_change(["roi", "trial"], self.refresh_data)
        self.update_example_session(self.state)

    def update_example_session(self, state):
        """Offer this mouse's sessions in chronological order."""
        labels = list(self._sessions_by_mouse[state["mouse"]])
        current = state["example_session"] if state["example_session"] in labels else labels[0]
        self.update_selection("example_session", value=current, options=labels)
        self.update_example_bounds({**state, "example_session": current})

    def _get_env_maps(self, state):
        session = self._sessions_by_mouse[state["mouse"]][state["example_session"]]
        previous_spks_type = session.params.spks_type
        session.params.spks_type = self._example_config.spks_type
        try:
            return session_cache.get_env_maps(session)
        finally:
            session.params.spks_type = previous_spks_type

    def update_example_bounds(self, state):
        """Clamp selections when the session or environment has fewer examples."""
        maps = self._get_env_maps(state)
        if not len(maps.environments):
            raise ValueError("Selected session has no environment maps")
        env = min(state["env"], len(maps.environments) - 1)
        nroi, ntrial, _ = maps.spkmap[env].shape
        if not nroi or not ntrial:
            raise ValueError("Selected environment has no ROIs or full trials")
        self.update_integer("env", value=env, max=len(maps.environments) - 1)
        self.update_integer("roi", value=min(state["roi"], nroi - 1), max=nroi - 1)
        self.update_integer("trial", value=min(state["trial"], ntrial - 1), max=ntrial - 1)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        maps = self._get_env_maps(state)
        self.distcenters = maps.distcenters
        self.environment = maps.environments[state["env"]]
        self.spkmap = np.asarray(maps.spkmap[state["env"]][state["roi"]], dtype=float)
        self.placefield = np.nanmean(self.spkmap, axis=0)
        self.trial_activity = self.spkmap[state["trial"]]
        gains, _ = least_squares_gain_matrix(self.trial_activity[None, None, :], self.placefield[None, :])
        self.gain = float(gains[0, 0])

    def plot(self, state):
        fontsize = state["fontsize"]
        show_gain_fit = state["show_gain_fit"]
        fig, axes = self.new_subplots(
            1, 2 if show_gain_fit else 1, figsize=self.figsize, layout="constrained", squeeze=False,
        )
        ax_field = axes[0, 0]
        color = ROLE_COLOR["internal"]
        trial_label = f"Single Trial\nGain={self.gain:.2f}"
        ax_field.plot(self.distcenters, self.placefield, color="black", lw=1.5, label="Avg PF")
        ax_field.plot(self.distcenters, self.trial_activity, color=color, lw=1.5, label=trial_label)
        ax_field.set_xlabel("Position (cm)", fontsize=fontsize)
        ax_field.set_ylabel("Norm. Fluor.", fontsize=fontsize, labelpad=-2)
        xbounds = (0, 200)
        finite_amplitudes = np.concatenate([self.placefield, self.trial_activity])
        finite_amplitudes = finite_amplitudes[np.isfinite(finite_amplitudes)]
        curve_max = float(np.max(finite_amplitudes)) if finite_amplitudes.size else 0.0
        ymax = max(1, int(np.ceil(curve_max)))
        ax_field.set_xlim(xbounds)
        ax_field.set_ylim(-.05 * ymax, ymax)
        label = ax_field.text(
            .02, .98, "Avg PF", transform=ax_field.transAxes,
            ha="left", va="top", color="black", fontsize=fontsize,
        )
        ax_field.annotate(
            trial_label, xy=(0, 0), xycoords=label, xytext=(0, -2), textcoords="offset points",
            ha="left", va="top", color=color, fontsize=fontsize,
        )
        format_spines(
            ax_field, x_pos=-.02, y_pos=0, spines_visible=["left", "bottom"],
            xbounds=xbounds, xticks=xbounds, ybounds=(0, ymax), yticks=[0, ymax],
            tick_length=2, tick_fontsize=fontsize,
        )
        ax_field.xaxis.set_label_coords(
            .5, 0,
            transform=ax_field.transAxes + ScaledTranslation(0, -(fontsize + 8) / 72, fig.dpi_scale_trans),
        )
        ax_field.xaxis.label.set_verticalalignment("top")
        ax_field.yaxis.set_label_coords(
            0, .5,
            transform=ax_field.transAxes + ScaledTranslation(-(fontsize + 7) / 72, 0, fig.dpi_scale_trans),
        )
        ax_field.yaxis.label.set_verticalalignment("center")

        if show_gain_fit:
            ax_fit = axes[0, 1]
            valid = np.isfinite(self.placefield) & np.isfinite(self.trial_activity)
            x, y = self.placefield[valid], self.trial_activity[valid]
            ax_fit.scatter(x, y, color="black", s=10, edgecolors="none")
            if np.isfinite(self.gain):
                line_x = np.array([min(0.0, float(x.min())), max(0.0, float(x.max()))])
                ax_fit.plot(line_x, self.gain * line_x, color=color, lw=1.5)
                label = f"Gain (slope) = {self.gain:.2f}"
            else:
                label = "Gain undefined (zero PF energy)"
            ax_fit.text(.04, .96, label, transform=ax_fit.transAxes, va="top", fontsize=fontsize)
            ax_fit.set_xlabel(r"Average place field ($\sigma$)", fontsize=fontsize)
            ax_fit.set_ylabel(r"Trial activity ($\sigma$)", fontsize=fontsize)
            format_spines(
                ax_fit, x_pos=-.02, y_pos=-.02, spines_visible=["left", "bottom"],
                tick_length=2, tick_fontsize=fontsize,
            )
        return fig
