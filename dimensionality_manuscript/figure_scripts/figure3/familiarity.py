"""Variance ratio and total variance over familiarity, whole-session or per env slot."""

from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.figure_scripts.panels import FigureViewer

from ._familiarity import ENV_FULL_SCOPES, FAMILIARITY_STYLES, familiarity_curves, render_familiarity_panels
from ._selection import add_stimspace_selection_widgets, stimspace_selection

FAMILIARITY_MODES = ["all", "by_env"]


class SubspaceFamiliarityViewer(FigureViewer):
    """Variance-ratio-over-familiarity figure: whole-session or per-env-experience-slot curves.

    Shows, per mouse and averaged across mice, the fraction of total activity variance shared with
    the stimulus (placefield) subspace across sessions ("Variance Ratio") and the total activity
    variance itself ("Total Variance"), as a 1x2 grid. ``plot_mode="all"`` uses the whole-session
    spectra (``sf_cv`` / ``ff``), x-axis = overall session number; ``plot_mode="by_env"`` uses the
    per-environment-experience-slot spectra, overlaying all ``MAX_ENV_SLOTS`` experience-order
    slots together, x-axis = session number within that env.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``StimSpaceSpectraConfig`` results providing ``param_axes``, ``sel``,
        ``unique_mice``, ``mouse_names`` and ``sessions``.
    plot_mode : {"all", "by_env"}
        Whole-session curves, or per-environment-experience-slot curves.
    env_full_scope : {"within_env", "outside_env", "with_iti", "with_spontaneous"}
        Only used when ``plot_mode="by_env"``. Selects which per-env key pairing / ITI condition
        to plot: ``"within_env"`` normalizes by the env-restricted total variance
        (``sf_cv_env_full1`` / ``ff_env_full1``; ``include_iti`` has no effect on this scope, since
        its func side is always env-only VR frames). The other three all use ``sf_cv_env_fullall``
        / ``ff_env_full1_fullall`` (env-stim vs all-env-func): ``"outside_env"`` is the
        behaving-only condition, ``"with_iti"`` includes ITIs restricted to non-spontaneous
        sessions, and ``"with_spontaneous"`` includes ITIs restricted to sessions with a genuine
        spontaneous window.
    full_within_env : bool
        Only used when ``plot_mode="by_env"``. If True, the total-variance denominator is the
        env-only variance (``ff_env_full1``); if False, it is the whole-session variance (``ff``).
    within_condition : bool
        Only used when ``plot_mode="all"``. If True, each curve's x-axis is the session index
        within its own kept subset (a mouse's first spontaneous session is bin 0, regardless of
        its overall session number). If False, bins track the mouse's overall chronological
        session index instead.
    style : {"errorPlot", "all"}
        ``"all"`` plots every mouse's curve as a faint line plus the mouse-mean as a bold line;
        ``"errorPlot"`` drops the per-mouse lines and shows the mouse mean +/- SE as a band.
    fontsize : float
        Font size for axis labels, tick labels and the legend.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the data-selection widgets (``activity_parameters_name``,
        ``smooth_widths``, ``reliability_fraction_active_thresholds``). Tuple-valued axes accept
        native tuples; they are encoded to the widget's string labels internally. ``include_iti``
        is not selectable -- it is set per curve by the Behaving/ITI/Spontaneous split and by
        ``env_full_scope``.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        plot_mode: str = "all",
        env_full_scope: str = "within_env",
        full_within_env: bool = True,
        within_condition: bool = True,
        style: str = "errorPlot",
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (6.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize

        self.selection_names, self._tuple_labels = add_stimspace_selection_widgets(self, results, selection_defaults)

        self.add_selection("plot_mode", value=plot_mode, options=FAMILIARITY_MODES)
        self.add_selection("env_full_scope", value=env_full_scope, options=ENV_FULL_SCOPES)
        self.add_boolean("full_within_env", value=full_within_env)
        self.add_boolean("within_condition", value=within_condition)
        self.add_selection("style", value=style, options=FAMILIARITY_STYLES)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0, step=0.5)

        for name in (*self.selection_names, "plot_mode", "env_full_scope", "full_within_env", "within_condition"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Rebuild the per-mouse variance-ratio / total-variance curves for the current selection."""
        self._curves = familiarity_curves(
            self.results,
            stimspace_selection(state, self.selection_names, self._tuple_labels),
            state["plot_mode"],
            env_full_scope=state["env_full_scope"],
            full_within_env=state["full_within_env"],
            within_condition=state["within_condition"],
        )

    def plot(self, state: dict):
        fig, ax = self.new_subplots(1, 2, figsize=self.figsize, layout="constrained")
        render_familiarity_panels(ax[0], ax[1], self._curves, state["plot_mode"], state["style"], state["fontsize"])
        return fig
