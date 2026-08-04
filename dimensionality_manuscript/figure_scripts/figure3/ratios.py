"""Placefield-component vs. reliable-CA1 variance spectra, plus cumulative-variance ratios."""

from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.figure_scripts.panels import FigureViewer

from ._ratios import plot_ratios_beeswarms, plot_ratios_spectrum, ratios_arrays
from ._selection import add_stimspace_selection_widgets, stimspace_selection


class SubspaceCurvesRatiosViewer(FigureViewer):
    """Placefield-component vs. reliable-CA1 variance spectra, plus cumulative-variance ratios.

    Left panel: mouse-averaged normalized ``sf_cv`` (placefield component) and ``ff`` (reliable
    CA1) spectra vs. shared dimension (log-log), each mouse faint plus the cross-mouse mean bold.
    Right panels: beeswarm of the fraction of total variance captured in the first 10 dims and in
    all dims, split by ``Behaving`` (every session), ``w/ ITIs`` (non-spontaneous sessions with
    ITIs included) and ``w/ Spont`` (sessions with a genuine spontaneous window, ITIs included).

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``StimSpaceSpectraConfig`` results providing ``param_axes``, ``sel``,
        ``mouse_names`` and ``sessions``.
    fontsize : float
        Font size for axis labels, tick labels, the legend and the inline "1st 10" / "All"
        annotations.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the data-selection widgets (``activity_parameters_name``,
        ``smooth_widths``, ``reliability_fraction_active_thresholds``). Tuple-valued axes accept
        native tuples; they are encoded to the widget's string labels internally.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (5.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize

        self.selection_names, self._tuple_labels = add_stimspace_selection_widgets(self, results, selection_defaults)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0, step=0.5)

        for name in self.selection_names:
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select the spectra and cumulative-variance-ratio arrays."""
        self._arrays = ratios_arrays(self.results, stimspace_selection(state, self.selection_names, self._tuple_labels))

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        fig, ax = self.new_subplots(1, 3, figsize=self.figsize, layout="constrained", width_ratios=[1, 0.3, 0.3])
        fig.get_layout_engine().set(wspace=0.02, w_pad=0.00)
        ax[2].sharex(ax[1])
        ax[2].sharey(ax[1])

        plot_ratios_spectrum(ax[0], self._arrays, fontsize)
        plot_ratios_beeswarms(ax[1], ax[2], self._arrays, fontsize)
        return fig
