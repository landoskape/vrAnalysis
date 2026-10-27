"""Standalone Head/Full placefield-component curves over environment familiarity."""

from matplotlib.ticker import MaxNLocator

from dimensionality_manuscript.figure_scripts.panels import FigureViewer
from dimensionality_manuscript.pipeline import ResultsAggregator
from vrAnalysis.helpers.plotting import format_spines

from ._familiarity import ENV_FULL_SCOPES, FAMILIARITY_STYLES, familiarity_curves, familiarity_panel
from ._selection import add_stimspace_selection_widgets, stimspace_selection


class PlacefieldComponentFamiliarityViewer(FigureViewer):
    """The environment-familiarity panels from :class:`ConciseSpectrumViewer`.

    ``component="both"`` overlays First N and Full on one axis with text labels.
    ``"head"`` or ``"full"`` draws one component without a text label.
    ``head_text_xy`` and ``full_text_xy`` position the labels independently in
    axes-fraction coordinates, with separate x/y widgets for each label.
    Head sums the first ``first10`` dimensions of both paired spectra; Full sums
    all dimensions. Curves, mouse averaging, and SE bands match the composite.
    The compact defaults hide the y axis and legend, as in the manuscript panel.
    ``figsize`` is the size of the entire figure in inches in every mode.
    Data-selection kwargs use the usual StimSpaceSpectra defaults and widgets.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        component: str = "both",
        first10: int = 10,
        env_full_scope: str = "within_env",
        full_within_env: bool = False,
        style_by_env: str = "errorPlot",
        as_percent: bool = False,
        shared_variance_ratio_ylims: tuple[float, float] = (0.0, 1.0),
        show_yaxis: bool = False,
        show_legend: bool = False,
        head_text_xy: tuple[float, float] = (0.05, 0.95),
        full_text_xy: tuple[float, float] = (0.05, 0.45),
        fontsize: float = 7.0,
        figsize: tuple[float, float] = (1.6, 2.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.selection_names, self._tuple_labels = add_stimspace_selection_widgets(self, results, selection_defaults)
        self.add_selection("component", value=component, options=("full", "head", "both"))
        self.add_integer("first10", value=first10, min=1, max=30)
        self.add_selection("env_full_scope", value=env_full_scope, options=ENV_FULL_SCOPES)
        self.add_boolean("full_within_env", value=full_within_env)
        self.add_selection("style_by_env", value=style_by_env, options=FAMILIARITY_STYLES)
        self.add_boolean("as_percent", value=as_percent)
        self.add_float_range("shared_variance_ratio_ylims", value=shared_variance_ratio_ylims, min=0.0, max=1.0, step=0.01)
        self.add_boolean("show_yaxis", value=show_yaxis)
        self.add_boolean("show_legend", value=show_legend)
        for prefix, xy in (("head_text", head_text_xy), ("full_text", full_text_xy)):
            self.add_float(f"{prefix}_x", value=xy[0], min=0.0, max=1.0, step=0.01)
            self.add_float(f"{prefix}_y", value=xy[1], min=0.0, max=1.0, step=0.01)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0, step=0.5)
        for name in (*self.selection_names, "component", "first10", "env_full_scope", "full_within_env"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        selection = stimspace_selection(state, self.selection_names, self._tuple_labels)
        modes = ("head", "full") if state["component"] == "both" else (state["component"],)
        self._curves = {
            mode: familiarity_curves(
                self.results,
                selection,
                "by_env",
                env_full_scope=state["env_full_scope"],
                full_within_env=state["full_within_env"],
                max_dims=state["first10"] if mode == "head" else None,
            )
            for mode in modes
        }

    def plot(self, state):
        modes = ("head", "full") if state["component"] == "both" else (state["component"],)
        fig, ax = self.new_subplots(figsize=self.figsize, layout="constrained")
        fontsize = state["fontsize"]
        scale = 100.0 if state["as_percent"] else 1.0
        ylabel = "Placefield Component (%)" if state["as_percent"] else "Placefield Component"
        for mode in modes:
            familiarity_panel(
                ax,
                self._curves[mode],
                "svr",
                "Env Session #",
                ylabel if state["show_yaxis"] else "",
                state["style_by_env"],
                fontsize,
                value_scale=scale,
            )
            if len(modes) == 2:
                ax.text(
                    state[f"{mode}_text_x"],
                    state[f"{mode}_text_y"],
                    f"First {state['first10']}" if mode == "head" else "Full",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=fontsize,
                )
        ax.set_ylim(*(limit * scale for limit in state["shared_variance_ratio_ylims"]))
        spines = (["left"] if state["show_yaxis"] else []) + ["bottom"]
        format_spines(ax, x_pos=-0.02, y_pos=-0.02, spines_visible=spines, tick_fontsize=fontsize)
        ax.tick_params(axis="y", left=state["show_yaxis"], labelleft=state["show_yaxis"])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines["bottom"].set_bounds(0, ax.get_xlim()[1])
        if state["show_legend"]:
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="upper left", fontsize=fontsize, frameon=False, handlelength=0.8, handletextpad=0.5)
        return fig
