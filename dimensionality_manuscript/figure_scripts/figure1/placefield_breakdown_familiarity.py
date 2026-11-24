"""Temporal PF/gain/residual variance fractions in the holistic familiarity layout."""

from collections.abc import Mapping

from dimensionality_manuscript.configs.pfpred_quality import PF_VARIANCE_COMPARISONS
from dimensionality_manuscript.pipeline import ResultsAggregator

from .holistic_placefield_familiarity import HolisticPlacefieldFamiliarityViewer
from .timeline import ExperimentTimeline


class PlacefieldBreakdownFamiliarityViewer(HolisticPlacefieldFamiliarityViewer):
    """Timeline and PF, gain, and residual components in three session coordinates.

    Uses the same schematic, single-mouse chronological/relative columns, and equal-mouse
    relative-time column as HolisticPlacefieldFamiliarityViewer. Quality selection, mean
    or median cell summaries, mouse support, error bands, and optional first-environment
    baseline normalization have the same behavior.

    Each component is its temporal variance divided by data variance, using the per-ROI
    ``pf_frac_var_{template,gain,additive}_<fold>_slot`` arrays from schema v12.
    ``fold`` selects all_from_all, f1_from_f1, f2_from_f1, f1_from_f2, f2_from_f2,
    cv, or within_cv. The default ``cv`` is the arithmetic per-cell average of
    f2_from_f1 and f1_from_f2; both directions must be finite. ``within_cv`` averages
    f1_from_f1 and f2_from_f2. Fold averaging precedes the across-cell mean/median.
    Only PF templates are held out: trial gains are fitted on evaluation frames.

    Component rows default to [0, 1] display limits. Set ``ylim_full_components=False``
    to autoscale and see fractions above one. Relative baseline ratios always autoscale.

    Example::

        viewer = PlacefieldBreakdownFamiliarityViewer(
            results, schematic_config=timeline, cells="split", fold="cv",
        )
        viewer.show()
    """

    _METRICS = ("template", "gain", "additive")
    _METRIC_LABELS = {"template": "PF Component", "gain": "Gain Component", "additive": "Residual Component"}
    _RELOAD_CONTROLS = ("fold",)
    _SUMMARY_CONTROLS = ()

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        schematic_config: ExperimentTimeline | Mapping | None = None,
        mouse: str | None = None,
        fold: str = "cv",
        cells: str = "all",
        reliability_threshold: float = 0.7,
        fraction_active_threshold: float = 0.2,
        summary_stat: str = "mean",
        plot_style: str = "errorPlot",
        hide_error: bool = False,
        show_legend: bool = True,
        relative_values: bool = False,
        baseline_sessions: int = 1,
        min_mice: int = 2,
        ylim_full_components: bool = True,
        fontsize: float = 9.0,
        schema_height_ratio: float = 0.8,
        figsize: tuple[float, float] = (12.0, 10.0),
        **selection_defaults,
    ):
        self._breakdown_defaults = dict(fold=fold, ylim_full_components=ylim_full_components)
        super().__init__(
            results, schematic_config=schematic_config, mouse=mouse, cells=cells,
            reliability_threshold=reliability_threshold, fraction_active_threshold=fraction_active_threshold,
            summary_stat=summary_stat, plot_style=plot_style, hide_error=hide_error,
            show_legend=show_legend, relative_values=relative_values, baseline_sessions=baseline_sessions,
            min_mice=min_mice, fontsize=fontsize, schema_height_ratio=schema_height_ratio,
            figsize=figsize, **selection_defaults,
        )

    def _add_metric_widgets(self, defaults):
        self.add_selection("fold", value=self._breakdown_defaults["fold"], options=list(PF_VARIANCE_COMPARISONS))
        self.add_boolean("ylim_full_components", value=self._breakdown_defaults["ylim_full_components"])

    def reload_arrays(self, state):
        self._load_metric_arrays(state, {
            metric: f"pf_frac_var_{metric}_{state['fold']}_slot" for metric in self._METRICS
        }, schema_version="v12")

    def _summary_values(self, state):
        return self.arrays

    def _visible_metrics(self, state):
        return self._METRICS

    def _set_metric_ylim(self, ax, metric, state):
        if state["ylim_full_components"] and not state["relative_values"]:
            ax.set_ylim(0.0, 1.0)
