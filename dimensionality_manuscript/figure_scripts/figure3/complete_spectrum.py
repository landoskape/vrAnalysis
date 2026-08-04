"""Composite figure: the ratios spectrum and beeswarms beside both familiarity ratio panels."""

import pandas as pd

from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, apply_legend, update_legend_widgets
from dimensionality_manuscript.figure_scripts.panels import FigureViewer

from ._familiarity import ENV_FULL_SCOPES, FAMILIARITY_STYLES, familiarity_curves, render_familiarity_ratio_panel
from ._ratios import COMPOSITE_SPINE_OFFSET, plot_ratios_beeswarms_combined, plot_ratios_spectrum, ratios_arrays
from ._slopes import ENV_SLOPE_STYLES, env_slope_stats, env_slope_table, format_env_slope_stats, plot_env_slopes
from ._selection import add_stimspace_selection_widgets, stimspace_selection

# Legend widget prefix -> the settings its panel helper already draws with, so the widgets start
# out reproducing the current figure rather than matplotlib's (or LEGEND_KNOBS') defaults. ``loc``
# stays "auto" and is supplied per panel as ``auto_loc`` at draw time.
LEGEND_DEFAULTS = {
    "spectrum_legend": dict(fontsize_scale=1.0),
    "all_legend": dict(fontsize_scale=1.0, handlelength=0.8, handletextpad=0.5),
    "env_legend": dict(fontsize_scale=1.0, handlelength=0.8, handletextpad=0.5),
}
# Placement each panel falls back to at ``{prefix}_loc="auto"``, matching its helper's own call.
LEGEND_AUTO_LOCS = {"spectrum_legend": "upper right", "all_legend": "upper left", "env_legend": "upper left"}


def _width_ratios(ax1_width_ratio: float, ax23_width_ratio: float, show_slope_panel: bool) -> tuple[float, ...]:
    """Width ratios for the composite figure's 1xN grid: ``[1, ax1, ax23, ax23]``, plus 1 for ax[4].

    ax[2] and ax[3] share one knob because they're the same kind of panel (a familiarity curve
    over sessions) and reading them against each other depends on their x axes matching.
    """
    return (1.0, ax1_width_ratio, ax23_width_ratio, ax23_width_ratio) + ((1.0,) if show_slope_panel else ())


class CompleteSpectrumViewer(FigureViewer):
    """Composite figure: ratios spectrum+beeswarm alongside both familiarity Variance Ratio panels.

    A plain 1x5 ``subplots``, or 1x4 when ``show_slope_panel`` is off. ``ax[0]`` is the
    PF-structure-vs-reliable-CA1 spectrum panel from
    :class:`~.ratios.SubspaceCurvesRatiosViewer`. ``ax[1]`` combines that same figure's two
    cumulative-variance-ratio beeswarm groups onto one axis (1st 10 dims, then all dims, same
    Behaving/w ITIs/w Spont color order in each), separated by spacing rather than by labels, with
    a segmented x spine drawing one bracket per group. Only the group centers get tick labels
    ("1st 10" / "All") -- the per-color condition labels aren't repeated here since ``ax[2]``'s
    legend already maps these same colors to Behaving/w ITIs/w Spont. ``ax[2]`` is the
    whole-session familiarity Variance Ratio panel (``plot_mode="all"``; Total Variance omitted).
    ``ax[3]`` is the per-env-experience-slot familiarity Variance Ratio panel
    (``plot_mode="by_env"``). ``ax[4]`` collapses each of ``ax[3]``'s curves to its slope, one
    point per mouse per env slot, and tests them with ``slope ~ env + (1 | mouse)`` -- so it lives
    on a slope scale and keeps its own y-axis. ``ax[1]``, ``ax[2]`` and ``ax[3]`` share one y-axis,
    fixed to ``[0, 1]`` and labeled "Shared Variance Ratio" once, on the beeswarm panel; the other
    two hide their y spine/ticks/label. All panels share one set of ``StimSpaceSpectraConfig``
    param-axis selections, since every source figure reads from the same aggregated results.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``StimSpaceSpectraConfig`` results, shared by every panel.
    within_condition : bool
        Only affects ``ax[2]``. If True, each curve's x-axis is the session index within its own
        kept subset; if False, bins track the mouse's overall chronological session index.
    style_all : {"errorPlot", "all"}
        Rendering style for ``ax[2]``. ``"errorPlot"`` shows the mouse mean +/- SE band; ``"all"``
        also shows every mouse's faint curve.
    env_full_scope : {"within_env", "outside_env", "with_iti", "with_spontaneous"}
        Only affects ``ax[3]``. Selects which per-env key pairing / ITI condition to plot (see
        :class:`~.familiarity.SubspaceFamiliarityViewer`).
    full_within_env : bool
        Only affects ``ax[3]``. If True, the total-variance denominator is the env-only variance;
        if False, it is the whole-session variance. (Affects the ratio's normalization even though
        the Total Variance panel itself is omitted.)
    style_by_env : {"errorPlot", "all"}
        Rendering style for ``ax[3]``, same options as ``style_all``.
    within_xspace : float
        Only affects ``ax[1]``. Spacing (in that panel's x units) between the three conditions of
        a beeswarm group. The swarm spread and mean-line width scale with it.
    between_xspace : float
        Only affects ``ax[1]``. Extra spacing added at the group boundary, on top of the
        ``within_xspace`` step.
    ax1_width_ratio : float
        Width of ``ax[1]`` relative to ``ax[0]`` and ``ax[4]``, which are fixed at 1.
    ax23_width_ratio : float
        Width of ``ax[2]`` and ``ax[3]`` on that same scale. They share one knob because they are
        the same kind of panel and are meant to be read against each other.
    show_slope_panel : bool
        Include ``ax[4]``. When False the figure is a 1x4 grid and none of the ``slope_*``
        arguments apply.
    slope_style : {"beeswarm", "all", "errorPlot"}
        Only affects ``ax[4]``. ``"beeswarm"`` swarms the per-mouse slopes at each env tick with a
        mean line; ``"all"`` draws one faint line per mouse across the env slots plus the mean;
        ``"errorPlot"`` shows only the mean +/- SE band.
    min_slope_sessions : int
        Only affects ``ax[4]``. Minimum finite sessions a mouse needs in an env slot to contribute
        a slope there (and so to enter the mixed-effects model). Floored at 3, since two points
        always fit a slope exactly and so carry no evidence of a trend.
    show_slope_stats : bool
        Only affects ``ax[4]``. Annotate the panel with the omnibus, trend and pairwise p-values.
    spectrum_legend, all_legend, env_legend : dict or None
        Legend styling for ``ax[0]``, ``ax[2]`` and ``ax[3]`` respectively, as ``{knob: value}``.
        ``"loc"`` is a matplotlib placement, ``"auto"`` (the panel's own default placement) or
        ``"none"`` (draw no legend); the rest are :meth:`~matplotlib.axes.Axes.legend` kwargs --
        see :data:`~dimensionality_manuscript.figure_scripts.legends.LEGEND_KNOBS`. Only the keys
        given are overridden.
    fontsize : float
        Font size (points) shared by every panel: axis labels, tick labels, legends and inline
        annotations.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the data-selection widgets (``activity_parameters_name``,
        ``smooth_widths``, ``reliability_fraction_active_thresholds``), shared by every panel.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        within_condition: bool = False,
        style_all: str = "errorPlot",
        env_full_scope: str = "within_env",
        full_within_env: bool = False,
        style_by_env: str = "errorPlot",
        within_xspace: float = 0.8,
        between_xspace: float = 1.6,
        ax1_width_ratio: float = 1.2,
        ax23_width_ratio: float = 1.0,
        show_slope_panel: bool = True,
        slope_style: str = "beeswarm",
        min_slope_sessions: int = 4,
        show_slope_stats: bool = True,
        spectrum_legend: dict | None = None,
        all_legend: dict | None = None,
        env_legend: dict | None = None,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (17.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize

        self.selection_names, self._tuple_labels = add_stimspace_selection_widgets(self, results, selection_defaults)

        # ax[2]: familiarity plot_mode="all" knobs (env_full_scope/full_within_env don't apply).
        self.add_boolean("within_condition", value=within_condition)
        self.add_selection("style_all", value=style_all, options=FAMILIARITY_STYLES)

        # ax[3]: familiarity plot_mode="by_env" knobs (within_condition doesn't apply).
        self.add_selection("env_full_scope", value=env_full_scope, options=ENV_FULL_SCOPES)
        self.add_boolean("full_within_env", value=full_within_env)
        self.add_selection("style_by_env", value=style_by_env, options=FAMILIARITY_STYLES)

        # ax[1]: beeswarm group layout. The two spacings are in the panel's own x units and are
        # built up from 0, so only their ratio matters for the look -- widening between_xspace
        # pulls the groups apart, shrinking within_xspace tightens each group (and its swarms).
        self.add_float("within_xspace", value=within_xspace, min=0.1, max=3.0, step=0.05)
        self.add_float("between_xspace", value=between_xspace, min=0.0, max=6.0, step=0.05)
        self.add_float("ax1_width_ratio", value=ax1_width_ratio, min=0.2, max=4.0, step=0.05)
        self.add_float("ax23_width_ratio", value=ax23_width_ratio, min=0.2, max=4.0, step=0.05)

        # ax[4]: per-env slopes of ax[3]'s curves, dropped entirely when show_slope_panel is off.
        self.add_boolean("show_slope_panel", value=show_slope_panel)
        self.add_selection("slope_style", value=slope_style, options=ENV_SLOPE_STYLES)
        self.add_integer("min_slope_sessions", value=min_slope_sessions, min=3, max=20)
        self.add_boolean("show_slope_stats", value=show_slope_stats)

        # One independent legend control set per panel that draws a legend. Each is seeded to the
        # placement and spacing its panel helper already used, so the knobs start as a no-op: the
        # "auto" loc defers to the helper's own placement (see the auto_loc passes in plot).
        overrides = dict(spectrum_legend=spectrum_legend, all_legend=all_legend, env_legend=env_legend)
        for prefix, defaults in LEGEND_DEFAULTS.items():
            add_legend_widgets(self, prefix=prefix)
            update_legend_widgets(self, {**defaults, **(overrides[prefix] or {})}, prefix=prefix)

        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0, step=0.5)

        for name in (
            *self.selection_names,
            "within_condition",
            "env_full_scope",
            "full_within_env",
            "show_slope_panel",
            "show_slope_stats",
            "min_slope_sessions",
        ):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _sel_params(self, state: dict) -> dict:
        """Params to forward to ``results.sel``, decoding tuple labels back to tuples."""
        return stimspace_selection(state, self.selection_names, self._tuple_labels)

    def _by_env_curves(self, state: dict) -> dict:
        """``mode="by_env"`` curves for the current state, shared by ax[3] and the ax[4] slopes."""
        return familiarity_curves(
            self.results,
            self._sel_params(state),
            "by_env",
            env_full_scope=state["env_full_scope"],
            full_within_env=state["full_within_env"],
        )

    def refresh_data(self, state):
        """Rebuild every panel's arrays: spectra, both familiarity curve sets, and the slopes."""
        sel_params = self._sel_params(state)
        self._ratios_arrays = ratios_arrays(self.results, sel_params)
        self._curves_all = familiarity_curves(self.results, sel_params, "all", within_condition=state["within_condition"])
        self._curves_by_env = self._by_env_curves(state)
        self._slope_table = env_slope_table(self._curves_by_env, state["min_slope_sessions"])
        # Fitting three mixed models is the expensive part, so it is skipped whenever the panel
        # that would show the numbers isn't drawn.
        show_stats = state["show_slope_panel"] and state["show_slope_stats"]
        self._slope_stats_text = format_env_slope_stats(env_slope_stats(self._slope_table), list(self._curves_by_env)) if show_stats else None

    def slope_table_and_stats(self, state: dict | None = None) -> tuple[pd.DataFrame, dict | None]:
        """The ax[4] per-mouse slope table and its mixed-effects tests, for reporting numbers.

        The panel itself only has room for the headline p-values; this returns everything behind
        them (see :func:`~._slopes.env_slope_stats`) for the current viewer state.
        """
        state = self.state if state is None else state
        table = env_slope_table(self._by_env_curves(state), state["min_slope_sessions"])
        return table, env_slope_stats(table)

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        show_slope_panel = state["show_slope_panel"]
        width_ratios = _width_ratios(state["ax1_width_ratio"], state["ax23_width_ratio"], show_slope_panel)
        fig, ax = self.new_subplots(1, len(width_ratios), figsize=self.figsize, layout="constrained", width_ratios=width_ratios)

        def _legend(axis, prefix: str) -> None:
            """Restyle a panel's already-drawn legend under its own ``{prefix}_*`` widgets."""
            apply_legend(axis, state, fontsize, prefix=prefix, auto_loc=LEGEND_AUTO_LOCS[prefix])

        plot_ratios_spectrum(ax[0], self._ratios_arrays, fontsize)
        _legend(ax[0], "spectrum_legend")
        plot_ratios_beeswarms_combined(
            ax[1],
            self._ratios_arrays,
            fontsize,
            within_xspace=state["within_xspace"],
            between_xspace=state["between_xspace"],
        )

        ax[2].sharey(ax[1])
        render_familiarity_ratio_panel(ax[2], self._curves_all, "all", state["style_all"], fontsize)
        _legend(ax[2], "all_legend")

        ax[3].sharey(ax[1])
        render_familiarity_ratio_panel(ax[3], self._curves_by_env, "by_env", state["style_by_env"], fontsize)
        _legend(ax[3], "env_legend")

        # ax[4] summarizes ax[3]: one slope per mouse per env slot, so it keeps its own y axis.
        if show_slope_panel:
            plot_env_slopes(
                ax[4],
                self._slope_table,
                list(self._curves_by_env),
                state["slope_style"],
                fontsize,
                self._slope_stats_text,
            )

        # ax[1]-ax[3] share one y-axis: fixed [0, 1] range, spine/ticks/label shown once (on the
        # beeswarm panel), the other two hidden. Setting ylim on any one member of a sharey group
        # re-syncs the whole group, so this override wins regardless of whatever each panel's own
        # helper set internally.
        ax[1].set_ylim(0, 1)
        ax[1].set_ylabel("Shared Variance Ratio", fontsize=fontsize)
        for a in (ax[2], ax[3]):
            a.set_ylabel("")
            a.spines["left"].set_visible(False)
            a.tick_params(axis="y", left=False, labelleft=False)

        # Each panel's helper already ran format_spines, which bakes its y_pos fraction into a data
        # coordinate using the ylim at that moment -- and both of those ylims differed from the final
        # (0, 1). Re-pin the bottom spines here, where the fraction and the data coordinate coincide,
        # so every panel's x spine lines up: ax[0]'s continuous spine, ax[1]'s hand-drawn segments
        # and these two all sit at the same fractional offset.
        for a in (ax[2], ax[3]):
            a.spines["bottom"].set_position(("data", COMPOSITE_SPINE_OFFSET))

        return fig
