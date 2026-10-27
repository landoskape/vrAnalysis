"""Environment familiarity of CA1/PF dimensionality and the placefield component."""

import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter

from dimensionality_manuscript.env_order import ENV_SLOT_COLORS
from dimensionality_manuscript.figure_scripts.figure4._spectrum_math import (
    _clip_at_first_negative,
    _signed_participation_ratio,
    _smooth_spectrum,
)
from dimensionality_manuscript.figure_scripts.figure4.familiarity import DimensionalityFamiliarityViewer
from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, apply_legend, update_legend_widgets
from dimensionality_manuscript.figure_scripts.panels import FigureViewer
from dimensionality_manuscript.pipeline import ResultsAggregator
from vrAnalysis.helpers.plotting import format_spines

from ._curves import chronological_mouse_sessions
from ._familiarity import ENV_FULL_SCOPES, select_by_env_spectra
from ._selection import add_stimspace_selection_widgets, stimspace_selection


class DimensionalityComponentFamiliarityViewer(FigureViewer):
    """Three shared-x rows: Full CA1 PR, PF-fraction PR, and placefield component.

    Uses the same environment-specific ``sf_cv``/``ff`` selection as the full-spectrum
    familiarity panel in ConciseSpectrumViewer (never Figure 4's SVD Full estimator).
    ``env_full_scope`` and ``full_within_env`` have the same meaning as in that viewer.
    PR is calculated per session using Figure 4's signed participation ratio. Optional
    clipping and rank smoothing affect PR only; the component always uses the raw
    full-spectrum sum(sf_cv) / sum(ff).

    Each environment's sessions are sorted within mouse, then numbered from zero.
    All rows share the raw paired-spectrum coverage mask, so an undefined PR leaves
    a gap without shifting later sessions. Population means/SE require two mice.
    ``display="each"`` additionally draws individual mice. Data-selection kwargs seed
    the usual StimSpaceSpectra widgets, including tuple-valued smoothing/thresholds.
    Dimensionality defaults to log scaling with decade-only labels, minor ticks, and limits rounded
    out to enclosing decades. ``sharey=True`` shares the top two panels' y-axis.
    ``legend_panel`` selects the environment legend's row (0=top, 1=middle, 2=bottom).
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        env_full_scope: str = "within_env",
        full_within_env: bool = False,
        display: str = "errorPlot",
        pr_pre_smooth: bool = True,
        pf_smooth_method: str = "none",
        pf_smooth_width: float = 3.0,
        full_smooth_method: str = "none",
        full_smooth_width: float = 3.0,
        clip_negative: bool = False,
        log_y: bool = True,
        sharey: bool = False,
        as_percent: bool = False,
        shared_variance_ratio_ylims: tuple[float, float] = (0.0, 1.0),
        env_legend: dict | None = None,
        legend_panel: int = 1,
        height_ratios: tuple[float, float, float] = (1.0, 1.0, 1.0),
        figsize: tuple[float, float] = (3.5, 5.0),
        fontsize: float = 7.0,
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        if len(height_ratios) != 3 or not all(np.isfinite(value) and value > 0 for value in height_ratios):
            raise ValueError("height_ratios must contain three positive finite values")
        self.height_ratios = height_ratios
        self.selection_names, self._tuple_labels = add_stimspace_selection_widgets(self, results, selection_defaults)
        self.add_selection("env_full_scope", value=env_full_scope, options=ENV_FULL_SCOPES)
        self.add_boolean("full_within_env", value=full_within_env)
        self.add_selection("display", value=display, options=("errorPlot", "each"))
        self.add_boolean("pr_pre_smooth", value=pr_pre_smooth)
        for prefix, method, width in (
            ("pf", pf_smooth_method, pf_smooth_width),
            ("full", full_smooth_method, full_smooth_width),
        ):
            self.add_selection(f"{prefix}_smooth_method", value=method, options=("none", "boxcar", "gaussian"))
            self.add_float(f"{prefix}_smooth_width", value=width, min=0.0, max=50.0, step=0.5)
        self.add_boolean("clip_negative", value=clip_negative)
        self.add_boolean("log_y", value=log_y)
        self.add_boolean("sharey", value=sharey)
        self.add_boolean("as_percent", value=as_percent)
        self.add_float_range("shared_variance_ratio_ylims", value=shared_variance_ratio_ylims, min=0.0, max=1.0, step=0.01)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0, step=0.5)
        self.add_integer("legend_panel", value=legend_panel, min=0, max=2)
        add_legend_widgets(self, prefix="env_legend")
        update_legend_widgets(self, {"handlelength": 0.8, "handletextpad": 0.5, **(env_legend or {})}, prefix="env_legend")
        for name in (
            *self.selection_names, "env_full_scope", "full_within_env", "pr_pre_smooth",
            "pf_smooth_method", "pf_smooth_width", "full_smooth_method", "full_smooth_width", "clip_negative",
        ):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _sel_params(self, state):
        return stimspace_selection(state, self.selection_names, self._tuple_labels)

    def _pr(self, spectrum, state, prefix):
        spectrum = _clip_at_first_negative(spectrum) if state["clip_negative"] else spectrum
        if not state["pr_pre_smooth"]:
            shape = spectrum.shape
            spectrum = _smooth_spectrum(
                spectrum.reshape(-1, shape[-1]), state[f"{prefix}_smooth_method"], state[f"{prefix}_smooth_width"]
            ).reshape(shape)
        return _signed_participation_ratio(spectrum)

    def refresh_data(self, state):
        sf, ff, session_mask = select_by_env_spectra(
            self.results, self._sel_params(state), state["env_full_scope"], state["full_within_env"]
        )
        # A whole-session ff denominator is reused for each environment when selected.
        if ff.ndim == 2:
            ff = np.broadcast_to(ff[:, None, :], (ff.shape[0], sf.shape[1], ff.shape[-1]))
        valid = np.isfinite(sf).any(axis=-1) & np.isfinite(ff).any(axis=-1) & session_mask[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            component = np.nansum(sf, axis=-1) / np.nansum(ff, axis=-1)
        component[~np.isfinite(component)] = np.nan
        metrics = (self._pr(ff, state, "full"), self._pr(sf, state, "pf"), component)
        self._curves = [{} for _ in metrics]
        for slot in range(sf.shape[1]):
            for curves in self._curves:
                curves[slot] = {}
            for mouse in self.results.unique_mice:
                rows = chronological_mouse_sessions(self.results, mouse, exclude_bad_envs=False)
                rows = rows[valid[rows, slot]]
                for curves, values in zip(self._curves, metrics):
                    curves[slot][str(mouse)] = values[rows, slot]

    def plot(self, state):
        fontsize = state["fontsize"]
        fig, axes = self.new_subplots(
            3, 1, sharex=True, figsize=self.figsize, layout="constrained", height_ratios=self.height_ratios
        )
        if state["sharey"]:
            axes[1].sharey(axes[0])
        labels = ("Full CA1", "PF Fraction", "Placefield\nComponent")
        xmax = 1
        for row, (ax, curves, ylabel) in enumerate(zip(axes, self._curves, labels)):
            if row < 2 and state["log_y"]:
                ax.set_yscale("log")
            scale = 100.0 if row == 2 and state["as_percent"] else 1.0
            for slot, per_mouse in curves.items():
                length = DimensionalityFamiliarityViewer._draw_curves(
                    ax, {mouse: values * scale for mouse, values in per_mouse.items()},
                    ENV_SLOT_COLORS[slot], f"Env #{slot + 1}", state["display"],
                )
                xmax = max(xmax, length - 1)
            if row == 2:
                ax.set_ylim(*(limit * scale for limit in state["shared_variance_ratio_ylims"]))
                ax.set_yticks(np.array([0.0, 0.5, 1.0]) * scale)
                # set_yticks can expand the limits; retain the selected component range.
                ax.set_ylim(*(limit * scale for limit in state["shared_variance_ratio_ylims"]))
                if state["as_percent"]:
                    ylabel += " (%)"
            ax.set_ylabel(ylabel, fontsize=fontsize)
            format_spines(ax, x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"], tick_fontsize=fontsize)
            if row < 2:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        if state["log_y"]:
            for ax in axes[:2]:
                low, high = ax.get_ylim()
                ax.set_ylim(10.0 ** np.floor(np.log10(low)), 10.0 ** np.ceil(np.log10(high)))
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=100))
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
                ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
                ax.yaxis.set_minor_formatter(NullFormatter())
        axes[-1].set_xlabel("Env Session #", fontsize=fontsize)
        axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[-1].set_xlim(0, xmax)
        axes[-1].spines["bottom"].set_bounds(0, xmax)
        apply_legend(axes[state["legend_panel"]], state, fontsize, prefix="env_legend", auto_loc="upper left")
        # Let constrained layout reserve room on the left for the common PR label,
        # then center it over the actual extent of the two dimensionality panels.
        dimensionality_label = fig.supylabel("Dimensionality (PR)", fontsize=fontsize)
        fig.draw_without_rendering()
        dimensionality_label.set_y((axes[0].get_position().y1 + axes[1].get_position().y0) / 2)
        return fig
