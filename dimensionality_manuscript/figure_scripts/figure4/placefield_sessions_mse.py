"""Per-session Tilbury-fit parameter distributions and decay-law goodness-of-fit.

Two viewers over Tilbury-fit results:

- :class:`PlacefieldParameterSessionsViewer` -- session-resolved distributions of one fitted
  parameter (or selected penalty weight): one mouse's sessions, the population grouped by
  within-mouse session number, and a running summary statistic across session number.
- :class:`PlacefieldSpectrumMSEViewer` -- power-law vs. exponential decay-law fit quality
  (:data:`~._spectrum_math.DECAY_MODELS`) for the Tilbury eigenspectra and, optionally, a data
  spectrum, using the same adaptive/fixed fit-window machinery as
  :mod:`.spectra_diagnostics`.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from vrAnalysis.helpers import edge2center
from vrAnalysis.helpers.plotting import errorPlot, format_spines

from dimensionality_manuscript import ResultsAggregator, average_by_mouse

from ._alpha_config import ADAPTIVE_ALPHA_CONFIG_REGISTRY, AdaptiveAlphaConfig
from ._param_axes import (
    CVPCA_KEYS,
    FIT_KEY_PARAM_KEYS,
    POP_ALPHA_COLORS,
    POP_ALPHA_LABELS,
    PREFERRED_DEFAULTS,
    SOURCE_OF_KEY,
    STIMSPACE_KEYS,
    add_merged_param_axis_widgets,
    merged_axis_names as _merged_axis_names,
    sel_params,
)
from ._spectrum_math import (
    DECAY_MODELS,
    _align_rows_to_sessions,
    _clip_at_first_negative,
    _decay_fit_per_session,
    _eig_to_ss_scale,
    _smooth_spectrum,
)
from ._stats import _decay_stat_panel
from ..panels import FigureViewer

# Unregularized-generalized-fit color when only one generalized variant is shown (labelled
# "Generalized" either way). Kept private to this module, matching the same small, figure-local
# color constant already duplicated in placefield_fits.py -- neither panel shares enough with the
# other to justify a shared color module.
_GENERALIZED_COLOR = "blue"


class PlacefieldParameterSessionsViewer(FigureViewer):
    """Session-resolved distributions of Tilbury fit parameters and selected penalties.

    ``ax[0]`` shows every retained session of one mouse, ``ax[1]`` groups the same
    distributions by within-mouse session number across the population, and ``ax[2]``
    reduces every mouse/session distribution to a configurable summary statistic.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``TilburyFitConfig`` results.
    mouse : str or None
        Mouse shown in ax[0]. If None, the first of ``results.unique_mice`` is used.
    parameter : {"peak_exponent", "asymmetry", "peak_penalty", "asymmetry_penalty"}
        Which per-neuron quantity to show: the fitted peak exponent, the absolute log
        left/right width ratio, or one of the two validation-selected penalty weights.
    fit : {"generalized", "shrinkage"}
        Which fitted parameters to use for ``"peak_exponent"``/``"asymmetry"``. Ignored for the
        penalty options, which always use the validation-selected shrinkage weights.
    display : {"each", "errorPlot"}
        ax[1]/ax[2] rendering: per-mouse curves plus their mean, or the mean +/- SE band.
    skip_sessions : int
        Thin the colored session groups shown in ax[0]/ax[1] while always keeping the first and
        last retained session.
    summary : {"mean", "median", "percentile"}
        Per-session scalar statistic shown in ax[2].
    percentile : float
        Percentile used when ``summary="percentile"``.
    num_bins : int
        Bin count for the histogram fallback when a session's KDE is singular (constant-valued).
    fontsize : float
        Font size for every tick label, axis label, title, and colorbar label.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the Tilbury-fit param-axis widgets (e.g.
        ``activity_parameters_name``), overriding :data:`~._param_axes.PREFERRED_DEFAULTS`.
    """

    _PARAMETERS = ["peak_exponent", "asymmetry", "peak_penalty", "asymmetry_penalty"]
    _FITS = ["generalized", "shrinkage"]
    _SUMMARIES = ["mean", "median", "percentile"]

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        parameter: str = "peak_exponent",
        fit: str = "shrinkage",
        display: str = "errorPlot",
        skip_sessions: int = 0,
        summary: str = "median",
        percentile: float = 75.0,
        num_bins: int = 80,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (9.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.config = results.config_class
        self.figsize = figsize

        mouse_options = list(results.unique_mice)
        if mouse is None:
            self.add_selection("mouse", options=mouse_options)
        else:
            self.add_selection("mouse", options=mouse_options, value=mouse)
        self.add_selection("parameter", options=self._PARAMETERS, value=parameter)
        # The fit selector affects fitted parameters. Penalty weights exist only for the
        # validation-selected shrinkage fit, so it is deliberately ignored for penalty views.
        self.add_selection("fit", options=self._FITS, value=fit)
        self.add_selection("display", options=["each", "errorPlot"], value=display)
        self.add_integer("skip_sessions", value=skip_sessions, min=0, max=len(results.sessions))
        self.add_selection("summary", options=self._SUMMARIES, value=summary)
        self.add_float("percentile", value=percentile, min=0.0, max=100.0, step=0.5)
        self.add_integer("num_bins", value=num_bins, min=5, max=200)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        self._tuple_labels = add_merged_param_axis_widgets(self, results, preferred_defaults={**PREFERRED_DEFAULTS, **selection_defaults})
        self._axis_names = _merged_axis_names(results)

        for name in (*self._axis_names, "parameter", "fit", "num_bins", "summary", "percentile"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    @staticmethod
    def _shown_session_indices(n_sessions: int, skip_sessions: int) -> np.ndarray:
        """Evenly thin session indices while always retaining the first and last."""
        if n_sessions <= 2 or skip_sessions <= 0:
            return np.arange(n_sessions)
        n_points = max(2, int(round((n_sessions - 1) / (skip_sessions + 1))) + 1)
        return np.unique(np.round(np.linspace(0, n_sessions - 1, n_points)).astype(int))

    def _session_values(self, state: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return retained per-neuron values and their mouse names, preserving session order."""
        selected = self.results.sel(
            keys=["params", "params_shrinkage", "lambda_selected", "r2_test"],
            load_ragged=True,
            squeeze_ones=False,
            **sel_params(state, self._tuple_labels, self.results.param_axes),
        )
        r2_test = np.asarray(selected["r2_test"], dtype=float)
        valid_sessions = np.sum(np.isfinite(r2_test), axis=-1) >= 200
        parameter = state["parameter"]

        if parameter in ("peak_penalty", "asymmetry_penalty"):
            penalty_idx = 0 if parameter == "peak_penalty" else 1
            values = np.asarray(selected["lambda_selected"], dtype=float)[..., penalty_idx]
        else:
            params_key = "params_shrinkage" if state["fit"] == "shrinkage" else "params"
            params = np.asarray(selected[params_key], dtype=float)
            if parameter == "peak_exponent":
                values = params[..., self.config.param_names.index("p")]
            else:
                left = params[..., self.config.param_names.index("sigma_left")]
                right = params[..., self.config.param_names.index("sigma_right")]
                with np.errstate(divide="ignore", invalid="ignore"):
                    # Magnitude of the same log-width ratio used by the asymmetry penalty.
                    values = np.abs(np.log(left / right))
                values[~np.isfinite(values)] = np.nan
        return values[valid_sessions], np.asarray(self.results.mouse_names)[valid_sessions]

    def _distribution_axis(self, values: np.ndarray, state: dict) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
        """Build per-session density/mass curves and their shared x coordinates."""
        parameter = state["parameter"]
        finite_all = values[np.isfinite(values)]
        if parameter in ("peak_penalty", "asymmetry_penalty"):
            grid = np.unique(finite_all)
            if grid.size == 0:
                grid = np.array([0.0])
            x = np.arange(grid.size, dtype=float)
            curves = np.full((values.shape[0], grid.size), np.nan)
            for row_idx, row in enumerate(values):
                finite = row[np.isfinite(row)]
                if finite.size:
                    curves[row_idx] = np.array([np.mean(finite == value) for value in grid])
            labels = [f"{value:g}" for value in grid]
            return x, curves, labels

        if parameter == "peak_exponent":
            lo, hi = 0.0, 10.0
        elif finite_all.size:
            lo = 0.0
            hi = float(np.nanpercentile(finite_all, 99.5))
            if not np.isfinite(hi) or hi <= lo:
                hi = 1.0
        else:
            lo, hi = 0.0, 1.0
        x = edge2center(np.linspace(lo, hi, state["num_bins"] + 1))
        curves = np.full((values.shape[0], x.size), np.nan)
        for row_idx, row in enumerate(values):
            finite = row[np.isfinite(row)]
            if finite.size < 2:
                continue
            try:
                curves[row_idx] = gaussian_kde(finite)(x)
            except np.linalg.LinAlgError:
                # Constant-valued sessions make a KDE singular; a normalized histogram still
                # gives a useful, deterministic distribution on the same grid.
                hist, _ = np.histogram(finite, bins=state["num_bins"], range=(lo, hi), density=True)
                curves[row_idx] = hist
        return x, curves, None

    @staticmethod
    def _by_mouse_session(values: np.ndarray, mouse_names: np.ndarray) -> tuple[np.ndarray, list]:
        """Pad session-ordered rows to ``(mice, session_number, ...)``."""
        mice = list(dict.fromkeys(mouse_names))
        grouped = [values[mouse_names == mouse] for mouse in mice]
        max_sessions = max((len(rows) for rows in grouped), default=0)
        trailing_shape = values.shape[1:]
        output = np.full((len(mice), max_sessions, *trailing_shape), np.nan)
        for mouse_idx, rows in enumerate(grouped):
            output[mouse_idx, : len(rows)] = rows
        return output, mice

    def _summarize(self, values: np.ndarray, state: dict) -> np.ndarray:
        """Reduce each session's neurons to the selected scalar statistic."""
        if state["summary"] == "mean":
            return np.nanmean(values, axis=1)
        if state["summary"] == "median":
            return np.nanmedian(values, axis=1)
        return np.nanpercentile(values, state["percentile"], axis=1)

    @staticmethod
    def _render_group(
        ax,
        x: np.ndarray,
        data: np.ndarray,
        color,
        display: str,
        label: str | None = None,
        min_support: int = 2,
    ):
        """Render mouse rows plus their mean, or their mean +/- SE."""
        valid_count = np.sum(np.isfinite(data), axis=0)
        value_sum = np.nansum(data, axis=0)
        mean = np.divide(value_sum, valid_count, out=np.full_like(value_sum, np.nan), where=valid_count > 0)
        mean[valid_count < min_support] = np.nan
        if display == "each":
            ax.plot(x, data.T, color=color, linewidth=0.5, alpha=0.25)
            ax.plot(x, mean, color=color, linewidth=1.8, label=label)
        else:
            masked = np.where(valid_count[None, :] >= min_support, data, np.nan)
            errorPlot(x, masked, axis=0, se=True, ax=ax, color=color, linewidth=1.8, alpha=0.2, label=label)

    def refresh_data(self, state: dict) -> None:
        """Re-select the retained per-session values and their session-grouped distributions."""
        values, mouse_names = self._session_values(state)
        x_distribution, distributions, category_labels = self._distribution_axis(values, state)
        distributions_by_session, _ = self._by_mouse_session(distributions, mouse_names)
        summary_by_session, _ = self._by_mouse_session(self._summarize(values, state), mouse_names)

        self._values = values
        self._mouse_names = mouse_names
        self._x_distribution = x_distribution
        self._distributions = distributions
        self._category_labels = category_labels
        self._distributions_by_session = distributions_by_session
        self._summary_by_session = summary_by_session

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        values = self._values
        mouse_names = self._mouse_names
        x_distribution = self._x_distribution
        distributions = self._distributions
        category_labels = self._category_labels
        distributions_by_session = self._distributions_by_session
        summary_by_session = self._summary_by_session

        max_sessions = distributions_by_session.shape[1]
        support = np.sum(np.isfinite(distributions_by_session).any(axis=-1), axis=0) if max_sessions else np.array([])
        kept_sessions = np.where(support > 1)[0]
        shown_population = kept_sessions[self._shown_session_indices(len(kept_sessions), state["skip_sessions"])]
        colors = plt.get_cmap("coolwarm")(np.linspace(0, 1, max(max_sessions, 1)))

        fig, ax = self.new_subplots(1, 3, figsize=self.figsize, layout="constrained")

        # ax[0]: one selected mouse, one density/mass curve per retained session.
        mouse_mask = mouse_names == state["mouse"]
        mouse_distributions = distributions[mouse_mask]
        shown_mouse = self._shown_session_indices(len(mouse_distributions), state["skip_sessions"])
        for session_idx in shown_mouse:
            ax[0].plot(x_distribution, mouse_distributions[session_idx], color=colors[session_idx], linewidth=1.2)
        ax[0].set_title(str(state["mouse"]), fontsize=fontsize)

        # ax[1]: one session-number group across mice; style controls mouse traces vs mean +/- SE.
        for session_idx in shown_population:
            self._render_group(
                ax[1],
                x_distribution,
                distributions_by_session[:, session_idx],
                colors[session_idx],
                state["display"],
            )
        ax[1].set_title("Across mice", fontsize=fontsize)

        # ax[2]: the requested per-session scalar, followed across session number for every mouse.
        session_x = np.arange(max_sessions) + 1
        if max_sessions:
            self._render_group(ax[2], session_x, summary_by_session, "k", state["display"])

        parameter = state["parameter"]
        xlabel = {
            "peak_exponent": "Peak Exponent",
            "asymmetry": r"Asymmetry $|\log(\sigma_L/\sigma_R)|$",
            "peak_penalty": r"Selected $\lambda_p$",
            "asymmetry_penalty": r"Selected $\lambda_{\rm asym}$",
        }[parameter]
        ylabel = "Probability" if parameter.endswith("penalty") else "Density"
        for axis in ax[:2]:
            axis.set_xlabel(xlabel, fontsize=fontsize)
            axis.set_ylabel(ylabel, fontsize=fontsize)
            if category_labels is not None:
                axis.set_xticks(x_distribution, category_labels, fontsize=fontsize)
        ax[2].set_xlabel("Session #", fontsize=fontsize)
        summary_label = f"{state['percentile']:g}th percentile" if state["summary"] == "percentile" else state["summary"].capitalize()
        ax[2].set_ylabel(f"{summary_label}\n{xlabel}", fontsize=fontsize)

        if parameter == "peak_exponent":
            ax[0].axvline(2.0, color="k", linestyle=":", linewidth=0.8)
            ax[1].axvline(2.0, color="k", linestyle=":", linewidth=0.8)
        if parameter.endswith("penalty"):
            positive = values[np.isfinite(values) & (values > 0)]
            if positive.size:
                ax[2].set_yscale("symlog", linthresh=float(np.min(positive)) / 2)

        if max_sessions:
            sm = ScalarMappable(norm=Normalize(vmin=1, vmax=max(max_sessions, 2)), cmap="coolwarm")
            colorbar = fig.colorbar(sm, ax=ax[:2], location="bottom", fraction=0.08, pad=0.16, aspect=30)
            colorbar.set_label("Session #", fontsize=fontsize)
            colorbar.ax.tick_params(labelsize=fontsize)
            colorbar.set_ticks([1, max_sessions] if max_sessions > 1 else [1])
        # Not routed through panels.style_model_axis: that helper applies tick_params with
        # which="both", but the original figure only sized major ticks -- ax[2] can be on a
        # symlog scale (penalty parameters) whose minor ticks would then pick up a font size
        # they never had before.
        for axis in ax:
            format_spines(axis, x_pos=-0.02, y_pos=-0.02, spines_visible=["bottom", "left"])
            axis.tick_params(labelsize=fontsize)
        return fig


class PlacefieldSpectrumMSEViewer(FigureViewer):
    """Decay-law goodness-of-fit for the Tilbury placefield eigenspectra.

    Two candidate decay laws are compared -- power law ``n^-alpha`` and exponential
    ``exp(-n^2 / 2 M^2)`` (:data:`~._spectrum_math.DECAY_MODELS`, the two xtick positions in each
    panel). Both are fit (log-space) to every spectrum: the ``source_key`` data spectrum (from
    ``results_spectra``/``results_cvpca``) plus the generalized-shrinkage, unregularized
    generalized and plain-Gaussian fit eigenspectra from the Tilbury-fit aggregator (``results``).

    - ax[0]: each fit's characteristic parameter -- the power-law exponent ``alpha`` at the
      power-law tick and the exponential characteristic dimension ``M`` at the exponential tick
      (y-axis "Characteristic Dim.").
    - ax[1]: the log-space MSE of each fit at the same two x-positions. A spectrum that follows
      one law reads low there and high at the other -- the point of the comparison, since
      Gaussian-tuned populations decay too fast to be genuine power laws.

    Every curve option is one colour; ``display="each"`` draws a faint per-mouse line across the
    two x-positions plus a bold across-mouse mean, ``display="errorPlot"`` draws the across-mouse
    mean +/- SE band, and ``display="swarm"`` drops the per-mouse connections for one beeswarm
    column per (decay model, curve) with a short horizontal mean line (spread set by
    ``beewidth``). The fit window is either fixed (``fit_zone="fixed"``, the ``fixed_range``
    integer range) or per-session adaptive (``fit_zone="adaptive"``, the same
    peak-curvature-to-noise-floor window and :class:`~._alpha_config.AdaptiveAlphaConfig`
    machinery as the other spectrum-diagnostics panels).

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``TilburyFitConfig`` results (source of the ``eig_*`` spectra).
    results_spectra : ResultsAggregator or None
        Aggregated StimSpaceSpectra results, source of the ``source_key`` data spectrum and of
        the fixed ``ss_cvpca`` adaptive-window fallback. If None, the data column is dropped and
        every column must locate its own adaptive window.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results; when given, ``reg_covariances_fixed`` is also a valid
        ``source_key``.
    generalized_fit : {"both", "generalized", "shrinkage"}
        Which generalized fit column(s) to show alongside the data and Gaussian columns.
        ``"both"`` shows shrinkage (purple) and unregularized generalized (blue); a single choice
        shows only that fit, drawn blue and labelled "Generalized".
    fit_zone : {"adaptive", "fixed"}
        ``"adaptive"`` locates each session's own peak-curvature-to-noise-floor window (with the
        ``ss_cvpca`` fallback for non-cross-validated spectra); ``"fixed"`` fits every session
        over ``fixed_range``.
    fixed_range : tuple[int, int]
        ``(start, end)`` index window used when ``fit_zone="fixed"``.
    display : {"each", "errorPlot", "swarm"}
        Rendering mode, see above.
    beewidth : float
        Beeswarm point spread in x-axis units, used only when ``display="swarm"``.
    source_key : str
        Which spectrum is the data column: ``ss_cv``/``ss_direct``/``ss_cvpca``/``sf_cv``/
        ``sf_direct`` (from ``results_spectra``) or ``reg_covariances_fixed`` (from
        ``results_cvpca``). Only offered as a widget when ``results_spectra`` is given.
    normalize : bool
        Normalize each spectrum by its sum (per session) before smoothing. Does not affect the
        MSE (a constant rescale only shifts the log-space intercept), kept for parity with the
        other spectrum figures.
    clip_negative : bool
        Replace each spectrum's first negative entry and all later ranks with NaN before fitting.
    source_cfg : AdaptiveAlphaConfig or None
        Log-space smoothing + adaptive-window configuration shared by every column. Defaults to
        ``ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]`` when None; its fields seed the
        ``source_smooth_method``/``source_smooth_width``/``source_fpd_window_size``/
        ``source_adaptive_buffer``/``source_minimum_window_size`` widgets.
    fontsize : float
        Font size for every tick label, axis label, and legend text.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the shared param-axis widgets (e.g. ``activity_parameters_name``,
        ``smooth_widths``, ``reliability_fraction_active_thresholds``), overriding
        :data:`~._param_axes.PREFERRED_DEFAULTS`.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_spectra: ResultsAggregator | None = None,
        results_cvpca: ResultsAggregator | None = None,
        *,
        generalized_fit: str = "both",
        fit_zone: str = "adaptive",
        fixed_range: tuple[int, int] = (10, 50),
        display: str = "each",
        beewidth: float = 0.2,
        source_key: str = "ss_cv",
        normalize: bool = True,
        clip_negative: bool = False,
        source_cfg: AdaptiveAlphaConfig | None = None,
        fontsize: float = 9.0,
        # The caller function this viewer replaces (`placefield_spectrum_mse`) always passed its
        # own figsize=(7.0, 3.0) default into the constructor, overriding whatever default lived
        # here; that is the figsize every notebook actually saw, so it wins over the (8.0, 3.0)
        # this class previously defaulted to when built directly.
        figsize: tuple[float, float] = (7.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.results_spectra = results_spectra
        self.results_cvpca = results_cvpca
        self.source_cfg = source_cfg if source_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]
        # Data spectra come from these, resolved per key via SOURCE_OF_KEY.
        self._agg = {"stimspace": results_spectra, "cvpca": results_cvpca}
        self.figsize = figsize

        self.add_selection("generalized_fit", options=["both", "generalized", "shrinkage"], value=generalized_fit)
        self.add_selection("fit_zone", options=["adaptive", "fixed"], value=fit_zone)
        self.add_integer_range("fixed_range", value=tuple(fixed_range), min=1, max=500)
        self.add_selection("display", options=["each", "errorPlot", "swarm"], value=display)
        self.add_float("beewidth", value=beewidth, min=0.0, max=1.0, step=0.01)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        # Data-spectrum source (the first column); options mirror the other spectrum panels.
        if results_spectra is not None:
            source_options = list(STIMSPACE_KEYS) + (list(CVPCA_KEYS) if results_cvpca is not None else [])
            self.add_selection("source_key", options=source_options, value=source_key)

        # One widget per shared param-axis name (tuple-valued axes encoded as string labels), so
        # both the data spectrum and the Tilbury-fit eig spectra can be sliced.
        self._tuple_labels = add_merged_param_axis_widgets(
            self,
            results_spectra,
            results_cvpca,
            results,
            preferred_defaults={**PREFERRED_DEFAULTS, **selection_defaults},
        )
        self._fit_axes = list(results.param_axes)
        self._axis_names = _merged_axis_names(results_spectra, results_cvpca, results)

        # Log-space smoothing + adaptive-window controls, shared by every column.
        self.add_boolean("normalize", value=normalize)
        self.add_boolean("clip_negative", value=clip_negative)
        cfg = self.source_cfg
        self.add_selection("source_smooth_method", options=["none", "boxcar", "gaussian"], value=cfg.smooth_method)
        self.add_float("source_smooth_width", value=cfg.smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_integer("source_fpd_window_size", value=cfg.fpd_window_size, min=1, max=50)
        self.add_integer("source_adaptive_buffer", value=cfg.adaptive_buffer, min=0, max=50)
        self.add_integer("source_minimum_window_size", value=cfg.minimum_window_size, min=1, max=500)

        refresh_names = [
            "generalized_fit",
            "fit_zone",
            "fixed_range",
            *self._axis_names,
            "normalize",
            "clip_negative",
            "source_smooth_method",
            "source_smooth_width",
            "source_fpd_window_size",
            "source_adaptive_buffer",
            "source_minimum_window_size",
        ]
        if results_spectra is not None:
            refresh_names.append("source_key")
        for name in refresh_names:
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    @staticmethod
    def _cfg_from_state(state: dict, prefix: str) -> AdaptiveAlphaConfig:
        """Build the adaptive-fit settings from the ``{prefix}_*`` widgets."""
        return AdaptiveAlphaConfig(
            smooth_method=state[f"{prefix}_smooth_method"],
            smooth_width=state[f"{prefix}_smooth_width"],
            fpd_window_size=int(state[f"{prefix}_fpd_window_size"]),
            adaptive_buffer=int(state[f"{prefix}_adaptive_buffer"]),
            minimum_window_size=int(state[f"{prefix}_minimum_window_size"]),
        )

    def _spectrum_sessions(self, state: dict, key: str, cfg: AdaptiveAlphaConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Per-session raw and smoothed ``(sessions, dims)`` data spectrum for ``key``, with mouse/session ids.

        Normalize is applied per session (row) rather than after mouse-averaging, so the adaptive
        alpha fit can find each session's own first-negative crossover before any cross-session
        averaging blurs it. Both the raw (pre-smoothing) and smoothed spectrum are returned:
        smoothing maps every non-positive entry to NaN before exponentiating back
        (:func:`~._spectrum_math._smooth_spectrum`), so a smoothed row is never negative --
        first-negative detection must use the raw one, while the exponent fit itself uses the
        smoothed one.
        """
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        spec = agg.sel(keys=[key], avg_by_mouse=False, **sel_params(state, self._tuple_labels, agg.param_axes))[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["clip_negative"]:
            spec = _clip_at_first_negative(spec)
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)
        return spec, smoothed, agg.mouse_names, agg.session_ids

    def _fit_spectrum_raw_sessions(self, state: dict, key: str) -> np.ndarray:
        """Raw (unnormalized, unsmoothed) per-session Tilbury-fit eigenvalue spectrum for ``key``.

        Each session is converted from the PCA convention to the ``ss_cv`` covariance convention
        with ``P / (P - 1)``, gated by whichever ``params*`` arrays back ``key`` (see
        :data:`~._param_axes.FIT_KEY_PARAM_KEYS`).
        """
        fit_params = sel_params(state, self._tuple_labels, self._fit_axes)
        param_keys = FIT_KEY_PARAM_KEYS[key]
        selected = self.results.sel(keys=[key, *param_keys], squeeze_ones=False, **fit_params)
        dist_centers = self.results.sel_objects(keys=["dist_centers"], **fit_params)["dist_centers"]
        param_arrays = [selected[pk] for pk in param_keys]
        return np.stack(
            [
                _eig_to_ss_scale(eig, list(params_row), theta)
                for eig, theta, *params_row in zip(
                    np.asarray(selected[key], dtype=float),
                    dist_centers,
                    *param_arrays,
                )
            ]
        )

    def _fit_spectrum_sessions(self, state: dict, key: str, cfg: AdaptiveAlphaConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Per-session raw+smoothed Tilbury-fit eigenvalue spectrum for ``key``, with mouse/session ids."""
        session_spec = self._fit_spectrum_raw_sessions(state, key)
        if state["clip_negative"]:
            session_spec = _clip_at_first_negative(session_spec)
        if state["normalize"]:
            session_spec = session_spec / np.nansum(session_spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(session_spec, cfg.smooth_method, cfg.smooth_width)
        return session_spec, smoothed, self.results.mouse_names, self.results.session_ids

    def _columns(self, state: dict, cfg: AdaptiveAlphaConfig) -> list[tuple]:
        """Assemble the spectrum columns: ``(raw, smooth, mouse_names, session_ids, color, label)``.

        Order matches the request -- data, then the selected generalized fit(s), then the
        Gaussian control (see :data:`~._param_axes.POP_ALPHA_COLORS`/``POP_ALPHA_LABELS``).
        """
        fit_sel = state["generalized_fit"]
        columns: list[tuple] = []
        if self.results_spectra is not None:
            raw, smooth, mouse_names, session_ids = self._spectrum_sessions(state, state["source_key"], cfg)
            columns.append((raw, smooth, mouse_names, session_ids, POP_ALPHA_COLORS["source_key"], "Data"))
        if fit_sel == "both":
            eig_keys = ["eig_shrinkage", "eig_tilbury"]
        elif fit_sel == "shrinkage":
            eig_keys = ["eig_shrinkage"]
        else:  # "generalized"
            eig_keys = ["eig_tilbury"]
        eig_keys.append("eig_control")
        for key in eig_keys:
            raw, smooth, mouse_names, session_ids = self._fit_spectrum_sessions(state, key, cfg)
            if fit_sel != "both" and key in ("eig_shrinkage", "eig_tilbury"):
                color, label = _GENERALIZED_COLOR, "Generalized"
            else:
                color, label = POP_ALPHA_COLORS[key], POP_ALPHA_LABELS[key]
            columns.append((raw, smooth, mouse_names, session_ids, color, label))
        return columns

    def refresh_data(self, state: dict) -> None:
        """Fit both decay models to every column's spectrum, averaged by mouse."""
        cfg = self._cfg_from_state(state, "source")
        fit_zone = state["fit_zone"]
        fixed_range = tuple(int(v) for v in state["fixed_range"])
        columns = self._columns(state, cfg)

        # Fixed fallback window source (ss_cvpca) for non-cross-validated fit spectra in adaptive
        # mode; harmless when a column already is cross-validated (it locates its own window then).
        cvpca = self._spectrum_sessions(state, "ss_cvpca", cfg) if self.results_spectra is not None else None

        model_keys = [m for m, _ in DECAY_MODELS]
        colors, labels = [], []
        mse_mats, param_mats = [], []
        for raw, smooth, mouse_names, session_ids, color, label in columns:
            fb_raw = fb_smooth = None
            if fit_zone == "adaptive" and cvpca is not None:
                cvpca_raw, cvpca_smooth, _, cvpca_session_ids = cvpca
                fb_raw = _align_rows_to_sessions(session_ids, cvpca_session_ids, cvpca_raw)
                fb_smooth = _align_rows_to_sessions(session_ids, cvpca_session_ids, cvpca_smooth)
            mse_cols, param_cols = [], []
            for model_key in model_keys:
                mse_s, param_s = _decay_fit_per_session(raw, smooth, model_key, fit_zone, fixed_range, cfg.adaptive_buffer, fb_raw, fb_smooth)
                mse_cols.append(average_by_mouse(mse_s, mouse_names))
                param_cols.append(average_by_mouse(param_s, mouse_names))
            mse_mats.append(np.stack(mse_cols, axis=1))  # (n_mice, 2)
            param_mats.append(np.stack(param_cols, axis=1))
            colors.append(color)
            labels.append(label)

        self._mse_mats = mse_mats
        self._param_mats = param_mats
        self._colors = colors
        self._labels = labels

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        display = state["display"]
        beewidth = state["beewidth"]
        xtick_labels = [lbl for _, lbl in DECAY_MODELS]

        fig, axes = self.new_subplots(1, 2, figsize=self.figsize, layout="constrained")

        # ax[0]: each fit's characteristic parameter -- power-law exponent (alpha) at the
        # power-law tick, exponential characteristic dimension (M) at the exponential tick.
        _decay_stat_panel(axes[0], self._param_mats, self._colors, self._labels, display, beewidth, fontsize, xtick_labels)
        axes[0].set_ylabel("Characteristic Dim.", fontsize=fontsize)
        axes[0].legend(fontsize=fontsize * 0.8, frameon=False)

        # ax[1]: log-space MSE of each fit, same format.
        _decay_stat_panel(axes[1], self._mse_mats, self._colors, self._labels, display, beewidth, fontsize, xtick_labels)
        axes[1].set_ylabel("Log-space MSE", fontsize=fontsize)
        return fig
