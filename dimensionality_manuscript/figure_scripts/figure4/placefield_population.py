"""Tilbury generalized-Gaussian placefield fits: population-summary panel (no examples)."""

import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from vrAnalysis.helpers import edge2center
from vrAnalysis.helpers.plotting import beeswarm, format_spines

from dimensionality_manuscript import ResultsAggregator, average_by_mouse

from ._alpha_config import ADAPTIVE_ALPHA_CONFIG_REGISTRY, AdaptiveAlphaConfig
from ._param_axes import (
    CVPCA_KEYS,
    FIT_KEY_PARAM_KEYS,
    POP_ALPHA_COLORS,
    POP_ALPHA_LABELS,
    POP_EIG_KEYS,
    PREFERRED_DEFAULTS,
    SOURCE_OF_KEY,
    STIMSPACE_KEYS,
    add_merged_param_axis_widgets,
)
from ._param_axes import encode_param as _encode_param
from ._param_axes import sel_params as _sel_params_for
from ._spectrum_math import (
    DECAY_MODELS,
    _align_rows_to_sessions,
    _clip_at_first_negative,
    _decay_fit_per_session,
    _eig_to_ss_scale,
    _median_fpd_alpha_per_session,
    _smooth_spectrum,
)
from ._stats import _beeswarm_panel, _decay_stat_panel, _paired_pvalue, _significance_stars, _zero_to_max_ticks
from ..legends import add_legend_widgets, apply_legend, update_legend_widgets
from ..panels import FigureViewer, style_model_axis


class PlacefieldPopulationViewer(FigureViewer):
    """Tilbury generalized-Gaussian placefield fits: population summaries (no examples).

    Four panels driven by every session in
    :class:`~dimensionality_manuscript.configs.tilbury_fit.TilburyFitConfig` results (one fit per
    neuron; the reported quality is held-out test R^2):

    - gs[0]: per-mouse peak-exponent (``p``) density for both generalized fits — unregularized
      (blue) and shrinkage (purple) — thin per-mouse lines with the bold across-mouse mean, and a
      reference line at ``p = 2`` (the ordinary-Gaussian exponent).
    - gs[1]: per-mouse median test R^2 for the shrinkage, generalized and Gaussian models (paired,
      thin gray, in that column order) with the across-mouse mean (bold dark line).
    - gs[2]: fraction of neurons where the generalized fit beats the Gaussian, either pooled to one
      per-mouse beeswarm (``fraction_view="pooled"``) or broken down with one beeswarm of per-session
      values per mouse (``fraction_view="by_mouse"``). Omitted entirely when ``fraction_view="none"``.
    - gs[-1]: an across-mouse spectrum-decay statistic for the selected ``source_key`` spectrum (from
      ``results_spectra``/``results_cvpca``) plus the :data:`~._param_axes.POP_EIG_KEYS` fit spectra
      (colors in :data:`~._param_axes.POP_ALPHA_COLORS`), selected by ``last_axis``:

      - ``"decay_exponent"``: one beeswarm column per curve of the power-law exponent estimated as
        the median five-point-derivative local exponent over each session's own
        peak-curvature-to-noise-floor window, computed per session and averaged by mouse, under one
        :class:`~._alpha_config.AdaptiveAlphaConfig` (the ``source_*`` widgets). Keys with no
        negative entry borrow their window from that session's ``ss_cvpca`` row when
        ``results_spectra`` is given.
      - ``"characteristic_dim"`` / ``"mse"``: each curve fit under both
        :data:`~._spectrum_math.DECAY_MODELS` (power law and exponential, the two x-positions of the
        "Fit Type" axis) over the ``fit_zone`` window — the characteristic parameter (``alpha`` /
        ``M``, y-label "Dimensionality", spanning ``[0, max]`` with ticks every 5 units) or the
        log-space MSE — laid out by ``display`` (``each``/``errorPlot``/``swarm``).

      The panel's legend is styled by the ``legend_*`` widgets (:func:`~..legends.add_legend_widgets`).

    Data selection is one widget per param axis shared across ``results``, ``results_spectra`` and
    ``results_cvpca`` (:func:`~._param_axes.add_merged_param_axis_widgets`), so every panel slices to
    the same selection; tuple-valued axes (e.g. ``smooth_widths``) are encoded as string labels since
    a Syd selection widget only takes scalars.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated :class:`TilburyFitConfig` results (source of the ``eig_*`` spectra and of every
        per-neuron fit statistic in gs[0]/gs[1]/gs[2]).
    results_spectra : ResultsAggregator or None
        Aggregated StimSpaceSpectra results, source of the ``source_key`` spectrum in gs[-1] and of
        the fixed ``ss_cvpca`` window-fallback source. If None, gs[-1] shows only the eig fit
        spectra, and every key must locate its own adaptive window.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results; when given, ``reg_covariances_fixed`` is also a valid
        ``source_key``.
    source_cfg : AdaptiveAlphaConfig or None
        Fixed adaptive-fit configuration (smoothing, five-point-derivative window, adaptive buffer,
        minimum window size) shared by every gs[-1] curve, whichever ``last_axis`` is shown. Defaults
        to ``ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]`` when None. Editable afterwards through
        the ``source_*`` widgets, which seed from it. The exponent is the median
        five-point-derivative local exponent over each session's own peak-curvature-to-noise-floor
        window, computed per session and averaged by mouse; sessions with fewer than
        ``minimum_window_size`` finite local-exponent values in that window are NaN. Keys whose row
        has no negative entry borrow the window from that session's ``ss_cvpca`` row (only available
        when ``results_spectra`` is given).
    num_bins : int
        Bin count for the per-session KDE of the peak-exponent density (gs[0]).
    fraction_view : {"pooled", "by_mouse", "none"}
        gs[2] layout: one pooled per-mouse beeswarm, one per-session beeswarm per mouse, or the
        panel omitted entirely.
    beewidth : float
        Beeswarm point spread in x-axis units (gs[2] and gs[-1]).
    source_key : str
        Which spectrum drives the gs[-1] orange curve: ``ss_cv``/``ss_direct``/``ss_cvpca`` (from
        ``results_spectra``) or ``reg_covariances_fixed`` (from ``results_cvpca``). Only offered as
        a widget when ``results_spectra`` is given.
    metric : {"r2", "cc"}
        Which performance metric drives the gs[1] and gs[2] panels: held-out test R^2 or Pearson
        correlation.
    generalized_fit : {"both", "generalized", "shrinkage"}
        Which generalized fit to show in gs[0]/gs[1]/gs[-1]. ``"both"`` plots the unregularized
        (blue) and shrinkage (purple) fits side by side; a single choice plots only that fit, drawn
        black in gs[0]/gs[1] and blue in gs[-1] (matching :class:`SpectrumFigureViewer`), labelled
        "Generalized".
    include_better : bool
        Include the per-neuron "better" composite eig spectrum in the gs[-1] decay panel.
    paired_test : {"ttest", "wilcoxon"}
        Two-sided paired test used for the gs[1] fit-vs-Gaussian asterisks (Bonferroni-corrected when
        ``generalized_fit="both"``): paired t-test or Wilcoxon signed-rank, on the per-mouse averages.
    last_axis : {"decay_exponent", "characteristic_dim", "mse"}
        Which spectrum-decay statistic gs[-1] shows.
    fit_zone : {"adaptive", "fixed"}
        Fit window for the ``"characteristic_dim"``/``"mse"`` options (ignored by
        ``"decay_exponent"``, which always uses its own adaptive window): ``"adaptive"`` locates each
        session's own peak-curvature-to-noise-floor window (with the ``ss_cvpca`` fallback for
        non-cross-validated spectra); ``"fixed"`` fits every session over ``fixed_range``.
    fixed_range : tuple[int, int]
        ``(start, end)`` index window used when ``fit_zone="fixed"``.
    display : {"each", "errorPlot", "swarm"}
        gs[-1] layout for the ``"characteristic_dim"``/``"mse"`` options: ``"each"`` a faint per-mouse
        line across the two decay-model x-positions plus a bold across-mouse mean; ``"errorPlot"`` the
        across-mouse mean +/- SE band; ``"swarm"`` one beeswarm column per (decay model, curve) with a
        short horizontal mean line (spread set by ``beewidth``).
    normalize : bool
        If True, normalize each gs[-1] spectrum by its sum (per session) before smoothing.
    clip_negative : bool
        If True, replace each gs[-1] spectrum's first negative entry and every later rank with NaN
        before further processing.
    legend : dict or None
        gs[-1] legend styling, seeded via :func:`~..legends.update_legend_widgets` (``{knob:
        value}``, see :data:`~..legends.LEGEND_KNOBS`). None leaves the defaults (``loc="auto"``:
        "best" for the ``"characteristic_dim"``/``"mse"`` options, no legend for
        ``"decay_exponent"``, whose columns are labelled on the x-axis).
    **selection_defaults
        Starting values for the shared param-axis widgets (:data:`~._param_axes.PREFERRED_DEFAULTS`
        remains the fallback for axes not named here), keyed by raw ``param_axes`` name of ``results``
        (``activity_parameters_name``) or of ``results_spectra``/``results_cvpca`` (e.g.
        ``smooth_widths``, ``reliability_fraction_active_thresholds``).
    fontsize : float
        Base font size, threaded explicitly into every label/tick/legend.
    figsize : tuple[float, float]
        Figure size in inches.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_spectra: ResultsAggregator | None = None,
        results_cvpca: ResultsAggregator | None = None,
        source_cfg: AdaptiveAlphaConfig | None = None,
        num_bins: int = 80,
        fraction_view: str = "pooled",
        beewidth: float = 0.2,
        source_key: str = "ss_cv",
        metric: str = "r2",
        generalized_fit: str = "both",
        include_better: bool = True,
        paired_test: str = "ttest",
        last_axis: str = "decay_exponent",
        fit_zone: str = "adaptive",
        fixed_range: tuple[int, int] = (10, 50),
        display: str = "each",
        normalize: bool = True,
        clip_negative: bool = False,
        legend: dict | None = None,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (8.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.results_spectra = results_spectra
        self.results_cvpca = results_cvpca
        # Alias: the fit-aggregator helper methods below (adapted from SpectrumFigureViewer) fetch
        # the eig spectra from ``self.results_fit``, which is this viewer's Tilbury-fit aggregator.
        self.results_fit = results
        self.source_cfg = source_cfg if source_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]
        # Reused by _spectrum_sessions/_sel_params: the source_key spectrum for the gs[-1] alpha
        # panel comes from these, resolved via SOURCE_OF_KEY.
        self._agg = {"stimspace": results_spectra, "cvpca": results_cvpca}
        self.config = results.config_class
        self.fontsize = fontsize
        self.figsize = figsize

        # Bin count for the per-session KDE of the peak-exponent density (gs[0]).
        self.add_integer("num_bins", value=num_bins, min=5, max=200)
        self.add_selection("fraction_view", options=["pooled", "by_mouse", "none"], value=fraction_view)
        self.add_float("beewidth", value=beewidth, min=0.0, max=1.0, step=0.01)
        self.add_selection("metric", value=metric, options=["r2", "cc"])
        # Which generalized fit to show. "both" plots the unregularized (blue) and the shrinkage
        # (purple) fits side by side. "generalized"/"shrinkage" plot only that one fit, labelled
        # "Generalized". "include_better" toggles the per-neuron "better" composite in gs[-1].
        self.add_selection("generalized_fit", options=["both", "generalized", "shrinkage"], value=generalized_fit)
        self.add_boolean("include_better", value=include_better)
        # Paired test for the gs[1] fit-vs-Gaussian comparison(s): two-sided paired t-test or
        # Wilcoxon signed-rank, on the per-mouse averages. Bonferroni-corrected when "both".
        self.add_selection("paired_test", options=["ttest", "wilcoxon"], value=paired_test)

        # --- gs[-1] population spectrum-decay panel: source_key spectrum + eig fit spectra ---
        self.add_selection("last_axis", options=["decay_exponent", "characteristic_dim", "mse"], value=last_axis)
        self.add_selection("fit_zone", options=["adaptive", "fixed"], value=fit_zone)
        self.add_integer_range("fixed_range", value=tuple(int(v) for v in fixed_range), min=1, max=500)
        self.add_selection("display", options=["each", "errorPlot", "swarm"], value=display)
        # gs[-1] legend styling, available under every ``last_axis`` option.
        add_legend_widgets(self)
        if legend is not None:
            update_legend_widgets(self, legend)

        # source_key options mirror SpectrumFigureViewer (StimSpace keys, plus the CVPCA key when
        # given). Only offered when there is a spectra aggregator to slice.
        if results_spectra is not None:
            source_options = list(STIMSPACE_KEYS) + (list(CVPCA_KEYS) if results_cvpca is not None else [])
            self.add_selection("source_key", options=source_options, value=source_key)

        # One widget per shared param-axis name (tuple-valued axes encoded as string labels), so the
        # source_key spectrum can be sliced (activity_parameters_name, smooth_widths, ...). The
        # TilburyFitConfig axes are merged in too: every panel here slices the fit results, so each
        # of its axes must be pinned by a widget as well.
        self._fit_axes = list(results.param_axes)
        merged_defaults = {**PREFERRED_DEFAULTS, **selection_defaults}
        self._tuple_labels = add_merged_param_axis_widgets(
            self,
            results_spectra,
            results_cvpca,
            results,
            preferred_defaults=merged_defaults,
        )
        self._merged_axis_names: list[str] = []
        for agg in (results_spectra, results_cvpca, results):
            if agg is None:
                continue
            for name in agg.param_axes:
                if name not in self._merged_axis_names:
                    self._merged_axis_names.append(name)

        # Adaptive median-FPD estimation controls, shared by every gs[-1] curve: one widget per
        # AdaptiveAlphaConfig field (the "source"-prefixed scheme also used by SpectrumFigureViewer).
        self.add_boolean("normalize", value=normalize)
        self.add_boolean("clip_negative", value=clip_negative)
        cfg = self.source_cfg
        self.add_selection("source_smooth_method", options=["none", "boxcar", "gaussian"], value=cfg.smooth_method)
        self.add_float("source_smooth_width", value=cfg.smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_integer("source_fpd_window_size", value=cfg.fpd_window_size, min=1, max=50)
        self.add_integer("source_adaptive_buffer", value=cfg.adaptive_buffer, min=0, max=50)
        self.add_integer("source_minimum_window_size", value=cfg.minimum_window_size, min=1, max=500)

        # Widgets that change which numbers get computed (data selection, fit-spectrum fetches, the
        # adaptive alpha / decay-law estimates) trigger a recompute; pure style knobs (beewidth,
        # fraction_view's layout, display, legend placement, fontsize) are read directly by plot().
        data_triggers = [
            "num_bins",
            "metric",
            "generalized_fit",
            "include_better",
            "last_axis",
            "fit_zone",
            "fixed_range",
            "normalize",
            "clip_negative",
            "source_smooth_method",
            "source_smooth_width",
            "source_fpd_window_size",
            "source_adaptive_buffer",
            "source_minimum_window_size",
            *self._merged_axis_names,
        ]
        if results_spectra is not None:
            data_triggers.append("source_key")
        for name in data_triggers:
            self.on_change(name, self.refresh_data)

        self.refresh_data(self.state)

    def encode_param(self, name: str, value):
        """Map a raw param value to its widget value (tuple -> string label; else unchanged)."""
        return _encode_param(self._tuple_labels, name, value)

    def _sel_params(self, state: dict, source: str) -> dict:
        """Select the params relevant to ``source`` ("stimspace"/"cvpca"), decoding tuple labels."""
        return _sel_params_for(state, self._tuple_labels, self._agg[source].param_axes)

    def _fit_sel_params(self, state: dict) -> dict:
        """Params pinning every Tilbury-fit param axis, decoding tuple labels back to tuples."""
        return _sel_params_for(state, self._tuple_labels, self._fit_axes)

    @staticmethod
    def _cfg_from_state(state: dict, prefix: str) -> AdaptiveAlphaConfig:
        """Build the adaptive-fit config from the ``{prefix}_*`` widgets."""
        return AdaptiveAlphaConfig(
            smooth_method=state[f"{prefix}_smooth_method"],
            smooth_width=state[f"{prefix}_smooth_width"],
            fpd_window_size=int(state[f"{prefix}_fpd_window_size"]),
            adaptive_buffer=int(state[f"{prefix}_adaptive_buffer"]),
            minimum_window_size=int(state[f"{prefix}_minimum_window_size"]),
        )

    def _spectrum_sessions(
        self,
        state: dict,
        key: str,
        cfg: AdaptiveAlphaConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Per-session raw and smoothed ``(sessions, dims)`` spectrum for ``key``, with mouse/session ids.

        Normalize is applied per session (row), so the adaptive alpha fit can find each session's
        own first-negative crossover before any cross-session averaging blurs it. Both the raw
        (pre-smoothing) and smoothed spectrum are returned: smoothing maps every non-positive entry
        to NaN before exponentiating back, so a smoothed row is never negative -- first-negative
        detection must use the raw one, while the exponent fit itself uses the smoothed one.
        """
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        spec = agg.sel(keys=[key], avg_by_mouse=False, **self._sel_params(state, source))[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state.get("clip_negative", False):
            spec = _clip_at_first_negative(spec)
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)
        return spec, smoothed, agg.mouse_names, agg.session_ids

    def _fit_spectrum_raw_sessions(self, state: dict, key: str) -> np.ndarray:
        """Raw (unnormalized, unsmoothed) per-session Tilbury-fit eigenvalue spectrum for ``key``.

        ``key`` is one of :data:`~._param_axes.POP_EIG_KEYS` from the Tilbury-fit aggregator. Each
        session is converted from the PCA convention to the ``ss_cv`` covariance convention with
        ``P / (P - 1)``, gated by whichever ``params*`` arrays back ``key``
        (:data:`~._param_axes.FIT_KEY_PARAM_KEYS`).
        """
        fit_params = self._fit_sel_params(state)
        param_keys = FIT_KEY_PARAM_KEYS[key]
        selected = self.results_fit.sel(
            keys=[key, *param_keys],
            squeeze_ones=False,
            **fit_params,
        )
        dist_centers = self.results_fit.sel_objects(
            keys=["dist_centers"],
            **fit_params,
        )["dist_centers"]
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

    def _fit_spectrum_sessions(
        self,
        state: dict,
        key: str,
        cfg: AdaptiveAlphaConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Per-session raw+smoothed Tilbury-fit eigenvalue spectrum for ``key``, with mouse/session ids."""
        session_spec = self._fit_spectrum_raw_sessions(state, key)
        if state.get("clip_negative", False):
            session_spec = _clip_at_first_negative(session_spec)
        if state["normalize"]:
            session_spec = session_spec / np.nansum(session_spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(session_spec, cfg.smooth_method, cfg.smooth_width)
        return session_spec, smoothed, self.results_fit.mouse_names, self.results_fit.session_ids

    def _aggregate_stats(self, state: dict) -> dict:
        """Per-session and per-mouse summary arrays for the gs[0]/gs[1]/gs[2] population panels.

        Covers all three fits: the unregularized generalized Gaussian (``params`` / ``*_test``), the
        plain-Gaussian control (``*_test_control``) and the generalized-shrinkage fit at its
        validation-selected lambda (``params_shrinkage`` / ``*_test_shrinkage``).
        """
        sel = self.results.sel(
            keys=[
                "params",
                "params_shrinkage",
                "r2_test",
                "r2_test_control",
                "r2_test_shrinkage",
                "pearson_test",
                "pearson_test_control",
                "pearson_test_shrinkage",
                "idx_keep",
            ],
            load_ragged=True,
            squeeze_ones=False,
            **self._fit_sel_params(state),
        )
        params = sel["params"]  # (n_sess, N, 6)
        params_shrinkage = sel["params_shrinkage"]  # (n_sess, N, 6)
        _suffix = "r2_test" if state["metric"] == "r2" else "pearson_test"
        performance_test = sel[_suffix]  # (n_sess, N)
        performance_test_control = sel[f"{_suffix}_control"]  # (n_sess, N)
        performance_test_shrinkage = sel[f"{_suffix}_shrinkage"]  # (n_sess, N)
        idx_keep = sel["idx_keep"]  # (n_sess,) object array of bool masks

        # Drop sessions with less than 200 fitted neurons (all-NaN r2 rows).
        idx_valid = np.sum(~np.isnan(performance_test), axis=-1) >= 200
        idx_peak = self.config.param_names.index("p")
        peak = params[..., idx_peak][idx_valid]
        # param_names_shrinkage matches param_names (same generalized-Gaussian layout), so the
        # exponent lives in the same column.
        peak_shrinkage = params_shrinkage[..., idx_peak][idx_valid]
        performance_test = performance_test[idx_valid]
        performance_test_control = performance_test_control[idx_valid]
        performance_test_shrinkage = performance_test_shrinkage[idx_valid]
        idx_keep = idx_keep[idx_valid]
        mouse_names = self.results.mouse_names[idx_valid]

        # Per-session median test R2 (shrinkage, generalized, gaussian -- the gs[1] column order) and
        # the fraction of kept neurons where the generalized fit beats the Gaussian.
        avg_performance = np.full((performance_test.shape[0], 3), np.nan)
        avg_performance[:, 0] = np.nanmedian(performance_test_shrinkage, axis=1)
        avg_performance[:, 1] = np.nanmedian(performance_test, axis=1)
        avg_performance[:, 2] = np.nanmedian(performance_test_control, axis=1)
        improvement = performance_test - performance_test_control
        fraction_better = np.full(performance_test.shape[0], np.nan)
        for i, imp in enumerate(improvement):
            num_keep = int(np.nansum(idx_keep[i]))
            if num_keep > 0:
                fraction_better[i] = np.nansum(imp > 0) / num_keep

        # Per-session KDE of the peak exponent over a fixed [0, 10] grid, for both generalized fits.
        edges_peak = np.linspace(0.0, 10.0, state["num_bins"] + 1)
        centers_peak = edge2center(edges_peak)

        def _densities(peaks: np.ndarray) -> np.ndarray:
            density = np.full((peaks.shape[0], len(centers_peak)), np.nan)
            for i, row in enumerate(peaks):
                row = row[np.isfinite(row)]
                if len(row) < 2:
                    continue
                density[i] = gaussian_kde(row)(centers_peak)
            return density

        mouse_avg_performance, mouse_avg_names = average_by_mouse(avg_performance, mouse_names, include_mouse_names=True)
        return {
            "centers_peak": centers_peak,
            "mouse_density_peak": average_by_mouse(_densities(peak), mouse_names),
            "mouse_density_peak_shrinkage": average_by_mouse(_densities(peak_shrinkage), mouse_names),
            "mouse_avg_performance": mouse_avg_performance,
            "mouse_fraction_better": average_by_mouse(fraction_better, mouse_names),
            "fraction_better": fraction_better,
            "mouse_names": mouse_names,
            "mouse_avg_names": mouse_avg_names,
        }

    def _compute_last_axis(self, state: dict) -> dict:
        """Curves, colors, labels and the selected decay statistic for the gs[-1] panel.

        Fetches the source_key spectrum (if any) and every wanted eig fit spectrum
        (:data:`~._param_axes.POP_EIG_KEYS`, filtered by ``generalized_fit``/``include_better``),
        then reduces each curve's per-session spectra to the statistic named by ``state["last_axis"]``:
        the adaptive median-FPD exponent, or a decay-law fit's characteristic parameter / MSE.
        """
        cfg = self._cfg_from_state(state, "source")
        fit_zone = state["fit_zone"]
        fixed_range = tuple(int(v) for v in state["fixed_range"])
        fit_sel = state["generalized_fit"]

        # Fixed fallback window source: ss_cvpca, fetched unconditionally (harmless self-fallback
        # when source_key already is that key). Keys that aren't cross-validated have no negative
        # entry to locate a noise floor with, so they borrow that session's ss_cvpca window.
        cvpca = self._spectrum_sessions(state, "ss_cvpca", cfg) if self.results_spectra is not None else None

        def _fallback_rows(session_ids) -> tuple[np.ndarray | None, np.ndarray | None]:
            """The ss_cvpca (raw, smoothed) rows aligned to ``session_ids``, for window borrowing."""
            if cvpca is None:
                return None, None
            cvpca_raw, cvpca_smooth, _, cvpca_session_ids = cvpca
            return (
                _align_rows_to_sessions(session_ids, cvpca_session_ids, cvpca_raw),
                _align_rows_to_sessions(session_ids, cvpca_session_ids, cvpca_smooth),
            )

        def _adaptive_alpha(raw: np.ndarray, smooth: np.ndarray, mouse_names, session_ids) -> np.ndarray:
            """Per-mouse adaptive exponent for one key's per-session raw/smoothed spectrum."""
            fb_raw, fb_smooth = _fallback_rows(session_ids)
            return average_by_mouse(
                _median_fpd_alpha_per_session(
                    raw,
                    smooth,
                    cfg.fpd_window_size,
                    cfg.adaptive_buffer,
                    cfg.minimum_window_size,
                    fb_raw,
                    fb_smooth,
                ),
                mouse_names,
            )

        def _decay_stats(raw: np.ndarray, smooth: np.ndarray, mouse_names, session_ids) -> dict[str, np.ndarray]:
            """Per-mouse decay-law statistics for one key, as ``(n_mice, 2)`` arrays.

            The two columns are the power-law and exponential fits (:data:`DECAY_MODELS`' order);
            ``"characteristic_dim"`` is that model's parameter (``alpha`` / ``M``) and ``"mse"`` its
            log-space MSE.
            """
            fb_raw, fb_smooth = _fallback_rows(session_ids) if fit_zone == "adaptive" else (None, None)
            mse_cols, param_cols = [], []
            for model_key, _ in DECAY_MODELS:
                mse_s, param_s = _decay_fit_per_session(raw, smooth, model_key, fit_zone, fixed_range, cfg.adaptive_buffer, fb_raw, fb_smooth)
                mse_cols.append(average_by_mouse(mse_s, mouse_names))
                param_cols.append(average_by_mouse(param_s, mouse_names))
            return {"characteristic_dim": np.stack(param_cols, axis=1), "mse": np.stack(mse_cols, axis=1)}

        # Assemble the curves: the source_key data spectrum, then the eig fit spectra -- "better"
        # composite (if requested), the selected generalized fit(s), then the gaussian control,
        # keeping POP_EIG_KEYS' order. A single-fit selection draws that fit blue and labelled
        # "Generalized" (matching gs[0]/gs[1], which instead draw it black -- see class docstring).
        curves: list[tuple[tuple, str, str]] = []  # (per-session spectra tuple, color, label)
        if self.results_spectra is not None:
            spectra = self._spectrum_sessions(state, state["source_key"], cfg)
            curves.append((spectra, POP_ALPHA_COLORS["source_key"], "Data"))

        def _wanted(key: str) -> bool:
            if key == "eig_better":
                return state["include_better"]
            if key == "eig_shrinkage":
                return fit_sel in ("both", "shrinkage")
            if key == "eig_tilbury":
                return fit_sel in ("both", "generalized")
            return True  # eig_control is always shown

        eig_keys = [key for key in POP_EIG_KEYS if _wanted(key)]
        for key in eig_keys:
            if fit_sel != "both" and key in ("eig_shrinkage", "eig_tilbury"):
                color, label = POP_ALPHA_COLORS["eig_tilbury"], "Generalized"
            else:
                color, label = POP_ALPHA_COLORS[key], POP_ALPHA_LABELS[key]
            curves.append((self._fit_spectrum_sessions(state, key, cfg), color, label))

        colors = [color for _, color, _ in curves]
        labels = [label for _, _, label in curves]
        last_axis = state["last_axis"]
        result = {"last_axis": last_axis, "colors": colors, "labels": labels}
        if last_axis == "decay_exponent":
            result["values"] = [_adaptive_alpha(*spectra) for spectra, _, _ in curves]
        else:
            result["values"] = [_decay_stats(*spectra)[last_axis] for spectra, _, _ in curves]
        return result

    def refresh_data(self, state: dict) -> None:
        """Recompute the population statistics and the gs[-1] curves for the current selection."""
        self._stats = self._aggregate_stats(state)
        self._last_axis_result = self._compute_last_axis(state)

    def plot(self, state: dict):
        stats = self._stats
        fontsize = self.fontsize

        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        num_cols = 4 if state["fraction_view"] != "none" else 3
        width_ratios = [1, 0.65, 1, 1] if state["fraction_view"] != "none" else [1, 0.65, 1]
        outer = fig.add_gridspec(1, num_cols, width_ratios=width_ratios)

        # --- gs[0]: per-mouse peak-exponent density + across-mouse mean, reference at p=2 ---
        # Both generalized fits are shown: the unregularized one (blue) and the shrinkage one
        # (purple), whose penalty pulls p toward the Gaussian value of 2.
        ax1 = fig.add_subplot(outer[0, 0])
        centers_peak = stats["centers_peak"]
        fit_sel = state["generalized_fit"]
        if fit_sel == "both":
            peak_densities = (
                (stats["mouse_density_peak"], POP_ALPHA_COLORS["eig_tilbury"], "Generalized"),
                (stats["mouse_density_peak_shrinkage"], POP_ALPHA_COLORS["eig_shrinkage"], "Shrinkage"),
            )
        elif fit_sel == "shrinkage":
            peak_densities = ((stats["mouse_density_peak_shrinkage"], "black", "Generalized"),)
        else:  # "generalized"
            peak_densities = ((stats["mouse_density_peak"], "black", "Generalized"),)
        for density, color, label in peak_densities:
            ax1.plot(centers_peak, density.T, color=color, linewidth=0.8, alpha=0.3)
            ax1.plot(centers_peak, np.nanmean(density, axis=0), color=color, linewidth=2.0, label=label)
        ax1.axvline(x=2.0, color="k", linestyle=":", linewidth=0.8)
        ax1.set_xlabel("Peak Exponent", fontsize=fontsize)
        ax1.set_ylabel("Density", fontsize=fontsize)
        if fit_sel == "both":
            ax1.legend(fontsize=fontsize * 0.8, frameon=False, loc="upper right")
        style_model_axis(
            ax1,
            fontsize=fontsize,
            xbounds=(0, 10),
            ybounds=(0, np.round(max(np.nanmax(d) for d, _, _ in peak_densities), 2)),
            spines_visible=["bottom", "left"],
            xticks=[0, 2, 4, 6, 8, 10],
        )

        # --- gs[1]: per-mouse median test R2, shrinkage vs generalized vs gaussian, paired ---
        ax2 = fig.add_subplot(outer[0, 1])
        # mouse_avg_performance columns are [shrinkage, generalized, gaussian]; pick the selected
        # generalized fit (or both) plus the gaussian control, relabelling the chosen fit "Generalized".
        if fit_sel == "both":
            perf_cols, perf_labels = [0, 1, 2], ["Shrinkage", "Generalized", "Gaussian"]
        elif fit_sel == "shrinkage":
            perf_cols, perf_labels = [0, 2], ["Generalized", "Gaussian"]
        else:  # "generalized"
            perf_cols, perf_labels = [1, 2], ["Generalized", "Gaussian"]
        mouse_avg_performance = stats["mouse_avg_performance"][:, perf_cols]
        x_models = list(range(len(perf_cols)))
        ax2.plot(x_models, mouse_avg_performance.T, color="0.7", marker="o", markersize=3, linewidth=0.8)
        ax2.plot(x_models, np.nanmean(mouse_avg_performance, axis=0), color="k", marker="o", markersize=5, linewidth=2.0)
        ax2.set_ylabel("Test R²" if state["metric"] == "r2" else "Test Correlation", fontsize=fontsize)
        ylims = ax2.get_ylim()
        if state["metric"] == "cc":
            ymin = min(0, ylims[0])
            ymax = 1
        else:
            ymin = ylims[0]
            ymax = ylims[1]
        # Headroom above the data for the significance asterisks (spine still bounded at ymax).
        y_headroom = 0.05 * (ymax - ymin)
        ax2.set_xlim(-0.5, len(x_models) - 0.5)
        ax2.set_ylim(ymin, ymax + y_headroom)
        # Not panels.style_model_axis here: it sets rotation_mode="anchor" on the categorical tick
        # labels, which shifts the rotated label position relative to the original rotation_mode
        # default -- see the migration report for this behavior-preservation call.
        format_spines(
            ax2,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=(0, len(x_models) - 1),
            ybounds=(ymin, ymax),
            spines_visible=["bottom", "left"],
            xticks=x_models,
            tick_fontsize=fontsize,
        )
        ax2.set_xticklabels(perf_labels, rotation=45, ha="right", fontsize=fontsize)

        # Planned two-sided paired comparison of each generalized fit against the Gaussian control
        # (always the last column). Bonferroni-corrected for the number of comparisons (1 when a
        # single fit is shown, 2 when "both"). An asterisk tier is drawn over each generalized column.
        gaussian_perf = mouse_avg_performance[:, -1]
        n_comparisons = mouse_avg_performance.shape[1] - 1
        for i in range(n_comparisons):
            p = _paired_pvalue(mouse_avg_performance[:, i], gaussian_perf, state["paired_test"])
            p_corrected = min(p * n_comparisons, 1.0) if np.isfinite(p) else np.nan
            ax2.text(
                x_models[i],
                ymax,
                _significance_stars(p_corrected),
                ha="center",
                va="bottom",
                fontsize=fontsize * 1.3,
                fontweight="bold",
            )

        # --- gs[2]: fraction generalized > gaussian, pooled or broken down by mouse ---
        if state["fraction_view"] != "none":
            ax3 = fig.add_subplot(outer[0, 2])
            beewidth = state["beewidth"]
            if state["fraction_view"] == "pooled":
                vals = stats["mouse_fraction_better"]
                xbounds = (0, 0)
                offsets = beeswarm(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else np.zeros_like(vals)
                x = np.zeros_like(vals)
                x[np.isfinite(vals)] = beewidth * offsets
                ax3.plot(x, vals, linestyle="none", color="black", marker="o", markersize=4, alpha=0.8)
                ax3.plot([-0.25, 0.25], [np.nanmean(vals)] * 2, color="black", linewidth=2.0)
                ax3.set_xlim(-0.5, 0.5)
                xticks = []
            else:
                # One beeswarm per mouse, sorted by mean fraction from highest to lowest.
                mouse_names = stats["mouse_names"]
                mice = list(dict.fromkeys(mouse_names))
                xbounds = (0, len(mice) - 1)
                mice.sort(key=lambda m: np.nanmean(stats["fraction_better"][mouse_names == m]), reverse=True)
                for xi, mouse in enumerate(mice):
                    vals = stats["fraction_better"][mouse_names == mouse]
                    finite = np.isfinite(vals)
                    offsets = np.zeros_like(vals)
                    if finite.any():
                        offsets[finite] = beeswarm(vals[finite])
                    ax3.plot(xi + beewidth * offsets, vals, linestyle="none", color="black", marker=".", markersize=5, alpha=0.3)
                    ax3.plot(xi + np.array([-0.4, 0.4]), [np.nanmean(vals)] * 2, color="black", linewidth=1.2)
                ax3.set_xlim(-1.0, len(mice))
                ax3.set_xlabel("Mice", fontsize=fontsize)
                xticks = range(len(mice))

            ax3.set_ylim(0, 1)
            ax3.set_ylabel("Fraction Cells\nGeneralized > Gaussian", fontsize=fontsize)
            style_model_axis(
                ax3,
                fontsize=fontsize,
                xbounds=xbounds,
                ybounds=(0, 1),
                spines_visible=["bottom", "left"],
                yticks=[0, 0.5, 1],
            )
            ax3.set_xticks(xticks, labels=[])

        # --- gs[-1]: across-mouse spectrum decay statistic, source_key spectrum + eig fit spectra ---
        ax4 = fig.add_subplot(outer[0, -1])
        last = self._last_axis_result
        last_axis = last["last_axis"]
        colors, labels = last["colors"], last["labels"]
        if last_axis == "decay_exponent":
            _beeswarm_panel(ax4, last["values"], colors, labels, fontsize, state["beewidth"])
            ax4.set_ylabel("Decay exponent", fontsize=fontsize)
            # The beeswarm columns are labelled on the x-axis, so no legend unless one is asked for --
            # its points carry no labels, hence the proxy handles.
            handles = [Line2D([], [], color=color, marker="o", markersize=3, linestyle="none") for color in colors]
            apply_legend(ax4, state, fontsize, handles=handles, labels=labels)
        else:
            data_list = last["values"]
            xtick_labels = [lbl for _, lbl in DECAY_MODELS]
            # The characteristic dimension is non-negative: span [0, max] with ticks every 5 units.
            ybounds, yticks = _zero_to_max_ticks(data_list) if last_axis == "characteristic_dim" else (None, None)
            _decay_stat_panel(
                ax4,
                data_list,
                colors,
                labels,
                state["display"],
                state["beewidth"],
                fontsize,
                xtick_labels,
                ybounds=ybounds,
                yticks=yticks,
            )
            ax4.set_xlabel("Fit Type", fontsize=fontsize)
            ax4.set_ylabel("Dimensionality" if last_axis == "characteristic_dim" else "Log-space MSE", fontsize=fontsize)
            apply_legend(ax4, state, fontsize, auto_loc="best")
        return fig
