"""Placefield vs. Full-CA1 dimensionality (or decay exponent) over session number."""

import numpy as np

from vrAnalysis.helpers.plotting import errorPlot
from dimensionality_manuscript import ResultsAggregator
from dimensionality_manuscript.env_order import ENV_SLOT_COLORS
from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, apply_legend, update_legend_widgets

from ._alpha_config import ADAPTIVE_ALPHA_CONFIG_REGISTRY, AdaptiveAlphaConfig
from ._param_axes import (
    CVPCA_KEYS,
    FF_KEY,
    PER_ENV_PF_KEYS as _PER_ENV_PF_KEYS,
    PER_ENV_PF_RESULT_KEYS as _PER_ENV_PF_RESULT_KEYS,
    PER_ENV_RESIDUAL_RESULT_KEYS as _PER_ENV_RESIDUAL_RESULT_KEYS,
    PER_ENV_SESSION_ALIGNMENTS as _PER_ENV_SESSION_ALIGNMENTS,
    PREFERRED_DEFAULTS,
    SOURCE_OF_KEY,
    STIMSPACE_KEYS,
    add_merged_param_axis_widgets,
    full_key_options as _full_key_options,
    per_env_full_options as _per_env_full_options,
    sel_params,
)
from ._spectrum_math import (
    _align_rows_to_sessions,
    _clip_at_first_negative,
    _median_fpd_alpha_per_session,
    _signed_participation_ratio,
    _smooth_spectrum,
    _truncated_participation_ratio,
)
from ..panels import FigureViewer, style_model_axis


SVCA_PF_KEYS = {"SVCA": "ss", "SVCA_PRED": "ss_pred"}
SVCA_PF_ENV_KEYS = {"SVCA": "ss_env", "SVCA_PRED": "ss_pred_env"}
SVCA_FULL_KEYS = {"SVCA": "ff", "SVCA_RES": "ff_res"}
SVCA_FULL_ENV_KEYS = {"SVCA": "ff_env", "SVCA_RES": "ff_res_env"}
LEGACY_SVCA_PF_KEYS = {"SVCA": "variance_placefield_placefield", "SVCA_PRED": "variance_placefield_prediction"}
LEGACY_SVCA_PF_ENV_KEYS = {
    "SVCA": "variance_placefield_placefield_env",
    "SVCA_PRED": "variance_placefield_prediction_env",
}
LEGACY_SVCA_FULL_KEYS = {"SVCA": "variance_activity", "SVCA_RES": "variance_activity_residual"}
LEGACY_SVCA_FULL_ENV_KEYS = {"SVCA": "variance_activity_env", "SVCA_RES": "variance_activity_residual_env"}


def _ordinal(number: int) -> str:
    """Return a compact English ordinal label (1st, 2nd, 3rd, ...)."""
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


class DimensionalityFamiliarityViewer(FigureViewer):
    """Participation-ratio dimensionality or adaptive decay exponent over session number.

    This is the familiarity analogue of ``SpectrumFigureViewer``: one selected placefield
    spectrum and one selected Full-CA1 spectrum are reduced to either a signed participation ratio
    or adaptive median-FPD decay exponent for every session. Sessions are then sorted
    chronologically within mouse and the two sources are reindexed onto the union of their session
    IDs, so missing coverage leaves a NaN gap instead of shifting a curve to the wrong session
    number.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, supplying the ``ss_*``/``sf_*`` PF spectra and the
        SVD Full-CA1 ``ff`` spectrum.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results. When provided, ``reg_covariances_fixed`` is added to the
        selectable PF ``source_key`` options.
    results_svca : ResultsAggregator or None
        Aggregated StimspaceSVCAConfig results. Adds ``SVCA``/``SVCA_PRED`` as PF sources backed
        by ``ss``/``ss_pred`` and ``SVCA``/``SVCA_RES`` as Full sources backed by ``ff``/``ff_res``.
    results_subspace : ResultsAggregator or None
        Legacy SubspaceConfig backend for the same public SVCA choices, using its ``variance_*``
        keys. When both backends are supplied, ``results_svca`` takes precedence.
    source_cfg, full_cfg : AdaptiveAlphaConfig or None
        Independent adaptive-alpha configurations for the PF and Full-CA1 curves, defaulting to
        ``ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]``/``["full"]``. Each seeds its own Syd
        smoothing, FPD-window, buffer, and minimum-window-size controls (``source_*``/``full_*``
        widgets below); these settings are used only for ``metric="alpha"``. Environment-derived PF
        spectra use ``ss_cvpca_env`` as their context-matched fallback (averaged across slots for
        ``"avg_env"``); Full spectra use ``ff_env`` (SVCA), aligned by both
        session and environment slot.
    source_key : str
        Placefield spectrum key: one of ``STIMSPACE_KEYS``, plus ``CVPCA_KEYS`` when
        ``results_cvpca`` is given and ``SVCA``/``SVCA_PRED`` when ``results_svca`` is given.
    full_key : {"SVD", "SVD_RES", "SVCA", "SVCA_RES"}
        Full-CA1 estimator for both plot modes. ``SVD_RES`` is exposed only when residual spectra
        are discoverable; ``SVCA``/``SVCA_RES`` require ``results_svca`` (``SVCA_RES`` also
        requires ``ff_res`` to be discoverable there). In ``"avg_env"``/``"by_env"``, they read
        the per-environment ``ff_env``/``ff_res_env`` keys.
    full_scope : {"full1", "fullall"}
        Functional-data scope for environment-derived spectra. ``"full1"`` uses the selected
        environment alone; ``"fullall"`` compares it with all-session functional activity.
        In ``"by_env"`` and ``"avg_env"`` this selects the environment-specific Full SVD/SVD_RES
        spectrum and the paired SF source. It is ignored for SVCA and an ``"all"`` PF source.
    plot_mode : {"all", "avg_env", "by_env"}
        ``"all"`` uses the overall PF and Full spectra. ``"avg_env"`` averages both selected
        per-environment spectra across available environment slots within each session.
        ``"by_env"`` plots each environment slot separately.
    by_env_layout : {"row", "col", "shared"}
        Arrange the two ``plot_mode="by_env"`` panels side by side (``"row"``) or vertically
        (``"col"``), or draw both sources on one axis (``"shared"``). The vertical layout places
        Full/PF Residual above the placefields and shares their x-axis, showing its spine, ticks,
        and label only on the bottom panel.
    session_alignment : {"within_env", "overall"}
        Per-environment x values: densified within-environment session number or the mouse's
        overall chronological session number.
    metric : {"participation_ratio", "truncated_pr", "alpha"}
        Per-session spectrum summary. ``"participation_ratio"`` uses the signed whole-spectrum
        participation ratio. ``"truncated_pr"`` uses ordinary PR for the selected PF spectrum and
        computes Full-CA1 PR only over ranks ``k:end``. ``"alpha"`` uses the adaptive median-FPD
        decay exponent: PF window boundaries fall back to ``ss_cvpca`` and Full-CA1 boundaries
        fall back to SVCA when the selected spectrum has no negative crossover.
    k_method : {"hardcode", "adaptive"}
        How ``k`` is chosen for ``metric="truncated_pr"``. ``"hardcode"`` uses ``k_value``.
        ``"adaptive"`` uses the paired PF spectrum's ordinary PR, rounded to the nearest integer,
        independently for every session and environment slot.
    k_value : int
        Hardcoded truncation rank in the inclusive range 0--100. Ignored for adaptive ``k``.
    display : {"each", "errorPlot"}
        ``"each"`` draws every mouse as a faint line plus the across-mouse mean.
        ``"errorPlot"`` draws the mean +/- SE band. Population summaries require at least two
        mice at a session number.
    log_y : bool
        Use a logarithmic y-axis.
    ylim_start_at_one : bool
        Pin the lower y-axis limit to 1 while retaining the autoscaled upper limit.
    sharey : bool
        Share the PF and Full-CA1 y-axis in ``plot_mode="by_env"``.
    clip_negative : bool
        Replace each spectrum's first negative entry and all later ranks with NaN before use.
    pr_pre_smooth : bool
        For participation-ratio metrics, use the raw spectrum (True) or apply ``source_cfg``'s /
        ``full_cfg``'s smoothing first (False).
    fontsize : float
        Font size for axis labels, tick labels, titles, and the legend.
    figsize : tuple[float, float]
        Figure size in inches.
    pf_color, ff_color : str
        Colors of the placefield and Full-CA1 curves.
    pf_label, ff_label : str
        Legend labels of the placefield and Full-CA1 curves.
    pf_text_x, pf_text_y, ff_text_x, ff_text_y : float
        Independent axes-fraction positions for the Placefields and Full/PF Residual panel text
        in ``plot_mode="by_env"``.
    legend_anchor_x, legend_anchor_y : float
        Axes-fraction offsets for the legend's anchor box in the by-environment layout. ``(0, 0)``
        retains the normal position. A nonzero offset makes the legend an overlay, so constrained
        layout does not move the axes to accommodate it.
    legend_options : dict or None
        Legend knobs forwarded to :mod:`~dimensionality_manuscript.figure_scripts.legends`
        (``{"loc": ..., "ncols": ...}``); ``{"loc": "none"}`` hides it.
    **param_defaults
        Starting values for the shared data-selection widgets built from ``results`` and
        ``results_cvpca``'s param axes (keyed by raw ``param_axes`` name), overriding
        ``PREFERRED_DEFAULTS``.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        results_svca: ResultsAggregator | None = None,
        *,
        source_cfg: AdaptiveAlphaConfig | None = None,
        full_cfg: AdaptiveAlphaConfig | None = None,
        source_key: str = "ss_cv",
        full_key: str = "SVD",
        full_scope: str = "full1",
        plot_mode: str = "all",
        by_env_layout: str = "row",
        session_alignment: str = "within_env",
        metric: str = "participation_ratio",
        k_method: str = "hardcode",
        k_value: int = 0,
        display: str = "errorPlot",
        log_y: bool = False,
        ylim_start_at_one: bool = False,
        sharey: bool = False,
        clip_negative: bool = False,
        pr_pre_smooth: bool = True,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (3.5, 2.5),
        pf_color: str = "orange",
        ff_color: str = "black",
        pf_label: str = "Placefields",
        ff_label: str = "Full CA1",
        pf_text_x: float = 0.05,
        pf_text_y: float = 0.9,
        ff_text_x: float = 0.05,
        ff_text_y: float = 0.9,
        legend_anchor_x: float = 0.0,
        legend_anchor_y: float = 0.0,
        legend_options: dict | None = None,
        **param_defaults,
    ):
        svca_results = results_svca if results_svca is not None else results_subspace
        use_stimspace_svca = results_svca is not None

        self.results = results
        self.results_cvpca = results_cvpca
        self.results_svca = results_svca
        self.results_subspace = results_subspace
        self._svca_results = svca_results
        self._svca_pf_keys = SVCA_PF_KEYS if use_stimspace_svca else LEGACY_SVCA_PF_KEYS
        self._svca_pf_env_keys = SVCA_PF_ENV_KEYS if use_stimspace_svca else LEGACY_SVCA_PF_ENV_KEYS
        self._svca_full_keys = SVCA_FULL_KEYS if use_stimspace_svca else LEGACY_SVCA_FULL_KEYS
        self._svca_full_env_keys = SVCA_FULL_ENV_KEYS if use_stimspace_svca else LEGACY_SVCA_FULL_ENV_KEYS
        self.source_cfg = source_cfg if source_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]
        self.full_cfg = full_cfg if full_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["full"]
        self._agg = {"stimspace": results, "cvpca": results_cvpca, "svca": svca_results}
        self.figsize = figsize
        self.pf_color = pf_color
        self.ff_color = ff_color
        self.pf_label = pf_label
        self.ff_label = ff_label

        # Populated by refresh_data(); plot() only reads these.
        self._mode = "curves"
        self._pf_curves: dict = {}
        self._ff_curves: dict = {}
        self._ylabel = ""

        pf_options = list(STIMSPACE_KEYS)
        if results_cvpca is not None:
            pf_options += list(CVPCA_KEYS)
        if svca_results is not None:
            pf_options += list(SVCA_PF_KEYS)
        self.add_selection("source_key", options=pf_options, value=source_key)

        full_options = [key for key in _full_key_options(results) if not key.startswith("SVCA")]
        if svca_results is not None:
            full_options.append("SVCA")
            full_options.append("SVCA_RES")
        self.add_selection("full_key", options=full_options, value=full_key)
        self.add_selection("full_scope", options=["full1", "fullall"], value=full_scope)
        self.add_selection("plot_mode", options=["all", "avg_env", "by_env"], value=plot_mode)
        self.add_selection("by_env_layout", options=["row", "col", "shared"], value=by_env_layout)
        self.add_selection("session_alignment", options=_PER_ENV_SESSION_ALIGNMENTS, value=session_alignment)
        self.add_boolean("sharey", value=sharey)
        self.add_boolean("clip_negative", value=clip_negative)
        self.add_boolean("pr_pre_smooth", value=pr_pre_smooth)

        # Shared data-selection widgets, matching SpectrumFigureViewer/PlacefieldSpectraViewer:
        # one widget per param-axis name across all aggregators, tuple-valued axes encoded as
        # string labels.
        self._tuple_labels = add_merged_param_axis_widgets(
            self,
            results,
            results_cvpca,
            svca_results,
            preferred_defaults={**PREFERRED_DEFAULTS, **param_defaults},
        )

        self.add_selection("metric", options=["participation_ratio", "truncated_pr", "alpha"], value=metric)
        self.add_selection("k_method", options=["hardcode", "adaptive"], value=k_method)
        self.add_integer("k_value", value=int(k_value), min=0, max=100)
        self.add_selection("display", options=["each", "errorPlot"], value=display)
        self.add_boolean("log_y", value=log_y)
        self.add_boolean("ylim_start_at_one", value=ylim_start_at_one)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        self.add_float("pf_text_x", value=pf_text_x, min=0.0, max=1.0, step=0.01)
        self.add_float("pf_text_y", value=pf_text_y, min=0.0, max=1.0, step=0.01)
        self.add_float("ff_text_x", value=ff_text_x, min=0.0, max=1.0, step=0.01)
        self.add_float("ff_text_y", value=ff_text_y, min=0.0, max=1.0, step=0.01)
        self.add_float("legend_anchor_x", value=legend_anchor_x, min=-2.0, max=2.0, step=0.01)
        self.add_float("legend_anchor_y", value=legend_anchor_y, min=-2.0, max=2.0, step=0.01)
        add_legend_widgets(self)
        update_legend_widgets(self, legend_options or {})

        # Independent PF/Full-CA1 adaptive-fit controls. Inert unless metric="alpha".
        for prefix, cfg in (("source", self.source_cfg), ("full", self.full_cfg)):
            self.add_selection(f"{prefix}_smooth_method", options=["none", "boxcar", "gaussian"], value=cfg.smooth_method)
            self.add_float(f"{prefix}_smooth_width", value=cfg.smooth_width, min=0.0, max=50.0, step=0.5)
            self.add_integer(f"{prefix}_fpd_window_size", value=cfg.fpd_window_size, min=1, max=50)
            self.add_integer(f"{prefix}_adaptive_buffer", value=cfg.adaptive_buffer, min=0, max=50)
            self.add_integer(f"{prefix}_minimum_window_size", value=cfg.minimum_window_size, min=1, max=500)

        axis_names = set(results.param_axes)
        if results_cvpca is not None:
            axis_names |= set(results_cvpca.param_axes)
        if svca_results is not None:
            axis_names |= set(svca_results.param_axes)
        data_widgets = (
            *axis_names,
            "source_key",
            "full_key",
            "full_scope",
            "plot_mode",
            "session_alignment",
            "clip_negative",
            "pr_pre_smooth",
            "metric",
            "k_method",
            "k_value",
            "source_smooth_method",
            "source_smooth_width",
            "source_fpd_window_size",
            "source_adaptive_buffer",
            "source_minimum_window_size",
            "full_smooth_method",
            "full_smooth_width",
            "full_fpd_window_size",
            "full_adaptive_buffer",
            "full_minimum_window_size",
        )
        for name in data_widgets:
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _sel_params(self, state: dict, source: str) -> dict:
        """Select the params relevant to ``source`` ("stimspace" or "cvpca"), decoding tuple labels."""
        agg = self._agg.get(source)
        if agg is None:
            return {}
        return sel_params(state, self._tuple_labels, agg.param_axes)

    def _cfg_from_state(self, state: dict, prefix: str) -> AdaptiveAlphaConfig:
        """Build one side's adaptive-alpha configuration from its Syd controls."""
        return AdaptiveAlphaConfig(
            smooth_method=state[f"{prefix}_smooth_method"],
            smooth_width=state[f"{prefix}_smooth_width"],
            fpd_window_size=int(state[f"{prefix}_fpd_window_size"]),
            adaptive_buffer=int(state[f"{prefix}_adaptive_buffer"]),
            minimum_window_size=int(state[f"{prefix}_minimum_window_size"]),
        )

    def _spectrum_sessions(self, state: dict, key: str) -> tuple[np.ndarray, ResultsAggregator]:
        """Return the selected PF spectrum as ``(sessions, dimensions)`` and its aggregator."""
        source = "svca" if key in SVCA_PF_KEYS else SOURCE_OF_KEY[key]
        agg = self._agg[source]
        stored_key = self._svca_pf_keys.get(key, key)
        spec = agg.sel(keys=[stored_key], avg_by_mouse=False, **self._sel_params(state, source))[stored_key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["clip_negative"]:
            spec = _clip_at_first_negative(spec)
        return spec, agg

    def _full_spectrum_sessions(self, state: dict) -> tuple[np.ndarray, ResultsAggregator]:
        """Return the selected Full-CA1 spectrum and the aggregator supplying its session rows."""
        full_key = state["full_key"]
        if full_key == "SVD":
            return self._spectrum_sessions(state, FF_KEY)
        if full_key == "SVD_RES":
            if "ffres" not in self.results.arrays:
                raise ValueError("The overall residual spectrum 'ffres' is not present in results.")
            return self._spectrum_sessions(state, "ffres")

        key = self._svca_full_keys[full_key]
        spec = self._svca_results.sel(keys=[key], avg_by_mouse=False, **self._sel_params(state, "svca"))[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["clip_negative"]:
            spec = _clip_at_first_negative(spec)
        return spec, self._svca_results

    @staticmethod
    def _per_env_result_keys(source_key: str, full_key: str, full_scope: str) -> tuple[str, str]:
        """Map public choices to StimspaceSVCA/StimSpace per-environment result keys."""
        if source_key not in _PER_ENV_PF_KEYS and source_key not in SVCA_PF_KEYS:
            options = [*_PER_ENV_PF_KEYS, *SVCA_PF_KEYS]
            raise ValueError(f"source_key={source_key!r} is unavailable with plot_mode='by_env'. Options: {options}")
        use_fullall = full_scope == "fullall"
        if source_key in SVCA_PF_ENV_KEYS:
            pf_key = SVCA_PF_ENV_KEYS[source_key]
        else:
            pf_key = f"{source_key}_env_{full_scope}" if source_key in ("sf_cv", "sf_direct") else _PER_ENV_PF_RESULT_KEYS[source_key]
        if full_key in SVCA_FULL_ENV_KEYS:
            full_stored_key = SVCA_FULL_ENV_KEYS[full_key]
        elif full_key == "SVD_RES":
            full_stored_key = "ffres_env_full1_fullall" if use_fullall else "ffres_env_full1"
        else:
            full_stored_key = "ff_env_full1_fullall" if use_fullall else "ff_env_full1"
        return pf_key, full_stored_key

    def _selected_per_env_result_keys(self, source_key: str, full_key: str, full_scope: str) -> tuple[str, str]:
        """Map public choices using the selected new or legacy SVCA backend."""
        pf_key, full_stored_key = self._per_env_result_keys(source_key, full_key, full_scope)
        if source_key in self._svca_pf_env_keys:
            pf_key = self._svca_pf_env_keys[source_key]
        if full_key in self._svca_full_env_keys:
            full_stored_key = self._svca_full_env_keys[full_key]
        return pf_key, full_stored_key

    def _per_env_spectra(self, state: dict) -> tuple[np.ndarray, ResultsAggregator, np.ndarray, ResultsAggregator]:
        """Return PF and Full spectra with shape ``(sessions, env slots, dimensions)``."""
        pf_key, full_key = self._selected_per_env_result_keys(state["source_key"], state["full_key"], state["full_scope"])
        pf_source = "svca" if state["source_key"] in SVCA_PF_KEYS else "stimspace"
        pf_agg = self._agg[pf_source]
        pf = pf_agg.sel(keys=[pf_key], squeeze_ones=False, avg_by_mouse=False, **self._sel_params(state, pf_source))[pf_key]
        if full_key in self._svca_full_env_keys.values():
            full_agg = self._svca_results
            full = full_agg.sel(keys=[full_key], squeeze_ones=False, avg_by_mouse=False, **self._sel_params(state, "svca"))[full_key]
        else:
            full_agg = self.results
            full = self.results.sel(keys=[full_key], squeeze_ones=False, avg_by_mouse=False, **self._sel_params(state, "stimspace"))[full_key]
        pf = np.asarray(pf, dtype=float)
        full = np.asarray(full, dtype=float)
        if state["clip_negative"]:
            pf = _clip_at_first_negative(pf)
            full = _clip_at_first_negative(full)
        return pf, pf_agg, full, full_agg

    @staticmethod
    def _average_env_slots(spec: np.ndarray) -> np.ndarray:
        """Average available environment slots without warning on entirely missing sessions."""
        spec = np.asarray(spec, dtype=float)
        count = np.sum(np.isfinite(spec), axis=1)
        total = np.nansum(spec, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)

    def _avg_env_source_sessions(self, state: dict, source_key: str) -> tuple[np.ndarray, ResultsAggregator]:
        """Average a per-environment PF spectrum across slots for every session."""
        pf_key, _ = self._selected_per_env_result_keys(source_key, state["full_key"], state["full_scope"])
        source = "svca" if source_key in SVCA_PF_KEYS else "stimspace"
        agg = self._agg[source]
        spec = agg.sel(
            keys=[pf_key],
            squeeze_ones=False,
            avg_by_mouse=False,
            **self._sel_params(state, source),
        )[pf_key]
        spec = np.asarray(spec, dtype=float)
        if state["clip_negative"]:
            spec = _clip_at_first_negative(spec)
        return self._average_env_slots(spec), agg

    @staticmethod
    def _align_env_spectra(target_session_ids: list, source_session_ids: list, spectra: np.ndarray) -> np.ndarray:
        """Align a ``(sessions, env slots, dimensions)`` fallback on session id."""
        index = {sid: i for i, sid in enumerate(source_session_ids)}
        out = np.full((len(target_session_ids),) + spectra.shape[1:], np.nan, dtype=float)
        for i, sid in enumerate(target_session_ids):
            j = index.get(sid)
            if j is not None:
                out[i] = spectra[j]
        return out

    def _per_env_alpha(
        self,
        state: dict,
        pf_raw: np.ndarray,
        pf_agg: ResultsAggregator,
        ff_raw: np.ndarray,
        ff_agg: ResultsAggregator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adaptive alpha by environment, using only environment-matched fallback spectra."""
        source_cfg = self._cfg_from_state(state, "source")
        full_cfg = self._cfg_from_state(state, "full")
        pf_smooth = _smooth_spectrum(pf_raw.reshape(-1, pf_raw.shape[-1]), source_cfg.smooth_method, source_cfg.smooth_width)
        ff_smooth = _smooth_spectrum(ff_raw.reshape(-1, ff_raw.shape[-1]), full_cfg.smooth_method, full_cfg.smooth_width)

        cv_raw = self.results.sel(
            keys=["ss_cvpca_env"],
            squeeze_ones=False,
            avg_by_mouse=False,
            **self._sel_params(state, "stimspace"),
        )["ss_cvpca_env"]
        cv_raw = self._align_env_spectra(pf_agg.session_ids, self.results.session_ids, np.asarray(cv_raw, dtype=float))
        cv_flat = cv_raw.reshape(-1, cv_raw.shape[-1])
        cv_smooth = _smooth_spectrum(cv_flat, source_cfg.smooth_method, source_cfg.smooth_width)

        ff_fb_raw = ff_fb_smooth = None
        if self._svca_results is not None:
            svca_key = self._svca_full_env_keys["SVCA"]
            svca = self._svca_results.sel(
                keys=[svca_key],
                squeeze_ones=False,
                avg_by_mouse=False,
                **self._sel_params(state, "svca"),
            )[svca_key]
            svca = self._align_env_spectra(ff_agg.session_ids, self._svca_results.session_ids, np.asarray(svca, dtype=float))
            ff_fb_raw = svca.reshape(-1, svca.shape[-1])
            ff_fb_smooth = _smooth_spectrum(ff_fb_raw, full_cfg.smooth_method, full_cfg.smooth_width)

        pf_shape = pf_raw.shape[:-1]
        ff_shape = ff_raw.shape[:-1]
        pf_alpha = _median_fpd_alpha_per_session(
            pf_raw.reshape(-1, pf_raw.shape[-1]),
            pf_smooth,
            source_cfg.fpd_window_size,
            source_cfg.adaptive_buffer,
            source_cfg.minimum_window_size,
            cv_flat,
            cv_smooth,
        ).reshape(pf_shape)
        ff_alpha = _median_fpd_alpha_per_session(
            ff_raw.reshape(-1, ff_raw.shape[-1]),
            ff_smooth,
            full_cfg.fpd_window_size,
            full_cfg.adaptive_buffer,
            full_cfg.minimum_window_size,
            ff_fb_raw,
            ff_fb_smooth,
        ).reshape(ff_shape)
        return pf_alpha, ff_alpha

    @staticmethod
    def _align_values_to_sessions(target_session_ids: list, source_session_ids: list, values: np.ndarray) -> np.ndarray:
        """Align arbitrary session-leading values to another aggregator's session order."""
        index = {sid: i for i, sid in enumerate(source_session_ids)}
        out = np.full((len(target_session_ids),) + values.shape[1:], np.nan, dtype=float)
        for i, sid in enumerate(target_session_ids):
            j = index.get(sid)
            if j is not None:
                out[i] = values[j]
        return out

    def _full_truncated_pr(
        self,
        state: dict,
        pf_spec: np.ndarray,
        pf_agg: ResultsAggregator,
        ff_spec: np.ndarray,
        ff_agg: ResultsAggregator,
    ) -> np.ndarray:
        """Full-spectrum PR after hardcoded or paired-PF adaptive truncation."""
        if state["k_method"] == "hardcode":
            k = int(state["k_value"])
        else:
            source_pr = _signed_participation_ratio(pf_spec)
            k = self._align_values_to_sessions(ff_agg.session_ids, pf_agg.session_ids, source_pr)
        return _truncated_participation_ratio(ff_spec, k)

    def _pr_spectrum(self, state: dict, spec: np.ndarray, prefix: str) -> np.ndarray:
        """Return the raw or smoothed spectrum used for participation-ratio metrics."""
        if state["pr_pre_smooth"]:
            return spec
        cfg = self._cfg_from_state(state, prefix)
        shape = spec.shape
        smoothed = _smooth_spectrum(spec.reshape(-1, shape[-1]), cfg.smooth_method, cfg.smooth_width)
        return smoothed.reshape(shape)

    @staticmethod
    def _per_env_curves(values: np.ndarray, aggregator: ResultsAggregator, alignment: str):
        """Build environment-slot curves using the same x conventions as spectrum_dim_familiarity."""
        curves = {}
        for slot in range(values.shape[1]):
            per_mouse = {}
            for mouse in aggregator.unique_mice:
                rows = np.where(aggregator.mouse_names == mouse)[0]
                dates = np.array([aggregator.sessions[row].date for row in rows])
                rows = rows[np.argsort(dates)]
                curve = np.asarray(values[rows, slot], dtype=float)
                if alignment == "within_env":
                    curve = curve[np.isfinite(curve)]
                if np.any(np.isfinite(curve)):
                    per_mouse[str(mouse)] = curve
            curves[slot] = per_mouse
        return curves

    def _alpha_per_session(
        self,
        state: dict,
        pf_raw: np.ndarray,
        pf_agg: ResultsAggregator,
        ff_raw: np.ndarray,
        ff_agg: ResultsAggregator,
        pf_fallback: tuple[np.ndarray, ResultsAggregator] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adaptive median-FPD alpha per PF/Full-CA1 session, with standard fallbacks.

        The selected PF spectrum borrows window boundaries from ``ss_cvpca`` whenever its own
        spectrum has no negative crossover. The Full-CA1 spectrum analogously borrows from SVCA
        when a subspace aggregator is available. In both cases the local exponent itself is still
        estimated from the selected spectrum; the fallback supplies only the adaptive window.
        """
        source_cfg = self._cfg_from_state(state, "source")
        full_cfg = self._cfg_from_state(state, "full")
        pf_smooth = _smooth_spectrum(pf_raw, source_cfg.smooth_method, source_cfg.smooth_width)
        ff_smooth = _smooth_spectrum(ff_raw, full_cfg.smooth_method, full_cfg.smooth_width)

        if pf_fallback is None:
            cvpca_raw, cvpca_agg = self._spectrum_sessions(state, "ss_cvpca")
        else:
            cvpca_raw, cvpca_agg = pf_fallback
        cvpca_smooth = _smooth_spectrum(cvpca_raw, source_cfg.smooth_method, source_cfg.smooth_width)
        pf_fb_raw = _align_rows_to_sessions(pf_agg.session_ids, cvpca_agg.session_ids, cvpca_raw)
        pf_fb_smooth = _align_rows_to_sessions(pf_agg.session_ids, cvpca_agg.session_ids, cvpca_smooth)

        ff_fb_raw, ff_fb_smooth = None, None
        if self._svca_results is not None:
            svca_key = self._svca_full_keys["SVCA"]
            svca_raw = self._svca_results.sel(keys=[svca_key], avg_by_mouse=False, **self._sel_params(state, "svca"))[svca_key]
            svca_raw = np.atleast_2d(np.asarray(svca_raw, dtype=float))
            svca_smooth = _smooth_spectrum(svca_raw, full_cfg.smooth_method, full_cfg.smooth_width)
            ff_fb_raw = _align_rows_to_sessions(ff_agg.session_ids, self._svca_results.session_ids, svca_raw)
            ff_fb_smooth = _align_rows_to_sessions(ff_agg.session_ids, self._svca_results.session_ids, svca_smooth)

        pf_alpha = _median_fpd_alpha_per_session(
            pf_raw,
            pf_smooth,
            source_cfg.fpd_window_size,
            source_cfg.adaptive_buffer,
            source_cfg.minimum_window_size,
            pf_fb_raw,
            pf_fb_smooth,
        )
        ff_alpha = _median_fpd_alpha_per_session(
            ff_raw,
            ff_smooth,
            full_cfg.fpd_window_size,
            full_cfg.adaptive_buffer,
            full_cfg.minimum_window_size,
            ff_fb_raw,
            ff_fb_smooth,
        )
        return pf_alpha, ff_alpha

    @staticmethod
    def _aligned_mouse_curves(
        pf_values: np.ndarray,
        pf_agg: ResultsAggregator,
        ff_values: np.ndarray,
        ff_agg: ResultsAggregator,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Align two session-level value arrays to the same chronological slots within each mouse."""
        session_meta: dict[str, tuple[str, object]] = {}
        mouse_order: list[str] = []
        for agg in (pf_agg, ff_agg):
            for session_id, mouse, session in zip(agg.session_ids, agg.mouse_names, agg.sessions):
                mouse = str(mouse)
                session_meta.setdefault(session_id, (mouse, session.date))
                if mouse not in mouse_order:
                    mouse_order.append(mouse)

        pf_by_session = dict(zip(pf_agg.session_ids, pf_values))
        ff_by_session = dict(zip(ff_agg.session_ids, ff_values))
        pf_curves: dict[str, np.ndarray] = {}
        ff_curves: dict[str, np.ndarray] = {}
        for mouse in mouse_order:
            session_ids = [session_id for session_id, (name, _) in session_meta.items() if name == mouse]
            session_ids.sort(key=lambda session_id: (str(session_meta[session_id][1]), str(session_id)))
            pf_curve = np.array([pf_by_session.get(session_id, np.nan) for session_id in session_ids], dtype=float)
            ff_curve = np.array([ff_by_session.get(session_id, np.nan) for session_id in session_ids], dtype=float)
            if np.any(np.isfinite(pf_curve)) or np.any(np.isfinite(ff_curve)):
                pf_curves[mouse] = pf_curve
                ff_curves[mouse] = ff_curve
        return pf_curves, ff_curves

    @staticmethod
    def _pad_curves(curves: dict[str, np.ndarray]) -> np.ndarray:
        """NaN-pad ragged per-mouse curves to ``(mice, max_sessions)``."""
        max_sessions = max((len(curve) for curve in curves.values()), default=0)
        stack = np.full((len(curves), max_sessions), np.nan)
        for row, curve in enumerate(curves.values()):
            stack[row, : len(curve)] = curve
        return stack

    @classmethod
    def _draw_curves(
        cls,
        ax,
        curves: dict[str, np.ndarray],
        color: str,
        label: str,
        display: str,
        x_start: int = 0,
    ) -> int:
        """Draw individual+mean or mean+SE curves, returning the number of x slots shown."""
        stack = cls._pad_curves(curves)
        if not stack.size:
            return 0

        # Log axes cannot show non-positive participation ratios or alpha estimates. Treat them as
        # missing for both display modes so the two modes have identical support.
        if ax.get_yscale() == "log":
            stack[stack <= 0] = np.nan

        support = np.sum(np.isfinite(stack), axis=0)
        # Match the manuscript's other familiarity panels: a population summary is only drawn
        # where at least two mice contribute. Individual traces still retain their full extent.
        summary_columns = np.where(support >= 2)[0]

        if display == "each":
            for curve in stack:
                ax.plot(np.arange(curve.size) + x_start, curve, color=(color, 0.3), linewidth=0.5)
            if summary_columns.size:
                ax.plot(
                    summary_columns + x_start,
                    np.nanmean(stack[:, summary_columns], axis=0),
                    color=color,
                    linewidth=2.0,
                    label=label,
                )
            return stack.shape[1]

        if summary_columns.size:
            errorPlot(
                summary_columns + x_start,
                stack[:, summary_columns],
                axis=0,
                se=True,
                ax=ax,
                color=color,
                linewidth=2.0,
                alpha=0.25,
                label=label,
            )
            return int(summary_columns[-1]) + 1
        return 0

    def _refresh_curves(self, state: dict) -> None:
        """Recompute per-mouse PF/Full-CA1 curves for ``plot_mode in {"all", "avg_env"}``."""
        if state["plot_mode"] == "avg_env":
            pf_env, pf_agg, ff_env, ff_agg = self._per_env_spectra(state)
            pf_spec = self._average_env_slots(pf_env)
            ff_spec = self._average_env_slots(ff_env)
        else:
            pf_spec, pf_agg = self._spectrum_sessions(state, state["source_key"])
            ff_spec, ff_agg = self._full_spectrum_sessions(state)
        if state["metric"] == "alpha":
            pf_fallback = self._avg_env_source_sessions(state, "ss_cvpca") if state["plot_mode"] == "avg_env" else None
            pf_values, ff_values = self._alpha_per_session(state, pf_spec, pf_agg, ff_spec, ff_agg, pf_fallback)
            ylabel = "Decay exponent"
        else:
            pf_spec = self._pr_spectrum(state, pf_spec, "source")
            ff_spec = self._pr_spectrum(state, ff_spec, "full")
            pf_values = _signed_participation_ratio(pf_spec)
            ff_values = (
                self._full_truncated_pr(state, pf_spec, pf_agg, ff_spec, ff_agg)
                if state["metric"] == "truncated_pr"
                else _signed_participation_ratio(ff_spec)
            )
            ylabel = "Dimensionality"
        pf_curves, ff_curves = self._aligned_mouse_curves(pf_values, pf_agg, ff_values, ff_agg)
        self._mode = "curves"
        self._pf_curves, self._ff_curves, self._ylabel = pf_curves, ff_curves, ylabel

    def _refresh_by_env(self, state: dict) -> None:
        """Recompute per-environment-slot PF/Full-CA1 curves for ``plot_mode="by_env"``."""
        pf_spec, pf_agg, ff_spec, ff_agg = self._per_env_spectra(state)
        if state["metric"] == "alpha":
            pf_values, ff_values = self._per_env_alpha(state, pf_spec, pf_agg, ff_spec, ff_agg)
            ylabel = "Decay exponent"
        else:
            pf_spec = self._pr_spectrum(state, pf_spec, "source")
            ff_spec = self._pr_spectrum(state, ff_spec, "full")
            pf_values = _signed_participation_ratio(pf_spec)
            ff_values = (
                self._full_truncated_pr(state, pf_spec, pf_agg, ff_spec, ff_agg)
                if state["metric"] == "truncated_pr"
                else _signed_participation_ratio(ff_spec)
            )
            ylabel = "Dimensionality"
        self._mode = "by_env"
        self._pf_curves = self._per_env_curves(pf_values, pf_agg, state["session_alignment"])
        self._ff_curves = self._per_env_curves(ff_values, ff_agg, state["session_alignment"])
        self._ylabel = ylabel

    def refresh_data(self, state: dict) -> None:
        """Recompute the PF/Full-CA1 curves for the current selection (see ``plot``)."""
        if state["plot_mode"] == "by_env":
            self._refresh_by_env(state)
        else:
            self._refresh_curves(state)

    def _full_label(self, state: dict) -> str:
        """Return the displayed CA1 label for the selected full-spectrum estimator."""
        return "PF Residual" if state["full_key"].endswith("_RES") else self.ff_label

    def _plot_curves(self, state: dict, fontsize: float):
        """Draw the single-axis ``plot_mode in {"all", "avg_env"}`` view."""
        fig, ax = self.new_subplots(1, 1, figsize=self.figsize, layout="constrained")
        if state["log_y"]:
            ax.set_yscale("log")

        extents = [
            self._draw_curves(ax, self._pf_curves, self.pf_color, self.pf_label, state["display"]),
            self._draw_curves(ax, self._ff_curves, self.ff_color, self._full_label(state), state["display"]),
        ]
        xmax = max(max(extents) - 1, 1)
        ax.set_xlabel("Session #", fontsize=fontsize)
        ax.set_ylabel(self._ylabel, fontsize=fontsize)

        ax.set_xlim(left=0)
        if state["ylim_start_at_one"]:
            ax.set_ylim(bottom=1)
        ylim = ax.get_ylim()
        xticks = ax.get_xticks()
        xticks = np.unique(np.append(xticks[(xticks >= 0) & (xticks <= ax.get_xlim()[1])], 0.0))
        style_model_axis(ax, fontsize=fontsize, xbounds=[0, xmax], ybounds=ylim, xticks=xticks)
        apply_legend(ax, state, fontsize, auto_loc="best")
        return fig

    def _plot_by_env(self, state: dict, fontsize: float):
        """Draw the separate- or shared-axis ``plot_mode="by_env"`` view."""
        layout = state["by_env_layout"]
        column_layout = layout == "col"
        shared_layout = layout == "shared"
        nrows, ncols = (1, 1) if shared_layout else ((2, 1) if column_layout else (1, 2))
        fig, axes = self.new_subplots(
            nrows,
            ncols,
            figsize=self.figsize,
            layout="constrained",
            sharex=column_layout,
            sharey=state["sharey"] and not shared_layout,
        )
        axes = np.atleast_1d(axes)
        if state["log_y"]:
            for axis in axes:
                axis.set_yscale("log")
        extents = []
        pf_spec = (
            self._pf_curves,
            self.pf_label,
            state["pf_text_x"],
            state["pf_text_y"],
            "purple",
        )
        ff_spec = (
            self._ff_curves,
            self._full_label(state),
            state["ff_text_x"],
            state["ff_text_y"],
            "brown",
        )
        ordered_specs = (ff_spec, pf_spec) if column_layout else (pf_spec, ff_spec)
        panel_specs = (
            tuple((axes[0], *spec) for spec in ordered_specs)
            if shared_layout
            else tuple((axis, *spec) for axis, spec in zip(axes, ordered_specs))
        )
        xlabel = "Env session #" if state["session_alignment"] == "within_env" else "Overall session #"
        for axis, curves, text, text_x, text_y, text_color in panel_specs:
            extent = 0
            for slot, per_mouse in curves.items():
                extent = max(
                    extent,
                    self._draw_curves(
                        axis,
                        per_mouse,
                        ENV_SLOT_COLORS[slot % len(ENV_SLOT_COLORS)],
                        _ordinal(slot + 1),
                        state["display"],
                    ),
                )
            extents.append(extent)
            axis.text(
                text_x,
                text_y,
                text,
                transform=axis.transAxes,
                ha="left",
                va="center",
                fontsize=fontsize,
                color=text_color,
                fontweight="bold",
            )
            axis.set_ylabel(self._ylabel, fontsize=fontsize)
            axis.set_xlabel(xlabel, fontsize=fontsize)
        # Both axes must be fully drawn before either is formatted: with sharey, formatting one
        # while the other is empty positions its offset spine against stale (empty-axis) limits.
        xmax = max(max(extents) - 1, 1)
        for ia, axis in enumerate(axes):
            axis.set_xlim(0, xmax + 0.2)
            if state["ylim_start_at_one"]:
                axis.set_ylim(bottom=1)
            xticks = axis.get_xticks()
            xticks = np.unique(np.append(xticks[(xticks >= 0) & (xticks <= axis.get_xlim()[1])], 0.0))
            is_shared_x_top = column_layout and ia == 0
            spines_visible = [] if is_shared_x_top else ["bottom"]
            if column_layout or ia == 0 or not state["sharey"]:
                spines_visible.append("left")
            style_model_axis(axis, fontsize=fontsize, xbounds=[0, xmax], ybounds=axis.get_ylim(), xticks=xticks, spines_visible=spines_visible)
            if is_shared_x_top:
                axis.set_xlabel("")
                axis.xaxis.set_visible(False)
            if not column_layout and ia == 1 and state["sharey"]:
                axis.yaxis.set_visible(False)
        legend_axis = axes[0]
        handles = labels = None
        if shared_layout:
            all_handles, all_labels = legend_axis.get_legend_handles_labels()
            unique = dict(zip(all_labels, all_handles))
            labels, handles = list(unique), list(unique.values())
        apply_legend(legend_axis, state, fontsize, auto_loc="best", handles=handles, labels=labels)
        legend = legend_axis.get_legend()
        if legend is not None:
            if layout in ("col", "shared"):
                legend.set_bbox_to_anchor(
                    (state["legend_anchor_x"], state["legend_anchor_y"], 1.0, 1.0),
                    transform=legend_axis.transAxes,
                )
                if state["legend_anchor_x"] != 0.0 or state["legend_anchor_y"] != 0.0:
                    legend.set_in_layout(False)
            legend.set_title("Env #")
            legend.get_title().set_fontsize(fontsize * state["legend_fontsize_scale"])
        return fig

    def plot(self, state: dict):
        """Draw the current PF/Full-CA1 curves. Assumes ``refresh_data`` already ran."""
        fontsize = state["fontsize"]
        if self._mode == "by_env":
            return self._plot_by_env(state, fontsize)
        return self._plot_curves(state, fontsize)
