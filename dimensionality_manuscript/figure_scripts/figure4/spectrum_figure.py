"""Placefield-vs-full spectrum figure: spectra, participation ratio, and adaptive decay exponent."""

import numpy as np

from vrAnalysis.helpers.plotting import format_spines
from dimensionality_manuscript import ResultsAggregator, average_by_mouse
from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, apply_legend, update_legend_widgets

from ._alpha_config import AdaptiveAlphaConfig, SpectrumSmoothing, SpectrumSmoothingConfig, ADAPTIVE_ALPHA_CONFIG_REGISTRY
from ._param_axes import (
    CVPCA_KEYS,
    FF_KEY,
    FIT_KEY_COLORS,
    FIT_KEY_LABELS,
    FIT_KEY_PARAM_KEYS,
    FIT_KEYS,
    PER_ENV_PF_RESULT_KEYS,
    PREFERRED_DEFAULTS,
    SOURCE_OF_KEY,
    STIMSPACE_KEYS,
    TILBURY_REL_FA,
    add_merged_param_axis_widgets,
    merged_axis_names as _merged_axis_names,
    sel_params,
)
from ._spectrum_math import (
    _align_rows_to_sessions,
    _clip_at_first_negative,
    _eig_to_ss_scale,
    _median_fpd_alpha_per_session,
    _signed_participation_ratio,
    _smooth_spectrum,
    _xvals,
)
from ._stats import _beeswarm_panel, _horizontal_beeswarm_panel
from ..panels import FigureViewer

SPECTRUM_LEGEND_DEFAULTS = {
    "handlelength": 1.0,
    "handletextpad": 0.25,
    "markerfirst": False,
}

# Selectable spectra stored by StimspaceSVCAConfig. Keep viewer-facing names separate from the
# compact result keys used by that config.
SVCA_PF_KEYS = {
    "SVCA": "ss",
    "SVCA_PRED": "ss_pred",
}
SVCA_FULL_KEYS = {
    "SVCA": "ff",
    "SVCA_RES": "ff_res",
}
LEGACY_SVCA_PF_KEYS = {
    "SVCA": "variance_placefield_placefield",
    "SVCA_PRED": "variance_placefield_prediction",
}
LEGACY_SVCA_FULL_KEYS = {
    "SVCA": "variance_activity",
    "SVCA_RES": "variance_activity_residual",
}


def draw_reliable_spectrum_annotation(
    ax,
    xy: tuple[float, float],
    y_spacing: float,
    definition_x_offset: float,
    fontsize: float,
    *,
    top_label: str = r"$ROIs_A$",
    bottom_label: str = r"$ROIs_B$",
    definition_label: str = "reliable spectrum",
    color: str = "black",
):
    """Draw the compact ``ROI_A times ROI_B`` reliable-spectrum annotation.

    Positions are in axes-fraction coordinates. ``xy`` is the upper-left position of the
    first row, ``y_spacing`` separates the three stacked rows, and
    ``definition_x_offset`` moves the explanatory text rightward from that first column.

    Returns
    -------
    tuple[matplotlib.text.Text, ...]
        The four text artists, useful to callers that want further local styling.
    """
    x, y = xy
    text_kwargs = {
        "transform": ax.transAxes,
        "fontsize": fontsize,
        "color": color,
        "ha": "left",
        "va": "top",
    }
    middle_y = y - y_spacing
    return (
        ax.text(x, y, top_label, **text_kwargs),
        ax.text(x, y - 2 * y_spacing, bottom_label, **text_kwargs),
        ax.text(x + definition_x_offset, middle_y, rf"$\times$ = {definition_label}", **text_kwargs),
    )


def draw_xcov_spectrum_annotation(
    ax,
    xy: tuple[float, float],
    y_spacing: float,
    fontsize: float,
    *,
    color: str = "black",
):
    """Draw the two-line cross-covariance definition of the reliable spectrum."""
    x, y = xy
    text_kwargs = {
        "transform": ax.transAxes,
        "fontsize": fontsize,
        "color": color,
        "ha": "left",
        "va": "top",
    }
    return (
        ax.text(x, y, r"x-cov(source, target)$\rightarrow$ reliable spectrum", **text_kwargs),
        # ax.text(x, y - y_spacing, r"", **text_kwargs),
    )


def draw_dimensionality_equation_annotation(
    ax,
    xy: tuple[float, float],
    fontsize: float,
    *,
    pf_label: str,
    ff_label: str,
    pf_color: str,
    ff_color: str,
    linewidth: float,
    yoffset: float,
    yheight: float,
    arrow_mutation_scale: float,
):
    """Draw ``CA1 = PF + residual`` with dimension arrows sized to the term text."""
    x, y = xy
    ax.figure.draw_without_rendering()
    renderer = ax.figure._get_renderer()
    text_artists = []
    term_artists = []
    for label, color, weight, is_term in (
        ("CA1", "black", "bold", False),
        ("=", "black", "normal", False),
        (pf_label, pf_color, "bold", True),
        ("+", "black", "normal", False),
        (ff_label, ff_color, "bold", True),
    ):
        artist = ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            ha="left",
            va="top",
        )
        text_artists.append(artist)
        if is_term:
            term_artists.append((artist, color))
        bbox_axes = artist.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        x = bbox_axes.x1

    diagram_artists = []
    for term, color in term_artists:
        bbox_axes = term.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        center_x = (bbox_axes.x0 + bbox_axes.x1) / 2
        bar_y = bbox_axes.y0 - yoffset
        bar = ax.plot(
            [bbox_axes.x0, bbox_axes.x1],
            [bar_y, bar_y],
            transform=ax.transAxes,
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
            clip_on=False,
        )[0]
        arrow = ax.annotate(
            "",
            xy=(center_x, bar_y - yheight),
            xytext=(center_x, bar_y),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": linewidth,
                "mutation_scale": arrow_mutation_scale,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
        )
        dimension = ax.text(
            center_x,
            bar_y - yheight - yoffset,
            "dim?",
            transform=ax.transAxes,
            fontsize=fontsize,
            color=color,
            ha="center",
            va="top",
            clip_on=False,
        )
        diagram_artists.extend((bar, arrow, dimension))
    return (*text_artists, *diagram_artists)


class SpectrumFigureViewer(FigureViewer):
    """Placefield-vs-full spectrum figure: spectra and participation-ratio dimensionality.

    Two vertically stacked panels comparing one placefield (PF) spectrum against the full/functional
    (FF) spectrum:

    - ax[0]: the selected PF ``source_key`` spectrum (StimSpace, CVPCA, or optional SVCA variants) and the FF
      spectrum, both log-space pre-smoothed and drawn log-log (faint per-mouse lines + bold
      mouse-average). The FF curve source is set by ``full_source_key``: ``"SVD"`` and ``"SVD_RES"``
      are the ``ff`` and ``ffres`` keys from the StimSpaceSpectra aggregator; ``"SVCA"`` and
      ``"SVCA_RES"`` are the ``ff`` and ``ff_res`` keys from the StimspaceSVCA aggregator. Optional Tilbury-fit eigenvalue-spectrum
      overlays (``fit_key``) are drawn between PF and FF, always at the fit's fixed
      reliability/fraction-active threshold (:data:`~._param_axes.TILBURY_REL_FA`); a mismatch with
      the shared ``reliability_fraction_active_thresholds`` selection is flagged in the ax[0] title.
    - ax[1]: the signed participation ratio measured from the raw (unsmoothed) mouse spectra on the
      same log-scaled dimension axis as ax[0]. PF, fit-overlay, and FF points share one unlabeled
      categorical y-position; their colors are identified by ax[0]'s legend.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*`` and ``ff``/``ffres`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results, source of the ``reg_covariances_fixed`` key. Required if
        ``source_key`` is that CVPCA key; if None only StimSpace PF keys are selectable.
    results_svca : ResultsAggregator or None
        Aggregated StimspaceSVCAConfig results. Supplies ``ss``/``ss_pred`` for
        ``source_key="SVCA"``/``"SVCA_PRED"`` and ``ff``/``ff_res`` for
        ``full_source_key="SVCA"``/``"SVCA_RES"``.
    results_subspace : ResultsAggregator or None
        Legacy SubspaceConfig backend for the same public SVCA choices, using its ``variance_*``
        keys. When both backends are supplied, ``results_svca`` takes precedence.
    results_fit : ResultsAggregator or None
        Aggregated TilburyFitConfig results, source of the :data:`~._param_axes.FIT_KEYS` overlays
        selected by ``fit_key``. Required for those overlays; if None ``fit_key`` must be empty.
    source_key : str
        Which PF spectrum to show in ax[0]. One of ``ss_cv``/``ss_direct``/``ss_cvpca`` (from
        ``results``), ``reg_covariances_fixed`` (from ``results_cvpca``), or ``SVCA``/``SVCA_PRED``
        (from ``results_svca``).
    source_mode : {"all", "avg_env"}
        ``"all"`` preserves the overall-spectrum behavior. ``"avg_env"`` reads the independently
        computed per-environment ``source_key`` spectra, averages the available environment slots
        within each session, then averages sessions within mouse before measuring dimensionality.
        ``full_source_key`` and Tilbury-fit overlays retain their existing whole-session behavior.
        This mode supports StimSpace and SVCA PF sources; CVPCAConfig PF sources have no stored
        per-environment equivalent. ``SVCA`` reads ``ss_env`` and ``SVCA_PRED`` reads
        ``ss_pred_env`` from ``results_svca``.
    full_source_key : {"SVD", "SVD_RES", "SVCA", "SVCA_RES"}
        Source of the FF ("Reliable CA1 Spectrum") curve. ``"SVD"``/``"SVD_RES"`` use the
        StimSpaceSpectra ``ff``/``ffres`` keys; ``"SVCA"``/``"SVCA_RES"`` use StimspaceSVCA
        ``ff``/``ff_res`` and require ``results_svca``.
    fit_key : str or list of str
        Extra ax[0] overlays from the Tilbury-fit aggregator: any of ``eig_tilbury`` (blue, the
        unregularized generalized Gaussian), ``eig_control`` (green, the plain-Gaussian control),
        ``eig_shrinkage`` (purple, the generalized fit with the Gaussian-centered shrinkage penalty
        at its per-neuron validation-selected lambdas), or ``eig_better`` (red, the per-neuron
        generalized/Gaussian composite). Requires ``results_fit``.
    source_smooth_method, full_smooth_method : {"none", "boxcar", "gaussian"}
        Independent log-space smoothing methods for the PF and full-CA1 spectra.
    source_smooth_width, full_smooth_width : float
        Corresponding smoothing widths in rank units.
    normalize : bool
        If True, normalize each spectrum by its sum (does not affect the participation ratio).
    clip_negative : bool
        If True, clip each source and full-CA1 session spectrum at its first negative value before
        mouse averaging: the first negative rank and every later rank are replaced with NaN.
    pr_pre_smooth : bool
        If True, calculate the participation ratio before display smoothing. If False, calculate it
        from the smoothed spectra.
    ylim_min, ylim_max : float
        ax[0] y-limits in log10 units; the applied range is ``10 ** ylim_min`` to ``10 ** ylim_max``.
    beewidth : float
        Vertical spread of the ax[1] participation-ratio swarms around the shared y-position.
    each_line_alpha : float
        Alpha of individual-mouse spectrum lines in ax[0].
    point_alpha : float
        Alpha of individual-mouse participation-ratio points in ax[1].
    markersize : float
        Participation-ratio marker size in points.
    mean_linewidth : float
        Width of the vertical mean markers in ax[1].
    fontsize : float
        Font size for every text element (labels, ticks, legend, title).
    show_annotation : {"ROIs", "X-Cov", "None"} or bool
        Annotation shown on ax[0]. ``"ROIs"`` draws the ROI-product annotation, ``"X-Cov"``
        draws the two-line cross-covariance definition, and ``"None"`` hides it. The legacy
        boolean values ``True``/``False`` are accepted as aliases for ``"ROIs"``/``"None"``.
    annotation_x, annotation_y : float
        Axes-fraction coordinates of the annotation's upper-left corner.
    annotation_y_spacing : float
        Axes-fraction vertical spacing between ``ROI_A``, the multiplication sign, and ``ROI_B``.
    annotation_definition_x_offset : float
        Axes-fraction horizontal offset from the stacked labels to ``times = reliable spectrum``.
    show_equation : bool
        Whether to show the dimensionality equation independently of ``show_annotation``.
    equation_x, equation_y : float
        Axes-fraction coordinates of the equation's upper-left corner.
    equation_linewidth : float
        Width of the colored bars and arrows beneath the PF and full-spectrum terms.
    equation_yoffset : float
        Gap from each term to its bar, and from each arrow tip to its ``dim?`` label.
    equation_yheight : float
        Height of the downward arrows in axes-fraction units.
    equation_arrow_mutation_scale : float
        Matplotlib mutation scale controlling the arrow-head size.
    legend_options : dict or None
        Initial legend settings forwarded to
        :mod:`~dimensionality_manuscript.figure_scripts.legends`. These seed the Syd
        ``legend_*`` controls; unspecified settings preserve this panel's prior legend style.
    figsize : tuple[float, float]
        Figure size in inches.
    height_ratios : tuple[float, float]
        Relative heights of the spectrum and participation-ratio rows.
    pf_color, ff_color : str
        Colors shared by each spectrum and its participation-ratio points.
    pf_label, ff_label : str
        Legend labels for the PF and FF spectra.
    **selection_defaults
        Starting values for the merged parameter-axis widgets, keyed by raw ``param_axes`` name
        (e.g. ``activity_parameters_name``). Falls back to
        :data:`~._param_axes.PREFERRED_DEFAULTS` for names not given here.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        results_fit: ResultsAggregator | None = None,
        results_svca: ResultsAggregator | None = None,
        *,
        source_key: str = "ss_cv",
        source_mode: str = "all",
        full_source_key: str = "SVD",
        fit_key: str | list[str] = (),
        source_smooth_method: str = "gaussian",
        source_smooth_width: float = 3.0,
        full_smooth_method: str = "gaussian",
        full_smooth_width: float = 20.0,
        normalize: bool = True,
        clip_negative: bool = False,
        pr_pre_smooth: bool = True,
        ylim_min: float = -5.5,
        ylim_max: float = 0.0,
        beewidth: float = 0.15,
        each_line_alpha: float = 0.3,
        point_alpha: float = 0.6,
        markersize: float = 3.0,
        mean_linewidth: float = 2.0,
        fontsize: float = 9.0,
        show_annotation: str | bool = "ROIs",
        annotation_x: float = 0.43,
        annotation_y: float = 0.92,
        annotation_y_spacing: float = 0.08,
        annotation_definition_x_offset: float = 0.08,
        show_equation: bool = False,
        equation_x: float = 0.02,
        equation_y: float = 0.98,
        equation_linewidth: float = 1.5,
        equation_yoffset: float = 0.015,
        equation_yheight: float = 0.10,
        equation_arrow_mutation_scale: float = 10.0,
        legend_options: dict | None = None,
        figsize: tuple[float, float] = (3.25, 3.25),
        height_ratios: tuple[float, float] = (1.0, 0.22),
        pf_color: str = "purple",
        ff_color: str = "black",
        pf_label: str = "Placefields",
        ff_label: str = "Full CA1",
        **selection_defaults,
    ):
        svca_results = results_svca if results_svca is not None else results_subspace
        svca_pf_keys = SVCA_PF_KEYS if results_svca is not None else LEGACY_SVCA_PF_KEYS
        svca_full_keys = SVCA_FULL_KEYS if results_svca is not None else LEGACY_SVCA_FULL_KEYS

        pf_options = list(STIMSPACE_KEYS)
        if results_cvpca is not None:
            pf_options += list(CVPCA_KEYS)
        if svca_results is not None:
            pf_options.extend(SVCA_PF_KEYS)
        if source_key not in pf_options:
            raise ValueError(f"Unknown PF source_key {source_key!r}. Options: {pf_options}")

        full_options = ["SVD"]
        if "ffres" in results.arrays:
            full_options.append("SVD_RES")
        if svca_results is not None:
            full_options.extend(SVCA_FULL_KEYS)
        if full_source_key not in full_options:
            raise ValueError(f"Unknown full_source_key {full_source_key!r}. Options: {full_options}")

        fit_keys = [fit_key] if isinstance(fit_key, str) else list(fit_key)
        if fit_keys and results_fit is None:
            raise ValueError("fit_key requires results_fit to be provided.")
        for fk in fit_keys:
            if fk not in FIT_KEYS:
                raise ValueError(f"Unknown fit_key {fk!r}. Options: {FIT_KEYS}")

        self.results = results
        self.results_cvpca = results_cvpca
        self.results_svca = results_svca
        self.results_subspace = results_subspace
        self._svca_results = svca_results
        self._svca_pf_keys = svca_pf_keys
        self._svca_full_keys = svca_full_keys
        self.results_fit = results_fit
        self._agg = {"stimspace": results, "cvpca": results_cvpca, "svca": svca_results}
        self.figsize = figsize
        self.height_ratios = height_ratios
        self.pf_color = pf_color
        self.ff_color = ff_color
        self.pf_label = pf_label
        self.ff_label = ff_label

        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        self.add_selection("source_key", options=pf_options, value=source_key)
        self.add_selection("source_mode", options=["all", "avg_env"], value=source_mode)
        self.add_selection("full_source_key", options=full_options, value=full_source_key)
        if results_fit is not None:
            self.add_multiple_selection("fit_key", options=list(FIT_KEYS), value=fit_keys)

        # One widget per param-axis name, shared across sources; fit-only axes (not already covered
        # by results/results_cvpca) get their own widget via extra_axes so a fit_key selection still
        # pins every axis the fit aggregator needs.
        extra_axes = results_fit.param_axes if results_fit is not None else None
        preferred_defaults = {**PREFERRED_DEFAULTS, **selection_defaults}
        self._tuple_labels = add_merged_param_axis_widgets(
            self,
            results,
            results_cvpca,
            svca_results,
            extra_axes=extra_axes,
            preferred_defaults=preferred_defaults,
        )
        self._fit_axes: list[str] = list(results_fit.param_axes) if results_fit is not None else []

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_float("ylim_max", value=ylim_max, min=-8.0, max=12.0, step=0.1)
        self.add_float("beewidth", value=beewidth, min=0.0, max=1.0, step=0.01)
        self.add_boolean("normalize", value=normalize)
        self.add_boolean("clip_negative", value=clip_negative)
        self.add_boolean("pr_pre_smooth", value=pr_pre_smooth)
        self.add_float("each_line_alpha", value=each_line_alpha, min=0.0, max=1.0, step=0.01)
        self.add_float("point_alpha", value=point_alpha, min=0.0, max=1.0, step=0.01)
        self.add_float("markersize", value=markersize, min=0.5, max=12.0, step=0.5)
        self.add_float("mean_linewidth", value=mean_linewidth, min=0.25, max=6.0, step=0.25)
        annotation_mode = {True: "ROIs", False: "None"}.get(show_annotation, show_annotation)
        self.add_selection("show_annotation", options=["ROIs", "X-Cov", "None"], value=annotation_mode)
        self.add_float("annotation_x", value=annotation_x, min=-1.0, max=2.0, step=0.01)
        self.add_float("annotation_y", value=annotation_y, min=-1.0, max=2.0, step=0.01)
        self.add_float("annotation_y_spacing", value=annotation_y_spacing, min=0.0, max=0.5, step=0.005)
        self.add_float(
            "annotation_definition_x_offset",
            value=annotation_definition_x_offset,
            min=-1.0,
            max=2.0,
            step=0.01,
        )
        self.add_boolean("show_equation", value=show_equation)
        self.add_float("equation_x", value=equation_x, min=-1.0, max=2.0, step=0.01)
        self.add_float("equation_y", value=equation_y, min=-1.0, max=2.0, step=0.01)
        self.add_float("equation_linewidth", value=equation_linewidth, min=0.1, max=10.0, step=0.1)
        self.add_float("equation_yoffset", value=equation_yoffset, min=0.0, max=0.5, step=0.005)
        self.add_float("equation_yheight", value=equation_yheight, min=0.0, max=0.5, step=0.005)
        self.add_float(
            "equation_arrow_mutation_scale",
            value=equation_arrow_mutation_scale,
            min=1.0,
            max=50.0,
            step=1.0,
        )
        add_legend_widgets(self)
        update_legend_widgets(self, {**SPECTRUM_LEGEND_DEFAULTS, **(legend_options or {})})

        # Independent spectrum smoothing; there are deliberately no adaptive-alpha controls here.
        self.add_selection("source_smooth_method", options=["none", "boxcar", "gaussian"], value=source_smooth_method)
        self.add_float("source_smooth_width", value=source_smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_selection("full_smooth_method", options=["none", "boxcar", "gaussian"], value=full_smooth_method)
        self.add_float("full_smooth_width", value=full_smooth_width, min=0.0, max=50.0, step=0.5)

        merged_axis_names = _merged_axis_names(results, results_cvpca, svca_results, extra_axes=extra_axes)
        refresh_names = (
            *merged_axis_names,
            "source_key",
            "source_mode",
            "full_source_key",
            "normalize",
            "clip_negative",
            "pr_pre_smooth",
            "source_smooth_method",
            "source_smooth_width",
            "full_smooth_method",
            "full_smooth_width",
        )
        for name in refresh_names:
            self.on_change(name, self.refresh_data)
        if results_fit is not None:
            self.on_change("fit_key", self.refresh_data)
        self.refresh_data(self.state)

    def _smoothing_from_state(self, state: dict, prefix: str) -> SpectrumSmoothingConfig:
        """Return only the smoothing state needed by this viewer."""
        return SpectrumSmoothingConfig(
            smooth_method=state[f"{prefix}_smooth_method"],
            smooth_width=state[f"{prefix}_smooth_width"],
        )

    @staticmethod
    def _cfg_from_state(state: dict, prefix: str) -> AdaptiveAlphaConfig:
        """Build adaptive-fit settings for alpha-viewer subclasses and compatibility aliases."""
        return AdaptiveAlphaConfig(
            smooth_method=state[f"{prefix}_smooth_method"],
            smooth_width=state[f"{prefix}_smooth_width"],
            fpd_window_size=int(state[f"{prefix}_fpd_window_size"]),
            adaptive_buffer=int(state[f"{prefix}_adaptive_buffer"]),
            minimum_window_size=int(state[f"{prefix}_minimum_window_size"]),
        )

    def _spectrum(
        self,
        state: dict,
        key: str,
        cfg: SpectrumSmoothing,
        *,
        apply_source_mode: bool = True,
    ) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` spectrum for ``key``, normalized per ``state``, smoothed per ``cfg``."""
        if apply_source_mode and state.get("source_mode", "all") == "avg_env":
            return self._avg_env_spectrum(state, key, cfg)
        if key in SVCA_PF_KEYS:
            return self._svca_placefield_spectrum(state, key, cfg)
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        params = sel_params(state, self._tuple_labels, agg.param_axes)
        if state.get("clip_negative", False):
            spec = agg.sel(keys=[key], avg_by_mouse=False, **params)[key]
            spec = _clip_at_first_negative(np.atleast_2d(np.asarray(spec, dtype=float)))
            spec = average_by_mouse({key: spec}, agg.mouse_names)[key]
        else:
            spec = agg.sel(keys=[key], avg_by_mouse=True, **params)[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)

    @staticmethod
    def _average_env_slots(spec: np.ndarray) -> np.ndarray:
        """Average available environment slots in ``(sessions, slots, dims)`` spectra."""
        spec = np.asarray(spec, dtype=float)
        count = np.sum(np.isfinite(spec), axis=1)
        total = np.nansum(spec, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)

    @staticmethod
    def _per_env_pf_key(key: str) -> str:
        """Map a selectable StimSpace PF source to its stored per-environment key."""
        if key in PER_ENV_PF_RESULT_KEYS:
            return PER_ENV_PF_RESULT_KEYS[key]
        if key in ("sf_cv", "sf_direct"):
            return f"{key}_env_full1"
        raise ValueError(
            f"source_mode='avg_env' has no independently computed per-environment spectrum for source_key={key!r}. " f"Options: {STIMSPACE_KEYS}"
        )

    def _avg_env_stimspace_sessions(self, state: dict, key: str) -> np.ndarray:
        """Per-session StimSpace spectrum after averaging independently computed environments."""
        stored_key = self._per_env_pf_key(key)
        params = sel_params(state, self._tuple_labels, self.results.param_axes)
        spec = self.results.sel(keys=[stored_key], squeeze_ones=False, avg_by_mouse=False, **params)[stored_key]
        spec = np.asarray(spec, dtype=float)
        if state.get("clip_negative", False):
            spec = _clip_at_first_negative(spec)
        return self._average_env_slots(spec)

    def _avg_env_svca_placefield_sessions(self, state: dict, source_key: str) -> np.ndarray:
        """Per-session SVCA placefield spectrum after averaging environment slots."""
        params = sel_params(state, self._tuple_labels, self._svca_results.param_axes)
        key = f"{self._svca_pf_keys[source_key]}_env"
        spec = self._svca_results.sel(
            keys=[key],
            squeeze_ones=False,
            avg_by_mouse=False,
            **params,
        )[key]
        spec = np.asarray(spec, dtype=float)
        if state.get("clip_negative", False):
            spec = _clip_at_first_negative(spec)
        return self._average_env_slots(spec)

    def _avg_env_spectrum(self, state: dict, key: str, cfg: SpectrumSmoothing) -> np.ndarray:
        """Environment-then-mouse averaged PF spectrum for ``source_mode='avg_env'``."""
        if key in SVCA_PF_KEYS:
            session_spec = self._avg_env_svca_placefield_sessions(state, key)
            mouse_names = self._svca_results.mouse_names
        else:
            if SOURCE_OF_KEY.get(key) != "stimspace":
                self._per_env_pf_key(key)  # raises the source-specific error
            session_spec = self._avg_env_stimspace_sessions(state, key)
            mouse_names = self.results.mouse_names
        spec = np.atleast_2d(average_by_mouse(session_spec, mouse_names))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)

    def _spectrum_sessions(
        self,
        state: dict,
        key: str,
        cfg: SpectrumSmoothing,
        *,
        apply_source_mode: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Per-session raw and smoothed ``(sessions, dims)`` spectrum for ``key``, with mouse/session ids.

        Normalize is applied per session (row) instead of after mouse-averaging, so the adaptive
        alpha fit can find each session's own first-negative crossover before any cross-session
        averaging blurs it. Both the raw (pre-smoothing) and smoothed spectrum are returned: smoothing
        maps every non-positive entry to NaN before exponentiating back (:func:`_smooth_spectrum`), so
        a smoothed row is never negative -- first-negative detection must use the raw one, while the
        exponent fit itself uses the smoothed one (matching every other alpha method).
        """
        if apply_source_mode and state.get("source_mode", "all") == "avg_env":
            if key in SVCA_PF_KEYS:
                spec = self._avg_env_svca_placefield_sessions(state, key)
                agg = self._svca_results
            else:
                if SOURCE_OF_KEY.get(key) != "stimspace":
                    self._per_env_pf_key(key)  # raises the source-specific error
                spec = self._avg_env_stimspace_sessions(state, key)
                agg = self.results
            if state["normalize"]:
                spec = spec / np.nansum(spec, axis=1)[:, None]
            smoothed = _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)
            return spec, smoothed, agg.mouse_names, agg.session_ids
        if key in SVCA_PF_KEYS:
            return self._svca_placefield_spectrum_sessions(state, key, cfg)
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        params = sel_params(state, self._tuple_labels, agg.param_axes)
        spec = agg.sel(keys=[key], avg_by_mouse=False, **params)[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state.get("clip_negative", False):
            spec = _clip_at_first_negative(spec)
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)
        return spec, smoothed, agg.mouse_names, agg.session_ids

    def _svca_placefield_spectrum(self, state: dict, source_key: str, cfg: SpectrumSmoothing) -> np.ndarray:
        """Mouse-averaged SVCA placefield spectrum from StimspaceSVCA results."""
        params = sel_params(state, self._tuple_labels, self._svca_results.param_axes)
        key = self._svca_pf_keys[source_key]
        if state.get("clip_negative", False):
            spec = self._svca_results.sel(keys=[key], avg_by_mouse=False, **params)[key]
            spec = _clip_at_first_negative(np.atleast_2d(np.asarray(spec, dtype=float)))
            spec = average_by_mouse({key: spec}, self._svca_results.mouse_names)[key]
        else:
            spec = self._svca_results.sel(keys=[key], avg_by_mouse=True, **params)[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)

    def _svca_placefield_spectrum_sessions(
        self,
        state: dict,
        source_key: str,
        cfg: SpectrumSmoothing,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Per-session raw and smoothed SVCA placefield spectrum from StimspaceSVCA results."""
        params = sel_params(state, self._tuple_labels, self._svca_results.param_axes)
        key = self._svca_pf_keys[source_key]
        spec = self._svca_results.sel(keys=[key], avg_by_mouse=False, **params)[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state.get("clip_negative", False):
            spec = _clip_at_first_negative(spec)
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)
        return spec, smoothed, self._svca_results.mouse_names, self._svca_results.session_ids

    def _svca_spectrum_sessions(
        self,
        state: dict,
        cfg: SpectrumSmoothing,
        key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list] | None:
        """Per-session raw+smoothed StimspaceSVCA activity spectrum for ``key``.

        This is both the ``full_source_key="SVCA"`` spectrum and the fixed FF-side window-fallback
        source (see :meth:`SpectrumAlphaFigureViewer.refresh_data`), fetched independently of the
        current ``full_source_key`` selection. Returns None if ``results_svca`` was not provided
        (fallback then simply isn't available).
        """
        if self._svca_results is None:
            return None
        if key is None:
            key = self._svca_full_keys["SVCA"]
        params = sel_params(state, self._tuple_labels, self._svca_results.param_axes)
        spec = self._svca_results.sel(keys=[key], avg_by_mouse=False, **params)[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state.get("clip_negative", False):
            spec = _clip_at_first_negative(spec)
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)
        return spec, smoothed, self._svca_results.mouse_names, self._svca_results.session_ids

    def _ff_spectrum_sessions(
        self,
        state: dict,
        cfg: SpectrumSmoothing,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Per-session raw+smoothed "Reliable CA1" spectrum, per ``state['full_source_key']`` (see :meth:`_ff_spectrum`)."""
        full_source_key = state.get("full_source_key", "SVD")
        if full_source_key not in SVCA_FULL_KEYS:
            return self._spectrum_sessions(
                state,
                "ffres" if full_source_key == "SVD_RES" else FF_KEY,
                cfg,
                apply_source_mode=False,
            )
        return self._svca_spectrum_sessions(state, cfg, key=self._svca_full_keys[full_source_key])

    def _ff_spectrum(self, state: dict, cfg: SpectrumSmoothing) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` "Reliable CA1" spectrum, per ``state['full_source_key']``.

        ``"SVD"``/``"SVD_RES"`` use the StimSpaceSpectra ``ff``/``ffres`` keys (via :meth:`_spectrum`).
        ``"SVCA"``/``"SVCA_RES"`` use the StimspaceSVCA ``ff``/``ff_res`` keys. All share the
        same normalize/log-space smoothing (per ``cfg``) as every other spectrum.
        """
        full_source_key = state.get("full_source_key", "SVD")
        if full_source_key not in SVCA_FULL_KEYS:
            return self._spectrum(
                state,
                "ffres" if full_source_key == "SVD_RES" else FF_KEY,
                cfg,
                apply_source_mode=False,
            )
        params = sel_params(state, self._tuple_labels, self._svca_results.param_axes)
        key = self._svca_full_keys[full_source_key]
        if state.get("clip_negative", False):
            spec = self._svca_results.sel(keys=[key], avg_by_mouse=False, **params)[key]
            spec = _clip_at_first_negative(np.atleast_2d(np.asarray(spec, dtype=float)))
            spec = average_by_mouse({key: spec}, self._svca_results.mouse_names)[key]
        else:
            spec = self._svca_results.sel(keys=[key], avg_by_mouse=True, **params)[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)

    def _ff_plot_style(self, state: dict) -> tuple[str, str]:
        """Return the label and color appropriate for the selected full-CA1 source."""
        if state.get("full_source_key", "SVD").upper().endswith("_RES"):
            return "PF Residual", "brown"
        return self.ff_label, self.ff_color

    def _draw_spectrum_annotation(
        self,
        ax,
        state: dict,
        fontsize: float,
        ff_label: str,
        ff_color: str,
    ) -> None:
        """Draw the selected definition annotation and optional dimensionality equation."""
        mode = state["show_annotation"]
        xy = (state["annotation_x"], state["annotation_y"])
        if mode == "ROIs":
            draw_reliable_spectrum_annotation(
                ax,
                xy,
                state["annotation_y_spacing"],
                state["annotation_definition_x_offset"],
                fontsize,
            )
        elif mode == "X-Cov":
            draw_xcov_spectrum_annotation(
                ax,
                xy,
                state["annotation_y_spacing"],
                fontsize,
            )
        if state["show_equation"]:
            draw_dimensionality_equation_annotation(
                ax,
                (state["equation_x"], state["equation_y"]),
                fontsize,
                pf_label=self.pf_label,
                ff_label=ff_label,
                pf_color=self.pf_color,
                ff_color=ff_color,
                linewidth=state["equation_linewidth"],
                yoffset=state["equation_yoffset"],
                yheight=state["equation_yheight"],
                arrow_mutation_scale=state["equation_arrow_mutation_scale"],
            )

    def _fit_spectrum_raw_sessions(self, state: dict, key: str) -> np.ndarray:
        """Raw (unnormalized, unsmoothed) per-session Tilbury-fit eigenvalue spectrum for ``key``.

        ``key`` is one of :data:`~._param_axes.FIT_KEYS` from the Tilbury-fit aggregator. These
        spectra vary in length across sessions but are stored as ``"pad"`` keys, so the aggregator
        NaN-pads them to a common length. Each session is converted from the PCA convention to the
        ``ss_cv`` covariance convention with ``P / (P - 1)``, gated by whichever ``params*`` arrays
        back ``key`` (see :data:`~._param_axes.FIT_KEY_PARAM_KEYS`). Every fit param axis
        (``activity_parameters_name``, ...) follows its syd widget; the reliability/fraction-active
        threshold is not a fit axis and remains fixed at :data:`~._param_axes.TILBURY_REL_FA`.
        """
        fit_params = sel_params(state, self._tuple_labels, self._fit_axes)
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

    def _fit_spectrum(
        self,
        state: dict,
        key: str,
        cfg: SpectrumSmoothing,
    ) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` Tilbury-fit eigenvalue spectrum for ``key``.

        Normalize/log-space smoothing (matching every other spectrum) are applied after averaging the
        raw per-session spectrum (:meth:`_fit_spectrum_raw_sessions`) by mouse.
        """
        session_spec = self._fit_spectrum_raw_sessions(state, key)
        if state.get("clip_negative", False):
            session_spec = _clip_at_first_negative(session_spec)
        spec = np.atleast_2d(average_by_mouse(session_spec, self.results_fit.mouse_names))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)

    def _fit_spectrum_sessions(
        self,
        state: dict,
        key: str,
        cfg: SpectrumSmoothing,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Per-session raw+smoothed Tilbury-fit eigenvalue spectrum for ``key``, with mouse/session ids.

        Normalize is applied per session rather than after mouse-averaging (see
        :meth:`_spectrum_sessions`); both the raw (pre-smoothing) and smoothed spectrum are returned
        for the adaptive alpha fit, which needs the raw one to locate the first-negative crossover.
        """
        session_spec = self._fit_spectrum_raw_sessions(state, key)
        if state.get("clip_negative", False):
            session_spec = _clip_at_first_negative(session_spec)
        if state["normalize"]:
            session_spec = session_spec / np.nansum(session_spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(session_spec, cfg.smooth_method, cfg.smooth_width)
        return session_spec, smoothed, self.results_fit.mouse_names, self.results_fit.session_ids

    def _rel_fa_matches_fit(self, state: dict) -> bool:
        """Whether the shared reliability/fraction-active selection equals the Tilbury-fit threshold.

        The Tilbury eig spectra are fixed at :data:`~._param_axes.TILBURY_REL_FA`; if the shared
        ``reliability_fraction_active_thresholds`` widget selects a different value the overlaid PF/FF
        curves are computed on a different neuron population. Returns True when there is no such shared
        widget (nothing to mismatch).
        """
        rel_axis = "reliability_fraction_active_thresholds"
        if rel_axis not in state:
            return True
        value = state[rel_axis]
        if rel_axis in self._tuple_labels:
            value = self._tuple_labels[rel_axis][value]
        return tuple(value) == TILBURY_REL_FA

    def refresh_data(self, state: dict) -> None:
        """Recompute the PF/FF/fit-overlay spectra and their signed participation ratios.

        Runs on every widget change that affects which numbers get computed (data selection,
        smoothing, ``normalize``/``clip_negative``/``pr_pre_smooth``). ``plot`` only reads the
        results cached here.
        """
        pf_key = state["source_key"]
        source_smoothing = self._smoothing_from_state(state, "source")
        full_smoothing = self._smoothing_from_state(state, "full")
        raw_smoothing = SpectrumSmoothingConfig(smooth_method="none", smooth_width=0.0)

        pf_spec = self._spectrum(state, pf_key, source_smoothing)
        ff_spec = self._ff_spectrum(state, full_smoothing)
        pf_raw = self._spectrum(state, pf_key, raw_smoothing)
        ff_raw = self._ff_spectrum(state, raw_smoothing)

        fit_keys = list(state.get("fit_key", []))
        fit_specs = {k: self._fit_spectrum(state, k, source_smoothing) for k in fit_keys}
        fit_raw = {k: self._fit_spectrum(state, k, raw_smoothing) for k in fit_keys}

        # ss_cvpca uses covariance across neurons (denominator N - 1), whereas ss_cv uses covariance
        # across positions (denominator P - 1). The required N/P counts are not stored with the
        # aggregated spectra, so align ss_cvpca post hoc using the robust low-rank amplitude ratio.
        # eig_* is already converted analytically to the ss_cv convention in _eig_to_ss_scale and
        # must not receive this factor. Note: only pf_spec (the display-smoothed curve) is rescaled,
        # matching the original behavior -- pf_raw (used for pr_pre_smooth) is left unscaled.
        if pf_key == "ss_cvpca":
            num_dim_for_scaling = 5
            pf_spec_reference = self._spectrum(state, "ss_cv", source_smoothing)
            ratio = pf_spec_reference[:, :num_dim_for_scaling] / pf_spec[:, :num_dim_for_scaling]
            scaling = np.nanmedian(ratio, axis=1)
            pf_spec = pf_spec * scaling[:, np.newaxis]

        # Optionally measure participation ratio before display smoothing. Clipping and
        # normalization precede both choices.
        pf_pr = _signed_participation_ratio(pf_raw if state["pr_pre_smooth"] else pf_spec)
        ff_pr = _signed_participation_ratio(ff_raw if state["pr_pre_smooth"] else ff_spec)
        fit_pr = {k: _signed_participation_ratio(fit_raw[k] if state["pr_pre_smooth"] else fit_specs[k]) for k in fit_keys}

        self._pf_key = pf_key
        self._fit_keys = fit_keys
        self._pf_spec = pf_spec
        self._ff_spec = ff_spec
        self._fit_specs = fit_specs
        self._pf_pr = pf_pr
        self._ff_pr = ff_pr
        self._fit_pr = fit_pr

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        each_alpha = state["each_line_alpha"]
        ylim_min = state["ylim_min"]
        ylim_max = state["ylim_max"]
        pf_color = self.pf_color
        pf_label = self.pf_label
        ff_label, ff_color = self._ff_plot_style(state)
        pf_spec, ff_spec = self._pf_spec, self._ff_spec
        fit_keys, fit_specs = self._fit_keys, self._fit_specs
        pf_pr, ff_pr, fit_pr = self._pf_pr, self._ff_pr, self._fit_pr

        fig, ax = self.new_subplots(
            2,
            1,
            figsize=self.figsize,
            layout="constrained",
            sharex=True,
            height_ratios=self.height_ratios,
        )

        # --- ax[0]: PF and FF spectra (faint per-mouse + bold average) ---
        # No fixed fit-window shading: the adaptive window is per-session, not a shared [start, end).
        for spec, label, color in (
            (pf_spec, pf_label, pf_color),
            (ff_spec, ff_label, ff_color),
        ):
            spec_positive = np.where(spec > 0, spec, np.nan)
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(_xvals(spec), np.nanmean(spec_positive, axis=0), color=color, label=label, linewidth=2.0)

        # --- ax[0] extra overlays: Tilbury-fit eig spectra (fixed rel/frac-active threshold) ---
        for key in fit_keys:
            spec = fit_specs[key]
            spec_positive = np.where(spec > 0, spec, np.nan)
            color = FIT_KEY_COLORS.get(key, "gray")
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(_xvals(spec), np.nanmean(spec_positive, axis=0), color=color, label=FIT_KEY_LABELS.get(key, key), linewidth=2.0)
        if fit_keys and not self._rel_fa_matches_fit(state):
            ax[0].set_title("REL-FA Not MATCHED!", fontsize=fontsize, color="red")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(10**ylim_min, 10**ylim_max)
        ax[0].set_ylabel("Variance", fontsize=fontsize)
        apply_legend(ax[0], state, fontsize, auto_loc="upper right")

        # Participation-ratio groups share one categorical row and the spectrum's x-axis.
        beeswarm_colors = [pf_color] + [FIT_KEY_COLORS.get(k, "gray") for k in fit_keys] + [ff_color]
        pr_values = [pf_pr] + [fit_pr[k] for k in fit_keys] + [ff_pr]

        _horizontal_beeswarm_panel(
            ax[1],
            pr_values,
            beeswarm_colors,
            beewidth=state["beewidth"],
            each_alpha=state["point_alpha"],
            markersize=state["markersize"],
            mean_linewidth=state["mean_linewidth"],
        )
        ax[1].set_xlabel("Dimension", fontsize=fontsize)

        xlim = ax[0].get_xlim()
        ax[0].set_xlim(1, xlim[1])
        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[1, xlim[1]],
            ybounds=[10**ylim_min, 10**ylim_max],
            tick_fontsize=fontsize,
        )
        format_spines(
            ax[1],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["bottom"],
            xbounds=[1, xlim[1]],
            tick_fontsize=fontsize,
        )
        self._draw_spectrum_annotation(ax[0], state, fontsize, ff_label, ff_color)
        return fig


class SpectrumAlphaFigureViewer(SpectrumFigureViewer):
    """Spectrum comparison with the original adaptive decay-exponent panel.

    ax[0] retains the selectable PF and full-CA1 spectra from :class:`SpectrumFigureViewer`. ax[1]
    shows the per-mouse adaptive median-FPD decay exponent for those spectra (and any selected
    Tilbury-fit overlays). There is no participation-ratio panel.

    Parameters
    ----------
    results, results_cvpca, results_svca, results_fit : ResultsAggregator or None
        See :class:`SpectrumFigureViewer`.
    source_key, full_source_key, fit_key : see :class:`SpectrumFigureViewer`.
    source_cfg, full_cfg : AdaptiveAlphaConfig or None
        Independent smoothing and adaptive median-FPD settings for the PF and full-CA1 sides.
        Defaults to :data:`~._alpha_config.ADAPTIVE_ALPHA_CONFIG_REGISTRY`'s ``"placefields"``/
        ``"full"`` entries. These also seed the inherited ``source_smooth_*``/``full_smooth_*``
        widgets.
    normalize, clip_negative : see :class:`SpectrumFigureViewer`. (``pr_pre_smooth`` is inherited
        but unused here -- there is no participation-ratio panel.)
    ylim_min, ylim_max : float
        ax[0] y-limits in log10 units.
    beewidth : float
        Horizontal spread of each decay-exponent swarm around its categorical x-position.
    each_line_alpha : float
        Alpha of individual-mouse spectrum lines in ax[0].
    point_alpha : float
        Alpha of individual-mouse decay-exponent points in ax[1].
    markersize : float
        Decay-exponent marker size in points.
    mean_linewidth : float
        Width of the mean markers in the decay-exponent panel.
    fontsize : float
        Font size for every text element.
    figsize : tuple[float, float]
        Figure size in inches.
    width_ratios : tuple[float, float]
        Relative widths of the spectrum and decay-exponent panels.
    pf_color, ff_color, pf_label, ff_label : see :class:`SpectrumFigureViewer`.
    **selection_defaults
        Forwarded to :class:`SpectrumFigureViewer`.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        results_fit: ResultsAggregator | None = None,
        results_svca: ResultsAggregator | None = None,
        *,
        source_key: str = "ss_cv",
        source_mode: str = "all",
        full_source_key: str = "SVD",
        fit_key: str | list[str] = (),
        source_cfg: AdaptiveAlphaConfig | None = None,
        full_cfg: AdaptiveAlphaConfig | None = None,
        normalize: bool = True,
        clip_negative: bool = False,
        ylim_min: float = -5.5,
        ylim_max: float = 0.0,
        beewidth: float = 0.15,
        each_line_alpha: float = 0.3,
        point_alpha: float = 0.3,
        markersize: float = 3.0,
        mean_linewidth: float = 2.0,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (5.0, 3.0),
        width_ratios: tuple[float, float] = (1.0, 0.5),
        pf_color: str = "orange",
        ff_color: str = "black",
        pf_label: str = "Placefields",
        ff_label: str = "Full CA1",
        **selection_defaults,
    ):
        self.source_cfg = source_cfg if source_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]
        self.full_cfg = full_cfg if full_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["full"]

        # Registered before super().__init__() (which ends by calling self.refresh_data, dispatched
        # dynamically to this class's override below) so that override's state lookups already have
        # these widgets to read.
        for prefix, cfg in (("source", self.source_cfg), ("full", self.full_cfg)):
            self.add_integer(f"{prefix}_fpd_window_size", value=cfg.fpd_window_size, min=1, max=50)
            self.add_integer(f"{prefix}_adaptive_buffer", value=cfg.adaptive_buffer, min=0, max=50)
            self.add_integer(f"{prefix}_minimum_window_size", value=cfg.minimum_window_size, min=1, max=500)

        super().__init__(
            results,
            results_cvpca=results_cvpca,
            results_subspace=results_subspace,
            results_fit=results_fit,
            results_svca=results_svca,
            source_key=source_key,
            source_mode=source_mode,
            full_source_key=full_source_key,
            fit_key=fit_key,
            source_smooth_method=self.source_cfg.smooth_method,
            source_smooth_width=self.source_cfg.smooth_width,
            full_smooth_method=self.full_cfg.smooth_method,
            full_smooth_width=self.full_cfg.smooth_width,
            normalize=normalize,
            clip_negative=clip_negative,
            ylim_min=ylim_min,
            ylim_max=ylim_max,
            beewidth=beewidth,
            each_line_alpha=each_line_alpha,
            point_alpha=point_alpha,
            markersize=markersize,
            mean_linewidth=mean_linewidth,
            fontsize=fontsize,
            figsize=figsize,
            pf_color=pf_color,
            ff_color=ff_color,
            pf_label=pf_label,
            ff_label=ff_label,
            **selection_defaults,
        )
        self.width_ratios = width_ratios

    def refresh_data(self, state: dict) -> None:
        """Recompute the base spectra plus each side's per-mouse adaptive median-FPD decay exponent."""
        super().refresh_data(state)
        source_cfg = self._cfg_from_state(state, "source")
        full_cfg = self._cfg_from_state(state, "full")
        pf_key = self._pf_key
        fit_keys = self._fit_keys

        # Estimate alpha per session using each session's own peak-curvature-to-noise-floor window,
        # then average the estimates by mouse. Non-cross-validated PF/fit spectra borrow window
        # boundaries from ss_cvpca; the FF side borrows them from SVCA when available.
        pf_raw, pf_smooth, pf_mouse_names, pf_session_ids = self._spectrum_sessions(state, pf_key, source_cfg)
        ff_raw, ff_smooth, ff_mouse_names, ff_session_ids = self._ff_spectrum_sessions(state, full_cfg)
        cvpca_raw, cvpca_smooth, _, cvpca_session_ids = self._spectrum_sessions(state, "ss_cvpca", source_cfg)
        fit_cvpca_raw, fit_cvpca_smooth, fit_cvpca_session_ids = cvpca_raw, cvpca_smooth, cvpca_session_ids
        if state.get("source_mode", "all") == "avg_env" and fit_keys:
            # source_mode belongs only to source_key. Fit overlays retain their whole-session
            # ss_cvpca fallback just as their displayed spectra retain whole-session behavior.
            fit_cvpca_raw, fit_cvpca_smooth, _, fit_cvpca_session_ids = self._spectrum_sessions(
                state,
                "ss_cvpca",
                source_cfg,
                apply_source_mode=False,
            )
        svca = self._svca_spectrum_sessions(state, full_cfg)

        pf_fb_raw = _align_rows_to_sessions(pf_session_ids, cvpca_session_ids, cvpca_raw)
        pf_fb_smooth = _align_rows_to_sessions(pf_session_ids, cvpca_session_ids, cvpca_smooth)
        ff_fb_raw, ff_fb_smooth = None, None
        if svca is not None:
            svca_raw, svca_smooth, _, svca_session_ids = svca
            ff_fb_raw = _align_rows_to_sessions(ff_session_ids, svca_session_ids, svca_raw)
            ff_fb_smooth = _align_rows_to_sessions(ff_session_ids, svca_session_ids, svca_smooth)

        self._pf_alpha = average_by_mouse(
            _median_fpd_alpha_per_session(
                pf_raw,
                pf_smooth,
                source_cfg.fpd_window_size,
                source_cfg.adaptive_buffer,
                source_cfg.minimum_window_size,
                pf_fb_raw,
                pf_fb_smooth,
            ),
            pf_mouse_names,
        )
        self._ff_alpha = average_by_mouse(
            _median_fpd_alpha_per_session(
                ff_raw,
                ff_smooth,
                full_cfg.fpd_window_size,
                full_cfg.adaptive_buffer,
                full_cfg.minimum_window_size,
                ff_fb_raw,
                ff_fb_smooth,
            ),
            ff_mouse_names,
        )
        fit_alpha = {}
        for key in fit_keys:
            fit_raw, fit_smooth, fit_mouse_names, fit_session_ids = self._fit_spectrum_sessions(state, key, source_cfg)
            fit_fb_raw = _align_rows_to_sessions(fit_session_ids, fit_cvpca_session_ids, fit_cvpca_raw)
            fit_fb_smooth = _align_rows_to_sessions(fit_session_ids, fit_cvpca_session_ids, fit_cvpca_smooth)
            fit_alpha[key] = average_by_mouse(
                _median_fpd_alpha_per_session(
                    fit_raw,
                    fit_smooth,
                    source_cfg.fpd_window_size,
                    source_cfg.adaptive_buffer,
                    source_cfg.minimum_window_size,
                    fit_fb_raw,
                    fit_fb_smooth,
                ),
                fit_mouse_names,
            )
        self._fit_alpha = fit_alpha

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        each_alpha = state["each_line_alpha"]
        ylim_min = state["ylim_min"]
        ylim_max = state["ylim_max"]
        pf_spec, ff_spec = self._pf_spec, self._ff_spec
        fit_keys, fit_specs = self._fit_keys, self._fit_specs
        ff_label, ff_color = self._ff_plot_style(state)

        fig, ax = self.new_subplots(1, 2, figsize=self.figsize, layout="constrained", width_ratios=self.width_ratios)

        for spec, label, color in (
            (pf_spec, self.pf_label, self.pf_color),
            (ff_spec, ff_label, ff_color),
        ):
            spec_positive = np.where(spec > 0, spec, np.nan)
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(_xvals(spec), np.nanmean(spec_positive, axis=0), color=color, label=label, linewidth=2.0)

        for key in fit_keys:
            spec = fit_specs[key]
            spec_positive = np.where(spec > 0, spec, np.nan)
            color = FIT_KEY_COLORS.get(key, "gray")
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(
                _xvals(spec),
                np.nanmean(spec_positive, axis=0),
                color=color,
                label=FIT_KEY_LABELS.get(key, key),
                linewidth=2.0,
            )
        if fit_keys and not self._rel_fa_matches_fit(state):
            ax[0].set_title("REL-FA Not MATCHED!", fontsize=fontsize, color="red")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(10**ylim_min, 10**ylim_max)
        ax[0].set_xlabel("Shared Dimension", fontsize=fontsize)
        ax[0].set_ylabel("Variance", fontsize=fontsize)
        apply_legend(ax[0], state, fontsize, auto_loc="upper right")
        xlim = ax[0].get_xlim()
        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[1, xlim[1]],
            ybounds=[10**ylim_min, 10**ylim_max],
            tick_fontsize=fontsize,
        )

        colors = [self.pf_color] + [FIT_KEY_COLORS.get(key, "gray") for key in fit_keys] + [ff_color]
        labels = ["PF"] + [FIT_KEY_LABELS.get(key, key) for key in fit_keys] + ["CA1"]
        alpha_values = [self._pf_alpha] + [self._fit_alpha[key] for key in fit_keys] + [self._ff_alpha]
        _beeswarm_panel(
            ax[1],
            alpha_values,
            colors,
            labels,
            fontsize,
            beewidth=state["beewidth"],
            each_alpha=state["point_alpha"],
            markersize=state["markersize"],
            mean_linewidth=state["mean_linewidth"],
        )
        ax[1].set_ylabel("Decay exponent", fontsize=fontsize)
        self._draw_spectrum_annotation(ax[0], state, fontsize, ff_label, ff_color)
        return fig
