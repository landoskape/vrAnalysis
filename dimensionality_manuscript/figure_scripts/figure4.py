from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from scipy.stats import gaussian_kde, ttest_rel, wilcoxon
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from syd import Viewer

from vrAnalysis.helpers import edge2center
from vrAnalysis.helpers.plotting import save_figure, beeswarm, format_spines, errorPlot
from dimilibi.helpers import fit_powerlaw_decay, fit_powerlaw_derivatives
from dimensionality_manuscript import ResultsAggregator, average_by_mouse
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.configs.tilbury_fit import TilburyFitConfig, _eval_tilbury, _eval_gaussian, _SPLITS
from dimensionality_manuscript.env_order import ENV_SLOT_COLORS, MAX_ENV_SLOTS
from .legends import LEGEND_LOCS, LEGEND_KNOBS, add_legend_widgets, update_legend_widgets, apply_legend

# Selectable spectrum keys and which aggregator each one comes from. StimSpace keys resolve
# against the StimSpaceSpectra aggregator; CVPCA keys against the CVPCAConfig aggregator.
_STIMSPACE_KEYS = ["ss_cv", "ss_direct", "ss_cvpca", "sf_cv", "sf_direct"]
_CVPCA_KEYS = ["reg_covariances_fixed"]
# The full (functional) spectrum key, also from the StimSpaceSpectra aggregator.
_FF_KEY = "ff"

# Tilbury-fit eigenvalue spectra selectable as extra ax[0] overlays in SpectrumFigureViewer, with
# fixed colors. These come from the TilburyFitConfig aggregator, which fits only reliable/active
# neurons at a single fixed reliability/fraction-active threshold (no selection axis).
_FIT_KEYS = [
    "eig_tilbury",
    "eig_control",
    "eig_shrinkage",
    "eig_better",
]
_FIT_KEY_COLORS = {
    "eig_tilbury": "blue",
    "eig_control": "green",
    "eig_shrinkage": "purple",
    "eig_better": "red",
}
_FIT_KEY_LABELS = {
    "eig_tilbury": "Generalized",
    "eig_control": "Gaussian",
    "eig_shrinkage": "Generalized (shrinkage)",
    "eig_better": "Best (Single)",
}
# Which per-neuron fit ``params*`` arrays gate each fit_key's neuron count (matching the
# ``ok_*`` masks TilburyFitConfig.process used to build that key's underlying curve matrix):
# eig_tilbury/eig_control/eig_better all come from ``mat_tilbury``/``mat_control``/``mat_better``,
# which TilburyFitConfig built from ``ok_both`` (params & params_control both finite);
# eig_shrinkage comes from ``ok_s`` (params_shrinkage finite).
_FIT_KEY_PARAM_KEYS = {
    "eig_tilbury": ["params", "params_control"],
    "eig_control": ["params", "params_control"],
    "eig_shrinkage": ["params_shrinkage"],
    "eig_better": ["params", "params_control"],
}
_TILBURY_REL_FA = (0.3, 0.1)

# Population alpha-comparison panel (ax[-1] of PlacefieldPopulationViewer): the selected source_key
# spectrum plus the four Tilbury-fit eigenvalue spectra, each a per-mouse power-law-exponent
# beeswarm, in plotting order (shrinkage sits between the composite and the unregularized
# generalized fit). "eig_gaussian" in the request is the plain-Gaussian control key ``eig_control``.
_POP_EIG_KEYS = ["eig_better", "eig_shrinkage", "eig_tilbury", "eig_control"]
_POP_ALPHA_COLORS = {
    "source_key": "darkorange",
    "eig_better": "red",
    "eig_tilbury": "blue",
    "eig_shrinkage": "purple",
    "eig_control": "black",
}
_POP_ALPHA_LABELS = {
    "eig_better": "Better",
    "eig_tilbury": "Generalized",
    "eig_shrinkage": "Shrinkage",
    "eig_control": "Gaussian",
}
# key -> "stimspace" | "cvpca"
SOURCE_OF_KEY = {
    **{k: "stimspace" for k in _STIMSPACE_KEYS},
    **{k: "cvpca" for k in _CVPCA_KEYS},
    _FF_KEY: "stimspace",
}

# Fixed color per curve option, used for the per-mouse alpha scatter (ax[1]) and the
# local-exponent curves (ax[2]) so a given curve reads the same across panels.
_KEY_COLORS = {
    "ss_cv": "black",
    "ss_direct": "blue",
    "ss_cvpca": "red",
    "sf_cv": "orange",
    "sf_direct": "cyan",
    "reg_covariances_fixed": "green",
    "ff": "purple",
}

# Preferred default values for shared param widgets, keyed by raw param-axis name. A widget is
# seeded with this value when the axis exists (in any source) and the value is among its options.
_PREFERRED_DEFAULTS = {
    "activity_parameters_name": "default",
    "include_iti": False,
    "spks_type": "sigrebase",
    "center": True,
    "use_fast_sampling": True,
}


def _xvals(x: np.ndarray) -> np.ndarray:
    """Return 1-based dimension indices for a (mice, dims) spectrum array."""
    return np.arange(x.shape[1]) + 1


def _smooth_kernel(kind: str, width: float) -> np.ndarray:
    """Normalized 1-D smoothing kernel over rank index.

    ``width`` is the boxcar full-width in rank units. The Gaussian uses ``sigma = width / 2`` so
    its ``+/- 1 sigma`` bulk spans the same window as the boxcar.
    """
    if kind == "boxcar":
        length = max(1, int(round(width)))
        return np.ones(length) / length
    if kind == "gaussian":
        sigma = width / 2.0
        radius = max(1, int(np.ceil(3.0 * sigma)))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return kernel / kernel.sum()
    raise ValueError(f"Unknown smoothing kind {kind!r}.")


def _smooth_spectrum(spec: np.ndarray, kind: str, width: float) -> np.ndarray:
    """Geometric-mean (log-space) smoothing of a ``(mice, dims)`` spectrum along ranks.

    Smoothing is done on ``log(spec)`` (slope-preserving for power laws) with a NaN-aware weighted
    convolution: non-positive/NaN entries are excluded and the kernel is renormalized per output
    point (which also handles edges). ``kind == "none"`` or ``width <= 0`` returns ``spec`` unchanged.
    """
    if kind == "none" or width <= 0:
        return spec
    kernel = _smooth_kernel(kind, width)
    with np.errstate(invalid="ignore"):
        logspec = np.where(spec > 0, np.log(spec), np.nan)
    mask = np.isfinite(logspec)
    filled = np.where(mask, logspec, 0.0)
    num = np.stack([np.convolve(row, kernel, mode="same") for row in filled])
    den = np.stack([np.convolve(row, kernel, mode="same") for row in mask.astype(float)])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return np.exp(out)


def _eig_to_ss_scale(
    eig: np.ndarray,
    params_list: list[np.ndarray],
    dist_centers: np.ndarray,
) -> np.ndarray:
    """Put one fitted placefield eigenspectrum on the ``ss_cv`` variance scale.

    ``_curve_spectrum`` divides squared singular values by the number of positions ``P``,
    while ``ss_cv`` uses the sample-covariance convention and divides by ``P - 1``.
    Thus the conversion factor is ``P / (P - 1)``. ``params_list`` holds every per-neuron fit
    ``params*`` array that gates this eig key's underlying neuron count (see
    :data:`_FIT_KEY_PARAM_KEYS`); a neuron counts only if finite in all of them, so sessions
    without at least two jointly valid fitted neurons still produce NaNs.
    """
    valid_fit = np.ones(np.asarray(params_list[0]).shape[0], dtype=bool)
    for params in params_list:
        valid_fit &= np.isfinite(params).all(axis=-1)
    n_fit = int(np.count_nonzero(valid_fit))
    n_positions = int(np.asarray(dist_centers).size)
    if n_fit < 2 or n_positions < 2:
        return np.full_like(np.asarray(eig, dtype=float), np.nan)
    return np.asarray(eig, dtype=float) * n_positions / (n_positions - 1)


def _tuple_label(value: tuple) -> str:
    """Render a tuple param value (elements are float or None) as a widget-safe string label."""
    return "-".join("None" if v is None else str(v) for v in value)


def _clamp_range(start: int, end: int, n: int) -> tuple[int, int]:
    """Clamp a [start, end) index window to a spectrum of length ``n``."""
    end = int(min(end, n))
    start = int(min(max(start, 0), max(end - 1, 0)))
    return start, end


def _decay_alpha_per_mouse(spec: np.ndarray, start: int, end: int) -> np.ndarray:
    """Power-law exponent per mouse from a log-log fit over ranks ``[start, end)``.

    ``spec`` is ``(mice, dims)``; returns one alpha per row (NaN where the fit fails).
    """
    start, end = _clamp_range(start, end, spec.shape[1])
    alphas = np.full(spec.shape[0], np.nan)
    if end - start < 2:
        return alphas
    for m in range(spec.shape[0]):
        try:
            alphas[m], _ = fit_powerlaw_decay(spec[m], start_idx=start, end_idx=end, ignore_nans=True, verbose=False)
        except (ValueError, RuntimeError, TypeError):
            alphas[m] = np.nan
    return alphas


def _local_alpha_curve(spec: np.ndarray, width: int, eps: float = 1e-8) -> np.ndarray:
    """Per-rank local exponent (five-point derivative) for each mouse.

    ``spec`` is ``(mice, dims)``; returns a same-shaped array, NaN-padded where the stencil
    does not reach (edges) or the spectrum is non-positive.
    """
    n = spec.shape[1]
    out = np.full(spec.shape, np.nan, dtype=float)
    if n >= 4 * width + 1:
        alpha_local, idx_slice = fit_powerlaw_derivatives(spec, width=width, axis=1, eps=eps)
        out[:, idx_slice] = np.asarray(alpha_local, dtype=float)
    return out


def _deriv_alpha_per_mouse(local_alpha: np.ndarray, start: int, end: int) -> np.ndarray:
    """Per-mouse exponent from the mean local exponent over ranks ``[start, end)``."""
    start, end = _clamp_range(start, end, local_alpha.shape[1])
    if end - start < 1:
        return np.full(local_alpha.shape[0], np.nan)
    return np.nanmean(local_alpha[:, start:end], axis=1)


def _first_negative_index(row: np.ndarray) -> int:
    """Index of the first negative entry in ``row``, or ``row.size`` if none is negative."""
    neg = np.where(row < 0)[0]
    return int(neg[0]) if neg.size else row.size


_MEDIAN_FPD_MIN_VALUES = 5


def _second_derivative_window(raw_row: np.ndarray, fit_row: np.ndarray, buffer: int) -> tuple[int, int]:
    """Peak-curvature-to-noise-floor window on ``fit_row`` (a smoothed spectrum row).

    The window start is ``buffer`` dims after the peak of ``fit_row``'s second derivative (its
    point of maximum curvature -- the knee where the decay rate settles), searched only up to
    ``raw_row``'s (pre-smoothing) first negative entry -- past that point the second derivative is
    noise-floor curvature, not signal, and can dwarf the real peak. The peak's derivative-array index
    is converted back to ``fit_row``'s index (a second difference at derivative-index ``j`` is
    centered on original index ``j + 1``). The window end is ``buffer`` dims before that same first
    negative entry -- smoothing maps every non-positive entry to NaN before exponentiating back
    (:func:`_smooth_spectrum`), so a smoothed row is never negative and the noise-floor onset must be
    found on the raw one.

    Returns ``(start, end)`` as 0-based indices into ``fit_row``/``raw_row`` (``end`` exclusive);
    ``end <= start`` if no valid window exists (e.g. ``fit_row`` too short, or an all-NaN second
    derivative before the first negative entry).
    """
    second_derivative = np.diff(fit_row, n=2)
    first_neg = _first_negative_index(raw_row)
    # second_derivative[j] is centered on original index j + 1, so restrict the peak search to
    # j + 1 < first_neg.
    search_limit = min(max(first_neg - 1, 0), second_derivative.size)
    candidate = second_derivative[:search_limit]
    if candidate.size == 0 or not np.any(np.isfinite(candidate)):
        return 0, 0
    start = int(np.nanargmax(candidate)) + 1 + buffer
    end = first_neg - buffer
    return start, end


def _align_rows_to_sessions(target_session_ids: list, source_session_ids: list, rows: np.ndarray) -> np.ndarray:
    """Reindex ``rows`` (row ``i`` <-> ``source_session_ids[i]``) onto ``target_session_ids`` order.

    A target session absent from ``source_session_ids`` gets an all-NaN row. Needed because a
    fallback source and its target can come from different :class:`ResultsAggregator` instances
    with different session coverage/ordering -- a positional zip would silently misalign sessions.
    """
    index = {sid: i for i, sid in enumerate(source_session_ids)}
    out = np.full((len(target_session_ids), rows.shape[1]), np.nan, dtype=float)
    for i, sid in enumerate(target_session_ids):
        j = index.get(sid)
        if j is not None:
            out[i] = rows[j]
    return out


def _median_fpd_alpha_session(
    raw_row: np.ndarray,
    fit_row: np.ndarray,
    deriv_width: int,
    buffer: int,
    min_window_size: int = _MEDIAN_FPD_MIN_VALUES,
    fallback_raw_row: np.ndarray | None = None,
    fallback_fit_row: np.ndarray | None = None,
) -> float:
    """Median five-point-derivative local exponent over one session's peak-to-noise-floor window.

    See :func:`_second_derivative_window` for the window definition. Returns NaN if the window has
    fewer than ``min_window_size`` finite local-exponent values to take the median of.

    If ``raw_row`` has no negative entry (not cross-validated, so no noise-floor onset to find) and
    a ``fallback_raw_row``/``fallback_fit_row`` pair is given, the window is instead located on the
    fallback pair -- the local-exponent curve (and its median) is still always computed from
    ``fit_row`` itself, only the window boundaries are borrowed.
    """
    window_raw_row, window_fit_row = raw_row, fit_row
    if fallback_raw_row is not None and _first_negative_index(raw_row) == raw_row.size and np.any(np.isfinite(fallback_raw_row)):
        window_raw_row, window_fit_row = fallback_raw_row, fallback_fit_row
    start, end = _second_derivative_window(window_raw_row, window_fit_row, buffer)
    if end <= start:
        return np.nan
    local_alpha = _local_alpha_curve(fit_row[None, :], deriv_width)[0]
    window = local_alpha[start:end]
    finite = window[np.isfinite(window)]
    if finite.size < min_window_size:
        return np.nan
    return float(np.median(finite))


def _median_fpd_alpha_per_session(
    raw_spec: np.ndarray,
    fit_spec: np.ndarray,
    deriv_width: int,
    buffer: int,
    min_window_size: int = _MEDIAN_FPD_MIN_VALUES,
    fallback_raw_spec: np.ndarray | None = None,
    fallback_fit_spec: np.ndarray | None = None,
) -> np.ndarray:
    """Median-FPD exponent for every session row (see :func:`_median_fpd_alpha_session`).

    ``raw_spec`` and ``fit_spec`` are paired row-wise (raw locates each session's noise-floor onset;
    fit is what the local exponent is computed from). ``fallback_raw_spec``/``fallback_fit_spec``, if
    given, are paired row-wise with ``raw_spec``/``fit_spec`` too (align them first, e.g. via
    :func:`_align_rows_to_sessions`, if they come from a different aggregator).
    """
    if fallback_raw_spec is None:
        return np.array(
            [_median_fpd_alpha_session(raw_row, fit_row, deriv_width, buffer, min_window_size) for raw_row, fit_row in zip(raw_spec, fit_spec)],
            dtype=float,
        )
    return np.array(
        [
            _median_fpd_alpha_session(raw_row, fit_row, deriv_width, buffer, min_window_size, fb_raw, fb_fit)
            for raw_row, fit_row, fb_raw, fb_fit in zip(raw_spec, fit_spec, fallback_raw_spec, fallback_fit_spec)
        ],
        dtype=float,
    )


# --- Log-space decay-model fits: power law vs exponential ---------------------------------------
# For each spectrum, fit two candidate decay laws and report both how well each describes the
# spectrum (log-space MSE) and each fit's characteristic parameter, following Tilbury et al.'s
# comparison of ``n^-alpha`` against ``exp(-n^2 / 2 M^2)`` fits. Both models are linear in log-space:
#   power law      log(lambda) = c - alpha * log(n)      (feature log n; parameter alpha)
#   exponential    log(lambda) = c - n^2 / (2 M^2)       (feature n^2;   parameter M)
# so each fit is one ordinary least squares of ``log(lambda)`` on the model's feature. The MSE is
# the mean squared log-space residual over the fit window; because both share ``log(lambda)`` as the
# response, a constant rescaling of the spectrum (normalization, ss_cvpca scaling) only shifts the
# intercept and leaves the MSE unchanged. The reported parameter comes from the fitted slope:
# ``alpha = -slope`` for the power law, ``M = 1 / sqrt(-2 * slope)`` for the exponential. The xtick
# labels are the reduced equations, as drawn in the Tilbury paper.
_DECAY_MODELS: tuple[tuple[str, str], ...] = (
    ("power", r"$n^{-\alpha}$"),
    ("exp2", r"$e^{-n^2/2M^2}$"),
)


def _decay_feature(model: str, n: np.ndarray) -> np.ndarray:
    """Log-space regressor for one decay model at 1-based ranks ``n``."""
    if model == "power":
        return np.log(n)
    if model == "exp2":
        return n**2
    raise ValueError(f"Unknown decay model {model!r}. Available: {[m for m, _ in _DECAY_MODELS]}")


def _decay_param_from_slope(model: str, slope: float) -> float:
    """Characteristic parameter of a decay fit from its log-space slope.

    Power law ``log(lambda) = c - alpha * log(n)`` -> ``alpha = -slope``. Exponential
    ``log(lambda) = c - n^2 / (2 M^2)`` -> slope on ``n^2`` is ``-1 / (2 M^2)``, so
    ``M = 1 / sqrt(-2 * slope)`` (NaN for a non-decaying, non-negative slope).
    """
    if model == "power":
        return float(-slope)
    if slope < 0:
        return float(1.0 / np.sqrt(-2.0 * slope))
    return np.nan


def _logspace_decay_fit(row: np.ndarray, start: int, end: int, model: str) -> tuple[float, float]:
    """Log-space MSE and characteristic parameter of one decay-model fit over ranks ``[start, end)``.

    ``row`` is one spectrum (a single mouse or session). The fit is a degree-1 least squares of
    ``log(row)`` on :func:`_decay_feature`; ranks are 1-based (``index + 1``), matching
    :func:`~dimilibi.helpers.fit_powerlaw_decay`. Non-positive/NaN entries are dropped. Returns
    ``(mse, param)`` -- ``param`` is ``alpha`` (power law) or ``M`` (exponential), see
    :func:`_decay_param_from_slope`. Both are NaN with fewer than three usable points (too few to fit
    two parameters and leave a residual).
    """
    start, end = _clamp_range(start, end, row.size)
    if end - start < 3:
        return np.nan, np.nan
    n = np.arange(start, end) + 1.0
    values = np.asarray(row[start:end], dtype=float)
    with np.errstate(invalid="ignore"):
        log_lambda = np.where(values > 0, np.log(values), np.nan)
    feature = _decay_feature(model, n)
    mask = np.isfinite(log_lambda) & np.isfinite(feature)
    if int(mask.sum()) < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(feature[mask], log_lambda[mask], 1)
    residual = log_lambda[mask] - (slope * feature[mask] + intercept)
    return float(np.mean(residual**2)), _decay_param_from_slope(model, slope)


def _decay_window_session(
    raw_row: np.ndarray,
    fit_row: np.ndarray,
    buffer: int,
    fallback_raw_row: np.ndarray | None = None,
    fallback_fit_row: np.ndarray | None = None,
) -> tuple[int, int]:
    """Adaptive peak-to-noise-floor fit window for one session (see :func:`_second_derivative_window`).

    Mirrors :func:`_median_fpd_alpha_session`'s window selection: if ``raw_row`` has no negative
    entry (not cross-validated, so no noise-floor onset to find) and a fallback pair is given, the
    window boundaries are borrowed from the fallback spectrum instead.
    """
    window_raw_row, window_fit_row = raw_row, fit_row
    if fallback_raw_row is not None and _first_negative_index(raw_row) == raw_row.size and np.any(np.isfinite(fallback_raw_row)):
        window_raw_row, window_fit_row = fallback_raw_row, fallback_fit_row
    return _second_derivative_window(window_raw_row, window_fit_row, buffer)


def _decay_fit_per_session(
    raw_spec: np.ndarray,
    fit_spec: np.ndarray,
    model: str,
    fit_zone: str,
    fixed_range: tuple[int, int],
    buffer: int,
    fallback_raw_spec: np.ndarray | None = None,
    fallback_fit_spec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-session log-space MSE and characteristic parameter of one decay model.

    ``fit_zone == "fixed"`` uses ``fixed_range`` (a ``(start, end)`` index window) for every session;
    ``"adaptive"`` locates each session's own peak-curvature-to-noise-floor window via
    :func:`_decay_window_session` (with the ``ss_cvpca`` fallback for non-cross-validated rows). The
    fit itself always uses the smoothed spectrum ``fit_spec`` (matching the adaptive-alpha path);
    ``raw_spec`` only locates the adaptive window.

    Returns ``(mse, param)`` arrays, each shape ``(n_sessions,)`` -- ``param`` is ``alpha`` (power
    law) or ``M`` (exponential), see :func:`_logspace_decay_fit`.
    """
    mse = np.full(raw_spec.shape[0], np.nan)
    param = np.full(raw_spec.shape[0], np.nan)
    for i in range(raw_spec.shape[0]):
        if fit_zone == "fixed":
            start, end = fixed_range
        else:
            fb_raw = fallback_raw_spec[i] if fallback_raw_spec is not None else None
            fb_fit = fallback_fit_spec[i] if fallback_fit_spec is not None else None
            start, end = _decay_window_session(raw_spec[i], fit_spec[i], buffer, fb_raw, fb_fit)
        mse[i], param[i] = _logspace_decay_fit(fit_spec[i], start, end, model)
    return mse, param


def _zero_to_max_ticks(data_list, step: float = 5.0) -> tuple[tuple[float, float], list[float]]:
    """``(ybounds, yticks)`` for a non-negative statistic: the range ``[0, max]`` over ``data_list``.

    Ticks are ``0`` and the largest multiple of ``step`` at or below the data maximum (just ``[0]``
    when the maximum falls below one step). ``data_list`` is any collection of arrays; non-finite
    entries are ignored.
    """
    finite = np.concatenate([np.asarray(d, dtype=float).ravel() for d in data_list]) if len(data_list) else np.array([])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0), [0.0]
    ymax = float(np.max(finite))
    top_tick = step * np.floor(ymax / step)
    return (0.0, ymax), [0.0] if top_tick <= 0 else [0.0, float(top_tick)]


def _format_stat_spines(ax, xbounds, ybounds, yticks) -> None:
    """:func:`format_spines` for a decay-stat panel, honoring an explicit y range / ticks when given.

    With ``ybounds`` the axis bottom is pinned there (the top keeps matplotlib's padded auto limit) and
    the y spine is bounded to it; without it the spine spans the current limits, as before.
    """
    if ybounds is not None:
        ax.set_ylim(bottom=ybounds[0])
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=list(xbounds),
        ybounds=list(ybounds) if ybounds is not None else list(ax.get_ylim()),
        **({} if yticks is None else {"yticks": list(yticks)}),
    )


def _decay_stat_panel(
    ax,
    data_list,
    colors,
    labels,
    display: str,
    beewidth: float,
    fontsize: float,
    xtick_labels,
    ybounds: tuple[float, float] | None = None,
    yticks=None,
) -> None:
    """One panel of a per-curve statistic at the two decay-model x-positions (power law, exponential).

    ``data_list`` holds one ``(n_mice, 2)`` array per curve option (its two columns are the power-law
    and exponential values). NaN mice are dropped by every mode.

    - ``display == "each"``: one faint per-mouse line across x=``[0, 1]`` per curve, plus a bold
      across-mouse mean line.
    - ``display == "errorPlot"``: the across-mouse mean +/- SE band per curve (via
      :func:`~vrAnalysis.helpers.plotting.errorPlot`).
    - ``display == "swarm"``: no per-mouse connections -- one beeswarm column per (decay model, curve)
      at ``x = model_index * n_curves + curve_index`` (blocks ``[0 .. n_curves-1]`` for the power law,
      then ``[n_curves .. 2*n_curves-1]`` for the exponential), each with a short horizontal mean line.
      ``beewidth`` sets the point spread; the reduced-equation label sits under each block.

    ``ybounds``/``yticks`` override the y range and ticks (see :func:`_format_stat_spines`); by default
    the y spine spans the automatic limits and keeps matplotlib's ticks.
    """
    if display == "swarm":
        n_curves = len(data_list)
        line_extent = np.array([-0.25, 0.25])
        for j in range(2):  # decay model column: 0 = power law, 1 = exponential
            for k, (data, color) in enumerate(zip(data_list, colors)):
                vals = np.asarray(data, dtype=float)[:, j]
                x = j * n_curves + k
                offsets = np.zeros_like(vals)
                finite = np.isfinite(vals)
                if finite.any():
                    offsets[finite] = beeswarm(vals[finite])
                ax.plot(
                    x + beewidth * offsets,
                    vals,
                    color=color,
                    linestyle="none",
                    marker="o",
                    markersize=3,
                    alpha=0.3,
                    label=labels[k] if j == 0 else None,
                )
                ax.plot(x + line_extent, [np.nanmean(vals)] * 2, color=color, linewidth=2.0)
        n_pos = 2 * n_curves
        ax.set_xlim(-0.5, n_pos - 0.5)
        centers = [(n_curves - 1) / 2.0, n_curves + (n_curves - 1) / 2.0]
        _format_stat_spines(ax, [0, n_pos - 1], ybounds, yticks)
        ax.set_xticks(centers, labels=xtick_labels, fontsize=fontsize)
        return

    x = np.array([0.0, 1.0])
    for data, color, label in zip(data_list, colors, labels):
        data = np.atleast_2d(np.asarray(data, dtype=float))
        if display == "errorPlot":
            errorPlot(x, data, axis=0, se=True, ax=ax, color=color, alpha=0.25, label=label, linewidth=2.0)
        else:  # "each"
            ax.plot(x, data.T, color=color, alpha=0.3, linewidth=0.8)
            ax.plot(x, np.nanmean(data, axis=0), color=color, linewidth=2.0, label=label)
    ax.set_xlim(-0.3, 1.3)
    _format_stat_spines(ax, [0, 1], ybounds, yticks)
    ax.set_xticks([0, 1], labels=xtick_labels, fontsize=fontsize)


def _signed_participation_ratio(spec: np.ndarray) -> np.ndarray:
    """Signed participation ratio ``(sum lambda)^2 / sum(lambda^2)`` per mouse.

    ``spec`` is ``(mice, dims)``. "Signed" because negative spectrum entries are used as-is (no
    clipping); scale-invariant so normalization does not matter.
    """
    s1 = np.nansum(spec, axis=1)
    s2 = np.nansum(spec**2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s2 > 0, s1**2 / s2, np.nan)


def _paired_pvalue(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Two-sided paired-sample p-value of ``a`` vs ``b`` over the mice finite in both.

    ``method`` is ``"ttest"`` (:func:`scipy.stats.ttest_rel`) or ``"wilcoxon"``
    (:func:`scipy.stats.wilcoxon`, signed-rank). Returns NaN with fewer than two paired
    observations, and ``1.0`` when the two are numerically identical (no difference to test).
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return np.nan
    a, b = a[mask], b[mask]
    if np.allclose(a, b):
        return 1.0
    if method == "wilcoxon":
        return float(wilcoxon(a, b, alternative="two-sided").pvalue)
    return float(ttest_rel(a, b).pvalue)


def _significance_stars(p: float) -> str:
    """Tiered significance label: ``***`` p<0.001, ``**`` p<0.01, ``*`` p<0.05, else ``ns``."""
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _beeswarm_panel(
    ax,
    values_list,
    colors,
    labels,
    fontsize,
    beewidth: float = 0.2,
    each_alpha: float = 0.3,
    yscale: str = "linear",
    markersize: float = 3.0,
    mean_linewidth: float = 2.0,
) -> None:
    """Per-mouse beeswarm (points + bold mean line) at integer x-positions ``0, 1, ...``."""
    line_extent = np.array([-0.25, 0.25])
    for x, (vals, color) in enumerate(zip(values_list, colors)):
        vals = np.asarray(vals, dtype=float)
        offsets = np.zeros_like(vals)
        finite = np.isfinite(vals)
        if finite.any():
            offsets[finite] = beeswarm(vals[finite])
        ax.plot(x + beewidth * offsets, vals, color=color, linestyle="none", marker="o", markersize=markersize, alpha=each_alpha)
        ax.plot(x + line_extent, [np.nanmean(vals)] * 2, color=color, linewidth=mean_linewidth)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_yscale(yscale)
    ymin = 0 if yscale == "linear" else 1
    _ax_ylim = ax.get_ylim()
    ax.set_ylim(ymin, _ax_ylim[1])

    xticks = range(len(labels))
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=[0, max(xticks)],
        ybounds=[ymin, _ax_ylim[1]],
    )
    if len(labels) > 2:
        ax.set_xticks(xticks, labels=labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_xticks(xticks, labels=labels, fontsize=fontsize)


def _horizontal_beeswarm_panel(
    ax,
    values_list,
    colors,
    beewidth: float = 0.15,
    each_alpha: float = 0.6,
    markersize: float = 3.0,
    mean_linewidth: float = 2.0,
) -> None:
    """Draw several horizontal swarms around one shared, unlabeled y-position.

    The x-coordinate is the statistic itself. Swarm offsets are computed in log space because
    this helper is used below a log-rank spectrum axis. When groups contain the same number of
    mice, later groups reuse the PF offsets so corresponding CA1/PF points are y-aligned; otherwise
    the group gets its own centered offsets. Group identity is intentionally left to the spectrum
    legend in the shared panel above.
    """
    mean_extent = np.array([-0.22, 0.22])
    reference_offsets = None
    for vals, color in zip(values_list, colors):
        vals = np.asarray(vals, dtype=float)
        finite = np.isfinite(vals) & (vals > 0)
        offsets = np.zeros_like(vals)
        if reference_offsets is not None and len(reference_offsets) == len(vals):
            offsets = reference_offsets.copy()
        elif finite.any():
            offsets[finite] = beeswarm(np.log10(vals[finite]))
        if reference_offsets is None:
            reference_offsets = offsets.copy()
        ax.plot(
            vals,
            beewidth * offsets,
            color=color,
            linestyle="none",
            marker="o",
            markersize=markersize,
            alpha=each_alpha,
        )
        if finite.any():
            mean = np.nanmean(vals[finite])
            ax.plot([mean, mean], mean_extent, color=color, linewidth=mean_linewidth)

    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, right=False, labelleft=False)


# The legend widget machinery moved to ``legends.py`` when figure3's composite figure needed it too.
# Aliased rather than renamed at the call sites so this module's references stay as they were.
_LEGEND_LOCS = LEGEND_LOCS
_LEGEND_KNOBS = LEGEND_KNOBS
_add_legend_widgets = add_legend_widgets
_update_legend_widgets = update_legend_widgets
_apply_legend = apply_legend


class PlacefieldSpectraViewer(Viewer):
    """Interactive shared-variance spectrum + power-law exponent over aggregated results.

    Three panels: ax[0] the ``source_key`` example spectrum (one faint line per mouse, bold
    mouse-average, log-log); ax[1] a per-mouse beeswarm of the power-law exponent over ranks
    ``[start, end)`` for every curve option, grouped by method (log-log fit vs mean five-point
    derivative); ax[2] the per-rank five-point-derivative local-exponent curves for every option.
    The example spectrum is chosen by the ``source_key`` selection: StimSpaceSpectra keys
    (``ss_cv``, ``ss_direct``) are pulled from ``results`` and the ``reg_covariances_fixed`` key
    from ``results_cvpca``. The implementation knows which aggregator each key belongs to via
    :data:`SOURCE_OF_KEY`.

    Both aggregators may expose param axes with the same name (e.g. ``activity_parameters_name``).
    These share a single widget keyed by the raw axis name; at plot time only the params present in
    the active source's ``param_axes`` are forwarded to :meth:`ResultsAggregator.sel`. Tuple-valued
    axes (e.g. ``smooth_widths``) are auto-detected and encoded as string labels for the dropdown,
    then decoded back to tuples before selection.

    The lower y-limit is controlled in log10 units by a float slider (the applied floor is
    ``10 ** state["ylim_min"]``); the upper limit is autoscaled to the data.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        ylim_min: float = -5.5,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (5.0, 3.0),
    ):
        self.results = results
        self.results_cvpca = results_cvpca
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.fontsize = fontsize
        self.figsize = figsize

        available = list(_STIMSPACE_KEYS)
        if results_cvpca is not None:
            available += list(_CVPCA_KEYS)
        self.add_multiple_selection("source_key", options=available, value=["ss_cv"])

        # One widget per param-axis name, shared across sources. Options are the union of each
        # source's options for that axis (in first-seen order).
        merged_axes: dict[str, list] = {}
        for agg in self._agg.values():
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(opt for opt in options if opt not in existing)

        # Axes whose options are tuples (e.g. smooth_widths) can't be dropdown values directly, so
        # they are encoded as string labels; ``_tuple_labels[name]`` maps label -> original tuple.
        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for name, options in merged_axes.items():
            if any(isinstance(opt, tuple) for opt in options):
                label_map = {_tuple_label(opt): opt for opt in options}
                self._tuple_labels[name] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(name, options=widget_options)
            if name in _PREFERRED_DEFAULTS:
                default = self.encode_param(name, _PREFERRED_DEFAULTS[name])
                if default in widget_options:
                    self.update_selection(name, value=default)

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_boolean("normalize", value=True)
        # Rank window (0-based, [start, end)) the exponent is estimated over, for both methods.
        self.add_integer_range("fit_range", value=(10, 20), min=1, max=200)
        # Stencil half-width for the five-point-derivative local exponent.
        self.add_integer("deriv_width", value=1, min=1, max=10)
        # Log-space (geometric-mean) pre-smoothing of the spectrum before fitting.
        self.add_selection("smooth_kind", options=["none", "boxcar", "gaussian"], value="none")
        self.add_float("smooth_width", value=3.0, min=0.0, max=50.0, step=0.5)

    def encode_param(self, name: str, value):
        """Map a raw param value to its widget value (tuple -> string label; else unchanged)."""
        if name in self._tuple_labels and isinstance(value, tuple):
            return _tuple_label(value)
        return value

    def _sel_params(self, state: dict, source: str) -> dict:
        """Select the params relevant to this source, decoding tuple labels back to tuples."""
        agg = self._agg[source]
        params = {}
        for name in agg.param_axes:
            if name not in state:
                continue
            value = state[name]
            if name in self._tuple_labels:
                value = self._tuple_labels[name][value]
            params[name] = value
        return params

    def _available_keys(self) -> list[str]:
        """Curve options available to estimate exponents for (source_key options)."""
        keys = list(_STIMSPACE_KEYS)
        if self.results_cvpca is not None:
            keys += list(_CVPCA_KEYS)
        return keys

    def _spectrum(self, state: dict, key: str) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` spectrum for ``key``, normalized per ``state``."""
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        spec = agg.sel(keys=[key], avg_by_mouse=True, **self._sel_params(state, source))[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, state["smooth_kind"], state["smooth_width"])

    def plot(self, state: dict):
        selected_keys = list(state["source_key"])
        keys_all = self._available_keys()
        start, end = (int(v) for v in state["fit_range"])
        deriv_width = int(state["deriv_width"])
        each_alpha = 0.3
        ylim_min = state["ylim_min"]

        # Per-curve spectra, local-exponent curves, and both per-mouse exponent estimates.
        spectra = {k: self._spectrum(state, k) for k in keys_all}
        local_alpha = {k: _local_alpha_curve(spectra[k], deriv_width) for k in keys_all}
        decay_alpha = {k: _decay_alpha_per_mouse(spectra[k], start, end) for k in keys_all}
        deriv_alpha = {k: _deriv_alpha_per_mouse(local_alpha[k], start, end) for k in keys_all}

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(1, 3, figsize=self.figsize, layout="constrained", width_ratios=[1.0, 0.9, 0.9])

        # --- ax[0]: the selected example spectra (one faint line per mouse + bold average per key) ---
        for key in selected_keys:
            spec = spectra[key]
            spec_positive = np.where(spec > 0, spec, np.nan)
            ex_color = _KEY_COLORS.get(key, "blue")
            ax[0].plot(_xvals(spec), spec_positive.T, color=ex_color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(_xvals(spec), np.nanmean(spec_positive, axis=0), color=ex_color, label=key, linewidth=2.0)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(bottom=10**ylim_min)
        yticks = ax[0].get_yticks()
        ytick_power = [np.log10(yt) for yt in yticks]
        ax[0].set_yticks(yticks, labels=ytick_power)
        ax[0].set_ylim(bottom=10**ylim_min)
        ax[0].set_xlabel("Shared Dimension")
        ax[0].set_ylabel("Variance")
        ax[0].legend(loc="upper right", fontsize=self.fontsize, frameon=False)

        # --- ax[1]: per-mouse exponent, beeswarm, two method groups x each curve option ---
        methods = [("power-law fit", decay_alpha), ("5-pt deriv", deriv_alpha)]
        n = len(keys_all)
        beewidth = 0.2
        line_extent = np.array([-0.25, 0.25])
        np1 = np.array([1, 1])
        xticks = []
        all_vals = []
        for g, (_, alphas_by_key) in enumerate(methods):
            for i, k in enumerate(keys_all):
                x = g * (n + 1) + i
                xticks.append(x)
                vals = alphas_by_key[k]
                all_vals.append(vals)
                color = _KEY_COLORS.get(k, "gray")
                offsets = np.zeros_like(vals)
                finite = np.isfinite(vals)
                if finite.any():
                    offsets[finite] = beeswarm(vals[finite])
                ax[1].plot(x + beewidth * offsets, vals, color=color, linestyle="none", marker="o", markersize=3, alpha=each_alpha)
                ax[1].plot(x + line_extent, np1 * np.nanmean(vals), color=color, linewidth=2.0)

        flat = np.concatenate([v[np.isfinite(v)] for v in all_vals]) if all_vals else np.array([0.0, 1.0])
        if flat.size == 0:
            flat = np.array([0.0, 1.0])
        ylo, yhi = float(np.min(flat)), float(np.max(flat))
        pad = 0.1 * (yhi - ylo + 1e-9)
        yline = 0
        for g, (mname, _) in enumerate(methods):
            group_ticks = xticks[g * n : (g + 1) * n]
            ax[1].annotate(
                "",
                xy=(group_ticks[0], yline),
                xytext=(group_ticks[-1], yline),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0),
                annotation_clip=False,
            )
            ax[1].text(np.mean(group_ticks), yline - 0.3 * pad, mname, fontsize=self.fontsize, ha="center", va="bottom")
        ax[1].set_xlim(-0.5, max(xticks) + 0.5)
        ax[1].set_ylim(yline - pad, yhi + pad)
        ax[1].set_ylabel("Power-law exponent")
        ax[1].axhline(4.0, color="0.8", linestyle="--", linewidth=1.0)
        ax[1].set_xticks(xticks, labels=keys_all * len(methods), rotation=45, ha="right")

        # --- ax[2]: five-point-derivative local-exponent curves (per mouse + bold average) ---
        for k in keys_all:
            la = local_alpha[k]
            color = _KEY_COLORS.get(k, "gray")
            xv = np.arange(la.shape[1]) + 1
            ax[2].plot(xv, la.T, color=color, alpha=0.2, linewidth=0.8)
            ax[2].plot(xv, np.nanmean(la, axis=0), color=color, linewidth=2.0, label=k)
        ax[2].axvspan(start + 1, end, color="0.8", alpha=0.4)
        ax[2].set_xscale("log")
        ax[2].set_xlabel("Shared Dimension")
        ax[2].set_ylabel("Local exponent")
        ax[2].set_ylim(-1, 10)
        ax[2].legend(loc="upper left", fontsize=self.fontsize, frameon=False)
        return fig


class SessionSpectraViewer(Viewer):
    """Interactive per-session view of every shared-variance spectrum on one axis.

    The single-session analogue of :class:`PlacefieldSpectraViewer`. Instead of showing one
    ``source_key`` spectrum across all mice, ax[0] overlays every available curve option
    (``ss_cv``, ``ss_direct``, ..., ``reg_covariances_fixed``) for a single selected session,
    each colored per :data:`_KEY_COLORS`. ax[1] and ax[2] are unchanged in structure but now
    estimate the power-law exponent from that one session's spectra (one point per curve in the
    beeswarm, one local-exponent curve per option).

    A ``session`` selection widget replaces ``source_key``; its options are the union of the
    ``session_ids`` of every provided aggregator (in first-seen order). For each curve the session
    is resolved against its own source aggregator (via :data:`SOURCE_OF_KEY`); a session missing
    from a source yields an all-NaN spectrum for that curve (it simply does not draw).

    Param-axis widgets, tuple-label encoding, and the log10 y-floor behave exactly as in
    :class:`PlacefieldSpectraViewer`.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        ylim_min: float = -5.5,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (9.0, 3.0),
    ):
        self.results = results
        self.results_cvpca = results_cvpca
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.fontsize = fontsize
        self.figsize = figsize

        # Session options: union of every aggregator's session_ids, first-seen order.
        sessions: list[str] = []
        for agg in self._agg.values():
            if agg is None:
                continue
            sessions.extend(uid for uid in agg.session_ids if uid not in sessions)
        self.add_selection("session", options=sessions, value=sessions[0] if sessions else None)

        # One widget per param-axis name, shared across sources (same scheme as PlacefieldSpectraViewer).
        merged_axes: dict[str, list] = {}
        for agg in self._agg.values():
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(opt for opt in options if opt not in existing)

        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for name, options in merged_axes.items():
            if any(isinstance(opt, tuple) for opt in options):
                label_map = {_tuple_label(opt): opt for opt in options}
                self._tuple_labels[name] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(name, options=widget_options)
            if name in _PREFERRED_DEFAULTS:
                default = self.encode_param(name, _PREFERRED_DEFAULTS[name])
                if default in widget_options:
                    self.update_selection(name, value=default)

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_boolean("normalize", value=True)
        self.add_integer_range("fit_range", value=(10, 20), min=1, max=200)
        self.add_integer("deriv_width", value=1, min=1, max=10)
        self.add_selection("smooth_kind", options=["none", "boxcar", "gaussian"], value="none")
        self.add_float("smooth_width", value=3.0, min=0.0, max=50.0, step=0.5)

    encode_param = PlacefieldSpectraViewer.encode_param
    _sel_params = PlacefieldSpectraViewer._sel_params
    _available_keys = PlacefieldSpectraViewer._available_keys

    def _spectrum(self, state: dict, key: str) -> np.ndarray:
        """Single-session ``(1, dims)`` spectrum for ``key``, normalized per ``state``.

        Returns an all-NaN row if the selected session is absent from ``key``'s source aggregator.
        """
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        arr = agg.sel(keys=[key], squeeze_ones=False, **self._sel_params(state, source))[key]
        arr = np.asarray(arr, dtype=float)
        sess_idx = agg._session_index.get(state["session"])
        if sess_idx is None:
            return np.full((1, arr.shape[-1]), np.nan)
        spec = np.atleast_2d(arr[sess_idx])
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, state["smooth_kind"], state["smooth_width"])

    def plot(self, state: dict):
        keys_all = self._available_keys()
        start, end = (int(v) for v in state["fit_range"])
        deriv_width = int(state["deriv_width"])
        each_alpha = 0.3
        ylim_min = state["ylim_min"]

        # Per-curve single-session spectra, local-exponent curves, and both exponent estimates.
        spectra = {k: self._spectrum(state, k) for k in keys_all}
        local_alpha = {k: _local_alpha_curve(spectra[k], deriv_width) for k in keys_all}
        decay_alpha = {k: _decay_alpha_per_mouse(spectra[k], start, end) for k in keys_all}
        deriv_alpha = {k: _deriv_alpha_per_mouse(local_alpha[k], start, end) for k in keys_all}

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(1, 3, figsize=self.figsize, layout="constrained", width_ratios=[1.0, 0.9, 0.9])

        # --- ax[0]: every curve option for the selected session (one line each) ---
        for k in keys_all:
            spec = spectra[k]
            spec_positive = np.where(spec > 0, spec, np.nan)
            color = _KEY_COLORS.get(k, "gray")
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, label=k, linewidth=1.5)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(bottom=10**ylim_min)
        yticks = ax[0].get_yticks()
        ytick_power = [np.log10(yt) for yt in yticks]
        ax[0].set_yticks(yticks, labels=ytick_power)
        ax[0].set_ylim(bottom=10**ylim_min)
        ax[0].set_xlabel("Shared Dimension")
        ax[0].set_ylabel("Variance")
        ax[0].set_title(str(state["session"]), fontsize=self.fontsize)
        ax[0].legend(loc="upper right", fontsize=self.fontsize, frameon=False)

        # --- ax[1]: per-session exponent, two method groups x each curve option ---
        methods = [("power-law fit", decay_alpha), ("5-pt deriv", deriv_alpha)]
        n = len(keys_all)
        xticks = []
        all_vals = []
        for g, (_, alphas_by_key) in enumerate(methods):
            for i, k in enumerate(keys_all):
                x = g * (n + 1) + i
                xticks.append(x)
                vals = alphas_by_key[k]
                all_vals.append(vals)
                color = _KEY_COLORS.get(k, "gray")
                ax[1].plot(np.full_like(vals, x), vals, color=color, linestyle="none", marker="o", markersize=4)

        flat = np.concatenate([v[np.isfinite(v)] for v in all_vals]) if all_vals else np.array([0.0, 1.0])
        if flat.size == 0:
            flat = np.array([0.0, 1.0])
        ylo, yhi = float(np.min(flat)), float(np.max(flat))
        pad = 0.1 * (yhi - ylo + 1e-9)
        yline = ylo - 2 * pad
        for g, (mname, _) in enumerate(methods):
            group_ticks = xticks[g * n : (g + 1) * n]
            ax[1].annotate(
                "",
                xy=(group_ticks[0], yline),
                xytext=(group_ticks[-1], yline),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0),
                annotation_clip=False,
            )
            ax[1].text(np.mean(group_ticks), yline - 0.3 * pad, mname, fontsize=self.fontsize, ha="center", va="top")
        ax[1].set_xlim(-0.5, max(xticks) + 0.5)
        ax[1].set_ylim(yline - pad, yhi + pad)
        ax[1].set_ylabel("Power-law exponent")
        ax[1].axhline(4.0, color="0.8", linestyle="--", linewidth=1.0)
        ax[1].set_xticks(xticks, labels=keys_all * len(methods), rotation=45, ha="right")

        # --- ax[2]: five-point-derivative local-exponent curves (one per curve option) ---
        for k in keys_all:
            la = local_alpha[k]
            color = _KEY_COLORS.get(k, "gray")
            xv = np.arange(la.shape[1]) + 1
            ax[2].plot(xv, la.T, color=color, linewidth=1.5, label=k)
        ax[2].axvspan(start + 1, end, color="0.8", alpha=0.4)
        ax[2].set_xscale("log")
        ax[2].set_xlabel("Shared Dimension")
        ax[2].set_ylabel("Local exponent")
        ax[2].set_ylim(-1, 10)
        ax[2].legend(loc="upper left", fontsize=self.fontsize, frameon=False)
        return fig


_ASE_SVCA_KEY = "svca"


class AdaptiveSpectraEstimationViewer(Viewer):
    """Diagnostic figure for the ``alpha_method="adaptive"`` median-FPD fit, on one session's spectrum.

    The single-session, single-curve analogue of the ``"adaptive"`` branch of
    :class:`SpectrumFigureViewer` (see :func:`_second_derivative_window` /
    :func:`_median_fpd_alpha_session`), for inspecting the mechanics of that method directly:

    - ax[0]: the raw and smoothed spectrum on log-log axes, with vertical lines marking the smoothed
      spectrum's 2nd-derivative peak (max curvature) and the raw spectrum's first negative entry,
      plus the buffered window start/end actually used for the median (``adaptive_buffer`` dims
      inside each of those two landmarks).
    - ax[1]: the five-point-derivative local exponent curve restricted to that buffered window, with
      a horizontal line at its median -- the value :class:`SpectrumFigureViewer` would report (NaN,
      annotated as such, if fewer than :data:`_MEDIAN_FPD_MIN_VALUES` finite values fall inside it).
    - ax[2]: the two other (fixed-window) methods for comparison, over an independent ``fit_range``:
      the five-point-derivative local-exponent curve (window shaded) plus horizontal reference lines
      for the window power-law fit and the window-mean FPD exponent.
    - ax[3]: the first and second derivative of the smoothed spectrum.

    ``key`` options are the PF-like StimSpace/CVPCA curves plus the two "Reliable CA1" curves also
    offered by :class:`SpectrumFigureViewer`'s ``full_source_key``: ``"ff"`` (SVD, the StimSpaceSpectra
    ``ff`` key) and ``"svca"`` (the subspace ``variance_activity`` key, ``subspace_name='svca_subspace'``,
    requires ``results_subspace``). Session selection, param-axis widgets, tuple-label encoding and
    the log10 y-floor behave as in :class:`SessionSpectraViewer`.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        ylim_min: float = -5.5,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (9.0, 3.0),
    ):
        self.results = results
        self.results_cvpca = results_cvpca
        self.results_subspace = results_subspace
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.fontsize = fontsize
        self.figsize = figsize

        sessions: list[str] = []
        for agg in self._agg.values():
            if agg is None:
                continue
            sessions.extend(uid for uid in agg.session_ids if uid not in sessions)
        if results_subspace is not None:
            sessions.extend(uid for uid in results_subspace.session_ids if uid not in sessions)
        self.add_selection("session", options=sessions, value=sessions[0] if sessions else None)

        key_options = (
            list(_STIMSPACE_KEYS)
            + [_FF_KEY]
            + (list(_CVPCA_KEYS) if results_cvpca is not None else [])
            + ([_ASE_SVCA_KEY] if results_subspace is not None else [])
        )
        self.add_selection("key", options=key_options, value="ss_cv")

        # One widget per param-axis name, shared across sources (same scheme as SessionSpectraViewer).
        # ``results_subspace`` is deliberately excluded here (and from ``self._agg``): its own
        # ``smooth_width``/``subspace_name`` axes are fixed internally by ``_spectrum_smoothed`` for
        # the "svca" key, matching :meth:`SpectrumFigureViewer._ff_spectrum_sessions`.
        merged_axes: dict[str, list] = {}
        for agg in self._agg.values():
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(opt for opt in options if opt not in existing)

        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for name, options in merged_axes.items():
            if any(isinstance(opt, tuple) for opt in options):
                label_map = {_tuple_label(opt): opt for opt in options}
                self._tuple_labels[name] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(name, options=widget_options)
            if name in _PREFERRED_DEFAULTS:
                default = self.encode_param(name, _PREFERRED_DEFAULTS[name])
                if default in widget_options:
                    self.update_selection(name, value=default)

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_boolean("normalize", value=True)
        self.add_selection("smooth_kind", options=["none", "boxcar", "gaussian"], value="none")
        self.add_float("smooth_width", value=3.0, min=0.0, max=50.0, step=0.5)

        # Adaptive median-FPD fit (see _second_derivative_window / _median_fpd_alpha_session),
        # matching SpectrumFigureViewer.
        self.add_integer("adaptive_buffer", value=2, min=0, max=20)

        # Fixed-window comparison methods (ax[2]): independent of the adaptive window.
        self.add_integer_range("fit_range", value=(10, 20), min=1, max=500)
        self.add_integer("deriv_width", value=1, min=1, max=10)

    encode_param = PlacefieldSpectraViewer.encode_param
    _sel_params = PlacefieldSpectraViewer._sel_params

    def _spectrum_raw_and_smooth(self, state: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
        """Single-session raw (normalized only) and smoothed 1-D spectrum for ``key``.

        Mirrors :meth:`SessionSpectraViewer._spectrum` but returns the pre-smoothing row alongside
        the smoothed one: the window end is found on the raw values (see
        :func:`_second_derivative_window`), since smoothing maps every non-positive entry to NaN
        before exponentiating back (:func:`_smooth_spectrum`) and so never produces a negative
        output. The ``"svca"`` key instead mirrors :meth:`SpectrumFigureViewer._ff_spectrum_sessions`'s
        SVCA branch (subspace ``variance_activity``, ``subspace_name='svca_subspace'``,
        ``smooth_width=None`` fixed at the SubspaceConfig level -- distinct from this viewer's own
        ``smooth_width`` widget below, which is the log-space post-hoc smoothing applied here).
        """
        if key == _ASE_SVCA_KEY:
            params = {"subspace_name": "svca_subspace", "smooth_width": None}
            if "activity_parameters_name" in state:
                params["activity_parameters_name"] = state["activity_parameters_name"]
            agg = self.results_subspace
            arr = agg.sel(keys=["variance_activity"], squeeze_ones=False, **params)["variance_activity"]
        else:
            source = SOURCE_OF_KEY[key]
            agg = self._agg[source]
            arr = agg.sel(keys=[key], squeeze_ones=False, **self._sel_params(state, source))[key]
        arr = np.asarray(arr, dtype=float)
        sess_idx = agg._session_index.get(state["session"])
        if sess_idx is None:
            raw = np.full(arr.shape[-1], np.nan)
            return raw, raw.copy()
        raw = np.array(arr[sess_idx], dtype=float)
        if state["normalize"]:
            raw = raw / np.nansum(raw)
        smoothed = _smooth_spectrum(raw[None, :], state["smooth_kind"], state["smooth_width"])[0]
        return raw, smoothed

    def plot(self, state: dict):
        key = state["key"]
        ylim_min = state["ylim_min"]
        buffer = int(state["adaptive_buffer"])
        fit_start, fit_end = (int(v) for v in state["fit_range"])
        deriv_width = int(state["deriv_width"])

        raw, smoothed = self._spectrum_raw_and_smooth(state, key)
        first_derivative = np.diff(smoothed, n=1)
        second_derivative = np.diff(smoothed, n=2)
        start, end = _second_derivative_window(raw, smoothed, buffer)
        peak = start - 1 - buffer if start > 0 else -1  # the search-restricted peak _second_derivative_window used
        smoothed_2d = smoothed[None, :]
        local_alpha_2d = _local_alpha_curve(smoothed_2d, deriv_width)
        local_alpha = local_alpha_2d[0]
        median_alpha = _median_fpd_alpha_session(raw, smoothed, deriv_width, buffer)

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(1, 4, figsize=self.figsize, layout="constrained")

        # --- ax[0]: smoothed spectrum with the adaptive-window landmark vlines ---
        # ``peak``/``start``/``end``/``first_neg`` are 0-based array indices; the plotted axis is
        # 1-based dim numbers, so a +1 converts (matching the ``axvspan(start + 1, end)`` convention
        # used elsewhere in this file). ``first_neg`` (raw spectrum) is what ``end`` is buffered from.
        first_neg = _first_negative_index(raw) + 1
        xv_full = np.arange(smoothed.shape[0]) + 1
        raw_positive = np.where(raw > 0, raw, np.nan)
        smoothed_positive = np.where(smoothed > 0, smoothed, np.nan)
        ax[0].plot(xv_full, raw_positive, color="0.7", linewidth=1.0, label="raw")
        ax[0].plot(xv_full, smoothed_positive, color="black", linewidth=1.5, label="smoothed")
        vline_specs = [
            (peak + 1, "orange", "peak 2nd deriv"),
            (first_neg, "0.5", "first negative (raw)"),
            (start + 1, "blue", "window start"),
            (end, "red", "window end"),
        ]
        for xpos, color, vlabel in vline_specs:
            if 0 < xpos <= smoothed.shape[0]:
                ax[0].axvline(xpos, color=color, linestyle="--", linewidth=1.0, label=vlabel)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(bottom=10**ylim_min)
        yticks = ax[0].get_yticks()
        ax[0].set_yticks(yticks, labels=[np.log10(yt) for yt in yticks])
        ax[0].set_ylim(bottom=10**ylim_min)
        ax[0].set_xlabel("Shared Dimension")
        ax[0].set_ylabel("Variance")
        ax[0].set_title(f"{state['session']} : {key}", fontsize=self.fontsize)
        ax[0].legend(loc="upper right", fontsize=0.8 * self.fontsize, frameon=False)

        # --- ax[1]: FPD local exponent restricted to the adaptive window, median marked ---
        if end > start:
            window_mask = (xv_full >= start + 1) & (xv_full <= end)
            ax[1].plot(xv_full[window_mask], local_alpha[window_mask], color="purple", linewidth=1.5, label="FPD (window)")
        alpha_label = "median (NaN: too few values)" if np.isnan(median_alpha) else f"median ({median_alpha:.2f})"
        if not np.isnan(median_alpha):
            ax[1].axhline(median_alpha, color="red", linestyle="--", linewidth=1.0, label=alpha_label)
        else:
            ax[1].text(0.5, 0.5, alpha_label, transform=ax[1].transAxes, ha="center", va="center", color="red", fontsize=self.fontsize)
        ax[1].set_xlabel("Dim")
        ax[1].set_ylabel("Power-law exponent")
        ax[1].set_title("Adaptive window (median FPD)", fontsize=self.fontsize)
        ax[1].legend(loc="best", fontsize=0.8 * self.fontsize, frameon=False)

        # --- ax[2]: fixed-window comparison methods (window fit, window-avg FPD) ---
        window_alpha = _decay_alpha_per_mouse(smoothed_2d, fit_start, fit_end)[0]
        deriv_alpha = _deriv_alpha_per_mouse(local_alpha_2d, fit_start, fit_end)[0]
        ax[2].plot(xv_full, local_alpha, color="black", linewidth=1.5, label="5-pt deriv")
        ax[2].axvspan(fit_start + 1, fit_end, color="0.8", alpha=0.4)
        ax[2].axhline(window_alpha, color="blue", linestyle="--", linewidth=1.0, label=f"window fit ({window_alpha:.2f})")
        ax[2].axhline(deriv_alpha, color="orange", linestyle="--", linewidth=1.0, label=f"window-avg FPD ({deriv_alpha:.2f})")
        ax[2].set_xscale("log")
        ax[2].set_ylim(-1, 10)
        ax[2].set_xlabel("Shared Dimension")
        ax[2].set_ylabel("Local exponent")
        ax[2].set_title("Fixed-window methods", fontsize=self.fontsize)
        ax[2].legend(loc="upper left", fontsize=0.8 * self.fontsize, frameon=False)

        ax[3].plot(xv_full[1:], first_derivative, color="blue", linewidth=1.5, label="1st deriv")
        ax[3].plot(xv_full[2:], second_derivative, color="orange", linewidth=1.5, label="2nd deriv")
        ax[3].set_xscale("log")
        ax[3].set_xlabel("Shared Dimension")
        ax[3].set_ylabel("Derivative")
        ax[3].set_title("Derivatives of smoothed spectrum", fontsize=self.fontsize)
        ax[3].legend(loc="upper right", fontsize=0.8 * self.fontsize, frameon=False)
        return fig


@dataclass(frozen=False)
class AdaptiveAlphaConfig:
    """Fixed configuration for the adaptive median-FPD power-law exponent fit.

    See :func:`_second_derivative_window` / :func:`_median_fpd_alpha_session` for the estimation
    procedure this configures. ``frozen=False`` so instances can be edited in place (e.g. from a
    syd viewer) if that's wired up later; not frozen does not by itself add any such widgets.
    """

    smooth_method: str
    """Log-space (geometric-mean) pre-smoothing kind: ``"none"``, ``"boxcar"``, or ``"gaussian"``."""
    smooth_width: float
    """Boxcar full-width in rank units; the Gaussian uses ``sigma = smooth_width / 2``."""
    fpd_window_size: int
    """Five-point-derivative stencil half-width (``deriv_width`` in the estimation functions)."""
    adaptive_buffer: int
    """Dims of margin applied on both sides of the second-derivative window."""
    minimum_window_size: int
    """Minimum finite local-exponent count inside the window; below this the fit is NaN."""


@dataclass(frozen=True)
class SpectrumSmoothingConfig:
    """The smoothing settings needed by a spectrum-only panel."""

    smooth_method: str
    smooth_width: float


class SpectrumSmoothing(Protocol):
    """Structural type shared by smoothing-only and adaptive-alpha settings."""

    smooth_method: str
    smooth_width: float


# Fixed per-side adaptive-fit configs for spectrum_alpha_figure: "placefields" governs source_key
# (and the fit_key Tilbury overlays); "full" governs full_source_key.
ADAPTIVE_ALPHA_CONFIG_REGISTRY: dict[str, AdaptiveAlphaConfig] = {
    "placefields": AdaptiveAlphaConfig(
        smooth_method="gaussian",
        smooth_width=3.0,
        fpd_window_size=1,
        adaptive_buffer=2,
        minimum_window_size=10,
    ),
    "full": AdaptiveAlphaConfig(
        smooth_method="gaussian",
        smooth_width=20.0,
        fpd_window_size=20,
        adaptive_buffer=10,
        minimum_window_size=100,
    ),
}
ADAPTIVE_ALPHA_CONFIG_NAMES: tuple[str, ...] = tuple(ADAPTIVE_ALPHA_CONFIG_REGISTRY.keys())


def get_adaptive_alpha_config(name: str) -> AdaptiveAlphaConfig:
    if name not in ADAPTIVE_ALPHA_CONFIG_REGISTRY:
        raise ValueError(f"Unknown adaptive alpha config name {name!r}. Available: {list(ADAPTIVE_ALPHA_CONFIG_REGISTRY)}")
    return ADAPTIVE_ALPHA_CONFIG_REGISTRY[name]


class SpectrumFigureViewer(Viewer):
    """Placefield-vs-full spectrum figure: spectra and participation-ratio dimensionality.

    Two vertically stacked panels comparing one placefield (PF) spectrum against the
    full/functional (FF) spectrum:

    - ax[0]: the selected PF ``source_key`` spectrum (one of the four curve options) and the FF
      spectrum, both log-space pre-smoothed and drawn log-log (faint per-mouse lines + bold
      mouse-average). The FF curve source is set by ``full_source_key``: ``"SVD"`` is the ``ff`` key
      from the StimSpaceSpectra aggregator, ``"SVCA"`` is the subspace ``variance_activity`` key
      (``svca_subspace``, ``smooth_width=None``).
    - ax[1]: the signed participation ratio per mouse on the same log-scaled dimension axis as
      ax[0]. PF and FF points share one unlabeled categorical y-position; their colors are identified
      by ax[0]'s legend.

    Smoothing, param-axis widgets, tuple-label encoding, and the log10 y-floor behave as in
    :class:`PlacefieldSpectraViewer`.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        results_fit: ResultsAggregator | None = None,
        source_smooth_method: str = "gaussian",
        source_smooth_width: float = 3.0,
        full_smooth_method: str = "gaussian",
        full_smooth_width: float = 20.0,
        ylim_min: float = -5.5,
        ylim_max: float = 0.0,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (3.25, 3.25),
        height_ratios: tuple[float, float] = (1.0, 0.22),
        pf_color: str = "orange",
        ff_color: str = "black",
        pf_label: str = "Placefields",
        ff_label: str = "Full CA1",
    ):
        self.results = results
        self.results_cvpca = results_cvpca
        self.results_subspace = results_subspace
        self.results_fit = results_fit
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.fontsize = fontsize
        self.figsize = figsize
        self.height_ratios = height_ratios
        self.pf_color = pf_color
        self.ff_color = ff_color
        self.pf_label = pf_label
        self.ff_label = ff_label

        pf_options = list(_STIMSPACE_KEYS)
        if results_cvpca is not None:
            pf_options += list(_CVPCA_KEYS)
        self.add_selection("source_key", options=pf_options, value="ss_cv")

        # The "Reliable CA1 Spectrum" (FF) curve source: "SVD" is the StimSpaceSpectra ``ff`` key;
        # "SVCA" is the subspace ``variance_activity`` key (svca_subspace, smooth_width=None). SVCA
        # is only offered when a subspace aggregator is provided.
        full_options = ["SVD"] + (["SVCA"] if results_subspace is not None else [])
        self.add_selection("full_source_key", options=full_options, value="SVD")

        # Extra ax[0] overlays from the Tilbury-fit aggregator: generalized-Gaussian,
        # generalized-shrinkage and plain-Gaussian eigenvalue spectra, plus the per-neuron "better"
        # (lower validation-MSE) generalized/Gaussian composite (see :data:`_FIT_KEYS`). These are
        # always at the fit's fixed reliability/fraction-active threshold (:data:`_TILBURY_REL_FA`);
        # a mismatch with the shared ``reliability_fraction_active_thresholds`` selection is flagged
        # in the ax[0] title.
        if results_fit is not None:
            self.add_multiple_selection("fit_key", options=list(_FIT_KEYS), value=[])

        # One widget per param-axis name, shared across sources (same scheme as PlacefieldSpectraViewer).
        merged_axes: dict[str, list] = {}
        for agg in self._agg.values():
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(opt for opt in options if opt not in existing)

        # Param axes only the Tilbury-fit aggregator has get their own widget: a fit_key selection
        # must pin all of them, otherwise the sliced spectrum keeps an extra param dimension. Axes
        # shared with the spectra aggregators reuse the widget above. (With the current
        # TilburyFitConfig every axis is shared, so this branch is a no-op -- it stays general so an
        # added fit-only axis keeps working.)
        self._fit_axes: list[str] = list(results_fit.param_axes) if results_fit is not None else []
        if results_fit is not None:
            for name, options in results_fit.param_axes.items():
                if name not in merged_axes:
                    merged_axes[name] = list(options)

        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for name, options in merged_axes.items():
            if any(isinstance(opt, tuple) for opt in options):
                label_map = {_tuple_label(opt): opt for opt in options}
                self._tuple_labels[name] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(name, options=widget_options)
            if name in _PREFERRED_DEFAULTS:
                default = self.encode_param(name, _PREFERRED_DEFAULTS[name])
                if default in widget_options:
                    self.update_selection(name, value=default)

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_float("ylim_max", value=ylim_max, min=-8.0, max=12.0, step=0.1)
        self.add_float("beewidth", value=0.15, min=0.0, max=1.0, step=0.01)
        self.add_boolean("normalize", value=True)
        self.add_float("each_line_alpha", value=0.3, min=0.0, max=1.0, step=0.01)
        self.add_float("point_alpha", value=0.6, min=0.0, max=1.0, step=0.01)
        self.add_float("markersize", value=3.0, min=0.5, max=12.0, step=0.5)
        self.add_float("mean_linewidth", value=2.0, min=0.25, max=6.0, step=0.25)

        # Independent spectrum smoothing; there are deliberately no adaptive-alpha controls here.
        self.add_selection(
            "source_smooth_method",
            options=["none", "boxcar", "gaussian"],
            value=source_smooth_method,
        )
        self.add_float("source_smooth_width", value=source_smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_selection(
            "full_smooth_method",
            options=["none", "boxcar", "gaussian"],
            value=full_smooth_method,
        )
        self.add_float("full_smooth_width", value=full_smooth_width, min=0.0, max=50.0, step=0.5)

    encode_param = PlacefieldSpectraViewer.encode_param
    _sel_params = PlacefieldSpectraViewer._sel_params

    def _fit_sel_params(self, state: dict) -> dict:
        """Params pinning every Tilbury-fit param axis, decoding tuple labels back to tuples.

        The fit aggregator's own :meth:`_sel_params` analogue: it selects over
        ``results_fit.param_axes`` (``activity_parameters_name``, plus any fit-only axes) rather
        than over a spectra aggregator's axes.
        """
        params = {}
        for name in self._fit_axes:
            if name not in state:
                continue
            value = state[name]
            if name in self._tuple_labels:
                value = self._tuple_labels[name][value]
            params[name] = value
        return params

    def _smoothing_from_state(self, state: dict, prefix: str) -> SpectrumSmoothingConfig:
        """Return only the smoothing state needed by this viewer."""
        return SpectrumSmoothingConfig(
            smooth_method=state[f"{prefix}_smooth_method"],
            smooth_width=state[f"{prefix}_smooth_width"],
        )

    @staticmethod
    def _cfg_from_state(state: dict, prefix: str) -> AdaptiveAlphaConfig:
        """Build the shared adaptive-alpha configuration used by the alpha-based viewers."""
        return AdaptiveAlphaConfig(
            smooth_method=state[f"{prefix}_smooth_method"],
            smooth_width=state[f"{prefix}_smooth_width"],
            fpd_window_size=int(state[f"{prefix}_fpd_window_size"]),
            adaptive_buffer=int(state[f"{prefix}_adaptive_buffer"]),
            minimum_window_size=int(state[f"{prefix}_minimum_window_size"]),
        )

    def _spectrum(self, state: dict, key: str, cfg: SpectrumSmoothing) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` spectrum for ``key``, normalized per ``state``, smoothed per ``cfg``."""
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        spec = agg.sel(keys=[key], avg_by_mouse=True, **self._sel_params(state, source))[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)

    def _spectrum_sessions(
        self,
        state: dict,
        key: str,
        cfg: SpectrumSmoothing,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Per-session raw and smoothed ``(sessions, dims)`` spectrum for ``key``, with mouse/session ids.

        Normalize is applied per session (row) instead of after mouse-averaging, so the adaptive
        alpha fit can find each session's own first-negative crossover before any cross-session
        averaging blurs it. Both the raw (pre-smoothing) and smoothed spectrum are returned: smoothing
        maps every non-positive entry to NaN before exponentiating back (:func:`_smooth_spectrum`), so
        a smoothed row is never negative -- first-negative detection must use the raw one, while the
        exponent fit itself uses the smoothed one (matching every other alpha method).
        """
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        spec = agg.sel(keys=[key], avg_by_mouse=False, **self._sel_params(state, source))[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)
        return spec, smoothed, agg.mouse_names, agg.session_ids

    def _svca_spectrum_sessions(
        self,
        state: dict,
        cfg: SpectrumSmoothing,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list] | None:
        """Per-session raw+smoothed SVCA subspace ``variance_activity`` spectrum.

        This is both the ``full_source_key="SVCA"`` spectrum and the fixed FF-side window-fallback
        source (see :meth:`plot`), fetched independently of the current ``full_source_key`` selection.
        Returns None if ``results_subspace`` was not provided (fallback then simply isn't available).
        """
        if self.results_subspace is None:
            return None
        params = {"subspace_name": "svca_subspace", "smooth_width": None}
        if "activity_parameters_name" in state:
            params["activity_parameters_name"] = state["activity_parameters_name"]
        spec = self.results_subspace.sel(keys=["variance_activity"], avg_by_mouse=False, **params)["variance_activity"]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)
        return spec, smoothed, self.results_subspace.mouse_names, self.results_subspace.session_ids

    def _ff_spectrum_sessions(
        self,
        state: dict,
        cfg: SpectrumSmoothing,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Per-session raw+smoothed "Reliable CA1" spectrum, per ``state['full_source_key']`` (see :meth:`_ff_spectrum`)."""
        if state.get("full_source_key", "SVD") != "SVCA":
            return self._spectrum_sessions(state, _FF_KEY, cfg)
        return self._svca_spectrum_sessions(state, cfg)

    def _ff_spectrum(self, state: dict, cfg: SpectrumSmoothing) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` "Reliable CA1" spectrum, per ``state['full_source_key']``.

        ``"SVD"`` uses the StimSpaceSpectra ``ff`` key (via :meth:`_spectrum`). ``"SVCA"`` uses the
        subspace ``variance_activity`` key with ``subspace_name='svca_subspace'`` and
        ``smooth_width=None``; ``activity_parameters_name`` follows the shared widget. Both share the
        same normalize/log-space smoothing (per ``cfg``) as every other spectrum.
        """
        if state.get("full_source_key", "SVD") != "SVCA":
            return self._spectrum(state, _FF_KEY, cfg)
        params = {"subspace_name": "svca_subspace", "smooth_width": None}
        if "activity_parameters_name" in state:
            params["activity_parameters_name"] = state["activity_parameters_name"]
        spec = self.results_subspace.sel(keys=["variance_activity"], avg_by_mouse=True, **params)["variance_activity"]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)

    def _fit_spectrum_raw_sessions(self, state: dict, key: str) -> np.ndarray:
        """Raw (unnormalized, unsmoothed) per-session Tilbury-fit eigenvalue spectrum for ``key``.

        ``key`` is one of :data:`_FIT_KEYS` from the Tilbury-fit aggregator. These spectra vary in
        length across sessions but are stored as ``"pad"`` keys, so the aggregator NaN-pads them to a
        common length. Each session is converted from the PCA convention to the ``ss_cv`` covariance
        convention with ``P / (P - 1)``, gated by whichever ``params*`` arrays back ``key`` (see
        :data:`_FIT_KEY_PARAM_KEYS`). Every fit param axis (``activity_parameters_name``, ...)
        follows its syd widget; the reliability/fraction-active threshold is
        not a fit axis and remains fixed at :data:`_TILBURY_REL_FA`.
        """
        fit_params = self._fit_sel_params(state)
        param_keys = _FIT_KEY_PARAM_KEYS[key]
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

        Normalize/log-space smoothing (matching every other spectrum) are applied after averaging
        the raw per-session spectrum (:meth:`_fit_spectrum_raw_sessions`) by mouse.
        """
        session_spec = self._fit_spectrum_raw_sessions(state, key)
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
        if state["normalize"]:
            session_spec = session_spec / np.nansum(session_spec, axis=1)[:, None]
        smoothed = _smooth_spectrum(session_spec, cfg.smooth_method, cfg.smooth_width)
        return session_spec, smoothed, self.results_fit.mouse_names, self.results_fit.session_ids

    def _rel_fa_matches_fit(self, state: dict) -> bool:
        """Whether the shared reliability/fraction-active selection equals the Tilbury-fit threshold.

        The Tilbury eig spectra are fixed at :data:`_TILBURY_REL_FA`; if the shared
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
        return tuple(value) == _TILBURY_REL_FA

    def plot(self, state: dict):
        pf_key = state["source_key"]
        do_posthoc_scaling = pf_key == "ss_cvpca"
        source_smoothing = self._smoothing_from_state(state, "source")
        full_smoothing = self._smoothing_from_state(state, "full")
        each_alpha = state["each_line_alpha"]
        ylim_min = state["ylim_min"]
        ylim_max = state["ylim_max"]

        pf_spec = self._spectrum(state, pf_key, source_smoothing)
        ff_spec = self._ff_spectrum(state, full_smoothing)
        pf_color = self.pf_color
        ff_color = self.ff_color
        pf_label = self.pf_label
        ff_label = self.ff_label

        # Tilbury-fit overlays (PF-like spectra), placed between PF and CA1 in every panel.
        fit_keys = list(state.get("fit_key", []))
        fit_specs = {k: self._fit_spectrum(state, k, source_smoothing) for k in fit_keys}

        # ss_cvpca uses covariance across neurons (denominator N - 1), whereas ss_cv
        # uses covariance across positions (denominator P - 1). The required N/P counts
        # are not stored with the aggregated spectra, so align ss_cvpca post hoc using
        # the robust low-rank amplitude ratio. eig_* is already converted analytically
        # to the ss_cv convention in _eig_to_ss_scale and must not receive this factor.
        if do_posthoc_scaling:
            num_dim_for_scaling = 5
            pf_spec_reference = self._spectrum(state, "ss_cv", source_smoothing)
            ratio = pf_spec_reference[:, :num_dim_for_scaling] / pf_spec[:, :num_dim_for_scaling]
            scaling = np.nanmedian(ratio, axis=1)
            pf_spec *= scaling[:, np.newaxis]

        pf_pr = _signed_participation_ratio(pf_spec)
        ff_pr = _signed_participation_ratio(ff_spec)
        fit_pr = {k: _signed_participation_ratio(fit_specs[k]) for k in fit_keys}

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(
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
            color = _FIT_KEY_COLORS.get(key, "gray")
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(_xvals(spec), np.nanmean(spec_positive, axis=0), color=color, label=_FIT_KEY_LABELS.get(key, key), linewidth=2.0)
        if fit_keys and not self._rel_fa_matches_fit(state):
            ax[0].set_title("REL-FA Not MATCHED!", fontsize=self.fontsize, color="red")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(10**ylim_min, 10**ylim_max)
        ax[0].set_ylabel("Variance")
        ax[0].legend(loc="upper right", fontsize=self.fontsize, frameon=False, markerfirst=False, handlelength=1.0, handletextpad=0.25)

        # Participation-ratio groups share one categorical row and the spectrum's x-axis.
        beeswarm_colors = [pf_color] + [_FIT_KEY_COLORS.get(k, "gray") for k in fit_keys] + [ff_color]
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
        ax[1].set_xlabel("Shared Dimension")

        xlim = ax[0].get_xlim()
        ax[0].set_xlim(1, xlim[1])
        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[1, xlim[1]],
            ybounds=[10**ylim_min, 10**ylim_max],
        )
        format_spines(
            ax[1],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["bottom"],
            xbounds=[1, xlim[1]],
        )
        return fig


class SpectrumAlphaFigureViewer(SpectrumFigureViewer):
    """Spectrum comparison with the original adaptive decay-exponent panel.

    ax[0] retains the selectable PF and full-CA1 spectra from
    :class:`SpectrumFigureViewer`. ax[1] shows the per-mouse adaptive median-FPD decay exponent for
    those spectra (and any selected Tilbury-fit overlays). There is no participation-ratio panel.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        results_fit: ResultsAggregator | None = None,
        source_cfg: AdaptiveAlphaConfig | None = None,
        full_cfg: AdaptiveAlphaConfig | None = None,
        ylim_min: float = -5.5,
        ylim_max: float = 0.0,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (5.0, 3.0),
        width_ratios: tuple[float, float] = (1.0, 0.5),
        pf_color: str = "orange",
        ff_color: str = "black",
        pf_label: str = "Placefields",
        ff_label: str = "Full CA1",
    ):
        self.source_cfg = source_cfg if source_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]
        self.full_cfg = full_cfg if full_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["full"]
        super().__init__(
            results,
            results_cvpca=results_cvpca,
            results_subspace=results_subspace,
            results_fit=results_fit,
            source_smooth_method=self.source_cfg.smooth_method,
            source_smooth_width=self.source_cfg.smooth_width,
            full_smooth_method=self.full_cfg.smooth_method,
            full_smooth_width=self.full_cfg.smooth_width,
            ylim_min=ylim_min,
            ylim_max=ylim_max,
            fontsize=fontsize,
            figsize=figsize,
            pf_color=pf_color,
            ff_color=ff_color,
            pf_label=pf_label,
            ff_label=ff_label,
        )
        self.width_ratios = width_ratios

        # Restore every adaptive-fit control used by the former spectrum_figure alpha panel.
        for prefix, cfg in (("source", self.source_cfg), ("full", self.full_cfg)):
            self.add_integer(f"{prefix}_fpd_window_size", value=cfg.fpd_window_size, min=1, max=50)
            self.add_integer(f"{prefix}_adaptive_buffer", value=cfg.adaptive_buffer, min=0, max=50)
            self.add_integer(f"{prefix}_minimum_window_size", value=cfg.minimum_window_size, min=1, max=500)

    def plot(self, state: dict):
        pf_key = state["source_key"]
        source_cfg = self._cfg_from_state(state, "source")
        full_cfg = self._cfg_from_state(state, "full")
        each_alpha = state["each_line_alpha"]
        ylim_min = state["ylim_min"]
        ylim_max = state["ylim_max"]

        pf_spec = self._spectrum(state, pf_key, source_cfg)
        ff_spec = self._ff_spectrum(state, full_cfg)
        fit_keys = list(state.get("fit_key", []))
        fit_specs = {key: self._fit_spectrum(state, key, source_cfg) for key in fit_keys}

        # ss_cvpca uses covariance across neurons (denominator N - 1), whereas ss_cv uses
        # covariance across positions (denominator P - 1). Match the original post-hoc scaling.
        if pf_key == "ss_cvpca":
            num_dim_for_scaling = 5
            pf_reference = self._spectrum(state, "ss_cv", source_cfg)
            ratio = pf_reference[:, :num_dim_for_scaling] / pf_spec[:, :num_dim_for_scaling]
            pf_spec *= np.nanmedian(ratio, axis=1)[:, np.newaxis]

        # Estimate alpha per session using each session's own peak-curvature-to-noise-floor window,
        # then average the estimates by mouse. Non-cross-validated PF/fit spectra borrow window
        # boundaries from ss_cvpca; the FF side borrows them from SVCA when available.
        pf_raw, pf_smooth, pf_mouse_names, pf_session_ids = self._spectrum_sessions(state, pf_key, source_cfg)
        ff_raw, ff_smooth, ff_mouse_names, ff_session_ids = self._ff_spectrum_sessions(state, full_cfg)
        cvpca_raw, cvpca_smooth, _, cvpca_session_ids = self._spectrum_sessions(state, "ss_cvpca", source_cfg)
        svca = self._svca_spectrum_sessions(state, full_cfg)

        pf_fb_raw = _align_rows_to_sessions(pf_session_ids, cvpca_session_ids, cvpca_raw)
        pf_fb_smooth = _align_rows_to_sessions(pf_session_ids, cvpca_session_ids, cvpca_smooth)
        ff_fb_raw, ff_fb_smooth = None, None
        if svca is not None:
            svca_raw, svca_smooth, _, svca_session_ids = svca
            ff_fb_raw = _align_rows_to_sessions(ff_session_ids, svca_session_ids, svca_raw)
            ff_fb_smooth = _align_rows_to_sessions(ff_session_ids, svca_session_ids, svca_smooth)

        pf_alpha = average_by_mouse(
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
        ff_alpha = average_by_mouse(
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
            fit_fb_raw = _align_rows_to_sessions(fit_session_ids, cvpca_session_ids, cvpca_raw)
            fit_fb_smooth = _align_rows_to_sessions(fit_session_ids, cvpca_session_ids, cvpca_smooth)
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

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(1, 2, figsize=self.figsize, layout="constrained", width_ratios=self.width_ratios)

        for spec, label, color in (
            (pf_spec, self.pf_label, self.pf_color),
            (ff_spec, self.ff_label, self.ff_color),
        ):
            spec_positive = np.where(spec > 0, spec, np.nan)
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(_xvals(spec), np.nanmean(spec_positive, axis=0), color=color, label=label, linewidth=2.0)

        for key in fit_keys:
            spec = fit_specs[key]
            spec_positive = np.where(spec > 0, spec, np.nan)
            color = _FIT_KEY_COLORS.get(key, "gray")
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(
                _xvals(spec),
                np.nanmean(spec_positive, axis=0),
                color=color,
                label=_FIT_KEY_LABELS.get(key, key),
                linewidth=2.0,
            )
        if fit_keys and not self._rel_fa_matches_fit(state):
            ax[0].set_title("REL-FA Not MATCHED!", fontsize=self.fontsize, color="red")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(10**ylim_min, 10**ylim_max)
        ax[0].set_xlabel("Shared Dimension")
        ax[0].set_ylabel("Variance")
        ax[0].legend(
            loc="upper right",
            fontsize=self.fontsize,
            frameon=False,
            markerfirst=False,
            handlelength=1.0,
            handletextpad=0.25,
        )
        xlim = ax[0].get_xlim()
        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[1, xlim[1]],
            ybounds=[10**ylim_min, 10**ylim_max],
        )

        colors = [self.pf_color] + [_FIT_KEY_COLORS.get(key, "gray") for key in fit_keys] + [self.ff_color]
        labels = ["PF"] + [_FIT_KEY_LABELS.get(key, key) for key in fit_keys] + ["CA1"]
        alpha_values = [pf_alpha] + [fit_alpha[key] for key in fit_keys] + [ff_alpha]
        _beeswarm_panel(
            ax[1],
            alpha_values,
            colors,
            labels,
            self.fontsize,
            beewidth=state["beewidth"],
            each_alpha=state["point_alpha"],
            markersize=state["markersize"],
            mean_linewidth=state["mean_linewidth"],
        )
        ax[1].set_ylabel("Decay exponent")
        return fig


def placefield_spectra(
    results: ResultsAggregator,
    results_cvpca: ResultsAggregator | None = None,
    source_key: str | list[str] = "ss_cv",
    ylim_min: float = -5.5,
    normalize: bool = True,
    fit_range: tuple[int, int] = (10, 20),
    deriv_width: int = 1,
    smooth_kind: str = "none",
    smooth_width: float = 3.0,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (9.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Shared-variance spectrum figure with power-law exponent estimates.

    Three panels: ax[0] the ``source_key`` example spectrum on log-log axes (one faint line per
    mouse, bold mouse-average); ax[1] a per-mouse beeswarm of the power-law exponent estimated
    over ranks ``[start, end)`` for every curve option, grouped by method (log-log ``power-law
    fit`` vs the mean five-point-derivative ``5-pt deriv``); ax[2] the full per-rank
    five-point-derivative local-exponent curves for every curve option (with the ``[start, end)``
    window shaded). Which aggregator each key comes from is resolved via :data:`SOURCE_OF_KEY`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results, source of the ``reg_covariances_fixed`` key. Required if
        ``source_key`` names a CVPCA key; if None only StimSpace keys are available.
    source_key : str or list of str
        Which spectrum/spectra to overlay in ax[0]. A single key or list drawn from
        ``ss_cv``/``ss_direct`` (from ``results``) and ``reg_covariances_fixed`` (from
        ``results_cvpca``); see :data:`SOURCE_OF_KEY`. Each is colored per :data:`_KEY_COLORS`. The
        exponent panels always cover all available curve options regardless of this choice.
    ylim_min : float
        Lower y-limit of the spectrum panel in log10 units; the applied floor is ``10 ** ylim_min``.
        The upper limit is autoscaled to the data.
    normalize : bool
        If True, normalize the spectrum by the sum of the spectrum.
    fit_range : tuple[int, int]
        0-based ``[start, end)`` rank window the exponent is estimated over (both methods).
    deriv_width : int
        Stencil half-width for the five-point-derivative local exponent.
    smooth_kind : {"none", "boxcar", "gaussian"}
        Log-space (geometric-mean) pre-smoothing applied to each spectrum before both exponent
        fits. ``"none"`` disables smoothing.
    smooth_width : float
        Boxcar full-width in rank units; the Gaussian uses ``sigma = smooth_width / 2``.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here via
        ``save_figure``.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections, keyed by raw ``param_axes`` name (e.g.
        ``activity_parameters_name``, ``include_iti``, ``smooth_widths``). Each key must be a
        ``param_axes`` name of at least one provided aggregator; it applies to whichever source(s)
        have that axis. Tuple-valued axes (e.g. ``smooth_widths=(5.0, None)``) are passed as native
        tuples; they are encoded to the widget's string labels internally.

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldSpectraViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    source_keys = [source_key] if isinstance(source_key, str) else list(source_key)
    for sk in source_keys:
        if sk not in SOURCE_OF_KEY:
            raise ValueError(f"Unknown source_key {sk!r}. Options: {list(SOURCE_OF_KEY)}")
        if SOURCE_OF_KEY[sk] == "cvpca" and results_cvpca is None:
            raise ValueError(f"source_key {sk!r} is a CVPCA key but results_cvpca was not provided.")

    viewer = PlacefieldSpectraViewer(results, results_cvpca=results_cvpca, ylim_min=ylim_min, fontsize=fontsize, figsize=figsize)
    viewer.update_multiple_selection("source_key", value=source_keys)

    valid_selections = set()
    for agg in viewer._agg.values():
        if agg is None:
            continue
        valid_selections.update(agg.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))

    viewer.update_float("ylim_min", value=ylim_min)
    viewer.update_boolean("normalize", value=normalize)
    viewer.update_integer_range("fit_range", value=tuple(fit_range))
    viewer.update_integer("deriv_width", value=deriv_width)
    viewer.update_selection("smooth_kind", value=smooth_kind)
    viewer.update_float("smooth_width", value=smooth_width)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


def session_spectra(
    results: ResultsAggregator,
    results_cvpca: ResultsAggregator | None = None,
    session: str | None = None,
    ylim_min: float = -5.5,
    normalize: bool = True,
    fit_range: tuple[int, int] = (10, 20),
    deriv_width: int = 1,
    smooth_kind: str = "none",
    smooth_width: float = 3.0,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (9.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Per-session view of every shared-variance spectrum with power-law exponent estimates.

    The single-session analogue of :func:`placefield_spectra`. Three panels: ax[0] overlays every
    curve option (``ss_cv``, ``ss_direct``, ..., ``reg_covariances_fixed``) for one session on
    log-log axes, colored per :data:`_KEY_COLORS`; ax[1] a beeswarm of the power-law exponent for
    each curve (one point per curve, both estimation methods) over ranks ``[start, end)``; ax[2] the
    per-rank five-point-derivative local-exponent curve for each option (with the window shaded).
    Which aggregator each key comes from is resolved via :data:`SOURCE_OF_KEY`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results, source of the ``reg_covariances_fixed`` key. If None only
        StimSpace keys are drawn.
    session : str or None
        session_uid to show. Must be a session of at least one provided aggregator. If None, the
        first session (union of the aggregators' ``session_ids``) is used.
    ylim_min : float
        Lower y-limit of the spectrum panel in log10 units; the applied floor is ``10 ** ylim_min``.
    normalize : bool
        If True, normalize each spectrum by its sum.
    fit_range : tuple[int, int]
        0-based ``[start, end)`` rank window the exponent is estimated over (both methods).
    deriv_width : int
        Stencil half-width for the five-point-derivative local exponent.
    smooth_kind : {"none", "boxcar", "gaussian"}
        Log-space (geometric-mean) pre-smoothing applied to each spectrum before both exponent
        fits. ``"none"`` disables smoothing.
    smooth_width : float
        Boxcar full-width in rank units; the Gaussian uses ``sigma = smooth_width / 2``.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections, keyed by raw ``param_axes`` name. See
        :func:`placefield_spectra`.

    Returns
    -------
    matplotlib.figure.Figure or SessionSpectraViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = SessionSpectraViewer(results, results_cvpca=results_cvpca, ylim_min=ylim_min, fontsize=fontsize, figsize=figsize)
    if session is not None:
        viewer.update_selection("session", value=session)

    valid_selections = set()
    for agg in viewer._agg.values():
        if agg is None:
            continue
        valid_selections.update(agg.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))

    viewer.update_float("ylim_min", value=ylim_min)
    viewer.update_boolean("normalize", value=normalize)
    viewer.update_integer_range("fit_range", value=tuple(fit_range))
    viewer.update_integer("deriv_width", value=deriv_width)
    viewer.update_selection("smooth_kind", value=smooth_kind)
    viewer.update_float("smooth_width", value=smooth_width)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


def adaptive_spectra_estimation(
    results: ResultsAggregator,
    results_cvpca: ResultsAggregator | None = None,
    results_subspace: ResultsAggregator | None = None,
    session: str | None = None,
    key: str = "ss_cv",
    ylim_min: float = -5.5,
    normalize: bool = True,
    smooth_kind: str = "none",
    smooth_width: float = 3.0,
    adaptive_buffer: int = 2,
    fit_range: tuple[int, int] = (10, 20),
    deriv_width: int = 1,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (9.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Diagnostic figure for the ``alpha_method="adaptive"`` median-FPD fit, on one session's spectrum.

    Four panels (see :class:`AdaptiveSpectraEstimationViewer`): ax[0] the selected spectrum
    (smoothed per ``smooth_kind``/``smooth_width``) with vlines for the smoothed spectrum's 2nd-
    derivative peak and the raw spectrum's first negative entry, and the buffered window start/end
    actually used for the median; ax[1]
    the five-point-derivative local exponent restricted to that window, with its median marked (NaN,
    annotated, if too few finite values fall inside it); ax[2] the two other (fixed-window) methods
    over an independent ``fit_range`` for comparison: the five-point-derivative local-exponent curve
    plus horizontal reference lines for the window power-law fit and the window-mean FPD exponent;
    ax[3] the first and second derivative of the smoothed spectrum.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*``/``sf_*``/``ff`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results, source of the ``reg_covariances_fixed`` key. If None only
        StimSpace keys (including ``ff``) are selectable.
    results_subspace : ResultsAggregator or None
        Aggregated SubspaceConfig results, source of the ``"svca"`` key (subspace
        ``variance_activity``, ``subspace_name='svca_subspace'``). Required for that option; if None
        it is not selectable. Matches :class:`SpectrumFigureViewer`'s ``full_source_key="SVCA"``.
    session : str or None
        session_uid to show. Must be a session of at least one provided aggregator. If None, the
        first session (union of the aggregators' ``session_ids``) is used.
    key : str
        Which spectrum to inspect: one of ``ss_cv``/``ss_direct``/``ss_cvpca``/``sf_cv``/``sf_direct``/
        ``ff`` (from ``results``), ``reg_covariances_fixed`` (from ``results_cvpca``), or ``"svca"``
        (from ``results_subspace``). ``"ff"`` and ``"svca"`` are the two "Reliable CA1" curves also
        offered by :class:`SpectrumFigureViewer`'s ``full_source_key`` (``"SVD"``/``"SVCA"``).
    ylim_min : float
        Lower y-limit of the spectrum panel (ax[0]) in log10 units; the applied floor is
        ``10 ** ylim_min``.
    normalize : bool
        If True, normalize the spectrum by its sum before smoothing/fitting.
    smooth_kind : {"none", "boxcar", "gaussian"}
        Log-space (geometric-mean) pre-smoothing applied before every fit in this figure.
        ``"none"`` disables smoothing.
    smooth_width : float
        Boxcar full-width in rank units; the Gaussian uses ``sigma = smooth_width / 2``.
    adaptive_buffer : int
        Dims of margin applied on both sides of the second-derivative window: it starts this many
        dims after the second derivative's peak and ends this many dims before its first negative
        crossing. Matches :func:`spectrum_figure`'s ``adaptive_buffer``.
    fit_range : tuple[int, int]
        0-based ``[start, end)`` rank window for the two fixed-window comparison methods in ax[2]
        (independent of the adaptive window).
    deriv_width : int
        Stencil half-width for the five-point-derivative local exponent (ax[1]/ax[2]).
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections, keyed by raw ``param_axes`` name. See
        :func:`placefield_spectra`.

    Returns
    -------
    matplotlib.figure.Figure or AdaptiveSpectraEstimationViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    key_options = (
        list(_STIMSPACE_KEYS)
        + [_FF_KEY]
        + (list(_CVPCA_KEYS) if results_cvpca is not None else [])
        + ([_ASE_SVCA_KEY] if results_subspace is not None else [])
    )
    if key not in key_options:
        raise ValueError(f"Unknown key {key!r}. Options: {key_options}")

    viewer = AdaptiveSpectraEstimationViewer(
        results,
        results_cvpca=results_cvpca,
        results_subspace=results_subspace,
        ylim_min=ylim_min,
        fontsize=fontsize,
        figsize=figsize,
    )
    if session is not None:
        viewer.update_selection("session", value=session)
    viewer.update_selection("key", value=key)

    valid_selections = set()
    for agg in viewer._agg.values():
        if agg is None:
            continue
        valid_selections.update(agg.param_axes)
    for sel_key, value in selections.items():
        if sel_key not in valid_selections:
            raise ValueError(f"Unknown selection {sel_key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(sel_key, value=viewer.encode_param(sel_key, value))

    viewer.update_float("ylim_min", value=ylim_min)
    viewer.update_boolean("normalize", value=normalize)
    viewer.update_selection("smooth_kind", value=smooth_kind)
    viewer.update_float("smooth_width", value=smooth_width)
    viewer.update_integer("adaptive_buffer", value=adaptive_buffer)
    viewer.update_integer_range("fit_range", value=tuple(fit_range))
    viewer.update_integer("deriv_width", value=deriv_width)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


def spectrum_figure(
    results: ResultsAggregator,
    results_cvpca: ResultsAggregator | None = None,
    results_subspace: ResultsAggregator | None = None,
    results_fit: ResultsAggregator | None = None,
    source_key: str = "ss_cv",
    full_source_key: str = "SVD",
    fit_key: str | list[str] = (),
    ylim_min: float = -5.5,
    ylim_max: float = 0.0,
    beewidth: float = 0.15,
    normalize: bool = True,
    source_smooth_method: str = "gaussian",
    source_smooth_width: float = 3.0,
    full_smooth_method: str = "gaussian",
    full_smooth_width: float = 20.0,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (3.25, 3.25),
    height_ratios: tuple[float, float] = (1.0, 0.22),
    each_line_alpha: float = 0.3,
    point_alpha: float = 0.6,
    markersize: float = 3.0,
    mean_linewidth: float = 2.0,
    pf_color: str = "orange",
    ff_color: str = "black",
    pf_label: str = "Placefields",
    ff_label: str = "Full CA1",
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Placefield-vs-full spectrum figure: spectra and participation-ratio dimensionality.

    Two rows share a log-scaled dimension axis. ax[0] shows the selected PF ``source_key`` spectrum
    and FF spectrum (faint per-mouse + bold mouse-average, both log-space pre-smoothed). ax[1]
    shows their per-mouse signed participation ratios as horizontal swarms centered on the same
    unlabeled y-position; colors are identified by ax[0]'s legend.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*`` and ``ff`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results, source of the ``reg_covariances_fixed`` key. Required if
        ``source_key`` is that CVPCA key; if None only StimSpace PF keys are selectable.
    results_subspace : ResultsAggregator or None
        Aggregated SubspaceConfig results, source of the ``variance_activity`` key used when
        ``full_source_key="SVCA"``. Required for that option; if None only ``"SVD"`` is selectable.
    results_fit : ResultsAggregator or None
        Aggregated TilburyFitConfig results, source of the :data:`_FIT_KEYS` overlays selected by
        ``fit_key``. Required for those overlays; if None ``fit_key`` must be empty.
    source_key : str
        Which PF spectrum to show in ax[0]. One of ``ss_cv``/``ss_direct``/``ss_cvpca`` (from
        ``results``) or ``reg_covariances_fixed`` (from ``results_cvpca``).
    full_source_key : {"SVD", "SVCA"}
        Source of the FF ("Reliable CA1 Spectrum") curve. ``"SVD"`` uses the StimSpaceSpectra ``ff``
        key. ``"SVCA"`` uses the subspace ``variance_activity`` key with
        ``subspace_name='svca_subspace'`` and ``smooth_width=None`` (``activity_parameters_name``
        follows the shared selection), and requires ``results_subspace``.
    fit_key : str or list of str
        Extra ax[0] overlays from the Tilbury-fit aggregator: any of ``eig_tilbury`` (blue, the
        unregularized generalized Gaussian), ``eig_control`` (green, the plain-Gaussian control),
        ``eig_shrinkage`` (purple, the generalized fit with the Gaussian-centered shrinkage penalty
        at its per-neuron validation-selected lambdas), or ``eig_better`` (red, the per-neuron
        generalized/Gaussian composite) -- see :data:`_FIT_KEYS`/:data:`_FIT_KEY_COLORS`. The fit
        aggregator's own param axes (``activity_parameters_name``) are set
        through ``**selections`` like any other axis. These are always at the fit's fixed
        reliability/fraction-active threshold ``(0.3, 0.1)``; if the shared
        ``reliability_fraction_active_thresholds`` selection differs, ax[0] is titled
        ``"REL-FA Not MATCHED!"``. Requires ``results_fit``.
    ylim_min : float
        Lower y-limit of the spectrum panel in log10 units; the applied floor is ``10 ** ylim_min``.
    ylim_max : float
        Upper y-limit of the spectrum panel in log10 units; the applied ceiling is ``10 ** ylim_max``.
    beewidth : float
        Vertical spread of the participation-ratio swarms around ax[1]'s shared y-position.
    normalize : bool
        If True, normalize each spectrum by its sum (does not affect the participation ratio).
    source_smooth_method, full_smooth_method : {"none", "boxcar", "gaussian"}
        Independent log-space smoothing methods for the PF and full-CA1 spectra.
    source_smooth_width, full_smooth_width : float
        Corresponding smoothing widths in rank units.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    height_ratios : tuple[float, float]
        Relative heights of the spectrum and participation-ratio rows.
    each_line_alpha : float
        Alpha of individual-mouse spectrum lines in ax[0].
    point_alpha : float
        Alpha of individual-mouse participation-ratio points in ax[1].
    markersize : float
        Participation-ratio marker size in points.
    mean_linewidth : float
        Width of the vertical mean markers in ax[1].
    pf_color, ff_color : str
        Colors shared by each spectrum and its participation-ratio points.
    pf_label, ff_label : str
        Legend labels for the PF and FF spectra.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections, keyed by raw ``param_axes`` name. See
        :func:`placefield_spectra`.

    Returns
    -------
    matplotlib.figure.Figure or SpectrumFigureViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    pf_options = list(_STIMSPACE_KEYS) + (list(_CVPCA_KEYS) if results_cvpca is not None else [])
    if source_key not in pf_options:
        raise ValueError(f"Unknown PF source_key {source_key!r}. Options: {pf_options}")
    full_options = ["SVD"] + (["SVCA"] if results_subspace is not None else [])
    if full_source_key not in full_options:
        raise ValueError(f"Unknown full_source_key {full_source_key!r}. Options: {full_options}")
    fit_keys = [fit_key] if isinstance(fit_key, str) else list(fit_key)
    if fit_keys and results_fit is None:
        raise ValueError("fit_key requires results_fit to be provided.")
    for fk in fit_keys:
        if fk not in _FIT_KEYS:
            raise ValueError(f"Unknown fit_key {fk!r}. Options: {_FIT_KEYS}")

    viewer = SpectrumFigureViewer(
        results,
        results_cvpca=results_cvpca,
        results_subspace=results_subspace,
        results_fit=results_fit,
        source_smooth_method=source_smooth_method,
        source_smooth_width=source_smooth_width,
        full_smooth_method=full_smooth_method,
        full_smooth_width=full_smooth_width,
        ylim_min=ylim_min,
        ylim_max=ylim_max,
        fontsize=fontsize,
        figsize=figsize,
        height_ratios=height_ratios,
        pf_color=pf_color,
        ff_color=ff_color,
        pf_label=pf_label,
        ff_label=ff_label,
    )
    viewer.update_selection("source_key", value=source_key)
    viewer.update_selection("full_source_key", value=full_source_key)
    if results_fit is not None:
        viewer.update_multiple_selection("fit_key", value=fit_keys)

    valid_selections = set()
    for agg in viewer._agg.values():
        if agg is None:
            continue
        valid_selections.update(agg.param_axes)
    if results_fit is not None:
        valid_selections.update(results_fit.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))

    viewer.update_float("ylim_min", value=ylim_min)
    viewer.update_float("ylim_max", value=ylim_max)
    viewer.update_boolean("normalize", value=normalize)
    viewer.update_float("beewidth", value=beewidth)
    viewer.update_float("each_line_alpha", value=each_line_alpha)
    viewer.update_float("point_alpha", value=point_alpha)
    viewer.update_float("markersize", value=markersize)
    viewer.update_float("mean_linewidth", value=mean_linewidth)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


def spectrum_alpha_figure(
    results: ResultsAggregator,
    results_cvpca: ResultsAggregator | None = None,
    results_subspace: ResultsAggregator | None = None,
    results_fit: ResultsAggregator | None = None,
    source_key: str = "ss_cv",
    full_source_key: str = "SVD",
    fit_key: str | list[str] = (),
    ylim_min: float = -5.5,
    ylim_max: float = 0.0,
    beewidth: float = 0.15,
    normalize: bool = True,
    source_cfg: AdaptiveAlphaConfig | None = None,
    full_cfg: AdaptiveAlphaConfig | None = None,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (5.0, 3.0),
    width_ratios: tuple[float, float] = (1.0, 0.5),
    each_line_alpha: float = 0.3,
    point_alpha: float = 0.3,
    markersize: float = 3.0,
    mean_linewidth: float = 2.0,
    pf_color: str = "orange",
    ff_color: str = "black",
    pf_label: str = "Placefields",
    ff_label: str = "Full CA1",
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """Plot PF/full-CA1 spectra and their adaptive power-law decay exponents.

    This restores the spectra and adaptive-alpha portions of the former
    :func:`spectrum_figure` layout as a two-column figure. ax[0] contains the selectable spectra;
    ax[1] contains per-mouse adaptive median-FPD decay-exponent swarms. There is no participation-
    ratio panel.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*`` and ``ff`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results. Required for ``source_key="reg_covariances_fixed"``.
    results_subspace : ResultsAggregator or None
        Aggregated SubspaceConfig results. Required for ``full_source_key="SVCA"`` and otherwise
        used, when available, as the full-CA1 adaptive-window fallback.
    results_fit : ResultsAggregator or None
        Aggregated TilburyFitConfig results, required when ``fit_key`` is non-empty.
    source_key : str
        PF spectrum shown in ax[0] and summarized in ax[1].
    full_source_key : {"SVD", "SVCA"}
        Full-CA1 spectrum source.
    fit_key : str or list of str
        Optional Tilbury-fit spectrum overlays, also summarized in ax[1].
    ylim_min, ylim_max : float
        Spectrum y-limits in log10 units.
    beewidth : float
        Horizontal spread of each decay-exponent swarm around its categorical x-position.
    normalize : bool
        Whether to normalize each spectrum by its sum.
    source_cfg, full_cfg : AdaptiveAlphaConfig or None
        Independent smoothing and adaptive median-FPD settings for the PF and full-CA1 sides.
    fontsize : float
        Base font size.
    figsize : tuple[float, float]
        Figure size in inches.
    width_ratios : tuple[float, float]
        Relative widths of the spectrum and decay-exponent panels.
    each_line_alpha : float
        Alpha of individual-mouse spectrum lines.
    point_alpha : float
        Alpha of individual-mouse decay-exponent points.
    markersize : float
        Decay-exponent marker size in points.
    mean_linewidth : float
        Width of the mean markers in the decay-exponent panel.
    pf_color, ff_color : str
        Colors shared by each spectrum and its decay-exponent points.
    pf_label, ff_label : str
        Legend labels for the PF and full-CA1 spectra.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the configured :class:`SpectrumAlphaFigureViewer`.
    **selections
        Parameter-axis selection overrides, keyed by raw ``param_axes`` name.

    Returns
    -------
    matplotlib.figure.Figure or SpectrumAlphaFigureViewer
        The rendered figure, or its configured Syd viewer.
    """
    pf_options = list(_STIMSPACE_KEYS) + (list(_CVPCA_KEYS) if results_cvpca is not None else [])
    if source_key not in pf_options:
        raise ValueError(f"Unknown PF source_key {source_key!r}. Options: {pf_options}")
    full_options = ["SVD"] + (["SVCA"] if results_subspace is not None else [])
    if full_source_key not in full_options:
        raise ValueError(f"Unknown full_source_key {full_source_key!r}. Options: {full_options}")
    fit_keys = [fit_key] if isinstance(fit_key, str) else list(fit_key)
    if fit_keys and results_fit is None:
        raise ValueError("fit_key requires results_fit to be provided.")
    for key in fit_keys:
        if key not in _FIT_KEYS:
            raise ValueError(f"Unknown fit_key {key!r}. Options: {_FIT_KEYS}")

    viewer = SpectrumAlphaFigureViewer(
        results,
        results_cvpca=results_cvpca,
        results_subspace=results_subspace,
        results_fit=results_fit,
        source_cfg=source_cfg,
        full_cfg=full_cfg,
        ylim_min=ylim_min,
        ylim_max=ylim_max,
        fontsize=fontsize,
        figsize=figsize,
        width_ratios=width_ratios,
        pf_color=pf_color,
        ff_color=ff_color,
        pf_label=pf_label,
        ff_label=ff_label,
    )
    viewer.update_selection("source_key", value=source_key)
    viewer.update_selection("full_source_key", value=full_source_key)
    if results_fit is not None:
        viewer.update_multiple_selection("fit_key", value=fit_keys)

    valid_selections = set()
    for agg in viewer._agg.values():
        if agg is not None:
            valid_selections.update(agg.param_axes)
    if results_fit is not None:
        valid_selections.update(results_fit.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))

    viewer.update_float("ylim_min", value=ylim_min)
    viewer.update_float("ylim_max", value=ylim_max)
    viewer.update_boolean("normalize", value=normalize)
    viewer.update_float("beewidth", value=beewidth)
    viewer.update_float("each_line_alpha", value=each_line_alpha)
    viewer.update_float("point_alpha", value=point_alpha)
    viewer.update_float("markersize", value=markersize)
    viewer.update_float("mean_linewidth", value=mean_linewidth)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


class DimensionalityFamiliarityViewer(Viewer):
    """Participation-ratio dimensionality or adaptive decay exponent over session number.

    This is the familiarity analogue of :class:`SpectrumFigureViewer`: one selected placefield
    spectrum and one selected Full-CA1 spectrum are reduced to either a signed participation ratio
    or adaptive median-FPD decay exponent for every session. Sessions are then sorted
    chronologically within mouse and the two sources are reindexed onto the union of their session
    IDs, so missing coverage leaves a NaN gap instead of shifting a curve to the wrong session
    number.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        source_cfg: AdaptiveAlphaConfig | None = None,
        full_cfg: AdaptiveAlphaConfig | None = None,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (3.5, 2.5),
        pf_color: str = "orange",
        ff_color: str = "black",
        pf_label: str = "Placefields",
        ff_label: str = "Full CA1",
    ):
        self.results = results
        self.results_cvpca = results_cvpca
        self.results_subspace = results_subspace
        self.source_cfg = source_cfg if source_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]
        self.full_cfg = full_cfg if full_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["full"]
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.fontsize = fontsize
        self.figsize = figsize
        self.pf_color = pf_color
        self.ff_color = ff_color
        self.pf_label = pf_label
        self.ff_label = ff_label

        pf_options = list(_STIMSPACE_KEYS)
        if results_cvpca is not None:
            pf_options += list(_CVPCA_KEYS)
        self.add_selection("source_key", options=pf_options, value="ss_cv")

        full_options = ["SVD"] + (["SVCA"] if results_subspace is not None else [])
        self.add_selection("full_source_key", options=full_options, value="SVD")

        # Match SpectrumFigureViewer's shared data-selection widgets. The subspace-only
        # ``subspace_name``/``smooth_width`` axes stay fixed for SVCA; a shared
        # ``activity_parameters_name`` selection is forwarded below.
        merged_axes: dict[str, list] = {}
        for agg in self._agg.values():
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(option for option in options if option not in existing)

        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for name, options in merged_axes.items():
            if any(isinstance(option, tuple) for option in options):
                label_map = {_tuple_label(option): option for option in options}
                self._tuple_labels[name] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(name, options=widget_options)
            if name in _PREFERRED_DEFAULTS:
                default = self.encode_param(name, _PREFERRED_DEFAULTS[name])
                if default in widget_options:
                    self.update_selection(name, value=default)

        self.add_selection("metric", options=["participation_ratio", "alpha"], value="participation_ratio")
        self.add_selection("display", options=["each", "errorPlot"], value="errorPlot")
        self.add_boolean("log_y", value=False)

        # Independent PF/Full-CA1 adaptive-fit controls, matching SpectrumAlphaFigureViewer.
        # These are inert while metric="participation_ratio".
        for prefix, cfg in (("source", self.source_cfg), ("full", self.full_cfg)):
            self.add_selection(f"{prefix}_smooth_method", options=["none", "boxcar", "gaussian"], value=cfg.smooth_method)
            self.add_float(f"{prefix}_smooth_width", value=cfg.smooth_width, min=0.0, max=50.0, step=0.5)
            self.add_integer(f"{prefix}_fpd_window_size", value=cfg.fpd_window_size, min=1, max=50)
            self.add_integer(f"{prefix}_adaptive_buffer", value=cfg.adaptive_buffer, min=0, max=50)
            self.add_integer(f"{prefix}_minimum_window_size", value=cfg.minimum_window_size, min=1, max=500)

    encode_param = PlacefieldSpectraViewer.encode_param
    _sel_params = PlacefieldSpectraViewer._sel_params

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
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        spec = agg.sel(keys=[key], avg_by_mouse=False, **self._sel_params(state, source))[key]
        return np.atleast_2d(np.asarray(spec, dtype=float)), agg

    def _full_spectrum_sessions(self, state: dict) -> tuple[np.ndarray, ResultsAggregator]:
        """Return the selected Full-CA1 spectrum and the aggregator supplying its session rows."""
        if state["full_source_key"] == "SVD":
            return self._spectrum_sessions(state, _FF_KEY)

        params = {"subspace_name": "svca_subspace", "smooth_width": None}
        if "activity_parameters_name" in state:
            params["activity_parameters_name"] = state["activity_parameters_name"]
        spec = self.results_subspace.sel(keys=["variance_activity"], avg_by_mouse=False, **params)["variance_activity"]
        return np.atleast_2d(np.asarray(spec, dtype=float)), self.results_subspace

    def _alpha_per_session(
        self,
        state: dict,
        pf_raw: np.ndarray,
        pf_agg: ResultsAggregator,
        ff_raw: np.ndarray,
        ff_agg: ResultsAggregator,
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

        cvpca_raw, cvpca_agg = self._spectrum_sessions(state, "ss_cvpca")
        cvpca_smooth = _smooth_spectrum(cvpca_raw, source_cfg.smooth_method, source_cfg.smooth_width)
        pf_fb_raw = _align_rows_to_sessions(pf_agg.session_ids, cvpca_agg.session_ids, cvpca_raw)
        pf_fb_smooth = _align_rows_to_sessions(pf_agg.session_ids, cvpca_agg.session_ids, cvpca_smooth)

        ff_fb_raw, ff_fb_smooth = None, None
        if self.results_subspace is not None:
            params = {"subspace_name": "svca_subspace", "smooth_width": None}
            if "activity_parameters_name" in state:
                params["activity_parameters_name"] = state["activity_parameters_name"]
            svca_raw = self.results_subspace.sel(keys=["variance_activity"], avg_by_mouse=False, **params)["variance_activity"]
            svca_raw = np.atleast_2d(np.asarray(svca_raw, dtype=float))
            svca_smooth = _smooth_spectrum(svca_raw, full_cfg.smooth_method, full_cfg.smooth_width)
            ff_fb_raw = _align_rows_to_sessions(ff_agg.session_ids, self.results_subspace.session_ids, svca_raw)
            ff_fb_smooth = _align_rows_to_sessions(ff_agg.session_ids, self.results_subspace.session_ids, svca_smooth)

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
                ax.plot(np.arange(curve.size), curve, color=(color, 0.3), linewidth=0.5)
            if summary_columns.size:
                ax.plot(
                    summary_columns,
                    np.nanmean(stack[:, summary_columns], axis=0),
                    color=color,
                    linewidth=2.0,
                    label=label,
                )
            return stack.shape[1]

        if summary_columns.size:
            errorPlot(
                summary_columns,
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

    def plot(self, state: dict):
        pf_spec, pf_agg = self._spectrum_sessions(state, state["source_key"])
        ff_spec, ff_agg = self._full_spectrum_sessions(state)
        if state["metric"] == "alpha":
            pf_values, ff_values = self._alpha_per_session(state, pf_spec, pf_agg, ff_spec, ff_agg)
            ylabel = "Decay exponent"
        else:
            pf_values = _signed_participation_ratio(pf_spec)
            ff_values = _signed_participation_ratio(ff_spec)
            ylabel = "Dimensionality"
        pf_curves, ff_curves = self._aligned_mouse_curves(pf_values, pf_agg, ff_values, ff_agg)

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, layout="constrained")
        if state["log_y"]:
            ax.set_yscale("log")

        extents = [
            self._draw_curves(ax, pf_curves, self.pf_color, self.pf_label, state["display"]),
            self._draw_curves(ax, ff_curves, self.ff_color, self.ff_label, state["display"]),
        ]
        xmax = max(max(extents) - 1, 1)
        ax.set_xlim(-0.2, xmax + 0.2)
        ax.set_xlabel("Session #")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=self.fontsize, frameon=False)

        ylim = ax.get_ylim()
        format_spines(
            ax,
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[0, xmax],
            ybounds=ylim,
        )
        return fig


def dimensionality_familiarity(
    results: ResultsAggregator,
    results_cvpca: ResultsAggregator | None = None,
    results_subspace: ResultsAggregator | None = None,
    source_key: str = "ss_cv",
    full_source_key: str = "SVD",
    metric: str = "participation_ratio",
    display: str = "errorPlot",
    log_y: bool = False,
    source_cfg: AdaptiveAlphaConfig | None = None,
    full_cfg: AdaptiveAlphaConfig | None = None,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (3.5, 2.5),
    pf_color: str = "orange",
    ff_color: str = "black",
    pf_label: str = "Placefields",
    ff_label: str = "Full CA1",
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """Plot placefield and Full-CA1 dimensionality or decay exponent over session number.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, supplying the ``ss_*``/``sf_*`` PF spectra and the
        SVD Full-CA1 ``ff`` spectrum.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results. When provided, ``reg_covariances_fixed`` is added to the
        selectable PF ``source_key`` options, matching :func:`spectrum_figure`.
    results_subspace : ResultsAggregator or None
        Aggregated SubspaceConfig results. When provided, ``"SVCA"`` is added to the selectable
        ``full_source_key`` options and reads ``variance_activity`` at
        ``subspace_name="svca_subspace"`` and ``smooth_width=None``.
    source_key : str
        Placefield spectrum key, with the same options as :func:`spectrum_figure`.
    full_source_key : {"SVD", "SVCA"}
        Full-CA1 spectrum source, with the same availability rules as :func:`spectrum_figure`.
    metric : {"participation_ratio", "alpha"}
        Per-session spectrum summary. ``"participation_ratio"`` uses the signed whole-spectrum
        participation ratio. ``"alpha"`` uses the adaptive median-FPD decay exponent: PF window
        boundaries fall back to ``ss_cvpca`` and Full-CA1 boundaries fall back to SVCA when the
        selected spectrum has no negative crossover.
    display : {"each", "errorPlot"}
        ``"each"`` draws every mouse as a faint line plus the across-mouse mean.
        ``"errorPlot"`` draws the mean +/- SE band. Population summaries require at least two
        mice at a session number.
    log_y : bool
        Use a logarithmic y-axis.
    source_cfg, full_cfg : AdaptiveAlphaConfig or None
        Independent adaptive-alpha configurations for the PF and Full-CA1 curves. Each seeds its
        own Syd smoothing, FPD-window, buffer, and minimum-window-size controls. These settings are
        ignored for ``metric="participation_ratio"``.
    fontsize : float
        Base font size.
    figsize : tuple[float, float]
        Figure size in inches.
    pf_color, ff_color : str
        Colors of the placefield and Full-CA1 curves.
    pf_label, ff_label : str
        Legend labels of the placefield and Full-CA1 curves.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections, keyed by raw ``param_axes`` name.

    Returns
    -------
    matplotlib.figure.Figure or DimensionalityFamiliarityViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    pf_options = list(_STIMSPACE_KEYS) + (list(_CVPCA_KEYS) if results_cvpca is not None else [])
    if source_key not in pf_options:
        raise ValueError(f"Unknown PF source_key {source_key!r}. Options: {pf_options}")
    full_options = ["SVD"] + (["SVCA"] if results_subspace is not None else [])
    if full_source_key not in full_options:
        raise ValueError(f"Unknown full_source_key {full_source_key!r}. Options: {full_options}")
    if metric not in ("participation_ratio", "alpha"):
        raise ValueError(f"Unknown metric {metric!r}. Options: ['participation_ratio', 'alpha']")
    if display not in ("each", "errorPlot"):
        raise ValueError(f"Unknown display {display!r}. Options: ['each', 'errorPlot']")

    viewer = DimensionalityFamiliarityViewer(
        results,
        results_cvpca=results_cvpca,
        results_subspace=results_subspace,
        source_cfg=source_cfg,
        full_cfg=full_cfg,
        fontsize=fontsize,
        figsize=figsize,
        pf_color=pf_color,
        ff_color=ff_color,
        pf_label=pf_label,
        ff_label=ff_label,
    )
    viewer.update_selection("source_key", value=source_key)
    viewer.update_selection("full_source_key", value=full_source_key)

    valid_selections = set()
    for agg in viewer._agg.values():
        if agg is not None:
            valid_selections.update(agg.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))

    viewer.update_selection("metric", value=metric)
    viewer.update_selection("display", value=display)
    viewer.update_boolean("log_y", value=log_y)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


_PER_ENV_PF_KEYS = ["ss_cv", "ss_direct", "ss_cvpca", "sf_cv", "sf_direct"]
_PER_ENV_PF_RESULT_KEYS = {
    "ss_cv": "ss_cv_env",
    "ss_direct": "ss_direct_env",
    "ss_cvpca": "ss_cvpca_env",
}
_PER_ENV_FULL_SCOPES = ["full1", "fullall"]


class SpectrumDimFamiliarityViewer(Viewer):
    """Per-environment participation-ratio dimensionality over familiarity.

    Every environment is represented by its experience-order slot and is reindexed to session
    number within that environment for each mouse. PF and Full spectra are shown on separate axes.
    ``full_scope`` is shared by every estimator that has environment-only and all-session func-side
    variants: ``"full1"`` uses environment-only activity, while ``"fullall"`` uses the stored
    environment-vs-all-session cross-spectrum.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (6.0, 2.5),
        pf_label: str = "PF",
        full_label: str = "Full",
    ):
        self.results = results
        self.fontsize = fontsize
        self.figsize = figsize
        self.pf_label = pf_label
        self.full_label = full_label

        self.add_selection("source_key", options=_PER_ENV_PF_KEYS, value="ss_cvpca")
        self.add_selection("full_scope", options=_PER_ENV_FULL_SCOPES, value="full1")
        self.add_selection("display", options=["each", "errorPlot"], value="errorPlot")
        self.add_selection("yscale", options=["linear", "log"], value="linear")
        self.add_boolean("sharey", value=False)

        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for name, options in results.param_axes.items():
            if any(isinstance(option, tuple) for option in options):
                label_map = {_tuple_label(option): option for option in options}
                self._tuple_labels[name] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(name, options=widget_options)
            if name in _PREFERRED_DEFAULTS:
                default = self.encode_param(name, _PREFERRED_DEFAULTS[name])
                if default in widget_options:
                    self.update_selection(name, value=default)

    def encode_param(self, name: str, value):
        """Map raw tuple-valued aggregator parameters to widget-safe labels."""
        if name in self._tuple_labels and isinstance(value, tuple):
            return _tuple_label(value)
        return value

    def _sel_params(self, state: dict) -> dict:
        """Return only aggregator parameter selections, decoding tuple-valued axes."""
        params = {}
        for name in self.results.param_axes:
            if name not in state:
                continue
            value = state[name]
            if name in self._tuple_labels:
                value = self._tuple_labels[name][value]
            params[name] = value
        return params

    @staticmethod
    def _result_keys(source_key: str, full_scope: str) -> tuple[str, str]:
        """Map viewer choices to the two stored per-environment spectrum keys."""
        if source_key in ("sf_cv", "sf_direct"):
            pf_key = f"{source_key}_env_{full_scope}"
        else:
            pf_key = _PER_ENV_PF_RESULT_KEYS[source_key]
        full_key = "ff_env_full1" if full_scope == "full1" else "ff_env_full1_fullall"
        return pf_key, full_key

    @staticmethod
    def _participation_ratio(spec: np.ndarray) -> np.ndarray:
        """Signed participation ratio over the final (spectrum-rank) axis."""
        spec = np.asarray(spec, dtype=float)
        s1 = np.nansum(spec, axis=-1)
        s2 = np.nansum(spec**2, axis=-1)
        valid = np.isfinite(spec).any(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(valid & (s2 > 0), s1**2 / s2, np.nan)

    def _curves(self, values: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
        """Reindex ``(sessions, env_slots)`` values by within-environment session number."""
        curves: dict[int, dict[str, np.ndarray]] = {}
        for slot in range(MAX_ENV_SLOTS):
            per_mouse: dict[str, np.ndarray] = {}
            for mouse in self.results.unique_mice:
                rows = np.where(self.results.mouse_names == mouse)[0]
                dates = np.array([self.results.sessions[row].date for row in rows])
                rows = rows[np.argsort(dates)]
                curve = np.asarray(values[rows, slot], dtype=float)
                curve = curve[np.isfinite(curve)]
                if curve.size:
                    per_mouse[str(mouse)] = curve
            curves[slot] = per_mouse
        return curves

    @staticmethod
    def _pad_curves(curves: dict[str, np.ndarray]) -> np.ndarray:
        """NaN-pad ragged per-mouse curves to a common session axis."""
        max_sessions = max((len(curve) for curve in curves.values()), default=0)
        stack = np.full((len(curves), max_sessions), np.nan)
        for row, curve in enumerate(curves.values()):
            stack[row, : len(curve)] = curve
        return stack

    def _draw_axis(self, ax, curves: dict[int, dict[str, np.ndarray]], display: str) -> int:
        """Draw every environment slot on one axis and return the longest x extent."""
        max_length = 0
        for slot, per_mouse in curves.items():
            stack = self._pad_curves(per_mouse)
            if not stack.size:
                continue
            if ax.get_yscale() == "log":
                stack[stack <= 0] = np.nan

            color = ENV_SLOT_COLORS[slot % len(ENV_SLOT_COLORS)]
            support = np.sum(np.isfinite(stack), axis=0)
            summary_columns = np.where(support >= 2)[0]
            label = f"Env #{slot + 1}"

            if display == "each":
                for curve in stack:
                    ax.plot(np.arange(1, curve.size + 1), curve, color=(color, 0.3), linewidth=0.5)
                if summary_columns.size:
                    ax.plot(
                        summary_columns + 1,
                        np.nanmean(stack[:, summary_columns], axis=0),
                        color=color,
                        linewidth=2.0,
                        label=label,
                    )
                max_length = max(max_length, stack.shape[1])
            elif summary_columns.size:
                errorPlot(
                    summary_columns + 1,
                    stack[:, summary_columns],
                    axis=0,
                    se=True,
                    ax=ax,
                    color=color,
                    linewidth=2.0,
                    alpha=0.25,
                    label=label,
                )
                max_length = max(max_length, int(summary_columns[-1]) + 1)
        return max_length

    def plot(self, state: dict):
        pf_key, full_key = self._result_keys(state["source_key"], state["full_scope"])
        out = self.results.sel(
            keys=[pf_key, full_key],
            squeeze_ones=False,
            avg_by_mouse=False,
            **self._sel_params(state),
        )
        pf_values = self._participation_ratio(out[pf_key])
        full_values = self._participation_ratio(out[full_key])
        pf_curves = self._curves(pf_values)
        full_curves = self._curves(full_values)

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(1, 2, figsize=self.figsize, layout="constrained", sharey=state["sharey"])
        for axis in ax:
            axis.set_yscale(state["yscale"])

        extents = [
            self._draw_axis(ax[0], pf_curves, state["display"]),
            self._draw_axis(ax[1], full_curves, state["display"]),
        ]
        xmax = max(max(extents), 1)
        for axis, title in zip(ax, (self.pf_label, self.full_label)):
            axis.set_xlim(0.8, xmax + 0.2)
            axis.set_xlabel("Env session #")
            axis.set_ylabel("Dimensionality")
            axis.set_title(title)
            ylim = axis.get_ylim()
            format_spines(
                axis,
                x_pos=-0.02,
                y_pos=-0.02,
                spines_visible=["left", "bottom"],
                xbounds=[1, xmax],
                ybounds=ylim,
            )
        ax[0].legend(
            loc="best",
            fontsize=self.fontsize,
            frameon=False,
            title="Environment",
            handlelength=0.8,
            handletextpad=0.5,
        )
        return fig


def spectrum_dim_familiarity(
    results: ResultsAggregator,
    source_key: str = "ss_cvpca",
    full_scope: str = "full1",
    display: str = "errorPlot",
    yscale: str = "linear",
    sharey: bool = False,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (6.0, 2.5),
    pf_label: str = "PF",
    full_label: str = "Full",
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """Plot per-environment PF and Full participation-ratio dimensionality over familiarity.

    Each mouse's environments are aligned by experience-order slot and each slot is plotted against
    session number within that environment. PF dimensionality is drawn in ``ax[0]`` and Full
    dimensionality in ``ax[1]``, using the same environment colors as the manuscript's other
    familiarity figures.

    ``source_key`` may be ``ss_cv``, ``ss_direct``, ``ss_cvpca``, ``sf_cv``, or ``sf_direct``.
    ``full_scope="full1"`` uses environment-only functional activity for every applicable PF/Full
    estimator. ``"fullall"`` uses the corresponding environment-vs-all-session functional
    cross-spectrum. The SS estimators have no functional side, so ``full_scope`` changes only their
    paired Full curve. ``sharey=True`` gives the PF and Full panels one shared y-axis.
    """
    if source_key not in _PER_ENV_PF_KEYS:
        raise ValueError(f"Unknown source_key {source_key!r}. Options: {_PER_ENV_PF_KEYS}")
    if full_scope not in _PER_ENV_FULL_SCOPES:
        raise ValueError(f"Unknown full_scope {full_scope!r}. Options: {_PER_ENV_FULL_SCOPES}")
    if display not in ("each", "errorPlot"):
        raise ValueError(f"Unknown display {display!r}. Options: ['each', 'errorPlot']")
    if yscale not in ("linear", "log"):
        raise ValueError(f"Unknown yscale {yscale!r}. Options: ['linear', 'log']")

    viewer = SpectrumDimFamiliarityViewer(
        results,
        fontsize=fontsize,
        figsize=figsize,
        pf_label=pf_label,
        full_label=full_label,
    )
    viewer.update_selection("source_key", value=source_key)
    viewer.update_selection("full_scope", value=full_scope)
    viewer.update_selection("display", value=display)
    viewer.update_selection("yscale", value=yscale)
    viewer.update_boolean("sharey", value=sharey)
    for key, value in selections.items():
        if key not in results.param_axes:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(results.param_axes)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


# Fixed colors for the Tilbury-fit panels: generalized (Tilbury) vs plain-Gaussian control.
_GENERALIZED_COLOR = "blue"
_GAUSSIAN_COLOR = "black"
_SHRINKAGE_COLOR = "purple"


def _short_mouse_name(name: str) -> str:
    """Shorten ``CR_Hippocannula*`` mouse names to ``CR*``; other names unchanged."""
    prefix = "CR_Hippocannula"
    if name.startswith(prefix):
        return "CR" + name[len(prefix) :]
    return name


def _add_param_axis_widgets(viewer: Viewer, axes: dict[str, list]) -> dict[str, dict[str, tuple]]:
    """Add one selection widget per param axis of an aggregator, seeded with preferred defaults.

    Tuple-valued options are shown as string labels (:func:`_tuple_label`), since syd selections take
    scalars; the returned maps decode a widget value back to its tuple (see :func:`_decode_params`).

    Parameters
    ----------
    viewer : Viewer
        Viewer to add the widgets to.
    axes : dict
        ``ResultsAggregator.param_axes``: axis name -> list of stored values.

    Returns
    -------
    dict
        Label -> raw-value map per tuple-valued axis; axes with scalar options are absent.
    """
    tuple_labels: dict[str, dict[str, tuple]] = {}
    for name, options in axes.items():
        if any(isinstance(opt, tuple) for opt in options):
            label_map = {_tuple_label(opt): opt for opt in options}
            tuple_labels[name] = label_map
            widget_options = list(label_map)
        else:
            widget_options = list(options)
        viewer.add_selection(name, options=widget_options)
        if name in _PREFERRED_DEFAULTS:
            default = _PREFERRED_DEFAULTS[name]
            default = _tuple_label(default) if name in tuple_labels and isinstance(default, tuple) else default
            if default in widget_options:
                viewer.update_selection(name, value=default)
    return tuple_labels


def _decode_params(state: dict, names: list[str], tuple_labels: dict[str, dict[str, tuple]]) -> dict:
    """Params pinning each axis in ``names``, decoding tuple labels back to tuples.

    Every param axis of an aggregator must be pinned before slicing it, otherwise ``sel`` leaves that
    axis as an extra dimension of the returned arrays.
    """
    params = {}
    for name in names:
        if name not in state:
            continue
        value = state[name]
        if name in tuple_labels:
            value = tuple_labels[name][value]
        params[name] = value
    return params


class PlacefieldExampleFitViewer(Viewer):
    """Tilbury generalized-Gaussian placefield fits: grid of example single-neuron fits.

    An ``n_rows x n_cols`` grid of example single-neuron fits from one session (the top neurons by
    test R^2 that also clear the improvement threshold). Each panel overlays the held-out test
    placefield (points) against the three fitted curves: generalized-Gaussian (Tilbury, blue),
    plain-Gaussian control (black) and generalized-shrinkage (purple). The stored
    :class:`~dimensionality_manuscript.configs.tilbury_fit.TilburyFitConfig`
    results already hold the fitted parameters and R^2; only the held-out test curve is not stored,
    so it is rebuilt from the deterministic train/test split and trial-averaging (no re-fit). The fit
    param axes (``activity_parameters_name``) are selectable widgets.

    The population summaries live in the separate :class:`PlacefieldPopulationViewer`.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        registry: PopulationRegistry,
        n_rows: int = 1,
        n_cols: int = 3,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (8.0, 3.0),
    ):
        self.results = results
        self.registry = registry
        self.config = results.config_class
        self.fontsize = fontsize
        self.figsize = figsize
        # Rebuilding the test curve is cheap (deterministic trial-average), but cache by
        # (session_uid, fit params) so switching back to a session in the viewer is instant.
        self._fit_cache: dict[tuple, dict] = {}

        self.add_selection("example_session", options=list(results.session_ids), value=results.session_ids[0])
        # One widget per TilburyFitConfig param axis (activity_parameters_name, ...):
        # the stored fits exist once per combination, so every one must be pinned before slicing.
        self._fit_axes = list(results.param_axes)
        self._tuple_labels = _add_param_axis_widgets(self, results.param_axes)
        self.add_integer("n_rows", value=n_rows, min=1, max=6)
        self.add_integer("n_cols", value=n_cols, min=1, max=6)
        # Example neurons are drawn at random from those with generalized test R2 above r2_threshold
        # AND (generalized - gaussian) OR (shrinkage - gaussian) test R2 above improvement_threshold
        # (so the example fits well and beats the plain Gaussian via either the generalized or the
        # shrinkage fit). The seed makes the draw reproducible; if too few clear both thresholds the
        # extra panels are left empty.
        self.add_float("r2_threshold", value=0.5, min=-1.0, max=1.0, step=0.05)
        self.add_float("improvement_threshold", value=0.0, min=0.0, max=1.0, step=0.01)
        self.add_integer("random_seed", value=0, min=0, max=100000)
        # Per-panel normalization (see PlacefieldFitFigureViewer): divide the curve group by a
        # statistic (std / sum / max / none). normalize_independent scales each of the four curves
        # by its own statistic (shape-only); otherwise the group shares the test-data curve's scale,
        # keeping the fits overlaid on the data.
        self.add_selection("normalize", options=list(_FIT_FIGURE_NORMALIZATIONS), value="sum")
        self.add_boolean("normalize_independent", value=True)

    def _fit_sel_params(self, state: dict) -> dict:
        """Params pinning every TilburyFitConfig param axis, decoded from the widgets."""
        return _decode_params(state, self._fit_axes, self._tuple_labels)

    def _example_fit(self, session_uid: str, fit_params: dict) -> dict:
        """Return the (cached) example fit for ``session_uid`` at ``fit_params``, loading it on a miss."""
        cache_key = (session_uid, tuple(sorted(fit_params.items())))
        if cache_key not in self._fit_cache:
            self._fit_cache[cache_key] = self._load_example_fit(session_uid, fit_params)
        return self._fit_cache[cache_key]

    def _load_example_fit(self, session_uid: str, fit_params: dict) -> dict:
        """Assemble one session's example fit from stored results plus a rebuilt test curve.

        The fitted parameters and R^2 come straight from the aggregated
        :class:`TilburyFitConfig` results (no gradient descent). The held-out test placefield is
        not stored, so it is rebuilt with the same deterministic split (``registry.time_split``) and
        trial-averaging (``_avg_placefield``) the fit used; ``best_env``, the bin edges and the
        dropped-bin mask are recomputed exactly as :meth:`TilburyFitConfig.process` does.

        Parameters
        ----------
        session_uid : str
            Session to load.
        fit_params : dict
            One value per TilburyFitConfig param axis (see :meth:`_fit_sel_params`).

        Returns
        -------
        dict
            ``theta`` (P,), ``test_curve`` (n_kept, P), ``params`` (n_kept, 6),
            ``params_control`` (n_kept, 4), ``params_shrinkage`` (n_kept, 6), ``r2_test`` (n_kept,),
            ``r2_test_control`` (n_kept,), ``r2_test_shrinkage`` (n_kept,),
            ``lambda_selected`` (n_kept, 2), aligned so panel ``n`` uses row ``n`` of each.
        """
        config = self.config
        idx = self.results._session_index[session_uid]
        session = self.results.sessions[idx]

        sel = self.results.sel(
            keys=[
                "params",
                "params_control",
                "params_shrinkage",
                "r2_test",
                "r2_test_control",
                "r2_test_shrinkage",
                "lambda_selected",
                "idx_keep",
            ],
            load_ragged=True,
            squeeze_ones=False,
            **fit_params,
        )
        idx_keep = sel["idx_keep"][idx]  # (N_total,) bool; kept neurons, in order
        n_kept = int(np.sum(idx_keep))
        # Stored per-neuron arrays are padded with NaN to the max neuron count; kept rows are the
        # first n_kept, in the same order as idx_keep selects them below.
        params = sel["params"][idx][:n_kept]
        params_control = sel["params_control"][idx][:n_kept]
        params_shrinkage = sel["params_shrinkage"][idx][:n_kept]
        r2_test = sel["r2_test"][idx][:n_kept]
        r2_test_control = sel["r2_test_control"][idx][:n_kept]
        r2_test_shrinkage = sel["r2_test_shrinkage"][idx][:n_kept]
        lambda_selected = sel["lambda_selected"][idx][:n_kept]  # (n_kept, 2): per-neuron (lam_p, lam_asym)

        # Original ROI indices that entered the fit: population.idx_neurons (the AND of
        # session.idx_rois across all spks_types), NOT the current-spks_type session.idx_rois.
        # idx_keep indexes into this array, so idx_neurons[j] recovers a neuron's original index.
        population, _ = self.registry.get_population(session, config.spks_type)
        idx_neurons = np.asarray(population.idx_neurons)

        # Recompute the fit's fixed choices (all "skip"ped from storage, all deterministic).
        num_per_env = {i: int(np.sum(session.trial_environment == i)) for i in session.environments}
        best_env = max(num_per_env, key=num_per_env.get)
        dist_edges = np.linspace(0, session.env_length[0], config.num_bins + 1)
        dist_centers = edge2center(dist_edges)

        # Trial-average every split's placefield over the kept neurons; the counts give the
        # dropped-bin mask (bins empty in any split) so theta matches the stored params' support.
        spks, fb = config._get_split_data(session, self.registry)
        for s in _SPLITS:
            spks[s] = spks[s][:, idx_keep]
        curves, counts = {}, {}
        for s in _SPLITS:
            curves[s], counts[s] = config._avg_placefield(spks[s], fb[s], dist_edges, best_env, session)
        bad = np.zeros(config.num_bins, dtype=bool)
        for s in _SPLITS:
            bad |= counts[s] == 0
        good = ~bad

        return {
            "theta": dist_centers[good],
            "test_curve": curves["test"][:, good],
            "params": params,
            "params_control": params_control,
            "params_shrinkage": params_shrinkage,
            "r2_test": r2_test,
            "r2_test_control": r2_test_control,
            "r2_test_shrinkage": r2_test_shrinkage,
            "lambda_selected": lambda_selected,
            "idx_keep": idx_keep,
            "idx_neurons": idx_neurons,
        }

    def plot(self, state: dict):
        method = state["normalize"]
        independent = bool(state["normalize_independent"])

        n_rows = int(state["n_rows"])
        n_cols = int(state["n_cols"])
        fit = self._example_fit(state["example_session"], self._fit_sel_params(state))

        plt.rcParams["font.size"] = self.fontsize
        fig = plt.figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(n_rows, n_cols)

        theta = fit["theta"]
        n_show = n_rows * n_cols
        r2 = fit["r2_test"]
        r2c = fit["r2_test_control"]
        r2s = fit["r2_test_shrinkage"]
        # Well-fit by the generalized model AND beating the plain Gaussian by improvement_threshold
        # via *either* the generalized or the shrinkage fit (OR logic).
        improves = ((r2 - r2c) > state["improvement_threshold"]) | ((r2s - r2c) > state["improvement_threshold"])
        eligible = np.flatnonzero(np.isfinite(r2) & np.isfinite(r2c) & np.isfinite(r2s) & (r2 > state["r2_threshold"]) & improves)
        rng = np.random.default_rng(int(state["random_seed"]))
        # Random draw without replacement; if too few clear the threshold, extra panels stay empty.
        chosen = rng.choice(eligible, size=n_show, replace=False) if eligible.size > n_show else eligible
        share_ax = None
        for cell in range(n_show):
            r, c = divmod(cell, n_cols)
            # Share x (common position axis) but not y: each neuron gets its own optimal y-range.
            ax = fig.add_subplot(gs[r, c], sharex=share_ax)
            share_ax = share_ax or ax
            if r == n_rows - 1:
                ax.set_xlabel("Position (cm)")
            if c == 0:
                ax.set_ylabel("Activity")
            if cell >= len(chosen):
                continue  # not enough eligible neurons -> leave this panel empty
            n = chosen[cell]

            # Original ROI index of this neuron: n indexes the kept arrays, np.where(idx_keep) maps
            # it to a row of idx_neurons (population.idx_neurons), which already holds original indices.
            idx_within_fit_neurons = np.where(fit["idx_keep"])[0][n]
            idx_within_idx_rois = fit["idx_neurons"][idx_within_fit_neurons]

            data = fit["test_curve"][n]
            gen = _eval_tilbury(theta, fit["params"][n])
            gauss = _eval_gaussian(theta, fit["params_control"][n])
            shrink = _eval_tilbury(theta, fit["params_shrinkage"][n])
            # normalize_independent: each curve divided by its own statistic (shape-only). Otherwise
            # the whole set shares the test-data curve's scale, keeping the fits overlaid on the data.
            if independent:
                data = data / _fit_figure_scale(data, method)
                gen = gen / _fit_figure_scale(gen, method)
                gauss = gauss / _fit_figure_scale(gauss, method)
                shrink = shrink / _fit_figure_scale(shrink, method)
            else:
                scale = _fit_figure_scale(data, method)
                data, gen, gauss, shrink = data / scale, gen / scale, gauss / scale, shrink / scale

            first = cell == 0
            ax.plot(theta, data, "o", color="red", ms=2.5, alpha=0.5, label="Test data" if first else None)
            ax.plot(theta, gen, "-", color=_GENERALIZED_COLOR, lw=1.5, label="Generalized" if first else None)
            ax.plot(theta, gauss, "-", color=_GAUSSIAN_COLOR, lw=1.5, label="Gaussian" if first else None)
            ax.plot(theta, shrink, "-", color=_SHRINKAGE_COLOR, lw=1.5, label="Generalized (shrinkage)" if first else None)
            lam_p, lam_asym = fit["lambda_selected"][n]
            ax.set_title(
                f"{state['example_session']} | Neuron: {idx_within_idx_rois} | R²={fit['r2_test'][n]:.2f} | λ=({lam_p:g}, {lam_asym:g})",
                fontsize=self.fontsize,
            )
            if first:
                ax.legend(fontsize=self.fontsize * 0.8, frameon=False, loc="upper right")
        return fig


# Normalization presets for PlacefieldFitFigureViewer: each maps a curve to the scalar its group
# (test data + both fits) is divided by, computed on the *test-data* curve so the fits stay overlaid
# on the data while every panel shares a common scale (needed for sharey).
_FIT_FIGURE_NORMALIZATIONS = ("std", "sum", "max", "none")

# Which generalized-family fit(s) PlacefieldFitFigureViewer overlays alongside the plain-Gaussian
# control: just the unregularized generalized fit, just the shrinkage fit, or both. When only one is
# shown it is drawn in the generalized (blue) color; "both" keeps generalized blue and shrinkage purple.
_FIT_MODEL_OPTIONS = ("generalized", "shrinkage", "both")

# Matplotlib ``loc`` strings offered for the placement of the fit-figure legend.
_LEGEND_POSITIONS = (
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
    "best",
)


def _fit_figure_scale(ref: np.ndarray, method: str) -> float:
    """Scalar to divide a curve group by, from the reference (test-data) curve.

    ``method`` is one of :data:`_FIT_FIGURE_NORMALIZATIONS`. Returns ``1.0`` when the statistic is
    non-finite or non-positive (flat / empty curve) so normalization is a no-op instead of blowing up.
    """
    if method == "none":
        return 1.0
    if method == "std":
        s = float(np.nanstd(ref))
    elif method == "sum":
        s = float(np.nansum(ref))
    elif method == "max":
        s = float(np.nanmax(ref))
    else:
        raise ValueError(f"Unknown normalization {method!r}. Options: {list(_FIT_FIGURE_NORMALIZATIONS)}")
    return s if np.isfinite(s) and s > 0 else 1.0


class PlacefieldFitFigureViewer(Viewer):
    """Tilbury placefield fits for a hand-picked list of (session, neuron) examples.

    Unlike :class:`PlacefieldExampleFitViewer` (which *draws* well-fit neurons at random from one
    session), this viewer plots an explicit, ordered list of neurons the user selected by eye — each
    identified by its session_uid and its **original ROI index** (the index into the session's full
    ROI set, i.e. ``np.where(session.idx_rois)[0][k]``). That original index is the stable identifier
    to write down for a figure: it survives regardless of how many neurons cleared the reliability /
    fraction-active thresholds in :class:`TilburyFitConfig`.

    Each requested ROI is traced back to its fit: it must be one of the pipeline's available neurons
    (``population.idx_neurons``) *and* have been kept by the fit's inclusion thresholds
    (``idx_keep``). A neuron that was never available or was dropped before fitting raises (opt-in via
    ``strict``) or is flagged with an empty, titled panel.

    The first ``n_rows * n_cols`` entries of the list are plotted, in order, into a shared-axes grid
    (``sharex``/``sharey``); each panel overlays the held-out test placefield (points) against the
    plain-Gaussian control (always shown) and the generalized-family fit(s) selected by ``fit_model``
    (generalized, shrinkage, or both). The whole group in a panel is normalized by the test-data
    curve's statistic (``normalize``: std / sum / max / none), so the fits stay overlaid on the data
    while panels remain comparable under ``sharey``. Only one panel is labelled and carries the
    legend, picked by ``legend_axis`` (a flat row-major panel index).
    """

    def __init__(
        self,
        results: ResultsAggregator,
        registry: PopulationRegistry,
        session_uids: list[str],
        neurons: list[int],
        n_rows: int = 2,
        n_cols: int = 3,
        legend_axis: int = 0,
        legend_position: str = "upper right",
        legend_x_offset: float = 0.0,
        legend_y_offset: float = 0.0,
        legend_handlelength: float = 2.0,
        legend_handletextpad: float = 0.8,
        legend_markerfirst: bool = True,
        normalize: str = "std",
        normalize_independent: bool = False,
        fit_model: str = "both",
        strict: bool = True,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (8.0, 4.0),
    ):
        if len(session_uids) != len(neurons):
            raise ValueError(f"session_uids and neurons must be the same length, got {len(session_uids)} and {len(neurons)}.")
        if normalize not in _FIT_FIGURE_NORMALIZATIONS:
            raise ValueError(f"Unknown normalize {normalize!r}. Options: {list(_FIT_FIGURE_NORMALIZATIONS)}")
        if fit_model not in _FIT_MODEL_OPTIONS:
            raise ValueError(f"Unknown fit_model {fit_model!r}. Options: {list(_FIT_MODEL_OPTIONS)}")
        if legend_position not in _LEGEND_POSITIONS:
            raise ValueError(f"Unknown legend_position {legend_position!r}. Options: {list(_LEGEND_POSITIONS)}")

        self.results = results
        self.registry = registry
        self.config = results.config_class
        self.session_uids = list(session_uids)
        self.neurons = [int(n) for n in neurons]
        self.strict = strict
        self.fontsize = fontsize
        self.figsize = figsize
        # Rebuilding a session's test curves is cheap but cache by (session_uid, fit params) so the
        # same session appearing for several requested neurons is only loaded once.
        self._fit_cache: dict[tuple, dict] = {}

        self.add_integer("n_rows", value=n_rows, min=1, max=8)
        self.add_integer("n_cols", value=n_cols, min=1, max=8)
        # Flat panel index (row-major, resolved with divmod by n_cols) of the panel that carries the
        # legend; its upper bound tracks the current grid size.
        self.add_integer("legend_axis", value=legend_axis, min=0, max=n_rows * n_cols - 1)
        # Legend layout, passed straight through to ax.legend: placement, the length of the sample
        # line in each entry, the gap between that sample and its label, and whether the sample is
        # drawn to the left of the label (False puts it on the right).
        self.add_selection("legend_position", options=list(_LEGEND_POSITIONS), value=legend_position)
        # Shift the anchor box the legend is placed against, in axes-fraction units (1.0 = one full
        # axes width/height), so the legend can sit outside its panel — e.g. in the gap between two
        # panels. 0 leaves it flush with the axes, as a plain ``loc`` would put it.
        self.add_float("legend_x_offset", value=legend_x_offset, min=-2.0, max=2.0, step=0.01)
        self.add_float("legend_y_offset", value=legend_y_offset, min=-2.0, max=2.0, step=0.01)
        self.add_float("legend_handlelength", value=legend_handlelength, min=0.0, max=6.0, step=0.1)
        self.add_float("legend_handletextpad", value=legend_handletextpad, min=0.0, max=3.0, step=0.05)
        self.add_boolean("legend_markerfirst", value=legend_markerfirst)
        # One widget per TilburyFitConfig param axis (see PlacefieldExampleFitViewer).
        self._fit_axes = list(results.param_axes)
        self._tuple_labels = _add_param_axis_widgets(self, results.param_axes)
        self.add_selection("normalize", options=list(_FIT_FIGURE_NORMALIZATIONS), value=normalize)
        # normalize_independent: scale each of the three curves (test data, generalized, gaussian) by
        # its own statistic (shape-only comparison), instead of the whole group by the test-data curve.
        self.add_boolean("normalize_independent", value=normalize_independent)
        # Which generalized-family fit(s) to overlay (the plain-Gaussian control is always shown).
        self.add_selection("fit_model", options=list(_FIT_MODEL_OPTIONS), value=fit_model)

        self.on_change(["n_rows", "n_cols"], self.update_legend_bounds)

    _fit_sel_params = PlacefieldExampleFitViewer._fit_sel_params

    def update_legend_bounds(self, state: dict):
        """Keep the legend panel index inside the current ``n_rows * n_cols`` grid."""
        self.update_integer("legend_axis", max=int(state["n_rows"]) * int(state["n_cols"]) - 1)

    def _session_fit(self, session_uid: str, fit_params: dict) -> dict:
        """Return the (cached) per-session fit bundle for ``session_uid`` at ``fit_params``."""
        cache_key = (session_uid, tuple(sorted(fit_params.items())))
        if cache_key not in self._fit_cache:
            self._fit_cache[cache_key] = self._load_session_fit(session_uid, fit_params)
        return self._fit_cache[cache_key]

    def _load_session_fit(self, session_uid: str, fit_params: dict) -> dict:
        """Assemble one session's fit bundle: kept-neuron fits, rebuilt test curves, and the
        original-ROI-index -> kept-row map needed to resolve a hand-picked neuron.

        Fitted parameters and R^2 come straight from the stored :class:`TilburyFitConfig` results; the
        held-out test placefield is rebuilt with the same deterministic split and trial-averaging the
        fit used (no re-fit), exactly as :meth:`PlacefieldExampleFitViewer._load_example_fit` does.

        Returns
        -------
        dict
            ``theta`` (P,), ``test_curve`` (n_kept, P), ``params`` (n_kept, 6),
            ``params_control`` (n_kept, 4), ``params_shrinkage`` (n_kept, 6), ``r2_test`` (n_kept,),
            ``r2_test_control`` (n_kept,),
            ``idx_neurons`` (N_available,) original ROI indices that entered the pipeline, and
            ``idx_keep`` (N_available,) bool mask of which of those were fitted (kept rows are the
            finite prefix of the per-neuron arrays, in ``idx_neurons`` order).
        """
        if session_uid not in self.results._session_index:
            raise KeyError(f"session_uid {session_uid!r} not in results (options: {list(self.results.session_ids)}).")

        config = self.config
        idx = self.results._session_index[session_uid]
        session = self.results.sessions[idx]

        sel = self.results.sel(
            keys=["params", "params_control", "params_shrinkage", "r2_test", "r2_test_control", "idx_keep"],
            load_ragged=True,
            squeeze_ones=False,
            **fit_params,
        )
        idx_keep = np.asarray(sel["idx_keep"][idx], dtype=bool)  # (N_available,) over population.idx_neurons
        n_kept = int(np.sum(idx_keep))
        params = sel["params"][idx][:n_kept]
        params_control = sel["params_control"][idx][:n_kept]
        params_shrinkage = sel["params_shrinkage"][idx][:n_kept]
        r2_test = sel["r2_test"][idx][:n_kept]
        r2_test_control = sel["r2_test_control"][idx][:n_kept]

        # Original ROI indices that entered the fit (population.idx_neurons), used to map a
        # hand-picked original index -> row of idx_keep. Same registry/spks_type as the fit.
        population, _ = self.registry.get_population(session, config.spks_type)
        idx_neurons = np.asarray(population.idx_neurons)

        # Recompute the fit's deterministic choices and rebuild each split's trial-averaged curve.
        num_per_env = {i: int(np.sum(session.trial_environment == i)) for i in session.environments}
        best_env = max(num_per_env, key=num_per_env.get)
        dist_edges = np.linspace(0, session.env_length[0], config.num_bins + 1)
        dist_centers = edge2center(dist_edges)

        spks, fb = config._get_split_data(session, self.registry)
        for s in _SPLITS:
            spks[s] = spks[s][:, idx_keep]
        curves, counts = {}, {}
        for s in _SPLITS:
            curves[s], counts[s] = config._avg_placefield(spks[s], fb[s], dist_edges, best_env, session)
        bad = np.zeros(config.num_bins, dtype=bool)
        for s in _SPLITS:
            bad |= counts[s] == 0
        good = ~bad

        return {
            "theta": dist_centers[good],
            "test_curve": curves["test"][:, good],
            "params": params,
            "params_control": params_control,
            "params_shrinkage": params_shrinkage,
            "r2_test": r2_test,
            "r2_test_control": r2_test_control,
            "idx_neurons": idx_neurons,
            "idx_keep": idx_keep,
            "session": session,
        }

    def _resolve(self, session_uid: str, roi: int, fit_params: dict) -> tuple[dict, Optional[int], str]:
        """Map a hand-picked ``(session_uid, original ROI index)`` to its kept-row in the fit bundle.

        Returns ``(fit, kept_row, status)`` where ``status`` is ``"ok"`` (``kept_row`` is the row of
        ``params`` / ``test_curve`` for this neuron), ``"not_available"`` (ROI never entered the
        pipeline — silent / filtered out), or ``"not_fit"`` (available but dropped by the reliability
        / fraction-active thresholds). ``kept_row`` is ``None`` for the two failure statuses.
        """
        fit = self._session_fit(session_uid, fit_params)
        idx_neurons = fit["idx_neurons"]
        pos = np.flatnonzero(idx_neurons == roi)
        if pos.size == 0:
            return fit, None, "not_available"
        j = int(pos[0])
        if not fit["idx_keep"][j]:
            return fit, None, "not_fit"
        kept_row = int(np.sum(fit["idx_keep"][:j]))
        return fit, kept_row, "ok"

    def plot(self, state: dict):
        n_rows = int(state["n_rows"])
        n_cols = int(state["n_cols"])
        method = state["normalize"]
        independent = bool(state["normalize_independent"])
        n_show = n_rows * n_cols
        # Panel that carries the legend, as a flat row-major index into the grid.
        legend_cell = min(int(state["legend_axis"]), n_show - 1)
        fit_params = self._fit_sel_params(state)

        plt.rcParams["font.size"] = self.fontsize
        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.figsize, squeeze=False, layout="constrained")

        ylims = {}
        for cell in range(n_show):
            r, c = divmod(cell, n_cols)
            ax = axs[r, c]
            if r == n_rows - 1:
                ax.set_xlabel("Position (cm)")
            if c == 0:
                ax.set_ylabel("Activity")
            if cell >= len(self.session_uids):
                ax.set_visible(False)  # fewer requested neurons than panels -> hide the extras
                continue

            session_uid, roi = self.session_uids[cell], self.neurons[cell]
            fit, kept_row, status = self._resolve(session_uid, roi, fit_params)
            if status != "ok":
                # Traced but not fittable: flag loudly (strict) or leave a titled empty panel.
                if self.strict:
                    raise ValueError(f"Neuron roi={roi} in session {session_uid!r} is '{status}' (not a fitted neuron).")
                ax.set_title(f"{session_uid}\nroi {roi}: {status}", fontsize=self.fontsize * 0.8, color="red")
                continue

            theta = fit["theta"]
            data = fit["test_curve"][kept_row]
            gen = _eval_tilbury(theta, fit["params"][kept_row])
            gauss = _eval_gaussian(theta, fit["params_control"][kept_row])
            shrink = _eval_tilbury(theta, fit["params_shrinkage"][kept_row])
            # normalize_independent: each curve divided by its own statistic (shape-only). Otherwise
            # the whole set shares the test-data curve's scale, keeping the fits overlaid on the data.
            if independent:
                data = data / _fit_figure_scale(data, method)
                gen = gen / _fit_figure_scale(gen, method)
                gauss = gauss / _fit_figure_scale(gauss, method)
                shrink = shrink / _fit_figure_scale(shrink, method)
            else:
                scale = _fit_figure_scale(data, method)
                data, gen, gauss, shrink = data / scale, gen / scale, gauss / scale, shrink / scale

            # The legend lives on one panel, so only that panel's curves get labelled.
            first = (r, c) == divmod(legend_cell, n_cols)
            ax.plot(theta, data, "o", color="gray", ms=2.5, alpha=0.5, label="Test data" if first else None)
            ax.plot(theta, gauss, "-", color=_GAUSSIAN_COLOR, lw=1.5, label="Gaussian" if first else None)
            # Overlay the requested generalized-family fit(s). A lone fit is drawn blue (the
            # generalized color); "both" keeps generalized blue and shrinkage purple.
            fit_model = state["fit_model"]
            if fit_model in ("generalized", "both"):
                ax.plot(theta, gen, "-", color=_GENERALIZED_COLOR, lw=1.5, label="Generalized" if first else None)
            if fit_model in ("shrinkage", "both"):
                shrink_color = _SHRINKAGE_COLOR if fit_model == "both" else _GENERALIZED_COLOR
                shrink_label = "Generalized (shrinkage)" if fit_model == "both" else "Generalized"
                ax.plot(theta, shrink, "-", color=shrink_color, lw=1.5, label=shrink_label if first else None)
            # ax.set_title(f"{session_uid}\nroi {roi}  R²={fit['r2_test'][kept_row]:.2f}", fontsize=self.fontsize * 0.8)
            if first:
                # The anchor box defaults to the axes box (0, 0, 1, 1); offsetting its origin slides
                # the legend off the panel without moving where ``loc`` pins it within the box.
                leg = ax.legend(
                    fontsize=self.fontsize - 1,
                    frameon=False,
                    loc=state["legend_position"],
                    bbox_to_anchor=(float(state["legend_x_offset"]), float(state["legend_y_offset"]), 1.0, 1.0),
                    markerfirst=bool(state["legend_markerfirst"]),
                    handlelength=float(state["legend_handlelength"]),
                    handletextpad=float(state["legend_handletextpad"]),
                )
                # Keep the constrained layout from reserving space for an off-panel legend, which
                # would shrink the panels and break the uniform grid.
                leg.set_in_layout(False)
                # A legend spilling past its panel is drawn under any axes created after this one, so
                # the neighbour's opaque background would clip it. Draw this panel last, over a
                # transparent patch so raising it does not hide anything underneath.
                ax.set_zorder(10)
                ax.patch.set_visible(False)

        xbounds = (0, theta[-1] + (theta[1] - theta[0]) / 2)
        xticks = xbounds
        ylims = [ax.get_ylim() for ax in axs.flat if ax.get_visible()]
        ymax = max(yl[1] for yl in ylims)
        # Extend the drawn y-range a touch below 0 so the test-data points sitting at ~0 are not
        # clipped by the bottom edge; the spine (ybounds) still starts at exactly 0.
        ylims = (-0.05 * ymax, ymax)
        ybounds = (0, np.floor(ymax * 10) / 10)
        for cell in range(n_show):
            r, c = divmod(cell, n_cols)
            ax = axs[r, c]
            on_left = c == 0
            on_bottom = r == n_rows - 1
            spines_visible = ["bottom"]
            if on_left:
                spines_visible.append("left")
            xticks = xbounds if on_bottom else []
            yticks = ybounds if on_left else []
            ylabels = [0, 1] if on_left else []
            ax.set_ylim(ylims)
            format_spines(
                ax,
                x_pos=-0.02,
                y_pos=-0.02,
                xbounds=xbounds,
                ybounds=ybounds,
                spines_visible=spines_visible,
                xticks=xticks,
                yticks=yticks,
                ylabels=ylabels,
            )
        return fig


class PlacefieldPopulationViewer(Viewer):
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
      values per mouse (``fraction_view="by_mouse"``).
    - gs[-1]: an across-mouse spectrum-decay statistic for the selected ``source_key`` spectrum (from
      ``results_spectra``/``results_cvpca``) plus the :data:`_POP_EIG_KEYS` fit spectra (colors in
      :data:`_POP_ALPHA_COLORS`), selected by ``last_axis``:

      - ``"decay_exponent"``: one beeswarm column per curve of the power-law exponent estimated
        exactly as in :class:`SpectrumFigureViewer` — the median five-point-derivative local exponent
        over each session's own peak-curvature-to-noise-floor window
        (:func:`_second_derivative_window`), computed per session and averaged by mouse, under one
        :class:`AdaptiveAlphaConfig` (the ``source_*`` widgets). Keys with no negative entry borrow
        their window from that session's ``ss_cvpca`` row when ``results_spectra`` is given.
      - ``"characteristic_dim"`` / ``"mse"``: the two statistics of
        :class:`PlacefieldSpectrumMSEViewer`, each curve fit under both :data:`_DECAY_MODELS` (power
        law and exponential, the two x-positions of the "Fit Type" axis) over the ``fit_zone`` window
        — the characteristic parameter (``alpha`` / ``M``, y-label "Dimensionality", spanning
        ``[0, max]`` with ticks every 5 units) or the log-space MSE — laid out by ``display``
        (``each``/``errorPlot``/``swarm``).

      The panel's legend is styled by the ``legend_*`` widgets (:func:`_add_legend_widgets`).

    The example single-neuron fits live in the separate :class:`PlacefieldExampleFitViewer`.
    ``TilburyFitConfig``'s param axes (``activity_parameters_name``) are merged
    with the spectra aggregators' axes into one widget per axis; every panel is sliced to that
    selection.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_spectra: ResultsAggregator | None = None,
        results_cvpca: ResultsAggregator | None = None,
        source_cfg: AdaptiveAlphaConfig | None = None,
        num_bins: int = 80,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (6.0, 3.0),
    ):
        self.results = results
        self.results_spectra = results_spectra
        self.results_cvpca = results_cvpca
        # Alias so the SpectrumFigureViewer methods borrowed below (which fetch the eig spectra from
        # ``results_fit``) resolve to this viewer's Tilbury-fit aggregator.
        self.results_fit = results
        self.source_cfg = source_cfg if source_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]
        # Reused by _spectrum_sessions/_sel_params (borrowed from the spectra viewers): the source_key
        # spectrum for the gs[-1] alpha panel comes from these, resolved via SOURCE_OF_KEY.
        self._agg = {"stimspace": results_spectra, "cvpca": results_cvpca}
        self.config = results.config_class
        self.fontsize = fontsize
        self.figsize = figsize

        # Bin count for the per-session KDE of the peak-exponent density (gs[0]).
        self.add_integer("num_bins", value=num_bins, min=5, max=200)
        self.add_selection("fraction_view", options=["pooled", "by_mouse", "none"], value="none")
        self.add_float("beewidth", value=0.2, min=0.0, max=1.0, step=0.01)
        self.add_selection("metric", value="cc", options=["r2", "cc"])
        # Which generalized fit to show. "both" plots the unregularized (blue) and the shrinkage
        # (purple) fits side by side, as before. "generalized"/"shrinkage" plot only that one fit,
        # always drawn blue and labelled "Generalized" -- the caller knows which it is from the
        # parameter request. "include_better" toggles the per-neuron "better" composite in gs[-1].
        self.add_selection("generalized_fit", options=["both", "generalized", "shrinkage"], value="both")
        self.add_boolean("include_better", value=True)
        # Paired test for the gs[1] fit-vs-Gaussian comparison(s): two-sided paired t-test or
        # Wilcoxon signed-rank, on the per-mouse averages. Bonferroni-corrected when "both".
        self.add_selection("paired_test", options=["ttest", "wilcoxon"], value="ttest")

        # --- ax[3] population spectrum-decay panel: source_key spectrum + eig fit spectra ---
        # Which statistic the panel shows: the adaptive median-FPD power-law exponent (one value per
        # curve, beeswarm), or one of the two decay-law statistics of PlacefieldSpectrumMSEViewer --
        # the characteristic parameter (power-law ``alpha`` / exponential ``M``) or the log-space MSE
        # -- each evaluated at both _DECAY_MODELS x-positions.
        self.add_selection("last_axis", options=["decay_exponent", "characteristic_dim", "mse"], value="decay_exponent")
        # Decay-law fit window and layout, used only by the "characteristic_dim"/"mse" options (the
        # exponent option always uses its own adaptive window and a beeswarm). Same semantics as
        # PlacefieldSpectrumMSEViewer: per-session adaptive window or one fixed [start, end) window,
        # and "each" per-mouse lines / "errorPlot" mean +/- SE band / "swarm" beeswarm columns.
        self.add_selection("fit_zone", options=["adaptive", "fixed"], value="adaptive")
        self.add_integer_range("fixed_range", value=(10, 50), min=1, max=500)
        self.add_selection("display", options=["each", "errorPlot", "swarm"], value="each")
        # gs[-1] legend styling, available under every ``last_axis`` option. At the default
        # ``legend_loc="auto"`` the decay-law panels place a legend at "best" and the exponent beeswarm
        # shows none (its columns are labelled on the x-axis); any explicit loc draws one either way.
        _add_legend_widgets(self)
        # source_key options mirror spectrum_figure (StimSpace keys, plus the CVPCA key when given).
        if results_spectra is not None:
            source_options = list(_STIMSPACE_KEYS) + (list(_CVPCA_KEYS) if results_cvpca is not None else [])
            self.add_selection("source_key", options=source_options, value="ss_cv")

        # One widget per shared param-axis name (same tuple-label scheme as SpectrumFigureViewer), so
        # the source_key spectrum can be sliced (activity_parameters_name, smooth_widths, ...). The
        # TilburyFitConfig axes (activity_parameters_name) are merged in: every panel
        # here slices the fit results, so each of its axes must be pinned by a widget too.
        merged_axes: dict[str, list] = {}
        for agg in list(self._agg.values()) + [results]:
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(opt for opt in options if opt not in existing)
        self._fit_axes = list(results.param_axes)
        self._tuple_labels = _add_param_axis_widgets(self, merged_axes)

        # Adaptive median-FPD estimation controls, shared by every gs[-1] curve: one widget per
        # AdaptiveAlphaConfig field (same "source"-prefixed scheme as SpectrumFigureViewer).
        self.add_boolean("normalize", value=True)
        cfg = self.source_cfg
        self.add_selection("source_smooth_method", options=["none", "boxcar", "gaussian"], value=cfg.smooth_method)
        self.add_float("source_smooth_width", value=cfg.smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_integer("source_fpd_window_size", value=cfg.fpd_window_size, min=1, max=50)
        self.add_integer("source_adaptive_buffer", value=cfg.adaptive_buffer, min=0, max=50)
        self.add_integer("source_minimum_window_size", value=cfg.minimum_window_size, min=1, max=500)

    encode_param = PlacefieldSpectraViewer.encode_param
    _sel_params = PlacefieldSpectraViewer._sel_params
    # Adaptive-alpha machinery is shared verbatim with the spectrum figure: same per-session spectra
    # (raw + smoothed), same fit-aggregator eig spectra, same AdaptiveAlphaConfig assembly.
    # staticmethod on the source class: re-wrap, otherwise the plain function rebinds as a method.
    _cfg_from_state = staticmethod(SpectrumFigureViewer._cfg_from_state)
    _spectrum_sessions = SpectrumFigureViewer._spectrum_sessions
    _fit_spectrum_raw_sessions = SpectrumFigureViewer._fit_spectrum_raw_sessions
    _fit_spectrum_sessions = SpectrumFigureViewer._fit_spectrum_sessions
    _fit_sel_params = SpectrumFigureViewer._fit_sel_params

    def _aggregate_stats(self, state: dict) -> dict:
        """Per-session and per-mouse summary arrays for the population panels.

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

    def plot(self, state: dict):
        stats = self._aggregate_stats(state)

        plt.rcParams["font.size"] = self.fontsize
        fig = plt.figure(figsize=self.figsize, layout="constrained")
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
                (stats["mouse_density_peak"], _GENERALIZED_COLOR, "Generalized"),
                (stats["mouse_density_peak_shrinkage"], _SHRINKAGE_COLOR, "Shrinkage"),
            )
        elif fit_sel == "shrinkage":
            peak_densities = ((stats["mouse_density_peak_shrinkage"], "black", "Generalized"),)
        else:  # "generalized"
            peak_densities = ((stats["mouse_density_peak"], "black", "Generalized"),)
        for density, color, label in peak_densities:
            ax1.plot(centers_peak, density.T, color=color, linewidth=0.8, alpha=0.3)
            ax1.plot(centers_peak, np.nanmean(density, axis=0), color=color, linewidth=2.0, label=label)
        ax1.axvline(x=2.0, color="k", linestyle=":", linewidth=0.8)
        ax1.set_xticks([0, 2, 4, 6, 8, 10])
        ax1.set_xlabel("Peak Exponent")
        ax1.set_ylabel("Density")
        if fit_sel == "both":
            ax1.legend(fontsize=self.fontsize * 0.8, frameon=False, loc="upper right")
        format_spines(
            ax1,
            x_pos=-0.02,
            y_pos=-0.02,
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
        ax2.set_ylabel("Test R²" if state["metric"] == "r2" else "Test Correlation")
        ylims = ax2.get_ylim()
        if state["metric"] == "cc":
            ymin = min(0, ylims[0])
            ymax = 1
        else:
            ymin = ylims[0]
            ymax = ylims[1]
            ybounds = (np.fix(ymin * 10) / 10, np.fix(ymax * 10) / 10)
        # Headroom above the data for the significance asterisks (spine still bounded at ymax).
        y_headroom = 0.05 * (ymax - ymin)
        ax2.set_xticks(x_models)
        ax2.set_xlim(-0.5, len(x_models) - 0.5)
        ax2.set_ylim(ymin, ymax + y_headroom)
        format_spines(
            ax2,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=(0, len(x_models) - 1),
            ybounds=(ymin, ymax),
            spines_visible=["bottom", "left"],
            xticks=x_models,
            # yticks=[ymin, 0.5, ymax],
        )
        ax2.set_xticklabels(perf_labels, rotation=45, ha="right")

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
                fontsize=self.fontsize * 1.3,
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
                ax3.set_xlabel("Mice")
                xticks = range(len(mice))

            ax3.set_yticks([0, 0.5, 1])
            ax3.set_ylim(0, 1)
            ax3.set_ylabel("Fraction Cells\nGeneralized > Gaussian")
            format_spines(
                ax3,
                x_pos=-0.02,
                y_pos=-0.02,
                xbounds=xbounds,
                ybounds=(0, 1),
                spines_visible=["bottom", "left"],
                yticks=[0, 0.5, 1],
            )
            ax3.set_xticks(xticks, labels=[])

        # --- gs[-1]: across-mouse spectrum decay statistic, source_key spectrum + eig fit spectra ---
        # Which statistic is set by ``last_axis``. "decay_exponent": each session's exponent is the
        # median five-point-derivative local exponent over its own peak-curvature-to-noise-floor
        # window (_second_derivative_window), then averaged by mouse -- the same estimator
        # spectrum_figure uses, under the single "source" AdaptiveAlphaConfig, drawn as one beeswarm
        # column per curve. "characteristic_dim"/"mse": the decay-law statistics of
        # PlacefieldSpectrumMSEViewer -- each curve fit under both _DECAY_MODELS and drawn across the
        # two model x-positions in the ``display`` layout.
        ax4 = fig.add_subplot(outer[0, -1])
        cfg = self._cfg_from_state(state, "source")
        fit_zone = state["fit_zone"]
        fixed_range = tuple(int(v) for v in state["fixed_range"])

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

            The two columns are the power-law and exponential fits (:data:`_DECAY_MODELS`' order);
            ``"characteristic_dim"`` is that model's parameter (``alpha`` / ``M``) and ``"mse"`` its
            log-space MSE.
            """
            fb_raw, fb_smooth = _fallback_rows(session_ids) if fit_zone == "adaptive" else (None, None)
            mse_cols, param_cols = [], []
            for model_key, _ in _DECAY_MODELS:
                mse_s, param_s = _decay_fit_per_session(raw, smooth, model_key, fit_zone, fixed_range, cfg.adaptive_buffer, fb_raw, fb_smooth)
                mse_cols.append(average_by_mouse(mse_s, mouse_names))
                param_cols.append(average_by_mouse(param_s, mouse_names))
            return {"characteristic_dim": np.stack(param_cols, axis=1), "mse": np.stack(mse_cols, axis=1)}

        # Assemble the curves: the source_key data spectrum, then the eig fit spectra -- "better"
        # composite (if requested), the selected generalized fit(s), then the gaussian control,
        # keeping _POP_EIG_KEYS' order. A single-fit selection draws that fit blue and labelled
        # "Generalized" (matching gs[0]/gs[1]).
        curves: list[tuple[tuple, str, str]] = []  # (per-session spectra tuple, color, label)
        if self.results_spectra is not None:
            spectra = self._spectrum_sessions(state, state["source_key"], cfg)
            curves.append((spectra, _POP_ALPHA_COLORS["source_key"], "Data"))
        eig_keys = ["eig_better"] if state["include_better"] else []
        if fit_sel == "both":
            eig_keys += ["eig_shrinkage", "eig_tilbury"]
        elif fit_sel == "shrinkage":
            eig_keys.append("eig_shrinkage")
        else:  # "generalized"
            eig_keys.append("eig_tilbury")
        eig_keys.append("eig_control")
        for key in eig_keys:
            if fit_sel != "both" and key in ("eig_shrinkage", "eig_tilbury"):
                color, label = _GENERALIZED_COLOR, "Generalized"
            else:
                color, label = _POP_ALPHA_COLORS[key], _POP_ALPHA_LABELS[key]
            curves.append((self._fit_spectrum_sessions(state, key, cfg), color, label))

        colors = [color for _, color, _ in curves]
        labels = [label for _, _, label in curves]
        last_axis = state["last_axis"]
        if last_axis == "decay_exponent":
            alpha_values = [_adaptive_alpha(*spectra) for spectra, _, _ in curves]
            _beeswarm_panel(ax4, alpha_values, colors, labels, self.fontsize, state["beewidth"])
            ax4.set_ylabel("Decay exponent")
            # The beeswarm columns are labelled on the x-axis, so no legend unless one is asked for --
            # its points carry no labels, hence the proxy handles.
            handles = [Line2D([], [], color=color, marker="o", markersize=3, linestyle="none") for color in colors]
            _apply_legend(ax4, state, self.fontsize, handles=handles, labels=labels)
        else:
            data_list = [_decay_stats(*spectra)[last_axis] for spectra, _, _ in curves]
            xtick_labels = [lbl for _, lbl in _DECAY_MODELS]
            # The characteristic dimension is non-negative: span [0, max] with ticks every 5 units.
            ybounds, yticks = _zero_to_max_ticks(data_list) if last_axis == "characteristic_dim" else (None, None)
            _decay_stat_panel(
                ax4,
                data_list,
                colors,
                labels,
                state["display"],
                state["beewidth"],
                self.fontsize,
                xtick_labels,
                ybounds=ybounds,
                yticks=yticks,
            )
            ax4.set_xlabel("Fit Type")
            ax4.set_ylabel("Dimensionality" if last_axis == "characteristic_dim" else "Log-space MSE")
            _apply_legend(ax4, state, self.fontsize, auto_loc="best")
        return fig


def placefield_example_fits(
    results: ResultsAggregator,
    registry: PopulationRegistry,
    example_session: str | None = None,
    n_rows: int = 2,
    n_cols: int = 3,
    r2_threshold: float = 0.5,
    improvement_threshold: float = 0.0,
    random_seed: int = 0,
    normalize: str = "sum",
    normalize_independent: bool = True,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (8.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Tilbury generalized-Gaussian placefield-fit figure: grid of example single-neuron fits.

    An ``n_rows x n_cols`` grid of example single-neuron fits (test placefield vs the generalized-
    Gaussian, plain-Gaussian and generalized-shrinkage curves) for ``example_session``. The fitted
    parameters and R^2 come from the stored results; only the held-out test curve is rebuilt on the
    fly (deterministic trial-average, no re-fit). Population summaries are in
    :func:`placefield_population`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated :class:`TilburyFitConfig` results.
    registry : PopulationRegistry
        Registry used to rebuild the example session's test curve (must match the one the results
        were built with).
    example_session : str or None
        session_uid to show. If None, the first session in ``results`` is used.
    n_rows, n_cols : int
        Grid shape of the example-fit panel (``n_rows * n_cols`` example neurons).
    r2_threshold : float
        Example neurons must have generalized test R^2 above this threshold.
    improvement_threshold : float
        Example neurons must also beat the plain Gaussian by at least this much test R^2 via
        *either* the generalized or the shrinkage fit (OR logic):
        ``max(r2_generalized, r2_shrinkage) - r2_gaussian > improvement_threshold``.
    random_seed : int
        Seed for the random example draw (reproducible). If fewer than ``n_rows * n_cols`` neurons
        clear both thresholds, the leftover panels are left empty.
    normalize : {"std", "sum", "max", "none"}
        Per-panel, the curve group is divided by this statistic (of the test-data curve unless
        ``normalize_independent``).
    normalize_independent : bool
        If True, divide each of the four curves (test data, generalized, gaussian, shrinkage) by its
        *own* statistic instead of the shared test-data one — a shape-only comparison that removes
        amplitude differences (they no longer overlay). Default True.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the fit's parameter-axis selections, keyed by raw ``param_axes`` name of
        ``results`` (``activity_parameters_name``).

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldExampleFitViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = PlacefieldExampleFitViewer(
        results,
        registry,
        n_rows=n_rows,
        n_cols=n_cols,
        fontsize=fontsize,
        figsize=figsize,
    )
    if example_session is not None:
        viewer.update_selection("example_session", value=example_session)
    viewer.update_integer("n_rows", value=n_rows)
    viewer.update_integer("n_cols", value=n_cols)
    viewer.update_float("r2_threshold", value=r2_threshold)
    viewer.update_float("improvement_threshold", value=improvement_threshold)
    viewer.update_integer("random_seed", value=random_seed)
    viewer.update_selection("normalize", value=normalize)
    viewer.update_boolean("normalize_independent", value=normalize_independent)
    for key, value in selections.items():
        if key not in results.param_axes:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(results.param_axes)}")
        viewer.update_selection(key, value=_tuple_label(value) if isinstance(value, tuple) else value)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


def placefield_fit_figure(
    results: ResultsAggregator,
    registry: PopulationRegistry,
    session_uids: list[str],
    neurons: list[int],
    n_rows: int = 2,
    n_cols: int = 3,
    legend_axis: int = 0,
    legend_position: str = "upper right",
    legend_x_offset: float = 0.0,
    legend_y_offset: float = 0.0,
    legend_handlelength: float = 2.0,
    legend_handletextpad: float = 0.8,
    legend_markerfirst: bool = True,
    normalize: str = "std",
    normalize_independent: bool = False,
    fit_model: str = "both",
    strict: bool = True,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (8.0, 4.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """Tilbury placefield-fit figure for a hand-picked, ordered list of ``(session, neuron)`` examples.

    Plots the first ``n_rows * n_cols`` neurons of ``(session_uids, neurons)`` — two parallel lists —
    into a shared-axes grid, in list order, each panel overlaying the held-out test placefield against
    the fitted generalized-Gaussian and plain-Gaussian curves. Fitted parameters and R^2 come from the
    stored :class:`TilburyFitConfig` results; the test curve is rebuilt on the fly (no re-fit). For the
    random-draw counterpart see :func:`placefield_example_fits`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated :class:`TilburyFitConfig` results.
    registry : PopulationRegistry
        Registry used to rebuild each session's test curve and to map an original ROI index to its
        fit row (must match the one the results were built with).
    session_uids : list of str
        session_uid per requested neuron.
    neurons : list of int
        The **original ROI index** per requested neuron — the index into the session's full ROI set
        (``np.where(session.idx_rois)[0][k]``), which is the stable identifier to record for a figure.
        Must be the same length as ``session_uids`` and aligned to it.
    n_rows, n_cols : int
        Grid shape; the first ``n_rows * n_cols`` list entries are plotted (extra panels hidden).
    legend_axis : int
        Flat row-major index of the panel that carries the legend, resolved to a grid position with
        ``divmod(legend_axis, n_cols)``. Must be in ``[0, n_rows * n_cols - 1]``. Default 0 (top-left).
    legend_position : str
        Matplotlib ``loc`` for the legend, one of :data:`_LEGEND_POSITIONS`. Default ``"upper right"``.
    legend_x_offset, legend_y_offset : float
        Shift of the legend's anchor box in axes-fraction units (1.0 = one full axes width/height),
        letting the legend sit outside its panel — e.g. ``legend_x_offset=0.4`` with
        ``legend_position="upper right"`` parks it in the gap to the right of the panel. The legend is
        excluded from the constrained layout, so offsetting it never resizes the panels. Default 0.
    legend_handlelength : float
        Length of the sample line drawn in each legend entry, in font-size units. Default 2.0.
    legend_handletextpad : float
        Gap between a legend entry's sample line and its label, in font-size units. Default 0.8.
    legend_markerfirst : bool
        If True (default), the sample line is drawn to the left of its label; False puts it on the right.
    normalize : {"std", "sum", "max", "none"}
        Per-panel, the test-data curve and both fits are divided by this statistic of the test-data
        curve, so the fits stay overlaid on the data while panels share a scale under ``sharey``.
    normalize_independent : bool
        If True, divide each of the three curves (test data, generalized, gaussian) by its *own*
        statistic instead of the shared test-data one — a shape-only comparison that removes amplitude
        differences between the fits and the data (they no longer overlay). Default False.
    fit_model : {"generalized", "shrinkage", "both"}
        Which generalized-family fit(s) to overlay alongside the always-shown plain-Gaussian control.
        A lone fit is drawn in the generalized (blue) color; ``"both"`` keeps generalized blue and
        shrinkage purple. Default ``"both"``.
    strict : bool
        If True (default), a requested ROI that never entered the pipeline or was dropped before
        fitting raises ``ValueError``. If False, that panel is left empty with a red status title.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the fit's parameter-axis selections, keyed by raw ``param_axes`` name of
        ``results`` (``activity_parameters_name``).

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldFitFigureViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = PlacefieldFitFigureViewer(
        results,
        registry,
        session_uids=session_uids,
        neurons=neurons,
        n_rows=n_rows,
        n_cols=n_cols,
        legend_axis=legend_axis,
        legend_position=legend_position,
        legend_x_offset=legend_x_offset,
        legend_y_offset=legend_y_offset,
        legend_handlelength=legend_handlelength,
        legend_handletextpad=legend_handletextpad,
        legend_markerfirst=legend_markerfirst,
        normalize=normalize,
        normalize_independent=normalize_independent,
        fit_model=fit_model,
        strict=strict,
        fontsize=fontsize,
        figsize=figsize,
    )
    viewer.update_integer("n_rows", value=n_rows)
    viewer.update_integer("n_cols", value=n_cols)
    # Seeding via update_* may not fire on_change, so set the legend bound alongside its value.
    viewer.update_integer("legend_axis", value=legend_axis, max=n_rows * n_cols - 1)
    viewer.update_selection("legend_position", value=legend_position)
    viewer.update_float("legend_x_offset", value=legend_x_offset)
    viewer.update_float("legend_y_offset", value=legend_y_offset)
    viewer.update_float("legend_handlelength", value=legend_handlelength)
    viewer.update_float("legend_handletextpad", value=legend_handletextpad)
    viewer.update_boolean("legend_markerfirst", value=legend_markerfirst)
    viewer.update_selection("normalize", value=normalize)
    viewer.update_boolean("normalize_independent", value=normalize_independent)
    viewer.update_selection("fit_model", value=fit_model)
    for key, value in selections.items():
        if key not in results.param_axes:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(results.param_axes)}")
        viewer.update_selection(key, value=_tuple_label(value) if isinstance(value, tuple) else value)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


def placefield_population(
    results: ResultsAggregator,
    results_spectra: ResultsAggregator | None = None,
    results_cvpca: ResultsAggregator | None = None,
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
    legend: dict | None = None,
    normalize: bool = True,
    source_cfg: AdaptiveAlphaConfig | None = None,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (8.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Tilbury generalized-Gaussian placefield-fit figure: population summaries (no examples).

    Four panels over every session in ``results``: gs[0] the per-mouse peak-exponent density for the
    generalized (blue) and shrinkage (purple) fits with the across-mouse mean (bold) and a reference
    at ``p = 2``; gs[1] the per-mouse median test R^2 for the shrinkage, generalized and Gaussian
    models (paired, in that order); gs[2] the fraction of neurons where the generalized fit beats the
    Gaussian, either pooled (one per-mouse beeswarm) or broken down by mouse; gs[-1] an across-mouse
    spectrum-decay statistic (``last_axis``) for the selected ``source_key`` spectrum and the
    ``eig_better``/``eig_shrinkage``/``eig_tilbury``/``eig_control`` fit spectra (colors
    orange/red/purple/blue/black) -- either the power-law exponent from the same fixed adaptive
    median-FPD fit as :func:`spectrum_figure`, or the characteristic parameter / log-space MSE of the
    power-law-vs-exponential comparison of :func:`placefield_spectrum_mse`. Example single-neuron fits
    are in :func:`placefield_example_fits`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated :class:`TilburyFitConfig` results (source of the ``eig_*`` spectra).
    results_spectra : ResultsAggregator or None
        Aggregated StimSpaceSpectra results, source of the ``source_key`` spectrum in gs[-1] and of
        the fixed ``ss_cvpca`` window-fallback source. If None, gs[-1] shows only the four eig fit
        spectra, and every key must locate its own adaptive window.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results; when given, ``reg_covariances_fixed`` is also a valid
        ``source_key``.
    num_bins : int
        Bin count for the per-session KDE of the peak-exponent density (gs[0]).
    fraction_view : {"pooled", "by_mouse"}
        gs[2] layout: one pooled per-mouse beeswarm, or one per-session beeswarm per mouse.
    beewidth : float
        Beeswarm point spread in x-axis units (gs[2] and gs[-1]).
    source_key : str
        Which spectrum drives the gs[-1] orange curve: ``ss_cv``/``ss_direct``/``ss_cvpca`` (from
        ``results_spectra``) or ``reg_covariances_fixed`` (from ``results_cvpca``).
    metric : {"r2", "cc"}
        Which performance metric drives the gs[1] and gs[2] panels: held-out test R^2 or Pearson correlation.
    generalized_fit : {"both", "generalized", "shrinkage"}
        Which generalized fit to show in gs[0]/gs[1]/gs[-1]. ``"both"`` plots the unregularized (blue)
        and shrinkage (purple) fits side by side; a single choice plots only that fit, drawn blue and
        labelled "Generalized".
    include_better : bool
        Include the per-neuron "better" composite eig spectrum in the gs[-1] decay panel.
    paired_test : {"ttest", "wilcoxon"}
        Two-sided paired test used for the gs[1] fit-vs-Gaussian asterisks (Bonferroni-corrected when
        ``generalized_fit="both"``): paired t-test or Wilcoxon signed-rank, on the per-mouse averages.
    last_axis : {"decay_exponent", "characteristic_dim", "mse"}
        Which spectrum-decay statistic gs[-1] shows. ``"decay_exponent"``: the adaptive median-FPD
        power-law exponent, one beeswarm column per curve. ``"characteristic_dim"``/``"mse"``: the two
        statistics of :func:`placefield_spectrum_mse` -- every curve fit as a power law (``n^-alpha``)
        and as an exponential (``exp(-n^2 / 2 M^2)``), drawn across those two x-positions, showing
        either the characteristic parameter (``alpha`` / ``M``) or the log-space MSE of each fit.
    fit_zone : {"adaptive", "fixed"}
        Fit window for the ``"characteristic_dim"``/``"mse"`` options (ignored by
        ``"decay_exponent"``, which always uses its own adaptive window): ``"adaptive"`` locates each
        session's own peak-curvature-to-noise-floor window (with the ``ss_cvpca`` fallback for
        non-cross-validated spectra); ``"fixed"`` fits every session over ``fixed_range``.
    fixed_range : tuple[int, int]
        ``(start, end)`` index window used when ``fit_zone="fixed"`` (default ``(10, 50)``).
    display : {"each", "errorPlot", "swarm"}
        gs[-1] layout for the ``"characteristic_dim"``/``"mse"`` options: ``"each"`` a faint per-mouse
        line across the two decay-model x-positions plus a bold across-mouse mean; ``"errorPlot"`` the
        across-mouse mean +/- SE band; ``"swarm"`` one beeswarm column per (decay model, curve) with a
        short horizontal mean line (spread set by ``beewidth``).
    legend : dict or None
        gs[-1] legend styling, as ``{knob: value}``. ``"loc"`` is a matplotlib placement,
        ``"auto"`` (the default -- ``"best"`` for the ``"characteristic_dim"``/``"mse"`` options, no
        legend for ``"decay_exponent"``, whose columns are labelled on the x-axis) or ``"none"`` (never
        draw one). The rest are :meth:`~matplotlib.axes.Axes.legend` kwargs at their matplotlib
        defaults -- ``ncols``, ``handlelength``, ``handletextpad``, ``labelspacing``, ``borderpad``,
        ``borderaxespad``, ``markerfirst``, ``frameon`` -- plus ``fontsize_scale``, which multiplies
        ``fontsize`` (default 0.8).
    normalize : bool
        If True, normalize each gs[-1] spectrum by its sum (per session) before smoothing.
    source_cfg : AdaptiveAlphaConfig or None
        Fixed adaptive-fit configuration (smoothing, five-point-derivative window, adaptive buffer,
        minimum window size) shared by every gs[-1] curve, whichever ``last_axis`` is shown. Defaults to
        ``ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]`` when None. The exponent is the median
        five-point-derivative local exponent over each session's own peak-curvature-to-noise-floor
        window, computed per session and averaged by mouse; sessions with fewer than
        ``source_cfg.minimum_window_size`` finite local-exponent values in that window are NaN. Keys
        whose row has no negative entry borrow the window from that session's ``ss_cvpca`` row (only
        available when ``results_spectra`` is given). See :func:`spectrum_figure`.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections, keyed by raw ``param_axes`` name of ``results``
        (``activity_parameters_name``) or of
        ``results_spectra``/``results_cvpca`` (e.g. ``smooth_widths``,
        ``reliability_fraction_active_thresholds``). See :func:`spectrum_figure`.

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldPopulationViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = PlacefieldPopulationViewer(
        results,
        results_spectra=results_spectra,
        results_cvpca=results_cvpca,
        source_cfg=source_cfg,
        num_bins=num_bins,
        fontsize=fontsize,
        figsize=figsize,
    )
    viewer.update_integer("num_bins", value=num_bins)
    viewer.update_selection("fraction_view", value=fraction_view)
    viewer.update_float("beewidth", value=beewidth)
    if results_spectra is not None:
        viewer.update_selection("source_key", value=source_key)

    valid_selections = set(results.param_axes)
    for agg in viewer._agg.values():
        if agg is None:
            continue
        valid_selections.update(agg.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))
    viewer.update_boolean("normalize", value=normalize)
    viewer.update_selection("metric", value=metric)
    viewer.update_selection("generalized_fit", value=generalized_fit)
    viewer.update_boolean("include_better", value=include_better)
    viewer.update_selection("paired_test", value=paired_test)
    viewer.update_selection("last_axis", value=last_axis)
    viewer.update_selection("fit_zone", value=fit_zone)
    viewer.update_integer_range("fixed_range", value=tuple(fixed_range))
    viewer.update_selection("display", value=display)
    if legend is not None:
        _update_legend_widgets(viewer, legend)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


class PlacefieldSpectrumMSEViewer(Viewer):
    """Decay-law goodness-of-fit for the Tilbury placefield eigenspectra.

    Two candidate decay laws are compared -- power law ``n^-alpha`` and exponential
    ``exp(-n^2 / 2 M^2)`` (:data:`_DECAY_MODELS`, the two xtick positions in each panel). Both are fit
    (log-space) to every spectrum: the ``source_key`` data spectrum (from
    ``results_spectra``/``results_cvpca``) plus the generalized-shrinkage, unregularized generalized
    and plain-Gaussian fit eigenspectra from the Tilbury-fit aggregator.

    - ax[0]: each fit's characteristic parameter -- the power-law exponent ``alpha`` at the power-law
      tick and the exponential characteristic dimension ``M`` at the exponential tick (y-axis
      "Characteristic Dim.").
    - ax[1]: the log-space MSE of each fit at the same two x-positions. A spectrum that follows one
      law reads low there and high at the other -- the point of the comparison, since Gaussian-tuned
      populations decay too fast to be genuine power laws.

    Every curve option is one colour; ``display="each"`` draws a faint per-mouse line across the two
    x-positions plus a bold across-mouse mean, ``display="errorPlot"`` draws the across-mouse mean
    +/- SE band, and ``display="swarm"`` drops the per-mouse connections for one beeswarm column per
    (decay model, curve) with a short horizontal mean line. The fit window is either fixed
    (``fit_zone="fixed"``, the ``fixed_range`` integer
    range, default ``(10, 50)``) or per-session adaptive (``fit_zone="adaptive"``, the same
    peak-curvature-to-noise-floor window and :class:`AdaptiveAlphaConfig` machinery as
    :class:`PlacefieldPopulationViewer`'s exponent panel). Spectra, log-space smoothing, param-axis
    widgets and mouse-averaging match that viewer exactly.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_spectra: ResultsAggregator | None = None,
        results_cvpca: ResultsAggregator | None = None,
        source_cfg: AdaptiveAlphaConfig | None = None,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (8.0, 3.0),
    ):
        self.results = results
        self.results_spectra = results_spectra
        self.results_cvpca = results_cvpca
        # Alias so the borrowed SpectrumFigureViewer methods (which fetch eig spectra from
        # ``results_fit``) resolve to this viewer's Tilbury-fit aggregator.
        self.results_fit = results
        self.source_cfg = source_cfg if source_cfg is not None else ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]
        # Reused by _spectrum_sessions/_sel_params (borrowed): the data spectrum column comes from
        # these, resolved via SOURCE_OF_KEY.
        self._agg = {"stimspace": results_spectra, "cvpca": results_cvpca}
        self.fontsize = fontsize
        self.figsize = figsize

        # Which generalized fit(s) to include as columns, mirroring PlacefieldPopulationViewer:
        # "both" -> shrinkage (purple) and unregularized generalized (blue); a single choice draws
        # just that fit, blue and labelled "Generalized".
        self.add_selection("generalized_fit", options=["both", "generalized", "shrinkage"], value="both")
        # Fit-window mode: per-session adaptive window, or the same fixed [start, end) for every session.
        self.add_selection("fit_zone", options=["adaptive", "fixed"], value="adaptive")
        self.add_integer_range("fixed_range", value=(10, 50), min=1, max=500)
        # How each curve is drawn across the two decay-model x-positions: "each" -> per-mouse lines +
        # bold mean; "errorPlot" -> across-mouse mean +/- SE band; "swarm" -> one beeswarm column per
        # (decay model, curve), no per-mouse connections (uses ``beewidth``).
        self.add_selection("display", options=["each", "errorPlot", "swarm"], value="each")
        self.add_float("beewidth", value=0.2, min=0.0, max=1.0, step=0.01)

        # Data-spectrum source (the first column); options mirror spectrum_figure.
        if results_spectra is not None:
            source_options = list(_STIMSPACE_KEYS) + (list(_CVPCA_KEYS) if results_cvpca is not None else [])
            self.add_selection("source_key", options=source_options, value="ss_cv")

        # One widget per shared param-axis name (same tuple-label scheme as SpectrumFigureViewer), so
        # both the data spectrum and the Tilbury-fit eig spectra can be sliced.
        merged_axes: dict[str, list] = {}
        for agg in list(self._agg.values()) + [results]:
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(opt for opt in options if opt not in existing)
        self._fit_axes = list(results.param_axes)
        self._tuple_labels = _add_param_axis_widgets(self, merged_axes)

        # Log-space smoothing + adaptive-window controls, shared by every column (same
        # "source"-prefixed AdaptiveAlphaConfig scheme as SpectrumFigureViewer).
        self.add_boolean("normalize", value=True)
        cfg = self.source_cfg
        self.add_selection("source_smooth_method", options=["none", "boxcar", "gaussian"], value=cfg.smooth_method)
        self.add_float("source_smooth_width", value=cfg.smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_integer("source_fpd_window_size", value=cfg.fpd_window_size, min=1, max=50)
        self.add_integer("source_adaptive_buffer", value=cfg.adaptive_buffer, min=0, max=50)
        self.add_integer("source_minimum_window_size", value=cfg.minimum_window_size, min=1, max=500)

    encode_param = PlacefieldSpectraViewer.encode_param
    _sel_params = PlacefieldSpectraViewer._sel_params
    # Adaptive-window / spectra machinery shared verbatim with the spectrum figure.
    _cfg_from_state = staticmethod(SpectrumFigureViewer._cfg_from_state)
    _spectrum_sessions = SpectrumFigureViewer._spectrum_sessions
    _fit_spectrum_raw_sessions = SpectrumFigureViewer._fit_spectrum_raw_sessions
    _fit_spectrum_sessions = SpectrumFigureViewer._fit_spectrum_sessions
    _fit_sel_params = SpectrumFigureViewer._fit_sel_params

    def _columns(self, state: dict, cfg: AdaptiveAlphaConfig) -> list[tuple]:
        """Assemble the spectrum columns: ``(raw, smooth, mouse_names, session_ids, color, label)``.

        Order matches the request -- data, then the selected generalized fit(s), then the Gaussian
        control (see :data:`_POP_ALPHA_COLORS` / :data:`_POP_ALPHA_LABELS`).
        """
        fit_sel = state["generalized_fit"]
        columns: list[tuple] = []
        if self.results_spectra is not None:
            raw, smooth, mouse_names, session_ids = self._spectrum_sessions(state, state["source_key"], cfg)
            columns.append((raw, smooth, mouse_names, session_ids, _POP_ALPHA_COLORS["source_key"], "Data"))
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
                color, label = _POP_ALPHA_COLORS[key], _POP_ALPHA_LABELS[key]
            columns.append((raw, smooth, mouse_names, session_ids, color, label))
        return columns

    def plot(self, state: dict):
        cfg = self._cfg_from_state(state, "source")
        fit_zone = state["fit_zone"]
        fixed_range = tuple(int(v) for v in state["fixed_range"])
        display = state["display"]
        columns = self._columns(state, cfg)

        # Fixed fallback window source (ss_cvpca) for non-cross-validated fit spectra in adaptive
        # mode; harmless when a column already is cross-validated (it locates its own window then).
        cvpca = self._spectrum_sessions(state, "ss_cvpca", cfg) if self.results_spectra is not None else None

        # For every column, compute the per-mouse (mse, param) at both decay models as (n_mice, 2)
        # matrices (column 0 = power law, column 1 = exponential), matching _DECAY_MODELS' order.
        model_keys = [m for m, _ in _DECAY_MODELS]
        xtick_labels = [lbl for _, lbl in _DECAY_MODELS]
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

        beewidth = state["beewidth"]
        plt.rcParams["font.size"] = self.fontsize
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, layout="constrained")

        # ax[0]: each fit's characteristic parameter -- power-law exponent (alpha) at the power-law
        # tick, exponential characteristic dimension (M) at the exponential tick.
        _decay_stat_panel(axes[0], param_mats, colors, labels, display, beewidth, self.fontsize, xtick_labels)
        axes[0].set_ylabel("Characteristic Dim.")
        axes[0].legend(fontsize=self.fontsize * 0.8, frameon=False)

        # ax[1]: log-space MSE of each fit, same format.
        _decay_stat_panel(axes[1], mse_mats, colors, labels, display, beewidth, self.fontsize, xtick_labels)
        axes[1].set_ylabel("Log-space MSE")
        return fig


def placefield_spectrum_mse(
    results: ResultsAggregator,
    results_spectra: ResultsAggregator | None = None,
    results_cvpca: ResultsAggregator | None = None,
    generalized_fit: str = "both",
    fit_zone: str = "adaptive",
    fixed_range: tuple[int, int] = (10, 50),
    display: str = "each",
    beewidth: float = 0.2,
    source_key: str = "ss_cv",
    normalize: bool = True,
    source_cfg: AdaptiveAlphaConfig | None = None,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (7.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """Decay-law goodness-of-fit for the Tilbury placefield eigenspectra.

    Two panels comparing a power-law (``n^-alpha``) against an exponential (``exp(-n^2 / 2 M^2)``)
    fit of every spectrum: the ``source_key`` data spectrum plus the
    ``eig_shrinkage``/``eig_tilbury``/``eig_control`` Tilbury-fit eigenspectra (colors
    purple/blue/black, data orange). ax[0] shows each fit's characteristic parameter (power-law
    exponent ``alpha`` at x=0, exponential dimension ``M`` at x=1); ax[1] shows the log-space MSE of
    each fit at the same two x-positions. Reading MSE across the two ticks shows which decay law each
    spectrum follows -- the comparison Tilbury et al. use to distinguish high-dimensional power-law
    codes from the fast-decaying Gaussian-tuned code.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated :class:`TilburyFitConfig` results (source of the ``eig_*`` spectra).
    results_spectra : ResultsAggregator or None
        Aggregated StimSpaceSpectra results, source of the ``source_key`` data spectrum and of the
        fixed ``ss_cvpca`` adaptive-window fallback. If None, the data column is dropped and every
        column must locate its own adaptive window.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results; when given, ``reg_covariances_fixed`` is also a valid
        ``source_key``.
    generalized_fit : {"both", "generalized", "shrinkage"}
        Which generalized fit column(s) to show alongside the data and Gaussian columns. ``"both"``
        shows shrinkage (purple) and unregularized generalized (blue); a single choice shows only
        that fit, drawn blue and labelled "Generalized".
    fit_zone : {"adaptive", "fixed"}
        ``"adaptive"`` locates each session's own peak-curvature-to-noise-floor window (the
        :class:`AdaptiveAlphaConfig` machinery, with the ``ss_cvpca`` fallback for non-cross-validated
        spectra); ``"fixed"`` fits every session over ``fixed_range``.
    fixed_range : tuple[int, int]
        ``(start, end)`` index window used when ``fit_zone="fixed"`` (default ``(10, 50)``).
    display : {"each", "errorPlot", "swarm"}
        ``"each"`` draws a faint per-mouse line across the two decay-model x-positions plus a bold
        across-mouse mean; ``"errorPlot"`` draws the across-mouse mean +/- SE band; ``"swarm"`` drops
        the per-mouse connections and draws one beeswarm column per (decay model, curve) with a short
        horizontal mean line (spread set by ``beewidth``).
    beewidth : float
        Beeswarm point spread in x-axis units, used only when ``display="swarm"``.
    source_key : str
        Which spectrum is the data column: ``ss_cv``/``ss_direct``/``ss_cvpca`` (from
        ``results_spectra``) or ``reg_covariances_fixed`` (from ``results_cvpca``).
    normalize : bool
        Normalize each spectrum by its sum (per session) before smoothing. Does not affect the MSE
        (a constant rescale only shifts the log-space intercept), kept for parity with the other
        spectrum figures.
    source_cfg : AdaptiveAlphaConfig or None
        Log-space smoothing + adaptive-window configuration shared by every column. Defaults to
        ``ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]`` when None.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections, keyed by raw ``param_axes`` name of ``results``
        or of ``results_spectra``/``results_cvpca`` (e.g. ``activity_parameters_name``,
        ``smooth_widths``, ``reliability_fraction_active_thresholds``).

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldSpectrumMSEViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = PlacefieldSpectrumMSEViewer(
        results,
        results_spectra=results_spectra,
        results_cvpca=results_cvpca,
        source_cfg=source_cfg,
        fontsize=fontsize,
        figsize=figsize,
    )
    viewer.update_selection("generalized_fit", value=generalized_fit)
    viewer.update_selection("fit_zone", value=fit_zone)
    viewer.update_integer_range("fixed_range", value=tuple(fixed_range))
    viewer.update_selection("display", value=display)
    viewer.update_float("beewidth", value=beewidth)
    if results_spectra is not None:
        viewer.update_selection("source_key", value=source_key)

    valid_selections = set(results.param_axes)
    for agg in viewer._agg.values():
        if agg is None:
            continue
        valid_selections.update(agg.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))
    viewer.update_boolean("normalize", value=normalize)

    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig
