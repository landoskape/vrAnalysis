"""Cross-panel numeric helpers: spectrum smoothing, power-law/decay fits, and participation ratio.

Pure functions only -- no Syd widgets, no plotting. Every figure4 panel that estimates a
power-law exponent, smooths a spectrum, or measures a participation ratio builds on these.
"""

import numpy as np

from dimilibi.helpers import fit_powerlaw_decay, fit_powerlaw_derivatives


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
    :data:`_param_axes._FIT_KEY_PARAM_KEYS`); a neuron counts only if finite in all of them, so
    sessions without at least two jointly valid fitted neurons still produce NaNs.
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


def _clip_at_first_negative(spec: np.ndarray) -> np.ndarray:
    """Replace each spectrum's first negative entry and all later ranks with NaN."""
    spec = np.array(spec, dtype=float, copy=True)
    rows = spec.reshape(-1, spec.shape[-1])
    for row in rows:
        row[_first_negative_index(row) :] = np.nan
    return spec


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
DECAY_MODELS: tuple[tuple[str, str], ...] = (
    ("power", r"$n^{-\alpha}$"),
    ("exp2", r"$e^{-n^2/2M^2}$"),
)


def _decay_feature(model: str, n: np.ndarray) -> np.ndarray:
    """Log-space regressor for one decay model at 1-based ranks ``n``."""
    if model == "power":
        return np.log(n)
    if model == "exp2":
        return n**2
    raise ValueError(f"Unknown decay model {model!r}. Available: {[m for m, _ in DECAY_MODELS]}")


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


def _signed_participation_ratio(spec: np.ndarray) -> np.ndarray:
    """Signed participation ratio ``(sum lambda)^2 / sum(lambda^2)`` over spectrum rank.

    Spectrum rank is the final axis, so both ``(mice, dims)`` and
    ``(sessions, environment slots, dims)`` inputs are supported. "Signed" means negative spectrum
    entries are used as-is (no clipping); the result is scale-invariant.
    """
    s1 = np.nansum(spec, axis=-1)
    s2 = np.nansum(spec**2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s2 > 0, s1**2 / s2, np.nan)


def _truncated_participation_ratio(spec: np.ndarray, k: int | np.ndarray) -> np.ndarray:
    """Signed participation ratio after dropping the first ``k`` spectrum ranks.

    ``spec`` may have any leading shape and spectrum rank on its final axis. ``k`` is either one
    scalar or an array matching that leading shape. Non-finite adaptive ``k`` values yield NaN;
    finite values are rounded to the nearest integer and clamped to the available rank range.
    """
    spec = np.asarray(spec, dtype=float)
    leading_shape = spec.shape[:-1]
    rows = spec.reshape(-1, spec.shape[-1])
    k_values = np.broadcast_to(np.asarray(k, dtype=float), leading_shape).reshape(-1)
    out = np.full(rows.shape[0], np.nan)
    for i, (row, row_k) in enumerate(zip(rows, k_values)):
        if not np.isfinite(row_k):
            continue
        start = int(np.clip(np.rint(row_k), 0, row.size))
        out[i] = _signed_participation_ratio(row[None, start:])[0]
    return out.reshape(leading_shape)
