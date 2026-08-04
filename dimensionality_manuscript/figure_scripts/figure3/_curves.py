"""Cross-panel curve maths for figure 3.

Everything here is shared by two or more panels and knows nothing about drawing: rank-axis
smoothing, the two cross-spectrum overlap metrics, and the reshaping helpers that turn a flat
per-session array into per-mouse or per-session-number stacks.
"""

import numpy as np
from matplotlib import pyplot as plt

from dimensionality_manuscript.pipeline import ResultsAggregator

# Smoothing options offered by the cross-spectrum panels; see :func:`smooth_fraction`.
SMOOTH_KINDS = ["none", "boxcar", "gaussian", "median"]

# Metrics summarizing how much full-activity variance the placefield span misses; see
# :func:`distribution_metric`.
DISTRIBUTION_METRICS = ["gini", "weighted_missing", "missing_structure"]

# Colormap the session-ordered curve groups and their colorbar insets are drawn with.
SESSION_CMAP = "coolwarm"


def gini(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute the equality measure (1 - Gini coefficient).

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, default=-1
        Axis along which to compute the Gini coefficient.

    Returns
    -------
    np.ndarray
        1 - Gini coefficient, measuring equality rather than inequality.
    """
    n = x.shape[axis]
    x = np.sort(x, axis=axis)  # Sort values
    weights = np.moveaxis((1 + np.arange(n))[(...,) + (None,) * axis], 0, axis)
    gini_coefficient = 2 * np.sum(weights * x, axis=axis) / n / (np.sum(x, axis=axis) + 1e-10) - (n + 1) / n
    return 1 - gini_coefficient


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


def _median_smooth(curves: np.ndarray, width: float) -> np.ndarray:
    """NaN-aware centered running-median of ``(curves, dims)`` along the dim axis.

    ``width`` is the window length in dim units. Unlike the convolution kernels, the median
    filter is edge-preserving: it removes spikes/outliers without rounding off kinks. Windows are
    clipped at the ends and reduced with ``nanmedian`` (all-NaN windows stay NaN).
    """
    window = max(1, int(round(width)))
    half = window // 2
    n_dims = curves.shape[1]
    out = np.full_like(curves, np.nan, dtype=float)
    with np.errstate(invalid="ignore"):
        for i in range(n_dims):
            lo, hi = max(0, i - half), min(n_dims, i + half + 1)
            block = curves[:, lo:hi]
            valid = np.isfinite(block).any(axis=1)
            out[valid, i] = np.nanmedian(block[valid], axis=1)
    return out


def smooth_fraction(curves: np.ndarray, kind: str, width: float) -> np.ndarray:
    """Linear NaN-aware smoothing of ``(curves, dims)`` fraction curves along the dim axis.

    Unlike the log-space spectrum smoothing in ``figure4`` (slope-preserving for power laws),
    fraction curves live in ``[0, 1]``, so smoothing is done in linear space. ``"boxcar"`` and
    ``"gaussian"`` use a NaN-aware weighted convolution (NaN entries excluded, kernel renormalized
    per output point, which also handles edges); ``"median"`` uses an edge/kink-preserving running
    median. ``kind == "none"`` or ``width <= 0`` returns ``curves`` unchanged.
    """
    if kind == "none" or width <= 0 or curves.shape[0] == 0:
        return curves
    curves = np.asarray(curves, dtype=float)
    if kind == "median":
        return _median_smooth(curves, width)
    kernel = _smooth_kernel(kind, width)
    mask = np.isfinite(curves)
    filled = np.where(mask, curves, 0.0)
    num = np.stack([np.convolve(row, kernel, mode="same") for row in filled])
    den = np.stack([np.convolve(row, kernel, mode="same") for row in mask.astype(float)])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def weighted_fraction(cross: np.ndarray, variance_activity: np.ndarray) -> np.ndarray:
    """Variance-weighted fraction of each full PC's variance recovered from the PF subspace.

    The unweighted metric ``energy_on_full[i] = ||P u_i||^2`` (with ``P = V Vᵀ`` the projector
    onto the placefield span) is purely geometric: it counts every placefield direction equally,
    so a near-full-rank placefield basis saturates it even when the overlap lands on directions
    carrying little neural variance. This weights the overlap by full-activity variance instead.

    For each full PC ``i`` it approximates ``Var(X P u_i) / λ_i`` under the PCA covariance model
    ``C = Σ_k λ_k u_k u_kᵀ``:

        w_i = (Σ_k λ_k ⟨r_i, r_k⟩²) / λ_i,   r_i = cross[i, :],   λ = variance_activity

    where ``⟨r_i, r_k⟩ = (cross crossᵀ)_{ik}``. Equal to the unweighted metric when ``C ∝ I``;
    departs from it by down-weighting overlap onto low-variance full PCs.

    Parameters
    ----------
    cross : np.ndarray
        Full-vs-placefield cross matrices, shape ``(sessions, n_full, n_pf)``. NaN padding
        (ragged dims) is treated as zero overlap.
    variance_activity : np.ndarray
        Full-activity variance per full PC (the ``λ_k``), shape ``(sessions, n_full)``.

    Returns
    -------
    np.ndarray
        Weighted fraction per full PC, shape ``(sessions, F)`` with
        ``F = min(n_full, len(variance_activity))``. Padded full dims are NaN.
    """
    cross = np.nan_to_num(np.asarray(cross, dtype=float), nan=0.0)
    lam = np.asarray(variance_activity, dtype=float)
    num_full = min(cross.shape[1], lam.shape[1])
    cross = cross[:, :num_full, :]
    lam = lam[:, :num_full]
    lam_weight = np.nan_to_num(lam, nan=0.0)

    weighted = np.empty((cross.shape[0], num_full), dtype=float)
    for s in range(cross.shape[0]):
        overlap = cross[s] @ cross[s].T  # (F, F): ⟨r_i, r_k⟩
        weighted[s] = (overlap * overlap) @ lam_weight[s]  # Σ_k λ_k ⟨r_i, r_k⟩²
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = weighted / lam  # λ_i == 0 (padded/degenerate dims) -> inf/NaN
    weighted[~np.isfinite(weighted)] = np.nan  # drop padded and zero-variance full dims
    return weighted


# ======================================================================================
# Cross-spectrum overlap curves and their per-session summaries
# ======================================================================================


def energy_on_full(cross: np.ndarray, smooth_kind: str, smooth_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed ``||P u_i||^2`` per full PC, plus the mask of full dims a session actually has.

    Returns ``(curves, valid_full_dims)``, both ``(sessions, n_full)``. Padded full dims (a
    session with fewer dims than the widest one) are NaN in ``curves`` and False in the mask.
    """
    energy = cross**2
    valid_full_dims = np.isfinite(cross).any(axis=2)
    curves = np.where(valid_full_dims, np.nansum(energy, axis=2), np.nan)
    return smooth_fraction(curves, smooth_kind, smooth_width), valid_full_dims


def kink_positions(curves: np.ndarray, threshold: float) -> np.ndarray:
    """First full dimension at which each session's curve drops to ``threshold`` of its max."""
    max_energy = np.nanmax(curves, axis=1)
    condition = curves <= threshold * max_energy[:, None]
    return np.where(condition.any(axis=1), condition.argmax(axis=1), np.nan)


def distribution_metric(
    metric: str,
    curves: np.ndarray,
    valid_full_dims: np.ndarray,
    variance_activity: np.ndarray,
    *,
    gini_equality: bool = False,
) -> np.ndarray:
    """One number per session summarizing how much full-space structure the PF span misses.

    ``"missing_structure"`` is the mean uncaptured fraction over a session's valid full dims;
    ``"weighted_missing"`` weights that same uncaptured fraction by each full dim's activity
    variance; ``"gini"`` measures how unevenly the captured energy is spread over dims.

    ``gini_equality`` picks which way round the Gini option is reported, because the two
    cross-space panels disagree: the per-mouse panel plots :func:`gini`'s equality measure
    (``1 - G``) while the summary panel plots its complement, the Gini coefficient itself. Both
    conventions are preserved rather than unified, since each panel's y range was tuned to its own.
    """
    if metric == "gini":
        equality = gini(curves, axis=1)
        return equality if gini_equality else 1 - equality
    if metric == "weighted_missing":
        numerator = np.where(valid_full_dims, (1 - curves) * variance_activity, np.nan)
        denominator = np.where(valid_full_dims, variance_activity, np.nan)
        return np.nansum(numerator, axis=1) / np.nansum(denominator, axis=1)
    if metric == "missing_structure":
        return np.nanmean(np.where(valid_full_dims, 1 - curves, np.nan), axis=1)
    raise ValueError(f"Unknown distribution_metric {metric!r}. Options: {DISTRIBUTION_METRICS}")


# ======================================================================================
# Reshaping: sessions -> per-mouse / per-session-number stacks
# ======================================================================================


def supported_xmax(xvals: np.ndarray, data: np.ndarray, min_support: int = 1) -> float:
    """Rightmost ``xvals`` entry whose column has at least ``min_support`` finite series.

    This is the x extent :func:`~dimensionality_manuscript.figure_scripts.panels.render_curve_group`
    actually covers with its mean/band, and so the range the offset x spine should span.
    Falls back to ``xvals[0]`` when no column qualifies.
    """
    valid_count = np.sum(np.isfinite(data), axis=0)
    supported = np.where(valid_count >= min_support)[0]
    return float(xvals[supported[-1]]) if supported.size else float(xvals[0])


def stack_sessions_by_mouse(values: np.ndarray, mouse_names: np.ndarray, unique_mice) -> np.ndarray:
    """``(n_mice, max_n_sessions)`` stack of one per-session scalar, NaN-padding short histories.

    Session order within a mouse is preserved as given (the aggregator's own order).
    """
    mouse_values = [values[mouse_names == mouse] for mouse in unique_mice]
    organized = np.full((len(mouse_values), max(map(len, mouse_values))), np.nan)
    for i, values_for_mouse in enumerate(mouse_values):
        organized[i, : len(values_for_mouse)] = values_for_mouse
    return organized


def by_session_groups(curves: np.ndarray, mouse_names: np.ndarray, unique_mice, skip_sessions: int):
    """Group per-session curves by *within-mouse session number* for the population curve panels.

    Each mouse's curves are laid out in session order and NaN-padded to the longest history, then
    session numbers backed by at most one mouse are dropped -- a "mean" over one mouse is that
    mouse. ``skip_sessions`` thins which of the kept session numbers are actually drawn (the first
    and last are always kept), so a dense session count doesn't overplot the panel; the colors
    still span the full kept range, so the gaps don't compress the colormap.

    Returns
    -------
    tuple
        ``(by_session, kept, colors, show_idx)``: the ``(n_mice, n_session_numbers, n_dims)``
        stack, the kept session numbers, one color per kept session number, and the indices
        *into* ``kept`` that should be drawn.
    """
    mouse_curves = [curves[mouse_names == mouse] for mouse in unique_mice]
    max_n_sessions = max(map(len, mouse_curves), default=0)
    by_session = np.full((len(mouse_curves), max_n_sessions, curves.shape[1]), np.nan)
    for i, mouse_stack in enumerate(mouse_curves):
        by_session[i, : len(mouse_stack)] = mouse_stack

    support = np.array([np.sum(np.isfinite(by_session[:, j, :]).any(axis=1)) for j in range(max_n_sessions)])
    kept = np.where(support > 1)[0]
    colors = plt.get_cmap(SESSION_CMAP)(np.linspace(0, 1, max(len(kept), 1)))

    n_kept = len(kept)
    step = skip_sessions + 1
    if n_kept <= 2 or step <= 1:
        show_idx = np.arange(n_kept)
    else:
        n_points = max(2, int(round((n_kept - 1) / step)) + 1)
        show_idx = np.unique(np.round(np.linspace(0, n_kept - 1, n_points)).astype(int))
    return by_session, kept, colors, show_idx


def chronological_mouse_sessions(results: ResultsAggregator, mouse: str, exclude_bad_envs: bool = True) -> np.ndarray:
    """Session indices for one mouse, sorted chronologically by date.

    When ``exclude_bad_envs``, sessions carrying the invalid environment sentinel (``-1``) are
    dropped first, matching the whole-session ("all") figure's original filtering.
    """
    idx_mouse = np.where(results.mouse_names == mouse)[0]
    if exclude_bad_envs:
        bad = np.array([-1 in results.sessions[i].environments for i in idx_mouse], dtype=bool)
        idx_mouse = idx_mouse[~bad]
    dates = np.array([results.sessions[i].date for i in idx_mouse])
    return idx_mouse[np.argsort(dates)]


def pad_stack_by_mouse(curves: dict[str, np.ndarray]) -> np.ndarray:
    """Stack ragged per-mouse 1D curves into a NaN-padded ``(n_mice, max_len)`` array."""
    max_len = max((len(v) for v in curves.values()), default=0)
    stack = np.full((len(curves), max_len), np.nan)
    for i, values in enumerate(curves.values()):
        stack[i, : len(values)] = values
    return stack


def support_length(pad_stack: np.ndarray, min_support: int = 1) -> int:
    """Number of leading columns where more than ``min_support`` mice have finite data."""
    support = np.sum(np.isfinite(pad_stack), axis=0)
    valid = np.where(support > min_support)[0]
    return int(valid[-1] + 1) if valid.size else 0


def mean_with_min_support(pad_stack: np.ndarray, min_support: int = 1) -> np.ndarray:
    """Nanmean across mice (axis 0), truncated where at most ``min_support`` mice have data."""
    length = support_length(pad_stack, min_support)
    return np.nanmean(pad_stack[:, :length], axis=0)
