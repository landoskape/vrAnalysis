import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from syd import make_viewer, Viewer
from tqdm import tqdm

from vrAnalysis.database import get_database
from vrAnalysis.helpers import Timer
from vrAnalysis.helpers.plotting import format_spines, beeswarm, errorPlot, save_figure, add_scaled_limits
from vrAnalysis.processors.placefields import get_placefield
from dimilibi import PCA, SVCA
from dimensionality_manuscript.registry import PopulationRegistry, get_subspace, SubspaceName, RegistryPaths
from dimensionality_manuscript.subspace_analysis.base import Subspace
from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldHyperparameters
from dimensionality_manuscript.simulations import sqrtm_spd
from dimensionality_manuscript import SubspaceConfig, StimSpaceConfig, StimSpaceSpectraConfig
from dimensionality_manuscript import ResultsAggregator, ResultsStore, get_data_config
from dimensionality_manuscript.scripts.status import status
from dimensionality_manuscript.subspace_analysis.stimspace import StimSpaceSubspace
from dimensionality_manuscript import average_by_mouse
from ..env_order import MAX_ENV_SLOTS


def _gini(x: np.ndarray, axis: int = -1) -> np.ndarray:
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


def _smooth_fraction(curves: np.ndarray, kind: str, width: float) -> np.ndarray:
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


def _weighted_fraction(cross: np.ndarray, variance_activity: np.ndarray) -> np.ndarray:
    """Variance-weighted fraction of each full PC's variance recovered from the PF subspace.

    The unweighted right-panel metric ``energy_on_full[i] = ||P u_i||^2`` (with
    ``P = V Vᵀ`` the projector onto the placefield span) is purely geometric: it
    counts every placefield direction equally, so a near-full-rank placefield basis
    saturates it even when the overlap lands on directions carrying little neural
    variance. This weights the overlap by full-activity variance instead.

    For each full PC ``i`` it approximates ``Var(X P u_i) / λ_i`` under the PCA
    covariance model ``C = Σ_k λ_k u_k u_kᵀ``:

        w_i = (Σ_k λ_k ⟨r_i, r_k⟩²) / λ_i,   r_i = cross[i, :],   λ = variance_activity

    where ``⟨r_i, r_k⟩ = (cross crossᵀ)_{ik}``. Equal to the unweighted metric when
    ``C ∝ I``; departs from it by down-weighting overlap onto low-variance full PCs.

    Parameters
    ----------
    cross : np.ndarray
        Full-vs-placefield cross matrices, shape ``(sessions, n_full, n_pf)``. NaN
        padding (ragged dims) is treated as zero overlap.
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


def _render_curve_group(
    ax,
    xvals: np.ndarray,
    data: np.ndarray,
    color,
    plot_style: str,
    label: str | None = None,
    hide_error: bool = False,
    min_support: int = 1,
    line_alpha: float = 0.3,
    linewidth: float = 2.0,
) -> float:
    """Render one group of ``(n_series, n_x)`` curves as either individual+mean lines or an errorPlot band.

    ``plot_style == "each"`` draws every row as a thin line plus a solid mean line.
    ``plot_style == "errorPlot"`` draws a mean +/- std band via :func:`errorPlot`, or just the
    mean line when ``hide_error`` is True. In both styles, x-columns with fewer than
    ``min_support`` finite rows are excluded from the mean/band (but not from the thin lines).

    Returns
    -------
    float
        The largest ``xvals`` entry whose column met ``min_support`` (``xvals[0]`` if none did),
        i.e. the rightmost x actually covered by the rendered mean/band.
    """
    valid_count = np.sum(np.isfinite(data), axis=0)
    value_sum = np.nansum(data, axis=0)
    mean_curve = np.divide(value_sum, valid_count, out=np.full_like(value_sum, np.nan, dtype=float), where=valid_count > 0)
    mean_curve[valid_count < min_support] = np.nan

    if plot_style == "each":
        ax.plot(xvals, data.T, color=(color, line_alpha), linewidth=0.5)
        ax.plot(xvals, mean_curve, color=color, linewidth=linewidth, label=label)
    elif plot_style == "errorPlot":
        if hide_error:
            ax.plot(xvals, mean_curve, color=color, linewidth=linewidth, label=label)
        else:
            masked = np.where(valid_count[None, :] >= min_support, data, np.nan)
            errorPlot(xvals, masked, axis=0, ax=ax, color=color, linewidth=linewidth, alpha=0.2, label=label)
    else:
        raise ValueError(f"Unknown plot_style {plot_style!r}. Options: ['each', 'errorPlot']")

    supported = np.where(valid_count >= min_support)[0]
    return float(xvals[supported[-1]]) if supported.size else float(xvals[0])


# Fixed color lookup for the ax[1] metric-explainer decorations (arrows/patch), kept separate
# from the CONDITION_COLORS/env-slot palettes used elsewhere in the manuscript. Edit here to
# restyle every decoration at once.
_CROSS_METRIC_COLORS = {
    "max_captured": "black",
    "kink": "darkviolet",
    "missing_structure": "dimgrey",
}


class SubspaceCrossspaceViewer(Viewer):
    """Interactive cross-spectrum energy viewer over aggregated subspace results."""

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (12.0, 3.0)):
        self.results = results
        self.figsize = figsize
        for key, value in results.param_axes.items():
            self.add_selection(key, options=value)

        preferred_state = {
            "smooth_width": None,
            "activity_parameters_name": "default",
        }
        for key, value in preferred_state.items():
            self.update_selection(key, value=value)

        self.add_integer("idx_cross", value=0, min=0, max=len(results.sessions) - 1)
        self.add_integer("num_cross_show", value=25, min=1, max=100)
        self.add_boolean("plot_energy", value=True)
        self.add_boolean("weighted", value=False)
        self.add_selection("curve_mode", options=["average", "by_session"], value="average")
        self.add_selection("plot_style", options=["each", "errorPlot"], value="each")
        self.add_boolean("hide_error", value=False)
        self.add_integer("skip_sessions", value=0, min=0, max=len(results.sessions))
        self.add_selection("curve_smooth_kind", options=["none", "boxcar", "gaussian", "median"], value="none")
        self.add_float("curve_smooth_width", value=3.0, min=0.0, max=50.0, step=0.5)
        self.add_float("kink_threshold", value=0.95, min=0.0, max=1.0, step=0.001)
        self.add_selection("distribution_metric", options=["gini", "weighted_missing", "missing_structure"], value="weighted_missing")

        # ax[1] metric-explainer decorations (colors fixed in _CROSS_METRIC_COLORS).
        self.add_boolean("show_decorations", value=True)
        self.add_boolean("show_marker_labels", value=True)
        self.add_float("arrow_linewidth", value=1.5, min=0.1, max=10.0, step=0.1)
        self.add_float("arrow_head_size", value=0.4, min=0.05, max=2.0, step=0.05)
        self.add_float("max_arrow_x", value=1.0, min=1.0, max=1000.0, step=0.1)
        self.add_float("max_arrow_y_start", value=0.0, min=0.0, max=1.5, step=0.01)
        self.add_float("max_arrow_y_end", value=0.9, min=0.0, max=1.5, step=0.01)
        self.add_float("kink_arrow_x_start", value=1.0, min=1.0, max=1000.0, step=0.1)
        self.add_float("kink_arrow_x_end", value=100.0, min=1.0, max=1000.0, step=0.1)
        self.add_float("kink_arrow_y", value=0.9, min=0.0, max=1.5, step=0.01)
        self.add_float("missing_structure_x_offset", value=10.0, min=0.0, max=1000.0, step=0.1)
        self.add_float("missing_structure_y_offset", value=0.05, min=0.0, max=1.0, step=0.01)
        self.add_float("missing_structure_alpha", value=0.15, min=0.0, max=1.0, step=0.01)
        self.add_float("fontsize", value=9.0, min=4.0, max=24.0, step=0.5)

    def plot(self, state: dict):
        _sel_state = {k: v for k, v in state.items() if k in self.results.param_axes}
        _out = self.results.sel(**_sel_state)
        cross = _out["cross"]
        energy = cross**2
        variance_activity = _out["variance_activity"][:, : cross.shape[1]]

        # Energy of PF dimensions on full space
        energy_on_full = np.nansum(energy, axis=2)
        valid_full_dims = np.isfinite(cross).any(axis=2)
        energy_on_full = np.where(valid_full_dims, energy_on_full, np.nan)
        energy_on_full = _smooth_fraction(energy_on_full, state["curve_smooth_kind"], state["curve_smooth_width"])
        energy_on_diagonal = np.cumsum(np.diagonal(energy, axis1=1, axis2=2), axis=1)
        energy_expanding_dims = np.diagonal(np.cumsum(np.cumsum(energy, axis=1), axis=2), axis1=1, axis2=2)

        max_energy = np.nanmax(energy_on_full, axis=1)
        condition = energy_on_full <= state["kink_threshold"] * max_energy[:, None]
        kink_position = np.where(condition.any(axis=1), condition.argmax(axis=1), np.nan)
        if state["distribution_metric"] == "gini":
            distribution_metric = 1 - _gini(energy_on_full, axis=1)
        elif state["distribution_metric"] == "weighted_missing":
            _numerator = np.where(valid_full_dims, (1 - energy_on_full) * variance_activity, np.nan)
            _denominator = np.where(valid_full_dims, variance_activity, np.nan)
            distribution_metric = np.nansum(_numerator, axis=1) / np.nansum(_denominator, axis=1)
        else:
            distribution_metric = np.nanmean(np.where(valid_full_dims, 1 - energy_on_full, np.nan), axis=1)

        # Preserve session order within each mouse and pad shorter histories with NaN.
        def sessions_by_mouse(values):
            mouse_values = [values[self.results.mouse_names == mouse] for mouse in self.results.unique_mice]
            organized = np.full((len(mouse_values), max(map(len, mouse_values))), np.nan)
            for i, values_for_mouse in enumerate(mouse_values):
                organized[i, : len(values_for_mouse)] = values_for_mouse
            return organized

        max_energy_by_session = sessions_by_mouse(max_energy)
        distribution_by_session = sessions_by_mouse(distribution_metric)
        kink_by_session = sessions_by_mouse(kink_position)

        big_first_10 = np.mean(np.diagonal(energy, axis1=1, axis2=2)[:, :10], axis=1)
        idx_aligned = np.argsort(-big_first_10)

        # Right-panel fraction: unweighted subspace overlap ||P u_i||² (sum over PF dims
        # of squared alignment) or the variance-weighted recovery Var(X P u_i)/λ_i.
        if state["weighted"]:
            variance_activity = self.results.sel(**_sel_state)["variance_activity"]
            panel_fraction = _weighted_fraction(cross, variance_activity)
            panel_ylabel = "Variance-Weighted Fraction\nFull Variance Recovered"
        else:
            panel_fraction = energy_on_full
            panel_ylabel = "Fraction Full Variance\nCaptured By Placefields"

        # Average by mouse
        panel_fraction_avg = average_by_mouse(panel_fraction, self.results.mouse_names)
        energy_on_diagonal_avg = average_by_mouse(energy_on_diagonal, self.results.mouse_names)
        energy_expanding_dims_avg = average_by_mouse(energy_expanding_dims, self.results.mouse_names)
        aligned_ratio = energy_on_diagonal_avg / energy_expanding_dims_avg

        fig, ax = plt.subplots(1, 4, figsize=self.figsize, layout="constrained", width_ratios=[1, 1.2, 0.9, 0.9])

        idx_plot = idx_aligned[state["idx_cross"]]
        if state["plot_energy"]:
            imshow_data = energy[idx_plot][:100, :100]
            cmap = "gray_r"
            vmin = 0
        else:
            imshow_data = cross[idx_plot][:100, :100]
            cmap = "bwr"
            vmin = -1

        xlims0 = [-0.5, state["num_cross_show"] + 0.5]
        ylims0 = [state["num_cross_show"] + 0.5, -0.5]
        xbounds0 = [0, state["num_cross_show"]]
        ybounds0 = [state["num_cross_show"], 0]
        extent = [0, 100, 100, 0]
        fontsize = state["fontsize"]
        ax[0].imshow(imshow_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=1, extent=extent)
        ax[0].set_xlabel("Placefield Dimension", fontsize=fontsize)
        ax[0].set_ylabel("Full Dimension", fontsize=fontsize)
        ax[0].set_xlim(xlims0)
        ax[0].set_ylim(ylims0)

        xvals = np.arange(panel_fraction.shape[1]) + 1
        xbounds1 = [1, panel_fraction.shape[1] + 1]
        if state["curve_mode"] == "by_session":
            # Organize each mouse's curves by session order, padding shorter histories with NaN.
            mouse_curves = [panel_fraction[self.results.mouse_names == mouse] for mouse in self.results.unique_mice]
            max_n_sessions = max(map(len, mouse_curves), default=0)
            by_session = np.full((len(mouse_curves), max_n_sessions, panel_fraction.shape[1]), np.nan)
            for i, curves in enumerate(mouse_curves):
                by_session[i, : len(curves)] = curves

            support = np.array([np.sum(np.isfinite(by_session[:, j, :]).any(axis=1)) for j in range(max_n_sessions)])
            kept_js = np.where(support > 1)[0]
            # Color only over the kept session numbers, so a full coolwarm range is used even
            # when trailing/sparse session numbers (<=1 mouse) get excluded below.
            session_colors = plt.get_cmap("coolwarm")(np.linspace(0, 1, max(len(kept_js), 1)))

            # Thin out which kept sessions actually get drawn (always first + last), so dense
            # session counts don't overplot ax[1]. Colors still index the full kept_js range.
            n_kept = len(kept_js)
            step = state["skip_sessions"] + 1
            if n_kept <= 2 or step <= 1:
                show_idx = np.arange(n_kept)
            else:
                n_points = max(2, int(round((n_kept - 1) / step)) + 1)
                show_idx = np.unique(np.round(np.linspace(0, n_kept - 1, n_points)).astype(int))

            finite_chunks = []
            for color_idx in show_idx:
                session_data = by_session[:, kept_js[color_idx], :]
                _render_curve_group(
                    ax[1],
                    xvals,
                    session_data,
                    session_colors[color_idx],
                    state["plot_style"],
                    hide_error=state["hide_error"],
                    linewidth=1.5,
                )
                finite_chunks.append(np.nanmean(session_data, axis=0))
            finite_curve = np.concatenate(finite_chunks) if finite_chunks else np.array([])

            if len(kept_js):
                cmap_data = plt.get_cmap("coolwarm")(np.linspace(0, 1, 255))[np.newaxis, :, :]
                cbar_inset = ax[1].inset_axes([0.05, 0.04, 0.6, 0.075])
                cbar_inset.imshow(cmap_data, aspect="auto")
                cbar_inset.set_xticks([])
                cbar_inset.set_yticks([])
                cbar_inset.text(0.02, 0.5, f"{kept_js[0]}", transform=cbar_inset.transAxes, ha="left", va="center", color="white", fontsize=fontsize)
                cbar_inset.text(
                    0.98, 0.5, f"{kept_js[-1]}", transform=cbar_inset.transAxes, ha="right", va="center", color="white", fontsize=fontsize
                )
                cbar_inset.text(0.5, 0.5, "session #", transform=cbar_inset.transAxes, ha="center", va="center", color="black", fontsize=fontsize)
        else:
            _render_curve_group(
                ax[1],
                xvals,
                panel_fraction_avg,
                "k",
                state["plot_style"],
                hide_error=state["hide_error"],
                linewidth=2.0,
            )
            mean_curve = np.nanmean(panel_fraction_avg, axis=0)
            finite_curve = mean_curve[np.isfinite(mean_curve)]
        ax[1].set_xlabel("Full Dimension", fontsize=fontsize)
        ax[1].set_ylabel(panel_ylabel, fontsize=fontsize)
        ax[1].set_xscale("log")
        ymax = 1.0
        if state["weighted"] and finite_curve.size:
            ymax = max(1.0, float(finite_curve.max()))
        ybounds1 = [0, ymax]
        ax[1].set_xlim(xbounds1)
        ax[1].set_ylim(ybounds1)

        if state["show_decorations"]:
            arrow_style = f"-|>,head_width={state['arrow_head_size']},head_length={2 * state['arrow_head_size']}"
            # Max captured: vertical arrow at a fixed x, from y_start up to y_end.
            ax[1].annotate(
                "",
                xy=(state["max_arrow_x"], state["max_arrow_y_end"]),
                xytext=(state["max_arrow_x"], state["max_arrow_y_start"]),
                arrowprops=dict(arrowstyle=arrow_style, color=_CROSS_METRIC_COLORS["max_captured"], lw=state["arrow_linewidth"]),
                annotation_clip=False,
            )
            # Kink: horizontal arrow at a fixed y, from x_start to x_end.
            ax[1].annotate(
                "",
                xy=(state["kink_arrow_x_end"], state["kink_arrow_y"]),
                xytext=(state["kink_arrow_x_start"], state["kink_arrow_y"]),
                arrowprops=dict(arrowstyle=arrow_style, color=_CROSS_METRIC_COLORS["kink"], lw=state["arrow_linewidth"]),
                annotation_clip=False,
            )
            # Missing structure: patch hugging the curve envelope (max over all sessions at
            # each x), offset right (x_offset) and up (y_offset) from it, filling to the
            # panel's top-right corner.
            envelope = np.nanmax(panel_fraction, axis=0)
            patch_x_start = xbounds1[0] + state["missing_structure_x_offset"]
            patch_bottom = np.minimum(envelope + state["missing_structure_y_offset"], ybounds1[1])
            ax[1].fill_between(
                xvals,
                patch_bottom,
                ybounds1[1],
                where=xvals >= patch_x_start,
                color=_CROSS_METRIC_COLORS["missing_structure"],
                alpha=state["missing_structure_alpha"],
                linewidth=0,
            )

            if state["show_marker_labels"]:
                # "max": vertical text to the right of the max-captured arrow, starting at its bottom.
                ax[1].text(
                    state["max_arrow_x"] * 1.15,
                    state["max_arrow_y_start"],
                    "max",
                    ha="left",
                    va="bottom",
                    rotation=90,
                    color=_CROSS_METRIC_COLORS["max_captured"],
                    fontsize=fontsize,
                )
                # "kink": horizontal text below the kink arrow, right-aligned to its right end.
                ax[1].text(
                    state["kink_arrow_x_end"],
                    state["kink_arrow_y"] - 0.05,
                    "kink",
                    ha="right",
                    va="top",
                    color=_CROSS_METRIC_COLORS["kink"],
                    fontsize=fontsize,
                )
                # "missing structure": top-right corner of the panel.
                ax[1].text(
                    0.98,
                    0.98,
                    "missing\nstructure",
                    transform=ax[1].transAxes,
                    ha="right",
                    va="top",
                    color="black",
                    fontsize=fontsize,
                )

        xticks = [1, 10, 100, 1000]

        session_xvals = np.arange(max_energy_by_session.shape[1])

        def plot_session_metric(axis, values, color, label):
            return _render_curve_group(axis, session_xvals, values, color, state["plot_style"], label=label, min_support=2)

        ax2_xmax = max(
            plot_session_metric(ax[2], max_energy_by_session, _CROSS_METRIC_COLORS["max_captured"], "max variance"),
            plot_session_metric(ax[2], distribution_by_session, _CROSS_METRIC_COLORS["missing_structure"], "missing fraction"),
        )
        ax[2].set_xlabel("Session #", fontsize=fontsize)
        ax[2].legend(fontsize=fontsize, loc="lower left", frameon=False, handlelength=0.8, handletextpad=0.5)

        ax3_xmax = plot_session_metric(ax[3], kink_by_session, _CROSS_METRIC_COLORS["kink"], "kink dimension")
        ax[3].set_xlabel("Session #", fontsize=fontsize)
        ax[3].legend(fontsize=fontsize, loc="upper left", frameon=False, handlelength=0.8, handletextpad=0.5)

        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=xbounds0,
            ybounds=ybounds0,
            tick_fontsize=fontsize,
        )
        format_spines(
            ax[1],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=xbounds1,
            ybounds=ybounds1,
            xticks=xticks,
            yticks=[0, round(ymax, 2)],
            tick_fontsize=fontsize,
        )
        ylim = ax[2].get_ylim()
        ylim = [min(ylim[0], 0.0), max(ylim[1], 1.0)]
        ybounds = [0, 1]
        ax[2].set_ylim(ylim)
        format_spines(
            ax[2], x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"], xbounds=[0, ax2_xmax], ybounds=ybounds, tick_fontsize=fontsize
        )

        ylim = ax[3].get_ylim()
        ylim = [min(ylim[0], 0.0), ylim[1]]
        ybounds = [0, np.floor(ylim[1])]
        format_spines(
            ax[3], x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"], xbounds=[0, ax3_xmax], ybounds=ybounds, tick_fontsize=fontsize
        )
        # format_spines uses tick_params, which doesn't reach minor ticks on ax[1]'s log x-axis.
        ax[1].tick_params(axis="both", which="both", labelsize=fontsize)
        return fig


class SubspaceCrossPerMouseViewer(Viewer):
    """Per-mouse cross-spectrum viewer.

    Left panel: cross energy heat map of one chosen session. Right panel: every
    session of the chosen mouse, showing the fraction of full-activity variance
    captured by placefields, color-coded by session order with ``coolwarm``.
    """

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (6.0, 3.0)):
        self.results = results
        self.figsize = figsize
        for key, value in results.param_axes.items():
            self.add_selection(key, options=value)

        preferred_state = {
            "smooth_width": None,
            "activity_parameters_name": "default",
        }
        for key, value in preferred_state.items():
            self.update_selection(key, value=value)

        self.add_selection("mouse", options=list(results.unique_mice))
        self.add_integer("session", value=0, min=0, max=1)
        self.add_integer("num_cross_show", value=30, min=1, max=100)
        self.add_selection("curve_smooth_kind", options=["none", "boxcar", "gaussian", "median"], value="none")
        self.add_float("curve_smooth_width", value=3.0, min=0.0, max=50.0, step=0.5)
        self.add_float("kink_threshold", value=0.95, min=0.0, max=1.0, step=0.01)
        self.add_selection("distribution_metric", options=["gini", "weighted_missing", "missing_structure"], value="gini")

        self.on_change("mouse", self.update_mouse)
        self.update_mouse(self.state)

    def update_mouse(self, state):
        n_sess = int(np.sum(self.results.mouse_names == state["mouse"]))
        self.update_integer("session", max=max(n_sess - 1, 0))

    def _mouse_results(self, state):
        """Selected results for the chosen mouse; ``cross`` has shape (n_sess_mouse, n_full, n_pf)."""
        _sel_state = {k: v for k, v in state.items() if k in self.results.param_axes}
        return self.results.sel(mouse=state["mouse"], squeeze_ones=False, **_sel_state)

    def _mouse_session_ids(self, mouse):
        return [sid for sid, m in zip(self.results.session_ids, self.results.mouse_names) if m == mouse]

    def plot(self, state):
        mouse = state["mouse"]
        _out = self._mouse_results(state)
        cross = _out["cross"]
        energy = cross**2
        variance_activity = _out["variance_activity"][:, : cross.shape[1]]
        energy_on_full = np.nansum(energy, axis=2)  # (n_sess, n_full)
        valid_full_dims = np.isfinite(cross).any(axis=2)
        energy_on_full = np.where(valid_full_dims, energy_on_full, np.nan)
        energy_on_full = _smooth_fraction(energy_on_full, state["curve_smooth_kind"], state["curve_smooth_width"])

        # Get metrics for energy on full
        max_energy = np.nanmax(energy_on_full, axis=1)
        condition = energy_on_full <= state["kink_threshold"] * max_energy[:, None]
        kink_position = np.where(condition.any(axis=1), condition.argmax(axis=1), np.nan)
        if state["distribution_metric"] == "gini":
            distribution_metric = _gini(energy_on_full, axis=1)
            distribution_label = "Gini Equality"
        elif state["distribution_metric"] == "weighted_missing":
            _numerator = np.where(valid_full_dims, (1 - energy_on_full) * variance_activity, np.nan)
            _denominator = np.where(valid_full_dims, variance_activity, np.nan)
            distribution_metric = np.nansum(_numerator, axis=1) / np.nansum(_denominator, axis=1)
            distribution_label = "Weighted Missing"
        else:
            distribution_metric = np.nanmean(np.where(valid_full_dims, 1 - energy_on_full, np.nan), axis=1)
            distribution_label = "Missing Structure"

        n_sess = cross.shape[0]
        sess = min(state["session"], n_sess - 1)

        fig, ax = plt.subplots(1, 3, figsize=self.figsize, layout="constrained")

        # Left: cross energy heat map of chosen session
        imshow_data = energy[sess][:100, :100]
        extent = [0, 100, 100, 0]
        ax[0].imshow(imshow_data.T, cmap="gray_r", aspect="auto", vmin=0, vmax=1, extent=extent)
        ax[0].set_xlabel("Full Dim.")
        ax[0].set_ylabel("Placefield Dim.")
        xbounds0 = [0, state["num_cross_show"]]
        ybounds0 = [state["num_cross_show"], 0]
        ax[0].set_xlim([-0.5, state["num_cross_show"] + 0.5])
        ax[0].set_ylim([state["num_cross_show"] + 0.5, -0.5])
        session_ids = self._mouse_session_ids(mouse)
        ax[0].set_title(session_ids[sess] if sess < len(session_ids) else mouse)

        # Right: fraction full captured by PF for every session, colored by session order
        colors = plt.get_cmap("coolwarm")(np.linspace(0, 1, max(n_sess, 1)))
        xvals = np.arange(energy_on_full.shape[1]) + 1
        for i in range(n_sess):
            ax[1].plot(xvals, energy_on_full[i], color=colors[i], linewidth=1.0)
        ax[1].set_xlabel("Full Dim.")
        ax[1].set_ylabel("Fraction Captured\nBy Placefields")
        ax[1].set_xscale("log")
        xbounds1 = [1, energy_on_full.shape[1] + 1]
        ybounds1 = [0, 1.0]
        ax[1].set_xlim(xbounds1)
        ax[1].set_ylim(ybounds1)

        xvals = range(len(kink_position))
        ax[2].plot(xvals, max_energy, color="k", linewidth=1.0, label="Max Energy")
        ax[2].plot(xvals, distribution_metric, color="b", linewidth=1.0, label=distribution_label)
        kink_ax = ax[2].twinx()
        kink_ax.plot(xvals, kink_position, color="r", linewidth=1.0, label="Kink Position")
        ax[2].set_xlabel("Session Index")
        ax[2].set_ylabel(f"Max Energy / {distribution_label}")
        kink_ax.set_ylabel("Kink Position", color="r")
        kink_ax.tick_params(axis="y", colors="r")
        lines, labels = ax[2].get_legend_handles_labels()
        kink_lines, kink_labels = kink_ax.get_legend_handles_labels()
        ax[2].legend(lines + kink_lines, labels + kink_labels, fontsize=8)

        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=xbounds0,
            ybounds=ybounds0,
        )
        format_spines(
            ax[1],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=xbounds1,
            ybounds=ybounds1,
            xticks=[1, 10, 100, 1000],
            yticks=[0, 1],
        )
        format_spines(
            ax[2],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
        )
        return fig


def subspace_crossspace_per_mouse(
    results: ResultsAggregator,
    mouse: str | None = None,
    session: int = 0,
    num_cross_show: int = 30,
    curve_smooth_kind: str = "none",
    curve_smooth_width: float = 3.0,
    kink_threshold: float = 0.95,
    distribution_metric: str = "gini",
    figsize: tuple[float, float] = (6.0, 3.0),
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Per-mouse cross-spectrum figure.

    Left panel shows the cross energy heat map of the chosen session. Right panel
    shows, for every session of the chosen mouse, the fraction of full-activity
    variance captured by placefields, color-coded in session order with ``coolwarm``.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated subspace results providing ``param_axes``, ``sel``, ``unique_mice``,
        ``mouse_names`` and ``session_ids``.
    mouse : str, optional
        Mouse to show. Defaults to the first mouse in ``results.unique_mice``.
    session : int
        Index (within the chosen mouse's sessions) of the example cross matrix in the
        left panel.
    num_cross_show : int
        Number of cross dimensions to show in the left panel.
    curve_smooth_kind : {"none", "boxcar", "gaussian", "median"}
        Linear NaN-aware smoothing applied to each session's fraction curve in the right
        panel. ``"median"`` is edge/kink-preserving; ``"none"`` disables smoothing.
    curve_smooth_width : float
        Boxcar/median full-width in dim units; the Gaussian uses ``sigma = curve_smooth_width / 2``.
    kink_threshold : float
        Threshold (fraction of max) for the kink position metric in the right panel.
    distribution_metric : {"gini", "weighted_missing", "missing_structure"}
        Metric shown with max energy. Missing structure is the mean uncaptured
        fraction over valid full dimensions; weighted missing is the same uncaptured
        fraction weighted by each full dimension's activity variance.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections (e.g. ``smooth_width``,
        ``activity_parameters_name``). Each key must be a valid ``results.param_axes`` name.

    Returns
    -------
    matplotlib.figure.Figure or SubspaceCrossPerMouseViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = SubspaceCrossPerMouseViewer(results, figsize=figsize)
    for key, value in selections.items():
        if key not in results.param_axes:
            raise ValueError(f"Unknown selection {key!r}. Options: {list(results.param_axes)}")
        viewer.update_selection(key, value=value)
    if mouse is not None:
        viewer.update_selection("mouse", value=mouse)
        viewer.update_mouse(viewer.state)
    viewer.update_integer("session", value=session)
    viewer.update_integer("num_cross_show", value=num_cross_show)
    viewer.update_selection("curve_smooth_kind", value=curve_smooth_kind)
    viewer.update_float("curve_smooth_width", value=curve_smooth_width)
    viewer.update_float("kink_threshold", value=kink_threshold)
    viewer.update_selection("distribution_metric", value=distribution_metric)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


def subspace_crossspace(
    results: ResultsAggregator,
    idx_cross: int = 0,
    plot_energy: bool = True,
    num_cross_show: int = 30,
    weighted: bool = False,
    curve_mode: str = "average",
    plot_style: str = "each",
    hide_error: bool = False,
    skip_sessions: int = 0,
    curve_smooth_kind: str = "none",
    curve_smooth_width: float = 3.0,
    kink_threshold: float = 0.95,
    distribution_metric: str = "weighted_missing",
    show_decorations: bool = True,
    show_marker_labels: bool = True,
    arrow_linewidth: float = 1.5,
    arrow_head_size: float = 0.4,
    max_arrow_x: float = 1.0,
    max_arrow_y_start: float = 0.0,
    max_arrow_y_end: float = 0.9,
    kink_arrow_x_start: float = 1.0,
    kink_arrow_x_end: float = 100.0,
    kink_arrow_y: float = 0.9,
    missing_structure_x_offset: float = 10.0,
    missing_structure_y_offset: float = 0.05,
    missing_structure_alpha: float = 0.15,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (12.0, 3.0),
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Cross-spectrum energy figure for aggregated subspace results.

    The left panel shows an example placefield-vs-full cross matrix for one session
    (chosen by ``idx_cross`` from sessions sorted by descending mean top-10 diagonal
    energy). The second panel shows the mouse-averaged fraction of full-activity
    variance captured by placefields. The final panels show per-mouse session
    trajectories and their supported cross-mouse averages for max energy/Gini and
    kink position.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated subspace results providing ``param_axes``, ``sel`` and ``mouse_names``.
    idx_cross : int
        Index (into sessions sorted by descending mean top-10 diagonal energy) of the
        example cross matrix shown in the left panel.
    plot_energy : bool
        Show squared cross energy (``gray_r``, from 0) when True, otherwise the signed
        cross values (``bwr``, from -1) in the left panel.
    num_cross_show : int
        Number of cross dimensions to show in the left panel.
    weighted : bool
        Right panel shows the variance-weighted recovery ``Var(X P u_i)/λ_i`` when True
        (down-weights overlap onto low-variance full PCs), otherwise the unweighted
        subspace overlap ``||P u_i||²``.
    curve_mode : {"average", "by_session"}
        ``"average"`` groups all sessions of a mouse into one curve per mouse for the second
        panel. ``"by_session"`` instead groups curves by within-mouse session number (first
        session of each mouse, second of each mouse, ...), one color-coded group per session
        number (``coolwarm``, early to late); session numbers with data from only one mouse are
        skipped. Either way, rendering is controlled by ``plot_style``.
    plot_style : {"each", "errorPlot"}
        How every curve group (in the second panel, and the per-mouse session-index panels) is
        rendered. ``"each"`` draws each underlying curve thin plus a solid mean, as before.
        ``"errorPlot"`` draws a mean +/- std band instead of the individual thin curves.
    hide_error : bool
        When ``plot_style == "errorPlot"``, suppress the std band in the second panel and show
        only the mean curve(s). No effect on ``plot_style == "each"`` or on the other panels.
    skip_sessions : int
        Only used when ``curve_mode == "by_session"``. Thins out which session-number groups are
        drawn in the second panel: the first and last kept session number are always shown, with
        the rest spaced as evenly as possible to skip roughly ``skip_sessions`` kept session
        numbers between each drawn one. ``0`` (default) draws every kept session number. Colors
        still span the full kept-session range, so gaps don't compress the ``coolwarm`` scale.
    curve_smooth_kind : {"none", "boxcar", "gaussian", "median"}
        Smoothing applied to energy-on-full curves before computing session metrics.
    curve_smooth_width : float
        Smoothing width in full-dimension units.
    kink_threshold : float
        Fraction of maximum energy used to locate the first below-threshold dimension.
    distribution_metric : {"gini", "missing_structure"}
        Metric shown with max energy. Missing structure is the mean uncaptured
        fraction over valid full dimensions.
    show_decorations : bool
        Overlay the ax[1] metric-explainer decorations (colors fixed in
        ``_CROSS_METRIC_COLORS``): a vertical arrow for max captured, a horizontal arrow for
        kink position, and a patch hugging the curve envelope for missing structure.
    show_marker_labels : bool
        Overlay text labels next to each decoration (only when ``show_decorations`` is also
        True): "max" rotated vertically to the right of the max-captured arrow, "kink"
        horizontally below the kink arrow, and "missing\\nstructure" in the panel's top-right
        corner.
    arrow_linewidth : float
        Line width of both the "max captured" and "kink" arrows.
    arrow_head_size : float
        Arrow head width (in points-ish arrowstyle units); head length is fixed at
        ``2 * arrow_head_size``. Shared by both arrows.
    max_arrow_x : float
        X position (dim units) of the vertical "max captured" arrow.
    max_arrow_y_start, max_arrow_y_end : float
        Start/end y of the "max captured" arrow.
    kink_arrow_x_start, kink_arrow_x_end : float
        Start/end x (dim units) of the horizontal "kink" arrow.
    kink_arrow_y : float
        Y position of the "kink" arrow.
    missing_structure_x_offset : float
        Right offset (dim units, added to the panel's left edge) of where the missing-structure
        patch begins.
    missing_structure_y_offset : float
        Vertical gap added above the curve envelope (max over all sessions at each x) for the
        missing-structure patch's bottom edge.
    missing_structure_alpha : float
        Fill alpha of the missing-structure patch.
    fontsize : float
        Font size applied uniformly across every panel: axis labels, tick labels (major and
        minor, including ax[1]'s log-scale x-axis), the colorbar-inset session-number labels,
        and both legends.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections (e.g. ``smooth_width``,
        ``activity_parameters_name``). Each key must be a valid ``results.param_axes`` name.

    Returns
    -------
    matplotlib.figure.Figure or SubspaceCrossspaceViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = SubspaceCrossspaceViewer(results, figsize=figsize)
    for key, value in selections.items():
        if key not in results.param_axes:
            raise ValueError(f"Unknown selection {key!r}. Options: {list(results.param_axes)}")
        viewer.update_selection(key, value=value)
    viewer.update_integer("idx_cross", value=idx_cross)
    viewer.update_boolean("plot_energy", value=plot_energy)
    viewer.update_integer("num_cross_show", value=num_cross_show)
    viewer.update_boolean("weighted", value=weighted)
    viewer.update_selection("curve_mode", value=curve_mode)
    viewer.update_selection("plot_style", value=plot_style)
    viewer.update_boolean("hide_error", value=hide_error)
    viewer.update_integer("skip_sessions", value=skip_sessions)
    viewer.update_selection("curve_smooth_kind", value=curve_smooth_kind)
    viewer.update_float("curve_smooth_width", value=curve_smooth_width)
    viewer.update_boolean("show_decorations", value=show_decorations)
    viewer.update_boolean("show_marker_labels", value=show_marker_labels)
    viewer.update_float("arrow_linewidth", value=arrow_linewidth)
    viewer.update_float("arrow_head_size", value=arrow_head_size)
    viewer.update_float("max_arrow_x", value=max_arrow_x)
    viewer.update_float("max_arrow_y_start", value=max_arrow_y_start)
    viewer.update_float("max_arrow_y_end", value=max_arrow_y_end)
    viewer.update_float("kink_arrow_x_start", value=kink_arrow_x_start)
    viewer.update_float("kink_arrow_x_end", value=kink_arrow_x_end)
    viewer.update_float("kink_arrow_y", value=kink_arrow_y)
    viewer.update_float("missing_structure_x_offset", value=missing_structure_x_offset)
    viewer.update_float("missing_structure_y_offset", value=missing_structure_y_offset)
    viewer.update_float("missing_structure_alpha", value=missing_structure_alpha)
    viewer.update_float("fontsize", value=fontsize)
    viewer.update_float("kink_threshold", value=kink_threshold)
    viewer.update_selection("distribution_metric", value=distribution_metric)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Helpers for subspace_familiarity
# ---------------------------------------------------------------------------

# Shared orange-family palette for behaving/ITI/spontaneous condition splits, used by
# both SubspaceFamiliarityViewer (plot_mode="all") and SubspaceCurvesRatiosViewer.
CONDITION_COLORS = {
    "behaving": "darkorange",
    "itis": "orangered",
    "spontaneous": "sienna",
}

_FAMILIARITY_COLORS = {
    "Behaving": CONDITION_COLORS["behaving"],
    "w/ ITIs": CONDITION_COLORS["itis"],
    "w/ Spont.": CONDITION_COLORS["spontaneous"],
    "Env #1": "k",
    "Env #2": "blue",
    "Env #3": "green",
}
_ENV_FULL_SCOPES = ["within_env", "outside_env", "with_iti", "with_spontaneous"]


def _tuple_label(value: tuple) -> str:
    """Render a tuple param value (elements are float or None) as a widget-safe string label."""
    return "-".join("None" if v is None else str(v) for v in value)


_STIMSPACE_PREFERRED_STATE = {
    "smooth_widths": (5.0, None),
    "activity_parameters_name": "default",
    "reliability_fraction_active_thresholds": (None, None),
}


def _register_stimspace_selections(viewer: Viewer, results: ResultsAggregator) -> dict[str, dict[str, tuple]]:
    """Register one ``add_selection`` per ``StimSpaceSpectraConfig`` param axis (skipping ``include_iti``).

    Tuple-valued axes (e.g. ``smooth_widths``) can't be dropdown values directly, so they are
    encoded as string labels; the returned ``tuple_labels`` map (``{axis_name: {label: original
    tuple}}``) is what :func:`_stimspace_sel_params` uses to decode them back. Shared setup for
    every viewer built on these results (ratios, familiarity, composite).
    """
    tuple_labels: dict[str, dict[str, tuple]] = {}
    for key, options in results.param_axes.items():
        if key == "include_iti":
            continue
        if any(isinstance(opt, tuple) for opt in options):
            label_map = {_tuple_label(opt): opt for opt in options}
            tuple_labels[key] = label_map
            widget_options = list(label_map)
        else:
            widget_options = options
        viewer.add_selection(key, options=widget_options)

    for key, value in _STIMSPACE_PREFERRED_STATE.items():
        if key in results.param_axes:
            encoded = _tuple_label(value) if key in tuple_labels and isinstance(value, tuple) else value
            viewer.update_selection(key, value=encoded)
    return tuple_labels


def _stimspace_sel_params(results: ResultsAggregator, tuple_labels: dict[str, dict[str, tuple]], state: dict) -> dict:
    """Params to forward to ``results.sel``, decoding tuple labels back to tuples."""
    params = {}
    for key, value in state.items():
        if key not in results.param_axes or key == "include_iti":
            continue
        if key in tuple_labels:
            value = tuple_labels[key][value]
        params[key] = value
    return params


def _chronological_mouse_sessions(results: ResultsAggregator, mouse: str, exclude_bad_envs: bool = True) -> np.ndarray:
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


def _pad_stack_by_mouse(curves: dict[str, np.ndarray]) -> np.ndarray:
    """Stack ragged per-mouse 1D curves into a NaN-padded ``(n_mice, max_len)`` array."""
    max_len = max((len(v) for v in curves.values()), default=0)
    stack = np.full((len(curves), max_len), np.nan)
    for i, values in enumerate(curves.values()):
        stack[i, : len(values)] = values
    return stack


def _support_length(pad_stack: np.ndarray, min_support: int = 1) -> int:
    """Number of leading columns where more than ``min_support`` mice have finite data."""
    support = np.sum(np.isfinite(pad_stack), axis=0)
    valid = np.where(support > min_support)[0]
    return int(valid[-1] + 1) if valid.size else 0


def _mean_with_min_support(pad_stack: np.ndarray, min_support: int = 1) -> np.ndarray:
    """Nanmean across mice (axis 0), truncated where at most ``min_support`` mice have data."""
    length = _support_length(pad_stack, min_support)
    return np.nanmean(pad_stack[:, :length], axis=0)


def _curves_from_defs(
    results: ResultsAggregator,
    curve_defs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    exclude_bad_envs: bool,
    within_condition: bool = True,
) -> dict[str, dict]:
    """Turn ``{curve_label: (ratio, total, keep_mask)}`` session-level arrays into per-mouse curves.

    Shared tail end of :func:`_familiarity_curves` for both ``mode="all"`` (curve_label = ITI
    status) and ``mode="by_env"`` (curve_label = env slot): reorders each mouse's sessions
    chronologically, then densifies to sessions where ``keep`` is True.

    When ``within_condition``, kept sessions are renumbered 0..n_kept-1 (the env/ITI-conditioned
    session index), matching the original behavior. When not, kept sessions instead keep their
    position in the mouse's full chronological session order (so e.g. a mouse's first
    spontaneous session, if it's their 6th session overall, lands in bin 5, not bin 0), with
    NaN filling the dropped-session gaps in between.
    """
    unique_mice = results.unique_mice
    curves: dict[str, dict] = {}
    for curve_label, (ratio, total, keep) in curve_defs.items():
        svr_per_mouse, total_per_mouse = {}, {}
        for mouse in unique_mice:
            idx_sorted = _chronological_mouse_sessions(results, mouse, exclude_bad_envs=exclude_bad_envs)
            mouse_keep = keep[idx_sorted]
            if within_condition:
                idx_use = idx_sorted[mouse_keep]
                svr_per_mouse[mouse] = ratio[idx_use]
                total_per_mouse[mouse] = total[idx_use]
            else:
                svr_full = np.full(len(idx_sorted), np.nan)
                total_full = np.full(len(idx_sorted), np.nan)
                svr_full[mouse_keep] = ratio[idx_sorted[mouse_keep]]
                total_full[mouse_keep] = total[idx_sorted[mouse_keep]]
                svr_per_mouse[mouse] = svr_full
                total_per_mouse[mouse] = total_full
        curves[curve_label] = {"svr": svr_per_mouse, "total": total_per_mouse}
    return curves


def _familiarity_curves(
    results: ResultsAggregator,
    sel_params: dict,
    mode: str,
    env_full_scope: str = "within_env",
    full_within_env: bool = True,
    within_condition: bool = True,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Per-mouse variance-ratio / total-variance curves, keyed by curve label.

    Returns ``{curve_label: {"svr": {mouse: 1D array}, "total": {mouse: 1D array}}}``. In both
    modes the x-axis is, by default, the env/ITI-conditioned session index (session number
    within the kept subset, not the mouse's overall session number).

    ``within_condition`` (``mode == "all"`` only; ignored for ``"by_env"``): if False, curves are
    instead aligned to the mouse's overall chronological session index, so e.g. ``"w/ Spont."``
    data from a mouse's 6th session lands in bin 5 rather than bin 0.

    ``mode == "all"``: whole-session ``sf_cv``/``ff`` keys, curve labels ``"Behaving"`` /
    ``"w/ ITIs"`` / ``"w/ Spont."`` overlaid together (unaffected by ``env_full_scope``).

    ``mode == "by_env"``: ``env_full_scope`` selects exactly one ``(sf_key, ff_key, include_iti,
    session_mask)`` combination — ``"within_env"`` (``sf_cv_env_full1`` / ``ff_env_full1``,
    env-only frames, no ITI variant exists), ``"outside_env"`` (``sf_cv_env_fullall`` /
    ``ff_env_full1_fullall``, env-stim vs all-env-func, behaving-only), ``"with_iti"`` (same keys,
    ``include_iti=True``, non-spontaneous sessions), ``"with_spontaneous"`` (same keys,
    ``include_iti=True``, spontaneous sessions). Curve labels are ``"Env #1"``/``"Env #2"``/``"Env #3"``
    (all ``MAX_ENV_SLOTS`` experience-order slots are always overlaid together).
    """
    if mode == "all":
        session_has_spontaneous = np.array([s.has_spontaneous() for s in results.sessions])
        all_sessions = np.ones_like(session_has_spontaneous, dtype=bool)

        def _fetch(include_iti: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            out = results.sel(keys=["sf_cv", "ff"], squeeze_ones=False, avg_by_mouse=False, include_iti=include_iti, **sel_params)
            total = np.nansum(out["ff"], axis=1)
            ratio = np.nansum(out["sf_cv"], axis=1) / total
            valid = np.isfinite(out["ff"]).any(axis=1)
            return ratio, total, valid

        ratio_b, total_b, valid_b = _fetch(False)
        ratio_i, total_i, valid_i = _fetch(True)
        curve_defs = {
            "Behaving": (ratio_b, total_b, valid_b & all_sessions),
            "w/ ITIs": (ratio_i, total_i, valid_i & ~session_has_spontaneous),
            "w/ Spont.": (ratio_i, total_i, valid_i & session_has_spontaneous),
        }
        return _curves_from_defs(results, curve_defs, exclude_bad_envs=True, within_condition=within_condition)

    if env_full_scope not in _ENV_FULL_SCOPES:
        raise ValueError(f"Unknown env_full_scope {env_full_scope!r}. Options: {_ENV_FULL_SCOPES}")

    if env_full_scope == "within_env":
        sf_key = "sf_cv_env_full1"
        ff_key = "ff_env_full1"
        include_iti = False
    else:
        sf_key = "sf_cv_env_fullall"
        ff_key = "ff_env_full1_fullall" if full_within_env else "ff"
        include_iti = env_full_scope != "outside_env"

    out = results.sel(keys=[sf_key, ff_key], squeeze_ones=False, avg_by_mouse=False, include_iti=include_iti, **sel_params)
    sf_all_slots, ff_all_slots = out[sf_key], out[ff_key]

    if env_full_scope == "with_iti":
        session_mask = ~np.array([s.has_spontaneous() for s in results.sessions])
    elif env_full_scope == "with_spontaneous":
        session_mask = np.array([s.has_spontaneous() for s in results.sessions])
    else:
        session_mask = np.ones(len(results.sessions), dtype=bool)

    curve_defs = {}
    for env_slot in range(MAX_ENV_SLOTS):
        sf = sf_all_slots[:, env_slot, :]
        if ff_all_slots.ndim == 3:
            ff = ff_all_slots[:, env_slot, :]
        else:
            ff = ff_all_slots
        total = np.nansum(ff, axis=1)
        ratio = np.nansum(sf, axis=1) / total
        valid = np.isfinite(ff).any(axis=1) & np.isfinite(sf).any(axis=1)
        curve_defs[f"Env #{env_slot + 1}"] = (ratio, total, valid & session_mask)

    return _curves_from_defs(results, curve_defs, exclude_bad_envs=False)


def _familiarity_panel(ax, axis_curves: dict, metric: str, xlabel: str, ylabel: str, style: str) -> float:
    """Plot one familiarity metric panel (svr or total) across curve labels; returns the max finite value drawn."""
    max_val = 0.0
    for curve_label, data in axis_curves.items():
        color = _FAMILIARITY_COLORS[curve_label]
        per_mouse = data[metric]
        stack = _pad_stack_by_mouse(per_mouse)
        length = _support_length(stack)
        stack = stack[:, :length]
        if style == "all":
            for values in per_mouse.values():
                ax.plot(np.arange(len(values)), values, color=(color, 0.3), linewidth=0.5)
            ax.plot(np.arange(length), _mean_with_min_support(stack), color=color, linewidth=2.0, label=curve_label)
            visible = stack
        elif length:
            errorPlot(np.arange(length), stack, axis=0, se=True, ax=ax, color=color, linewidth=2.0, label=curve_label, alpha=0.25)
            # Only the mean +/- SE band is actually drawn, not the raw per-mouse traces, so
            # the ymax should track the band's upper edge rather than individual-mouse outliers.
            num_valid = np.sum(~np.isnan(stack), axis=0)
            se = np.nanstd(stack, axis=0) / np.sqrt(num_valid)
            visible = np.nanmean(stack, axis=0) + se
        else:
            visible = stack
        finite = visible[np.isfinite(visible)]
        if finite.size:
            max_val = max(max_val, float(finite.max()))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return max_val


def _format_familiarity_ylim(ax, max_val: float, round_to_tenth: bool = True) -> None:
    """Set ``ylim = (0, 1.05 * max_val)`` and spine ``xbounds``/``ybounds`` up to clean values.

    ``ybounds`` top is ``ceil(ylim_top * 10) / 10`` when ``round_to_tenth`` (0-1 ratio
    panels), else ``ceil(ylim_top)`` (unbounded total-variance panel).
    """
    ylim_top = 1.05 * max_val if max_val > 0 else 1.0
    ybound_top = np.ceil(ylim_top * 10) / 10 if round_to_tenth else np.ceil(ylim_top)
    ylim_top = max(ylim_top, ybound_top)  # ceil can push the bound above the padded max; keep it on-screen
    ax.set_ylim(0, ylim_top)
    xbounds = [0, ax.get_xlim()[1]]
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=xbounds,
        ybounds=[0, ybound_top],
    )


def _render_familiarity_panels(ax_ratio, ax_total, curves: dict, mode: str, style: str, fontsize: int = 9) -> None:
    """Render the Variance Ratio / Total Variance panel pair shared by the familiarity figure and the composite figure."""
    xlabel = "Session #" if mode == "all" else "Env Session #"
    max_ratio = _familiarity_panel(ax_ratio, curves, "svr", xlabel, "Variance Ratio", style)
    max_total = _familiarity_panel(ax_total, curves, "total", xlabel, "Total Variance", style)
    ax_ratio.legend(loc="lower right", fontsize=fontsize, frameon=False, markerfirst=False)
    _format_familiarity_ylim(ax_ratio, max_ratio)
    _format_familiarity_ylim(ax_total, max_total, round_to_tenth=False)


def _render_familiarity_ratio_panel(ax_ratio, curves: dict, mode: str, style: str, fontsize: int = 9) -> None:
    """Render only the Variance Ratio panel (no Total Variance) for the composite figure."""
    xlabel = "Session #" if mode == "all" else "Env Session #"
    max_ratio = _familiarity_panel(ax_ratio, curves, "svr", xlabel, "Variance Ratio", style)
    ax_ratio.legend(loc="upper left", fontsize=fontsize, frameon=False, markerfirst=True, handlelength=0.8, handletextpad=0.5)
    _format_familiarity_ylim(ax_ratio, max_ratio)


class SubspaceFamiliarityViewer(Viewer):
    """Variance-ratio-over-familiarity viewer: whole-session or per-env-experience-slot curves.

    Both modes render a 1x2 grid (Variance Ratio, Total Variance). ``plot_mode="all"`` reproduces
    the original whole-session figure, x-axis = overall session number. ``plot_mode="by_env"``
    shows the same two metrics for every environment-experience slot overlaid together (colored
    black/blue/green for slot 0/1/2), x-axis = session number within that env. ``env_full_scope``
    (only used in ``by_env``) picks which per-env key pairing / ITI condition to show.
    ``within_condition`` (only used in ``all``) toggles whether each curve's x-axis is renumbered
    to its own kept-session index (True, default) or left at the mouse's overall chronological
    session index (False).
    """

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (6.0, 3.0)):
        self.results = results
        self.figsize = figsize
        self._tuple_labels = _register_stimspace_selections(self, results)

        self.add_selection("plot_mode", options=["all", "by_env"], value="all")
        self.add_selection("env_full_scope", options=_ENV_FULL_SCOPES, value="within_env")
        self.add_boolean("full_within_env", value=True)
        self.add_boolean("within_condition", value=True)
        self.add_selection("style", options=["errorPlot", "all"], value="all")

    def encode_param(self, name: str, value):
        """Map a raw param value to its widget value (tuple -> string label; else unchanged)."""
        if name in self._tuple_labels and isinstance(value, tuple):
            return _tuple_label(value)
        return value

    def _sel_params(self, state: dict) -> dict:
        """Params to forward to ``results.sel``, decoding tuple labels back to tuples."""
        return _stimspace_sel_params(self.results, self._tuple_labels, state)

    def plot(self, state: dict):
        sel_params = self._sel_params(state)
        mode = state["plot_mode"]
        curves = _familiarity_curves(
            self.results,
            sel_params,
            mode,
            env_full_scope=state["env_full_scope"],
            full_within_env=state["full_within_env"],
            within_condition=state["within_condition"],
        )
        fontsize = 9
        plt.rcParams["font.size"] = fontsize

        style = state["style"]
        xlabel = "Session #" if mode == "all" else "Env Session #"
        fig, ax = plt.subplots(1, 2, figsize=self.figsize, layout="constrained")
        _render_familiarity_panels(ax[0], ax[1], curves, mode, style, fontsize)
        return fig


def subspace_familiarity(
    results: ResultsAggregator,
    plot_mode: str = "all",
    env_full_scope: str = "within_env",
    full_within_env: bool = True,
    within_condition: bool = True,
    style: str = "errorPlot",
    figsize: tuple[float, float] = (6.0, 3.0),
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Variance-ratio-over-familiarity figure.

    Shows, per mouse and averaged across mice, the fraction of total activity variance shared
    with the stimulus (placefield) subspace across sessions ("Variance Ratio") and the total
    activity variance itself ("Total Variance"). ``plot_mode="all"`` uses the whole-session
    spectra (``sf_cv`` / ``ff``), x-axis = overall session number; ``plot_mode="by_env"`` uses
    the per-environment-experience-slot spectra from ``StimSpaceSpectraConfig._per_env_spectra``,
    overlaying all ``MAX_ENV_SLOTS`` experience-order slots together (black/blue/green for slot
    0/1/2), x-axis = session number within that env.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``StimSpaceSpectraConfig`` results providing ``param_axes``, ``sel``,
        ``unique_mice``, ``mouse_names`` and ``sessions``.
    plot_mode : {"all", "by_env"}
        Whole-session curves, or per-environment-experience-slot curves.
    env_full_scope : {"within_env", "outside_env", "with_iti", "with_spontaneous"}
        Only used when ``plot_mode == "by_env"`` (ignored for ``"all"``). Selects which per-env
        key pairing / ITI condition to plot: ``"within_env"`` normalizes by the env-restricted
        total variance (``sf_cv_env_full1`` / ``ff_env_full1``; ``include_iti`` has no effect on
        this scope, since its func side is always env-only VR frames). The other three all use
        ``sf_cv_env_fullall`` / ``ff_env_full1_fullall`` (env-stim vs all-env-func, normalized by
        the matching mixed-scope total-variance estimate): ``"outside_env"`` is the behaving-only
        condition (``include_iti=False``), ``"with_iti"`` is ``include_iti=True`` restricted to
        non-spontaneous sessions, and ``"with_spontaneous"`` is ``include_iti=True`` restricted to
        sessions with a genuine spontaneous window.
    full_within_env : bool
        Only used when ``plot_mode == "by_env"`` (ignored for ``"all"``). If True, the total-variance
        denominator is the env-only variance (``ff_env_full1``); if False, the denominator is the
        whole-session variance (``ff``).
    within_condition : bool
        Only used when ``plot_mode == "all"`` (ignored for ``"by_env"``). If True (default), each
        curve's x-axis is the session index within its own kept subset (e.g. a mouse's first
        spontaneous session is bin 0, regardless of its overall session number). If False, bins
        instead track the mouse's overall chronological session index, so e.g. a mouse's 6th
        session (0-indexed: 5) that happens to be its first spontaneous session lands in bin 5.
    style : {"all", "errorPlot"}
        ``"all"`` plots every mouse's curve as a faint line plus the mouse-mean as a bold line
        (as before). ``"errorPlot"`` drops the per-mouse lines and instead shows the mouse
        mean +/- SE as a shaded band (via ``vrAnalysis.helpers.plotting.errorPlot``).
    figsize : tuple[float, float]
        Figure size in inches. Both ``plot_mode`` values render a 1x2 grid.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections (e.g. ``smooth_widths``,
        ``activity_parameters_name``, ``reliability_fraction_active_thresholds``). Each key
        must be a valid ``results.param_axes`` name other than ``include_iti`` (handled
        internally via ``env_full_scope`` / the Behaving/ITI/Spontaneous curve split).
        Tuple-valued axes (e.g. ``smooth_widths=(5.0, None)``) are passed as native tuples; they
        are encoded to the widget's string labels internally.

    Returns
    -------
    matplotlib.figure.Figure or SubspaceFamiliarityViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = SubspaceFamiliarityViewer(results, figsize=figsize)
    valid_keys = [k for k in results.param_axes if k != "include_iti"]
    for key, value in selections.items():
        if key not in valid_keys:
            raise ValueError(f"Unknown selection {key!r}. Options: {valid_keys}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))
    viewer.update_selection("plot_mode", value=plot_mode)
    viewer.update_selection("env_full_scope", value=env_full_scope)
    viewer.update_selection("style", value=style)
    viewer.update_boolean("full_within_env", value=full_within_env)
    viewer.update_boolean("within_condition", value=within_condition)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Helpers for subspace_curves_ratios
# ---------------------------------------------------------------------------


def _ratios_arrays(results: ResultsAggregator, sel_params: dict) -> dict[str, np.ndarray]:
    """Mouse-averaged spectra plus per-mouse cumulative-variance-ratio arrays for the ratios figure.

    Shared data prep for both :func:`_plot_ratios_spectrum` and :func:`_plot_ratios_beeswarms`.
    """
    out = results.sel(keys=["sf_cv", "ff"], **sel_params, avg_by_mouse=False, include_iti=False)
    out_iti = results.sel(keys=["sf_cv", "ff"], **sel_params, avg_by_mouse=False, include_iti=True)

    full_sum = np.nansum(out["ff"], axis=1, keepdims=True)
    full_sum_iti = np.nansum(out_iti["ff"], axis=1, keepdims=True)
    sf_cv = out["sf_cv"] / full_sum
    sf_cv_iti = out_iti["sf_cv"] / full_sum_iti
    ff = out["ff"] / full_sum

    full_sum_10 = np.nansum(out["ff"][:, :10], axis=1, keepdims=True)
    full_sum_iti_10 = np.nansum(out_iti["ff"][:, :10], axis=1, keepdims=True)
    sf_cv_10 = out["sf_cv"][:, :10] / full_sum_10
    sf_cv_iti_10 = out_iti["sf_cv"][:, :10] / full_sum_iti_10

    session_has_spontaneous = np.array([session.has_spontaneous() for session in results.sessions])

    # Measure cumulative variance after normalizing
    sf_cv_total = average_by_mouse(np.nansum(sf_cv, axis=1), results.mouse_names)
    sf_cv_total_10 = average_by_mouse(np.nansum(sf_cv_10, axis=1), results.mouse_names)
    sf_cv_total_iti = average_by_mouse(np.nansum(sf_cv_iti[~session_has_spontaneous], axis=1), results.mouse_names[~session_has_spontaneous])
    sf_cv_total_iti_10 = average_by_mouse(np.nansum(sf_cv_iti_10[~session_has_spontaneous], axis=1), results.mouse_names[~session_has_spontaneous])
    sf_cv_total_spont = average_by_mouse(np.nansum(sf_cv_iti[session_has_spontaneous], axis=1), results.mouse_names[session_has_spontaneous])
    sf_cv_total_spont_10 = average_by_mouse(np.nansum(sf_cv_iti_10[session_has_spontaneous], axis=1), results.mouse_names[session_has_spontaneous])

    # Average curves by mouse
    sf_cv = average_by_mouse(sf_cv, results.mouse_names)
    ff = average_by_mouse(ff, results.mouse_names)

    return dict(
        sf_cv=sf_cv,
        ff=ff,
        sf_cv_total=sf_cv_total,
        sf_cv_total_10=sf_cv_total_10,
        sf_cv_total_iti=sf_cv_total_iti,
        sf_cv_total_iti_10=sf_cv_total_iti_10,
        sf_cv_total_spont=sf_cv_total_spont,
        sf_cv_total_spont_10=sf_cv_total_spont_10,
    )


def _plot_ratios_spectrum(ax, arrays: dict[str, np.ndarray], fontsize: int = 9) -> None:
    """Left panel of the ratios figure: mouse-averaged normalized ``sf_cv`` / ``ff`` spectra (log-log)."""
    sf_cv, ff = arrays["sf_cv"], arrays["ff"]
    ylim_min = -5.5
    ylim_max = -0.8
    yline = -5.25
    sf_color = CONDITION_COLORS["behaving"]
    ff_color = "black"
    each_alpha = 0.3

    def xvals(x):
        return np.arange(x.shape[1]) + 1

    ax.plot(xvals(sf_cv), sf_cv.T, color=sf_color, alpha=each_alpha, linewidth=1.0)
    ax.plot(xvals(ff), ff.T, color=ff_color, alpha=each_alpha, linewidth=1.0)
    ax.plot(xvals(sf_cv), np.nanmean(sf_cv, axis=0), color=sf_color, label="PF Structure", linewidth=2.0)
    ax.plot(xvals(ff), np.nanmean(ff, axis=0), color=ff_color, label="Reliable CA1", linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ylim = (10**ylim_min, 10**ylim_max)
    yticks = ax.get_yticks()
    ytick_power = [np.log10(yt) for yt in yticks]
    ax.set_yticks(yticks, labels=ytick_power)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Shared Dimension")
    ax.set_ylabel("Variance")
    xlim = ax.get_xlim()
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=[1, xlim[1]],
        ybounds=[ylim[0], ylim[1]],
    )
    ax.legend(loc="upper right", fontsize=fontsize, frameon=False)
    xticks = ax.get_xticks()
    xticks = xticks[xticks >= 1]
    ax.set_xticks(xticks)
    ax.annotate(
        "",
        xy=(10, 10**yline),
        xytext=(1, 10**yline),
        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0),
        annotation_clip=False,
    )
    ax.text(np.sqrt(10), 10 ** (yline + 0.1), "1st 10", fontsize=fontsize, ha="center", va="bottom")


def _plot_ratios_beeswarms(ax1, ax2, arrays: dict[str, np.ndarray], fontsize: int = 9) -> None:
    """Right two panels of the ratios figure: beeswarm of cumulative variance ratio in the first 10 dims (``ax1``) and all dims (``ax2``)."""
    xticks = [0, 1, 2]
    xticklabels = ["Behaving", "w/ ITIs", "w/ Spont"]

    beewidth = 0.2
    alpha = 0.3
    line_extent = np.array([-0.25, 0.25])
    np1 = np.array([1, 1])
    linewidth = 2.0
    color_behaving = CONDITION_COLORS["behaving"]
    color_itis = CONDITION_COLORS["itis"]
    color_spontaneous = CONDITION_COLORS["spontaneous"]

    def _swarm(ax, x, values, color):
        ax.plot(
            x + beewidth * beeswarm(values),
            values,
            color=color,
            linestyle="none",
            linewidth=0.5,
            marker="o",
            markersize=3,
            alpha=alpha,
        )

    _swarm(ax1, xticks[0], arrays["sf_cv_total_10"], color_behaving)
    _swarm(ax1, xticks[1], arrays["sf_cv_total_iti_10"], color_itis)
    _swarm(ax1, xticks[2], arrays["sf_cv_total_spont_10"], color_spontaneous)
    _swarm(ax2, xticks[0], arrays["sf_cv_total"], color_behaving)
    _swarm(ax2, xticks[1], arrays["sf_cv_total_iti"], color_itis)
    _swarm(ax2, xticks[2], arrays["sf_cv_total_spont"], color_spontaneous)

    ax1.plot(xticks[0] + line_extent, np1 * np.nanmean(arrays["sf_cv_total_10"]), color=color_behaving, linewidth=linewidth)
    ax1.plot(xticks[1] + line_extent, np1 * np.nanmean(arrays["sf_cv_total_iti_10"]), color=color_itis, linewidth=linewidth)
    ax1.plot(xticks[2] + line_extent, np1 * np.nanmean(arrays["sf_cv_total_spont_10"]), color=color_spontaneous, linewidth=linewidth)
    ax2.plot(xticks[0] + line_extent, np1 * np.nanmean(arrays["sf_cv_total"]), color=color_behaving, linewidth=linewidth)
    ax2.plot(xticks[1] + line_extent, np1 * np.nanmean(arrays["sf_cv_total_iti"]), color=color_itis, linewidth=linewidth)
    ax2.plot(xticks[2] + line_extent, np1 * np.nanmean(arrays["sf_cv_total_spont"]), color=color_spontaneous, linewidth=linewidth)
    ax1.set_ylim(-0.12, 1.10)
    ax1.set_xlim(-0.5, max(xticks) + 0.5)
    ax1.set_ylabel("Variance Ratio")
    yline_ratio = -0.10
    ax1.annotate(
        "",
        xy=(xticks[0], yline_ratio),
        xytext=(xticks[2], yline_ratio),
        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0),
        annotation_clip=False,
    )
    ax1.text(np.mean(xticks), yline_ratio + 0.02, "1st 10", fontsize=fontsize, ha="center", va="bottom")
    ax2.annotate(
        "",
        xy=(xticks[0], yline_ratio),
        xytext=(xticks[2], yline_ratio),
        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0),
        annotation_clip=False,
    )
    ax2.text(np.mean(xticks), yline_ratio + 0.02, "All", fontsize=fontsize, ha="center", va="bottom")
    format_spines(
        ax1,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=[0, max(xticks)],
        ybounds=[0, 1],
        yticks=[0, 0.5, 1.0],
    )
    format_spines(
        ax2,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["bottom"],
        xbounds=[0, max(xticks)],
    )
    ax2.tick_params(axis="y", left=False, labelleft=False)
    ax1.set_xticks(xticks, labels=xticklabels, rotation=45, ha="right")
    ax2.set_xticks(xticks, labels=xticklabels, rotation=45, ha="right")


def _plot_ratios_beeswarms_combined(ax, arrays: dict[str, np.ndarray], fontsize: int = 9) -> None:
    """Both beeswarm groups (1st 10 dims, all dims) on one axis, for the composite figure.

    x = 0,1,2 is the 1st-10-dims group (Behaving/w ITIs/w Spont), x = 3,4,5 is the all-dims group,
    same order. Only the group centers (x=1, x=4) get tick labels ("1st 10" / "All") -- the
    per-color condition labels aren't repeated here since the neighboring familiarity legend
    already maps these same colors to Behaving/w ITIs/w Spont.
    """
    beewidth = 0.2
    alpha = 0.3
    line_extent = np.array([-0.25, 0.25])
    np1 = np.array([1, 1])
    linewidth = 2.0
    color_behaving = CONDITION_COLORS["behaving"]
    color_itis = CONDITION_COLORS["itis"]
    color_spontaneous = CONDITION_COLORS["spontaneous"]

    def _swarm(x, values, color):
        ax.plot(
            x + beewidth * beeswarm(values),
            values,
            color=color,
            linestyle="none",
            linewidth=0.5,
            marker="o",
            markersize=3,
            alpha=alpha,
        )

    def _mean_line(x, values, color):
        ax.plot(x + line_extent, np1 * np.nanmean(values), color=color, linewidth=linewidth)

    groups = {
        0: ("sf_cv_total_10", "sf_cv_total_iti_10", "sf_cv_total_spont_10"),
        3: ("sf_cv_total", "sf_cv_total_iti", "sf_cv_total_spont"),
    }
    for offset, (behaving_key, itis_key, spont_key) in groups.items():
        _swarm(offset + 0, arrays[behaving_key], color_behaving)
        _swarm(offset + 1, arrays[itis_key], color_itis)
        _swarm(offset + 2, arrays[spont_key], color_spontaneous)
        _mean_line(offset + 0, arrays[behaving_key], color_behaving)
        _mean_line(offset + 1, arrays[itis_key], color_itis)
        _mean_line(offset + 2, arrays[spont_key], color_spontaneous)

    ax.set_ylabel("Variance Ratio")
    xmax = max(groups) + 2
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=[0, xmax],
        ybounds=[0, 1],
        yticks=[0, 0.5, 1.0],
    )
    ax.set_xlim(-0.5, xmax + 0.5)
    ax.set_xticks([1, 4], labels=["1st 10", "All"])


class SubspaceCurvesRatiosViewer(Viewer):
    """Placefield-component vs. reliable-CA1 variance spectra, plus cumulative-variance ratios.

    Left panel: mouse-averaged normalized ``sf_cv`` (placefield component) and ``ff`` (reliable
    CA1) spectra vs. shared dimension (log-log), each mouse faint plus the cross-mouse mean bold.
    Right panel: beeswarm of the fraction of total variance captured in the first 10 dims and in
    all dims, split by ``Behaving`` / ``w/ ITIs`` / ``w/ Spont`` (ITI and spontaneous sessions are
    mutually exclusive subsets of the session set; ``Behaving`` uses every session).
    """

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (5.0, 3.0)):
        self.results = results
        self.figsize = figsize
        self._tuple_labels = _register_stimspace_selections(self, results)

    def encode_param(self, name: str, value):
        """Map a raw param value to its widget value (tuple -> string label; else unchanged)."""
        if name in self._tuple_labels and isinstance(value, tuple):
            return _tuple_label(value)
        return value

    def _sel_params(self, state: dict) -> dict:
        """Params to forward to ``results.sel``, decoding tuple labels back to tuples."""
        return _stimspace_sel_params(self.results, self._tuple_labels, state)

    def plot(self, state: dict):
        sel_params = self._sel_params(state)
        arrays = _ratios_arrays(self.results, sel_params)

        fontsize = 9
        plt.rcParams["font.size"] = fontsize

        fig, ax = plt.subplots(1, 3, figsize=self.figsize, layout="constrained", width_ratios=[1, 0.3, 0.3])
        fig.get_layout_engine().set(wspace=0.02, w_pad=0.00)
        ax[2].sharex(ax[1])
        ax[2].sharey(ax[1])

        _plot_ratios_spectrum(ax[0], arrays, fontsize)
        _plot_ratios_beeswarms(ax[1], ax[2], arrays, fontsize)
        return fig


def subspace_curves_ratios(
    results: ResultsAggregator,
    figsize: tuple[float, float] = (5.0, 3.0),
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Placefield-component vs. reliable-CA1 variance spectra, plus cumulative-variance ratios.

    Left panel shows the mouse-averaged normalized ``sf_cv`` (placefield component) and ``ff``
    (reliable CA1) spectra against shared dimension (log-log), each mouse faint plus the
    cross-mouse mean bold. Right panel shows a beeswarm of the fraction of total variance
    captured in the first 10 dimensions and across all dimensions, split by ``Behaving``
    (every session), ``w/ ITIs`` (non-spontaneous sessions with ITIs included), and ``w/ Spont``
    (sessions with a genuine spontaneous window, ITIs included).

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``StimSpaceSpectraConfig`` results providing ``param_axes``, ``sel``,
        ``mouse_names`` and ``sessions``.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections (``activity_parameters_name``,
        ``smooth_widths``, ``reliability_fraction_active_thresholds``). Tuple-valued axes
        (e.g. ``smooth_widths=(5.0, None)``) are passed as native tuples; they are encoded to
        the widget's string labels internally.

    Returns
    -------
    matplotlib.figure.Figure or SubspaceCurvesRatiosViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = SubspaceCurvesRatiosViewer(results, figsize=figsize)
    valid_keys = [k for k in results.param_axes if k != "include_iti"]
    for key, value in selections.items():
        if key not in valid_keys:
            raise ValueError(f"Unknown selection {key!r}. Options: {valid_keys}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Helpers for complete_spectrum_figure
# ---------------------------------------------------------------------------

# 1x4 subplots width ratios: ax[0] (spectrum) ~matches the ratios figure's own [1, 0.3, 0.3] split
# scaled to its default figsize; ax[1] (combined beeswarm, 6 x-positions) is the other two of
# those three; ax[2] and ax[3] are each a single Variance Ratio panel (no Total Variance).
_COMPLETE_SPECTRUM_WIDTH_RATIOS = (1.5, 1.2, 1.0, 1.0)


class CompleteSpectrumViewer(Viewer):
    """Composite figure: ratios spectrum+beeswarm alongside both familiarity Variance Ratio panels.

    A plain 1x4 ``subplots``. ``ax[0]`` is the PF-structure-vs-reliable-CA1 spectrum panel from
    :func:`subspace_curves_ratios`. ``ax[1]`` combines that same figure's two cumulative-variance-
    ratio beeswarm groups onto one axis (1st 10 dims at x=0-2, all dims at x=3-5, same
    Behaving/w ITIs/w Spont color order in each); only the group centers (x=1, x=4) get tick labels
    ("1st 10" / "All") -- the per-color condition labels aren't repeated here since the
    neighboring familiarity legend (``ax[2]``, in ``plot_mode="all"``) already maps these same
    colors to Behaving/w ITIs/w Spont. ``ax[2]`` is the whole-session familiarity Variance Ratio
    panel from :func:`subspace_familiarity` (``plot_mode="all"``; Total Variance omitted). ``ax[3]``
    is the per-env-experience-slot familiarity Variance Ratio panel (``plot_mode="by_env"``; Total
    Variance omitted). ``ax[1]``, ``ax[2]`` and ``ax[3]`` share one y-axis, fixed to ``[0, 1]`` and
    labeled "Shared Variance Ratio" once, on the beeswarm panel; the other two hide their y
    spine/ticks/label. All four panels also share one set of ``StimSpaceSpectraConfig`` param-axis
    selections, since all three source figures read from the same aggregated results.
    """

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (14.0, 3.0)):
        self.results = results
        self.figsize = figsize
        self._tuple_labels = _register_stimspace_selections(self, results)

        # gs[2]: familiarity plot_mode="all" knobs (env_full_scope/full_within_env don't apply here).
        self.add_boolean("within_condition", value=False)
        self.add_selection("style_all", options=["errorPlot", "all"], value="errorPlot")

        # gs[3]: familiarity plot_mode="by_env" knobs (within_condition doesn't apply here).
        self.add_selection("env_full_scope", options=_ENV_FULL_SCOPES, value="within_env")
        self.add_boolean("full_within_env", value=False)
        self.add_selection("style_by_env", options=["errorPlot", "all"], value="errorPlot")

        self.add_float("fontsize", value=9.0, min=4.0, max=24.0, step=0.5)

    def encode_param(self, name: str, value):
        """Map a raw param value to its widget value (tuple -> string label; else unchanged)."""
        if name in self._tuple_labels and isinstance(value, tuple):
            return _tuple_label(value)
        return value

    def _sel_params(self, state: dict) -> dict:
        """Params to forward to ``results.sel``, decoding tuple labels back to tuples."""
        return _stimspace_sel_params(self.results, self._tuple_labels, state)

    def plot(self, state: dict):
        sel_params = self._sel_params(state)
        fontsize = state["fontsize"]
        plt.rcParams["font.size"] = fontsize

        fig, ax = plt.subplots(1, 4, figsize=self.figsize, layout="constrained", width_ratios=_COMPLETE_SPECTRUM_WIDTH_RATIOS)

        ratios_arrays = _ratios_arrays(self.results, sel_params)
        _plot_ratios_spectrum(ax[0], ratios_arrays, fontsize)
        _plot_ratios_beeswarms_combined(ax[1], ratios_arrays, fontsize)

        ax[2].sharey(ax[1])
        curves_all = _familiarity_curves(self.results, sel_params, "all", within_condition=state["within_condition"])
        _render_familiarity_ratio_panel(ax[2], curves_all, "all", state["style_all"], fontsize)

        ax[3].sharey(ax[1])
        curves_by_env = _familiarity_curves(
            self.results,
            sel_params,
            "by_env",
            env_full_scope=state["env_full_scope"],
            full_within_env=state["full_within_env"],
        )
        _render_familiarity_ratio_panel(ax[3], curves_by_env, "by_env", state["style_by_env"], fontsize)

        # ax[1]-ax[3] share one y-axis: fixed [0, 1] range, spine/ticks/label shown once (on the
        # beeswarm panel), the other two hidden. Setting ylim on any one member of a sharey group
        # re-syncs the whole group, so this override wins regardless of whatever each panel's own
        # helper set internally.
        ax[1].set_ylim(0, 1)
        ax[1].set_ylabel("Shared Variance Ratio")
        for a in (ax[2], ax[3]):
            a.set_ylabel("")
            a.spines["left"].set_visible(False)
            a.tick_params(axis="y", left=False, labelleft=False)

        return fig


def complete_spectrum_figure(
    results: ResultsAggregator,
    within_condition: bool = False,
    style_all: str = "errorPlot",
    env_full_scope: str = "within_env",
    full_within_env: bool = False,
    style_by_env: str = "errorPlot",
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (14.0, 3.0),
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Composite figure combining the ratios spectrum+beeswarm with both familiarity Variance Ratio panels.

    A plain 1x4 ``subplots``: ``ax[0]`` is the PF-structure-vs-reliable-CA1 spectrum panel (as in
    :func:`subspace_curves_ratios`); ``ax[1]`` combines that figure's two cumulative-variance-ratio
    beeswarm groups onto one axis (1st 10 dims at x=0-2, all dims at x=3-5), with tick labels only
    at the group centers ("1st 10" / "All") -- the Behaving/w ITIs/w Spont color coding is explained
    by ``ax[2]``'s legend instead of being repeated here; ``ax[2]`` is the whole-session familiarity
    Variance Ratio panel (as in :func:`subspace_familiarity` with ``plot_mode="all"``, Total
    Variance omitted); ``ax[3]`` is the per-env-experience-slot familiarity Variance Ratio panel
    (``plot_mode="by_env"``, Total Variance omitted). ``ax[1]``, ``ax[2]`` and ``ax[3]`` share one
    y-axis fixed to ``[0, 1]``, labeled "Shared Variance Ratio" once on the beeswarm panel.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``StimSpaceSpectraConfig`` results providing ``param_axes``, ``sel``,
        ``mouse_names`` and ``sessions``. Shared by all four panels.
    within_condition : bool
        Only affects ``ax[2]`` (whole-session familiarity, ``plot_mode="all"``). If True, each
        curve's x-axis is the session index within its own kept subset; if False (default), bins
        track the mouse's overall chronological session index.
    style_all : {"errorPlot", "all"}
        Rendering style for ``ax[2]`` (whole-session familiarity panel). ``"errorPlot"`` shows
        the mouse mean +/- SE band; ``"all"`` also shows every mouse's faint curve.
    env_full_scope : {"within_env", "outside_env", "with_iti", "with_spontaneous"}
        Only affects ``ax[3]`` (per-env familiarity, ``plot_mode="by_env"``). Selects which per-env
        key pairing / ITI condition to plot (see :func:`subspace_familiarity`).
    full_within_env : bool
        Only affects ``ax[3]``. If True, the total-variance denominator is the env-only variance;
        if False (default), it's the whole-session variance. (Affects the ratio's normalization
        even though the Total Variance panel itself is omitted.)
    style_by_env : {"errorPlot", "all"}
        Rendering style for ``ax[3]`` (per-env familiarity panel), same options as ``style_all``.
    fontsize : float
        Font size (points) shared by every panel: axis labels, tick labels, legends and inline
        annotations, via ``plt.rcParams["font.size"]`` plus explicit ``fontsize=`` passes.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections (e.g. ``smooth_widths``,
        ``activity_parameters_name``, ``reliability_fraction_active_thresholds``), shared by all
        four panels. Tuple-valued axes are passed as native tuples; they are encoded to the
        widget's string labels internally.

    Returns
    -------
    matplotlib.figure.Figure or CompleteSpectrumViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = CompleteSpectrumViewer(results, figsize=figsize)
    valid_keys = [k for k in results.param_axes if k != "include_iti"]
    for key, value in selections.items():
        if key not in valid_keys:
            raise ValueError(f"Unknown selection {key!r}. Options: {valid_keys}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))
    viewer.update_boolean("within_condition", value=within_condition)
    viewer.update_selection("style_all", value=style_all)
    viewer.update_selection("env_full_scope", value=env_full_scope)
    viewer.update_boolean("full_within_env", value=full_within_env)
    viewer.update_selection("style_by_env", value=style_by_env)
    viewer.update_float("fontsize", value=fontsize)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Helpers for placefield_structure_over_time
# ---------------------------------------------------------------------------

# Per-cell feature keys produced by ``PlaceFieldStructureConfig._compute_pf_features`` that are
# pickable as plot curves (excludes ``env_slot_ids``, which has no neuron axis; excludes
# ``reliability``/``fraction_active``, which are used as per-ROI inclusion thresholds instead —
# see ``_pf_env_curves``).
_PF_FEATURE_KEYS = [
    "pf_mean",
    "pf_var",
    "pf_norm",
    "pf_max",
    "pf_cv",
    "pf_gauss_amp",
    "pf_gauss_center",
    "pf_gauss_width",
    "pf_gauss_r2",
    "spatial_participation",
    "pf_tdot_mean",
    "pf_tdot_std",
    "pf_tdot_cv",
    "pf_tcorr_mean",
    "pf_tcorr_std",
]


def _oom_bucket(scale: float) -> int:
    """Order-of-magnitude bucket for a positive value scale: 0 covers ``(0, 1]``, 1 covers ``(1, 10]``, etc."""
    if not np.isfinite(scale) or scale <= 0:
        return 0
    return max(0, int(np.ceil(np.log10(scale))))


def _zscore_curve(values: np.ndarray) -> np.ndarray:
    """NaN-aware zscore of a 1D curve; returns it unchanged if empty or constant."""
    if values.size == 0:
        return values
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values - mean
    return (values - mean) / std


def _pf_env_curves(
    results: ResultsAggregator,
    sel_params: dict,
    keys: list[str],
    env_slot: int,
    reliability_threshold: float = -1.0,
    fraction_active_threshold: float = 0.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-mouse, env-session-indexed curves for each feature key at one env-experience slot.

    ROIs are filtered *within each session* first (``reliability >= reliability_threshold`` and
    ``fraction_active >= fraction_active_threshold``), then averaged across the surviving ROIs to
    get one value per session. Each mouse's sessions are then reindexed to the chronological
    session number *within* this env slot (0 = the mouse's first session containing this
    environment) rather than the mouse's overall session number — the same ``by_env`` x-axis
    convention used by :func:`subspace_familiarity`. A session counts as present for this slot
    when ``pf_mean`` (always computed whenever the slot has data, independent of the ROI filter)
    is finite for at least one neuron; a present session with zero surviving ROIs still keeps its
    slot in the sequence, just with a NaN value.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``{feature_key: {mouse: 1D array of env-session-indexed values}}``.
    """
    fetch_keys = list(dict.fromkeys(keys + ["pf_mean", "reliability", "fraction_active"]))
    out = results.sel(keys=fetch_keys, squeeze_ones=False, avg_by_mouse=False, **sel_params)
    presence = np.isfinite(out["pf_mean"][:, env_slot, :]).any(axis=1)
    roi_mask = (out["reliability"][:, env_slot, :] >= reliability_threshold) & (out["fraction_active"][:, env_slot, :] >= fraction_active_threshold)

    curves: dict[str, dict[str, np.ndarray]] = {}
    for key in keys:
        values = np.nanmean(np.where(roi_mask, out[key][:, env_slot, :], np.nan), axis=1)
        per_mouse = {}
        for mouse in results.unique_mice:
            idx_sorted = _chronological_mouse_sessions(results, mouse, exclude_bad_envs=False)
            mouse_keep = presence[idx_sorted]
            per_mouse[mouse] = values[idx_sorted[mouse_keep]]
        curves[key] = per_mouse
    return curves


class PlaceFieldStructureOverTimeViewer(Viewer):
    """Per-cell placefield feature values across sessions of one env-experience slot.

    The x-axis is each mouse's chronological session index *within* the chosen env slot
    (position within env, following ``subspace_familiarity``'s ``by_env`` indexing), not the
    mouse's overall session number. Selected feature keys (population mean over neurons per
    session) are overlaid, each drawn in a fixed color from a large colormap and labeled.
    Panels are auto-split by order of magnitude, so e.g. a ``[0, 1]``-ranged feature and a
    ``[0, 10]``-ranged one (like ``pf_cv``) don't get flattened onto the same axis. ``zscore``
    (fed through before that OOM split) zscores each mouse's curve individually, showing relative
    variation across sessions instead of absolute values — useful for comparing shape across
    features/mice with very different baselines/scales. ``reliability_threshold`` and
    ``fraction_active_threshold`` filter which ROIs are included in each session's population
    mean (see :func:`_pf_env_curves`), rather than being plottable curves themselves.
    """

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (8.0, 3.0)):
        self.results = results
        self.figsize = figsize

        for key, value in results.param_axes.items():
            self.add_selection(key, options=value)

        preferred_state = {
            "smooth_width": None,
            "activity_parameters_name": "default",
        }
        for key, value in preferred_state.items():
            if key in results.param_axes:
                self.update_selection(key, value=value)

        self.add_integer("env_slot", value=0, min=0, max=MAX_ENV_SLOTS - 1)
        self.add_multiple_selection("feature_keys", options=list(_PF_FEATURE_KEYS), value=["pf_mean"])
        self.add_selection("style", options=["all", "errorPlot"], value="all")
        self.add_boolean("zscore", value=False)
        self.add_float("reliability_threshold", value=-1.0, min=-1.0, max=1.0, step=0.001)
        self.add_float("fraction_active_threshold", value=0.0, min=0.0, max=1.0, step=0.001)

        cmap = plt.get_cmap("gist_ncar")
        n = len(_PF_FEATURE_KEYS)
        self._colors = {key: cmap(i / max(n - 1, 1))[:3] for i, key in enumerate(_PF_FEATURE_KEYS)}

    def _sel_params(self, state: dict) -> dict:
        return {k: v for k, v in state.items() if k in self.results.param_axes}

    def plot(self, state: dict):
        sel_params = self._sel_params(state)
        keys = list(state["feature_keys"]) or [_PF_FEATURE_KEYS[0]]
        env_slot = state["env_slot"]
        curves = _pf_env_curves(
            self.results,
            sel_params,
            keys,
            env_slot,
            reliability_threshold=state["reliability_threshold"],
            fraction_active_threshold=state["fraction_active_threshold"],
        )

        if state["zscore"]:
            curves = {key: {mouse: _zscore_curve(values) for mouse, values in per_mouse.items()} for key, per_mouse in curves.items()}

        buckets: dict[int, list[str]] = {}
        for key in keys:
            stack = _pad_stack_by_mouse(curves[key])
            finite = stack[np.isfinite(stack)]
            scale = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
            buckets.setdefault(_oom_bucket(scale), []).append(key)
        bucket_ids = sorted(buckets)

        fig, axes = plt.subplots(1, len(bucket_ids), figsize=self.figsize, layout="constrained", squeeze=False)
        axes = axes[0]

        for ax, bucket in zip(axes, bucket_ids):
            for key in buckets[bucket]:
                stack = _pad_stack_by_mouse(curves[key])
                length = _support_length(stack)
                stack = stack[:, :length]
                xvals = np.arange(length)
                color = self._colors[key]
                if state["style"] == "all":
                    for values in curves[key].values():
                        ax.plot(np.arange(len(values)), values, color=color + (0.3,), linewidth=0.5)
                    ax.plot(xvals, _mean_with_min_support(stack), color=color, linewidth=2.0, label=key)
                elif length:
                    errorPlot(xvals, stack, axis=0, se=True, ax=ax, color=color, linewidth=2.0, label=key, alpha=0.25)
            ax.set_xlabel("Env Session #")
            ax.set_ylabel(f"Value (OOM ~1e{bucket})")
            ax.legend(fontsize=8, frameon=False)
            format_spines(ax, x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"])
        return fig


def placefield_structure_over_time(
    results: ResultsAggregator,
    env_slot: int = 0,
    feature_keys: list[str] | None = None,
    style: str = "all",
    zscore: bool = False,
    reliability_threshold: float = -1.0,
    fraction_active_threshold: float = 0.0,
    figsize: tuple[float, float] = (8.0, 3.0),
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Per-cell placefield feature values across sessions of one env-experience slot.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``PlaceFieldStructureConfig`` results providing ``param_axes``, ``sel``,
        ``unique_mice``, ``mouse_names`` and ``sessions``.
    env_slot : int
        Env-experience-order slot (0-indexed, ``< MAX_ENV_SLOTS``) to plot.
    feature_keys : list of str, optional
        Feature keys to overlay (see ``_PF_FEATURE_KEYS`` for the full set). Defaults to
        ``["pf_mean"]``.
    style : {"all", "errorPlot"}
        ``"all"`` plots every mouse's curve as a faint line plus the mouse-mean as a bold line.
        ``"errorPlot"`` shows only the mouse mean +/- SE as a shaded band.
    zscore : bool
        If True, zscore each mouse's curve individually (before the OOM panel split), showing
        relative variation across sessions rather than absolute values.
    reliability_threshold : float
        Minimum per-ROI ``reliability`` (range ``[-1, 1]``) required for a session's ROI to count
        toward that session's population mean. Filtering happens within each session, before
        averaging across surviving ROIs.
    fraction_active_threshold : float
        Minimum per-ROI ``fraction_active`` (range ``[0, 1]``) required, same filtering rule as
        ``reliability_threshold``.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections (``activity_parameters_name``,
        ``smooth_width``). Each key must be a valid ``results.param_axes`` name.

    Returns
    -------
    matplotlib.figure.Figure or PlaceFieldStructureOverTimeViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = PlaceFieldStructureOverTimeViewer(results, figsize=figsize)
    for key, value in selections.items():
        if key not in results.param_axes:
            raise ValueError(f"Unknown selection {key!r}. Options: {list(results.param_axes)}")
        viewer.update_selection(key, value=value)
    viewer.update_integer("env_slot", value=env_slot)
    if feature_keys is not None:
        viewer.update_multiple_selection("feature_keys", value=list(feature_keys))
    viewer.update_selection("style", value=style)
    viewer.update_boolean("zscore", value=zscore)
    viewer.update_float("reliability_threshold", value=reliability_threshold)
    viewer.update_float("fraction_active_threshold", value=fraction_active_threshold)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig
