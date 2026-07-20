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
        self.add_selection("curve_smooth_kind", options=["none", "boxcar", "gaussian", "median"], value="none")
        self.add_float("curve_smooth_width", value=3.0, min=0.0, max=50.0, step=0.5)
        self.add_float("kink_threshold", value=0.95, min=0.0, max=1.0, step=0.01)
        self.add_selection("distribution_metric", options=["gini", "weighted_missing", "missing_structure"], value="weighted_missing")

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
            distribution_metric = _gini(energy_on_full, axis=1)
            distribution_label = "Gini Equality"
        elif state["distribution_metric"] == "weighted_missing":
            _numerator = np.where(valid_full_dims, (1 - energy_on_full) * variance_activity, np.nan)
            _denominator = np.where(valid_full_dims, variance_activity, np.nan)
            distribution_metric = np.nansum(_numerator, axis=1) / np.nansum(_denominator, axis=1)
            distribution_label = "Weighted Missing Structure"
        else:
            distribution_metric = np.nanmean(np.where(valid_full_dims, 1 - energy_on_full, np.nan), axis=1)
            distribution_label = "Missing Structure"

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

        line_alpha = 0.3
        fig, ax = plt.subplots(1, 4, figsize=self.figsize, layout="constrained")

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
        ax[0].imshow(imshow_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=1, extent=extent)
        ax[0].set_xlabel("Placefield Dim.")
        ax[0].set_ylabel("Full Dim.")
        ax[0].set_xlim(xlims0)
        ax[0].set_ylim(ylims0)

        xvals = np.arange(panel_fraction_avg.shape[1]) + 1
        xbounds1 = [1, panel_fraction_avg.shape[1] + 1]
        mean_curve = np.nanmean(panel_fraction_avg, axis=0)
        ax[1].plot(xvals, panel_fraction_avg.T, color=("k", line_alpha), linewidth=0.5)
        ax[1].plot(xvals, mean_curve, color="k", linewidth=2.0)
        ax[1].set_xlabel("Full Dim.")
        ax[1].set_ylabel(panel_ylabel)
        ax[1].set_xscale("log")
        ymax = 1.0
        finite_curve = mean_curve[np.isfinite(mean_curve)]
        if state["weighted"] and finite_curve.size:
            ymax = max(1.0, float(finite_curve.max()))
        ybounds1 = [0, ymax]
        ax[1].set_xlim(xbounds1)
        ax[1].set_ylim(ybounds1)
        xticks = [1, 10, 100, 1000]

        session_xvals = np.arange(max_energy_by_session.shape[1])

        def plot_session_metric(axis, values, color, label):
            axis.plot(session_xvals, values.T, color=(color, line_alpha), linewidth=0.5)
            valid_count = np.sum(np.isfinite(values), axis=0)
            value_sum = np.nansum(values, axis=0)
            mean_values = np.divide(
                value_sum,
                valid_count,
                out=np.full_like(value_sum, np.nan, dtype=float),
                where=valid_count > 0,
            )
            mean_values[valid_count <= 1] = np.nan
            axis.plot(session_xvals, mean_values, color=color, linewidth=2.0, label=label)

        plot_session_metric(ax[2], max_energy_by_session, "k", "Max Energy")
        plot_session_metric(ax[2], distribution_by_session, "b", distribution_label)
        ax[2].set_xlabel("Session Index")
        ax[2].set_ylabel("Metric Value")
        ax[2].legend(fontsize=8, loc="best")

        plot_session_metric(ax[3], kink_by_session, "r", "Kink Position")
        ax[3].set_xlabel("Session Index")
        ax[3].set_ylabel("Kink Position")
        ax[3].legend(fontsize=8, loc="best")

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
            xticks=xticks,
            yticks=[0, round(ymax, 2)],
        )
        format_spines(ax[2], x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"])
        format_spines(ax[3], x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"])
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
        self.add_selection("distribution_metric", options=["gini", "missing_structure"], value="gini")

        self.on_change("mouse", self.update_mouse)
        self.update_mouse(self.state)

    def update_mouse(self, state):
        n_sess = int(np.sum(self.results.mouse_names == state["mouse"]))
        self.update_integer("session", max=max(n_sess - 1, 0))

    def _mouse_cross(self, state):
        """Cross matrices for the selected mouse, shape (n_sess_mouse, n_full, n_pf)."""
        _sel_state = {k: v for k, v in state.items() if k in self.results.param_axes}
        return self.results.sel(mouse=state["mouse"], squeeze_ones=False, **_sel_state)["cross"]

    def _mouse_session_ids(self, mouse):
        return [sid for sid, m in zip(self.results.session_ids, self.results.mouse_names) if m == mouse]

    def plot(self, state):
        mouse = state["mouse"]
        cross = self._mouse_cross(state)
        energy = cross**2
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
    distribution_metric : {"gini", "missing_structure"}
        Metric shown with max energy. Missing structure is the mean uncaptured
        fraction over valid full dimensions.
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
    curve_smooth_kind: str = "none",
    curve_smooth_width: float = 3.0,
    kink_threshold: float = 0.95,
    distribution_metric: str = "weighted_missing",
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
    curve_smooth_kind : {"none", "boxcar", "gaussian", "median"}
        Smoothing applied to energy-on-full curves before computing session metrics.
    curve_smooth_width : float
        Smoothing width in full-dimension units.
    kink_threshold : float
        Fraction of maximum energy used to locate the first below-threshold dimension.
    distribution_metric : {"gini", "missing_structure"}
        Metric shown with max energy. Missing structure is the mean uncaptured
        fraction over valid full dimensions.
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
    viewer.update_selection("curve_smooth_kind", value=curve_smooth_kind)
    viewer.update_float("curve_smooth_width", value=curve_smooth_width)
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

_FAMILIARITY_COLORS = {
    "Behaving": "k",
    "With ITIs": "blue",
    "With Spontaneous": "green",
    "Env #1": "k",
    "Env #2": "blue",
    "Env #3": "green",
}
_FAMILIARITY_XLABELS = {"env_session_number": "Env Session #", "overall_session_number": "Session #"}
_ENV_FULL_SCOPES = ["within_env", "outside_env", "with_iti", "with_spontaneous"]


def _tuple_label(value: tuple) -> str:
    """Render a tuple param value (elements are float or None) as a widget-safe string label."""
    return "-".join("None" if v is None else str(v) for v in value)


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
    x_axis_labels: list[str],
    exclude_bad_envs: bool,
) -> dict[str, dict]:
    """Turn ``{curve_label: (ratio, total, keep_mask)}`` session-level arrays into per-mouse curves.

    Shared tail end of :func:`_familiarity_curves` for both ``mode="all"`` (curve_label = ITI
    status) and ``mode="by_env"`` (curve_label = env slot): reorders each mouse's sessions
    chronologically and either densifies to valid-only sessions (``"env_session_number"``) or
    keeps every session with invalid ones set to NaN (``"overall_session_number"``).
    """
    unique_mice = results.unique_mice
    curves: dict[str, dict] = {label: {} for label in x_axis_labels}
    for curve_label, (ratio, total, keep) in curve_defs.items():
        for x_axis_label in x_axis_labels:
            svr_per_mouse, total_per_mouse = {}, {}
            for mouse in unique_mice:
                idx_sorted = _chronological_mouse_sessions(results, mouse, exclude_bad_envs=exclude_bad_envs)
                if x_axis_label == "env_session_number":
                    idx_use = idx_sorted[keep[idx_sorted]]
                    svr_per_mouse[mouse] = ratio[idx_use]
                    total_per_mouse[mouse] = total[idx_use]
                else:
                    svr_per_mouse[mouse] = np.where(keep[idx_sorted], ratio[idx_sorted], np.nan)
                    total_per_mouse[mouse] = np.where(keep[idx_sorted], total[idx_sorted], np.nan)
            curves[x_axis_label][curve_label] = {"svr": svr_per_mouse, "total": total_per_mouse}
    return curves


def _familiarity_curves(
    results: ResultsAggregator,
    sel_params: dict,
    mode: str,
    env_full_scope: str = "within_env",
    full_within_env: bool = True,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Per-mouse variance-ratio / total-variance curves, keyed by x-axis convention then curve label.

    Returns ``{x_axis_label: {curve_label: {"svr": {mouse: 1D array}, "total": {mouse: 1D array}}}}``.

    ``mode == "all"``: single x-axis (``"overall_session_number"``), whole-session ``sf_cv``/``ff``
    keys, curve labels ``"Behaving"`` / ``"With ITIs"`` / ``"With Spontaneous"`` overlaid together
    (unaffected by ``env_full_scope``).

    ``mode == "by_env"``: both x-axis conventions (``"env_session_number"`` and
    ``"overall_session_number"``). ``env_full_scope`` selects exactly one ``(sf_key, ff_key,
    include_iti, session_mask)`` combination — ``"within_env"`` (``sf_cv_env_full1`` /
    ``ff_env_full1``, env-only frames, no ITI variant exists), ``"outside_env"`` (``sf_cv_env_fullall``
    / ``ff_env_full1_fullall``, env-stim vs all-env-func, behaving-only), ``"with_iti"`` (same keys,
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
            "With ITIs": (ratio_i, total_i, valid_i & ~session_has_spontaneous),
            "With Spontaneous": (ratio_i, total_i, valid_i & session_has_spontaneous),
        }
        return _curves_from_defs(results, curve_defs, ["overall_session_number"], exclude_bad_envs=True)

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

    return _curves_from_defs(results, curve_defs, ["env_session_number", "overall_session_number"], exclude_bad_envs=False)


class SubspaceFamiliarityViewer(Viewer):
    """Variance-ratio-over-familiarity viewer: whole-session or per-env-experience-slot curves.

    ``plot_mode="all"`` reproduces the original whole-session figure (1x2: Variance Ratio,
    Total Variance, x-axis = overall session number). ``plot_mode="by_env"`` shows the same two
    metrics for every environment-experience slot overlaid together (colored black/blue/green
    for slot 0/1/2), with both candidate x-axis conventions side by side (2x2 grid): session
    number within that env vs. the mouse's overall session number. ``env_full_scope`` (only used
    in ``by_env``) picks which per-env key pairing / ITI condition to show.
    """

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (6.0, 3.0)):
        self.results = results
        self.figsize = figsize

        # Axes whose options are tuples (e.g. smooth_widths, reliability_fraction_active_thresholds)
        # can't be dropdown values directly, so they are encoded as string labels;
        # ``_tuple_labels[name]`` maps label -> original tuple (same pattern as figure4.py's
        # PlacefieldSpectraViewer).
        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for key, options in results.param_axes.items():
            if key == "include_iti":
                continue
            if any(isinstance(opt, tuple) for opt in options):
                label_map = {_tuple_label(opt): opt for opt in options}
                self._tuple_labels[key] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(key, options=widget_options)

        preferred_state = {
            "smooth_widths": (5.0, None),
            "activity_parameters_name": "default",
            "reliability_fraction_active_thresholds": (None, None),
        }
        for key, value in preferred_state.items():
            if key in results.param_axes:
                self.update_selection(key, value=self.encode_param(key, value))

        self.add_selection("plot_mode", options=["all", "by_env"], value="all")
        self.add_selection("env_full_scope", options=_ENV_FULL_SCOPES, value="within_env")
        self.add_boolean("full_within_env", value=True)
        self.add_selection("style", options=["errorPlot", "all"], value="all")

    def encode_param(self, name: str, value):
        """Map a raw param value to its widget value (tuple -> string label; else unchanged)."""
        if name in self._tuple_labels and isinstance(value, tuple):
            return _tuple_label(value)
        return value

    def _sel_params(self, state: dict) -> dict:
        """Params to forward to ``results.sel``, decoding tuple labels back to tuples."""
        params = {}
        for key, value in state.items():
            if key not in self.results.param_axes or key == "include_iti":
                continue
            if key in self._tuple_labels:
                value = self._tuple_labels[key][value]
            params[key] = value
        return params

    def _plot_panel(self, ax, axis_curves: dict, metric: str, xlabel: str, ylabel: str, style: str) -> float:
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
            elif length:
                errorPlot(np.arange(length), stack, axis=0, se=True, ax=ax, color=color, linewidth=2.0, label=curve_label, alpha=0.25)
            finite = stack[np.isfinite(stack)]
            if finite.size:
                max_val = max(max_val, float(finite.max()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return max_val

    def plot(self, state: dict):
        sel_params = self._sel_params(state)
        mode = state["plot_mode"]
        curves = _familiarity_curves(self.results, sel_params, mode, env_full_scope=state["env_full_scope"], full_within_env=state["full_within_env"])
        fontsize = 9
        plt.rcParams["font.size"] = fontsize

        style = state["style"]
        if mode == "all":
            fig, ax = plt.subplots(1, 2, figsize=self.figsize, layout="constrained")
            axis_curves = curves["overall_session_number"]
            xlabel = _FAMILIARITY_XLABELS["overall_session_number"]
            max_ratio = self._plot_panel(ax[0], axis_curves, "svr", xlabel, "Variance Ratio", style)
            self._plot_panel(ax[1], axis_curves, "total", xlabel, "Total Variance", style)
            ax[0].legend(loc="lower right", fontsize=fontsize, frameon=False, markerfirst=False)
            format_spines(ax[0], x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"], ybounds=[0, round(max_ratio, 1)])
            format_spines(ax[1], x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"])
        else:
            figsize = (self.figsize[0], self.figsize[1] * 2)
            fig, ax = plt.subplots(2, 2, figsize=figsize, layout="constrained")
            for col, x_axis_label in enumerate(["env_session_number", "overall_session_number"]):
                axis_curves = curves[x_axis_label]
                xlabel = _FAMILIARITY_XLABELS[x_axis_label]
                max_ratio = self._plot_panel(ax[0, col], axis_curves, "svr", xlabel, "Variance Ratio", style)
                self._plot_panel(ax[1, col], axis_curves, "total", xlabel, "Total Variance", style)
                format_spines(
                    ax[0, col],
                    x_pos=-0.02,
                    y_pos=-0.02,
                    spines_visible=["left", "bottom"],
                    ybounds=[0, round(max_ratio, 1)],
                )
                format_spines(ax[1, col], x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"])
            ax[0, 0].legend(loc="lower right", fontsize=fontsize, frameon=False, markerfirst=False)
            fig.suptitle(state["env_full_scope"], fontsize=fontsize)
        return fig


def subspace_familiarity(
    results: ResultsAggregator,
    plot_mode: str = "all",
    env_full_scope: str = "within_env",
    full_within_env: bool = True,
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
    spectra (``sf_cv`` / ``ff``); ``plot_mode="by_env"`` uses the per-environment-experience-slot
    spectra from ``StimSpaceSpectraConfig._per_env_spectra``, overlaying all ``MAX_ENV_SLOTS``
    experience-order slots together (black/blue/green for slot 0/1/2), with side-by-side panels
    for the two x-axis conventions (session number within that env vs. the mouse's overall
    session number).

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
    style : {"all", "errorPlot"}
        ``"all"`` plots every mouse's curve as a faint line plus the mouse-mean as a bold line
        (as before). ``"errorPlot"`` drops the per-mouse lines and instead shows the mouse
        mean +/- SE as a shaded band (via ``vrAnalysis.helpers.plotting.errorPlot``).
    figsize : tuple[float, float]
        Figure size in inches. Pass a larger size (e.g. ``(8.0, 6.0)``) for
        ``plot_mode="by_env"``, which renders a 2x2 grid instead of the 1x2 grid used by
        ``plot_mode="all"``.
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
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig
