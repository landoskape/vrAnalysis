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

        self.add_integer("idx_cross", value=0, min=0, max=len(results.sessions) - 1)
        self.add_integer("num_cross_show", value=25, min=1, max=100)
        self.add_boolean("plot_energy", value=True)
        self.add_boolean("weighted", value=False)

    def plot(self, state: dict):
        _sel_state = {k: v for k, v in state.items() if k in self.results.param_axes}
        cross = self.results.sel(**_sel_state)["cross"]
        energy = cross**2

        # Energy of PF dimensions on full space
        energy_on_full = np.nansum(energy, axis=2)
        energy_on_diagonal = np.cumsum(np.diagonal(energy, axis1=1, axis2=2), axis=1)
        energy_expanding_dims = np.diagonal(np.cumsum(np.cumsum(energy, axis=1), axis=2), axis1=1, axis2=2)

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
        fig, ax = plt.subplots(1, 2, figsize=self.figsize, layout="constrained")

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
        energy_on_full = _smooth_fraction(energy_on_full, state["curve_smooth_kind"], state["curve_smooth_width"])
        n_sess = cross.shape[0]
        sess = min(state["session"], n_sess - 1)

        fig, ax = plt.subplots(1, 2, figsize=self.figsize, layout="constrained")

        # Left: cross energy heat map of chosen session
        imshow_data = energy[sess][:100, :100]
        extent = [0, 100, 100, 0]
        ax[0].imshow(imshow_data, cmap="gray_r", aspect="auto", vmin=0, vmax=1, extent=extent)
        ax[0].set_xlabel("Placefield Dim.")
        ax[0].set_ylabel("Full Dim.")
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
        ax[1].set_ylabel("Fraction Full Variance\nCaptured By Placefields")
        ax[1].set_xscale("log")
        xbounds1 = [1, energy_on_full.shape[1] + 1]
        ybounds1 = [0, 1.0]
        ax[1].set_xlim(xbounds1)
        ax[1].set_ylim(ybounds1)

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
        return fig


def subspace_crossspace_per_mouse(
    results: ResultsAggregator,
    mouse: str | None = None,
    session: int = 0,
    num_cross_show: int = 30,
    curve_smooth_kind: str = "none",
    curve_smooth_width: float = 3.0,
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
    figsize: tuple[float, float] = (6.0, 3.0),
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Cross-spectrum energy figure for aggregated subspace results.

    The left panel shows an example placefield-vs-full cross matrix for one session
    (chosen by ``idx_cross`` from sessions sorted by descending mean top-10 diagonal
    energy). The right panel shows the mouse-averaged fraction of full-activity
    variance captured by placefields as a function of full-activity dimension.

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
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig
