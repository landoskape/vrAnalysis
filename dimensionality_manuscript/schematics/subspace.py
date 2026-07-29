"""Schematics for the shared-variance-ratio exposition (see ``dimensionality_manuscript/docs/shared_variance.md``).

Parked illustration ideas, not yet built here (revisit if the kappa-overlap panel needs a sequel):

1. Nested-ellipse Loewner proof. Draw the stim ellipse literally inside the full ellipse
   (Sigma_stim preceq Sigma_full) as a visual proof that SVR in [0, 1] when A is a variance
   subset of B.
2. Train/test reliability cartoon. Three small ellipses of the same shape (dotted = latent
   truth, solid = noisy train/test draws) motivating why kappa(train, test) != kappa(true, true)
   before the reader hits the cross-validation algebra.
3. Amplitude vs. energy bar comparison. Same pair of matrices, bar charts of sqrt(eigenvalues)
   (kappa/SVR scale) vs. raw eigenvalues (omega/cvSER/CKA scale), showing how the energy scale
   exaggerates the dominant mode relative to the amplitude scale.
4. Rotating alignment strip. 4-5 static frames sweeping the relative orientation of two
   ellipses from aligned to orthogonal, to build intuition that kappa tracks orientation
   agreement, not just size.
5. CKA vs. SVR denominator diagram. Small schematic fractions (icons, not formulas) contrasting
   SVR's reliability-based denominator (train-vs-test self overlap) against CKA's total-energy
   denominator (self-vs-self, same sample).
"""

from typing import Literal, Sequence, Tuple
from dataclasses import asdict, dataclass, field, fields, replace
import json
from pathlib import Path
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch
from syd import Viewer
from vrAnalysis.helpers.plotting import format_spines


def _get_covariance_ellipse(covariance: np.ndarray, n_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the coordinates of an ellipse representing the covariance matrix.

    Parameters
    ----------
    covariance : np.ndarray
        A 2x2 covariance matrix.
    n_std : float, optional
        The number of standard deviations to determine the ellipse's radii, by default 2.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The x and y coordinates of the ellipse.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_coords = np.array([np.cos(theta), np.sin(theta)])
    ellipse_coords = (eigenvectors @ np.diag(np.sqrt(eigenvalues) * n_std)) @ ellipse_coords

    return ellipse_coords[0], ellipse_coords[1]


@dataclass
class StimNuisance2D:
    """
    A dataclass representing a 2D stimulus nuisance variable, which is a function of two variables (e.g., position and speed).
    """

    name: str
    nuisance_direction: np.ndarray
    nuisance_amplitude: float
    stim_direction: np.ndarray = field(default_factory=lambda: np.array([-1.0, 1.0]))
    stim_amplitude: float = 1.0
    noise_amplitude: float = 0.1

    def __post_init__(self):
        # Normalize the stimulus and nuisance directions
        self.stim_direction = self.stim_direction / np.linalg.norm(self.stim_direction)
        self.nuisance_direction = self.nuisance_direction / np.linalg.norm(self.nuisance_direction)


def plot_stim_nuisance_2D(
    cfg: StimNuisance2D,
    ax: tuple[plt.Axes, ...],
    stim_color: str = "orange",
    nuisance_color: str = "black",
    extend_arrow_factor: float = 1.0,
    arrow_width: float = 0.5,
    arrow_length: float = 0.5,
    linewidth: float = 1.0,
    point_alpha: float = 0.5,
    point_size: float = 10.0,
    max_lim_factor: float = 4.5,
    n_samples: int = 1000,
) -> None:
    # unpack axes
    ax_direction, ax_stimcov, ax_fullcov, ax_value = ax

    covariance_stim = np.outer(cfg.stim_direction, cfg.stim_direction) * cfg.stim_amplitude
    covariance_nuisance = np.outer(cfg.nuisance_direction, cfg.nuisance_direction) * cfg.nuisance_amplitude
    covariance_noise = np.eye(2) * cfg.noise_amplitude

    stim_ellipse = _get_covariance_ellipse(covariance_stim + covariance_noise)
    full_ellipse = _get_covariance_ellipse(covariance_stim + covariance_nuisance + covariance_noise)

    samples_stim = np.random.multivariate_normal(mean=np.zeros(2), cov=covariance_stim + covariance_noise, size=n_samples)
    samples_full = np.random.multivariate_normal(mean=np.zeros(2), cov=covariance_stim + covariance_nuisance + covariance_noise, size=n_samples)

    # In ax_direction, plot stimulus and nuisance directions as bidirectional arrows from the origin
    max_amplitude = max(cfg.stim_amplitude, cfg.nuisance_amplitude) + cfg.noise_amplitude
    max_lims = max_amplitude * max_lim_factor
    ax_direction.arrow(
        0,
        0,
        cfg.stim_direction[0] * extend_arrow_factor,
        cfg.stim_direction[1] * extend_arrow_factor,
        head_width=arrow_width,
        head_length=arrow_length,
        fc=stim_color,
        ec=stim_color,
        linewidth=linewidth,
    )
    ax_direction.arrow(
        0,
        0,
        cfg.nuisance_direction[0] * cfg.nuisance_amplitude * extend_arrow_factor,
        cfg.nuisance_direction[1] * cfg.nuisance_amplitude * extend_arrow_factor,
        head_width=arrow_width,
        head_length=arrow_length,
        fc=nuisance_color,
        ec=nuisance_color,
        linewidth=linewidth,
    )
    ax_direction.arrow(
        0,
        0,
        -cfg.stim_direction[0] * extend_arrow_factor,
        -cfg.stim_direction[1] * extend_arrow_factor,
        head_width=arrow_width,
        head_length=arrow_length,
        fc=stim_color,
        ec=stim_color,
        linewidth=linewidth,
    )
    ax_direction.arrow(
        0,
        0,
        -cfg.nuisance_direction[0] * cfg.nuisance_amplitude * extend_arrow_factor,
        -cfg.nuisance_direction[1] * cfg.nuisance_amplitude * extend_arrow_factor,
        head_width=arrow_width,
        head_length=arrow_length,
        fc=nuisance_color,
        ec=nuisance_color,
        linewidth=linewidth,
    )
    ax_direction.set_xlim(max_lims * -1, max_lims)
    ax_direction.set_ylim(max_lims * -1, max_lims)

    # In ax_stimcov / ax_fullcov, generate samples of the stimulus and full(stim+nuisance), including independent noise
    # scatter the sample data, and plot the covariance ellipsoids with a thin line colored appropriately
    ax_stimcov.scatter(samples_stim[:, 0], samples_stim[:, 1], alpha=point_alpha, s=point_size, color=stim_color)
    ax_fullcov.scatter(samples_full[:, 0], samples_full[:, 1], alpha=point_alpha, s=point_size, color=nuisance_color)
    ax_stimcov.plot(stim_ellipse[0], stim_ellipse[1], color=stim_color, linewidth=linewidth)
    ax_fullcov.plot(full_ellipse[0], full_ellipse[1], color=nuisance_color, linewidth=linewidth)
    ax_stimcov.set_xlim(max_lims * -1, max_lims)
    ax_stimcov.set_ylim(max_lims * -1, max_lims)
    ax_fullcov.set_xlim(max_lims * -1, max_lims)
    ax_fullcov.set_ylim(max_lims * -1, max_lims)

    # In ax_value, clear any axes stuff and just report the value (which is stim_amplitude / (nuisance_amplitude + stim_amplitude + noise_amplitude)) as text
    ax_value.text(
        0.5,
        0.5,
        f"Value: {cfg.stim_amplitude / (cfg.stim_amplitude + cfg.nuisance_amplitude + cfg.noise_amplitude):.2f}",
        ha="center",
        va="center",
        fontsize=16,
    )
    ax_value.axis("off")


def _mat_sqrt(M: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    evals, evecs = np.linalg.eigh(M)
    return evecs @ np.diag(np.sqrt(np.maximum(evals, 0.0))) @ evecs.T


def _svr_2d(
    cov_stim: npt.NDArray[np.floating],
    cov_full: npt.NDArray[np.floating],
    mode: Literal["eigenvalue", "nuclear_norm"] = "eigenvalue",
) -> float:
    """SVR = κ(stim, full) / κ(full, full) for 2x2 covariances.

    Parameters
    ----------
    mode : {"eigenvalue", "nuclear_norm"}
        "eigenvalue" — tr(sqrt(stim^{1/2} full stim^{1/2})), eigenvalue form.
        "nuclear_norm" — ||stim^{1/2} full^{1/2}||_*, explicit nuclear norm via SVD.
        Both are mathematically equivalent; denominator is tr(full) in either case.
    """
    sqrt_s = _mat_sqrt(cov_stim)
    if mode == "nuclear_norm":
        sqrt_f = _mat_sqrt(cov_full)
        numerator = float(np.sum(np.linalg.svd(sqrt_s @ sqrt_f, compute_uv=False)))
    else:
        kappa = np.sqrt(np.maximum(np.linalg.eigvalsh(sqrt_s @ cov_full @ sqrt_s), 0.0))
        numerator = float(np.sum(kappa))
    return numerator / np.trace(cov_full)


@dataclass
class StimNuisanceArray2D:
    """Config for an array of stim+nuisance configs varying angle or amplitude."""

    stim_direction: npt.NDArray[np.floating]
    vary_type: Literal["angle", "amplitude"]
    stim_amplitude: float = 1.0
    stim_orth_amplitude: float = 0.0
    nuisance_amplitude: float = 1.0
    nuisance_orth_amplitude: float = 0.0
    noise_amplitude: float = 0.1
    n_nuisance: int = 8
    min_nuisance_amplitude: float = 0.0
    max_nuisance_amplitude: float = 1.0
    nuisance_angle: float = 0.0
    n_samples: int = 500
    svr_mode: Literal["eigenvalue", "nuclear_norm"] = "eigenvalue"

    def __post_init__(self) -> None:
        self.stim_direction = np.asarray(self.stim_direction, dtype=float)
        self.stim_direction = self.stim_direction / np.linalg.norm(self.stim_direction)


def _build_nuisance_configs(
    cfg: StimNuisanceArray2D,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return (angles_rad, amplitudes, directions) arrays of shape (n_nuisance,) / (n_nuisance, 2)."""
    if cfg.vary_type == "angle":
        angles = np.linspace(0, np.pi, cfg.n_nuisance, endpoint=False)
        amplitudes = np.full(cfg.n_nuisance, cfg.nuisance_amplitude)
    else:
        angles = np.full(cfg.n_nuisance, cfg.nuisance_angle)
        amplitudes = np.linspace(cfg.min_nuisance_amplitude, cfg.max_nuisance_amplitude, cfg.n_nuisance)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return angles, amplitudes, directions


def plot_stim_nuisance_array_2D(
    cfg: StimNuisanceArray2D,
    ax: tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes],
    stim_color: str = "black",
    cmap_name: str = "hsv",
    example_nuisance_idx: int = 0,
    distribution_layout: Literal["separate", "overlapped_full"] = "separate",
    arrow_scale: float = 3.0,
    arrow_width: float = 0.15,
    schematic_linewidth: float = 2.0,
    schematic_angle_smudge: float = 0.0,
    point_alpha: float = 0.4,
    point_size: float = 8.0,
    linewidth: float = 1.0,
    stim_arrow_height: float = 0.2,
    nuisance_arrow_height: float = 0.25,
    n_std: float = 1.0,
) -> None:
    """Plot a 4-panel schematic sweeping nuisance angle or amplitude.

    Parameters
    ----------
    cfg : StimNuisanceArray2D
    ax : tuple of 4 Axes
        (ax_stim, ax_nuisance, ax_schematic, ax_svr)
    stim_color : str
    cmap_name : str
        Colormap for nuisance configs.
    example_nuisance_idx : int
        Which nuisance config to show in the example nuisance panel(s).
    distribution_layout : {"separate", "overlapped_full"}
        "separate" — stim on ax[0], example nuisance on ax[1] (default).
        "overlapped_full" — stim and example nuisance overlaid on ax[0], then the
        combined stim+nuisance+noise distribution on ax[1].
    arrow_scale : float
        Half-length of the longest arrow in ax_schematic (data coords). All arrows are
        normalized so max amplitude maps to arrow_scale; schematic axis spans ±arrow_scale * 1.3.
    arrow_width : float
        Arrow head width (data coords) in ax_schematic; head_length = arrow_width / 2.
    schematic_linewidth : float
        Line width for arrow stems in ax_schematic (independent of ellipse linewidth).
    schematic_angle_smudge : float
        Total angular spread (radians) to fan nuisance arrows so overlapping ones are visible.
        Each arrow i is offset by (i/n - 0.5) * smudge. Visual only — does not affect SVR.
    point_alpha, point_size, linewidth : float
    n_std : float
        Number of standard deviations to plot the ellipses.
    """
    ax_stim, ax_nuisance, ax_schematic, ax_svr = ax

    v_s = cfg.stim_direction
    v_orth = np.array([-v_s[1], v_s[0]])

    cov_stim = cfg.stim_amplitude * np.outer(v_s, v_s) + cfg.stim_orth_amplitude * np.outer(v_orth, v_orth)
    cov_noise = cfg.noise_amplitude * np.eye(2)

    angles, amplitudes, directions = _build_nuisance_configs(cfg)

    cmap: Colormap = plt.get_cmap(cmap_name)
    n = max(cfg.n_nuisance - 1, 1)
    if cfg.vary_type == "angle":
        # Shift HSV so the example nuisance lands at blue (hue 2/3)
        offset = (2 / 3 - example_nuisance_idx / n) % 1.0
        colors = [cmap((i / n + offset) % 1.0) for i in range(cfg.n_nuisance)]
    else:
        colors = [cmap(i / n) for i in range(cfg.n_nuisance)]

    svr_values = []
    cov_nuisances = []
    for i in range(cfg.n_nuisance):
        v_n = directions[i]
        v_n_orth = np.array([-v_n[1], v_n[0]])
        cov_n = amplitudes[i] * np.outer(v_n, v_n) + cfg.nuisance_orth_amplitude * np.outer(v_n_orth, v_n_orth)
        cov_nuisances.append(cov_n)
        cov_full = cov_stim + cov_n + cov_noise
        svr_values.append(_svr_2d(cov_stim, cov_full, mode=cfg.svr_mode))

    max_amp = max(cfg.stim_amplitude, float(np.max(amplitudes))) + cfg.noise_amplitude
    max_lim = max_amp * 4.5

    idx = example_nuisance_idx
    cov_n_ex = cov_nuisances[idx]
    nuisance_color = colors[idx]

    samples_stim = np.random.multivariate_normal(np.zeros(2), cov_stim + cov_noise, size=cfg.n_samples)
    stim_ellipse = _get_covariance_ellipse(cov_stim + cov_noise, n_std=n_std)
    samples_n = np.random.multivariate_normal(np.zeros(2), cov_n_ex + cov_noise, size=cfg.n_samples)
    n_ellipse = _get_covariance_ellipse(cov_n_ex + cov_noise, n_std=n_std)

    def _set_cov_panel(axis: plt.Axes) -> None:
        axis.set_xlim(-max_lim, max_lim)
        axis.set_ylim(-max_lim, max_lim)
        axis.set_aspect("equal")

    if distribution_layout == "separate":
        # --- ax[0]: stim scatter + ellipse ---
        ax_stim.scatter(samples_stim[:, 0], samples_stim[:, 1], alpha=point_alpha, s=point_size, color=stim_color)
        ax_stim.plot(stim_ellipse[0], stim_ellipse[1], color=stim_color, linewidth=linewidth)
        _set_cov_panel(ax_stim)

        # --- ax[1]: example nuisance scatter + ellipse ---
        ax_nuisance.scatter(samples_n[:, 0], samples_n[:, 1], alpha=point_alpha, s=point_size, color=nuisance_color)
        ax_nuisance.plot(n_ellipse[0], n_ellipse[1], color=nuisance_color, linewidth=linewidth)
        _set_cov_panel(ax_nuisance)
    elif distribution_layout == "overlapped_full":
        # --- ax[0]: stim + example nuisance overlaid ---
        ax_stim.scatter(samples_stim[:, 0], samples_stim[:, 1], alpha=point_alpha, s=point_size, color=stim_color)
        ax_stim.scatter(samples_n[:, 0], samples_n[:, 1], alpha=point_alpha, s=point_size, color=nuisance_color)
        ax_stim.plot(stim_ellipse[0], stim_ellipse[1], color=stim_color, linewidth=linewidth)
        ax_stim.plot(n_ellipse[0], n_ellipse[1], color=nuisance_color, linewidth=linewidth)
        _set_cov_panel(ax_stim)

        # --- ax[1]: full (stim + nuisance + noise) scatter + ellipse ---
        cov_full_ex = cov_stim + cov_n_ex + cov_noise
        samples_full = np.random.multivariate_normal(np.zeros(2), cov_full_ex, size=cfg.n_samples)
        full_ellipse = _get_covariance_ellipse(cov_full_ex, n_std=n_std)
        ax_nuisance.scatter(samples_full[:, 0], samples_full[:, 1], alpha=point_alpha, s=point_size, color=nuisance_color)
        ax_nuisance.plot(full_ellipse[0], full_ellipse[1], color=nuisance_color, linewidth=linewidth)
        _set_cov_panel(ax_nuisance)
    else:
        raise ValueError(f"Unknown distribution_layout: {distribution_layout!r}")

    # --- ax_schematic: stim (black) + all nuisance (colored) arrows ---
    def _draw_bidirectional_arrow(axis: plt.Axes, dx: float, dy: float, color: str) -> None:
        kw = dict(head_width=arrow_width, head_length=arrow_width / 2, fc=color, ec=color, linewidth=schematic_linewidth)
        axis.arrow(0, 0, dx, dy, **kw)
        axis.arrow(0, 0, -dx, -dy, **kw)

    if cfg.vary_type == "amplitude":
        stim_len = cfg.stim_amplitude
        schematic_max_lim = max(cfg.stim_amplitude, float(np.max(amplitudes))) * 1.3

        def _nuisance_len(i: int) -> float:
            return float(amplitudes[i])

    else:
        arrow_norm = max(cfg.stim_amplitude, float(np.max(amplitudes)))
        stim_len = cfg.stim_amplitude / arrow_norm * arrow_scale
        schematic_max_lim = arrow_scale * 1.3

        def _nuisance_len(i: int) -> float:
            return float(amplitudes[i] / arrow_norm * arrow_scale)

    _draw_bidirectional_arrow(ax_schematic, v_s[0] * stim_len, v_s[1] * stim_len, stim_color)
    draw_order = range(cfg.n_nuisance - 1, -1, -1) if cfg.vary_type == "amplitude" else range(cfg.n_nuisance)
    n = max(cfg.n_nuisance - 1, 1)
    for i in draw_order:
        n_len = _nuisance_len(i)
        base_angle = np.arctan2(directions[i][1], directions[i][0])
        smudge = (i / n - 0.5) * schematic_angle_smudge
        v_n_draw = np.array([np.cos(base_angle + smudge), np.sin(base_angle + smudge)])
        _draw_bidirectional_arrow(ax_schematic, v_n_draw[0] * n_len, v_n_draw[1] * n_len, colors[i])
    ax_schematic.set_xlim(-schematic_max_lim, schematic_max_lim)
    ax_schematic.set_ylim(-schematic_max_lim, schematic_max_lim)
    ax_schematic.set_aspect("equal")

    # --- ax_svr: SVR curve ---
    if cfg.vary_type == "angle":
        x_vals = np.degrees(angles)
        x_label = "Nuisance angle (°)"
    else:
        x_vals = amplitudes
        x_label = "Nuisance amplitude"

    ax_svr.plot(x_vals, svr_values, color="gray", linewidth=linewidth, zorder=1)
    for i in range(cfg.n_nuisance):
        ax_svr.scatter(x_vals[i], svr_values[i], color=colors[i], s=point_size * 4, zorder=2)
    ax_svr.set_xlabel(x_label)
    ax_svr.set_ylabel("SVR")
    ax_svr.set_ylim(0, 1)

    def _svr_annotate_arrow(axis: plt.Axes, x: float, height: float, label: str, color: str) -> None:
        axis.annotate(
            "",
            xy=(x, 0),
            xytext=(x, height),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=linewidth),
            zorder=3,
        )
        axis.text(
            x,
            height,
            label,
            ha="center",
            va="bottom",
            color=color,
            fontsize=plt.rcParams["font.size"] * 0.8,
            zorder=3,
        )

    if cfg.vary_type == "angle":
        stim_angle_deg = np.degrees(np.arctan2(v_s[1], v_s[0])) % 180
        _svr_annotate_arrow(ax_svr, stim_angle_deg, stim_arrow_height, "stim", stim_color)
        nuisance_angle_deg = float(np.degrees(angles[example_nuisance_idx]) % 180)
        _svr_annotate_arrow(ax_svr, nuisance_angle_deg, nuisance_arrow_height, "nuisance", colors[example_nuisance_idx])
    else:
        _svr_annotate_arrow(ax_svr, cfg.stim_amplitude, stim_arrow_height, "stim", stim_color)
        _svr_annotate_arrow(ax_svr, float(amplitudes[example_nuisance_idx]), nuisance_arrow_height, "nuisance", colors[example_nuisance_idx])


@dataclass
class StimNuisanceCombined2D:
    """Config for a combined angle+amplitude sweep schematic in 5–6 panels."""

    stim_direction: npt.NDArray[np.floating]
    stim_amplitude: float = 1.0
    stim_orth_amplitude: float = 0.0
    noise_amplitude: float = 0.1
    nuisance_orth_amplitude: float = 0.0
    n_angle: int = 8
    n_amplitude: int = 8
    min_amplitude: float = 0.1
    max_amplitude: float = 2.0
    n_samples: int = 500
    svr_mode: Literal["eigenvalue", "nuclear_norm"] = "eigenvalue"

    def __post_init__(self) -> None:
        self.stim_direction = np.asarray(self.stim_direction, dtype=float)
        self.stim_direction /= np.linalg.norm(self.stim_direction)


def plot_stim_nuisance_combined_2D(
    cfg: StimNuisanceCombined2D,
    ax: tuple[plt.Axes, ...],
    stim_color: str = "orange",
    nuisance_color: str = "black",
    angle_cmap_name: str = "hsv",
    amplitude_cmap_name: str = "gist_heat",
    example_angle_idx: int = 0,
    example_amplitude_idx: int = 0,
    arrow_width: float = 0.15,
    schematic_linewidth: float = 1.0,
    schematic_min_amplitude: float = 0.0,
    point_alpha: float = 0.4,
    point_size: float = 8.0,
    linewidth: float = 1.0,
    include_grid: bool = False,
    n_std: float = 1.0,
    x_padding: float = 0.02,
    fontsize: float = 12,
) -> None:
    """Unified 5- or 6-panel schematic combining angle and amplitude sweeps.

    Parameters
    ----------
    cfg : StimNuisanceCombined2D
    ax : tuple of 5 Axes, or 6 when include_grid is True
        (ax_combined, ax_full, ax_schematic, ax_amplitude, ax_angle[, ax_heatmap])
        ax_combined  — stim + example nuisance scatter with both ellipses
        ax_full      — full (stim+nuisance+noise) scatter with stim/nuisance ellipses overlaid
        ax_schematic — bidirectional amplitude arrows + clockwise arc angle arrows
        ax_amplitude — SVR vs nuisance amplitude
        ax_angle     — SVR vs relative nuisance angle (clockwise from stim, 0–180°)
        ax_heatmap   — SVR grid (relative angle × amplitude); only when include_grid=True
    stim_color : str
    nuisance_color : str
    angle_cmap_name : str
        Colormap for angle-sweep arcs and SVR curve.
    amplitude_cmap_name : str
        Colormap for amplitude-sweep arrows and SVR curve.
    example_angle_idx : int
        Index into angle sweep to highlight.
    example_amplitude_idx : int
        Index into amplitude sweep to highlight; auto-computed if None.
    arrow_width, schematic_linewidth : float
    schematic_min_amplitude : float
        Amplitude threshold below which arrows are omitted from ax_schematic. Default 0 (all shown).
    point_alpha, point_size, linewidth : float
    include_grid : bool
        If True, draw SVR heatmap on ax[5].
    n_std : float
        Number of standard deviations to plot the ellipses.
    fontsize : float
        Font size for all text elements, by default 12
    x_padding : float
        Padding for the x-axis of the summary panels, by default 0.02
    """
    if include_grid:
        ax_combined, ax_full, ax_schematic, ax_amplitude, ax_angle, ax_heatmap = ax[:6]
    else:
        ax_combined, ax_full, ax_schematic, ax_amplitude, ax_angle = ax[:5]

    v_s = cfg.stim_direction
    v_orth = np.array([-v_s[1], v_s[0]])

    cov_stim = cfg.stim_amplitude * np.outer(v_s, v_s) + cfg.stim_orth_amplitude * np.outer(v_orth, v_orth)
    cov_noise = cfg.noise_amplitude * np.eye(2)

    # --- Build sweep arrays ---
    stim_angle_rad = float(np.arctan2(v_s[1], v_s[0]))
    angle_vals = stim_angle_rad + np.linspace(0, np.pi, cfg.n_angle, endpoint=False)
    amplitude_vals = np.linspace(cfg.min_amplitude, cfg.max_amplitude, cfg.n_amplitude)
    angle_vals_degrees = np.degrees(angle_vals)

    def _cov_nuisance(angle: float, amplitude: float) -> npt.NDArray[np.floating]:
        v_n = np.array([np.cos(angle), np.sin(angle)])
        v_n_orth = np.array([-v_n[1], v_n[0]])
        return amplitude * np.outer(v_n, v_n) + cfg.nuisance_orth_amplitude * np.outer(v_n_orth, v_n_orth)

    # SVR along angle sweep (amplitude fixed at cfg.example_nuisance_amplitude)
    svr_angle = []
    for a in angle_vals:
        cov_n = _cov_nuisance(a, amplitude_vals[example_amplitude_idx])
        svr_angle.append(_svr_2d(cov_stim, cov_stim + cov_n + cov_noise, mode=cfg.svr_mode))
    svr_angle = np.array(svr_angle)

    # SVR along amplitude sweep (angle fixed at cfg.example_nuisance_angle)
    svr_amplitude = []
    for amp in amplitude_vals:
        cov_n = _cov_nuisance(angle_vals[example_angle_idx], amp)
        svr_amplitude.append(_svr_2d(cov_stim, cov_stim + cov_n + cov_noise, mode=cfg.svr_mode))

    # --- Colormaps ---
    angle_cmap: Colormap = plt.get_cmap(angle_cmap_name)
    amp_cmap: Colormap = plt.get_cmap(amplitude_cmap_name)
    n_a = max(cfg.n_angle - 1, 1)
    n_amp = max(cfg.n_amplitude - 1, 1)
    # HSV: shift so example angle lands at blue (hue 2/3)
    angle_offset = (2 / 3 - example_angle_idx / n_a) % 1.0
    angle_colors = [angle_cmap((i / n_a + angle_offset) % 1.0) for i in range(cfg.n_angle)]
    amp_colors = [amp_cmap(v) for v in np.linspace(0, 0.9, cfg.n_amplitude)]

    example_nuisance_color = angle_colors[example_angle_idx]

    # --- Example nuisance covariance ---
    cov_n_ex = _cov_nuisance(angle_vals[example_angle_idx], amplitude_vals[example_amplitude_idx])

    # --- Distribution panels setup ---
    samples_stim = np.random.multivariate_normal(np.zeros(2), cov_stim + cov_noise, size=cfg.n_samples)
    samples_n = np.random.multivariate_normal(np.zeros(2), cov_n_ex + cov_noise, size=cfg.n_samples)
    stim_ellipse = _get_covariance_ellipse(cov_stim + cov_noise, n_std=n_std)
    n_ellipse = _get_covariance_ellipse(cov_n_ex + cov_noise, n_std=n_std)
    dist_lim = max(cfg.stim_amplitude, cfg.max_amplitude) * 1.3
    schematic_max_lim = max(cfg.stim_amplitude, cfg.max_amplitude) * 1.025

    def _set_dist_panel(axis: plt.Axes) -> None:
        axis.set_xlim(-dist_lim, dist_lim)
        axis.set_ylim(-dist_lim, dist_lim)
        axis.set_aspect("equal")

    # --- ax_combined: stim + example nuisance scatter, both ellipses ---
    ax_combined.scatter(samples_stim[:, 0], samples_stim[:, 1], alpha=point_alpha, s=point_size, color=stim_color)
    ax_combined.scatter(samples_n[:, 0], samples_n[:, 1], alpha=point_alpha, s=point_size, color=nuisance_color)
    ax_combined.plot(stim_ellipse[0], stim_ellipse[1], color=stim_color, linewidth=linewidth, label="stim")
    ax_combined.plot(n_ellipse[0], n_ellipse[1], color=nuisance_color, linewidth=linewidth, label="nuisance")
    ax_combined.legend(frameon=True, loc="lower left", fontsize=fontsize)
    _set_dist_panel(ax_combined)

    # --- ax_full: full (stim+nuisance+noise) scatter, stim + nuisance ellipses overlaid ---
    cov_full_ex = cov_stim + cov_n_ex + cov_noise
    samples_full = np.random.multivariate_normal(np.zeros(2), cov_full_ex, size=cfg.n_samples)
    ax_full.scatter(samples_full[:, 0], samples_full[:, 1], alpha=point_alpha, s=point_size, color=stim_color, label="full data")
    ax_full.plot(stim_ellipse[0], stim_ellipse[1], color=stim_color, linewidth=linewidth, label="stim")
    ax_full.plot(n_ellipse[0], n_ellipse[1], color=nuisance_color, linewidth=linewidth, label="nuisance")
    ax_full.legend(frameon=True, loc="lower left", fontsize=fontsize)
    _set_dist_panel(ax_full)

    max_tick = np.floor(dist_lim)
    format_spines(
        ax_combined,
        x_pos=-0.02,
        y_pos=-0.02,
        xticks=(),  # (-max_tick, 0, max_tick),
        yticks=(),  # (-max_tick, 0, max_tick),
        xbounds=(-max_tick, max_tick),
        ybounds=(-max_tick, max_tick),
        spines_visible=[],  # ["bottom", "left"],
    )
    format_spines(
        ax_full,
        x_pos=-0.02,
        y_pos=-0.02,
        xbounds=(-max_tick, max_tick),
        ybounds=(-max_tick, max_tick),
        xticks=(),  # (-max_tick, 0, max_tick),
        yticks=(),
        spines_visible=[],  # ["bottom"],
    )

    # --- ax_schematic: amplitude arrows (back) + angle arcs (front) + stim arrow (top) ---
    def _draw_bidirectional_arrow(axis: plt.Axes, dx: float, dy: float, color) -> None:
        axis.plot([-dx, dx], [-dy, dy], color=color, linewidth=schematic_linewidth, solid_capstyle="round")
        norm = np.hypot(dx, dy)
        if norm < 1e-9:
            return
        eps = norm * 1e-4
        ux, uy = dx / norm * eps, dy / norm * eps
        ap = dict(arrowstyle="-|>", color=color, lw=schematic_linewidth, mutation_scale=arrow_width * 80)
        axis.annotate("", xy=(dx, dy), xytext=(dx - ux, dy - uy), arrowprops=ap)
        axis.annotate("", xy=(-dx, -dy), xytext=(-dx + ux, -dy + uy), arrowprops=ap)

    arc_radius = cfg.stim_amplitude

    v_n_amp = np.array([np.cos(angle_vals[example_angle_idx]), np.sin(angle_vals[example_angle_idx])])
    for i in range(cfg.n_amplitude - 1, -1, -1):
        if amplitude_vals[i] < schematic_min_amplitude:
            continue
        _draw_bidirectional_arrow(ax_schematic, v_n_amp[0] * amplitude_vals[i], v_n_amp[1] * amplitude_vals[i], amp_colors[i])

    def _draw_arc_arrow(axis: plt.Axes, from_angle: float, to_angle: float, color) -> None:
        if abs(to_angle - from_angle) < 1e-6:
            return
        n_pts = max(int(abs(to_angle - from_angle) * 180 / np.pi) + 2, 3)
        arc_th = np.linspace(from_angle, to_angle, n_pts)
        axis.plot(arc_radius * np.cos(arc_th), arc_radius * np.sin(arc_th), color=color, linewidth=schematic_linewidth, solid_capstyle="round")
        eps = np.sign(to_angle - from_angle) * 1e-4
        axis.annotate(
            "",
            xy=(arc_radius * np.cos(to_angle), arc_radius * np.sin(to_angle)),
            xytext=(arc_radius * np.cos(to_angle - eps), arc_radius * np.sin(to_angle - eps)),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=schematic_linewidth, mutation_scale=arrow_width * 80),
        )

    angle_draw_order = [0] + list(range(cfg.n_angle - 1, 0, -1))
    for idx, angle_idx in enumerate(angle_draw_order):
        _draw_arc_arrow(ax_schematic, stim_angle_rad, angle_vals[angle_idx], angle_colors[idx])

    _draw_bidirectional_arrow(ax_schematic, v_s[0] * arc_radius, v_s[1] * arc_radius, stim_color)
    ax_schematic.set_xlim(-schematic_max_lim, schematic_max_lim)
    ax_schematic.set_ylim(-schematic_max_lim, schematic_max_lim)
    ax_schematic.set_aspect("equal")

    # --- Insets: amplitude colorbar (bottom-left) and angle colorbar (bottom-right) ---
    _ifs = plt.rcParams["font.size"] * 0.7
    ax_amp_ins = ax_schematic.inset_axes([0.02, 0.0, 0.42, 0.04])
    ax_amp_ins.imshow(np.array(amp_colors).reshape(1, -1, 4), aspect="auto")
    ax_amp_ins.set_xticks([])
    ax_amp_ins.set_yticks([])
    ax_amp_ins.set_title("Amplitude", fontsize=_ifs)
    ax_amp_ins.tick_params(axis="x", labelsize=_ifs)

    ax_ang_ins = ax_schematic.inset_axes([0.56, 0.00, 0.42, 0.04])
    ax_ang_ins.imshow(np.array(angle_colors).reshape(1, -1, 4), aspect="auto")
    ax_ang_ins.set_xticks([])
    ax_ang_ins.set_yticks([])
    ax_ang_ins.set_title("Angle", fontsize=_ifs)
    ax_ang_ins.tick_params(axis="x", labelsize=_ifs)

    format_spines(
        ax_schematic,
        x_pos=-0.02,
        y_pos=-0.02,
        xticks=(),
        yticks=(),
        spines_visible=[],
    )

    # --- ax_amplitude: SVR vs nuisance amplitude ---
    ax_amplitude.plot(amplitude_vals, svr_amplitude, color="gray", linewidth=linewidth, zorder=1)
    for i in range(cfg.n_amplitude):
        ax_amplitude.scatter(amplitude_vals[i], svr_amplitude[i], color=amp_colors[i], s=point_size * 4, zorder=2)
    # ax_amplitude.scatter(
    #     amplitude_vals[example_amplitude_idx],
    #     svr_amplitude[example_amplitude_idx],
    #     color=example_nuisance_color,
    #     s=point_size * 8,
    #     zorder=3,
    #     edgecolors=stim_color,
    #     linewidths=linewidth,
    # )
    ax_amplitude.set_xlabel("Nuisance amplitude")
    ax_amplitude.set_ylabel("SVR")
    xlims = (0, cfg.max_amplitude * 1.1)
    xrange = xlims[1] - xlims[0]
    xlims = (xlims[0] - xrange * x_padding, xlims[1] + xrange * x_padding)
    ax_amplitude.set_xlim(xlims)
    ax_amplitude.set_ylim(0, 1)
    format_spines(
        ax_amplitude,
        x_pos=-0.02,
        y_pos=-0.02,
        xticks=(0, cfg.max_amplitude),
        yticks=(0, 0.5, 1),
        xbounds=(0, cfg.max_amplitude),
        ybounds=(0, 1),
        spines_visible=["bottom", "left"],
    )

    # --- ax_angle: SVR vs relative (clockwise) angle from stim ---
    # Parallel-to-stim point sits at 180°; % 180 wraps it to 0°
    relative_angles_degrees = angle_vals_degrees - stim_angle_rad * 180 / np.pi
    ax_angle.plot(relative_angles_degrees, svr_angle, color="gray", linewidth=linewidth, zorder=1)
    for idx, angle_idx in enumerate(angle_draw_order):
        ax_angle.scatter(relative_angles_degrees[angle_idx], svr_angle[angle_idx], color=angle_colors[idx], s=point_size * 4, zorder=2)
    # ax_angle.scatter(
    #     relative_angles_degrees[example_angle_idx],
    #     svr_angle[example_angle_idx],
    #     color=example_nuisance_color,
    #     s=point_size * 8,
    #     zorder=3,
    #     edgecolors=stim_color,
    #     linewidths=linewidth,
    # )
    ex_angle_deg = float(relative_angles_degrees[example_angle_idx])
    ax_angle.set_xlabel("Relative nuisance angle (°)")
    ax_angle.set_xlim(0 - 180 * x_padding, 180 + 180 * x_padding)
    ax_angle.set_ylim(0, 1)
    format_spines(
        ax_angle,
        x_pos=-0.02,
        y_pos=-0.02,
        xticks=(0, 90, 180),
        yticks=(),
        xbounds=(0, 180),
        ybounds=(0, 1),
        spines_visible=["bottom"],
    )

    # --- ax_heatmap (optional): SVR grid, y = relative angle (0 → max cw_dist) ---
    if include_grid:
        svr_grid = np.zeros((cfg.n_angle, cfg.n_amplitude))
        for ai, a in enumerate(angle_vals):
            for ampi, amp in enumerate(amplitude_vals):
                cov_n = _cov_nuisance(a, amp)
                svr_grid[ai, ampi] = _svr_2d(cov_stim, cov_stim + cov_n + cov_noise, mode=cfg.svr_mode)

        max_angle_deg = float(angle_vals_degrees[-1])
        im = ax_heatmap.imshow(
            svr_grid,
            origin="lower",
            aspect="auto",
            extent=[amplitude_vals[0], amplitude_vals[-1], 0, max_angle_deg],
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        ax_heatmap.scatter(
            amplitude_vals[example_amplitude_idx],
            ex_angle_deg,
            marker="x",
            color="white",
            s=point_size * 8,
            linewidths=linewidth * 2,
            zorder=3,
        )
        ax_heatmap.set_xlabel("Nuisance amplitude")
        ax_heatmap.set_ylabel("Relative nuisance angle (°)")
        plt.colorbar(im, ax=ax_heatmap, label="SVR")


def _kappa_optimal_point(
    cov_a: npt.NDArray[np.floating],
    cov_b: npt.NDArray[np.floating],
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Top mode of kappa(A, B): the point on ellipse A maximizing <A^{1/2}u, B^{1/2}v> over unit u, v.

    Returns
    -------
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        (point_a, sqrt_a) where point_a = A^{1/2} u* is the optimal point on ellipse A
        (radius 1, i.e. n_std=1 scale) and sqrt_a = A^{1/2}.
    """
    sqrt_a = _mat_sqrt(cov_a)
    eigvals, eigvecs = np.linalg.eigh(sqrt_a @ cov_b @ sqrt_a)
    u_star = eigvecs[:, np.argmax(eigvals)]
    point_a = sqrt_a @ u_star
    return point_a, sqrt_a


@dataclass
class KappaOverlap2D:
    """Config for the kappa(A, B) geometric-overlap schematic.

    Shows the optimal point u* on ellipse A (black arrow), a sweep of points v(theta) around
    ellipse B (rainbow), and the dot product <u*, v(theta)> as a function of theta, with the
    maximizing theta marked. See ``docs/shared_variance.md``, "Shared variance overlap" section.
    """

    cov_a: npt.NDArray[np.floating]
    cov_b: npt.NDArray[np.floating]
    n_sweep: int = 64
    n_samples: int = 500
    n_std: float = 2.0


def plot_kappa_overlap_2D(
    cfg: KappaOverlap2D,
    ax: tuple[plt.Axes, plt.Axes, plt.Axes],
    color_a: str = "black",
    cmap_name: str = "hsv",
    point_alpha: float = 0.4,
    point_size: float = 8.0,
    linewidth: float = 1.0,
    arrow_width: float = 0.15,
    sweep_point_size: float = 10.0,
) -> None:
    """Plot the kappa(A, B) optimal-overlap schematic: ellipse A + u*, ellipse B + v(theta) sweep, dot-product stem plot.

    Parameters
    ----------
    cfg : KappaOverlap2D
    ax : tuple of 3 Axes
        (ax_a, ax_b, ax_stem)
    color_a : str
        Color for ellipse A, its scatter, and the optimal arrow.
    cmap_name : str
        Colormap for the theta sweep on ellipse B and the stem plot.
    point_alpha, point_size, linewidth : float
    arrow_width : float
        Arrow head width (data coords) for the two highlighted arrows.
    sweep_point_size : float
        Marker size for the swept points on ellipse B.
    """
    ax_a, ax_b, ax_stem = ax

    point_a, _ = _kappa_optimal_point(cfg.cov_a, cfg.cov_b)
    sqrt_b = _mat_sqrt(cfg.cov_b)

    thetas = np.linspace(0, 2 * np.pi, cfg.n_sweep, endpoint=False)
    v_sweep = np.stack([np.cos(thetas), np.sin(thetas)], axis=0)
    points_b = sqrt_b @ v_sweep
    dot_products = point_a @ points_b
    idx_max = int(np.argmax(dot_products))

    cmap: Colormap = plt.get_cmap(cmap_name)
    colors = [cmap(i / cfg.n_sweep) for i in range(cfg.n_sweep)]

    ellipse_a = _get_covariance_ellipse(cfg.cov_a, n_std=cfg.n_std)
    ellipse_b = _get_covariance_ellipse(cfg.cov_b, n_std=cfg.n_std)
    samples_a = np.random.multivariate_normal(np.zeros(2), cfg.cov_a, size=cfg.n_samples)
    samples_b = np.random.multivariate_normal(np.zeros(2), cfg.cov_b, size=cfg.n_samples)

    def _draw_arrow(axis: plt.Axes, xy: npt.NDArray[np.floating], color) -> None:
        axis.arrow(
            0,
            0,
            xy[0],
            xy[1],
            head_width=arrow_width,
            head_length=arrow_width / 2,
            fc=color,
            ec=color,
            linewidth=linewidth,
            length_includes_head=True,
        )

    lim = np.max(np.abs(np.concatenate([samples_a, samples_b], axis=1))) * 1.01

    # --- ax_a: ellipse A, scatter, optimal arrow u* ---
    ax_a.scatter(samples_a[:, 0], samples_a[:, 1], alpha=point_alpha, s=point_size, color=color_a)
    ax_a.plot(ellipse_a[0], ellipse_a[1], color=color_a, linewidth=linewidth)
    _draw_arrow(ax_a, point_a * cfg.n_std, color_a)
    ax_a.set_xlim(-lim, lim)
    ax_a.set_ylim(-lim, lim)
    ax_a.set_aspect("equal")

    # --- ax_b: ellipse B, scatter, rainbow sweep, peak marked with star ---
    ax_b.scatter(samples_b[:, 0], samples_b[:, 1], alpha=point_alpha, s=point_size, color="gray")
    ax_b.plot(ellipse_b[0], ellipse_b[1], color="gray", linewidth=linewidth)
    ax_b.plot(ellipse_a[0], ellipse_a[1], color=color_a, linewidth=linewidth)
    ax_b.scatter(points_b[0] * cfg.n_std, points_b[1] * cfg.n_std, c=colors, s=sweep_point_size)
    _draw_arrow(ax_b, point_a * cfg.n_std, color_a)
    _draw_arrow(ax_b, points_b[:, idx_max] * cfg.n_std, "black")
    ax_b.set_xlim(-lim, lim)
    ax_b.set_ylim(-lim, lim)
    ax_b.set_aspect("equal")

    # --- ax_stem: dot product <u*, v(theta)> vs theta, peak marked ---
    theta_deg = np.degrees(thetas)
    ax_stem.scatter(theta_deg, dot_products, c=colors, s=sweep_point_size, zorder=2)
    ax_stem.vlines(theta_deg, 0, dot_products, colors=colors, linewidth=linewidth, zorder=1)
    ax_stem.scatter(
        theta_deg[idx_max],
        dot_products[idx_max],
        marker="*",
        s=sweep_point_size,
        color=colors[idx_max],
        edgecolors="black",
        linewidths=linewidth * 2,
        zorder=3,
    )
    ax_stem.vlines(theta_deg[idx_max], 0, dot_products[idx_max], colors="black", linewidth=linewidth * 2, zorder=1)
    ax_stem.axhline(0, color="black", linewidth=linewidth * 0.5, zorder=0)
    ax_stem.set_xlabel("theta (deg)")
    ax_stem.set_ylabel("<u*, v(theta)>")
    _max_dot = np.max(np.abs(dot_products))
    _ylims = _max_dot * 1.1 * np.array([-1, 1])
    _yticks = np.round(_max_dot, 1) * np.array([-1, 1])
    ax_stem.set_ylim(_ylims)
    format_spines(
        ax_stem,
        x_pos=-0.02,
        y_pos=-0.02,
        xticks=(0, 90, 180, 270, 360),
        yticks=_yticks,
        xbounds=(0, 360),
        ybounds=_yticks[[0, -1]],
        spines_visible=["bottom", "left"],
    )


Kind = Literal["pf", "full"]


@dataclass(frozen=True)
class ComparisonSpec:
    """One matrix-product schematic."""

    label: str
    left: Kind
    right: Kind
    left_fold: str = r"$i$"
    right_fold: str = r"$j$"
    left_cv: bool = False


@dataclass(frozen=True)
class StimSpaceSchematicConfig:
    """Visual configuration for the SS_cv / SF_cv / FF schematic."""

    comparisons: tuple[ComparisonSpec, ...] = field(
        default_factory=lambda: (
            ComparisonSpec("ss_cv", "pf", "pf", left_cv=True),
            ComparisonSpec("sf_cv", "pf", "full", left_cv=True),
            ComparisonSpec("ff", "full", "full"),
        )
    )

    # Canvas
    figsize: tuple[float, float] = (11.0, 3.25)
    dpi: int = 200
    background: str = "white"
    font_family: str = "Arial"

    # Colors
    pf_color: str = "#F05A19"
    full_color: str = "#111111"
    box_facecolor: str = "white"

    # Typography
    matrix_label_size: float = 20
    fold_label_size: float = 13
    font_weight: str = "semibold"

    # Box geometry, in axes coordinates
    box_width: float = 0.105
    box_height: float = 0.28
    box_rounding: float = 0.014
    box_linewidth: float = 1.8
    matrix_y: float = 0.47
    fold_pad_x: float = 0.014
    fold_pad_y: float = 0.02

    # Horizontal layout
    panel_centers: tuple[float, ...] = (0.17, 0.50, 0.83)
    pair_gap: float = 0.075
    operator: str = r"$\bullet$"
    operator_size: float = 22

    # CV badge
    cv_text: str = "CV"
    cv_badge_size: float = 10
    cv_badge_width: float = 0.034
    cv_badge_height: float = 0.055
    cv_badge_pad_x: float = 0.009
    cv_badge_pad_y: float = 0.012
    cv_badge_linewidth: float = 1.4

    # Export
    bbox_inches: str = "tight"
    transparent: bool = False


def _kind_label(kind: Kind) -> str:
    return "PF" if kind == "pf" else "Full"


def _kind_color(kind: Kind, cfg: StimSpaceSchematicConfig) -> str:
    return cfg.pf_color if kind == "pf" else cfg.full_color


def _draw_matrix_box(
    ax: Axes,
    *,
    center_x: float,
    kind: Kind,
    fold: str,
    cfg: StimSpaceSchematicConfig,
    show_cv: bool = False,
) -> None:
    """Draw one rounded matrix box in axes coordinates."""

    color = _kind_color(kind, cfg)
    left = center_x - cfg.box_width / 2
    bottom = cfg.matrix_y - cfg.box_height / 2

    box = FancyBboxPatch(
        (left, bottom),
        cfg.box_width,
        cfg.box_height,
        boxstyle=f"round,pad=0.008,rounding_size={cfg.box_rounding}",
        transform=ax.transAxes,
        facecolor=cfg.box_facecolor,
        edgecolor=color,
        linewidth=cfg.box_linewidth,
        clip_on=False,
    )
    ax.add_patch(box)

    ax.text(
        center_x,
        cfg.matrix_y,
        _kind_label(kind),
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=color,
        fontsize=cfg.matrix_label_size,
        fontweight=cfg.font_weight,
    )

    ax.text(
        left + cfg.box_width - cfg.fold_pad_x,
        bottom + cfg.fold_pad_y,
        fold,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=cfg.full_color,
        fontsize=cfg.fold_label_size,
    )

    if show_cv:
        badge_left = left + cfg.box_width - cfg.cv_badge_width - cfg.cv_badge_pad_x
        badge_bottom = bottom + cfg.box_height - cfg.cv_badge_height - cfg.cv_badge_pad_y

        badge = FancyBboxPatch(
            (badge_left, badge_bottom),
            cfg.cv_badge_width,
            cfg.cv_badge_height,
            boxstyle=f"round,pad=0.003,rounding_size={cfg.box_rounding * 0.8}",
            transform=ax.transAxes,
            facecolor=cfg.box_facecolor,
            edgecolor=cfg.pf_color,
            linewidth=cfg.cv_badge_linewidth,
            clip_on=False,
        )
        ax.add_patch(badge)
        ax.text(
            badge_left + cfg.cv_badge_width / 2,
            badge_bottom + cfg.cv_badge_height / 2,
            cfg.cv_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=cfg.pf_color,
            fontsize=cfg.cv_badge_size,
            fontweight="bold",
        )


def make_stimspace_schematic(
    cfg: StimSpaceSchematicConfig | None = None,
) -> tuple[Figure, Axes]:
    """Create the minimalist stimulus-space spectra schematic."""

    cfg = cfg or StimSpaceSchematicConfig()

    if len(cfg.panel_centers) != len(cfg.comparisons):
        raise ValueError("panel_centers and comparisons must have the same length: " f"{len(cfg.panel_centers)} != {len(cfg.comparisons)}")

    plt.rcParams.update(
        {
            "font.family": cfg.font_family,
            "svg.fonttype": "none",  # keep text editable in Illustrator
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    fig.patch.set_facecolor(cfg.background)
    ax.set_facecolor(cfg.background)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for center, spec in zip(cfg.panel_centers, cfg.comparisons, strict=True):
        left_x = center - cfg.pair_gap
        right_x = center + cfg.pair_gap

        _draw_matrix_box(
            ax,
            center_x=left_x,
            kind=spec.left,
            fold=spec.left_fold,
            cfg=cfg,
            show_cv=spec.left_cv,
        )
        _draw_matrix_box(
            ax,
            center_x=right_x,
            kind=spec.right,
            fold=spec.right_fold,
            cfg=cfg,
        )

        ax.text(
            center,
            cfg.matrix_y,
            cfg.operator,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=cfg.operator_size,
            color=cfg.full_color,
        )

    fig.tight_layout(pad=0.15)
    return fig, ax


def save_stimspace_schematic(
    output_stem: str | Path,
    cfg: StimSpaceSchematicConfig | None = None,
    formats: Sequence[str] = ("svg", "png"),
) -> list[Path]:
    """Render and save the schematic in one or more formats."""

    cfg = cfg or StimSpaceSchematicConfig()
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = make_stimspace_schematic(cfg)
    paths: list[Path] = []

    try:
        for extension in formats:
            extension = extension.lower().lstrip(".")
            path = output_stem.with_suffix(f".{extension}")
            fig.savefig(
                path,
                dpi=cfg.dpi,
                bbox_inches=cfg.bbox_inches,
                transparent=cfg.transparent,
                facecolor=cfg.background,
            )
            paths.append(path)
    finally:
        plt.close(fig)

    return paths


# StimSpaceSchematicConfig fields exposed as live Syd controls.
_STIMSPACE_TUNABLES = [
    "matrix_label_size",
    "fold_label_size",
    "box_width",
    "box_height",
    "box_rounding",
    "box_linewidth",
    "matrix_y",
    "fold_pad_x",
    "fold_pad_y",
    "pair_gap",
    "operator_size",
    "cv_badge_size",
    "cv_badge_width",
    "cv_badge_height",
    "cv_badge_pad_x",
    "cv_badge_pad_y",
    "cv_badge_linewidth",
]


class StimSpaceSchematicViewer(Viewer):
    """Interactive SS_cv / SF_cv / FF schematic driven by a :class:`StimSpaceSchematicConfig`.

    All layout/typography fields in ``_STIMSPACE_TUNABLES`` are exposed as live sliders;
    everything else (colors, labels, comparisons) comes straight from ``config``.
    """

    def __init__(self, config: StimSpaceSchematicConfig):
        self.cfg = config
        limits = {
            "matrix_label_size": (8.0, 36.0),
            "fold_label_size": (6.0, 24.0),
            "box_width": (0.02, 0.3),
            "box_height": (0.05, 0.6),
            "box_rounding": (0.0, 0.05),
            "box_linewidth": (0.5, 6.0),
            "matrix_y": (0.2, 0.8),
            "fold_pad_x": (0.0, 0.05),
            "fold_pad_y": (0.0, 0.05),
            "pair_gap": (0.02, 0.2),
            "operator_size": (8.0, 40.0),
            "cv_badge_size": (4.0, 20.0),
            "cv_badge_width": (0.01, 0.1),
            "cv_badge_height": (0.02, 0.15),
            "cv_badge_pad_x": (0.0, 0.03),
            "cv_badge_pad_y": (0.0, 0.03),
            "cv_badge_linewidth": (0.5, 4.0),
        }
        for name in _STIMSPACE_TUNABLES:
            lo, hi = limits[name]
            self.add_float(name, value=float(getattr(config, name)), min=lo, max=hi, step=0.001)

    def plot(self, state):
        cfg = replace(
            self.cfg,
            **{name: state[name] for name in _STIMSPACE_TUNABLES},
        )
        fig, _ = make_stimspace_schematic(cfg)
        return fig


def stimspace_schematic(
    config: StimSpaceSchematicConfig | None = None,
    return_syd_viewer: bool = False,
):
    """Minimalist stimulus-space spectra schematic (SS_cv / SF_cv / FF).

    Every visual knob comes from ``config``; the layout/typography fields in
    ``_STIMSPACE_TUNABLES`` are also exposed as live Syd sliders.

    Parameters
    ----------
    config : StimSpaceSchematicConfig or None
        Full style/layout config. A default one is created when None.
    return_syd_viewer : bool
        If True, return the Syd viewer instead of a rendered figure.
    """
    viewer = StimSpaceSchematicViewer(config or StimSpaceSchematicConfig())
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Pipeline flow schematics: "dim" (spectra -> dimensionality) and "svr" (ratio)
# ---------------------------------------------------------------------------

FlowStyle = Literal["pf", "full", "process"]
FlowRole = Literal["pair", "process", "spectrum", "outcome"]
FlowConnector = Literal["dot", "arrow"]

# Draw order: arrows tuck behind the boxes they point into; labels sit on top.
_Z_ARROW = 1
_Z_BOX = 2
_Z_TEXT = 3


@dataclass(frozen=True)
class FlowNode:
    """One rounded box in a pipeline row.

    Parameters
    ----------
    text : str
        Box label. Newlines wrap; mathtext (``$..$``) is supported.
    style : {"pf", "full", "process"}
        Color scheme: placefield (orange), full CA1 (black), or processing step (gray).
    role : {"pair", "process", "spectrum", "outcome"}
        What the box is in the pipeline. Selects its default width -- ``pair_width``,
        ``process_width``, ``spectrum_width`` or ``outcome_width`` -- so each stage can be
        sized independently. "pair" boxes also use ``pair_label_size`` for their text.
    superscript : str
        Small corner tag drawn at the top-right inside the box (e.g. ``"cv"``).
    width : float or None
        Column-width override in layout units. None uses the role's width.
    """

    text: str
    style: FlowStyle
    role: FlowRole = "process"
    superscript: str = ""
    width: float | None = None


@dataclass(frozen=True)
class FlowRow:
    """One row of the pipeline: a sequence of boxes joined by dot/arrow connectors."""

    title: str
    cells: tuple[FlowNode, ...]
    connectors: tuple[FlowConnector, ...]

    def __post_init__(self) -> None:
        if len(self.connectors) != len(self.cells) - 1:
            raise ValueError(f"connectors must have len(cells) - 1 entries: {len(self.connectors)} != {len(self.cells) - 1}")


@dataclass(frozen=True)
class FlowVariant:
    """A registered schematic: the rows plus the optional variance-circle summary.

    ``fig_width`` is the variant's default canvas width in inches -- the one fixed
    physical dimension of the drawing. Everything else is relative: the layout is solved
    to fill that width and the canvas height follows from it (see :func:`flow_figsize`).
    """

    rows: tuple[FlowRow, ...]
    fig_width: float
    show_circle: bool = False
    divide_label: str = "divide"
    divide_start_col: int = 2


_PF_I = r"$\mathrm{PF}_\mathrm{i}$"
_PF_J = r"$\mathrm{PF}_\mathrm{j}$"
_FULL_I = r"$\mathrm{Full}_\mathrm{i}$"
_FULL_J = r"$\mathrm{Full}_\mathrm{j}$"

FLOW_VARIANTS: dict[str, FlowVariant] = {
    "dim": FlowVariant(
        fig_width=4.475,
        rows=(
            FlowRow(
                title="reliable placefield structure",
                cells=(
                    FlowNode(_PF_I, "pf", role="pair", superscript="cv"),
                    FlowNode(_PF_J, "pf", role="pair", superscript="cv"),
                    FlowNode("x-cov\nsvd", "process", role="process"),
                    FlowNode("placefield\nspectrum", "pf", role="spectrum"),
                    FlowNode("dimensionality", "process", role="outcome"),
                ),
                connectors=("dot", "arrow", "arrow", "arrow"),
            ),
            FlowRow(
                title="reliable CA1 structure",
                cells=(
                    FlowNode(_FULL_I, "full", role="pair"),
                    FlowNode(_FULL_J, "full", role="pair"),
                    FlowNode("svd", "process", role="process"),
                    FlowNode("full CA1\nspectrum", "full", role="spectrum"),
                    FlowNode("dimensionality", "process", role="outcome"),
                ),
                connectors=("dot", "arrow", "arrow", "arrow"),
            ),
        ),
    ),
    "svr": FlowVariant(
        fig_width=6.8,
        rows=(
            FlowRow(
                title="reliable placefield structure",
                cells=(
                    FlowNode(_PF_I, "pf", role="pair", superscript="cv"),
                    FlowNode(_FULL_J, "full", role="pair"),
                    FlowNode("x-cov\nsvd", "process", role="process"),
                    FlowNode("PF Shared\nSpectrum", "pf", role="spectrum"),
                    FlowNode("sum of x-val'd\nspectral mass", "process", role="outcome"),
                ),
                connectors=("dot", "arrow", "arrow", "arrow"),
            ),
            FlowRow(
                title="reliable CA1 structure",
                cells=(
                    FlowNode(_FULL_I, "full", role="pair"),
                    FlowNode(_FULL_J, "full", role="pair"),
                    FlowNode("svd", "process", role="process"),
                    FlowNode("full CA1\nspectrum", "full", role="spectrum"),
                    FlowNode("sum of x-val'd\nspectral mass", "process", role="outcome"),
                ),
                connectors=("dot", "arrow", "arrow", "arrow"),
            ),
        ),
        show_circle=True,
        divide_label="divide",
        divide_start_col=4,
    ),
}

# Per-variant deltas from the (dim-tuned) dataclass defaults. "dim" needs no delta. See
# :func:`default_flow_config`, which layers these onto FlowSchematicConfig.
_VARIANT_PRESETS: dict[str, dict[str, float]] = {
    "dim": {},
    "svr": {
        "spectrum_width": 1.3,
        "circle_gap": 0.62,
        "circle_outer_label_y": -0.55,
        "inner_offset_x": -0.30,
        "question_size": 18.0,
    },
}


@dataclass(frozen=True)
class FlowSchematicConfig:
    """Full configuration for both pipeline schematics.

    ``fig_width`` is the only fixed physical quantity: the canvas is exactly that many
    inches wide. Everything else is relative, expressed in abstract layout units. The
    inches-per-unit scale is *solved* rather than configured, from

        ``unit_scale * (2 * margin_x + drawing_width_in_units) == fig_width``

    and the canvas height then follows as ``unit_scale * (2 * margin_y + drawing_height)``,
    so the page always grows to fit the drawing instead of the drawing being cropped to
    the page. x and y share the one scale, so the drawing is never distorted.

    Font sizes are the exception -- they stay in points, because a publication figure needs
    real type sizes. That means changing the layout changes how big the type looks relative
    to the boxes, so re-check the labels after retuning widths.
    """

    variant: str = "dim"

    # Canvas. fig_width None means "use the variant's width" (see FlowVariant); set it to
    # pin a different one. Margins are layout units, applied on all four sides. The bare
    # dataclass defaults below are the "dim" variant; per-variant deltas live in
    # ``_VARIANT_PRESETS`` and are applied by :func:`default_flow_config`.
    fig_width: float | None = None
    margin_x: float = 0.14
    margin_y: float = 0.02
    dpi: int = 200
    background: str = "white"
    font_family: str = "Arial"
    # Applies to every string in the schematic. Note that the mathtext labels ($..$) render
    # through mathtext.rm and ignore it, so anything other than "normal" makes the plain
    # labels disagree in weight with the PF_i / Full_i boxes.
    font_weight: str = "normal"

    # Colors (face / edge / text per style)
    pf_face: str = "#F05A19"
    pf_edge: str = "#1A1A1A"
    pf_text: str = "white"
    full_face: str = "#0A0A0A"
    full_edge: str = "#0A0A0A"
    full_text: str = "white"
    process_face: str = "#CFCFCF"
    process_edge: str = "#3F4A52"
    process_text: str = "#111111"
    arrow_color: str = "#0A0A0A"
    title_color: str = "#111111"

    # Box geometry. One shared height for every box; one width per pipeline role, so the
    # input pair, the svd step, the spectrum and the outcome each size independently.
    box_height: float = 0.5
    pair_width: float = 1.0
    process_width: float = 1.05
    spectrum_width: float = 1.25
    outcome_width: float = 1.45
    box_rounding: float = 0.14
    box_linewidth: float = 0.8

    # Connectors. Thin shaft with a proportional head.
    dot_gap: float = 0.28
    dot_radius: float = 0.05
    arrow_len: float = 0.52
    arrow_shaft: float = 0.03
    arrow_head_width: float = 0.1
    arrow_head_length: float = 0.09

    # Rows and titles. Titles center on the whole canvas width.
    row_pitch: float = 0.86
    title_pad: float = 0.05
    title_size: float = 8.0

    # Typography. Every size here is a font size in points.
    label_size: float = 8.0
    pair_label_size: float = 8.0
    superscript_size: float = 5.0
    superscript_pad: float = 0.1
    linespacing: float = 1.0

    # Divide arrow (svr only). Sits below the row midline (divide_y_offset < 0) with the
    # "divide" label centered on the sum-of-spectral-mass boxes, above the arrow.
    divide_label_size: float = 8.0
    divide_label_pad: float = 0.05
    divide_y_offset: float = -0.07

    # Variance circle (svr only). The outer radius is NOT configured -- it is derived so
    # the circle spans the row band (see _flow_geometry). Knobs: horizontal gap to the
    # pipeline, inner-circle size/offset, label font size, and question-mark placement.
    circle_gap: float = 0.35
    inner_radius_frac: float = 0.52
    inner_offset_x: float = -0.28
    inner_offset_y: float = 0.34
    circle_label_size: float = 8.0
    circle_outer_label: str = "All CA1\nvariance"
    circle_inner_label: str = "placefield\nvariance"
    circle_outer_label_y: float = -0.58
    question_mark: str = "?"
    question_size: float = 30.0
    question_x: float = 0.5
    question_y: float = 0.05

    # Export. bbox_inches is None so the saved file is exactly the solved canvas -- in
    # particular exactly ``fig_width`` wide, which is the whole point. "tight" would crop
    # back to the ink instead and give up that guarantee.
    bbox_inches: str | None = None
    transparent: bool = False


def _flow_colors(style: FlowStyle, cfg: FlowSchematicConfig) -> tuple[str, str, str]:
    """Return (facecolor, edgecolor, textcolor) for a node style."""
    if style == "pf":
        return cfg.pf_face, cfg.pf_edge, cfg.pf_text
    if style == "full":
        return cfg.full_face, cfg.full_edge, cfg.full_text
    return cfg.process_face, cfg.process_edge, cfg.process_text


def _role_width(role: FlowRole, cfg: FlowSchematicConfig) -> float:
    """Default box width in layout units for a pipeline role."""
    return {
        "pair": cfg.pair_width,
        "process": cfg.process_width,
        "spectrum": cfg.spectrum_width,
        "outcome": cfg.outcome_width,
    }[role]


def _flow_layout(variant: FlowVariant, cfg: FlowSchematicConfig) -> tuple[list[float], list[float], list[float]]:
    """Compute shared column geometry.

    Returns
    -------
    tuple of lists
        (column_widths, column_lefts, connector_widths). Columns are shared across rows
        so every row stays aligned; each width is the max requested by any row.
    """
    n_cols = len(variant.rows[0].cells)
    if any(len(row.cells) != n_cols for row in variant.rows):
        raise ValueError("all rows in a variant must have the same number of cells")

    widths: list[float] = []
    for col in range(n_cols):
        candidates = []
        for row in variant.rows:
            node = row.cells[col]
            candidates.append(node.width if node.width is not None else _role_width(node.role, cfg))
        widths.append(max(candidates))

    connector_widths: list[float] = []
    for col in range(n_cols - 1):
        kinds = {row.connectors[col] for row in variant.rows}
        connector_widths.append(cfg.dot_gap if kinds == {"dot"} else cfg.arrow_len)

    lefts: list[float] = []
    cursor = 0.0
    for col in range(n_cols):
        lefts.append(cursor)
        cursor += widths[col]
        if col < n_cols - 1:
            cursor += connector_widths[col]
    return widths, lefts, connector_widths


def _draw_flow_node(ax: Axes, node: FlowNode, left: float, width: float, y_center: float, cfg: FlowSchematicConfig) -> None:
    """Draw one rounded box with its label and optional corner superscript."""
    face, edge, text_color = _flow_colors(node.style, cfg)
    bottom = y_center - cfg.box_height / 2

    ax.add_patch(
        FancyBboxPatch(
            (left, bottom),
            width,
            cfg.box_height,
            boxstyle=f"round,pad=0,rounding_size={cfg.box_rounding}",
            facecolor=face,
            edgecolor=edge,
            linewidth=cfg.box_linewidth,
            zorder=_Z_BOX,
        )
    )

    fontsize = cfg.pair_label_size if node.role == "pair" else cfg.label_size
    ax.text(
        left + width / 2,
        y_center,
        node.text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight=cfg.font_weight,
        linespacing=cfg.linespacing,
        zorder=_Z_TEXT,
    )

    if node.superscript:
        ax.text(
            left + width - cfg.superscript_pad,
            y_center + cfg.box_height / 2 - cfg.superscript_pad,
            node.superscript,
            ha="right",
            va="top",
            color=text_color,
            fontsize=cfg.superscript_size,
            fontweight=cfg.font_weight,
            zorder=_Z_TEXT,
        )


def _draw_flow_arrow(ax: Axes, x0: float, x1: float, y: float, cfg: FlowSchematicConfig) -> None:
    """Draw a horizontal block arrow from x0 to x1, behind the boxes."""
    ax.add_patch(
        FancyArrow(
            x0,
            y,
            x1 - x0,
            0.0,
            width=cfg.arrow_shaft,
            head_width=cfg.arrow_head_width,
            head_length=min(cfg.arrow_head_length, abs(x1 - x0)),
            length_includes_head=True,
            color=cfg.arrow_color,
            zorder=_Z_ARROW,
        )
    )


def _draw_variance_circle(ax: Axes, center_x: float, center_y: float, radius: float, cfg: FlowSchematicConfig) -> None:
    """Draw the nested placefield-variance / all-CA1-variance circle at a derived radius."""
    ax.add_patch(Circle((center_x, center_y), radius, facecolor=cfg.full_face, edgecolor="none", zorder=_Z_BOX))

    inner_radius = radius * cfg.inner_radius_frac
    inner_x = center_x + radius * cfg.inner_offset_x
    inner_y = center_y + radius * cfg.inner_offset_y
    ax.add_patch(Circle((inner_x, inner_y), inner_radius, facecolor=cfg.pf_face, edgecolor="none", zorder=_Z_BOX))

    ax.text(
        inner_x,
        inner_y,
        cfg.circle_inner_label,
        ha="center",
        va="center",
        color=cfg.pf_text,
        fontsize=cfg.circle_label_size,
        linespacing=cfg.linespacing,
        zorder=_Z_TEXT,
    )
    ax.text(
        center_x,
        center_y + radius * cfg.circle_outer_label_y,
        cfg.circle_outer_label,
        ha="center",
        va="center",
        color=cfg.full_text,
        fontsize=cfg.circle_label_size,
        linespacing=cfg.linespacing,
        zorder=_Z_TEXT,
    )
    ax.text(
        center_x + radius * cfg.question_x,
        center_y + radius * cfg.question_y,
        cfg.question_mark,
        ha="center",
        va="center",
        color=cfg.pf_face,
        fontsize=cfg.question_size,
        fontweight=cfg.font_weight,
        zorder=_Z_TEXT,
    )


@dataclass(frozen=True)
class _FlowGeometry:
    """Laid-out positions (layout units) shared by the renderer and the figsize helper.

    Bounds are the tight extent of the drawing itself; the origin offsets are applied
    later, when the layout is placed on the canvas.
    """

    widths: list[float]
    lefts: list[float]
    row_centers: list[float]
    mid_y: float
    circle_center_x: float
    circle_center_y: float
    circle_radius: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


def _flow_geometry(variant: FlowVariant, cfg: FlowSchematicConfig, unit_scale: float) -> _FlowGeometry:
    """Lay out one variant and return its positions and tight bounds in layout units.

    The variance circle (svr) is sized to span the row band exactly: its top meets the top
    of the first row's title text and its bottom meets the bottom of the last row's boxes,
    so its radius and vertical center are derived here, not configured.

    ``unit_scale`` (inches per layout unit) is needed only to convert the title's point
    size into layout units; it is what makes the layout mildly self-referential and is
    solved for in :func:`_flow_metrics`.
    """
    widths, lefts, _ = _flow_layout(variant, cfg)
    flow_right = lefts[-1] + widths[-1]
    row_centers = [-i * cfg.row_pitch for i in range(len(variant.rows))]
    mid_y = float(np.mean(row_centers))

    # Row band: top of the first row's title text down to the bottom of the last row's boxes.
    top = row_centers[0] + cfg.box_height / 2 + cfg.title_pad + cfg.title_size / 72 / unit_scale
    bottom = row_centers[-1] - cfg.box_height / 2

    circle_center_x = 0.0
    circle_center_y = mid_y
    circle_radius = 0.0
    if variant.show_circle:
        circle_radius = (top - bottom) / 2
        circle_center_y = (top + bottom) / 2
        circle_center_x = flow_right + cfg.circle_gap + circle_radius
        content_right = circle_center_x + circle_radius
    else:
        content_right = flow_right

    return _FlowGeometry(
        widths=widths,
        lefts=lefts,
        row_centers=row_centers,
        mid_y=mid_y,
        circle_center_x=circle_center_x,
        circle_center_y=circle_center_y,
        circle_radius=circle_radius,
        x_min=0.0,
        x_max=content_right,
        y_min=bottom,
        y_max=top,
    )


def default_flow_config(variant: str) -> FlowSchematicConfig:
    """The tuned default config for a variant.

    Layers the variant's entries in ``_VARIANT_PRESETS`` onto the (dim-tuned) dataclass
    defaults. This is what :func:`flow_schematic` uses when no config is supplied.

    Parameters
    ----------
    variant : str
        Registry key, e.g. "dim" or "svr".

    Returns
    -------
    FlowSchematicConfig
    """
    if variant not in _VARIANT_PRESETS:
        raise ValueError(f"unknown variant {variant!r}; options are {sorted(_VARIANT_PRESETS)}")
    return FlowSchematicConfig(variant=variant, **_VARIANT_PRESETS[variant])


@dataclass(frozen=True)
class FlowMetrics:
    """The solved page: layout positions plus the scale and canvas they imply."""

    geom: _FlowGeometry
    unit_scale: float
    fig_width: float
    fig_height: float


def _flow_variant(cfg: FlowSchematicConfig) -> FlowVariant:
    if cfg.variant not in FLOW_VARIANTS:
        raise ValueError(f"unknown variant {cfg.variant!r}; options are {sorted(FLOW_VARIANTS)}")
    return FLOW_VARIANTS[cfg.variant]


def flow_metrics(cfg: FlowSchematicConfig) -> FlowMetrics:
    """Solve the page for a config: scale, canvas size and laid-out geometry.

    Width is the fixed input. ``unit_scale`` is whatever makes the margined drawing exactly
    that wide, and the height is then read off the same scale, so the canvas always fits its
    contents. The drawing width depends (weakly) on the scale itself -- the title's point
    size becomes layout-unit headroom, which the svr variant's circle radius picks up -- so
    the equation is solved by fixed-point iteration; it converges in a couple of steps, and
    immediately for circle-free variants.

    Parameters
    ----------
    cfg : FlowSchematicConfig

    Returns
    -------
    FlowMetrics
    """
    variant = _flow_variant(cfg)
    fig_width = float(cfg.fig_width if cfg.fig_width is not None else variant.fig_width)

    unit_scale = 1.0
    for _ in range(32):
        new_scale = fig_width / (2 * cfg.margin_x + _flow_geometry(variant, cfg, unit_scale).width)
        converged = abs(new_scale - unit_scale) < 1e-9
        unit_scale = new_scale
        if converged:
            break

    geom = _flow_geometry(variant, cfg, unit_scale)
    return FlowMetrics(
        geom=geom,
        unit_scale=unit_scale,
        fig_width=fig_width,
        fig_height=unit_scale * (2 * cfg.margin_y + geom.height),
    )


def flow_figsize(cfg: FlowSchematicConfig) -> tuple[float, float]:
    """Canvas size in inches for a config: the fixed width and the height it implies.

    Parameters
    ----------
    cfg : FlowSchematicConfig

    Returns
    -------
    tuple of (float, float)
        Width and height in inches.
    """
    metrics = flow_metrics(cfg)
    return metrics.fig_width, metrics.fig_height


def make_flow_schematic(cfg: FlowSchematicConfig | None = None) -> tuple[Figure, Axes]:
    """Render a registered pipeline schematic ("dim" or "svr").

    Parameters
    ----------
    cfg : FlowSchematicConfig or None
        Full style/layout config; a default (variant "dim") is created when None.

    Returns
    -------
    tuple of (Figure, Axes)
    """
    cfg = cfg or FlowSchematicConfig()
    variant = _flow_variant(cfg)
    metrics = flow_metrics(cfg)
    fig_w, fig_h = metrics.fig_width, metrics.fig_height

    plt.rcParams.update(
        {
            "font.family": cfg.font_family,
            "mathtext.fontset": "custom",
            "mathtext.rm": cfg.font_family,
            "mathtext.it": f"{cfg.font_family}:italic",
            "mathtext.bf": f"{cfg.font_family}:bold",
            "svg.fonttype": "none",  # keep text editable in Illustrator
            "pdf.fonttype": 42,
        }
    )

    geom = metrics.geom
    widths, lefts = geom.widths, geom.lefts
    row_centers = geom.row_centers
    mid_y = geom.mid_y
    circle_center_x = geom.circle_center_x

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=cfg.dpi)
    fig.patch.set_facecolor(cfg.background)

    # The axes IS the page: it fills the figure and its view is exactly the drawing plus
    # the margins. Both canvas dimensions came from that same view times unit_scale, so the
    # x and y scales are equal by construction -- no aspect-driven refitting, and nothing
    # can fall off the edge.
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(cfg.background)
    ax.set_xlim(geom.x_min - cfg.margin_x, geom.x_max + cfg.margin_x)
    ax.set_ylim(geom.y_min - cfg.margin_y, geom.y_max + cfg.margin_y)
    ax.axis("off")
    # Titles center on the pipeline (the box row), not the whole canvas. For dim the
    # pipeline fills the canvas so this is figure-centered; for svr it keeps the titles
    # over the boxes rather than drifting right toward the circle.
    pipeline_center_x = (lefts[0] + lefts[-1] + widths[-1]) / 2

    for row, y_center in zip(variant.rows, row_centers, strict=True):
        for col, node in enumerate(row.cells):
            _draw_flow_node(ax, node, lefts[col], widths[col], y_center, cfg)

        for col, connector in enumerate(row.connectors):
            gap_start = lefts[col] + widths[col]
            gap_end = lefts[col + 1]
            if connector == "dot":
                ax.add_patch(
                    Circle(
                        ((gap_start + gap_end) / 2, y_center),
                        cfg.dot_radius,
                        facecolor=cfg.arrow_color,
                        edgecolor="none",
                        zorder=_Z_BOX,
                    )
                )
            else:
                _draw_flow_arrow(ax, gap_start, gap_end, y_center, cfg)

        ax.text(
            pipeline_center_x,
            y_center + cfg.box_height / 2 + cfg.title_pad,
            row.title,
            ha="center",
            va="bottom",
            color=cfg.title_color,
            fontsize=cfg.title_size,
            fontweight=cfg.font_weight,
            zorder=_Z_TEXT,
        )

    if variant.show_circle:
        # Divide arrow: left edge aligned with the sum-of-spectral-mass boxes (left edge of
        # the divide_start_col column), running into the circle, below the row midline.
        divide_col = variant.divide_start_col
        box_left = lefts[divide_col]
        box_right = lefts[divide_col] + widths[divide_col]
        x1 = circle_center_x - geom.circle_radius - cfg.circle_gap / 2
        divide_y = mid_y + cfg.divide_y_offset
        _draw_flow_arrow(ax, box_left, x1, divide_y, cfg)
        # "divide" centered on the boxes, above the arrow.
        ax.text(
            (box_left + box_right) / 2,
            divide_y + cfg.arrow_shaft / 2 + cfg.divide_label_pad,
            variant.divide_label,
            ha="center",
            va="bottom",
            color=cfg.title_color,
            fontsize=cfg.divide_label_size,
            fontweight=cfg.font_weight,
            zorder=_Z_TEXT,
        )
        _draw_variance_circle(ax, circle_center_x, geom.circle_center_y, geom.circle_radius, cfg)

    # The page is sized around the box geometry, but type is measured in points and can
    # still spill past it (an over-long label, an oversized title). The axes fills the
    # canvas, so clipping to it clips to the page. Patches clip by default; text does not,
    # and unclipped text still exists as geometry -- a tight bounding box (Jupyter's inline
    # backend uses one) would expand to swallow it and quietly change the figure width.
    for text in ax.texts:
        text.set_clip_on(True)

    return fig, ax


def save_flow_schematic(
    output_stem: str | Path,
    cfg: FlowSchematicConfig | None = None,
    formats: Sequence[str] = ("svg", "png"),
    fig_width: float | None = None,
) -> list[Path]:
    """Render and save a pipeline schematic in one or more formats.

    Parameters
    ----------
    output_stem : str or Path
        Output path without extension; one file is written per entry in ``formats``.
    cfg : FlowSchematicConfig or None
    formats : Sequence[str]
    fig_width : float or None
        Canvas width in inches, overriding ``cfg.fig_width``. The height follows from the
        layout. The file is written at exactly that size (``cfg.bbox_inches`` is None by
        default; setting it to "tight" would crop back to the ink).
    """
    cfg = cfg or FlowSchematicConfig()
    if fig_width is not None:
        cfg = replace(cfg, fig_width=fig_width)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = make_flow_schematic(cfg)
    paths: list[Path] = []
    try:
        for extension in formats:
            # Append rather than with_suffix: a stem like "figure_1.5in" would otherwise
            # have its ".5in" stripped as a suffix and silently overwrite a sibling file.
            path = output_stem.with_name(f"{output_stem.name}.{extension.lower().lstrip('.')}")
            fig.savefig(
                path,
                dpi=cfg.dpi,
                bbox_inches=cfg.bbox_inches,
                transparent=cfg.transparent,
                facecolor=cfg.background,
            )
            paths.append(path)
    finally:
        plt.close(fig)
    return paths


_DEFAULT_FLOW_CONFIG_DIR = Path(__file__).parent / "configs"

# Viewer-only footprint tint: shows where the canvas edge falls against the white GUI
# panel. Deliberately not a FlowSchematicConfig field so it can never reach an export.
_FOOTPRINT_FACECOLOR = "#F4F4F4"


def save_flow_config(cfg: FlowSchematicConfig, path: str | Path) -> Path:
    """Write a config to JSON.

    Parameters
    ----------
    cfg : FlowSchematicConfig
    path : str or Path
        Destination; a ``.json`` suffix is added when missing and parent directories
        are created.

    Returns
    -------
    Path
        The file that was written.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        # Append rather than with_suffix: a name like "flow_v1.2" has a spurious suffix.
        path = path.with_name(path.name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return path


def load_flow_config(path: str | Path) -> FlowSchematicConfig:
    """Read a config previously written by :func:`save_flow_config`.

    Keys that are no longer :class:`FlowSchematicConfig` fields (e.g. a since-removed
    ``circle_radius``) are dropped, so configs saved by older versions still load.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(FlowSchematicConfig)}
    data = {k: v for k, v in data.items() if k in known}
    return FlowSchematicConfig(**data)


# FlowSchematicConfig float fields exposed as live Syd sliders, with (min, max, step).
_FLOW_TUNABLES: dict[str, tuple[float, float, float]] = {
    "margin_x": (0.0, 3.0, 0.01),
    "margin_y": (0.0, 3.0, 0.01),
    "box_height": (0.05, 4.0, 0.01),
    "pair_width": (0.05, 8.0, 0.01),
    "process_width": (0.05, 8.0, 0.01),
    "spectrum_width": (0.05, 8.0, 0.01),
    "outcome_width": (0.05, 8.0, 0.01),
    "box_rounding": (0.0, 0.5, 0.005),
    "box_linewidth": (0.0, 6.0, 0.1),
    "dot_gap": (0.0, 2.0, 0.01),
    "dot_radius": (0.0, 0.4, 0.005),
    "arrow_len": (0.0, 3.0, 0.01),
    "arrow_shaft": (0.0, 0.6, 0.005),
    "arrow_head_width": (0.0, 1.2, 0.01),
    "arrow_head_length": (0.0, 1.0, 0.01),
    "row_pitch": (0.05, 4.0, 0.01),
    "title_pad": (0.0, 1.0, 0.01),
    "title_size": (2.0, 36.0, 0.5),
    "label_size": (2.0, 36.0, 0.5),
    "pair_label_size": (2.0, 36.0, 0.5),
    "superscript_size": (2.0, 36.0, 0.5),
    "superscript_pad": (0.0, 0.4, 0.005),
    "linespacing": (0.5, 2.5, 0.01),
    "divide_label_size": (2.0, 36.0, 0.5),
    "divide_label_pad": (0.0, 0.6, 0.01),
    "divide_y_offset": (-1.0, 1.0, 0.01),
    "circle_gap": (0.0, 2.0, 0.01),
    "inner_radius_frac": (0.1, 0.95, 0.01),
    "inner_offset_x": (-0.8, 0.8, 0.01),
    "inner_offset_y": (-0.8, 0.8, 0.01),
    "circle_label_size": (2.0, 36.0, 0.5),
    "circle_outer_label_y": (-0.95, 0.95, 0.01),
    "question_size": (2.0, 60.0, 0.5),
    "question_x": (-0.95, 0.95, 0.01),
    "question_y": (-0.95, 0.95, 0.01),
}


class FlowSchematicViewer(Viewer):
    """Interactive pipeline schematic driven by a :class:`FlowSchematicConfig`.

    The variant registry key plus every geometry/typography field in ``_FLOW_TUNABLES``
    is a live control; colors and label text come straight from ``config``. Every font
    control is a raw point size, and each pipeline role (pair / process / spectrum /
    outcome) has its own width slider. ``fig_width`` is the only physical dimension: the
    canvas is always that wide, the scale is solved to fill it, and the height follows
    from the layout -- so widening a box grows the page rather than overflowing it.

    The ``config_name`` text box and "Save config JSON" button dump the live parameters
    to ``config_dir/<config_name>.json``, reloadable with :func:`load_flow_config`.

    ``show_footprint`` tints the canvas and outlines its border so the figure extent is
    visible against the GUI panel. It is a viewer-only display aid: it is not a
    :class:`FlowSchematicConfig` field, so it never reaches a saved config or an export.
    """

    def __init__(self, config: FlowSchematicConfig, config_dir: str | Path | None = None):
        self.cfg = config
        self.config_dir = Path(config_dir) if config_dir is not None else _DEFAULT_FLOW_CONFIG_DIR

        self.add_selection("variant", value=config.variant, options=sorted(FLOW_VARIANTS))
        self.add_text("config_name", value=config.variant)
        self.add_button("save_config", label="Save config JSON", callback=self.save_config, replot=False)

        self.add_boolean("show_footprint", value=True)
        # None means "the variant's width"; resolve it so the slider starts somewhere real.
        # Switching variant afterwards therefore keeps the slider's width, not the new
        # variant's -- reset it by hand when comparing the two at their publication sizes.
        self.add_float("fig_width", value=flow_metrics(config).fig_width, min=0.5, max=30.0, step=0.025)
        for name, (lo, hi, step) in _FLOW_TUNABLES.items():
            # Widen the slider to admit the incoming value; syd otherwise clamps it to the
            # range and the viewer would silently render something other than the config.
            value = float(getattr(config, name))
            self.add_float(name, value=value, min=min(lo, value), max=max(hi, value), step=step)

    def config_from_state(self, state) -> FlowSchematicConfig:
        """Build the config the current control values describe."""
        return replace(
            self.cfg,
            variant=state["variant"],
            fig_width=state["fig_width"],
            **{name: state[name] for name in _FLOW_TUNABLES},
        )

    def save_config(self, state) -> None:
        """Button callback: write the live parameters to ``config_dir/<config_name>.json``."""
        name = state["config_name"].strip() or state["variant"]
        path = save_flow_config(self.config_from_state(state), self.config_dir / name)
        print(f"Saved flow schematic config to {path}")

    def plot(self, state):
        fig, _ = make_flow_schematic(self.config_from_state(state))
        if state["show_footprint"]:
            fig.patch.set_facecolor(_FOOTPRINT_FACECOLOR)
        return fig


def flow_schematic(
    variant: str = "dim",
    fig_width: float | None = None,
    config: FlowSchematicConfig | None = None,
    config_dir: str | Path | None = None,
    return_syd_viewer: bool = False,
    **overrides,
):
    """Pipeline schematic for the spectra ("dim") and shared-variance-ratio ("svr") analyses.

    Parameters
    ----------
    variant : str
        Registry key: "dim" (two matched spectra, each reduced to a dimensionality) or
        "svr" (PF-shared vs full spectral mass, divided, with the variance circle).
    fig_width : float or None
        Canvas width in inches -- the figure's one fixed dimension. The layout is scaled to
        fill it and the height follows from what the drawing needs. None keeps whatever
        ``config`` specifies, which by default falls back to the variant's publication
        width (:attr:`FlowVariant.fig_width`).
    config : FlowSchematicConfig or None
        Full style/layout config. When None, the variant's tuned default
        (:func:`default_flow_config`) is used.
    config_dir : str or Path or None
        Directory the viewer's "Save config JSON" button writes into. Defaults to
        ``schematics/configs``.
    return_syd_viewer : bool
        If True, return the un-deployed Syd viewer instead of a rendered figure.
    **overrides
        Any :class:`FlowSchematicConfig` field, applied on top of ``config``.

    Returns
    -------
    Figure or Viewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    base = config if config is not None else default_flow_config(variant)
    cfg = replace(base, variant=variant, **overrides)
    if fig_width is not None:
        cfg = replace(cfg, fig_width=fig_width)
    viewer = FlowSchematicViewer(cfg, config_dir=config_dir)

    if return_syd_viewer:
        return viewer

    # The footprint tint is a GUI aid only; the returned figure is export-ready.
    viewer.update_boolean("show_footprint", value=False)
    fig = viewer.plot(viewer.state)
    plt.show()
    return fig
