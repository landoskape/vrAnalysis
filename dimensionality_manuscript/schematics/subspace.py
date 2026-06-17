from typing import Literal, Tuple
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
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
    example_nuisance_angle: float = 0.0
    example_nuisance_amplitude: float = 1.0
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
    stim_color: str = "black",
    angle_cmap_name: str = "hsv",
    amplitude_cmap_name: str = "gist_heat",
    example_angle_idx: int | None = None,
    example_amplitude_idx: int | None = None,
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
    angle_cmap_name : str
        Colormap for angle-sweep arcs and SVR curve.
    amplitude_cmap_name : str
        Colormap for amplitude-sweep arrows and SVR curve.
    example_angle_idx : int or None
        Index into angle sweep to highlight; auto-computed if None.
    example_amplitude_idx : int or None
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
    angle_vals = np.linspace(0, np.pi, cfg.n_angle, endpoint=False)
    amplitude_vals = np.linspace(cfg.min_amplitude, cfg.max_amplitude, cfg.n_amplitude)

    # Clockwise distance from stim to each nuisance direction (in (0, π]); used by schematic + plots
    stim_angle_rad = float(np.arctan2(v_s[1], v_s[0]))
    cw_dists = np.array([(stim_angle_rad - a) % np.pi for a in angle_vals])
    # In covariance space 0° and 180° are identical; represent the 0-distance as π
    cw_dists = np.where(cw_dists < 1e-9, np.pi, cw_dists)

    # Auto-select example indices (nearest to cfg.example_nuisance_angle / amplitude)
    if example_angle_idx is None:
        example_angle_idx = int(np.argmin(np.abs(angle_vals - cfg.example_nuisance_angle)))
    if example_amplitude_idx is None:
        example_amplitude_idx = int(np.argmin(np.abs(amplitude_vals - cfg.example_nuisance_amplitude)))

    def _cov_nuisance(angle: float, amplitude: float) -> npt.NDArray[np.floating]:
        v_n = np.array([np.cos(angle), np.sin(angle)])
        v_n_orth = np.array([-v_n[1], v_n[0]])
        return amplitude * np.outer(v_n, v_n) + cfg.nuisance_orth_amplitude * np.outer(v_n_orth, v_n_orth)

    # SVR along angle sweep (amplitude fixed at cfg.example_nuisance_amplitude)
    svr_angle = []
    for a in angle_vals:
        cov_n = _cov_nuisance(a, cfg.example_nuisance_amplitude)
        svr_angle.append(_svr_2d(cov_stim, cov_stim + cov_n + cov_noise, mode=cfg.svr_mode))
    svr_angle = np.array(svr_angle)

    # SVR along amplitude sweep (angle fixed at cfg.example_nuisance_angle)
    svr_amplitude = []
    for amp in amplitude_vals:
        cov_n = _cov_nuisance(cfg.example_nuisance_angle, amp)
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

    # Sort angle sweep by cw_dist (0 → π); needed by schematic insets and angle panel
    rel_sort = np.argsort(cw_dists)
    cw_deg_sorted = np.degrees(cw_dists[rel_sort])
    svr_angle_sorted = svr_angle[rel_sort]

    example_nuisance_color = angle_colors[example_angle_idx]

    # --- Example nuisance covariance ---
    cov_n_ex = _cov_nuisance(cfg.example_nuisance_angle, cfg.example_nuisance_amplitude)

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
    ax_combined.scatter(samples_n[:, 0], samples_n[:, 1], alpha=point_alpha, s=point_size, color=example_nuisance_color)
    ax_combined.plot(stim_ellipse[0], stim_ellipse[1], color=stim_color, linewidth=linewidth, label="stim")
    ax_combined.plot(n_ellipse[0], n_ellipse[1], color=example_nuisance_color, linewidth=linewidth, label="nuisance")
    ax_combined.legend(frameon=True, loc="lower left", fontsize=fontsize)
    _set_dist_panel(ax_combined)

    # --- ax_full: full (stim+nuisance+noise) scatter, stim + nuisance ellipses overlaid ---
    cov_full_ex = cov_stim + cov_n_ex + cov_noise
    samples_full = np.random.multivariate_normal(np.zeros(2), cov_full_ex, size=cfg.n_samples)
    ax_full.scatter(samples_full[:, 0], samples_full[:, 1], alpha=point_alpha, s=point_size, color=stim_color, label="full data")
    ax_full.plot(stim_ellipse[0], stim_ellipse[1], color=stim_color, linewidth=linewidth, label="stim")
    ax_full.plot(n_ellipse[0], n_ellipse[1], color=example_nuisance_color, linewidth=linewidth, label="nuisance")
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

    v_n_amp = np.array([np.cos(cfg.example_nuisance_angle), np.sin(cfg.example_nuisance_angle)])
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

    angle_draw_order = sorted(range(cfg.n_angle), key=lambda i: -cw_dists[i])
    for i in angle_draw_order:
        _draw_arc_arrow(ax_schematic, stim_angle_rad, stim_angle_rad - cw_dists[i], angle_colors[i])

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
    ax_ang_ins.imshow(np.array([angle_colors[i] for i in rel_sort]).reshape(1, -1, 4), aspect="auto")
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
    ax_amplitude.scatter(
        amplitude_vals[example_amplitude_idx],
        svr_amplitude[example_amplitude_idx],
        color=example_nuisance_color,
        s=point_size * 8,
        zorder=3,
        edgecolors=stim_color,
        linewidths=linewidth,
    )
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
    cw_deg_plot = cw_deg_sorted % 180
    _plot_order = np.argsort(cw_deg_plot)
    ax_angle.plot(cw_deg_plot[_plot_order], svr_angle_sorted[_plot_order], color="gray", linewidth=linewidth, zorder=1)
    for k, i in enumerate(rel_sort):
        ax_angle.scatter(cw_deg_plot[k], svr_angle_sorted[k], color=angle_colors[i], s=point_size * 4, zorder=2)
    ax_angle.scatter(
        cw_deg_plot[example_angle_idx],
        svr_angle_sorted[example_angle_idx],
        color=example_nuisance_color,
        s=point_size * 8,
        zorder=3,
        edgecolors=stim_color,
        linewidths=linewidth,
    )
    ex_cw_deg = float(np.degrees(cw_dists[example_angle_idx]))
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

        # Reorder rows so y-axis = relative angle increasing from 0
        svr_grid_sorted = svr_grid[rel_sort, :]
        max_cw_deg = float(cw_deg_sorted[-1])
        im = ax_heatmap.imshow(
            svr_grid_sorted,
            origin="lower",
            aspect="auto",
            extent=[amplitude_vals[0], amplitude_vals[-1], 0, max_cw_deg],
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        ax_heatmap.scatter(
            cfg.example_nuisance_amplitude,
            ex_cw_deg,
            marker="x",
            color="white",
            s=point_size * 8,
            linewidths=linewidth * 2,
            zorder=3,
        )
        ax_heatmap.set_xlabel("Nuisance amplitude")
        ax_heatmap.set_ylabel("Relative nuisance angle (°)")
        plt.colorbar(im, ax=ax_heatmap, label="SVR")
