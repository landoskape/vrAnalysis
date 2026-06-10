from typing import Literal, Tuple
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap


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


def _svr_2d(cov_stim: npt.NDArray[np.floating], cov_full: npt.NDArray[np.floating]) -> float:
    """SVR = sum(kappa_modes(stim, full)) / tr(full) for 2x2 covariances."""
    evals_s, evecs_s = np.linalg.eigh(cov_stim)
    evals_s = np.maximum(evals_s, 0.0)
    sqrt_s = evecs_s @ np.diag(np.sqrt(evals_s)) @ evecs_s.T
    kappa = np.sqrt(np.maximum(np.linalg.eigvalsh(sqrt_s @ cov_full @ sqrt_s), 0.0))
    return float(np.sum(kappa) / np.trace(cov_full))


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
    arrow_scale: float = 1.0,
    arrow_width: float = 0.05,
    point_alpha: float = 0.4,
    point_size: float = 8.0,
    linewidth: float = 1.0,
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
        Which nuisance config to show in ax_nuisance.
    arrow_scale : float
        Multiplier on arrow lengths in ax_schematic.
    arrow_width, point_alpha, point_size, linewidth : float
    """
    ax_stim, ax_nuisance, ax_schematic, ax_svr = ax

    v_s = cfg.stim_direction
    v_orth = np.array([-v_s[1], v_s[0]])

    cov_stim = cfg.stim_amplitude * np.outer(v_s, v_s) + cfg.stim_orth_amplitude * np.outer(v_orth, v_orth)
    cov_noise = cfg.noise_amplitude * np.eye(2)

    angles, amplitudes, directions = _build_nuisance_configs(cfg)

    cmap: Colormap = plt.get_cmap(cmap_name)
    colors = [cmap(i / max(cfg.n_nuisance - 1, 1)) for i in range(cfg.n_nuisance)]

    svr_values = []
    cov_nuisances = []
    for i in range(cfg.n_nuisance):
        v_n = directions[i]
        v_n_orth = np.array([-v_n[1], v_n[0]])
        cov_n = amplitudes[i] * np.outer(v_n, v_n) + cfg.nuisance_orth_amplitude * np.outer(v_n_orth, v_n_orth)
        cov_nuisances.append(cov_n)
        cov_full = cov_stim + cov_n + cov_noise
        svr_values.append(_svr_2d(cov_stim, cov_full))

    max_amp = max(cfg.stim_amplitude, float(np.max(amplitudes))) + cfg.noise_amplitude
    max_lim = max_amp * 4.5

    # --- ax_stim: stim scatter + ellipse ---
    samples_stim = np.random.multivariate_normal(np.zeros(2), cov_stim + cov_noise, size=cfg.n_samples)
    stim_ellipse = _get_covariance_ellipse(cov_stim + cov_noise)
    ax_stim.scatter(samples_stim[:, 0], samples_stim[:, 1], alpha=point_alpha, s=point_size, color=stim_color)
    ax_stim.plot(stim_ellipse[0], stim_ellipse[1], color=stim_color, linewidth=linewidth)
    ax_stim.set_xlim(-max_lim, max_lim)
    ax_stim.set_ylim(-max_lim, max_lim)
    ax_stim.set_aspect("equal")

    # --- ax_nuisance: example nuisance scatter + ellipse ---
    idx = example_nuisance_idx
    cov_n_ex = cov_nuisances[idx]
    samples_n = np.random.multivariate_normal(np.zeros(2), cov_n_ex + cov_noise, size=cfg.n_samples)
    n_ellipse = _get_covariance_ellipse(cov_n_ex + cov_noise)
    ax_nuisance.scatter(samples_n[:, 0], samples_n[:, 1], alpha=point_alpha, s=point_size, color=colors[idx])
    ax_nuisance.plot(n_ellipse[0], n_ellipse[1], color=colors[idx], linewidth=linewidth)
    ax_nuisance.set_xlim(-max_lim, max_lim)
    ax_nuisance.set_ylim(-max_lim, max_lim)
    ax_nuisance.set_aspect("equal")

    # --- ax_schematic: stim (black) + all nuisance (colored) arrows ---
    def _draw_bidirectional_arrow(axis: plt.Axes, dx: float, dy: float, color: str) -> None:
        kw = dict(head_width=arrow_width * 2, head_length=arrow_width, fc=color, ec=color, linewidth=linewidth)
        axis.arrow(0, 0, dx, dy, **kw)
        axis.arrow(0, 0, -dx, -dy, **kw)

    stim_len = cfg.stim_amplitude * arrow_scale
    _draw_bidirectional_arrow(ax_schematic, v_s[0] * stim_len, v_s[1] * stim_len, stim_color)
    for i in range(cfg.n_nuisance):
        n_len = amplitudes[i] * arrow_scale
        v_n = directions[i]
        _draw_bidirectional_arrow(ax_schematic, v_n[0] * n_len, v_n[1] * n_len, colors[i])
    ax_schematic.set_xlim(-max_lim, max_lim)
    ax_schematic.set_ylim(-max_lim, max_lim)
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
