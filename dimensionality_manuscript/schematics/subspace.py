from typing import Tuple
from dataclasses import dataclass, field
import numpy as np
from matplotlib import pyplot as plt


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
    max_lims = max_amplitude * 4.5
    ax_direction.arrow(
        0,
        0,
        cfg.stim_direction[0],
        cfg.stim_direction[1],
        head_width=0.1,
        head_length=0.1,
        fc=stim_color,
        ec=stim_color,
    )
    ax_direction.arrow(
        0,
        0,
        cfg.nuisance_direction[0],
        cfg.nuisance_direction[1],
        head_width=0.1,
        head_length=0.1,
        fc=nuisance_color,
        ec=nuisance_color,
    )
    ax_direction.arrow(
        0,
        0,
        -cfg.stim_direction[0],
        -cfg.stim_direction[1],
        head_width=0.1,
        head_length=0.1,
        fc=stim_color,
        ec=stim_color,
    )
    ax_direction.arrow(
        0,
        0,
        -cfg.nuisance_direction[0],
        -cfg.nuisance_direction[1],
        head_width=0.1,
        head_length=0.1,
        fc=nuisance_color,
        ec=nuisance_color,
    )
    ax_direction.set_xlim(max_lims * -1, max_lims)
    ax_direction.set_ylim(max_lims * -1, max_lims)

    # In ax_stimcov / ax_fullcov, generate samples of the stimulus and full(stim+nuisance), including independent noise
    # scatter the sample data, and plot the covariance ellipsoids with a thin line colored appropriately
    ax_stimcov.scatter(samples_stim[:, 0], samples_stim[:, 1], alpha=0.5, color=stim_color)
    ax_fullcov.scatter(samples_full[:, 0], samples_full[:, 1], alpha=0.5, color=nuisance_color)
    ax_stimcov.plot(stim_ellipse[0], stim_ellipse[1], color=stim_color, linewidth=2)
    ax_fullcov.plot(full_ellipse[0], full_ellipse[1], color=nuisance_color, linewidth=2)
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
