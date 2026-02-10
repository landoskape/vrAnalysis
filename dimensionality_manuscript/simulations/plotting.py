from typing import Any, Tuple, Union
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


def plot_ellipse(
    ax: Axes,
    D: Union[npt.NDArray[np.floating], Tuple[float, float]],
    V: npt.NDArray[np.floating],
    mean: Union[Tuple[float, float], npt.NDArray[np.floating]] = (0.0, 0.0),
    r: float = 1.0,
    n: int = 256,
    **plot_kwargs: Any
) -> Tuple[Line2D, npt.NDArray[np.floating]]:
    """
    Plot the 2D ellipse defined by x = mean + V @ sqrt(D) @ u, ||u||=r.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to plot on.
    D : (2,) or (2,2) array_like
        Eigenvalues (as length-2 vector) or diagonal matrix.
    V : (2,2) array_like
        Eigenvectors as columns (typical np.linalg.eig / eigh output).
    mean : (2,) array_like, optional
        Center of the ellipse.
    r : float, optional
        Radius in "standard deviation units" (e.g. r=1 for 1-sigma).
    n : int, optional
        Number of points on the ellipse.
    **plot_kwargs :
        Passed to ax.plot (e.g. color='k', lw=2, alpha=0.8).

    Returns
    -------
    line : matplotlib.lines.Line2D
        The plotted line object.
    xy : (n,2) ndarray
        The ellipse points.
    """
    V = np.asarray(V, dtype=float)
    mean = np.asarray(mean, dtype=float).reshape(
        2,
    )

    D = np.asarray(D, dtype=float)
    if D.shape == (2, 2):
        evals = np.diag(D)
    else:
        evals = D.reshape(
            2,
        )

    # Guard against tiny negative eigenvalues from numerical error
    evals = np.maximum(evals, 0.0)

    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    circle = np.column_stack((np.cos(t), np.sin(t)))  # (n,2), ||u||=1

    # Map unit circle -> ellipse: u -> V @ diag(sqrt(evals)) @ (r*u)
    A = V @ np.diag(np.sqrt(evals))
    xy = mean + (r * circle) @ A.T  # (n,2)

    (line,) = ax.plot(xy[:, 0], xy[:, 1], **plot_kwargs)
    return line, xy
