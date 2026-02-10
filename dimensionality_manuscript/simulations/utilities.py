"""
Utility functions for dimensionality analysis simulations.

This module provides helper functions for generating orthogonal vectors
and orthonormal bases.
"""

from typing import Optional
import numpy as np
import numpy.typing as npt


def get_orthogonal_direction(vector: npt.NDArray[np.floating], eps: float = 1e-12) -> npt.NDArray[np.floating]:
    """
    Get a unit vector orthogonal to the given vector.

    For 2D vectors, this maintains the behavior of the original
    `get_orthogonal_direction_dim2` function. For higher dimensions,
    it returns a vector from the kernel/null space of the input vector.

    Parameters
    ----------
    vector : npt.NDArray[np.floating]
        Input vector of shape (N,) where N >= 2.
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-12.

    Returns
    -------
    npt.NDArray[np.floating]
        Unit vector orthogonal to the input vector, same dtype as input.

    Raises
    ------
    ValueError
        If vector is not 1D or has fewer than 2 elements.

    Examples
    --------
    >>> v = np.array([1.0, 2.0])
    >>> orth = get_orthogonal_direction(v)
    >>> np.abs(np.dot(v, orth)) < 1e-10
    True

    >>> v = np.array([1.0, 2.0, 3.0])
    >>> orth = get_orthogonal_direction(v)
    >>> np.abs(np.dot(v, orth)) < 1e-10
    True
    """
    vector = np.asarray(vector, dtype=float)

    if vector.ndim != 1:
        raise ValueError(f"Input must be a 1D array, got shape {vector.shape}")

    if vector.shape[0] < 2:
        raise ValueError(f"Input must have at least 2 elements, got {vector.shape[0]}")

    # Normalize input vector
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("Input vector is too close to zero")
    vector = vector / norm

    # Special case for 2D: maintain original behavior
    if vector.shape[0] == 2:
        c = 1.0
        d = -vector[0] / (vector[1] + eps)
        kernel = np.array([c, d], dtype=vector.dtype)
        kernel = kernel / np.linalg.norm(kernel)
        return kernel

    # For N > 2: use QR decomposition to find orthogonal basis
    # Create a matrix with the input vector as first column
    # and random vectors for the rest, then QR decompose
    N = vector.shape[0]

    # Create a matrix with the input vector as first column
    # and random vectors for the remaining columns
    # Using random vectors ensures robustness even if input is close to a basis vector
    rng = np.random.default_rng()
    A = np.zeros((N, N), dtype=vector.dtype)
    A[:, 0] = vector

    # Fill remaining columns with random vectors
    for i in range(1, N):
        A[:, i] = rng.standard_normal(N).astype(vector.dtype)

    # QR decomposition: Q will have orthonormal columns
    # The first column is the normalized input vector
    # The second column is orthogonal to the first
    Q, R = np.linalg.qr(A)

    # Return the second column (index 1) which is orthogonal to the first
    return Q[:, 1]


def generate_orthonormal(n: int, rank: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> npt.NDArray[np.floating]:
    """
    Generate an orthonormal matrix using QR decomposition.

    Parameters
    ----------
    n : int
        Number of rows (dimension of the space).
    rank : int, optional
        Number of columns (rank of the matrix). If None, defaults to n.
        Must satisfy 1 <= rank <= n.
    rng : np.random.Generator, optional
        Random number generator. If None, uses the default generator.

    Returns
    -------
    npt.NDArray[np.floating]
        Orthonormal matrix of shape (n, rank) with orthonormal columns.

    Raises
    ------
    ValueError
        If rank > n or rank < 1.

    Examples
    --------
    >>> Q = generate_orthonormal(5, rank=3)
    >>> Q.shape
    (5, 3)
    >>> np.allclose(Q.T @ Q, np.eye(3))
    True
    """
    if rank is None:
        rank = n

    if rank < 1:
        raise ValueError(f"Rank must be at least 1, got {rank}")
    if rank > n:
        raise ValueError(f"Rank cannot be greater than n: {rank} > {n}")

    if rng is None:
        rng = np.random.default_rng()

    # Generate random matrix and use QR decomposition
    # Q will have orthonormal columns
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)

    # Return only the first 'rank' columns
    return Q[:, :rank]
