"""
Matrix operators for symmetric positive definite (SPD) matrices.

This module provides functions for computing matrix square roots, inverse square roots,
geometric means, and root sandwich operations on SPD matrices.
"""

import numpy as np
import numpy.typing as npt


def sqrtm_spd(A: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Compute the matrix square root of a symmetric positive definite matrix.

    For a symmetric positive definite matrix A, this computes A^(1/2) such that
    A^(1/2) @ A^(1/2) = A.

    Parameters
    ----------
    A : npt.NDArray[np.floating]
        Symmetric positive definite matrix of shape (N, N).

    Returns
    -------
    npt.NDArray[np.floating]
        Matrix square root of A, shape (N, N).
    """
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 0.0)  # Ensure non-negative eigenvalues
    return V @ np.diag(np.sqrt(w)) @ V.T


def invsqrtm_spd(A: npt.NDArray[np.floating], eps: float = 1e-12) -> npt.NDArray[np.floating]:
    """
    Compute the inverse matrix square root of a symmetric positive definite matrix.

    For a symmetric positive definite matrix A, this computes A^(-1/2) such that
    A^(-1/2) @ A^(-1/2) = A^(-1).

    Parameters
    ----------
    A : npt.NDArray[np.floating]
        Symmetric positive definite matrix of shape (N, N).
    eps : float, optional
        Minimum eigenvalue threshold to avoid division by zero. Default is 1e-12.

    Returns
    -------
    npt.NDArray[np.floating]
        Inverse matrix square root of A, shape (N, N).
    """
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)  # Clamp eigenvalues to avoid division by zero
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def geometric_mean_spd(A: npt.NDArray[np.floating], B: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Compute the matrix geometric mean of two symmetric positive definite matrices.

    The matrix geometric mean of A and B is defined as:
    G = A^(1/2) @ (A^(-1/2) @ B @ A^(-1/2))^(1/2) @ A^(1/2)

    This operation is symmetric: geometric_mean_spd(A, B) = geometric_mean_spd(B, A).

    Parameters
    ----------
    A : npt.NDArray[np.floating]
        First symmetric positive definite matrix, shape (N, N).
    B : npt.NDArray[np.floating]
        Second symmetric positive definite matrix, shape (N, N).

    Returns
    -------
    npt.NDArray[np.floating]
        Matrix geometric mean of A and B, shape (N, N).
    """
    Ar = sqrtm_spd(A)
    Ais = invsqrtm_spd(A)
    M = Ais @ B @ Ais
    Mr = sqrtm_spd(M)
    return Ar @ Mr @ Ar


def root_sandwich(A: npt.NDArray[np.floating], B: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Compute the root sandwich operation: A^(1/2) @ B @ A^(1/2).

    This operation is commonly used in cross-covariance analysis and subspace methods.
    It transforms matrix B by "sandwiching" it between the square root of A.

    Parameters
    ----------
    A : npt.NDArray[np.floating]
        Symmetric positive definite matrix to take the square root of, shape (N, N).
    B : npt.NDArray[np.floating]
        Matrix to be sandwiched, shape (N, N).

    Returns
    -------
    npt.NDArray[np.floating]
        Result of A^(1/2) @ B @ A^(1/2), shape (N, N).

    Notes
    -----
    The result is symmetric if B is symmetric. This operation is not symmetric
    in A and B: root_sandwich(A, B) != root_sandwich(B, A) in general.
    """
    Aroot = sqrtm_spd(A)
    return Aroot @ B @ Aroot
