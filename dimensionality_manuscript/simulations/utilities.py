"""
Utility functions for dimensionality analysis simulations.

This module provides helper functions for generating orthogonal vectors
and orthonormal bases.
"""

from typing import Optional
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def generate_orthonormal(
    n: int,
    rank: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    *,
    kernel: Optional[npt.NDArray[np.floating]] = None,
    kernel_rtol: float = 1e-12,
) -> npt.NDArray[np.floating]:
    """
    Generate an orthonormal matrix with columns orthogonal to an optional kernel subspace.

    If `kernel` is provided, the returned columns span a random `rank`-dimensional
    subspace of the orthogonal complement of span(kernel).

    Parameters
    ----------
    n : int
        Ambient dimension (number of rows).
    rank : int, optional
        Number of columns to return. If None, defaults to n (or to the maximum feasible
        rank if `kernel` is provided).
    rng : np.random.Generator, optional
        Random number generator.
    kernel : array, optional
        Array of shape (n, k) whose columns span the subspace to be orthogonal to.
        Columns need not be orthonormal or independent.
    kernel_rtol : float
        Relative threshold used by SVD to determine numerical rank of `kernel`.

    Returns
    -------
    Q : ndarray
        Orthonormal matrix of shape (n, rank) with Q.T @ Q = I and, if kernel is not None,
        kernel.T @ Q ≈ 0.

    Raises
    ------
    ValueError
        If rank < 1, rank exceeds the available orthogonal-complement dimension,
        or if kernel has incompatible shape.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    if rng is None:
        rng = np.random.default_rng()

    if kernel is None:
        # Original behavior: random orthonormal columns in R^n
        if rank is None:
            rank = n
        if rank < 1:
            raise ValueError(f"Rank must be at least 1, got {rank}")
        if rank > n:
            raise ValueError(f"Rank cannot be greater than n: {rank} > {n}")

        A = rng.standard_normal((n, rank))
        Q, _ = np.linalg.qr(A, mode="reduced")
        return Q

    K = np.asarray(kernel, dtype=float)
    if K.ndim != 2 or K.shape[0] != n:
        raise ValueError(f"kernel must have shape (n, k); got {K.shape} with n={n}")

    # SVD to find an orthonormal basis for span(K) and its orthogonal complement
    # K = U S V^T, columns of U span column space of K.
    U, s, _ = np.linalg.svd(K, full_matrices=True)

    # Numerical rank of kernel
    if s.size == 0:
        k_rank = 0
    else:
        tol = kernel_rtol * s[0]
        k_rank = int(np.sum(s > tol))

    complement_dim = n - k_rank
    if complement_dim == 0:
        raise ValueError("kernel spans (numerically) all of R^n; no orthogonal complement exists.")

    if rank is None:
        rank = complement_dim

    if rank < 1:
        raise ValueError(f"Rank must be at least 1, got {rank}")
    if rank > complement_dim:
        raise ValueError(f"Requested rank={rank}, but orthogonal-complement dimension is {complement_dim}.")

    # Orthonormal basis for orthogonal complement: last n-k_rank columns of U
    Uc = U[:, k_rank:]  # shape (n, complement_dim)

    # Sample a random subspace within the complement and orthonormalize
    B = rng.standard_normal((complement_dim, rank))
    Qc, _ = np.linalg.qr(B, mode="reduced")  # (complement_dim, rank)
    Q = Uc @ Qc  # (n, rank), columns orthonormal and orthogonal to span(K)

    return Q


def random_orthonormal_complement(Q: np.ndarray, k: int, rng=np.random.default_rng()):
    """
    Return U with orthonormal columns, U^T U = I, and Q^T U = 0.
    Requires d - k >= k (i.e. d >= 2k) to get k columns.
    """
    d, kQ = Q.shape
    assert kQ == k
    A = rng.standard_normal((d, k))
    # Project A into the orthogonal complement of span(Q)
    A = A - Q @ (Q.T @ A)
    # Orthonormalize
    U, _ = np.linalg.qr(A)
    return U[:, :k]


def random_orthogonal(k: int, rng=np.random.default_rng()):
    M = rng.standard_normal((k, k))
    O, _ = np.linalg.qr(M)
    # Optional: enforce det=+1 (special orthogonal) if you care
    if np.linalg.det(O) < 0:
        O[:, 0] *= -1
    return O


def rotate_subspace_by_angle(Q: np.ndarray, theta: float, rng=np.random.default_rng()):
    """
    Rotate an orthonormal basis Q (d, k) by principal angle theta.
    Returns Q' with Q'^T Q' = I and principal angles all equal theta,
    assuming d >= 2k.
    """
    d, k = Q.shape
    if d < 2 * k:
        raise ValueError(f"Need d >= 2k for equal-angle rotation; got d={d}, k={k}.")
    U = random_orthonormal_complement(Q, k, rng)
    O = random_orthogonal(k, rng)
    return Q * np.cos(theta) + (U @ O) * np.sin(theta)


def find_commute_space_gated(A, B, K=50, num_steps=2000, lr=1e-2, lambda_gate=1e-3, prune_every=200, prune_thr=1e-2, min_dims=1, device="cuda"):
    A = torch.as_tensor(A, dtype=torch.float32, device=device)
    B = torch.as_tensor(B, dtype=torch.float32, device=device)
    n = A.shape[0]
    M = A @ B - B @ A

    W = torch.randn(n, K, device=device, requires_grad=True)
    log_g = torch.zeros(K, device=device, requires_grad=True)  # gates in R

    opt = torch.optim.Adam([W, log_g], lr=lr)

    alive = torch.ones(K, dtype=torch.bool, device=device)

    for t in tqdm(range(num_steps)):
        opt.zero_grad(set_to_none=True)

        # Orthonormal directions (economy QR). Keep only alive cols.
        W_alive = W[:, alive]
        Q, _ = torch.linalg.qr(W_alive, mode="reduced")  # n x k_alive

        g = F.softplus(log_g[alive])  # k_alive, positive
        V = Q * g.unsqueeze(0)  # n x k_alive (scaled)

        # Your energies (note: you're maximizing A,B energy while minimizing commutator)
        dot_A = (A @ V).pow(2).sum(dim=0)
        dot_B = (B @ V).pow(2).sum(dim=0)
        dot_M = (M @ V).pow(2).sum(dim=0)

        # Objective: maximize dot_A + dot_B, minimize dot_M, plus sparsity on g
        total = dot_A.sum() + dot_B.sum() - dot_M.sum() - lambda_gate * g.sum()

        # (Optional) small regularizer to prevent g blowing up if scale is free:
        # total = total - 1e-4 * (g.pow(2).sum())

        loss = -total
        loss.backward()
        opt.step()

        # Periodic hard prune
        if (t + 1) % prune_every == 0 and alive.sum() > min_dims:
            with torch.no_grad():
                g_full = torch.zeros_like(log_g)
                g_full[alive] = F.softplus(log_g[alive])

                # prune smallest gates
                to_prune = (g_full < prune_thr) & alive
                # keep at least min_dims
                if alive.sum() - to_prune.sum() < min_dims:
                    # keep top min_dims gates
                    vals = g_full[alive]
                    k_alive = vals.numel()
                    keep_k = min_dims
                    thresh = torch.topk(vals, keep_k, largest=True).values.min()
                    to_prune = (g_full < thresh) & alive

                alive[to_prune] = False

    # Final basis in data space (orthonormal Q, not scaled) and diagnostics
    with torch.no_grad():
        Q, _ = torch.linalg.qr(W[:, alive], mode="reduced")
        g = F.softplus(log_g[alive])
        V = Q  # return the subspace directions; keep g separately if you want
        props = {
            "gates": g.detach().cpu().numpy(),
            "k": int(alive.sum().item()),
            "comm_energy": (M @ V).pow(2).sum(dim=0).detach().cpu().numpy(),
            "A_energy": (A @ V).pow(2).sum(dim=0).detach().cpu().numpy(),
            "B_energy": (B @ V).pow(2).sum(dim=0).detach().cpu().numpy(),
        }

    return V.detach().cpu().numpy(), props


def find_commute_space(A, B, num_dims, num_steps=1000, learning_rate=0.01):
    A = torch.from_numpy(A).float().to(device)
    B = torch.from_numpy(B).float().to(device)
    n = A.shape[0]

    raw_V = torch.randn(n, num_dims, device=device, requires_grad=True)
    raw_V.data /= raw_V.norm(dim=0, keepdim=True)

    subtracted = A @ B - B @ A
    for _ in tqdm(range(num_steps)):
        V = raw_V / raw_V.norm(dim=0, keepdim=True)

        # Squared norms per column: ||M @ v||^2 as ((M @ V) ** 2).sum(dim=0)
        dot_A = ((A @ V) ** 2).sum(dim=0)
        dot_B = ((B @ V) ** 2).sum(dim=0)
        dot_subtraction = ((subtracted @ V) ** 2).sum(dim=0)
        # Gram matrix G[i,j] = v_i · v_j; orth penalty = sum of (v_i · v_j)^2 for i > j
        G = V.T @ V
        orth = torch.tril(G**2, diagonal=-1).sum()

        total = dot_A.sum() + dot_B.sum() - dot_subtraction.sum() - orth
        total.backward()
        raw_V.data.add_(raw_V.grad, alpha=learning_rate)
        raw_V.grad = None

    # Measure final properties
    candidates = raw_V / raw_V.norm(dim=0, keepdim=True)

    # Subspace projections
    proj_commute = torch.sum(candidates * (subtracted @ candidates), dim=0).cpu().detach().numpy()
    proj_A = torch.sum(candidates * (A @ candidates), dim=0).cpu().detach().numpy()
    proj_B = torch.sum(candidates * (B @ candidates), dim=0).cpu().detach().numpy()

    properties = dict(
        proj_commute=proj_commute,
        proj_A=proj_A,
        proj_B=proj_B,
    )
    return candidates.cpu().detach().numpy(), properties
