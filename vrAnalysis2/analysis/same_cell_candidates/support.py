import math
import numpy as np
import numba as nb
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@nb.njit
def condensed_to_square_indices(k, n):
    """
    Convert a condensed matrix index k to square matrix indices (i, j).

    Parameters:
    k : int
        Index in the condensed matrix
    n : int
        Number of points in the original data

    Returns:
    (i, j) : tuple of ints
        Indices in the square distance matrix
    """
    # First solve for i
    i = n - 2 - math.floor(math.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    # Then solve for j
    j = k + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2

    if i >= n or j >= n or i < 0 or j < 0:
        raise ValueError("Indices out of bounds")

    return i, j


@nb.njit
def square_to_condensed_index(i, j, n):
    """
    Convert square matrix indices (i, j) to a condensed matrix index.

    Parameters:
    i, j : int
        Indices in the square distance matrix where i < j
    n : int
        Number of points in the original data

    Returns:
    k : int
        Index in the condensed matrix
    """
    if i >= n or j >= n or i < 0 or j < 0:
        raise ValueError("Indices out of bounds")

    if i < j:
        k = (n * (n - 1) // 2) - (n - i) * ((n - i) - 1) // 2 + j - i - 1
    else:
        k = (n * (n - 1) // 2) - (n - j) * ((n - j) - 1) // 2 + i - j - 1
    return k


@nb.njit(parallel=True, fastmath=True)
def dist_between_points(xpoints, ypoints):
    N = len(xpoints)
    max_k = square_to_condensed_index(N - 1, N - 1, N) + 1
    dists = np.zeros(max_k)
    for k in nb.prange(max_k):
        i, j = condensed_to_square_indices(k, N)
        dists[k] = np.sqrt((xpoints[i] - xpoints[j]) ** 2 + (ypoints[i] - ypoints[j]) ** 2)
    return dists


@nb.njit(parallel=True)
def pair_val_from_vec(vector):
    N = len(vector)
    max_k = square_to_condensed_index(N - 1, N - 1, N) + 1
    value1 = np.zeros(max_k)
    value2 = np.zeros(max_k)
    for k in nb.prange(max_k):
        i, j = condensed_to_square_indices(k, N)
        value1[k] = vector[i]
        value2[k] = vector[j]
    return value1, value2


@torch.no_grad()
def torch_corrcoef(data: np.ndarray, undefined_val: float = np.nan) -> np.ndarray:
    """
    Compute the correlation matrix of the input data using PyTorch on the GPU,
    then return back the numpy array on the CPU.

    Only use when your data matrix is huge!
    """
    data_gpu = torch.tensor(data).to(device)
    torch_corr = torch.corrcoef(data_gpu)
    if ~np.isnan(undefined_val):
        no_variance = torch.var(data_gpu, dim=1) == 0
        idx_no_variance = torch.where(no_variance)[0]
        torch_corr.scatter_(dim=0, index=idx_no_variance.view(-1, 1).expand(-1, torch_corr.size(1)), value=undefined_val)
        torch_corr.scatter_(dim=1, index=idx_no_variance.view(1, -1).expand(torch_corr.size(0), -1), value=undefined_val)
    torch_corr = torch_corr.to("cpu").numpy()
    return torch_corr
