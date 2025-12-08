from typing import Union, Optional
import numpy as np
import torch
from torch.nn.functional import conv1d
from scipy.linalg import convolution_matrix
from tqdm import tqdm


@torch.no_grad()
def vector_correlation(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    covariance: bool = False,
    dim: int = -1,
    ignore_nan: bool = False,
) -> torch.Tensor:
    """
    Measure the correlation of every element in x with every element in y on dim=dim.

    If covariance=True, will measure the covariance instead of correlation.
    If ignore_nan=True, will ignore NaN values in the correlation calculation.

    Parameters
    ----------
    x : Union[torch.Tensor, np.ndarray]
        First input tensor. Must have the same shape as y.
    y : Union[torch.Tensor, np.ndarray]
        Second input tensor. Must have the same shape as x.
    covariance : bool
        If True, compute covariance instead of correlation (default is False).
    dim : int
        Dimension along which to compute correlation (default is -1).
    ignore_nan : bool
        If True, ignore NaN values in the calculation (default is False).

    Returns
    -------
    torch.Tensor
        The correlation (or covariance) between x and y along the specified dimension.
    """
    # Convert to torch tensors if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    assert x.shape == y.shape, "x and y need to have the same shape!"

    # Select appropriate functions based on ignore_nan flag
    mean_func = torch.nanmean if ignore_nan else torch.mean
    sum_func = torch.nansum if ignore_nan else torch.sum

    n = x.shape[dim]
    x_dev = x - mean_func(x, dim=dim, keepdim=True)
    y_dev = y - mean_func(y, dim=dim, keepdim=True)

    if not covariance:
        x_sample_std = torch.sqrt(sum_func(x_dev**2, dim=dim, keepdim=True) / (n - 1))
        y_sample_std = torch.sqrt(sum_func(y_dev**2, dim=dim, keepdim=True) / (n - 1))
        x_idx_valid = x_sample_std > 0
        y_idx_valid = y_sample_std > 0
        x_sample_std_corrected = x_sample_std + (~x_idx_valid).float()
        y_sample_std_corrected = y_sample_std + (~y_idx_valid).float()
    else:
        # For covariance, we don't normalize by standard deviation
        # Use scalar 1.0 which will broadcast correctly during division
        x_sample_std_corrected = 1.0
        y_sample_std_corrected = 1.0

    x_dev = x_dev / x_sample_std_corrected
    y_dev = y_dev / y_sample_std_corrected
    std = sum_func(x_dev * y_dev, dim=dim) / (n - 1)

    if not covariance:
        valid_mask = torch.squeeze(x_idx_valid & y_idx_valid)
        std = std * valid_mask.float()

    return std


def get_gauss_kernel(timestamps: Union[torch.Tensor, np.ndarray], width: float, nonzero: bool = True) -> np.ndarray:
    """
    Create Gaussian kernel (sum=1) around the timestamps array with width in units of timestamps.

    Parameters
    ----------
    timestamps : Union[torch.Tensor, np.ndarray]
        Array of timestamps/positions. If tensor, will be converted to numpy.
    width : float
        Width of the Gaussian kernel in units of timestamps.
    nonzero : bool, default=True
        If True, will remove zeros from the returned values (numerical zeros).
        This speeds up convolution_matrix computation.

    Returns
    -------
    np.ndarray
        Gaussian kernel array, normalized to sum to 1.
    """
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.cpu().numpy()

    kernel_domain = timestamps - np.mean(timestamps)
    kernel = np.exp(-(kernel_domain**2) / (2 * width) ** 2)
    kernel = kernel / np.sum(kernel)
    if nonzero:
        # since scipy.linalg.convolution_matrix only needs nonzero values, this is faster
        kernel = kernel[kernel > 0]
    return kernel


@torch.no_grad()
def gaussian_filter(
    data: torch.Tensor,
    smoothing_widths: Union[torch.Tensor, float, int],
    stimulus_positions: Optional[Union[torch.Tensor, np.ndarray]] = None,
    axis: int = -1,
) -> torch.Tensor:
    """
    Apply Gaussian smoothing filter to data along a specified axis.

    For cvPCA, data is typically (num_neurons, num_stimuli) and smoothing is applied
    along the stimulus dimension (axis=1). Each sample (e.g., neuron) is smoothed with
    its own smoothing width.

    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Input data tensor. Shape: (..., num_stimuli, ...) where num_stimuli is at `axis`.
    smoothing_widths : Union[torch.Tensor, float, int]
        Smoothing widths for Gaussian kernel. If tensor, should be shape (num_samples,)
        for per-sample smoothing. If scalar, same width is used for all samples.
    stimulus_positions : Optional[Union[torch.Tensor, np.ndarray]]
        Positions/timestamps for stimuli. If None, assumes evenly spaced [0, 1, ..., n-1].
    axis : int, default=-1
        Axis along which to apply smoothing (stimulus dimension).

    Returns
    -------
    torch.Tensor
        Smoothed data with same shape as input.
    """
    # Ensure torch tensor
    if isinstance(data, np.ndarray):
        data = torch.as_tensor(data)

    device = data.device
    dtype = data.dtype

    # Move target axis to last dimension for easier processing
    data_moved = torch.moveaxis(data, axis, -1)
    original_shape = data_moved.shape
    num_stimuli = original_shape[-1]

    # Flatten everything except the stimulus axis
    if data_moved.ndim > 2:
        num_samples = int(np.prod(original_shape[:-1]))
        data_flat = data_moved.reshape(num_samples, num_stimuli)
    else:
        num_samples = original_shape[0]
        data_flat = data_moved  # (num_samples, num_stimuli)

    # Set up stimulus positions (for kernel generation)
    if stimulus_positions is None:
        stimulus_positions = np.arange(num_stimuli, dtype=np.float32)
    elif isinstance(stimulus_positions, torch.Tensor):
        stimulus_positions = stimulus_positions.detach().cpu().numpy()
    else:
        stimulus_positions = np.asarray(stimulus_positions, dtype=np.float32)

    # Normalize smoothing_widths to a 1D tensor of length num_samples
    if isinstance(smoothing_widths, (int, float)):
        smoothing_widths_t = torch.full(
            (num_samples,),
            float(smoothing_widths),
            device=device,
            dtype=torch.float32,
        )
    else:
        smoothing_widths_t = torch.as_tensor(smoothing_widths, device=device, dtype=torch.float32).flatten()
        if smoothing_widths_t.numel() == 1:
            smoothing_widths_t = smoothing_widths_t.expand(num_samples)
        elif smoothing_widths_t.numel() != num_samples:
            raise ValueError(f"smoothing_widths must be scalar or have length {num_samples}, " f"got {smoothing_widths_t.numel()}")

    # Build one Gaussian kernel per sample (on CPU via your existing helper)
    widths_np = smoothing_widths_t.detach().cpu().numpy()
    kernels = []
    for w in widths_np:
        # Assumes get_gauss_kernel(stimulus_positions, width, nonzero=True)
        # returns a 1D kernel of length num_stimuli
        k = get_gauss_kernel(stimulus_positions, float(w), nonzero=False)
        kernels.append(k)

    kernels_np = np.stack(kernels, axis=0).astype(np.float32)  # (num_samples, num_stimuli)

    # Convert kernels to torch and reshape for grouped conv1d
    kernels_t = torch.from_numpy(kernels_np).to(device=device, dtype=dtype)
    # Shape: (out_channels=num_samples, in_channels_per_group=1, kernel_size=num_stimuli)
    kernels_t = kernels_t.unsqueeze(1)  # (num_samples, 1, num_stimuli)

    # Prepare data for grouped conv1d
    # We interpret each "sample" as a separate channel
    # Input: (batch=1, channels=num_samples, length=num_stimuli)
    data_conv = data_flat.to(dtype=dtype).unsqueeze(0)  # (1, num_samples, num_stimuli)

    # Grouped convolution: each input channel gets its own kernel
    padding = num_stimuli // 2  # 'same'-like for symmetric kernels
    smoothed_conv = conv1d(
        data_conv,
        weight=kernels_t,
        bias=None,
        padding=padding,
        groups=num_samples,
    )  # (1, num_samples, L_out)

    smoothed_flat = smoothed_conv.squeeze(0)[:, :num_stimuli]  # (num_samples, num_stimuli)

    # Reshape back to original shape
    smoothed_reshaped = smoothed_flat.reshape(*original_shape)

    # Move axis back to original position
    smoothed = torch.moveaxis(smoothed_reshaped, -1, axis)

    return smoothed


class VectorizedGoldenSectionSearch:
    """
    Support for managing vectorized golden section search optimization.

    This class manages the state and updates for golden section search when optimizing
    multiple independent 1D functions simultaneously (e.g., per-neuron smoothing widths).

    The golden section search algorithm finds the minimum of a 1D function by iteratively
    narrowing the search interval [a, b] using the golden ratio.
    """

    def __init__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        tolerance: float = 1e-5,
        max_iterations: int = 100,
    ):
        """
        Initialize golden section search for multiple independent optimizations.

        Parameters
        ----------
        a : torch.Tensor
            Lower bounds for search intervals. Shape: (n_optimizations,)
        b : torch.Tensor
            Upper bounds for search intervals. Shape: (n_optimizations,)
        tolerance : float, default=1e-5
            Convergence tolerance. Search stops when interval width < tolerance.
        max_iterations : int, default=100
            Maximum number of iterations before stopping.
        """
        assert a.shape == b.shape, "a and b must have the same shape"
        assert torch.all(a < b), "All elements of a must be less than b"

        self.a = a.clone()
        self.b = b.clone()
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Golden ratio constant
        self.invphi = (np.sqrt(5) - 1) / 2

        # Initialize search points
        self.c = self.b - (self.b - self.a) * self.invphi
        self.d = self.a + (self.b - self.a) * self.invphi

        # Track which optimizations have converged
        self.converged = torch.zeros(a.shape[0], dtype=torch.bool, device=a.device)
        self.iteration = 0

        # Store function values at c and d
        self.fc = None
        self.fd = None

    def max_iterations_possible(self) -> int:
        """Get the maximum number of iterations possible.

        Because the algorithm stops when the interval width is less than the tolerance,
        the maximum number of iterations is the number of times we can divide the max
        interval width by the golden ratio before it's smaller than tolerance.
        """
        max_interval_width = (self.b - self.a).max().item()
        return int(np.ceil(np.log(self.tolerance / max_interval_width) / np.log(self.invphi)))

    def update(self, fc: torch.Tensor, fd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update search intervals based on function values at c and d.

        Parameters
        ----------
        fc : torch.Tensor
            Function values at points c. Shape: (n_optimizations,)
        fd : torch.Tensor
            Function values at points d. Shape: (n_optimizations,)

        Returns
        -------
        new_c : torch.Tensor
            New evaluation points c for next iteration.
        new_d : torch.Tensor
            New evaluation points d for next iteration.
        """
        self.fc = fc
        self.fd = fd

        # Determine which interval to keep based on function values
        # If fc < fd, minimum is in [a, d], so update b = d
        # If fd < fc, minimum is in [c, b], so update a = c
        update_b = fc < fd
        update_a = ~update_b

        # Update intervals (only for non-converged optimizations)
        active = ~self.converged
        self.b[active & update_b] = self.d[active & update_b]
        self.a[active & update_a] = self.c[active & update_a]

        # Update c and d
        self.c = self.b - (self.b - self.a) * self.invphi
        self.d = self.a + (self.b - self.a) * self.invphi

        # Check convergence
        interval_widths = self.b - self.a
        c_d_too_close = torch.abs(self.c - self.d) < self.tolerance * self.invphi**3
        self.converged = (interval_widths < self.tolerance) | c_d_too_close

        self.iteration += 1

        return self.c, self.d

    def get_best_points(self) -> torch.Tensor:
        """
        Get the best points (midpoints of current intervals) for converged optimizations.

        Returns
        -------
        torch.Tensor
            Best points. Shape: (n_optimizations,)
        """
        return (self.a + self.b) / 2

    def is_converged(self) -> bool:
        """Check if all optimizations have converged."""
        return torch.all(self.converged).item()

    def get_active_mask(self) -> torch.Tensor:
        """Get boolean mask of optimizations that are still active (not converged)."""
        return ~self.converged
