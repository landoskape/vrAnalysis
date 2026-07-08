from dataclasses import dataclass, field
from typing import Callable, Union, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import conv1d
from scipy.optimize import curve_fit
from tqdm import tqdm


def fivePointDer(signal, h, axis=-1, returnIndex=False):
    assert isinstance(signal, np.ndarray), "signal must be a numpy array"
    assert -1 <= axis <= signal.ndim, "requested axis does not exist"
    N = signal.shape[axis]
    assert N >= 4 * h + 1, "h is too large for the given array -- it needs to be less than (N-1)/4!"
    signal = np.moveaxis(signal, axis, 0)
    n2 = slice(0, N - 4 * h)
    n1 = slice(h, N - 3 * h)
    p1 = slice(3 * h, N - h)
    p2 = slice(4 * h, N)
    fpd = (1 / (12 * h)) * (-signal[p2] + 8 * signal[p1] - 8 * signal[n1] + signal[n2])
    fpd = np.moveaxis(fpd, 0, axis)
    if returnIndex:
        return fpd, slice(2 * h, N - 2 * h)
    return fpd


@torch.no_grad()
def vector_correlation(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    covariance: bool = False,
    dim: int = -1,
    ignore_nan: bool = False,
    center: bool = True,
) -> torch.Tensor:
    """
    Measure the correlation of every element in x with every element in y on dim=dim.

    If covariance=True, will measure the covariance instead of correlation.
    If ignore_nan=True, will ignore NaN values in the correlation calculation.
    If center=False, uses uncentered second moments along ``dim``.

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
    center : bool
        If True, subtract the mean along ``dim`` before measuring covariance or
        correlation. If False, use uncentered moments. Default is True.

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
    denominator = n - int(center)
    if center:
        x_dev = x - mean_func(x, dim=dim, keepdim=True)
        y_dev = y - mean_func(y, dim=dim, keepdim=True)
    else:
        x_dev = x
        y_dev = y

    if not covariance:
        x_sample_std = torch.sqrt(sum_func(x_dev**2, dim=dim, keepdim=True) / denominator)
        y_sample_std = torch.sqrt(sum_func(y_dev**2, dim=dim, keepdim=True) / denominator)
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
    std = sum_func(x_dev * y_dev, dim=dim) / denominator

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
    full_stimulus_positions: Optional[Union[torch.Tensor, np.ndarray]] = None,
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
    full_stimulus_positions : Optional[Union[torch.Tensor, np.ndarray]]
        Complete regular stimulus grid used to normalize smoothing when
        ``stimulus_positions`` is a subset with missing bins. Missing bins are
        treated as zero-valued samples.
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
    if stimulus_positions.ndim != 1 or stimulus_positions.shape[0] != num_stimuli:
        raise ValueError(
            "stimulus_positions must be one-dimensional with length matching "
            f"the smoothing axis ({num_stimuli}), got shape {stimulus_positions.shape}."
        )
    if full_stimulus_positions is not None:
        if isinstance(full_stimulus_positions, torch.Tensor):
            full_stimulus_positions = full_stimulus_positions.detach().cpu().numpy()
        else:
            full_stimulus_positions = np.asarray(full_stimulus_positions, dtype=np.float32)
        if full_stimulus_positions.ndim != 1:
            raise ValueError("full_stimulus_positions must be one-dimensional.")

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

    position_steps = np.diff(stimulus_positions)
    positions_are_uniform = num_stimuli < 3 or np.allclose(position_steps, position_steps[0])
    if full_stimulus_positions is not None or not positions_are_uniform:
        if full_stimulus_positions is None:
            positive_steps = position_steps[position_steps > 0]
            if len(positive_steps) == 0:
                raise ValueError("stimulus_positions must span a positive range.")
            bin_width = np.min(positive_steps)
            full_stimulus_positions = np.arange(
                stimulus_positions[0],
                stimulus_positions[-1] + 0.5 * bin_width,
                bin_width,
                dtype=np.float32,
            )

        full_steps = np.diff(full_stimulus_positions)
        full_positions_are_uniform = len(full_stimulus_positions) < 3 or np.allclose(full_steps, full_steps[0])
        if not full_positions_are_uniform:
            raise ValueError("full_stimulus_positions must be a regular grid.")

        bin_width = 1.0 if len(full_stimulus_positions) < 2 else full_stimulus_positions[1] - full_stimulus_positions[0]
        dense_idx = np.rint((stimulus_positions - full_stimulus_positions[0]) / bin_width).astype(int)
        if (
            np.any(dense_idx < 0)
            or np.any(dense_idx >= len(full_stimulus_positions))
            or not np.allclose(full_stimulus_positions[dense_idx], stimulus_positions)
        ):
            raise ValueError("stimulus_positions must be a subset of full_stimulus_positions.")

        dense_flat = torch.zeros(
            (num_samples, len(full_stimulus_positions)),
            device=device,
            dtype=dtype,
        )
        dense_flat[:, dense_idx] = data_flat.to(dtype=dtype)
        smoothed_dense = gaussian_filter(
            dense_flat,
            smoothing_widths_t,
            stimulus_positions=full_stimulus_positions,
            axis=1,
        )
        smoothed_flat = smoothed_dense[:, dense_idx]
        smoothed_reshaped = smoothed_flat.reshape(*original_shape)
        return torch.moveaxis(smoothed_reshaped, -1, axis)

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


def fit_powerlaw_derivatives(eigenspectrum: torch.Tensor, width: int = 1, axis: int = 0, eps: float = 1e-8) -> float:
    """
    Fit powerlaw decay as a smoothed derivative of the eigenspectrum using a five-point stencil.

    Parameters
    ----------
    eigenspectrum : torch.Tensor
        The eigenspectrum to fit. Shape: (..., num_dimensions, ...)
    width : int, default=1
        The width of the smoothing window for the derivative.
    axis : int, default=0
        The axis along which to compute the derivative.
    eps : float, default=1e-8
        The epsilon value to add to the eigenspectrum to avoid log(0).

    Returns
    -------
    torch.Tensor
        The smoothed derivative of the eigenspectrum. Shape: (..., num_dimensions, ...)
    slice
        The slice of the eigenspectrum used for the fit.
    """
    lam = np.asarray(eigenspectrum)

    # mask invalid
    pos = lam > 0
    loglam = np.full_like(lam, np.nan, dtype=float)
    loglam[pos] = np.log(lam[pos] + eps)

    dloglam_dk, idx_slice = fivePointDer(loglam, width, axis=axis, returnIndex=True)

    # Build k (1-based) aligned to the derivative output slice along `axis`
    # Suppose axis=0 and idx_slice is a slice selecting the "center" k’s.
    k_full = np.arange(lam.shape[axis]) + 1  # 1..N
    k = k_full[idx_slice]  # centers used by derivative

    # reshape k to broadcast along other dims
    shape = [1] * lam.ndim
    shape[axis] = -1
    k = k.reshape(shape)

    alpha_local = -k * dloglam_dk  # ≈ alpha if power law holds
    return alpha_local, idx_slice


def fit_powerlaw_decay(
    eigenspectrum: torch.Tensor,
    start_idx: int = 0,
    end_idx: int = None,
    ignore_nans: bool = False,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Fit a powerlaw decay to the eigenspectrum. Returns the alpha value for n^(-alpha) decay.

    Parameters
    ----------
    eigenspectrum : torch.Tensor
        The eigenspectrum to fit. Shape: (num_dimensions,)
    start_idx : int, default=0
        The index of the first dimension to fit.
    end_idx : int, default=None
        The index of the last dimension to fit. If None, will fit until the end of the eigenspectrum.
    ignore_nans : bool, default=False
        If True, will ignore NaN values in the fit. If False, will raise an error if there are NaNs in the selected range.
    verbose : bool, default=True
        If True, will print warnings about ignored values. Set to False to suppress warnings.

    Returns
    -------
    float
        The alpha value for the powerlaw decay.
    float
        The amplitude value for the powerlaw decay.
    """
    if end_idx is None:
        end_idx = eigenspectrum.shape[0]

    if start_idx < 0 or start_idx >= eigenspectrum.shape[0]:
        raise ValueError(f"start_idx must be between 0 and {eigenspectrum.shape[0]}")
    if end_idx < 0 or end_idx > eigenspectrum.shape[0]:
        raise ValueError(f"end_idx must be between 0 and {eigenspectrum.shape[0]}")
    if start_idx >= end_idx:
        raise ValueError(f"start_idx must be less than end_idx")

    eigenspectrum = np.array(eigenspectrum[start_idx:end_idx])

    idx_use = eigenspectrum > 0
    if not np.all(idx_use) and verbose:
        print(f"Warning: some eigenspectrum values are negative or zero! Ignoring them.")

    idx_not_nan = ~np.isnan(eigenspectrum)
    if not np.all(idx_not_nan):
        if ignore_nans:
            if verbose:
                print(f"Warning: some eigenspectrum values are NaN! Ignoring them.")
            idx_use = idx_use & idx_not_nan
        else:
            raise ValueError("Eigenspectrum contains NaN values in the selected range. Set ignore_nans=True to ignore them.")

    if not np.any(idx_use) and ignore_nans:
        if verbose:
            print("Warning: no valid eigenspectrum values to fit after ignoring NaNs and non-positive values! Returning NaN.")
        return np.nan, np.nan

    x = np.log(np.arange(start_idx, end_idx)[idx_use] + 1)
    y = np.log(eigenspectrum[idx_use])  # log(n^(-1)) = -log(n)

    def _powerlaw(x, alpha, amplitude):
        # lambda = A * n^(-alpha)
        # in log space
        # log(lambda) = log(A) - alpha * log(n)
        return np.log(amplitude) - alpha * x

    popt, _ = curve_fit(_powerlaw, x, y)
    return popt[0], popt[1]


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
        a: Union[torch.Tensor, float],
        b: Union[torch.Tensor, float],
        tolerance: float = 1e-5,
        max_iterations: int = 100,
    ):
        """
        Initialize golden section search for multiple independent optimizations.

        Parameters
        ----------
        a : Union[torch.Tensor, float]
            Lower bounds for search intervals. Shape: (n_optimizations,) or scalar.
        b : Union[torch.Tensor, float]
            Upper bounds for search intervals. Shape: (n_optimizations,) or scalar.
        tolerance : float, default=1e-5
            Convergence tolerance. Search stops when interval width < tolerance.
        max_iterations : int, default=100
            Maximum number of iterations before stopping.
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor([a], dtype=torch.float32)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor([b], dtype=torch.float32)
        a = torch.atleast_1d(a)
        b = torch.atleast_1d(b)

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

    def run(
        self,
        objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        verbose: bool = False,
        maximize: bool = False,
    ) -> torch.Tensor:
        """
        Run the golden section search loop.

        Parameters
        ----------
        objective : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            Function with signature ``(points, active_mask) -> values``.
            ``points`` has shape (n_optimizations,); ``active_mask`` is boolean of same shape.
            Must return shape (n_optimizations,). Inactive entries are ignored — the caller
            may skip them or return zeros; ``run`` merges via ``torch.where``.
        verbose : bool, default=False
            Show tqdm progress bar.
        maximize : bool, default=False
            If True, maximize the objective; if False, minimize.

        Returns
        -------
        torch.Tensor
            Best points (midpoints of final intervals). Shape: (n_optimizations,).
        """
        sign = -1.0 if maximize else 1.0
        active_mask = torch.ones(self.a.shape[0], dtype=torch.bool, device=self.a.device)

        fc = sign * objective(self.c, active_mask)
        fd = sign * objective(self.d, active_mask)

        pbar = tqdm(total=self.max_iterations, desc="Golden section search", disable=not verbose, leave=False)
        try:
            while not self.is_converged() and self.iteration < self.max_iterations:
                self.c, self.d = self.update(fc, fd)
                active_mask = self.get_active_mask()
                if torch.any(active_mask):
                    fc_new = sign * objective(self.c, active_mask)
                    fd_new = sign * objective(self.d, active_mask)
                    fc = torch.where(active_mask, fc_new, fc)
                    fd = torch.where(active_mask, fd_new, fd)
                pbar.update(1)
                pbar.set_postfix({"active": active_mask.sum().item()})
        finally:
            if self.is_converged():
                pbar.n = self.max_iterations
            pbar.close()

        return self.get_best_points()

    def get_active_mask(self) -> torch.Tensor:
        """Get boolean mask of optimizations that are still active (not converged)."""
        return ~self.converged


# ---------------------------------------------------------------------------
# Robust power-law (alpha) estimation from spectra
# ---------------------------------------------------------------------------


@dataclass
class PowerlawFit:
    """Result of :func:`fit_powerlaw_spectrum`.

    The reported ``alpha`` is the exponent of the *longest* sub-window whose log-log fit
    is consistent (``R^2 >= r2_min``), so the fit covers as many ranks as the data allow
    (including the head when it stays straight) without extending into a floor or cliff.
    ``alpha_std`` / ``alpha_lo`` / ``alpha_hi`` summarise the spread of all consistent
    windows.

    Attributes
    ----------
    alpha, alpha_std, alpha_lo, alpha_hi
        Chosen exponent, robust spread (1.4826 * MAD), and percentile band of the
        consistent-window ensemble.
    amplitude
        Amplitude ``A`` of the chosen ``[head_end, tail_end]`` fit.
    head_end, tail_end
        Start and end (exclusive) 0-based ranks of the chosen fit window.
    search_end
        Region boundary the windows were drawn from: the first rank where the spectrum
        goes non-positive, thins out, or falls off a cliff (a big drop-off, detected as a
        local-exponent spike). Everything at/after this is treated as a bad tail.
    best_r2
        Log-log R^2 of the chosen window.
    n_valid
        Number of positive points in ``[head_end, tail_end]``.
    ensemble_alphas, ensemble_windows, ensemble_r2
        Per-window exponents, ``(M, 2)`` start/end index pairs, and log-log R^2 for the
        consistent windows the estimate is drawn from.
    local_alpha
        Full-length per-rank local exponent (NaN-padded to align with ``spectrum``).
    valid_mask
        Boolean mask of finite, positive entries.
    spectrum, smoothed
        Raw spectrum (numpy float) and the smoothed copy used for region detection.
    """

    alpha: float
    alpha_std: float
    alpha_lo: float
    alpha_hi: float
    amplitude: float
    head_end: int
    tail_end: int
    search_end: int
    best_r2: float
    n_valid: int
    ensemble_alphas: np.ndarray
    ensemble_windows: np.ndarray
    ensemble_r2: np.ndarray
    local_alpha: np.ndarray
    valid_mask: np.ndarray
    spectrum: np.ndarray
    smoothed: np.ndarray

    def plot(self, axes=None):
        """Plot the fit (see :func:`plot_powerlaw_fit`)."""
        return plot_powerlaw_fit(self, axes=axes)


def _nan_powerlaw_fit(spectrum: np.ndarray, smoothed: np.ndarray, valid_mask: np.ndarray) -> PowerlawFit:
    """All-NaN result for degenerate spectra that cannot be fit."""
    return PowerlawFit(
        alpha=np.nan,
        alpha_std=np.nan,
        alpha_lo=np.nan,
        alpha_hi=np.nan,
        amplitude=np.nan,
        head_end=0,
        tail_end=int(spectrum.shape[0]),
        search_end=int(spectrum.shape[0]),
        best_r2=np.nan,
        n_valid=int(valid_mask.sum()),
        ensemble_alphas=np.empty(0, dtype=float),
        ensemble_windows=np.empty((0, 2), dtype=int),
        ensemble_r2=np.empty(0, dtype=float),
        local_alpha=np.full(spectrum.shape[0], np.nan, dtype=float),
        valid_mask=valid_mask,
        spectrum=spectrum,
        smoothed=smoothed,
    )


def _boxcar_smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Edge-normalized moving average over ``width`` points (1-D)."""
    w = max(1, int(round(width)))
    if w == 1:
        return values.astype(float, copy=True)
    ones = np.ones(w)
    counts = np.convolve(np.ones(values.size), ones, mode="same")
    return np.convolve(values, ones, mode="same") / counts


def _safe_powerlaw_decay(spectrum: np.ndarray, start: int, end: int) -> Tuple[float, float]:
    """`fit_powerlaw_decay` that returns ``(nan, nan)`` instead of raising.

    ``curve_fit`` can fail to converge (``maxfev``) on windows spanning a huge dynamic
    range (e.g. an oracle spectrum decaying into a ~1e-21 numerical-zero floor), and the
    underlying fit raises ``ValueError`` on empty / all-invalid windows. Both are treated
    as a failed window here.
    """
    try:
        alpha, amplitude = fit_powerlaw_decay(spectrum, start, end, ignore_nans=True, verbose=False)
    except (RuntimeError, ValueError, TypeError):
        return np.nan, np.nan
    return float(alpha), float(amplitude)


def _loglog_r2(spectrum: np.ndarray, start: int, end: int, alpha: float, amplitude: float) -> float:
    """Coefficient of determination of a power-law fit in log-log space over [start, end)."""
    ranks = np.arange(start, end) + 1
    values = spectrum[start:end]
    keep = np.isfinite(values) & (values > 0)
    if keep.sum() < 2 or not (np.isfinite(alpha) and np.isfinite(amplitude) and amplitude > 0):
        return np.nan
    x = np.log(ranks[keep])
    y = np.log(values[keep])
    pred = np.log(amplitude) - alpha * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def fit_powerlaw_spectrum(
    spectrum: Union[np.ndarray, torch.Tensor],
    *,
    smooth_width: Optional[int] = 5,
    deriv_width: int = 1,
    min_head: int = 2,
    min_window: int = 5,
    tail_positive_frac: float = 0.9,
    cliff_alpha: float = 10.0,
    r2_min: float = 0.9,
    n_windows: int = 24,
    percentiles: Tuple[float, float] = (16.0, 84.0),
    eps: float = 1e-8,
) -> PowerlawFit:
    """Estimate the power-law exponent ``alpha`` of a spectrum, robustly.

    Real spectra follow a power law only over part of their range: a curved head, then a
    straight power-law stretch, then a bad tail (noisy / cross-validated spectra dip
    negative; oracle spectra fall off a cliff into a numerical-zero floor). This routine
    (1) finds the region boundary ``search_end`` from three signals -- the smoothed
    noise-floor crossing, the trailing positive fraction, and a **cliff** (a big drop-off,
    seen as a spike in the per-rank local exponent) -- then (2) fits
    ``lambda_k = A * (k+1)**(-alpha)`` (:func:`fit_powerlaw_decay`) over a grid of
    sub-windows inside ``[min_head, search_end)`` and reports the exponent of the
    **longest window whose fit is consistent** (``R^2 >= r2_min``). Maximising the fitted
    rank range this way naturally includes the head when it stays straight and excludes
    it when it curves.

    Parameters
    ----------
    spectrum : np.ndarray or torch.Tensor
        1-D spectrum, sorted descending (rank 0 largest).
    smooth_width : int or None, default=5
        Number of points in the boxcar moving average used to build a de-noised copy for
        region detection only (the raw spectrum is what gets fit). ``None`` uses the raw spectrum.
    deriv_width : int, default=1
        Stencil half-width for the local-exponent derivative.
    min_head : int, default=2
        Never begin the fit before this rank.
    min_window : int, default=5
        Minimum number of valid points required in each fit window.
    tail_positive_frac : float, default=0.9
        The region ends where the trailing positive fraction drops below this.
    cliff_alpha : float, default=10.0
        A rank whose local exponent exceeds this is treated as a cliff (a big drop-off /
        numerical floor); the region ends there. Real power laws sit well below it.
    r2_min : float, default=0.9
        Minimum log-log R^2 for a window to count as a consistent power-law fit.
    n_windows : int, default=24
        Target number of grid sub-windows.
    percentiles : tuple of float, default=(16.0, 84.0)
        Percentiles of the consistent-window ensemble reported as ``alpha_lo`` / ``alpha_hi``.
    eps : float, default=1e-8
        Added inside logs to avoid ``log(0)``.

    Returns
    -------
    PowerlawFit
        Structured fit result. Degenerate inputs yield an all-NaN result rather
        than raising.
    """
    spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
    N = spectrum.shape[0]
    finite_mask = np.isfinite(spectrum)
    valid_mask = finite_mask & (spectrum > 0)

    # Smoothed copy for region detection (removes random tail negatives). A boxcar moving
    # average over `smooth_width` points is applied over the finite entries only (NaNs, e.g.
    # trailing padding from a ragged aggregate, are skipped rather than propagated); NaN
    # positions stay NaN.
    finite_idx = np.flatnonzero(finite_mask)
    if smooth_width is not None and finite_idx.size >= 1:
        smoothed = np.full(N, np.nan, dtype=float)
        smoothed[finite_idx] = _boxcar_smooth(spectrum[finite_idx], int(round(smooth_width)))
    else:
        smoothed = spectrum.copy()

    if N < min_window or valid_mask.sum() < min_window:
        return _nan_powerlaw_fit(spectrum, smoothed, valid_mask)

    min_head = int(np.clip(min_head, 0, N - 1))

    # --- Per-rank local exponent (NaN-padded to full length). Flat = power law; a spike ---
    # --- marks a cliff (big drop-off), the floor after it reads back near zero. ---
    local_alpha = np.full(N, np.nan, dtype=float)
    if N >= 4 * deriv_width + 1:
        alpha_local, idx_slice = fit_powerlaw_derivatives(smoothed, width=deriv_width, axis=0, eps=eps)
        local_alpha[idx_slice] = np.asarray(alpha_local, dtype=float)

    # --- Region end from three signals; take the earliest. ---
    # (a) smoothed noise-floor crossing (first smoothed value <= 0)
    nonpositive = ~(smoothed > 0)
    nonpositive[:min_head] = False
    smoothed_cross = int(np.argmax(nonpositive)) if np.any(nonpositive) else N

    # (b) trailing positive fraction thinning out
    frac_window = max(min_window, int(round(2 * (smooth_width or 1))) + 1)
    trailing_frac = np.convolve(valid_mask.astype(float), np.ones(frac_window) / frac_window, mode="same")
    below = trailing_frac < tail_positive_frac
    below[: max(min_head, frac_window)] = False
    frac_cross = int(np.argmax(below)) if np.any(below) else N

    # (c) cliff: first rank whose local exponent spikes above cliff_alpha
    cliff = np.isfinite(local_alpha) & (local_alpha > cliff_alpha)
    cliff[:min_head] = False
    cliff_cross = int(np.argmax(cliff)) if np.any(cliff) else N

    search_end = int(min(smoothed_cross, frac_cross, cliff_cross, N))
    search_end = min(max(search_end, min_head + min_window), N)

    # --- Grid of window fits inside [min_head, search_end). ---
    n_side = max(2, int(np.ceil(np.sqrt(n_windows))))
    starts = np.unique(np.linspace(min_head, max(min_head, search_end - min_window), n_side).astype(int))
    ends = np.unique(np.linspace(min(min_head + min_window, search_end), search_end, n_side).astype(int))

    windows: list[tuple[int, int]] = []
    alphas: list[float] = []
    amps: list[float] = []
    r2s: list[float] = []
    for s in starts:
        for e in ends:
            if e - s < min_window or int(valid_mask[s:e].sum()) < min_window:
                continue
            alpha_w, amp_w = _safe_powerlaw_decay(spectrum, int(s), int(e))
            if not np.isfinite(alpha_w):
                continue
            windows.append((int(s), int(e)))
            alphas.append(float(alpha_w))
            amps.append(float(amp_w))
            r2s.append(_loglog_r2(spectrum, int(s), int(e), float(alpha_w), float(amp_w)))

    if len(windows) == 0:
        result = _nan_powerlaw_fit(spectrum, smoothed, valid_mask)
        result.search_end = search_end
        result.local_alpha = local_alpha
        return result

    windows_arr = np.asarray(windows, dtype=int)
    alphas_arr = np.asarray(alphas, dtype=float)
    r2s_arr = np.asarray(r2s, dtype=float)

    # --- Choose the longest consistent window: R^2 >= r2_min, then max rank span. ---
    consistent = r2s_arr >= r2_min
    pool = np.flatnonzero(consistent) if np.any(consistent) else np.flatnonzero(np.isfinite(r2s_arr))
    if pool.size == 0:
        pool = np.arange(alphas_arr.size)
    spans = windows_arr[:, 1] - windows_arr[:, 0]
    # sort key: longest span, then reaching the deepest rank, then starting earliest
    best = pool[np.lexsort((windows_arr[pool, 0], -windows_arr[pool, 1], -spans[pool]))[0]]
    best_start, best_end = int(windows_arr[best, 0]), int(windows_arr[best, 1])

    ensemble_alphas = alphas_arr[pool]
    alpha = float(alphas_arr[best])
    mad = float(np.median(np.abs(ensemble_alphas - np.median(ensemble_alphas)))) if ensemble_alphas.size else 0.0
    alpha_std = 1.4826 * mad
    if ensemble_alphas.size:
        alpha_lo, alpha_hi = (float(v) for v in np.percentile(ensemble_alphas, percentiles))
    else:
        alpha_lo = alpha_hi = alpha

    return PowerlawFit(
        alpha=alpha,
        alpha_std=alpha_std,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
        amplitude=float(amps[best]),
        head_end=best_start,
        tail_end=best_end,
        search_end=search_end,
        best_r2=float(r2s_arr[best]),
        n_valid=int(valid_mask[best_start:best_end].sum()),
        ensemble_alphas=ensemble_alphas,
        ensemble_windows=windows_arr[pool],
        ensemble_r2=r2s_arr[pool],
        local_alpha=local_alpha,
        valid_mask=valid_mask,
        spectrum=spectrum,
        smoothed=smoothed,
    )


def plot_powerlaw_fit(fit: PowerlawFit, axes=None):
    """Three-panel diagnostic for a :class:`PowerlawFit`.

    Panel A: log-log spectrum (positive points filled, non-positive hollow at ``|value|``),
    the smoothed curve, the shaded chosen ``[head_end, tail_end]`` fit window, the region
    boundary ``search_end`` and the fitted line.
    Panel B: the per-rank local exponent with the chosen exponent, the fit window and the
    ``search_end`` boundary.
    Panel C: the consistent-window exponents with the chosen value and percentile band.

    Parameters
    ----------
    fit : PowerlawFit
        Result of :func:`fit_powerlaw_spectrum`.
    axes : sequence of matplotlib.axes.Axes, optional
        Three axes to draw into. If ``None``, a ``(1, 3)`` figure is created.

    Returns
    -------
    (fig, axes)
        The figure (``None`` if ``axes`` was supplied) and the three axes.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    else:
        fig = None
    ax_spec, ax_local, ax_hist = axes

    spectrum = fit.spectrum
    N = spectrum.shape[0]
    ranks = np.arange(N) + 1
    pos = fit.valid_mask

    # --- Panel A: spectrum + fit ---
    ax_spec.loglog(ranks[pos], spectrum[pos], "o", ms=3, color="0.3", label="spectrum")
    nonpos = np.isfinite(spectrum) & ~pos
    if np.any(nonpos):
        ax_spec.loglog(ranks[nonpos], np.abs(spectrum[nonpos]), "o", ms=3, mfc="none", mec="0.7", label="|value<=0|")
    smooth_pos = fit.smoothed > 0
    ax_spec.loglog(ranks[smooth_pos], fit.smoothed[smooth_pos], "-", color="tab:blue", lw=1, alpha=0.7, label="smoothed")
    ax_spec.axvspan(fit.head_end + 1, fit.tail_end, color="tab:orange", alpha=0.15, label="fit window")
    ax_spec.axvline(fit.search_end, color="tab:green", ls=":", lw=1, label=f"search_end={fit.search_end}")
    if np.isfinite(fit.alpha) and np.isfinite(fit.amplitude):
        fit_ranks = np.arange(fit.head_end, fit.tail_end) + 1
        ax_spec.loglog(fit_ranks, fit.amplitude * fit_ranks.astype(float) ** (-fit.alpha), "r--", lw=1.5, label="fit")
    ax_spec.set_xlabel("rank")
    ax_spec.set_ylabel("value")
    ax_spec.set_title(f"alpha = {fit.alpha:.2f} +/- {fit.alpha_std:.2f}  (R2={fit.best_r2:.3f})")
    ax_spec.legend(fontsize=8)

    # --- Panel B: local exponent ---
    ax_local.plot(ranks, fit.local_alpha, "-", color="0.4", lw=1)
    if np.isfinite(fit.alpha):
        ax_local.axhline(fit.alpha, color="r", lw=1, label="chosen alpha")
        ax_local.axhspan(fit.alpha_lo, fit.alpha_hi, color="r", alpha=0.12)
    ax_local.axvspan(fit.head_end + 1, fit.tail_end, color="tab:orange", alpha=0.15, label="fit window")
    ax_local.axvline(fit.search_end, color="tab:green", ls=":", label="search_end")
    ax_local.set_xscale("log")
    ax_local.set_xlabel("rank")
    ax_local.set_ylabel("local alpha")
    ax_local.set_title("local exponent")
    finite_local = fit.local_alpha[np.isfinite(fit.local_alpha)]
    if finite_local.size:
        ax_local.set_ylim(np.nanpercentile(finite_local, 1) - 0.5, np.nanpercentile(finite_local, 99) + 0.5)
    ax_local.legend(fontsize=8)

    # --- Panel C: consistent-window distribution ---
    if fit.ensemble_alphas.size:
        ax_hist.hist(fit.ensemble_alphas, bins=min(20, max(5, fit.ensemble_alphas.size)), color="0.7")
        ax_hist.axvline(fit.alpha, color="r", lw=1.5, label="chosen")
        ax_hist.axvspan(fit.alpha_lo, fit.alpha_hi, color="r", alpha=0.12, label="16-84%")
        ax_hist.legend(fontsize=8)
    ax_hist.set_xlabel("windowed alpha")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"consistent windows (n={fit.ensemble_alphas.size})")

    return fig, axes
