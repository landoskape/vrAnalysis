from typing import Optional, Union
import torch
import numpy as np
from tqdm import tqdm
from .pca import PCA
from .helpers import vector_correlation, gaussian_filter, VectorizedGoldenSectionSearch
from .metrics import mse


class CVPCA:
    """
    Original Cross-Validated Principal Component Analysis

    This is from the Stringer et al 2019 nature paper.

    Fits PCA on repeat 1, then scores between repeat 1 and repeat 2.
    No smoothing is applied.
    """

    @torch.no_grad()
    def __init__(
        self,
        num_components: Optional[int] = None,
        verbose: Optional[bool] = False,
        use_svd: Optional[bool] = False,
    ):
        """
        Initialize a CVPCA object with the option of specifying supporting parameters.

        Note: PCA is implemented with sklearn.decomposition.PCA, which learns and implements
        centering by default (without the option to turn it off). This means that the PCA model that
        is learned on the first repeat will be centered *on the first repeat*, and the scoring will
        use the centering from repeat 1. Therefore, you don't really need to center data before training,
        but should understand how this works for interpreting the model.

        Parameters
        ----------
        num_components : Optional[int]
            Number of components to use in the SVD decomposition.
            (default is the minimum number of neurons in the two groups)
        verbose : Optional[bool]
            If True, will print updates and results as they are computed.
            (default is False)
        use_svd: Optional[bool]
            If True, will use the torch SVD instead of the sklearn PCA decomposition.
            (default is False)
        """

        self.num_components = num_components
        self.verbose = verbose
        self.use_svd = use_svd
        self.fitted = False

    @torch.no_grad()
    def fit(
        self,
        data_repeat1: torch.Tensor,
    ):
        """
        Fit the CVPCA model to the provided data. The provided data should be (neurons x stimuli)
        for the first repeat of stimuli.

        Parameters
        ----------
        data_repeat1 : torch.Tensor
            The data to be used for training (num_neurons, num_stimuli).

        Returns
        -------
        self : object
            The CVPCA object with the fitted model.
        """
        self.fitted = False
        self._validate_data(data_repeat1)

        self.pca = PCA(num_components=self.num_components, verbose=self.verbose, use_svd=self.use_svd).fit(data_repeat1)
        self.fitted = True
        return self

    @torch.no_grad()
    def score(self, data_repeat1: torch.Tensor, data_repeat2: torch.Tensor) -> torch.Tensor:
        """
        Score the CVPCA model on the provided data by measuring the covariance
        of repeats 1 and 2 on each dimension of the fitted model.

        Parameters
        ----------
        data_repeat1 : torch.Tensor
            The first repeat of data to be used for scoring (num_neurons, num_stimuli).
        data_repeat2 : torch.Tensor
            The second repeat of data to be used for scoring (num_neurons, num_stimuli).

        Returns
        -------
        torch.Tensor
            The covariance between the first and second repeats of data on each dimension of the fitted model.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before scoring data.")

        self._validate_data(data_repeat1)
        self._validate_data(data_repeat2)

        repeat1_proj = self.pca.model.transform(data_repeat1.T)
        repeat2_proj = self.pca.model.transform(data_repeat2.T)

        return vector_correlation(repeat1_proj, repeat2_proj, covariance=True, dim=0)

    @torch.no_grad()
    def _validate_data(self, data: torch.Tensor):
        """Check if source and target data are valid"""
        if self.fitted and data.shape[0] != self.pca.get_components().shape[0]:
            raise ValueError("Data must have the same number of neurons as the training data.")


class RegularizedCVPCA:
    """
    Regularized Cross-Validated Principal Component Analysis

    This class implements the regularized cvPCA method introduced in this paper:
    https://www.biorxiv.org/content/10.1101/2025.11.12.688086v2.full.pdf
    """

    @torch.no_grad()
    def __init__(
        self,
        num_components: Optional[int] = None,
        verbose: Optional[bool] = False,
        use_svd: Optional[bool] = False,
    ):
        """
        Initialize a RegularizedCVPCA object with the option of specifying supporting parameters.

        Note: PCA is implemented with sklearn.decomposition.PCA, which learns and implements
        centering by default (without the option to turn it off). This means that the PCA model that
        is learned on the first repeat will be centered *on the first repeat*, and the scoring will
        use the centering from repeat 1. Therefore, you don't really need to center data before training,
        but should understand how this works for interpreting the model.

        Parameters
        ----------
        num_components : Optional[int]
            Number of components to use in the SVD decomposition.
            (default is the minimum number of neurons in the two groups)
        verbose : Optional[bool]
            If True, will print updates and results as they are computed.
            (default is True)
        use_svd: Optional[bool]
            If True, will use the torch SVD instead of the sklearn PCA decomposition.
            (default is False)
        """

        self.num_components = num_components
        self.verbose = verbose
        self.fitted = False
        self.smoothing_fitted = False
        self.smoothing_widths = None
        self.use_svd = use_svd

    @torch.no_grad()
    def fit_smoothing(
        self,
        data_repeat1: torch.Tensor,
        data_repeat2: torch.Tensor,
        data_repeat3: torch.Tensor,
        smoothing_range: tuple[float, float] = (0.01, 0.5),
        tolerance: float = 1e-3,
        max_iterations: int = 100,
    ):
        """
        Run optimization to find the best smoothing width for each neuron in the data.

        Will find the best smoothing width, using a gaussian kernel, for each neuron in the data.
        It will be evaluated by measuring the sum of MSE between smoothed repeat 1 and repeat 2,
        and MSE between smoothed repeat 1 and repeat 3. Uses a vectorized golden section search
        since the optimization is roughly convex and 1D.

        Stimuli are assumed to be evenly spaced [0, 1, ..., num_stimuli-1].

        Parameters
        ----------
        data_repeat1 : torch.Tensor
            The data to be used for training (num_neurons, num_stimuli).
        data_repeat2 : torch.Tensor
            The data to be used for testing (num_neurons, num_stimuli).
        data_repeat3 : torch.Tensor
            The data to be used for testing (num_neurons, num_stimuli).
        smoothing_range : tuple[float, float], default=(0.01, 0.5)
            (min, max) range for smoothing width search, relative to the number of stimuli.
            For example, with num_stimuli=100 and smoothing_range=(0.01, 0.5), the search
            will be over widths [1.0, 50.0] in stimulus index units.
        tolerance : float, default=1e-3
            Convergence tolerance for golden section search.
        max_iterations : int, default=100
            Maximum iterations for golden section search.

        Returns
        -------
        self : object
            The RegularizedCVPCA object with optimized smoothing widths.
        """
        self._validate_data(data_repeat1)
        self._validate_data(data_repeat2)
        self._validate_data(data_repeat3)

        assert data_repeat1.shape == data_repeat2.shape == data_repeat3.shape, "All repeats must have the same shape"

        num_neurons, num_stimuli = data_repeat1.shape

        # Scale smoothing_range by number of stimuli to convert from relative to absolute units
        # Stimuli are evenly spaced [0, 1, ..., num_stimuli-1], so range is num_stimuli
        smoothing_min = smoothing_range[0] * num_stimuli
        smoothing_max = smoothing_range[1] * num_stimuli

        # Initialize smoothing widths for each neuron
        a = torch.full((num_neurons,), smoothing_min, device=data_repeat1.device)
        b = torch.full((num_neurons,), smoothing_max, device=data_repeat1.device)

        # Initialize golden section search
        gss = VectorizedGoldenSectionSearch(a, b, tolerance=tolerance, max_iterations=max_iterations)

        # Evaluate initial points c and d
        c, d = gss.c, gss.d

        # Evaluate function at initial points (all neurons are active initially)
        active_mask = torch.ones(num_neurons, dtype=torch.bool, device=data_repeat1.device)
        fc = self._evaluate_smoothing_mse(data_repeat1, data_repeat2, data_repeat3, c, active_mask)
        fd = self._evaluate_smoothing_mse(data_repeat1, data_repeat2, data_repeat3, d, active_mask)

        # Main optimization loop with progress bar
        pbar = tqdm(total=max_iterations, desc="Optimizing smoothing widths", disable=not self.verbose, leave=False)
        try:
            while not gss.is_converged() and gss.iteration < max_iterations:
                # Update search intervals
                c, d = gss.update(fc, fd)

                # Evaluate at new points (only for active optimizations)
                active_mask = gss.get_active_mask()
                if torch.any(active_mask):
                    fc_active = self._evaluate_smoothing_mse(data_repeat1, data_repeat2, data_repeat3, c, active_mask)
                    fd_active = self._evaluate_smoothing_mse(data_repeat1, data_repeat2, data_repeat3, d, active_mask)
                    # Only update active elements
                    fc = torch.where(active_mask, fc_active, fc)
                    fd = torch.where(active_mask, fd_active, fd)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"active_neurons": active_mask.sum().item()})
        finally:
            # Jump to end if converged early
            if gss.is_converged():
                pbar.n = max_iterations
            pbar.close()

        # Get optimal smoothing widths
        self.smoothing_widths = gss.get_best_points()
        self.smoothing_fitted = True

        if self.verbose:
            print(
                f"Optimized smoothing widths: min={self.smoothing_widths.min():.4f}, "
                f"max={self.smoothing_widths.max():.4f}, mean={self.smoothing_widths.mean():.4f}"
            )

        return self

    @torch.no_grad()
    def fit(
        self,
        data_repeat1: torch.Tensor,
        disable_smoothing: bool = False,
    ):
        """
        Fit the RegularizedCVPCA model to the provided data. The provided data should be (neurons x stimuli)
        for the first repeat of stimuli, which is smoothed to accurately estimate the eigenvectors.

        If smoothing has been optimized via fit_smoothing(), it will be applied automatically unless
        disable_smoothing=True. Stimuli are assumed to be evenly spaced [0, 1, ..., num_stimuli-1].

        Parameters
        ----------
        data_repeat1 : torch.Tensor
            The data to be used for training (num_neurons, num_stimuli).
        disable_smoothing : bool, default=False
            If True, no smoothing will be performed even if smoothing_widths are available.

        Returns
        -------
        self : object
            The RegularizedCVPCA object with the fitted model.
        """
        if not self.smoothing_fitted and not disable_smoothing:
            if self.verbose:
                print("Warning: Smoothing not fitted. Fitting without smoothing. " "Call fit_smoothing() first to optimize smoothing widths.")

        self.fitted = False
        self._validate_data(data_repeat1)

        # Apply smoothing if available and not disabled
        if self.smoothing_fitted and not disable_smoothing and self.smoothing_widths is not None:
            data_repeat1 = gaussian_filter(data_repeat1, self.smoothing_widths, axis=1)

        self.pca = PCA(num_components=self.num_components, verbose=self.verbose, use_svd=self.use_svd).fit(data_repeat1)
        self.fitted = True
        return self

    @torch.no_grad()
    def score(self, data_repeat2: torch.Tensor, data_repeat3: torch.Tensor) -> torch.Tensor:
        """
        Score the RegularizedCVPCA model on the provided data by measuring the covariance
        of repeats 2 and 3 on each dimension of the fitted model.

        Parameters
        ----------
        data_repeat2 : torch.Tensor
            The second repeat of data to be used for testing (num_neurons, num_stimuli).
        data_repeat3 : torch.Tensor
            The third repeat of data to be used for testing (num_neurons, num_stimuli).

        Returns
        -------
        torch.Tensor
            The covariance between the second and third repeats of data on each dimension of the fitted model.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before scoring data.")

        self._validate_data(data_repeat2)
        self._validate_data(data_repeat3)

        repeat2_proj = self.pca.model.transform(data_repeat2.T)
        repeat3_proj = self.pca.model.transform(data_repeat3.T)

        return vector_correlation(repeat2_proj, repeat3_proj, covariance=True, dim=0)

    @torch.no_grad()
    def _validate_data(self, data: torch.Tensor):
        """Check if source and target data are valid"""
        if self.fitted and data.shape[0] != self.pca.get_components().shape[0]:
            raise ValueError("Data must have the same number of neurons as the training data.")

    @torch.no_grad()
    def _evaluate_smoothing_mse(
        self,
        data_repeat1: torch.Tensor,
        data_repeat2: torch.Tensor,
        data_repeat3: torch.Tensor,
        smoothing_widths: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate MSE between smoothed repeat 1 and repeats 2 & 3 for given smoothing widths.

        Only computes MSE for active neurons (where active_mask is True), avoiding unnecessary
        computation for converged neurons.

        For each smoothing width:
        1. Smooth repeat 1 (each neuron with its own width)
        2. Compute MSE between smoothed r1 and r2 (per neuron)
        3. Compute MSE between smoothed r1 and r3 (per neuron)
        4. Return sum of MSEs for each neuron

        Parameters
        ----------
        data_repeat1 : torch.Tensor
            Repeat 1 data (num_neurons, num_stimuli).
        data_repeat2 : torch.Tensor
            Repeat 2 data (num_neurons, num_stimuli).
        data_repeat3 : torch.Tensor
            Repeat 3 data (num_neurons, num_stimuli).
        smoothing_widths : torch.Tensor
            Smoothing widths to evaluate. Shape: (num_neurons,).
        active_mask : torch.Tensor
            Boolean mask indicating which neurons are still being optimized. Shape: (num_neurons,).

        Returns
        -------
        torch.Tensor
            Sum of MSE values (MSE(smooth(r1), r2) + MSE(smooth(r1), r3)) for each neuron.
            Shape: (num_neurons,). Values for inactive neurons are set to 0 (will be replaced
            by previous values in the caller).
        """
        # Only smooth and compute MSE for active neurons
        if not torch.any(active_mask):
            # All neurons converged, return zeros (will be replaced by previous values)
            return torch.zeros_like(smoothing_widths)

        # Initialize result tensor
        result = torch.zeros_like(smoothing_widths)

        # Only process active neurons
        active_indices = torch.where(active_mask)[0]
        data_r1_active = data_repeat1[active_indices]
        data_r2_active = data_repeat2[active_indices]
        data_r3_active = data_repeat3[active_indices]
        smoothing_widths_active = smoothing_widths[active_indices]

        # Smooth active neurons with their own widths (stimuli are evenly spaced [0, 1, ..., num_stimuli-1])
        smoothed_r1_active = gaussian_filter(data_r1_active, smoothing_widths_active, axis=1)

        # Compute MSE between smoothed r1 and r2 (per neuron, along stimulus dimension)
        mse_r1_r2_active = mse(smoothed_r1_active, data_r2_active, reduce=None, dim=1)  # (num_active,)

        # Compute MSE between smoothed r1 and r3 (per neuron, along stimulus dimension)
        mse_r1_r3_active = mse(smoothed_r1_active, data_r3_active, reduce=None, dim=1)  # (num_active,)

        # Store results only for active neurons
        result[active_indices] = mse_r1_r2_active + mse_r1_r3_active

        return result
