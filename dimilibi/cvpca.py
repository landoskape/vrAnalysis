from typing import Optional, Union
import numpy as np
import torch
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
        center: Optional[bool] = True,
        on_stimuli: Optional[bool] = False,
    ):
        """
        Initialize a CVPCA object with the option of specifying supporting parameters.

        Parameters
        ----------
        num_components : Optional[int]
            Number of components to use in the SVD decomposition.
            (default is the minimum number of neurons in the two groups)
        verbose : Optional[bool]
            If True, will print updates and results as they are computed.
            (default is False)
        center : Optional[bool]
            If True, center the data before PCA. Default is True.
        on_stimuli : Optional[bool]
            If True, will perform PCA on the stimulus dimension instead of the neuron dimension.
        """

        self.num_components = num_components
        self.verbose = verbose
        self.center = center
        self.on_stimuli = on_stimuli
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
        if self.on_stimuli:
            data_repeat1 = data_repeat1.T

        self.pca = PCA(num_components=self.num_components, verbose=self.verbose, center=self.center).fit(data_repeat1)
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

        if self.on_stimuli:
            data_repeat1 = data_repeat1.T
            data_repeat2 = data_repeat2.T

        self._validate_data(data_repeat1)
        self._validate_data(data_repeat2)

        repeat1_proj = self.pca.transform(data_repeat1)
        repeat2_proj = self.pca.transform(data_repeat2)

        return vector_correlation(repeat1_proj, repeat2_proj, covariance=True, dim=1, center=bool(self.center))

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
        center: Optional[bool] = True,
        on_stimuli: Optional[bool] = False,
        stimulus_positions: Optional[Union[torch.Tensor, np.ndarray]] = None,
        full_stimulus_positions: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        """
        Initialize a RegularizedCVPCA object with the option of specifying supporting parameters.

        Parameters
        ----------
        num_components : Optional[int]
            Number of components to use in the SVD decomposition.
            (default is the minimum number of neurons in the two groups)
        verbose : Optional[bool]
            If True, will print updates and results as they are computed.
            (default is True)
        center : Optional[bool]
            If True, center the data before PCA. Default is True.
        on_stimuli : Optional[bool]
            If True, will perform PCA on the stimulus dimension instead of the neuron dimension.
            (Will still fit smoothing for each neuron!)
        stimulus_positions : torch.Tensor or np.ndarray or None
            The positions of the stimuli. If None, will assume evenly spaced
            ``[0, 1, ..., num_stimuli - 1]``.
        full_stimulus_positions : torch.Tensor or np.ndarray or None
            Complete regular stimulus grid for smoothing normalization when
            ``stimulus_positions`` omits bins.
        """

        self.num_components = num_components
        self.verbose = verbose
        self.fitted = False
        self.smoothing_fitted = False
        self.smoothing_widths = None
        self.center = center
        self.on_stimuli = on_stimuli
        self.stimulus_positions = stimulus_positions
        self.full_stimulus_positions = full_stimulus_positions

    def _stimulus_axis_length(self, num_stimuli: int) -> float:
        """Return the physical length of the stimulus axis."""
        if self.full_stimulus_positions is not None and self.stimulus_positions is not None:
            stimulus_positions = torch.as_tensor(self.stimulus_positions, dtype=torch.float32).flatten()
            if stimulus_positions.numel() != num_stimuli:
                raise ValueError(
                    "stimulus_positions must have length matching the stimulus axis " f"({num_stimuli}), got {stimulus_positions.numel()}."
                )

        axis_positions = self.full_stimulus_positions if self.full_stimulus_positions is not None else self.stimulus_positions
        if axis_positions is None:
            return float(num_stimuli)

        stimulus_positions = torch.as_tensor(axis_positions, dtype=torch.float32).flatten()
        if self.full_stimulus_positions is None and stimulus_positions.numel() != num_stimuli:
            raise ValueError("stimulus_positions must have length matching the stimulus axis " f"({num_stimuli}), got {stimulus_positions.numel()}.")
        if num_stimuli == 1:
            return 1.0

        bin_width = torch.median(torch.abs(torch.diff(stimulus_positions))).item()
        axis_length = (stimulus_positions.max() - stimulus_positions.min()).item() + bin_width
        if axis_length <= 0:
            raise ValueError("stimulus_positions must span a positive range.")
        return axis_length

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

        Stimuli are assumed to be evenly spaced ``[0, 1, ..., num_stimuli - 1]``
        when ``stimulus_positions`` is not provided.

        Parameters
        ----------
        data_repeat1 : torch.Tensor
            The data to be used for training (num_neurons, num_stimuli).
        data_repeat2 : torch.Tensor
            The data to be used for testing (num_neurons, num_stimuli).
        data_repeat3 : torch.Tensor
            The data to be used for testing (num_neurons, num_stimuli).
        smoothing_range : tuple[float, float], default=(0.01, 0.5)
            (min, max) range for smoothing width search, relative to the stimulus
            axis length. For example, with num_stimuli=100, no stimulus positions,
            and smoothing_range=(0.01, 0.5), the search will be over widths
            [1.0, 50.0] in stimulus index units.
        tolerance : float, default=1e-3
            Convergence tolerance for golden section search.
        max_iterations : int, default=100
            Maximum iterations for golden section search.

        Returns
        -------
        self : object
            The RegularizedCVPCA object with optimized smoothing widths.
        """
        assert data_repeat1.shape == data_repeat2.shape == data_repeat3.shape, "All repeats must have the same shape"

        num_neurons, num_stimuli = data_repeat1.shape

        # Convert relative search bounds into the same units as stimulus_positions.
        stimulus_axis_length = self._stimulus_axis_length(num_stimuli)
        smoothing_min = smoothing_range[0] * stimulus_axis_length
        smoothing_max = smoothing_range[1] * stimulus_axis_length

        # Initialize smoothing widths for each neuron
        a = torch.full((num_neurons,), smoothing_min, device=data_repeat1.device)
        b = torch.full((num_neurons,), smoothing_max, device=data_repeat1.device)

        # Initialize golden section search
        gss = VectorizedGoldenSectionSearch(a, b, tolerance=tolerance, max_iterations=max_iterations)

        def objective(widths: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
            return self._evaluate_smoothing_mse(data_repeat1, data_repeat2, data_repeat3, widths, active_mask)

        self.smoothing_widths = gss.run(objective, verbose=self.verbose)
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
        smoothing_factor: float = 1.0,
    ):
        """
        Fit the RegularizedCVPCA model to the provided data. The provided data should be (neurons x stimuli)
        for the first repeat of stimuli, which is smoothed to accurately estimate the eigenvectors.

        If smoothing has been optimized via fit_smoothing(), it will be applied automatically unless
        disable_smoothing=True. Stimuli are assumed to be evenly spaced ``[0, 1, ..., num_stimuli - 1]``
        when ``stimulus_positions`` is not provided.

        Parameters
        ----------
        data_repeat1 : torch.Tensor
            The data to be used for training (num_neurons, num_stimuli).
        disable_smoothing : bool, default=False
            If True, no smoothing will be performed even if smoothing_widths are available.
        smoothing_factor: float, default=1.0
            Factor by which to multiply the smoothing widths.

        Returns
        -------
        self : object
            The RegularizedCVPCA object with the fitted model.
        """
        if not self.smoothing_fitted and not disable_smoothing:
            if self.verbose:
                print("Warning: Smoothing not fitted. Fitting without smoothing. " "Call fit_smoothing() first to optimize smoothing widths.")

        self.fitted = False

        # Apply smoothing if available and not disabled
        if self.smoothing_fitted and not disable_smoothing and self.smoothing_widths is not None:
            data_repeat1 = gaussian_filter(
                data_repeat1,
                self.smoothing_widths * smoothing_factor,
                axis=1,
                stimulus_positions=self.stimulus_positions,
                full_stimulus_positions=self.full_stimulus_positions,
            )

        if self.on_stimuli:
            data_repeat1 = data_repeat1.T

        self.pca = PCA(num_components=self.num_components, verbose=self.verbose, center=self.center).fit(data_repeat1)
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

        if self.on_stimuli:
            data_repeat2 = data_repeat2.T
            data_repeat3 = data_repeat3.T

        self._validate_data(data_repeat2)
        self._validate_data(data_repeat3)

        repeat2_proj = self.pca.transform(data_repeat2)
        repeat3_proj = self.pca.transform(data_repeat3)

        return vector_correlation(repeat2_proj, repeat3_proj, covariance=True, dim=1, center=bool(self.center))

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

        # Smooth active neurons with their own widths in stimulus-axis units.
        smoothed_r1_active = gaussian_filter(
            data_r1_active,
            smoothing_widths_active,
            axis=1,
            stimulus_positions=self.stimulus_positions,
            full_stimulus_positions=self.full_stimulus_positions,
        )

        # Compute MSE between smoothed r1 and r2 (per neuron, along stimulus dimension)
        mse_r1_r2_active = mse(smoothed_r1_active, data_r2_active, reduce=None, dim=1)  # (num_active,)

        # Compute MSE between smoothed r1 and r3 (per neuron, along stimulus dimension)
        mse_r1_r3_active = mse(smoothed_r1_active, data_r3_active, reduce=None, dim=1)  # (num_active,)

        # Store results only for active neurons
        result[active_indices] = mse_r1_r2_active + mse_r1_r3_active

        return result


class LegacyCVPCA:
    """
    Legacy Cross-Validated Principal Component Analysis

    This is from the Stringer et al 2019 nature paper, and uses shuffle methods
    inherited from some of their code (which was used in the _old_vrAnalysis library).

    This is here to compare because the methods produce different results, so this
    will test the data on like-to-like conditions to identify which parts create the difference.

    Fits PCA on repeat 1, then scores between repeat 1 and repeat 2.
    No smoothing is applied.
    """

    @torch.no_grad()
    def __init__(
        self,
        num_components: Optional[int] = None,
        shuffle_fraction: Optional[float] = 0.0,
        center: Optional[bool] = True,
        on_stimuli: Optional[bool] = False,
        fraction_nan_permitted: Optional[float] = 0.1,
        true_legacy: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """
        Initialize a CVPCA object with the option of specifying supporting parameters.

        Parameters
        ----------
        num_components : Optional[int]
            Number of components to use in the SVD decomposition.
            (default is the minimum number of neurons in the two groups)
        shuffle_fraction: Optional[float]
            The fraction of trials to shuffle across repeat1 & repeat2.
            (default is 0.0)
        center : Optional[bool]
            If True, center the data before PCA. Default is True.
        on_stimuli : Optional[bool]
            If True, performs cvCPA on stimulus dimension instead of neural dimension.
        true_legacy: Optional[bool]
            If True, will use the true legacy cvPCA method, which will shuffle the data across repeats.
            (default is False)
        verbose : Optional[bool]
            If True, will print updates and results as they are computed.
            (default is False)
        """

        self.num_components = num_components
        self.shuffle_fraction = shuffle_fraction
        self.center = center
        self.on_stimuli = on_stimuli
        self.verbose = verbose
        self.true_legacy = true_legacy

    @torch.no_grad()
    def fit_score(
        self,
        data_repeat1: torch.Tensor,
        data_repeat2: torch.Tensor,
    ):
        """
        Fit the CVPCA model to the provided data. The provided data should be (neurons x stimuli)
        for the first and second repeats of stimuli.

        Parameters
        ----------
        data_repeat1 : torch.Tensor
            The data to be used for training (num_neurons, num_stimuli).
        data_repeat2 : torch.Tensor
            The data to be used for scoring (num_neurons, num_stimuli).

        Returns
        -------
        covariance : torch.Tensor
            The covariance between the first and second repeats of data on each dimension of the fitted model.
        """
        num_stimuli = data_repeat1.shape[1]
        if data_repeat1.shape != data_repeat2.shape:
            raise ValueError("Data repeats must have the same shape.")

        if self.true_legacy:
            if self.on_stimuli:
                data_repeat1 = data_repeat1.T
                data_repeat2 = data_repeat2.T
            from _old_vrAnalysis import helpers as old_helpers

            covariance = old_helpers.shuff_cvPCA(data_repeat1.T.numpy(), data_repeat2.T.numpy(), nshuff=1, center=self.center)
            covariance = np.nanmean(covariance, axis=0)
            return covariance

        data_repeat1_flipped = data_repeat1.clone()
        data_repeat2_flipped = data_repeat2.clone()
        if self.shuffle_fraction > 0.0:
            idx_flip = torch.rand(num_stimuli) < self.shuffle_fraction
            data_repeat1_flipped[:, idx_flip] = data_repeat2[:, idx_flip]
            data_repeat2_flipped[:, idx_flip] = data_repeat1[:, idx_flip]

        if self.on_stimuli:
            data_repeat1_flipped = data_repeat1_flipped.T
            data_repeat2_flipped = data_repeat2_flipped.T

        self.pca = PCA(num_components=self.num_components, verbose=self.verbose, center=self.center).fit(data_repeat1_flipped)

        repeat1_proj = self.pca.transform(data_repeat1_flipped)
        repeat2_proj = self.pca.transform(data_repeat2_flipped)

        return vector_correlation(repeat1_proj, repeat2_proj, covariance=True, dim=1, center=bool(self.center))
