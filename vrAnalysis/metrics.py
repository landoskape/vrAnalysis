from typing import Union, List, Tuple, Optional
import itertools
import numpy as np
from scipy.stats import gaussian_kde
from skimage.measure import find_contours
import matplotlib.pyplot as plt


class KernelDensityEstimator:
    """Estimates 2D probability density functions using kernel density estimation.

    This class provides a wrapper around scipy's gaussian_kde specifically for 2D data,
    with additional functionality for grid generation and plotting.

    This class can be used in two ways:
    1. Basic usage with automatic ranges:
        kde = KernelDensityEstimator(x_data, y_data).fit()
        density = kde.pdf()

    2. With specified ranges and bin count:
        kde = KernelDensityEstimator(x_data, y_data,
                                    xrange=(-1, 1),
                                    yrange=(-2, 2),
                                    nbins=201).fit()
        density = kde.pdf()

    Parameters
    ----------
    x : np.ndarray
        1D numpy array containing x coordinates
    y : np.ndarray
        1D numpy array containing y coordinates
    xrange : Optional[Tuple[float, float]], default=None
        Tuple of (min, max) specifying the range for x axis.
        If None, range is automatically determined from the data.
    yrange : Optional[Tuple[float, float]], default=None
        Tuple of (min, max) specifying the range for y axis.
        If None, range is automatically determined from the data.
    nbins : int, default=101
        Number of bins to use for the evaluation grid in each dimension.

    Attributes
    ----------
    x : np.ndarray
        The input x coordinates
    y : np.ndarray
        The input y coordinates
    nbins : int
        Number of bins in the evaluation grid
    xrange : Tuple[float, float]
        The range used for x axis evaluation
    yrange : Tuple[float, float]
        The range used for y axis evaluation
    kde : scipy.stats.gaussian_kde
        The underlying KDE estimator (available after calling fit())

    Raises
    ------
    ValueError
        If input arrays are not 1D numpy arrays
        If input arrays have different shapes
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xrange: Optional[Tuple[float, float]] = None,
        yrange: Optional[Tuple[float, float]] = None,
        nbins: int = 101,
    ):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be numpy arrays!")
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D arrays!")
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape!")

        self.x = x
        self.y = y
        self.nbins = nbins

        self.xrange = (np.min(x), np.max(x)) if xrange is None else xrange
        self.yrange = (np.min(y), np.max(y)) if yrange is None else yrange

    @property
    def grid(self) -> np.ndarray:
        """Get the evaluation grid points.

        Returns
        -------
        np.ndarray
            Array of shape (2, nbins^2) containing the coordinates
            of all grid points
        """
        if not hasattr(self, "_grid"):
            self._build_grid()
        return self._grid

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Get the shape of the evaluation grid.

        Returns
        -------
        Tuple[int, int]
            The shape of the grid as (nbins, nbins)
        """
        if not hasattr(self, "_grid_shape"):
            self._build_grid()
        return self._grid_shape

    def _build_grid(self) -> None:
        """Build the evaluation grid based on ranges and nbins."""
        x_grid = np.linspace(self.xrange[0], self.xrange[1], self.nbins)
        y_grid = np.linspace(self.yrange[0], self.yrange[1], self.nbins)
        mesh = np.meshgrid(x_grid, y_grid, indexing="ij")
        self._grid = np.vstack([m.ravel() for m in mesh])
        self._grid_shape = mesh[0].shape

    def fit(self) -> "KernelDensityEstimator":
        """Fit the kernel density estimator to the data.

        Returns
        -------
        KernelDensityEstimator
            The fitted estimator (self)
        """
        data = np.vstack([self.x, self.y])
        self.kde = gaussian_kde(data)
        return self

    @property
    def pdf(self):
        """Compute the probability density function on the evaluation grid.

        Returns
        -------
        np.ndarray
            The computed probability density values
        """
        if not hasattr(self, "_pdf"):
            self._pdf = np.reshape(self.kde(self.grid), self.grid_shape)
        return self._pdf

    @property
    def plot_data(self) -> np.ndarray:
        """Get density data in a format suitable for plotting.

        Returns
        -------
        np.ndarray
            The probability density values, rotated 90 degrees for correct
            orientation in common plotting functions
        """
        return np.rot90(self.pdf)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Get the plot extent in a format suitable for matplotlib.

        Returns
        -------
        Tuple[float, float, float, float]
            The extent as (xmin, xmax, ymin, ymax)
        """
        return self.xrange[0], self.xrange[1], self.yrange[0], self.yrange[1]

    def contours(self, levels: List[float]) -> List[np.ndarray]:
        """Find contours at specified levels using skimage.measure.find_contours.

        Parameters
        ----------
        levels : List[float]
            The levels at which to find contours.

        Returns
        -------
        List[np.ndarray]
            A list of contours, one for each level. The output of find_contours for any particular level
            is a list of arrays, each containing the (x, y) coordinates of the contour. The method returns
            as many groups of x, y coordinates as there are valid contours at that level (e.g. it's usually
            1 for unimodal distributions, can be 2 for bimodal, but will be more if a level is chosen that
            could be 1 but is cutoff by the edges of the distribution).

        Example Usage:
        --------------
        >>> kde = KernelDensityEstimator(x_data, y_data).fit()
        >>> contours = kde.contours([0.1, 0.5])
        >>> for contour in contours:
        ...     plt.plot(contour[:, 0], contour[:, 1], color='k', linewidth=2)
        """
        return [self._find_contours(level) for level in levels]

    def _find_contours(self, level):
        """Find contours at a given level using scipy.ndimage.find_contours and rescale
        to original data coordinates.

        See Also
        --------
        KernelDensityEstimator.contours : The public method that calls this with more documentation.
        plot_contours : A helper function to plot the contours.
        """
        # First find the contour at the given level (using the real percentile value)
        percentile = np.percentile(self.pdf, 100 * level)
        contours = find_contours(self.pdf, percentile)
        # Then map the contour back to the original data coordinates

        x_scale = (self.xrange[1] - self.xrange[0]) / (self.nbins - 1)
        y_scale = (self.yrange[1] - self.yrange[0]) / (self.nbins - 1)
        scaled_contours = [(c * [x_scale, y_scale] + [self.xrange[0], self.yrange[0]]) for c in contours]
        return scaled_contours


def plot_contours(contours, ax=None, **plot_kwargs):
    """Plot the contours on a given axis.

    Parameters
    ----------
    contours : list of numpy.ndarray
        The contours to plot.
    ax : matplotlib.axes.Axes, optional
        The axis to plot the contours on.
    plot_kwargs : dict, optional
        Keyword arguments to pass to the plot command (color, linewidth, etc.)

    See Also
    --------
    KernelDensityEstimator.contours : The method that finds the contours.
    """
    if ax is None:
        ax = plt.gca()
    for contour in contours:
        ax.plot(contour[:, 0], contour[:, 1], **plot_kwargs)


class FractionActive:
    """Computes the fraction of active elements across specified dimensions.


    This class can be used in two ways:
    1. As a callable object:
        fraction = FractionActive(activity_method='rms')(spks, 2, 1)

    2. As a reusable calculator:
        calculator = FractionActive(activity_method='rms')
        fraction1 = calculator(spks1, 2, 1)
        fraction2 = calculator(spks2, 2, 1)

    3. With the classmethod:
        fraction = FractionActive.compute(spks, 2, 1, activity_method='rms', fraction_method='participation')
    """

    activity_methods: List[str] = ["max", "mean", "rms", "any"]
    fraction_methods: List[str] = ["gini", "relative_entropy", "fraction", "participation"]

    def __init__(self, activity_method: str = "rms", fraction_method: str = "participation"):
        """Initialize FractionActive calculator with specified methods.

        Parameters
        ----------
        activity_method : str, default="rms"
            Method to compute activity across the activity axis. Options:
            - 'max': maximum value
            - 'mean': average value
            - 'rms': root mean square
            - 'any': presence of any non-zero value

        fraction_method : str, default="participation"
            Method to compute the fraction across the fraction axis. Options:
            - 'gini': 1 minus Gini coefficient (1 = perfect equality)
            - 'relative_entropy': normalized entropy (1 = uniform distribution)
            - 'fraction': simple fraction above zero
            - 'participation': participation ratio (1 = uniform participation)

        Raises
        ------
        ValueError
            If activity_method or fraction_method are not in their respective allowed values.
        """
        if activity_method not in self.activity_methods:
            raise ValueError(f"Invalid activity method: {activity_method}")
        if fraction_method not in self.fraction_methods:
            raise ValueError(f"Invalid fraction method: {fraction_method}")
        self.activity_method = activity_method
        self.fraction_method = fraction_method

    @classmethod
    def get_combinations(cls) -> List[Tuple[str, str]]:
        """Get all valid combinations of activity and fraction methods.

        Returns
        -------
        List[Tuple[str, str]]
            List of tuples containing (activity_method, fraction_method) pairs.
        """
        return list(itertools.product(cls.activity_methods, cls.fraction_methods))

    @classmethod
    def compute(
        cls,
        spkmap: np.ndarray,
        activity_axis: Union[int, tuple[int, ...]],
        fraction_axis: Union[int, tuple[int, ...]],
        activity_method: str = "rms",
        fraction_method: str = "participation",
    ) -> np.ndarray:
        """Convenience method for one-off calculations without creating an instance.

        This is a class method that provides a simpler interface for one-time calculations.
        Instead of creating an instance and calling it, you can use this method directly:
        >>> FractionActive.compute(spks, 2, 1)

        Parameters
        ----------
        spkmap : np.ndarray
            The spike map or activity data to analyze.
        activity_axis : Union[int, tuple[int, ...]]
            Axis or axes over which to compute the activity measure.
            For example, if your data is time x trials x neurons,
            and you want to compute activity over time, use axis=0.
        fraction_axis : Union[int, tuple[int, ...]]
            Axis or axes over which to compute the fraction measure
            after activity has been computed.
        activity_method : str, default="rms"
            Method to compute activity. Options are:
            - 'max': maximum value
            - 'mean': average value
            - 'rms': root mean square
            - 'any': presence of any non-zero value
        fraction_method : str, default="participation"
            Method to compute fraction. Options are:
            - 'gini': 1 minus Gini coefficient
            - 'relative_entropy': normalized entropy
            - 'fraction': simple fraction above zero
            - 'participation': participation ratio

        Returns
        -------
        np.ndarray
            The fraction of active elements. Interpretation depends on fraction_method.

        See Also
        --------
        FractionActive.__call__ : The instance method version of this computation
        """
        calculator = cls(activity_method=activity_method, fraction_method=fraction_method)
        return calculator(spkmap, activity_axis, fraction_axis)

    def fraction_active(
        self,
        spkmap: np.ndarray,
        activity_axis: Union[int, tuple[int, ...]],
        fraction_axis: Union[int, tuple[int, ...]],
    ) -> np.ndarray:
        """Calculate the fraction of active elements using specified methods.

        Parameters
        ----------
        spkmap : np.ndarray
            The spike map or activity data to analyze.
        activity_axis : Union[int, tuple[int, ...]]
            Axis or axes over which to compute the activity measure.
            For example, if your data is time x trials x neurons,
            and you want to compute activity over time, use axis=0.
        fraction_axis : Union[int, tuple[int, ...]]
            Axis or axes over which to compute the fraction measure
            after activity has been computed.

        Returns
        -------
        np.ndarray
            The fraction of active elements. The interpretation depends
            on the chosen fraction_method:
            - 'gini': 1 minus Gini coefficient (1 = perfect equality)
            - 'relative_entropy': normalized entropy (1 = uniform distribution)
            - 'fraction': simple fraction of elements above zero
            - 'participation': participation ratio (1 = uniform participation)
        """
        if self.activity_method == "max":
            # Use max activity on each trial
            activity = np.nanmax(spkmap, axis=activity_axis)

        elif self.activity_method == "mean":
            # Use mean activity on each trial
            activity = np.nanmean(spkmap, axis=activity_axis)

        elif self.activity_method == "rms":
            # Use rms activity on each trial
            activity = np.sqrt(np.nanmean(spkmap**2, axis=activity_axis))

        elif self.activity_method == "any":
            # Use any activity on each trial (better for significant transient data)
            activity = np.any(spkmap > 0, axis=activity_axis)

        else:
            raise ValueError(f"Invalid activity method: {self.activity_method}")

        if self.fraction_method == "gini":
            # Use gini coefficient of activity on each trial
            fraction_active = self.gini(activity, axis=fraction_axis)

        elif self.fraction_method == "relative_entropy":
            # Use relative entropy of activity on each trial
            fraction_active = self.entropy(activity, axis=fraction_axis, relative=True)

        elif self.fraction_method == "fraction":
            # Use fraction of trials with any activity
            if isinstance(fraction_axis, int):
                fraction_active = np.sum(activity > 0, axis=fraction_axis) / activity.shape[fraction_axis]
            else:
                fraction_active = np.sum(activity > 0, axis=fraction_axis) / np.prod([activity.shape[ax] for ax in fraction_axis])

        elif self.fraction_method == "participation":
            # Use participation ratio of activity on each trial
            fraction_active = self.participation_ratio(activity, axis=fraction_axis)

        else:
            raise ValueError(f"Invalid fraction method: {self.fraction_method}")

        return fraction_active.squeeze()

    __call__ = fraction_active

    def gini(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute the equality measure (1 - Gini coefficient).

        Parameters
        ----------
        x : np.ndarray
            Input array.
        axis : int, default=-1
            Axis along which to compute the Gini coefficient.

        Returns
        -------
        np.ndarray
            1 - Gini coefficient, measuring equality rather than inequality.
        """
        n = x.shape[axis]
        x = np.sort(x, axis=axis)  # Sort values
        weights = np.moveaxis((1 + np.arange(n))[(...,) + (None,) * axis], 0, axis)
        gini_coefficient = 2 * np.sum(weights * x, axis=axis) / n / (np.sum(x, axis=axis) + 1e-10) - (n + 1) / n
        return 1 - gini_coefficient

    def entropy(self, x: np.ndarray, axis: int = -1, relative: bool = True) -> np.ndarray:
        """Compute the entropy of the input array.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        axis : int, default=-1
            Axis along which to compute entropy.
        relative : bool, default=True
            If True, normalize the entropy by log(n) where n is the size of the axis.

        Returns
        -------
        np.ndarray
            Entropy values. If relative=True, values are between 0 and 1.
        """
        activity = x / (np.sum(x, axis=axis, keepdims=True) + 1e-10)
        # Handle zeros in activity by setting 0 * log(0) = 0
        log_activity = np.zeros_like(activity)
        nonzero = activity > 0
        log_activity[nonzero] = np.log(activity[nonzero])
        output = -np.sum(activity * log_activity, axis=axis)
        if relative:
            output = output / np.log(activity.shape[axis])
        return output

    def participation_ratio(self, x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
        """Compute the participation ratio of the input array.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        axis : int, default=-1
            Axis along which to compute the participation ratio.
        eps : float, default=1e-10
            Small value to prevent division by zero.

        Returns
        -------
        np.ndarray
            Participation ratio, normalized by the size of the axis.
            Values are between 0 and 1, where 1 indicates uniform participation.
        """
        relative_activity = x / (np.sum(x, axis=axis, keepdims=True) + eps)
        numerator = np.sum(relative_activity, axis=axis) ** 2
        denominator = np.sum(relative_activity**2, axis=axis)
        return numerator / (denominator + eps) / x.shape[axis]
