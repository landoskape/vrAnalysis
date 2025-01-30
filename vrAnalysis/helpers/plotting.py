from contextlib import contextmanager
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


@contextmanager
def batch_plot_context():
    plt.ioff()
    try:
        yield
    finally:
        plt.ion()


def save_figure(fig, path, **kwargs):
    """
    Save a figure with high resolution in png and svg formats
    """
    fig.savefig(path.with_suffix(".png"), dpi=300, **kwargs)
    fig.savefig(path.with_suffix(".svg"), **kwargs)


def short_mouse_names(mouse_names):
    short_names = []
    for name in mouse_names:
        if "_Hippocannula" in name:
            short_names.append("".join(name.split("_Hippocannula")))
        elif "ATL0" in name:
            short_names.append("".join(["ATL", *name.split("ATL0")]))
        else:
            short_names.append(name)
    return short_names


def fractional_histogram(*args, **kwargs):
    """wrapper of np.histogram() with relative counts instead of total or density"""
    counts, bins = np.histogram(*args, **kwargs)
    counts = counts / np.sum(counts)
    return counts, bins


# ---------------------------------- plotting helpers ----------------------------------
def scale(data, vmin=0, vmax=1, prctile=(0, 100)):
    """scale data to arbitrary range using conservative percentile estimate"""
    xmin = np.percentile(data, prctile[0])
    xmax = np.percentile(data, prctile[1])
    sdata = (data - xmin) / (xmax - xmin)
    sdata *= vmax - vmin
    sdata += vmin
    return sdata


def errorPlot(x, data, axis=-1, se=False, ax=None, handle_nans=True, **kwargs):
    """
    convenience method for making a plot with errorbars
    kwargs go into fill_between and plot, so they have to work for both...
    to make that more flexible, we could add a list of kwargs that work for
    one but not the other and pop them out as I did with the 'label'...
    """
    mean = np.nanmean if handle_nans else np.mean
    std = np.nanstd if handle_nans else np.std
    if handle_nans:
        num_valid_points = np.sum(~np.isnan(data), axis=axis)
    else:
        num_valid_points = data.shape[axis]
    if ax is None:
        ax = plt.gca()
    meanData = mean(data, axis=axis)
    correction = np.sqrt(num_valid_points) if se else 1
    errorData = std(data, axis=axis) / correction
    fillBetweenArgs = kwargs.copy()
    fillBetweenArgs.pop("label", None)
    ax.fill_between(x, meanData + errorData, meanData - errorData, **fillBetweenArgs)
    kwargs.pop("alpha", None)
    ax.plot(x, meanData, **kwargs)


def discreteColormap(name="Spectral", N=10):
    return mpl.colormaps[name].resampled(N)


def get_color_map(name, vmin=0, vmax=1):
    cmap = mpl.colormaps[name]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    get_color = lambda val: scalar_map.to_rgba(val)
    return get_color


def ncmap(name="Spectral", vmin=10, vmax=None):
    if vmax is None:
        vmax = vmin - 1
        vmin = 0

    cmap = mpl.cm.get_cmap(name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    def getcolor(val):
        return cmap(norm(val))

    return getcolor


def edge2center(edges):
    assert isinstance(edges, np.ndarray) and edges.ndim == 1, "edges must be a 1-d numpy array"
    return edges[:-1] + np.diff(edges) / 2


def beeswarm(y, nbins=None):
    """thanks to: https://python-graph-gallery.com/509-introduction-to-swarm-plot-in-matplotlib/"""
    # Convert y to a NumPy array
    y = np.asarray(y)

    # Calculate the number of bins if not provided
    if nbins is None:
        nbins = len(y) // 6

    # Get upper and lower bounds of the data
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)

    # Calculate the size of each bin based on the number of bins
    dy = (yhi - ylo) / nbins

    # Calculate the upper bounds of each bin using linspace
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide the indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins  # List to store indices for each bin
    ybs = [0] * nbins  # List to store values for each bin
    nmax = 0  # Variable to store the maximum number of data points in a bin
    for j, ybin in enumerate(ybins):

        # Create a boolean mask for elements that are less than or equal to the bin upper bound
        f = y <= ybin

        # Store the indices and values that belong to this bin
        ibs[j], ybs[j] = i[f], y[f]

        # Update nmax with the maximum number of elements in a bin so far
        nmax = max(nmax, len(ibs[j]))

        # Update i and y by excluding the elements already added to the current bin
        f = ~f
        i, y = i[f], y[f]

    # Add the remaining elements to the last bin
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices to the data points in each bin
    dx = 1 / (nmax // 2)

    for i, y in zip(ibs, ybs):
        if len(i) > 1:

            # Determine the index to start from based on whether the bin has an even or odd number of elements
            j = len(i) % 2

            # Sort the indices in the bin based on the corresponding values
            i = i[np.argsort(y)]

            # Separate the indices into two groups, 'a' and 'b'
            a = i[j::2]
            b = i[j + 1 :: 2]

            # Assign x values to the 'a' group using positive values and to the 'b' group using negative values
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


def clear_axis(ax):
    """use axis as empty space or something else without any of the default matplotlib stuff"""
    ax.clear()  # Clear all the artists (lines, patches, etc.)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.spines["top"].set_visible(False)  # Hide top spine
    ax.spines["right"].set_visible(False)  # Hide right spine
    ax.spines["bottom"].set_visible(False)  # Hide top spine
    ax.spines["left"].set_visible(False)  # Hide right spine


def refline(slope, intercept, ax=None, **kwargs):
    """Plot a line from slope and intercept"""
    if ax is None:
        ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, **kwargs)
