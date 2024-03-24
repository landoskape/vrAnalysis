import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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


def errorPlot(x, data, axis=-1, se=False, ax=None, **kwargs):
    """
    convenience method for making a plot with errorbars
    kwargs go into fill_between and plot, so they have to work for both...
    to make that more flexible, we could add a list of kwargs that work for
    one but not the other and pop them out as I did with the 'label'...
    """
    if ax is None:
        ax = plt.gca()
    meanData = np.mean(data, axis=axis)
    correction = data.shape[axis] if se else 1
    errorData = np.std(data, axis=axis) / correction
    fillBetweenArgs = kwargs.copy()
    fillBetweenArgs.pop("label")
    ax.fill_between(x, meanData + errorData, meanData - errorData, **fillBetweenArgs)
    kwargs.pop("alpha")
    ax.plot(x, meanData, **kwargs)


def discreteColormap(name="Spectral", N=10):
    return matplotlib.colormaps[name].resampled(N)


def ncmap(name="Spectral", vmin=10, vmax=None):
    if vmax is None:
        vmax = vmin - 1
        vmin = 0

    cmap = matplotlib.cm.get_cmap(name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    def getcolor(val):
        return cmap(norm(val))

    return getcolor


def edge2center(edges):
    assert isinstance(edges, np.ndarray) and edges.ndim == 1, "edges must be a 1-d numpy array"
    return edges[:-1] + np.diff(edges) / 2
