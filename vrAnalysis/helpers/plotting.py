from pathlib import Path
from contextlib import contextmanager
from copy import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
from typing import Literal


@contextmanager
def batch_plot_context():
    plt.ioff()
    try:
        yield
    finally:
        plt.ion()


def save_figure(fig: plt.Figure, path: Path, parents: bool = True, **kwargs) -> None:
    """
    Save a figure with high resolution in png and svg formats
    """
    # Add the .fig extension to the path so it will replace the right suffix.
    path = path.parent / (path.name + ".fig")
    if parents:
        path.parent.mkdir(parents=True, exist_ok=True)
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
    if counts.sum() > 0:
        counts = counts / counts.sum()
    else:
        counts = np.full_like(counts, np.nan, dtype=float)
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


def insert_nans_at_gaps(data: np.ndarray, idx: np.ndarray, keep_idx: bool = False):
    """
    Insert NaN values in array data at positions where idx has gaps.
    This prevents matplotlib from drawing lines across gaps.
    If keep_idx is True, the indices will be returned as is, otherwise
    they will be converted to range(len(data)).

    Parameters:
    -----------
    data : np.ndarray
        Array with len(data) == len(idx)
    idx : np.ndarray
        Array of consecutive indices with potential gaps
    keep_idx : bool, optional
        Whether to keep the indices as is, by default False.

    Returns:
    --------
    data_with_gaps : np.ndarray
        Array with NaN inserted at gap positions
    idx_with_gaps : np.ndarray
        Corresponding indices with NaN inserted at gap positions
    """
    if len(data) != len(idx):
        raise ValueError(f"Length mismatch: len(data)={len(data)}, len(idx)={len(idx)}")

    # Find where gaps occur (diff != 1)
    diffs = np.diff(idx)
    gap_mask = diffs != 1

    if not np.any(gap_mask):
        # No gaps, return as-is
        return data.copy(), idx.copy()

    # Build new arrays, inserting NaN between values where gaps occur
    data_list = []
    idx_list = []
    idx_value = idx if keep_idx else np.arange(len(data))

    for i in range(len(data)):
        data_list.append(data[i])
        idx_list.append(idx_value[i])

        # If there's a gap after this position, insert NaN
        if i < len(diffs) and gap_mask[i]:
            data_list.append(np.nan)
            idx_list.append(np.nan)

    return np.array(data_list), np.array(idx_list)


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


def color_violins(parts, facecolor=None, linecolor=None):
    """Helper to color parts manually."""
    if facecolor is not None:
        for pc in parts["bodies"]:
            pc.set_facecolor(facecolor)
    if linecolor is not None:
        for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            if partname in parts:
                lc = parts[partname]
                lc.set_edgecolor(linecolor)


def format_spines(
    ax,
    x_pos,
    y_pos,
    xbounds=None,
    ybounds=None,
    xticks=None,
    yticks=None,
    xlabels=None,
    ylabels=None,
    spine_linewidth=1,
    tick_length=4,
    tick_width=1,
    tick_fontsize=None,
    spines_visible: list[str] = ["bottom", "left"],
):
    """
    Format a matplotlib axis to have separated spines with data offset from axes.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to format
    x_pos : int or float, optional
        The fractional value of the y-axis to offset the x-axis
    y_pos : int or float, optional
        The fractional value of the x-axis to offset the y-axis
    xbounds : tuple, optional
        The x-axis bounds as (min, max)
    ybounds : tuple, optional
        The y-axis bounds as (min, max)
    xticks : list or array, optional
        Custom x-axis tick positions
    yticks : list or array, optional
        Custom y-axis tick positions
    xlabels : list or array, optional
        Custom x-axis tick labels
    ylabels : list or array, optional
        Custom y-axis tick labels
    spine_linewidth : int or float, optional
        Width of the axis spines
    tick_length : int or float, optional
        Length of the tick marks
    tick_width : int or float, optional
        Width of the tick marks
    tick_fontsize : int or float, optional
        Font size of the tick labels
    spines_visible : list[str], optional
        List of spines to keep visible

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The formatted axis
    """
    # Move bottom spine down and left spine left
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    x_scale = ax.get_xscale()
    y_scale = ax.get_yscale()

    # Compute x-axis position (for left spine)
    if x_scale == "log":
        # For log scale: position = min * (max/min)^fractional
        # This preserves the same relative positioning as linear scale
        x_pos_computed = x_lims[0] * (x_lims[1] / x_lims[0]) ** x_pos
    else:
        # For linear scale: position = fractional * range + min
        x_range = x_lims[1] - x_lims[0]
        x_pos_computed = x_pos * x_range + x_lims[0]

    # Compute y-axis position (for bottom spine)
    if y_scale == "log":
        # For log scale: position = min * (max/min)^fractional
        y_pos_computed = y_lims[0] * (y_lims[1] / y_lims[0]) ** y_pos
    else:
        # For linear scale: position = fractional * range + min
        y_range = y_lims[1] - y_lims[0]
        y_pos_computed = y_pos * y_range + y_lims[0]

    ax.spines["bottom"].set_position(("data", y_pos_computed))
    ax.spines["left"].set_position(("data", x_pos_computed))

    # Set axis limits if provided
    if xbounds is not None:
        ax.spines["bottom"].set_bounds(xbounds[0], xbounds[1])

    if ybounds is not None:
        ax.spines["left"].set_bounds(ybounds[0], ybounds[1])

    # Set custom ticks if provided
    if xticks is not None:
        ax.set_xticks(xticks)
    if xlabels is not None:
        ax.set_xticklabels(xlabels)

    if yticks is not None:
        ax.set_yticks(yticks)
    if ylabels is not None:
        ax.set_yticklabels(ylabels)

    # Adjust tick appearance
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=tick_length,
        width=tick_width,
        labelsize=tick_fontsize,
    )

    # Hide spines that are not in the spines_visible list
    for spine_name, spine in ax.spines.items():
        if spine_name not in spines_visible:
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_linewidth(spine_linewidth)

    return ax


def blinded_study_legend(
    ax: plt.Axes,
    xpos: float,
    ypos: float,
    pilot_colors: list[ColorType] | ColorType,
    blinded_colors: list[ColorType] | ColorType,
    blinded: bool = True,
    fontsize: float = 12,
    y_offset: float = 0.05,
    origin: Literal["upper_left", "upper_right", "lower_left", "lower_right"] = "upper_right",
):
    """Create a custom legend for blinded study plots with colored text.

    This function creates a special legend that displays "Pilot Mice" and "Blinded Study"
    with different colors for each character in "Blinded Study".

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to add the legend to
    xpos : float
        X-coordinate position for the legend text
    ypos : float
        Y-coordinate position for the legend text
    pilot_colors : list of ColorType
        List of two colors for "Pilot" and "Mice" text. First color is for "Pilot",
        second color is for "Mice"
    blinded_colors : list of ColorType
        List of colors to cycle through for each character in "Blinded Study"
    blinded : bool, optional
        Whether to use the blinded color scheme or unblinded color scheme. Default True.
        If False, will use Red for knockout and Black for control.
    fontsize : float, optional
        Font size for all text elements, by default 12
    y_offset : float, optional
        Vertical offset between text elements, by default 0.05
    origin : Literal["upper_left", "upper_right", "lower_left", "lower_right"], optional
        Whether to start from top left ("upper_left"), top right ("upper_right"),
        bottom left ("lower_left"), or bottom right ("lower_right"), by default "upper_right".
    """
    if blinded:
        first_word = "Pilot"
        full_text = "Blinded Study"
    else:
        first_word = "Knockout"
        full_text = "Control Mice"
        pilot_colors = ["purple", "purple"]
        blinded_colors = ["gray"]

    if origin.lower() == "upper_right":
        text = ax.text(xpos, ypos, "Mice", ha="right", va="top", fontsize=fontsize, color=pilot_colors[1])
        ax.annotate(first_word + " ", xycoords=text, xy=(0, 0), fontsize=fontsize, color=pilot_colors[0], va="bottom", ha="right")
        for i, cstr in enumerate(full_text[::-1]):
            ccolor = blinded_colors[i % len(blinded_colors)]
            if i == 0:
                text = ax.annotate(cstr, xycoords=text, xy=(1, -y_offset), fontsize=fontsize, color=ccolor, va="top", ha="right")
            else:
                text = ax.annotate(cstr, xycoords=text, xy=(0, 0), fontsize=fontsize, color=ccolor, va="bottom", ha="right")

    elif origin.lower() == "lower_right":
        initial_text = ax.text(xpos, ypos, full_text[-1], ha="right", va="bottom", fontsize=fontsize, color=blinded_colors[-1])
        text = copy(initial_text)
        for i, cstr in enumerate(full_text[:-1][::-1], start=1):
            ccolor = blinded_colors[i % len(blinded_colors)]
            text = ax.annotate(cstr, xycoords=text, xy=(0, 0), fontsize=fontsize, color=ccolor, va="bottom", ha="right")
        text = ax.annotate("Mice", xycoords=initial_text, xy=(1, 1 + y_offset), fontsize=fontsize, color=pilot_colors[1], va="bottom", ha="right")
        ax.annotate(first_word + " ", xycoords=text, xy=(0, 0), fontsize=fontsize, color=pilot_colors[0], va="bottom", ha="right")

    elif origin.lower() == "upper_left":
        text = ax.text(xpos, ypos, first_word, ha="left", va="top", fontsize=fontsize, color=pilot_colors[0])
        ax.annotate(" Mice", xycoords=text, xy=(1, 0), fontsize=fontsize, color=pilot_colors[1], va="bottom", ha="left")
        for i, cstr in enumerate(full_text):
            ccolor = blinded_colors[i % len(blinded_colors)]
            if i == 0:
                text = ax.annotate(cstr, xycoords=text, xy=(0, -y_offset), fontsize=fontsize, color=ccolor, va="top", ha="left")
            else:
                text = ax.annotate(cstr, xycoords=text, xy=(1, 0), fontsize=fontsize, color=ccolor, va="bottom", ha="left")

    elif origin.lower() == "lower_left":
        initial_text = ax.text(xpos, ypos, full_text[0], ha="left", va="bottom", fontsize=fontsize, color=blinded_colors[0])
        text = copy(initial_text)
        for i, cstr in enumerate(full_text[1:], start=1):
            ccolor = blinded_colors[i % len(blinded_colors)]
            text = ax.annotate(cstr, xycoords=text, xy=(1, 0), fontsize=fontsize, color=ccolor, va="bottom", ha="left")
        text = ax.annotate("Pilot", xycoords=initial_text, xy=(0, 1 + y_offset), fontsize=fontsize, color=pilot_colors[0], va="bottom", ha="left")
        ax.annotate(" Mice", xycoords=text, xy=(1, 0), fontsize=fontsize, color=pilot_colors[1], va="bottom", ha="left")

    else:
        raise ValueError(f"Invalid origin: {origin}, permitted values are 'upper_left', 'upper_right', 'lower_left', 'lower_right'")


def get_mouse_colors(
    mouse_names: list[str],
    blinded: bool = True,
    asdict: bool = False,
    mousedb=None,
) -> tuple[list] | tuple[dict]:
    """Generate consistent colors, line widths, and z-orders for mouse plotting.

    This function assigns colors to mice in a blinded study, with special handling for pilot mice.
    Pilot mice are assigned specific colors (black and dimgrey) while blinded mice get colors
    from a rainbow colormap. Pilot mice also get thicker lines and higher z-order for emphasis.

    Note: This could be slightly improved by using a fixed color for each mouse (based on mousedb <- tracked), but the
    choice of mouse names is pretty consistent so should almost always give the same results.

    Parameters
    ----------
    mouse_names : list of str
        List of mouse identifiers to generate colors for
    blinded : bool, optional
        Whether this is a blinded study, by default True. Non-blinded studies
        are not yet implemented
    asdict : bool, optional
        Whether to return colors, linewidths, and zorders as a dictionary, by default False.
    mousedb : MouseDB, optional
        The mousedb object to use for color mapping when blinded=False.

    Returns
    -------
    colors, linewidth, zorders, either as lists or dictionaries
        tuple[list[ColorType], list[float], list[int]]
            A tuple containing three lists:
            - colors: List of colors for each mouse
            - linewidth: List of line widths (2 for pilot mice, 1 for others)
            - zorder: List of z-orders (1 for pilot mice, 0 for others)
        tuple[dict[str, ColorType], dict[str, float], dict[str, float]]
            A tuple containing three dictionaries:
            - colors: Dictionary of colors for each mouse
            - linewidth: Dictionary of line widths (2 for pilot mice, 1 for others)
            - zorder: Dictionary of z-orders (1 for pilot mice, 0 for others)

    Notes
    -----
    Pilot mice (CR_Hippocannula6 and CR_Hippocannula7) are assigned fixed colors,
    while other mice get colors from a rainbow colormap distributed evenly across
    the spectrum.

    Raises
    ------
    NotImplementedError
        If blinded=False, as the unblinding has not been performed!!!!
    """
    if blinded:
        pilot_mice = dict(
            CR_Hippocannula6="black",
            CR_Hippocannula7="dimgrey",
        )
        blinded_mice = [mouse for mouse in mouse_names if mouse not in pilot_mice]
        num_blinded = len(blinded_mice)
        cmap = mpl.colormaps["rainbow"]
        color_options = cmap(np.linspace(0, 1, num_blinded))
        colors_blinded = {mouse: color_options[imouse] for imouse, mouse in enumerate(blinded_mice)}
        colors = {**pilot_mice, **colors_blinded}
        if "ATL076" in mouse_names:
            colors["ATL076"] = "rosybrown"
        if asdict:
            linewidth = {mouse: 2 if mouse in pilot_mice else 1 for mouse in mouse_names}
            zorder = {mouse: 1 if mouse in pilot_mice else 0 for mouse in mouse_names}
        else:
            colors = [colors[mouse] for mouse in mouse_names]
            linewidth = [2 if mouse in pilot_mice else 1 for mouse in mouse_names]
            zorder = [1 if mouse in pilot_mice else 0 for mouse in mouse_names]
        return colors, linewidth, zorder

    else:
        if mousedb is None:
            raise ValueError("mousedb must be provided when blinded=False")

        # Get the KO and blinded status for each mouse
        ko = dict(zip(mousedb.get_table()["mouseName"], mousedb.get_table()["KO"]))
        blinded_mice = dict(zip(mousedb.get_table()["mouseName"], mousedb.get_table()["Blinded"]))

        # For each mouse, if it is not blinded, overwrite the color with the desired color based on genotype
        colors = []
        linewidth = []
        zorder = []
        for mouse in mouse_names:
            # Set the color, linewidth, and zorder for the mouse
            if not blinded_mice[mouse]:
                _color = "purple" if ko[mouse] else "gray"
                _linewidth = 1
                _zorder = 2 if ko[mouse] else 1
            else:
                _color = "red"
                _linewidth = 1
                _zorder = 1

            colors.append(_color)
            linewidth.append(_linewidth)
            zorder.append(_zorder)

        # Convert to dictionary if requested
        if asdict:
            colors = dict(zip(mouse_names, colors))
            linewidth = dict(zip(mouse_names, linewidth))
            zorder = dict(zip(mouse_names, zorder))

        return colors, linewidth, zorder
