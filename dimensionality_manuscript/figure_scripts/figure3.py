import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from syd import make_viewer, Viewer
from tqdm import tqdm

from vrAnalysis.database import get_database
from vrAnalysis.helpers import Timer
from vrAnalysis.helpers.plotting import format_spines, beeswarm, errorPlot, save_figure, add_scaled_limits
from vrAnalysis.processors.placefields import get_placefield
from dimilibi import PCA, SVCA
from dimensionality_manuscript.registry import PopulationRegistry, get_subspace, SubspaceName, RegistryPaths
from dimensionality_manuscript.subspace_analysis.base import Subspace
from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldHyperparameters
from dimensionality_manuscript.simulations import sqrtm_spd
from dimensionality_manuscript import SubspaceConfig, StimSpaceConfig, StimSpaceSpectraConfig
from dimensionality_manuscript import ResultsAggregator, ResultsStore, get_data_config
from dimensionality_manuscript.scripts.status import status
from dimensionality_manuscript.subspace_analysis.stimspace import StimSpaceSubspace
from dimensionality_manuscript import average_by_mouse


class SubspaceCrossspaceViewer(Viewer):
    """Interactive cross-spectrum energy viewer over aggregated subspace results."""

    def __init__(self, results: ResultsAggregator, figsize: tuple[float, float] = (6.0, 3.0)):
        self.results = results
        self.figsize = figsize
        for key, value in results.param_axes.items():
            self.add_selection(key, options=value)

        preferred_state = {
            "smooth_width": None,
            "activity_parameters_name": "default",
        }
        for key, value in preferred_state.items():
            self.update_selection(key, value=value)

        self.add_integer("idx_cross", value=0, min=0, max=len(results.sessions) - 1)
        self.add_integer("num_cross_show", value=25, min=1, max=100)
        self.add_boolean("plot_energy", value=True)

    def plot(self, state: dict):
        _sel_state = {k: v for k, v in state.items() if k in self.results.param_axes}
        cross = self.results.sel(**_sel_state)["cross"]
        energy = cross**2

        # Energy of PF dimensions on full space
        energy_on_full = np.nansum(energy, axis=2)
        energy_on_diagonal = np.cumsum(np.diagonal(energy, axis1=1, axis2=2), axis=1)
        energy_expanding_dims = np.diagonal(np.cumsum(np.cumsum(energy, axis=1), axis=2), axis1=1, axis2=2)

        big_first_10 = np.mean(np.diagonal(energy, axis1=1, axis2=2)[:, :10], axis=1)
        idx_aligned = np.argsort(-big_first_10)

        # Average by mouse
        energy_on_full_avg = average_by_mouse(energy_on_full, self.results.mouse_names)
        energy_on_diagonal_avg = average_by_mouse(energy_on_diagonal, self.results.mouse_names)
        energy_expanding_dims_avg = average_by_mouse(energy_expanding_dims, self.results.mouse_names)
        aligned_ratio = energy_on_diagonal_avg / energy_expanding_dims_avg

        line_alpha = 0.3
        fig, ax = plt.subplots(1, 2, figsize=self.figsize, layout="constrained")

        idx_plot = idx_aligned[state["idx_cross"]]
        if state["plot_energy"]:
            imshow_data = energy[idx_plot][:100, :100]
            cmap = "gray_r"
            vmin = 0
        else:
            imshow_data = cross[idx_plot][:100, :100]
            cmap = "bwr"
            vmin = -1

        xlims0 = [-0.5, state["num_cross_show"] + 0.5]
        ylims0 = [state["num_cross_show"] + 0.5, -0.5]
        xbounds0 = [0, state["num_cross_show"]]
        ybounds0 = [state["num_cross_show"], 0]
        extent = [0, 100, 100, 0]
        ax[0].imshow(imshow_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=1, extent=extent)
        ax[0].set_xlabel("Placefield Dim.")
        ax[0].set_ylabel("Full Dim.")
        ax[0].set_xlim(xlims0)
        ax[0].set_ylim(ylims0)

        xvals = np.arange(energy_on_full_avg.shape[1]) + 1
        xbounds1 = [1, energy_on_full_avg.shape[1] + 1]
        ax[1].plot(xvals, energy_on_full_avg.T, color=("k", line_alpha), linewidth=0.5)
        ax[1].plot(xvals, np.nanmean(energy_on_full_avg, axis=0), color="k", linewidth=2.0)
        ax[1].set_xlabel("Full Dim.")
        ax[1].set_ylabel("Fraction Full Variance\nCaptured By Placefields")
        ax[1].set_xscale("log")
        ybounds1 = [0, 1]
        ax[1].set_xlim(xbounds1)
        ax[1].set_ylim(ybounds1)
        xticks = [1, 10, 100, 1000]

        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=xbounds0,
            ybounds=ybounds0,
        )
        format_spines(
            ax[1],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=xbounds1,
            ybounds=ybounds1,
            xticks=xticks,
            yticks=[0, 1.0],
        )
        return fig


def subspace_crossspace(
    results: ResultsAggregator,
    idx_cross: int = 0,
    plot_energy: bool = True,
    num_cross_show: int = 30,
    figsize: tuple[float, float] = (6.0, 3.0),
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Cross-spectrum energy figure for aggregated subspace results.

    The left panel shows an example placefield-vs-full cross matrix for one session
    (chosen by ``idx_cross`` from sessions sorted by descending mean top-10 diagonal
    energy). The right panel shows the mouse-averaged fraction of full-activity
    variance captured by placefields as a function of full-activity dimension.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated subspace results providing ``param_axes``, ``sel`` and ``mouse_names``.
    idx_cross : int
        Index (into sessions sorted by descending mean top-10 diagonal energy) of the
        example cross matrix shown in the left panel.
    plot_energy : bool
        Show squared cross energy (``gray_r``, from 0) when True, otherwise the signed
        cross values (``bwr``, from -1) in the left panel.
    num_cross_show : int
        Number of cross dimensions to show in the left panel.
    figsize : tuple[float, float]
        Figure size in inches.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections (e.g. ``smooth_width``,
        ``activity_parameters_name``). Each key must be a valid ``results.param_axes`` name.

    Returns
    -------
    matplotlib.figure.Figure or SubspaceCrossspaceViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = SubspaceCrossspaceViewer(results, figsize=figsize)
    for key, value in selections.items():
        if key not in results.param_axes:
            raise ValueError(f"Unknown selection {key!r}. Options: {list(results.param_axes)}")
        viewer.update_selection(key, value=value)
    viewer.update_integer("idx_cross", value=idx_cross)
    viewer.update_boolean("plot_energy", value=plot_energy)
    viewer.update_integer("num_cross_show", value=num_cross_show)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    plt.show()
    return fig
