import numpy as np
from matplotlib import pyplot as plt
from syd import Viewer

from vrAnalysis.helpers.plotting import save_figure
from dimensionality_manuscript import ResultsAggregator


def _xvals(x: np.ndarray) -> np.ndarray:
    """Return 1-based dimension indices for a (mice, dims) spectrum array."""
    return np.arange(x.shape[1]) + 1


class PlacefieldSpectraViewer(Viewer):
    """Interactive place-field shared-variance spectrum over aggregated stim-space results.

    The panel shows the cross-validated shared (SS) spectrum, one faint line per mouse and
    a bold mouse-average, on log-log axes. The y-limits are controlled in log10 units by a
    float-range slider (the actual limits are ``10 ** state["ylim_range"]``).
    """

    def __init__(
        self,
        results: ResultsAggregator,
        ylim_range: tuple[float, float] = (-5.5, -0.8),
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (5.0, 3.0),
    ):
        self.results = results
        self.fontsize = fontsize
        self.figsize = figsize
        for key, value in results.param_axes.items():
            self.add_selection(key, options=value)

        preferred_state = {
            "activity_parameters_name": "default",
            "include_iti": False,
        }
        for key, value in preferred_state.items():
            if key in results.param_axes:
                self.update_selection(key, value=value)

        self.add_float_range("ylim_range", value=ylim_range, min=-8.0, max=2.0, step=0.1)

    def plot(self, state: dict):
        sel_params = {k: v for k, v in state.items() if k in self.results.param_axes}
        out = self.results.sel(keys=["ss_direct", "ss_cv"], avg_by_mouse=True, **sel_params)
        ss_cv = out["ss_cv"]

        ss_cv_positive = np.where(ss_cv > 0, ss_cv, np.nan)

        ss_color = "blue"
        each_alpha = 0.3
        ylim_min, ylim_max = state["ylim_range"]

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(1, 2, figsize=self.figsize, layout="constrained", width_ratios=[1, 0.5])

        ax[0].plot(_xvals(ss_cv), ss_cv_positive.T, color=ss_color, alpha=each_alpha, linewidth=1.0)
        ax[0].plot(_xvals(ss_cv), np.nanmean(ss_cv_positive, axis=0), color=ss_color, label="PF Spectrum", linewidth=2.0)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(10 ** ylim_min, 10 ** ylim_max)
        yticks = ax[0].get_yticks()
        ytick_power = [np.log10(yt) for yt in yticks]
        ax[0].set_yticks(yticks, labels=ytick_power)
        ax[0].set_ylim(10 ** ylim_min, 10 ** ylim_max)
        ax[0].set_xlabel("Shared Dimension")
        ax[0].set_ylabel("Variance")
        ax[0].legend(loc="upper right", fontsize=self.fontsize, frameon=False)

        # Second panel reserved for a cumulative-variance ratio comparison (coming soon).
        ax[1].axis("off")
        return fig


def placefield_spectra(
    results: ResultsAggregator,
    ylim_range: tuple[float, float] = (-5.5, -0.8),
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (5.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Place-field shared-variance spectrum figure for aggregated stim-space results.

    The panel shows the cross-validated shared (SS) spectrum on log-log axes: one faint
    line per mouse and a bold mouse-average.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated stim-space spectra results providing ``param_axes`` and ``sel``.
    ylim_range : tuple[float, float]
        Y-limits of the spectrum panel in log10 units; the actual limits applied are
        ``10 ** ylim_range[0]`` and ``10 ** ylim_range[1]``.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here via
        ``save_figure``.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections (e.g. ``smooth_widths``,
        ``reliability_fraction_active_thresholds``, ``include_iti``). Each key must be a
        valid ``results.param_axes`` name.

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldSpectraViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = PlacefieldSpectraViewer(results, ylim_range=ylim_range, fontsize=fontsize, figsize=figsize)
    for key, value in selections.items():
        if key not in results.param_axes:
            raise ValueError(f"Unknown selection {key!r}. Options: {list(results.param_axes)}")
        viewer.update_selection(key, value=value)
    viewer.update_float_range("ylim_range", value=ylim_range)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig
