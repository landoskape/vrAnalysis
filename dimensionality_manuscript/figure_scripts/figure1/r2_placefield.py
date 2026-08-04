"""How well a place field predicts a cell: one ROI, one session, and the population."""

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from vrAnalysis.helpers import vectorRSquared
from vrAnalysis.processors import spkmaps as SMPs
from vrAnalysis.sessions import B2Session

from dimensionality_manuscript.configs.pfpred_quality import PFPredQualityConfig, _kde_r2
from dimensionality_manuscript.figure_scripts import session_cache
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import draw_example_roi_panel, round_up_labels, style_axis


def r2_placefield_arrays(session: B2Session, smp: SMPs.SpkmapProcessor, idx_env: int):
    """
    Compute valid-frame activity, PF predictions, and per-ROI R² for one environment.

    R² below -1 is set to NaN: a prediction that bad says nothing about the cell beyond "the
    place field does not describe it", and leaving it in would set the axis range by itself.

    Returns
    -------
    spks_valid, pfpred_valid, r2, reliability
    """
    spks = session_cache.get_spks(session, params=smp.params)
    reliability = session_cache.get_reliability(session, params=smp.params)
    placefield_prediction = session_cache.get_prediction(session, "spkmap", params=smp.params)
    extras = session_cache.get_prediction_extras(session, params=smp.params)
    idx_best_environment = extras["frame_environment_index"] == idx_env
    idx_keep = extras["idx_valid"] & idx_best_environment
    spks_valid = spks[idx_keep]
    pfpred_valid = placefield_prediction[idx_keep]
    r2 = vectorRSquared(pfpred_valid, spks_valid, axis=0)
    r2[r2 < -1] = np.nan
    return spks_valid, pfpred_valid, r2, reliability


class R2PlacefieldFocus(FigureViewer):
    """Example ROI, session R² vs spatial reliability, and the across-mouse R² summary.

    - ``ax[0]``: one ROI's activity against its place-field prediction, frame by frame.
    - ``ax[1]``: every ROI of the session, R² against spatial reliability, with the running
      average E[R² | reliability] over it and the example ROI marked in red.
    - ``ax[2]``: that same running average for every mouse, the example mouse highlighted, and
      the across-mouse average in solid black.

    The example ROI panel is either its own leftmost panel or, when ``example_as_inset`` is set,
    an inset on the session panel (which then leaves a two-panel figure). It is drawn identically
    either way apart from font size, which drops to ``inset_fontsize`` when it is an inset.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``PFPredQualityConfig`` results -- the source of ``ax[2]``.
    session : B2Session
        Session ``ax[0]`` and ``ax[1]`` are computed from.
    idx_env : int
        Environment index within the session; frames of other environments are dropped.
    roi : int
        Initial ROI, in session-filtered ROI coordinates.
    example_style : {"points", "density"}
        How ``ax[0]`` draws the frames: a translucent point cloud, or a filled 2D KDE of the same
        points. A unity line is drawn either way.
    example_alpha : float
        Opacity of ``ax[0]``'s point cloud. Ignored by ``"density"``.
    cloud_style : {"hex", "scatter"}
        How ``ax[1]`` draws all ROIs.
    cloud_alpha : float or None
        Opacity of that cloud. None takes 0.55 for hex and 0.1 for scatter, which are the
        densities the two marks read at.
    hex_count_norm : {"linear", "log"}
        Color mapping of the hexbin counts (ignored by ``cloud_style="scatter"``). ``log`` keeps
        sparse regions visible when a few bins dominate the count range.
    r2_ylim : tuple[float, float]
        Shared y limits of ``ax[1]`` and ``ax[2]``, within ``(-1, 1)``.
    example_as_inset : bool
        Draw ``ax[0]`` as an inset on the session panel, leaving a 1x2 figure.
    inset_position : tuple of float
        ``(x, y, width, height)`` of that inset, in axes fractions of the session panel.
    fontsize : float
        Font size of every axis label, tick label, and the legend.
    inset_fontsize : float
        Font size of the example panel when it is an inset.
    figsize : tuple[float, float]
        Figure size in inches. Not rescaled by ``example_as_inset``, so pass a narrower width for
        the two-panel version.
    **selection_defaults
        Starting values for the aggregator's own param axes, by name.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        session: B2Session,
        *,
        idx_env: int = 0,
        roi: int = 0,
        example_style: str = "points",
        example_alpha: float = 0.1,
        cloud_style: str = "hex",
        cloud_alpha: float | None = None,
        hex_count_norm: str = "linear",
        r2_ylim: tuple[float, float] = (-1.0, 1.0),
        example_as_inset: bool = False,
        inset_position: tuple[float, float, float, float] = (0.06, 0.58, 0.32, 0.38),
        fontsize: float = 9.0,
        inset_fontsize: float = 7.0,
        figsize: tuple[float, float] = (8.0, 2.0),
        **selection_defaults,
    ):
        self.results = results
        self.session = session
        self.smp = session_cache.get_smp(session)
        self.figsize = figsize
        self._kde_grid_spec = PFPredQualityConfig().kde_grid

        num_rois = session_cache.get_spks(session, params=self.smp.params).shape[1]
        num_envs = len(session_cache.get_env_maps(session, params=self.smp.params).environments)

        # --- data selection ---
        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_integer("env", value=idx_env, min=0, max=num_envs - 1)
        self.add_selection("roi", value=roi, options=list(range(num_rois)))

        # --- what the clouds look like ---
        self.add_selection("example_style", value=example_style, options=["points", "density"])
        self.add_float("example_alpha", value=example_alpha, min=0.0, max=1.0)
        self.add_selection("cloud_style", value=cloud_style, options=["hex", "scatter"])
        self.add_selection("hex_count_norm", value=hex_count_norm, options=["linear", "log"])
        # The two marks read at very different densities, so each carries its own default.
        self.add_float("cloud_alpha", value=cloud_alpha if cloud_alpha is not None else (0.55 if cloud_style == "hex" else 0.1), min=0.0, max=1.0)
        self.add_float_range("r2_ylim", value=tuple(r2_ylim), min=-1.0, max=1.0, step=0.05)

        # --- layout and style ---
        self.add_boolean("example_as_inset", value=example_as_inset)
        self.add_float("inset_x", value=inset_position[0], min=0.0, max=1.0, step=0.01)
        self.add_float("inset_y", value=inset_position[1], min=0.0, max=1.0, step=0.01)
        self.add_float("inset_width", value=inset_position[2], min=0.05, max=1.0, step=0.01)
        self.add_float("inset_height", value=inset_position[3], min=0.05, max=1.0, step=0.01)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        self.add_float("inset_fontsize", value=inset_fontsize, min=4.0, max=24.0)

        self.on_change([*self.selection_names, "env"], self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Recompute the session's per-ROI R², its running average, and the population summary."""
        self.idx_env = state["env"]
        self.spks_valid, self.pfpred_valid, self.r2, self.reliability = r2_placefield_arrays(self.session, self.smp, self.idx_env)
        kde_result = _kde_r2(self.r2, self.reliability.values[self.idx_env], self._kde_grid_spec)
        self.kde_grid = kde_result["r2_kde_grid"]
        self.kde_mean = kde_result["r2_kde_mean"]
        self.output = self.results.sel(avg_by_mouse=True, **data_selection(state, self.results, self.selection_names))

    def _draw_session_cloud(self, ax, state, roi):
        """Every ROI of the session, R² against reliability, plus the running average and the example."""
        rel_env = self.reliability.values[self.idx_env]
        valid = np.isfinite(self.r2) & np.isfinite(rel_env)
        cloud_alpha = state["cloud_alpha"]

        if state["cloud_style"] == "hex":
            hex_norm = LogNorm(vmin=1) if state["hex_count_norm"] == "log" else None
            ax.hexbin(
                rel_env[valid],
                self.r2[valid],
                gridsize=30,
                cmap="Greys",
                mincnt=1,
                linewidths=0,
                norm=hex_norm,
                alpha=cloud_alpha,
                zorder=1,
            )
            # Black reads on grey hexes; against a black scatter the running average needs a hue.
            kde_color = "black"
        elif state["cloud_style"] == "scatter":
            ax.plot(
                rel_env[valid],
                self.r2[valid],
                markerfacecolor="k",
                markeredgecolor="none",
                marker=".",
                markersize=2.5,
                linestyle="none",
                alpha=cloud_alpha,
                zorder=1,
            )
            kde_color = "blue"
        else:
            raise ValueError(f"Invalid cloud_style: {state['cloud_style']!r}")

        ax.plot(self.kde_grid, self.kde_mean, color=kde_color, linewidth=1, zorder=5)
        ax.plot(
            rel_env[roi],
            self.r2[roi],
            markerfacecolor="r",
            markeredgecolor="none",
            marker=".",
            markersize=7.5,
            linestyle="none",
            zorder=10,
        )

    def _draw_population(self, ax, fontsize):
        """One running average per mouse, the example mouse highlighted, and their average."""
        linewidth_example = 1
        linewidth_average = 1.5
        alpha_example = 0.3
        alpha_highlight = 0.7
        idx_to_example = self.results.unique_mice.index(self.session.mouse_name)
        kde_grid = self.output["r2_kde_grid"][0]
        kde_mean = self.output["r2_kde_mean"]
        ax.plot(kde_grid, kde_mean.T, color=("k", alpha_example), linewidth=linewidth_example)
        ax.plot(kde_grid, kde_mean[idx_to_example].T, color=("blue", alpha_highlight), linewidth=linewidth_example)
        ax.plot(kde_grid, np.nanmean(kde_mean, axis=0), color="k", linewidth=linewidth_average)
        ax.legend(
            handles=[
                Line2D([0], [0], color="k", alpha=alpha_example, linewidth=linewidth_example, label="mouse"),
                Line2D([0], [0], color="blue", alpha=alpha_highlight, linewidth=linewidth_example, label="example"),
                Line2D([0], [0], color="k", linewidth=linewidth_average, label="average"),
            ],
            fontsize=fontsize,
        )

    def plot(self, state):
        fontsize = state["fontsize"]
        roi = state["roi"]

        # The example ROI panel is either the leftmost axes or an inset on the session R² panel.
        if state["example_as_inset"]:
            fig, axs = self.new_subplots(1, 2, figsize=self.figsize, layout="constrained")
            ax_session, ax_summary = axs
            inset_rect = [state["inset_x"], state["inset_y"], state["inset_width"], state["inset_height"]]
            ax_example = ax_session.inset_axes(inset_rect)
        else:
            fig, axs = self.new_subplots(1, 3, figsize=self.figsize, layout="constrained")
            ax_example, ax_session, ax_summary = axs

        # The example panel is drawn the same way whether it is its own axes or an inset; only its
        # font size changes, since an inset has much less room for labels.
        draw_example_roi_panel(
            ax_example,
            self.spks_valid.T[roi],
            self.pfpred_valid.T[roi],
            style=state["example_style"],
            alpha=state["example_alpha"],
            fontsize=state["inset_fontsize"] if state["example_as_inset"] else fontsize,
        )

        self._draw_session_cloud(ax_session, state, roi)
        self._draw_population(ax_summary, fontsize)
        for ax in (ax_session, ax_summary):
            ax.set_xlim(-1, 1)
            ax.set_xlabel("Spatial Reliability", fontsize=fontsize)
            ax.set_ylabel(r"$R^2$", fontsize=fontsize)

        # ax_session and ax_summary share one user-set y range; spine bounds and ticks are clipped to it.
        ylim = tuple(state["r2_ylim"])
        ybounds = [max(float(np.nanmin(self.r2)), ylim[0]), min(float(np.nanmax(self.r2)), ylim[1])]
        # Tick the visible extremes (rounded inward so ticks never exceed the spine bounds), plus 0.
        yticks = [0, float(np.floor(ybounds[1] * 100) / 100)]

        # Format the spines only once both limits are set: format_spines reads the current ones.
        for ax in (ax_session, ax_summary):
            ax.set_ylim(ylim)
            style_axis(
                ax,
                fontsize=fontsize,
                xbounds=[-1, 1],
                ybounds=ybounds,
                xticks=[-1, 0, 1],
                yticks=yticks,
                xlabels=round_up_labels([-1, 0, 1]),
                ylabels=round_up_labels(yticks),
            )
        return fig
