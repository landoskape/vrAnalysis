"""Spatial-reliability distributions and the fraction of cells that count as place cells."""

import numpy as np
from matplotlib.lines import Line2D

from vrAnalysis.helpers import edge2center
from vrAnalysis.helpers.plotting import beeswarm

from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)
from dimensionality_manuscript.pipeline import ResultsAggregator, average_by_mouse

from ._shared import style_axis


class ReliabilityHistogramViewer(FigureViewer):
    """Mouse-average spatial-reliability histograms + fraction-of-place-cells beeswarm.

    Reads the ``PFPredQualityConfig`` aggregator, whose per-session ``reliability`` key holds one
    spatial-reliability value per ROI (best environment) -- the same measure
    :class:`~.r2_placefield.R2PlacefieldFocus` plots R² against.

    - ``ax[0]``: for each session a reliability histogram over ``[-1, 1]`` (normalized to a
      fraction of cells), averaged across sessions within a mouse, then drawn as one thin black
      alpha line per mouse plus a thicker black across-mouse average. A dotted line marks the
      place-cell threshold.
    - ``ax[1]``: fraction of place cells (reliability above the threshold) as a beeswarm.
      ``swarm_mode`` picks either a single pooled swarm of per-mouse averages (``"pooled"``) or
      one swarm per mouse of per-session fractions, sorted by mouse average (``"by_mouse"``).

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``PFPredQualityConfig`` results (source of the per-ROI ``reliability`` key).
    place_cell_threshold : float
        Reliability cutoff defining a place cell for ``ax[1]`` (and the dotted marker on ``ax[0]``).
    n_bins : int
        Number of histogram bins over ``[-1, 1]``.
    swarm_mode : {"pooled", "by_mouse"}
        Beeswarm layout for ``ax[1]``. ``"pooled"`` also narrows the panel's width ratio, since
        one swarm needs far less room than one per mouse.
    beewidth : float
        Horizontal spread of the beeswarm points.
    hist_alpha : float
        Opacity of the per-mouse histogram lines.
    fontsize : float
        Font size of every text element.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the aggregator's own param axes, by name.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        place_cell_threshold: float = 0.3,
        n_bins: int = 40,
        swarm_mode: str = "pooled",
        beewidth: float = 0.2,
        hist_alpha: float = 0.3,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (6.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.mouse_names = results.mouse_names

        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_integer("n_bins", value=n_bins, min=5, max=100)
        self.add_float("place_cell_threshold", value=place_cell_threshold, min=-1.0, max=1.0, step=0.05)
        self.add_selection("swarm_mode", options=["pooled", "by_mouse"], value=swarm_mode)
        self.add_float("beewidth", value=beewidth, min=0.0, max=1.0, step=0.01)
        self.add_float("hist_alpha", value=hist_alpha, min=0.0, max=1.0, step=0.05)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        self.on_change([*self.selection_names, "n_bins", "place_cell_threshold"], self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select the per-ROI reliabilities and rebuild the histograms and place-cell fractions."""
        selection = data_selection(state, self.results, self.selection_names)
        # Per-session per-ROI spatial reliability, NaN-padded to the max ROI count: (n_sess, max_rois).
        reliability = np.asarray(self.results.sel(keys=["reliability"], squeeze_ones=False, **selection)["reliability"], dtype=float)

        n_bins = int(state["n_bins"])
        edges = np.linspace(-1, 1, n_bins + 1)
        self.bin_centers = edge2center(edges)
        hist = np.full((reliability.shape[0], n_bins), np.nan)
        for i, rel in enumerate(reliability):
            rel = rel[np.isfinite(rel)]
            if rel.size == 0:
                continue
            counts, _ = np.histogram(rel, bins=edges)
            hist[i] = counts / counts.sum()
        self.mouse_hist = average_by_mouse(hist, self.mouse_names)

        finite = np.isfinite(reliability)
        n_finite = finite.sum(axis=1)
        n_place = np.sum(finite & (reliability > state["place_cell_threshold"]), axis=1)
        self.fraction_place_cells = np.where(n_finite > 0, n_place / np.maximum(n_finite, 1), np.nan)

    @staticmethod
    def _swarm_offsets(values: np.ndarray) -> np.ndarray:
        """Beeswarm x offsets for ``values``, zero where a value is missing."""
        finite = np.isfinite(values)
        offsets = np.zeros_like(values)
        if finite.any():
            offsets[finite] = beeswarm(values[finite])
        return offsets

    def _draw_histograms(self, ax, state, fontsize):
        """One thin line per mouse plus a thicker across-mouse average, with the threshold marked."""
        threshold = state["place_cell_threshold"]
        ax.plot(self.bin_centers, self.mouse_hist.T, color=("k", state["hist_alpha"]), linewidth=1.0)
        ax.plot(self.bin_centers, np.nanmean(self.mouse_hist, axis=0), color="k", linewidth=2.0)
        ax.axvline(threshold, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_xlim(-1, 1)
        ylim = ax.get_ylim()
        ax.text(threshold + 0.05, ylim[1] * 0.9, f"{threshold:.1f}", color="0.6", fontsize=fontsize - 1, ha="left", va="top")
        ax.set_xlabel("Spatial Reliability", fontsize=fontsize)
        ax.set_ylabel("Fraction of Cells", fontsize=fontsize)
        ax.legend(
            handles=[
                Line2D([0], [0], color="k", alpha=state["hist_alpha"], linewidth=1.0, label="each"),
                Line2D([0], [0], color="k", linewidth=2.0, label="avg"),
            ],
            fontsize=fontsize,
            frameon=False,
            handlelength=1.25,
        )
        style_axis(
            ax,
            fontsize=fontsize,
            xbounds=[-1, 1],
            ybounds=[0, np.nanmax(self.mouse_hist)],
            xticks=[-1, 0, 1],
        )

    def _draw_swarm(self, ax, state, fontsize):
        """The place-cell fraction, pooled over mice or one swarm of sessions per mouse."""
        beewidth = state["beewidth"]
        frac = self.fraction_place_cells
        if state["swarm_mode"] == "pooled":
            vals = average_by_mouse(frac, self.mouse_names)
            ax.plot(beewidth * self._swarm_offsets(vals), vals, linestyle="none", color="k", marker="o", markersize=2.5, alpha=0.8)
            ax.plot([-0.25, 0.25], [np.nanmean(vals)] * 2, color="k", linewidth=2.0)
            ax.set_xlim(-0.5, 0.5)
            xbounds, xticks = (0, 0), []
        else:
            mice = list(dict.fromkeys(self.mouse_names))
            mice.sort(key=lambda m: np.nanmean(frac[self.mouse_names == m]), reverse=True)
            for xi, mouse in enumerate(mice):
                vals = frac[self.mouse_names == mouse]
                ax.plot(
                    xi + beewidth * self._swarm_offsets(vals),
                    vals,
                    linestyle="none",
                    color="k",
                    marker=".",
                    markersize=5,
                    alpha=0.3,
                )
                ax.plot(xi + np.array([-0.4, 0.4]), [np.nanmean(vals)] * 2, color="k", linewidth=1.2)
            ax.set_xlim(-1.0, len(mice))
            ax.set_xlabel("Mice", fontsize=fontsize)
            xbounds, xticks = (0, len(mice) - 1), range(len(mice))

        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction Place Cells", fontsize=fontsize)
        style_axis(ax, fontsize=fontsize, xbounds=xbounds, ybounds=(0, 1), yticks=[0, 0.5, 1])
        # After style_axis: format_spines sets the x ticks from xbounds, and this panel labels none.
        ax.set_xticks(xticks, labels=[])

    def plot(self, state):
        fontsize = state["fontsize"]
        width_ratios = [1, 0.33] if state["swarm_mode"] == "pooled" else [1, 1]
        fig, ax = self.new_subplots(1, 2, figsize=self.figsize, layout="constrained", width_ratios=width_ratios)
        self._draw_histograms(ax[0], state, fontsize)
        self._draw_swarm(ax[1], state, fontsize)
        return fig
