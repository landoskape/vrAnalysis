"""Compare in-PF and out-of-PF contributions to the temporal PF × Gain score."""

import numpy as np
from matplotlib.ticker import MaxNLocator

from vrAnalysis.helpers.plotting import beeswarm

from dimensionality_manuscript.configs.pfpred_quality import PF_VARIANCE_COMPARISONS
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import env_slot_color, load_placefield_cell_arrays, ordinal, pad_stack, style_axis


_REGIONS = ("within", "outside")
_REGION_LABELS = ("In-PF", "Out-of-PF")
_METRICS = {
    "r2_shared": ("r2_{region}_shared", r"PF × Gain $R^2$ contribution"),
    "r2_weighted": ("frac_unexplained_{region}_weighted", r"Regional PF × Gain $R^2$"),
    "frac_unexplained_shared": ("frac_unexplained_{region}_shared", r"PF × Gain $1-R^2$ contribution"),
    "frac_unexplained_weighted": ("frac_unexplained_{region}_weighted", "Regional unexplained fraction"),
    "residual_rms": ("residual_rms_{region}", "PF × Gain residual RMS"),
}


class _PlacefieldRegionalViewer(FigureViewer):
    def __init__(
        self,
        results: ResultsAggregator,
        *,
        metric: str = "r2_shared",
        fold: str = "cv",
        reliability_threshold: float = 0.7,
        fraction_active_threshold: float = 0.2,
        fontsize: float = 9.0,
        **selection_defaults,
    ):
        self.results = results
        mouse_names = np.asarray(results.mouse_names)
        self._rows_by_mouse = {
            mouse: sorted(np.flatnonzero(mouse_names == mouse), key=lambda row: _session_sort_key(results.sessions[row]))
            for mouse in sorted(set(mouse_names))
        }
        if not self._rows_by_mouse:
            raise ValueError("The aggregator contains no sessions.")

        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults)
        self.add_selection("metric", value=metric, options=list(_METRICS))
        self.add_selection("fold", value=fold, options=list(PF_VARIANCE_COMPARISONS))
        self.add_float("reliability_threshold", value=reliability_threshold, min=-1.0, max=1.0, step=0.01)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0, step=0.01)
        self.add_float("fontsize", value=fontsize, min=3.0, max=30.0)
        self.on_change([*self.selection_names, "metric", "fold"], self.reload_arrays)
        self.reload_arrays(self.state)

    def reload_arrays(self, state):
        metric = state["metric"]
        template, _ = _METRICS[metric]
        schema_version = "v16" if metric == "residual_rms" else "v15"
        metric_keys = {
            region: f"pf_{template.format(region=region)}_{state['fold']}_slot"
            for region in _REGIONS
        }
        selection = data_selection(state, self.results, self.selection_names)
        self.arrays, _, self.env_present = load_placefield_cell_arrays(
            self.results, selection, metric_keys, schema_version=schema_version,
        )

        if metric == "r2_weighted":
            for region in _REGIONS:
                self.arrays[region] = 1.0 - self.arrays[region]

    def session_environment_means(self, state):
        """Return paired means with shape (sessions, environment slots, 2)."""
        reliability = self.arrays["reliability"]
        active = self.arrays["fraction_active"]
        within, outside = (self.arrays[region] for region in _REGIONS)
        keep = np.isfinite(reliability) & np.isfinite(active) & np.isfinite(within) & np.isfinite(outside)
        keep &= (reliability > state["reliability_threshold"]) & (active > state["fraction_active_threshold"])
        counts = keep.sum(axis=-1)
        sums = np.stack([np.where(keep, region, 0.0).sum(axis=-1) for region in (within, outside)], axis=-1)
        return np.divide(
            sums, counts[..., None], out=np.full_like(sums, np.nan),
            where=(self.env_present & (counts > 0))[..., None],
        )


class PlacefieldRegionalScatter(_PlacefieldRegionalViewer):
    """One paired mean per mouse or session, with swarms centered at x=1 and x=2.

    ``swarm_width`` scales the horizontal beeswarm offsets in x-axis units.

    Both values use cells with finite paired scores that exceed both quality
    thresholds. ``env_slot="all"`` averages environments equally within a session.
    Mouse mode averages the available sessions equally per mouse. Integer slots
    select a specific environment in experience order, starting at zero.

    The default shared-denominator contributions add to overall PF × Gain R².
    Select a different metric or stored fold using the corresponding widgets.
    Regional scores require pfpred_quality v15. The residual RMS in activity
    units (``metric="residual_rms"``) requires v16.
    """

    def __init__(
        self, results: ResultsAggregator, *, unit: str = "mouse", env_slot: int | str = "all",
        swarm_width: float = 0.2, figsize: tuple[float, float] = (3.4, 3.0), **kwargs,
    ):
        super().__init__(results, **kwargs)
        self.figsize = figsize
        self.add_selection("unit", value=unit, options=["mouse", "session"])
        self.add_selection("env_slot", value=env_slot, options=["all", *range(self.env_present.shape[1])])
        self.add_float("swarm_width", value=swarm_width, min=0.0, max=0.5, step=0.01)

    def points(self, state):
        """Return one finite row per plotted unit, ordered in-PF, out-of-PF."""
        means = self.session_environment_means(state)
        slot = state["env_slot"]
        if slot == "all":
            counts = np.isfinite(means[..., 0]).sum(axis=1)
            sessions = np.divide(
                np.nansum(means, axis=1), counts[:, None],
                out=np.full((len(means), 2), np.nan), where=counts[:, None] > 0,
            )
        else:
            sessions = means[:, slot]
        valid = np.isfinite(sessions).all(axis=1)
        if state["unit"] == "session":
            return sessions[valid]
        mouse_means = []
        for rows in self._rows_by_mouse.values():
            samples = sessions[rows][valid[rows]]
            if samples.size:
                mouse_means.append(samples.mean(axis=0))
        return np.asarray(mouse_means).reshape(-1, 2)

    def plot(self, state):
        points = self.points(state)
        fig, ax = self.new_subplots(figsize=self.figsize, layout="constrained")
        for x, values in enumerate(points.T, 1):
            nbins = max(1, len(values) // 6) if len(values) > 1 else 0
            positions = x + state["swarm_width"] * beeswarm(values, nbins=nbins)
            ax.scatter(positions, values, s=16, color="black", alpha=0.5, linewidths=0)
        ax.set_xticks([1, 2], _REGION_LABELS)
        padding = max(0.4, state["swarm_width"] + 0.05)
        ax.set_xlim(1 - padding, 2 + padding)
        ax.set_ylabel(_METRICS[state["metric"]][1], fontsize=state["fontsize"])
        style_axis(ax, fontsize=state["fontsize"])
        return fig


class PlacefieldRegionalFamiliarityViewer(_PlacefieldRegionalViewer):
    """Overlay both regions on one across-mouse familiarity axis.

    Colors identify environment experience order, and line style identifies the
    region. Each mouse contributes one paired cell mean per exposure to that
    environment. Exposures without eligible cells retain their session number.
    Bands show the standard deviation across mice. Regional scores require
    pfpred_quality v15, and ``metric="residual_rms"`` requires v16.
    """

    def __init__(
        self, results: ResultsAggregator, *, min_mice: int = 2, hide_error: bool = False,
        show_legend: bool = True, figsize: tuple[float, float] = (5.0, 3.0), **kwargs,
    ):
        super().__init__(results, **kwargs)
        self.figsize = figsize
        n_mice = len(self._rows_by_mouse)
        self.add_integer("min_mice", value=min(min_mice, n_mice), min=1, max=n_mice)
        self.add_boolean("hide_error", value=hide_error)
        self.add_boolean("show_legend", value=show_legend)

    def plot(self, state):
        means = self.session_environment_means(state)
        fig, ax = self.new_subplots(figsize=self.figsize, layout="constrained")
        for slot in range(means.shape[1]):
            exposures = [means[rows, slot][self.env_present[rows, slot]] for rows in self._rows_by_mouse.values()]
            stacks = [pad_stack([values[:, index] for values in exposures]) for index in range(len(_REGIONS))]
            supported = np.flatnonzero(np.isfinite(stacks[0]).sum(axis=0) >= state["min_mice"])
            if not supported.size:
                continue
            x = np.arange(supported[0] + 1, supported[-1] + 2)
            visible = supported - supported[0]
            color = env_slot_color(slot)
            for stack, region, linestyle in zip(stacks, _REGION_LABELS, ("-", "--")):
                samples = stack[:, supported]
                mean = np.full(x.shape, np.nan)
                mean[visible] = np.nanmean(samples, axis=0)
                ax.plot(
                    x, mean, color=color, linestyle=linestyle, linewidth=1.5,
                    marker="." if supported.size == 1 else None, label=f"{ordinal(slot + 1)} Env · {region}",
                )
                if not state["hide_error"]:
                    std = np.full_like(mean, np.nan)
                    std[visible] = np.nanstd(samples, axis=0)
                    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
        fontsize = state["fontsize"]
        ax.set_xlabel("Env session #", fontsize=fontsize)
        ax.set_ylabel(_METRICS[state["metric"]][1], fontsize=fontsize)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if not ax.lines:
            ax.set_xlim(0.5, 1.5)
        style_axis(ax, fontsize=fontsize)
        if state["show_legend"] and ax.lines:
            ax.legend(frameon=False, fontsize=fontsize)
        return fig
