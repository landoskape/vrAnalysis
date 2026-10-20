"""Signs of saved spectra and the probability of a negative component."""

import numpy as np
from matplotlib.colors import ListedColormap

from dimensionality_manuscript import ResultsAggregator
from ..panels import FigureViewer
from ._param_axes import (
    CVPCA_KEYS,
    PREFERRED_DEFAULTS,
    SOURCE_OF_KEY,
    STIMSPACE_KEYS,
    add_merged_param_axis_widgets,
    merged_axis_names,
    sel_params,
)
from .spectrum_figure import LEGACY_SVCA_PF_KEYS, SVCA_PF_KEYS, SpectrumFigureViewer


def _finite_mean(values, axis=0):
    """Mean of available values, preserving all-missing ranks without warnings."""
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _mouse_summary(values):
    """Across-mouse mean and sample SEM, separately at each rank."""
    mean = _finite_mean(values)
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    ss = np.where(valid, (values - mean) ** 2, 0.0).sum(axis=0)
    sem = np.sqrt(np.divide(ss, count * (count - 1), out=np.full_like(mean, np.nan), where=count > 1))
    return mean, sem


class SpectrumNegativeViewer(FigureViewer):
    """Binary sign heatmap above a mouse mean +/- SEM negative-probability curve.

    Accepts the same ``results``, optional ``results_cvpca``, ``results_subspace``
    (legacy SVCA), and ``results_svca`` aggregators and PF ``source_key`` choices
    as SpectrumFigureViewer. Shared parameter widgets, including tuple-valued
    axes, select the stored analysis variation via ``**selection_defaults``.

    ``source_mode='all'`` uses the spectrum computed jointly over environments:
    one heatmap row per session. ``source_mode='avg_env'`` uses independently
    computed environment spectra: one binary row per session/environment, with
    session boundaries marked and mouse names centered on their row blocks.
    Negative indicators are averaged over available
    environments within each session, then sessions within mouse, then mice.
    CVPCA has no per-environment counterpart in this interface and ignores
    ``avg_env``, using whole-session spectra instead.
    ``sf_*`` per-environment spectra use the same ``full1`` scope as
    SpectrumFigureViewer. Missing environments/ranks are gray and excluded from
    averages; zero is nonnegative. SEM requires at least two mice at that rank.

    Fold combinations have already been averaged in the saved analysis results,
    *before* testing their sign. Individual-fold signs cannot be recovered, so
    no all-fold-combinations toggle is offered. No normalization, clipping, or
    display smoothing is applied. ``show_mice`` overlays each mouse's curve;
    ``xscale`` selects linear/log dimension axes. Both are Syd widgets.

    Example::

        viewer = SpectrumNegativeViewer(results, results_cvpca=results_cvpca,
                                        results_svca=results_svca)
        viewer.show()
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        results_svca: ResultsAggregator | None = None,
        *,
        source_key: str = "ss_cv",
        source_mode: str = "all",
        show_mice: bool = True,
        xscale: str = "linear",
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (7.0, 6.0),
        **selection_defaults,
    ):
        svca = results_svca if results_svca is not None else results_subspace
        self._svca_pf_keys = SVCA_PF_KEYS if results_svca is not None else LEGACY_SVCA_PF_KEYS
        self._agg = {"stimspace": results, "cvpca": results_cvpca, "svca": svca}
        self.figsize = figsize
        options = list(STIMSPACE_KEYS)
        if results_cvpca is not None:
            options.extend(CVPCA_KEYS)
        if svca is not None:
            options.extend(SVCA_PF_KEYS)
        if source_key not in options:
            raise ValueError(f"Unknown or unavailable source_key {source_key!r}. Options: {options}")
        self.add_selection("source_key", options=options, value=source_key)
        self.add_selection("source_mode", options=["all", "avg_env"], value=source_mode)
        self._tuple_labels = add_merged_param_axis_widgets(
            self, *self._agg.values(), preferred_defaults={**PREFERRED_DEFAULTS, **selection_defaults}
        )
        self.add_boolean("show_mice", value=show_mice)
        self.add_selection("xscale", options=["linear", "log"], value=xscale)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        for name in ("source_key", "source_mode", *merged_axis_names(*self._agg.values())):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _uses_per_env(self, state):
        return state["source_mode"] == "avg_env" and state["source_key"] not in CVPCA_KEYS

    def refresh_data(self, state):
        key = state["source_key"]
        source = "svca" if key in SVCA_PF_KEYS else SOURCE_OF_KEY[key]
        agg = self._agg[source]
        per_env = self._uses_per_env(state)
        stored_key = self._svca_pf_keys[key] if source == "svca" else key
        if per_env:
            stored_key = f"{stored_key}_env" if source == "svca" else SpectrumFigureViewer._per_env_pf_key(key)
        if stored_key not in agg.arrays:
            raise ValueError(f"Stored spectrum {stored_key!r} is unavailable; load results containing this key.")
        spectra = np.asarray(
            agg.sel(keys=[stored_key], squeeze_ones=False, avg_by_mouse=False,
                    **sel_params(state, self._tuple_labels, agg.param_axes))[stored_key],
            dtype=float,
        )
        expected_ndim = 3 if per_env else 2
        if spectra.ndim != expected_ndim:
            raise ValueError(f"Expected {expected_ndim} spectrum axes, got shape {spectra.shape}.")
        signs = np.where(np.isfinite(spectra), (spectra < 0).astype(float), np.nan)
        session_values = _finite_mean(signs, axis=1) if per_env else signs
        names = np.asarray(agg.mouse_names)
        self._mice = list(dict.fromkeys(names))
        self._mouse_probabilities = np.stack([_finite_mean(session_values[names == mouse]) for mouse in self._mice])
        self._mean, self._sem = _mouse_summary(self._mouse_probabilities)
        # Group rows by mouse, preserving session and environment order within mouse.
        rows, labels, groups = [], [], []
        mouse_ticks, mouse_labels = [], []
        for mouse in self._mice:
            start = len(rows)
            for i in np.flatnonzero(names == mouse):
                block = signs[i] if per_env else signs[i:i + 1]
                for slot, row in enumerate(block):
                    if not np.any(np.isfinite(row)):
                        continue
                    rows.append(row)
                    suffix = f" / env slot {slot + 1}" if per_env else ""
                    labels.append(f"{mouse}: {agg.session_ids[i]}{suffix}")
                    groups.append(i)
            if len(rows) > start:
                mouse_ticks.append((start + len(rows) - 1) / 2)
                mouse_labels.append(mouse)
        self._heatmap = np.asarray(rows).reshape(-1, spectra.shape[-1])
        self._row_labels = labels
        self._row_sessions = np.asarray(groups)
        self._mouse_ticks = mouse_ticks
        self._mouse_labels = mouse_labels

    def plot(self, state):
        fig, axes = self.new_subplots(2, 1, sharex=True, figsize=self.figsize,
                                     height_ratios=[2, 1], layout="constrained")
        heat, curve = axes
        cmap = ListedColormap(["white", "black"]).with_extremes(bad="0.75")
        nrows, ndims = self._heatmap.shape
        if nrows and ndims:
            mesh = heat.pcolormesh(np.arange(ndims + 1) + 0.5, np.arange(nrows + 1) - 0.5,
                                  np.ma.masked_invalid(self._heatmap), cmap=cmap, vmin=0, vmax=1,
                                  shading="flat", rasterized=True)
            heat.set_ylim(nrows - 0.5, -0.5)
            heat.set_yticks(self._mouse_ticks, labels=self._mouse_labels)
            for boundary in np.flatnonzero(np.diff(self._row_sessions)) + 0.5:
                heat.axhline(boundary, color="tab:blue", linewidth=0.4, alpha=0.5)
            bar = fig.colorbar(mesh, ax=heat, ticks=[0, 1], pad=0.02)
            bar.ax.set_yticklabels(["Nonnegative", "Negative"])
            bar.ax.tick_params(labelsize=state["fontsize"])
        else:
            heat.text(0.5, 0.5, "No finite spectra for this selection", transform=heat.transAxes, ha="center")
        x = np.arange(1, ndims + 1)
        if state["show_mice"]:
            curve.plot(x, self._mouse_probabilities.T, color="tab:blue", alpha=0.25, linewidth=0.8)
        curve.plot(x, self._mean, color="tab:blue", label="Mouse mean +/- SEM", linewidth=2)
        curve.fill_between(x, self._mean - self._sem, self._mean + self._sem, color="tab:blue", alpha=0.2)
        curve.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        curve.set_ylim(-0.03, 1.03)
        curve.set_xlim(0.5, max(1.5, ndims + 0.5))
        curve.set_xscale(state["xscale"])
        per_env = self._uses_per_env(state)
        mode = "avg_env" if per_env else "all"
        heat.set_title(f"{state['source_key']} | {mode} | saved fold-averaged spectra")
        heat.set_ylabel("Session / environment" if per_env else "Session")
        curve.set_ylabel("Probability < 0")
        curve.set_xlabel("Dimension (1-based)")
        curve.legend(fontsize=state["fontsize"])
        for ax in axes:
            ax.tick_params(labelsize=state["fontsize"])
            for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
                text.set_fontsize(state["fontsize"])
        return fig
