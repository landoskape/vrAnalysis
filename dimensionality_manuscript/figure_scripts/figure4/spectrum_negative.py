"""Signs of saved spectra and the fraction of negative components."""

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from vrAnalysis.helpers.plotting import format_spines

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
    """Mean and sample SEM across contributing units, separately at each rank."""
    mean = _finite_mean(values)
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    ss = np.where(valid, (values - mean) ** 2, 0.0).sum(axis=0)
    sem = np.sqrt(np.divide(ss, count * (count - 1), out=np.full_like(mean, np.nan), where=count > 1))
    return mean, sem


class SpectrumNegativeViewer(FigureViewer):
    """Binary sign heatmap above a mean +/- SEM negative-fraction curve.

    Accepts the same ``results``, optional ``results_cvpca``, ``results_subspace``
    (legacy SVCA), and ``results_svca`` aggregators and PF ``source_key`` choices
    as SpectrumFigureViewer. Shared parameter widgets, including tuple-valued
    axes, select the stored analysis variation via ``**selection_defaults``.

    ``source_mode='all'`` uses the spectrum computed jointly over environments:
    one heatmap row per session. ``source_mode='avg_env'`` uses independently
    computed environment spectra: one binary row per session/environment, with
    rows grouped by mouse and numbered consecutively.
    Negative indicators are averaged over available environments within each
    session. ``average_by="mouse"`` (default) then averages sessions within each
    mouse and reports the mean and SEM across mice. ``average_by="session"``
    instead reports the mean and SEM across all sessions, weighting sessions
    equally regardless of mouse. This choice is also a viewer widget.
    CVPCA has no per-environment counterpart in this interface and ignores
    ``avg_env``, using whole-session spectra instead.
    ``sf_*`` per-environment spectra use the same ``full1`` scope as
    SpectrumFigureViewer. Missing environments/ranks are gray and excluded from
    averages; zero is nonnegative. SEM requires at least two contributing mice
    or sessions at that rank.

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
        average_by: str = "mouse",
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
        self.add_selection("average_by", options=["mouse", "session"], value=average_by)
        self.add_boolean("show_mice", value=show_mice)
        self.add_selection("xscale", options=["linear", "log"], value=xscale)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        for name in ("source_key", "source_mode", "average_by", *merged_axis_names(*self._agg.values())):
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
        summary_values = self._mouse_probabilities if state["average_by"] == "mouse" else session_values
        self._mean, self._sem = _mouse_summary(summary_values)
        # Group rows by mouse, preserving session and environment order within mouse.
        rows, labels, groups = [], [], []
        for mouse in self._mice:
            for i in np.flatnonzero(names == mouse):
                block = signs[i] if per_env else signs[i:i + 1]
                for slot, row in enumerate(block):
                    if not np.any(np.isfinite(row)):
                        continue
                    rows.append(row)
                    suffix = f" / env slot {slot + 1}" if per_env else ""
                    labels.append(f"{mouse}: {agg.session_ids[i]}{suffix}")
                    groups.append(i)
        self._heatmap = np.asarray(rows).reshape(-1, spectra.shape[-1])
        self._row_labels = labels
        self._row_sessions = np.asarray(groups)

    def plot(self, state):
        fig, axes = self.new_subplots(2, 1, sharex=True, figsize=self.figsize,
                                     height_ratios=[2, 1], layout="constrained")
        heat, curve = axes
        cmap = ListedColormap(["white", "black"]).with_extremes(bad="0.75")
        nrows, ndims = self._heatmap.shape
        if nrows and ndims:
            heat.pcolormesh(np.arange(ndims + 1) + 0.5, np.arange(nrows + 1) + 0.5,
                            np.ma.masked_invalid(self._heatmap), cmap=cmap, vmin=0, vmax=1,
                            shading="flat", rasterized=True)
            heat.set_ylim(nrows + 0.5, 0.5)
            heat.set_yticks([1, nrows] if nrows > 1 else [1])
        else:
            heat.text(0.5, 0.5, "No finite spectra for this selection", transform=heat.transAxes, ha="center")
        heat.legend(handles=[Patch(facecolor="black", label="negative\nvariance")],
                    loc="upper left", fontsize=state["fontsize"], frameon=True,
                    facecolor="white", edgecolor="none", framealpha=0.9,
                    handlelength=0.8, handleheight=0.8)
        x = np.arange(1, ndims + 1)
        if state["show_mice"]:
            curve.plot(x, self._mouse_probabilities.T, color="black", alpha=0.25, linewidth=0.8)
        curve.plot(x, self._mean, color="black", linewidth=2)
        curve.fill_between(x, self._mean - self._sem, self._mean + self._sem, color="black", alpha=0.2)
        curve.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        curve.set_ylim(-0.05, 1.05)
        curve.set_xlim(0.5, max(1.5, ndims + 0.5))
        curve.set_xscale(state["xscale"])
        per_env = self._uses_per_env(state)
        heat.set_ylabel("Session \u00d7 environment" if per_env else "Session")
        curve.set_ylabel("Fraction < 0")
        curve.set_xlabel("Dimension")
        format_spines(heat, x_pos=-0.02, y_pos=-0.02, spines_visible=["left"],
                      ybounds=[1, max(1, nrows)], tick_fontsize=state["fontsize"])
        heat.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        format_spines(curve, x_pos=-0.02, y_pos=-0.02, xbounds=[1, max(1, ndims)],
                      ybounds=[0, 1], yticks=[0, 0.5, 1], tick_fontsize=state["fontsize"])
        for ax in axes:
            ax.tick_params(labelsize=state["fontsize"])
            for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
                text.set_fontsize(state["fontsize"])
        return fig
