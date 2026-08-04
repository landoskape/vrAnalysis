"""Per-cell placefield feature values across the sessions of one env-experience slot."""

import numpy as np
from matplotlib import pyplot as plt

from vrAnalysis.helpers.plotting import errorPlot, format_spines
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)
from ...env_order import MAX_ENV_SLOTS

from ._curves import chronological_mouse_sessions, mean_with_min_support, pad_stack_by_mouse, support_length
from ._familiarity import FAMILIARITY_STYLES
from ._selection import ACTIVITY_SELECTION_DEFAULTS

# Per-cell feature keys produced by ``PlaceFieldStructureConfig._compute_pf_features`` that are
# pickable as plot curves (excludes ``env_slot_ids``, which has no neuron axis; excludes
# ``reliability``/``fraction_active``, which are used as per-ROI inclusion thresholds instead --
# see :func:`pf_env_curves`).
PF_FEATURE_KEYS = [
    "pf_mean",
    "pf_var",
    "pf_norm",
    "pf_max",
    "pf_cv",
    "pf_gauss_amp",
    "pf_gauss_center",
    "pf_gauss_width",
    "pf_gauss_r2",
    "spatial_participation",
    "pf_tdot_mean",
    "pf_tdot_std",
    "pf_tdot_cv",
    "pf_tcorr_mean",
    "pf_tcorr_std",
]


def _oom_bucket(scale: float) -> int:
    """Order-of-magnitude bucket for a positive value scale: 0 covers ``(0, 1]``, 1 covers ``(1, 10]``, etc."""
    if not np.isfinite(scale) or scale <= 0:
        return 0
    return max(0, int(np.ceil(np.log10(scale))))


def _zscore_curve(values: np.ndarray) -> np.ndarray:
    """NaN-aware zscore of a 1D curve; returns it unchanged if empty or constant."""
    if values.size == 0:
        return values
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values - mean
    return (values - mean) / std


def pf_env_curves(
    results: ResultsAggregator,
    sel_params: dict,
    keys: list[str],
    env_slot: int,
    reliability_threshold: float = -1.0,
    fraction_active_threshold: float = 0.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-mouse, env-session-indexed curves for each feature key at one env-experience slot.

    ROIs are filtered *within each session* first (``reliability >= reliability_threshold`` and
    ``fraction_active >= fraction_active_threshold``), then averaged across the surviving ROIs to
    get one value per session. Each mouse's sessions are then reindexed to the chronological
    session number *within* this env slot (0 = the mouse's first session containing this
    environment) rather than the mouse's overall session number -- the same ``by_env`` x-axis
    convention used by the familiarity panel. A session counts as present for this slot when
    ``pf_mean`` (always computed whenever the slot has data, independent of the ROI filter) is
    finite for at least one neuron; a present session with zero surviving ROIs still keeps its
    slot in the sequence, just with a NaN value.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``{feature_key: {mouse: 1D array of env-session-indexed values}}``.
    """
    fetch_keys = list(dict.fromkeys(keys + ["pf_mean", "reliability", "fraction_active"]))
    out = results.sel(keys=fetch_keys, squeeze_ones=False, avg_by_mouse=False, **sel_params)
    presence = np.isfinite(out["pf_mean"][:, env_slot, :]).any(axis=1)
    roi_mask = (out["reliability"][:, env_slot, :] >= reliability_threshold) & (out["fraction_active"][:, env_slot, :] >= fraction_active_threshold)

    curves: dict[str, dict[str, np.ndarray]] = {}
    for key in keys:
        values = np.nanmean(np.where(roi_mask, out[key][:, env_slot, :], np.nan), axis=1)
        per_mouse = {}
        for mouse in results.unique_mice:
            idx_sorted = chronological_mouse_sessions(results, mouse, exclude_bad_envs=False)
            mouse_keep = presence[idx_sorted]
            per_mouse[mouse] = values[idx_sorted[mouse_keep]]
        curves[key] = per_mouse
    return curves


class PlaceFieldStructureOverTimeViewer(FigureViewer):
    """Per-cell placefield feature values across sessions of one env-experience slot.

    The x-axis is each mouse's chronological session index *within* the chosen env slot (position
    within env, following the familiarity panel's ``by_env`` indexing), not the mouse's overall
    session number. Selected feature keys (population mean over neurons per session) are overlaid,
    each drawn in a fixed color from a large colormap and labeled. Panels are auto-split by order
    of magnitude, so e.g. a ``[0, 1]``-ranged feature and a ``[0, 10]``-ranged one (like ``pf_cv``)
    don't get flattened onto the same axis.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``PlaceFieldStructureConfig`` results providing ``param_axes``, ``sel``,
        ``unique_mice``, ``mouse_names`` and ``sessions``.
    env_slot : int
        Env-experience-order slot (0-indexed, ``< MAX_ENV_SLOTS``) to plot.
    feature_keys : list of str or None
        Feature keys to overlay (see :data:`PF_FEATURE_KEYS` for the full set). Defaults to
        ``["pf_mean"]``.
    style : {"errorPlot", "all"}
        ``"all"`` plots every mouse's curve as a faint line plus the mouse-mean as a bold line;
        ``"errorPlot"`` shows only the mouse mean +/- SE as a shaded band.
    zscore : bool
        If True, zscore each mouse's curve individually (before the OOM panel split), showing
        relative variation across sessions rather than absolute values -- useful for comparing
        shape across features/mice with very different baselines and scales.
    reliability_threshold : float
        Minimum per-ROI ``reliability`` (range ``[-1, 1]``) required for a session's ROI to count
        toward that session's population mean. Filtering happens within each session, before
        averaging across surviving ROIs.
    fraction_active_threshold : float
        Minimum per-ROI ``fraction_active`` (range ``[0, 1]``) required, same filtering rule as
        ``reliability_threshold``.
    fontsize : float
        Font size for axis labels and tick labels. The per-panel legend is drawn at 0.8x this.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the data-selection widgets (``activity_parameters_name``,
        ``smooth_width``).
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        env_slot: int = 0,
        feature_keys: list[str] | None = None,
        style: str = "all",
        zscore: bool = False,
        reliability_threshold: float = -1.0,
        fraction_active_threshold: float = 0.0,
        fontsize: float = 10.0,
        figsize: tuple[float, float] = (8.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize

        self.selection_names = add_data_selection_widgets(
            self, results, defaults={**ACTIVITY_SELECTION_DEFAULTS, **selection_defaults}
        )

        self.add_integer("env_slot", value=env_slot, min=0, max=MAX_ENV_SLOTS - 1)
        self.add_multiple_selection("feature_keys", value=list(feature_keys or ["pf_mean"]), options=list(PF_FEATURE_KEYS))
        self.add_selection("style", value=style, options=FAMILIARITY_STYLES)
        self.add_boolean("zscore", value=zscore)
        self.add_float("reliability_threshold", value=reliability_threshold, min=-1.0, max=1.0, step=0.001)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0, step=0.001)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0, step=0.5)

        cmap = plt.get_cmap("gist_ncar")
        n = len(PF_FEATURE_KEYS)
        self._colors = {key: cmap(i / max(n - 1, 1))[:3] for i, key in enumerate(PF_FEATURE_KEYS)}

        for name in (
            *self.selection_names,
            "env_slot",
            "feature_keys",
            "zscore",
            "reliability_threshold",
            "fraction_active_threshold",
        ):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select the per-mouse feature curves and re-bucket them by order of magnitude."""
        keys = list(state["feature_keys"]) or [PF_FEATURE_KEYS[0]]
        curves = pf_env_curves(
            self.results,
            data_selection(state, self.results, self.selection_names),
            keys,
            state["env_slot"],
            reliability_threshold=state["reliability_threshold"],
            fraction_active_threshold=state["fraction_active_threshold"],
        )
        if state["zscore"]:
            curves = {key: {mouse: _zscore_curve(values) for mouse, values in per_mouse.items()} for key, per_mouse in curves.items()}
        self._curves = curves

        buckets: dict[int, list[str]] = {}
        for key in keys:
            stack = pad_stack_by_mouse(curves[key])
            finite = stack[np.isfinite(stack)]
            scale = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
            buckets.setdefault(_oom_bucket(scale), []).append(key)
        self._buckets = buckets

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        bucket_ids = sorted(self._buckets)
        fig, axes = self.new_subplots(1, len(bucket_ids), figsize=self.figsize, layout="constrained", squeeze=False)
        axes = axes[0]

        for ax, bucket in zip(axes, bucket_ids):
            for key in self._buckets[bucket]:
                stack = pad_stack_by_mouse(self._curves[key])
                length = support_length(stack)
                stack = stack[:, :length]
                xvals = np.arange(length)
                color = self._colors[key]
                if state["style"] == "all":
                    for values in self._curves[key].values():
                        ax.plot(np.arange(len(values)), values, color=color + (0.3,), linewidth=0.5)
                    ax.plot(xvals, mean_with_min_support(stack), color=color, linewidth=2.0, label=key)
                elif length:
                    errorPlot(xvals, stack, axis=0, se=True, ax=ax, color=color, linewidth=2.0, label=key, alpha=0.25)
            ax.set_xlabel("Env Session #", fontsize=fontsize)
            ax.set_ylabel(f"Value (OOM ~1e{bucket})", fontsize=fontsize)
            ax.legend(fontsize=fontsize * 0.8, frameon=False)
            format_spines(ax, x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"], tick_fontsize=fontsize)
        return fig
