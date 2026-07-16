from typing import Optional

import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from syd import Viewer

from vrAnalysis.helpers import edge2center
from vrAnalysis.helpers.plotting import save_figure, beeswarm, format_spines
from dimilibi.helpers import fit_powerlaw_decay, fit_powerlaw_derivatives
from dimensionality_manuscript import ResultsAggregator, average_by_mouse
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.configs.tilbury_fit import TilburyFitConfig, _eval_tilbury, _eval_gaussian, _SPLITS

# Selectable spectrum keys and which aggregator each one comes from. StimSpace keys resolve
# against the StimSpaceSpectra aggregator; CVPCA keys against the CVPCAConfig aggregator.
_STIMSPACE_KEYS = ["ss_cv", "ss_direct", "ss_cvpca"]
_CVPCA_KEYS = ["reg_covariances_fixed"]
# The full (functional) spectrum key, also from the StimSpaceSpectra aggregator.
_FF_KEY = "ff"

# Tilbury-fit eigenvalue spectra selectable as extra ax[0] overlays in SpectrumFigureViewer, with
# fixed colors. These come from the TilburyFitConfig aggregator, which fits only reliable/active
# neurons at a single fixed reliability/fraction-active threshold (no selection axis).
_FIT_KEYS = ["eig_tilbury", "eig_control"]
_FIT_KEY_COLORS = {"eig_tilbury": "blue", "eig_control": "green"}
_FIT_KEY_LABELS = {"eig_tilbury": "Generalized", "eig_control": "Gaussian"}
_TILBURY_REL_FA = (0.3, 0.1)

# Population alpha-comparison panel (ax[3] of PlacefieldPopulationViewer): the selected source_key
# spectrum plus the three Tilbury-fit eigenvalue spectra, each a per-mouse power-law-exponent
# beeswarm. "eig_gaussian" in the request is the plain-Gaussian control key ``eig_control``.
_POP_EIG_KEYS = ["eig_better", "eig_tilbury", "eig_control"]
_POP_ALPHA_COLORS = {"source_key": "orange", "eig_better": "red", "eig_tilbury": "blue", "eig_control": "black"}
_POP_ALPHA_LABELS = {"eig_better": "Better", "eig_tilbury": "Generalized", "eig_control": "Gaussian"}
# key -> "stimspace" | "cvpca"
SOURCE_OF_KEY = {
    **{k: "stimspace" for k in _STIMSPACE_KEYS},
    **{k: "cvpca" for k in _CVPCA_KEYS},
    _FF_KEY: "stimspace",
}

# Fixed color per curve option, used for the per-mouse alpha scatter (ax[1]) and the
# local-exponent curves (ax[2]) so a given curve reads the same across panels.
_KEY_COLORS = {
    "ss_cv": "black",
    "ss_direct": "blue",
    "ss_cvpca": "red",
    "reg_covariances_fixed": "green",
    "ff": "purple",
}

# Preferred default values for shared param widgets, keyed by raw param-axis name. A widget is
# seeded with this value when the axis exists (in any source) and the value is among its options.
_PREFERRED_DEFAULTS = {
    "activity_parameters_name": "default",
    "include_iti": False,
    "spks_type": "sigrebase",
    "center": True,
    "use_fast_sampling": True,
}


def _xvals(x: np.ndarray) -> np.ndarray:
    """Return 1-based dimension indices for a (mice, dims) spectrum array."""
    return np.arange(x.shape[1]) + 1


def _smooth_kernel(kind: str, width: float) -> np.ndarray:
    """Normalized 1-D smoothing kernel over rank index.

    ``width`` is the boxcar full-width in rank units. The Gaussian uses ``sigma = width / 2`` so
    its ``+/- 1 sigma`` bulk spans the same window as the boxcar.
    """
    if kind == "boxcar":
        length = max(1, int(round(width)))
        return np.ones(length) / length
    if kind == "gaussian":
        sigma = width / 2.0
        radius = max(1, int(np.ceil(3.0 * sigma)))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return kernel / kernel.sum()
    raise ValueError(f"Unknown smoothing kind {kind!r}.")


def _smooth_spectrum(spec: np.ndarray, kind: str, width: float) -> np.ndarray:
    """Geometric-mean (log-space) smoothing of a ``(mice, dims)`` spectrum along ranks.

    Smoothing is done on ``log(spec)`` (slope-preserving for power laws) with a NaN-aware weighted
    convolution: non-positive/NaN entries are excluded and the kernel is renormalized per output
    point (which also handles edges). ``kind == "none"`` or ``width <= 0`` returns ``spec`` unchanged.
    """
    if kind == "none" or width <= 0:
        return spec
    kernel = _smooth_kernel(kind, width)
    with np.errstate(invalid="ignore"):
        logspec = np.where(spec > 0, np.log(spec), np.nan)
    mask = np.isfinite(logspec)
    filled = np.where(mask, logspec, 0.0)
    num = np.stack([np.convolve(row, kernel, mode="same") for row in filled])
    den = np.stack([np.convolve(row, kernel, mode="same") for row in mask.astype(float)])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return np.exp(out)


def _tuple_label(value: tuple) -> str:
    """Render a tuple param value (elements are float or None) as a widget-safe string label."""
    return "-".join("None" if v is None else str(v) for v in value)


def _clamp_range(start: int, end: int, n: int) -> tuple[int, int]:
    """Clamp a [start, end) index window to a spectrum of length ``n``."""
    end = int(min(end, n))
    start = int(min(max(start, 0), max(end - 1, 0)))
    return start, end


def _decay_alpha_per_mouse(spec: np.ndarray, start: int, end: int) -> np.ndarray:
    """Power-law exponent per mouse from a log-log fit over ranks ``[start, end)``.

    ``spec`` is ``(mice, dims)``; returns one alpha per row (NaN where the fit fails).
    """
    start, end = _clamp_range(start, end, spec.shape[1])
    alphas = np.full(spec.shape[0], np.nan)
    if end - start < 2:
        return alphas
    for m in range(spec.shape[0]):
        try:
            alphas[m], _ = fit_powerlaw_decay(spec[m], start_idx=start, end_idx=end, ignore_nans=True, verbose=False)
        except (ValueError, RuntimeError, TypeError):
            alphas[m] = np.nan
    return alphas


def _local_alpha_curve(spec: np.ndarray, width: int, eps: float = 1e-8) -> np.ndarray:
    """Per-rank local exponent (five-point derivative) for each mouse.

    ``spec`` is ``(mice, dims)``; returns a same-shaped array, NaN-padded where the stencil
    does not reach (edges) or the spectrum is non-positive.
    """
    n = spec.shape[1]
    out = np.full(spec.shape, np.nan, dtype=float)
    if n >= 4 * width + 1:
        alpha_local, idx_slice = fit_powerlaw_derivatives(spec, width=width, axis=1, eps=eps)
        out[:, idx_slice] = np.asarray(alpha_local, dtype=float)
    return out


def _deriv_alpha_per_mouse(local_alpha: np.ndarray, start: int, end: int) -> np.ndarray:
    """Per-mouse exponent from the mean local exponent over ranks ``[start, end)``."""
    start, end = _clamp_range(start, end, local_alpha.shape[1])
    if end - start < 1:
        return np.full(local_alpha.shape[0], np.nan)
    return np.nanmean(local_alpha[:, start:end], axis=1)


def _signed_participation_ratio(spec: np.ndarray) -> np.ndarray:
    """Signed participation ratio ``(sum lambda)^2 / sum(lambda^2)`` per mouse.

    ``spec`` is ``(mice, dims)``. "Signed" because negative spectrum entries are used as-is (no
    clipping); scale-invariant so normalization does not matter.
    """
    s1 = np.nansum(spec, axis=1)
    s2 = np.nansum(spec**2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s2 > 0, s1**2 / s2, np.nan)


def _beeswarm_panel(ax, values_list, colors, labels, fontsize, beewidth: float = 0.2, each_alpha: float = 0.3, yscale: str = "linear") -> None:
    """Per-mouse beeswarm (points + bold mean line) at integer x-positions ``0, 1, ...``."""
    line_extent = np.array([-0.25, 0.25])
    for x, (vals, color) in enumerate(zip(values_list, colors)):
        vals = np.asarray(vals, dtype=float)
        offsets = np.zeros_like(vals)
        finite = np.isfinite(vals)
        if finite.any():
            offsets[finite] = beeswarm(vals[finite])
        ax.plot(x + beewidth * offsets, vals, color=color, linestyle="none", marker="o", markersize=3, alpha=each_alpha)
        ax.plot(x + line_extent, [np.nanmean(vals)] * 2, color=color, linewidth=2.0)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_yscale(yscale)
    ymin = 0 if yscale == "linear" else 1
    _ax_ylim = ax.get_ylim()
    ax.set_ylim(ymin, _ax_ylim[1])

    xticks = range(len(labels))
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=[0, max(xticks)],
        ybounds=[ymin, _ax_ylim[1]],
    )
    if len(labels) > 2:
        ax.set_xticks(xticks, labels=labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_xticks(xticks, labels=labels, fontsize=fontsize)


class PlacefieldSpectraViewer(Viewer):
    """Interactive shared-variance spectrum + power-law exponent over aggregated results.

    Three panels: ax[0] the ``source_key`` example spectrum (one faint line per mouse, bold
    mouse-average, log-log); ax[1] a per-mouse beeswarm of the power-law exponent over ranks
    ``[start, end)`` for every curve option, grouped by method (log-log fit vs mean five-point
    derivative); ax[2] the per-rank five-point-derivative local-exponent curves for every option.
    The example spectrum is chosen by the ``source_key`` selection: StimSpaceSpectra keys
    (``ss_cv``, ``ss_direct``) are pulled from ``results`` and the ``reg_covariances_fixed`` key
    from ``results_cvpca``. The implementation knows which aggregator each key belongs to via
    :data:`SOURCE_OF_KEY`.

    Both aggregators may expose param axes with the same name (e.g. ``activity_parameters_name``).
    These share a single widget keyed by the raw axis name; at plot time only the params present in
    the active source's ``param_axes`` are forwarded to :meth:`ResultsAggregator.sel`. Tuple-valued
    axes (e.g. ``smooth_widths``) are auto-detected and encoded as string labels for the dropdown,
    then decoded back to tuples before selection.

    The lower y-limit is controlled in log10 units by a float slider (the applied floor is
    ``10 ** state["ylim_min"]``); the upper limit is autoscaled to the data.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        ylim_min: float = -5.5,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (5.0, 3.0),
    ):
        self.results = results
        self.results_cvpca = results_cvpca
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.fontsize = fontsize
        self.figsize = figsize

        available = list(_STIMSPACE_KEYS)
        if results_cvpca is not None:
            available += list(_CVPCA_KEYS)
        self.add_multiple_selection("source_key", options=available, value=["ss_cv"])

        # One widget per param-axis name, shared across sources. Options are the union of each
        # source's options for that axis (in first-seen order).
        merged_axes: dict[str, list] = {}
        for agg in self._agg.values():
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(opt for opt in options if opt not in existing)

        # Axes whose options are tuples (e.g. smooth_widths) can't be dropdown values directly, so
        # they are encoded as string labels; ``_tuple_labels[name]`` maps label -> original tuple.
        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for name, options in merged_axes.items():
            if any(isinstance(opt, tuple) for opt in options):
                label_map = {_tuple_label(opt): opt for opt in options}
                self._tuple_labels[name] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(name, options=widget_options)
            if name in _PREFERRED_DEFAULTS:
                default = self.encode_param(name, _PREFERRED_DEFAULTS[name])
                if default in widget_options:
                    self.update_selection(name, value=default)

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_boolean("normalize", value=True)
        # Rank window (0-based, [start, end)) the exponent is estimated over, for both methods.
        self.add_integer_range("fit_range", value=(10, 20), min=1, max=200)
        # Stencil half-width for the five-point-derivative local exponent.
        self.add_integer("deriv_width", value=1, min=1, max=10)
        # Log-space (geometric-mean) pre-smoothing of the spectrum before fitting.
        self.add_selection("smooth_kind", options=["none", "boxcar", "gaussian"], value="none")
        self.add_float("smooth_width", value=3.0, min=0.0, max=50.0, step=0.5)

    def encode_param(self, name: str, value):
        """Map a raw param value to its widget value (tuple -> string label; else unchanged)."""
        if name in self._tuple_labels and isinstance(value, tuple):
            return _tuple_label(value)
        return value

    def _sel_params(self, state: dict, source: str) -> dict:
        """Select the params relevant to this source, decoding tuple labels back to tuples."""
        agg = self._agg[source]
        params = {}
        for name in agg.param_axes:
            if name not in state:
                continue
            value = state[name]
            if name in self._tuple_labels:
                value = self._tuple_labels[name][value]
            params[name] = value
        return params

    def _available_keys(self) -> list[str]:
        """Curve options available to estimate exponents for (source_key options)."""
        keys = list(_STIMSPACE_KEYS)
        if self.results_cvpca is not None:
            keys += list(_CVPCA_KEYS)
        return keys

    def _spectrum(self, state: dict, key: str) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` spectrum for ``key``, normalized per ``state``."""
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        spec = agg.sel(keys=[key], avg_by_mouse=True, **self._sel_params(state, source))[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, state["smooth_kind"], state["smooth_width"])

    def plot(self, state: dict):
        selected_keys = list(state["source_key"])
        keys_all = self._available_keys()
        start, end = (int(v) for v in state["fit_range"])
        deriv_width = int(state["deriv_width"])
        each_alpha = 0.3
        ylim_min = state["ylim_min"]

        # Per-curve spectra, local-exponent curves, and both per-mouse exponent estimates.
        spectra = {k: self._spectrum(state, k) for k in keys_all}
        local_alpha = {k: _local_alpha_curve(spectra[k], deriv_width) for k in keys_all}
        decay_alpha = {k: _decay_alpha_per_mouse(spectra[k], start, end) for k in keys_all}
        deriv_alpha = {k: _deriv_alpha_per_mouse(local_alpha[k], start, end) for k in keys_all}

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(1, 3, figsize=self.figsize, layout="constrained", width_ratios=[1.0, 0.9, 0.9])

        # --- ax[0]: the selected example spectra (one faint line per mouse + bold average per key) ---
        for key in selected_keys:
            spec = spectra[key]
            spec_positive = np.where(spec > 0, spec, np.nan)
            ex_color = _KEY_COLORS.get(key, "blue")
            ax[0].plot(_xvals(spec), spec_positive.T, color=ex_color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(_xvals(spec), np.nanmean(spec_positive, axis=0), color=ex_color, label=key, linewidth=2.0)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(bottom=10**ylim_min)
        yticks = ax[0].get_yticks()
        ytick_power = [np.log10(yt) for yt in yticks]
        ax[0].set_yticks(yticks, labels=ytick_power)
        ax[0].set_ylim(bottom=10**ylim_min)
        ax[0].set_xlabel("Shared Dimension")
        ax[0].set_ylabel("Variance")
        ax[0].legend(loc="upper right", fontsize=self.fontsize, frameon=False)

        # --- ax[1]: per-mouse exponent, beeswarm, two method groups x each curve option ---
        methods = [("power-law fit", decay_alpha), ("5-pt deriv", deriv_alpha)]
        n = len(keys_all)
        beewidth = 0.2
        line_extent = np.array([-0.25, 0.25])
        np1 = np.array([1, 1])
        xticks = []
        all_vals = []
        for g, (_, alphas_by_key) in enumerate(methods):
            for i, k in enumerate(keys_all):
                x = g * (n + 1) + i
                xticks.append(x)
                vals = alphas_by_key[k]
                all_vals.append(vals)
                color = _KEY_COLORS.get(k, "gray")
                offsets = np.zeros_like(vals)
                finite = np.isfinite(vals)
                if finite.any():
                    offsets[finite] = beeswarm(vals[finite])
                ax[1].plot(x + beewidth * offsets, vals, color=color, linestyle="none", marker="o", markersize=3, alpha=each_alpha)
                ax[1].plot(x + line_extent, np1 * np.nanmean(vals), color=color, linewidth=2.0)

        flat = np.concatenate([v[np.isfinite(v)] for v in all_vals]) if all_vals else np.array([0.0, 1.0])
        if flat.size == 0:
            flat = np.array([0.0, 1.0])
        ylo, yhi = float(np.min(flat)), float(np.max(flat))
        pad = 0.1 * (yhi - ylo + 1e-9)
        yline = 0
        for g, (mname, _) in enumerate(methods):
            group_ticks = xticks[g * n : (g + 1) * n]
            ax[1].annotate(
                "",
                xy=(group_ticks[0], yline),
                xytext=(group_ticks[-1], yline),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0),
                annotation_clip=False,
            )
            ax[1].text(np.mean(group_ticks), yline - 0.3 * pad, mname, fontsize=self.fontsize, ha="center", va="bottom")
        ax[1].set_xlim(-0.5, max(xticks) + 0.5)
        ax[1].set_ylim(yline - pad, yhi + pad)
        ax[1].set_ylabel("Power-law exponent")
        ax[1].axhline(4.0, color="0.8", linestyle="--", linewidth=1.0)
        ax[1].set_xticks(xticks, labels=keys_all * len(methods), rotation=45, ha="right")

        # --- ax[2]: five-point-derivative local-exponent curves (per mouse + bold average) ---
        for k in keys_all:
            la = local_alpha[k]
            color = _KEY_COLORS.get(k, "gray")
            xv = np.arange(la.shape[1]) + 1
            ax[2].plot(xv, la.T, color=color, alpha=0.2, linewidth=0.8)
            ax[2].plot(xv, np.nanmean(la, axis=0), color=color, linewidth=2.0, label=k)
        ax[2].axvspan(start + 1, end, color="0.8", alpha=0.4)
        ax[2].set_xscale("log")
        ax[2].set_xlabel("Shared Dimension")
        ax[2].set_ylabel("Local exponent")
        ax[2].set_ylim(-1, 10)
        ax[2].legend(loc="upper left", fontsize=self.fontsize, frameon=False)
        return fig


class SessionSpectraViewer(Viewer):
    """Interactive per-session view of every shared-variance spectrum on one axis.

    The single-session analogue of :class:`PlacefieldSpectraViewer`. Instead of showing one
    ``source_key`` spectrum across all mice, ax[0] overlays every available curve option
    (``ss_cv``, ``ss_direct``, ..., ``reg_covariances_fixed``) for a single selected session,
    each colored per :data:`_KEY_COLORS`. ax[1] and ax[2] are unchanged in structure but now
    estimate the power-law exponent from that one session's spectra (one point per curve in the
    beeswarm, one local-exponent curve per option).

    A ``session`` selection widget replaces ``source_key``; its options are the union of the
    ``session_ids`` of every provided aggregator (in first-seen order). For each curve the session
    is resolved against its own source aggregator (via :data:`SOURCE_OF_KEY`); a session missing
    from a source yields an all-NaN spectrum for that curve (it simply does not draw).

    Param-axis widgets, tuple-label encoding, and the log10 y-floor behave exactly as in
    :class:`PlacefieldSpectraViewer`.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        ylim_min: float = -5.5,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (9.0, 3.0),
    ):
        self.results = results
        self.results_cvpca = results_cvpca
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.fontsize = fontsize
        self.figsize = figsize

        # Session options: union of every aggregator's session_ids, first-seen order.
        sessions: list[str] = []
        for agg in self._agg.values():
            if agg is None:
                continue
            sessions.extend(uid for uid in agg.session_ids if uid not in sessions)
        self.add_selection("session", options=sessions, value=sessions[0] if sessions else None)

        # One widget per param-axis name, shared across sources (same scheme as PlacefieldSpectraViewer).
        merged_axes: dict[str, list] = {}
        for agg in self._agg.values():
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(opt for opt in options if opt not in existing)

        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for name, options in merged_axes.items():
            if any(isinstance(opt, tuple) for opt in options):
                label_map = {_tuple_label(opt): opt for opt in options}
                self._tuple_labels[name] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(name, options=widget_options)
            if name in _PREFERRED_DEFAULTS:
                default = self.encode_param(name, _PREFERRED_DEFAULTS[name])
                if default in widget_options:
                    self.update_selection(name, value=default)

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_boolean("normalize", value=True)
        self.add_integer_range("fit_range", value=(10, 20), min=1, max=200)
        self.add_integer("deriv_width", value=1, min=1, max=10)
        self.add_selection("smooth_kind", options=["none", "boxcar", "gaussian"], value="none")
        self.add_float("smooth_width", value=3.0, min=0.0, max=50.0, step=0.5)

    encode_param = PlacefieldSpectraViewer.encode_param
    _sel_params = PlacefieldSpectraViewer._sel_params
    _available_keys = PlacefieldSpectraViewer._available_keys

    def _spectrum(self, state: dict, key: str) -> np.ndarray:
        """Single-session ``(1, dims)`` spectrum for ``key``, normalized per ``state``.

        Returns an all-NaN row if the selected session is absent from ``key``'s source aggregator.
        """
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        arr = agg.sel(keys=[key], squeeze_ones=False, **self._sel_params(state, source))[key]
        arr = np.asarray(arr, dtype=float)
        sess_idx = agg._session_index.get(state["session"])
        if sess_idx is None:
            return np.full((1, arr.shape[-1]), np.nan)
        spec = np.atleast_2d(arr[sess_idx])
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, state["smooth_kind"], state["smooth_width"])

    def plot(self, state: dict):
        keys_all = self._available_keys()
        start, end = (int(v) for v in state["fit_range"])
        deriv_width = int(state["deriv_width"])
        each_alpha = 0.3
        ylim_min = state["ylim_min"]

        # Per-curve single-session spectra, local-exponent curves, and both exponent estimates.
        spectra = {k: self._spectrum(state, k) for k in keys_all}
        local_alpha = {k: _local_alpha_curve(spectra[k], deriv_width) for k in keys_all}
        decay_alpha = {k: _decay_alpha_per_mouse(spectra[k], start, end) for k in keys_all}
        deriv_alpha = {k: _deriv_alpha_per_mouse(local_alpha[k], start, end) for k in keys_all}

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(1, 3, figsize=self.figsize, layout="constrained", width_ratios=[1.0, 0.9, 0.9])

        # --- ax[0]: every curve option for the selected session (one line each) ---
        for k in keys_all:
            spec = spectra[k]
            spec_positive = np.where(spec > 0, spec, np.nan)
            color = _KEY_COLORS.get(k, "gray")
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, label=k, linewidth=1.5)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(bottom=10**ylim_min)
        yticks = ax[0].get_yticks()
        ytick_power = [np.log10(yt) for yt in yticks]
        ax[0].set_yticks(yticks, labels=ytick_power)
        ax[0].set_ylim(bottom=10**ylim_min)
        ax[0].set_xlabel("Shared Dimension")
        ax[0].set_ylabel("Variance")
        ax[0].set_title(str(state["session"]), fontsize=self.fontsize)
        ax[0].legend(loc="upper right", fontsize=self.fontsize, frameon=False)

        # --- ax[1]: per-session exponent, two method groups x each curve option ---
        methods = [("power-law fit", decay_alpha), ("5-pt deriv", deriv_alpha)]
        n = len(keys_all)
        xticks = []
        all_vals = []
        for g, (_, alphas_by_key) in enumerate(methods):
            for i, k in enumerate(keys_all):
                x = g * (n + 1) + i
                xticks.append(x)
                vals = alphas_by_key[k]
                all_vals.append(vals)
                color = _KEY_COLORS.get(k, "gray")
                ax[1].plot(np.full_like(vals, x), vals, color=color, linestyle="none", marker="o", markersize=4)

        flat = np.concatenate([v[np.isfinite(v)] for v in all_vals]) if all_vals else np.array([0.0, 1.0])
        if flat.size == 0:
            flat = np.array([0.0, 1.0])
        ylo, yhi = float(np.min(flat)), float(np.max(flat))
        pad = 0.1 * (yhi - ylo + 1e-9)
        yline = ylo - 2 * pad
        for g, (mname, _) in enumerate(methods):
            group_ticks = xticks[g * n : (g + 1) * n]
            ax[1].annotate(
                "",
                xy=(group_ticks[0], yline),
                xytext=(group_ticks[-1], yline),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0),
                annotation_clip=False,
            )
            ax[1].text(np.mean(group_ticks), yline - 0.3 * pad, mname, fontsize=self.fontsize, ha="center", va="top")
        ax[1].set_xlim(-0.5, max(xticks) + 0.5)
        ax[1].set_ylim(yline - pad, yhi + pad)
        ax[1].set_ylabel("Power-law exponent")
        ax[1].axhline(4.0, color="0.8", linestyle="--", linewidth=1.0)
        ax[1].set_xticks(xticks, labels=keys_all * len(methods), rotation=45, ha="right")

        # --- ax[2]: five-point-derivative local-exponent curves (one per curve option) ---
        for k in keys_all:
            la = local_alpha[k]
            color = _KEY_COLORS.get(k, "gray")
            xv = np.arange(la.shape[1]) + 1
            ax[2].plot(xv, la.T, color=color, linewidth=1.5, label=k)
        ax[2].axvspan(start + 1, end, color="0.8", alpha=0.4)
        ax[2].set_xscale("log")
        ax[2].set_xlabel("Shared Dimension")
        ax[2].set_ylabel("Local exponent")
        ax[2].set_ylim(-1, 10)
        ax[2].legend(loc="upper left", fontsize=self.fontsize, frameon=False)
        return fig


class SpectrumFigureViewer(Viewer):
    """Placefield-vs-full spectrum figure: spectra, power-law exponent, participation ratio.

    Three panels comparing one placefield (PF) spectrum against the full/functional (FF) spectrum:

    - ax[0]: the selected PF ``source_key`` spectrum (one of the four curve options) and the FF
      spectrum, both log-space pre-smoothed and drawn log-log (faint per-mouse lines + bold
      mouse-average). The FF curve source is set by ``full_source_key``: ``"SVD"`` is the ``ff`` key
      from the StimSpaceSpectra aggregator, ``"SVCA"`` is the subspace ``variance_activity`` key
      (``svca_subspace``, ``smooth_width=None``). The two fit windows are shaded.
    - ax[1]: the power-law exponent per mouse for PF (x=0) and FF (x=1), estimated either by a
      window log-log fit or by averaging the five-point-derivative local exponent, selected via
      ``alpha_method``. PF and FF use independent fit windows (``pf_fit_range`` / ``ff_fit_range``)
      because the FF tail sits at much higher rank.
    - ax[2]: the signed participation ratio per mouse for PF (x=0) and FF (x=1).

    Smoothing, param-axis widgets, tuple-label encoding, and the log10 y-floor behave as in
    :class:`PlacefieldSpectraViewer`.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        results_fit: ResultsAggregator | None = None,
        ylim_min: float = -5.5,
        ylim_max: float = 0.0,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (6.5, 3.0),
    ):
        self.results = results
        self.results_cvpca = results_cvpca
        self.results_subspace = results_subspace
        self.results_fit = results_fit
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.fontsize = fontsize
        self.figsize = figsize

        pf_options = list(_STIMSPACE_KEYS)
        if results_cvpca is not None:
            pf_options += list(_CVPCA_KEYS)
        self.add_selection("source_key", options=pf_options, value="ss_cv")

        # The "Reliable CA1 Spectrum" (FF) curve source: "SVD" is the StimSpaceSpectra ``ff`` key;
        # "SVCA" is the subspace ``variance_activity`` key (svca_subspace, smooth_width=None). SVCA
        # is only offered when a subspace aggregator is provided.
        full_options = ["SVD"] + (["SVCA"] if results_subspace is not None else [])
        self.add_selection("full_source_key", options=full_options, value="SVD")

        # Extra ax[0] overlays from the Tilbury-fit aggregator: the generalized-Gaussian (tilbury)
        # and plain-Gaussian (control) eigenvalue spectra. These are always at the fit's fixed
        # reliability/fraction-active threshold (:data:`_TILBURY_REL_FA`); a mismatch with the shared
        # ``reliability_fraction_active_thresholds`` selection is flagged in the ax[0] title.
        if results_fit is not None:
            self.add_multiple_selection("fit_key", options=list(_FIT_KEYS), value=[])

        # One widget per param-axis name, shared across sources (same scheme as PlacefieldSpectraViewer).
        merged_axes: dict[str, list] = {}
        for agg in self._agg.values():
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(opt for opt in options if opt not in existing)

        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for name, options in merged_axes.items():
            if any(isinstance(opt, tuple) for opt in options):
                label_map = {_tuple_label(opt): opt for opt in options}
                self._tuple_labels[name] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(name, options=widget_options)
            if name in _PREFERRED_DEFAULTS:
                default = self.encode_param(name, _PREFERRED_DEFAULTS[name])
                if default in widget_options:
                    self.update_selection(name, value=default)

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_float("ylim_max", value=ylim_max, min=-8.0, max=12.0, step=0.1)
        self.add_float("beewidth", value=0.15, min=0.0, max=1.0, step=0.01)
        self.add_boolean("normalize", value=True)
        # How the ax[1] exponent is estimated: a log-log window fit or the mean local FPD exponent.
        self.add_selection("alpha_method", options=["window", "5-pt deriv"], value="window")
        # Independent fit windows: the FF tail is at much higher rank than the PF tail.
        self.add_integer_range("pf_fit_range", value=(10, 20), min=1, max=500)
        self.add_integer_range("ff_fit_range", value=(50, 150), min=1, max=5000)
        self.add_integer("deriv_width", value=1, min=1, max=10)
        self.add_selection("smooth_kind", options=["none", "boxcar", "gaussian"], value="none")
        self.add_float("smooth_width", value=3.0, min=0.0, max=50.0, step=0.5)
        self.add_selection("yscale", options=["linear", "log"], value="log")
        self.add_float("each_line_alpha", value=0.3, min=0.0, max=1.0, step=0.01)

    encode_param = PlacefieldSpectraViewer.encode_param
    _sel_params = PlacefieldSpectraViewer._sel_params
    _spectrum = PlacefieldSpectraViewer._spectrum

    def _alpha_per_mouse(self, spec: np.ndarray, start: int, end: int, deriv_width: int, method: str) -> np.ndarray:
        """Per-mouse power-law exponent via ``method`` over ranks ``[start, end)``."""
        if method == "window":
            return _decay_alpha_per_mouse(spec, start, end)
        local = _local_alpha_curve(spec, deriv_width)
        return _deriv_alpha_per_mouse(local, start, end)

    def _ff_spectrum(self, state: dict) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` "Reliable CA1" spectrum, per ``state['full_source_key']``.

        ``"SVD"`` uses the StimSpaceSpectra ``ff`` key (via :meth:`_spectrum`). ``"SVCA"`` uses the
        subspace ``variance_activity`` key with ``subspace_name='svca_subspace'`` and
        ``smooth_width=None``; ``activity_parameters_name`` follows the shared widget. Both share the
        same normalize/log-space smoothing as every other spectrum.
        """
        if state.get("full_source_key", "SVD") != "SVCA":
            return self._spectrum(state, _FF_KEY)
        params = {"subspace_name": "svca_subspace", "smooth_width": None}
        if "activity_parameters_name" in state:
            params["activity_parameters_name"] = state["activity_parameters_name"]
        spec = self.results_subspace.sel(keys=["variance_activity"], avg_by_mouse=True, **params)["variance_activity"]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, state["smooth_kind"], state["smooth_width"])

    def _fit_spectrum(self, state: dict, key: str) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` Tilbury-fit eigenvalue spectrum for ``key``.

        ``key`` is ``eig_tilbury`` or ``eig_control`` from the Tilbury-fit aggregator. These spectra
        vary in length across sessions but are stored as ``"pad"`` keys, so the aggregator NaN-pads
        them to a common length and ``avg_by_mouse`` (nanmean) averages them directly. The aggregator
        has no selection axes (the fit is at the fixed :data:`_TILBURY_REL_FA` threshold). Normalize/
        log-space smoothing match every other spectrum.
        """
        spec = self.results_fit.sel(keys=[key], avg_by_mouse=True)[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, state["smooth_kind"], state["smooth_width"])

    def _rel_fa_matches_fit(self, state: dict) -> bool:
        """Whether the shared reliability/fraction-active selection equals the Tilbury-fit threshold.

        The Tilbury eig spectra are fixed at :data:`_TILBURY_REL_FA`; if the shared
        ``reliability_fraction_active_thresholds`` widget selects a different value the overlaid PF/FF
        curves are computed on a different neuron population. Returns True when there is no such shared
        widget (nothing to mismatch).
        """
        rel_axis = "reliability_fraction_active_thresholds"
        if rel_axis not in state:
            return True
        value = state[rel_axis]
        if rel_axis in self._tuple_labels:
            value = self._tuple_labels[rel_axis][value]
        return tuple(value) == _TILBURY_REL_FA

    def plot(self, state: dict):
        pf_key = state["source_key"]
        do_posthoc_scaling = pf_key == "ss_cvpca"
        deriv_width = int(state["deriv_width"])
        pf_start, pf_end = (int(v) for v in state["pf_fit_range"])
        ff_start, ff_end = (int(v) for v in state["ff_fit_range"])
        method = state["alpha_method"]
        each_alpha = state["each_line_alpha"]
        ylim_min = state["ylim_min"]
        ylim_max = state["ylim_max"]

        pf_spec = self._spectrum(state, pf_key)
        ff_spec = self._ff_spectrum(state)
        pf_color = "orange"
        ff_color = "black"
        pf_label = "Placefields"
        ff_label = "Reliable CA1"

        # Tilbury-fit overlays (PF-like spectra), placed between PF and CA1 in every panel.
        fit_keys = list(state.get("fit_key", []))
        fit_specs = {k: self._fit_spectrum(state, k) for k in fit_keys}

        if do_posthoc_scaling:
            num_dim_for_scaling = 5
            pf_spec_reference = self._spectrum(state, "ss_cv")
            ratio = pf_spec_reference[:, :num_dim_for_scaling] / pf_spec[:, :num_dim_for_scaling]
            scaling = np.nanmedian(ratio, axis=1)
            pf_spec *= scaling[:, np.newaxis]
            for k in fit_keys:
                fit_ratio = pf_spec_reference[:, :num_dim_for_scaling] / fit_specs[k][:, :num_dim_for_scaling]
                fit_specs[k] = fit_specs[k] * np.nanmedian(fit_ratio, axis=1)[:, np.newaxis]

        pf_alpha = self._alpha_per_mouse(pf_spec, pf_start, pf_end, deriv_width, method)
        ff_alpha = self._alpha_per_mouse(ff_spec, ff_start, ff_end, deriv_width, method)
        pf_pr = _signed_participation_ratio(pf_spec)
        ff_pr = _signed_participation_ratio(ff_spec)
        # Fit spectra are placefield-like, so estimate their exponent over the PF window.
        fit_alpha = {k: self._alpha_per_mouse(fit_specs[k], pf_start, pf_end, deriv_width, method) for k in fit_keys}
        fit_pr = {k: _signed_participation_ratio(fit_specs[k]) for k in fit_keys}

        plt.rcParams["font.size"] = self.fontsize
        fig, ax = plt.subplots(1, 3, figsize=self.figsize, layout="constrained", width_ratios=[1.0, 0.5, 0.5])

        # --- ax[0]: PF and FF spectra (faint per-mouse + bold average), fit windows shaded ---
        for spec, label, color, (start, end) in (
            (pf_spec, pf_label, pf_color, (pf_start, pf_end)),
            (ff_spec, ff_label, ff_color, (ff_start, ff_end)),
        ):
            spec_positive = np.where(spec > 0, spec, np.nan)
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(_xvals(spec), np.nanmean(spec_positive, axis=0), color=color, label=label, linewidth=2.0)
            ax[0].axvspan(start + 1, end, color=color, alpha=0.1)

        # --- ax[0] extra overlays: Tilbury-fit eig spectra (fixed rel/frac-active threshold) ---
        for key in fit_keys:
            spec = fit_specs[key]
            spec_positive = np.where(spec > 0, spec, np.nan)
            color = _FIT_KEY_COLORS.get(key, "gray")
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(_xvals(spec), np.nanmean(spec_positive, axis=0), color=color, label=_FIT_KEY_LABELS.get(key, key), linewidth=2.0)
        if fit_keys and not self._rel_fa_matches_fit(state):
            ax[0].set_title("REL-FA Not MATCHED!", fontsize=self.fontsize, color="red")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(10**ylim_min, 10**ylim_max)
        yticks = ax[0].get_yticks()
        ytick_power = [np.log10(yt) for yt in yticks]
        ax[0].set_yticks(yticks, labels=ytick_power)
        ax[0].set_ylim(10**ylim_min, 10**ylim_max)
        ax[0].set_xlabel("Shared Dimension")
        ax[0].set_ylabel("Variance")
        ax[0].legend(loc="upper right", fontsize=self.fontsize, frameon=False, markerfirst=False)
        xlim = ax[0].get_xlim()
        format_spines(
            ax[0],
            x_pos=-0.02,
            y_pos=-0.02,
            spines_visible=["left", "bottom"],
            xbounds=[1, xlim[1]],
            ybounds=[10**ylim_min, 10**ylim_max],
        )

        # Beeswarm groups: PF (x=0), each selected fit overlay, then CA1 last (x=1..n+1).
        beeswarm_colors = [pf_color] + [_FIT_KEY_COLORS.get(k, "gray") for k in fit_keys] + [ff_color]
        beeswarm_labels = ["PF"] + [_FIT_KEY_LABELS.get(k, k) for k in fit_keys] + ["CA1"]
        alpha_values = [pf_alpha] + [fit_alpha[k] for k in fit_keys] + [ff_alpha]
        pr_values = [pf_pr] + [fit_pr[k] for k in fit_keys] + [ff_pr]

        # --- ax[1]: per-mouse power-law exponent, PF / fits / CA1 ---
        _beeswarm_panel(ax[1], alpha_values, beeswarm_colors, beeswarm_labels, self.fontsize, state["beewidth"])
        ax[1].set_ylabel(f"Power-law exponent")

        # --- ax[2]: signed participation ratio, PF / fits / CA1 ---
        _beeswarm_panel(ax[2], pr_values, beeswarm_colors, beeswarm_labels, self.fontsize, state["beewidth"], yscale=state["yscale"])
        ax[2].set_ylabel("Dimensionality (Participation Ratio)")
        return fig


def placefield_spectra(
    results: ResultsAggregator,
    results_cvpca: ResultsAggregator | None = None,
    source_key: str | list[str] = "ss_cv",
    ylim_min: float = -5.5,
    normalize: bool = True,
    fit_range: tuple[int, int] = (10, 20),
    deriv_width: int = 1,
    smooth_kind: str = "none",
    smooth_width: float = 3.0,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (9.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Shared-variance spectrum figure with power-law exponent estimates.

    Three panels: ax[0] the ``source_key`` example spectrum on log-log axes (one faint line per
    mouse, bold mouse-average); ax[1] a per-mouse beeswarm of the power-law exponent estimated
    over ranks ``[start, end)`` for every curve option, grouped by method (log-log ``power-law
    fit`` vs the mean five-point-derivative ``5-pt deriv``); ax[2] the full per-rank
    five-point-derivative local-exponent curves for every curve option (with the ``[start, end)``
    window shaded). Which aggregator each key comes from is resolved via :data:`SOURCE_OF_KEY`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results, source of the ``reg_covariances_fixed`` key. Required if
        ``source_key`` names a CVPCA key; if None only StimSpace keys are available.
    source_key : str or list of str
        Which spectrum/spectra to overlay in ax[0]. A single key or list drawn from
        ``ss_cv``/``ss_direct`` (from ``results``) and ``reg_covariances_fixed`` (from
        ``results_cvpca``); see :data:`SOURCE_OF_KEY`. Each is colored per :data:`_KEY_COLORS`. The
        exponent panels always cover all available curve options regardless of this choice.
    ylim_min : float
        Lower y-limit of the spectrum panel in log10 units; the applied floor is ``10 ** ylim_min``.
        The upper limit is autoscaled to the data.
    normalize : bool
        If True, normalize the spectrum by the sum of the spectrum.
    fit_range : tuple[int, int]
        0-based ``[start, end)`` rank window the exponent is estimated over (both methods).
    deriv_width : int
        Stencil half-width for the five-point-derivative local exponent.
    smooth_kind : {"none", "boxcar", "gaussian"}
        Log-space (geometric-mean) pre-smoothing applied to each spectrum before both exponent
        fits. ``"none"`` disables smoothing.
    smooth_width : float
        Boxcar full-width in rank units; the Gaussian uses ``sigma = smooth_width / 2``.
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
        Overrides for the parameter-axis selections, keyed by raw ``param_axes`` name (e.g.
        ``activity_parameters_name``, ``include_iti``, ``smooth_widths``). Each key must be a
        ``param_axes`` name of at least one provided aggregator; it applies to whichever source(s)
        have that axis. Tuple-valued axes (e.g. ``smooth_widths=(5.0, None)``) are passed as native
        tuples; they are encoded to the widget's string labels internally.

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldSpectraViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    source_keys = [source_key] if isinstance(source_key, str) else list(source_key)
    for sk in source_keys:
        if sk not in SOURCE_OF_KEY:
            raise ValueError(f"Unknown source_key {sk!r}. Options: {list(SOURCE_OF_KEY)}")
        if SOURCE_OF_KEY[sk] == "cvpca" and results_cvpca is None:
            raise ValueError(f"source_key {sk!r} is a CVPCA key but results_cvpca was not provided.")

    viewer = PlacefieldSpectraViewer(results, results_cvpca=results_cvpca, ylim_min=ylim_min, fontsize=fontsize, figsize=figsize)
    viewer.update_multiple_selection("source_key", value=source_keys)

    valid_selections = set()
    for agg in viewer._agg.values():
        if agg is None:
            continue
        valid_selections.update(agg.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))

    viewer.update_float("ylim_min", value=ylim_min)
    viewer.update_boolean("normalize", value=normalize)
    viewer.update_integer_range("fit_range", value=tuple(fit_range))
    viewer.update_integer("deriv_width", value=deriv_width)
    viewer.update_selection("smooth_kind", value=smooth_kind)
    viewer.update_float("smooth_width", value=smooth_width)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


def session_spectra(
    results: ResultsAggregator,
    results_cvpca: ResultsAggregator | None = None,
    session: str | None = None,
    ylim_min: float = -5.5,
    normalize: bool = True,
    fit_range: tuple[int, int] = (10, 20),
    deriv_width: int = 1,
    smooth_kind: str = "none",
    smooth_width: float = 3.0,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (9.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Per-session view of every shared-variance spectrum with power-law exponent estimates.

    The single-session analogue of :func:`placefield_spectra`. Three panels: ax[0] overlays every
    curve option (``ss_cv``, ``ss_direct``, ..., ``reg_covariances_fixed``) for one session on
    log-log axes, colored per :data:`_KEY_COLORS`; ax[1] a beeswarm of the power-law exponent for
    each curve (one point per curve, both estimation methods) over ranks ``[start, end)``; ax[2] the
    per-rank five-point-derivative local-exponent curve for each option (with the window shaded).
    Which aggregator each key comes from is resolved via :data:`SOURCE_OF_KEY`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results, source of the ``reg_covariances_fixed`` key. If None only
        StimSpace keys are drawn.
    session : str or None
        session_uid to show. Must be a session of at least one provided aggregator. If None, the
        first session (union of the aggregators' ``session_ids``) is used.
    ylim_min : float
        Lower y-limit of the spectrum panel in log10 units; the applied floor is ``10 ** ylim_min``.
    normalize : bool
        If True, normalize each spectrum by its sum.
    fit_range : tuple[int, int]
        0-based ``[start, end)`` rank window the exponent is estimated over (both methods).
    deriv_width : int
        Stencil half-width for the five-point-derivative local exponent.
    smooth_kind : {"none", "boxcar", "gaussian"}
        Log-space (geometric-mean) pre-smoothing applied to each spectrum before both exponent
        fits. ``"none"`` disables smoothing.
    smooth_width : float
        Boxcar full-width in rank units; the Gaussian uses ``sigma = smooth_width / 2``.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections, keyed by raw ``param_axes`` name. See
        :func:`placefield_spectra`.

    Returns
    -------
    matplotlib.figure.Figure or SessionSpectraViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = SessionSpectraViewer(results, results_cvpca=results_cvpca, ylim_min=ylim_min, fontsize=fontsize, figsize=figsize)
    if session is not None:
        viewer.update_selection("session", value=session)

    valid_selections = set()
    for agg in viewer._agg.values():
        if agg is None:
            continue
        valid_selections.update(agg.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))

    viewer.update_float("ylim_min", value=ylim_min)
    viewer.update_boolean("normalize", value=normalize)
    viewer.update_integer_range("fit_range", value=tuple(fit_range))
    viewer.update_integer("deriv_width", value=deriv_width)
    viewer.update_selection("smooth_kind", value=smooth_kind)
    viewer.update_float("smooth_width", value=smooth_width)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


def spectrum_figure(
    results: ResultsAggregator,
    results_cvpca: ResultsAggregator | None = None,
    results_subspace: ResultsAggregator | None = None,
    results_fit: ResultsAggregator | None = None,
    source_key: str = "ss_cv",
    full_source_key: str = "SVD",
    fit_key: str | list[str] = (),
    ylim_min: float = -5.5,
    ylim_max: float = 0.0,
    beewidth: float = 0.15,
    normalize: bool = True,
    alpha_method: str = "window",
    pf_fit_range: tuple[int, int] = (10, 20),
    ff_fit_range: tuple[int, int] = (50, 150),
    deriv_width: int = 1,
    smooth_kind: str = "none",
    smooth_width: float = 3.0,
    fontsize: float = 9.0,
    yscale: str = "log",
    figsize: tuple[float, float] = (6.5, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Placefield-vs-full spectrum figure: spectra, power-law exponent, participation ratio.

    Three panels: ax[0] the selected PF ``source_key`` spectrum and the FF (``ff``) spectrum on
    log-log axes (faint per-mouse + bold mouse-average, both log-space pre-smoothed, fit windows
    shaded); ax[1] the per-mouse power-law exponent for PF (x=0) and FF (x=1) via ``alpha_method``
    over independent windows; ax[2] the per-mouse signed participation ratio for PF and FF.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*`` and ``ff`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results, source of the ``reg_covariances_fixed`` key. Required if
        ``source_key`` is that CVPCA key; if None only StimSpace PF keys are selectable.
    results_subspace : ResultsAggregator or None
        Aggregated SubspaceConfig results, source of the ``variance_activity`` key used when
        ``full_source_key="SVCA"``. Required for that option; if None only ``"SVD"`` is selectable.
    results_fit : ResultsAggregator or None
        Aggregated TilburyFitConfig results, source of the ``eig_tilbury``/``eig_control`` overlays
        selected by ``fit_key``. Required for those overlays; if None ``fit_key`` must be empty.
    source_key : str
        Which PF spectrum to show in ax[0]. One of ``ss_cv``/``ss_direct``/``ss_cvpca`` (from
        ``results``) or ``reg_covariances_fixed`` (from ``results_cvpca``).
    full_source_key : {"SVD", "SVCA"}
        Source of the FF ("Reliable CA1 Spectrum") curve. ``"SVD"`` uses the StimSpaceSpectra ``ff``
        key. ``"SVCA"`` uses the subspace ``variance_activity`` key with
        ``subspace_name='svca_subspace'`` and ``smooth_width=None`` (``activity_parameters_name``
        follows the shared selection), and requires ``results_subspace``.
    fit_key : str or list of str
        Extra ax[0] overlays from the Tilbury-fit aggregator: any of ``eig_tilbury`` (blue) and
        ``eig_control`` (green). These are always at the fit's fixed reliability/fraction-active
        threshold ``(0.3, 0.1)``; if the shared ``reliability_fraction_active_thresholds`` selection
        differs, ax[0] is titled ``"REL-FA Not MATCHED!"``. Requires ``results_fit``.
    ylim_min : float
        Lower y-limit of the spectrum panel in log10 units; the applied floor is ``10 ** ylim_min``.
    ylim_max : float
        Upper y-limit of the spectrum panel in log10 units; the applied ceiling is ``10 ** ylim_max``.
    beewidth : float
        Width of the beeswarm points in ax[1] and ax[2], in x-axis units.
    normalize : bool
        If True, normalize each spectrum by its sum (does not affect the participation ratio).
    alpha_method : {"window", "5-pt deriv"}
        How the ax[1] exponent is estimated: a log-log window fit, or the mean of the five-point
        derivative local exponent over the window.
    pf_fit_range : tuple[int, int]
        0-based ``[start, end)`` rank window for the PF exponent.
    ff_fit_range : tuple[int, int]
        0-based ``[start, end)`` rank window for the FF exponent (typically higher rank).
    deriv_width : int
        Stencil half-width for the five-point-derivative local exponent (``alpha_method="5-pt deriv"``).
    smooth_kind : {"none", "boxcar", "gaussian"}
        Log-space (geometric-mean) pre-smoothing applied to both spectra before fitting.
    smooth_width : float
        Boxcar full-width in rank units; the Gaussian uses ``sigma = smooth_width / 2``.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    yscale : {"linear", "log"}
        y-axis scale for the participation-ratio panel (ax[2]).
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the parameter-axis selections, keyed by raw ``param_axes`` name. See
        :func:`placefield_spectra`.

    Returns
    -------
    matplotlib.figure.Figure or SpectrumFigureViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    pf_options = list(_STIMSPACE_KEYS) + (list(_CVPCA_KEYS) if results_cvpca is not None else [])
    if source_key not in pf_options:
        raise ValueError(f"Unknown PF source_key {source_key!r}. Options: {pf_options}")
    full_options = ["SVD"] + (["SVCA"] if results_subspace is not None else [])
    if full_source_key not in full_options:
        raise ValueError(f"Unknown full_source_key {full_source_key!r}. Options: {full_options}")
    fit_keys = [fit_key] if isinstance(fit_key, str) else list(fit_key)
    if fit_keys and results_fit is None:
        raise ValueError("fit_key requires results_fit to be provided.")
    for fk in fit_keys:
        if fk not in _FIT_KEYS:
            raise ValueError(f"Unknown fit_key {fk!r}. Options: {_FIT_KEYS}")

    viewer = SpectrumFigureViewer(
        results,
        results_cvpca=results_cvpca,
        results_subspace=results_subspace,
        results_fit=results_fit,
        ylim_min=ylim_min,
        ylim_max=ylim_max,
        fontsize=fontsize,
        figsize=figsize,
    )
    viewer.update_selection("source_key", value=source_key)
    viewer.update_selection("full_source_key", value=full_source_key)
    if results_fit is not None:
        viewer.update_multiple_selection("fit_key", value=fit_keys)

    valid_selections = set()
    for agg in viewer._agg.values():
        if agg is None:
            continue
        valid_selections.update(agg.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))

    viewer.update_float("ylim_min", value=ylim_min)
    viewer.update_float("ylim_max", value=ylim_max)
    viewer.update_boolean("normalize", value=normalize)
    viewer.update_selection("alpha_method", value=alpha_method)
    viewer.update_integer_range("pf_fit_range", value=tuple(pf_fit_range))
    viewer.update_integer_range("ff_fit_range", value=tuple(ff_fit_range))
    viewer.update_integer("deriv_width", value=deriv_width)
    viewer.update_selection("smooth_kind", value=smooth_kind)
    viewer.update_float("smooth_width", value=smooth_width)
    viewer.update_float("beewidth", value=beewidth)
    viewer.update_selection("yscale", value=yscale)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


# Fixed colors for the Tilbury-fit panels: generalized (Tilbury) vs plain-Gaussian control.
_GENERALIZED_COLOR = "blue"
_GAUSSIAN_COLOR = "black"


def _short_mouse_name(name: str) -> str:
    """Shorten ``CR_Hippocannula*`` mouse names to ``CR*``; other names unchanged."""
    prefix = "CR_Hippocannula"
    if name.startswith(prefix):
        return "CR" + name[len(prefix) :]
    return name


class PlacefieldExampleFitViewer(Viewer):
    """Tilbury generalized-Gaussian placefield fits: grid of example single-neuron fits.

    An ``n_rows x n_cols`` grid of example single-neuron fits from one session (the top neurons by
    test R^2 that also clear the improvement threshold). Each panel overlays the held-out test
    placefield (points) against the fitted generalized-Gaussian (Tilbury) and plain-Gaussian control
    curves. The stored :class:`~dimensionality_manuscript.configs.tilbury_fit.TilburyFitConfig`
    results already hold the fitted parameters and R^2; only the held-out test curve is not stored,
    so it is rebuilt from the deterministic train/test split and trial-averaging (no re-fit).

    The population summaries live in the separate :class:`PlacefieldPopulationViewer`.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        registry: PopulationRegistry,
        n_rows: int = 1,
        n_cols: int = 3,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (8.0, 3.0),
    ):
        self.results = results
        self.registry = registry
        self.config = results.config_class
        self.fontsize = fontsize
        self.figsize = figsize
        # Rebuilding the test curve is cheap (deterministic trial-average), but cache by session_uid
        # so switching back to a session in the viewer is instant.
        self._fit_cache: dict[str, dict] = {}

        self.add_selection("example_session", options=list(results.session_ids), value=results.session_ids[0])
        self.add_integer("n_rows", value=n_rows, min=1, max=6)
        self.add_integer("n_cols", value=n_cols, min=1, max=6)
        # Example neurons are drawn at random from those with generalized test R2 above r2_threshold
        # AND (generalized - gaussian) test R2 above improvement_threshold (so the example both fits
        # well and beats the plain Gaussian). The seed makes the draw reproducible; if too few clear
        # both thresholds the extra panels are left empty.
        self.add_float("r2_threshold", value=0.5, min=-1.0, max=1.0, step=0.05)
        self.add_float("improvement_threshold", value=0.0, min=0.0, max=1.0, step=0.01)
        self.add_integer("random_seed", value=0, min=0, max=100000)
        self.add_boolean("normalize_curves", value=True)

    def _example_fit(self, session_uid: str) -> dict:
        """Return the (cached) example fit for ``session_uid``, loading it on a miss."""
        if session_uid not in self._fit_cache:
            self._fit_cache[session_uid] = self._load_example_fit(session_uid)
        return self._fit_cache[session_uid]

    def _load_example_fit(self, session_uid: str) -> dict:
        """Assemble one session's example fit from stored results plus a rebuilt test curve.

        The fitted parameters and R^2 come straight from the aggregated
        :class:`TilburyFitConfig` results (no gradient descent). The held-out test placefield is
        not stored, so it is rebuilt with the same deterministic split (``registry.time_split``) and
        trial-averaging (``_avg_placefield``) the fit used; ``best_env``, the bin edges and the
        dropped-bin mask are recomputed exactly as :meth:`TilburyFitConfig.process` does.

        Returns
        -------
        dict
            ``theta`` (P,), ``test_curve`` (n_kept, P), ``params`` (n_kept, 6),
            ``params_control`` (n_kept, 4), ``r2_test`` (n_kept,), ``r2_test_control`` (n_kept,),
            aligned so panel ``n`` uses row ``n`` of each.
        """
        config = self.config
        idx = self.results._session_index[session_uid]
        session = self.results.sessions[idx]

        sel = self.results.sel(
            keys=["params", "params_control", "r2_test", "r2_test_control", "idx_keep"],
            load_ragged=True,
            squeeze_ones=False,
        )
        idx_keep = sel["idx_keep"][idx]  # (N_total,) bool; kept neurons, in order
        n_kept = int(np.sum(idx_keep))
        # Stored per-neuron arrays are padded with NaN to the max neuron count; kept rows are the
        # first n_kept, in the same order as idx_keep selects them below.
        params = sel["params"][idx][:n_kept]
        params_control = sel["params_control"][idx][:n_kept]
        r2_test = sel["r2_test"][idx][:n_kept]
        r2_test_control = sel["r2_test_control"][idx][:n_kept]

        # Original ROI indices that entered the fit: population.idx_neurons (the AND of
        # session.idx_rois across all spks_types), NOT the current-spks_type session.idx_rois.
        # idx_keep indexes into this array, so idx_neurons[j] recovers a neuron's original index.
        population, _ = self.registry.get_population(session, config.spks_type)
        idx_neurons = np.asarray(population.idx_neurons)

        # Recompute the fit's fixed choices (all "skip"ped from storage, all deterministic).
        num_per_env = {i: int(np.sum(session.trial_environment == i)) for i in session.environments}
        best_env = max(num_per_env, key=num_per_env.get)
        dist_edges = np.linspace(0, session.env_length[0], config.num_bins + 1)
        dist_centers = edge2center(dist_edges)

        # Trial-average every split's placefield over the kept neurons; the counts give the
        # dropped-bin mask (bins empty in any split) so theta matches the stored params' support.
        spks, fb = config._get_split_data(session, self.registry)
        for s in _SPLITS:
            spks[s] = spks[s][:, idx_keep]
        curves, counts = {}, {}
        for s in _SPLITS:
            curves[s], counts[s] = config._avg_placefield(spks[s], fb[s], dist_edges, best_env, session)
        bad = np.zeros(config.num_bins, dtype=bool)
        for s in _SPLITS:
            bad |= counts[s] == 0
        good = ~bad

        return {
            "theta": dist_centers[good],
            "test_curve": curves["test"][:, good],
            "params": params,
            "params_control": params_control,
            "r2_test": r2_test,
            "r2_test_control": r2_test_control,
            "idx_keep": idx_keep,
            "idx_neurons": idx_neurons,
        }

    def plot(self, state: dict):
        def _optional_normalization(curve: np.ndarray) -> np.ndarray:
            if state["normalize_curves"]:
                sum = np.nansum(curve)
                if sum == 0:
                    return curve
                return curve / np.nansum(curve)
            return curve

        n_rows = int(state["n_rows"])
        n_cols = int(state["n_cols"])
        fit = self._example_fit(state["example_session"])

        plt.rcParams["font.size"] = self.fontsize
        fig = plt.figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(n_rows, n_cols)

        theta = fit["theta"]
        n_show = n_rows * n_cols
        r2 = fit["r2_test"]
        r2c = fit["r2_test_control"]
        # Well-fit by the generalized model AND beating the plain Gaussian by improvement_threshold.
        eligible = np.flatnonzero(np.isfinite(r2) & np.isfinite(r2c) & (r2 > state["r2_threshold"]) & (r2 - r2c > state["improvement_threshold"]))
        rng = np.random.default_rng(int(state["random_seed"]))
        # Random draw without replacement; if too few clear the threshold, extra panels stay empty.
        chosen = rng.choice(eligible, size=n_show, replace=False) if eligible.size > n_show else eligible
        share_ax = None
        for cell in range(n_show):
            r, c = divmod(cell, n_cols)
            # Share x (common position axis) but not y: each neuron gets its own optimal y-range.
            ax = fig.add_subplot(gs[r, c], sharex=share_ax)
            share_ax = share_ax or ax
            if r == n_rows - 1:
                ax.set_xlabel("Position (cm)")
            if c == 0:
                ax.set_ylabel("Activity")
            if cell >= len(chosen):
                continue  # not enough eligible neurons -> leave this panel empty
            n = chosen[cell]

            # Original ROI index of this neuron: n indexes the kept arrays, np.where(idx_keep) maps
            # it to a row of idx_neurons (population.idx_neurons), which already holds original indices.
            idx_within_fit_neurons = np.where(fit["idx_keep"])[0][n]
            idx_within_idx_rois = fit["idx_neurons"][idx_within_fit_neurons]

            first = cell == 0
            ax.plot(
                theta,
                _optional_normalization(fit["test_curve"][n]),
                "o",
                color="red",
                ms=2.5,
                alpha=0.5,
                label="Test data" if first else None,
            )
            ax.plot(
                theta,
                _optional_normalization(_eval_tilbury(theta, fit["params"][n])),
                "-",
                color=_GENERALIZED_COLOR,
                lw=1.5,
                label="Generalized" if first else None,
            )
            ax.plot(
                theta,
                _optional_normalization(_eval_gaussian(theta, fit["params_control"][n])),
                "-",
                color=_GAUSSIAN_COLOR,
                lw=1.5,
                label="Gaussian" if first else None,
            )
            ax.set_title(f"{state['example_session']} | Neuron: {idx_within_idx_rois} | R²={fit['r2_test'][n]:.2f}", fontsize=self.fontsize)
            if first:
                ax.legend(fontsize=self.fontsize * 0.8, frameon=False, loc="upper right")
        return fig


# Normalization presets for PlacefieldFitFigureViewer: each maps a curve to the scalar its trio
# (test data + both fits) is divided by, computed on the *test-data* curve so the fits stay overlaid
# on the data while every panel shares a common scale (needed for sharey).
_FIT_FIGURE_NORMALIZATIONS = ("std", "sum", "max", "none")


def _fit_figure_scale(ref: np.ndarray, method: str) -> float:
    """Scalar to divide a curve trio by, from the reference (test-data) curve.

    ``method`` is one of :data:`_FIT_FIGURE_NORMALIZATIONS`. Returns ``1.0`` when the statistic is
    non-finite or non-positive (flat / empty curve) so normalization is a no-op instead of blowing up.
    """
    if method == "none":
        return 1.0
    if method == "std":
        s = float(np.nanstd(ref))
    elif method == "sum":
        s = float(np.nansum(ref))
    elif method == "max":
        s = float(np.nanmax(ref))
    else:
        raise ValueError(f"Unknown normalization {method!r}. Options: {list(_FIT_FIGURE_NORMALIZATIONS)}")
    return s if np.isfinite(s) and s > 0 else 1.0


class PlacefieldFitFigureViewer(Viewer):
    """Tilbury placefield fits for a hand-picked list of (session, neuron) examples.

    Unlike :class:`PlacefieldExampleFitViewer` (which *draws* well-fit neurons at random from one
    session), this viewer plots an explicit, ordered list of neurons the user selected by eye — each
    identified by its session_uid and its **original ROI index** (the index into the session's full
    ROI set, i.e. ``np.where(session.idx_rois)[0][k]``). That original index is the stable identifier
    to write down for a figure: it survives regardless of how many neurons cleared the reliability /
    fraction-active thresholds in :class:`TilburyFitConfig`.

    Each requested ROI is traced back to its fit: it must be one of the pipeline's available neurons
    (``population.idx_neurons``) *and* have been kept by the fit's inclusion thresholds
    (``idx_keep``). A neuron that was never available or was dropped before fitting raises (opt-in via
    ``strict``) or is flagged with an empty, titled panel.

    The first ``n_rows * n_cols`` entries of the list are plotted, in order, into a shared-axes grid
    (``sharex``/``sharey``); each panel overlays the held-out test placefield (points) against the
    fitted generalized-Gaussian (Tilbury) and plain-Gaussian control curves. The whole trio in a
    panel is normalized by the test-data curve's statistic (``normalize``: std / sum / max / none),
    so the fits stay overlaid on the data while panels remain comparable under ``sharey``.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        registry: PopulationRegistry,
        session_uids: list[str],
        neurons: list[int],
        n_rows: int = 2,
        n_cols: int = 3,
        normalize: str = "std",
        normalize_independent: bool = False,
        strict: bool = True,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (8.0, 4.0),
    ):
        if len(session_uids) != len(neurons):
            raise ValueError(f"session_uids and neurons must be the same length, got {len(session_uids)} and {len(neurons)}.")
        if normalize not in _FIT_FIGURE_NORMALIZATIONS:
            raise ValueError(f"Unknown normalize {normalize!r}. Options: {list(_FIT_FIGURE_NORMALIZATIONS)}")

        self.results = results
        self.registry = registry
        self.config = results.config_class
        self.session_uids = list(session_uids)
        self.neurons = [int(n) for n in neurons]
        self.strict = strict
        self.fontsize = fontsize
        self.figsize = figsize
        # Rebuilding a session's test curves is cheap but cache by session_uid so the same session
        # appearing for several requested neurons is only loaded once.
        self._fit_cache: dict[str, dict] = {}

        self.add_integer("n_rows", value=n_rows, min=1, max=8)
        self.add_integer("n_cols", value=n_cols, min=1, max=8)
        self.add_selection("normalize", options=list(_FIT_FIGURE_NORMALIZATIONS), value=normalize)
        # normalize_independent: scale each of the three curves (test data, generalized, gaussian) by
        # its own statistic (shape-only comparison), instead of the whole trio by the test-data curve.
        self.add_boolean("normalize_independent", value=normalize_independent)

    def _session_fit(self, session_uid: str) -> dict:
        """Return the (cached) per-session fit bundle for ``session_uid``, loading it on a miss."""
        if session_uid not in self._fit_cache:
            self._fit_cache[session_uid] = self._load_session_fit(session_uid)
        return self._fit_cache[session_uid]

    def _load_session_fit(self, session_uid: str) -> dict:
        """Assemble one session's fit bundle: kept-neuron fits, rebuilt test curves, and the
        original-ROI-index -> kept-row map needed to resolve a hand-picked neuron.

        Fitted parameters and R^2 come straight from the stored :class:`TilburyFitConfig` results; the
        held-out test placefield is rebuilt with the same deterministic split and trial-averaging the
        fit used (no re-fit), exactly as :meth:`PlacefieldExampleFitViewer._load_example_fit` does.

        Returns
        -------
        dict
            ``theta`` (P,), ``test_curve`` (n_kept, P), ``params`` (n_kept, 6),
            ``params_control`` (n_kept, 4), ``r2_test`` (n_kept,), ``r2_test_control`` (n_kept,),
            ``idx_neurons`` (N_available,) original ROI indices that entered the pipeline, and
            ``idx_keep`` (N_available,) bool mask of which of those were fitted (kept rows are the
            finite prefix of the per-neuron arrays, in ``idx_neurons`` order).
        """
        if session_uid not in self.results._session_index:
            raise KeyError(f"session_uid {session_uid!r} not in results (options: {list(self.results.session_ids)}).")

        config = self.config
        idx = self.results._session_index[session_uid]
        session = self.results.sessions[idx]

        sel = self.results.sel(
            keys=["params", "params_control", "r2_test", "r2_test_control", "idx_keep"],
            load_ragged=True,
            squeeze_ones=False,
        )
        idx_keep = np.asarray(sel["idx_keep"][idx], dtype=bool)  # (N_available,) over population.idx_neurons
        n_kept = int(np.sum(idx_keep))
        params = sel["params"][idx][:n_kept]
        params_control = sel["params_control"][idx][:n_kept]
        r2_test = sel["r2_test"][idx][:n_kept]
        r2_test_control = sel["r2_test_control"][idx][:n_kept]

        # Original ROI indices that entered the fit (population.idx_neurons), used to map a
        # hand-picked original index -> row of idx_keep. Same registry/spks_type as the fit.
        population, _ = self.registry.get_population(session, config.spks_type)
        idx_neurons = np.asarray(population.idx_neurons)

        # Recompute the fit's deterministic choices and rebuild each split's trial-averaged curve.
        num_per_env = {i: int(np.sum(session.trial_environment == i)) for i in session.environments}
        best_env = max(num_per_env, key=num_per_env.get)
        dist_edges = np.linspace(0, session.env_length[0], config.num_bins + 1)
        dist_centers = edge2center(dist_edges)

        spks, fb = config._get_split_data(session, self.registry)
        for s in _SPLITS:
            spks[s] = spks[s][:, idx_keep]
        curves, counts = {}, {}
        for s in _SPLITS:
            curves[s], counts[s] = config._avg_placefield(spks[s], fb[s], dist_edges, best_env, session)
        bad = np.zeros(config.num_bins, dtype=bool)
        for s in _SPLITS:
            bad |= counts[s] == 0
        good = ~bad

        return {
            "theta": dist_centers[good],
            "test_curve": curves["test"][:, good],
            "params": params,
            "params_control": params_control,
            "r2_test": r2_test,
            "r2_test_control": r2_test_control,
            "idx_neurons": idx_neurons,
            "idx_keep": idx_keep,
            "session": session,
        }

    def _resolve(self, session_uid: str, roi: int) -> tuple[dict, Optional[int], str]:
        """Map a hand-picked ``(session_uid, original ROI index)`` to its kept-row in the fit bundle.

        Returns ``(fit, kept_row, status)`` where ``status`` is ``"ok"`` (``kept_row`` is the row of
        ``params`` / ``test_curve`` for this neuron), ``"not_available"`` (ROI never entered the
        pipeline — silent / filtered out), or ``"not_fit"`` (available but dropped by the reliability
        / fraction-active thresholds). ``kept_row`` is ``None`` for the two failure statuses.
        """
        fit = self._session_fit(session_uid)
        idx_neurons = fit["idx_neurons"]
        pos = np.flatnonzero(idx_neurons == roi)
        if pos.size == 0:
            return fit, None, "not_available"
        j = int(pos[0])
        if not fit["idx_keep"][j]:
            return fit, None, "not_fit"
        kept_row = int(np.sum(fit["idx_keep"][:j]))
        return fit, kept_row, "ok"

    def plot(self, state: dict):
        n_rows = int(state["n_rows"])
        n_cols = int(state["n_cols"])
        method = state["normalize"]
        independent = bool(state["normalize_independent"])
        n_show = n_rows * n_cols

        plt.rcParams["font.size"] = self.fontsize
        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.figsize, squeeze=False, layout="constrained")

        ylims = {}
        for cell in range(n_show):
            r, c = divmod(cell, n_cols)
            ax = axs[r, c]
            if r == n_rows - 1:
                ax.set_xlabel("Position (cm)")
            if c == 0:
                ax.set_ylabel("Activity")
            if cell >= len(self.session_uids):
                ax.set_visible(False)  # fewer requested neurons than panels -> hide the extras
                continue

            session_uid, roi = self.session_uids[cell], self.neurons[cell]
            fit, kept_row, status = self._resolve(session_uid, roi)
            if status != "ok":
                # Traced but not fittable: flag loudly (strict) or leave a titled empty panel.
                if self.strict:
                    raise ValueError(f"Neuron roi={roi} in session {session_uid!r} is '{status}' (not a fitted neuron).")
                ax.set_title(f"{session_uid}\nroi {roi}: {status}", fontsize=self.fontsize * 0.8, color="red")
                continue

            theta = fit["theta"]
            data = fit["test_curve"][kept_row]
            gen = _eval_tilbury(theta, fit["params"][kept_row])
            gauss = _eval_gaussian(theta, fit["params_control"][kept_row])
            # normalize_independent: each curve divided by its own statistic (shape-only). Otherwise
            # the whole trio shares the test-data curve's scale, keeping the fits overlaid on the data.
            if independent:
                data = data / _fit_figure_scale(data, method)
                gen = gen / _fit_figure_scale(gen, method)
                gauss = gauss / _fit_figure_scale(gauss, method)
            else:
                scale = _fit_figure_scale(data, method)
                data, gen, gauss = data / scale, gen / scale, gauss / scale

            first = cell == 0
            last = cell == n_show - 1
            ax.plot(theta, data, "o", color="gray", ms=2.5, alpha=0.5, label="Test data" if first else None)
            ax.plot(theta, gen, "-", color=_GENERALIZED_COLOR, lw=1.5, label="Generalized" if first else None)
            ax.plot(theta, gauss, "-", color=_GAUSSIAN_COLOR, lw=1.5, label="Gaussian" if first else None)
            # ax.set_title(f"{session_uid}\nroi {roi}  R²={fit['r2_test'][kept_row]:.2f}", fontsize=self.fontsize * 0.8)
            if first:
                ax.legend(fontsize=self.fontsize - 1, frameon=False, loc="upper left", markerfirst=True)

        xbounds = (0, theta[-1] + (theta[1] - theta[0]) / 2)
        xticks = xbounds
        ylims = [ax.get_ylim() for ax in axs.flat if ax.get_visible()]
        ymin = 0
        ymax = max(yl[1] for yl in ylims)
        ylims = (ymin, ymax)
        ybounds = (ymin, np.floor(ymax * 10) / 10)
        for cell in range(n_show):
            r, c = divmod(cell, n_cols)
            ax = axs[r, c]
            on_left = c == 0
            on_bottom = r == n_rows - 1
            spines_visible = ["bottom"]
            if on_left:
                spines_visible.append("left")
            xticks = xbounds if on_bottom else []
            yticks = ybounds if on_left else []
            ylabels = [0, 1] if on_left else []
            ax.set_ylim(ylims)
            format_spines(
                ax,
                x_pos=-0.02,
                y_pos=-0.02,
                xbounds=xbounds,
                ybounds=ybounds,
                spines_visible=spines_visible,
                xticks=xticks,
                yticks=yticks,
                ylabels=ylabels,
            )
        return fig


class PlacefieldPopulationViewer(Viewer):
    """Tilbury generalized-Gaussian placefield fits: population summaries (no examples).

    Four panels driven by every session in
    :class:`~dimensionality_manuscript.configs.tilbury_fit.TilburyFitConfig` results (one fit per
    neuron; the reported quality is held-out test R^2):

    - gs[0]: per-mouse peak-exponent (``p``) density (thin gray lines) with the across-mouse mean
      (bold dark line) and a reference line at ``p = 2`` (the ordinary-Gaussian exponent).
    - gs[1]: per-mouse median test R^2 for the generalized vs Gaussian model (paired, thin gray)
      with the across-mouse mean (bold dark line).
    - gs[2]: fraction of neurons where the generalized fit beats the Gaussian, either pooled to one
      per-mouse beeswarm (``fraction_view="pooled"``) or broken down with one beeswarm of per-session
      values per mouse (``fraction_view="by_mouse"``).
    - gs[3]: across-mouse power-law exponent beeswarms for four spectra — the selected ``source_key``
      spectrum (from ``results_spectra``/``results_cvpca``) and the ``eig_better``/``eig_tilbury``/
      ``eig_control`` fit spectra (colors orange/red/blue/black) — estimated by a window log-log fit
      or the mean five-point-derivative local exponent, with optional log-space pre-smoothing.

    The example single-neuron fits live in the separate :class:`PlacefieldExampleFitViewer`.
    ``TilburyFitConfig`` has no param grid; the gs[3] param-axis widgets come from the spectra
    aggregators.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_spectra: ResultsAggregator | None = None,
        results_cvpca: ResultsAggregator | None = None,
        num_bins: int = 80,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (6.0, 3.0),
    ):
        self.results = results
        self.results_spectra = results_spectra
        self.results_cvpca = results_cvpca
        # Reused by _spectrum/_sel_params (borrowed from PlacefieldSpectraViewer): the source_key
        # spectrum for the ax[3] alpha panel comes from these, resolved via SOURCE_OF_KEY.
        self._agg = {"stimspace": results_spectra, "cvpca": results_cvpca}
        self.config = results.config_class
        self.fontsize = fontsize
        self.figsize = figsize

        # Bin count for the per-session KDE of the peak-exponent density (gs[0]).
        self.add_integer("num_bins", value=num_bins, min=5, max=200)
        self.add_selection("fraction_view", options=["pooled", "by_mouse"], value="pooled")
        self.add_float("beewidth", value=0.2, min=0.0, max=1.0, step=0.01)
        self.add_selection("metric", value="cc", options=["r2", "cc"])

        # --- ax[3] population power-law-exponent panel: source_key spectrum + eig fit spectra ---
        # source_key options mirror spectrum_figure (StimSpace keys, plus the CVPCA key when given).
        if results_spectra is not None:
            source_options = list(_STIMSPACE_KEYS) + (list(_CVPCA_KEYS) if results_cvpca is not None else [])
            self.add_selection("source_key", options=source_options, value="ss_cv")

        # One widget per shared param-axis name (same tuple-label scheme as SpectrumFigureViewer), so
        # the source_key spectrum can be sliced (activity_parameters_name, smooth_widths, ...).
        merged_axes: dict[str, list] = {}
        for agg in self._agg.values():
            if agg is None:
                continue
            for name, options in agg.param_axes.items():
                existing = merged_axes.setdefault(name, [])
                existing.extend(opt for opt in options if opt not in existing)

        self._tuple_labels: dict[str, dict[str, tuple]] = {}
        for name, options in merged_axes.items():
            if any(isinstance(opt, tuple) for opt in options):
                label_map = {_tuple_label(opt): opt for opt in options}
                self._tuple_labels[name] = label_map
                widget_options = list(label_map)
            else:
                widget_options = options
            self.add_selection(name, options=widget_options)
            if name in _PREFERRED_DEFAULTS:
                default = self.encode_param(name, _PREFERRED_DEFAULTS[name])
                if default in widget_options:
                    self.update_selection(name, value=default)

        # Alpha-estimation controls (shared by all four ax[3] curves), mirroring spectrum_figure.
        self.add_boolean("normalize", value=True)
        self.add_selection("alpha_method", options=["window", "5-pt deriv"], value="window")
        self.add_integer_range("fit_range", value=(10, 30), min=1, max=500)
        self.add_integer("deriv_width", value=1, min=1, max=10)
        self.add_selection("smooth_kind", options=["none", "boxcar", "gaussian"], value="none")
        self.add_float("smooth_width", value=3.0, min=0.0, max=50.0, step=0.5)

    encode_param = PlacefieldSpectraViewer.encode_param
    _sel_params = PlacefieldSpectraViewer._sel_params
    _spectrum = PlacefieldSpectraViewer._spectrum
    _alpha_per_mouse = SpectrumFigureViewer._alpha_per_mouse

    def _eig_spectrum(self, state: dict, key: str) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` Tilbury-fit eigenvalue spectrum for ``key``.

        ``key`` is one of ``eig_better``/``eig_tilbury``/``eig_control`` (``"pad"`` keys, so
        ``avg_by_mouse`` nanmean works). Normalize/log-space smoothing match the source_key spectrum
        so all four ax[3] curves are estimated identically.
        """
        spec = self.results.sel(keys=[key], avg_by_mouse=True)[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, state["smooth_kind"], state["smooth_width"])

    def _aggregate_stats(self, state: dict) -> dict:
        """Per-session and per-mouse summary arrays for the population panels."""
        sel = self.results.sel(
            keys=["params", "r2_test", "r2_test_control", "pearson_test", "pearson_test_control", "idx_keep"],
            load_ragged=True,
            squeeze_ones=False,
        )
        params = sel["params"]  # (n_sess, N, 6)
        _performance_key = "r2_test" if state["metric"] == "r2" else "pearson_test"
        _performance_control_key = "r2_test_control" if state["metric"] == "r2" else "pearson_test_control"
        performance_test = sel[_performance_key]  # (n_sess, N)
        performance_test_control = sel[_performance_control_key]  # (n_sess, N)
        idx_keep = sel["idx_keep"]  # (n_sess,) object array of bool masks

        # Drop sessions with less than 200 fitted neurons (all-NaN r2 rows).
        idx_valid = np.sum(~np.isnan(performance_test), axis=1) >= 200
        idx_peak = self.config.param_names.index("p")
        peak = params[..., idx_peak][idx_valid]
        performance_test = performance_test[idx_valid]
        performance_test_control = performance_test_control[idx_valid]
        idx_keep = idx_keep[idx_valid]
        mouse_names = self.results.mouse_names[idx_valid]

        # Per-session median test R2 (generalized, gaussian) and fraction of kept neurons improved.
        avg_performance = np.full((performance_test.shape[0], 2), np.nan)
        avg_performance[:, 0] = np.nanmedian(performance_test, axis=1)
        avg_performance[:, 1] = np.nanmedian(performance_test_control, axis=1)
        improvement = performance_test - performance_test_control
        fraction_better = np.full(performance_test.shape[0], np.nan)
        for i, imp in enumerate(improvement):
            num_keep = int(np.nansum(idx_keep[i]))
            if num_keep > 0:
                fraction_better[i] = np.nansum(imp > 0) / num_keep

        # Per-session KDE of the peak exponent over a fixed [0, 10] grid.
        edges_peak = np.linspace(0.0, 10.0, state["num_bins"] + 1)
        centers_peak = edge2center(edges_peak)
        density_peak = np.full((peak.shape[0], len(centers_peak)), np.nan)
        for i, row in enumerate(peak):
            row = row[np.isfinite(row)]
            if len(row) < 2:
                continue
            density_peak[i] = gaussian_kde(row)(centers_peak)

        mouse_avg_performance, mouse_avg_names = average_by_mouse(avg_performance, mouse_names, include_mouse_names=True)
        return {
            "centers_peak": centers_peak,
            "mouse_density_peak": average_by_mouse(density_peak, mouse_names),
            "mouse_avg_performance": mouse_avg_performance,
            "mouse_fraction_better": average_by_mouse(fraction_better, mouse_names),
            "fraction_better": fraction_better,
            "mouse_names": mouse_names,
            "mouse_avg_names": mouse_avg_names,
        }

    def plot(self, state: dict):
        stats = self._aggregate_stats(state)

        plt.rcParams["font.size"] = self.fontsize
        fig = plt.figure(figsize=self.figsize, layout="constrained")
        outer = fig.add_gridspec(1, 4, width_ratios=[1, 0.65, 1, 1])

        # --- gs[0]: per-mouse peak-exponent density + across-mouse mean, reference at p=2 ---
        ax1 = fig.add_subplot(outer[0, 0])
        centers_peak = stats["centers_peak"]
        mouse_density_peak = stats["mouse_density_peak"]
        ax1.plot(centers_peak, mouse_density_peak.T, color="0.7", linewidth=0.8)
        ax1.plot(centers_peak, np.nanmean(mouse_density_peak, axis=0), color="k", linewidth=2.0)
        ax1.axvline(x=2.0, color="k", linestyle=":", linewidth=0.8)
        ax1.set_xticks([0, 2, 4, 6, 8, 10])
        ax1.set_xlabel("Peak Exponent")
        ax1.set_ylabel("Density")
        format_spines(
            ax1,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=(0, 10),
            ybounds=(0, np.round(np.nanmax(mouse_density_peak), 2)),
            spines_visible=["bottom", "left"],
            xticks=[0, 2, 4, 6, 8, 10],
        )

        # --- gs[1]: per-mouse median test R2, generalized vs gaussian, paired ---
        ax2 = fig.add_subplot(outer[0, 1])
        mouse_avg_performance = stats["mouse_avg_performance"]
        ax2.plot([0, 1], mouse_avg_performance.T, color="0.7", marker="o", markersize=3, linewidth=0.8)
        ax2.plot([0, 1], np.nanmean(mouse_avg_performance, axis=0), color="k", marker="o", markersize=5, linewidth=2.0)
        ax2.set_ylabel("Test R²" if state["metric"] == "r2" else "Test Correlation")
        ylims = ax2.get_ylim()
        ymin = min(0, ylims[0])
        ymax = 1
        ax2.set_xticks([0, 1])
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(ymin, ymax)
        format_spines(
            ax2,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=(0, 1),
            ybounds=(ymin, ymax),
            spines_visible=["bottom", "left"],
            xticks=[0, 1],
            # yticks=[ymin, 0.5, ymax],
        )
        ax2.set_xticklabels(["Generalized", "Gaussian"], rotation=45, ha="right")

        # --- gs[2]: fraction generalized > gaussian, pooled or broken down by mouse ---
        ax3 = fig.add_subplot(outer[0, 2])
        beewidth = state["beewidth"]
        if state["fraction_view"] == "pooled":
            vals = stats["mouse_fraction_better"]
            xbounds = (0, 0)
            offsets = beeswarm(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else np.zeros_like(vals)
            x = np.zeros_like(vals)
            x[np.isfinite(vals)] = beewidth * offsets
            ax3.plot(x, vals, linestyle="none", color="black", marker="o", markersize=4, alpha=0.8)
            ax3.plot([-0.25, 0.25], [np.nanmean(vals)] * 2, color="black", linewidth=2.0)
            ax3.set_xlim(-0.5, 0.5)
            xticks = []
        else:
            # One beeswarm per mouse, sorted by mean fraction from highest to lowest.
            mouse_names = stats["mouse_names"]
            mice = list(dict.fromkeys(mouse_names))
            xbounds = (0, len(mice) - 1)
            mice.sort(key=lambda m: np.nanmean(stats["fraction_better"][mouse_names == m]), reverse=True)
            for xi, mouse in enumerate(mice):
                vals = stats["fraction_better"][mouse_names == mouse]
                finite = np.isfinite(vals)
                offsets = np.zeros_like(vals)
                if finite.any():
                    offsets[finite] = beeswarm(vals[finite])
                ax3.plot(xi + beewidth * offsets, vals, linestyle="none", color="black", marker=".", markersize=5, alpha=0.3)
                ax3.plot(xi + np.array([-0.4, 0.4]), [np.nanmean(vals)] * 2, color="black", linewidth=1.2)
            ax3.set_xlim(-1.0, len(mice))
            ax3.set_xlabel("Mice")
            xticks = range(len(mice))

        ax3.set_yticks([0, 0.5, 1])
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Fraction Cells\nGeneralized > Gaussian")
        format_spines(
            ax3,
            x_pos=-0.02,
            y_pos=-0.02,
            xbounds=xbounds,
            ybounds=(0, 1),
            spines_visible=["bottom", "left"],
            yticks=[0, 0.5, 1],
        )
        ax3.set_xticks(xticks, labels=[])

        # --- gs[3]: across-mouse power-law exponent for source_key spectrum + eig fit spectra ---
        ax4 = fig.add_subplot(outer[0, 3])
        start, end = (int(v) for v in state["fit_range"])
        deriv_width = int(state["deriv_width"])
        method = state["alpha_method"]
        alpha_values, alpha_colors, alpha_labels = [], [], []
        if self.results_spectra is not None:
            source_key = state["source_key"]
            spec = self._spectrum(state, source_key)
            alpha_values.append(self._alpha_per_mouse(spec, start, end, deriv_width, method))
            alpha_colors.append(_POP_ALPHA_COLORS["source_key"])
            alpha_labels.append(source_key)
        for key in _POP_EIG_KEYS:
            spec = self._eig_spectrum(state, key)
            alpha_values.append(self._alpha_per_mouse(spec, start, end, deriv_width, method))
            alpha_colors.append(_POP_ALPHA_COLORS[key])
            alpha_labels.append(_POP_ALPHA_LABELS[key])
        _beeswarm_panel(ax4, alpha_values, alpha_colors, alpha_labels, self.fontsize, state["beewidth"])
        ax4.set_ylabel("Power-law exponent")
        return fig


def placefield_example_fits(
    results: ResultsAggregator,
    registry: PopulationRegistry,
    example_session: str | None = None,
    n_rows: int = 2,
    n_cols: int = 3,
    r2_threshold: float = 0.5,
    improvement_threshold: float = 0.0,
    random_seed: int = 0,
    normalize_curves: bool = True,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (8.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
):
    """
    Tilbury generalized-Gaussian placefield-fit figure: grid of example single-neuron fits.

    An ``n_rows x n_cols`` grid of example single-neuron fits (test placefield vs generalized-Gaussian
    and plain-Gaussian curves) for ``example_session``. The fitted parameters and R^2 come from the
    stored results; only the held-out test curve is rebuilt on the fly (deterministic trial-average,
    no re-fit). Population summaries are in :func:`placefield_population`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated :class:`TilburyFitConfig` results.
    registry : PopulationRegistry
        Registry used to rebuild the example session's test curve (must match the one the results
        were built with).
    example_session : str or None
        session_uid to show. If None, the first session in ``results`` is used.
    n_rows, n_cols : int
        Grid shape of the example-fit panel (``n_rows * n_cols`` example neurons).
    r2_threshold : float
        Example neurons must have generalized test R^2 above this threshold.
    improvement_threshold : float
        Example neurons must also beat the plain Gaussian by at least this much test R^2
        (``r2_generalized - r2_gaussian > improvement_threshold``).
    random_seed : int
        Seed for the random example draw (reproducible). If fewer than ``n_rows * n_cols`` neurons
        clear both thresholds, the leftover panels are left empty.
    normalize_curves : bool
        If True, normalize each curve (data and fits) by its sum
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldExampleFitViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = PlacefieldExampleFitViewer(
        results,
        registry,
        n_rows=n_rows,
        n_cols=n_cols,
        fontsize=fontsize,
        figsize=figsize,
    )
    if example_session is not None:
        viewer.update_selection("example_session", value=example_session)
    viewer.update_integer("n_rows", value=n_rows)
    viewer.update_integer("n_cols", value=n_cols)
    viewer.update_float("r2_threshold", value=r2_threshold)
    viewer.update_float("improvement_threshold", value=improvement_threshold)
    viewer.update_integer("random_seed", value=random_seed)
    viewer.update_boolean("normalize_curves", value=normalize_curves)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


def placefield_fit_figure(
    results: ResultsAggregator,
    registry: PopulationRegistry,
    session_uids: list[str],
    neurons: list[int],
    n_rows: int = 2,
    n_cols: int = 3,
    normalize: str = "std",
    normalize_independent: bool = False,
    strict: bool = True,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (8.0, 4.0),
    save_path=None,
    return_syd_viewer: bool = False,
):
    """Tilbury placefield-fit figure for a hand-picked, ordered list of ``(session, neuron)`` examples.

    Plots the first ``n_rows * n_cols`` neurons of ``(session_uids, neurons)`` — two parallel lists —
    into a shared-axes grid, in list order, each panel overlaying the held-out test placefield against
    the fitted generalized-Gaussian and plain-Gaussian curves. Fitted parameters and R^2 come from the
    stored :class:`TilburyFitConfig` results; the test curve is rebuilt on the fly (no re-fit). For the
    random-draw counterpart see :func:`placefield_example_fits`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated :class:`TilburyFitConfig` results.
    registry : PopulationRegistry
        Registry used to rebuild each session's test curve and to map an original ROI index to its
        fit row (must match the one the results were built with).
    session_uids : list of str
        session_uid per requested neuron.
    neurons : list of int
        The **original ROI index** per requested neuron — the index into the session's full ROI set
        (``np.where(session.idx_rois)[0][k]``), which is the stable identifier to record for a figure.
        Must be the same length as ``session_uids`` and aligned to it.
    n_rows, n_cols : int
        Grid shape; the first ``n_rows * n_cols`` list entries are plotted (extra panels hidden).
    normalize : {"std", "sum", "max", "none"}
        Per-panel, the test-data curve and both fits are divided by this statistic of the test-data
        curve, so the fits stay overlaid on the data while panels share a scale under ``sharey``.
    normalize_independent : bool
        If True, divide each of the three curves (test data, generalized, gaussian) by its *own*
        statistic instead of the shared test-data one — a shape-only comparison that removes amplitude
        differences between the fits and the data (they no longer overlay). Default False.
    strict : bool
        If True (default), a requested ROI that never entered the pipeline or was dropped before
        fitting raises ``ValueError``. If False, that panel is left empty with a red status title.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldFitFigureViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = PlacefieldFitFigureViewer(
        results,
        registry,
        session_uids=session_uids,
        neurons=neurons,
        n_rows=n_rows,
        n_cols=n_cols,
        normalize=normalize,
        normalize_independent=normalize_independent,
        strict=strict,
        fontsize=fontsize,
        figsize=figsize,
    )
    viewer.update_integer("n_rows", value=n_rows)
    viewer.update_integer("n_cols", value=n_cols)
    viewer.update_selection("normalize", value=normalize)
    viewer.update_boolean("normalize_independent", value=normalize_independent)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig


def placefield_population(
    results: ResultsAggregator,
    results_spectra: ResultsAggregator | None = None,
    results_cvpca: ResultsAggregator | None = None,
    num_bins: int = 80,
    fraction_view: str = "pooled",
    beewidth: float = 0.2,
    source_key: str = "ss_cv",
    normalize: bool = True,
    alpha_method: str = "window",
    fit_range: tuple[int, int] = (10, 30),
    deriv_width: int = 1,
    smooth_kind: str = "none",
    smooth_width: float = 3.0,
    fontsize: float = 9.0,
    figsize: tuple[float, float] = (8.0, 3.0),
    save_path=None,
    return_syd_viewer: bool = False,
    **selections,
):
    """
    Tilbury generalized-Gaussian placefield-fit figure: population summaries (no examples).

    Four panels over every session in ``results``: gs[0] the per-mouse peak-exponent density (thin
    gray) with the across-mouse mean (bold) and a reference at ``p = 2``; gs[1] the per-mouse median
    test R^2 for the generalized vs Gaussian model (paired); gs[2] the fraction of neurons where the
    generalized fit beats the Gaussian, either pooled (one per-mouse beeswarm) or broken down by
    mouse; gs[3] the across-mouse power-law exponent for the selected ``source_key`` spectrum and the
    ``eig_better``/``eig_tilbury``/``eig_control`` fit spectra (colors orange/red/blue/black),
    estimated by the same window or five-point-derivative method as :func:`spectrum_figure`. Example
    single-neuron fits are in :func:`placefield_example_fits`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated :class:`TilburyFitConfig` results (source of the ``eig_*`` spectra).
    results_spectra : ResultsAggregator or None
        Aggregated StimSpaceSpectra results, source of the ``source_key`` spectrum in gs[3]. If None,
        gs[3] shows only the three eig fit spectra.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results; when given, ``reg_covariances_fixed`` is also a valid
        ``source_key``.
    num_bins : int
        Bin count for the per-session KDE of the peak-exponent density (gs[0]).
    fraction_view : {"pooled", "by_mouse"}
        gs[2] layout: one pooled per-mouse beeswarm, or one per-session beeswarm per mouse.
    beewidth : float
        Beeswarm point spread in x-axis units (gs[2] and gs[3]).
    source_key : str
        Which spectrum drives the gs[3] orange curve: ``ss_cv``/``ss_direct``/``ss_cvpca`` (from
        ``results_spectra``) or ``reg_covariances_fixed`` (from ``results_cvpca``).
    normalize : bool
        If True, normalize each gs[3] spectrum by its sum before smoothing (does not affect alpha).
    alpha_method : {"window", "5-pt deriv"}
        gs[3] exponent estimator: a log-log window fit, or the mean five-point-derivative local
        exponent over the window.
    fit_range : tuple[int, int]
        0-based ``[start, end)`` rank window for the gs[3] exponent (all four curves).
    deriv_width : int
        Stencil half-width for the five-point-derivative local exponent (``alpha_method="5-pt deriv"``).
    smooth_kind : {"none", "boxcar", "gaussian"}
        Log-space pre-smoothing applied to each gs[3] spectrum before fitting.
    smooth_width : float
        Boxcar full-width in rank units; the Gaussian uses ``sigma = smooth_width / 2``.
    fontsize : float
        Base font size applied via ``plt.rcParams``.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str or pathlib.Path or None
        If given (and ``return_syd_viewer`` is False), save the rendered figure here.
    return_syd_viewer : bool
        If True, return the Syd viewer with state seeded from the other arguments.
    **selections
        Overrides for the gs[3] parameter-axis selections, keyed by raw ``param_axes`` name of
        ``results_spectra``/``results_cvpca`` (e.g. ``activity_parameters_name``, ``smooth_widths``,
        ``reliability_fraction_active_thresholds``). See :func:`spectrum_figure`.

    Returns
    -------
    matplotlib.figure.Figure or PlacefieldPopulationViewer
        The rendered figure, or the Syd viewer when ``return_syd_viewer`` is True.
    """
    viewer = PlacefieldPopulationViewer(
        results,
        results_spectra=results_spectra,
        results_cvpca=results_cvpca,
        num_bins=num_bins,
        fontsize=fontsize,
        figsize=figsize,
    )
    viewer.update_integer("num_bins", value=num_bins)
    viewer.update_selection("fraction_view", value=fraction_view)
    viewer.update_float("beewidth", value=beewidth)
    if results_spectra is not None:
        viewer.update_selection("source_key", value=source_key)

    valid_selections = set()
    for agg in viewer._agg.values():
        if agg is None:
            continue
        valid_selections.update(agg.param_axes)
    for key, value in selections.items():
        if key not in valid_selections:
            raise ValueError(f"Unknown selection {key!r}. Options: {sorted(valid_selections)}")
        viewer.update_selection(key, value=viewer.encode_param(key, value))
    viewer.update_boolean("normalize", value=normalize)
    viewer.update_selection("alpha_method", value=alpha_method)
    viewer.update_integer_range("fit_range", value=tuple(fit_range))
    viewer.update_integer("deriv_width", value=deriv_width)
    viewer.update_selection("smooth_kind", value=smooth_kind)
    viewer.update_float("smooth_width", value=smooth_width)
    if return_syd_viewer:
        return viewer

    fig = viewer.plot(viewer.state)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig
