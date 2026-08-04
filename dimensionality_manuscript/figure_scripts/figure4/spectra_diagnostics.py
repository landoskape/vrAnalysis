"""Interactive spectrum diagnostics: example spectra, power-law exponents, and the adaptive fit.

Three viewers, from coarsest to most zoomed-in:

- :class:`PlacefieldSpectraViewer` -- one or more example spectra averaged over mice, plus
  per-mouse power-law-exponent estimates for every curve option.
- :class:`SessionSpectraViewer` -- the single-session analogue: every curve option overlaid for
  one session.
- :class:`AdaptiveSpectraEstimationViewer` -- mechanics of the ``alpha_method="adaptive"``
  median-five-point-derivative fit on one session's spectrum.

All three read spectra from a StimSpaceSpectra aggregator (``results``) and, optionally, a CVPCA
aggregator (``results_cvpca``); :class:`AdaptiveSpectraEstimationViewer` additionally accepts a
subspace aggregator (``results_subspace``) for the ``"svca"`` key. Which aggregator a given key
comes from is resolved via :data:`SOURCE_OF_KEY`.
"""

import numpy as np

from vrAnalysis.helpers.plotting import beeswarm
from dimensionality_manuscript import ResultsAggregator, average_by_mouse

from ._param_axes import (
    ASE_SVCA_KEY,
    CVPCA_KEYS,
    FF_KEY,
    KEY_COLORS,
    PREFERRED_DEFAULTS,
    SOURCE_OF_KEY,
    STIMSPACE_KEYS,
    add_merged_param_axis_widgets,
    merged_axis_names as _merged_axis_names,
    sel_params,
)
from ._spectrum_math import (
    _clip_at_first_negative,
    _decay_alpha_per_mouse,
    _deriv_alpha_per_mouse,
    _first_negative_index,
    _local_alpha_curve,
    _median_fpd_alpha_session,
    _second_derivative_window,
    _smooth_spectrum,
    _xvals,
)
from ..panels import FigureViewer


class PlacefieldSpectraViewer(FigureViewer):
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
        ``results_cvpca``); see :data:`SOURCE_OF_KEY`. Each is colored per :data:`KEY_COLORS`. The
        exponent panels always cover all available curve options regardless of this choice.
    ylim_min : float
        Lower y-limit of the spectrum panel in log10 units; the applied floor is ``10 ** ylim_min``.
        The upper limit is autoscaled to the data.
    normalize : bool
        If True, normalize the spectrum by the sum of the spectrum.
    clip_negative : bool
        If True, replace each spectrum's first negative entry and all later ranks with NaN before
        averaging across mice.
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
        Font size for every tick label, axis label, and legend text in the figure.
    figsize : tuple[float, float]
        Figure size in inches.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        *,
        source_key: str | list[str] = "ss_cv",
        ylim_min: float = -5.5,
        normalize: bool = True,
        clip_negative: bool = False,
        fit_range: tuple[int, int] = (10, 20),
        deriv_width: int = 1,
        smooth_kind: str = "none",
        smooth_width: float = 3.0,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (9.0, 3.0),
        **selection_defaults,
    ):
        source_keys = [source_key] if isinstance(source_key, str) else list(source_key)
        for sk in source_keys:
            if sk not in SOURCE_OF_KEY:
                raise ValueError(f"Unknown source_key {sk!r}. Options: {list(SOURCE_OF_KEY)}")
            if SOURCE_OF_KEY[sk] == "cvpca" and results_cvpca is None:
                raise ValueError(f"source_key {sk!r} is a CVPCA key but results_cvpca was not provided.")

        self.results = results
        self.results_cvpca = results_cvpca
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.figsize = figsize

        available = list(STIMSPACE_KEYS)
        if results_cvpca is not None:
            available += list(CVPCA_KEYS)
        self.add_multiple_selection("source_key", options=available, value=source_keys)

        # One widget per param-axis name, shared across sources; tuple-valued axes (e.g.
        # smooth_widths) are encoded as string labels -- see add_merged_param_axis_widgets.
        # ``selection_defaults`` lets a caller seed any of these axes by name (e.g.
        # activity_parameters_name="default", smooth_widths=(5.0, None)).
        self._tuple_labels = add_merged_param_axis_widgets(
            self, results, results_cvpca, preferred_defaults={**PREFERRED_DEFAULTS, **selection_defaults}
        )
        self._param_axis_names = _merged_axis_names(results, results_cvpca)

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_boolean("normalize", value=normalize)
        self.add_boolean("clip_negative", value=clip_negative)
        # Rank window (0-based, [start, end)) the exponent is estimated over, for both methods.
        self.add_integer_range("fit_range", value=tuple(fit_range), min=1, max=200)
        # Stencil half-width for the five-point-derivative local exponent.
        self.add_integer("deriv_width", value=deriv_width, min=1, max=10)
        # Log-space (geometric-mean) pre-smoothing of the spectrum before fitting.
        self.add_selection("smooth_kind", options=["none", "boxcar", "gaussian"], value=smooth_kind)
        self.add_float("smooth_width", value=smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        for name in ("source_key", *self._param_axis_names, "normalize", "clip_negative", "smooth_kind", "smooth_width", "fit_range", "deriv_width"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _available_keys(self) -> list[str]:
        """Curve options available to estimate exponents for (source_key options)."""
        keys = list(STIMSPACE_KEYS)
        if self.results_cvpca is not None:
            keys += list(CVPCA_KEYS)
        return keys

    def _sel_params(self, state: dict, source: str) -> dict:
        """Select the params relevant to this source, decoding tuple labels back to tuples."""
        return sel_params(state, self._tuple_labels, self._agg[source].param_axes)

    def _spectrum(self, state: dict, key: str) -> np.ndarray:
        """Mouse-averaged ``(mice, dims)`` spectrum for ``key``, normalized per ``state``."""
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        if state["clip_negative"]:
            spec = agg.sel(keys=[key], avg_by_mouse=False, **self._sel_params(state, source))[key]
            spec = _clip_at_first_negative(np.atleast_2d(np.asarray(spec, dtype=float)))
            spec = average_by_mouse({key: spec}, agg.mouse_names)[key]
        else:
            spec = agg.sel(keys=[key], avg_by_mouse=True, **self._sel_params(state, source))[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, state["smooth_kind"], state["smooth_width"])

    def refresh_data(self, state: dict) -> None:
        """Re-select every curve option's spectrum and both exponent estimates."""
        keys_all = self._available_keys()
        start, end = (int(v) for v in state["fit_range"])
        deriv_width = int(state["deriv_width"])
        self._spectra = {k: self._spectrum(state, k) for k in keys_all}
        self._local_alpha = {k: _local_alpha_curve(self._spectra[k], deriv_width) for k in keys_all}
        self._decay_alpha = {k: _decay_alpha_per_mouse(self._spectra[k], start, end) for k in keys_all}
        self._deriv_alpha = {k: _deriv_alpha_per_mouse(self._local_alpha[k], start, end) for k in keys_all}

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        selected_keys = list(state["source_key"])
        keys_all = self._available_keys()
        start, end = (int(v) for v in state["fit_range"])
        each_alpha = 0.3
        ylim_min = state["ylim_min"]

        spectra = self._spectra
        local_alpha = self._local_alpha
        decay_alpha = self._decay_alpha
        deriv_alpha = self._deriv_alpha

        fig, ax = self.new_subplots(1, 3, figsize=self.figsize, layout="constrained", width_ratios=[1.0, 0.9, 0.9])

        # --- ax[0]: the selected example spectra (one faint line per mouse + bold average per key) ---
        for key in selected_keys:
            spec = spectra[key]
            spec_positive = np.where(spec > 0, spec, np.nan)
            ex_color = KEY_COLORS.get(key, "blue")
            ax[0].plot(_xvals(spec), spec_positive.T, color=ex_color, alpha=each_alpha, linewidth=1.0)
            ax[0].plot(_xvals(spec), np.nanmean(spec_positive, axis=0), color=ex_color, label=key, linewidth=2.0)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(bottom=10**ylim_min)
        yticks = ax[0].get_yticks()
        ytick_power = [np.log10(yt) for yt in yticks]
        ax[0].set_yticks(yticks, labels=ytick_power, fontsize=fontsize)
        ax[0].set_ylim(bottom=10**ylim_min)
        ax[0].set_xlabel("Shared Dimension", fontsize=fontsize)
        ax[0].set_ylabel("Variance", fontsize=fontsize)
        ax[0].tick_params(axis="both", which="both", labelsize=fontsize)
        ax[0].legend(loc="upper right", fontsize=fontsize, frameon=False)

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
                color = KEY_COLORS.get(k, "gray")
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
            ax[1].text(np.mean(group_ticks), yline - 0.3 * pad, mname, fontsize=fontsize, ha="center", va="bottom")
        ax[1].set_xlim(-0.5, max(xticks) + 0.5)
        ax[1].set_ylim(yline - pad, yhi + pad)
        ax[1].set_ylabel("Power-law exponent", fontsize=fontsize)
        ax[1].axhline(4.0, color="0.8", linestyle="--", linewidth=1.0)
        ax[1].set_xticks(xticks, labels=keys_all * len(methods), rotation=45, ha="right", fontsize=fontsize)
        ax[1].tick_params(axis="both", which="both", labelsize=fontsize)

        # --- ax[2]: five-point-derivative local-exponent curves (per mouse + bold average) ---
        for k in keys_all:
            la = local_alpha[k]
            color = KEY_COLORS.get(k, "gray")
            xv = np.arange(la.shape[1]) + 1
            ax[2].plot(xv, la.T, color=color, alpha=0.2, linewidth=0.8)
            ax[2].plot(xv, np.nanmean(la, axis=0), color=color, linewidth=2.0, label=k)
        ax[2].axvspan(start + 1, end, color="0.8", alpha=0.4)
        ax[2].set_xscale("log")
        ax[2].set_xlabel("Shared Dimension", fontsize=fontsize)
        ax[2].set_ylabel("Local exponent", fontsize=fontsize)
        ax[2].set_ylim(-1, 10)
        ax[2].tick_params(axis="both", which="both", labelsize=fontsize)
        ax[2].legend(loc="upper left", fontsize=fontsize, frameon=False)
        return fig


class SessionSpectraViewer(FigureViewer):
    """Interactive per-session view of every shared-variance spectrum on one axis.

    The single-session analogue of :class:`PlacefieldSpectraViewer`. Instead of showing one
    ``source_key`` spectrum across all mice, ax[0] overlays every available curve option
    (``ss_cv``, ``ss_direct``, ..., ``reg_covariances_fixed``) for a single selected session,
    each colored per :data:`KEY_COLORS`. ax[1] and ax[2] are unchanged in structure but now
    estimate the power-law exponent from that one session's spectra (one point per curve in the
    beeswarm, one local-exponent curve per option).

    A ``session`` selection widget replaces ``source_key``; its options are the union of the
    ``session_ids`` of every provided aggregator (in first-seen order). For each curve the session
    is resolved against its own source aggregator (via :data:`SOURCE_OF_KEY`); a session missing
    from a source yields an all-NaN spectrum for that curve (it simply does not draw).

    Param-axis widgets, tuple-label encoding, and the log10 y-floor behave exactly as in
    :class:`PlacefieldSpectraViewer`.

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
    clip_negative : bool
        If True, replace each spectrum's first negative entry and all later ranks with NaN.
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
        Font size for every tick label, axis label, title, and legend text in the figure.
    figsize : tuple[float, float]
        Figure size in inches.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        *,
        session: str | None = None,
        ylim_min: float = -5.5,
        normalize: bool = True,
        clip_negative: bool = False,
        fit_range: tuple[int, int] = (10, 20),
        deriv_width: int = 1,
        smooth_kind: str = "none",
        smooth_width: float = 3.0,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (9.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.results_cvpca = results_cvpca
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.figsize = figsize

        # Session options: union of every aggregator's session_ids, first-seen order.
        sessions: list[str] = []
        for agg in self._agg.values():
            if agg is None:
                continue
            sessions.extend(uid for uid in agg.session_ids if uid not in sessions)
        self.add_selection("session", options=sessions, value=session if session is not None else (sessions[0] if sessions else None))

        # One widget per param-axis name, shared across sources (same scheme as PlacefieldSpectraViewer).
        self._tuple_labels = add_merged_param_axis_widgets(
            self, results, results_cvpca, preferred_defaults={**PREFERRED_DEFAULTS, **selection_defaults}
        )
        self._param_axis_names = _merged_axis_names(results, results_cvpca)

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_boolean("normalize", value=normalize)
        self.add_boolean("clip_negative", value=clip_negative)
        self.add_integer_range("fit_range", value=tuple(fit_range), min=1, max=200)
        self.add_integer("deriv_width", value=deriv_width, min=1, max=10)
        self.add_selection("smooth_kind", options=["none", "boxcar", "gaussian"], value=smooth_kind)
        self.add_float("smooth_width", value=smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        for name in ("session", *self._param_axis_names, "normalize", "clip_negative", "smooth_kind", "smooth_width", "fit_range", "deriv_width"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _available_keys(self) -> list[str]:
        """Curve options available to estimate exponents for."""
        keys = list(STIMSPACE_KEYS)
        if self.results_cvpca is not None:
            keys += list(CVPCA_KEYS)
        return keys

    def _sel_params(self, state: dict, source: str) -> dict:
        """Select the params relevant to this source, decoding tuple labels back to tuples."""
        return sel_params(state, self._tuple_labels, self._agg[source].param_axes)

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
        if state["clip_negative"]:
            spec = _clip_at_first_negative(spec)
        if state["normalize"]:
            spec = spec / np.nansum(spec, axis=1)[:, None]
        return _smooth_spectrum(spec, state["smooth_kind"], state["smooth_width"])

    def refresh_data(self, state: dict) -> None:
        """Re-select every curve option's single-session spectrum and both exponent estimates."""
        keys_all = self._available_keys()
        start, end = (int(v) for v in state["fit_range"])
        deriv_width = int(state["deriv_width"])
        self._spectra = {k: self._spectrum(state, k) for k in keys_all}
        self._local_alpha = {k: _local_alpha_curve(self._spectra[k], deriv_width) for k in keys_all}
        self._decay_alpha = {k: _decay_alpha_per_mouse(self._spectra[k], start, end) for k in keys_all}
        self._deriv_alpha = {k: _deriv_alpha_per_mouse(self._local_alpha[k], start, end) for k in keys_all}

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        keys_all = self._available_keys()
        start, end = (int(v) for v in state["fit_range"])
        ylim_min = state["ylim_min"]

        spectra = self._spectra
        local_alpha = self._local_alpha
        decay_alpha = self._decay_alpha
        deriv_alpha = self._deriv_alpha

        fig, ax = self.new_subplots(1, 3, figsize=self.figsize, layout="constrained", width_ratios=[1.0, 0.9, 0.9])

        # --- ax[0]: every curve option for the selected session (one line each) ---
        for k in keys_all:
            spec = spectra[k]
            spec_positive = np.where(spec > 0, spec, np.nan)
            color = KEY_COLORS.get(k, "gray")
            ax[0].plot(_xvals(spec), spec_positive.T, color=color, label=k, linewidth=1.5)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(bottom=10**ylim_min)
        yticks = ax[0].get_yticks()
        ytick_power = [np.log10(yt) for yt in yticks]
        ax[0].set_yticks(yticks, labels=ytick_power, fontsize=fontsize)
        ax[0].set_ylim(bottom=10**ylim_min)
        ax[0].set_xlabel("Shared Dimension", fontsize=fontsize)
        ax[0].set_ylabel("Variance", fontsize=fontsize)
        ax[0].tick_params(axis="both", which="both", labelsize=fontsize)
        ax[0].set_title(str(state["session"]), fontsize=fontsize)
        ax[0].legend(loc="upper right", fontsize=fontsize, frameon=False)

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
                color = KEY_COLORS.get(k, "gray")
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
            ax[1].text(np.mean(group_ticks), yline - 0.3 * pad, mname, fontsize=fontsize, ha="center", va="top")
        ax[1].set_xlim(-0.5, max(xticks) + 0.5)
        ax[1].set_ylim(yline - pad, yhi + pad)
        ax[1].set_ylabel("Power-law exponent", fontsize=fontsize)
        ax[1].axhline(4.0, color="0.8", linestyle="--", linewidth=1.0)
        ax[1].set_xticks(xticks, labels=keys_all * len(methods), rotation=45, ha="right", fontsize=fontsize)
        ax[1].tick_params(axis="both", which="both", labelsize=fontsize)

        # --- ax[2]: five-point-derivative local-exponent curves (one per curve option) ---
        for k in keys_all:
            la = local_alpha[k]
            color = KEY_COLORS.get(k, "gray")
            xv = np.arange(la.shape[1]) + 1
            ax[2].plot(xv, la.T, color=color, linewidth=1.5, label=k)
        ax[2].axvspan(start + 1, end, color="0.8", alpha=0.4)
        ax[2].set_xscale("log")
        ax[2].set_xlabel("Shared Dimension", fontsize=fontsize)
        ax[2].set_ylabel("Local exponent", fontsize=fontsize)
        ax[2].set_ylim(-1, 10)
        ax[2].tick_params(axis="both", which="both", labelsize=fontsize)
        ax[2].legend(loc="upper left", fontsize=fontsize, frameon=False)
        return fig


class AdaptiveSpectraEstimationViewer(FigureViewer):
    """Diagnostic figure for the ``alpha_method="adaptive"`` median-FPD fit, on one session's spectrum.

    The single-session, single-curve analogue of the ``"adaptive"`` branch of the placefield/session
    spectra viewers (see :func:`_second_derivative_window` / :func:`_median_fpd_alpha_session`), for
    inspecting the mechanics of that method directly:

    - ax[0]: the raw and smoothed spectrum on log-log axes, with vertical lines marking the smoothed
      spectrum's 2nd-derivative peak (max curvature) and the raw spectrum's first negative entry,
      plus the buffered window start/end actually used for the median (``adaptive_buffer`` dims
      inside each of those two landmarks).
    - ax[1]: the five-point-derivative local exponent curve restricted to that buffered window, with
      a horizontal line at its median -- the value the adaptive method would report (NaN, annotated
      as such, if fewer than five finite values fall inside it).
    - ax[2]: the two other (fixed-window) methods for comparison, over an independent ``fit_range``:
      the five-point-derivative local-exponent curve (window shaded) plus horizontal reference lines
      for the window power-law fit and the window-mean FPD exponent.
    - ax[3]: the first and second derivative of the smoothed spectrum.

    ``key`` options are the PF-like StimSpace/CVPCA curves plus the two "Reliable CA1" curves:
    ``"ff"`` (SVD, the StimSpaceSpectra ``ff`` key) and ``"svca"`` (the subspace
    ``variance_activity`` key, ``subspace_name='svca_subspace'``, requires ``results_subspace``).
    Session selection, param-axis widgets, tuple-label encoding and the log10 y-floor behave as in
    :class:`SessionSpectraViewer`.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*``/``sf_*``/``ff`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results, source of the ``reg_covariances_fixed`` key. If None only
        StimSpace keys (including ``ff``) are selectable.
    results_subspace : ResultsAggregator or None
        Aggregated SubspaceConfig results, source of the ``"svca"`` key (subspace
        ``variance_activity``, ``subspace_name='svca_subspace'``). Required for that option; if None
        it is not selectable.
    session : str or None
        session_uid to show. Must be a session of at least one provided aggregator (including
        ``results_subspace``). If None, the first session (union of the aggregators' ``session_ids``)
        is used.
    key : str
        Which spectrum to inspect: one of ``ss_cv``/``ss_direct``/``ss_cvpca``/``sf_cv``/``sf_direct``/
        ``ff`` (from ``results``), ``reg_covariances_fixed`` (from ``results_cvpca``), or ``"svca"``
        (from ``results_subspace``).
    ylim_min : float
        Lower y-limit of the spectrum panel (ax[0]) in log10 units; the applied floor is
        ``10 ** ylim_min``.
    normalize : bool
        If True, normalize the spectrum by its sum before smoothing/fitting.
    clip_negative : bool
        If True, replace the spectrum's first negative entry and all later ranks with NaN.
    smooth_kind : {"none", "boxcar", "gaussian"}
        Log-space (geometric-mean) pre-smoothing applied before every fit in this figure.
        ``"none"`` disables smoothing.
    smooth_width : float
        Boxcar full-width in rank units; the Gaussian uses ``sigma = smooth_width / 2``.
    adaptive_buffer : int
        Dims of margin applied on both sides of the second-derivative window: it starts this many
        dims after the second derivative's peak and ends this many dims before its first negative
        crossing.
    fit_range : tuple[int, int]
        0-based ``[start, end)`` rank window for the two fixed-window comparison methods in ax[2]
        (independent of the adaptive window).
    deriv_width : int
        Stencil half-width for the five-point-derivative local exponent (ax[1]/ax[2]).
    fontsize : float
        Font size for every tick label, axis label, title, and legend text in the figure.
    figsize : tuple[float, float]
        Figure size in inches.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        *,
        session: str | None = None,
        key: str = "ss_cv",
        ylim_min: float = -5.5,
        normalize: bool = True,
        clip_negative: bool = False,
        smooth_kind: str = "none",
        smooth_width: float = 3.0,
        adaptive_buffer: int = 2,
        fit_range: tuple[int, int] = (10, 20),
        deriv_width: int = 1,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (9.0, 3.0),
        **selection_defaults,
    ):
        key_options = (
            list(STIMSPACE_KEYS)
            + [FF_KEY]
            + (list(CVPCA_KEYS) if results_cvpca is not None else [])
            + ([ASE_SVCA_KEY] if results_subspace is not None else [])
        )
        if key not in key_options:
            raise ValueError(f"Unknown key {key!r}. Options: {key_options}")

        self.results = results
        self.results_cvpca = results_cvpca
        self.results_subspace = results_subspace
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.figsize = figsize

        sessions: list[str] = []
        for agg in self._agg.values():
            if agg is None:
                continue
            sessions.extend(uid for uid in agg.session_ids if uid not in sessions)
        if results_subspace is not None:
            sessions.extend(uid for uid in results_subspace.session_ids if uid not in sessions)
        self.add_selection("session", options=sessions, value=session if session is not None else (sessions[0] if sessions else None))

        self.add_selection("key", options=key_options, value=key)

        # One widget per param-axis name, shared across sources (same scheme as SessionSpectraViewer).
        # ``results_subspace`` is deliberately excluded here (and from ``self._agg``): its own
        # ``smooth_width``/``subspace_name`` axes are fixed internally by ``_spectrum_raw_and_smooth``
        # for the "svca" key.
        self._tuple_labels = add_merged_param_axis_widgets(
            self, results, results_cvpca, preferred_defaults={**PREFERRED_DEFAULTS, **selection_defaults}
        )
        self._param_axis_names = _merged_axis_names(results, results_cvpca)

        self.add_float("ylim_min", value=ylim_min, min=-8.0, max=2.0, step=0.1)
        self.add_boolean("normalize", value=normalize)
        self.add_boolean("clip_negative", value=clip_negative)
        self.add_selection("smooth_kind", options=["none", "boxcar", "gaussian"], value=smooth_kind)
        self.add_float("smooth_width", value=smooth_width, min=0.0, max=50.0, step=0.5)

        # Adaptive median-FPD fit (see _second_derivative_window / _median_fpd_alpha_session).
        self.add_integer("adaptive_buffer", value=adaptive_buffer, min=0, max=20)

        # Fixed-window comparison methods (ax[2]): independent of the adaptive window.
        self.add_integer_range("fit_range", value=tuple(fit_range), min=1, max=500)
        self.add_integer("deriv_width", value=deriv_width, min=1, max=10)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        for name in (
            "session",
            "key",
            *self._param_axis_names,
            "normalize",
            "clip_negative",
            "smooth_kind",
            "smooth_width",
            "adaptive_buffer",
            "fit_range",
            "deriv_width",
        ):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _sel_params(self, state: dict, source: str) -> dict:
        """Select the params relevant to this source, decoding tuple labels back to tuples."""
        return sel_params(state, self._tuple_labels, self._agg[source].param_axes)

    def _spectrum_raw_and_smooth(self, state: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
        """Single-session raw (normalized only) and smoothed 1-D spectrum for ``key``.

        Returns the pre-smoothing row alongside the smoothed one: the window end is found on the
        raw values (see :func:`_second_derivative_window`), since smoothing maps every non-positive
        entry to NaN before exponentiating back (:func:`_smooth_spectrum`) and so never produces a
        negative output. The ``"svca"`` key instead reads the subspace aggregator's own
        ``variance_activity`` (``subspace_name='svca_subspace'``, ``smooth_width=None`` fixed at the
        SubspaceConfig level -- distinct from this viewer's own ``smooth_width`` widget below, which
        is the log-space post-hoc smoothing applied here).
        """
        if key == ASE_SVCA_KEY:
            params = {"subspace_name": "svca_subspace", "smooth_width": None}
            if "activity_parameters_name" in state:
                params["activity_parameters_name"] = state["activity_parameters_name"]
            agg = self.results_subspace
            arr = agg.sel(keys=["variance_activity"], squeeze_ones=False, **params)["variance_activity"]
        else:
            source = SOURCE_OF_KEY[key]
            agg = self._agg[source]
            arr = agg.sel(keys=[key], squeeze_ones=False, **self._sel_params(state, source))[key]
        arr = np.asarray(arr, dtype=float)
        sess_idx = agg._session_index.get(state["session"])
        if sess_idx is None:
            raw = np.full(arr.shape[-1], np.nan)
            return raw, raw.copy()
        raw = np.array(arr[sess_idx], dtype=float)
        if state["clip_negative"]:
            raw = _clip_at_first_negative(raw[None])[0]
        if state["normalize"]:
            raw = raw / np.nansum(raw)
        smoothed = _smooth_spectrum(raw[None, :], state["smooth_kind"], state["smooth_width"])[0]
        return raw, smoothed

    def refresh_data(self, state: dict) -> None:
        """Re-select the spectrum and every derived quantity this figure draws."""
        key = state["key"]
        buffer = int(state["adaptive_buffer"])
        fit_start, fit_end = (int(v) for v in state["fit_range"])
        deriv_width = int(state["deriv_width"])

        raw, smoothed = self._spectrum_raw_and_smooth(state, key)
        self._raw = raw
        self._smoothed = smoothed
        self._first_derivative = np.diff(smoothed, n=1)
        self._second_derivative = np.diff(smoothed, n=2)
        self._first_neg = _first_negative_index(raw) + 1

        start, end = _second_derivative_window(raw, smoothed, buffer)
        self._start = start
        self._end = end
        self._peak = start - 1 - buffer if start > 0 else -1  # search-restricted peak _second_derivative_window used

        smoothed_2d = smoothed[None, :]
        local_alpha_2d = _local_alpha_curve(smoothed_2d, deriv_width)
        self._local_alpha = local_alpha_2d[0]
        self._median_alpha = _median_fpd_alpha_session(raw, smoothed, deriv_width, buffer)
        self._window_alpha = _decay_alpha_per_mouse(smoothed_2d, fit_start, fit_end)[0]
        self._window_deriv_alpha = _deriv_alpha_per_mouse(local_alpha_2d, fit_start, fit_end)[0]

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        key = state["key"]
        ylim_min = state["ylim_min"]
        fit_start, fit_end = (int(v) for v in state["fit_range"])

        raw, smoothed = self._raw, self._smoothed
        first_derivative, second_derivative = self._first_derivative, self._second_derivative
        start, end, peak, first_neg = self._start, self._end, self._peak, self._first_neg
        local_alpha = self._local_alpha
        median_alpha = self._median_alpha
        window_alpha, deriv_alpha = self._window_alpha, self._window_deriv_alpha
        small_fontsize = 0.8 * fontsize

        fig, ax = self.new_subplots(1, 4, figsize=self.figsize, layout="constrained")

        # --- ax[0]: smoothed spectrum with the adaptive-window landmark vlines ---
        # ``peak``/``start``/``end``/``first_neg`` are 0-based array indices; the plotted axis is
        # 1-based dim numbers, so a +1 converts (matching the ``axvspan(start + 1, end)`` convention
        # used elsewhere in this module). ``first_neg`` (raw spectrum) is what ``end`` is buffered from.
        xv_full = np.arange(smoothed.shape[0]) + 1
        raw_positive = np.where(raw > 0, raw, np.nan)
        smoothed_positive = np.where(smoothed > 0, smoothed, np.nan)
        ax[0].plot(xv_full, raw_positive, color="0.7", linewidth=1.0, label="raw")
        ax[0].plot(xv_full, smoothed_positive, color="black", linewidth=1.5, label="smoothed")
        vline_specs = [
            (peak + 1, "orange", "peak 2nd deriv"),
            (first_neg, "0.5", "first negative (raw)"),
            (start + 1, "blue", "window start"),
            (end, "red", "window end"),
        ]
        for xpos, color, vlabel in vline_specs:
            if 0 < xpos <= smoothed.shape[0]:
                ax[0].axvline(xpos, color=color, linestyle="--", linewidth=1.0, label=vlabel)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(bottom=10**ylim_min)
        yticks = ax[0].get_yticks()
        ax[0].set_yticks(yticks, labels=[np.log10(yt) for yt in yticks], fontsize=fontsize)
        ax[0].set_ylim(bottom=10**ylim_min)
        ax[0].set_xlabel("Shared Dimension", fontsize=fontsize)
        ax[0].set_ylabel("Variance", fontsize=fontsize)
        ax[0].tick_params(axis="both", which="both", labelsize=fontsize)
        ax[0].set_title(f"{state['session']} : {key}", fontsize=fontsize)
        ax[0].legend(loc="upper right", fontsize=small_fontsize, frameon=False)

        # --- ax[1]: FPD local exponent restricted to the adaptive window, median marked ---
        if end > start:
            window_mask = (xv_full >= start + 1) & (xv_full <= end)
            ax[1].plot(xv_full[window_mask], local_alpha[window_mask], color="purple", linewidth=1.5, label="FPD (window)")
        alpha_label = "median (NaN: too few values)" if np.isnan(median_alpha) else f"median ({median_alpha:.2f})"
        if not np.isnan(median_alpha):
            ax[1].axhline(median_alpha, color="red", linestyle="--", linewidth=1.0, label=alpha_label)
        else:
            ax[1].text(0.5, 0.5, alpha_label, transform=ax[1].transAxes, ha="center", va="center", color="red", fontsize=fontsize)
        ax[1].set_xlabel("Dim", fontsize=fontsize)
        ax[1].set_ylabel("Power-law exponent", fontsize=fontsize)
        ax[1].tick_params(axis="both", which="both", labelsize=fontsize)
        ax[1].set_title("Adaptive window (median FPD)", fontsize=fontsize)
        ax[1].legend(loc="best", fontsize=small_fontsize, frameon=False)

        # --- ax[2]: fixed-window comparison methods (window fit, window-avg FPD) ---
        ax[2].plot(xv_full, local_alpha, color="black", linewidth=1.5, label="5-pt deriv")
        ax[2].axvspan(fit_start + 1, fit_end, color="0.8", alpha=0.4)
        ax[2].axhline(window_alpha, color="blue", linestyle="--", linewidth=1.0, label=f"window fit ({window_alpha:.2f})")
        ax[2].axhline(deriv_alpha, color="orange", linestyle="--", linewidth=1.0, label=f"window-avg FPD ({deriv_alpha:.2f})")
        ax[2].set_xscale("log")
        ax[2].set_ylim(-1, 10)
        ax[2].set_xlabel("Shared Dimension", fontsize=fontsize)
        ax[2].set_ylabel("Local exponent", fontsize=fontsize)
        ax[2].tick_params(axis="both", which="both", labelsize=fontsize)
        ax[2].set_title("Fixed-window methods", fontsize=fontsize)
        ax[2].legend(loc="upper left", fontsize=small_fontsize, frameon=False)

        ax[3].plot(xv_full[1:], first_derivative, color="blue", linewidth=1.5, label="1st deriv")
        ax[3].plot(xv_full[2:], second_derivative, color="orange", linewidth=1.5, label="2nd deriv")
        ax[3].set_xscale("log")
        ax[3].set_xlabel("Shared Dimension", fontsize=fontsize)
        ax[3].set_ylabel("Derivative", fontsize=fontsize)
        ax[3].tick_params(axis="both", which="both", labelsize=fontsize)
        ax[3].set_title("Derivatives of smoothed spectrum", fontsize=fontsize)
        ax[3].legend(loc="upper right", fontsize=small_fontsize, frameon=False)
        return fig
