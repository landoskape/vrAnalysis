"""Session examples of measured and fitted placefield spectra with decay-law overlays."""

import numpy as np
from matplotlib.ticker import LogFormatterMathtext, NullLocator

from vrAnalysis.helpers.plotting import format_spines
from ._alpha_config import ADAPTIVE_ALPHA_CONFIG_REGISTRY, AdaptiveAlphaConfig
from ._param_axes import (
    CVPCA_KEYS, STIMSPACE_KEYS, POP_ALPHA_LABELS, PREFERRED_DEFAULTS, SOURCE_OF_KEY,
    add_merged_param_axis_widgets, merged_axis_names,
)
from ._placefield_spectrum_access import PlacefieldSpectrumAccess
from ._spectrum_math import (
    _clamp_range, _decay_feature, _decay_window_session,
    _logspace_decay_fit, _median_fpd_alpha_session,
)
from ..legends import add_legend_widgets, apply_legend, update_legend_widgets
from ..panels import FigureViewer


_MATCHED_SOURCES = {
    "matched_ss_cvpca": "ss_cvpca_matched",
    "matched_pca_train": "eig_spatial_raw_train",
    "matched_pca_test": "eig_spatial_raw_test",
}


def _rescaled_model_spectra(fit, covariance_axis="positions", center=True):
    """Scale each neuron's fitted curve before forming its sample covariance.

    A neuron-dependent activity conversion cannot be applied to eigenvalues:
    it changes both the spectrum's amplitude and its shape. Use the common
    successful-neuron set from TilburyFitConfig.process for every model.
    """
    from dimensionality_manuscript.configs.tilbury_fit import _eval_gaussian, _eval_tilbury

    theta = fit["theta"]
    valid = np.logical_and.reduce([
        np.isfinite(fit[key]).all(axis=1)
        for key in ("params", "params_control", "params_shrinkage")
    ])
    scale = fit["fluorescence_scale"][valid, None]
    matrices = {
        "eig_control": np.asarray([_eval_gaussian(theta, p) for p in fit["params_control"][valid]]),
        "eig_tilbury": np.asarray([_eval_tilbury(theta, p) for p in fit["params"][valid]]),
        "eig_shrinkage": np.asarray([_eval_tilbury(theta, p) for p in fit["params_shrinkage"][valid]]),
    }
    if "r2_val" in fit:
        better = fit["r2_val"][valid] > fit["r2_val_control"][valid]
        matrices["eig_better"] = np.where(better[:, None], matrices["eig_tilbury"], matrices["eig_control"])
    spectra = {}
    for key, matrix in matrices.items():
        if scale.shape[0] < 2 or theta.size < 2:
            spectra[key] = np.full(theta.size, np.nan)
            continue
        matrix = matrix * scale
        # ss_cvpca treats neurons as samples of spatial features. ss_cv and
        # ss_direct instead treat positions as samples of neural features.
        axis = 0 if covariance_axis == "neurons" else 1
        if center:
            matrix -= matrix.mean(axis=axis, keepdims=True)
        spectra[key] = np.linalg.svd(matrix, compute_uv=False) ** 2 / (matrix.shape[axis] - int(center))
    return spectra


def _example_decay(raw, smooth, model, state, cfg, fallback=None):
    """Return a parameter, prediction, and zero-based, end-exclusive fit window."""
    empty = np.full(raw.shape, np.nan)
    if state["fit_zone"] == "fixed":
        start, end = state["fixed_range"]
    else:
        fb_raw, fb_smooth = fallback if fallback is not None else (None, None)
        start, end = _decay_window_session(raw, smooth, cfg.adaptive_buffer, fb_raw, fb_smooth)
    if end <= start:
        return np.nan, empty, (0, 0)
    start, end = _clamp_range(start, end, raw.size)
    if model == "none":
        return np.nan, empty, (start, end)
    ranks = np.arange(1, raw.size + 1, dtype=float)
    indices = np.arange(start, end)
    indices = indices[np.isfinite(smooth[indices]) & (smooth[indices] > 0)]
    if indices.size < 3:
        return np.nan, empty, (start, end)
    feature = _decay_feature(model, ranks)
    if model == "power" and state["alpha_method"] == "median_fpd":
        if state["fit_zone"] == "adaptive":
            fb_raw, fb_smooth = fallback if fallback is not None else (None, None)
            parameter = _median_fpd_alpha_session(
                raw, smooth, cfg.fpd_window_size, cfg.adaptive_buffer,
                cfg.minimum_window_size, fb_raw, fb_smooth,
            )
        else:
            from ._spectrum_math import _local_alpha_curve
            local = _local_alpha_curve(smooth[None], cfg.fpd_window_size)[0, start:end]
            finite = local[np.isfinite(local)]
            parameter = np.median(finite) if finite.size >= cfg.minimum_window_size else np.nan
        slope = -parameter
        intercept = np.mean(np.log(smooth[indices]) - slope * feature[indices])
    else:
        _, parameter = _logspace_decay_fit(smooth, start, end, model)
        slope, intercept = np.polyfit(feature[indices], np.log(smooth[indices]), 1)
    if not np.isfinite(parameter):
        return np.nan, empty, (start, end)
    with np.errstate(over="ignore", under="ignore"):
        prediction = np.exp(intercept + slope * feature)
    return parameter, prediction, (start, end)


class PlacefieldSpectrumExampleViewer(PlacefieldSpectrumAccess, FigureViewer):
    """One session's data, Gaussian, and generalized placefield eigenspectra.

    Takes the same fit/spectrum aggregators and shared parameter-axis defaults as
    PlacefieldPopulationViewer. ``session_id`` is an ID from ``results.session_ids``;
    the widget selects it directly. Sources are matched by ID, never row number.
    Missing source sessions are omitted and identified on the plot.

    With v11 results, the default ``source_key="matched_ss_cvpca"`` reads stored
    four-fold data cvPCA and ordinary spatial PCA of the models on the exact same
    successful neurons, environment, positions, and activity units. No activity
    reload or display-time conversion is needed. ``matched_pca_train`` and
    ``matched_pca_test`` instead show ordinary spatial PCA of measured curves.
    ``comparison_metadata`` exposes the selected stored units/support conventions.

    External sources remain available: StimSpace is labelled "Pooled data" for
    v11 results. For these comparisons fitted curves are converted neuron by
    neuron into the source spectrum's activity units, using full-recording
    scaling statistics for StimSpace.
    Covariance centering and denominator also follow the selected source:
    ss_cvpca centers across neurons and divides by N-1; ss_cv/ss_direct center
    across positions and divide by P-1. ``registry`` supplies the
    original activity and deterministic splits; defaults to PopulationRegistry
    with autosave disabled. Pass the registry used to build your results if it
    uses custom settings. Converted spectra are cached for the selected session.

    The stored StimSpace data pool environments and select reliable/active
    neurons using all trials; fits use the best-sampled environment and a
    train-split selection, further restricted to successful fits. Matching
    units and covariance conventions does not remove those differences.

    Positive values are dots; negative *raw data* values are crosses pinned to
    the lower y-limit. ``show_unfiltered=False`` displays the smoothed positive
    curves while retaining those negative markers. Smoothing affects estimation
    in either mode. Normalization divides each session/curve by its signed sum,
    matching PlacefieldPopulationViewer.

    ``force_scale=True`` multiplies each model spectrum and its decay overlay
    by exp(mean(log(data / model))) within that model's fit window, minimizing
    squared log-space error against the data spectrum. Both spectra use the
    selected estimation smoothing; ranks with nonpositive raw data or invalid
    model/data values are excluded. No overlap leaves the scale at 1. This is
    a display adjustment: alpha, M, fit windows, and stored spectra are unchanged.
    Applied factors are available in ``_example_fits[key]['scale_factor']``.

    Default overlays fit the Gaussian with exp(-n**2 / (2*M**2)), and data and
    generalized spectra with n**(-alpha). Dashed lines extrapolate the fitted law;
    solid lines mark the fitted interval. ``alpha_method='regression'`` matches
    the population viewer's characteristic_dim statistic; ``'median_fpd'`` uses
    its adaptive decay_exponent estimator and all source_cfg fields instead.
    Adaptive model windows borrow the same session's ss_cvpca window when the
    model has no negative entries. Fixed ranges are zero-based [start, end).

    ``ylim_log10=(-8, 0)`` means actual limits (1e-8, 1). ``xlim`` uses component
    ranks, with a log x-axis by default. Colors, marker size, line width, font
    size, fit guides, title, and the standard legend knobs are editable widgets.
    """

    def __init__(
        self, results, results_spectra=None, results_cvpca=None, *,
        session_id=None, source_key=None, generalized_fit="shrinkage", registry=None,
        include_better=False, source_cfg: AdaptiveAlphaConfig | None = None,
        fit_zone="adaptive", fixed_range=(10, 50), alpha_method="regression",
        gaussian_decay="exp2", generalized_decay="power", data_decay="power",
        normalize=False, force_scale=False, show_unfiltered=True, show_negative=True,
        ylim_log10=(-8, 0), xlim=(1, 80), xscale="log",
        show_fit_window=False, show_title=False,
        data_color="#4c7898", gaussian_color="#70a35b", generalized_color="#b75a50",
        shrinkage_color="purple", better_color="red", data_zorder=3.0,
        markersize=2.5, linewidth=1.25, xticks=(1, 50), legend=None, fontsize=7.0,
        figsize=(2.5, 2.5), **selection_defaults,
    ):
        self.results = self.results_fit = results
        self.results_spectra = results_spectra
        self.results_cvpca = results_cvpca
        self.registry = registry
        self._rescaled_spectra_cache = {}
        self._agg = {"stimspace": results_spectra, "cvpca": results_cvpca}
        self._fit_axes = list(results.param_axes)
        self.figsize = figsize
        self.xticks = tuple(xticks) if xticks is not None else None
        cfg = source_cfg or ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]
        self.source_cfg = cfg
        self._session_ids = {str(sid): sid for sid in results.session_ids}
        if not self._session_ids or len(self._session_ids) != len(results.session_ids):
            raise ValueError("Fit results must have nonempty, unique session IDs.")
        self.add_selection("session_id", options=list(self._session_ids),
                           value=str(session_id) if session_id is not None else next(iter(self._session_ids)))
        self._has_matched = (getattr(results.config_class, "schema_version", None) != "v10"
                             and "ss_cvpca_matched" in results.arrays)
        source_options = list(_MATCHED_SOURCES) if self._has_matched else []
        source_options += list(STIMSPACE_KEYS) if results_spectra is not None else []
        if results_cvpca is not None:
            source_options += list(CVPCA_KEYS)
        if source_options:
            if source_key is None:
                source_key = "matched_ss_cvpca" if self._has_matched else "ss_cvpca" if "ss_cvpca" in source_options else source_options[0]
            self.add_selection("source_key", options=source_options, value=source_key)
        self._tuple_labels = add_merged_param_axis_widgets(
            self, results_spectra, results_cvpca, results,
            preferred_defaults={**PREFERRED_DEFAULTS,
                                **({"activity_parameters_name": "std"} if self._has_matched else {}),
                                **selection_defaults},
        )
        self.add_selection("generalized_fit", options=["both", "generalized", "shrinkage"], value=generalized_fit)
        self.add_boolean("include_better", value=include_better)
        self.add_boolean("normalize", value=normalize)
        self.add_boolean("force_scale", value=force_scale)
        self.add_selection("fit_zone", options=["adaptive", "fixed"], value=fit_zone)
        self.add_integer_range("fixed_range", value=tuple(fixed_range), min=0, max=max(500, fixed_range[1]))
        self.add_selection("alpha_method", options=["regression", "median_fpd"], value=alpha_method)
        for name, value in (("gaussian_decay", gaussian_decay), ("generalized_decay", generalized_decay), ("data_decay", data_decay)):
            self.add_selection(name, options=["none", "power", "exp2"], value=value)
        self.add_selection("source_smooth_method", options=["none", "boxcar", "gaussian"], value=cfg.smooth_method)
        self.add_float("source_smooth_width", value=cfg.smooth_width, min=0., max=50., step=.5)
        self.add_integer("source_fpd_window_size", value=cfg.fpd_window_size, min=1, max=50)
        self.add_integer("source_adaptive_buffer", value=cfg.adaptive_buffer, min=0, max=50)
        self.add_integer("source_minimum_window_size", value=cfg.minimum_window_size, min=1, max=500)
        self.add_integer_range("ylim_log10", value=tuple(ylim_log10), min=-20, max=10)
        self.add_integer_range("xlim", value=tuple(xlim), min=1, max=max(500, xlim[1]))
        self.add_selection("xscale", options=["log", "linear"], value=xscale)
        for name, value in (("show_unfiltered", show_unfiltered), ("show_negative", show_negative),
                            ("show_fit_window", show_fit_window), ("show_title", show_title)):
            self.add_boolean(name, value=value)
        for name, value in (("data_color", data_color), ("gaussian_color", gaussian_color),
                            ("generalized_color", generalized_color), ("shrinkage_color", shrinkage_color),
                            ("better_color", better_color)):
            self.add_text(name, value=value)
        self.add_float("markersize", value=markersize, min=0., max=15.)
        self.add_float("linewidth", value=linewidth, min=0., max=10.)
        self.add_float("data_zorder", value=data_zorder, min=0., max=20.)
        self.add_float("fontsize", value=fontsize, min=4., max=24.)
        add_legend_widgets(self)
        update_legend_widgets(self, {"loc": "lower left", "frameon": True, **(legend or {})})
        self._load_names = [
            "session_id", "generalized_fit", "include_better", "normalize", "source_smooth_method", "source_smooth_width",
            *merged_axis_names(results_spectra, results_cvpca, results),
        ] + (["source_key"] if source_options else [])
        for name in self._load_names:
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _stored_spectrum_sessions(self, state, key, cfg):
        selected = self.results_fit.sel(keys=[key], squeeze_ones=False, **self._fit_sel_params(state))
        if key not in selected:
            raise ValueError(f"Missing {key!r}; compute TilburyFitConfig v11 for this activity selection.")
        return (*self._preprocess(selected[key], state, cfg),
                self.results_fit.mouse_names, self.results_fit.session_ids)

    def _fit_spectrum_sessions(self, state, key, cfg):
        if state.get("source_key") in _MATCHED_SOURCES:
            return self._stored_spectrum_sessions(state, key.replace("eig_", "eig_spatial_", 1), cfg)
        fit_params = self._fit_sel_params(state)
        fit_preset = fit_params.get("activity_parameters_name", getattr(self.results_fit.config_class, "activity_parameters_name", None))
        source = SOURCE_OF_KEY[state["source_key"]] if "source_key" in state else None
        agg = self._agg[source] if source is not None else None
        target = self._sel_params(state, source).get(
            "activity_parameters_name", getattr(agg.config_class, "activity_parameters_name", None)
        ) if agg is not None else fit_preset
        if target is None and fit_preset is None:
            # Lightweight precomputed aggregators without activity metadata.
            return super()._fit_spectrum_sessions(state, key, cfg)
        if target is None or fit_preset is None:
            raise ValueError("Activity presets are required to convert fitted spectra to data units.")
        sid = self._session_ids[state["session_id"]]
        source_params = self._sel_params(state, source) if source is not None else {}
        spatial = state.get("source_key") == "ss_cvpca" or (
            source == "cvpca" and source_params.get(
                "use_spatial_eigenvectors", getattr(agg.config_class, "use_spatial_eigenvectors", True)
            )
        )
        center = source_params.get("center", getattr(agg.config_class, "center", True)) if source == "cvpca" else True
        covariance_axis = "neurons" if spatial else "positions"
        full_session = source == "stimspace"
        if target == fit_preset and not full_session and not spatial and center:
            return super()._fit_spectrum_sessions(state, key, cfg)
        cache_key = (sid, repr(sorted(fit_params.items())), target, state["include_better"], covariance_axis, center, full_session)
        if cache_key not in self._rescaled_spectra_cache:
            from dimensionality_manuscript.registry import PopulationRegistry
            from .placefield_fits import _load_fit_curves

            if self.registry is None:
                self.registry = PopulationRegistry(autosave=False)
            fit = _load_fit_curves(
                self.results_fit, self.registry, sid, fit_params,
                extra_keys=("r2_val", "r2_val_control") if state["include_better"] else (),
                target_activity_parameters_name=target,
                target_full_session=full_session,
            )
            self._rescaled_spectra_cache[cache_key] = _rescaled_model_spectra(fit, covariance_axis, center)
        raw = self._rescaled_spectra_cache[cache_key][key][None, :]
        i = list(self.results_fit.session_ids).index(sid)
        return (*self._preprocess(raw, state, cfg), np.asarray(self.results_fit.mouse_names)[[i]], [sid])

    def refresh_data(self, state):
        cfg = self._cfg_from_state(state, "source")
        self._spectra = {}
        source = state.get("source_key")
        self.comparison_metadata = None
        self._data_label = "Data"
        if source in _MATCHED_SOURCES:
            self._data_label = "Matched data"
            self._spectra["data"] = self._stored_spectrum_sessions(state, _MATCHED_SOURCES[source], cfg)
            self._fallback = self._stored_spectrum_sessions(state, "ss_cvpca_matched", cfg)
            objects = self.results_fit.sel_objects(keys=["spectrum_metadata"], **self._fit_sel_params(state))
            if "spectrum_metadata" in objects:
                i = list(self.results_fit.session_ids).index(self._session_ids[state["session_id"]])
                self.comparison_metadata = objects["spectrum_metadata"][i]
        else:
            if source is not None:
                if self._has_matched and SOURCE_OF_KEY[source] == "stimspace":
                    self._data_label = "Pooled data"
                self._spectra["data"] = self._spectrum_sessions(state, source, cfg)
            if self.results_spectra is not None and "ss_cvpca" in self.results_spectra.arrays:
                self._fallback = self._spectrum_sessions(state, "ss_cvpca", cfg)
            else:
                self._fallback = None
        keys = ["eig_control"]
        if state["generalized_fit"] in ("both", "generalized"):
            keys.append("eig_tilbury")
        if state["generalized_fit"] in ("both", "shrinkage"):
            keys.append("eig_shrinkage")
        if state["include_better"]:
            keys.append("eig_better")
        for key in keys:
            self._spectra[key] = self._fit_spectrum_sessions(state, key, cfg)
        self._loaded_state = tuple(state[name] for name in self._load_names)

    @staticmethod
    def _session_rows(spectra, sid):
        if spectra is None or sid not in spectra[3]:
            return None
        i = list(spectra[3]).index(sid)
        if not np.isfinite(spectra[0][i]).any():
            return None
        return spectra[0][i], spectra[1][i]

    def plot(self, state):
        if tuple(state[name] for name in self._load_names) != self._loaded_state:
            self.refresh_data(state)
        sid = self._session_ids[state["session_id"]]
        cfg = self._cfg_from_state(state, "source")
        fallback = self._session_rows(self._fallback, sid)
        data_rows = self._session_rows(self._spectra.get("data"), sid)
        fig, ax = self.new_subplots(figsize=self.figsize, layout="constrained")
        missing = []
        self._example_fits = {}
        order = [key for key in self._spectra if key != "data"]
        if "data" in self._spectra:
            order.append("data")
        for key in order:
            spectra = self._spectra[key]
            rows = self._session_rows(spectra, sid)
            if rows is None:
                missing.append("Data" if key == "data" else POP_ALPHA_LABELS[key])
                continue
            raw, smooth = rows
            if key == "data":
                label, color, decay = self._data_label, state["data_color"], state["data_decay"]
            elif key == "eig_control":
                label, color, decay = "Gaussian", state["gaussian_color"], state["gaussian_decay"]
            else:
                label = POP_ALPHA_LABELS[key] if state["generalized_fit"] == "both" or key == "eig_better" else "Generalized"
                color_key = "better_color" if key == "eig_better" else "shrinkage_color" if key == "eig_shrinkage" and state["generalized_fit"] == "both" else "generalized_color"
                color, decay = state[color_key], state["generalized_decay"]
            parameter, prediction, window = _example_decay(raw, smooth, decay, state, cfg, fallback)
            scale_factor = 1.0
            if state["force_scale"] and key != "data" and data_rows is not None:
                start, end = window
                end = min(end, smooth.size, data_rows[0].size, data_rows[1].size)
                model_zone = smooth[start:end]
                data_zone = data_rows[1][start:end]
                valid = (np.isfinite(model_zone) & (model_zone > 0)
                         & np.isfinite(data_zone) & (data_zone > 0)
                         & (data_rows[0][start:end] > 0))
                if np.any(valid):
                    scale_factor = float(np.exp(np.mean(np.log(data_zone[valid]) - np.log(model_zone[valid]))))
                prediction = prediction * scale_factor
            self._example_fits[key] = dict(parameter=parameter, prediction=prediction, window=window,
                                           model=decay, scale_factor=scale_factor)
            if decay != "none":
                symbol = r"\alpha" if decay == "power" else "M"
                label += f": ${symbol} = {parameter:.2f}$" if np.isfinite(parameter) else ": fit unavailable"
            x = np.arange(1, raw.size + 1)
            shown = (raw if state["show_unfiltered"] else smooth) * scale_factor
            zorder = state["data_zorder"] if key == "data" else 2.
            ax.plot(x, np.where(shown > 0, shown, np.nan), linestyle="none", marker="o",
                    markersize=state["markersize"], color=color, label=label, zorder=zorder)
            ax.plot(x, prediction, "--", color=color, alpha=.4, linewidth=state["linewidth"], zorder=1.)
            start, end = window
            ax.plot(x[start:end], prediction[start:end], color=color, linewidth=state["linewidth"], zorder=2.)
            if state["show_fit_window"] and end > start:
                for boundary in (start + 1, end):
                    ax.axvline(boundary, color=color, alpha=.25, linestyle=":", linewidth=.7, zorder=0)
            if key == "data" and state["show_negative"]:
                negative = (np.isfinite(raw) & (raw < 0)
                            & (x >= state["xlim"][0]) & (x <= state["xlim"][1]))
                bottom = np.full(np.count_nonzero(negative), 10. ** state["ylim_log10"][0])
                ax.plot(x[negative], bottom, linestyle="none", marker="x", color=color, clip_on=False,
                        markersize=state["markersize"] + 1, markeredgewidth=.7,
                        label=f"{self._data_label}: -ve values", zorder=zorder)
        ax.set(xscale=state["xscale"], yscale="log", xlim=state["xlim"],
               ylim=10. ** np.asarray(state["ylim_log10"], dtype=float),
               xlabel="Component", ylabel="Variance explained")
        ax.set_yticks(10. ** np.asarray(state["ylim_log10"], dtype=float))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_locator(NullLocator())
        if self.xticks is not None:
            ticks = [tick for tick in self.xticks if state["xlim"][0] <= tick <= state["xlim"][1]]
            ax.set_xticks(ticks, labels=[f"{tick:g}" for tick in ticks])
        if state["show_title"]:
            ax.set_title(str(sid), fontsize=state["fontsize"])
        if missing:
            ax.text(.02, .98, f"Unavailable: {', '.join(missing)}", transform=ax.transAxes,
                    va="top", fontsize=state["fontsize"])
        format_spines(ax, x_pos=0, y_pos=0)
        ax.tick_params(labelsize=state["fontsize"])
        ax.xaxis.label.set_size(state["fontsize"])
        ax.yaxis.label.set_size(state["fontsize"])
        apply_legend(ax, state, state["fontsize"], auto_loc="lower left")
        return fig
