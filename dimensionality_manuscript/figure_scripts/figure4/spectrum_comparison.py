"""Compact comparison of spectrum estimators, dimensionality, and adaptive alpha."""

import numpy as np
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator

from dimensionality_manuscript import average_by_mouse
from vrAnalysis.helpers.plotting import beeswarm, format_spines

from ._alpha_config import ADAPTIVE_ALPHA_CONFIG_REGISTRY, SpectrumSmoothingConfig
from ._param_axes import PREFERRED_DEFAULTS, add_merged_param_axis_widgets, merged_axis_names
from ._spectrum_math import (
    _align_rows_to_sessions,
    _median_fpd_alpha_per_session,
    _signed_participation_ratio,
)
from .spectrum_figure import SpectrumFigureViewer, SVCA_PF_KEYS, SVCA_FULL_KEYS


METHOD_LABELS = {
    "sf_cv": "PF x CA1",
    "ss_cv": "PF x PF",
    "ss_cvpca": "r-cvPCA",
    "ff": "CA1 x CA1",
    "SVCA": "SVCA",
    "reg_covariances_fixed": "reg_covariances_fixed",
}


class SpectrumComparisonViewer(SpectrumFigureViewer):
    """Two rows (placefields/full activity) by spectra, dimensionality, and alpha.

    Thin curves are session-averaged mouse spectra; thick curves are their arithmetic
    mean. Dimensionality is signed participation ratio of each mouse spectrum (before
    smoothing by default). Alpha uses the existing adaptive median five-point-derivative
    estimator per session, then averages by mouse. Non-cross-validated spectra borrow
    the window from session-aligned ss_cvpca (PF) or SVCA (full activity).

    ``source_mode='avg_env'`` averages independently computed PF environment spectra
    before averaging sessions. Full activity always uses whole-session spectra.
    Normalization divides each mouse spectrum by its signed sum before display
    smoothing, with no post-hoc amplitude matching between methods.

    ``show_unfiltered=True`` displays the unsmoothed mouse spectra regardless of
    smoothing controls. Those controls still govern pre-alpha smoothing (and PR
    when ``pr_pre_smooth=False``); normalization and clipping still apply.

    Summary axes run from zero to the ceiling of their finite data maximum.
    ``dimensionality_log`` applies only to full activity and starts that axis at
    one. All logarithmic axes use powers-of-ten labels with no minor ticks.

    Both primary aggregators are required. Optional ``results_cvpca`` supplies
    ``reg_covariances_fixed`` when ``include_reg_covariances_fixed=True``. This
    method always uses whole-session spectra (it has no per-environment results),
    including its ss_cvpca alpha-window fallback. Data-axis defaults
    beyond the named keywords can be passed through ``selection_defaults``.
    """

    def __init__(
        self,
        results,
        results_svca,
        *,
        results_cvpca=None,
        include_reg_covariances_fixed=False,
        source_mode="avg_env",
        activity_parameters_name="std",
        smooth_widths=(5.0, None),
        reliability_fraction_active_thresholds=(None, None),
        include_iti=False,
        normalize=True,
        pr_pre_smooth=True,
        clip_negative=False,
        show_unfiltered=False,
        source_cfg=None,
        full_cfg=None,
        ylim_min=-5.5,
        ylim_max=0.0,
        dimensionality_log=True,
        each_line_alpha=0.3,
        subject_linewidth=0.5,
        mean_linewidth=1.5,
        point_alpha=0.7,
        markersize=3.0,
        beewidth=0.15,
        fontsize=7.0,
        figsize=(7.2, 4.4),
        width_ratios=(1.6, 1.0, 1.0),
        method_colors=None,
        show_legend=True,
        **selection_defaults,
    ):
        if results_svca is None:
            raise ValueError("results_svca is required for the SVCA comparisons.")
        self.results = results
        self.results_cvpca = results_cvpca
        self.results_svca = self._svca_results = results_svca
        self._svca_pf_keys = SVCA_PF_KEYS
        self._svca_full_keys = SVCA_FULL_KEYS
        self._agg = {"stimspace": results, "svca": results_svca, "cvpca": results_cvpca}
        self.figsize, self.width_ratios = figsize, width_ratios
        # Okabe-Ito colors, held constant across all three columns and both rows.
        self.method_colors = {
            "sf_cv": "#E69F00", "ss_cv": "#0072B2", "ss_cvpca": "#CC79A7",
            "SVCA": "#009E73", "ff": "#333333",
            "reg_covariances_fixed": "#D55E00",
            **(method_colors or {}),
        }
        self.add_selection("source_mode", options=["all", "avg_env"], value=source_mode)
        defaults = dict(
            PREFERRED_DEFAULTS,
            activity_parameters_name=activity_parameters_name,
            smooth_widths=smooth_widths,
            reliability_fraction_active_thresholds=reliability_fraction_active_thresholds,
            include_iti=include_iti,
        )
        defaults.update(selection_defaults)
        self._tuple_labels = add_merged_param_axis_widgets(
            self, results, results_svca, results_cvpca, preferred_defaults=defaults,
        )
        for name, value in (
            ("normalize", normalize), ("pr_pre_smooth", pr_pre_smooth),
            ("clip_negative", clip_negative), ("dimensionality_log", dimensionality_log),
            ("show_legend", show_legend),
            ("show_unfiltered", show_unfiltered),
            ("include_reg_covariances_fixed", include_reg_covariances_fixed),
        ):
            self.add_boolean(name, value=value)
        for name, value, low, high in (
            ("ylim_min", ylim_min, -10., 2.), ("ylim_max", ylim_max, -10., 12.),
            ("each_line_alpha", each_line_alpha, 0., 1.),
            ("subject_linewidth", subject_linewidth, 0., 3.),
            ("mean_linewidth", mean_linewidth, .25, 6.),
            ("point_alpha", point_alpha, 0., 1.), ("markersize", markersize, .5, 12.),
            ("beewidth", beewidth, 0., .5), ("fontsize", fontsize, 4., 24.),
        ):
            self.add_float(name, value=value, min=low, max=high)
        for prefix, cfg in (
            ("source", source_cfg or ADAPTIVE_ALPHA_CONFIG_REGISTRY["placefields"]),
            ("full", full_cfg or ADAPTIVE_ALPHA_CONFIG_REGISTRY["full"]),
        ):
            self.add_selection(f"{prefix}_smooth_method", options=["none", "boxcar", "gaussian"], value=cfg.smooth_method)
            self.add_float(f"{prefix}_smooth_width", value=cfg.smooth_width, min=0., max=50.)
            for name, high in (("fpd_window_size", 50), ("adaptive_buffer", 50), ("minimum_window_size", 500)):
                self.add_integer(f"{prefix}_{name}", value=getattr(cfg, name), min=0 if name == "adaptive_buffer" else 1, max=high)
        refresh_names = [
            *merged_axis_names(results, results_svca, results_cvpca), "source_mode", "normalize",
            "pr_pre_smooth", "clip_negative", "include_reg_covariances_fixed",
            *(f"{prefix}_{name}" for prefix in ("source", "full") for name in (
                "smooth_method", "smooth_width", "fpd_window_size", "adaptive_buffer", "minimum_window_size",
            )),
        ]
        for name in refresh_names:
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Use the same aggregation and adaptive-alpha conventions as the main viewers."""
        raw_cfg = SpectrumSmoothingConfig("none", 0.)
        pf_keys = ["sf_cv", "ss_cvpca", "ss_cv"]
        if state["include_reg_covariances_fixed"]:
            if self.results_cvpca is None:
                raise ValueError("include_reg_covariances_fixed=True requires results_cvpca.")
            pf_keys.append("reg_covariances_fixed")
        rows = []
        for prefix, keys in (("source", pf_keys), ("full", ("ff", "SVCA"))):
            cfg = self._cfg_from_state(state, prefix)

            def session_data(key):
                if prefix == "source":
                    return self._spectrum_sessions(state, key, cfg, apply_source_mode=key != "reg_covariances_fixed")
                return self._ff_spectrum_sessions(dict(state, full_source_key="SVCA" if key == "SVCA" else "SVD"), cfg)

            fb_raw, fb_fit, _, fb_ids = session_data("ss_cvpca" if prefix == "source" else "SVCA")
            methods = {}
            for key in keys:
                if prefix == "source":
                    spec = self._spectrum(state, key, cfg, apply_source_mode=key != "reg_covariances_fixed")
                    raw = self._spectrum(state, key, raw_cfg, apply_source_mode=key != "reg_covariances_fixed")
                else:
                    selection = dict(state, full_source_key="SVCA" if key == "SVCA" else "SVD")
                    spec = self._ff_spectrum(selection, cfg)
                    raw = self._ff_spectrum(selection, raw_cfg)
                session_raw, session_fit, names, ids = session_data(key)
                window_raw, window_fit, window_ids = fb_raw, fb_fit, fb_ids
                if key == "reg_covariances_fixed":
                    window_raw, window_fit, _, window_ids = self._spectrum_sessions(
                        state, "ss_cvpca", cfg, apply_source_mode=False,
                    )
                alpha = _median_fpd_alpha_per_session(
                    session_raw, session_fit, cfg.fpd_window_size, cfg.adaptive_buffer,
                    cfg.minimum_window_size,
                    _align_rows_to_sessions(ids, window_ids, window_raw),
                    _align_rows_to_sessions(ids, window_ids, window_fit),
                )
                methods[key] = dict(
                    spectrum=spec,
                    raw_spectrum=raw,
                    dimensionality=_signed_participation_ratio(raw if state["pr_pre_smooth"] else spec),
                    alpha=average_by_mouse(alpha, names),
                )
            rows.append(methods)
        self._comparison_rows = rows

    def plot(self, state):
        fig, axes = self.new_subplots(2, 3, figsize=self.figsize, gridspec_kw=dict(width_ratios=self.width_ratios), layout="constrained")
        fontsize = state["fontsize"]
        for row, (title, methods) in enumerate(zip(("Placefields", "Full activity"), self._comparison_rows)):
            ax = axes[row, 0]
            for position, (key, data) in enumerate(methods.items()):
                color = self.method_colors[key]
                spec = data["raw_spectrum" if state["show_unfiltered"] else "spectrum"]
                ranks = np.arange(spec.shape[1]) + 1
                # Mask nonpositive values for log display without altering signed PR.
                shown = np.where(spec > 0, spec, np.nan)
                ax.loglog(ranks, shown.T, color=color, linewidth=state["subject_linewidth"], alpha=state["each_line_alpha"])
                count = np.isfinite(spec).sum(axis=0)
                mean = np.divide(np.nansum(spec, axis=0), count, out=np.full(spec.shape[1], np.nan), where=count > 0)
                ax.loglog(ranks, np.where(mean > 0, mean, np.nan), color=color, linewidth=state["mean_linewidth"], label=METHOD_LABELS[key])
                for col, metric in ((1, "dimensionality"), (2, "alpha")):
                    target = axes[row, col]
                    values = data[metric]
                    values = values[np.isfinite(values)]
                    if values.size:
                        target.plot(position + state["beewidth"] * beeswarm(values), values, "o", color=color, markersize=state["markersize"], alpha=state["point_alpha"])
                        target.plot(position + np.array([-.2, .2]), [values.mean()] * 2, color=color, linewidth=state["mean_linewidth"])
                    else:
                        target.text(position, .04, "n/a", transform=target.get_xaxis_transform(), ha="center", fontsize=fontsize, color=color)
            ax.set(title=title, xlabel="Rank", ylabel="Normalized variance" if state["normalize"] else "Variance", ylim=(10. ** state["ylim_min"], 10. ** state["ylim_max"]))
            if state["show_legend"]:
                ax.legend(frameon=False, fontsize=fontsize, loc="lower left", handlelength=1.2)
            for col, metric, label in ((1, "dimensionality", "Dimensionality (PR)"), (2, "alpha", r"Decay exponent $\alpha$")):
                target = axes[row, col]
                logarithmic = row == 1 and col == 1 and state["dimensionality_log"]
                if logarithmic:
                    target.set_yscale("log")
                values = np.concatenate([data[metric] for data in methods.values()])
                finite = values[np.isfinite(values)]
                upper = max(2 if logarithmic else 1, np.ceil(finite.max()) if finite.size else 1)
                target.set_ylim(1 if logarithmic else 0, upper)
                target.set_ylabel(label, fontsize=fontsize)
                target.set_xticks(range(len(methods)), [METHOD_LABELS[key] for key in methods], rotation=35, ha="right")
                target.set_xlim(-.5, len(methods) - .5)
        for ax in axes.flat:
            format_spines(ax, x_pos=-0.02, y_pos=-0.02, spines_visible=["left", "bottom"])
            ax.tick_params(axis="both", which="both", labelsize=fontsize)
            ax.xaxis.label.set_fontsize(fontsize)
            ax.yaxis.label.set_fontsize(fontsize)
            ax.title.set_fontsize(fontsize)
        for ax in axes[:, 1:].flat:
            ax.tick_params(axis="x", labelrotation=35)
        # All logarithmic axes show decade ticks only, including spectra axes.
        for ax in axes.flat:
            for axis, scale, limits in ((ax.xaxis, ax.get_xscale(), ax.get_xlim()), (ax.yaxis, ax.get_yscale(), ax.get_ylim())):
                if scale == "log":
                    powers = np.arange(np.ceil(np.log10(limits[0])), np.floor(np.log10(limits[1])) + 1)
                    axis.set_major_locator(FixedLocator(10. ** powers))
                    axis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
                    axis.set_minor_locator(NullLocator())
        return fig
