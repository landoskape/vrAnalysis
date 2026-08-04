"""Spectrum-by-session-familiarity panels: raw curves and participation-ratio dimensionality.

Two viewers, both indexing sessions by within-mouse experience order rather than by session
identity:

- :class:`SpectrumCurvesByFamiliarityViewer` -- PF and Full-CA1 spectra themselves, color-coded
  by session number (one example mouse plus a cross-mouse average per session number).
- :class:`SpectrumDimFamiliarityViewer` -- per-environment participation-ratio dimensionality
  over familiarity, one curve per environment experience-order slot.

Both read from a StimSpaceSpectra aggregator (``results``) and, for some ``full_key`` choices, a
subspace aggregator (``results_subspace``); :class:`SpectrumCurvesByFamiliarityViewer` additionally
accepts a CVPCA aggregator (``results_cvpca``). Which aggregator a spectrum key comes from is
resolved via :data:`~.SOURCE_OF_KEY`.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from vrAnalysis.helpers.plotting import errorPlot
from dimensionality_manuscript import ResultsAggregator
from dimensionality_manuscript.env_order import ENV_SLOT_COLORS, MAX_ENV_SLOTS

from ._param_axes import (
    CVPCA_KEYS,
    FF_KEY,
    PER_ENV_PF_KEYS as _PER_ENV_PF_KEYS,
    PER_ENV_PF_RESULT_KEYS as _PER_ENV_PF_RESULT_KEYS,
    PER_ENV_RESIDUAL_RESULT_KEYS as _PER_ENV_RESIDUAL_RESULT_KEYS,
    PER_ENV_SESSION_ALIGNMENTS as _PER_ENV_SESSION_ALIGNMENTS,
    PREFERRED_DEFAULTS,
    SOURCE_OF_KEY,
    STIMSPACE_KEYS,
    SVCA_FULL_ENV_KEYS,
    SVCA_FULL_KEYS,
    add_merged_param_axis_widgets,
    full_key_options as _full_key_options,
    merged_axis_names as _merged_axis_names,
    per_env_full_options as _per_env_full_options,
    sel_params,
)
from ._spectrum_math import _clip_at_first_negative, _signed_participation_ratio, _smooth_spectrum
from ..panels import FigureViewer, style_model_axis


class SpectrumCurvesByFamiliarityViewer(FigureViewer):
    """Four-panel spectrum view with curves colored by within-mouse session number.

    ax[0]/ax[2] show every available session for one example mouse (PF, then Full-CA1); ax[1]/ax[3]
    group the same spectra by within-mouse session number and average across mice (session groups
    with fewer than two mice are omitted). ``plot_mode`` selects the familiarity source: ``"all"``
    reads the overall spectrum directly; ``"avg_env"``/``"by_env"`` read the per-environment spectrum
    and either average across experience-order slots or show one selected slot. ``full_key="SVD_RES"``
    reads ``ffres`` in ``"all"`` mode and the scope-selected ``ffres_env_*`` key in the environment
    modes. When ``log_y`` is enabled, ``ymin`` is a base-10 exponent (``ymin=-4`` places the lower
    limit at ``10**-4``).

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the ``ss_*``/``sf_*``/``ff``/``ffres`` keys.
    results_cvpca : ResultsAggregator or None
        Aggregated CVPCAConfig results, source of the ``reg_covariances_fixed`` key. If None only
        StimSpace PF keys are selectable.
    results_subspace : ResultsAggregator or None
        Aggregated SubspaceConfig results, source of the ``"SVCA"``/``"variance_activity[_env]"``
        Full choice. Required for that choice; if None it is not selectable.
    source_key : str
        Which PF spectrum to show: one of ``ss_cv``, ``ss_direct``, ``ss_cvpca``, ``sf_cv``,
        ``sf_direct`` (from ``results``), or ``reg_covariances_fixed`` (from ``results_cvpca``).
    full_key : {"SVD", "SVD_RES", "SVCA", "SVCA_RES"}
        Which Full-CA1 estimator to show. ``SVCA_RES`` requires ``results_subspace`` with a
        discoverable ``variance_activity_residual`` key; in per-environment modes it reads
        ``variance_activity_residual_env``.
    full_scope : {"full1", "fullall"}
        For per-environment modes, whether the Full spectrum is fit within one environment
        (``full1``) or across all sessions (``fullall``).
    plot_mode : {"all", "avg_env", "by_env"}
        Familiarity source: overall spectrum, average over experience-order slots, or one slot
        (``env_slot``).
    session_alignment : {"within_env", "overall"}
        In ``"by_env"``, whether within-mouse session numbering is densified within the
        environment (``within_env``) or kept as the mouse's overall session index (``overall``).
    env_slot : int
        Experience-order slot shown in ``"by_env"`` mode.
    mouse : str or None
        Example mouse for ax[0]/ax[2]. If None, the first of ``results.unique_mice`` is used.
    plot_style : {"each", "errorPlot"}
        Population-panel (ax[1]/ax[3]) style: individual + mean lines, or mean +/- std band.
    hide_error : bool
        With ``"errorPlot"``, draw only the mean line.
    skip_sessions : int
        Thin the population panel's session groups, keeping every ``skip_sessions + 1``-th group
        (always keeping the first and last).
    normalize : bool
        Normalize each spectrum by its sum before smoothing/plotting.
    clip_negative : bool
        Replace each spectrum's first negative entry and all later ranks with NaN.
    curve_smooth_kind : {"none", "boxcar", "gaussian"}
        Log-space (geometric-mean) pre-smoothing applied to every spectrum.
    curve_smooth_width : float
        Boxcar full-width in rank units; the Gaussian uses ``sigma = curve_smooth_width / 2``.
    log_x, log_y : bool
        Log-scale the x/y axes.
    ymin : float
        Lower y-limit in base-10 exponent units, applied when ``log_y`` is True.
    fontsize : float
        Font size for every tick label, axis label, title, and colorbar label.
    figsize : tuple[float, float]
        Figure size in inches.
    pf_label, ff_label : str
        Panel-title labels for the PF and Full-CA1 columns.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_cvpca: ResultsAggregator | None = None,
        results_subspace: ResultsAggregator | None = None,
        *,
        source_key: str = "ss_cv",
        full_key: str = "SVD",
        full_scope: str = "full1",
        plot_mode: str = "all",
        session_alignment: str = "within_env",
        env_slot: int = 0,
        mouse: str | None = None,
        plot_style: str = "errorPlot",
        hide_error: bool = False,
        skip_sessions: int = 0,
        normalize: bool = True,
        clip_negative: bool = False,
        curve_smooth_kind: str = "none",
        curve_smooth_width: float = 3.0,
        log_x: bool = True,
        log_y: bool = True,
        ymin: float = -4.0,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (11.0, 2.8),
        pf_label: str = "Placefields",
        ff_label: str = "Full CA1",
        **selection_defaults,
    ):
        self.results = results
        self.results_cvpca = results_cvpca
        self.results_subspace = results_subspace
        self._agg = {"stimspace": results, "cvpca": results_cvpca}
        self.figsize = figsize
        self.pf_label = pf_label
        self.ff_label = ff_label

        pf_options = list(STIMSPACE_KEYS)
        if results_cvpca is not None:
            pf_options += list(CVPCA_KEYS)
        self.add_selection("source_key", options=pf_options, value=source_key)
        self.add_selection("full_key", options=_full_key_options(results, results_subspace), value=full_key)
        self.add_selection("full_scope", options=["full1", "fullall"], value=full_scope)
        self.add_selection("plot_mode", options=["all", "avg_env", "by_env"], value=plot_mode)
        self.add_selection("session_alignment", options=_PER_ENV_SESSION_ALIGNMENTS, value=session_alignment)

        # One widget per param-axis name, shared across the stimspace/cvpca sources; tuple-valued
        # axes (e.g. smooth_widths) are encoded as string labels -- see add_merged_param_axis_widgets.
        # ``selection_defaults`` lets a caller seed any of these axes by name.
        self._tuple_labels = add_merged_param_axis_widgets(
            self, results, results_cvpca, preferred_defaults={**PREFERRED_DEFAULTS, **selection_defaults}
        )
        self._param_axis_names = _merged_axis_names(results, results_cvpca)

        mouse_options = list(results.unique_mice)
        if mouse is not None:
            self.add_selection("mouse", options=mouse_options, value=mouse)
        else:
            self.add_selection("mouse", options=mouse_options)
        self.add_integer("env_slot", value=int(env_slot), min=0, max=MAX_ENV_SLOTS - 1)
        self.add_selection("plot_style", options=["each", "errorPlot"], value=plot_style)
        self.add_boolean("hide_error", value=hide_error)
        self.add_integer("skip_sessions", value=int(skip_sessions), min=0, max=max(len(results.sessions), 1))
        self.add_boolean("normalize", value=normalize)
        self.add_boolean("clip_negative", value=clip_negative)
        self.add_selection("curve_smooth_kind", options=["none", "boxcar", "gaussian"], value=curve_smooth_kind)
        self.add_float("curve_smooth_width", value=curve_smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_boolean("log_x", value=log_x)
        self.add_boolean("log_y", value=log_y)
        self.add_float("ymin", value=ymin, min=-12.0, max=2.0, step=0.25)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        for name in (
            "source_key",
            "full_key",
            "full_scope",
            "plot_mode",
            "session_alignment",
            "env_slot",
            *self._param_axis_names,
            "normalize",
            "clip_negative",
            "curve_smooth_kind",
            "curve_smooth_width",
        ):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _spectrum_sessions(self, state: dict, key: str) -> tuple[np.ndarray, ResultsAggregator]:
        """Return the selected PF spectrum as ``(sessions, dimensions)`` and its aggregator."""
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        spec = agg.sel(keys=[key], avg_by_mouse=False, **sel_params(state, self._tuple_labels, agg.param_axes))[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["clip_negative"]:
            spec = _clip_at_first_negative(spec)
        return spec, agg

    def _full_spectrum_sessions(self, state: dict) -> tuple[np.ndarray, ResultsAggregator]:
        """Return the selected Full-CA1 spectrum and the aggregator supplying its session rows."""
        full_key = state["full_key"]
        if full_key == "SVD":
            return self._spectrum_sessions(state, FF_KEY)
        if full_key == "SVD_RES":
            if "ffres" not in self.results.arrays:
                raise ValueError("The overall residual spectrum 'ffres' is not present in results.")
            return self._spectrum_sessions(state, "ffres")

        params = {"subspace_name": "svca_subspace", "smooth_width": None}
        if "activity_parameters_name" in state:
            params["activity_parameters_name"] = state["activity_parameters_name"]
        key = SVCA_FULL_KEYS[full_key]
        spec = self.results_subspace.sel(keys=[key], avg_by_mouse=False, **params)[key]
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state["clip_negative"]:
            spec = _clip_at_first_negative(spec)
        return spec, self.results_subspace

    @staticmethod
    def _per_env_result_keys(source_key: str, full_key: str, full_scope: str) -> tuple[str, str]:
        """Map the public per-environment choices to stored spectrum keys."""
        if source_key not in _PER_ENV_PF_KEYS:
            raise ValueError(f"source_key={source_key!r} is unavailable with plot_mode='by_env'. Options: {_PER_ENV_PF_KEYS}")
        use_fullall = full_scope == "fullall"
        pf_key = f"{source_key}_env_{full_scope}" if source_key in ("sf_cv", "sf_direct") else _PER_ENV_PF_RESULT_KEYS[source_key]
        if full_key in SVCA_FULL_ENV_KEYS:
            full_stored_key = SVCA_FULL_ENV_KEYS[full_key]
        elif full_key == "SVD_RES":
            full_stored_key = "ffres_env_full1_fullall" if use_fullall else "ffres_env_full1"
        else:
            full_stored_key = "ff_env_full1_fullall" if use_fullall else "ff_env_full1"
        return pf_key, full_stored_key

    def _per_env_spectra(self, state: dict) -> tuple[np.ndarray, ResultsAggregator, np.ndarray, ResultsAggregator]:
        """Return PF and Full spectra with shape ``(sessions, env slots, dimensions)``."""
        pf_key, full_key = self._per_env_result_keys(state["source_key"], state["full_key"], state["full_scope"])
        stimspace_params = sel_params(state, self._tuple_labels, self.results.param_axes)
        pf = self.results.sel(keys=[pf_key], squeeze_ones=False, avg_by_mouse=False, **stimspace_params)[pf_key]
        if full_key in SVCA_FULL_ENV_KEYS.values():
            params = {"subspace_name": "svca_subspace", "smooth_width": None}
            if "activity_parameters_name" in state:
                params["activity_parameters_name"] = state["activity_parameters_name"]
            full_agg = self.results_subspace
            full = full_agg.sel(keys=[full_key], squeeze_ones=False, avg_by_mouse=False, **params)[full_key]
        else:
            full_agg = self.results
            full = self.results.sel(keys=[full_key], squeeze_ones=False, avg_by_mouse=False, **stimspace_params)[full_key]
        pf = np.asarray(pf, dtype=float)
        full = np.asarray(full, dtype=float)
        if state["clip_negative"]:
            pf = _clip_at_first_negative(pf)
            full = _clip_at_first_negative(full)
        return pf, self.results, full, full_agg

    @staticmethod
    def _average_env_slots(spec: np.ndarray) -> np.ndarray:
        """Average available environment slots without warning on entirely missing sessions."""
        spec = np.asarray(spec, dtype=float)
        count = np.sum(np.isfinite(spec), axis=1)
        total = np.nansum(spec, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)

    def _selected_spectra(self, state: dict) -> tuple[np.ndarray, ResultsAggregator, np.ndarray, ResultsAggregator]:
        """Apply the familiarity source mode, returning session-leading PF and Full spectra."""
        if state["plot_mode"] == "by_env":
            pf, pf_agg, full, full_agg = self._per_env_spectra(state)
            slot = int(state["env_slot"])
            return pf[:, slot, :], pf_agg, full[:, slot, :], full_agg
        if state["plot_mode"] == "avg_env":
            pf_env, pf_agg, full_env, full_agg = self._per_env_spectra(state)
            pf = self._average_env_slots(pf_env)
            full = self._average_env_slots(full_env)
        else:
            pf, pf_agg = self._spectrum_sessions(state, state["source_key"])
            full, full_agg = self._full_spectrum_sessions(state)
        return pf, pf_agg, full, full_agg

    @staticmethod
    def _session_records(
        spectra: np.ndarray,
        aggregator: ResultsAggregator,
        mouse: str,
        *,
        dense: bool,
    ) -> list[tuple[int, np.ndarray]]:
        """Return ``(zero-based session number, curve)`` records in chronological order."""
        rows = np.where(aggregator.mouse_names == mouse)[0]
        rows = rows[np.argsort([str(aggregator.sessions[row].date) for row in rows])]
        records = []
        dense_idx = 0
        for overall_idx, row in enumerate(rows):
            curve = np.asarray(spectra[row], dtype=float)
            if not np.any(np.isfinite(curve)):
                continue
            session_idx = dense_idx if dense else overall_idx
            records.append((session_idx, curve))
            dense_idx += 1
        return records

    @classmethod
    def _records_by_mouse(cls, spectra: np.ndarray, aggregator: ResultsAggregator, *, dense: bool) -> dict[str, list[tuple[int, np.ndarray]]]:
        return {str(mouse): cls._session_records(spectra, aggregator, str(mouse), dense=dense) for mouse in aggregator.unique_mice}

    @staticmethod
    def _prepare_spectra(spec: np.ndarray, state: dict) -> np.ndarray:
        spec = np.asarray(spec, dtype=float)
        if state["normalize"]:
            denom = np.nansum(spec, axis=-1, keepdims=True)
            spec = np.divide(spec, denom, out=np.full_like(spec, np.nan), where=denom != 0)
        return _smooth_spectrum(spec, state["curve_smooth_kind"], state["curve_smooth_width"])

    @staticmethod
    def _shown_sessions(session_numbers: list[int], skip_sessions: int) -> list[int]:
        """Thin ordered session numbers while retaining the first and last."""
        if len(session_numbers) <= 2 or skip_sessions <= 0:
            return session_numbers
        step = skip_sessions + 1
        n_points = max(2, int(round((len(session_numbers) - 1) / step)) + 1)
        indices = np.unique(np.round(np.linspace(0, len(session_numbers) - 1, n_points)).astype(int))
        return [session_numbers[i] for i in indices]

    def _plot_example(self, axis, records, cmap, norm) -> None:
        for session_idx, curve in records:
            axis.plot(
                np.arange(curve.size) + 1,
                np.where(curve > 0, curve, np.nan),
                color=cmap(norm(session_idx + 1)),
                linewidth=1.0,
            )

    def _plot_population(self, axis, records_by_mouse, state, cmap, norm) -> None:
        session_numbers = sorted({session_idx for records in records_by_mouse.values() for session_idx, _ in records})
        supported = [
            session_idx
            for session_idx in session_numbers
            if sum(any(idx == session_idx for idx, _ in records) for records in records_by_mouse.values()) >= 2
        ]
        for session_idx in self._shown_sessions(supported, int(state["skip_sessions"])):
            curves = [curve for records in records_by_mouse.values() for idx, curve in records if idx == session_idx]
            max_dims = max(map(len, curves))
            stack = np.full((len(curves), max_dims), np.nan)
            for row, curve in enumerate(curves):
                stack[row, : len(curve)] = curve
            stack[stack <= 0] = np.nan
            xvals = np.arange(max_dims) + 1
            color = cmap(norm(session_idx + 1))
            valid = np.sum(np.isfinite(stack), axis=0)
            mean = np.divide(
                np.nansum(stack, axis=0),
                valid,
                out=np.full(max_dims, np.nan),
                where=valid > 0,
            )
            if state["plot_style"] == "each":
                axis.plot(xvals, stack.T, color=(*color[:3], 0.2), linewidth=0.4)
                axis.plot(xvals, mean, color=color, linewidth=1.5)
            elif state["hide_error"]:
                axis.plot(xvals, mean, color=color, linewidth=1.5)
            else:
                errorPlot(xvals, stack, axis=0, ax=axis, color=color, linewidth=1.5, alpha=0.2)

    def refresh_data(self, state: dict) -> None:
        """Reselect the PF/Full spectra and re-key their session records by mouse."""
        pf, pf_agg, full, full_agg = self._selected_spectra(state)
        pf = self._prepare_spectra(pf, state)
        full = self._prepare_spectra(full, state)
        self._dense = state["plot_mode"] == "by_env" and state["session_alignment"] == "within_env"
        self._pf_records = self._records_by_mouse(pf, pf_agg, dense=self._dense)
        self._full_records = self._records_by_mouse(full, full_agg, dense=self._dense)

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        pf_records, full_records, dense = self._pf_records, self._full_records, self._dense

        all_indices = [idx + 1 for collection in (pf_records, full_records) for records in collection.values() for idx, _ in records]
        vmax = max(all_indices, default=1)
        cmap = plt.get_cmap("coolwarm")
        norm = Normalize(vmin=1, vmax=max(vmax, 2))

        fig, axes = self.new_subplots(1, 4, figsize=self.figsize, layout="constrained", sharey=True)
        mouse = str(state["mouse"])
        self._plot_example(axes[0], pf_records.get(mouse, []), cmap, norm)
        self._plot_population(axes[1], pf_records, state, cmap, norm)
        self._plot_example(axes[2], full_records.get(mouse, []), cmap, norm)
        self._plot_population(axes[3], full_records, state, cmap, norm)

        axes[0].set_title(f"{self.pf_label}: {mouse}", fontsize=fontsize)
        axes[1].set_title(f"{self.pf_label}: all mice", fontsize=fontsize)
        axes[2].set_title(f"{self.ff_label}: {mouse}", fontsize=fontsize)
        axes[3].set_title(f"{self.ff_label}: all mice", fontsize=fontsize)
        for axis in axes:
            if state["log_x"]:
                axis.set_xscale("log")
            if state["log_y"]:
                axis.set_yscale("log")
                axis.set_ylim(bottom=10 ** state["ymin"])
            axis.set_xlabel("Dimension", fontsize=fontsize)
            style_model_axis(axis, fontsize=fontsize, xbounds=axis.get_xlim(), ybounds=axis.get_ylim())
        axes[0].set_ylabel("Fraction variance" if state["normalize"] else "Variance", fontsize=fontsize)
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=axes,
            orientation="horizontal",
            fraction=0.08,
            pad=0.12,
            aspect=45,
        )
        colorbar.set_label("Environment session #" if dense else "Overall session #", fontsize=fontsize)
        colorbar.ax.tick_params(labelsize=fontsize)
        return fig


class SpectrumDimFamiliarityViewer(FigureViewer):
    """Per-environment participation-ratio dimensionality over familiarity.

    Every environment is represented by its experience-order slot and is reindexed to session
    number within that environment for each mouse. PF and Full spectra are shown on separate axes
    (ax[0], ax[1]). ``full_key`` selects environment-only SVD, environment-vs-all-session SVD, or
    per-environment SVCA. Curves can be indexed either within each environment or by the mouse's
    overall session number.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated StimSpaceSpectra results, source of the per-environment PF/Full SVD keys.
    results_subspace : ResultsAggregator or None
        Aggregated SubspaceConfig results, source of the ``"svca"`` Full choice
        (``variance_activity_env``, ``subspace_name='svca_subspace'``). Required for that choice.
    source_key : str
        Per-environment PF spectrum: one of ``ss_cv``, ``ss_direct``, ``ss_cvpca``, ``sf_cv``,
        ``sf_direct``.
    full_key : {"svd_full1", "svd_fullall", "svd_res_full1", "svd_res_fullall", "svca"}
        Per-environment Full-CA1 estimator. Residual choices appear only when the corresponding
        ``ffres_env_*`` keys are present; ``"svca"`` requires ``results_subspace``.
    session_alignment : {"within_env", "overall"}
        Densify each curve to session number within its environment (``within_env``), or keep
        each observation's position in the mouse's full chronological session sequence
        (``overall``).
    display : {"each", "errorPlot"}
        Per-mouse lines + bold mean, or mean +/- SE band.
    yscale : {"linear", "log"}
        Y-axis scale, shared by both panels.
    sharey : bool
        Share one y-axis between the PF and Full panels.
    clip_negative : bool
        Replace each spectrum's first negative entry and all later ranks with NaN before reducing
        to a participation ratio.
    pr_pre_smooth : bool
        If True (default), reduce the raw spectrum straight to a participation ratio. If False,
        log-space smooth each spectrum first, using the independent PF/Full ``*_smooth_*`` knobs.
    source_smooth_method, full_smooth_method : {"none", "boxcar", "gaussian"}
        Pre-smoothing kind for the PF / Full spectrum, used only when ``pr_pre_smooth`` is False.
    source_smooth_width, full_smooth_width : float
        Boxcar full-width in rank units for the PF / Full pre-smoothing; the Gaussian uses
        ``sigma = width / 2``.
    fontsize : float
        Font size for every tick label, axis label, title, and legend.
    figsize : tuple[float, float]
        Figure size in inches.
    pf_label, full_label : str
        Panel titles for the PF and Full-CA1 axes.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        results_subspace: ResultsAggregator | None = None,
        *,
        source_key: str = "ss_cvpca",
        full_key: str = "svd_full1",
        session_alignment: str = "within_env",
        display: str = "errorPlot",
        yscale: str = "linear",
        sharey: bool = False,
        clip_negative: bool = False,
        pr_pre_smooth: bool = True,
        source_smooth_method: str = "gaussian",
        source_smooth_width: float = 3.0,
        full_smooth_method: str = "gaussian",
        full_smooth_width: float = 20.0,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (6.0, 2.5),
        pf_label: str = "PF",
        full_label: str = "Full",
        **selection_defaults,
    ):
        self.results = results
        self.results_subspace = results_subspace
        self.figsize = figsize
        self.pf_label = pf_label
        self.full_label = full_label

        self.add_selection("source_key", options=_PER_ENV_PF_KEYS, value=source_key)
        self.add_selection("full_key", options=_per_env_full_options(results, results_subspace), value=full_key)
        self.add_selection("session_alignment", options=_PER_ENV_SESSION_ALIGNMENTS, value=session_alignment)
        self.add_selection("display", options=["each", "errorPlot"], value=display)
        self.add_selection("yscale", options=["linear", "log"], value=yscale)
        self.add_boolean("sharey", value=sharey)
        self.add_boolean("clip_negative", value=clip_negative)
        self.add_boolean("pr_pre_smooth", value=pr_pre_smooth)
        self.add_selection("source_smooth_method", options=["none", "boxcar", "gaussian"], value=source_smooth_method)
        self.add_float("source_smooth_width", value=source_smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_selection("full_smooth_method", options=["none", "boxcar", "gaussian"], value=full_smooth_method)
        self.add_float("full_smooth_width", value=full_smooth_width, min=0.0, max=50.0, step=0.5)

        # ``selection_defaults`` lets a caller seed any shared param-axis by name.
        self._tuple_labels = add_merged_param_axis_widgets(
            self, results, preferred_defaults={**PREFERRED_DEFAULTS, **selection_defaults}
        )
        self._param_axis_names = _merged_axis_names(results)

        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        for name in (
            "source_key",
            "full_key",
            "session_alignment",
            "clip_negative",
            "pr_pre_smooth",
            "source_smooth_method",
            "source_smooth_width",
            "full_smooth_method",
            "full_smooth_width",
            *self._param_axis_names,
        ):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _sel_params(self, state: dict) -> dict:
        """Return only aggregator parameter selections, decoding tuple-valued axes."""
        return sel_params(state, self._tuple_labels, self.results.param_axes)

    @staticmethod
    def _result_keys(source_key: str, full_key: str) -> tuple[str, str]:
        """Map viewer choices to the two stored per-environment spectrum keys."""
        full_scope = "fullall" if full_key in ("svd_fullall", "svd_res_fullall") else "full1"
        if source_key in ("sf_cv", "sf_direct"):
            pf_key = f"{source_key}_env_{full_scope}"
        else:
            pf_key = _PER_ENV_PF_RESULT_KEYS[source_key]
        stored_full_key = {
            "svd_full1": "ff_env_full1",
            "svd_fullall": "ff_env_full1_fullall",
            **_PER_ENV_RESIDUAL_RESULT_KEYS,
            "svca": "variance_activity_env",
        }[full_key]
        return pf_key, stored_full_key

    def _curves(
        self,
        values: np.ndarray,
        aggregator: ResultsAggregator,
        session_alignment: str,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Index ``(sessions, env_slots)`` values within environment or across all sessions."""
        curves: dict[int, dict[str, np.ndarray]] = {}
        for slot in range(MAX_ENV_SLOTS):
            per_mouse: dict[str, np.ndarray] = {}
            for mouse in aggregator.unique_mice:
                rows = np.where(aggregator.mouse_names == mouse)[0]
                dates = np.array([aggregator.sessions[row].date for row in rows])
                rows = rows[np.argsort(dates)]
                curve = np.asarray(values[rows, slot], dtype=float)
                if session_alignment == "within_env":
                    curve = curve[np.isfinite(curve)]
                if np.any(np.isfinite(curve)):
                    per_mouse[str(mouse)] = curve
            curves[slot] = per_mouse
        return curves

    @staticmethod
    def _pad_curves(curves: dict[str, np.ndarray]) -> np.ndarray:
        """NaN-pad ragged per-mouse curves to a common session axis."""
        max_sessions = max((len(curve) for curve in curves.values()), default=0)
        stack = np.full((len(curves), max_sessions), np.nan)
        for row, curve in enumerate(curves.values()):
            stack[row, : len(curve)] = curve
        return stack

    def _draw_axis(self, ax, curves: dict[int, dict[str, np.ndarray]], display: str) -> int:
        """Draw every environment slot on one axis and return the longest x extent."""
        max_length = 0
        for slot, per_mouse in curves.items():
            stack = self._pad_curves(per_mouse)
            if not stack.size:
                continue
            if ax.get_yscale() == "log":
                stack[stack <= 0] = np.nan

            color = ENV_SLOT_COLORS[slot % len(ENV_SLOT_COLORS)]
            support = np.sum(np.isfinite(stack), axis=0)
            summary_columns = np.where(support >= 2)[0]
            label = f"Env #{slot + 1}"

            if display == "each":
                for curve in stack:
                    ax.plot(np.arange(1, curve.size + 1), curve, color=(color, 0.3), linewidth=0.5)
                if summary_columns.size:
                    ax.plot(
                        summary_columns + 1,
                        np.nanmean(stack[:, summary_columns], axis=0),
                        color=color,
                        linewidth=2.0,
                        label=label,
                    )
                max_length = max(max_length, stack.shape[1])
            elif summary_columns.size:
                errorPlot(
                    summary_columns + 1,
                    stack[:, summary_columns],
                    axis=0,
                    se=True,
                    ax=ax,
                    color=color,
                    linewidth=2.0,
                    alpha=0.25,
                    label=label,
                )
                max_length = max(max_length, int(summary_columns[-1]) + 1)
        return max_length

    def refresh_data(self, state: dict) -> None:
        """Reselect the per-environment PF/Full spectra and reduce them to per-mouse PR curves."""
        pf_key, stored_full_key = self._result_keys(state["source_key"], state["full_key"])
        params = self._sel_params(state)
        pf_out = self.results.sel(keys=[pf_key], squeeze_ones=False, avg_by_mouse=False, **params)
        if state["full_key"] == "svca":
            subspace_params = {"subspace_name": "svca_subspace", "smooth_width": None}
            if "activity_parameters_name" in state:
                subspace_params["activity_parameters_name"] = state["activity_parameters_name"]
            full_agg = self.results_subspace
            full_out = full_agg.sel(keys=[stored_full_key], squeeze_ones=False, avg_by_mouse=False, **subspace_params)
        else:
            full_agg = self.results
            full_out = self.results.sel(keys=[stored_full_key], squeeze_ones=False, avg_by_mouse=False, **params)
        if state["clip_negative"]:
            pf_out[pf_key] = _clip_at_first_negative(pf_out[pf_key])
            full_out[stored_full_key] = _clip_at_first_negative(full_out[stored_full_key])
        if not state["pr_pre_smooth"]:
            pf_shape = pf_out[pf_key].shape
            full_shape = full_out[stored_full_key].shape
            pf_out[pf_key] = _smooth_spectrum(
                pf_out[pf_key].reshape(-1, pf_shape[-1]), state["source_smooth_method"], state["source_smooth_width"]
            ).reshape(pf_shape)
            full_out[stored_full_key] = _smooth_spectrum(
                full_out[stored_full_key].reshape(-1, full_shape[-1]), state["full_smooth_method"], state["full_smooth_width"]
            ).reshape(full_shape)
        pf_values = _signed_participation_ratio(pf_out[pf_key])
        full_values = _signed_participation_ratio(full_out[stored_full_key])
        self._pf_curves = self._curves(pf_values, self.results, state["session_alignment"])
        self._full_curves = self._curves(full_values, full_agg, state["session_alignment"])

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        fig, ax = self.new_subplots(1, 2, figsize=self.figsize, layout="constrained", sharey=state["sharey"])
        for axis in ax:
            axis.set_yscale(state["yscale"])

        extents = [
            self._draw_axis(ax[0], self._pf_curves, state["display"]),
            self._draw_axis(ax[1], self._full_curves, state["display"]),
        ]
        xmax = max(max(extents), 1)
        for axis, title in zip(ax, (self.pf_label, self.full_label)):
            axis.set_xlim(0.8, xmax + 0.2)
            xlabel = "Env session #" if state["session_alignment"] == "within_env" else "Overall session #"
            axis.set_xlabel(xlabel, fontsize=fontsize)
            axis.set_ylabel("Dimensionality", fontsize=fontsize)
            axis.set_title(title, fontsize=fontsize)
            style_model_axis(axis, fontsize=fontsize, xbounds=[1, xmax], ybounds=axis.get_ylim())
        ax[0].legend(
            loc="best",
            fontsize=fontsize,
            title_fontsize=fontsize,
            frameon=False,
            title="Environment",
            handlelength=0.8,
            handletextpad=0.5,
        )
        return fig
