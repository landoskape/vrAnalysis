"""Tilbury generalized-Gaussian placefield fits: example-neuron grids for a single session, and a
hand-picked, ordered figure of specific (session, neuron) examples.

Both viewers overlay a held-out test placefield against the fitted generalized-Gaussian (Tilbury),
plain-Gaussian control, and generalized-shrinkage curves. The fitted parameters and R^2 come
straight from the stored :class:`~dimensionality_manuscript.configs.tilbury_fit.TilburyFitConfig`
results (no re-fit); only the held-out test curve is not stored, so both viewers rebuild it from
the deterministic train/test split and trial-averaging the fit used (:func:`_load_fit_curves`).

:class:`PlacefieldExampleFitViewer` draws a random sample of well-fit neurons from one session;
:class:`PlacefieldFitFigureViewer` plots an explicit, ordered list of neurons picked by eye. The
population summaries live elsewhere (``PlacefieldPopulationViewer``).
"""

from typing import Optional

import numpy as np

from vrAnalysis.helpers import edge2center
from vrAnalysis.helpers.plotting import format_spines
from dimensionality_manuscript import ResultsAggregator
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.configs.tilbury_fit import TilburyFitConfig, _eval_tilbury, _eval_gaussian, _SPLITS

from ..panels import FigureViewer
from ._param_axes import add_merged_param_axis_widgets, encode_param, sel_params

# Fixed curve colors shared by both viewers.
_GENERALIZED_COLOR = "blue"
_GAUSSIAN_COLOR = "black"
_SHRINKAGE_COLOR = "purple"

# Normalization presets: each maps a curve group to the scalar it is divided by, computed on the
# *test-data* curve so the fits stay overlaid on the data while every panel shares a common scale
# (needed for sharey).
_FIT_FIGURE_NORMALIZATIONS = ("std", "sum", "max", "none")

# Which generalized-family fit(s) PlacefieldFitFigureViewer overlays alongside the plain-Gaussian
# control: just the unregularized generalized fit, just the shrinkage fit, or both. When only one is
# shown it is drawn in the generalized (blue) color; "both" keeps generalized blue and shrinkage purple.
_FIT_MODEL_OPTIONS = ("generalized", "shrinkage", "both")

# Matplotlib ``loc`` strings offered for the placement of the fit-figure legend.
_LEGEND_POSITIONS = (
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
    "best",
)

# Per-neuron keys every Tilbury fit needs, sliced to the kept-neuron prefix (see _load_fit_curves).
_BASE_FIT_KEYS = ["params", "params_control", "params_shrinkage", "r2_test", "r2_test_control", "idx_keep"]


def _fit_figure_scale(ref: np.ndarray, method: str) -> float:
    """Scalar to divide a curve group by, from the reference (test-data) curve.

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


def _load_fit_curves(
    results: ResultsAggregator,
    registry: PopulationRegistry,
    session_uid: str,
    fit_params: dict,
    extra_keys: tuple[str, ...] = (),
) -> dict:
    """Load one session's kept-neuron Tilbury fits and rebuild its held-out test curve.

    Shared by :class:`PlacefieldExampleFitViewer` and :class:`PlacefieldFitFigureViewer`. The fitted
    parameters and R^2 come straight from the aggregated :class:`TilburyFitConfig` results (no
    gradient descent). The held-out test placefield is not stored, so it is rebuilt with the same
    deterministic split (``registry.time_split``) and trial-averaging (``_avg_placefield``) the fit
    used; ``best_env``, the bin edges and the dropped-bin mask are recomputed exactly as
    :meth:`TilburyFitConfig.process` does.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated TilburyFitConfig results.
    registry : PopulationRegistry
        Registry used to rebuild the split data (must match the one the results were built with).
    session_uid : str
        Session to load.
    fit_params : dict
        One value per TilburyFitConfig param axis.
    extra_keys : tuple of str
        Additional per-neuron result keys to select alongside :data:`_BASE_FIT_KEYS`, sliced to the
        kept-neuron prefix the same way (e.g. ``r2_test_shrinkage``, ``lambda_selected``).

    Returns
    -------
    dict
        ``theta`` (P,), ``test_curve`` (n_kept, P), ``params`` (n_kept, 6),
        ``params_control`` (n_kept, 4), ``params_shrinkage`` (n_kept, 6), ``r2_test`` (n_kept,),
        ``r2_test_control`` (n_kept,), ``idx_keep`` (N_available,) bool, ``idx_neurons``
        (N_available,) original ROI indices, ``session``, plus one entry per name in ``extra_keys``,
        all aligned so row ``n`` of every kept-neuron array is the same neuron.
    """
    if session_uid not in results._session_index:
        raise KeyError(f"session_uid {session_uid!r} not in results (options: {list(results.session_ids)}).")

    config = results.config_class
    idx = results._session_index[session_uid]
    session = results.sessions[idx]

    keys = [*_BASE_FIT_KEYS, *extra_keys]
    sel = results.sel(keys=keys, load_ragged=True, squeeze_ones=False, **fit_params)
    idx_keep = np.asarray(sel["idx_keep"][idx], dtype=bool)  # (N_available,) over population.idx_neurons
    n_kept = int(np.sum(idx_keep))
    fit = {key: sel[key][idx][:n_kept] for key in keys if key != "idx_keep"}
    fit["idx_keep"] = idx_keep

    # Original ROI indices that entered the fit: population.idx_neurons (the AND of
    # session.idx_rois across all spks_types), NOT the current-spks_type session.idx_rois.
    # idx_keep indexes into this array, so idx_neurons[j] recovers a neuron's original index.
    population, _ = registry.get_population(session, config.spks_type)
    idx_neurons = np.asarray(population.idx_neurons)

    # Recompute the fit's fixed choices (all "skip"ped from storage, all deterministic).
    num_per_env = {i: int(np.sum(session.trial_environment == i)) for i in session.environments}
    best_env = max(num_per_env, key=num_per_env.get)
    dist_edges = np.linspace(0, session.env_length[0], config.num_bins + 1)
    dist_centers = edge2center(dist_edges)

    # Trial-average every split's placefield over the kept neurons; the counts give the
    # dropped-bin mask (bins empty in any split) so theta matches the stored params' support.
    spks, fb = config._get_split_data(session, registry)
    for s in _SPLITS:
        spks[s] = spks[s][:, idx_keep]
    curves, counts = {}, {}
    for s in _SPLITS:
        curves[s], counts[s] = config._avg_placefield(spks[s], fb[s], dist_edges, best_env, session)
    bad = np.zeros(config.num_bins, dtype=bool)
    for s in _SPLITS:
        bad |= counts[s] == 0
    good = ~bad

    fit["theta"] = dist_centers[good]
    fit["test_curve"] = curves["test"][:, good]
    fit["idx_neurons"] = idx_neurons
    fit["session"] = session
    return fit


class PlacefieldExampleFitViewer(FigureViewer):
    """Tilbury generalized-Gaussian placefield fits: grid of example single-neuron fits.

    An ``n_rows x n_cols`` grid of example single-neuron fits from one session (a reproducible
    random sample of the top neurons by test R^2 that also clear the improvement threshold). Each
    panel overlays the held-out test placefield (points) against the three fitted curves:
    generalized-Gaussian (Tilbury, blue), plain-Gaussian control (black) and generalized-shrinkage
    (purple). The fit param axes (e.g. ``activity_parameters_name``) are selectable widgets.

    The population summaries live in the separate ``PlacefieldPopulationViewer``.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated TilburyFitConfig results.
    registry : PopulationRegistry
        Registry used to rebuild the example session's test curve (must match the one the results
        were built with).
    example_session : str or None
        session_uid to show. If None (default), the first session in ``results`` is used.
    n_rows, n_cols : int
        Grid shape of the example-fit panel (``n_rows * n_cols`` example neurons).
    r2_threshold : float
        Example neurons must have generalized test R^2 above this threshold.
    improvement_threshold : float
        Example neurons must also beat the plain Gaussian by at least this much test R^2 via
        *either* the generalized or the shrinkage fit (OR logic):
        ``max(r2_generalized, r2_shrinkage) - r2_gaussian > improvement_threshold``.
    random_seed : int
        Seed for the random example draw (reproducible). If fewer than ``n_rows * n_cols`` neurons
        clear both thresholds, the leftover panels are left empty.
    normalize : {"std", "sum", "max", "none"}
        Per-panel, the curve group is divided by this statistic (of the test-data curve unless
        ``normalize_independent``).
    normalize_independent : bool
        If True, divide each of the four curves (test data, generalized, gaussian, shrinkage) by its
        *own* statistic instead of the shared test-data one -- a shape-only comparison that removes
        amplitude differences (they no longer overlay). Default True.
    fontsize : float
        Font size for tick labels, titles, and the legend.
    figsize : tuple[float, float]
        Figure size in inches.
    **selections
        Overrides for the fit's parameter-axis selections, keyed by raw ``param_axes`` name of
        ``results`` (e.g. ``activity_parameters_name``).
    """

    def __init__(
        self,
        results: ResultsAggregator,
        registry: PopulationRegistry,
        *,
        example_session: Optional[str] = None,
        n_rows: int = 2,
        n_cols: int = 3,
        r2_threshold: float = 0.5,
        improvement_threshold: float = 0.0,
        random_seed: int = 0,
        normalize: str = "sum",
        normalize_independent: bool = True,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (8.0, 3.0),
        **selections,
    ):
        self.results = results
        self.registry = registry
        self.config = results.config_class
        self.figsize = figsize
        # Rebuilding the test curve is cheap (deterministic trial-average), but cache by
        # (session_uid, fit params) so switching back to a session in the viewer is instant.
        self._fit_cache: dict[tuple, dict] = {}
        self._fit: dict = {}
        self._chosen: np.ndarray = np.array([], dtype=int)

        session_options = list(results.session_ids)
        self.add_selection("example_session", options=session_options, value=example_session or session_options[0])
        # One widget per TilburyFitConfig param axis (activity_parameters_name, ...): the stored
        # fits exist once per combination, so every one must be pinned before slicing.
        self._fit_axes = list(results.param_axes)
        self._tuple_labels = add_merged_param_axis_widgets(self, results)
        for name, value in selections.items():
            if name not in results.param_axes:
                raise ValueError(f"Unknown selection {name!r}. Options: {sorted(results.param_axes)}")
            self.set_parameter_value(name, encode_param(self._tuple_labels, name, value))

        self.add_integer("n_rows", value=n_rows, min=1, max=6)
        self.add_integer("n_cols", value=n_cols, min=1, max=6)
        # Example neurons are drawn at random from those with generalized test R2 above r2_threshold
        # AND (generalized - gaussian) OR (shrinkage - gaussian) test R2 above improvement_threshold
        # (so the example fits well and beats the plain Gaussian via either the generalized or the
        # shrinkage fit). The seed makes the draw reproducible; if too few clear both thresholds the
        # extra panels are left empty.
        self.add_float("r2_threshold", value=r2_threshold, min=-1.0, max=1.0, step=0.05)
        self.add_float("improvement_threshold", value=improvement_threshold, min=0.0, max=1.0, step=0.01)
        self.add_integer("random_seed", value=random_seed, min=0, max=100000)
        # Per-panel normalization: divide the curve group by a statistic (std / sum / max / none).
        # normalize_independent scales each of the four curves by its own statistic (shape-only);
        # otherwise the group shares the test-data curve's scale, keeping the fits overlaid on the data.
        self.add_selection("normalize", options=list(_FIT_FIGURE_NORMALIZATIONS), value=normalize)
        self.add_boolean("normalize_independent", value=normalize_independent)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        # Everything that changes which neurons are loaded or chosen goes through refresh_data;
        # normalize / normalize_independent / fontsize are pure style, read straight from state in plot().
        refresh_names = ("example_session", *self._fit_axes, "n_rows", "n_cols", "r2_threshold", "improvement_threshold", "random_seed")
        self.on_change(refresh_names, self.refresh_data)
        self.refresh_data(self.state)

    def _fit_sel_params(self, state: dict) -> dict:
        """Params pinning every TilburyFitConfig param axis, decoded from the widgets."""
        return sel_params(state, self._tuple_labels, self._fit_axes)

    def _example_fit(self, session_uid: str, fit_params: dict) -> dict:
        """Return the (cached) example fit for ``session_uid`` at ``fit_params``, loading it on a miss."""
        cache_key = (session_uid, tuple(sorted(fit_params.items())))
        if cache_key not in self._fit_cache:
            self._fit_cache[cache_key] = _load_fit_curves(
                self.results, self.registry, session_uid, fit_params, extra_keys=("r2_test_shrinkage", "lambda_selected")
            )
        return self._fit_cache[cache_key]

    def refresh_data(self, state: dict) -> None:
        """Load the selected session's fit and draw a fresh reproducible sample of example neurons."""
        fit = self._example_fit(state["example_session"], self._fit_sel_params(state))
        n_show = int(state["n_rows"]) * int(state["n_cols"])
        r2 = fit["r2_test"]
        r2c = fit["r2_test_control"]
        r2s = fit["r2_test_shrinkage"]
        # Well-fit by the generalized model AND beating the plain Gaussian by improvement_threshold
        # via *either* the generalized or the shrinkage fit (OR logic).
        improves = ((r2 - r2c) > state["improvement_threshold"]) | ((r2s - r2c) > state["improvement_threshold"])
        eligible = np.flatnonzero(np.isfinite(r2) & np.isfinite(r2c) & np.isfinite(r2s) & (r2 > state["r2_threshold"]) & improves)
        rng = np.random.default_rng(int(state["random_seed"]))
        # Random draw without replacement; if too few clear the threshold, extra panels stay empty.
        chosen = rng.choice(eligible, size=n_show, replace=False) if eligible.size > n_show else eligible
        self._fit = fit
        self._chosen = chosen

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        method = state["normalize"]
        independent = bool(state["normalize_independent"])
        n_rows = int(state["n_rows"])
        n_cols = int(state["n_cols"])
        fit = self._fit
        chosen = self._chosen

        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        gs = fig.add_gridspec(n_rows, n_cols)

        theta = fit["theta"]
        n_show = n_rows * n_cols
        share_ax = None
        for cell in range(n_show):
            r, c = divmod(cell, n_cols)
            # Share x (common position axis) but not y: each neuron gets its own optimal y-range.
            ax = fig.add_subplot(gs[r, c], sharex=share_ax)
            share_ax = share_ax or ax
            if r == n_rows - 1:
                ax.set_xlabel("Position (cm)", fontsize=fontsize)
            if c == 0:
                ax.set_ylabel("Activity", fontsize=fontsize)
            ax.tick_params(axis="both", which="both", labelsize=fontsize)
            if cell >= len(chosen):
                continue  # not enough eligible neurons -> leave this panel empty
            n = chosen[cell]

            # Original ROI index of this neuron: n indexes the kept arrays, np.where(idx_keep) maps
            # it to a row of idx_neurons (population.idx_neurons), which already holds original indices.
            idx_within_fit_neurons = np.where(fit["idx_keep"])[0][n]
            idx_within_idx_rois = fit["idx_neurons"][idx_within_fit_neurons]

            data = fit["test_curve"][n]
            gen = _eval_tilbury(theta, fit["params"][n])
            gauss = _eval_gaussian(theta, fit["params_control"][n])
            shrink = _eval_tilbury(theta, fit["params_shrinkage"][n])
            # normalize_independent: each curve divided by its own statistic (shape-only). Otherwise
            # the whole set shares the test-data curve's scale, keeping the fits overlaid on the data.
            if independent:
                data = data / _fit_figure_scale(data, method)
                gen = gen / _fit_figure_scale(gen, method)
                gauss = gauss / _fit_figure_scale(gauss, method)
                shrink = shrink / _fit_figure_scale(shrink, method)
            else:
                scale = _fit_figure_scale(data, method)
                data, gen, gauss, shrink = data / scale, gen / scale, gauss / scale, shrink / scale

            first = cell == 0
            ax.plot(theta, data, "o", color="red", ms=2.5, alpha=0.5, label="Test data" if first else None)
            ax.plot(theta, gen, "-", color=_GENERALIZED_COLOR, lw=1.5, label="Generalized" if first else None)
            ax.plot(theta, gauss, "-", color=_GAUSSIAN_COLOR, lw=1.5, label="Gaussian" if first else None)
            ax.plot(theta, shrink, "-", color=_SHRINKAGE_COLOR, lw=1.5, label="Generalized (shrinkage)" if first else None)
            lam_p, lam_asym = fit["lambda_selected"][n]
            ax.set_title(
                f"{state['example_session']} | Neuron: {idx_within_idx_rois} | R²={fit['r2_test'][n]:.2f} | λ=({lam_p:g}, {lam_asym:g})",
                fontsize=fontsize,
            )
            if first:
                ax.legend(fontsize=fontsize * 0.8, frameon=False, loc="upper right")
        return fig


class PlacefieldFitFigureViewer(FigureViewer):
    """Tilbury placefield fits for a hand-picked list of (session, neuron) examples.

    Unlike :class:`PlacefieldExampleFitViewer` (which *draws* well-fit neurons at random from one
    session), this viewer plots an explicit, ordered list of neurons the user selected by eye — each
    identified by its session_uid and its **original ROI index** (the index into the session's full
    ROI set, i.e. ``np.where(session.idx_rois)[0][k]``). That original index is the stable identifier
    to write down for a figure: it survives regardless of how many neurons cleared the reliability /
    fraction-active thresholds in TilburyFitConfig.

    Each requested ROI is traced back to its fit: it must be one of the pipeline's available neurons
    (``population.idx_neurons``) *and* have been kept by the fit's inclusion thresholds
    (``idx_keep``). A neuron that was never available or was dropped before fitting raises (opt-in via
    ``strict``) or is flagged with an empty, titled panel.

    The first ``n_rows * n_cols`` entries of the list are plotted, in order, into a shared-axes grid
    (``sharex``/``sharey``); each panel overlays the held-out test placefield (points) against the
    plain-Gaussian control (always shown) and the generalized-family fit(s) selected by ``fit_model``
    (generalized, shrinkage, or both). The whole group in a panel is normalized by the test-data
    curve's statistic (``normalize``: std / sum / max / none), so the fits stay overlaid on the data
    while panels remain comparable under ``sharey``. Only one panel is labelled and carries the
    legend, picked by ``legend_axis`` (a flat row-major panel index).

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated TilburyFitConfig results.
    registry : PopulationRegistry
        Registry used to rebuild each session's test curve and to map an original ROI index to its
        fit row (must match the one the results were built with).
    session_uids : list of str
        session_uid per requested neuron.
    neurons : list of int
        The **original ROI index** per requested neuron -- the index into the session's full ROI set
        (``np.where(session.idx_rois)[0][k]``), which is the stable identifier to record for a figure.
        Must be the same length as ``session_uids`` and aligned to it.
    n_rows, n_cols : int
        Grid shape; the first ``n_rows * n_cols`` list entries are plotted (extra panels hidden).
    legend_axis : int
        Flat row-major index of the panel that carries the legend, resolved to a grid position with
        ``divmod(legend_axis, n_cols)``. Must be in ``[0, n_rows * n_cols - 1]``. Default 0 (top-left).
    legend_position : str
        Matplotlib ``loc`` for the legend, one of :data:`_LEGEND_POSITIONS`. Default ``"upper right"``.
    legend_x_offset, legend_y_offset : float
        Shift of the legend's anchor box in axes-fraction units (1.0 = one full axes width/height),
        letting the legend sit outside its panel -- e.g. ``legend_x_offset=0.4`` with
        ``legend_position="upper right"`` parks it in the gap to the right of the panel. The legend is
        excluded from the constrained layout, so offsetting it never resizes the panels. Default 0.
    legend_handlelength : float
        Length of the sample line drawn in each legend entry, in font-size units. Default 2.0.
    legend_handletextpad : float
        Gap between a legend entry's sample line and its label, in font-size units. Default 0.8.
    legend_markerfirst : bool
        If True (default), the sample line is drawn to the left of its label; False puts it on the right.
    normalize : {"std", "sum", "max", "none"}
        Per-panel, the test-data curve and both fits are divided by this statistic of the test-data
        curve, so the fits stay overlaid on the data while panels share a scale under ``sharey``.
    normalize_independent : bool
        If True, divide each of the three curves (test data, generalized, gaussian) by its *own*
        statistic instead of the shared test-data one -- a shape-only comparison that removes amplitude
        differences between the fits and the data (they no longer overlay). Default False.
    fit_model : {"generalized", "shrinkage", "both"}
        Which generalized-family fit(s) to overlay alongside the always-shown plain-Gaussian control.
        A lone fit is drawn in the generalized (blue) color; ``"both"`` keeps generalized blue and
        shrinkage purple. Default ``"both"``.
    strict : bool
        If True (default), a requested ROI that never entered the pipeline or was dropped before
        fitting raises ``ValueError``. If False, that panel is left empty with a red status title.
    fontsize : float
        Font size for tick labels and the legend.
    figsize : tuple[float, float]
        Figure size in inches.
    **selections
        Overrides for the fit's parameter-axis selections, keyed by raw ``param_axes`` name of
        ``results`` (e.g. ``activity_parameters_name``).
    """

    def __init__(
        self,
        results: ResultsAggregator,
        registry: PopulationRegistry,
        session_uids: list[str],
        neurons: list[int],
        *,
        n_rows: int = 2,
        n_cols: int = 3,
        legend_axis: int = 0,
        legend_position: str = "upper right",
        legend_x_offset: float = 0.0,
        legend_y_offset: float = 0.0,
        legend_handlelength: float = 2.0,
        legend_handletextpad: float = 0.8,
        legend_markerfirst: bool = True,
        normalize: str = "std",
        normalize_independent: bool = False,
        fit_model: str = "both",
        strict: bool = True,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (8.0, 4.0),
        **selections,
    ):
        if len(session_uids) != len(neurons):
            raise ValueError(f"session_uids and neurons must be the same length, got {len(session_uids)} and {len(neurons)}.")
        if normalize not in _FIT_FIGURE_NORMALIZATIONS:
            raise ValueError(f"Unknown normalize {normalize!r}. Options: {list(_FIT_FIGURE_NORMALIZATIONS)}")
        if fit_model not in _FIT_MODEL_OPTIONS:
            raise ValueError(f"Unknown fit_model {fit_model!r}. Options: {list(_FIT_MODEL_OPTIONS)}")
        if legend_position not in _LEGEND_POSITIONS:
            raise ValueError(f"Unknown legend_position {legend_position!r}. Options: {list(_LEGEND_POSITIONS)}")

        self.results = results
        self.registry = registry
        self.config = results.config_class
        self.session_uids = list(session_uids)
        self.neurons = [int(n) for n in neurons]
        self.strict = strict
        self.figsize = figsize
        # Rebuilding a session's test curves is cheap but cache by (session_uid, fit params) so the
        # same session appearing for several requested neurons is only loaded once.
        self._fit_cache: dict[tuple, dict] = {}
        # (fit, kept_row, status) per requested (session_uid, roi), refreshed only when the fit
        # parameters change (see refresh_data) -- not on every style-knob change.
        self._resolved: list[tuple[dict, Optional[int], str]] = []

        self.add_integer("n_rows", value=n_rows, min=1, max=8)
        self.add_integer("n_cols", value=n_cols, min=1, max=8)
        # Flat panel index (row-major, resolved with divmod by n_cols) of the panel that carries the
        # legend; its upper bound tracks the current grid size.
        self.add_integer("legend_axis", value=legend_axis, min=0, max=n_rows * n_cols - 1)
        # Legend layout, passed straight through to ax.legend: placement, the length of the sample
        # line in each entry, the gap between that sample and its label, and whether the sample is
        # drawn to the left of the label (False puts it on the right).
        self.add_selection("legend_position", options=list(_LEGEND_POSITIONS), value=legend_position)
        # Shift the anchor box the legend is placed against, in axes-fraction units (1.0 = one full
        # axes width/height), so the legend can sit outside its panel — e.g. in the gap between two
        # panels. 0 leaves it flush with the axes, as a plain ``loc`` would put it.
        self.add_float("legend_x_offset", value=legend_x_offset, min=-2.0, max=2.0, step=0.01)
        self.add_float("legend_y_offset", value=legend_y_offset, min=-2.0, max=2.0, step=0.01)
        self.add_float("legend_handlelength", value=legend_handlelength, min=0.0, max=6.0, step=0.1)
        self.add_float("legend_handletextpad", value=legend_handletextpad, min=0.0, max=3.0, step=0.05)
        self.add_boolean("legend_markerfirst", value=legend_markerfirst)
        # One widget per TilburyFitConfig param axis (see PlacefieldExampleFitViewer).
        self._fit_axes = list(results.param_axes)
        self._tuple_labels = add_merged_param_axis_widgets(self, results)
        for name, value in selections.items():
            if name not in results.param_axes:
                raise ValueError(f"Unknown selection {name!r}. Options: {sorted(results.param_axes)}")
            self.set_parameter_value(name, encode_param(self._tuple_labels, name, value))

        self.add_selection("normalize", options=list(_FIT_FIGURE_NORMALIZATIONS), value=normalize)
        # normalize_independent: scale each of the three curves (test data, generalized, gaussian) by
        # its own statistic (shape-only comparison), instead of the whole group by the test-data curve.
        self.add_boolean("normalize_independent", value=normalize_independent)
        # Which generalized-family fit(s) to overlay (the plain-Gaussian control is always shown).
        self.add_selection("fit_model", options=list(_FIT_MODEL_OPTIONS), value=fit_model)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        self.on_change(["n_rows", "n_cols"], self.update_legend_bounds)
        self.on_change(self._fit_axes, self.refresh_data)
        self.refresh_data(self.state)

    def _fit_sel_params(self, state: dict) -> dict:
        """Params pinning every TilburyFitConfig param axis, decoded from the widgets."""
        return sel_params(state, self._tuple_labels, self._fit_axes)

    def update_legend_bounds(self, state: dict) -> None:
        """Keep the legend panel index inside the current ``n_rows * n_cols`` grid."""
        self.update_integer("legend_axis", max=int(state["n_rows"]) * int(state["n_cols"]) - 1)

    def _session_fit(self, session_uid: str, fit_params: dict) -> dict:
        """Return the (cached) per-session fit bundle for ``session_uid`` at ``fit_params``."""
        cache_key = (session_uid, tuple(sorted(fit_params.items())))
        if cache_key not in self._fit_cache:
            self._fit_cache[cache_key] = _load_fit_curves(self.results, self.registry, session_uid, fit_params)
        return self._fit_cache[cache_key]

    def _resolve(self, session_uid: str, roi: int, fit_params: dict) -> tuple[dict, Optional[int], str]:
        """Map a hand-picked ``(session_uid, original ROI index)`` to its kept-row in the fit bundle.

        Returns ``(fit, kept_row, status)`` where ``status`` is ``"ok"`` (``kept_row`` is the row of
        ``params`` / ``test_curve`` for this neuron), ``"not_available"`` (ROI never entered the
        pipeline — silent / filtered out), or ``"not_fit"`` (available but dropped by the reliability
        / fraction-active thresholds). ``kept_row`` is ``None`` for the two failure statuses.
        """
        fit = self._session_fit(session_uid, fit_params)
        idx_neurons = fit["idx_neurons"]
        pos = np.flatnonzero(idx_neurons == roi)
        if pos.size == 0:
            return fit, None, "not_available"
        j = int(pos[0])
        if not fit["idx_keep"][j]:
            return fit, None, "not_fit"
        kept_row = int(np.sum(fit["idx_keep"][:j]))
        return fit, kept_row, "ok"

    def refresh_data(self, state: dict) -> None:
        """Resolve every requested (session_uid, roi) against the current fit parameters.

        Resolves the whole requested list regardless of the current grid size, since the fit
        parameters (not ``n_rows``/``n_cols``) are what changes which numbers get computed.
        """
        fit_params = self._fit_sel_params(state)
        self._resolved = [self._resolve(session_uid, roi, fit_params) for session_uid, roi in zip(self.session_uids, self.neurons)]

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        n_rows = int(state["n_rows"])
        n_cols = int(state["n_cols"])
        method = state["normalize"]
        independent = bool(state["normalize_independent"])
        n_show = n_rows * n_cols
        # Panel that carries the legend, as a flat row-major index into the grid.
        legend_cell = min(int(state["legend_axis"]), n_show - 1)

        fig, axs = self.new_subplots(n_rows, n_cols, figsize=self.figsize, squeeze=False, layout="constrained")

        theta = None
        for cell in range(n_show):
            r, c = divmod(cell, n_cols)
            ax = axs[r, c]
            if r == n_rows - 1:
                ax.set_xlabel("Position (cm)", fontsize=fontsize)
            if c == 0:
                ax.set_ylabel("Activity", fontsize=fontsize)
            if cell >= len(self._resolved):
                ax.set_visible(False)  # fewer requested neurons than panels -> hide the extras
                continue

            session_uid, roi = self.session_uids[cell], self.neurons[cell]
            fit, kept_row, status = self._resolved[cell]
            if status != "ok":
                # Traced but not fittable: flag loudly (strict) or leave a titled empty panel.
                if self.strict:
                    raise ValueError(f"Neuron roi={roi} in session {session_uid!r} is '{status}' (not a fitted neuron).")
                ax.set_title(f"{session_uid}\nroi {roi}: {status}", fontsize=fontsize * 0.8, color="red")
                continue

            theta = fit["theta"]
            data = fit["test_curve"][kept_row]
            gen = _eval_tilbury(theta, fit["params"][kept_row])
            gauss = _eval_gaussian(theta, fit["params_control"][kept_row])
            shrink = _eval_tilbury(theta, fit["params_shrinkage"][kept_row])
            # normalize_independent: each curve divided by its own statistic (shape-only). Otherwise
            # the whole set shares the test-data curve's scale, keeping the fits overlaid on the data.
            if independent:
                data = data / _fit_figure_scale(data, method)
                gen = gen / _fit_figure_scale(gen, method)
                gauss = gauss / _fit_figure_scale(gauss, method)
                shrink = shrink / _fit_figure_scale(shrink, method)
            else:
                scale = _fit_figure_scale(data, method)
                data, gen, gauss, shrink = data / scale, gen / scale, gauss / scale, shrink / scale

            # The legend lives on one panel, so only that panel's curves get labelled.
            first = (r, c) == divmod(legend_cell, n_cols)
            ax.plot(theta, data, "o", color="gray", ms=2.5, alpha=0.5, label="Test data" if first else None)
            ax.plot(theta, gauss, "-", color=_GAUSSIAN_COLOR, lw=1.5, label="Gaussian" if first else None)
            # Overlay the requested generalized-family fit(s). A lone fit is drawn blue (the
            # generalized color); "both" keeps generalized blue and shrinkage purple.
            fit_model = state["fit_model"]
            if fit_model in ("generalized", "both"):
                ax.plot(theta, gen, "-", color=_GENERALIZED_COLOR, lw=1.5, label="Generalized" if first else None)
            if fit_model in ("shrinkage", "both"):
                shrink_color = _SHRINKAGE_COLOR if fit_model == "both" else _GENERALIZED_COLOR
                shrink_label = "Generalized (shrinkage)" if fit_model == "both" else "Generalized"
                ax.plot(theta, shrink, "-", color=shrink_color, lw=1.5, label=shrink_label if first else None)
            if first:
                # The anchor box defaults to the axes box (0, 0, 1, 1); offsetting its origin slides
                # the legend off the panel without moving where ``loc`` pins it within the box.
                leg = ax.legend(
                    fontsize=fontsize - 1,
                    frameon=False,
                    loc=state["legend_position"],
                    bbox_to_anchor=(float(state["legend_x_offset"]), float(state["legend_y_offset"]), 1.0, 1.0),
                    markerfirst=bool(state["legend_markerfirst"]),
                    handlelength=float(state["legend_handlelength"]),
                    handletextpad=float(state["legend_handletextpad"]),
                )
                # Keep the constrained layout from reserving space for an off-panel legend, which
                # would shrink the panels and break the uniform grid.
                leg.set_in_layout(False)
                # A legend spilling past its panel is drawn under any axes created after this one, so
                # the neighbour's opaque background would clip it. Draw this panel last, over a
                # transparent patch so raising it does not hide anything underneath.
                ax.set_zorder(10)
                ax.patch.set_visible(False)

        xbounds = (0, theta[-1] + (theta[1] - theta[0]) / 2)
        ylims = [ax.get_ylim() for ax in axs.flat if ax.get_visible()]
        ymax = max(yl[1] for yl in ylims)
        # Extend the drawn y-range a touch below 0 so the test-data points sitting at ~0 are not
        # clipped by the bottom edge; the spine (ybounds) still starts at exactly 0.
        ylims = (-0.05 * ymax, ymax)
        ybounds = (0, np.floor(ymax * 10) / 10)
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
                tick_fontsize=fontsize,
            )
        return fig
