"""Regression performance over familiarity: R^2 against a mouse's own session number."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import ModelName
from dimensionality_manuscript.env_order import ENV_SLOT_COLORS
from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, apply_legend, update_legend_widgets
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
    style_model_axis,
)

from ._scores import (
    SESSION_XLABEL,
    add_familiarity_curve_widgets,
    chronological_mouse_sessions,
    draw_familiarity_series,
    mouse_session_curves,
    performance_scores,
)
from .model_style import ROLE_COLOR

from .performance import (
    STRUCTURED_ADDITIVE_MODEL_NAME,
    STRUCTURED_ADDITIVE_MODEL_LABEL,
    STRUCTURED_ADDITIVE_MODEL_COLOR,
)

# In improvement mode, comparison curves are measured relative to the internal place-field model,
# which is a subtraction baseline rather than a displayed series.
FAMILIARITY_IMPROVEMENT_CONTROL: ModelName = "internal_placefield_1d"
FAMILIARITY_PF_GAIN_MODEL: ModelName = "internal_placefield_1d_gain"
FAMILIARITY_STRUCTURED_ADDITIVE_MODEL: ModelName = STRUCTURED_ADDITIVE_MODEL_NAME
FAMILIARITY_PEER_MODEL: ModelName = "rrr"

FAMILIARITY_PF_COLOR = ROLE_COLOR["external"]
FAMILIARITY_PF_GAIN_COLOR = ROLE_COLOR["internal"]
FAMILIARITY_STRUCTURED_ADDITIVE_COLOR = STRUCTURED_ADDITIVE_MODEL_COLOR
FAMILIARITY_PEER_COLOR = ROLE_COLOR["neural"]
FAMILIARITY_PF_LABEL = "PF"
FAMILIARITY_PF_GAIN_LABEL = "Gain"
FAMILIARITY_STRUCTURED_ADDITIVE_LABEL = "Residual"
FAMILIARITY_PEER_LABEL = "Peer"

# Plain PF is the reference for every inferential comparison. The interaction coefficients are
# therefore the changes in each enabled model's advantage over PF per session.
FAMILIARITY_STAT_MODEL_LABELS = {
    FAMILIARITY_IMPROVEMENT_CONTROL: "pf",
    FAMILIARITY_PF_GAIN_MODEL: "pf_gain",
    FAMILIARITY_STRUCTURED_ADDITIVE_MODEL: "peer_residual",
    FAMILIARITY_PEER_MODEL: "peer",
}

FAMILIARITY_MODEL_LABELS = {
    FAMILIARITY_IMPROVEMENT_CONTROL: FAMILIARITY_PF_LABEL,
    FAMILIARITY_PF_GAIN_MODEL: FAMILIARITY_PF_GAIN_LABEL,
    FAMILIARITY_STRUCTURED_ADDITIVE_MODEL: FAMILIARITY_STRUCTURED_ADDITIVE_LABEL,
    FAMILIARITY_PEER_MODEL: FAMILIARITY_PEER_LABEL,
}
FAMILIARITY_MODEL_COLORS = {
    FAMILIARITY_IMPROVEMENT_CONTROL: FAMILIARITY_PF_COLOR,
    FAMILIARITY_PF_GAIN_MODEL: FAMILIARITY_PF_GAIN_COLOR,
    FAMILIARITY_STRUCTURED_ADDITIVE_MODEL: FAMILIARITY_STRUCTURED_ADDITIVE_COLOR,
    FAMILIARITY_PEER_MODEL: FAMILIARITY_PEER_COLOR,
}

# y-axis label per single-axis value mode. "improvement" subtracts the internal place-field
# model's score session by session, matching the Delta R^2 convention of the other figure-2
# performance panels. "both" composes those two single-axis views side by side.
FAMILIARITY_VALUE_LABELS: dict[str, str] = {
    "absolute": r"R$^2$",
    "improvement": r"$\Delta$ R$^2$",
}
FAMILIARITY_VALUE_MODES = (*FAMILIARITY_VALUE_LABELS, "both")


class RegressionFamiliarityViewer(FigureViewer):
    """R^2 over sessions for PF+Gain, optional Peer Residual, and Peer Prediction.

    The counterpart of figure3's familiarity panel for regression scores: each mouse's sessions
    are ordered chronologically and renumbered from 0, then the per-mouse curves are averaged
    across mice. A ``RegressionConfig`` score is a whole-session number, so the x-axis is the
    mouse's own session index -- there is nothing per-environment to align to.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``RegressionConfig`` results, with ``model_name`` as a param axis covering
        ``"internal_placefield_1d_gain"``, ``"rrr"``, and ``"internal_placefield_1d"``.
    include_structured_additive : bool
        Include the structured additive model in the curves and stats.
    value_mode : {"absolute", "improvement", "both"}
        ``"absolute"`` plots each displayed model's R^2. ``"improvement"`` subtracts the
        ``"internal_placefield_1d"`` score from each displayed model session by session. The
        subtraction baseline is not drawn. ``"both"`` places the absolute view on the left and
        the improvement view on the right.
    single_env_only : bool
        Keep only sessions that ran exactly one environment (the invalid negative environment
        sentinel is not counted). In practice that is each mouse's run of early sessions, before
        a second environment was introduced, so the curves stay within one familiarity regime.
        Dropped sessions are removed rather than left as gaps, so index 0 is still a mouse's
        first plotted session.
    auto_ylim : bool
        Fit the y limits and the upper y tick to the drawn data, ignoring ``ylim`` and
        ``ytick_max``.
    ylim : tuple[float, float]
        Y limits when ``auto_ylim`` is False, spanned by the offset left spine.
    ytick_max : float
        Upper y tick when ``auto_ylim`` is False (the lower one is 0).
    show_zero_line : bool
        Draw the dashed y = 0 reference line. Only applies to ``value_mode="improvement"``.
    fontsize : float
        Font size for axis labels, tick labels, and the legend.
    figsize : tuple[float, float]
        Figure size in inches for one value-mode axis. ``value_mode="both"`` doubles the width.
    legend_options : dict or None
        Legend knobs forwarded to :mod:`~dimensionality_manuscript.figure_scripts.legends`
        (``{"loc": ..., "ncols": ...}``); ``x_offset`` and ``y_offset`` shift the legend
        from that location in axes-relative coordinates, and ``{"loc": "none"}`` hides it.
    **curve_and_selection
        Curve-rendering knobs (``style``, ``se``, ``min_mice``, ``linewidth``,
        ``subject_linewidth``, ``subject_alpha``, ``fill_alpha``) and starting values for the
        data-selection widgets built from the aggregator's param axes.
    """

    _CURVE_KNOBS = ("style", "se", "min_mice", "linewidth", "subject_linewidth", "subject_alpha", "fill_alpha")

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        include_structured_additive: bool = False,
        value_mode: str = "absolute",
        single_env_only: bool = False,
        auto_ylim: bool = True,
        ylim: tuple[float, float] = (0.0, 0.3),
        ytick_max: float = 0.3,
        show_zero_line: bool = True,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (3.5, 2.5),
        legend_options: dict | None = None,
        **curve_and_selection,
    ):
        if value_mode not in FAMILIARITY_VALUE_MODES:
            raise ValueError(f"value_mode must be one of {list(FAMILIARITY_VALUE_MODES)}, got {value_mode!r}")

        self.results = results
        self.figsize = figsize
        self.include_structured_additive = include_structured_additive
        self.comparison_model_names = [FAMILIARITY_PF_GAIN_MODEL]
        if include_structured_additive:
            self.comparison_model_names.append(FAMILIARITY_STRUCTURED_ADDITIVE_MODEL)
        self.comparison_model_names.append(FAMILIARITY_PEER_MODEL)

        # PF is loaded even in improvement mode because it is the subtraction baseline. Whether
        # it is displayed is decided per value mode in _plot_value_mode(), which also lets the
        # "both" view share this one score selection.
        self.score_model_names = [FAMILIARITY_IMPROVEMENT_CONTROL, *self.comparison_model_names]
        self._scores: dict[str, np.ndarray] = {}

        curve_kwargs = {name: curve_and_selection.pop(name) for name in self._CURVE_KNOBS if name in curve_and_selection}
        self.selection_names = add_data_selection_widgets(self, results, skip=("model_name",), defaults=curve_and_selection)
        self.add_selection("value_mode", value=value_mode, options=list(FAMILIARITY_VALUE_MODES))
        self.add_boolean("single_env_only", value=single_env_only)

        add_familiarity_curve_widgets(self, results, **curve_kwargs)

        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        self.add_boolean("auto_ylim", value=auto_ylim)
        self.add_float_range("ylim", value=ylim, min=-1.0, max=2.0, step=0.01)
        self.add_float("ytick_max", value=ytick_max, min=0.01, max=2.0, step=0.01)
        self.add_boolean("show_zero_line", value=show_zero_line)

        legend_options = dict(legend_options or {})
        self.add_float(
            "legend_x_offset",
            value=float(legend_options.pop("x_offset", 0.0)),
            min=-2.0,
            max=2.0,
            step=0.01,
        )
        self.add_float(
            "legend_y_offset",
            value=float(legend_options.pop("y_offset", 0.0)),
            min=-2.0,
            max=2.0,
            step=0.01,
        )
        add_legend_widgets(self)
        update_legend_widgets(self, legend_options)

        for name in self.selection_names:
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Re-select per-session R^2 for the displayed comparisons and PF baseline."""
        scores = performance_scores(
            self.results,
            self.score_model_names,
            "r2",
            data_selection(state, self.results, self.selection_names),
            avg_by_mouse=False,
        )
        self._scores = dict(zip(self.score_model_names, scores))

    def measure_statistics(self, state: dict) -> dict:
        """Fit the absolute-score familiarity model for PF and the enabled comparisons.

        The fitted model is::

            r2 ~ model * session + (1 + session | mouse)

        where ``session`` is numeric (each mouse's chronological, zero-based session index),
        ``model`` is a factor with plain PF as its reference level. Consequently, each
        ``model:session`` coefficient is the change in that model's advantage over PF per
        session. When enabled, Peer Residual is treated exactly like the other comparisons.

        This method always re-selects the models' stored, absolute R^2 values.  In particular,
        ``state["value_mode"]`` has no effect: an ``"improvement"`` or ``"both"`` view is never
        analyzed as a difference score.  Data-selection state and ``single_env_only`` still define
        which underlying observations are in scope, just as they do for the displayed curves.

        Parameters
        ----------
        state : dict
            A viewer state, normally ``viewer.state``.

        Returns
        -------
        dict
            ``model`` is the fitted statsmodels ``MixedLMResults`` object and ``data`` is its tidy
            input table. ``fixed_effects`` contains beta, SE, z, p, and 95% Wald-CI columns for
            every fixed effect. ``interactions`` repeats the model-by-session estimates in
            manuscript-friendly records and gives ``minimum_symmetric_equivalence_bound`` for
            each: the smallest +/- bound containing that 95% CI. ``interaction_omnibus`` is the
            joint two-df Wald test. Sample sizes, model specification, and convergence status are
            also included.
        """
        selection = data_selection(state, self.results, self.selection_names)
        model_names = self.score_model_names
        absolute_scores = performance_scores(
            self.results,
            model_names,
            "r2",
            selection,
            avg_by_mouse=False,
        )

        rows = []
        single_env_only = bool(state.get("single_env_only", False))
        for mouse in self.results.unique_mice:
            session_indices = chronological_mouse_sessions(
                self.results,
                mouse,
                single_env_only=single_env_only,
            )
            for session, result_index in enumerate(session_indices):
                for model_index, model_name in enumerate(model_names):
                    r2 = absolute_scores[model_index, result_index]
                    if np.isfinite(r2):
                        rows.append(
                            {
                                "mouse": str(mouse),
                                "session": float(session),
                                "model": FAMILIARITY_STAT_MODEL_LABELS[model_name],
                                "r2": float(r2),
                            }
                        )

        table = pd.DataFrame(rows, columns=["mouse", "session", "model", "r2"])
        model_categories = [FAMILIARITY_STAT_MODEL_LABELS[name] for name in model_names]
        table["model"] = pd.Categorical(table["model"], categories=model_categories, ordered=True)
        if table.empty:
            raise ValueError("No finite absolute R^2 observations are available for the mixed model")
        if table["mouse"].nunique() < 2:
            raise ValueError("The mixed model requires absolute R^2 observations from at least two mice")
        if table["session"].nunique() < 2:
            raise ValueError("The mixed model requires at least two numeric session indices")
        if table["model"].nunique() < len(model_names):
            required = ", ".join(FAMILIARITY_MODEL_LABELS[name] for name in model_names)
            raise ValueError(f"The mixed model requires finite observations from {required}")

        formula = "r2 ~ model * session"
        random_effects_formula = "~session"
        fitted = smf.mixedlm(
            formula,
            table,
            groups=table["mouse"],
            re_formula=random_effects_formula,
        ).fit(reml=True, method=["lbfgs", "powell"])

        fixed_names = list(fitted.fe_params.index)
        beta = fitted.fe_params.astype(float)
        se = pd.Series(np.asarray(fitted.bse_fe, dtype=float), index=fixed_names)
        ci = fitted.conf_int(alpha=0.05).loc[fixed_names].astype(float)
        z = beta / se
        p = fitted.pvalues.loc[fixed_names].astype(float)
        fixed_effects = pd.DataFrame(
            {
                "beta": beta,
                "se": se,
                "z": z,
                "p": p,
                "ci_low": ci.iloc[:, 0],
                "ci_high": ci.iloc[:, 1],
            }
        )

        interaction_terms = {
            f"{FAMILIARITY_STAT_MODEL_LABELS[name]}_vs_pf": f"model[T.{FAMILIARITY_STAT_MODEL_LABELS[name]}]:session"
            for name in self.comparison_model_names
        }
        interactions = {}
        for comparison, interaction_term in interaction_terms.items():
            interaction_row = fixed_effects.loc[interaction_term]
            ci_low = float(interaction_row["ci_low"])
            ci_high = float(interaction_row["ci_high"])
            interactions[comparison] = {
                "term": interaction_term,
                "beta": float(interaction_row["beta"]),
                "se": float(interaction_row["se"]),
                "z": float(interaction_row["z"]),
                "p": float(interaction_row["p"]),
                "ci_level": 0.95,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "minimum_symmetric_equivalence_bound": max(abs(ci_low), abs(ci_high)),
                "units": "R2 per session",
            }

        # Joint Wald test that all model-by-session coefficients are zero. MixedLM's full
        # parameter vector also contains covariance parameters, hence the explicit full-width R.
        restriction = np.zeros((len(interaction_terms), len(fitted.params)))
        for row, term in enumerate(interaction_terms.values()):
            restriction[row, list(fitted.params.index).index(term)] = 1.0
        omnibus_test = fitted.wald_test(restriction, scalar=True)
        interaction_omnibus = {
            "statistic": float(np.asarray(omnibus_test.statistic).squeeze()),
            "df": len(interaction_terms),
            "p": float(np.asarray(omnibus_test.pvalue).squeeze()),
            "test": "Wald chi-square",
        }

        sessions_per_mouse = table.groupby("mouse", observed=True)["session"].nunique().astype(int).to_dict()
        return {
            "model": fitted,
            "data": table,
            "formula": formula,
            "random_effects_formula": random_effects_formula,
            "estimation": "REML",
            "baseline_model": FAMILIARITY_IMPROVEMENT_CONTROL,
            "comparison_models": list(self.comparison_model_names),
            "fixed_effects": fixed_effects,
            "interactions": interactions,
            "interaction_omnibus": interaction_omnibus,
            "n_observations": int(fitted.nobs),
            "n_mice": int(table["mouse"].nunique()),
            "n_mouse_sessions": int(table[["mouse", "session"]].drop_duplicates().shape[0]),
            "sessions_per_mouse": sessions_per_mouse,
            "observations_by_model": {str(model): int(count) for model, count in table.groupby("model", observed=True).size().items()},
            "converged": bool(fitted.converged),
        }

    def _plot_value_mode(self, ax, state, value_mode: str) -> None:
        """Draw one axis exactly as the corresponding single-mode view."""
        fontsize = state["fontsize"]
        improvement = value_mode == "improvement"
        model_names = self.comparison_model_names if improvement else self.score_model_names
        control = self._scores[FAMILIARITY_IMPROVEMENT_CONTROL]
        series = [
            (
                FAMILIARITY_MODEL_LABELS[model_name],
                FAMILIARITY_MODEL_COLORS[model_name],
                self._scores[model_name] - control if improvement else self._scores[model_name],
            )
            for model_name in model_names
        ]

        if improvement and state["show_zero_line"]:
            ax.axhline(0.0, color="k", linewidth=0.5, linestyle="--")

        extents = []
        visible_values = []
        for label, color, values in series:
            extent, visible = draw_familiarity_series(
                ax,
                mouse_session_curves(self.results, values, state["single_env_only"]),
                color,
                label,
                state,
            )
            extents.append(extent)
            visible_values.append(visible[np.isfinite(visible)])

        finite = np.concatenate(visible_values) if visible_values else np.array([])
        if state["auto_ylim"] and finite.size:
            low, high = min(float(finite.min()), 0.0), float(finite.max())
            pad = 0.05 * (high - low) if high > low else 0.05
            ylim = (low - pad, high + pad)
            ytick_max = round(high, 2)
        else:
            ylim = state["ylim"]
            ytick_max = state["ytick_max"]

        ax.set_ylim(ylim)
        style_model_axis(
            ax,
            fontsize=fontsize,
            xbounds=[0, max(max(extents) - 1, 1)],
            ybounds=ylim,
            yticks=[0, ytick_max],
        )
        ax.set_xlabel(SESSION_XLABEL, fontsize=fontsize)
        ax.set_ylabel(FAMILIARITY_VALUE_LABELS[value_mode], labelpad=-10, fontsize=fontsize)
        apply_legend(
            ax,
            state,
            fontsize,
            auto_loc="lower right",
            offset=(state["legend_x_offset"], state["legend_y_offset"]),
        )

    def plot(self, state):
        value_modes = ("absolute", "improvement") if state["value_mode"] == "both" else (state["value_mode"],)
        fig, axes = self.new_subplots(
            1,
            len(value_modes),
            figsize=self.figsize,
            layout="constrained",
            squeeze=False,
        )
        for ax, value_mode in zip(axes.ravel(), value_modes):
            self._plot_value_mode(ax, state, value_mode)
        return fig


class RegressionFamiliarityByEnvViewer(FigureViewer):
    """Per-environment R^2 and improvement over the internal place-field control.

    Each curve is one experience-order environment slot, using the shared black/blue/green
    palette.  The left axis shows ``r2_env`` for the model selected by ``model_name``; the right
    axis shows that score minus ``internal_placefield_1d`` for the same session and slot.

    When ``within_env_session`` is true, each mouse and slot is compacted to the sessions in
    which that environment has a finite score.  Otherwise values retain their position in the
    mouse's complete chronological session history, leaving gaps where the environment was not
    run.
    """

    _CURVE_KNOBS = RegressionFamiliarityViewer._CURVE_KNOBS

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        model_name: ModelName = FAMILIARITY_PF_GAIN_MODEL,
        within_env_session: bool = True,
        auto_ylim: bool = True,
        ylim: tuple[float, float] = (0.0, 0.3),
        ytick_max: float = 0.3,
        show_zero_line: bool = True,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (7.0, 2.5),
        legend_options: dict | None = None,
        **curve_and_selection,
    ):
        available_models = list(results.param_axes["model_name"])
        if model_name not in available_models:
            raise ValueError(f"model_name must be one of {available_models}, got {model_name!r}")
        if FAMILIARITY_IMPROVEMENT_CONTROL not in available_models:
            raise ValueError(f"Results must include the {FAMILIARITY_IMPROVEMENT_CONTROL!r} control model")

        self.results = results
        self.figsize = figsize
        self._model_scores = np.empty((0, 0))
        self._control_scores = np.empty((0, 0))

        curve_kwargs = {name: curve_and_selection.pop(name) for name in self._CURVE_KNOBS if name in curve_and_selection}
        self.selection_names = add_data_selection_widgets(
            self,
            results,
            skip=("model_name",),
            defaults=curve_and_selection,
        )
        self.add_selection("model_name", value=model_name, options=available_models)
        self.add_boolean("within_env_session", value=within_env_session)
        add_familiarity_curve_widgets(self, results, **curve_kwargs)

        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
        self.add_boolean("auto_ylim", value=auto_ylim)
        self.add_float_range("ylim", value=ylim, min=-1.0, max=2.0, step=0.01)
        self.add_float("ytick_max", value=ytick_max, min=0.01, max=2.0, step=0.01)
        self.add_boolean("show_zero_line", value=show_zero_line)
        add_legend_widgets(self)
        update_legend_widgets(self, legend_options or {})

        for name in (*self.selection_names, "model_name"):
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _select_scores(self, model_name: ModelName, selection: dict) -> np.ndarray:
        scores = self.results.sel(
            model_name=model_name,
            avg_by_mouse=False,
            **selection,
        )["r2_env"]
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2:
            raise ValueError(f"Expected r2_env with shape (sessions, environment slots), got {scores.shape}")
        return scores

    def refresh_data(self, state):
        """Select the current model and its session-matched PF control scores."""
        selection = data_selection(state, self.results, self.selection_names)
        self._model_scores = self._select_scores(state["model_name"], selection)
        self._control_scores = self._select_scores(FAMILIARITY_IMPROVEMENT_CONTROL, selection)
        if self._model_scores.shape != self._control_scores.shape:
            raise ValueError("Selected model and internal_placefield_1d r2_env arrays must have matching shapes")

    def _slot_curves(self, values: np.ndarray, slot: int, within_env_session: bool) -> dict[str, np.ndarray]:
        """Return one slot's curves in within-environment or overall-session coordinates."""
        curves = {}
        for mouse in self.results.unique_mice:
            rows = chronological_mouse_sessions(self.results, mouse)
            mouse_values = np.asarray(values[rows, slot], dtype=float)
            if within_env_session:
                mouse_values = mouse_values[np.isfinite(mouse_values)]
            if mouse_values.size:
                curves[str(mouse)] = mouse_values
        return curves

    @staticmethod
    def _fit_axis(ax, finite: np.ndarray, extents: list[int], state, ylabel: str) -> None:
        if state["auto_ylim"] and finite.size:
            low, high = min(float(finite.min()), 0.0), float(finite.max())
            pad = 0.05 * (high - low) if high > low else 0.05
            ylim = (low - pad, high + pad)
            ytick_max = round(high, 2)
        else:
            ylim = state["ylim"]
            ytick_max = state["ytick_max"]
        ax.set_ylim(ylim)
        style_model_axis(
            ax,
            fontsize=state["fontsize"],
            xbounds=[0, max(max(extents, default=0) - 1, 1)],
            ybounds=ylim,
            yticks=[0, ytick_max],
        )
        ax.set_ylabel(ylabel, labelpad=-10, fontsize=state["fontsize"])

    def _draw_panel(self, ax, values: np.ndarray, state, *, improvement: bool) -> None:
        if improvement and state["show_zero_line"]:
            ax.axhline(0.0, color="k", linewidth=0.5, linestyle="--")

        extents = []
        visible_values = []
        for slot in range(min(values.shape[1], len(ENV_SLOT_COLORS))):
            extent, visible = draw_familiarity_series(
                ax,
                self._slot_curves(values, slot, state["within_env_session"]),
                ENV_SLOT_COLORS[slot],
                f"Env #{slot + 1}",
                state,
            )
            extents.append(extent)
            visible_values.append(visible[np.isfinite(visible)])

        finite = np.concatenate(visible_values) if visible_values else np.array([])
        self._fit_axis(
            ax,
            finite,
            extents,
            state,
            FAMILIARITY_VALUE_LABELS["improvement" if improvement else "absolute"],
        )
        xlabel = "Environment Session #" if state["within_env_session"] else SESSION_XLABEL
        ax.set_xlabel(xlabel, fontsize=state["fontsize"])
        apply_legend(ax, state, state["fontsize"], auto_loc="lower right")

    def plot(self, state):
        fig, axes = self.new_subplots(
            1,
            2,
            figsize=self.figsize,
            layout="constrained",
            squeeze=False,
        )
        absolute = self._model_scores
        improvement = self._model_scores - self._control_scores
        self._draw_panel(axes[0, 0], absolute, state, improvement=False)
        self._draw_panel(axes[0, 1], improvement, state, improvement=True)
        return fig
