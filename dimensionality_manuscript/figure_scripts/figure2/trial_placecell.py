"""Familiarity and place-field residual views of held-out trial place-cell fits."""

import numpy as np

from dimensionality_manuscript.configs.regression import (
    RESIDUAL_FOLD_SPLITS,
    RESIDUAL_STATISTICS,
    _finite_mean,
    _finite_median,
    _membership_placefield,
    _membership_prediction,
    _placefield_peak,
    _residual_fold_metrics,
)
from dimensionality_manuscript.figure_scripts.panels import data_selection
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import ModelName

from .familiarity import RegressionFamiliarityByEnvViewer
from .pf_residual import ModelPlacefieldResidualInsetViewer, PF_RESIDUAL_REGIONS, _residual_metric_key
from .trial_performance import (
    ENVIRONMENT_OPTIONS,
    SQRT_GAIN_MODEL_COLOR,
    SQRT_GAIN_MODEL_LABEL,
    SQRT_GAIN_MODEL_NAME,
    TRIAL_PERFORMANCE_MODEL_COLORS,
    TRIAL_PERFORMANCE_MODEL_LABELS,
    TRIAL_PERFORMANCE_MODEL_NAMES,
    _reduce_sessions_by_mouse,
)
from .trial_predictions import replay_trial_folds


def _require_trial_placecell_results(results):
    if getattr(getattr(results, "config_class", None), "display_name", None) != "trial_placecell_regression":
        raise ValueError("Pass a ResultsAggregator built from TrialPlacecellRegressionConfig")


class TrialRegressionFamiliarityByEnvViewer(RegressionFamiliarityByEnvViewer):
    """Compare environment-specific held-out trial R² across familiarity.

    Uses the stored trial place-cell R² for each experience-order slot. The default
    comparison is Trial Gain versus external PF. Both models are selectable.
    """

    _score_key = "r2"
    _score_source = "TrialPlacecellRegressionConfig"
    _ylabel_pad = 2

    def __init__(
        self, results: ResultsAggregator, *, reference_model: ModelName = "internal_placefield_1d_structured_gain", **kwargs,
    ):
        _require_trial_placecell_results(results)
        super().__init__(results, reference_model=reference_model, **kwargs)


def _trial_residual_summaries(config, saved, session):
    """Reduce ROI scores within each scored fold, then take the fold median per slot."""
    previous_spks_type = session.params.spks_type
    by_slot = [{} for _ in saved["env_slot_ids"]]
    reducers = {"mean": _finite_mean, "median": _finite_median}
    dist_edges = np.linspace(0, float(np.asarray(session.env_length).flat[0]), 101)
    try:
        session.params.spks_type = config.spks_type
        for slot, model, report, _ in replay_trial_folds(config, saved, session):
            _, _, test_behavior = model.get_session_data(session, config.spks_type, "test")
            if report.extras.get("predictions_were_filtered", False):
                test_behavior = test_behavior.filter(report.extras["idx_valid_predictions"])
            target = np.asarray(report.target_data)
            prediction = np.asarray(report.predicted_data)
            variance = np.var(target, axis=1)
            for fold, split in RESIDUAL_FOLD_SPLITS.items():
                field = _membership_placefield(model, session, config.spks_type, split, dist_edges, None)
                membership = _membership_prediction(field, test_behavior, fold, session)
                metrics = _residual_fold_metrics(target, prediction, membership, _placefield_peak(field), variance)
                for key, values in metrics.items():
                    for statistic in RESIDUAL_STATISTICS:
                        summary_key = f"{statistic}_{fold}_{key}"
                        by_slot[slot].setdefault(summary_key, []).append(reducers[statistic](values))
    finally:
        session.params.spks_type = previous_spks_type
    return [{key: _finite_median(values) for key, values in scores.items()} for scores in by_slot]


class TrialModelPlacefieldResidualInsetViewer(ModelPlacefieldResidualInsetViewer):
    """Within- and outside-PF residuals for trial place-cell models.

    Replays each saved fit with its original target place cells and held-out trials.
    The first load refits using the saved hyperparameters. Changing the metric,
    membership fold, ROI statistic, or environment reuses the cached summaries.
    Each mouse's point averages sessions, with either one environment slot per session
    or an equal-weight average of the finite slots when ``environment="Average"``.
    """

    _roi_subsets = ("all",)

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        model_names: list[ModelName] | None = None,
        include_external_placefield: bool = False,
        environment: str = "Average",
        **kwargs,
    ):
        _require_trial_placecell_results(results)
        if environment not in ENVIRONMENT_OPTIONS:
            raise ValueError(f"environment must be one of {ENVIRONMENT_OPTIONS}")
        available = results.param_axes["model_name"]
        if model_names is None:
            model_names = [name for name in TRIAL_PERFORMANCE_MODEL_NAMES if name in available]
        self._initial_environment = environment
        self._residual_cache = {}
        super().__init__(
            results, model_names=model_names, include_external_placefield=include_external_placefield, **kwargs,
        )
        self.add_selection("environment", value=environment, options=list(ENVIRONMENT_OPTIONS))
        self.on_change("environment", self.refresh_data)

    def _model_styles(self, model_names, include_structured_additive, include_external_placefield):
        if include_structured_additive:
            raise ValueError("Shared Residual is unavailable in TrialPlacecellRegressionConfig")
        model_names = list(model_names)
        if include_external_placefield:
            model_names = ["external_placefield_1d", *(name for name in model_names if name != "external_placefield_1d")]
        available = self.results.param_axes["model_name"]
        if not model_names or any(name not in available for name in model_names):
            raise ValueError(f"model_names must be a nonempty selection from {available}")
        styles = dict(zip(TRIAL_PERFORMANCE_MODEL_NAMES, zip(TRIAL_PERFORMANCE_MODEL_LABELS, TRIAL_PERFORMANCE_MODEL_COLORS)))
        styles.update({
            "external_placefield_1d": ("Ext. PF", "#666666"),
            "external_placefield_1d_gain": ("Ext. Global\nGain", "#a44a4a"),
            SQRT_GAIN_MODEL_NAME: (SQRT_GAIN_MODEL_LABEL, SQRT_GAIN_MODEL_COLOR),
            "external_placefield_1d_structured_gain_residual": ("Ext. Gain\n+ Resid.", "#b66b00"),
            "internal_placefield_1d_structured_gain_residual": ("Gain\n+ Resid.", "#e69f00"),
        })
        labels, colors = zip(*(styles[name] for name in model_names))
        return model_names, list(labels), list(colors)

    def refresh_data(self, state):
        """Read saved fits once and reduce trial-fold summaries by session and mouse."""
        selection = self.results._sel_index(None, data_selection(state, self.results, self.selection_names))[2]
        environment = state.get("environment", self._initial_environment)
        slot = None if environment == "Average" else int(environment.removeprefix("Slot ")) - 1
        scores = {
            region: np.full((len(self.model_names), len(self.results.sessions)), np.nan)
            for region in PF_RESIDUAL_REGIONS
        }
        for model_index, name in enumerate(self.model_names):
            indices = {**selection, "model_name": self.results.param_axes["model_name"].index(name)}
            for ref in self.results._select_cells(param_indices=indices):
                row = ref.sess_idx
                config_key = ref.row["analysis_key"]
                cache_key = (row, config_key)
                if cache_key not in self._residual_cache:
                    saved = self.results._fetch_result(ref)
                    if saved is None:
                        continue
                    config = self.results._key_to_config[config_key]
                    self._residual_cache[cache_key] = _trial_residual_summaries(config, saved, self.results.sessions[row])
                summaries = self._residual_cache[cache_key]
                if slot is not None:
                    summaries = summaries[slot : slot + 1]
                for region in PF_RESIDUAL_REGIONS:
                    key = f"{state['statistic']}_{state['fold']}_{_residual_metric_key(region, state['metric'])}"
                    scores[region][model_index, row] = _finite_mean([summary.get(key, np.nan) for summary in summaries])
        self._scores = {
            region: _reduce_sessions_by_mouse(values.T, self.results.mouse_names, "mean").T
            for region, values in scores.items()
        }
