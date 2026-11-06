"""Existing trial-gain prediction plus residual RRR and exact posthoc projection audits."""

from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from dimilibi import ReducedRankRegression, measure_r2
from vrAnalysis.helpers.optimization import golden_section_search
from vrAnalysis.processors.placefields import get_placefield_prediction

from .hyperparameters import PlaceFieldGainResidualHyperparameters
from .models import PlaceFieldStructuredGainModel, least_squares_gain


GAIN_RESIDUAL_SCORE_NAMES = (
    "mse_gain",
    "mse_residual",
    "r2_gain",
    "r2_residual",
    "r2_template",
    "delta_r2_gain",
    "delta_r2_residual",
    "r2_gain_component",
    "r2_residual_component",
    "sst_activity",
    "sse_template",
    "sse_gain",
    "sse_residual",
    "sse_combined",
    *(
        f"{prefix}_{suffix}"
        for prefix in ("source_residual", "target_residual", "predicted_residual")
        for suffix in ("energy", "gain_energy", "off_template_energy", "gain_energy_fraction")
    ),
)


def template_projection_energy(activity, template, unit_index):
    """Exact per-cell/unit projection, reported as uncentered squared energies.

    Normalize templates before taking dot products, avoiding ill-conditioned raw
    gain ratios even in this posthoc audit. Zero templates project to zero.
    The projection does not modify model inputs or predictions.
    """
    activity = np.asarray(activity, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    total = float(np.sum(activity**2))
    gain_energy = off_energy = 0.0
    for unit in np.unique(unit_index):
        keep = unit_index == unit
        values, field = activity[:, keep], template[:, keep]
        scale = np.max(np.abs(field), axis=1, keepdims=True)
        normalized = np.divide(field, scale, out=np.zeros_like(field), where=scale > 0)
        denominator = np.sum(normalized**2, axis=1, keepdims=True)
        coefficient = np.divide(
            np.sum(values * normalized, axis=1, keepdims=True),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        projection = coefficient * normalized
        gain_energy += float(np.sum(projection**2))
        off_energy += float(np.sum((values - projection) ** 2))
    return dict(
        energy=total, gain_energy=gain_energy, off_template_energy=off_energy, gain_energy_fraction=gain_energy / total if total > 0 else np.nan
    )


class PlaceFieldGainResidualModel(PlaceFieldStructuredGainModel):
    """Reuse structured gain exactly, then predict Y - g_regularized PF.

    Gain hyperparameters are inherited from the matching standalone gain model.
    The inherited gain training and readout preserve its shrinkage and frame rules:
    source gains are measured BEFORE target-template frame filtering. E uses measured,
    regularized gains, not predictions, and can retain template-shaped activity.
    Process metrics quantify that activity with an exact posthoc projection.
    E-only activity predictions are PF + predicted residual; G+E still uses
    predicted gain activity + predicted residual. Raw residual predictions and
    G+E are signed, unprojected, and unrectified.

    Implementation revision 2 uses new optimization/score cache keys.
    TrialPlacecellRegressionConfig retains schema v2; old gain-residual DB results
    must be invalidated when adopting this revised implementation.
    """

    def __init__(
        self,
        registry,
        internal=False,
        fit_intercept=True,
        min_frames_per_unit=5,
        hyperparameters=None,
        activity_parameters=None,
        autosave=True,
    ):
        kwargs = {} if activity_parameters is None else {"activity_parameters": activity_parameters}
        super().__init__(
            registry,
            internal=internal,
            fit_intercept=fit_intercept,
            min_frames_per_unit=min_frames_per_unit,
            hyperparameters=hyperparameters or PlaceFieldGainResidualHyperparameters(),
            autosave=autosave,
            **kwargs,
        )

    @property
    def _model_hyperparameters(self):
        return PlaceFieldGainResidualHyperparameters

    def _get_model_name(self):
        prefix = "internal" if self.internal else "external"
        name = f"{prefix}_placefield_1d_structured_gain_residual"
        return name if self.fit_intercept else name + "_no_intercept"

    def _get_hyperparameter_cache_key(self, *args, **kwargs):
        return super()._get_hyperparameter_cache_key(*args, **kwargs) + "_regularized_v2"

    def _resolve_optimization_method(self, method, allow_best=False):
        if method == "best":
            raise ValueError("Gain-residual models require one reference optimization method; use preferred, golden, grid, or optuna")
        return super()._resolve_optimization_method(method, allow_best=allow_best)

    def regressor_dimensionality(self, num_environments=1, hyperparameters=None):
        hp = hyperparameters or self.hyperparameters
        return hp.num_bins * num_environments + hp.rank + hp.rank_residual

    def inherited_gain_hyperparameters(self, session, spks_type, train_split, validation_split, method="preferred"):
        """Use the same registry, scaling, split names and method as standalone G."""
        reference = PlaceFieldStructuredGainModel(
            self.registry,
            internal=self.internal,
            fit_intercept=self.fit_intercept,
            min_frames_per_unit=self.min_frames_per_unit,
            gain_regularization=self.gain_regularization,
            activity_parameters=self.activity_parameters,
            autosave=self.autosave,
        )
        return reference.get_best_hyperparameters(
            session,
            spks_type=spks_type,
            train_split=train_split,
            validation_split=validation_split,
            method=method,
        )[0]

    def decomposition_inputs(
        self,
        session,
        target_placefield,
        source_placefield,
        spks_type=None,
        split="test",
        nan_safe=False,
        include_target=False,
    ):
        """Keep baseline source gain measurements, then align the two branches.

        Never remeasure source gains on the smaller target-valid output mask.
        Target activity is inspected only for explicitly requested fit/scoring labels.
        Target regularized labels are defined on that output support; units retain
        the original split-wide IDs, including units with no remaining output frames.
        """
        inputs = super().prediction_inputs(
            session,
            target_placefield,
            source_placefield,
            spks_type,
            split,
            nan_safe=nan_safe,
        )
        source, target, _ = self.get_session_data(session, spks_type, split)
        source = np.asarray(source, dtype=np.float64)[:, inputs["idx_valid_prediction"]]
        behavior = inputs.get("frame_behavior_internal", inputs["frame_behavior"])
        source_template = get_placefield_prediction(source_placefield, behavior)[0].T
        valid = ~np.any(np.isnan(inputs["target_prediction"]), axis=0)
        if nan_safe and not np.all(valid):
            raise ValueError("NaN target template predictions")
        if not np.any(valid):
            raise ValueError("No finite template predictions on this split")
        inputs["target_prediction"] = inputs["target_prediction"][:, valid]
        inputs["unit_index"] = inputs["unit_index"][valid]
        inputs["idx_valid_prediction"] = inputs["idx_valid_prediction"][valid]
        inputs["was_filtered"] = inputs["was_filtered"] or not np.all(valid)
        for key in ("frame_behavior", "frame_behavior_internal"):
            if key in inputs:
                inputs[key] = inputs[key].filter(np.flatnonzero(valid))
        source_template = source_template[:, valid]
        inputs["source_template"] = source_template
        inputs["source_residual"] = source[:, valid] - source_template * inputs["source_gain"][:, inputs["unit_index"]]
        if include_target:
            target = np.asarray(target, dtype=np.float64)[:, inputs["idx_valid_prediction"]]
            if not np.all(np.isfinite(target)):
                raise ValueError("Non-finite target activity in fitting/scoring data")
            gain = least_squares_gain(
                target,
                inputs["target_prediction"],
                inputs["unit_index"],
                inputs["num_units"],
                self.gain_regularization,
            )
            inputs["target_gain_activity"] = inputs["target_prediction"] * gain[:, inputs["unit_index"]]
            inputs["target_residual"] = target - inputs["target_gain_activity"]
        return inputs

    def _fit_residual(self, inputs, alpha):
        return ReducedRankRegression(alpha=alpha, fit_intercept=self.fit_intercept).fit(
            torch.tensor(inputs["source_residual"].T, dtype=torch.float64),
            torch.tensor(inputs["target_residual"].T, dtype=torch.float64),
        )

    def train(self, session, spks_type=None, split="train", hyperparameters=None):
        hp = hyperparameters or self.hyperparameters
        target_pf, source_pf, gain_model = super().train(session, spks_type, split, hp)
        inputs = self.decomposition_inputs(session, target_pf, source_pf, spks_type, split, include_target=True)
        return target_pf, source_pf, gain_model, self._fit_residual(inputs, hp.alpha_residual)

    def apply_models(self, inputs, gain_model, residual_model, hyperparameters):
        gain_activity, extras = self.apply_gain_model(inputs, gain_model, hyperparameters.rank, nan_safe=True)
        residual = (
            residual_model.predict(
                torch.tensor(inputs["source_residual"].T, dtype=torch.float64),
                rank=min(int(hyperparameters.rank_residual), int(residual_model.max_rank)),
                nonnegative=False,
            )
            .numpy()
            .T
        )
        if not np.all(np.isfinite(gain_activity)) or not np.all(np.isfinite(residual)):
            raise ValueError("Non-finite gain/residual predictions")
        extras.update(
            gain_prediction=gain_activity,
            residual_prediction=residual,
            placefield_prediction=inputs["target_prediction"],
            residual_source=inputs["source_residual"],
            source_template=inputs["source_template"],
            gain_regularization=self.gain_regularization,
            num_gain_units=inputs["num_units"],
        )
        return gain_activity + residual, extras

    def predict(self, session, coefficients, spks_type=None, split="test", hyperparameters=None, nan_safe=False):
        target_pf, source_pf, gain_model, residual_model = coefficients
        inputs = self.decomposition_inputs(session, target_pf, source_pf, spks_type, split, nan_safe=nan_safe)
        return self.apply_models(inputs, gain_model, residual_model, hyperparameters or self.hyperparameters)

    def process(self, *args, **kwargs):
        report = super().process(*args, **kwargs)
        report.metrics.update(gain_residual_scores(report))
        return report

    def _optimize_residual(self, session, spks_type, train_split, validation_split, method, nan_safe=False):
        """Freeze reference G settings; tune only E prediction on validation trials."""
        gain_hp = self.inherited_gain_hyperparameters(session, spks_type, train_split, validation_split, method)
        hp = PlaceFieldGainResidualHyperparameters(
            num_bins=gain_hp.num_bins,
            smooth_width=gain_hp.smooth_width,
            alpha=gain_hp.alpha,
            rank=gain_hp.rank,
        )
        target_pf, source_pf, gain_model = super().train(session, spks_type, train_split, hp)
        train = self.decomposition_inputs(session, target_pf, source_pf, spks_type, train_split, nan_safe=nan_safe, include_target=True)
        validation = self.decomposition_inputs(session, target_pf, source_pf, spks_type, validation_split, nan_safe=nan_safe, include_target=True)
        max_rank = int(self._fit_residual(train, hp.alpha_residual).max_rank)
        records = []

        def evaluate(alpha, rank, fitted=None):
            fitted = fitted if fitted is not None else self._fit_residual(train, alpha)
            prediction = (
                fitted.predict(
                    torch.tensor(validation["source_residual"].T, dtype=torch.float64),
                    rank=int(rank),
                    nonnegative=False,
                )
                .numpy()
                .T
            )
            score = float(np.mean((validation["target_residual"] - prediction) ** 2))
            records.append(dict(alpha_residual=float(alpha), rank_residual=int(rank), score=score if np.isfinite(score) else np.inf))
            return records[-1]["score"]

        if method == "golden":
            alpha = golden_section_search(
                func=lambda alpha: evaluate(alpha, max_rank),
                a=1e-2,
                b=1e6,
                tolerance_param=1e-2,
                tolerance_score=1e-3,
                max_iterations=25,
                minimize=True,
                logspace=True,
            )[0]
            fitted = self._fit_residual(train, alpha)
            if max_rank == 1:
                evaluate(alpha, 1, fitted)
            else:
                golden_section_search(
                    func=lambda rank: evaluate(alpha, int(rank), fitted),
                    a=1.0,
                    b=float(max_rank),
                    tolerance_param=1.0,
                    tolerance_score=1e-3,
                    max_iterations=25,
                    minimize=True,
                    logspace=False,
                )
        elif method == "grid":
            space = hp.get_search_space()
            for alpha in space["training"]["alpha_residual"]:
                fitted = self._fit_residual(train, alpha)
                for rank in sorted({min(int(r), max_rank) for r in space["prediction"]["rank_residual"]}):
                    evaluate(alpha, rank, fitted)
        else:
            import optuna

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
            study.optimize(
                lambda trial: evaluate(
                    trial.suggest_float("alpha_residual", 1e-2, 1e10, log=True),
                    trial.suggest_int("rank_residual", 1, max_rank),
                ),
                n_trials=50,
                show_progress_bar=False,
            )
        best = min(records, key=lambda row: row["score"])
        if not np.isfinite(best["score"]):
            raise ValueError("No finite validation score for residual regression")
        hp = replace(hp, alpha_residual=best["alpha_residual"], rank_residual=best["rank_residual"])
        residual_model = self._fit_residual(train, hp.alpha_residual)
        prediction, _ = self.apply_models(validation, gain_model, residual_model, hp)
        target = validation["target_gain_activity"] + validation["target_residual"]
        params = {key: getattr(hp, key) for key in ("num_bins", "smooth_width", "alpha", "rank", "alpha_residual", "rank_residual")}
        table = pd.DataFrame(records).assign(alpha=hp.alpha, rank=hp.rank, num_bins=hp.num_bins, smooth_width=hp.smooth_width)
        return params, float(np.mean((target - prediction) ** 2)), table

    def _optimize_golden(self, session, spks_type, train_split, validation_split, nan_safe=False):
        return self._optimize_residual(session, spks_type, train_split, validation_split, "golden", nan_safe)

    def _optimize_grid(self, session, spks_type, train_split, validation_split, nan_safe=False):
        return self._optimize_residual(session, spks_type, train_split, validation_split, "grid", nan_safe)

    def _optimize_optuna(self, session, spks_type, train_split, validation_split, nan_safe=False):
        return self._optimize_residual(session, spks_type, train_split, validation_split, "optuna", nan_safe)


def gain_residual_scores(report):
    """Common-denominator prediction scores and exact posthoc template energies.

    Fractions use squared norm, not centered variance. E uses measured regularized
    gains. Predicted E is audited separately. Energies sum across folds; fractions
    must be pooled using those energies rather than averaged over folds.
    E-only activity scores use template + predicted E. Component scores and
    projection audits continue to use raw E.
    """
    metrics = getattr(report, "metrics", {})
    if all(key in metrics for key in GAIN_RESIDUAL_SCORE_NAMES):
        return {key: metrics[key] for key in GAIN_RESIDUAL_SCORE_NAMES}
    target = np.asarray(report.target_data, dtype=np.float64)
    gain, residual = report.extras["gain_prediction"], report.extras["residual_prediction"]
    template, units = report.extras["placefield_prediction"], report.extras["gain_unit_index"]
    residual_activity = template + residual
    sst = float(np.sum((target - target.mean()) ** 2))
    sse_template, sse_gain = float(np.sum((target - template) ** 2)), float(np.sum((target - gain) ** 2))
    sse_residual, sse_combined = float(np.sum((target - residual_activity) ** 2)), float(np.sum((target - gain - residual) ** 2))
    measured_gains = least_squares_gain(
        target,
        template,
        units,
        report.extras["num_gain_units"],
        report.extras["gain_regularization"],
    )
    measured_gain = template * measured_gains[:, units]
    measured_residual = target - measured_gain
    scores = dict(
        mse_gain=float(np.mean((target - gain) ** 2)),
        mse_residual=float(np.mean((target - residual_activity) ** 2)),
        r2_gain=float(measure_r2(gain, target, dim=None)),
        r2_residual=float(measure_r2(residual_activity, target, dim=None)),
        r2_template=float(measure_r2(template, target, dim=None)),
        r2_gain_component=float(measure_r2(gain, measured_gain, dim=None)),
        r2_residual_component=float(measure_r2(residual, measured_residual, dim=None)),
        delta_r2_gain=(sse_template - sse_gain) / sst if sst > 0 else np.nan,
        delta_r2_residual=(sse_gain - sse_combined) / sst if sst > 0 else np.nan,
        sst_activity=sst,
        sse_template=sse_template,
        sse_gain=sse_gain,
        sse_residual=sse_residual,
        sse_combined=sse_combined,
    )
    for prefix, activity, field in (
        ("source_residual", report.extras["residual_source"], report.extras["source_template"]),
        ("target_residual", measured_residual, template),
        ("predicted_residual", residual, template),
    ):
        scores.update({f"{prefix}_{key}": value for key, value in template_projection_energy(activity, field, units).items()})
    return scores
