"""Predict trial-template amplitudes and the remaining temporal activity separately."""

from copy import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from dimilibi import ReducedRankRegression, measure_r2
from vrAnalysis.processors.placefields import get_placefield_prediction
from vrAnalysis.helpers.optimization import golden_section_search

from .hyperparameters import PlaceFieldGainResidualHyperparameters
from .models import PlaceFieldModel, PlaceFieldStructuredGainModel, least_squares_gain, make_gain_unit_index


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
)


class PlaceFieldGainResidualModel(PlaceFieldStructuredGainModel):
    """Decompose Y = G + E and learn g_source -> g_target and E_source -> E_target.

    G is the unregularized least-squares projection onto each cell's template within
    each gain unit (a whole trial with TrialRegistry). E = Y - G is measured, rather
    than formed by subtracting predicted gains. Internal position is decoded with
    frozen, actual-position training maps, exactly as in the structured gain model.

    Held-out prediction uses only source activity and frozen maps. Target components
    are measured only during fitting, validation, and scoring. Residual predictions
    are signed and neither projected off-template nor rectified; their sum with G
    is also left unrectified so the two predictions remain literally additive.
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
            gain_regularization=0.0,
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

    def regressor_dimensionality(self, num_environments=1, hyperparameters=None):
        hp = hyperparameters or self.hyperparameters
        return hp.num_bins * num_environments + hp.rank + hp.rank_residual

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
        """Use one common frame mask for both branches; never inspect held-out Y."""
        source, target, behavior = self.get_session_data(session, spks_type, split)
        true_behavior = copy(behavior)
        if self.internal:
            behavior = self.decode_internal_frame_behavior(source, source_placefield, behavior)
        source_template = get_placefield_prediction(source_placefield, behavior)[0].T
        target_template = get_placefield_prediction(target_placefield, behavior)[0].T
        source = np.asarray(source, dtype=np.float64)
        valid = np.all(np.isfinite(source), axis=0) & np.all(np.isfinite(source_template), axis=0) & np.all(np.isfinite(target_template), axis=0)
        if nan_safe and not np.all(valid):
            raise ValueError("Non-finite source activity or template predictions")
        if not np.any(valid):
            raise ValueError("No finite template predictions on this split")
        units, _ = make_gain_unit_index(
            self.get_split_chunk_index(session, spks_type, split),
            behavior.trial,
        )
        # Remap after filtering: trials with no usable frames must not affect shrinkage,
        # rank bounds, or the training observation count.
        _, units = np.unique(units[valid], return_inverse=True)
        num_units = int(units.max()) + 1
        source = source[:, valid]
        source_template = source_template[:, valid]
        target_template = target_template[:, valid]
        source_gain = least_squares_gain(source, source_template, units, num_units)
        inputs = dict(
            source_gain=source_gain,
            source_residual=source - source_template * source_gain[:, units],
            target_prediction=target_template,
            unit_index=units,
            num_units=num_units,
            frame_behavior=true_behavior.filter(np.flatnonzero(valid)),
            idx_valid_prediction=np.flatnonzero(valid),
            was_filtered=not np.all(valid),
        )
        if self.internal:
            inputs["frame_behavior_internal"] = behavior.filter(np.flatnonzero(valid))
        if include_target:
            target = np.asarray(target, dtype=np.float64)[:, valid]
            if not np.all(np.isfinite(target)):
                raise ValueError("Non-finite target activity in fitting/scoring data")
            target_gain = least_squares_gain(target, target_template, units, num_units)
            inputs["target_gain"] = target_gain
            inputs["target_gain_activity"] = target_template * target_gain[:, units]
            inputs["target_residual"] = target - inputs["target_gain_activity"]
        return inputs

    def _fit_branch(self, inputs, branch, alpha):
        if branch == "gain":
            keep = np.bincount(inputs["unit_index"], minlength=inputs["num_units"]) >= self.min_frames_per_unit
            if not np.any(keep):
                raise ValueError(f"No gain unit has at least {self.min_frames_per_unit} frames")
            x, y = inputs["source_gain"][:, keep], inputs["target_gain"][:, keep]
        else:
            x, y = inputs["source_residual"], inputs["target_residual"]
        return ReducedRankRegression(alpha=alpha, fit_intercept=self.fit_intercept).fit(
            torch.tensor(x.T, dtype=torch.float64),
            torch.tensor(y.T, dtype=torch.float64),
        )

    def train(self, session, spks_type=None, split="train", hyperparameters=None):
        hp = hyperparameters or self.hyperparameters
        target_pf, source_pf = PlaceFieldModel.train(self, session, spks_type, split, hp)
        inputs = self.decomposition_inputs(
            session,
            target_pf,
            source_pf,
            spks_type,
            split,
            include_target=True,
        )
        return (
            target_pf,
            source_pf,
            self._fit_branch(inputs, "gain", hp.alpha),
            self._fit_branch(inputs, "residual", hp.alpha_residual),
        )

    def apply_models(self, inputs, gain_model, residual_model, hyperparameters):
        """Predict the two components without accessing measured target components."""
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
        )
        return gain_activity + residual, extras

    def predict(
        self,
        session,
        coefficients,
        spks_type=None,
        split="test",
        hyperparameters=None,
        nan_safe=False,
    ):
        target_pf, source_pf, gain_model, residual_model = coefficients
        inputs = self.decomposition_inputs(
            session,
            target_pf,
            source_pf,
            spks_type,
            split,
            nan_safe=nan_safe,
        )
        return self.apply_models(inputs, gain_model, residual_model, hyperparameters or self.hyperparameters)

    def _optimize_golden(self, session, spks_type, train_split, validation_split, nan_safe=False):
        """Tune each branch against its measured activity component on validation trials.

        Both branches inherit the same plain-PF hyperparameters. Gain is scored in
        reconstructed activity units, so low-energy trials do not dominate the loss
        through large, poorly determined gain ratios. The combined score is recorded
        for the returned pair; independent tuning need not minimize combined MSE.
        """
        pf_hp = self.inherited_placefield_hyperparameters(session, spks_type, train_split, validation_split)
        hp = PlaceFieldGainResidualHyperparameters(num_bins=pf_hp.num_bins, smooth_width=pf_hp.smooth_width)
        target_pf, source_pf = PlaceFieldModel.train(self, session, spks_type, train_split, hp)
        train = self.decomposition_inputs(
            session,
            target_pf,
            source_pf,
            spks_type,
            train_split,
            nan_safe=nan_safe,
            include_target=True,
        )
        validation = self.decomposition_inputs(
            session,
            target_pf,
            source_pf,
            spks_type,
            validation_split,
            nan_safe=nan_safe,
            include_target=True,
        )
        results = []
        fitted = {}
        for branch in ("gain", "residual"):
            max_model = self._fit_branch(train, branch, 1e2)
            max_rank = int(max_model.max_rank)
            records = []

            def evaluate(alpha, rank, model=None):
                model = model if model is not None else self._fit_branch(train, branch, alpha)
                if branch == "gain":
                    prediction = self.apply_gain_model(validation, model, rank, nan_safe=nan_safe)[0]
                    target = validation["target_gain_activity"]
                else:
                    prediction = (
                        model.predict(
                            torch.tensor(validation["source_residual"].T, dtype=torch.float64),
                            rank=rank,
                            nonnegative=False,
                        )
                        .numpy()
                        .T
                    )
                    target = validation["target_residual"]
                score = float(np.mean((target - prediction) ** 2))
                if not np.isfinite(score):
                    score = float("inf")
                records.append(dict(branch=branch, alpha=float(alpha), rank=int(rank), score=score))
                return score

            best_alpha = golden_section_search(
                func=lambda alpha: evaluate(alpha, max_rank),
                a=1e-2,
                b=1e6,
                tolerance_param=1e-2,
                tolerance_score=1e-3,
                max_iterations=25,
                minimize=True,
                logspace=True,
            )[0]
            model = self._fit_branch(train, branch, best_alpha)
            if max_rank == 1:
                evaluate(best_alpha, 1, model)
            else:
                golden_section_search(
                    func=lambda rank: evaluate(best_alpha, int(rank), model),
                    a=1.0,
                    b=float(max_rank),
                    tolerance_param=1.0,
                    tolerance_score=1e-3,
                    max_iterations=25,
                    minimize=True,
                    logspace=False,
                )
            best = min(records, key=lambda record: record["score"])
            if not np.isfinite(best["score"]):
                raise ValueError(f"No finite validation score for {branch} regression")
            fitted[branch] = self._fit_branch(train, branch, best["alpha"])
            if branch == "gain":
                hp = replace(hp, alpha=best["alpha"], rank=best["rank"])
            else:
                hp = replace(hp, alpha_residual=best["alpha"], rank_residual=best["rank"])
            results.extend(records)
        prediction, _ = self.apply_models(validation, fitted["gain"], fitted["residual"], hp)
        target = validation["target_gain_activity"] + validation["target_residual"]
        score = float(np.mean((target - prediction) ** 2))
        params = dict(
            num_bins=hp.num_bins,
            smooth_width=hp.smooth_width,
            alpha=hp.alpha,
            rank=hp.rank,
            alpha_residual=hp.alpha_residual,
            rank_residual=hp.rank_residual,
        )
        return params, score, pd.DataFrame(results)


def gain_residual_scores(report):
    """Score both predictions against original activity with a shared denominator.

    Per-fold increments telescope exactly: R2(PF) + gain increment + residual
    increment = R2(G+E). Standalone G and E scores are also saved. Summed SSE and
    SST permit additive pooling across folds without taking separate medians.
    """
    target = np.asarray(report.target_data, dtype=np.float64)
    gain = report.extras["gain_prediction"]
    residual = report.extras["residual_prediction"]
    template = report.extras["placefield_prediction"]
    sst = float(np.sum((target - target.mean()) ** 2))
    sse_template = float(np.sum((target - template) ** 2))
    sse_gain = float(np.sum((target - gain) ** 2))
    sse_residual = float(np.sum((target - residual) ** 2))
    sse_combined = float(np.sum((target - gain - residual) ** 2))
    units = report.extras["gain_unit_index"]
    target_gain = least_squares_gain(target, template, units, int(units.max()) + 1)
    measured_gain = template * target_gain[:, units]
    measured_residual = target - measured_gain
    return dict(
        mse_gain=float(np.mean((target - gain) ** 2)),
        mse_residual=float(np.mean((target - residual) ** 2)),
        r2_gain=float(measure_r2(gain, target, dim=None)),
        r2_residual=float(measure_r2(residual, target, dim=None)),
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
