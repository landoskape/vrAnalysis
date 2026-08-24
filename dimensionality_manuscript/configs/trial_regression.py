"""Regression evaluation on an even train/test split of whole trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from dimilibi import measure_r2, mse
from vrAnalysis.sessions import B2Session, SpksTypes

from ..pipeline.base import AnalysisConfigBase
from ..registry import ACTIVITY_PARAMETERS_NAMES, MODEL_NAMES, ModelName, PopulationRegistry, get_model
from ..regression_models.models import make_gain_unit_index
from ..trial_registry import TrialRegistry, TrialRegistryParameters
from .regression import (
    KEY_FIGURE_MODELS,
    VALID_SPKS_TYPES,
    _per_environment_scores,
    _pooled_subset_r2,
    _regression_quality_filter,
)


STRUCTURED_GAIN_MODELS: tuple[ModelName, ...] = (
    "external_placefield_1d_structured_gain",
    "internal_placefield_1d_structured_gain",
)


def _assert_one_gain_unit_per_trial(model, session: B2Session, spks_type: SpksTypes, split: str) -> int:
    """Verify that structured gain sees exactly one unit for every trial in a split."""
    _, _, frame_behavior = model.get_session_data(session, spks_type, split)
    unit_index, num_units = make_gain_unit_index(
        model.get_split_chunk_index(session, spks_type, split),
        frame_behavior.trial,
    )
    trials = np.asarray(frame_behavior.trial, dtype=np.int64)
    unique_trials = np.unique(trials)
    if num_units != unique_trials.size:
        raise ValueError(f"Structured gain produced {num_units} units for {unique_trials.size} {split} trials")
    for trial in unique_trials:
        if np.unique(unit_index[trials == trial]).size != 1:
            raise ValueError(f"Trial {trial} was divided across gain units on split {split!r}")
    return int(unique_trials.size)


@dataclass(frozen=True)
class TrialSplitRegressionConfig(AnalysisConfigBase):
    """Score fixed-hyperparameter regression models on held-out whole trials.

    This analysis deliberately uses the named ``even`` data config and does not expose data config
    as a parameter-grid axis.  The ordinary ``even`` :class:`PopulationRegistry` supplies the
    exact ROI population, source/target cell split, and already-optimized model hyperparameters.
    A derived :class:`~dimensionality_manuscript.trial_registry.TrialRegistry` then replaces only
    the time split: ``cross_validate_trials`` assigns half of every environment's trials to train
    and half to test using relative sizes ``(1, 1)``.

    There is no validation fold.  Hyperparameters are never optimized or loaded through the trial
    registry; they are inherited from the matching ordinary ``even`` model and passed directly to
    ``model.process``.  This gives both train and test complete trials while preserving an entirely
    separate population-cache namespace and leaving all existing registry hashes and paths intact.

    Every model is fit and predicts with all source and target cells.  Quality-filtered and
    complementary R² values slice only the completed held-out prediction, matching
    :class:`RegressionConfig`.  For structured gain, the registry explicitly collapses the
    existing ``chunk | trial`` grouping to one gain unit per trial; this invariant is checked on
    both train and test before results are returned.

    Notes
    -----
    ``TrialTimeSplit.train0`` and ``train1`` alias the one training fold.  Accordingly the model
    grid is restricted to :data:`KEY_FIGURE_MODELS`, which do not require independent encoder and
    decoder training folds.  A latent model with double cross-validation would need a distinct
    trial split with at least three folds.
    """

    schema_version: str = "v1"
    data_config_name: str = "even"

    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "sigrebase"
    method: str = "preferred"
    activity_parameters_name: str = "std"
    reliability_fraction_active_threshold: tuple[float, float] = (0.3, 0.1)
    trial_split_seed: int = 0

    display_name: ClassVar[str] = "trial_split_regression"

    @staticmethod
    def _param_grid() -> dict:
        return {"model_name": list(KEY_FIGURE_MODELS)}

    def validate(self):
        if self.data_config_name != "even":
            raise ValueError("TrialSplitRegressionConfig requires data_config_name='even'")
        if self.model_name not in MODEL_NAMES or self.model_name not in KEY_FIGURE_MODELS:
            raise ValueError(f"Unsupported model_name {self.model_name!r}. Available: {', '.join(KEY_FIGURE_MODELS)}")
        if self.spks_type not in VALID_SPKS_TYPES:
            raise ValueError(f"Unknown spks_type {self.spks_type!r}. Available: {VALID_SPKS_TYPES}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. "
                f"Available: {', '.join(ACTIVITY_PARAMETERS_NAMES)}"
            )
        if self.method == "best":
            raise ValueError("method='best' cannot identify one inherited hyperparameter optimization run")
        if len(self.reliability_fraction_active_threshold) != 2:
            raise ValueError("reliability_fraction_active_threshold must contain (reliability, fraction_active)")
        reliability_threshold, fraction_active_threshold = self.reliability_fraction_active_threshold
        if not -1 <= reliability_threshold <= 1:
            raise ValueError("the reliability threshold must be between -1 and 1")
        if not 0 <= fraction_active_threshold <= 1:
            raise ValueError("the fraction-active threshold must be between 0 and 1")

    def summary(self) -> str:
        reliability_threshold, fraction_active_threshold = self.reliability_fraction_active_threshold
        return "_".join(
            [
                self.display_name,
                self.model_name,
                "split=even-trial",
                f"spks={self.spks_type}",
                f"method={self.method}",
                f"ap={self.activity_parameters_name}",
                f"rel={reliability_threshold:g}",
                f"fa={fraction_active_threshold:g}",
                f"seed={self.trial_split_seed}",
                self.schema_version,
            ]
        )

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Load ordinary-even hyperparameters, then refit and score on even whole-trial folds."""
        if registry.registry_params != self.data_config.to_registry_params():
            raise ValueError("The supplied base registry does not match the named 'even' data config")

        base_model = get_model(self.model_name, registry, activity_parameters=self.activity_parameters_name)
        hyperparameters = base_model.get_best_hyperparameters(
            session,
            spks_type=self.spks_type,
            method=self.method,
        )[0]

        trial_registry = TrialRegistry(
            registry,
            TrialRegistryParameters(
                name="even",
                relative_size=(1, 1),
                split_seed=self.trial_split_seed,
                speed_threshold=registry.registry_params.speed_threshold,
            ),
        )
        model = get_model(self.model_name, trial_registry, activity_parameters=self.activity_parameters_name)
        report = model.process(
            session,
            self.spks_type,
            train_split="train",
            test_split="test",
            hyperparameters=hyperparameters,
        )

        target = np.asarray(report.target_data)
        prediction = np.asarray(report.predicted_data)
        if prediction.shape != target.shape:
            raise ValueError(f"Held-out prediction and target are misaligned: {prediction.shape} versus {target.shape}")

        quality = _regression_quality_filter(
            model,
            session,
            self.spks_type,
            self.reliability_fraction_active_threshold,
        )
        quality_mask = quality["quality_filtered_roi_mask"]
        if quality_mask.shape != (target.shape[0],):
            raise ValueError(
                "The quality filter is not aligned with the held-out target rows: "
                f"mask={quality_mask.shape}, target={target.shape}"
            )

        train_behavior = model.get_session_data(session, self.spks_type, "train")[2]
        test_behavior = model.get_session_data(session, self.spks_type, "test")[2]
        n_trials_train = int(np.unique(train_behavior.trial).size)
        n_trials_test = int(np.unique(test_behavior.trial).size)

        requested_rank = np.nan
        effective_rank = np.nan
        max_trial_rank = np.nan
        if self.model_name in STRUCTURED_GAIN_MODELS:
            n_trials_train = _assert_one_gain_unit_per_trial(model, session, self.spks_type, "train")
            n_trials_test = _assert_one_gain_unit_per_trial(model, session, self.spks_type, "test")
            requested_rank = float(hyperparameters.rank)
            gain_model = report.trained_model[2]
            max_trial_rank = float(gain_model.max_rank)
            effective_rank = float(min(int(hyperparameters.rank), int(gain_model.max_rank)))

        scores = {
            "mse": float(mse(prediction, target, reduce="mean", dim=None)),
            "r2": _pooled_subset_r2(prediction, target, np.ones(target.shape[0], dtype=bool)),
            "mse_roi": np.asarray(mse(prediction, target, reduce="none", dim=1)),
            "r2_roi": np.asarray(measure_r2(prediction, target, reduce="none", dim=1)),
            "r2_quality_filtered": _pooled_subset_r2(prediction, target, quality_mask),
            "r2_notquality_filtered": _pooled_subset_r2(prediction, target, ~quality_mask),
            "num_quality_filtered_rois": int(np.sum(quality_mask)),
            "num_notquality_filtered_rois": int(np.sum(~quality_mask)),
            "n_trials_train": n_trials_train,
            "n_trials_test": n_trials_test,
            "structured_gain_requested_rank": requested_rank,
            "structured_gain_effective_rank": effective_rank,
            "structured_gain_max_trial_rank": max_trial_rank,
        }
        return {
            **scores,
            **_per_environment_scores(model, session, self.spks_type, report),
            **quality,
        }

