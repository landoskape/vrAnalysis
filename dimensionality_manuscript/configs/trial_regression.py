"""Regression evaluation on an even train/test split of whole trials."""

from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
from dimilibi import measure_r2, mse
from vrAnalysis.helpers import reliability_loo, stable_hash
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.placefields import get_placefield
from vrAnalysis.sessions import B2Session, SpksTypes
from ..env_order import MAX_ENV_SLOTS, load_env_order
from ..pipeline.base import AnalysisConfigBase
from ..registry import ACTIVITY_PARAMETERS_NAMES, MODEL_NAMES, ModelName, PopulationRegistry, get_model
from ..regression_models.models import make_gain_unit_index
from ..trial_registry import (
    EnvironmentTrialRegistry,
    TrialRegistry,
    TrialRegistryParameters,
    split_trials_by_environment,
)
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

ALL_CELL_TRIAL_MODELS: tuple[ModelName, ...] = (
    "external_placefield_1d",
    "external_placefield_1d_gain",
    "internal_placefield_1d",
    "internal_placefield_1d_gain",
    "rrr",
)

PLACECELL_TRIAL_MODELS: tuple[ModelName, ...] = (
    *ALL_CELL_TRIAL_MODELS[:-1],
    "internal_placefield_1d_structured_gain",
    "rrr",
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
    """Optimize all-cell models on 50/25/25 whole-trial folds and score test once.

    The v2 analysis preserves the ordinary registry's ROI population and source/target split but
    replaces its sample folds with environment-stratified trial folds.  Hyperparameters are chosen
    from train/validation and the untouched test fold is used only for the returned scores.

    Explicit ``schema_version="v1"`` retains the prior 50/50 behavior and ordinary-even inherited
    hyperparameters, so existing results and cache identities remain reproducible.
    """

    schema_version: str = "v2"
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
        return {"model_name": list(ALL_CELL_TRIAL_MODELS)}

    def validate(self):
        if self.data_config_name != "even":
            raise ValueError("TrialSplitRegressionConfig requires data_config_name='even'")
        allowed = KEY_FIGURE_MODELS if self.schema_version == "v1" else ALL_CELL_TRIAL_MODELS
        if self.model_name not in MODEL_NAMES or self.model_name not in allowed:
            raise ValueError(f"Unsupported model_name {self.model_name!r}. Available: {', '.join(allowed)}")
        if self.spks_type not in VALID_SPKS_TYPES:
            raise ValueError(f"Unknown spks_type {self.spks_type!r}. Available: {VALID_SPKS_TYPES}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. " f"Available: {', '.join(ACTIVITY_PARAMETERS_NAMES)}"
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
        """Optimize, refit, and score on whole-trial folds (or reproduce legacy v1)."""
        if registry.registry_params != self.data_config.to_registry_params():
            raise ValueError("The supplied base registry does not match the named 'even' data config")

        trial_registry = TrialRegistry(
            registry,
            TrialRegistryParameters(
                name="even",
                relative_size=(1, 1) if self.schema_version == "v1" else (2, 1, 1),
                split_seed=self.trial_split_seed,
                speed_threshold=registry.registry_params.speed_threshold,
            ),
        )
        model = get_model(self.model_name, trial_registry, activity_parameters=self.activity_parameters_name)
        if self.schema_version == "v1":
            base_model = get_model(self.model_name, registry, activity_parameters=self.activity_parameters_name)
            hyperparameters = base_model.get_best_hyperparameters(session, spks_type=self.spks_type, method=self.method)[0]
        else:
            hyperparameters = model.get_best_hyperparameters(
                session,
                spks_type=self.spks_type,
                train_split="train",
                validation_split="validation",
                method=self.method,
            )[0]
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
            raise ValueError("The quality filter is not aligned with the held-out target rows: " f"mask={quality_mask.shape}, target={target.shape}")

        train_behavior = model.get_session_data(session, self.spks_type, "train")[2]
        test_behavior = model.get_session_data(session, self.spks_type, "test")[2]
        n_trials_train = int(np.unique(train_behavior.trial).size)
        n_trials_test = int(np.unique(test_behavior.trial).size)
        n_trials_validation = np.nan
        if self.schema_version != "v1":
            validation_behavior = model.get_session_data(session, self.spks_type, "validation")[2]
            n_trials_validation = int(np.unique(validation_behavior.trial).size)

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
            "n_trials_validation": n_trials_validation,
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


def _environment_placecell_rois(
    session: B2Session,
    registry: PopulationRegistry,
    spks_type: SpksTypes,
    activity_parameters_name: str,
    environment: int,
    thresholds: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Select PCs in one environment and retain the base source/target identities."""
    model = get_model("rrr", registry, activity_parameters=activity_parameters_name)
    source, target, behavior = model.get_session_data(session, spks_type, "full")
    population, _ = registry.get_population(session, spks_type)
    base_neurons = np.asarray(population.idx_neurons.detach().cpu(), dtype=np.int64)
    source_rois = base_neurons[np.asarray(population.cell_split_indices[0].detach().cpu(), dtype=np.int64)]
    target_rois = base_neurons[np.asarray(population.cell_split_indices[1].detach().cpu(), dtype=np.int64)]
    activity = np.concatenate([np.asarray(source), np.asarray(target)], axis=0)
    roi_indices = np.concatenate([source_rois, target_rois])

    placefields = get_placefield(
        activity.T,
        behavior,
        dist_edges=np.linspace(0, float(np.asarray(session.env_length).flat[0]), 101),
        speed_threshold=None,
        average=False,
        use_fast_sampling=True,
        session=session,
    ).filter_by_environment(environment)
    trial_maps = np.transpose(placefields.placefield, (2, 0, 1))
    if trial_maps.shape[1] < 2:
        reliability = np.full(activity.shape[0], np.nan)
    else:
        reliability = reliability_loo(trial_maps)
    if trial_maps.shape[1] == 0:
        fraction_active = np.full(activity.shape[0], np.nan)
    else:
        fraction_active = FractionActive.compute(
            trial_maps,
            activity_axis=2,
            fraction_axis=1,
            activity_method="rms",
            fraction_method="participation",
        )
    keep = _placecell_mask(reliability, fraction_active, thresholds)
    return {
        "roi_indices": roi_indices[keep],
        "source_roi_indices": source_rois[keep[: source_rois.size]],
        "target_roi_indices": target_rois[keep[source_rois.size :]],
        "reliability": reliability,
        "fraction_active": fraction_active,
        "roi_lookup": roi_indices,
        "placecell_mask": keep,
    }


def _placecell_mask(
    reliability: np.ndarray,
    fraction_active: np.ndarray,
    thresholds: tuple[float, float],
) -> np.ndarray:
    """Require reliability and fraction-active to pass for the same environment."""
    reliability = np.asarray(reliability, dtype=float)
    fraction_active = np.asarray(fraction_active, dtype=float)
    if reliability.shape != fraction_active.shape:
        raise ValueError("reliability and fraction_active must have the same shape")
    reliability_threshold, fraction_active_threshold = thresholds
    return (
        np.isfinite(reliability)
        & np.isfinite(fraction_active)
        & (reliability >= reliability_threshold)
        & (fraction_active >= fraction_active_threshold)
    )


@dataclass(frozen=True)
class TrialPlacecellRegressionConfig(AnalysisConfigBase):
    """Optimize and score temporal models separately in each environment's place cells."""

    schema_version: str = "v1"
    data_config_name: str = "even"
    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "sigrebase"
    method: str = "preferred"
    activity_parameters_name: str = "std"
    reliability_fraction_active_threshold: tuple[float, float] = (0.3, 0.1)
    trial_split_seed: int = 0

    display_name: ClassVar[str] = "trial_placecell_regression"
    _result_handling: ClassVar[dict[str, str]] = {
        "selected_roi_indices": "skip",
        "source_roi_indices": "skip",
        "target_roi_indices": "skip",
        "selected_hyperparameters": "skip",
    }

    @staticmethod
    def _param_grid() -> dict:
        return {"model_name": list(PLACECELL_TRIAL_MODELS)}

    def validate(self):
        if self.data_config_name != "even":
            raise ValueError("TrialPlacecellRegressionConfig requires data_config_name='even'")
        if self.model_name not in PLACECELL_TRIAL_MODELS:
            raise ValueError(f"Unsupported model_name {self.model_name!r}")
        if self.spks_type not in VALID_SPKS_TYPES:
            raise ValueError(f"Unknown spks_type {self.spks_type!r}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(f"Unknown activity_parameters_name {self.activity_parameters_name!r}")
        if len(self.reliability_fraction_active_threshold) != 2:
            raise ValueError("reliability_fraction_active_threshold must contain two values")
        reliability_threshold, fraction_active_threshold = self.reliability_fraction_active_threshold
        if not -1 <= reliability_threshold <= 1:
            raise ValueError("the reliability threshold must be between -1 and 1")
        if not 0 <= fraction_active_threshold <= 1:
            raise ValueError("the fraction-active threshold must be between 0 and 1")

    def summary(self) -> str:
        rel, frac = self.reliability_fraction_active_threshold
        return "_".join(
            [
                self.display_name,
                self.model_name,
                f"spks={self.spks_type}",
                f"method={self.method}",
                f"ap={self.activity_parameters_name}",
                f"rel={rel:g}",
                f"fa={frac:g}",
                f"seed={self.trial_split_seed}",
                self.schema_version,
            ]
        )

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        if registry.registry_params != self.data_config.to_registry_params():
            raise ValueError("The supplied base registry does not match the named 'even' data config")
        mouse_order = load_env_order().get(session.mouse_name)
        env_slot_ids = np.full(MAX_ENV_SLOTS, np.nan)
        if mouse_order is not None:
            env_slot_ids[: len(mouse_order)] = mouse_order
        scalar_names = (
            "mse",
            "r2",
            "n_trials_train",
            "n_trials_validation",
            "n_trials_test",
            "n_source_rois",
            "n_target_rois",
            "structured_gain_requested_rank",
            "structured_gain_effective_rank",
            "structured_gain_max_trial_rank",
        )
        output = {name: np.full(MAX_ENV_SLOTS, np.nan) for name in scalar_names}
        selected_rois = [None] * MAX_ENV_SLOTS
        source_rois = [None] * MAX_ENV_SLOTS
        target_rois = [None] * MAX_ENV_SLOTS
        selected_hyperparameters = [None] * MAX_ENV_SLOTS

        for environment in sorted(int(env) for env in np.asarray(session.environments) if env >= 0):
            if mouse_order is None or environment not in mouse_order:
                continue
            slot = mouse_order.index(environment)
            if slot >= MAX_ENV_SLOTS:
                continue
            environment_trials = np.flatnonzero(np.asarray(session.trial_environment) == environment)
            try:
                local_folds = split_trials_by_environment(
                    np.full(environment_trials.size, environment),
                    (2, 1, 1),
                    int(stable_hash(session.session_uid, environment, self.trial_split_seed, "trial-split"), 16),
                )
            except ValueError:
                continue
            for fold, key in zip(local_folds, ("n_trials_train", "n_trials_validation", "n_trials_test")):
                output[key][slot] = fold.size
            selection = _environment_placecell_rois(
                session,
                registry,
                self.spks_type,
                self.activity_parameters_name,
                environment,
                self.reliability_fraction_active_threshold,
            )
            selected_rois[slot] = selection["roi_indices"]
            source_rois[slot] = selection["source_roi_indices"]
            target_rois[slot] = selection["target_roi_indices"]
            output["n_source_rois"][slot] = selection["source_roi_indices"].size
            output["n_target_rois"][slot] = selection["target_roi_indices"].size
            if min(selection["source_roi_indices"].size, selection["target_roi_indices"].size) < 2:
                continue
            try:
                env_registry = EnvironmentTrialRegistry(
                    registry,
                    environment,
                    selection["roi_indices"],
                    TrialRegistryParameters(
                        relative_size=(2, 1, 1),
                        split_seed=self.trial_split_seed,
                        speed_threshold=registry.registry_params.speed_threshold,
                    ),
                )
                model = get_model(self.model_name, env_registry, activity_parameters=self.activity_parameters_name)
                hyperparameters = model.get_best_hyperparameters(
                    session,
                    spks_type=self.spks_type,
                    train_split="train",
                    validation_split="validation",
                    method=self.method,
                )[0]
                report = model.process(
                    session,
                    self.spks_type,
                    train_split="train",
                    test_split="test",
                    hyperparameters=hyperparameters,
                )
            except ValueError as error:
                # Sparse environments are expected omissions; programming/alignment errors should
                # remain loud rather than turning into plausible-looking missing data.
                message = str(error).lower()
                expected_sparse_error = any(
                    phrase in message
                    for phrase in (
                        "too few to appear",
                        "empty trial fold",
                        "no gain unit has at least",
                        "at least two source and two target",
                    )
                )
                if not expected_sparse_error:
                    raise
                continue

            prediction = np.asarray(report.predicted_data)
            target = np.asarray(report.target_data)
            output["mse"][slot] = float(mse(prediction, target, reduce="mean", dim=None))
            output["r2"][slot] = float(measure_r2(prediction, target, reduce="mean", dim=None))
            selected_hyperparameters[slot] = vars(hyperparameters).copy()
            if self.model_name in STRUCTURED_GAIN_MODELS:
                _assert_one_gain_unit_per_trial(model, session, self.spks_type, "train")
                _assert_one_gain_unit_per_trial(model, session, self.spks_type, "test")
                gain_model = report.trained_model[2]
                output["structured_gain_requested_rank"][slot] = float(hyperparameters.rank)
                output["structured_gain_max_trial_rank"][slot] = float(gain_model.max_rank)
                output["structured_gain_effective_rank"][slot] = float(min(hyperparameters.rank, gain_model.max_rank))

        return {
            **output,
            "env_slot_ids": env_slot_ids,
            "selected_roi_indices": selected_rois,
            "source_roi_indices": source_rois,
            "target_roi_indices": target_rois,
            "selected_hyperparameters": selected_hyperparameters,
        }
