"""RRRToExternalLatentsConfig — latent-to-latent predictability between RRR and regression-model latents.

Uses Ridge Regression (Optuna-optimized alpha) to measure how much variance in a regression
model's latents is explained by RRR latents and vice versa. Two *different* models supply the
two latent sets, because each is the correct model for its own question:

- **external** latents come from ``EXTERNAL_MODEL_NAME`` (``predict_latents=False``). Its basis
  functions are built directly from behavior, so they are ground-truth external variables that
  never touch neural activity.
- **internal** latents come from ``INTERNAL_MODEL_NAME`` (``predict_latents=True``,
  ``split_train=True``). Its encoder-predicted basis is the population's *estimate* of those
  external variables.

The internal model is deliberately the double-cross-validated one, **not** the ``_leak`` variant.
A leak model trains its encoder and decoder on the same split, which lets the basis act as a
generic rank/capacity channel between source and target activity rather than as a representation
of the external variables it is named after.

Because the two models optimize their hyperparameters independently, their bases generally have
different dimensionality, so the (position, speed, reward) column counts are reported separately
for each — see ``num_*_params`` (external) and ``num_*_params_pred`` (internal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Union

import numpy as np
import optuna
import torch
from dimilibi import RidgeRegression
from vrAnalysis.sessions import B2Session, SpksTypes

from ..registry import ACTIVITY_PARAMETERS_NAMES, PopulationRegistry, get_model
from ..regression_models.hyperparameters import FullRegressorHyperparameters
from ..pipeline.base import AnalysisConfigBase

VALID_SPKS_TYPES: list[SpksTypes] = ["oasis", "sigrebase"]
VALID_ACTIVITY_PARAMETERS: list[str] = ["default", "preserved"]
VALID_RRR_VARIANCE: list[Union[float, str]] = [1.0, 0.95, "match"]

# Both models are hard-coded: these comparisons only make sense against one fixed regressor
# structure, since the pos/speed/reward dimensionality breakdown below is derived from
# FullRegressorModel with speed_basis=False, no_reward=False.
#
# EXTERNAL supplies the true (behavior-derived) basis; it has predict_latents=False and so
# exposes only "basis_functions". INTERNAL supplies the encoder-predicted basis; it needs
# predict_latents=True to expose "basis_functions_predicted".
EXTERNAL_MODEL_NAME: str = "fullregressor_decoder_only_1dspeed"
INTERNAL_MODEL_NAME: str = "fullregressor_1dspeed"


def _zscore(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    std = x.std(dim=dim, keepdims=True).clamp(min=eps)
    return (x - x.mean(dim=dim, keepdims=True)) / std


def _valid_indices(extras: dict, num_samples: int) -> np.ndarray:
    """Indices, into the split's original frames, that a model's outputs correspond to.

    ``RegressionModel.predict`` drops samples whose prediction contains NaN and records the
    surviving indices in ``idx_valid_predictions``. When nothing was dropped the arrays still
    span every frame of the split.

    Parameters
    ----------
    extras : dict
        The extras dict returned by ``RegressionModel.predict``.
    num_samples : int
        Number of samples in the model's returned arrays (used only when nothing was filtered).

    Returns
    -------
    np.ndarray
        Ascending frame indices within the split.
    """
    if extras.get("predictions_were_filtered", False):
        return np.asarray(extras["idx_valid_predictions"])
    return np.arange(num_samples)


def _align_split(ext_extras: dict, int_extras: dict, rrr_extras: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict one split's external, internal, and RRR outputs to their common frames.

    The three models filter NaN samples independently, so their arrays are not necessarily
    row-aligned even though they describe the same split. Regressing them against each other
    without intersecting first would silently pair up mismatched timepoints.

    Parameters
    ----------
    ext_extras, int_extras, rrr_extras : dict
        Extras dicts from the external, internal, and RRR models for the same split.

    Returns
    -------
    tuple of np.ndarray
        ``(external_true, internal_pred, rrr_latents)`` restricted to shared frames. The first
        two are (frames, dims); ``rrr_latents`` keeps its native (rank, frames) layout.
    """
    external_true = np.asarray(ext_extras["basis_functions"])
    internal_pred = np.asarray(int_extras["basis_functions_predicted"])
    rrr_latents = np.asarray(rrr_extras["latents"])

    idx_ext = _valid_indices(ext_extras, external_true.shape[0])
    idx_int = _valid_indices(int_extras, internal_pred.shape[0])
    idx_rrr = _valid_indices(rrr_extras, rrr_latents.shape[1])

    common = np.intersect1d(np.intersect1d(idx_ext, idx_int), idx_rrr)
    return (
        external_true[np.searchsorted(idx_ext, common)],
        internal_pred[np.searchsorted(idx_int, common)],
        rrr_latents[:, np.searchsorted(idx_rrr, common)],
    )


def _gather_latents(
    session: B2Session,
    registry: PopulationRegistry,
    spks_type: SpksTypes,
    activity_parameters_name: str,
    method: str,
    rrr_variance: Optional[Union[float, str]],
    normalize: bool,
) -> dict[str, torch.Tensor]:
    normalize_func = _zscore if normalize else lambda x, dim=None: x

    ext_model = get_model(EXTERNAL_MODEL_NAME, registry, activity_parameters=activity_parameters_name)
    int_model = get_model(INTERNAL_MODEL_NAME, registry, activity_parameters=activity_parameters_name)
    rrr_model = get_model("rrr", registry, activity_parameters=activity_parameters_name)

    # Boundary check: the internal model must expose an encoder-predicted basis, and the external
    # model's basis must be behavior-derived rather than activity-derived.
    if not int_model.predict_latents:
        raise ValueError(f"INTERNAL_MODEL_NAME={INTERNAL_MODEL_NAME!r} must have predict_latents=True to expose a predicted basis.")
    if ext_model.predict_latents:
        raise ValueError(f"EXTERNAL_MODEL_NAME={EXTERNAL_MODEL_NAME!r} must have predict_latents=False so its basis is behavior-derived.")

    hyperparameters_ext = ext_model.get_best_hyperparameters(session, spks_type=spks_type, method=method)[0]
    hyperparameters_int = int_model.get_best_hyperparameters(session, spks_type=spks_type, method=method)[0]
    hyperparameters_rrr = rrr_model.get_best_hyperparameters(session, spks_type=spks_type, method=method)[0]

    report_ext = ext_model.process(session, spks_type=spks_type, hyperparameters=hyperparameters_ext)
    report_int = int_model.process(session, spks_type=spks_type, hyperparameters=hyperparameters_int)
    report_rrr = rrr_model.process(session, spks_type=spks_type, hyperparameters=hyperparameters_rrr)

    aligned = {"test": _align_split(report_ext.extras, report_int.extras, report_rrr.extras)}
    for split in ("train", "validation"):
        aligned[split] = _align_split(
            ext_model.predict(session, report_ext.trained_model, split=split, hyperparameters=hyperparameters_ext)[1],
            int_model.predict(session, report_int.trained_model, split=split, hyperparameters=hyperparameters_int)[1],
            rrr_model.predict(session, report_rrr.trained_model, split=split, hyperparameters=hyperparameters_rrr)[1],
        )

    train_external_true, train_external_pred, train_latents = aligned["train"]
    val_external_true, val_external_pred, val_latents = aligned["validation"]
    test_external_true, test_external_pred, test_latents = aligned["test"]

    if rrr_variance is not None:
        if rrr_variance == "match":
            idx_last_rrr_latent = min(train_external_true.shape[1], train_latents.shape[0])
        elif isinstance(rrr_variance, float):
            rrr_latents_var = torch.tensor(train_latents).var(dim=1)
            rrr_latents_var_cumsum = rrr_latents_var.cumsum(dim=0)
            rrr_latents_var_cumsum = rrr_latents_var_cumsum / rrr_latents_var_cumsum[-1]
            idx_last_rrr_latent = torch.where(rrr_latents_var_cumsum <= rrr_variance)[0][-1]
        else:
            raise ValueError(f"Invalid value for rrr_variance: {rrr_variance}")
    else:
        idx_last_rrr_latent = train_latents.shape[0]

    return {
        "train_external_true": normalize_func(torch.tensor(train_external_true), dim=0),
        "train_external_pred": normalize_func(torch.tensor(train_external_pred), dim=0),
        "train_rrr": normalize_func(torch.tensor(train_latents[:idx_last_rrr_latent]), dim=1).T,
        "val_external_true": normalize_func(torch.tensor(val_external_true), dim=0),
        "val_external_pred": normalize_func(torch.tensor(val_external_pred), dim=0),
        "val_rrr": normalize_func(torch.tensor(val_latents[:idx_last_rrr_latent]), dim=1).T,
        "test_external_true": normalize_func(torch.tensor(test_external_true), dim=0),
        "test_external_pred": normalize_func(torch.tensor(test_external_pred), dim=0),
        "test_rrr": normalize_func(torch.tensor(test_latents[:idx_last_rrr_latent]), dim=1).T,
        "hyperparameters_ext": hyperparameters_ext,
        "hyperparameters_int": hyperparameters_int,
        "num_environments": len(session.environments),
    }


def _optimize_alpha(
    Xtrain: torch.Tensor,
    Xval: torch.Tensor,
    Ytrain: torch.Tensor,
    Yval: torch.Tensor,
) -> tuple[float, float]:
    def _objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-3, 1e4, log=True)
        model = RidgeRegression(alpha=alpha, fit_intercept=True).fit(Xtrain, Ytrain)
        return model.score(Xval, Yval, dim=None)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=100)
    return study.best_params["alpha"], study.best_value


def _pos_speed_reward_dims(hyperparameters: FullRegressorHyperparameters, num_environments: int) -> tuple[int, int, int]:
    """Split a model's basis dimensionality into (position, speed, reward) components.

    Applies to both ``EXTERNAL_MODEL_NAME`` and ``INTERNAL_MODEL_NAME``, which share a regressor
    structure: ``speed_basis=False`` (single z-scored speed regressor) and ``no_reward=False``
    with all three reward components (expectation, delivered response, omission response)
    included and ``expectation_symmetric=True`` (see ``FullRegressorModel.build_regressors``).
    The two models optimize hyperparameters independently, so call this once per model.
    """
    num_pos = hyperparameters.num_basis * num_environments
    num_speed = 1
    num_reward = 4 * hyperparameters.reward_num_basis_lags + 3
    return num_pos, num_speed, num_reward


@dataclass(frozen=True)
class RRRToExternalLatentsConfig(AnalysisConfigBase):
    """Measure latent-to-latent predictability between RRR and regression-model latents.

    The ``*_true`` latents come from ``EXTERNAL_MODEL_NAME`` (behavior-derived basis) and the
    ``*_pred`` latents from ``INTERNAL_MODEL_NAME`` (encoder-predicted basis). See the module
    docstring for why these are two different models.

    Parameters
    ----------
    spks_type : SpksTypes
        Spike type to use for the population.
    activity_parameters_name : str
        Activity scaling method.
    method : str
        Hyperparameter optimization method.
    rrr_variance : float or str
        Fraction of RRR variance to retain (float), or ``"match"`` to use
        the same number of latents as the external model's true basis dimension.
    normalize : bool
        Whether to z-score latents before regression.
    """

    schema_version: str = "v6"
    # v5: hard-code external_model_name to EXTERNAL_MODEL_NAME and add
    # num_pos_params/num_speed_params/num_reward_params to the output.
    # v6: take the true basis from EXTERNAL_MODEL_NAME (decoder-only) and the predicted basis
    # from INTERNAL_MODEL_NAME (double-cross-validated) instead of both from one leak model;
    # report per-model column counts (num_*_params for external, num_*_params_pred for
    # internal) since the two optimize hyperparameters independently; align the three models'
    # frames before regressing.

    data_config_name: str = "default"
    spks_type: SpksTypes = "sigrebase"
    activity_parameters_name: str = "default"
    method: str = "preferred"
    rrr_variance: Union[float, str] = 0.95
    normalize: bool = False

    display_name: ClassVar[str] = "rrr_to_external_latents"

    @staticmethod
    def _param_grid() -> dict:
        return {
            # "spks_type": list(VALID_SPKS_TYPES), # no longer analyzing anything except sigrebase
            "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
            "rrr_variance": list(VALID_RRR_VARIANCE),
            "normalize": [True, False],
        }

    def validate(self):
        if self.spks_type not in VALID_SPKS_TYPES:
            raise ValueError(f"Unknown spks_type {self.spks_type!r}. Available: {VALID_SPKS_TYPES}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. " f"Available: {', '.join(ACTIVITY_PARAMETERS_NAMES)}"
            )
        if not (isinstance(self.rrr_variance, float) or self.rrr_variance == "match"):
            raise ValueError(f"rrr_variance must be a float or 'match', got {self.rrr_variance!r}")

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"method={self.method}",
            f"ext={EXTERNAL_MODEL_NAME}",
            f"int={INTERNAL_MODEL_NAME}",
            f"rrr_var={self.rrr_variance}",
            f"norm={self.normalize}",
        ]
        if self.activity_parameters_name != "default":
            parts.append(f"ap={self.activity_parameters_name}")
        parts.append(self.schema_version)
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        data = _gather_latents(
            session,
            registry,
            spks_type=self.spks_type,
            activity_parameters_name=self.activity_parameters_name,
            method=self.method,
            rrr_variance=self.rrr_variance,
            normalize=self.normalize,
        )

        # Reported separately per model: the external and internal models optimize their
        # hyperparameters independently, so their basis column counts generally differ.
        num_pos_params_true, num_speed_params_true, num_reward_params_true = _pos_speed_reward_dims(
            data["hyperparameters_ext"],
            data["num_environments"],
        )
        num_pos_params_pred, num_speed_params_pred, num_reward_params_pred = _pos_speed_reward_dims(
            data["hyperparameters_int"],
            data["num_environments"],
        )

        alpha_rrr_to_true, score_rrr_to_true = _optimize_alpha(
            data["train_rrr"],
            data["val_rrr"],
            data["train_external_true"],
            data["val_external_true"],
        )
        alpha_true_to_rrr, score_true_to_rrr = _optimize_alpha(
            data["train_external_true"],
            data["val_external_true"],
            data["train_rrr"],
            data["val_rrr"],
        )
        alpha_rrr_to_pred, score_rrr_to_pred = _optimize_alpha(
            data["train_rrr"],
            data["val_rrr"],
            data["train_external_pred"],
            data["val_external_pred"],
        )
        alpha_pred_to_rrr, score_pred_to_rrr = _optimize_alpha(
            data["train_external_pred"],
            data["val_external_pred"],
            data["train_rrr"],
            data["val_rrr"],
        )

        rrr_to_true = RidgeRegression(alpha=alpha_rrr_to_true, fit_intercept=True).fit(data["train_rrr"], data["train_external_true"])
        true_to_rrr = RidgeRegression(alpha=alpha_true_to_rrr, fit_intercept=True).fit(data["train_external_true"], data["train_rrr"])
        rrr_to_pred = RidgeRegression(alpha=alpha_rrr_to_pred, fit_intercept=True).fit(data["train_rrr"], data["train_external_pred"])
        pred_to_rrr = RidgeRegression(alpha=alpha_pred_to_rrr, fit_intercept=True).fit(data["train_external_pred"], data["train_rrr"])

        return {
            "alpha_rrr_to_true": alpha_rrr_to_true,
            "score_rrr_to_true": score_rrr_to_true,
            "alpha_true_to_rrr": alpha_true_to_rrr,
            "score_true_to_rrr": score_true_to_rrr,
            "alpha_rrr_to_pred": alpha_rrr_to_pred,
            "score_rrr_to_pred": score_rrr_to_pred,
            "alpha_pred_to_rrr": alpha_pred_to_rrr,
            "score_pred_to_rrr": score_pred_to_rrr,
            "test_score_rrr_to_true": rrr_to_true.score(data["test_rrr"], data["test_external_true"], dim=None),
            "test_score_true_to_rrr": true_to_rrr.score(data["test_external_true"], data["test_rrr"], dim=None),
            "test_score_rrr_to_pred": rrr_to_pred.score(data["test_rrr"], data["test_external_pred"], dim=None),
            "test_score_pred_to_rrr": pred_to_rrr.score(data["test_external_pred"], data["test_rrr"], dim=None),
            "test_score_each_rrr_to_true": rrr_to_true.score(data["test_rrr"], data["test_external_true"], dim=0, reduce="none"),
            "test_score_each_true_to_rrr": true_to_rrr.score(data["test_external_true"], data["test_rrr"], dim=0, reduce="none"),
            "test_score_each_rrr_to_pred": rrr_to_pred.score(data["test_rrr"], data["test_external_pred"], dim=0, reduce="none"),
            "test_score_each_pred_to_rrr": pred_to_rrr.score(data["test_external_pred"], data["test_rrr"], dim=0, reduce="none"),
            "num_pos_params_true": num_pos_params_true,
            "num_speed_params_true": num_speed_params_true,
            "num_reward_params_true": num_reward_params_true,
            "num_pos_params_pred": num_pos_params_pred,
            "num_speed_params_pred": num_speed_params_pred,
            "num_reward_params_pred": num_reward_params_pred,
        }
