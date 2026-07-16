"""RRRToExternalLatentsConfig — latent-to-latent predictability between RRR and a leak regression model.

Uses Ridge Regression (Optuna-optimized alpha) to measure how much variance
in the external model's latents (true and encoder-predicted) is explained by
RRR latents and vice versa. Only leak models (predict_latents=True) are valid
external models because they expose both true and predicted basis functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Union

import optuna
import torch
from dimilibi import RidgeRegression
from vrAnalysis.sessions import B2Session, SpksTypes

from ..registry import ACTIVITY_PARAMETERS_NAMES, PopulationRegistry, get_model
from ..pipeline.base import AnalysisConfigBase

VALID_SPKS_TYPES: list[SpksTypes] = ["oasis", "sigrebase"]
VALID_ACTIVITY_PARAMETERS: list[str] = ["default", "preserved"]
VALID_RRR_VARIANCE: list[Union[float, str]] = [1.0, 0.95, "match"]
VALID_LEAK_MODELS: list[str] = [
    "rbfpos_leak",
    "rbfpos_leak_no_intercept",
    "pos_speed_leak",
    "pos_speed_leak_1dspeed",
    "pos_speed_leak_no_intercept",
    "fullregressor_leak",
    "fullregressor_leak_1dspeed",
    "fullregressor_leak_no_intercept",
]


def _zscore(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    std = x.std(dim=dim, keepdims=True).clamp(min=eps)
    return (x - x.mean(dim=dim, keepdims=True)) / std


def _get_external_keys(model_name: str) -> tuple[str, str]:
    """Return (true_key, pred_key) for the extras dict of the given model."""
    if model_name.startswith("rbfpos"):
        return "position_basis", "position_basis_predicted"
    return "basis_functions", "basis_functions_predicted"


def _gather_latents(
    session: B2Session,
    registry: PopulationRegistry,
    spks_type: SpksTypes,
    activity_parameters_name: str,
    method: str,
    external_model_name: str,
    rrr_variance: Optional[Union[float, str]],
    normalize: bool,
) -> dict[str, torch.Tensor]:
    normalize_func = _zscore if normalize else lambda x, dim=None: x
    true_key, pred_key = _get_external_keys(external_model_name)

    ext_model = get_model(external_model_name, registry, activity_parameters=activity_parameters_name)
    rrr_model = get_model("rrr", registry, activity_parameters=activity_parameters_name)

    hyperparameters_ext = ext_model.get_best_hyperparameters(session, spks_type=spks_type, method=method)[0]
    hyperparameters_rrr = rrr_model.get_best_hyperparameters(session, spks_type=spks_type, method=method)[0]

    report_ext = ext_model.process(session, spks_type=spks_type, hyperparameters=hyperparameters_ext)
    report_rrr = rrr_model.process(session, spks_type=spks_type, hyperparameters=hyperparameters_rrr)

    train_extras_ext = ext_model.predict(session, report_ext.trained_model, split="train", hyperparameters=hyperparameters_ext)[1]
    train_extras_rrr = rrr_model.predict(session, report_rrr.trained_model, split="train", hyperparameters=hyperparameters_rrr)[1]
    val_extras_ext = ext_model.predict(session, report_ext.trained_model, split="validation", hyperparameters=hyperparameters_ext)[1]
    val_extras_rrr = rrr_model.predict(session, report_rrr.trained_model, split="validation", hyperparameters=hyperparameters_rrr)[1]

    train_external_true = normalize_func(torch.tensor(train_extras_ext[true_key]), dim=0)
    train_external_pred = normalize_func(torch.tensor(train_extras_ext[pred_key]), dim=0)
    val_external_true = normalize_func(torch.tensor(val_extras_ext[true_key]), dim=0)
    val_external_pred = normalize_func(torch.tensor(val_extras_ext[pred_key]), dim=0)
    test_external_true = normalize_func(torch.tensor(report_ext.extras[true_key]), dim=0)
    test_external_pred = normalize_func(torch.tensor(report_ext.extras[pred_key]), dim=0)

    if rrr_variance is not None:
        if rrr_variance == "match":
            idx_last_rrr_latent = min(
                train_external_true.shape[1],
                train_extras_rrr["latents"].shape[0],
            )
        elif isinstance(rrr_variance, float):
            rrr_latents = torch.tensor(train_extras_rrr["latents"])
            rrr_latents_var = rrr_latents.var(dim=1)
            rrr_latents_var_cumsum = rrr_latents_var.cumsum(dim=0)
            rrr_latents_var_cumsum = rrr_latents_var_cumsum / rrr_latents_var_cumsum[-1]
            idx_last_rrr_latent = torch.where(rrr_latents_var_cumsum <= rrr_variance)[0][-1]
        else:
            raise ValueError(f"Invalid value for rrr_variance: {rrr_variance}")
    else:
        idx_last_rrr_latent = train_extras_rrr["latents"].shape[0]

    train_rrr = normalize_func(torch.tensor(train_extras_rrr["latents"][:idx_last_rrr_latent]), dim=1).T
    val_rrr = normalize_func(torch.tensor(val_extras_rrr["latents"][:idx_last_rrr_latent]), dim=1).T
    test_rrr = normalize_func(torch.tensor(report_rrr.extras["latents"][:idx_last_rrr_latent]), dim=1).T

    return {
        "train_external_true": train_external_true,
        "train_external_pred": train_external_pred,
        "train_rrr": train_rrr,
        "val_external_true": val_external_true,
        "val_external_pred": val_external_pred,
        "val_rrr": val_rrr,
        "test_external_true": test_external_true,
        "test_external_pred": test_external_pred,
        "test_rrr": test_rrr,
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


@dataclass(frozen=True)
class RRRToExternalLatentsConfig(AnalysisConfigBase):
    """Measure latent-to-latent predictability between RRR and a leak regression model.

    Parameters
    ----------
    spks_type : SpksTypes
        Spike type to use for the population.
    activity_parameters_name : str
        Activity scaling method.
    method : str
        Hyperparameter optimization method.
    external_model_name : str
        Name of the external (leak) model whose latents are regressed against RRR.
        Must be one of VALID_LEAK_MODELS.
    rrr_variance : float or str
        Fraction of RRR variance to retain (float), or ``"match"`` to use
        the same number of latents as the external model's true basis dimension.
    normalize : bool
        Whether to z-score latents before regression.
    """

    schema_version: str = "v3"

    data_config_name: str = "default"
    spks_type: SpksTypes = "sigrebase"
    activity_parameters_name: str = "default"
    method: str = "preferred"
    external_model_name: str = "fullregressor_leak_1dspeed"
    rrr_variance: Union[float, str] = 0.95
    normalize: bool = False

    display_name: ClassVar[str] = "rrr_to_external_latents"

    @staticmethod
    def _param_grid() -> dict:
        return {
            # "spks_type": list(VALID_SPKS_TYPES), # no longer analyzing anything except sigrebase
            "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
            "external_model_name": ["fullregressor_leak_1dspeed"],
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
        if self.external_model_name not in VALID_LEAK_MODELS:
            raise ValueError(f"external_model_name {self.external_model_name!r} is not a leak model. " f"Available: {VALID_LEAK_MODELS}")
        if not (isinstance(self.rrr_variance, float) or self.rrr_variance == "match"):
            raise ValueError(f"rrr_variance must be a float or 'match', got {self.rrr_variance!r}")

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"method={self.method}",
            f"ext={self.external_model_name}",
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
            external_model_name=self.external_model_name,
            rrr_variance=self.rrr_variance,
            normalize=self.normalize,
        )

        alpha_rrr_to_true, score_rrr_to_true = _optimize_alpha(
            data["train_rrr"], data["val_rrr"], data["train_external_true"], data["val_external_true"]
        )
        alpha_true_to_rrr, score_true_to_rrr = _optimize_alpha(
            data["train_external_true"], data["val_external_true"], data["train_rrr"], data["val_rrr"]
        )
        alpha_rrr_to_pred, score_rrr_to_pred = _optimize_alpha(
            data["train_rrr"], data["val_rrr"], data["train_external_pred"], data["val_external_pred"]
        )
        alpha_pred_to_rrr, score_pred_to_rrr = _optimize_alpha(
            data["train_external_pred"], data["val_external_pred"], data["train_rrr"], data["val_rrr"]
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
            "test_score_each_rrr_to_true": rrr_to_true.score(data["test_rrr"], data["test_external_true"], dim=0),
            "test_score_each_true_to_rrr": true_to_rrr.score(data["test_external_true"], data["test_rrr"], dim=0),
            "test_score_each_rrr_to_pred": rrr_to_pred.score(data["test_rrr"], data["test_external_pred"], dim=0),
            "test_score_each_pred_to_rrr": pred_to_rrr.score(data["test_external_pred"], data["test_rrr"], dim=0),
        }
