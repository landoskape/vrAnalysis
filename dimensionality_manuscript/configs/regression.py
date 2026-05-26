"""ScoreModelsConfig — wraps regression model scoring from score_models.py.

This is a self-caching workflow: the regression model infrastructure manages
its own file-based cache (joblib files in ``score_path``).  The pipeline's
``ResultsStore`` records *that* the computation was done (with
``result_stored=False``), and ``get_result`` knows how to retrieve the
score dict from the model's own cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from vrAnalysis.sessions import B2Session, SpksTypes
from ..registry import (
    MODEL_NAMES,
    ModelName,
    PopulationRegistry,
    get_model,
    ACTIVITY_PARAMETERS_NAMES,
)
from ..pipeline.base import AnalysisConfigBase


@dataclass(frozen=True)
class RegressionConfig(AnalysisConfigBase):
    """Configuration for regression model scoring.

    Parameters
    ----------
    model_name : ModelName
        Name of the regression model (must be in ``MODEL_NAMES``).
    spks_type : SpksTypes
        Spike type to use for the population.
    method : str
        Hyperparameter optimization method.
    """

    schema_version: str = "v2"

    data_config_name: str = "default"
    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "oasis"
    method: str = "preferred"
    activity_parameters_name: str = "default"

    display_name: ClassVar[str] = "regression"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "model_name": list(MODEL_NAMES),
            "activity_parameters_name": list(ACTIVITY_PARAMETERS_NAMES),
        }

    def validate(self):
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model_name {self.model_name!r}. " f"Available: {', '.join(MODEL_NAMES)}")
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. Available: {', '.join(list(ACTIVITY_PARAMETERS_NAMES))}"
            )

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"{self.model_name}",
            f"spks={self.spks_type}",
            f"method={self.method}",
        ]
        if self.activity_parameters_name != "default":
            parts.append(f"ap={self.activity_parameters_name}")
        parts.append(self.schema_version)
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Score the model on this session.

        The model infrastructure caches results to its own file store,
        so we return None (completion marker) — no blob in ResultsStore.
        """
        model = get_model(self.model_name, registry, activity_parameters=self.activity_parameters_name)
        score = model.get_best_score(
            session,
            spks_type=self.spks_type,
            method=self.method,
        )
        return score
