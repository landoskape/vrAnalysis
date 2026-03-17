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
from dimensionality_manuscript.registry import (
    MODEL_NAMES,
    ModelName,
    PopulationRegistry,
    get_model,
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

    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "oasis"
    method: str = "best"

    display_name: ClassVar[str] = "regression"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "model_name": list(MODEL_NAMES),
            "spks_type": ["oasis"],
            "method": ["best"],
        }

    def validate(self):
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model_name {self.model_name!r}. " f"Available: {', '.join(MODEL_NAMES)}")

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"{self.model_name}",
            f"spks={self.spks_type}",
            f"method={self.method}",
            self.schema_version,
        ]
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> None:
        """Score the model on this session.

        The model infrastructure caches results to its own file store,
        so we return None (completion marker) — no blob in ResultsStore.
        """
        model = get_model(self.model_name, registry)
        model.get_best_score(
            session,
            spks_type=self.spks_type,
            method=self.method,
        )
        return None

    def get_result(
        self,
        store,
        row: dict,
        *,
        session: B2Session,
        registry: PopulationRegistry,
    ) -> dict | None:
        """Load the score dict from the model's own file cache.

        Parameters
        ----------
        store : ResultsStore
            Not used — result lives in the model's cache, not the store.
        row : dict
            Summary-table row (needs ``session_id``).
        session : B2Session
            Live session object (required).
        registry : PopulationRegistry
            Population registry (required).

        Returns
        -------
        dict
            Score metrics dict (keys like ``mse``, ``r2``, ``mse_roi``, etc.).
        """
        model = get_model(self.model_name, registry)
        return model.get_best_score(
            session,
            spks_type=self.spks_type,
            method=self.method,
        )
