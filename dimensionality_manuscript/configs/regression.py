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
from collections import defaultdict

import numpy as np
from dimilibi import measure_r2, mse
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_placefield_prediction
from ..registry import (
    MODEL_NAMES,
    ModelName,
    PopulationRegistry,
    get_model,
    ACTIVITY_PARAMETERS_NAMES,
    get_activity_parameters,
)
from ..regression_models.models import PlaceFieldModel
from ..pipeline.base import AnalysisConfigBase


VALID_ACTIVITY_PARAMETERS: list[str] = ["default", "preserved"]
VALID_SPKS_TYPES: list[SpksTypes] = ["oasis", "sigrebase"]


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
            "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
            "spks_type": list(VALID_SPKS_TYPES),
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


_VECTOR_GAIN_RANK_MAX_RANK: int = 100


@dataclass(frozen=True)
class VectorGainRankConfig(AnalysisConfigBase):
    """Score external_placefield_1d_vector_gain at each SVD rank from 1 to max_rank.

    Fits N=100 SVD components in one pass using existing best hyperparameters
    (rank-agnostic cache shared with RegressionConfig), then evaluates MSE and R²
    at each rank 1…100 on the test split.

    Parameters
    ----------
    spks_type : SpksTypes
        Spike type to use for the population.
    method : str
        Hyperparameter optimization method.
    activity_parameters_name : str
        Activity scaling method.
    """

    schema_version: str = "v1"

    data_config_name: str = "default"
    spks_type: SpksTypes = "oasis"
    method: str = "preferred"
    activity_parameters_name: str = "default"

    display_name: ClassVar[str] = "vector_gain_rank"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
            "spks_type": list(VALID_SPKS_TYPES),
        }

    def validate(self):
        if self.activity_parameters_name not in ACTIVITY_PARAMETERS_NAMES:
            raise ValueError(
                f"Unknown activity_parameters_name {self.activity_parameters_name!r}. Available: {', '.join(list(ACTIVITY_PARAMETERS_NAMES))}"
            )

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"spks={self.spks_type}",
            f"method={self.method}",
        ]
        if self.activity_parameters_name != "default":
            parts.append(f"ap={self.activity_parameters_name}")
        parts.append(self.schema_version)
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Fit external_placefield_1d_vector_gain with N=200 SVD ranks and score at each rank."""
        max_rank = _VECTOR_GAIN_RANK_MAX_RANK
        activity_parameters = get_activity_parameters(self.activity_parameters_name)
        _shared_kwargs = dict(
            registry=registry,
            internal=False,
            gain=True,
            vector_gain=True,
            activity_parameters=activity_parameters,
        )
        # rank=1 for hyp lookup/optimization — same cache key as RegressionConfig, same SVD cost
        hyp_model = PlaceFieldModel(**_shared_kwargs, rank=1)
        # rank=max_rank for the actual multi-rank fit
        fit_model = PlaceFieldModel(**_shared_kwargs, rank=max_rank)

        hyperparameters = hyp_model.get_best_hyperparameters(
            session,
            spks_type=self.spks_type,
            method=self.method,
        )[0]

        # Train once — SVD produces U of shape (n_cells, max_rank)
        target_placefield, source_placefield, (U_target, U_source) = fit_model.train(
            session,
            spks_type=self.spks_type,
            split="train",
            hyperparameters=hyperparameters,
        )

        # Get test data
        source_data, target_data, frame_behavior = fit_model.get_session_data(session, self.spks_type, "test")
        source_data_np = source_data.numpy()

        # Source prediction and NaN filtering (mirrors predict() logic)
        source_prediction = get_placefield_prediction(source_placefield, frame_behavior)[0].T
        idx_nan = np.any(np.isnan(source_prediction), axis=0) | np.any(np.isnan(source_data_np), axis=0)
        idx_valid = ~idx_nan
        source_prediction = source_prediction[:, idx_valid]
        source_data_np = source_data_np[:, idx_valid]
        frame_behavior_filtered = frame_behavior.filter(np.where(idx_valid)[0])

        source_deviation = source_data_np - source_prediction  # (n_source, T)

        # Base target prediction on filtered frames
        target_prediction = get_placefield_prediction(target_placefield, frame_behavior_filtered)[0].T  # (n_target, T)

        # Filtered target activity for scoring
        target_data_np = target_data.numpy()[:, idx_valid]

        # Precompute latent projections for all ranks at once: (max_rank, T)
        latent = U_source.T @ source_deviation

        scores: dict = defaultdict(lambda: np.full(max_rank, np.nan))
        for rank in range(1, max_rank + 1):
            arousal_activity = U_target[:, :rank] @ latent[:rank, :]  # (n_target, T)
            prediction = target_prediction + arousal_activity
            _mse = float(mse(prediction, target_data_np, reduce="mean", dim=None))
            _r2 = float(measure_r2(prediction, target_data_np, reduce="mean", dim=None))
            scores["mse"][rank - 1] = _mse
            scores["r2"][rank - 1] = _r2

        return dict(scores)
