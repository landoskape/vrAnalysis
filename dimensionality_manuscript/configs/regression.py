"""ScoreModelsConfig — wraps regression model scoring from score_models.py.

This is a self-caching workflow: the regression model infrastructure manages
its own file-based cache (joblib files in ``score_path``).  The pipeline's
``ResultsStore`` records *that* the computation was done (with
``result_stored=False``), and ``get_result`` knows how to retrieve the
score dict from the model's own cache.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from ..regression_models.models import (
    PlaceFieldModel,
    RBFPosModel,
    FullRegressorModel,
    ReducedRankRegressionModel,
)
from ..pipeline.base import AnalysisConfigBase

VALID_ACTIVITY_PARAMETERS: list[str] = ["default", "preserved"]
VALID_SPKS_TYPES: list[SpksTypes] = ["oasis", "sigrebase"]

# Models plotted in the main R2 figure (figure 2). The dimensionality sweep
# below runs only over these. Placefield vector-gain models are excluded here
# because their rank sweep is handled by VectorGainRankConfig.
FIGURE_MODEL_NAMES: list[ModelName] = [
    "external_placefield_1d",
    "rbfpos_decoder_only",
    "pos_speed_decoder_only_1dspeed",
    "fullregressor_decoder_only_1dspeed",
    "internal_placefield_1d",
    "rbfpos",
    "pos_speed_1dspeed",
    "fullregressor_1dspeed",
    "external_placefield_1d_gain",
    "internal_placefield_1d_gain",
    "rrr",
]


def _log_int_values(start: int, stop: int, num: int = 25) -> np.ndarray:
    """Unique integer values spaced logarithmically over ``[start, stop]``."""
    raw = np.logspace(np.log10(start), np.log10(stop), num=num)
    return np.unique(np.round(raw).astype(int))


# Dimensionality sweep grids (integer, log-spaced).
NUM_BINS_VALUES: np.ndarray = _log_int_values(1, 200)  # placefield num_bins
NUM_BASIS_VALUES: np.ndarray = _log_int_values(1, 200)  # rbfpos / full num_basis
RANK_VALUES: np.ndarray = _log_int_values(1, 2000)  # rrr rank (clipped to data)


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

    schema_version: str = "v3"
    # v3: recompute with numerically improved placefield code

    data_config_name: str = "default"
    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "sigrebase"
    method: str = "preferred"
    activity_parameters_name: str = "default"

    display_name: ClassVar[str] = "regression"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "model_name": list(MODEL_NAMES),
            "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
            # "spks_type": list(VALID_SPKS_TYPES), # no longer analyzing anything except sigrebase
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


@dataclass(frozen=True)
class VectorGainRankConfig(AnalysisConfigBase):
    """Score external_placefield_1d_vector_gain at each SVD rank from 1 to max_rank.

    Fits N=200 SVD components in one pass using existing best hyperparameters
    (rank-agnostic cache shared with RegressionConfig), then evaluates MSE and R²
    at each rank 1…200 on the test split.

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
    spks_type: SpksTypes = "sigrebase"
    method: str = "preferred"
    activity_parameters_name: str = "default"

    # Separate this from other parameters
    max_rank: ClassVar[int] = 200
    display_name: ClassVar[str] = "vector_gain_rank"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "activity_parameters_name": list(VALID_ACTIVITY_PARAMETERS),
            # "spks_type": list(VALID_SPKS_TYPES), # no longer analyzing anything except sigrebase
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
        fit_model = PlaceFieldModel(**_shared_kwargs, rank=self.max_rank)

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

        scores: dict = defaultdict(lambda: np.full(self.max_rank, np.nan))
        for rank in range(1, self.max_rank + 1):
            arousal_activity = U_target[:, :rank] @ latent[:rank, :]  # (n_target, T)
            prediction = target_prediction + arousal_activity
            _mse = float(mse(prediction, target_data_np, reduce="mean", dim=None))
            _r2 = float(measure_r2(prediction, target_data_np, reduce="mean", dim=None))
            scores["mse"][rank - 1] = _mse
            scores["r2"][rank - 1] = _r2

        return dict(scores)


def _pack_sweep(name: str, values: np.ndarray, dim: np.ndarray, mse_arr: np.ndarray, r2_arr: np.ndarray) -> dict:
    """Flatten one dimensionality sweep into ``{name}_{values,dim,mse,r2}`` arrays."""
    return {
        f"{name}_values": np.asarray(values, dtype=float),
        f"{name}_dim": np.asarray(dim, dtype=float),
        f"{name}_mse": np.asarray(mse_arr, dtype=float),
        f"{name}_r2": np.asarray(r2_arr, dtype=float),
    }


@dataclass(frozen=True)
class RegressionDimensionalitySweepConfig(AnalysisConfigBase):
    """Sweep test performance as a function of regressor dimensionality.

    For each figure-2 model, holds the best (cached) hyperparameters fixed and
    sweeps the model's dimensionality knob on the test split:

    - Placefield models: ``num_bins`` (log 1..200).
    - RBFPos / pos_speed / full-regressor models: ``num_basis`` (log 1..200).
    - RRR: ``rank`` (log 1..2000, clipped to the achievable rank). The model is
      fit once and re-scored at each rank, since RRR training is rank-agnostic.

    For the ``num_bins``/``num_basis`` sweeps, the Gaussian smoothing width
    (``smooth_width`` / ``basis_width``) is re-derived at every value as
    ``SMOOTH_SCALE * env_length / value`` instead of held fixed at the best-hyperparameter
    value. Fixed smoothing means resolution plateaus once bin spacing drops well below the
    smoothing width — the extra bins carry no new information because the smoothing kernel
    already blends neighbors together. Scaling smoothing to bin spacing keeps the kernel
    covering roughly one neighboring bin (adjacent-bin correlation ~= exp(-1) at
    SMOOTH_SCALE=0.5) regardless of ``num_bins``/``num_basis``, so the sweep reflects
    resolution rather than a fixed low-pass filter.

    Results are flat ``{sweep}_{values,dim,mse,r2}`` arrays, where ``dim`` is the
    nominal regressor dimensionality for the swept configuration. Always uses the
    ``"default"`` activity parameters.

    Parameters
    ----------
    model_name : ModelName
        Name of the regression model (must be in ``FIGURE_MODEL_NAMES``).
    spks_type : SpksTypes
        Spike type to use for the population.
    method : str
        Hyperparameter selection method used to fix the baseline hyperparameters.
    """

    schema_version: str = "v2"
    # v2: scale smooth_width/basis_width to bin spacing during the num_bins/num_basis
    # sweep instead of holding it fixed, so resolution doesn't plateau from oversmoothing.

    data_config_name: str = "default"
    model_name: ModelName = "external_placefield_1d"
    spks_type: SpksTypes = "sigrebase"
    method: str = "best"

    SMOOTH_SCALE: ClassVar[float] = 0.5
    display_name: ClassVar[str] = "regression_dim_sweep"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "model_name": list(FIGURE_MODEL_NAMES),
        }

    def validate(self):
        if self.model_name not in FIGURE_MODEL_NAMES:
            raise ValueError(f"Unknown model_name {self.model_name!r}. Available: {', '.join(FIGURE_MODEL_NAMES)}")

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"{self.model_name}",
            f"spks={self.spks_type}",
            f"method={self.method}",
            self.schema_version,
        ]
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Run the dimensionality sweep appropriate for this model."""
        model = get_model(self.model_name, registry, activity_parameters="default")
        num_env = len(session.environments)
        base_hp = model.get_best_hyperparameters(session, spks_type=self.spks_type, method=self.method)[0]

        if isinstance(model, PlaceFieldModel):
            if np.unique(session.env_length).size != 1:
                raise ValueError("All trials must have the same environment length!")
            env_length = float(session.env_length[0])
            return _pack_sweep(
                "num_bins", *self._sweep_param(model, session, base_hp, num_env, "num_bins", NUM_BINS_VALUES, "smooth_width", env_length)
            )

        if isinstance(model, FullRegressorModel) or isinstance(model, RBFPosModel):
            if np.unique(session.env_length).size != 1:
                raise ValueError("All trials must have the same environment length!")
            env_length = float(session.env_length[0])
            return _pack_sweep(
                "num_basis", *self._sweep_param(model, session, base_hp, num_env, "num_basis", NUM_BASIS_VALUES, "basis_width", env_length)
            )

        if isinstance(model, ReducedRankRegressionModel):
            return self._sweep_rrr(model, session, base_hp)

        raise TypeError(f"No dimensionality sweep defined for model type {type(model).__name__}")

    def _sweep_param(
        self,
        model,
        session: B2Session,
        base_hp,
        num_env: int,
        param_name: str,
        values: np.ndarray,
        smooth_param_name: str,
        env_length: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Refit and score ``model`` at each value of a single integer hyperparameter.

        At each ``value``, ``smooth_param_name`` (``smooth_width``/``basis_width``) is
        overridden to ``SMOOTH_SCALE * env_length / value`` so the smoothing kernel tracks
        bin spacing instead of staying fixed at the best-hyperparameter width.
        """
        n = len(values)
        dim = np.full(n, np.nan)
        mse_arr = np.full(n, np.nan)
        r2_arr = np.full(n, np.nan)
        for i, value in enumerate(values):
            bin_spacing = env_length / float(value)
            smooth_width = self.SMOOTH_SCALE * bin_spacing
            hyperparameters = replace(base_hp, **{param_name: int(value), smooth_param_name: smooth_width})
            dim[i] = model.regressor_dimensionality(num_env, hyperparameters=hyperparameters)
            report = model.process(session, spks_type=self.spks_type, hyperparameters=hyperparameters)
            mse_arr[i] = float(report.metrics["mse"])
            r2_arr[i] = float(report.metrics["r2"])
        return values, dim, mse_arr, r2_arr

    def _sweep_rrr(self, model: ReducedRankRegressionModel, session: B2Session, base_hp) -> dict:
        """Fit RRR once at the best alpha, then re-score the test split at each rank."""
        source_data, target_data_train, _ = model.get_session_data(session, self.spks_type, "train")
        max_rank = int(min(source_data.shape[0], target_data_train.shape[0]))
        ranks = RANK_VALUES[RANK_VALUES <= max_rank]

        trained_model = model.train(session, spks_type=self.spks_type, split="train", hyperparameters=base_hp)
        target_test = model.get_session_data(session, self.spks_type, "test")[1]

        n = len(ranks)
        mse_arr = np.full(n, np.nan)
        r2_arr = np.full(n, np.nan)
        for i, rank in enumerate(ranks):
            hyperparameters = replace(base_hp, rank=int(rank))
            prediction, extras = model.predict(
                session,
                trained_model,
                spks_type=self.spks_type,
                split="test",
                hyperparameters=hyperparameters,
            )
            target = target_test
            if extras.get("predictions_were_filtered", False):
                target = target_test[:, extras["idx_valid_predictions"]]
            metrics = model.evaluate(prediction, target)
            mse_arr[i] = float(metrics["mse"])
            r2_arr[i] = float(metrics["r2"])

        return _pack_sweep("rank", ranks, ranks.astype(float), mse_arr, r2_arr)
