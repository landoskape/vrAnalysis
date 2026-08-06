from typing import Optional, Union, TYPE_CHECKING, Type
from copy import copy
import numpy as np
from scipy import sparse
from scipy.stats import norm, linregress
from sklearn.decomposition import randomized_svd
import torch
import pandas as pd
from vrAnalysis.helpers import edge2center, vectorCorrelation
from vrAnalysis.helpers.optimization import golden_section_search
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_placefield, get_placefield_prediction, Placefield, FrameBehavior, get_frame_behavior
from dimilibi import RidgeRegression, ReducedRankRegression
from .base import RegressionModel, ActivityParameters, OptimizationMethod
from .hyperparameters import (
    PlaceFieldHyperparameters,
    PlaceFieldStructuredGainHyperparameters,
    RBFPosHyperparameters,
    FullRegressorHyperparameters,
    ReducedRankRegressionHyperparameters,
)

if TYPE_CHECKING:
    from ..registry import PopulationRegistry, SplitName


def get_regressor_dimensionality_from_hyperparameters(
    hyperparameters: (
        PlaceFieldHyperparameters
        | PlaceFieldStructuredGainHyperparameters
        | RBFPosHyperparameters
        | FullRegressorHyperparameters
        | ReducedRankRegressionHyperparameters
    ),
    num_environments: int = 1,
    gain: bool = False,
    vector_gain: bool = False,
    gain_rank: int = 1,
    speed_basis: bool = True,
    no_reward: bool = False,
    reward_inclusion: Optional[dict[str, bool]] = None,
    expectation_symmetric: bool = True,
) -> int:
    """Compute effective regressor dimensionality from hyperparameters.

    Parameters
    ----------
    hyperparameters : PlaceFieldHyperparameters | RBFPosHyperparameters | FullRegressorHyperparameters | ReducedRankRegressionHyperparameters
        Hyperparameter object that defines basis counts and/or latent rank.
    num_environments : int, default=1
        Number of environments used to tile spatial basis functions.
    gain : bool, default=False
        If True for place-field models, include gain-related regressors.
    vector_gain : bool, default=False
        If True with ``gain=True`` for place-field models, include ``gain_rank`` dimensions.
        Otherwise include one scalar gain regressor.
    gain_rank : int, default=1
        Number of gain dimensions used by vector-gain place-field models.
    speed_basis : bool, default=True
        If True for full-regressor models, use ``speed_num_basis`` dimensions for speed.
        If False, use a single z-scored speed regressor.
    no_reward : bool, default=False
        If True for full-regressor models, omit all reward regressors.
    reward_inclusion : Optional[dict[str, bool]], default=None
        Inclusion mask for full-regressor reward components. Keys:
        ``expectation``, ``delivered_response``, ``omission_response``.
        If None, all three are included.
    expectation_symmetric : bool, default=True
        If True, expectation reward basis uses symmetric lags (``2 * lags + 1``);
        if False, it uses predictive-only lags (``lags + 1``).

    Returns
    -------
    int
        Effective dimensionality of model regressors/latents implied by the
        provided hyperparameters and model flags.
    """
    if num_environments < 1:
        raise ValueError("num_environments must be >= 1")

    if isinstance(hyperparameters, ReducedRankRegressionHyperparameters):
        return hyperparameters.rank

    # Checked before PlaceFieldHyperparameters because it subclasses it: the spatial maps plus
    # the latent dimensions of the structured gain.
    if isinstance(hyperparameters, PlaceFieldStructuredGainHyperparameters):
        return hyperparameters.num_bins * num_environments + hyperparameters.rank

    if isinstance(hyperparameters, PlaceFieldHyperparameters):
        base_dim = hyperparameters.num_bins * num_environments
        if not gain:
            return base_dim
        if vector_gain:
            if gain_rank < 1:
                raise ValueError("gain_rank must be >= 1 when vector_gain=True")
            return base_dim + gain_rank
        return base_dim + 1

    if isinstance(hyperparameters, RBFPosHyperparameters):
        return hyperparameters.num_basis * num_environments

    if isinstance(hyperparameters, FullRegressorHyperparameters):
        dim = hyperparameters.num_basis * num_environments
        dim += hyperparameters.speed_num_basis if speed_basis else 1
        if no_reward:
            return dim

        if reward_inclusion is None:
            reward_inclusion = {
                "expectation": True,
                "delivered_response": True,
                "omission_response": True,
            }

        if reward_inclusion.get("expectation", False):
            if expectation_symmetric:
                dim += hyperparameters.reward_num_basis_lags * 2 + 1
            else:
                dim += hyperparameters.reward_num_basis_lags + 1
        if reward_inclusion.get("delivered_response", False):
            dim += hyperparameters.reward_num_basis_lags + 1
        if reward_inclusion.get("omission_response", False):
            dim += hyperparameters.reward_num_basis_lags + 1
        return dim

    raise TypeError(f"Unsupported hyperparameters type: {type(hyperparameters)!r}")


class PlaceFieldModel(RegressionModel[PlaceFieldHyperparameters]):
    preferred_optimization_method: OptimizationMethod = "optuna"

    def __init__(
        self,
        registry: "PopulationRegistry",
        internal: bool = False,
        gain: bool = False,
        vector_gain: bool = False,
        rank: int = 1,
        hyperparameters: PlaceFieldHyperparameters = PlaceFieldHyperparameters(),
        activity_parameters: ActivityParameters = ActivityParameters(),
        autosave: bool = True,
    ):
        super().__init__(
            registry,
            activity_parameters=activity_parameters,
            autosave=autosave,
        )
        self.internal = internal
        self.gain = gain
        self.vector_gain = vector_gain
        self.rank = rank
        self.hyperparameters = hyperparameters

    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "train",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
    ) -> Union[Placefield, tuple[Placefield, Placefield], tuple[Placefield, Placefield, tuple[np.ndarray, np.ndarray]]]:
        """Train the model by predicting the place field activity on train timepoints.

        Parameters
        ----------
        session : B2Session
            The session to train the placefield model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split: Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        hyperparameters : Optional[PlaceFieldHyperparameters]
            The hyperparameters to use for the placefield model. If None, uses the default hyperparameters for the model.

        Returns
        -------
        Placefield or tuple[Placefield, Placefield] or tuple[Placefield, Placefield, tuple[np.ndarray, np.ndarray]]
            - When internal=False, returns a single Placefield object corresponding to the target cells.
            - When internal=True, returns a tuple of Placefield objects corresponding to the target and source cells.
            - When gain=True and vector_gain=True, returns a tuple of Placefield objects corresponding to the target and source cells, and a tuple of arousal coefficients for the target and source cells.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        if np.unique(session.env_length).size != 1:
            raise ValueError("All trials must have the same environment length!")

        # Set up the distance edges for the placefield model
        env_length = session.env_length[0]
        dist_edges = np.linspace(0, env_length, hyperparameters.num_bins + 1)

        # Get session data for the requested split
        source_data, target_data, frame_behavior = self.get_session_data(session, spks_type, split)

        # Then we can get the placefields
        train_target_placefield = get_placefield(
            target_data.T.numpy(),
            frame_behavior,
            dist_edges,
            speed_threshold=None,  # because we've already filtered by speed threshold!!!
            average=True,
            idx_to_spks=None,
            smooth_width=hyperparameters.smooth_width,
            zero_to_nan=True,
        )

        if self.internal or self.gain:
            # Then we also need the placefield for source cells
            train_source_placefield = get_placefield(
                source_data.T.numpy(),
                frame_behavior,
                dist_edges,
                speed_threshold=None,  # because we've already filtered by speed threshold!!!
                average=True,
                idx_to_spks=None,
                smooth_width=hyperparameters.smooth_width,
                zero_to_nan=True,
            )

        if self.gain and self.vector_gain:
            # We need to extract arousal coefficients for the source and target neurons
            # We do it together using a rank 1 decomposition of the deviation between prediction and true activity
            source_prediction = get_placefield_prediction(train_source_placefield, frame_behavior)[0].T
            target_prediction = get_placefield_prediction(train_target_placefield, frame_behavior)[0].T
            source_deviation = source_data.numpy() - source_prediction
            target_deviation = target_data.numpy() - target_prediction
            full_deviation = np.concatenate([source_deviation, target_deviation], axis=0)
            U = randomized_svd(full_deviation, n_components=self.rank, n_iter=100)[0]

            num_source = source_data.shape[0]
            if self.rank == 1:
                arousal_coefficients_source = U[:num_source, 0]
                arousal_coefficients_target = U[num_source:, 0]
            else:
                arousal_coefficients_source = U[:num_source, :]
                arousal_coefficients_target = U[num_source:, :]
            arousal_coefficients = (arousal_coefficients_target, arousal_coefficients_source)

        if self.gain and self.vector_gain:
            return train_target_placefield, train_source_placefield, arousal_coefficients

        if self.internal or self.gain:
            return train_target_placefield, train_source_placefield

        return train_target_placefield

    def decode_internal_frame_behavior(
        self,
        source_data: torch.Tensor,
        source_placefield: Placefield,
        frame_behavior: FrameBehavior,
        block_size: int = 128,
    ) -> FrameBehavior:
        """Replace behavioral position/environment with the estimate decoded from source cells.

        For each frame, finds the (environment, position bin) whose source place-field vector
        has the lowest mean squared error against the observed source activity. Bins where any
        source cell has an undefined place field (unvisited on the training split) produce a NaN
        error and are excluded from the search.

        Frames are processed in blocks: the error tensor is
        ``(environments, bins, source cells, frames)`` before the mean over cells, which is
        several GB for a whole training split. Blocking does not change the result.

        Parameters
        ----------
        source_data : torch.Tensor
            Source cell activity of shape (source cells, frames).
        source_placefield : Placefield
            Place fields of the source cells, estimated on the training split.
        frame_behavior : FrameBehavior
            Behavior for the same frames as ``source_data``.
        block_size : int
            Number of frames decoded at once. Peak memory is
            ``environments * bins * source cells * block_size`` doubles, so the default of 128
            keeps a whole-session decode to a few hundred MB.

        Returns
        -------
        FrameBehavior
            A new FrameBehavior whose position and environment are the decoded estimates. All
            other fields (speed, trial, reward, idx) are carried over unchanged.
        """
        placefield = torch.tensor(source_placefield.placefield[..., None])
        num_bins = placefield.size(1)
        num_frames = source_data.size(1)
        idx_env = np.empty(num_frames, dtype=np.int64)
        idx_pos = np.empty(num_frames, dtype=np.int64)
        for start in range(0, num_frames, block_size):
            stop = min(start + block_size, num_frames)
            error = torch.mean((source_data[None, None, :, start:stop] - placefield) ** 2, dim=2)
            error = torch.nan_to_num(error, nan=float("inf"))
            argmin = error.view(-1, error.size(-1)).argmin(dim=0).numpy()
            idx_env[start:stop] = argmin // num_bins
            idx_pos[start:stop] = argmin % num_bins

        dist_centers = edge2center(source_placefield.dist_edges)
        decoded = copy(frame_behavior)
        decoded.position = dist_centers[idx_pos]
        decoded.environment = source_placefield.environment[idx_env]
        return decoded

    def predict(
        self,
        session: B2Session,
        coefficients: Union[Placefield, tuple[Placefield, Placefield], tuple[Placefield, Placefield, tuple[np.ndarray, np.ndarray]]],
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "test",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
        nan_safe: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Predict the target place field activity for a session.

        Parameters
        ----------
        session : B2Session
            The session to predict the target place field activity for.
        coefficients : Union[Placefield, tuple[Placefield, Placefield], tuple[Placefield, Placefield, tuple[np.ndarray, np.ndarray]]]
            The "coefficients" for making a prediction, in the form of Placefield objects.
            If internal=False, coefficients should be a single Placefield object corresponding to the target cells.
            If internal=True, coefficients should be a tuple of Placefield objects corresponding to the target and source cells.
            If gain=True and vector_gain=True, coefficients should be a tuple of Placefield objects corresponding to the target and source cells, and a tuple of arousal coefficients for the target and source cells.
            Either way it is the output of self.train() given the self.internal and self.gain and self.vector_gain flags.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session
            provided as input.
        split : Optional["SplitName"]
            The split to use for the prediction. If None, uses the split from the session
            provided as input. Default is "test".
        hyperparameters : Optional[PlaceFieldHyperparameters]
            The hyperparameters used for the model. These are not actually used for prediction so the presence of this parameter
            is ignored and only here for consistency with other model types.
        nan_safe : bool
            If True, will check for NaN values in predictions and raise an error if found.
            If False, will filter out NaN samples from predictions.

        Returns
        -------
        prediction : np.ndarray
            The predicted target place field activity for the requested timepoints.
        extras : dict
            Extra information about the prediction.
        """
        # Not actually used for prediction but this is here for consistency and future proofing
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        # If internal=True or gain=True, coefficients is a tuple of Placefield objects corresponding to the target and source cells
        if self.gain and self.vector_gain:
            target_placefield, source_placefield, (arousal_coefficients_target, arousal_coefficients_source) = coefficients
        elif self.internal or self.gain:
            target_placefield, source_placefield = coefficients
        else:
            target_placefield = coefficients

        # Get session data for the requested split
        source_data, _, frame_behavior = self.get_session_data(session, spks_type, split)

        # Track original number of samples for idx_valid_predictions
        num_samples_prediction = len(frame_behavior)
        idx_valid_prediction = np.arange(num_samples_prediction, dtype=np.int64)  # Track which original indices are still valid

        extras = {"frame_behavior": copy(frame_behavior)}

        # Get the source data to predict the internal position estimates
        if self.internal:
            frame_behavior = self.decode_internal_frame_behavior(source_data, source_placefield, frame_behavior)
            extras["frame_behavior_internal"] = frame_behavior

        if self.gain:
            # Either gain or vector gain, we need to get the prediction for the source neurons
            source_prediction = torch.tensor(get_placefield_prediction(source_placefield, frame_behavior)[0].T)

            # Check for NaNs in source_prediction and source_data
            idx_nan_gain = torch.any(torch.isnan(source_prediction) | torch.isnan(source_data), dim=0)

            if nan_safe:
                if torch.any(idx_nan_gain):
                    raise ValueError(f"{torch.sum(idx_nan_gain)} / {len(idx_nan_gain)} samples have nan predictions in {session.session_print()}!!!")
            else:
                # Filter out NaN samples before computing gain
                idx_valid_gain = ~idx_nan_gain
                source_prediction = source_prediction[:, idx_valid_gain]
                source_data = source_data[:, idx_valid_gain]
                frame_behavior = frame_behavior.filter(np.where(idx_valid_gain.numpy())[0])
                # Update tracking of valid indices from original
                idx_valid_prediction = idx_valid_prediction[idx_valid_gain.numpy()]

            if self.vector_gain:
                # If the model has a vector gain component, we need to estimate the "arousal" estimates for each sample
                # by multiplying the deviation of source neuron activity from prediction by their arousal coefficients
                # Then multiplying the arousal estimate by target neuron arousal coefficients
                # Then adding that to the prediction for the target neurons
                source_deviation = source_data.numpy() - source_prediction.numpy()
                if arousal_coefficients_source.ndim == 1:
                    # rank-1: scalar arousal estimate per time point
                    arousal_estimate = arousal_coefficients_source @ source_deviation
                    arousal_activity_target = np.reshape(arousal_coefficients_target, (-1, 1)) * np.reshape(arousal_estimate, (1, -1))
                else:
                    # rank > 1: full low-rank reconstruction
                    arousal_activity_target = arousal_coefficients_target @ (arousal_coefficients_source.T @ source_deviation)

            else:
                # If the model has a gain component, we need to fit the a scalar gain value for
                # each sample. To do this, we minimize the MSE loss between the predicted and target
                # data ***for the source neurons*** which were recorded at the same time as the
                # target neurons. We assume that the gain value is the same for the whole brain.
                # This way the estimator is cross-validated by neurons.
                # -----------------------------------------------------
                with torch.no_grad():
                    gain = torch.sum(source_prediction * source_data, dim=0) / torch.sum(source_prediction**2, dim=0)
                    gain = gain.numpy()

                if nan_safe:
                    if np.any(np.isnan(gain)):
                        raise ValueError(f"{np.sum(np.isnan(gain))} / {len(gain)} gains have nan values in {session.session_print()}!!!")

        # Get prediction for the test timepoints
        prediction = get_placefield_prediction(target_placefield, frame_behavior)[0]

        if self.gain:
            if self.vector_gain:
                prediction = prediction + arousal_activity_target.T
                extras["arousal_activity_target"] = arousal_activity_target
            else:
                # Apply the gain to the prediction
                prediction = prediction * gain.reshape(-1, 1)
                extras["gain"] = gain

        # Convert to numpy for consistency
        prediction = np.array(prediction.T)

        # Check for NaNs in prediction and handle based on nan_safe
        idx_nan_samples = np.any(np.isnan(prediction), axis=0)

        if nan_safe:
            if np.any(idx_nan_samples):
                num_nan = np.sum(idx_nan_samples)
                total = len(idx_nan_samples)
                raise ValueError(f"{num_nan} / {total} samples have NaN values in prediction!")
        else:
            # Filter out NaN samples
            idx_valid_final = ~idx_nan_samples
            if np.any(idx_nan_samples):
                # Filtering occurred
                prediction = prediction[:, idx_valid_final]
                if "frame_behavior" in extras:
                    extras["frame_behavior"] = extras["frame_behavior"].filter(np.where(idx_valid_final)[0])
                if "frame_behavior_internal" in extras:
                    extras["frame_behavior_internal"] = extras["frame_behavior_internal"].filter(np.where(idx_valid_final)[0])
                if "gain" in extras:
                    # Filter gain to match filtered prediction
                    extras["gain"] = extras["gain"][idx_valid_final]

                # Update tracking: idx_valid_final is relative to current data, map back to original
                extras["idx_valid_predictions"] = idx_valid_prediction[idx_valid_final]
                extras["predictions_were_filtered"] = True
            else:
                # No NaNs, no filtering needed
                extras["predictions_were_filtered"] = False

        return prediction, extras

    @property
    def _model_hyperparameters(self) -> Type[PlaceFieldHyperparameters]:
        """Return the hyperparameter class constructor for PlaceFieldModel.

        Returns
        -------
        type[PlaceFieldHyperparameters]
            The PlaceFieldHyperparameters class constructor.
        """
        return PlaceFieldHyperparameters

    def _get_model_name(self) -> str:
        """Get the model name identifier based on the internal and gain flags.

        Returns
        -------
        str
            The model name identifier, e.g., "internal_placefield_1d_gain" or "external_placefield_1d".
        """
        # Get model name from internal and gain attributes
        model_type = "internal" if self.internal else "external"
        if self.gain:
            gain_suffix = "_vector_gain" if self.vector_gain else "_gain"
        else:
            gain_suffix = ""
        model_name = f"{model_type}_placefield_1d{gain_suffix}"
        return model_name

    def regressor_dimensionality(
        self,
        num_environments: int = 1,
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
    ) -> int:
        """Return effective dimensionality implied by place-field hyperparameters.

        Parameters
        ----------
        hyperparameters : Optional[PlaceFieldHyperparameters]
            Hyperparameters to evaluate. If None, uses ``self.hyperparameters``.

        Returns
        -------
        int
            Regressor dimensionality for this model configuration.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        return get_regressor_dimensionality_from_hyperparameters(
            num_environments=num_environments,
            hyperparameters=hyperparameters,
            gain=self.gain,
            vector_gain=self.vector_gain,
            gain_rank=self.rank,
        )

    def measure_internals(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        test_split: Optional["SplitName"] = "test",
        dev_bin_edges: np.ndarray = np.linspace(-100, 100, 101),
    ) -> tuple[np.ndarray, float, float, float]:
        """Measure the internals of the model.

        Specifically, we measure the deviation of the internal position from the true position whenever the internal
        estimate is for the correct environment (on a histogram with bins set in kwargs). We also measure the fraction
        of samples that switch environment. Lastly, we measure the R2 of the activity variance (of population sum)
        vs. internal gain estimate for the target and source neurons.

        Parameters
        ----------
        session : B2Session
            The session to measure the internals of.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        train_split : Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        test_split : Optional["SplitName"]
            The split to use for the measurement. If None, uses the split from the session provided as input. Default is "test".
        dev_bin_edges : np.ndarray
            The edges of the bins to use for the measurement. Default is np.linspace(-100, 100, 101).

        Returns
        -------
        dev_bin_counts : np.ndarray
            The number of samples in each bin of the deviation between true and internal position.
        fraction_switch_env : float
            The fraction of samples that switch environment.
        r2_gain_target : float
            The R2 of the activity variance (of population sum) vs. internal gain estimate for the target neurons.
        r2_gain_source : float
            The R2 of the activity variance (of population sum) vs. internal gain estimate for the source neurons.
        """
        if not self.internal or not self.gain:
            raise ValueError("Can only measure internals for internal+gain models!")

        # Model internals for internal placefield model & gain model
        hyperparameters = self.get_best_hyperparameters(
            session,
            spks_type=spks_type,
            method="best",
        )[0]
        report = self.process(
            session,
            spks_type=spks_type,
            train_split=train_split,
            test_split=test_split,
            hyperparameters=hyperparameters,
        )

        # Get position by environment from true behavior and internal estimate
        position_by_environment = report.extras["frame_behavior"].position_by_environment()
        position_by_environment_internal = report.extras["frame_behavior_internal"].position_by_environment()

        # Measure deviation between true and internal position within / across environments
        deviation_internal = position_by_environment_internal - position_by_environment
        env_switch_internal = np.all(np.isnan(deviation_internal), axis=0)
        truedev_internal = np.nansum(deviation_internal, axis=0)[~env_switch_internal]

        dev_bin_counts = np.histogram(truedev_internal, bins=dev_bin_edges)[0]
        fraction_switch_env = np.sum(env_switch_internal) / len(env_switch_internal)

        # Measure R2 of activity variance (of population sum) vs. internal gain estimate
        source_population_sum = torch.sum(report.extras["source_data"], axis=0)
        source_population_zscore = (source_population_sum - source_population_sum.mean()) / source_population_sum.std()
        target_population_sum = torch.sum(report.target_data, axis=0)
        target_population_zscore = (target_population_sum - target_population_sum.mean()) / target_population_sum.std()
        r2_gain_source = vectorCorrelation(np.array(source_population_zscore), np.array(report.extras["gain"]))
        r2_gain_target = vectorCorrelation(np.array(target_population_zscore), np.array(report.extras["gain"]))

        regression = linregress(np.array(source_population_zscore), np.array(report.extras["gain"]))
        slope_source = regression.slope
        yint_source = regression.intercept
        regression = linregress(np.array(target_population_zscore), np.array(report.extras["gain"]))
        slope_target = regression.slope
        yint_target = regression.intercept

        internals = {
            "dev_bin_counts": dev_bin_counts,
            "fraction_switch_env": fraction_switch_env,
            "r2_gain_target": r2_gain_target,
            "r2_gain_source": r2_gain_source,
            "slope_gain_source": slope_source,
            "yint_gain_source": yint_source,
            "slope_gain_target": slope_target,
            "yint_gain_target": yint_target,
        }

        return internals


def make_gain_unit_index(chunk_index: np.ndarray, trial: np.ndarray) -> tuple[np.ndarray, int]:
    """Group a split's frames into gain units: contiguous time chunks intersected with trials.

    A time chunk belongs to exactly one split (see
    :meth:`RegressionModel.get_split_chunk_index`), so a quantity estimated within a unit never
    mixes data across the train/validation/test folds. Chunks that span a trial boundary produce
    one unit per trial, so a unit is always a contiguous stretch of a single trial.

    Parameters
    ----------
    chunk_index : np.ndarray
        Chunk id of each frame, from ``get_split_chunk_index``.
    trial : np.ndarray
        Trial number of each frame (``FrameBehavior.trial``).

    Returns
    -------
    unit_index : np.ndarray
        Unit id of each frame, in ``[0, num_units)``.
    num_units : int
        Number of distinct units.
    """
    chunk_index = np.asarray(chunk_index)
    trial = np.asarray(trial, dtype=float)
    if len(chunk_index) != len(trial):
        raise ValueError(f"chunk_index and trial must be the same length, got {len(chunk_index)} and {len(trial)}")
    if not np.all(np.isfinite(trial)):
        raise ValueError(f"{np.sum(~np.isfinite(trial))} / {len(trial)} frames have no trial number, cannot define gain units")
    if len(chunk_index) == 0:
        return np.zeros(0, dtype=np.int64), 0

    pairs = np.stack([chunk_index.astype(float), trial])
    unit_index = np.unique(pairs, axis=1, return_inverse=True)[1]
    unit_index = np.reshape(unit_index, -1).astype(np.int64)
    return unit_index, int(unit_index.max()) + 1


def least_squares_gain(
    activity: np.ndarray,
    placefield_prediction: np.ndarray,
    unit_index: np.ndarray,
    num_units: int,
    regularization: float = 0.0,
) -> np.ndarray:
    """Least-squares scale factor between activity and its place-field prediction, per unit.

    For cell ``i`` and unit ``u``, the gain is the scalar that best rescales the cell's place
    field to match its activity over that unit's frames::

        gain[i, u] = sum_f a[i, f] * p[i, f] / sum_f p[i, f] ** 2

    so ``gain == 1`` means the cell behaved exactly as its place field predicts. This is the same
    estimator :meth:`PlaceFieldModel.predict` uses for its scalar gain, with the pooling axis
    transposed: that one pools over cells within a frame, this one pools over a unit's frames
    within a cell.

    Frames where either input is not finite are excluded (place-field predictions are NaN in bins
    that were never visited on the training split).

    The ratio is only well posed where the cell's place field predicts something over the unit.
    Where it predicts almost nothing the denominator approaches zero and the gain explodes -- in
    practice to values above 1e30, which are meaningless and destroy any downstream fit.
    ``regularization`` shrinks the estimate toward the neutral gain of 1 by adding
    ``regularization * (that cell's mean place-field energy per unit)`` to both sides of the
    ratio. Well-measured units are essentially unaffected, badly-measured ones fall back toward 1
    instead of diverging, and a cell with no place-field energy at all in a unit lands exactly on
    1 -- its prediction there is ~0 regardless of gain.

    Parameters
    ----------
    activity : np.ndarray
        Activity of shape (cells, frames).
    placefield_prediction : np.ndarray
        Place-field prediction of shape (cells, frames).
    unit_index : np.ndarray
        Unit id of each frame, from ``make_gain_unit_index``.
    num_units : int
        Number of units (columns of the result).
    regularization : float
        Shrinkage toward a gain of 1, relative to each cell's mean place-field energy per unit.
        Default is 0.0, the unregularized ratio.

    Returns
    -------
    np.ndarray
        Gains of shape (cells, num_units).
    """
    activity = np.asarray(activity, dtype=float)
    placefield_prediction = np.asarray(placefield_prediction, dtype=float)
    if activity.shape != placefield_prediction.shape:
        raise ValueError(f"activity and placefield_prediction must match, got {activity.shape} and {placefield_prediction.shape}")
    if activity.shape[1] != len(unit_index):
        raise ValueError(f"activity has {activity.shape[1]} frames but unit_index has {len(unit_index)}")
    if regularization < 0:
        raise ValueError(f"regularization must be non-negative, got {regularization}")

    valid = np.isfinite(activity) & np.isfinite(placefield_prediction)
    prediction = np.where(valid, placefield_prediction, 0.0)
    observed = np.where(valid, activity, 0.0)

    num_frames = len(unit_index)
    unit_matrix = sparse.csr_matrix(
        (np.ones(num_frames), (unit_index, np.arange(num_frames))),
        shape=(num_units, num_frames),
    )
    numerator = (unit_matrix @ (observed * prediction).T).T
    denominator = (unit_matrix @ (prediction * prediction).T).T

    if regularization > 0:
        shrinkage = regularization * np.mean(denominator, axis=1, keepdims=True)
        numerator = numerator + shrinkage
        denominator = denominator + shrinkage

    gain = np.ones_like(numerator)
    np.divide(numerator, denominator, out=gain, where=denominator > 0)
    return gain


class PlaceFieldStructuredGainModel(PlaceFieldModel):
    """Place fields modulated by a gain that is structured across cells and slow in time.

    The model sits between ``*_placefield_1d`` (no gain) and ``*_placefield_1d_gain`` (one scalar
    gain per frame, shared by the whole population and read directly off the source cells). Here
    each cell gets its own gain on each short epoch, and the map from source-cell gains to
    target-cell gains is *learned* as a reduced-rank regression, so the gain fluctuations are
    constrained to a low-dimensional shared subspace.

    Training (on the train split):

    1. Place fields for source and target cells, exactly as in :class:`PlaceFieldModel`.
    2. Frames are grouped into gain units -- one contiguous time chunk intersected with one trial.
    3. A least-squares gain per cell per unit (:func:`least_squares_gain`).
    4. A reduced-rank regression from source gains to target gains, over units.

    Prediction (on a held-out split): gains of the *source* cells are measured on that split's own
    frames, mapped through the regression to predicted target gains, and each target cell's place
    field prediction is scaled by its predicted gain for the unit that frame belongs to.

    Note on timescale: the unit is a time chunk, not a trial. The population's time splits cut a
    session into contiguous chunks that are shorter than one traversal of the track, so a per-trial
    gain would be estimated from frames spanning several folds -- a cross-validation breach, since
    most test trials also contribute frames to the training split. Estimating within a chunk keeps
    every fold's gains independent of the others, at the cost of a shorter (few second) timescale.

    Parameters
    ----------
    registry : PopulationRegistry
        The population registry to use for the model.
    internal : bool
        If True, position and environment are decoded from source-cell activity instead of read
        from behavior. The decoded estimate is used everywhere the model reads a position, on the
        training split as well as at prediction time, so the model never sees true position
        outside place-field estimation.
    fit_intercept : bool
        Whether the gain regression fits an intercept. Default is True; gains are centered near 1,
        so the intercept matters.
    min_frames_per_unit : int
        Units with fewer frames than this are excluded from the regression fit, since their gains
        are dominated by noise. They are still predicted at test time, so the held-out sample stays
        comparable with the other models. Default is 5.
    gain_regularization : float
        Shrinkage of the measured gains toward 1, relative to each cell's mean place-field energy
        per unit (see :func:`least_squares_gain`). Without it the ratio diverges for cells whose
        place field predicts nothing over a unit -- observed above 1e30 in real sessions, which is
        enough to make the gain regression non-finite. Default is 0.01.
    hyperparameters : PlaceFieldStructuredGainHyperparameters
        Place-field and gain-regression hyperparameters.
    activity_parameters : ActivityParameters
        Activity scaling parameters.
    autosave : bool
        Whether to save optimization results to the cache.
    """

    preferred_optimization_method: OptimizationMethod = "golden"

    def __init__(
        self,
        registry: "PopulationRegistry",
        internal: bool = False,
        fit_intercept: bool = True,
        min_frames_per_unit: int = 5,
        gain_regularization: float = 0.01,
        hyperparameters: PlaceFieldStructuredGainHyperparameters = PlaceFieldStructuredGainHyperparameters(),
        activity_parameters: ActivityParameters = ActivityParameters(),
        autosave: bool = True,
    ):
        super().__init__(
            registry,
            internal=internal,
            gain=True,
            vector_gain=False,
            hyperparameters=hyperparameters,
            activity_parameters=activity_parameters,
            autosave=autosave,
        )
        self.fit_intercept = fit_intercept
        self.min_frames_per_unit = min_frames_per_unit
        self.gain_regularization = gain_regularization

    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "train",
        hyperparameters: Optional[PlaceFieldStructuredGainHyperparameters] = None,
    ) -> tuple[Placefield, Placefield, ReducedRankRegression]:
        """Estimate place fields and fit the source->target gain regression.

        Parameters
        ----------
        session : B2Session
            The session to train on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        split : Optional["SplitName"]
            The split to train on. Default is "train".
        hyperparameters : Optional[PlaceFieldStructuredGainHyperparameters]
            The hyperparameters to use. If None, uses the model's default hyperparameters.

        Returns
        -------
        target_placefield : Placefield
            Place fields of the target cells.
        source_placefield : Placefield
            Place fields of the source cells.
        gain_model : ReducedRankRegression
            The fitted gain regression, mapping source-cell gains to target-cell gains.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        target_placefield, source_placefield = super().train(session, spks_type, split, hyperparameters)
        gain_model = self.fit_gain_model(
            session,
            target_placefield,
            source_placefield,
            spks_type=spks_type,
            split=split,
            alpha=hyperparameters.alpha,
        )
        return target_placefield, source_placefield, gain_model

    def fit_gain_model(
        self,
        session: B2Session,
        target_placefield: Placefield,
        source_placefield: Placefield,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "train",
        alpha: float = 1e2,
    ) -> ReducedRankRegression:
        """Fit the source->target gain regression for given place fields.

        Separated from :meth:`train` so hyperparameter optimization can hold the place fields
        fixed (they do not depend on ``alpha`` or ``rank``) and refit only the regression.

        Parameters
        ----------
        session : B2Session
            The session to fit on.
        target_placefield : Placefield
            Place fields of the target cells.
        source_placefield : Placefield
            Place fields of the source cells.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        split : Optional["SplitName"]
            The split to fit on. Default is "train".
        alpha : float
            Ridge regularization for the gain regression.

        Returns
        -------
        ReducedRankRegression
            The fitted gain regression.
        """
        source_data, target_data, frame_behavior = self.get_session_data(session, spks_type, split)
        if self.internal:
            frame_behavior = self.decode_internal_frame_behavior(source_data, source_placefield, frame_behavior)

        unit_index, num_units = make_gain_unit_index(
            self.get_split_chunk_index(session, spks_type, split),
            frame_behavior.trial,
        )
        source_prediction = get_placefield_prediction(source_placefield, frame_behavior)[0].T
        target_prediction = get_placefield_prediction(target_placefield, frame_behavior)[0].T
        source_gain = least_squares_gain(source_data.numpy(), source_prediction, unit_index, num_units, self.gain_regularization)
        target_gain = least_squares_gain(target_data.numpy(), target_prediction, unit_index, num_units, self.gain_regularization)

        idx_fit = np.bincount(unit_index, minlength=num_units) >= self.min_frames_per_unit
        if not np.any(idx_fit):
            raise ValueError(f"No gain unit has at least {self.min_frames_per_unit} frames on split {split!r} " f"in {session.session_print()}!!!")

        # Fit in double precision: gains are ratios with a long tail, and rounding them to single
        # precision has been enough to make the solve non-finite.
        gain_model = ReducedRankRegression(alpha=alpha, fit_intercept=self.fit_intercept)
        return gain_model.fit(
            torch.tensor(source_gain[:, idx_fit].T, dtype=torch.float64),
            torch.tensor(target_gain[:, idx_fit].T, dtype=torch.float64),
        )

    def prediction_inputs(
        self,
        session: B2Session,
        target_placefield: Placefield,
        source_placefield: Placefield,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "test",
        nan_safe: bool = False,
    ) -> dict:
        """Everything needed to predict a split except the gain regression itself.

        Measures the source cells' gains on this split's own frames and reads out the target place
        fields. Splitting this from :meth:`apply_gain_model` lets hyperparameter optimization reuse
        one pass over the data across every ``alpha`` and ``rank``.

        Parameters
        ----------
        session : B2Session
            The session to predict.
        target_placefield : Placefield
            Place fields of the target cells.
        source_placefield : Placefield
            Place fields of the source cells.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        split : Optional["SplitName"]
            The split to predict. Default is "test".
        nan_safe : bool
            If True, raises when any source prediction is NaN instead of dropping those frames.

        Returns
        -------
        dict
            Keys: ``target_prediction`` (targets, frames), ``source_gain`` (sources, units),
            ``unit_index`` (frames,), ``num_units``, ``frame_behavior``,
            ``frame_behavior_internal`` (only when ``internal``), ``idx_valid_prediction``
            (indices into the split's original frames) and ``was_filtered``.
        """
        source_data, _, frame_behavior = self.get_session_data(session, spks_type, split)
        unit_index, num_units = make_gain_unit_index(
            self.get_split_chunk_index(session, spks_type, split),
            frame_behavior.trial,
        )
        idx_valid_prediction = np.arange(len(frame_behavior), dtype=np.int64)
        true_frame_behavior = copy(frame_behavior)

        if self.internal:
            frame_behavior = self.decode_internal_frame_behavior(source_data, source_placefield, frame_behavior)

        source_prediction = get_placefield_prediction(source_placefield, frame_behavior)[0].T
        source_activity = source_data.numpy()

        # Frames where a source cell's place field is undefined carry no gain information; the
        # scalar-gain model drops them the same way.
        idx_nan_gain = np.any(np.isnan(source_prediction), axis=0) | np.any(np.isnan(source_activity), axis=0)
        was_filtered = False
        if nan_safe:
            if np.any(idx_nan_gain):
                raise ValueError(f"{np.sum(idx_nan_gain)} / {len(idx_nan_gain)} samples have nan predictions in {session.session_print()}!!!")
        elif np.any(idx_nan_gain):
            idx_valid_gain = np.where(~idx_nan_gain)[0]
            source_prediction = source_prediction[:, idx_valid_gain]
            source_activity = source_activity[:, idx_valid_gain]
            frame_behavior = frame_behavior.filter(idx_valid_gain)
            true_frame_behavior = true_frame_behavior.filter(idx_valid_gain)
            unit_index = unit_index[idx_valid_gain]
            idx_valid_prediction = idx_valid_prediction[idx_valid_gain]
            was_filtered = True

        inputs = {
            "target_prediction": get_placefield_prediction(target_placefield, frame_behavior)[0].T,
            "source_gain": least_squares_gain(source_activity, source_prediction, unit_index, num_units, self.gain_regularization),
            "unit_index": unit_index,
            "num_units": num_units,
            "frame_behavior": true_frame_behavior,
            "idx_valid_prediction": idx_valid_prediction,
            "was_filtered": was_filtered,
        }
        if self.internal:
            inputs["frame_behavior_internal"] = frame_behavior
        return inputs

    def apply_gain_model(
        self,
        inputs: dict,
        gain_model: ReducedRankRegression,
        rank: int,
        nan_safe: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Scale place-field predictions by the gains the regression predicts for each unit.

        Parameters
        ----------
        inputs : dict
            Output of :meth:`prediction_inputs`. Not modified.
        gain_model : ReducedRankRegression
            The fitted gain regression.
        rank : int
            Requested rank. Clipped to the regression's achievable rank, which here is bounded by
            the number of gain units and is therefore much smaller than for frame-wise models.
        nan_safe : bool
            If True, raises when the prediction contains NaN instead of dropping those frames.

        Returns
        -------
        prediction : np.ndarray
            Predicted target activity of shape (targets, frames).
        extras : dict
            Prediction extras, including the measured source gains, the predicted target gains,
            and the per-frame gain that was applied.
        """
        unit_index = inputs["unit_index"]
        source_gain = inputs["source_gain"]
        predicted_gain = (
            gain_model.predict(
                torch.tensor(source_gain.T, dtype=torch.float64),
                rank=min(int(rank), int(gain_model.max_rank)),
                nonnegative=False,
            )
            .numpy()
            .T
        )

        gain_applied = predicted_gain[:, unit_index]
        prediction = inputs["target_prediction"] * gain_applied

        extras = {
            "frame_behavior": inputs["frame_behavior"],
            "gain_source": source_gain,
            "gain_predicted": predicted_gain,
            "gain_applied": gain_applied,
            "gain_unit_index": unit_index,
        }
        if "frame_behavior_internal" in inputs:
            extras["frame_behavior_internal"] = inputs["frame_behavior_internal"]

        idx_valid_prediction = inputs["idx_valid_prediction"]
        was_filtered = inputs["was_filtered"]

        idx_nan_samples = np.any(np.isnan(prediction), axis=0)
        if nan_safe:
            if np.any(idx_nan_samples):
                raise ValueError(f"{np.sum(idx_nan_samples)} / {len(idx_nan_samples)} samples have NaN values in prediction!")
        elif np.any(idx_nan_samples):
            idx_valid_final = np.where(~idx_nan_samples)[0]
            prediction = prediction[:, idx_valid_final]
            extras["gain_applied"] = extras["gain_applied"][:, idx_valid_final]
            extras["gain_unit_index"] = extras["gain_unit_index"][idx_valid_final]
            extras["frame_behavior"] = extras["frame_behavior"].filter(idx_valid_final)
            if "frame_behavior_internal" in extras:
                extras["frame_behavior_internal"] = extras["frame_behavior_internal"].filter(idx_valid_final)
            idx_valid_prediction = idx_valid_prediction[idx_valid_final]
            was_filtered = True

        # Any dropped frame -- at the gain stage or here -- has to be reported, otherwise callers
        # line the prediction up against an unfiltered target.
        extras["predictions_were_filtered"] = was_filtered
        if was_filtered:
            extras["idx_valid_predictions"] = idx_valid_prediction

        return prediction, extras

    def predict(
        self,
        session: B2Session,
        coefficients: tuple[Placefield, Placefield, ReducedRankRegression],
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "test",
        hyperparameters: Optional[PlaceFieldStructuredGainHyperparameters] = None,
        nan_safe: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Predict target activity as place fields scaled by the predicted structured gain.

        Parameters
        ----------
        session : B2Session
            The session to predict.
        coefficients : tuple[Placefield, Placefield, ReducedRankRegression]
            Output of :meth:`train`: target place fields, source place fields, gain regression.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        split : Optional["SplitName"]
            The split to predict. Default is "test".
        hyperparameters : Optional[PlaceFieldStructuredGainHyperparameters]
            The hyperparameters to use. Only ``rank`` is used at prediction time.
        nan_safe : bool
            If True, raises when predictions contain NaN instead of dropping those frames.

        Returns
        -------
        prediction : np.ndarray
            The predicted target activity for the requested timepoints.
        extras : dict
            Extra information about the prediction.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        target_placefield, source_placefield, gain_model = coefficients
        inputs = self.prediction_inputs(
            session,
            target_placefield,
            source_placefield,
            spks_type=spks_type,
            split=split,
            nan_safe=nan_safe,
        )
        return self.apply_gain_model(inputs, gain_model, hyperparameters.rank, nan_safe=nan_safe)

    @property
    def _model_hyperparameters(self) -> Type[PlaceFieldStructuredGainHyperparameters]:
        """Return the hyperparameter class constructor for PlaceFieldStructuredGainModel.

        Returns
        -------
        type[PlaceFieldStructuredGainHyperparameters]
            The PlaceFieldStructuredGainHyperparameters class constructor.
        """
        return PlaceFieldStructuredGainHyperparameters

    def _get_model_name(self) -> str:
        """Get the model name identifier based on the internal flag.

        Returns
        -------
        str
            The model name identifier, e.g. "external_placefield_1d_structured_gain".
        """
        model_type = "internal" if self.internal else "external"
        model_name = f"{model_type}_placefield_1d_structured_gain"
        if not self.fit_intercept:
            model_name += "_no_intercept"
        return model_name

    def regressor_dimensionality(
        self,
        num_environments: int = 1,
        hyperparameters: Optional[PlaceFieldStructuredGainHyperparameters] = None,
    ) -> int:
        """Return effective dimensionality: the spatial maps plus the gain latents.

        Parameters
        ----------
        num_environments : int
            Number of environments in the session.
        hyperparameters : Optional[PlaceFieldStructuredGainHyperparameters]
            Hyperparameters to evaluate. If None, uses ``self.hyperparameters``.

        Returns
        -------
        int
            Regressor dimensionality for this model configuration.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        return get_regressor_dimensionality_from_hyperparameters(
            num_environments=num_environments,
            hyperparameters=hyperparameters,
        )

    def max_gain_rank(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "train",
    ) -> int:
        """Largest rank the gain regression can take on a split.

        Bounded by the number of gain units, which is on the order of the number of time chunks --
        far smaller than the frame counts that bound frame-wise regressions.

        Parameters
        ----------
        session : B2Session
            The session to measure.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        split : Optional["SplitName"]
            The split the gain regression is fit on. Default is "train".

        Returns
        -------
        int
            The maximum achievable rank (at least 1).
        """
        source_data, target_data, frame_behavior = self.get_session_data(session, spks_type, split)
        unit_index, num_units = make_gain_unit_index(
            self.get_split_chunk_index(session, spks_type, split),
            frame_behavior.trial,
        )
        num_fit_units = int(np.sum(np.bincount(unit_index, minlength=num_units) >= self.min_frames_per_unit))
        num_features = source_data.shape[0] + int(self.fit_intercept)
        return max(1, min(num_features, target_data.shape[0], num_fit_units))

    def inherited_placefield_hyperparameters(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        train_split: Optional["SplitName"] = "train",
        validation_split: Optional["SplitName"] = "validation",
    ) -> PlaceFieldHyperparameters:
        """Best place-field hyperparameters of the matching plain place-field model.

        The place fields here are estimated exactly as in ``external_placefield_1d`` /
        ``internal_placefield_1d``, so their optimized ``num_bins`` and ``smooth_width`` are reused
        instead of being searched again. Same session, spike type, splits and activity scaling, so
        this is a cache hit whenever that model has already been run.

        Parameters
        ----------
        session : B2Session
            The session to look up.
        spks_type : Optional[SpksTypes]
            The type of spike data to use. If None, uses the spks_type from the session.
        train_split : Optional["SplitName"]
            The training split. Default is "train".
        validation_split : Optional["SplitName"]
            The validation split. Default is "validation".

        Returns
        -------
        PlaceFieldHyperparameters
            The best place-field hyperparameters for this session.
        """
        placefield_model = PlaceFieldModel(
            self.registry,
            internal=self.internal,
            gain=False,
            activity_parameters=self.activity_parameters,
            autosave=self.autosave,
        )
        return placefield_model.get_best_hyperparameters(
            session,
            spks_type=spks_type,
            train_split=train_split,
            validation_split=validation_split,
            method="best",
        )[0]

    def _optimize_golden(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: "SplitName",
        validation_split: "SplitName",
        nan_safe: bool = False,
    ) -> tuple[dict, float, pd.DataFrame]:
        """Optimize the gain regression with golden section search.

        The place-field hyperparameters are inherited from the matching plain place-field model
        (see :meth:`inherited_placefield_hyperparameters`) and held fixed, so the place fields and
        the measured gains are computed once. Then alpha is searched with rank at its maximum, and
        rank is searched at the best alpha -- the same two-stage scheme as
        :class:`ReducedRankRegressionModel`.

        Parameters
        ----------
        session : B2Session
            The session to optimize the hyperparameters for.
        spks_type : SpksTypes
            The type of spike data to use for the population.
        train_split : "SplitName"
            The split to use for the training.
        validation_split : "SplitName"
            The split to use for the validation.
        nan_safe : bool
            If True, will check for NaN values in predictions and metrics and raise errors if found.

        Returns
        -------
        best_params : dict
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            A DataFrame with all the results from the golden section search optimization.
        """
        placefield_hyperparameters = self.inherited_placefield_hyperparameters(
            session,
            spks_type=spks_type,
            train_split=train_split,
            validation_split=validation_split,
        )
        num_bins = placefield_hyperparameters.num_bins
        smooth_width = placefield_hyperparameters.smooth_width
        hyperparameters = PlaceFieldStructuredGainHyperparameters(num_bins=num_bins, smooth_width=smooth_width)

        # Place fields and measured gains don't depend on alpha or rank, so build them once.
        target_placefield, source_placefield = PlaceFieldModel.train(self, session, spks_type, train_split, hyperparameters)
        validation_inputs = self.prediction_inputs(
            session,
            target_placefield,
            source_placefield,
            spks_type=spks_type,
            split=validation_split,
            nan_safe=nan_safe,
        )
        target_validation = self.get_session_data(session, spks_type, validation_split)[1]
        max_rank = self.max_gain_rank(session, spks_type, train_split)

        results: list[dict] = []

        def evaluate(alpha: float, rank: int, gain_model: Optional[ReducedRankRegression] = None) -> tuple[float, ReducedRankRegression]:
            """Score one (alpha, rank) pair on the validation split."""
            if gain_model is None:
                gain_model = self.fit_gain_model(
                    session,
                    target_placefield,
                    source_placefield,
                    spks_type=spks_type,
                    split=train_split,
                    alpha=alpha,
                )
            prediction, extras = self.apply_gain_model(validation_inputs, gain_model, rank, nan_safe=nan_safe)
            target = target_validation
            if extras.get("predictions_were_filtered", False):
                target = target[:, extras["idx_valid_predictions"]]
            score = float(self.evaluate(prediction, target, nan_safe=nan_safe)["mse"])
            if np.isnan(score):
                score = float("inf")
            results.append({"num_bins": num_bins, "smooth_width": smooth_width, "alpha": alpha, "rank": int(rank), "score": score})
            return score, gain_model

        best_alpha = golden_section_search(
            func=lambda alpha: evaluate(alpha, max_rank)[0],
            a=1e-2,
            b=1e6,
            tolerance_param=1e-2,
            tolerance_score=1e-3,
            max_iterations=25,
            minimize=True,
            logspace=True,
        )[0]

        # Rank is a prediction-time knob: fit once at the best alpha and re-score each rank.
        best_gain_model = self.fit_gain_model(
            session,
            target_placefield,
            source_placefield,
            spks_type=spks_type,
            split=train_split,
            alpha=best_alpha,
        )
        golden_section_search(
            func=lambda rank: evaluate(best_alpha, int(rank), gain_model=best_gain_model)[0],
            a=1.0,
            b=float(max_rank),
            tolerance_param=1.0,
            tolerance_score=1e-3,
            max_iterations=25,
            minimize=True,
            logspace=False,
        )

        best_result = min(results, key=lambda result: result["score"])
        best_params = {
            "num_bins": best_result["num_bins"],
            "smooth_width": best_result["smooth_width"],
            "alpha": best_result["alpha"],
            "rank": best_result["rank"],
        }
        return best_params, best_result["score"], pd.DataFrame(results)


def make_position_basis(
    session: B2Session,
    frame_behavior: FrameBehavior,
    hyperparameters: RBFPosHyperparameters | FullRegressorHyperparameters,
) -> torch.Tensor:
    """Make the position basis for the Full Regressor model.

    The position basis is a tensor of shape (num_timepoints, num_basis * num_environments) where
    each column is a basis function for a given environment. When viewed as a 3-tensor with shape
    (num_timepoints, num_environments, num_basis) each timepoint in a particular environment will
    have a basis function represented in basis[timepoint, environment, :] with a structure depending
    on the number of basis functions and basis width (set by hyperparameters).

    Parameters
    ----------
    session : B2Session
        The session to make the position basis for. Used simply to get environment length.
    frame_behavior : FrameBehavior
        The frame behavior to make the position basis for.
    hyperparameters : RBFPosHyperparameters | FullRegressorHyperparameters
        The hyperparameters to use for the position_basis creation.

    Returns
    -------
    basis : torch.Tensor
        The position basis of shape (num_timepoints, num_basis * num_environments).
    """
    if np.unique(session.env_length).size != 1:
        raise ValueError("All trials must have the same environment length!")

    # Set up the basis centers
    env_length = session.env_length[0]
    basis_centers = edge2center(np.linspace(0, env_length, hyperparameters.num_basis + 1))
    basis_width = hyperparameters.basis_width

    # Create the position basis
    basis = torch.tensor(norm.pdf(frame_behavior.position[:, None], basis_centers, basis_width), dtype=torch.float32)

    # Now we need to divide it by environment (right now it's agnostic)
    environments = session.environments
    env_idx = torch.tensor(np.searchsorted(environments, frame_behavior.environment))
    basis_by_env = torch.zeros((len(frame_behavior.position), len(environments), hyperparameters.num_basis))

    # Scatter in basis by environment (so it's zero everywhere else)
    env_idx_for_scatter = env_idx.unsqueeze(-1).expand(-1, hyperparameters.num_basis).unsqueeze(1)
    basis_by_env.scatter_(1, env_idx_for_scatter, basis.unsqueeze(1))
    return basis_by_env.view(len(frame_behavior.position), -1)


def make_percentile_basis(signal: np.ndarray, num_basis: int):
    """Make a basis of percentile functions for the Full Regressor model.

    The percentile basis is a tensor of shape (num_timepoints, num_basis) where each
    column is a basis function corresponding to a percentile range of the input
    signal. The value of each basis function at a given timepoint is determined by the
    distance between the signal at that timepoint and the corresponding percentile value.

    Parameters
    ----------
    signal : np.ndarray
        The input signal to make the percentile basis for.
    num_basis : int
        The number of basis functions to create.

    Returns
    -------
    basis : torch.Tensor
        The percentile basis of shape (num_timepoints, num_basis).
    """
    percentiles = edge2center(np.linspace(0, 100, num_basis + 1))
    percentile_values = np.percentile(signal, percentiles)
    basis_width = (percentile_values[1] - percentile_values[0]) * 2.0
    basis = torch.tensor(norm.pdf(signal[:, None], percentile_values, basis_width), dtype=torch.float32)
    return basis


def make_temporal_basis(
    signal: np.ndarray,
    num_lags: int,
    basis_width: float,
    only_predictive: bool = False,
    only_responsive: bool = False,
    remove_empty: bool = True,
):
    """Make a basis of temporal functions for the Full Regressor model.

    The temporal basis is a tensor of shape (num_timepoints, num_basis) where each
    column is the input signal filtered by a raised-cosine temporal basis function.
    The middle basis function is centered at lag 0, and adjacent basis functions are
    spaced by ``basis_width`` time bins.

    Parameters
    ----------
    signal : np.ndarray
        The input signal to make the temporal basis for.
    num_lags : int
        The number of lags (in addition to lag=0) to include.
    basis_width : float
        The spacing, in time bins, between adjacent raised-cosine basis centers.
    only_predictive : bool
        If True, will only use predictive lags.
    only_responsive : bool
        If True, will only use responsive lags.
    remove_empty : bool = True
        If True, will remove basis functions that have no support after clipping (note this changes the effective
        num_basis!)

    Returns
    -------
    basis : torch.Tensor
        The temporal basis of shape (num_timepoints, num_bases).
    """
    if basis_width <= 0:
        raise ValueError("basis_width must be positive!")

    if only_predictive and only_responsive:
        raise ValueError("If you want to use both predictive and responsive lags, set only_predictive and only_responsive to False!")

    signal = np.asarray(signal, dtype=float)
    num_basis = num_lags * 2 + 1
    basis_centers = (np.arange(num_basis) - num_lags) * basis_width
    max_lag = int(np.ceil(np.max(np.abs(basis_centers)) + basis_width))
    lags = np.arange(-max_lag, max_lag + 1)

    scaled_lags = (lags[:, None] - basis_centers[None, :]) / basis_width
    filters = np.zeros_like(scaled_lags, dtype=float)
    idx_supported = np.abs(scaled_lags) <= 1
    filters[idx_supported] = 0.5 * (np.cos(np.pi * scaled_lags[idx_supported]) + 1.0)
    filter_sums = filters.sum(axis=0, keepdims=True)
    filters = np.divide(filters, filter_sums, out=np.zeros_like(filters), where=filter_sums > 0)

    if only_predictive:
        filters[lags > 0, :] = 0.0
    elif only_responsive:
        filters[lags < 0, :] = 0.0

    if remove_empty:
        idx_nonzero = np.any(filters != 0, axis=0)
        filters = filters[:, idx_nonzero]
        basis_centers = basis_centers[idx_nonzero]
        num_basis = filters.shape[1]

    padded_signal = np.pad(signal, (max_lag, max_lag), mode="constant")
    basis = np.column_stack([np.convolve(padded_signal, filters[:, i], mode="valid") for i in range(num_basis)])
    basis = torch.tensor(basis, dtype=torch.float32)
    return basis


class RBFPosModel(RegressionModel[RBFPosHyperparameters]):
    preferred_optimization_method: OptimizationMethod = "optuna"

    def __init__(
        self,
        registry: "PopulationRegistry",
        split_train: bool = True,
        predict_latents: bool = True,
        fit_intercept: bool = True,
        hyperparameters: RBFPosHyperparameters = RBFPosHyperparameters(),
        activity_parameters: ActivityParameters = ActivityParameters(),
        autosave: bool = True,
    ):
        super().__init__(
            registry,
            activity_parameters=activity_parameters,
            autosave=autosave,
        )
        self.hyperparameters = hyperparameters
        self.fit_intercept = fit_intercept
        self.nonnegative = True

        # This model requires double-cross-validation to prevent non-spatial leakage
        # between activity and position in the training set. To account for this, the
        # population registry created two training sets -- train_0 and train_1 -- which
        # are used to train the encoder and decoder respectively. (They're usually combined
        # in other models).
        # ------------------------------------------------------------------------------------
        # To keep the API consistent with other models, I didn't want to add an additional
        # split parameter for the train split - so instead we set a flag called _split_train
        # which tells us to split 'train' into 'train0' and 'train1'.... but we *won't* double
        # cross-validate if any other split is requested.
        self.predict_latents = predict_latents
        if not predict_latents:
            # When we're not predicting latents, we don't need to split the training set
            # into the decoder and encoder splits!
            self.split_train = False
        else:
            self.split_train = split_train

    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "train",
        hyperparameters: Optional[RBFPosHyperparameters] = None,
    ) -> Union[RidgeRegression, tuple[RidgeRegression, RidgeRegression]]:
        """Train the model by fitting the RBF(Pos) model to the training data.

        Parameters
        ----------
        session : B2Session
            The session to train the RBF(Pos) model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split: Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
            When _split_train is True, 'train' is split into 'train0' and 'train1' for the encoder and decoder.
        hyperparameters : Optional[RBFPosHyperparameters]
            The hyperparameters to use for the RBF(Pos) model. If None, uses the default hyperparameters for the model.

        Returns
        -------
        RidgeRegression or tuple[RidgeRegression, RidgeRegression]
            The trained encoder and decoder models. The encoder model predicts position basis from activity of the source
            neurons, and the decoder model predicts activity of the target neurons from the position basis.
            - When predict_latents is False, returns a single RidgeRegression object corresponding to the decoder model.
            - When predict_latents is True, returns a tuple of RidgeRegression objects corresponding to the encoder and decoder models.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        # Split the neural data
        if self.split_train and split == "train":
            encoder_split = "train0"
            decoder_split = "train1"
        else:
            encoder_split = split
            decoder_split = split

        if self.predict_latents:
            # Train the encoder model to predict the position basis from source neuron activity
            source_data_encoder, _, frame_behavior_encoder = self.get_session_data(session, spks_type, encoder_split)
            basis_for_encoder = make_position_basis(session, frame_behavior_encoder, hyperparameters)
            encoder = RidgeRegression(alpha=hyperparameters.alpha_encoder, fit_intercept=self.fit_intercept)
            encoder = encoder.fit(source_data_encoder.T, basis_for_encoder)

        # Train the decoder model to predict the target neuron activity from the position basis
        _, target_data_decoder, frame_behavior_decoder = self.get_session_data(session, spks_type, decoder_split)
        basis_for_decoder = make_position_basis(session, frame_behavior_decoder, hyperparameters)
        decoder = RidgeRegression(alpha=hyperparameters.alpha_decoder, fit_intercept=self.fit_intercept)
        decoder = decoder.fit(basis_for_decoder, target_data_decoder.T)

        if self.predict_latents:
            return encoder, decoder
        else:
            return decoder

    def predict(
        self,
        session: B2Session,
        rbfpos_model: Union[RidgeRegression, tuple[RidgeRegression, RidgeRegression]],
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "test",
        hyperparameters: Optional[RBFPosHyperparameters] = None,
        nan_safe: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Predict the target place field activity for a session.

        Parameters
        ----------
        session : B2Session
            The session to predict the target place field activity for.
        rbfpos_model : Union[RidgeRegression, tuple[RidgeRegression, RidgeRegression]]
            The trained encoder and decoder models. If predict_latents is False, rbfpos_model is a single RidgeRegression object corresponding to the decoder model.
            If predict_latents is True, rbfpos_model is a tuple of RidgeRegression objects corresponding to the encoder and decoder models. Otherwise, it is just the decoder model.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session
            provided as input.
        split : Optional["SplitName"]
            The split to use for the prediction. If None, uses the split from the session
            provided as input. Default is "test".
        hyperparameters : Optional[RBFPosHyperparameters]
            The hyperparameters used for the model. These are not actually used for prediction so the presence of this parameter
            is ignored and only here for consistency with other model types.
        nan_safe : bool
            If True, will check for NaN values in predictions and raise an error if found.
            If False, will filter out NaN samples from predictions.

        Returns
        -------
        prediction : np.ndarray
            The predicted target data for the requested timepoints.
        extras : dict
            Extra information about the prediction. Contains the "true" position basis and the predicted position basis.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        # Get source activity and frame_behavior for the requested split
        source_data, _, frame_behavior = self.get_session_data(session, spks_type, split)

        # Make the position basis... of the "true" position
        position_basis = make_position_basis(session, frame_behavior, hyperparameters)
        extras = {"position_basis": np.array(position_basis)}

        if self.predict_latents:
            # Predict the position basis with the encoder model, then the target from the predicted basis
            position_basis_predicted = rbfpos_model[0].predict(source_data.T, nonnegative=self.nonnegative)
            prediction = rbfpos_model[1].predict(position_basis_predicted, nonnegative=self.nonnegative).T
            extras["position_basis_predicted"] = np.array(position_basis_predicted)

        else:
            # Predict the target from the true position basis
            prediction = rbfpos_model.predict(position_basis, nonnegative=self.nonnegative).T

        prediction = np.array(prediction)

        # Check for NaNs in prediction and handle based on nan_safe
        idx_nan_samples = np.any(np.isnan(prediction), axis=0)

        if nan_safe:
            if np.any(idx_nan_samples):
                num_nan = np.sum(idx_nan_samples)
                total = len(idx_nan_samples)
                raise ValueError(f"{num_nan} / {total} samples have NaN values in prediction!")
        else:
            # Filter out NaN samples
            idx_valid = ~idx_nan_samples
            if np.any(idx_nan_samples):
                # Filtering occurred
                prediction = prediction[:, idx_valid]
                if "position_basis" in extras:
                    extras["position_basis"] = extras["position_basis"][idx_valid]
                if "position_basis_predicted" in extras:
                    extras["position_basis_predicted"] = extras["position_basis_predicted"][idx_valid]

                # Track which original samples are valid
                extras["idx_valid_predictions"] = np.where(idx_valid)[0]
                extras["predictions_were_filtered"] = True
            else:
                # No NaNs, no filtering needed
                extras["predictions_were_filtered"] = False

        return prediction, extras

    @property
    def _model_hyperparameters(self) -> Type[RBFPosHyperparameters]:
        """Return the hyperparameter class constructor for RBFPosModel.

        Returns
        -------
        type[RBFPosHyperparameters]
            The RBFPosHyperparameters class constructor.
        """
        return RBFPosHyperparameters

    def _get_model_name(self) -> str:
        """Get the model name identifier.

        Returns
        -------
        str
            The model name identifier, "rbfpos", "rbfpos_decoder_only", or "rbfpos_leak" for RBFPosModel.
            The "_decoder_only" suffix indicates that the model was trained to predict target neurons from
            True position basis, rather than a prediction of the position basis from source neurons..
            The "_leak" suffix indicates that the model was trained without double-cross-validation,
            which allows for non-spatial leakage between activity and position in the training set.

            Note that if predict_latents is False, split_train is ignored.
        """
        model_name = "rbfpos"
        if not self.predict_latents:
            model_name += "_decoder_only"
        elif not self.split_train:
            model_name += "_leak"
        if not self.fit_intercept:
            model_name += "_no_intercept"
        return model_name

    def regressor_dimensionality(
        self,
        num_environments: int = 1,
        hyperparameters: Optional[RBFPosHyperparameters] = None,
    ) -> int:
        """Return effective dimensionality implied by RBF position hyperparameters.

        Parameters
        ----------
        num_environments : int
            Number of environments in the session.
        hyperparameters : Optional[RBFPosHyperparameters]
            Hyperparameters to evaluate. If None, uses ``self.hyperparameters``.

        Returns
        -------
        int
            Regressor dimensionality for this model configuration.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        return get_regressor_dimensionality_from_hyperparameters(
            hyperparameters=hyperparameters,
            num_environments=num_environments,
        )


class FullRegressorModel(RegressionModel[FullRegressorHyperparameters]):
    preferred_optimization_method: OptimizationMethod = "optuna"

    def __init__(
        self,
        registry: "PopulationRegistry",
        split_train: bool = True,
        predict_latents: bool = True,
        speed_basis: bool = True,
        no_reward: bool = False,
        predictive_reward: bool = False,
        fit_intercept: bool = True,
        hyperparameters: FullRegressorHyperparameters = FullRegressorHyperparameters(),
        activity_parameters: ActivityParameters = ActivityParameters(),
        autosave: bool = True,
    ):
        super().__init__(
            registry,
            activity_parameters=activity_parameters,
            autosave=autosave,
        )
        self.hyperparameters = hyperparameters
        self.fit_intercept = fit_intercept
        self.nonnegative = True
        self.speed_basis = speed_basis
        self.no_reward = no_reward

        if predictive_reward and no_reward:
            raise ValueError("predictive_reward and no_reward are mutually exclusive!")
        self.predictive_reward = predictive_reward

        # Name the three components of the reward regressors for optional manual sculpting of the model
        # (Not going to put this in the main constructor API yet unless it helps)
        # ------------------------------------------------------------------------------------
        # The predictive_reward flag makes the reward regressors causally clean: the expectation
        # basis only looks forward in time (so it can't report on whether reward actually arrived)
        # and the omission response -- which is only defined after the fact -- is dropped entirely.
        # The delivered response is unchanged.
        self.reward_inclusion = {
            "expectation": True,
            "delivered_response": True,
            "omission_response": not predictive_reward,
        }
        self.expectation_symmetric = not predictive_reward

        # This model requires double-cross-validation to prevent non-spatial leakage
        # between activity and position in the training set. To account for this, the
        # population registry created two training sets -- train_0 and train_1 -- which
        # are used to train the encoder and decoder respectively. (They're usually combined
        # in other models).
        # ------------------------------------------------------------------------------------
        # To keep the API consistent with other models, I didn't want to add an additional
        # split parameter for the train split - so instead we set a flag called _split_train
        # which tells us to split 'train' into 'train0' and 'train1'.... but we *won't* double
        # cross-validate if any other split is requested.
        self.predict_latents = predict_latents
        if not predict_latents:
            # When we're not predicting latents, we don't need to split the training set
            # into the decoder and encoder splits!
            self.split_train = False
        else:
            self.split_train = split_train

    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "train",
        hyperparameters: Optional[FullRegressorHyperparameters] = None,
    ) -> Union[RidgeRegression, tuple[RidgeRegression, RidgeRegression]]:
        """Train the model by fitting the Full Regressor model to the training data.

        Parameters
        ----------
        session : B2Session
            The session to train the Full Regressor model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split: Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
            When _split_train is True, 'train' is split into 'train0' and 'train1' for the encoder and decoder.
        hyperparameters : Optional[FullRegressorHyperparameters]
            The hyperparameters to use for the Full Regressor model. If None, uses the default hyperparameters for the model.

        Returns
        -------
        RidgeRegression or tuple[RidgeRegression, RidgeRegression]
            The trained encoder and decoder models. The encoder model predicts position basis from activity of the source
            neurons, and the decoder model predicts activity of the target neurons from the position basis.
            The Full Regressor Model (in comparison to RBFPosModel) extends the regressors from *just* the position basis
            to include the running speed and a reward prediction signal.
            - When predict_latents is False, returns a single RidgeRegression object corresponding to the decoder model.
            - When predict_latents is True, returns a tuple of RidgeRegression objects corresponding to the encoder and decoder models.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        # Split the neural data
        if self.split_train and split == "train":
            encoder_split = "train0"
            decoder_split = "train1"
        else:
            encoder_split = split
            decoder_split = split

        if self.predict_latents:
            # Train the encoder model to predict the position basis from source neuron activity
            source_data_encoder, _, frame_behavior_encoder = self.get_session_data(session, spks_type, encoder_split)
            basis_for_encoder = self.build_regressors(session, frame_behavior_encoder, hyperparameters)
            encoder = RidgeRegression(alpha=hyperparameters.alpha_encoder, fit_intercept=self.fit_intercept)
            encoder = encoder.fit(source_data_encoder.T, basis_for_encoder)

        # Train the decoder model to predict the target neuron activity from the position basis
        _, target_data_decoder, frame_behavior_decoder = self.get_session_data(session, spks_type, decoder_split)
        basis_for_decoder = self.build_regressors(session, frame_behavior_decoder, hyperparameters)
        decoder = RidgeRegression(alpha=hyperparameters.alpha_decoder, fit_intercept=self.fit_intercept)
        decoder = decoder.fit(basis_for_decoder, target_data_decoder.T)

        if self.predict_latents:
            return encoder, decoder
        else:
            return decoder

    def predict(
        self,
        session: B2Session,
        fullreg_model: Union[RidgeRegression, tuple[RidgeRegression, RidgeRegression]],
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "test",
        hyperparameters: Optional[FullRegressorHyperparameters] = None,
        nan_safe: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Predict the target place field activity for a session.

        Parameters
        ----------
        session : B2Session
            The session to predict the target place field activity for.
        fullreg_model : Union[RidgeRegression, tuple[RidgeRegression, RidgeRegression]]
            The trained encoder and decoder models. If predict_latents is False, fullreg_model is a single RidgeRegression object corresponding to the decoder model.
            If predict_latents is True, fullreg_model is a tuple of RidgeRegression objects corresponding to the encoder and decoder models. Otherwise, it is just the decoder model.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session
            provided as input.
        split : Optional["SplitName"]
            The split to use for the prediction. If None, uses the split from the session
            provided as input. Default is "test".
        hyperparameters : Optional[FullRegressorHyperparameters]
            The hyperparameters used for the model. These are not actually used for prediction so the presence of this parameter
            is ignored and only here for consistency with other model types.
        nan_safe : bool
            If True, will check for NaN values in predictions and raise an error if found.
            If False, will filter out NaN samples from predictions.

        Returns
        -------
        prediction : np.ndarray
            The predicted target data for the requested timepoints.
        extras : dict
            Extra information about the prediction. Contains the "true" position basis and the predicted position basis.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        # Get source activity and frame_behavior for the requested split
        source_data, _, frame_behavior = self.get_session_data(session, spks_type, split)

        # Make the position basis... of the "true" position
        basis_functions = self.build_regressors(session, frame_behavior, hyperparameters)
        extras = {"basis_functions": np.array(basis_functions)}

        if self.predict_latents:
            # Predict the position basis with the encoder model, then the target from the predicted basis
            basis_functions_predicted = fullreg_model[0].predict(source_data.T, nonnegative=self.nonnegative)
            prediction = fullreg_model[1].predict(basis_functions_predicted, nonnegative=self.nonnegative).T
            extras["basis_functions_predicted"] = np.array(basis_functions_predicted)

        else:
            # Predict the target from the true position basis
            prediction = fullreg_model.predict(basis_functions, nonnegative=self.nonnegative).T

        prediction = np.array(prediction)

        # Check for NaNs in prediction and handle based on nan_safe
        idx_nan_samples = np.any(np.isnan(prediction), axis=0)

        if nan_safe:
            if np.any(idx_nan_samples):
                num_nan = np.sum(idx_nan_samples)
                total = len(idx_nan_samples)
                raise ValueError(f"{num_nan} / {total} samples have NaN values in prediction!")
        else:
            # Filter out NaN samples
            idx_valid = ~idx_nan_samples
            if np.any(idx_nan_samples):
                # Filtering occurred
                prediction = prediction[:, idx_valid]
                if "basis_functions" in extras:
                    extras["basis_functions"] = extras["basis_functions"][idx_valid]
                if "basis_functions_predicted" in extras:
                    extras["basis_functions_predicted"] = extras["basis_functions_predicted"][idx_valid]

                # Track which original samples are valid
                extras["idx_valid_predictions"] = np.where(idx_valid)[0]
                extras["predictions_were_filtered"] = True
            else:
                # No NaNs, no filtering needed
                extras["predictions_were_filtered"] = False

        return prediction, extras

    def build_regressors(
        self,
        session: B2Session,
        frame_behavior: FrameBehavior,
        hyperparameters: Optional[FullRegressorHyperparameters] = None,
        as_list: bool = False,
    ) -> torch.Tensor:
        """Make the position basis for the Full Regressor model.

        The position basis is a tensor of shape (num_timepoints, num_basis * num_environments) where
        each column is a basis function for a given environment. When viewed as a 3-tensor with shape
        (num_timepoints, num_environments, num_basis) each timepoint in a particular environment will
        have a basis function represented in basis[timepoint, environment, :] with a structure depending
        on the number of basis functions and basis width (set by hyperparameters).

        Parameters
        ----------
        session : B2Session
            The session to make the position basis for. Used simply to get environment length.
        frame_behavior : FrameBehavior
            The frame behavior to make the position basis for.
        hyperparameters : Optional[FullRegressorHyperparameters]
            The hyperparameters to use for the Full Regressor model. If None, uses the default hyperparameters for the model.
        as_list : bool
            If True, will return the different components of the basis as a list of tensors rather than concatenating them.
            The order of the list is [position_basis, speed_basis, reward_expectation_basis, reward_delivery_basis, reward_omission_basis].
            Reward components excluded by ``no_reward`` / ``predictive_reward`` are simply absent from the list, so index
            positionally at your own risk.

        Returns
        -------
        basis : torch.Tensor
            The position basis for the Full Regressor model of shape (num_timepoints, num_basis * num_environments).
        """
        _return_basis = lambda basis_list: basis_list if as_list else torch.cat(basis_list, dim=1)
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        # Get position basis (same as RBFPos Model)
        position_basis = make_position_basis(session, frame_behavior, hyperparameters)

        # Now make a speed basis from frame_behavior
        speed = frame_behavior.speed
        if self.speed_basis:
            speed_basis = make_percentile_basis(speed, hyperparameters.speed_num_basis)
        else:
            # Speed basis is just the speed itself after z-scoring
            speed_basis = torch.tensor((speed - np.mean(speed)) / np.std(speed), dtype=torch.float32).unsqueeze(-1)

        basis_list = [position_basis, speed_basis]
        if self.no_reward:
            # If no_reward flag is set, we won't include any reward-related basis functions
            return _return_basis(basis_list)

        # For the reward basis, we need to build the temporal basis from the *whole* session,
        # not just the split provided by get_session_data and passed through to here via frame_behavior.
        # This is because we use a temporal convolution with shifted basis, so we might need our basis
        # to include lags that respond (or predict) reward events that occur outside of the split!

        # Start by getting full frame_behavior
        frame_behavior_full = get_frame_behavior(session, clear_one_cache=False)

        # And also a reward prediction / response basis
        reward_delivery = frame_behavior_full.reward_delivery
        reward_omitted = frame_behavior_full.reward_omitted
        reward_expected = np.logical_or(reward_delivery, reward_omitted)

        if self.reward_inclusion["expectation"]:
            reward_expectation_basis = make_temporal_basis(
                reward_expected,
                hyperparameters.reward_num_basis_lags,
                hyperparameters.reward_basis_width,
                only_predictive=not self.expectation_symmetric,
            )[frame_behavior.idx]
            basis_list.append(reward_expectation_basis)

        if self.reward_inclusion["delivered_response"]:
            reward_delivery_basis = make_temporal_basis(
                reward_delivery,
                hyperparameters.reward_num_basis_lags,
                hyperparameters.reward_basis_width,
                only_responsive=True,
            )[frame_behavior.idx]
            basis_list.append(reward_delivery_basis)

        if self.reward_inclusion["omission_response"]:
            reward_omitted_basis = make_temporal_basis(
                reward_omitted,
                hyperparameters.reward_num_basis_lags,
                hyperparameters.reward_basis_width,
                only_responsive=True,
            )[frame_behavior.idx]
            basis_list.append(reward_omitted_basis)

        return _return_basis(basis_list)

    @property
    def _model_hyperparameters(self) -> Type[FullRegressorHyperparameters]:
        """Return the hyperparameter class constructor for FullRegressorModel.

        Returns
        -------
        type[FullRegressorHyperparameters]
            The FullRegressorHyperparameters class constructor.
        """
        return FullRegressorHyperparameters

    def _get_model_name(self) -> str:
        """Get the model name identifier.

        Returns
        -------
        str
            The model name identifier, "fullregressor", "fullregressor_decoder_only", or "fullregressor_leak" for FullRegressorModel.
            The "_decoder_only" suffix indicates that the model was trained to predict target neurons from
            True position basis, rather than a prediction of the position basis from source neurons..
            The "_leak" suffix indicates that the model was trained without double-cross-validation,
            which allows for non-spatial leakage between activity and position in the training set.
            The "_predreward" suffix indicates that the reward regressors are causally clean, i.e. the
            expectation basis is predictive-only and there is no reward omission response.

            Note that if predict_latents is False, split_train is ignored.
        """
        model_name = "fullregressor"
        if not self.predict_latents:
            model_name += "_decoder_only"
        elif not self.split_train:
            model_name += "_leak"
        if not self.speed_basis:
            model_name += "_1dspeed"
        if self.no_reward:
            model_name += "_noreward"
        if self.predictive_reward:
            model_name += "_predreward"
        if not self.fit_intercept:
            model_name += "_no_intercept"
        return model_name

    def regressor_dimensionality(
        self,
        num_environments: int = 1,
        hyperparameters: Optional[FullRegressorHyperparameters] = None,
    ) -> int:
        """Return effective dimensionality implied by full-regressor hyperparameters.

        Parameters
        ----------
        num_environments : int
            Number of environments in the session.
        hyperparameters : Optional[FullRegressorHyperparameters]
            Hyperparameters to evaluate. If None, uses ``self.hyperparameters``.

        Returns
        -------
        int
            Regressor dimensionality for this model configuration.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        return get_regressor_dimensionality_from_hyperparameters(
            hyperparameters=hyperparameters,
            num_environments=num_environments,
            speed_basis=self.speed_basis,
            no_reward=self.no_reward,
            reward_inclusion=self.reward_inclusion,
            expectation_symmetric=self.expectation_symmetric,
        )


class ReducedRankRegressionModel(RegressionModel[ReducedRankRegressionHyperparameters]):
    preferred_optimization_method: OptimizationMethod = "golden"

    def __init__(
        self,
        registry: "PopulationRegistry",
        fit_intercept: bool = True,
        hyperparameters: ReducedRankRegressionHyperparameters = ReducedRankRegressionHyperparameters(),
        activity_parameters: ActivityParameters = ActivityParameters(),
        autosave: bool = True,
    ):
        super().__init__(
            registry,
            activity_parameters=activity_parameters,
            autosave=autosave,
        )
        self.hyperparameters = hyperparameters
        self.fit_intercept = fit_intercept
        self.nonnegative = True

    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "train",
        hyperparameters: Optional[ReducedRankRegressionHyperparameters] = None,
    ) -> ReducedRankRegression:
        """Train the model by fitting the reduced rank regression model to the training data.

        Parameters
        ----------
        session : B2Session
            The session to train the reduced rank regression model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split: Optional["SplitName"]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        hyperparameters : Optional[ReducedRankRegressionHyperparameters]
            The hyperparameters to use for the reduced rank regression model. If None, uses the default hyperparameters for the model.
        fit_intercept : bool
            Whether to fit an intercept term in the regression model. Default is True.

        Returns
        -------
        ReducedRankRegression
            The trained ReducedRankRegression model.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        # Get the source and target activity data for the requested split
        source_data, target_data, _ = self.get_session_data(session, spks_type, split)

        # Fit a reduced rank regression model to the training data
        rrr_model = ReducedRankRegression(alpha=hyperparameters.alpha, fit_intercept=self.fit_intercept)
        return rrr_model.fit(source_data.T, target_data.T)

    def predict(
        self,
        session: B2Session,
        rrr_model: ReducedRankRegression,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "test",
        hyperparameters: Optional[ReducedRankRegressionHyperparameters] = None,
        nan_safe: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Predict the target place field activity for a session.

        Parameters
        ----------
        session : B2Session
            The session to predict the target place field activity for.
        rrr_model : ReducedRankRegression
            The trained ReducedRankRegression model.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session
            provided as input.
        split : Optional["SplitName"]
            The split to use for the prediction. If None, uses the split from the session
            provided as input. Default is "test".
        hyperparameters : Optional[ReducedRankRegressionHyperparameters]
            The hyperparameters used for the model. These are not actually used for prediction so the presence of this parameter
            is ignored and only here for consistency with other model types.
        nan_safe : bool
            If True, will check for NaN values in predictions and raise an error if found.
            If False, will filter out NaN samples from predictions.

        Returns
        -------
        prediction : np.ndarray
            The predicted target data for the requested timepoints.
        extras : dict
            Extra information about the prediction.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        # Get the source activity data for the requested split
        source_data, _, _ = self.get_session_data(session, spks_type, split)

        # Predict the target activity with the trained model
        prediction = rrr_model.predict(source_data.T, rank=hyperparameters.rank, nonnegative=self.nonnegative).T
        latents = rrr_model.predict_latent(source_data.T, rank=hyperparameters.rank)
        extras = {
            "latents": np.array(latents).T,
        }

        prediction = np.array(prediction)

        # Check for NaNs in prediction and handle based on nan_safe
        idx_nan_samples = np.any(np.isnan(prediction), axis=0)

        if nan_safe:
            if np.any(idx_nan_samples):
                num_nan = np.sum(idx_nan_samples)
                total = len(idx_nan_samples)
                raise ValueError(f"{num_nan} / {total} samples have NaN values in prediction!")
        else:
            # Filter out NaN samples
            idx_valid = ~idx_nan_samples
            if np.any(idx_nan_samples):
                # Filtering occurred
                prediction = prediction[:, idx_valid]
                if "latents" in extras:
                    extras["latents"] = extras["latents"][:, idx_valid]

                # Track which original samples are valid
                extras["idx_valid_predictions"] = np.where(idx_valid)[0]
                extras["predictions_were_filtered"] = True
            else:
                # No NaNs, no filtering needed
                extras["predictions_were_filtered"] = False

        return prediction, extras

    def score_curve(
        self,
        session: B2Session,
        rrr_model: ReducedRankRegression,
        spks_type: Optional[SpksTypes] = None,
        split: Optional["SplitName"] = "test",
        ranks: Optional[list[int]] = None,
        nan_safe: bool = False,
        reduce: str = "mean",
        verbose: bool = False,
    ) -> tuple[list[int], list]:
        """Score a trained RRR model across a range of ranks in a single pass.

        Thin wrapper around ``ReducedRankRegression.score_curve``. The trained model already
        holds the OLS coefficients and the rank-constraint basis, so scoring every rank reuses
        one set of latent projections rather than recomputing coefficients per rank. Returns
        R^2 (via ``measure_r2``), not MSE like ``score``.

        Parameters
        ----------
        session : B2Session
            The session to score the model on.
        rrr_model : ReducedRankRegression
            The trained ReducedRankRegression model (output of ``self.train``).
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split : Optional["SplitName"]
            The split to use for the scoring. If None, uses the split from the session provided as input. Default is "test".
        ranks : Optional[list[int]]
            The ranks at which to score the model. If None, scores every rank from 1 to the
            model's ``max_rank``. Values must be in ``[1, max_rank]``.
        nan_safe : bool
            If True, raises if any source/target sample contains NaN. If False, drops those
            samples before scoring.
        reduce : str
            Reduction passed through to ``measure_r2``: "mean" returns a float per rank, "none"
            returns a per-target tensor per rank. Default is "mean".
        verbose : bool
            If True, prints progress to stdout. Default is False.

        Returns
        -------
        ranks : list[int]
            The (sorted, de-duplicated) ranks that were scored.
        scores : list
            The R^2 score at each rank, aligned with ``ranks``.
        """
        source_data, target_data, _ = self.get_session_data(session, spks_type, split)
        X = source_data.T
        y = target_data.T

        idx_nan = torch.any(torch.isnan(X), dim=1) | torch.any(torch.isnan(y), dim=1)
        if torch.any(idx_nan):
            if nan_safe:
                raise ValueError(f"{torch.sum(idx_nan)} / {len(idx_nan)} samples have NaN values in {session.session_print()}!!!")
            X = X[~idx_nan]
            y = y[~idx_nan]

        return rrr_model.score_curve(X, y, ranks=ranks, nonnegative=self.nonnegative, reduce=reduce, verbose=verbose)

    @property
    def _model_hyperparameters(self) -> Type[ReducedRankRegressionHyperparameters]:
        """Return the hyperparameter class constructor for ReducedRankRegressionModel.

        Returns
        -------
        type[ReducedRankRegressionHyperparameters]
            The ReducedRankRegressionHyperparameters class constructor.
        """
        return ReducedRankRegressionHyperparameters

    def _get_model_name(self) -> str:
        """Get the model name identifier.

        Returns
        -------
        str
            The model name identifier, always "rrr" for ReducedRankRegressionModel.
        """
        model_name = "rrr"
        if not self.fit_intercept:
            model_name += "_no_intercept"
        return model_name

    def regressor_dimensionality(self, hyperparameters: Optional[ReducedRankRegressionHyperparameters] = None) -> int:
        """Return latent dimensionality implied by reduced-rank hyperparameters.

        Parameters
        ----------
        hyperparameters : Optional[ReducedRankRegressionHyperparameters]
            Hyperparameters to evaluate. If None, uses ``self.hyperparameters``.

        Returns
        -------
        int
            Latent dimensionality for this model configuration.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        return get_regressor_dimensionality_from_hyperparameters(hyperparameters=hyperparameters)

    def _optimize_golden(
        self,
        session: B2Session,
        spks_type: SpksTypes,
        train_split: "SplitName",
        validation_split: "SplitName",
        nan_safe: bool = False,
    ) -> tuple[dict, float, pd.DataFrame]:
        """Optimize hyperparameters using golden section search.

        First optimizes alpha (with rank=200 fixed), then optimizes rank (with best alpha).

        Parameters
        ----------
        session : B2Session
            The session to optimize the hyperparameters for.
        spks_type : SpksTypes
            The type of spike data to use for the population.
        train_split : "SplitName"
            The split to use for the training.
        validation_split : "SplitName"
            The split to use for the validation.
        nan_safe: bool = False
            If True, will check for NaN values in predictions and metrics and raise errors if found.

        Returns
        -------
        best_params : dict
            The best hyperparameters for the model.
        best_score : float
            The best score for the model.
        results_df : pd.DataFrame
            A DataFrame with all the results from the golden section search optimization.
        """
        # Get data to determine max rank
        source_data, target_data, _ = self.get_session_data(session, spks_type, train_split)
        max_rank = int(min(*source_data.shape, *target_data.shape))

        results: list[dict] = []

        # Step 1: Optimize alpha with rank=200 fixed
        def evaluate_alpha(alpha: float) -> float:
            """Evaluate alpha with rank=200."""
            hyperparameters = ReducedRankRegressionHyperparameters(alpha=alpha, rank=200)
            trained_model = self.train(
                session,
                spks_type=spks_type,
                split=train_split,
                hyperparameters=hyperparameters,
            )
            score = self.score(
                session,
                trained_model,
                spks_type=spks_type,
                split=validation_split,
                hyperparameters=hyperparameters,
                nan_safe=nan_safe,
            )
            if np.isnan(score):
                score = float("inf")

            # Record result
            result = {"alpha": alpha, "rank": 200, "score": score}
            results.append(result)

            return score

        best_alpha, best_alpha_score, alpha_history = golden_section_search(
            func=evaluate_alpha,
            a=1e-2,
            b=1e6,
            tolerance_param=1e-2,
            tolerance_score=1e-3,
            max_iterations=25,
            minimize=True,
            logspace=True,
        )

        # Step 2: Optimize rank with best alpha
        def evaluate_rank(rank: float) -> float:
            """Evaluate rank with best alpha."""
            rank = int(rank)
            hyperparameters = ReducedRankRegressionHyperparameters(alpha=best_alpha, rank=rank)
            trained_model = self.train(
                session,
                spks_type=spks_type,
                split=train_split,
                hyperparameters=hyperparameters,
            )
            score = self.score(
                session,
                trained_model,
                spks_type=spks_type,
                split=validation_split,
                hyperparameters=hyperparameters,
                nan_safe=nan_safe,
            )
            if np.isnan(score):
                score = float("inf")

            # Record result
            result = {"alpha": best_alpha, "rank": rank, "score": score}
            results.append(result)

            return score

        best_rank, best_rank_score, rank_history = golden_section_search(
            func=evaluate_rank,
            a=1.0,
            b=float(max_rank),
            tolerance_param=1.0,  # Tolerance of 1 rank unit
            tolerance_score=1e-3,
            max_iterations=25,
            minimize=True,
            logspace=False,
        )
        best_rank = int(best_rank)

        # Find overall best from all results
        best_result = min(results, key=lambda x: x["score"])
        best_params = {"alpha": best_result["alpha"], "rank": best_result["rank"]}
        best_score = best_result["score"]

        results_df = pd.DataFrame(results)
        return best_params, best_score, results_df
