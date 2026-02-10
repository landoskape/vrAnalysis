from typing import Optional, Union, TYPE_CHECKING, Type
from copy import copy
import numpy as np
from scipy.stats import norm, linregress
from sklearn.decomposition import randomized_svd
import torch
import pandas as pd
from vrAnalysis.helpers import edge2center, vectorCorrelation
from vrAnalysis.helpers.optimization import golden_section_search
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_placefield, get_placefield_prediction, Placefield, FrameBehavior
from dimilibi import RidgeRegression, ReducedRankRegression
from .base import RegressionModel, ActivityParameters
from .hyperparameters import PlaceFieldHyperparameters, RBFPosHyperparameters, ReducedRankRegressionHyperparameters

if TYPE_CHECKING:
    from ..registry import PopulationRegistry, SplitName


class PlaceFieldModel(RegressionModel[PlaceFieldHyperparameters]):
    def __init__(
        self,
        registry: "PopulationRegistry",
        internal: bool = False,
        gain: bool = False,
        vector_gain: bool = False,
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
            U = randomized_svd(full_deviation, n_components=1, n_iter=100)[0]

            num_source = source_data.shape[0]
            arousal_coefficients_source = U[:num_source, 0]
            arousal_coefficients_target = U[num_source:, 0]
            arousal_coefficients = (arousal_coefficients_target, arousal_coefficients_source)

        if self.gain and self.vector_gain:
            return train_target_placefield, train_source_placefield, arousal_coefficients

        if self.internal or self.gain:
            return train_target_placefield, train_source_placefield

        return train_target_placefield

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
            error = torch.mean((source_data[None, None] - torch.tensor(source_placefield.placefield[..., None])) ** 2, dim=2)
            error = torch.nan_to_num(error, nan=float("inf"))
            argmin = error.view(-1, error.size(-1)).argmin(dim=0)
            idx_env = argmin // error.size(1)
            idx_pos = argmin % error.size(1)

            dist_centers = edge2center(source_placefield.dist_edges)
            frame_behavior.position = dist_centers[idx_pos.numpy()]
            frame_behavior.environment = source_placefield.environment[idx_env.numpy()]

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
                arousal_estimate = arousal_coefficients_source @ source_deviation
                arousal_activity_target = np.reshape(arousal_coefficients_target, (-1, 1)) * np.reshape(arousal_estimate, (1, -1))

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


class RBFPosModel(RegressionModel[RBFPosHyperparameters]):
    def __init__(
        self,
        registry: "PopulationRegistry",
        split_train: bool = True,
        predict_latents: bool = True,
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
        self.fit_intercept = True
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
            basis_for_encoder = self.make_position_basis(session, frame_behavior_encoder, hyperparameters)
            encoder = RidgeRegression(alpha=hyperparameters.alpha_encoder, fit_intercept=self.fit_intercept)
            encoder = encoder.fit(source_data_encoder.T, basis_for_encoder)

        # Train the decoder model to predict the target neuron activity from the position basis
        _, target_data_decoder, frame_behavior_decoder = self.get_session_data(session, spks_type, decoder_split)
        basis_for_decoder = self.make_position_basis(session, frame_behavior_decoder, hyperparameters)
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
        # Get source activity and frame_behavior for the requested split
        source_data, _, frame_behavior = self.get_session_data(session, spks_type, split)

        # Make the position basis... of the "true" position
        position_basis = self.make_position_basis(session, frame_behavior, hyperparameters)
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

    def make_position_basis(
        self,
        session: B2Session,
        frame_behavior: FrameBehavior,
        hyperparameters: Optional[RBFPosHyperparameters] = None,
    ) -> torch.Tensor:
        """Make the position basis for the RBF(Pos) model.

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
        hyperparameters : Optional[RBFPosHyperparameters]
            The hyperparameters to use for the RBF(Pos) model. If None, uses the default hyperparameters for the model.

        Returns
        -------
        basis : torch.Tensor
            The position basis for the RBF(Pos) model of shape (num_timepoints, num_basis * num_environments).
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

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
        return model_name


class ReducedRankRegressionModel(RegressionModel[ReducedRankRegressionHyperparameters]):
    def __init__(
        self,
        registry: "PopulationRegistry",
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
        self.fit_intercept = True
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
        return "rrr"

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
