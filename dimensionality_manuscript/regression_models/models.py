from typing import Optional, Union, Literal, TYPE_CHECKING
import numpy as np
from scipy.stats import norm
import torch
from vrAnalysis.helpers import edge2center
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_placefield, get_placefield_prediction, Placefield, FrameBehavior
from dimilibi import ReducedRankRegression
from .base import RegressionModel, SplitName, ActivityParameters
from .hyperparameters import PlaceFieldHyperparameters, RBFPosHyperparameters, ReducedRankRegressionHyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if TYPE_CHECKING:
    from .registry import PopulationRegistry


class PlaceFieldModel(RegressionModel[PlaceFieldHyperparameters]):
    def __init__(
        self,
        registry: "PopulationRegistry",
        internal: bool = False,
        gain: bool = False,
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
        self.hyperparameters = hyperparameters

    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional[SplitName] = "train",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
    ) -> Union[Placefield, tuple[Placefield, Placefield]]:
        """Train the model by predicting the place field activity on train timepoints.

        Parameters
        ----------
        session : B2Session
            The session to train the placefield model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split: Optional[SplitName]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
        hyperparameters : Optional[PlaceFieldHyperparameters]
            The hyperparameters to use for the placefield model. If None, uses the default hyperparameters for the model.

        Returns
        -------
        Placefield or tuple[Placefield, Placefield]
            - When internal=False, returns a single Placefield object corresponding to the target cells.
            - When internal=True, returns a tuple of Placefield objects corresponding to the target and source cells.
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
            return train_target_placefield, train_source_placefield

        return train_target_placefield

    def predict(
        self,
        session: B2Session,
        coefficients: Union[Placefield, tuple[Placefield, Placefield]],
        spks_type: Optional[SpksTypes] = None,
        split: Optional[SplitName] = "test",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
    ) -> tuple[np.ndarray, dict]:
        """Predict the target place field activity for a session.

        Parameters
        ----------
        session : B2Session
            The session to predict the target place field activity for.
        coefficients : Union[Placefield, tuple[Placefield, Placefield]]
            The "coefficients" for making a prediction, in the form of Placefield objects.
            If internal=False, coefficients should be a single Placefield object corresponding to the target cells.
            If internal=True, coefficients should be a tuple of Placefield objects corresponding to the target and source cells.
            Either way it is the output of self.train() given the self.internal flag.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session
            provided as input.
        split : Optional[SplitName]
            The split to use for the prediction. If None, uses the split from the session
            provided as input. Default is "test".
        hyperparameters : Optional[PlaceFieldHyperparameters]
            The hyperparameters used for the model. These are not actually used for prediction so the presence of this parameter
            is ignored and only here for consistency with other model types.

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
        if self.internal or self.gain:
            target_placefield, source_placefield = coefficients
        else:
            target_placefield = coefficients

        # Get session data for the requested split
        source_data, _, frame_behavior = self.get_session_data(session, spks_type, split)

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

        if self.gain:
            # If the model has a gain component, we need to fit the a scalar gain value for
            # each sample. To do this, we minimize the MSE loss between the predicted and target
            # data ***for the source neurons*** which were recorded at the same time as the
            # target neurons. We assume that the gain value is the same for the whole brain.
            # This way the estimator is cross-validated by neurons.
            # -----------------------------------------------------
            # First get the prediction for the source neurons, and target source data
            source_prediction = torch.tensor(get_placefield_prediction(source_placefield, frame_behavior)[0].T).to(device)
            source_data = source_data.to(device)

            idx_nan = torch.any(torch.isnan(source_prediction) | torch.isnan(source_data), dim=0)
            if torch.any(idx_nan):
                raise ValueError(f"{np.sum(idx_nan)} / {len(idx_nan)} samples have nan predictions in {session.session_print()}!!!")

            # Our objective function is the MSE loss between the predicted and target data
            def _objective(gain, prediction, target):
                return torch.sum((prediction * gain - target) ** 2)

            # Initialize the gain and optimize it using gradient descent
            gain = torch.ones(source_prediction.size(1), dtype=source_prediction.dtype, requires_grad=True, device=device)
            optimizer = torch.optim.Adam([gain], lr=1e-2)

            # Early stopping setup
            best_loss = float("inf")
            max_epochs = 1000
            patience = 50
            patience_counter = 0
            min_delta = 1e-6
            loss_history = []

            for _ in range(max_epochs):
                optimizer.zero_grad()
                loss = _objective(gain, source_prediction, source_data)
                loss.backward()
                optimizer.step()

                # Early stopping check
                current_loss = loss.item()
                loss_history.append(current_loss)
                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Reset the gain to nan for bad samples
            gain = gain.detach().cpu().numpy()

            if np.any(np.isnan(gain)):
                raise ValueError(f"{np.sum(np.isnan(gain))} / {len(gain)} gains have nan values in {session.session_print()}!!!")

        # Get prediction for the test timepoints
        prediction, extras = get_placefield_prediction(target_placefield, frame_behavior)
        extras["frame_behavior"] = frame_behavior

        if self.gain:
            # Apply the gain to the prediction
            prediction = prediction * gain.reshape(-1, 1)
            extras["gain"] = gain

        if np.any(np.isnan(prediction)):
            raise ValueError(
                f"{np.sum(np.any(np.isnan(prediction), axis=1))} / {len(prediction)} predictions have nan values in {session.session_print()}!!!"
            )

        return prediction.T, extras

    @property
    def _model_hyperparameters(self) -> type[PlaceFieldHyperparameters]:
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
        gain_suffix = "_gain" if self.gain else ""
        model_name = f"{model_type}_placefield_1d{gain_suffix}"
        return model_name


class RBFPosModel(RegressionModel[RBFPosHyperparameters]):
    def __init__(
        self,
        registry: "PopulationRegistry",
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
        self._split_train = True

    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional[SplitName] = "train",
        hyperparameters: Optional[RBFPosHyperparameters] = None,
    ) -> ReducedRankRegression:
        """Train the model by fitting the RBF(Pos) model to the training data.

        Parameters
        ----------
        session : B2Session
            The session to train the RBF(Pos) model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split: Optional[SplitName]
            The split to use for the training. If None, uses the split from the session provided as input. Default is "train".
            When _split_train is True, 'train' is split into 'train0' and 'train1' for the encoder and decoder.
        hyperparameters : Optional[RBFPosHyperparameters]
            The hyperparameters to use for the RBF(Pos) model. If None, uses the default hyperparameters for the model.

        Returns
        -------
        tuple[ReducedRankRegression, ReducedRankRegression]
            The trained encoder and decoder models. The encoder model predicts position basis from activity of the source
            neurons, and the decoder model predicts activity of the target neurons from the position basis.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        # Split the neural data
        if self._split_train and split == "train":
            encoder_split = "train0"
            decoder_split = "train1"
        else:
            encoder_split = split
            decoder_split = split

        # Get source activity and frame_behavior for encoder split
        source_data_encoder, _, frame_behavior_encoder = self.get_session_data(session, spks_type, encoder_split)

        # Get target activity and frame_behavior for decoder split
        _, target_data_decoder, frame_behavior_decoder = self.get_session_data(session, spks_type, decoder_split)

        # Create a position basis and split it for encoder / decoder
        basis_for_encoder = self.make_position_basis(session, frame_behavior_encoder, hyperparameters)
        basis_for_decoder = self.make_position_basis(session, frame_behavior_decoder, hyperparameters)

        # Note that ReducedRankRegression uses full rank unless specified in its predict() method.
        encoder = ReducedRankRegression(alpha=hyperparameters.alpha, fit_intercept=self.fit_intercept)
        decoder = ReducedRankRegression(alpha=hyperparameters.alpha, fit_intercept=self.fit_intercept)
        encoder = encoder.fit(source_data_encoder.T, basis_for_encoder)
        decoder = decoder.fit(basis_for_decoder, target_data_decoder.T)
        return encoder, decoder

    def predict(
        self,
        session: B2Session,
        rbfpos_model: tuple[ReducedRankRegression, ReducedRankRegression],
        spks_type: Optional[SpksTypes] = None,
        split: Optional[SplitName] = "test",
        hyperparameters: Optional[RBFPosHyperparameters] = None,
    ) -> tuple[np.ndarray, dict]:
        """Predict the target place field activity for a session.

        Parameters
        ----------
        session : B2Session
            The session to predict the target place field activity for.
        rbfpos_model : tuple[ReducedRankRegression, ReducedRankRegression]
            The trained encoder and decoder models. The encoder model predicts position basis from activity of the source
            neurons, and the decoder model predicts activity of the target neurons from the position basis.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session
            provided as input.
        split : Optional[SplitName]
            The split to use for the prediction. If None, uses the split from the session
            provided as input. Default is "test".
        hyperparameters : Optional[RBFPosHyperparameters]
            The hyperparameters used for the model. These are not actually used for prediction so the presence of this parameter
            is ignored and only here for consistency with other model types.

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

        # Predict the position basis with the encoder model
        position_basis_predicted = rbfpos_model[0].predict(source_data.T, nonnegative=self.nonnegative)

        # Predict the activity with the decoder model
        prediction = rbfpos_model[1].predict(position_basis_predicted, nonnegative=self.nonnegative).T

        extras = {
            "position_basis": np.array(position_basis),
            "position_basis_predicted": np.array(position_basis_predicted),
        }
        return np.array(prediction), extras

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
        environments = np.unique(frame_behavior.environment)
        env_idx = torch.tensor(np.searchsorted(environments, frame_behavior.environment))
        basis_by_env = torch.zeros((len(frame_behavior.position), len(environments), hyperparameters.num_basis))

        # Scatter in basis by environment (so it's zero everywhere else)
        env_idx_for_scatter = env_idx.unsqueeze(-1).expand(-1, hyperparameters.num_basis).unsqueeze(1)
        basis_by_env.scatter_(1, env_idx_for_scatter, basis.unsqueeze(1))
        return basis_by_env.view(len(frame_behavior.position), -1)

    @property
    def _model_hyperparameters(self) -> type[RBFPosHyperparameters]:
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
            The model name identifier, either "rbfpos" or "rbfpos_leak" for RBFPosModel.
            The "_leak" suffix indicates that the model was trained without double-cross-validation,
            which allows for non-spatial leakage between activity and position in the training set.
        """
        model_name = "rbfpos"
        if not self._split_train:
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
        split: Optional[SplitName] = "train",
        hyperparameters: Optional[ReducedRankRegressionHyperparameters] = None,
    ) -> ReducedRankRegression:
        """Train the model by fitting the reduced rank regression model to the training data.

        Parameters
        ----------
        session : B2Session
            The session to train the reduced rank regression model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split: Optional[SplitName]
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
        split: Optional[SplitName] = "test",
        hyperparameters: Optional[ReducedRankRegressionHyperparameters] = None,
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
        split : Optional[SplitName]
            The split to use for the prediction. If None, uses the split from the session
            provided as input. Default is "test".
        hyperparameters : Optional[ReducedRankRegressionHyperparameters]
            The hyperparameters used for the model. These are not actually used for prediction so the presence of this parameter
            is ignored and only here for consistency with other model types.

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
        return np.array(prediction), {}

    @property
    def _model_hyperparameters(self) -> type[ReducedRankRegressionHyperparameters]:
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
