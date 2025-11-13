from typing import Optional, Union, Literal, TYPE_CHECKING
import numpy as np
import torch
from vrAnalysis.helpers import edge2center
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_placefield, get_placefield_prediction, Placefield
from dimilibi import ReducedRankRegression
from .base import RegressionModel
from .hyperparameters import PlaceFieldHyperparameters, ReducedRankRegressionHyperparameters

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
        center: bool = False,
        scale: bool = True,
        scale_type: Optional[str] = "preserve",
        presplit: bool = True,
        autosave: bool = True,
    ):
        super().__init__(
            registry,
            center=center,
            scale=scale,
            scale_type=scale_type,
            presplit=presplit,
            autosave=autosave,
        )
        self.internal = internal
        self.gain = gain
        self.hyperparameters = hyperparameters

    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional[Literal["train", "train0", "train1", "validation", "test", "full"]] = "train",
        hyperparameters: Optional[PlaceFieldHyperparameters] = None,
    ) -> Union[Placefield, tuple[Placefield, Placefield]]:
        """Train the model by predicting the place field activity on train timepoints.

        Parameters
        ----------
        session : B2Session
            The session to train the placefield model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split: Optional[Literal["train", "train0", "train1", "validation", "test", "full"]]
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

        # Get the population and frame behavior
        population, frame_behavior = self.registry.get_population(session, spks_type)

        # Filter the frame behavior for the training timepoints
        # Note: we use within_idx_samples=False because frame_behavior is already filtered by idx_samples!
        idx = np.array(population.get_split_times(self.registry.time_split[split], within_idx_samples=False))
        frame_behavior = frame_behavior.filter(idx)

        # Get the target spks for estimating placefields
        target_spks = population.get_split_data(
            self.registry.time_split[split],
            center=self.center,
            scale=self.scale,
            scale_type=self.scale_type,
            pre_split=self.presplit,
        )[1].T.numpy()

        # Then we can get the placefields
        train_target_placefield = get_placefield(
            target_spks,
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
            source_spks = population.get_split_data(
                self.registry.time_split[split],
                center=self.center,
                scale=self.scale,
                scale_type=self.scale_type,
                pre_split=self.presplit,
            )[0].T.numpy()
            train_source_placefield = get_placefield(
                source_spks,
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
        split: Optional[Literal["train", "train0", "train1", "validation", "test", "full"]] = "test",
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
        split : Optional[Literal["train", "train0", "train1", "validation", "test", "full"]]
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

        # Get the test indices for the population and filter the frame behavior
        population, frame_behavior = self.registry.get_population(session, spks_type)
        idx = np.array(population.get_split_times(self.registry.time_split[split], within_idx_samples=False))

        # Filter the frame behavior for the test timepoints
        frame_behavior = frame_behavior.filter(idx)

        # Get the source data to predict the internal position estimates
        if self.internal:
            source_data = population.get_split_data(
                self.registry.time_split[split],
                center=self.center,
                scale=self.scale,
                scale_type=self.scale_type,
                pre_split=self.presplit,
            )[0]
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
            source_data = population.get_split_data(
                self.registry.time_split[split],
                center=self.center,
                scale=self.scale,
                scale_type=self.scale_type,
                pre_split=self.presplit,
            )[0].to(device)

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


class ReducedRankRegressionModel(RegressionModel[ReducedRankRegressionHyperparameters]):
    def __init__(
        self,
        registry: "PopulationRegistry",
        hyperparameters: ReducedRankRegressionHyperparameters = ReducedRankRegressionHyperparameters(),
        center: bool = False,
        scale: bool = True,
        scale_type: Optional[str] = "preserve",
        presplit: bool = True,
        autosave: bool = True,
    ):
        super().__init__(
            registry,
            center=center,
            scale=scale,
            scale_type=scale_type,
            presplit=presplit,
            autosave=autosave,
        )
        self.hyperparameters = hyperparameters

    def train(
        self,
        session: B2Session,
        spks_type: Optional[SpksTypes] = None,
        split: Optional[Literal["train", "train0", "train1", "validation", "test", "full"]] = "train",
        hyperparameters: Optional[ReducedRankRegressionHyperparameters] = None,
    ) -> ReducedRankRegression:
        """Train the model by fitting the reduced rank regression model to the training data.

        Parameters
        ----------
        session : B2Session
            The session to train the reduced rank regression model on.
        spks_type : Optional[SpksTypes]
            The type of spike data to use for the population. If None, uses the spks_type from the session provided as input.
        split: Optional[Literal["train", "train0", "train1", "validation", "test", "full"]]
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

        # Get the population -- we don't need frame behavior for reduced rank regression
        population = self.registry.get_population(session, spks_type)[0]

        train_source, train_target = population.get_split_data(
            self.registry.time_split[split],
            center=self.center,
            scale=self.scale,
            scale_type=self.scale_type,
            pre_split=self.presplit,
        )

        rrr_model = ReducedRankRegression(alpha=hyperparameters.alpha, fit_intercept=True)
        return rrr_model.fit(train_source.T, train_target.T)

    def predict(
        self,
        session: B2Session,
        rrr_model: ReducedRankRegression,
        spks_type: Optional[SpksTypes] = None,
        split: Optional[Literal["train", "train0", "train1", "validation", "test", "full"]] = "test",
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
        split : Optional[Literal["train", "train0", "train1", "validation", "test", "full"]]
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
        population = self.registry.get_population(session, spks_type)[0]
        train_source = population.get_split_data(
            self.registry.time_split[split],
            center=self.center,
            scale=self.scale,
            scale_type=self.scale_type,
            pre_split=self.presplit,
        )[0]
        extras = {}
        return rrr_model.predict(train_source.T, rank=hyperparameters.rank).T, extras

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
