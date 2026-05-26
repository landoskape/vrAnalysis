from typing import Optional, TYPE_CHECKING, Any
from dataclasses import dataclass, field
import torch
from .base import HyperparametersBase

if TYPE_CHECKING:
    from optuna import Trial


@dataclass(frozen=True)
class PlaceFieldHyperparameters(HyperparametersBase):
    """Hyperparameters for the PlaceFieldModel.

    Parameters
    ----------
    num_bins : int, default=200
        Number of spatial bins for the place field. More bins provide higher
        spatial resolution but may require more data.
    smooth_width : float | None, default=1.0
        Width of the Gaussian smoothing kernel applied to place fields (in spatial units).
        If None, no smoothing is applied. Larger values result in smoother place fields.
    """

    num_bins: int = field(default=100, init=True, repr=True)
    smooth_width: Optional[float] = field(default=1.0, init=True, repr=True)

    @classmethod
    def get_search_space(cls) -> dict[str, tuple[Any, ...]]:
        """Get the search space for grid search.

        Returns
        -------
        search_space : dict[str, tuple[Any, ...]]
            Dictionary with hyperparameter names as keys and tuples of possible values.
        """
        return {
            "num_bins": (100, 40, 25, 10),
            "smooth_width": (None, 1.0, 5.0, 10.0, 20.0, 50.0),
        }

    @classmethod
    def get_optuna_space(cls, trial: "Trial") -> dict[str, Any]:
        """Get the Optuna search space for the PlaceFieldModel.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object to suggest hyperparameters from.

        Returns
        -------
        search_space : dict[str, Any]
            Dictionary with hyperparameter names as keys and the suggested values.
        """
        use_smoothing = trial.suggest_categorical("use_smoothing", [True, False])
        if use_smoothing:
            smooth_width = trial.suggest_float("smooth_width", 1.0, 50.0, log=True)
        else:
            smooth_width = None

        return {
            "num_bins": trial.suggest_int("num_bins", 10, 100, log=True),
            "smooth_width": smooth_width,
        }


@dataclass(frozen=True)
class RBFPosHyperparameters(HyperparametersBase):
    """Hyperparameters for the RBFPosModel.

    Parameters
    ----------
    num_basis : int, default=10
        Number of position basis functions for the RBF(Pos) model.
    basis_width : float, default=10.0
        Width of the Gaussian basis functions for the RBF(Pos) model.
    alpha_encoder : float, default=1e0
        The ridge regularization parameter for the encoder.
    alpha_decoder : float, default=1e0
        The ridge regularization parameter for the decoder.
    """

    num_basis: int = field(default=100, init=True, repr=True)
    basis_width: float = field(default=5.0, init=True, repr=True)
    alpha_encoder: float = field(default=1e0, init=True, repr=True)
    alpha_decoder: float = field(default=1e0, init=True, repr=True)

    @classmethod
    def get_search_space(cls) -> dict[str, tuple[Any, ...]]:
        """Get the search space for grid search.

        Returns
        -------
        search_space : dict[str, tuple[Any, ...]]
            Dictionary with hyperparameter names as keys and tuples of possible values.
        """
        return {
            "num_basis": (100, 40, 25, 10),
            "basis_width": (5.0, 15.0, 40.0),
            "alpha_encoder": tuple(torch.logspace(-3, 10, 13).tolist()),
            "alpha_decoder": tuple(torch.logspace(-3, 10, 13).tolist()),
        }

    @classmethod
    def get_optuna_space(cls, trial: "Trial") -> dict[str, Any]:
        """Get the Optuna search space for the RBFPosModel.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object to suggest hyperparameters from.

        Returns
        -------
        search_space : dict[str, Any]
            Dictionary with hyperparameter names as keys and the suggested values.
        """
        return {
            "num_basis": trial.suggest_int("num_basis", 10, 100, log=True),
            "basis_width": trial.suggest_float("basis_width", 1.0, 50.0, log=True),
            "alpha_encoder": trial.suggest_float("alpha_encoder", 1e-3, 1e10, log=True),
            "alpha_decoder": trial.suggest_float("alpha_decoder", 1e-3, 1e10, log=True),
        }

    @classmethod
    def _process_params(cls, params: dict) -> dict:
        """Process the parameters from the dictionary.

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameter values.
        """
        if "alpha" in params:
            if "alpha_encoder" not in params:
                params["alpha_encoder"] = params["alpha"]
            if "alpha_decoder" not in params:
                params["alpha_decoder"] = params["alpha"]
            params.pop("alpha")
        return params


@dataclass(frozen=True)
class FullRegressorHyperparameters(HyperparametersBase):
    """Hyperparameters for the FullRegressorModel.

    Parameters
    ----------
    num_basis : int, default=10
        Number of position basis functions for the full regressor model.
    basis_width : float, default=10.0
        Width of the Gaussian basis functions for the full regressor model.
    speed_num_basis : int, default=10
        Number of speed basis functions for the full regressor model.
    reward_num_basis_lags : int, default=11
        Number of lags on reward basis functions for the full regressor model.
        (Note - it'll be symmetric for reward delivery, so num_basis=num_lags*2 + 1)
        but it'll be only "responsive" for reward omission, so it'll be num_basis=num_lags + 1.
    reward_basis_width : float, default=5.0
        Width of the Gaussian basis functions for the reward regressor in the full regressor model.
    alpha_encoder : float, default=1e0
        The ridge regularization parameter for the encoder.
    alpha_decoder : float, default=1e0
        The ridge regularization parameter for the decoder.
    """

    num_basis: int = field(default=100, init=True, repr=True)
    basis_width: float = field(default=5.0, init=True, repr=True)
    speed_num_basis: int = field(default=10, init=True, repr=True)
    reward_num_basis_lags: int = field(default=11, init=True, repr=True)
    reward_basis_width: int = field(default=5, init=True, repr=True)
    alpha_encoder: float = field(default=1e0, init=True, repr=True)
    alpha_decoder: float = field(default=1e0, init=True, repr=True)

    @classmethod
    def get_search_space(cls) -> dict[str, tuple[Any, ...]]:
        """Get the search space for grid search.

        Returns
        -------
        search_space : dict[str, tuple[Any, ...]]
            Dictionary with hyperparameter names as keys and tuples of possible values.
        """
        return {
            "num_basis": (100, 40, 25, 10),
            "basis_width": (5.0, 15.0, 40.0),
            "speed_num_basis": (10, 5, 3),
            "reward_num_basis_lags": (11, 5, 3),
            "reward_basis_width": (5, 10),
            "alpha_encoder": tuple(torch.logspace(-3, 10, 13).tolist()),
            "alpha_decoder": tuple(torch.logspace(-3, 10, 13).tolist()),
        }

    @classmethod
    def get_optuna_space(cls, trial: "Trial") -> dict[str, Any]:
        """Get the Optuna search space for the RBFPosModel.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object to suggest hyperparameters from.

        Returns
        -------
        search_space : dict[str, Any]
            Dictionary with hyperparameter names as keys and the suggested values.
        """
        max_frames_reward_support = 35  # prevent a reward regressor from extending too far from a reward,
        reward_num_basis_lags = trial.suggest_int("reward_num_basis_lags", 1, 15)
        max_reward_width = max_frames_reward_support / (reward_num_basis_lags + 1)
        reward_basis_width = trial.suggest_float("reward_basis_width", 1.0, max_reward_width)
        return {
            "num_basis": trial.suggest_int("num_basis", 10, 100, log=True),
            "basis_width": trial.suggest_float("basis_width", 1.0, 50.0, log=True),
            "speed_num_basis": trial.suggest_int("speed_num_basis", 2, 20),
            "reward_num_basis_lags": reward_num_basis_lags,
            "reward_basis_width": reward_basis_width,
            "alpha_encoder": trial.suggest_float("alpha_encoder", 1e-3, 1e10, log=True),
            "alpha_decoder": trial.suggest_float("alpha_decoder", 1e-3, 1e10, log=True),
        }

    @classmethod
    def _process_params(cls, params: dict) -> dict:
        """Process the parameters from the dictionary.

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameter values.
        """
        if "alpha" in params:
            if "alpha_encoder" not in params:
                params["alpha_encoder"] = params["alpha"]
            if "alpha_decoder" not in params:
                params["alpha_decoder"] = params["alpha"]
            params.pop("alpha")
        return params


@dataclass(frozen=True)
class ReducedRankRegressionHyperparameters(HyperparametersBase):
    """Hyperparameters for the ReducedRankRegressionModel.

    Parameters
    ----------
    rank : int, default=100
        The rank of the model.
    alpha : float, default=1e6
        The ridge regularization parameter.
    """

    rank: int = field(default=50, init=True, repr=True)
    alpha: float = field(default=1e2, init=True, repr=True)
    independent_optimization: bool = field(default=True, init=False, repr=False)

    @classmethod
    def get_search_space(cls) -> dict[str, tuple[Any, ...]]:
        """Get the search space for grid search.

        Returns
        -------
        search_space : dict[str, tuple[Any, ...]]
            Dictionary with hyperparameter names as keys and tuples of possible values.
        """
        training = {
            "alpha": tuple(torch.logspace(0, 4, 9).tolist()),
        }
        prediction = {
            "rank": (1, 2, 3, 5, 8, 15, 50, 100),
        }
        return {"training": training, "prediction": prediction}

    @classmethod
    def get_optuna_space(cls, trial: "Trial") -> dict[str, Any]:
        """Get the Optuna search space for the ReducedRankRegressionModel.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object to suggest hyperparameters from.

        Returns
        -------
        search_space : dict[str, Any]
            Dictionary with hyperparameter names as keys and the suggested values.
        """
        training = {
            "alpha": trial.suggest_float("alpha", 1e-2, 1e10, log=True),
        }
        prediction = {
            "rank": (1, 2, 3, 5, 8, 15, 50, 100),
        }
        return {"training": training, "prediction": prediction}
