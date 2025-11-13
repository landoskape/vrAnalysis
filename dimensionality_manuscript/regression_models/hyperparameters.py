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
        """Get hyperparameters from Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object to suggest hyperparameters from.

        Returns
        -------
        params : dict[str, Any]
            Dictionary of hyperparameter values (not distributions) suggested by Optuna.
            Keys are hyperparameter names, values are the suggested parameter values.
            Example: {"num_bins": 200, "smooth_width": 2.5}
        """
        # Use categorical for num_bins (discrete choices)
        num_bins = trial.suggest_categorical("num_bins", (100, 40, 25, 10))

        # For smooth_width, first decide if smoothing should be used
        use_smoothing = trial.suggest_categorical("use_smoothing", (True, False))
        if use_smoothing:
            smooth_width = trial.suggest_float("smooth_width", 1.0, 50.0, log=True)
        else:
            smooth_width = None

        return {
            "num_bins": num_bins,
            "smooth_width": smooth_width,
        }


@dataclass(frozen=True)
class ReducedRankRegressionHyperparameters(HyperparametersBase):
    """Hyperparameters for the ReducedRankRegressionModel.

    Parameters
    ----------
    rank : int, default=200
        The rank of the model.
    alpha : float, default=0.0
        The ridge regularization parameter.
    """

    rank: int = field(default=50, init=True, repr=True)
    alpha: float = field(default=1e7, init=True, repr=True)

    @classmethod
    def get_search_space(cls) -> dict[str, tuple[Any, ...]]:
        """Get the search space for grid search.

        Returns
        -------
        search_space : dict[str, tuple[Any, ...]]
            Dictionary with hyperparameter names as keys and tuples of possible values.
        """
        return {
            "rank": (1, 2, 3, 5, 8, 15, 50, 100, 200),
            "alpha": tuple(torch.logspace(1, 9, 9).tolist()),
        }

    @classmethod
    def get_optuna_space(cls, trial: "Trial") -> dict[str, Any]:
        """Get hyperparameters from Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object to suggest hyperparameters from.

        Returns
        -------
        params : dict[str, Any]
            Dictionary of hyperparameter values (not distributions) suggested by Optuna.
            Keys are hyperparameter names, values are the suggested parameter values.
            Example: {"rank": 200, "alpha": 1e5}
        """
        rank = trial.suggest_categorical("rank", (1, 2, 3, 5, 8, 15, 50, 100, 200))
        alpha = trial.suggest_float("alpha", 1e1, 1e9, log=True)

        return {
            "rank": rank,
            "alpha": alpha,
        }
