from typing import Optional
from copy import copy
import torch
from .metrics import measure_r2


class RidgeRegression:
    def __init__(self, alpha: Optional[float] = 0.0, fit_intercept: Optional[bool] = False):
        """
        Initialize the RidgeRegression model.

        Parameters
        ----------
        alpha : Optional[float]
            The ridge regularization parameter (default is 0.0).
        fit_intercept : Optional[bool]
            If True, will fit an intercept term to the model (default is False).
        """
        assert alpha >= 0, "Ridge regularization parameter must be non-negative."
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the RidgeRegression model to the provided data.

        Parameters
        ----------
        X : torch.Tensor
            The input data (num_samples, num_features).
        y : torch.Tensor
            The target data (num_samples, num_targets).

        Returns
        -------
        self : object
            The RidgeRegression object with the fitted model.
        """
        if self.fit_intercept:
            X = self._add_intercept(X)

        # store all of these matrices for easy testing of prediction with different ranks
        self._beta_ols = self._solve_ols_ridge(X, y, self.alpha)

        # return self for standard usage
        return self

    def predict(self, X: torch.Tensor, nonnegative: Optional[bool] = False) -> torch.Tensor:
        """
        Predict the target data using the RidgeRegression model.

        Parameters
        ----------
        X : torch.Tensor
            The input data (num_samples, num_features).
        nonnegative : Optional[bool]
            If True, will apply a ReLU to the prediction (default is False).

        Returns
        -------
        y_pred : torch.Tensor
            The predicted target data (num_samples, num_targets).
        """
        if self.fit_intercept:
            X = self._add_intercept(X)

        prediction = X @ self._beta_ols

        if nonnegative:
            prediction = torch.relu(prediction)

        return prediction

    def score(self, X: torch.Tensor, y: torch.Tensor, nonnegative: Optional[bool] = False) -> torch.Tensor:
        """
        Score the RidgeRegression model on the provided data.

        Parameters
        ----------
        X : torch.Tensor
            The input data (num_samples, num_features).
        y : torch.Tensor
            The target data (num_samples, num_targets).
        nonnegative : Optional[bool]
            If True, will apply a ReLU to the prediction (default is False).

        Returns
        -------
        r2 : torch.Tensor
            The coefficient of determination (R^2) for the model.
        """
        y_pred = self.predict(X, nonnegative=nonnegative)
        return measure_r2(y_pred.T, y.T)

    def _add_intercept(self, X: torch.Tensor) -> torch.Tensor:
        """
        Add an intercept term to the input data.

        Parameters
        ----------
        X : torch.Tensor
            The input data (num_samples, num_features).

        Returns
        -------
        X : torch.Tensor
            The input data with an intercept term added (num_samples, num_features + 1).
        """
        return torch.cat([X, torch.ones(X.size(0), 1, device=X.device)], dim=1)

    def _solve_ols_ridge(self, X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Solve the ordinary least squares (OLS) problem with ridge regularization.

        Mathematically, this is equivalent to solving the following problem:
        beta_ols = (X^T X + alpha I)^{-1} X^T y.

        First, we decompose X with the singular value decomposition (SVD):
        X = U S V.T

        Then we have:
        beta_ols = (V S.T U.T U S V.T + alpha I)^{-1} V S.T U.T y
        beta_ols = (V S.T S V.T + alpha I)^{-1} V S.T U.T y      # U.T U = I
        beta_ols = V (S^T S + alpha I)^{-1} S^T U.T y            # V.T V = I

        Note: this method is adapted directly from the scikit-learn implementation.

        Parameters
        ----------
        X : torch.Tensor
            The input data (num_samples, num_features).
        y : torch.Tensor
            The target data (num_samples, num_targets).
        alpha : float
            The ridge regularization parameter.

        Returns
        -------
        beta_ols : torch.Tensor
            The OLS solution with ridge regularization.
        """
        U, s, Vt = torch.linalg.svd(X, full_matrices=False)
        idx = s > 1e-15  # same default value as scipy.linalg.pinv
        s_nnz = s[idx]
        d = torch.zeros((s.size(0)), dtype=X.dtype, device=X.device)
        d[idx] = s_nnz / (s_nnz**2 + alpha)
        return Vt.T @ torch.diag(d) @ U.T @ y


class ReducedRankRegression(RidgeRegression):
    """
    Reduced rank regression model for fitting, predicting, and scoring data.
    """

    def __init__(self, rank: Optional[int] = None, alpha: Optional[float] = 0.0, fit_intercept: Optional[bool] = False):
        """
        Initialize the ReducedRankRegression model.

        Parameters
        ----------
        rank : Optional[int]
            The rank of the model (default is None).
        alpha : Optional[float]
            The ridge regularization parameter (default is 0.0).
        fit_intercept : Optional[bool]
            If True, will fit an intercept term to the model (default is False).
        """
        assert alpha >= 0, "Ridge regularization parameter must be non-negative."
        self.rank = rank
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the ReducedRankRegression model to the provided data.

        Parameters
        ----------
        X : torch.Tensor
            The input data (num_samples, num_features).
        y : torch.Tensor
            The target data (num_samples, num_targets).

        Returns
        -------
        self : object
            The ReducedRankRegression object with the fitted model.
        """
        # Perform the OLS regression with ridge regularization
        super().fit(X, y)

        if self.fit_intercept:
            X = self._add_intercept(X)

        # Store the rank restraint matrix and coefficients
        num_features = X.size(1)
        num_targets = y.size(1)
        num_samples = X.size(0)
        self.max_rank = min(num_features, num_targets, num_samples)

        if self.rank is None:
            self.rank = copy(self.max_rank)
        else:
            assert self.rank <= self.max_rank, "Rank must be less than or equal to the number of features, targets, and samples."

        self._Xbeta_V = torch.svd(X @ self._beta_ols, some=True, compute_uv=True).V
        self._rank_restraint = self._Xbeta_V[:, : self.rank] @ self._Xbeta_V[:, : self.rank].T
        self._beta_rrr = self._make_coefficients(self.rank)

        return self

    def predict(self, X: torch.Tensor, rank: Optional[int] = None, nonnegative: Optional[bool] = False) -> torch.Tensor:
        """
        Predict the target data using the ReducedRankRegression model.

        Parameters
        ----------
        X : torch.Tensor
            The input data (num_samples, num_features).
        rank : Optional[int]
            The rank of the model (default is None).
        nonnegative : Optional[bool]
            If True, will apply a ReLU to the prediction (default is False).

        Returns
        -------
        y_pred : torch.Tensor
            The predicted target data (num_samples, num_targets).
        """
        if self.fit_intercept:
            X = self._add_intercept(X)

        beta_rrr = self._make_coefficients(rank or self.rank)
        prediction = X @ beta_rrr

        if nonnegative:
            prediction = torch.relu(prediction)

        return prediction

    def score(self, X: torch.Tensor, y: torch.Tensor, rank: Optional[int] = None, nonnegative: Optional[bool] = False) -> torch.Tensor:
        """
        Score the ReducedRankRegression model on the provided data.

        Parameters
        ----------
        X : torch.Tensor
            The input data (num_samples, num_features).
        y : torch.Tensor
            The target data (num_samples, num_targets).
        rank : Optional[int]
            The rank of the model (default is None, which will use the originally registered rank).
        nonnegative : Optional[bool]
            If True, will apply a ReLU to the prediction (default is False).

        Returns
        -------
        r2 : torch.Tensor
            The coefficient of determination (R^2) for the model.
        """
        y_pred = self.predict(X, rank=rank, nonnegative=nonnegative)
        return measure_r2(y_pred.T, y.T)

    def predict_latent(self, X: torch.Tensor, rank: Optional[int] = None) -> torch.Tensor:
        """
        Predict the latent variables using the ReducedRankRegression model.

        The coefficient matrix for ReducedRankRegression is calculated as:
        beta_rrr = beta_ols @ Xbeta_V[:, :rank] @ Xbeta_V[:, :rank].T

        So we can interpret beta_ols @ Xbeta_V[:, :rank] as "encoder" coefficients and
        interpret Xbeta_V[:, :rank].T as "decoder" coefficients such that the latent
        variables of the prediction are given by: X @ beta_ols @ Xbeta_V[:, :rank].

        Parameters
        ----------
        X : torch.Tensor
            The input data (num_samples, num_features).
        rank : Optional[int]
            The rank of the model (default is None).

        Returns
        -------
        latent : torch.Tensor
            The predicted latent variables (num_samples, rank).
        """
        if self.fit_intercept:
            X = self._add_intercept(X)

        encoder_coefficients = self._beta_ols @ self._Xbeta_V[:, : rank or self.rank]
        return X @ encoder_coefficients

    def _make_coefficients(self, rank) -> torch.Tensor:
        """
        Make the coefficients for the ReducedRankRegression model.

        Will use stored coefficients if the requested rank is equal to the stored rank,
        otherwise will attempt to make new coefficients with the requested rank.

        Returns
        -------
        beta_rrr : torch.Tensor
            The coefficients for the ReducedRankRegression model.
        """
        # update rank restraint matrix if required
        if rank != self.rank:
            assert rank <= self.max_rank, "Rank must be less than or equal to the number of features, targets, and samples."
            _rank_restraint = self._Xbeta_V[:, :rank] @ self._Xbeta_V[:, :rank].T
        else:
            _rank_restraint = self._rank_restraint

        # calculate the coefficients
        beta_rrr = self._beta_ols @ _rank_restraint
        return beta_rrr
