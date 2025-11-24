from typing import Optional
from copy import copy
import torch
from .metrics import measure_r2


class RidgeRegression:
    def __init__(self, alpha: Optional[float] = 0.0, fit_intercept: Optional[bool] = False, use_svd: Optional[bool] = False):
        """
        Initialize the RidgeRegression model.

        Parameters
        ----------
        alpha : Optional[float]
            The ridge regularization parameter (default is 0.0).
        fit_intercept : Optional[bool]
            If True, will fit an intercept term to the model (default is False).
        use_svd : Optional[bool]
            If True, will use the singular value decomposition to solve the problem (default is False).
        """
        assert alpha >= 0, "Ridge regularization parameter must be non-negative."
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._use_svd = use_svd

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
        return measure_r2(y_pred, y)

    def to(self, device):
        """
        Move the RidgeRegression model to a device.

        Parameters
        ----------
        device : torch.device or str
            The device to move the model to (e.g., 'cpu', 'cuda', torch.device('cuda:0')).

        Returns
        -------
        self : RidgeRegression
            The RidgeRegression object with all tensors moved to the specified device.
        """
        if hasattr(self, "_beta_ols"):
            self._beta_ols = self._beta_ols.to(device)
        return self

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

        Mathematically, we seek the ridge-regularized coefficient matrix β that solves:

            β = argmin_β ||y - Xβ||^2 + alpha*||β||^2.

        The closed-form solution is:

            β_ols = (X^T X + alpha*I)^{-1} X^T y.

        This method provides two numerically valid ways to compute β_ols:

        ----------------------------------------------------------------------
        **(1) SVD-based computation (numerically robust, typically slower)**

        We factor X using the singular value decomposition (SVD):

            X = U S V^T,

        where:
            U ∈ R^{n x r} contains left singular vectors,
            S ∈ R^{r x r} contains singular values,
            V ∈ R^{p x r} contains right singular vectors,
            r = rank(X).

        Substituting into the analytic ridge solution:

            β_ols = (X^T X + alpha*I)^{-1} X^T y
                   = (V S^2 V^T + alpha*I)^{-1} V S U^T y
                   = V (S^2 + alpha*I)^{-1} S Uᵀ y.

        Since (S^2 + alpha*I) is diagonal, its inverse is computed elementwise as:

            d_i = s_i / (s_i^2 + alpha),

        giving the practical form:

            β_ols = V diag(d) U^T y.

        This approach matches the formulation in scikit-learn and is highly stable,
        especially for ill-conditioned matrices, because it never forms X^T X explicitly.

        ----------------------------------------------------------------------
        **(2) Linear-system solution without SVD (typically much faster)**

        When SVD is not requested, we compute β_ols using linear algebra
        on either feature space or sample space, depending on problem shape.

        **Feature-space formulation** (preferred when n ≥ p):

            β_ols = (X^T X + alpha*I)^{-1} (X^T y).

        This requires forming the Gram matrix:

            G = X^T X ∈ R^{p x p},

        and solving:

            G β_ols = X^T y.

        **Sample-space formulation** (preferred when p > n):

        Using the Woodbury identity:

            (X^T X + alpha*I)^{-1} X^T = X^T (X X^T + alpha*I)^{-1},

        we can compute:

            β_ols = X^T (X X^T + alpha*I)^{-1} y,

        which avoids forming a large p x p system when p is much larger than n.

        Both linear-solve formulations avoid the explicit decomposition of X and
        are substantially faster than the SVD method for large problems, but they
        can accumulate more numerical error because X^T X (or X X^T) is explicitly
        formed and may be less well-conditioned.

        ----------------------------------------------------------------------

        Parameters
        ----------
        X : torch.Tensor
            Input data matrix of shape (num_samples, num_features).
        y : torch.Tensor
            Target data matrix of shape (num_samples, num_targets).
        alpha : float
            Ridge regularization parameter.

        Returns
        -------
        beta_ols : torch.Tensor
            The ridge-regularized OLS solution with shape (num_features, num_targets).
        """
        if self._use_svd:
            U, s, Vt = torch.linalg.svd(X, full_matrices=False)
            idx = s > 1e-15  # same default value as scipy.linalg.pinv
            s_nnz = s[idx]
            d = torch.zeros((s.size(0)), dtype=X.dtype, device=X.device)
            d[idx] = s_nnz / (s_nnz**2 + alpha)
            beta_ols = Vt.T @ torch.diag(d) @ U.T @ y

        else:
            n, p = X.shape

            # Work in feature space if p is not huge compared to n
            if n >= p:
                Xt = X.transpose(0, 1)
                gram = Xt @ X  # (p, p)
                if alpha > 0:
                    gram = gram + alpha * torch.eye(p, device=X.device, dtype=X.dtype)
                rhs = Xt @ y  # (p, q)
                beta_ols = torch.linalg.solve(gram, rhs)  # (p, q)
            else:
                # Work in sample space using Woodbury identity:
                # beta = X^T (X X^T + alpha I)^(-1) y
                XXt = X @ X.transpose(0, 1)  # (n, n)
                if alpha > 0:
                    XXt = XXt + alpha * torch.eye(n, device=X.device, dtype=X.dtype)
                tmp = torch.linalg.solve(XXt, y)  # (n, q)
                beta_ols = X.transpose(0, 1) @ tmp  # (p, q)

        return beta_ols


class ReducedRankRegression(RidgeRegression):
    """
    Reduced rank regression model for fitting, predicting, and scoring data.
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        alpha: Optional[float] = 0.0,
        fit_intercept: Optional[bool] = False,
        use_svd: Optional[bool] = False,
    ):
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
        use_svd : Optional[bool]
            If True, will use the singular value decomposition to solve the problem (default is False).
        """
        assert alpha >= 0, "Ridge regularization parameter must be non-negative."
        self.rank = rank
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._use_svd = use_svd

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the ReducedRankRegression model to the provided data.

        To get the reduced rank model, we apply a rank constraint to solution. The prediction is given by y_hat = X @ β_ols,
        but we want a rank "r" solution, so we estimate the right singular vectors of X @ β_ols.

        We can do this in two ways, either with SVD directly (using the V matrix for a rank constraint), or with
        eigendecomposition of y_hat.T @ y_hat, which is faster. This is preffered and it's the default unless you
        set use_svd=True.

        To do this,

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

        if self._use_svd:
            self._Xbeta_V = torch.svd(X @ self._beta_ols, some=True, compute_uv=True).V

        else:
            Xt = X.transpose(0, 1)
            gram = Xt @ X

            K = self._beta_ols.transpose(0, 1) @ gram @ self._beta_ols
            evals, V = torch.linalg.eigh(K)

            idx = torch.argsort(evals, descending=True)
            self._Xbeta_V = V[:, idx]  # (q, q), plays role of "V" from SVD

        # Store the rank restraint matrix and coefficients
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

    def score(self, X: torch.Tensor, y: torch.Tensor, rank: Optional[int] = None, nonnegative: Optional[bool] = False) -> float:
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
        r2 : float
            The coefficient of determination (R^2) for the model.
        """
        y_pred = self.predict(X, rank=rank, nonnegative=nonnegative)
        return measure_r2(y_pred, y, dim=0)

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

    def to(self, device):
        """
        Move the ReducedRankRegression model to a device.

        Parameters
        ----------
        device : torch.device or str
            The device to move the model to (e.g., 'cpu', 'cuda', torch.device('cuda:0')).

        Returns
        -------
        self : ReducedRankRegression
            The ReducedRankRegression object with all tensors moved to the specified device.
        """
        super().to(device)
        if hasattr(self, "_Xbeta_V"):
            self._Xbeta_V = self._Xbeta_V.to(device)
        if hasattr(self, "_rank_restraint"):
            self._rank_restraint = self._rank_restraint.to(device)
        if hasattr(self, "_beta_rrr"):
            self._beta_rrr = self._beta_rrr.to(device)
        return self

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
