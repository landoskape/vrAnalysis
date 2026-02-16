import warnings
from typing import Optional, Union
import numpy as np
import torch
from sklearn.decomposition import PCA as skPCA


def as_tensor(data: Union[np.ndarray, torch.Tensor, list, tuple], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Convert input data to a torch tensor, handling multiple input types.

    Parameters
    ----------
    data : Union[np.ndarray, torch.Tensor, list, tuple]
        The data to convert to a torch tensor.
    dtype : Optional[torch.dtype]
        The dtype to convert the data to.
        (default is None which corresponds to the dtype of the data)

    Returns
    -------
    torch.Tensor
        The data as a torch tensor.
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if dtype is not None:
        data = data.to(dtype)
    return data


class PytorchPCA:
    """
    PytorchPCA is a wrapper of the torch SVD implementation for PCA built
    to have the same interface as the sklearn PCA class.
    """

    def __init__(self, num_components: Optional[int] = None):
        """
        Initialize a PytorchPCA object.

        Parameters
        ----------
        num_components : Optional[int]
            Number of components to use in the PCA decomposition.
            (default is None which corresponds to using all components)
        """
        self.num_components = num_components
        self._components_ = None
        self._singular_values_ = None
        self._fitted = False

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        """
        Fit the PCA model to the provided data using SVD.

        Parameters
        ----------
        data : torch.Tensor
            The data to be used for training a PCA model (num_samples, num_features).
            Data is NOT centered before SVD.

        Returns
        -------
        self : object
            The PytorchPCA object with the fitted model.
        """
        data = as_tensor(data)

        # U: (num_features, num_features), S: (min(num_features, num_samples)), V: (num_samples, num_samples)
        U, S, _ = torch.linalg.svd(data.T, full_matrices=False)

        # Store singular values
        if self.num_components is not None:
            S = S[: self.num_components]
            # Components are the first num_components columns of U
            # Transpose to match sklearn convention: (num_components, num_features)
            U = U[:, : self.num_components]

        self._singular_values_ = S
        self._components_ = U.T
        self._fitted = True
        return self

    @torch.no_grad()
    def transform(self, data: torch.Tensor):
        """
        Transform the input data using the PCA model.

        Parameters
        ----------
        data : torch.Tensor
            The data to be transformed (num_samples, num_features).

        Returns
        -------
        torch.Tensor
            The transformed data (num_samples, num_components).
        """
        if not self._fitted:
            raise ValueError("PytorchPCA model must be fitted before transforming data.")

        data = as_tensor(data)

        # Project data onto components: components_ @ data
        return data @ self.components_.T

    @property
    def components_(self):
        """
        Principal components (eigenvectors of the covariance matrix).

        Returns
        -------
        torch.Tensor
            Components of shape (num_components, num_features).
        """
        if not self._fitted:
            raise ValueError("PytorchPCA model must be fitted before accessing components_.")
        return self._components_

    @property
    def singular_values_(self):
        """
        Singular values corresponding to the principal components.

        Returns
        -------
        torch.Tensor
            Singular values of shape (num_components,).
        """
        if not self._fitted:
            raise ValueError("PytorchPCA model must be fitted before accessing singular_values_.")
        return self._singular_values_


class PCA:
    """
    Principal Component Analysis

    This class is a wrapper of the sklearn PCA implementation used for dimensionality reduction
    and data analysis. It uses a torch implementation and has functionality to integrate with the
    rest of the dimilibi library.
    """

    @torch.no_grad()
    def __init__(
        self,
        num_components: Optional[int] = None,
        verbose: Optional[bool] = False,
        use_svd: Optional[bool] = None,
        center: Optional[bool] = True,
    ):
        """
        Initialize a PCA object with the option of specifying supporting parameters.

        Parameters
        ----------
        num_components : Optional[int]
            Number of components to use in the PCA decomposition.
            (default is None which corresponds to using all components)
        verbose : Optional[bool]
            If True, will print updates and results as they are computed.
            (default is False)
        use_svd : Optional[bool], deprecated
            If True, use torch SVD; if False, use sklearn PCA. Default is True.
            Deprecated: prefer not passing this; the torch SVD path is the default
            and has fewer hidden steps. When False, sklearn is used and supersedes
            ``center`` (sklearn always centers internally).
        center : Optional[bool]
            If True, center the data before decomposition. Default is True.
            Ignored when use_svd=False (sklearn always centers).
        """
        if use_svd is not None:
            warnings.warn(
                "use_svd is deprecated and will be removed. The torch SVD path is now " "the default. Use center=False for uncentered PCA.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.use_svd = use_svd
        else:
            self.use_svd = True

        self.num_components = num_components
        self.verbose = verbose
        self.fitted = False
        self.center = center

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        """
        Fit the PCA model to the provided data.

        Parameters
        ----------
        data : torch.Tensor
            The data to be used for training a PCA model (num_neurons, num_timepoints).

        Returns
        -------
        self : object
            The PCA object with the fitted model.
        """
        self.fitted = False
        self._validate_data(data)
        self._validate_components(data)

        data = as_tensor(data)
        if self.center:
            self._mean = data.mean(dim=1, keepdim=True)
            data = data - self._mean

        self.num_samples = data.size(1)
        self.dtype = data.dtype
        if self.use_svd:
            self.model = PytorchPCA(num_components=self.num_components).fit(data.T)
        else:
            self.model = skPCA(n_components=self.num_components).fit(data.T)

        self.fitted = True
        return self

    @torch.no_grad()
    def transform(self, data: torch.Tensor, whiten: Optional[bool] = False, k: Optional[int] = None):
        """
        Transform the input data using the PCA model.

        Parameters
        ----------
        data : torch.Tensor
            The data to be transformed (num_neurons, num_timepoints).
        whiten : Optional[bool]
            If True, the data will be whitened using the ZCA matrix.
            (default is False)
        k : Optional[int]
            The number of components to use in the ZCA whitening matrix.
            (default is None which corresponds to using all components)

        Returns
        -------
        torch.Tensor
            The transformed data.
        """
        if not self.fitted:
            raise ValueError("PCA model must be fitted before transforming data.")

        self._validate_data(data)

        if self.center:
            data = data - self._mean

        if whiten:
            zca = self.get_zca(k, k=k)
            return zca @ data

        else:
            U = self.get_components()
            if k is not None:
                U = U[:, :k]
            return U.T @ data

    @torch.no_grad()
    def get_components(self):
        """
        Get the components of the PCA model.

        Returns
        -------
        components : torch.Tensor
            The components of the PCA model.
        """
        return as_tensor(self.model.components_, dtype=self.dtype).T

    @torch.no_grad()
    def get_singular_values(self):
        """
        Get the singular values of the PCA model.

        Returns
        -------
        singular_values : torch.Tensor
            The singular values of the PCA model.
        """
        return as_tensor(self.model.singular_values_, dtype=self.dtype)

    @torch.no_grad()
    def get_eigenvalues(self):
        """
        Get the eigenvalues of the PCA model.

        Returns
        -------
        eigenvalues : torch.Tensor
            The eigenvalues of the PCA model.
        """
        return self.get_singular_values() ** 2 / self.num_samples

    @torch.no_grad()
    def get_zca(self, k=None, eps=1e-4):
        """
        Get the ZCA whitening matrix of the PCA model.

        Parameters
        ----------
        k : Optional[int]
            The number of components to use in the ZCA whitening matrix.
            (default is None which corresponds to using all components)
        eps : float
            A small value to add to the eigenvalues to prevent division by zero.
            (default is 1e-4)

        Returns
        -------
        zca : torch.Tensor
            The ZCA whitening matrix of the PCA model.
        """
        U = self.get_components()
        S = self.get_eigenvalues()
        if k is not None:
            U = U[:, :k]
            S = S[:k]
        return U @ torch.diag(1 / torch.sqrt(S + eps)) @ U.T

    @torch.no_grad()
    def _validate_data(self, data: torch.Tensor):
        pass

    @torch.no_grad()
    def _validate_components(self, data: torch.Tensor):
        pass
