from typing import Optional
from sklearn.decomposition import PCA as skPCA
import torch


class PCA:
    """
    Principal Component Analysis

    This class is a wrapper of the sklearn PCA implementation used for dimensionality reduction
    and data analysis. It uses a torch API and has functionality to integrate with the rest
    of the dimilibi library.
    """

    @torch.no_grad()
    def __init__(self, num_components: Optional[int] = None, verbose: Optional[bool] = False):
        """
        Initialize a PCA object with the option of specifying supporting parameters.

        Parameters
        ----------
        num_components : Optional[int]
            Number of components to use in the SVD decomposition.
            (default is the minimum number of neurons in the two groups)
        verbose : Optional[bool]
            If True, will print updates and results as they are computed.
            (default is True)
        """

        self.num_components = num_components
        self.verbose = verbose
        self.fitted = False

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        """
        Fit the PCA model to the provided data.

        Parameters
        ----------
        data : torch.Tensor
            The data to be used for training a PCA model (num_neurons_source, num_timepoints).

        Returns
        -------
        self : object
            The PCA object with the fitted model.
        """
        self.fitted = False
        self._validate_data(data)
        self._validate_components(data)

        # perform svd on the map from source to target neurons
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        self.num_samples = data.size(1)
        self.dtype = data.dtype
        self.model = skPCA(n_components=self.num_components).fit(data.T)

        self.fitted = True
        return self

    @torch.no_grad()
    def get_components(self):
        """
        Get the components of the PCA model.

        Returns
        -------
        components : torch.Tensor
            The components of the PCA model.
        """
        return torch.tensor(self.model.components_, dtype=self.dtype).T

    @torch.no_grad()
    def get_singular_values(self):
        """
        Get the singular values of the PCA model.

        Returns
        -------
        singular_values : torch.Tensor
            The singular values of the PCA model.
        """
        return torch.tensor(self.model.singular_values_, dtype=self.dtype)

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
    def _validate_data(self, data: torch.Tensor):
        pass

    @torch.no_grad()
    def _validate_components(self, data: torch.Tensor):
        pass
