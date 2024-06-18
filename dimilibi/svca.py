from typing import Optional
import torch


class SVCA:
    """
    Shared Variance Component Analysis

    This class implements the shared variance component analysis method that was introduced in the
    paper: https://www.science.org/doi/10.1126/science.aav7893
    """

    @torch.no_grad()
    def __init__(
        self,
        num_components: Optional[int] = None,
        centered: Optional[bool] = True,
        verbose: Optional[bool] = False,
    ):
        """
        Initialize an SVCA object with the option of specifying supporting parameters.

        Parameters
        ----------
        num_components : Optional[int]
            Number of components to use in the SVD decomposition.
            (default is the minimum number of neurons in the two groups)
        centered : Optional[bool]
            If True, will center the data before performing SVD.
            (default is True)
        verbose : Optional[bool]
            If True, will print updates and results as they are computed.
            (default is True)
        """

        self.num_components = num_components
        self.centered = centered
        self.verbose = verbose
        self.fitted = False

    @torch.no_grad()
    def fit(self, source: torch.Tensor, target: torch.Tensor):
        """
        Fit the SVCA model to the provided data by generating a map from source to target with SVD.

        Parameters
        ----------
        source : torch.Tensor
            The source data to be used for training (num_neurons_source, num_timepoints).
        target : torch.Tensor
            The target data to be used for training (num_neurons_target, num_timepoints).

        Returns
        -------
        self : object
            The SVCA object with the fitted model.
        """
        self.fitted = False
        self._validate_data(source, target)
        self._validate_components(source, target)

        if self.centered:
            source = source - source.mean(dim=1, keepdim=True)
            target = target - target.mean(dim=1, keepdim=True)

        # perform svd on the map from source to target neurons
        gram_matrix = source @ target.T
        self.u, self.s, self.v = torch.svd(gram_matrix, some=True, compute_uv=True)

        # keep only the top num_components
        self.u = self.u[:, : self.num_components]
        self.s = self.s[: self.num_components]
        self.v = self.v[:, : self.num_components]

        self.fitted = True
        return self

    @torch.no_grad()
    def score(self, source: torch.Tensor, target: torch.Tensor, normalize: bool = True):
        """
        Score the SVCA model on the provided data by projecting the data onto the shared space.

        Parameters
        ----------
        source : torch.Tensor
            The source data to be used for testing (num_neurons_source, num_timepoints).
        target : torch.Tensor
            The target data to be used for testing (num_neurons_target, num_timepoints).
        normalize : bool
            If True, will normalize the shared variance by the number of timepoints minus 1.

        Returns
        -------
        shared_variance : torch.Tensor
            The shared variance between the source and target data on each dimension of the fitted model.
        """
        assert self.fitted, "Model must be fitted before scoring data."

        self._validate_data(source, target)

        if self.centered:
            source = source - source.mean(dim=1, keepdim=True)
            target = target - target.mean(dim=1, keepdim=True)

        num_timepoints = source.shape[1]
        source_proj = self.u.T @ source
        target_proj = self.v.T @ target

        # measure shared variance across source and target neurons on fitted model
        norm_value = num_timepoints if normalize else 1
        shared_variance = (source_proj * target_proj).sum(dim=1) / norm_value

        # measure total (average) variance in source and target neurons on fitted model
        source_proj_var = (source_proj**2).sum(dim=1) / norm_value
        target_proj_var = (target_proj**2).sum(dim=1) / norm_value
        total_variance = (source_proj_var + target_proj_var) / 2

        return shared_variance, total_variance

    @torch.no_grad()
    def _validate_data(self, source: torch.Tensor, target: torch.Tensor):
        """Check if source and target data are valid"""
        if self.fitted:
            assert source.shape[0] == self.u.shape[0], "Source data must have the same number of neurons as the training data."
            assert target.shape[0] == self.v.shape[0], "Target data must have the same number of neurons as the training data."
        assert source.shape[1] == target.shape[1], "Number of timepoints must be the same for source and target data."

    @torch.no_grad()
    def _validate_components(self, source: torch.Tensor, target: torch.Tensor):
        """Check if the number of components is valid"""
        min_neurons = min(source.shape[0], target.shape[0])

        if self.num_components is not None:
            msg = "Number of components must be less than or equal to the minimum number of neurons in the two groups."
            assert self.num_components <= min_neurons, msg
        else:
            self.num_components = min_neurons
