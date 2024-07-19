from typing import Optional, Union
import torch
from torch import nn


class FlexibleFilter(nn.Module):
    def __init__(
        self,
        device: str = "cpu",
        baseline: float = 0.1,
        positive_scale: float = 1.0,
        negative_scale: float = 1.0,
        positive_center: float = 0.75,
        negative_center: float = -0.75,
        positive_steepness: float = 5.0,
        negative_steepness: float = 5.0,
    ) -> None:
        """
        Initialize the FlexibleFilter.

        Parameters
        ----------
        device : str, optional
            The device to run the computations on ('cpu' or 'cuda'). Default is 'cpu'.
        baseline : float, optional
            The baseline value of the filter. Default is 0.1.
        positive_scale : float, optional
            The scale for the positive side of the filter. Default is 1.0.
        negative_scale : float, optional
            The scale for the negative side of the filter. Default is 1.0.
        positive_center : float, optional
            The center for the positive side of the filter. Default is 0.75.
        negative_center : float, optional
            The center for the negative side of the filter. Default is -0.75.
        positive_steepness : float, optional
            The steepness for the positive side of the filter. Default is 5.0.
        negative_steepness : float, optional
            The steepness for the negative side of the filter. Default is 5.0.

        Returns
        -------
        None
        """
        super().__init__()
        self.device = torch.device(device)

        self.baseline = torch.tensor(baseline, device=self.device)
        self.positive_scale = torch.tensor(positive_scale, device=self.device)
        self.negative_scale = torch.tensor(negative_scale, device=self.device)
        self.positive_center = torch.tensor(positive_center, device=self.device)
        self.negative_center = torch.tensor(negative_center, device=self.device)
        self.positive_steepness = torch.tensor(positive_steepness, device=self.device)
        self.negative_steepness = torch.tensor(negative_steepness, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, should be in the range [-1, 1] but it doesn't have to be.

        Returns
        -------
        torch.Tensor
            Filtered output in the range [0, 1].
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        positive_activation = torch.sigmoid(self.positive_steepness * (x - self.positive_center))
        negative_activation = 1 - torch.sigmoid(self.negative_steepness * (x - self.negative_center))

        return self.baseline + self.positive_scale * positive_activation + self.negative_scale * negative_activation

    def plot(self, num_points: int = 1000) -> None:
        """
        Plot the filter's transfer function.

        Parameters
        ----------
        num_points : int, optional
            Number of points to use for plotting. Default is 1000.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt

        x = torch.linspace(-1, 1, num_points, device=self.device)
        y = self.forward(x).cpu().numpy()
        x = x.cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(x, y)
        plt.title("Filter Transfer Function")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.grid(True)
        plt.show()

    def to(self, device: Union[str, torch.device]) -> "FlexibleFilter":
        """
        Move the filter to the specified device.

        Parameters
        ----------
        device : str or torch.device
            The device to move the filter to ('cpu' or 'cuda').

        Returns
        -------
        FlexibleFilter
            The filter instance moved to the specified device.
        """
        self.device = torch.device(device)
        self.baseline = self.baseline.to(self.device)
        self.positive_scale = self.positive_scale.to(self.device)
        self.negative_scale = self.negative_scale.to(self.device)
        self.positive_center = self.positive_center.to(self.device)
        self.negative_center = self.negative_center.to(self.device)
        self.positive_steepness = self.positive_steepness.to(self.device)
        self.negative_steepness = self.negative_steepness.to(self.device)
        return self


class LocalSmoothness(nn.Module):
    """
    A class for computing a loss based on local smoothness of data.

    The regularizer attempts to constrain a model such that the output is smooth
    wherever the input is smooth, but the global structure can retain any form. This
    is accomplished by a loss function that penalizes the difference between the
    distance in datapoints so long as the distance is below a variable threshold.

    Mathematically:
    The network attempts to transform source data (x) to target data (y).

    source_distance_{ij} = ||x_i - x_j||   <---- or a generalized difference
    target_distance_{ij} = ||y_i - y_j||   <---- or a generalized difference

    smoothing_filter(distance) = exp(-distance^2 / (2 * threshold^2))

    distance_difference_{ij} = source_distance_{ij} - target_distance_{ij}
    adjusted_difference_{ij} = smoothing_filter(source_distance_{ij}) * distance_difference_{ij}

    loss = sum_{i, j} adjusted_difference_{ij}^2
    """

    def __init__(self, data: torch.Tensor, quantile_threshold: float = 0.1, threshold_value: float = 0.1, quantile_scaling: float = 0.9, reduction: Optional[str] = None):
        """
        Initialize the LocalSmoothness regularizer.

        Parameters
        ----------
        data : torch.Tensor
            The data to be used for defining the threshold. The model will measure the distance
            between all points in the data, then compute a threshold based on a percentile in the
            dataset based on the parameter quantile_threshold.
        quantile_threshold : float
            The threshold quantile for the local smoothness regularizer (default is 0.1).
        threshold_value : float
            The value of the smoothing filter at the threshold (default is 0.1).
        quantile_scaling : float
            The quantile for determining how much to scale distance differences (default is 0.9).
        reduction : Optional[str]
            The reduction to apply to the loss. If "mean", the mean of the loss is returned.
            If "sum", the sum of the loss is returned. Default is None.
        """
        super().__init__()
        self.quantile_threshold = quantile_threshold
        self.threshold_value = threshold_value
        self.quantile_scaling = quantile_scaling
        self.threshold, self.scaling = self._compute_threshold(data, quantile_threshold, threshold_value, quantile_scaling)
        self.reduction = reduction

    def forward(self, source: torch.Tensor, target: torch.Tensor, reduction: Optional[str] = None) -> float:
        """
        Compute the local smoothness loss between the source and target data.

        Parameters
        ----------
        source : torch.Tensor
            The source data (num_samples, num_features).
        target : torch.Tensor
            The target data (num_samples, num_features).
        reduction : Optional[str]
            The reduction to apply to the loss. If "mean", the mean of the loss is returned.
            If "sum", the sum of the loss is returned. Default is None (which will fallback to 
            whatever was set by the constructor method).

        Returns
        -------
        loss : float
            The local smoothness loss between the source and target data.
        """
        source_distance = self._distance(source)
        target_distance = self._distance(target)

        distance_difference = source_distance - target_distance
        adjusted_difference = self._smoothing_filter(source_distance) * distance_difference / self.scaling

        reduction = reduction or self.reduction
        if reduction == 'mean':
            return torch.mean(adjusted_difference**2)
        elif reduction == 'sum':
            return torch.sum(adjusted_difference**2)
        elif reduction is None:
            return adjusted_difference
        else:
            raise ValueError(f"Invalid reduction: {reduction}, must be 'mean', 'sum', or None.")
        
        
    def _distance(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise distance between data points.

        Parameters
        ----------
        data : torch.Tensor
            The data to compute the pairwise distance of (num_samples, num_features).

        Returns
        -------
        distance : torch.Tensor
            The pairwise distance between data points (num_samples, num_samples).
        """
        return torch.pdist(data)

    def _smoothing_filter(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Compute the smoothing filter for the distance matrix.

        Parameters
        ----------
        distance : torch.Tensor
            The distance matrix (num_samples, num_samples).

        Returns
        -------
        filter : torch.Tensor
            The smoothing filter for the distance matrix (num_samples, num_samples).
        """
        return torch.exp(-(distance**2) / (2 * self.threshold**2))

    def _compute_threshold(self, data: torch.Tensor, quantile_threshold: float, threshold_value: float, quantile_scaling: float):
        """
        Compute the threshold for the local smoothness regularizer.

        Parameters
        ----------
        data : torch.Tensor
            The data to be used for defining the threshold (num_samples, num_features).
        quantile_threshold : float
            The threshold quantile for the local smoothness regularizer.
        threshold_value : float
            The value of the filter at the threshold.
        quantile_scaling : float
            The quantile for determining how much to scale distance differences.

        Returns
        -------
        threshold : float
            The threshold for the local smoothness regularizer.
        scaling : float
            The scaling factor for the local smoothness regularizer.
        """
        pairwise_distances = torch.pdist(data)
        quantile = torch.quantile(pairwise_distances, quantile_threshold)
        filter_sigma = torch.sqrt(-(quantile**2) / (2 * torch.log(torch.tensor(threshold_value))))

        scaling = torch.quantile(pairwise_distances, quantile_scaling)
        return filter_sigma, scaling


class LocalSimilarity(nn.Module):
    """
    A class for computing a loss based on local cosine similarity of data.

    The regularizer attempts to constrain a model such that the output has similar
    representations wherever the input does, but the global structure can retain
    any form. This is accomplished by a loss function that penalizes the difference
    between the cosine similarity in datapoints so long as the similarity is below a
    variable threshold.

    Mathematically:
    The network attempts to transform source data (x) to target data (y).

    (assume that the mean is learned and the data is centered)
    source_similarity_{ij} = x_i * x_j / ||x_i|| / ||x_j||   <---- or a generalized similarity
    target_similarity_{ij} = y_i * y_j / ||y_i|| / ||y_j||   <---- or a generalized similarity

    smoothing_filter(similarity) = exp(-similarity^2 / (2 * threshold^2))

    distance_difference_{ij} = source_distance_{ij} - target_distance_{ij}
    adjusted_difference_{ij} = smoothing_filter(source_distance_{ij}) * distance_difference_{ij}

    loss = (1/N) * sum_{i, j} adjusted_difference_{ij}^2      <---- using mean such that the loss is scale-invariant
    """

    def __init__(
        self,
        dim_source: int,
        dim_target: int,
        filter: Optional[FlexibleFilter] = None,
        mean_learning_rate: float = 0.1,
        reduction: Optional[str] = 'mean',
    ):
        """
        Initialize the LocalSimilarity regularizer.

        The user is required to specify the dimensionality of the source and target data, because this is used to
        estimate the mean of the source and target data (the user can also specify the learning rate for the means).
        In addition, the user can specify a filter which will be applied to the source similarity data and then
        multiplied by the difference between source and target.


        Parameters
        ----------
        dim_source : int
            The dimensionality of the source data.
        dim_target : int
            The dimensionality of the target data.
        filter : FlexibleFilter, optional
            The filter to apply to the similarity values (default is None).
        mean_learning_rate : float
            The learning rate for the source and target means (default is 0.1).
        reduction : Optional[str]
            The reduction to apply to the loss. If "mean", the mean of the loss is returned.
            If "sum", the sum of the loss is returned. Default is 'mean'.
        """
        super().__init__()
        self.source_mean = torch.zeros(dim_source, requires_grad=False)
        self.target_mean = torch.zeros(dim_target, requires_grad=False)
        self.filter = filter
        self.mean_learning_rate = mean_learning_rate
        self.reduction = reduction

    def to(self, device):
        """
        Move the regularizer to a device.

        Parameters
        ----------
        device : torch.device
            The device to move the regularizer to.
        """
        super().to(device)
        self.source_mean = self.source_mean.to(device)
        self.target_mean = self.target_mean.to(device)
        if self.filter is not None:
            self.filter = self.filter.to(device)
        return self

    def forward(self, source: torch.Tensor, target: torch.Tensor, update_mean_estimate: bool = True, reduction: Optional[str] = None) -> float:
        """
        Compute the local similarity loss between the source and target data.

        Parameters
        ----------
        source : torch.Tensor
            The source data (num_samples, num_features).
        target : torch.Tensor
            The target data (num_samples, num_features).
        update_mean_estimate : bool
            Whether to update the mean estimate for the source and target data. (default=True)
        reduction: optional str
            The reduction to apply to the loss. If "mean", the mean of the loss is returned.
            If "sum", the sum of the loss is returned. Default is None (which will fallback to 
            whatever was set by the constructor method).

        Returns
        -------
        loss : float
            The local similarity loss between the source and target data.
        """
        with torch.no_grad():
            if update_mean_estimate:
                batch_source_mean = torch.mean(source, dim=0)
                batch_target_mean = torch.mean(target, dim=0)
                self.source_mean = self.source_mean + self.mean_learning_rate * (batch_source_mean - self.source_mean)
                self.target_mean = self.target_mean + self.mean_learning_rate * (batch_target_mean - self.target_mean)

        source_similarity = self._similarity(source, self.source_mean)
        target_similarity = self._similarity(target, self.target_mean)

        similarity_difference = self._clamped_arctanh(source_similarity) - self._clamped_arctanh(target_similarity)
        filtered_similarity = 1.0 if self.filter is None else self.filter(source_similarity)
        adjusted_difference = filtered_similarity * similarity_difference

        reduction = reduction or self.reduction
        if reduction == 'mean':
            return torch.mean(adjusted_difference**2)
        elif reduction == 'sum':
            return torch.sum(adjusted_difference**2)
        elif reduction is None:
            return adjusted_difference
        else:
            raise ValueError(f"Invalid reduction: {reduction}, must be 'mean', 'sum', or None.")


    def _similarity(self, data: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise cosine similarity between data points.

        Parameters
        ----------
        data : torch.Tensor
            The data to compute the pairwise similarity of (num_samples, num_features).
        center : torch.Tensor
            The center to compute the similarity with (num_features).

        Returns
        -------
        similarity : torch.Tensor
            The pairwise similarity between data points (num_samples, num_samples).
        """
        centered_data = data - center
        dot_product = torch.mm(centered_data, centered_data.T)
        norm = torch.linalg.norm(centered_data, ord=2, dim=1)
        similarity = dot_product / norm.unsqueeze(1) / norm

        # the following lines use the implementation in native torch.pdist()
        i = torch.triu_indices(similarity.shape[0], similarity.shape[1], offset=1, device=similarity.device)
        return similarity.flatten().index_select(0, i[0] * similarity.shape[0] + i[1])

    def _clamped_arctanh(self, value: torch.Tensor):
        """
        Compute the arctanh of the value, clamping the value to prevent infinite values.

        Parameters
        ----------
        value : torch.Tensor
            The value(s) to compute the arctanh for.

        Returns
        -------
        arctanh : torch.Tensor
            The arctanh of the value(s).
        """
        eps = 1e-7  # set to be as small as possible while preventing infinities
        return torch.atanh(torch.clamp(value, min=-1 + eps, max=1 - eps))


class BetaVAE_KLDiv(nn.Module):
    """
    KL Divergence regularizer for the Beta-Variational Autoencoder (β-VAE)

    This regularizer measures the KL divergence term of the parameterized
    latent variables, weighted by the beta parameter.

    Parameters
    ----------
    beta : float, optional
        Weight for the KL divergence term (default is 1.0)

    Attributes
    ----------
    beta : float
        Weight for the KL divergence term
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, reduction: Optional[str] = None) -> torch.Tensor:
        """
        Compute the β-VAE loss

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution
        logvar : torch.Tensor
            Log variance of the latent distribution

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the β-VAE loss
        """
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        reduction = reduction or self.reduction
        if reduction == "mean":
            kl_div /= mu.numel()
        return self.beta * kl_div


class EmptyRegularizer(nn.Module):
    """
    The empty regularizer acts like a regularizer but returns 0 for the loss
    regardless of the input. It is useful as a standin for a regularizer when
    no regularization is desired.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Return 0 for the loss regardless of the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, only used to determine the device.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the loss, always 0.
        """
        return torch.tensor(0.0, device=x.device)
