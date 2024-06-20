import torch
from torch import nn


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

    def __init__(self, data: torch.Tensor, quantile_threshold: float = 0.1, threshold_value: float = 0.1):
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
        """
        super().__init__()
        self.quantile_threshold = quantile_threshold
        self.threshold_value = threshold_value
        self.threshold = self._compute_threshold(data, quantile_threshold, threshold_value)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute the local smoothness loss between the source and target data.

        Parameters
        ----------
        source : torch.Tensor
            The source data (num_samples, num_features).
        target : torch.Tensor
            The target data (num_samples, num_features).

        Returns
        -------
        loss : float
            The local smoothness loss between the source and target data.
        """
        source_distance = self._distance(source)
        target_distance = self._distance(target)

        distance_difference = source_distance - target_distance
        adjusted_difference = self._smoothing_filter(source_distance) * distance_difference

        return torch.sum(adjusted_difference**2)

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

    def _compute_threshold(self, data: torch.Tensor, quantile_threshold: float, threshold_value: float):
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

        Returns
        -------
        threshold : float
            The threshold for the local smoothness regularizer.
        """
        pairwise_distances = torch.pdist(data)
        quantile = torch.quantile(pairwise_distances, quantile_threshold)
        filter_sigma = torch.sqrt(-(quantile**2) / (2 * torch.log(torch.tensor(threshold_value))))
        return filter_sigma
