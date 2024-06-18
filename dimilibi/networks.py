from typing import List
import torch
from torch import nn


class SVCANet(nn.Module):
    """
    SVCA Network for peer prediction analysis

    SVCA Net is a beta-VAE like autoencoder designed to perform peer-prediction analysis between neurons.
    The encoder and decoder are both fully connected neural networks with a variable number of hidden layers.
    """

    def __init__(self, num_neurons: int, width_hidden: List[int], num_latent: int, num_target_neurons: int = None, activation: nn.Module = nn.ReLU()):
        """
        Initialize the SVCA Network

        Parameters
        ----------
        num_neurons : int
            Number of neurons in the input layer
        width_hidden : List[int]
            List of integers representing the width of each hidden layer
        num_latent : int
            Number of neurons in the latent layer
        num_target_neurons : int
            Number of neurons in the output layer (default is to use num_neurons)
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.width_hidden = width_hidden
        self.num_latent = num_latent
        self.num_target_neurons = num_target_neurons or num_neurons
        self.encoder = self._build_encoder(activation)
        self.decoder = self._build_decoder(activation)

    def _build_encoder(self, activation: nn.Module) -> nn.Sequential:
        """
        Build the encoder network

        Parameters
        ----------
        activation : nn.Module
            Activation function to use in the hidden layers

        Returns
        -------
        nn.Sequential
            nn.Sequential object representing the encoder network
        """
        layers = []
        prev_width = self.num_neurons
        if self.width_hidden is not None:
            for width in self.width_hidden:
                layers.append(nn.Linear(prev_width, width))
                layers.append(activation)
                prev_width = width
        layers.append(nn.Linear(prev_width, self.num_latent))
        return nn.Sequential(*layers)

    def _build_decoder(self, activation: nn.Module) -> nn.Sequential:
        """
        Build the decoder network

        Parameters
        ----------
        activation : nn.Module
            Activation function to use in the hidden layers

        Returns
        -------
        nn.Sequential
            nn.Sequential object representing the decoder network
        """
        layers = []
        prev_width = self.num_latent
        if self.width_hidden is not None:
            for width in reversed(self.width_hidden):
                layers.append(nn.Linear(prev_width, width))
                layers.append(activation)
                prev_width = width
        layers.append(nn.Linear(prev_width, self.num_target_neurons))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, store_hidden=False) -> torch.Tensor:
        """
        Forward pass through the SVCA Network

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_neurons, num_timepoints) or (num_neurons, num_timepoints)
        store_hidden : bool
            If True, will store the latent representation of the input tensor
            (default is False)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_neurons, num_timepoints) or (num_neurons, num_timepoints)
        """
        latent = self.encoder(x)
        if store_hidden:
            self.latent = latent
        return self.decoder(latent)

    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Score the SVCANet prediction on target data.

        Parameters
        ----------
        x : torch.Tensor
            The input data (num_neurons, num_timepoints)
        y : torch.Tensor
            The target data (num_neurons, num_timepoints).

        Returns
        -------
        r2 : torch.Tensor
            The coefficient of determination (R^2) for the model.
        """
        y_pred = self(x)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean(dim=0, keepdim=True)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        return r2
