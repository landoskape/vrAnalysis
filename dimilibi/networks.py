from typing import List
import torch
from torch import nn


class SVCANet(nn.Module):
    """
    SVCA Network for peer prediction analysis

    SVCA Net is a beta-VAE like autoencoder designed to perform peer-prediction analysis between neurons.
    The encoder and decoder are both fully connected neural networks with a variable number of hidden layers.
    """

    def __init__(
        self,
        num_neurons: int,
        width_hidden: List[int],
        num_latent: int,
        num_target_neurons: int = None,
        include_nn_latent: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
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
        include_nn_latent : bool
            If True, include an additional nonnegative latent layer (default is False)
        activation : nn.Module
            Activation function to use in the hidden layers (default is nn.ReLU())
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.width_hidden = width_hidden
        self.num_latent = num_latent
        self.num_target_neurons = num_target_neurons or num_neurons
        self.include_nn_latent = include_nn_latent
        self.activation = activation
        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        """
        Build the encoder network. Composed of a series of hidden layers and a latent layer.

        The hidden layers are defined by the width_hidden parameter. They are all linear and
        are followed by the activation function defined in the constructor.

        The latent layer is a linear layer that generates a constrained latent representation
        of the input data. It can include an additional nonnegative latent layer with the same
        width if requested.
        """
        self.encoder_hidden = nn.ModuleList()

        prev_width = self.num_neurons
        for width in self.width_hidden:
            self.encoder_hidden.append(nn.Linear(prev_width, width))
            prev_width = width

        self.latent = nn.Linear(prev_width, self.num_latent)
        if self.include_nn_latent:
            self.nn_latent = nn.Sequential(nn.Linear(prev_width, self.num_latent), nn.ReLU())

    def _build_decoder(self):
        """
        Build the decoder network. Composed of a series of hidden layers and an output layer.

        The hidden layers are defined by the width_hidden parameter. They are all linear and
        are symmetric with the encoder. They are followed by the activation function defined
        in the constructor. The output layer is a linear layer that generates the output data.
        """
        self.decoder_hidden = nn.ModuleList()

        prev_width = self.num_latent + self.num_latent * self.include_nn_latent
        for width in self.width_hidden:
            self.decoder_hidden.append(nn.Linear(prev_width, width))
            prev_width = width

        self.output = nn.Linear(prev_width, self.num_target_neurons)

    def forward(self, x: torch.Tensor, store_hidden: bool = False) -> torch.Tensor:
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
        for layer in self.encoder_hidden:
            x = self.activation(layer(x))
        latent = self.latent(x)
        if store_hidden:
            self.latent = latent
        if self.include_nn_latent:
            nn_latent = self.nn_latent(x)
            if store_hidden:
                self.nn_latent = nn_latent
            x = torch.cat([latent, nn_latent], dim=-1)
        else:
            x = latent
        for layer in self.decoder_hidden:
            x = self.activation(layer(x))
        return self.output(x)

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
