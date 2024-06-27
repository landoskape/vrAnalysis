from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


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
        activation: nn.Module = nn.ReLU(),
        nonnegative: bool = True,
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
        activation : nn.Module
            Activation function to use in the hidden layers (default is nn.ReLU())
        nonnegative : bool
            If True, will apply a non-negative constraint to the output layer (default is True)
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.width_hidden = width_hidden
        self.num_latent = num_latent
        self.num_target_neurons = num_target_neurons or num_neurons
        self.activation = activation
        self.nonnegative = nonnegative
        self._build_encoder()
        self._build_decoder()
        self._init_weights()

    def _build_encoder(self):
        """
        Build the encoder network. Composed of a series of hidden layers and a latent layer.

        The hidden layers are defined by the width_hidden parameter. They are all linear and
        are followed by the activation function defined in the constructor.

        The latent layer is a linear layer that generates a constrained latent representation
        of the input data.
        """
        self.encoder_hidden = nn.ModuleList()

        prev_width = self.num_neurons
        for width in self.width_hidden:
            self.encoder_hidden.append(nn.Linear(prev_width, width))
            prev_width = width

        self.norm = nn.BatchNorm1d(prev_width)
        self.latent = nn.Linear(prev_width, self.num_latent)

    def _build_decoder(self):
        """
        Build the decoder network. Composed of a series of hidden layers and an output layer.

        The hidden layers are defined by the width_hidden parameter. They are all linear and
        are symmetric with the encoder. They are followed by the activation function defined
        in the constructor. The output layer is a linear layer that generates the output data.
        """
        self.decoder_hidden = nn.ModuleList()

        prev_width = self.num_latent
        for width in reversed(self.width_hidden):
            self.decoder_hidden.append(nn.Linear(prev_width, width))
            prev_width = width

        self.output = nn.Linear(prev_width, self.num_target_neurons)

        if self.nonnegative:
            self.output = nn.Sequential(self.output, nn.SELU())

    def _init_weights(self):
        """
        Initialize the weights of the network. Uses the Kaiming initialization method.
        """
        for m in self.modules():
            kaiming_init(m)

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
        x = self.norm(x)
        latent = self.latent(x)
        if store_hidden:
            self.latent = latent
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


class BetaVAE(nn.Module):
    """
    Beta-Variational Autoencoder (β-VAE) Network

    β-VAE is a type of variational autoencoder that learns disentangled latent representations.
    The encoder and decoder are both fully connected neural networks with a variable number of hidden layers.
    This implementation allows for balancing between reconstruction quality and disentanglement of latent factors.

    Parameters
    ----------
    num_neurons : int
        Number of neurons in the input layer
    width_hidden : List[int]
        List of integers representing the width of each hidden layer
    num_latent : int
        Number of neurons in the latent layer
    num_target_neurons : int, optional
        Number of neurons in the output layer (default is to use num_neurons)
    activation : nn.Module, optional
        Activation function to use in the hidden layers (default is nn.ReLU())
    nonnegative : bool, optional
        If True, will apply a non-negative constraint to the output layer (default is True)

    Attributes
    ----------
    encoder_hidden : nn.ModuleList
        List of hidden layers in the encoder
    fc_mu : nn.Linear
        Linear layer for generating the mean of the latent distribution
    fc_logvar : nn.Linear
        Linear layer for generating the log variance of the latent distribution
    decoder_hidden : nn.ModuleList
        List of hidden layers in the decoder
    output : nn.Linear
        Output layer of the decoder
    """

    def __init__(
        self,
        num_neurons: int,
        width_hidden: List[int],
        num_latent: int,
        num_target_neurons: int = None,
        activation: nn.Module = nn.ReLU(),
        nonnegative: bool = True,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.width_hidden = width_hidden
        self.num_latent = num_latent
        self.num_target_neurons = num_target_neurons or num_neurons
        self.activation = activation
        self.nonnegative = nonnegative
        self._build_encoder()
        self._build_decoder()
        self._init_weights()

    def _build_encoder(self):
        """
        Build the encoder network.

        The encoder is composed of a series of hidden layers followed by two parallel
        linear layers that generate the mean and log variance of the latent distribution.
        """
        self.encoder_hidden = nn.ModuleList()
        prev_width = self.num_neurons
        for width in self.width_hidden:
            self.encoder_hidden.append(nn.Linear(prev_width, width))
            prev_width = width
        self.norm = nn.BatchNorm1d(prev_width)
        self.to_mu = nn.Linear(prev_width, self.num_latent)
        self.to_logvar = nn.Linear(prev_width, self.num_latent)

    def _build_decoder(self):
        """
        Build the decoder network.

        The decoder is composed of a series of hidden layers followed by an output layer.
        The hidden layers are symmetric with the encoder.
        """
        self.decoder_hidden = nn.ModuleList()
        prev_width = self.num_latent
        for width in reversed(self.width_hidden):
            self.decoder_hidden.append(nn.Linear(prev_width, width))
            prev_width = width
        self.output = nn.Linear(prev_width, self.num_target_neurons)
        if self.nonnegative:
            self.output = nn.Sequential(self.output, nn.SELU())

    def _init_weights(self):
        """
        Initialize the weights of the network. Uses the Kaiming initialization method.
        """
        for m in self.modules():
            kaiming_init(m)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input data to produce the latent distribution parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_neurons)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the mean and log variance of the latent distribution
        """
        for layer in self.encoder_hidden:
            x = self.activation(layer(x))
        x = self.norm(x)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        return mu, logvar

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Perform the reparameterization trick to sample from the latent distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution
        logvar : torch.Tensor
            Log variance of the latent distribution

        Returns
        -------
        torch.Tensor
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent vector to reconstruct the input data.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape (batch_size, num_latent)

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor of shape (batch_size, num_target_neurons)
        """
        for layer in self.decoder_hidden:
            z = self.activation(layer(z))
        return self.output(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the β-VAE Network

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_neurons)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing:
            - Reconstructed input tensor of shape (batch_size, num_target_neurons)
            - Mean of the latent distribution
            - Log variance of the latent distribution
        """
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        return self._decode(z), mu, logvar

    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the coefficient of determination (R^2) for the model predictions

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
        y_pred = self(x)[0]
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean(dim=0, keepdim=True)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        return r2
