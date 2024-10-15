from typing import List, Tuple
import torch
from torch import nn
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


class _TransparentReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(_, grad_output):
        return grad_output


class TransparentReLU(nn.Module):
    """
    Transparent ReLU activation function

    This is an implementation of the ReLU where the gradient is passed through as if
    there wasn't a ReLU. This is useful when the distance from 0 (for negative numbers)
    is relevant to the model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return _TransparentReLU.apply(input)


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
        transparent_relu: bool = False,
        dropout_rate: float = 0.0,
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
        transparent_relu : bool
            If True, will use a transparent ReLU activation function (default is False)
        dropout_rate : float
            Dropout rate to apply to the hidden layers (default is 0.0)
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.width_hidden = width_hidden
        self.num_latent = num_latent
        self.num_target_neurons = num_target_neurons or num_neurons
        self.activation = activation
        self.nonnegative = nonnegative
        self.transparent_relu = transparent_relu
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
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

        if not self.nonnegative:
            self.final_nonlinearity = nn.Identity()
        elif self.transparent_relu:
            self.final_nonlinearity = TransparentReLU()
        else:
            self.final_nonlinearity = nn.ReLU()

        if self.activation == nn.ReLU() and self.transparent_relu:
            self.activation = TransparentReLU()

    def _init_weights(self):
        """
        Initialize the weights of the network. Uses the Kaiming initialization method.
        """
        for m in self.modules():
            kaiming_init(m)

    def update_dropout_rate(self, dropout_rate: float):
        """
        Update the dropout rate of the hidden layers.

        Parameters
        ----------
        dropout_rate : float
            New dropout rate to apply to the hidden layers
        """
        self.dropout_rate = dropout_rate
        self.dropout.p = dropout_rate

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
            x = self.dropout(self.activation(layer(x)))
        x = self.norm(x)
        x = self.latent(x)  # latent activation
        if store_hidden:
            self.latent = x.clone()
        for layer in self.decoder_hidden:
            x = self.dropout(self.activation(layer(x)))
        return self.final_nonlinearity(self.output(x))

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


class HurdleNet(SVCANet):
    """
    Network for peer prediction analysis with a hurdle model (allowing for prediction of zeros independent of non-zero prediction).

    HurdleNet is an two path autoencoder-like network designed to perform peer-prediction analysis between neurons.
    There are two paths with an encoder and decoder. All are both fully connected neural networks with a variable number of hidden layers
    parameterized by the user.

    The first "path" is the non-zero path, which predicts the non-zero values of target data. This path has a linear output (with optional ReLU).

    The second "path" is the zero path, which predicts whether to use a zero value or not. This path has a probabilistic output in which the output
    is 1 or 0 and simply scales the first path. This is the "hurdle" in the model.

    TODO: make second path non-probailistic in evaluation mode
    """

    def __init__(
        self,
        num_neurons: int,
        width_hidden: List[int],
        num_latent: int,
        num_target_neurons: int = None,
        activation: nn.Module = nn.ReLU(),
        nonnegative: bool = True,
        transparent_relu: bool = False,
        dropout_rate: float = 0.0,
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
        transparent_relu : bool
            If True, will use a transparent ReLU activation function (default is False)
        dropout_rate : float
            Dropout rate to apply to the hidden layers (default is 0.0)
        """
        nn.Module.__init__(self)
        self.num_neurons = num_neurons
        self.width_hidden = width_hidden
        self.num_latent = num_latent
        self.num_target_neurons = num_target_neurons or num_neurons
        self.activation = activation
        self.nonnegative = nonnegative
        self.transparent_relu = transparent_relu
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.use_hurdle = True
        self._build_encoder()
        self._build_decoder()
        self._build_hurdle_encoder()
        self._build_hurdle_decoder()
        self._init_weights()

    def _build_hurdle_encoder(self):
        """
        Build the encoder network of the hurdle path. Composed of a series of hidden layers and a latent layer.

        The hidden layers are defined by the width_hidden parameter. They are all linear and
        are followed by the activation function defined in the constructor.

        The latent layer is a linear layer that generates a constrained latent representation
        of the input data.
        """
        self.hurdle_encoder_hidden = nn.ModuleList()

        prev_width = self.num_neurons
        for width in self.width_hidden:
            self.hurdle_encoder_hidden.append(nn.Linear(prev_width, width))
            prev_width = width

        self.hurdle_norm = nn.BatchNorm1d(prev_width)
        self.hurdle_latent = nn.Linear(prev_width, self.num_latent)

    def _build_hurdle_decoder(self):
        """
        Build the hurdle decoder network. Composed of a series of hidden layers and an output layer.

        The hidden layers are defined by the width_hidden parameter. They are all linear and
        are symmetric with the encoder. They are followed by the activation function defined
        in the constructor. The output layer is a linear layer that generates the output data.
        """
        self.hurdle_decoder_hidden = nn.ModuleList()

        prev_width = self.num_latent
        for width in reversed(self.width_hidden):
            self.hurdle_decoder_hidden.append(nn.Linear(prev_width, width))
            prev_width = width

        self.hurdle_output = nn.Linear(prev_width, self.num_target_neurons)
        self.hurdle_activation = nn.Sigmoid()

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
        x_hurdle = x.clone()

        # Do prediction path
        for layer in self.encoder_hidden:
            x = self.dropout(self.activation(layer(x)))
        x = self.norm(x)
        x = self.latent(x)  # latent activation
        if store_hidden:
            self.latent = x.clone()
        for layer in self.decoder_hidden:
            x = self.dropout(self.activation(layer(x)))
        predict_output = self.final_nonlinearity(self.output(x))

        # Do hurdle path
        for layer in self.hurdle_encoder_hidden:
            x_hurdle = self.dropout(layer(x_hurdle))
        x_hurdle = self.hurdle_norm(x_hurdle)
        x_hurdle = self.hurdle_latent(x_hurdle)
        if store_hidden:
            self.hurdle_latent = x_hurdle.clone()
        for layer in self.hurdle_decoder_hidden:
            x_hurdle = self.dropout(layer(x_hurdle))
        hurdle_output = self.hurdle_activation(self.hurdle_output(x_hurdle))

        # If in eval mode, do a deterministic hurdle, otherwise random
        if not self.training:
            hurdle_output = hurdle_output.round()
        else:
            hurdle_output = hurdle_output.bernoulli()

        return predict_output * hurdle_output


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
        transparent_relu: bool = False,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.width_hidden = width_hidden
        self.num_latent = num_latent
        self.num_target_neurons = num_target_neurons or num_neurons
        self.activation = activation
        self.nonnegative = nonnegative
        self.transparent_relu = transparent_relu
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

        if not self.nonnegative:
            self.final_nonlinearity = nn.Identity()
        elif self.transparent_relu:
            self.final_nonlinearity = TransparentReLU()
        else:
            self.final_nonlinearity = nn.ReLU()

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
        return self.final_nonlinearity(self.output(z))

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
