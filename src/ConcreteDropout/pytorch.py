import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5,
                 init_min=0.1,
                 init_max=0.1,
                 is_mc_dropout=False,
                 temperature=0.1):

        super().__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(
            torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)
        self.temperature = temperature
        self.is_mc_dropout = is_mc_dropout

        self.regularization = 0

    def _get_noise_shape(self, x):
        raise NotImplementedError(
            "Subclasses of ConcreteDropout must implement the noise shape")

    def spatial_concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        p = self.p
        # machine precision epsilon for numerical stability inside the log
        eps = torch.finfo(x.dtype).eps

        # this is the shape of the dropout noise, same as tf.nn.dropout
        noise_shape = self._get_noise_shape(x)

        unif_noise = torch.rand(*noise_shape)  # uniform noise
        # bracket inside equation 5, where u=uniform_noise
        drop_prob = (
            torch.log(p + eps)
            - torch.log1p(eps - p)
            + torch.log(unif_noise + eps)
            - torch.log1p(eps - unif_noise)
        )
        drop_prob = torch.sigmoid(drop_prob / self.temperature)  # z of eq 5
        random_tensor = 1. - drop_prob

        retain_prob = 1. - p
        x *= random_tensor  # we multiply the input by the concrete dropout mask
        x /= retain_prob  # we normalise by the probability to retain

        return x

    def get_regularization(self, x, layer):
        p = self.p
        # We will now compute the KL terms following eq.3 of 1705.07832
        weight = layer.weight
        # The kernel regularizer corresponds to the first term
        # Note: we  divide by (1 - p) because  we  scaled  layer  output  by(1 - p)
        kernel_regularizer = self.weight_regularizer * torch.sum(torch.square(
            weight)) / (1. - p)
        # the dropout regularizer corresponds to the second term
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log1p(- p)
        dropout_regularizer *= self.dropout_regularizer * x.shape[1]
        # this is the KL term to be added as a loss
        # regularizer
        return torch.sum(kernel_regularizer + dropout_regularizer)

    def forward(self, x, layer):
        self.p = torch.sigmoid(self.p_logit)
        self.regularization = self.get_regularization(x, layer)
        if self.is_mc_dropout:
            return layer(self.spatial_concrete_dropout(x))
        else:
            if self.training:
                return layer(self.spatial_concrete_dropout(x))
            else:
                return layer(x)

# implementation


class ConcreteLinearDropout(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Linear input layer.
    ```python
        x = # some input layer
        cd = ConcreteLinearDropout()
        linear = torch.Linear(in_features, out_features)
        x = cd(x, linear)
    ```
    # Arguments
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, temperature=0.1, **kwargs):
        super(ConcreteLinearDropout, self).__init__(
            temperature=temperature, **kwargs)

    # implement the noise shape for regular dropout
    def _get_noise_shape(self, inputs):
        input_shape = inputs.shape
        return input_shape

# convolution layers


class ConcreteDropout1D(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Conv1d input layer. 
    It is the Concrete Dropout implementation of `Dropout1d`.

    ```python
        x = # some input layer
        cd = ConcreteDropout1D()
        conv1d = torch.Conv1d(in_channels,out_channels, kernel_size)
        x = cd(x, conv1d)
    ```
    # Arguments
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, temperature=2./3., **kwargs):
        super(ConcreteDropout1D, self).__init__(
            temperature=temperature, **kwargs)

    # implement the noise shape for spatial dropout
    def _get_noise_shape(self, inputs):
        input_shape = inputs.shape
        return (input_shape[0], input_shape[1], 1)


class ConcreteDropout2D(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Conv2d input layer. 
    It is the Concrete Dropout implementation of `Dropout2d`.

    ```python
        x = # some input layer
        cd = ConcreteDropout2D()
        conv2d = torch.Conv2d(in_channels,out_channels, kernel_size)
        x = cd(x, conv2d)
    ```
    # Arguments
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, temperature=2./3., **kwargs):
        super(ConcreteDropout2D, self).__init__(
            temperature=temperature, **kwargs)

    # implement the noise shape for spatial dropout
    def _get_noise_shape(self, inputs):
        input_shape = inputs.shape
        return (input_shape[0], input_shape[1], 1, 1)


class ConcreteDropout3D(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Conv3d input layer. 
    It is the Concrete Dropout implementation of `Dropout3d`.

    ```python
        x = # some input layer
        cd = ConcreteDropout3D()
        conv3d = torch.Conv3d(in_channels,out_channels, kernel_size)
        x = cd(x, conv3d)
    ```
    # Arguments
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, temperature=2./3., **kwargs):
        super(ConcreteDropout3D, self).__init__(
            temperature=temperature, **kwargs)

    # implement the noise shape for spatial dropout
    def _get_noise_shape(self, inputs):
        input_shape = inputs.shape
        return (input_shape[0], input_shape[1], 1, 1, 1)


def get_weight_regularizer(N, l=1e-2, tau=0.1):
    return l**2 / (tau * N)


def get_dropout_regularizer(N, tau=0.1, cross_entropy_loss=False):
    reg = 1 / (tau * N)
    if not cross_entropy_loss:
        reg *= 2
    return reg
