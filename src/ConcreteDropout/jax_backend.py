from __future__ import annotations
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import lax, random

from flax import nnx
from flax.nnx import Rngs
from flax.nnx import Module


# IMPORTANT: these layers perform dropout BEFORE the wrapped operation.

class CDRegLoss(nnx.Variable): pass

# base class


class ConcreteDropout(Module):
    """Flax.nnx implementation of Concrete Dropout."""

    def __init__(self, layer: Module, broadcast_dims: Sequence[int] = (), weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.2, temperature=0.1, *, deterministic: bool = False, rngs: nnx.Rngs | None = None):
        self.layer = layer
        self.broadcast_dims = broadcast_dims
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.init_min = init_min
        self.init_max = init_max
        self.temperature = temperature
        self.deterministic = deterministic
        self.reg_loss = CDRegLoss(jnp.array(0.0))
        self.rngs = rngs
        key = rngs.dropout()
        self.p_logit = nnx.Param(jnp.log(jax.random.uniform(key, (1,), minval=self.init_min, maxval=self.init_max)))

    def _get_noise_shape(self, inputs):
        """Determines the noise shape. Potentially needs to be adapted for different layer types."""
        broadcast_shape = list(inputs.shape)
        for dim in self.broadcast_dims:
            broadcast_shape[dim] = 1
        return broadcast_shape

    def spatial_concrete_dropout(self, x, p, *, key):
        """Performs concrete dropout."""
        eps = 1e-7
        noise_shape = self._get_noise_shape(x)
        unif_noise = jax.random.uniform(key, noise_shape)

        drop_prob = nnx.sigmoid((jnp.log(p + eps) - jnp.log1p(eps - p) +
                             jnp.log(unif_noise + eps) - jnp.log1p(eps - unif_noise)) / self.temperature)
        random_tensor = 1.0 - drop_prob
        retain_prob = 1.0 - p
        return x * random_tensor / retain_prob

    def __call__(self, inputs):
        """Forward pass."""
        p = nnx.sigmoid(self.p_logit.value)
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * jnp.sum(weight**2) / (1.0 - p)
        dropout_regularizer = p * jnp.log(p) + (1.0 - p) * jnp.log1p(-p)
        dropout_regularizer *= self.dropout_regularizer * inputs.shape[0]
        regularizer = kernel_regularizer + dropout_regularizer

        if not self.deterministic:
            subkey = self.rngs.dropout()
            x = self.spatial_concrete_dropout(inputs, p, key=subkey)
            outputs = self.layer(x)
        else:
            outputs = self.layer(inputs)
    
        self.reg_loss.value = regularizer # store the regularizer loss

        return outputs