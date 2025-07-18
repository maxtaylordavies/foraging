import functools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


class CNN(nn.Module):
    """Tiny CNN that maps (B, 7, 7, 2) → (B, F)."""

    features: int = 128  # final vector length
    activation: Callable = nn.relu  # easy to swap GELU, SiLU, etc.

    @nn.compact
    def __call__(self, x):
        # x shape: (B, 7, 7, 2)
        x = nn.Conv(4, (3, 3), padding="SAME")(x)
        x = self.activation(x)

        x = nn.Conv(4, (3, 3), padding="SAME")(x)
        x = self.activation(x)

        # Optional: global average pool instead of flattening
        # x = jnp.mean(x, axis=(1, 2))          # shape (B, 4)

        x = x.reshape(x.shape[0], -1)  # flatten -> shape (B, 7*7*4)
        x = nn.Dense(self.features)(x)  # shape (B, features)
        x = self.activation(x)
        return x


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        hidden_size = rnn_state[0].shape[0]
        new_rnn_state, y = nn.GRUCell(features=hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )
