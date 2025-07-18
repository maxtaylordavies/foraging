import jax
import jax.numpy as jnp
import chex
import gpjax as gpx


def rbf_covariance(grid_size, var=1.0, scale=1.0, jitter=1e-5):
    xx, yy = jnp.mgrid[0:grid_size, 0:grid_size]
    X = jnp.vstack((xx.flatten(), yy.flatten())).T
    k = gpx.kernels.RBF(lengthscale=scale, variance=var)
    cov = k.gram(X).to_dense()
    return cov + (jitter * jnp.eye(cov.shape[0]))


def get_n_neighbours(x: chex.Array) -> chex.Array:
    def helper(i, j):
        return (
            x[i - 1, j]
            + x[i + 1, j]
            + x[i, j - 1]
            + x[i, j + 1]
            + x[i - 1, j - 1]
            + x[i - 1, j + 1]
            + x[i + 1, j - 1]
            + x[i + 1, j + 1]
        )

    return jax.vmap(jax.vmap(helper, in_axes=(None, 0)), in_axes=(0, None))(
        jnp.arange(x.shape[0]), jnp.arange(x.shape[1])
    ).astype(jnp.int8)
