import jax
import jax.numpy as jnp
import gpjax as gpx
import matplotlib.pyplot as plt

from src.utils import get_n_neighbours
from test.inspect_fodder2 import multiline_sig


def rbf_covariance(grid_size, var=1.0, scale=1.0):
    xx, yy = jnp.mgrid[0:grid_size, 0:grid_size]
    X = jnp.vstack((xx.flatten(), yy.flatten())).T
    k = gpx.kernels.RBF(lengthscale=scale, variance=var)
    return k.gram(X).to_dense()


def generate_landscape(key, grid_size, cov, threshold_lo, threshold_hi):
    landscape = jax.random.multivariate_normal(
        key, jnp.zeros(cov.shape[0]), cov
    ).reshape(grid_size, grid_size)
    landscape = (landscape - landscape.min()) / (landscape.max() - landscape.min())

    food = (landscape >= threshold_hi).astype(jnp.int32)
    poison = (landscape <= threshold_lo).astype(jnp.int32)

    return food, poison


def evolve_landscape(key, current, p_spont, p_adj, p_death):
    neighbours = get_n_neighbours(current)

    p_adjs = p_adj * neighbours
    p_deaths = p_death * current * (8 - neighbours) / 8

    current -= jax.random.bernoulli(key, p_deaths, current.shape).astype(jnp.int32)
    current += jax.random.bernoulli(key, p_adjs, current.shape).astype(jnp.int32)
    current += jax.random.bernoulli(key, p_spont, current.shape).astype(jnp.int32)

    return jnp.clip(current, 0, 1).astype(jnp.int32)


def visualise(landscape, t):
    fig, ax = plt.subplots()
    ax.imshow(landscape)
    # remove axis labels and ticks
    ax.set(xticks=[], yticks=[], title=f"Time: {t} ({landscape.sum()} resources)")
    plt.show()
    plt.close()


key = jax.random.PRNGKey(1)
grid_size = 30
scale = 3.0
p_spont, p_adj, p_death = 0.001, 0.006, 0.1

cov = rbf_covariance(grid_size, scale=scale, var=1.0)
cov += 1e-5 * jnp.eye(cov.shape[0])

landscape = jax.random.multivariate_normal(key, jnp.zeros(cov.shape[0]), cov).reshape(
    grid_size, grid_size
)
landscape = (landscape - landscape.min()) / (landscape.max() - landscape.min())

plt.imshow(landscape)
plt.show()

# food, poison = generate_landscape(key, grid_size, cov, 0.2, 0.7)

# # scale landscape up to 30x30
# food = jnp.repeat(jnp.repeat(food, 3, axis=0), 3, axis=1)
# poison = jnp.repeat(jnp.repeat(poison, 3, axis=0), 3, axis=1)

# for t in range(500):
#     key, key_food, key_poison = jax.random.split(key, 3)
#     food = evolve_landscape(
#         key_food, food, p_spont=p_spont, p_adj=p_adj, p_death=p_death
#     )
#     poison = evolve_landscape(
#         key_poison, poison, p_spont=p_spont, p_adj=p_adj, p_death=p_death
#     )
#     landscape = food - poison
#     if t % 20 == 0:
#         visualise(landscape, t)
