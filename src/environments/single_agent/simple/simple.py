from enum import Enum
from functools import reduce
from typing import Any, Dict, Optional, Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from gymnax.environments import spaces

from ..environment import (
    Environment,
    EnvState as BaseEnvState,
    EnvParams as BaseEnvParams,
)
from src.utils import get_n_neighbours, rbf_covariance

Obs = chex.Array


class Action(Enum):
    NONE = 0
    FORWARD = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
    EAT = 4


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


DIRECTION_TO_DELTA = jnp.asarray(
    [
        (-1, 0),  # 0 = up
        (0, 1),  # 1 = right
        (1, 0),  # 2 = down
        (0, -1),  # 3 = left
    ]
)


@struct.dataclass
class EnvState(BaseEnvState):
    ego_agent_energy: chex.Array
    ego_agent_loc: chex.Array
    ego_agent_dir: chex.Array
    demonstrator_locs: chex.Array
    demonstrator_dirs: chex.Array
    resource_map: chex.Array
    consumed_counts: chex.Array


@struct.dataclass
class EnvParams(BaseEnvParams):
    spont_growth_rate: float = 0.0
    adj_growth_rate: float = 0.005
    death_rate: float = 0.05
    resource_value: float = 1.0
    base_energy_loss: float = 0.05
    move_energy_loss: float = 0.1
    max_energy: float = 10.0
    initial_energy: float = 5.0
    energy_threshold: float = 5.0
    reward_size: float = 0.1
    penalty_size: float = 0.0
    max_steps_in_episode: int = 500


class SimpleForagingEnv(Environment):
    def __init__(
        self,
        grid_size: int,
        num_demonstrators: int,
        resource_vis_range: int,
        demonstrator_vis_range: int,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_demonstrators = num_demonstrators
        self.resource_vis_range = resource_vis_range
        self.demonstrator_vis_range = demonstrator_vis_range

        # precompute cholesky root of RBF covariance matrix
        # for fast sampling of resource maps
        rbf_cov = rbf_covariance(30, scale=5.0)
        self.cholesky_root = jsp.cholesky(rbf_cov, lower=True)

        self.max_vis_range = max(resource_vis_range, demonstrator_vis_range)
        self.obs_size = (2 * self.max_vis_range) + 1

        self.blank_obs_map = jnp.zeros((self.obs_size, self.obs_size))

        self.resource_vis_indices_row, self.resource_vis_indices_col = jnp.meshgrid(
            jnp.arange(-self.resource_vis_range, self.resource_vis_range + 1),
            jnp.arange(-self.resource_vis_range, self.resource_vis_range + 1),
            indexing="ij",
        )
        self.demonstrator_vis_indices_row, self.demonstrator_vis_indices_col = (
            jnp.meshgrid(
                jnp.arange(
                    -self.demonstrator_vis_range, self.demonstrator_vis_range + 1
                ),
                jnp.arange(
                    -self.demonstrator_vis_range, self.demonstrator_vis_range + 1
                ),
                indexing="ij",
            )
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def make_resource_map(
        self, food_map: chex.Array, poison_map: chex.Array
    ) -> chex.Array:
        """Create a resource map from food and poison maps."""
        return jnp.where(poison_map, -1, food_map)

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[Obs, EnvState]:
        eps = jax.random.normal(key, (self.grid_size**2,))
        m = self.cholesky_root @ eps
        m = m.reshape(self.grid_size, self.grid_size)

        # get mean and std of the resource map
        mean, std = jnp.mean(m), jnp.std(m)
        m = (m - mean) / std  # normalize

        food_map = (m >= 1.0).astype(jnp.int8)
        poison_map = (m <= -1.0).astype(jnp.int8)
        buffer_map = ((m > -1.0) & (m < -0.5)).astype(jnp.int8)
        resource_map = self.make_resource_map(food_map, poison_map)

        ego_agent_loc, ego_agent_dir, demonstrator_locs, demonstrator_dirs = (
            self.spawn_agents(key, buffer_map)
        )

        state = EnvState(
            time=0,
            ego_agent_energy=params.initial_energy,
            ego_agent_loc=ego_agent_loc,
            ego_agent_dir=ego_agent_dir,
            demonstrator_locs=demonstrator_locs,
            demonstrator_dirs=demonstrator_dirs,
            resource_map=resource_map,
            consumed_counts=jnp.zeros(2),
        )
        obs = self.get_obs(state, params)
        return obs, state

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams
    ) -> Tuple[Obs, EnvState, float, bool, Dict[Any, Any]]:
        # sample demonstrator actions
        key, key_dem = jax.random.split(key)
        key_dems = jax.random.split(key_dem, self.num_demonstrators)
        demonstrator_actions = jax.vmap(
            self.sample_demonstrator_action, in_axes=(0, None, 0)
        )(key_dems, state, jnp.arange(self.num_demonstrators))

        # action dynamics: movement
        all_locs = jnp.concatenate([state.ego_agent_loc[None], state.demonstrator_locs])
        all_dirs = jnp.concatenate([state.ego_agent_dir[None], state.demonstrator_dirs])
        all_actions = jnp.concatenate([jnp.array([action]), demonstrator_actions])
        new_locs, new_dirs, energy_losses = jax.vmap(
            self.handle_movement, in_axes=(0, 0, 0, None)
        )(all_locs, all_dirs, all_actions, params)

        # action dynamics: eating
        new_resource_map, energy_gain_vec, consumed_vec = self.handle_eat(
            all_locs, all_actions, state.resource_map, params
        )
        energy_gain, consumed = energy_gain_vec[0], consumed_vec[0]
        delta_energy = energy_gain - energy_losses[0]

        # resource dynamics
        key_food, key_poison = jax.random.split(key)
        food_map = (new_resource_map == 1).astype(jnp.int8)
        poison_map = (new_resource_map == -1).astype(jnp.int8)
        new_food_map = self.update_resources(
            key_food,
            food_map,
            params.adj_growth_rate,
            params.spont_growth_rate,
            params.death_rate,
        )
        new_poison_map = self.update_resources(
            key_poison,
            poison_map,
            params.adj_growth_rate,
            params.spont_growth_rate,
            params.death_rate,
        )
        new_resource_map = self.make_resource_map(new_food_map, new_poison_map)

        food_eaten = (consumed == 1).astype(jnp.int8)
        poison_eaten = (consumed == -1).astype(jnp.int8)
        new_consumed_counts = state.consumed_counts + jnp.array(
            [food_eaten, poison_eaten]
        )

        new_state = state.replace(
            resource_map=new_resource_map,
            ego_agent_loc=new_locs[0],
            ego_agent_dir=new_dirs[0],
            ego_agent_energy=jnp.clip(
                state.ego_agent_energy + delta_energy, 0, params.max_energy
            ),
            consumed_counts=new_consumed_counts,
            demonstrator_locs=new_locs[1:],
            demonstrator_dirs=new_dirs[1:],
        )
        delta_energy = new_state.ego_agent_energy - state.ego_agent_energy

        return (
            self.get_obs(new_state, params),
            new_state,
            self.compute_reward(new_state, params, delta_energy),
            self.is_terminal(new_state, params),
            {},
        )

    def handle_movement(
        self, loc: chex.Array, dir: chex.Array, action: chex.Array, params: EnvParams
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        delta = DIRECTION_TO_DELTA[dir]
        is_fwd = action == Action.FORWARD.value
        new_loc = jnp.where(is_fwd, (loc + delta) % self.grid_size, loc)
        new_dir = (
            dir
            + (action == Action.ROTATE_RIGHT.value)
            - (action == Action.ROTATE_LEFT.value)
        ) % 4
        energy_loss = params.base_energy_loss + is_fwd * params.move_energy_loss
        return new_loc, new_dir, energy_loss

    def handle_eat(self, locs, actions, resource_map, params):
        is_eat = (actions == Action.EAT.value).astype(jnp.int8)
        r, c = locs[:, 0], locs[:, 1]
        gathered = resource_map[r, c]
        consumed = gathered * is_eat  # 0 ⇒ nothing eaten
        energy_gain = consumed * params.resource_value

        def scatter_zero(rm, rc):
            return rm.at[rc[0], rc[1]].set(0)

        def no_op(rm, _):
            return rm

        def body(rm, rc):
            return jax.lax.cond(
                consumed[rc[2]] > 0, scatter_zero, no_op, rm, rc[:2]  # rc = (r, c, idx)
            )

        idxs = jnp.arange(locs.shape[0], dtype=jnp.int8)
        rc_triplet = jnp.stack([r, c, idxs], axis=1)

        new_resource_map = jax.lax.fori_loop(
            0, locs.shape[0], lambda i, rm: body(rm, rc_triplet[i]), resource_map
        )

        return new_resource_map, energy_gain, consumed

    def get_obs(self, state: EnvState, params: EnvParams) -> Obs:
        agent_map = self.blank_obs_map.at[
            state.demonstrator_locs[:, 0], state.demonstrator_locs[:, 1]
        ].set(1)
        agent_map = agent_map.at[state.ego_agent_loc[0], state.ego_agent_loc[1]].set(
            state.ego_agent_energy / params.max_energy
        )

        ego_r, ego_c = state.ego_agent_loc
        resource_vis_r = ego_r + self.resource_vis_indices_row
        resource_vis_c = ego_c + self.resource_vis_indices_col
        dem_vis_r = ego_r + self.demonstrator_vis_indices_row
        dem_vis_c = ego_c + self.demonstrator_vis_indices_col

        resource_view = state.resource_map[resource_vis_r, resource_vis_c]
        agent_view = agent_map[dem_vis_r, dem_vis_c]

        # pad
        resource_pad = self.max_vis_range - self.resource_vis_range
        dem_pad = self.max_vis_range - self.demonstrator_vis_range
        resource_view = jnp.pad(
            resource_view,
            ((resource_pad, resource_pad), (resource_pad, resource_pad)),
            mode="constant",
        )
        agent_view = jnp.pad(
            agent_view, ((dem_pad, dem_pad), (dem_pad, dem_pad)), mode="constant"
        )

        obs = jnp.stack([resource_view, agent_view], axis=-1)

        rot0 = lambda x: x
        rot1 = lambda x: jnp.rot90(x, k=1, axes=(0, 1))
        rot2 = lambda x: jnp.rot90(x, k=2, axes=(0, 1))
        rot3 = lambda x: jnp.rot90(x, k=3, axes=(0, 1))

        return jax.lax.switch(state.ego_agent_dir, (rot0, rot1, rot2, rot3), obs)

    def compute_reward(
        self, state: EnvState, params: EnvParams, energy_delta: chex.Array
    ) -> chex.Array:
        return energy_delta
        # if the ego agent energy is above threshold, give a reward of 0.1
        # otherwise, if the ego agent has just consumed a resource, give a supplementary reward of 0.01
        # base_reward = 0.1 * (state.ego_agent_energy >= params.energy_threshold).astype(
        #     jnp.float32
        # )
        # extra_reward = 0.01 * (energy_delta > 0).astype(jnp.float32)
        # return jnp.maximum(base_reward, extra_reward)

    def is_terminal(self, state: EnvState, params: EnvParams) -> chex.Array:  # type: ignore
        """Check whether state is terminal."""
        time_limit_reached = state.time >= params.max_steps_in_episode  # type: ignore
        agent_dead = state.ego_agent_energy <= 0  # type: ignore
        return jnp.logical_or(time_limit_reached, agent_dead)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.obs_size, self.obs_size, 2),
            dtype=jnp.int8,
        )

    def update_resources(
        self,
        key: chex.PRNGKey,
        current: chex.Array,
        p_adj: float,
        p_spont: float,
        p_death: float,
    ) -> chex.Array:
        neighbours = get_n_neighbours(current)

        p_adjs = p_adj * neighbours
        p_deaths = p_death * current * (8 - neighbours) / 8

        u = jax.random.uniform(key, shape=(3,) + current.shape)
        deaths = (u[0] < p_deaths).astype(jnp.int8)
        births_adj = (u[1] < p_adjs).astype(jnp.int8)
        births_spnt = (u[2] < p_spont).astype(jnp.int8)

        delta = births_adj + births_spnt - deaths
        return jnp.clip(current + delta, 0, 1).astype(jnp.int8)

    def spawn_agents(
        self, key: chex.PRNGKey, buffer_map: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        ego_key, dem_key = jax.random.split(key)

        # spawn ego agent into random location such that buffer_map is 1
        probs = buffer_map.reshape(-1) / buffer_map.sum()
        idx = jax.random.choice(key, probs.shape[0], p=probs)
        i, j = divmod(idx, buffer_map.shape[1])
        ego_agent_loc = jnp.array([i, j])

        ego_agent_dir = jax.random.randint(ego_key, (), 0, 4)
        demonstrator_locs = jax.random.randint(
            dem_key, (self.num_demonstrators, 2), 0, self.grid_size
        )
        demonstrator_dirs = jax.random.randint(dem_key, (self.num_demonstrators,), 0, 4)
        return ego_agent_loc, ego_agent_dir, demonstrator_locs, demonstrator_dirs

    def sample_demonstrator_action(
        self, key: chex.PRNGKey, env_state: EnvState, idx: int, epsilon: float = 0.1
    ) -> chex.Array:
        loc, dir_ = env_state.demonstrator_locs[idx], env_state.demonstrator_dirs[idx]

        # get manhattan distances to all grid tiles
        r, c = jnp.indices((self.grid_size, self.grid_size))
        dists = jnp.abs(r - loc[0]) + jnp.abs(c - loc[1])

        # find the closest food resource
        dists = jnp.where(env_state.resource_map == 1, dists, jnp.inf)
        target = jnp.unravel_index(jnp.argmin(dists), dists.shape)

        action = self.move_toward_or_eat(loc, dir_, target)
        random_action = jax.random.randint(key, (), 0, len(Action))

        return jax.lax.select(jax.random.uniform(key) < 0.1, random_action, action)

    def move_toward_or_eat(
        self, loc: chex.Array, dir: chex.Array, target: chex.Array
    ) -> chex.Array:
        # get the direction to the target
        d_row, d_col = target[0] - loc[0], target[1] - loc[1]
        same_cell = (d_row == 0) & (d_col == 0)

        # pick the axis with the larger |delta|
        choose_row = jnp.abs(d_row) > jnp.abs(d_col)
        desired_dir_row = jnp.where(
            d_row > 0, Direction.SOUTH.value, Direction.NORTH.value
        )
        desired_dir_col = jnp.where(
            d_col > 0, Direction.EAST.value, Direction.WEST.value
        )
        desired_dir = jnp.where(choose_row, desired_dir_row, desired_dir_col)

        # select the action
        # diff : 0 => already facing  | 1 => 90° right  | 3 => 90° left | 2 => reverse
        diff = (desired_dir - dir) & 3  # mod 4
        return jnp.select(
            [
                same_cell,  # 0. standing on the food
                diff == 0,  # 1. go forward
                diff == 1,  # 2. rotate right
                diff == 3,  # 3. rotate left
            ],
            [
                Action.EAT.value,
                Action.FORWARD.value,
                Action.ROTATE_RIGHT.value,
                Action.ROTATE_LEFT.value,
            ],
            default=Action.ROTATE_RIGHT.value,  # diff==2 → pick one turn
        )

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer(
            (self.grid_size, self.grid_size),
            self.resource_vis_range,
            self.demonstrator_vis_range,
        )
        self.rendering_initialized = True

    def render(self, state: EnvState, mode="human"):
        if not self.rendering_initialized:
            self._init_render()
        if self.viewer is None:
            return
        return_rgb = mode == "rgb_array"
        return self.viewer.render(state, return_rgb_array=return_rgb)

    def close(self):
        if self.viewer:
            self.viewer.close()


def make(
    grid_size: int,
    num_demonstrators: int,
    resource_vis_range: int,
    demonstrator_vis_range: int,
):
    env = SimpleForagingEnv(
        grid_size, num_demonstrators, resource_vis_range, demonstrator_vis_range
    )
    return env, env.default_params
