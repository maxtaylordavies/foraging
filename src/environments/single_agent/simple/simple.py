from enum import Enum
from typing import Any, Dict, Optional, Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from ..environment import (
    Environment,
    EnvState as BaseEnvState,
    EnvParams as BaseEnvParams,
)

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


DIRECTION_TO_DELTA = [
    jnp.array([0, 1]),
    jnp.array([1, 0]),
    jnp.array([0, -1]),
    jnp.array([-1, 0]),
]


@struct.dataclass
class EnvState(BaseEnvState):
    ego_agent_energy: chex.Array
    ego_agent_loc: chex.Array
    ego_agent_dir: chex.Array
    demonstrator_locs: chex.Array
    demonstrator_dirs: chex.Array
    resource_map: chex.Array


@struct.dataclass
class EnvParams(BaseEnvParams):
    initial_p_resource: float = 0.05
    spont_growth_rate: float = 0.001
    adj_growth_rate: float = 0.005
    death_rate: float = 0.03
    resource_value: float = 1.0
    energy_decay_rate: float = 0.05
    max_energy: float = 10.0
    initial_energy: float = 5.0
    energy_threshold: float = 5.0
    reward_size: float = 0.1
    penalty_size: float = 0.0
    max_steps_in_episode: int = 1000


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

        self.max_vis_range = max(resource_vis_range, demonstrator_vis_range)
        self.obs_size = (2 * self.max_vis_range) + 1

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

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[Obs, EnvState]:
        resource_map = self.update_resources(
            key,
            jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32),
            0.0,
            params.initial_p_resource,
            0.0,
        )
        ego_agent_loc, ego_agent_dir, demonstrator_locs, demonstrator_dirs = (
            self.spawn_agents(key)
        )
        state = EnvState(
            time=0,
            ego_agent_energy=params.initial_energy,
            ego_agent_loc=ego_agent_loc,
            ego_agent_dir=ego_agent_dir,
            demonstrator_locs=demonstrator_locs,
            demonstrator_dirs=demonstrator_dirs,
            resource_map=resource_map,
        )
        obs = self.get_obs(state, params)
        return obs, state

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Action, params: EnvParams
    ) -> Tuple[Obs, EnvState, float, bool, Dict[Any, Any]]:
        # handle action
        new_loc, new_dir, delta_energy, new_resource_map = self.handle_action(
            state.ego_agent_loc, state.ego_agent_dir, action, state.resource_map, params
        )

        # resource dynamics
        new_resource_map = self.update_resources(
            key,
            new_resource_map,
            params.adj_growth_rate,
            params.spont_growth_rate,
            params.death_rate,
        )

        new_state = state.replace(
            resource_map=new_resource_map,
            ego_agent_loc=new_loc,
            ego_agent_dir=new_dir,
            ego_agent_energy=jnp.clip(
                state.ego_agent_energy + delta_energy, 0, params.max_energy
            ),
        )
        return (
            self.get_obs(new_state, params),
            new_state,
            self.compute_reward(new_state, params),
            self.is_terminal(new_state, params),
            {},
        )

    def handle_action(
        self,
        loc: chex.Array,
        dir: chex.Array,
        action: Action,
        resource_map: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        delta = jnp.select([dir == k for k in range(4)], DIRECTION_TO_DELTA)
        new_loc = jax.lax.select(
            action == Action.FORWARD.value,
            (loc + delta) % self.grid_size,
            loc,
        )
        new_dir = jax.lax.select(
            action == Action.ROTATE_LEFT.value,
            (dir - 1) % 4,
            jax.lax.select(
                action == Action.ROTATE_RIGHT.value,
                (dir + 1) % 4,
                dir,
            ),
        )

        resource_consumed = (action == Action.EAT.value) & (
            resource_map[loc[0], loc[1]] > 0
        )
        delta_energy = -params.energy_decay_rate + (
            resource_consumed * params.resource_value
        )

        new_resource_map = jax.lax.select(
            resource_consumed,
            resource_map.at[loc[0], loc[1]].set(0),
            resource_map,
        )

        return new_loc, new_dir, delta_energy, new_resource_map

    def get_obs(self, state: EnvState, params: EnvParams) -> Obs:
        obs = jnp.zeros((self.obs_size, self.obs_size, 2))

        demonstrator_map = jnp.zeros((self.obs_size, self.obs_size))
        for i, loc in enumerate(state.demonstrator_locs):
            demonstrator_map = demonstrator_map.at[loc[0], loc[1]].set(1)

        ego_r, ego_c = state.ego_agent_loc
        resource_vis_r = ego_r + self.resource_vis_indices_row
        resource_vis_c = ego_c + self.resource_vis_indices_col
        dem_vis_r = ego_r + self.demonstrator_vis_indices_row
        dem_vis_c = ego_c + self.demonstrator_vis_indices_col

        resource_view = state.resource_map[resource_vis_r, resource_vis_c]
        demonstrator_view = demonstrator_map[dem_vis_r, dem_vis_c]

        # pad
        resource_pad = self.max_vis_range - self.resource_vis_range
        dem_pad = self.max_vis_range - self.demonstrator_vis_range
        resource_view = jnp.pad(
            resource_view,
            ((resource_pad, resource_pad), (resource_pad, resource_pad)),
            mode="constant",
        )
        demonstrator_view = jnp.pad(
            demonstrator_view, ((dem_pad, dem_pad), (dem_pad, dem_pad)), mode="constant"
        )

        obs = obs.at[:, :, 0].set(resource_view)
        obs = obs.at[:, :, 1].set(demonstrator_view)

        return jnp.select(
            [state.ego_agent_dir == direction for direction in range(4)],
            [jnp.rot90(obs, k=k, axes=(0, 1)) for k in range(4)],
        )

    def compute_reward(self, state: EnvState, params: EnvParams) -> chex.Array:
        return jnp.where(
            state.ego_agent_energy >= params.energy_threshold,
            params.reward_size,
            params.penalty_size,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> chex.Array:  # type: ignore
        """Check whether state is terminal."""
        return state.time >= params.max_steps_in_episode  # type: ignore

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        raise NotImplementedError

    def update_resources(
        self,
        key: chex.PRNGKey,
        current: chex.Array,
        p_adj: float,
        p_spont: float,
        p_death: float,
    ) -> chex.Array:
        p_deaths = p_death * current
        current -= jax.random.bernoulli(key, p_deaths, current.shape).astype(jnp.int32)

        p_adjs = p_adj * self.get_n_neighbours(current)
        adj_growth = jax.random.bernoulli(
            key, p_adjs, (self.grid_size, self.grid_size)
        ).astype(jnp.int32)

        spont_growth = jax.random.bernoulli(
            key, p_spont, (self.grid_size, self.grid_size)
        ).astype(jnp.int32)

        return jnp.clip(current + adj_growth + spont_growth, 0, 1).astype(jnp.int32)

    def spawn_agents(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        ego_key, dem_key = jax.random.split(key)
        ego_agent_loc = jax.random.randint(ego_key, (2,), 0, self.grid_size)
        ego_agent_dir = jax.random.randint(ego_key, (), 0, 4)
        demonstrator_locs = jax.random.randint(
            dem_key, (self.num_demonstrators, 2), 0, self.grid_size
        )
        demonstrator_dirs = jax.random.randint(
            dem_key, (self.num_demonstrators, 2), 0, 4
        )
        return ego_agent_loc, ego_agent_dir, demonstrator_locs, demonstrator_dirs

    def get_n_neighbours(self, x: chex.Array) -> chex.Array:
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
        )

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.grid_size, self.grid_size))
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
