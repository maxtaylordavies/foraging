import time

import jax

from src.environments.single_agent.simple import make

key = jax.random.PRNGKey(0)
env, env_params = make(grid_size=20, num_demonstrators=10, obs_size=2)

obs, env_state = env.reset(key, env_params)

for _ in range(100):
    env.render(env_state)
    time.sleep(0.1)

    key, key_act, key_step = jax.random.split(key, 3)
    action = env.action_space(env_params).sample(key_act)

    obs, env_state, reward, done, info = env.step(
        key_step, env_state, action, env_params
    )
