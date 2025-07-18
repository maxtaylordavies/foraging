import time

import jax
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.environments.single_agent.simple import make
from src.algos.ppo_cnn import train_ppo_cnn

key = jax.random.PRNGKey(0)
env, env_params = make(
    grid_size=30, num_demonstrators=10, resource_vis_range=1, demonstrator_vis_range=4
)

# _, train_df, _ = train_ppo_cnn(key, env, env_params)

# fig, axs = plt.subplots(1, 2)
# sns.lineplot(data=train_df, x="timestep", y="return", ax=axs[0])
# sns.lineplot(data=train_df, x="timestep", y="length", ax=axs[1])
# plt.show()

# train_df["total_consumed"] = train_df["food_consumed"] + train_df["poison_consumed"]
# train_df["prop_food_consumed"] = train_df["food_consumed"] / train_df["total_consumed"]
# train_df["prop_poison_consumed"] = (
#     train_df["poison_consumed"] / train_df["total_consumed"]
# )

# fig, ax = plt.subplots()
# sns.lineplot(data=train_df, x="timestep", y="food_consumed", ax=ax)
# plt.show()

# fig, ax = plt.subplots()
# sns.lineplot(data=train_df, x="timestep", y="poison_consumed", ax=ax)
# plt.show()

# fig, ax = plt.subplots()
# sns.lineplot(data=train_df, x="timestep", y="prop_food_consumed", ax=ax)
# sns.lineplot(data=train_df, x="timestep", y="prop_poison_consumed", ax=ax)
# plt.legend(["Food", "Poison"])
# plt.show()

# for i in range(100):
#     obs, env_state = env.reset(key, env_params)
#     env.render(env_state)
#     time.sleep(1.0)
#     key, _ = jax.random.split(key)

# print(
#     f"Average Food: {sum(foods) / len(foods):.2f}, Average Poison: {sum(poisons) / len(poisons):.2f}"
# )

obs, env_state = env.reset(key, env_params)
for t in range(500):
    # if t % 20 == 0:
    env.render(env_state)

    key, key_act, key_step = jax.random.split(key, 3)
    action = env.action_space(env_params).sample(key_act)

    obs, env_state, reward, done, info = env.step(
        key_step, env_state, action, env_params
    )

    # print(env_state.ego_agent_energy, reward)

# num_steps = int(1e5)
# start_time = time.time()
# obs, env_state = env.reset(key, env_params)
# for t in tqdm(range(num_steps)):
#     key, key_act, key_step = jax.random.split(key, 3)
#     action = env.action_space(env_params).sample(key_act)

#     obs, env_state, reward, done, info = env.step(
#         key_step, env_state, action, env_params
#     )

# end_time = time.time()
# print(f"Time taken for {num_steps} steps: {end_time - start_time:.2f} seconds")
# print(f"Steps per second: {num_steps / (end_time - start_time):.2f}")
