import os
from typing import Any, Tuple
import shutil

import chex
from gymnax.environments.environment import Environment, EnvParams
import jax
import pandas as pd
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp
from rejax.algos.algorithm import Algorithm, register_init
from rejax.networks import DiscretePolicy, VNetwork
import orbax.checkpoint

from src.algos.mixins import (
    NormalizeObservationsMixin,
    OnPolicyMixin,
)
from src.algos.networks import CNN, ScannedRNN
from src.evaluate import evaluate, Policy

from src.environments.single_agent.simple.evals import track_resource_consumption

# from src.utils import recursive_dict_to_train_state


def _make_policy(algo, ts):
    def act(rng, obs, state, extra):
        if getattr(algo, "normalize_observations", False):
            obs = algo.normalize_obs(ts.rms_state, obs)

        obs = jnp.expand_dims(obs, 0)
        hidden, feats = algo.process_obs(
            obs, extra["done"], extra["hidden"], ts.cnn_ts.params, ts.rnn_ts.params
        )
        feats = jnp.expand_dims(feats, 0)

        action = algo.actor.apply(ts.actor_ts.params, feats, rng, method="act")
        return jnp.squeeze(action), {**extra, "hidden": hidden}

    init_extra = {
        "hidden": ScannedRNN.initialize_carry(1, 32),
        "done": jnp.zeros(1, dtype=jnp.bool_),
    }

    return act, init_extra


class Trajectory(struct.PyTreeNode):
    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    value: chex.Array
    done: chex.Array
    hidden: chex.Array


class AdvantageMinibatch(struct.PyTreeNode):
    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class PPO_RNN(OnPolicyMixin, NormalizeObservationsMixin, Algorithm):
    cnn: nn.Module = struct.field(pytree_node=False, default=None)
    rnn: nn.Module = struct.field(pytree_node=False, default=None)
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=8)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)

    def make_policy(self, ts):
        return _make_policy(self, ts)

    @classmethod
    def create(cls, **config):
        env, env_params = cls.create_env(config)
        agent = cls.create_agent(config, env, env_params)

        def eval_callback(algo, ts, rng):
            policy, init_extra = algo.make_policy(ts)
            max_steps = algo.env_params.max_steps_in_episode
            lengths, returns, metrics = evaluate(
                policy,
                rng,
                env,
                env_params,
                track_resource_consumption,
                init_extra=init_extra,
                max_steps_in_episode=max_steps,
            )
            consumed = metrics["consumed_count"]
            jax.debug.print(
                "iter: {}, mean return: {}, mean length: {}, mean consumed: {}",
                ts.global_step,
                returns.mean(),
                lengths.mean(),
                consumed.mean(),
            )
            return returns, lengths, consumed

        return cls(
            env=env,
            env_params=env_params,
            eval_callback=eval_callback,
            **agent,
            **config,
        )

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        cnn = CNN(features=32)
        rnn = ScannedRNN()
        actor = DiscretePolicy(action_space.n, **agent_kwargs)
        critic = VNetwork(**agent_kwargs)
        return {"actor": actor, "critic": critic, "cnn": cnn, "rnn": rnn}

    @register_init
    def initialize_network_params(self, rng):
        rng, rng_cnn, rng_rnn, rng_actor, rng_critic = jax.random.split(rng, 5)
        init_obs = jnp.empty([1, *self.env.observation_space(self.env_params).shape])
        init_cnn_out = jnp.empty([1, 1, 32])
        init_dones = jnp.empty([1, 1])
        init_hidden = ScannedRNN.initialize_carry(1, 32)
        init_rnn_out = jnp.empty([1, 32])

        cnn_params = self.cnn.init(rng_cnn, init_obs)
        rnn_params = self.rnn.init(rng_rnn, init_hidden, (init_cnn_out, init_dones))
        actor_params = self.actor.init(rng_actor, init_rnn_out, rng_actor)
        critic_params = self.critic.init(rng_critic, init_rnn_out)

        tx = optax.chain(
            optax.clip_by_global_norm(0.5), optax.adam(learning_rate=self.learning_rate)
        )
        cnn_ts = TrainState.create(apply_fn=(), params=cnn_params, tx=tx)
        rnn_ts = TrainState.create(apply_fn=(), params=rnn_params, tx=tx)
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)

        return {
            "actor_ts": actor_ts,
            "critic_ts": critic_ts,
            "cnn_ts": cnn_ts,
            "rnn_ts": rnn_ts,
        }

    @register_init
    def initialize_additional_inputs(self, rng):
        init_hidden = ScannedRNN.initialize_carry(self.num_envs, 32)
        return {"last_hidden": init_hidden}

    def process_obs(self, obs, done, hidden, cnn_params, rnn_params):
        cnn_feats = self.cnn.apply(cnn_params, obs)
        rnn_inputs = (jnp.expand_dims(cnn_feats, axis=1), jnp.expand_dims(done, axis=1))

        print(
            f"rnn_inputs[0]: {rnn_inputs[0].shape}, rnn_inputs[1]: {rnn_inputs[1].shape}, hidden: {hidden.shape}"
        )

        hidden, out = self.rnn.apply(rnn_params, hidden, rnn_inputs)
        return hidden, out.squeeze()

    def train_iteration(self, ts):
        ts, trajectories = self.collect_trajectories(ts)
        _, last_feats = self.process_obs(
            ts.last_obs,
            ts.last_done,
            ts.last_hidden,
            ts.cnn_ts.params,
            ts.rnn_ts.params,
        )
        last_val = self.critic.apply(ts.critic_ts.params, last_feats)
        last_val = jnp.where(ts.last_done, 0, last_val)
        advantages, targets = self.calculate_gae(trajectories, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def collect_trajectories(self, ts):
        def env_step(ts, unused):
            # Get keys for sampling action and stepping environment
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)

            # Sample action
            hidden, feats = self.process_obs(
                ts.last_obs,
                ts.last_done,
                ts.last_hidden,
                ts.cnn_ts.params,
                ts.rnn_ts.params,
            )

            action, log_prob = self.actor.apply(
                ts.actor_ts.params, feats, rng_action, method="action_log_prob"
            )
            value = self.critic.apply(ts.critic_ts.params, feats)

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, _ = t

            if self.normalize_observations:
                rms_state, next_obs = self.update_and_normalize(ts.rms_state, next_obs)
                ts = ts.replace(rms_state=rms_state)

            # Return updated runner state and transition
            transition = Trajectory(
                ts.last_obs, action, log_prob, reward, value, done, hidden
            )
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                last_hidden=hidden,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)
        return ts, trajectories

    def calculate_gae(self, trajectories, last_val):
        def get_advantages(advantage_and_next_value, transition):
            advantage, next_value = advantage_and_next_value
            delta = (
                transition.reward.squeeze()  # For gymnax envs that return shape (1, )
                + self.gamma * next_value * (1 - transition.done)
                - transition.value
            )
            advantage = (
                delta + self.gamma * self.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.value), advantage

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajectories,
            reverse=True,
        )
        return advantages, advantages + trajectories.value

    def update_actor(self, ts, batch):
        def loss_fn(cnn_params, rnn_params, actor_params):
            _, feats = self.process_obs(
                batch.trajectories.obs,
                batch.trajectories.done,
                batch.trajectories.hidden,
                cnn_params,
                rnn_params,
            )
            log_prob, entropy = self.actor.apply(
                actor_params,
                feats,
                batch.trajectories.action,
                method="log_prob_entropy",
            )
            entropy = entropy.mean()

            # Calculate actor loss
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()
            return pi_loss - self.ent_coef * entropy

        cnn_grads, rnn_grads, actor_grads = jax.grad(loss_fn, argnums=(0, 1, 2))(
            ts.cnn_ts.params, ts.rnn_ts.params, ts.actor_ts.params
        )

        return ts.replace(
            cnn_ts=ts.cnn_ts.apply_gradients(grads=cnn_grads),
            rnn_ts=ts.rnn_ts.apply_gradients(grads=rnn_grads),
            actor_ts=ts.actor_ts.apply_gradients(grads=actor_grads),
        )

    def update_critic(self, ts, batch):
        def loss_fn(cnn_params, rnn_params, critic_params):
            _, feats = self.process_obs(
                batch.trajectories.obs,
                batch.trajectories.done,
                batch.trajectories.hidden,
                cnn_params,
                rnn_params,
            )
            value = self.critic.apply(critic_params, feats)
            value_pred_clipped = batch.trajectories.value + (
                value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return self.vf_coef * value_loss

        cnn_grads, rnn_grads, critic_grads = jax.grad(loss_fn, argnums=(0, 1, 2))(
            ts.cnn_ts.params, ts.rnn_ts.params, ts.critic_ts.params
        )

        return ts.replace(
            cnn_ts=ts.cnn_ts.apply_gradients(grads=cnn_grads),
            rnn_ts=ts.rnn_ts.apply_gradients(grads=rnn_grads),
            critic_ts=ts.critic_ts.apply_gradients(grads=critic_grads),
        )

    def update(self, ts, batch):
        ts = self.update_actor(ts, batch)
        ts = self.update_critic(ts, batch)
        return ts


def train_ppo_rnn(
    key: chex.PRNGKey, env: Environment, env_params: EnvParams
) -> Tuple[Policy, pd.DataFrame, Any]:
    algo = PPO_RNN.create(
        env=env,
        env_params=env_params,
        total_timesteps=int(2e6),
        eval_freq=1e5,
        num_envs=16,
        num_steps=1024,
        num_epochs=4,
        num_minibatches=8,
        max_grad_norm=0.5,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        normalize_observations=False,
    )

    train_fn = jax.jit(algo.train)
    train_state, evaluation = jax.block_until_ready(train_fn(key))

    ep_returns, ep_lengths, consumed_counts = evaluation
    mean_returns, mean_lengths, mean_consumed = (
        ep_returns.mean(axis=1),
        ep_lengths.mean(axis=1),
        consumed_counts.mean(axis=1),
    )
    train_df = pd.DataFrame(
        {
            "timestep": jnp.linspace(0, algo.total_timesteps, len(mean_returns)),
            "return": mean_returns,
            "length": mean_lengths,
            "consumed": mean_consumed,
        }
    )

    policy = algo.make_policy(train_state)
    # policy: Policy = jax.jit(lambda key, obs, state, extra: (_policy(obs, key), extra))

    return policy, train_df, train_state


# def load_policy_from_checkpoint(
#     checkpoint_path: str, env: Environment, env_params: EnvParams
# ) -> Policy:
#     checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#     train_state = checkpointer.restore(checkpoint_path)
#     train_state = recursive_dict_to_train_state(train_state)
#     algo = PPO.create(env=env, env_params=env_params)
#     _policy = _make_policy(algo, train_state)
#     return jax.jit(lambda key, obs, state, extra: (_policy(obs, key), extra))
