import jax

from .simple import SimpleForagingEnv, EnvState


def track_resource_consumption(env: SimpleForagingEnv, final_state: EnvState):
    return {
        "food_consumed": final_state.consumed_counts[0],
        "poison_consumed": final_state.consumed_counts[1],
    }
