from typing import Callable

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

# TODO: we could use a typing.Protocol here to make sure the factory
# takes a `_seed` argument. On the other hand, type annotations with
# Sacred ingredients are a mess anyway (and venv_factory will typically
# be create_env).
VenvFactory = Callable[..., VecEnv]
