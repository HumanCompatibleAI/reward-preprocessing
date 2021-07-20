from typing import Any, Mapping

import gym
from sacred import Ingredient
from stable_baselines3.common.vec_env import DummyVecEnv

env_ingredient = Ingredient("env")


@env_ingredient.config
def config():
    name = "EmptyMaze-v0"
    options = {"size": 6, "random_start": True}
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.capture
def create_env(name: str, _seed: int, options: Mapping[str, Any]):
    env = DummyVecEnv([lambda: gym.make(name, **options)])
    env.seed(_seed)
    # the action space uses a distinct random seed from the environment
    # itself, which is important if we use randomly sampled actions
    env.action_space.np_random.seed(_seed)
    return env
