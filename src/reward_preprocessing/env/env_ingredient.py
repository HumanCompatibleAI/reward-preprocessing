from typing import Any, Mapping

import gym
from sacred import Ingredient
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env_ingredient = Ingredient("env")


@env_ingredient.config
def config():
    name = "EmptyMaze-v0"
    options = {}
    stats_path = None
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def empty_maze():
    # this is currently the default anyway, but it's needed to
    # make the wrapper script work
    name = "EmptyMaze-v0"
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def mountain_car():
    name = "MountainCar-v0"
    stats_path = "results/stats/MountainCar-v0.pkl"
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.capture
def create_env(name: str, _seed: int, options: Mapping[str, Any], stats_path: str):
    env = DummyVecEnv([lambda: gym.make(name, **options)])
    env.seed(_seed)
    # the action space uses a distinct random seed from the environment
    # itself, which is important if we use randomly sampled actions
    env.action_space.np_random.seed(_seed)

    if stats_path:
        env = VecNormalize.load(stats_path, env)
        # Deactivate training and reward normalization
        env.training = False
        env.norm_reward = False

    return env
