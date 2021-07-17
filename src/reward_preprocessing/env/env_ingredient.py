import gym
from sacred import Ingredient
from stable_baselines3.common.vec_env import DummyVecEnv

env_ingredient = Ingredient("env")


@env_ingredient.config
def config():
    name = "EmptyMaze-v0"
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.capture
def create_env(name: str, _seed: int):
    env = DummyVecEnv([lambda: gym.make(name)])
    env.seed(_seed)
    return env
