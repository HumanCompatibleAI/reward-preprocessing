import gym
import gym_minigrid  # noqa: F401
from gym_minigrid.wrappers import ImgObsWrapper
from sacred import Ingredient
from stable_baselines3.common.vec_env import DummyVecEnv

env_ingredient = Ingredient("env")


def _make_env(name: str) -> gym.Env:
    env = gym.make(name)
    env = ImgObsWrapper(env)
    return env


@env_ingredient.config
def config():
    name = "MiniGrid-Empty-Random-6x6-v0"
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.capture
def create_env(name: str):
    env = DummyVecEnv([lambda: _make_env(name)])
    return env
