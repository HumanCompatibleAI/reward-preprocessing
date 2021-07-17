import gym

from .env_ingredient import create_env, env_ingredient
from .maze import MazeEnv  # noqa: F401

gym.envs.register(id="EmptyMaze-v0", entry_point=MazeEnv, max_episode_steps=200)

__all__ = ["create_env", "env_ingredient"]
