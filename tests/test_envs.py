import gym
import gym_minigrid  # noqa: F401
from gym_minigrid.wrappers import FlatObsWrapper, FullyObsWrapper
import numpy as np


def test_random_minigrid():
    env = gym.make("MiniGrid-Empty-Random-6x6-v0")
    env = FullyObsWrapper(env)
    env = FlatObsWrapper(env)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, int))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    env.close()
