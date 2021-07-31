import gym
import numpy as np

# register the Maze env
import reward_preprocessing.env  # noqa: F401


def test_empty_maze():
    env = gym.make("EmptyMaze-v0")

    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    env.close()


def test_mujoco():
    env = gym.make("HalfCheetah-v3")

    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    env.close()


def test_mock_env(mock_env):
    env = mock_env
    assert env.reset() == 5
    for _ in range(20):
        # the mock env is inside a DummyVecEnv, so we need a list of actions
        action = [env.action_space.sample()]
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, np.ndarray)
        assert isinstance(done, np.ndarray)
        assert isinstance(info, list)
        assert len(obs) == 1
        assert len(reward) == 1
        assert len(done) == 1
        assert len(info) == 1
    env.close()
