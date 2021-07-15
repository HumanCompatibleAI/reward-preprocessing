import gym
import numpy as np
import pytest


class MockEnv(gym.Env):
    """Extremely simple deterministic environment for testing purposes.

    It implements a 1D grid from 0 to 10: the agent starts at 5, can move left or right,
    and the episode ends when it reaches 0 or 10.
    """

    def __init__(self):
        super().__init__()
        self.pos = 5
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(11)

    def step(self, action):
        assert self.action_space.contains(action), "Action must be 0 or 1"
        action = 2 * action - 1
        self.pos += action
        done = self.pos <= 0 or self.pos >= 10
        if done and self.pos == 0:
            reward = -1.0
        elif done and self.pos == 10:
            reward = 1.0
        else:
            reward = 0.0
        info = {}

        return self.pos, reward, done, info

    def seed(self, seed=None):
        pass

    def reset(self):
        self.pos = 5
        return self.pos

    def render(self):
        raise NotImplementedError()

    def close(self):
        pass


gym.envs.register(id="MockEnv-v0", entry_point=MockEnv, max_episode_steps=100)


@pytest.fixture
def env():
    """Return a gym environment."""
    from reward_preprocessing.env import create_env

    env = create_env("MiniGrid-Empty-Random-6x6-v0")
    yield env
    env.close()


@pytest.fixture
def mock_env():
    """Return a very simple mock gym environment."""

    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: gym.make("MockEnv-v0")])
    yield env
    env.close()


@pytest.fixture
def mock_venv():
    """Return a very simple mock vectorized environment containing two environments."""

    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: gym.make("MockEnv-v0")] * 2)
    yield env
    env.close()


@pytest.fixture
def venv():
    """Return a vectorized environment containing multiple environments."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from reward_preprocessing.env import _make_env

    env = DummyVecEnv([lambda: _make_env("MiniGrid-Empty-Random-6x6-v0")] * 5)
    yield env
    env.close()


@pytest.fixture
def model(env):
    """Return an (untrained) dummy agent model."""
    from stable_baselines3 import PPO

    return PPO("MlpPolicy", env)


@pytest.fixture
def model_path(model, tmp_path):
    """Return a path to a stored (untrained) agent model."""
    path = tmp_path / "agent"
    model.save(path)
    # we don't clean up here -- the tmp_path fixture takes care
    # of deleting the directory
    return path


@pytest.fixture
def data_path(env, tmp_path):
    """Return a path to a small dummy reward dataset."""
    from reward_preprocessing.transition import get_transitions

    path = tmp_path / "dataset.npz"
    train_samples = 5
    test_samples = 10

    states = {}
    actions = {}
    next_states = {}
    rewards = {}

    for mode, num_samples in zip(["train", "test"], [train_samples, test_samples]):
        states[mode] = []
        actions[mode] = []
        next_states[mode] = []
        rewards[mode] = []
        for transition, reward in get_transitions(env, num=num_samples):
            states[mode].append(transition.state)
            actions[mode].append(transition.action)
            next_states[mode].append(transition.next_state)
            rewards[mode].append(reward)

    np.savez(
        str(path),
        train_states=np.stack(states["train"], axis=0),
        train_actions=np.array(actions["train"]),
        train_next_states=np.stack(next_states["train"], axis=0),
        train_rewards=np.array(rewards["train"]),
        test_states=np.stack(states["test"], axis=0),
        test_actions=np.array(actions["test"]),
        test_next_states=np.stack(next_states["test"], axis=0),
        test_rewards=np.array(rewards["test"]),
    )

    # we don't clean up here -- the tmp_path fixture takes care
    # of deleting the directory
    return path
