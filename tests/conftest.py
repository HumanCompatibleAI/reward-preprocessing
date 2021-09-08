import gym
from imitation.data import types
from imitation.data.rollout import generate_trajectories, make_sample_until
from imitation.rewards.reward_nets import BasicRewardNet
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from reward_preprocessing.utils import get_env_name


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
        if self.pos == 0:
            reward = -1.0
        elif self.pos == 10:
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


gym.envs.register(
    id="reward_preprocessing/MockEnv-v0", entry_point=MockEnv, max_episode_steps=100
)


@pytest.fixture(params=["EmptyMaze-v0", "MountainCar-v0", "HalfCheetah-v3"])
def env(request):
    """Return a gym environment."""
    from reward_preprocessing.env import create_env

    env = create_env(request.param, _seed=0, options={}, stats_path=None)
    yield env
    env.close()


@pytest.fixture
def mock_env():
    """Return a very simple mock gym environment."""

    env = DummyVecEnv([lambda: gym.make("reward_preprocessing/MockEnv-v0")])
    yield env
    env.close()


@pytest.fixture
def mock_venv():
    """Return a very simple mock vectorized environment containing two environments."""

    env = DummyVecEnv([lambda: gym.make("reward_preprocessing/MockEnv-v0")] * 2)
    yield env
    env.close()


@pytest.fixture
def venv(env):
    """Return a vectorized environment containing multiple environments."""

    # The point of this complicated procedure is to reuse the parametrization
    # of the env fixture. If a test depends on both this venv fixture
    # and e.g. agent_path (which depends on env), we want to use the same
    # type of environment for both fixtures.
    # I think an alternative would be to declare the parameters in the top-level
    # tests and then access them in fixtures like this:
    # https://pytest.org/en/6.2.x/fixture.html#using-markers-to-pass-data-to-fixtures
    # Would be more flexible, so we might have to switch at some point.
    # But for now, this solution has the advantage that every test that relies
    # on environments will be run with all environments automatically.
    env_name = get_env_name(env)
    venv = DummyVecEnv([lambda: gym.make(env_name)] * 5)
    yield venv
    venv.close()


@pytest.fixture
def agent(env):
    """Return an (untrained) dummy agent model."""

    return PPO("MlpPolicy", env)


@pytest.fixture
def agent_path(agent, tmp_path):
    """Return a path to a stored (untrained) agent model."""
    path = tmp_path / "agent"
    agent.save(path)
    # we don't clean up here -- the tmp_path fixture takes care
    # of deleting the directory
    return path


@pytest.fixture
def model(env):
    """Return an (untrained) dummy reward model."""
    return BasicRewardNet(
        env.observation_space, env.action_space, use_action=False, use_next_state=True
    )


@pytest.fixture
def model_path(model, tmp_path):
    """Return a path to a stored (untrained) reward model."""
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    # we don't clean up here -- the tmp_path fixture takes care
    # of deleting the directory
    return path


@pytest.fixture
def data_path(env, tmp_path):
    """Return a path to a small dummy trajectory dataset."""

    path = tmp_path / "dataset.pkl"
    trajectories = generate_trajectories(
        policy=None,
        venv=env,
        sample_until=make_sample_until(min_episodes=1, min_timesteps=None),
    )
    types.save(path, trajectories)

    # we don't clean up here -- the tmp_path fixture takes care
    # of deleting the directory
    return path
