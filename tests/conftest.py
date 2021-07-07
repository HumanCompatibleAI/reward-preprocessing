import pytest


@pytest.fixture
def env():
    """Return a gym environment."""
    from reward_preprocessing.env import create_env

    env = create_env("MiniGrid-Empty-Random-6x6-v0")
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
