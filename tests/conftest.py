import numpy as np
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


@pytest.fixture
def data_path(env, tmp_path):
    """Return a path to a small dummy reward dataset."""
    path = tmp_path / "dataset.npz"
    train_samples = 5
    test_samples = 10

    states = {}
    actions = {}
    next_states = {}
    rewards = {}

    for mode, num_samples in zip(["train", "test"], [train_samples, test_samples]):
        obs = env.reset()
        states[mode] = []
        actions[mode] = []
        next_states[mode] = []
        rewards[mode] = []
        for _ in range(num_samples):
            states[mode].append(obs)
            # put the action into a list because we use a vectorized environment
            # (so env.step expects a list of actions)
            action = [env.action_space.sample()]
            actions[mode].append(action)
            obs, reward, done, _ = env.step(action)
            next_states[mode].append(obs)
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
