import os

import gym

from reward_preprocessing.create_rollouts import ex as create_rollouts_ex
from reward_preprocessing.env import maze  # noqa: F401
from reward_preprocessing.preprocess import preprocess_ex as preprocess_ex
from reward_preprocessing.utils import get_env_name


def test_create_rollouts(env, agent_path, tmp_path):
    create_rollouts_ex.run(
        config_updates={
            "rollouts": [(0.5, agent_path, "mixture")],
            "min_timesteps": 2,
            "out_dir": str(tmp_path),
            "env.name": get_env_name(env),
        },
    )

    assert os.path.exists(os.path.join(tmp_path, "mixture.pkl"))


def test_preprocess_ex(env, data_path, model_path, tmp_path):
    rollouts = None
    if isinstance(env.observation_space, gym.spaces.Box):
        rollouts = [(0, None, "dataset", data_path)]
    preprocess_ex.run(
        config_updates={
            "rollouts": rollouts,
            "model_path": model_path,
            "save_path": os.path.join(tmp_path, "model"),
            "env.name": get_env_name(env),
        },
    )

    assert os.path.exists(os.path.join(tmp_path, "model.unmodified.pt"))
