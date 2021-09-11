import pytest

from reward_preprocessing.train_reward_model import ex as train_reward_model_ex
from reward_preprocessing.utils import get_env_name


@pytest.mark.expensive
def test_reward_training_deterministic_with_agent(env, agent_path, tmp_path):
    r1 = train_reward_model_ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "batch_size": 2,
            "steps": 10,
            "rollouts": [(0, str(agent_path))],
            "env.name": get_env_name(env),
        }
    )

    r2 = train_reward_model_ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "batch_size": 2,
            "steps": 10,
            "rollouts": [(0, str(agent_path))],
            "env.name": get_env_name(env),
        }
    )

    # just a sanity check to ensure we actually evaluated
    assert isinstance(r1.result, float)

    assert r1.result == r2.result


@pytest.mark.expensive
def test_reward_training_deterministic_no_agent(env, tmp_path):
    r1 = train_reward_model_ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "batch_size": 2,
            "steps": 10,
            "env.name": get_env_name(env),
        },
        named_configs=["random_rollouts"],
    )

    r2 = train_reward_model_ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "batch_size": 2,
            "steps": 10,
            "env.name": get_env_name(env),
        },
        named_configs=["random_rollouts"],
    )

    # just a sanity check to ensure we actually evaluated
    assert isinstance(r1.result, float)

    assert r1.result == r2.result


@pytest.mark.expensive
def test_reward_training_different_seeds(env, tmp_path):
    """A sanity check: with different seeds, the results should be different.
    Otherwise, the determinism test apparently doesn't work.
    """
    r1 = train_reward_model_ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "batch_size": 2,
            "steps": 10,
            "env.name": get_env_name(env),
        },
        named_configs=["random_rollouts"],
    )

    r2 = train_reward_model_ex.run(
        config_updates={
            "seed": 1,
            "run_dir": str(tmp_path),
            "batch_size": 2,
            "steps": 10,
            "env.name": get_env_name(env),
        },
        named_configs=["random_rollouts"],
    )

    # just a sanity check to ensure we actually evaluated
    assert isinstance(r1.result, float)

    assert r1.result != r2.result
