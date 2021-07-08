import numpy as np
import pytest


@pytest.mark.expensive
def test_agent_training_deterministic(tmp_path):
    from reward_preprocessing.train_agent import ex

    # for now we just check that it works without errors
    r1 = ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "steps": 10,
            "num_frames": 10,
        }
    )

    r2 = ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "steps": 10,
            "num_frames": 10,
        }
    )

    assert np.array_equal(r1.result[2], r2.result[2])


@pytest.mark.expensive
def test_agent_training_different_seeds(tmp_path):
    """A sanity check: with different seeds, the results should be different.
    Otherwise, the determinism test apparently doesn't work.
    """
    from reward_preprocessing.train_agent import ex

    # for now we just check that it works without errors
    r1 = ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "steps": 10,
            "num_frames": 10,
        }
    )

    r2 = ex.run(
        config_updates={
            "seed": 1,
            "run_dir": str(tmp_path),
            "steps": 10,
            "num_frames": 10,
        }
    )

    assert not np.array_equal(r1.result[2], r2.result[2])


@pytest.mark.expensive
def test_reward_training_deterministic(data_path, tmp_path):
    from reward_preprocessing.train_reward_model import ex

    r1 = ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "epochs": 1,
            "batch_size": 2,
            "data_path": str(data_path),
        }
    )

    r2 = ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "epochs": 1,
            "batch_size": 2,
            "data_path": str(data_path),
        }
    )

    # just a sanity check to ensure we actually evaluated
    assert isinstance(r1.result, float)

    assert r1.result == r2.result


@pytest.mark.expensive
def test_reward_training_different_seeds(data_path, tmp_path):
    """A sanity check: with different seeds, the results should be different.
    Otherwise, the determinism test apparently doesn't work.
    """
    from reward_preprocessing.train_reward_model import ex

    r1 = ex.run(
        config_updates={
            "seed": 0,
            "run_dir": str(tmp_path),
            "epochs": 1,
            "batch_size": 2,
            "data_path": str(data_path),
        }
    )

    r2 = ex.run(
        config_updates={
            "seed": 1,
            "run_dir": str(tmp_path),
            "epochs": 1,
            "batch_size": 2,
            "data_path": str(data_path),
        }
    )

    # just a sanity check to ensure we actually evaluated
    assert isinstance(r1.result, float)

    assert r1.result != r2.result
