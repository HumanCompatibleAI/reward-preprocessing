import pytest


@pytest.mark.expensive
def test_agent_training_deterministic(tmp_path):
    from reward_preprocessing.train_agent import ex

    # for now we just check that it works without errors
    r1 = ex.run(
        config_updates={
            "seed": 0,
            "eval_episodes": 10,
            "run_dir": str(tmp_path),
            "steps": 10,
            "num_frames": 10,
        }
    )

    r2 = ex.run(
        config_updates={
            "seed": 0,
            "eval_episodes": 10,
            "run_dir": str(tmp_path),
            "steps": 10,
            "num_frames": 10,
        }
    )

    # just a sanity check to ensure we actually evaluated
    assert isinstance(r1.result[0], float)

    assert r1.result == r2.result


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
