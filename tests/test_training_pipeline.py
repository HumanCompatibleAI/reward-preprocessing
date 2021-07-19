from pathlib import Path
import tempfile

from reward_preprocessing.create_rollouts import ex as create_rollouts_ex
from reward_preprocessing.interpret import ex as interpret_ex
from reward_preprocessing.train_agent import ex as train_agent_ex
from reward_preprocessing.train_reward_model import ex as train_reward_model_ex


def test_agent_training_experiment(tmp_path):
    # for now we just check that it works without errors
    train_agent_ex.run(
        config_updates={"run_dir": str(tmp_path), "steps": 10, "num_frames": 10}
    )


def test_dataset_creation(agent_path):
    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname) / "dataset"
        create_rollouts_ex.run(
            config_updates={
                "model_path": str(agent_path),
                "save_path": str(path),
                "train_samples": 10,
                "test_samples": 10,
            }
        )


def test_reward_training_experiment(data_path, tmp_path):
    train_reward_model_ex.run(
        config_updates={
            "run_dir": str(tmp_path),
            "epochs": 1,
            "batch_size": 2,
            "data_path": str(data_path),
        }
    )


def test_interpret_experiment(model_path, agent_path, tmp_path):
    interpret_ex.run(
        config_updates={
            "run_dir": str(tmp_path),
            "model_path": str(model_path),
            "agent_path": str(agent_path),
        }
    )
