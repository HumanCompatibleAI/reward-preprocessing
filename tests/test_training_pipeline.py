from pathlib import Path
import tempfile

from reward_preprocessing.create_rollouts import ex as create_rollouts_ex
from reward_preprocessing.interpret import ex as interpret_ex
from reward_preprocessing.train_agent import ex as train_agent_ex
from reward_preprocessing.train_reward_model import ex as train_reward_model_ex
from reward_preprocessing.utils import get_env_name


def test_agent_training_experiment(env, tmp_path):
    # for now we just check that it works without errors
    train_agent_ex.run(
        config_updates={
            "run_dir": str(tmp_path),
            "steps": 10,
            "num_frames": 10,
            "env.name": get_env_name(env),
        }
    )


def test_dataset_creation(env, agent_path):
    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname) / "dataset"
        create_rollouts_ex.run(
            config_updates={
                "agent_path": str(agent_path),
                "save_path": str(path),
                "train_samples": 10,
                "test_samples": 10,
                "env.name": get_env_name(env),
            }
        )


def test_reward_training_experiment(env, data_path, tmp_path):
    train_reward_model_ex.run(
        config_updates={
            "run_dir": str(tmp_path),
            "epochs": 1,
            "batch_size": 2,
            "data_path": str(data_path),
            "steps": None,
            "test_steps": None,
            "env.name": get_env_name(env),
        }
    )


def test_interpret_experiment(env, model_path, agent_path, tmp_path):
    interpret_ex.run(
        config_updates={
            "run_dir": str(tmp_path),
            "model_path": str(model_path),
            "agent_path": str(agent_path),
            "sparsify.steps": 10,
            "sparsify.batch_size": 2,
            "transition_visualization.num_samples": 10,
            "transition_visualization.num": 2,
            "rollout_visualization.plot_shape": (2, 2),
            "env.name": get_env_name(env),
        },
        named_configs=["sparsify.random_rollouts"],
    )
