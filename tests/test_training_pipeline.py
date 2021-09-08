from reward_preprocessing.interpret import ex as interpret_ex
from reward_preprocessing.train_reward_model import ex as train_reward_model_ex
from reward_preprocessing.utils import get_env_name


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
            "sparsify.enabled": True,
            "sparsify.steps": 2,
            "sparsify.batch_size": 2,
            "noise.enabled": True,
            "rewards.rollout_steps": 2,
            "rewards.bins": 2,
            "transition_visualization.steps": 10,
            "transition_visualization.num": 2,
            "rollout_visualization.plot_shape": (2, 2),
            "env.name": get_env_name(env),
        },
        named_configs=[
            "sparsify.random_rollouts",
            "rewards.random_rollouts",
            "transition_visualization.random_rollouts",
            "rollout_visualization.random_rollouts",
        ],
    )
