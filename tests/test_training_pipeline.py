from reward_preprocessing.interpret import ex as interpret_ex
from reward_preprocessing.utils import get_env_name


def test_interpret_experiment(env, model_path, tmp_path):
    interpret_ex.run(
        config_updates={
            "run_dir": str(tmp_path),
            "model_path": str(model_path),
            "optimize.enabled": True,
            "optimize.steps": 2,
            "optimize.batch_size": 2,
            "noise.enabled": True,
            "rewards.rollout_steps": 2,
            "rewards.bins": 2,
            "rewards.steps": 4,
            "transition_visualization.steps": 10,
            "transition_visualization.num": 2,
            "rollout_visualization.plot_shape": (2, 2),
            "env.name": get_env_name(env),
        },
        named_configs=[
            "optimize.random_rollouts",
            "rewards.random_rollouts",
            "transition_visualization.random_rollouts",
            "rollout_visualization.random_rollouts",
        ],
    )
