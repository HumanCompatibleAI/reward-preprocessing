from pathlib import Path
from typing import Mapping, Sequence

from imitation.data.rollout import flatten_trajectories_with_rew
from imitation.rewards.reward_nets import RewardNet
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
import torch

from reward_preprocessing.data import (
    RolloutConfig,
    get_trajectories,
    transitions_collate_fn,
)
from reward_preprocessing.env.env_ingredient import create_env, env_ingredient

# Lowest priority is plotted first (in the top row)
SHAPINGS = {
    "unshaped": {
        "pretty_name": "Unshaped",
        "priority": 0,
    },
    "dense": {
        "pretty_name": "Value shaped",
        "priority": 1,
    },
}

PRETTY_OBJECTIVE_NAMES = {
    "unmodified": "Unmodified",
    "sparse_l1": "L1 sparse",
    "smooth_l1": "L1 smooth",
    "sparse_log": "Log sparse",
    "smooth_log": "Log smooth",
}

reward_curve_ex = Experiment("reward_curves", ingredients=[env_ingredient])


@reward_curve_ex.config
def config():
    objectives = ["unmodified", "sparse_l1", "smooth_log"]
    gamma = 0.99
    base_path = "results"
    model_base_paths = []
    font_size = 5.5
    rollout_steps = 1000  # length of the plotted rollouts
    rollout_cfg = None
    locals()  # make flake8 happy


@reward_curve_ex.named_config
def log():
    objectives = ["unmodified", "sparse_log", "smooth_log"]
    locals()  # make flake8 happy


@reward_curve_ex.named_config
def l1():
    objectives = ["unmodified", "sparse_l1", "smooth_l1"]
    locals()  # make flake8 happy


@reward_curve_ex.capture
def plot_reward_curves(
    models: Mapping[str, Mapping[str, RewardNet]],
    objectives: Sequence[str],
    font_size: int,
    rollout_steps: int,
    rollout_cfg: RolloutConfig,
    _seed,
) -> plt.Figure:
    rollout_cfg = RolloutConfig(*rollout_cfg)

    plt.rcParams.update({"font.size": font_size})

    venv = create_env()
    # TODO: this is very brittle
    env = venv.envs[0].unwrapped
    env.reset()

    trajectories = get_trajectories(
        rollout_cfg, create_env, min_timesteps=rollout_steps, seed=_seed
    )
    transitions = flatten_trajectories_with_rew(trajectories)

    dataloader = torch.utils.data.DataLoader(
        transitions,
        batch_size=256,
        collate_fn=transitions_collate_fn,
        num_workers=0,
        shuffle=False,
    )

    n_cols = len(objectives)
    n_rows = len(models)
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(5.5, 3), squeeze=False, sharex=True, sharey=True
    )

    for row, (model_name, model_versions) in enumerate(models.items()):
        for col, (objective, model) in enumerate(model_versions.items()):
            ax[row, col].axhline(0, linewidth=0.2, color="black")
            for x in [200, 400, 600, 800]:
                ax[row, col].axvline(x, linewidth=0.2, color="gray")
            predicted_rewards = np.array([])
            for transitions_batch in dataloader:
                with torch.no_grad():
                    predicted_rewards_batch = model(
                        *model.preprocess(
                            transitions_batch.obs,
                            transitions_batch.acts,
                            transitions_batch.next_obs,
                            transitions_batch.dones,
                        )
                    )
                    predicted_rewards = np.concatenate(
                        [predicted_rewards, predicted_rewards_batch.cpu().numpy()]
                    )
            predicted_rewards -= predicted_rewards[0]

            ax[row, col].plot(predicted_rewards, linewidth=0.4)
            ax[row, col].set_xlim(left=0, right=1000)

            ax[row, col].set(
                title=f"{model_name} / {PRETTY_OBJECTIVE_NAMES[objective]}"
            )

    for row in range(n_rows):
        ax[row, 0].set_ylabel("Reward")
    for col in range(n_cols):
        ax[n_rows - 1, col].set_xlabel("Time step")

    venv.close()
    return fig


@reward_curve_ex.automain
def main(
    base_path: str,
    model_base_paths: Sequence[str],
    objectives: Sequence[str],
    save_path: str,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    models = {}
    for model_path in model_base_paths:
        models[model_path] = {}
        for objective in objectives:
            path = Path(base_path) / (model_path + f".{objective}.pt")
            print(f"Loading model from {path}")
            models[model_path][objective] = torch.load(path, map_location=device)

    # sort in the order we want for the plot
    models = {
        k: v
        for k, v in sorted(
            models.items(),
            key=lambda item: SHAPINGS[item[0].split("_")[3]]["priority"],
        )
    }
    # convert to nicer names
    models = {
        SHAPINGS[model_path.split("_")[3]]["pretty_name"]: v
        for model_path, v in models.items()
    }

    fig = plot_reward_curves(models)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
