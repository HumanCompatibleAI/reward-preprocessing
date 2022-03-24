import os
from pathlib import Path
from typing import Mapping, Sequence

import gym
from imitation.rewards.reward_nets import RewardNet
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
import torch

from reward_preprocessing.env.env_ingredient import create_env, env_ingredient
from reward_preprocessing.gridworld_plot import plot_gridworld_rewards

# Lowest priority is plotted first (in the top row)
SHAPINGS = {
    "unshaped": {
        "pretty_name": "Unshaped",
        "priority": 0,
    },
    "dense": {
        "pretty_name": "Dense shaping",
        "priority": 1,
    },
    "antidense": {
        "pretty_name": "Negative shaping",
        "priority": 2,
    },
    "random": {
        "pretty_name": "Random shaping",
        "priority": 3,
    },
}

PRETTY_OBJECTIVE_NAMES = {
    "unmodified": "Unmodified",
    "sparse_l1": "L1 sparse",
    "smooth_l1": "L1 smooth",
    "sparse_log": "Log sparse",
    "smooth_log": "Log smooth",
}

heatmap_ex = Experiment("heatmaps", ingredients=[env_ingredient])


@heatmap_ex.config
def config():
    objectives = ["unmodified", "sparse_l1", "smooth_log"]
    gamma = 0.99
    base_path = "results"
    model_base_paths = []
    font_size = 6
    locals()  # make flake8 happy


@heatmap_ex.named_config
def log():
    objectives = ["unmodified", "sparse_log", "smooth_log"]
    locals()  # make flake8 happy


@heatmap_ex.named_config
def l1():
    objectives = ["unmodified", "sparse_l1", "smooth_l1"]
    locals()  # make flake8 happy


@heatmap_ex.capture
def plot_heatmaps(
    models: Mapping[str, Mapping[str, RewardNet]],
    gamma: float,
    objectives: Sequence[str],
    font_size: int,
) -> plt.Figure:

    plt.rcParams.update({"font.size": font_size})

    venv = create_env()
    # TODO: this is very brittle
    env = venv.envs[0].unwrapped
    env.reset()
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError("Heatmap only works for discrete state spaces")

    rewards = {}

    states = []
    next_states = []
    actions = []
    invalid_indices = []
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            next_state, valid = env._step(state, action)
            if not valid:
                invalid_indices.append(5 * state + action)
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
    invalid_indices = np.array(invalid_indices)

    states = torch.tensor(states, dtype=torch.long)
    next_states = torch.tensor(next_states, dtype=torch.long)
    dones = torch.zeros(env.observation_space.n * 5, dtype=torch.bool)

    for model_name, model_versions in models.items():
        rewards[model_name] = {}
        for objective, model in model_versions.items():
            with torch.no_grad():
                out = model(states, None, next_states, dones)

            np_out = out.detach().cpu().numpy()
            np_out[invalid_indices] = np.nan
            np_out = np_out.reshape(env.observation_space.n, 5)
            if objective not in {"sparse_l1", "sparse_log"}:
                np_out -= np_out[np.isfinite(np_out)].mean()
            rewards[model_name][objective] = np_out

    if env.using_key:
        assert (
            len(rewards) == 1
        ), "Plotting multiple models not supported when using key"
        # We want to split the rewards for each objective into two arrays,
        # one with and one without the key. These can then be plotted separately.
        stacked_rewards = next(iter(rewards.values()))
        rewards = {
            "without key": {
                objective: stacked_rewards[objective][: env.size ** 2, :]
                for objective in objectives
            },
            "with key": {
                objective: stacked_rewards[objective][env.size ** 2 :, :]
                for objective in objectives
            },
        }

    ncols = len(objectives)
    # flatten the reward dict and reshape for plotting
    rewards = {
        f"{name} / {PRETTY_OBJECTIVE_NAMES[objective]}": reward.reshape(
            env.size, env.size, 5
        )
        for name, reward_versions in rewards.items()
        for objective, reward in reward_versions.items()
    }
    venv.close()
    return plot_gridworld_rewards(
        rewards, ncols=ncols, discount=gamma, vmin=-1.2, vmax=1.2
    )


@heatmap_ex.automain
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
            key=lambda item: SHAPINGS[item[0].split("_")[-1]]["priority"],
        )
    }
    # convert to nicer names
    models = {
        SHAPINGS[model_path.split("_")[-1]]["pretty_name"]: v
        for model_path, v in models.items()
    }

    fig = plot_heatmaps(models)
    fig.set_tight_layout(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
