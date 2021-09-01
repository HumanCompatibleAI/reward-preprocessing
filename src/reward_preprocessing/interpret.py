from typing import Any, Mapping
import warnings

from sacred import Experiment
from stable_baselines3 import PPO
import torch
import wandb

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.interp import (
    add_noise_potential,
    noise_ingredient,
    rollout_ingredient,
    sparsify,
    sparsify_ingredient,
    transition_ingredient,
    visualize_rollout,
    visualize_transitions,
)
from reward_preprocessing.models import MlpRewardModel, SasRewardModel
from reward_preprocessing.utils import add_observers

ex = Experiment(
    "interpret",
    ingredients=[
        env_ingredient,
        rollout_ingredient,
        sparsify_ingredient,
        noise_ingredient,
        transition_ingredient,
    ],
)
add_observers(ex)


@ex.config
def config():
    run_dir = "runs/interpret"
    agent_path = None  # path to the agent to use for sampling (without extension)
    model_path = None  # path to the model to be interpreted (with extension)
    gamma = 0.99  # discount rate (used for all potential shapings)
    model_type = "ss"  # type of reward model, either 'ss' or 'sas'
    wb = {}  # kwargs for wandb.init()

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(
    model_path: str,
    agent_path: str,
    gamma: float,
    model_type: str,
    wb: Mapping[str, Any],
    _config,
):
    env = create_env()
    agent = None
    if agent_path:
        agent = PPO.load(agent_path)
        if agent.gamma != gamma:
            warnings.warn("Agent was trained with different gamma value")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_type == "ss":
        model = MlpRewardModel(env.observation_space.shape).to(device)
    elif model_type == "sas":
        model = SasRewardModel(env.observation_space.shape, env.action_space.shape).to(
            device
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}', expected 'ss' or 'sas'.")
    model.load_state_dict(torch.load(model_path))

    use_wandb = False
    if wb:
        wandb.init(
            project="reward_preprocessing",
            job_type="interpret",
            config=_config,
            **wb,
        )
        use_wandb = True

    model = add_noise_potential(model, gamma)
    model = sparsify(model, device=device, gamma=gamma, use_wandb=use_wandb)
    model.eval()
    visualize_transitions(model, env, device=device, agent=agent)
    visualize_rollout(model, env, device=device, agent=agent)
    env.close()
