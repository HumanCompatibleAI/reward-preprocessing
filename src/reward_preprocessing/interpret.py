import warnings

from sacred import Experiment
from stable_baselines3 import PPO
import torch

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
from reward_preprocessing.models import MlpRewardModel
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
    agent_path = None
    model_path = None
    gamma = 0.99

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(model_path: str, agent_path: str, gamma: float):
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
    model = MlpRewardModel(env.observation_space.shape).to(device)
    model.load_state_dict(torch.load(model_path))
    model = add_noise_potential(model, gamma)
    model = sparsify(model, gamma=gamma, agent=agent)
    model.eval()
    visualize_transitions(model, env, agent=agent)
    visualize_rollout(model, env, agent=agent)
    env.close()
