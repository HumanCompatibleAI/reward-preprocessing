from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3 import PPO
import torch

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.interp import rollout_ingredient, visualize_rollout
from reward_preprocessing.models import MlpRewardModel
from reward_preprocessing.preprocessing.potential_shaping import RandomPotentialShaping

ex = Experiment("interpret", ingredients=[env_ingredient, rollout_ingredient])


@ex.config
def config():
    run_dir = "runs/interpret"
    agent_path = None
    model_path = None
    gamma = 0.99

    # Just to be save, we check whether an observer already exists,
    # to avoid adding multiple copies of the same observer
    # (see https://github.com/IDSIA/sacred/issues/300)
    if len(ex.observers) == 0:
        ex.observers.append(FileStorageObserver(run_dir))
    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(model_path: str, agent_path: str, gamma: float):
    env = create_env()
    agent = PPO.load(agent_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = MlpRewardModel(env.observation_space.shape).to(device)
    model.load_state_dict(torch.load(model_path))
    model = RandomPotentialShaping(model, gamma=gamma)
    model.eval()
    visualize_rollout(model, env, agent=agent)
    env.close()
