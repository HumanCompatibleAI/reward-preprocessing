from pathlib import Path
import tempfile

from sacred import Ingredient

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.models import RewardModel
from reward_preprocessing.preprocessing.potential_shaping import instantiate_potential
from reward_preprocessing.utils import sacred_save_fig

noise_ingredient = Ingredient("noise", ingredients=[env_ingredient])


@noise_ingredient.config
def config():
    enabled = True
    std = 1.0
    mean = 0.0
    potential = None

    _ = locals()  # make flake8 happy
    del _


@noise_ingredient.capture
def add_noise_potential(
    model: RewardModel,
    gamma: float,
    enabled: bool,
    std: float,
    mean: float,
    potential: str,
    _run,
) -> RewardModel:
    if not enabled:
        return model

    env = create_env()

    env_name = env.envs[0].spec.id
    wrapped_model = instantiate_potential(env_name, potential, model=model, gamma=gamma)
    try:
        wrapped_model.random_init(std=std, mean=mean)
    except NotImplementedError:
        print("Potential can't be used as noise potential, skipping")
        return model

    model = wrapped_model

    try:
        fig = model.plot(env)
        fig.suptitle("Noise potential")
        sacred_save_fig(fig, _run, "noise_potential")
    except NotImplementedError:
        print("Potential can't be plotted, skipping")

    return model
