from typing import Any, Mapping

from imitation.rewards.reward_nets import RewardNet
from sacred import Ingredient

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.preprocessing import PotentialShaping
from reward_preprocessing.utils import get_env_name, instantiate, sacred_save_fig

fixed_ingredient = Ingredient("fixed", ingredients=[env_ingredient])

DEFAULT_POTENTIALS = {"HalfCheetah-v3": "SparseHalfCheetah"}


@fixed_ingredient.config
def config():
    enabled = False
    potential = None  # class name of the potential
    options = {}  # kwargs for the potential (other than the model and gamma)

    _ = locals()  # make flake8 happy
    del _


@fixed_ingredient.capture
def add_fixed_potential(
    model: RewardNet,
    gamma: float,
    enabled: bool,
    options: Mapping[str, Any],
    potential: str,
    _run,
) -> RewardNet:
    if not enabled:
        return model

    env = create_env()

    if potential is None:
        env_name = get_env_name(env)
        try:
            potential = DEFAULT_POTENTIALS[env_name]
        except KeyError:
            raise ValueError(
                "No potential name given and no default exists "
                f"for environment {env_name}"
            )

    wrapped_model: PotentialShaping = instantiate(
        "reward_preprocessing.preprocessing.fixed_potentials",
        potential,
        env=env,
        model=model,
        gamma=gamma,
        **options,
    )

    try:
        fig = wrapped_model.plot(env)
        fig.suptitle("Learned potential")
        sacred_save_fig(fig, _run, "potential")
    except NotImplementedError:
        print("Potential can't be plotted, skipping")

    return wrapped_model
