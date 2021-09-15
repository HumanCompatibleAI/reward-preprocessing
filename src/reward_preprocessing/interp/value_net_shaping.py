from imitation.rewards.reward_nets import RewardNet
from sacred import Ingredient

from reward_preprocessing.preprocessing.potential_shaping import CriticPotentialShaping
from reward_preprocessing.utils import sacred_save_fig

value_net_ingredient = Ingredient("value_net")


@value_net_ingredient.config
def config():
    enabled = False
    path = None  # path to the SB3 algorithm containing the critic

    _ = locals()  # make flake8 happy
    del _


@value_net_ingredient.capture
def add_value_net_potential(
    model: RewardNet,
    gamma: float,
    enabled: bool,
    path: str,
    _run,
) -> RewardNet:
    if not enabled:
        return model

    if path is None:
        raise ValueError("Path to algorithm containing critic must be set!")
    wrapped_model = CriticPotentialShaping(model, path, gamma)

    try:
        fig = wrapped_model.plot()
        fig.suptitle("Value net potential")
        sacred_save_fig(fig, _run, "value_net_potential")
    except NotImplementedError:
        print("Potential can't be plotted, skipping")

    return wrapped_model
