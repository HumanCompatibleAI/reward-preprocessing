from typing import Optional

from imitation.rewards.reward_nets import RewardNet
from sacred import Ingredient
import torch

from reward_preprocessing.preprocessing.potential_shaping import CriticPotentialShaping
from reward_preprocessing.utils import sacred_save_fig

value_net_ingredient = Ingredient("value_net")


@value_net_ingredient.config
def config():
    enabled = False
    path = None  # path to the SB3 algorithm containing the critic

    _ = locals()  # make flake8 happy
    del _


class ZeroNet(RewardNet):
    def forward(self, state, action, next_state, done):
        return torch.zeros_like(done)


@value_net_ingredient.capture
def value_net_potential(
    model: RewardNet,
    gamma: float,
    enabled: bool,
    path: str,
    _run,
) -> Optional[RewardNet]:
    if not enabled:
        return None

    if path is None:
        raise ValueError("Path to algorithm containing critic must be set!")
    zero_model = ZeroNet(model.observation_space, model.action_space, model.normalize_images)
    shaping_model = CriticPotentialShaping(zero_model, path, gamma)

    try:
        fig = shaping_model.plot()
        fig.suptitle("Value net potential")
        sacred_save_fig(fig, _run, "value_net_potential")
    except NotImplementedError:
        print("Potential can't be plotted, skipping")

    return shaping_model
