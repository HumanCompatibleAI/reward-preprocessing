from typing import Callable

import numpy as np
from torch import nn

from reward_preprocessing.models import RewardModel
from reward_preprocessing.transition import Transition

from .preprocessor import Preprocessor


class PotentialShaping(Preprocessor):
    """A preprocessor that adds a potential shaping to the reward.

    Args:
        model: the RewardModel to be wrapped
        potential: a Callable that receives a state and returns a scalar.
            If the potential is itself a torch.nn.Module, it will become
            a submodule of this Module.
        gamma: the discount factor
    """

    def __init__(self, model: RewardModel, potential: Callable, gamma: float):
        super().__init__(model)
        self.potential = potential
        self.gamma = gamma

    def forward(self, transition: Transition):
        reward = self.model(transition)
        return (
            reward
            + self.gamma * self.potential(transition.next_state)
            - self.potential(transition.state)
        )


class LinearPotentialShaping(PotentialShaping):
    """A potential shaping preprocessor with a learned linear potential."""

    def __init__(self, model: RewardModel, gamma: float):
        in_size = np.product(model.state_shape)
        potential = nn.Sequential(nn.Flatten(), nn.Linear(in_size, 1))
        super().__init__(model, potential, gamma)