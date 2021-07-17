from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from reward_preprocessing.transition import Transition

from .reward_model import RewardModel


class MlpRewardModel(RewardModel):
    """An MLP reward model with 2 hidden layers and (s, s') pairs as input.
    Uses ReLU activation functions.

    Args:
        state_shape: shape of the observations the environment produces
            (these observations have to be arrays of floats, discrete
            observations are not supported)
        hidden_size (optional): number of neurons in each hidden layer
    """

    def __init__(self, state_shape: Tuple[int, ...], hidden_size: int = 64):
        super().__init__(state_shape)
        num_features = 2 * np.product(state_shape)
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, transitions: Transition) -> torch.Tensor:
        # the new stacked axis should be the second one, since the first axis
        # should remain the batch axis
        x = torch.stack([transitions.state, transitions.next_state], dim=1)
        x = x.flatten(start_dim=1)
        # the squeeze gets rid of the final singleton dimension produced by the last
        # linear layer
        return self.net(x).squeeze(dim=-1)
