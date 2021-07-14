from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from reward_preprocessing.transition import Transition

from .reward_model import RewardModel


class MlpRewardModel(RewardModel):
    def __init__(self, state_shape: Tuple[int, ...]):
        super().__init__(state_shape)
        num_features = 2 * np.product(state_shape)
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

    def forward(self, transition: Transition):
        # the new stacked axis should be the second one, since the first axis
        # should remain the batch axis
        x = torch.stack([transition.state, transition.next_state], dim=1)
        x = x.flatten(start_dim=1)
        # the squeeze gets rid of the final singleton dimension produced by the last
        # linear layer
        return self.net(x).squeeze(dim=-1)
