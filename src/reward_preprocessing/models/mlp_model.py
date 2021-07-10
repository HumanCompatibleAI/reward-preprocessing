import torch
import torch.nn as nn

from reward_preprocessing.transition import Transition


class MlpRewardModel(nn.Module):
    def __init__(self, observation_size: int):
        super().__init__()
        num_features = 2 * observation_size
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
        return self.net(x)
