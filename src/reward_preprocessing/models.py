import torch
import torch.nn as nn


class MlpRewardModel(nn.Module):
    def __init__(self, observation_size: int):
        super().__init__()
        num_features = 2 * observation_size
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

    def forward(self, x: torch.Tensor):
        """Evaluate the reward model on an (s, s') input.

        Args:
            x: A tensor of shape (2, ...) which should be a stack of the features
                of the state s and the following state s'
        """
        return self.net(x)
