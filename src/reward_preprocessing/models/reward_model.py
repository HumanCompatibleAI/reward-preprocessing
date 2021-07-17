from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

from reward_preprocessing.transition import Transition


class RewardModel(nn.Module, ABC):
    """ABC for all reward models.

    For now, this serves mainly for type hints and to define
    a clear class hierachy, it may contain actual functionality
    in the future.
    """

    def __init__(self, state_shape: Tuple[int, ...]):
        super().__init__()
        self.state_shape = state_shape

    @abstractmethod
    def forward(self, transitions: Transition) -> torch.Tensor:
        """Predict the reward for a batch of transitions.

        The batch of transitions is represented as a single Transition instance
        which contains batches in each field.
        Returns a batch of associated rewards.
        """
        pass
