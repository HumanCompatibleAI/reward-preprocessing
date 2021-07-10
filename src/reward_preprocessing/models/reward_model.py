from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn as nn

from reward_preprocessing.transition import Transition


class RewardModel(nn.Module, ABC):
    """ABC for all reward models.

    For now, this serves mainly for type hints and to define
    a clear class hierachy, it may contain actual functionality
    in the future.
    """

    def __init__(self, state_shape: Tuple[int]):
        super().__init__()
        self.state_shape = state_shape

    @abstractmethod
    def forward(self, transition: Transition):
        pass
