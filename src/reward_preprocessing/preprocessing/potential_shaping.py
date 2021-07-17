"""This module defines several preprocessors that add
a potential shaping to the reward (a particular transformation
that leaves the optimal policy unchanged).

A potential shaping is fully defined by its potential, which maps
states to floats. PotentialShaping is the most general case,
while the other classes in this module are helper classes
that use a particular type of potential.
"""
from typing import Callable

import numpy as np
import torch
from torch import nn

import reward_preprocessing.env.maze as maze
from reward_preprocessing.env.maze import get_agent_positions
from reward_preprocessing.models import RewardModel
from reward_preprocessing.transition import Transition

from .preprocessor import Preprocessor


class PotentialShaping(Preprocessor):
    """A preprocessor that adds a potential shaping to the reward.

    Args:
        model: the RewardModel to be wrapped
        potential: a Callable that receives a batch of states and returns
            a batch of scalars.
            If the potential is itself a torch.nn.Module, it will become
            a submodule of this Module.
        gamma: the discount factor
        is_terminal (optional): a Callable that determines whether a given
            batch of states is terminal (statewise).
            The potential of terminal states is forced to be zero
            automatically. By default, no state is terminal, so this feature
            isn't used.
    """

    def __init__(
        self,
        model: RewardModel,
        potential: Callable,
        gamma: float,
        is_terminal: Callable = lambda x: torch.zeros(x.size(0), dtype=torch.bool),
    ):
        super().__init__(model)
        self._potential = potential
        self.gamma = gamma
        self.is_terminal = is_terminal

    def potential(self, states) -> torch.Tensor:
        return (~self.is_terminal(states)) * self._potential(states)

    def forward(self, transitions: Transition) -> torch.Tensor:
        rewards = self.model(transitions)
        return (
            rewards
            + self.gamma * self.potential(transitions.next_state)
            - self.potential(transitions.state)
        )


class LinearPotentialShaping(PotentialShaping):
    """A potential shaping preprocessor with a learned linear potential."""

    def __init__(self, model: RewardModel, gamma: float):
        in_size = np.product(model.state_shape)
        potential = nn.Sequential(nn.Flatten(), nn.Linear(in_size, 1))
        super().__init__(model, potential, gamma)


class RandomPotentialShaping(PotentialShaping):
    """A preprocessor that adds a random potential shaping."""

    def __init__(self, model: RewardModel, gamma: float):
        in_size = np.product(model.state_shape)
        self.potential_data = torch.randn(in_size)

        def potential(states):
            # The total number of possible states for mazelab is
            # 4 ** (n ** 2), where n is the size of the maze.
            # So we let the potential only depend on the agent position,
            # since this is the only non-fixed part of the state for
            # any single episode.
            positions = get_agent_positions(states)
            return self.potential_data[positions]

        super().__init__(model, potential, gamma, maze.is_terminal)


class LookupTable(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.potential = nn.Parameter(torch.zeros(size))

    def forward(self, states):
        positions = get_agent_positions(states)
        return self.potential[positions]


class TabularPotentialShaping(PotentialShaping):
    """A preprocessor that adds a learned potential shaping in a tabular setting."""

    def __init__(self, model: RewardModel, gamma: float):
        in_size = np.product(model.state_shape)
        potential = LookupTable(in_size)

        super().__init__(model, potential, gamma, maze.is_terminal)
