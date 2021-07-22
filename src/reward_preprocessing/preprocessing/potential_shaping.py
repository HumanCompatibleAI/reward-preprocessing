"""This module defines several preprocessors that add
a potential shaping to the reward (a particular transformation
that leaves the optimal policy unchanged).

A potential shaping is fully defined by its potential, which maps
states to floats. PotentialShaping is the most general case,
while the other classes in this module are helper classes
that use a particular type of potential.
"""
from typing import Callable

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

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
    """

    def __init__(
        self,
        model: RewardModel,
        potential: Callable,
        gamma: float,
    ):
        super().__init__(model)
        self.potential = potential
        self.gamma = gamma

    def forward(self, transitions: Transition) -> torch.Tensor:
        rewards = self.model(transitions)
        current_potential = self.potential(transitions.state)
        # if the next state is final, then we set it's potential to zero
        next_potential = torch.logical_not(transitions.done) * self.potential(
            transitions.next_state
        )
        return rewards + self.gamma * next_potential - current_potential

    def plot(self, env: gym.Env) -> plt.Figure:
        """Plot the potential if possible.
        The type of plot depends on the type of potential.

        Args:
            env (gym.Env): the environment for which this potential is used
                (needed for some types of plots)

        Raises:
            NotImplementedError: if plotting isn't implemented for this type
                of potential

        Returns:
            plt.Figure: a plot of the potential
        """
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or space.shape != (2,):
            # we don't know how to handle state spaces that aren't 2D in general
            raise NotImplementedError(
                "Potential plotting is not implemented for this type of potential."
            )

        grid_size = 100

        xs = np.linspace(space.low[0], space.high[0], grid_size, dtype=np.float32)
        ys = np.linspace(space.low[1], space.high[1], grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        values = np.stack([xx, yy], axis=2).reshape(-1, 2)
        out = self.potential(torch.as_tensor(values)).detach().numpy()
        out = out.reshape(grid_size, grid_size)
        fig, ax = plt.subplots()
        im = ax.imshow(out)
        ax.set_axis_off()
        ax.set(title="Learned potential")
        fig.colorbar(im, ax=ax)
        return fig


class LinearPotentialShaping(PotentialShaping):
    """A potential shaping preprocessor with a learned linear potential."""

    def __init__(self, model: RewardModel, gamma: float):
        in_size = np.product(model.state_shape)
        potential = nn.Sequential(nn.Flatten(), nn.Linear(in_size, 1))
        super().__init__(model, potential, gamma)


class MlpPotentialShaping(PotentialShaping):
    """A potential shaping preprocessor with a learned MLP potential.
    ReLU activation functions and 2 hidden layers.

    Args:
        state_shape: shape of the observations the environment produces
            (these observations have to be arrays of floats, discrete
            observations are not supported)
        hidden_size (optional): number of neurons in each hidden layer
    """

    def __init__(self, model: RewardModel, gamma: float, hidden_size: int = 64):
        in_size = np.product(model.state_shape)
        potential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        super().__init__(model, potential, gamma)


class TabularPotentialShaping(PotentialShaping):
    """A preprocessor that adds a learned potential shaping in a tabular setting."""

    def __init__(self, model: RewardModel, gamma: float):
        super().__init__(model, self._potential, gamma)
        in_size = np.product(model.state_shape)
        self._data = nn.Parameter(torch.zeros(in_size))

    def _potential(self, states):
        positions = get_agent_positions(states)
        return self._data[positions]

    @property
    def potential_data(self) -> torch.Tensor:
        return self._data

    def plot(self, env: gym.Env) -> plt.Figure:
        fig, ax = plt.subplots()

        im = ax.imshow(
            self.potential_data.detach()
            .cpu()
            .numpy()
            .reshape(*env.observation_space.shape)
        )
        ax.set_axis_off()
        ax.set(title="Learned potential")
        fig.colorbar(im, ax=ax)
        return fig


class RandomPotentialShaping(TabularPotentialShaping):
    """A preprocessor that adds a random potential shaping."""

    def __init__(
        self, model: RewardModel, gamma: float, mean: float = 0, std: float = 1
    ):
        super().__init__(model, gamma)
        self._data = torch.nn.Parameter(
            std * torch.randn_like(self._data) + mean, requires_grad=False
        )
