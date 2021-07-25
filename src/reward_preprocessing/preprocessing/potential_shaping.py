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
from reward_preprocessing.utils import instantiate

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

    def random_init(self, **kwargs) -> None:
        """Set the potentials parameters randomly and freeze the weights.

        The idea is to use this in order to get a noise potential,
        sampled from the space of potentials representable by this class.
        We freeze the weights because we usually don't want to train this
        noise.
        """
        # Since the potential may in general be given by any Callable,
        # we can't provide a general purpose implementation.
        # Classes that support random initialization should override
        # this method.
        raise NotImplementedError(
            "Random initialization is not implemented for this type of potential."
        )

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
        fig.colorbar(im, ax=ax)
        return fig


class LinearPotentialShaping(PotentialShaping):
    """A potential shaping preprocessor with a learned linear potential."""

    def __init__(self, model: RewardModel, gamma: float):
        in_size = np.product(model.state_shape)
        potential = nn.Sequential(nn.Flatten(), nn.Linear(in_size, 1))
        super().__init__(model, potential, gamma)

    def random_init(self, mean: float = 0, std: float = 1) -> None:
        for param in self.potential.parameters():  # pytype: disable=attribute-error
            nn.init.normal_(param, mean=mean, std=std)
            param.requires_grad = False


class MlpPotentialShaping(PotentialShaping):
    """A potential shaping preprocessor with a learned MLP potential.
    Uses ReLU activation functions.

    Args:
        state_shape: shape of the observations the environment produces
            (these observations have to be arrays of floats, discrete
            observations are not supported)
        hidden_size (optional): number of neurons in each hidden layer
    """

    def __init__(
        self,
        model: RewardModel,
        gamma: float,
        hidden_size: int = 64,
        num_hidden: int = 1,
    ):
        in_size = np.product(model.state_shape)
        layers = [nn.Flatten(), nn.Linear(in_size, hidden_size), nn.ReLU()]
        if num_hidden < 1:
            raise ValueError(
                "MLP must have at least one hidden layer. "
                "Did you mean to use LinearPotentialShaping instead?"
            )
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))

        potential = nn.Sequential(*layers)
        super().__init__(model, potential, gamma)

    def random_init(self, mean: float = 0, std: float = 1) -> None:
        for param in self.potential.parameters():  # pytype: disable=attribute-error
            nn.init.normal_(param, mean=mean, std=std)
            param.requires_grad = False


class MazelabPotentialShaping(PotentialShaping):
    """A preprocessor that adds a learned potential shaping in Mazelab environment."""

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

    def random_init(self, mean: float = 0, std: float = 1) -> None:
        nn.init.normal_(self._data, mean=mean, std=std)
        self._data.requires_grad = False

    def plot(self, env: gym.Env) -> plt.Figure:
        fig, ax = plt.subplots()

        im = ax.imshow(
            self.potential_data.detach()
            .cpu()
            .numpy()
            .reshape(*env.observation_space.shape)
        )
        ax.set_axis_off()
        fig.colorbar(im, ax=ax)
        return fig


# dict mapping environment name to potential that will be used by default
DEFAULT_POTENTIALS = {
    "EmptyMaze-v0": "MazelabPotentialShaping",
    "MountainCar-v0": "LinearPotentialShaping",
}


def instantiate_potential(
    env_name: str = None, potential_name: str = None, **kwargs
) -> PotentialShaping:
    """Create the right PotentialShaping instance for a given environment.

    Args:
        env_name (str): the environment name, only used if no potential_name is
            given
        potential_name (str, optional): can be the name of a PotentialShaping
            subclass, then that class is used. If not set (default), then
            the function tries to choose a reasonable type of potential shaping
            based on the environment.
        **kwargs: passed on to the PotentialShaping constructor

    Returns:
        a PotentialShaping instance of the given or automatically determined type
    """

    if potential_name is None:
        if env_name is None:
            raise ValueError(
                "Need to specify either a potential name or an environment."
            )
        if env_name not in DEFAULT_POTENTIALS:
            raise ValueError(
                f"No default potential shaping class for environment '{env_name}' "
                "is set. You need to specify the type of potential to use "
                "by setting sparsify.potential"
            )
        potential_name = DEFAULT_POTENTIALS[env_name]

    return instantiate(
        "reward_preprocessing.preprocessing.potential_shaping", potential_name, **kwargs
    )
