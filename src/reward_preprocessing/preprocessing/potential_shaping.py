"""This module defines several preprocessors that add
a potential shaping to the reward (a particular transformation
that leaves the optimal policy unchanged).

A potential shaping is fully defined by its potential, which maps
states to floats. PotentialShaping is the most general case,
while the other classes in this module are helper classes
that use a particular type of potential.
"""
import math
from typing import Callable, Optional, Tuple, Type

import gym
from imitation.data.types import AnyPath, Transitions, path_to_str
from imitation.rewards.reward_nets import RewardNet
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
import torch
from torch import nn

from reward_preprocessing.utils import instantiate

from .preprocessor import Preprocessor


class PotentialShaping(Preprocessor):
    """A preprocessor that adds a potential shaping to the reward.

    Args:
        model: the RewardNet to be wrapped
        potential: a Callable that receives a batch of states and returns
            a batch of scalars.
            If the potential is itself a torch.nn.Module, it will become
            a submodule of this Module.
        gamma: the discount factor
    """

    def __init__(
        self,
        model: RewardNet,
        potential: Callable,
        gamma: float,
        freeze_model: bool = True,
    ):
        super().__init__(model, freeze_model=freeze_model)
        self.potential = potential
        self.gamma = gamma

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        rewards = self.model(state, action, next_state, done)
        if rewards.ndim > 1:
            rewards = rewards.squeeze(dim=1)
        current_potential = self.potential(state)
        if current_potential.ndim > 1:
            current_potential = current_potential.squeeze(dim=1)
        # Make sure that there isn't any unwanted broadcasting
        # (which could happen if one of these has singleton dimensions)
        assert done.shape == current_potential.shape
        next_potential = self.potential(next_state)
        if next_potential.ndim > 1:
            next_potential = next_potential.squeeze(dim=1)
        # if the next state is final, then we set it's potential to zero
        # next_potential *= torch.logical_not(done)
        assert rewards.shape == next_potential.shape
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

    def plot(self) -> plt.Figure:
        """Plot the potential if possible.
        The type of plot depends on the type of potential.

        Raises:
            NotImplementedError: if plotting isn't implemented for this type
                of potential

        Returns:
            plt.Figure: a plot of the potential
        """
        space = self.observation_space
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

        out = self.potential(torch.as_tensor(values)).detach().cpu().numpy()

        out = out.reshape(grid_size, grid_size)
        fig, ax = plt.subplots()
        im = ax.imshow(out)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax)
        return fig


class CriticPotentialShaping(PotentialShaping):
    """Uses the critic from an SB3 ActorCriticPolicy as the potential."""

    def __init__(
        self,
        model: RewardNet,
        path: AnyPath,
        gamma: float,
        algorithm_cls: Type[OnPolicyAlgorithm] = PPO,
    ):
        """Initialize the potential shaping.

        Args:
            model: the RewardNet to be shaped
            path: path to an SB3 OnPolicyAlgorithm's .zip file.
                The algorithm's policy must be an ActorCriticPolicy.
            gamma: discount factor used for potential shaping
            algorithm_cls: the type of OnPolicyAlgorithm used to load
                the file from `path`.
        """
        algorithm = algorithm_cls.load(path_to_str(path))
        policy = algorithm.policy
        assert isinstance(policy, ActorCriticPolicy)

        # the value function isn't meant to be trained:
        for p in policy.parameters():
            p.requires_grad = False
        policy.eval()

        def potential(obs):
            # This function is equivalent to how policy.forward() computes
            # state values but we do only the computations necessary for the
            # value function (ignoring the action probabilities).
            preprocessed_obs = preprocess_obs(
                obs, self.observation_space, normalize_images=self.normalize_images
            )
            features = policy.extract_features(preprocessed_obs)
            shared_latent = policy.mlp_extractor.shared_net(features)
            latent_vf = policy.mlp_extractor.value_net(shared_latent)
            return policy.value_net(latent_vf)

        super().__init__(model, potential, gamma)


class PytorchPotentialShaping(PotentialShaping):
    """A potential shaping where the potential is a torch.nn.Module."""

    def __init__(
        self,
        model: RewardNet,
        potential: nn.Module,
        gamma: float,
        freeze_model: bool = True,
    ):
        super().__init__(model, potential, gamma, freeze_model=freeze_model)

    def plot(self) -> plt.Figure:
        space = self.observation_space
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

        # HACK: This assumes that all parameters are on the same device
        # (and that the module expects input to be on that device).
        # But I think that's always going to be the case for us,
        # and this is less hassle than passing around device arguments all the time
        device = next(
            self.potential.parameters()  # pytype: disable=attribute-error
        ).device
        out = (
            self.potential(torch.as_tensor(values, device=device))
            .detach()
            .cpu()
            .numpy()
        )

        out = out.reshape(grid_size, grid_size)
        fig, ax = plt.subplots()
        im = ax.imshow(out)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax)
        return fig

    def random_init(self, mean: float = 0, std: float = 1) -> None:
        for param in self.potential.parameters():  # pytype: disable=attribute-error
            nn.init.normal_(param, mean=mean, std=std)
            param.requires_grad = False


class LinearPotentialShaping(PytorchPotentialShaping):
    """A potential shaping preprocessor with a learned linear potential."""

    def __init__(self, model: RewardNet, gamma: float):
        in_size = np.product(model.observation_space.shape)
        potential = nn.Sequential(nn.Flatten(), nn.Linear(in_size, 1))
        super().__init__(model, potential, gamma)


class MlpPotentialShaping(PytorchPotentialShaping):
    """A potential shaping preprocessor with a learned MLP potential.
    Uses ReLU activation functions.

    Args:
        model: the RewardNet to be shaped
        gamma: discount factor
        hidden_size: number of neurons in each hidden layer
        num_hidden: number of hidden layers
    """

    def __init__(
        self,
        model: RewardNet,
        gamma: float,
        hidden_size: int = 64,
        num_hidden: int = 1,
        freeze_model: bool = True,
    ):
        in_size = np.product(model.observation_space.shape)
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
        super().__init__(model, potential, gamma, freeze_model=freeze_model)


class TabularPotentialShaping(PotentialShaping):
    """A preprocessor that adds a learned potential shaping in tabular environments."""

    def __init__(self, model: RewardNet, gamma: float, freeze_model: bool = True):
        super().__init__(model, self._potential, gamma, freeze_model=freeze_model)
        assert isinstance(model.observation_space, gym.spaces.Discrete)
        in_size = np.product(model.observation_space.n)
        self._data = nn.Parameter(torch.zeros(in_size))

    def _potential(self, states):
        return self._data[states]

    @property
    def potential_data(self) -> torch.Tensor:
        return self._data

    def random_init(self, mean: float = 0, std: float = 1) -> None:
        nn.init.normal_(self._data, mean=mean, std=std)
        self._data.requires_grad = False

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots()

        # TODO: this is not very robust, only works for square
        # mazes
        n = int(math.sqrt(self.observation_space.n))

        im = ax.imshow(self.potential_data.detach().cpu().numpy().reshape(n, n))
        ax.set_axis_off()
        fig.colorbar(im, ax=ax)
        return fig

    def preprocess(
        self, transitions: Transitions
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.
        """
        state_th = torch.as_tensor(
            transitions.obs, device=self.device, dtype=torch.long
        )
        action_th = torch.as_tensor(transitions.acts, device=self.device)
        next_state_th = torch.as_tensor(
            transitions.next_obs, device=self.device, dtype=torch.long
        )
        done_th = torch.as_tensor(transitions.dones, device=self.device)

        del transitions  # unused

        assert state_th.shape == next_state_th.shape
        return state_th, action_th, next_state_th, done_th

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions without gradients.
        Also performs some preprocessing and numpy conversion.
        """
        state_th = torch.as_tensor(state, device=self.device, dtype=torch.long)
        action_th = torch.as_tensor(action, device=self.device)
        next_state_th = torch.as_tensor(
            next_state, device=self.device, dtype=torch.long
        )
        done_th = torch.as_tensor(done, device=self.device)

        with torch.no_grad():
            rew_th = self(state_th, action_th, next_state_th, done_th)

        rew = rew_th.detach().cpu().numpy().flatten()
        assert rew.shape == state.shape[:1]
        return rew


# dict mapping environment name to potential that will be used by default
DEFAULT_POTENTIALS = {
    "imitation/EmptyMaze4-v0": "TabularPotentialShaping",
    "imitation/EmptyMaze10-v0": "TabularPotentialShaping",
    "seals/MountainCar-v0": "LinearPotentialShaping",
    "seals/HalfCheetah-v0": "LinearPotentialShaping",
}


def instantiate_potential(
    env_name: Optional[str] = None, potential_name: Optional[str] = None, **kwargs
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
                "by setting optimize.potential"
            )
        potential_name = DEFAULT_POTENTIALS[env_name]

    return instantiate(
        "reward_preprocessing.preprocessing.potential_shaping", potential_name, **kwargs
    )
