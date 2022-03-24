from typing import Tuple

import gym
from imitation.rewards.reward_nets import RewardNet, ShapedRewardNet
import numpy as np
import torch as th
from torch import nn


class TabularRewardNet(RewardNet):
    """Lookup table RewardNet for discrete state spaces."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalize_images: bool = True,
        use_next_state: bool = True,
        use_action: bool = False,
    ):
        if not isinstance(observation_space, gym.spaces.Discrete):
            raise TypeError(
                "TabularRewardNet can only be used with Discrete observation spaces."
            )
        if use_action:
            raise NotImplementedError(
                "use_action not yet implemented for TabularRewardNet"
            )
        # TODO(ejnnr): it's silly to pass normalize_images here, that indicates
        # the design of RewardNet isn't that great yet. Maybe we should have something
        # like a NonTabularRewardNet ABC and put that there?
        super().__init__(observation_space, action_space, normalize_images)
        self.use_next_state = use_next_state
        n = observation_space.n ** 2 if use_next_state else observation_space.n
        self.rewards = nn.Parameter(th.zeros(n))

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ):
        """Compute rewards for a batch of transitions and keep gradients."""
        if not self.use_next_state:
            return self.rewards[state]
        else:
            return self.rewards[self.observation_space.n * state + next_state]

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.
        """
        state_th = th.as_tensor(state, device=self.device, dtype=th.long)
        action_th = th.as_tensor(action, device=self.device)
        next_state_th = th.as_tensor(next_state, device=self.device, dtype=th.long)
        done_th = th.as_tensor(done, device=self.device)

        assert state_th.shape == next_state_th.shape
        return state_th, action_th, next_state_th, done_th


class TabularPotential(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.potential_weights = nn.Parameter(th.zeros(n))

    def forward(self, states):
        return self.potential_weights[states]


class ShapedTabularRewardNet(ShapedRewardNet):
    """Lookup table RewardNet for discrete state spaces with shaping."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalize_images: bool = True,
        use_next_state: bool = False,
        use_action: bool = False,
        discount_factor: float = 0.99,
    ):
        base = TabularRewardNet(
            observation_space,
            action_space,
            normalize_images,
            use_next_state,
            use_action,
        )

        potential = TabularPotential(observation_space.n)

        super().__init__(
            observation_space,
            action_space,
            base,
            potential,
            discount_factor,
            normalize_images,
        )

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        return self.base.preprocess(state, action, next_state, done)
