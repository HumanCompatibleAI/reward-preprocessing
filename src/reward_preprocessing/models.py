from typing import Tuple

import gym
from imitation.data.types import Transitions
from imitation.envs import maze
from imitation.rewards.reward_nets import RewardNet
import numpy as np
import torch


class EmptyMazeRewardNet(RewardNet):
    def __init__(self, **kwargs):
        env = gym.make("imitation/EmptyMaze-v0", **kwargs)
        self.rewards = env.rewards
        super().__init__(
            observation_space=env.observation_space, action_space=env.action_space
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        np_state = state.detach().cpu().numpy()
        np_next_state = next_state.detach().cpu().numpy()
        rewards = self.rewards[np_state, np_next_state]
        return torch.as_tensor(rewards, device=state.device)

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
