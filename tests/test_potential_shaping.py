import gym
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
import numpy as np
import torch

from reward_preprocessing.preprocessing import PotentialShaping


def test_dummy_potential():
    # create a reward model
    space = gym.spaces.Box(low=-1, high=1, shape=(5, 5))
    model = BasicRewardNet(space, space)
    gamma = 0.9

    # create a dummy potential shaping: the potential phi(s) is simply
    # the sum of all the entries of s
    def potential(s):
        # we don't want to sum away the batch dimension
        out = s.view(s.size(0), -1).sum(dim=1)
        assert out.size(0) == s.size(0)
        assert out.ndim == 1
        return out

    shaping = PotentialShaping(model, potential, gamma)

    np.random.seed(0)
    # create a random transition
    state = np.random.randn(1, 5, 5)
    action = np.random.randn(1, 5, 5)
    next_state = np.random.randn(1, 5, 5)
    done = np.array([False])
    transition = Transitions(state, action, None, next_state, done)

    original = model(*model.preprocess(transition))
    shaped = shaping(*model.preprocess(transition))

    assert shaped == original + gamma * potential(
        torch.as_tensor(next_state, dtype=torch.float32)
    ) - potential(torch.as_tensor(state, dtype=torch.float32))

    # now check that the terminal state has potential 0
    done = np.array([True])
    transition = Transitions(state, action, None, next_state, done)

    original = model(*model.preprocess(transition))
    shaped = shaping(*model.preprocess(transition))

    assert shaped == original - potential(torch.as_tensor(state, dtype=torch.float32))
