import torch

from reward_preprocessing.models import MlpRewardModel
from reward_preprocessing.preprocessing import PotentialShaping
from reward_preprocessing.transition import Transition


def test_dummy_potential():
    # create a reward model
    model = MlpRewardModel((5, 5))
    # create a dummy potential shaping: the potential phi(s) is simply
    # the sum of all the entries of s
    gamma = 0.9
    shaping = PotentialShaping(model, torch.sum, gamma)

    # create a random transition
    state = torch.randn(1, 5, 5)
    action = None
    next_state = torch.randn(1, 5, 5)
    transition = Transition(state, action, next_state)

    original = model(transition)
    shaped = shaping(transition)

    assert shaped == original + gamma * torch.sum(next_state) - torch.sum(state)
