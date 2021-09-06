import torch

from reward_preprocessing.models import MlpRewardModel
from reward_preprocessing.preprocessing import PotentialShaping
from reward_preprocessing.transition import Transition


def test_dummy_potential():
    # create a reward model
    model = MlpRewardModel((5, 5))
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

    torch.manual_seed(0)
    # create a random transition
    state = torch.randn(1, 5, 5)
    action = None
    next_state = torch.randn(1, 5, 5)
    done = torch.tensor([False])
    transition = Transition(state, action, next_state, done)

    original = model(transition)
    shaped = shaping(transition)

    assert shaped == original + gamma * potential(next_state) - potential(state)

    # now check that the terminal state has potential 0
    done = torch.tensor([True])
    transition = Transition(state, action, next_state, done)

    original = model(transition)
    shaped = shaping(transition)

    assert shaped == original - potential(state)
