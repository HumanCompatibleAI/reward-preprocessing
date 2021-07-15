"""Package for different reward models.
All reward models inherit from RewardModel, which in particular
makes them Pytorch nn.Modules which take a Transition as input
and return a float tensor (the predicted rewards).

Note that batches of Transitions are represented as a single
Transition with batches in each of its fields for the purposes
of model input. collate_fn in datasets.py takes care of this.
"""
from .mlp_model import MlpRewardModel
from .reward_model import RewardModel

__all__ = ["MlpRewardModel", "RewardModel"]
