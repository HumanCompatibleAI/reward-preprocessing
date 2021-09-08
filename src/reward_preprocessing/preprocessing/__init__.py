"""Package containing preprocessors, i.e. wrappers around
reward models that transform the reward in some way before it
is interpreted.

Each preprocessor inherits from Preprocessor and is itself
a RewardNet. If possible, all code dealing with RewardNets
should be agnostic to whether the RewardNet is an "unwrapped"
model such as BasicRewardNet, or a Preprocessor wrapping another
model.

Preprocessors can also be easily chained (since they are themselves
RewardNets).
"""
from .potential_shaping import LinearPotentialShaping, PotentialShaping
from .preprocessor import Preprocessor

__all__ = ["Preprocessor", "PotentialShaping", "LinearPotentialShaping"]
