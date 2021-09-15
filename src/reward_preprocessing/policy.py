import random
from typing import Callable, Optional, Union

import gym
from imitation.data.types import AnyPath, path_to_str
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy

# A PolicyCallable is a function that takes an array of observations
# and returns an array of corresponding actions.
PolicyCallable = Callable[[np.ndarray], np.ndarray]
# A Policy can be a PolicyCallable, but also a BaseAlgorithm/BasePolicy
# or a gym action space (which means that random actions will be chosen)
Policy = Union[gym.Space, PolicyCallable, BaseAlgorithm, BasePolicy]


def policy_to_callable(
    policy: Policy, deterministic_policy: bool = True
) -> PolicyCallable:
    """Turn a policy defined in any valid way into a Callable for getting actions.

    Args:
        policy (Policy): either a gym space (for random actions), a stable baselines
            policy or algorithm, or a Callable (which will be returned unchanged)
        deterministic_policy (bool, optional): Whether to use deterministic actions
            or stochastic ones. Only relevant if a stable-baselines policy is used.
            Defaults to True.

    Returns:
        PolicyCallable: a function that takes in an array of observations
            and returns an array of actions for those observations.
    """
    if isinstance(policy, gym.Space):

        def get_actions(states):
            acts = []
            for _ in range(len(states)):
                acts.append(policy.sample())  # type: ignore
            return np.stack(acts, axis=0)

    elif isinstance(policy, (BaseAlgorithm, BasePolicy)):

        def get_actions(states):
            acts, _ = policy.predict(  # type: ignore
                states, deterministic=deterministic_policy
            )
            return acts

    elif isinstance(policy, Callable):
        get_actions = policy

    else:
        raise TypeError(
            "Policy must be a gym.Space, a stable-baselines policy or algorithm, "
            f"or a Callable, got {type(policy)} instead"
        )

    return get_actions


def get_policy(
    random_prob: float = 1,
    agent_path: Optional[AnyPath] = None,
    action_space: Optional[gym.Space] = None,
) -> PolicyCallable:
    """Create a policy (which can be passed to get_transitions etc.)
    based on config values.

    Args:
        random_prob (float, optional): Probability at each step of
            using a random action. Defaults to 1, i.e. return a completely
            random policy. Set to 0 to use an expert policy without noise.
        expert_path (str, optional): path to a trained agent model.
            Needs to be set if random_prob < 1.
        action_space (gym.Space, optional): action space to use for sampling random
            actions, only needed if random_prob > 0.
    """
    if random_prob > 0 and action_space is None:
        raise ValueError("random_prob is > 0, so action_space must be set")
    if random_prob < 1 and agent_path is None:
        raise ValueError("random_prob is < 1, so expert_path must be set")

    if random_prob == 1:
        return policy_to_callable(action_space)

    assert agent_path is not None
    agent = PPO.load(path_to_str(agent_path))
    if random_prob == 0:
        return policy_to_callable(agent)

    return mix_policies(action_space, agent, random_prob)


def mix_policies(
    policy1: Policy, policy2: Policy, proportion: float, seed: int = 0
) -> PolicyCallable:
    """Create a policy that chooses randomly at each step between
    two policies.

    Args:
        policy1: first policy (can be a Callable, a stable baselines policy/algorithm
            or a gym.Space for random actions)
        policy2: same as policy1
        proportion: probability of selecting policy1, must be in [0, 1]
        seed (optional): the seed to use for the RNG that chooses between policies

    Returns:
        Callable: a function that takes in an array of observations
            and returns an array of actions, taken from either policy1
            or policy2 at random.
    """
    if proportion < 0 or proportion > 1:
        raise ValueError("proportion must be in [0, 1]")

    callable1 = policy_to_callable(policy1)
    callable2 = policy_to_callable(policy2)

    rng = random.Random(seed)

    def meta_policy(observations):
        if rng.random() < proportion:
            return callable1(observations)
        else:
            return callable2(observations)

    return meta_policy
