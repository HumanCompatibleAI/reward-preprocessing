import random
from typing import Callable, Optional, Union

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

# A PolicyCallable is a function that takes an array of observations
# and returns an array of corresponding actions.
PolicyCallable = Callable[[np.ndarray], np.ndarray]
# A Policy can be a PolicyCallable, but also a BaseAlgorithm/BasePolicy
# or a VecEnv (which means that random actions will be chosen)
Policy = Union[VecEnv, PolicyCallable, BaseAlgorithm, BasePolicy]


def policy_to_callable(
    policy: Policy, deterministic_policy: bool = True
) -> PolicyCallable:
    """Turn a policy defined in any valid way into a Callable for getting actions.

    Args:
        policy (Policy): either a VecEnv (for random actions), a stable baselines
            policy or algorithm, or a Callable (which will be returned unchanged)
        deterministic_policy (bool, optional): Whether to use deterministic actions
            or stochastic ones. Only relevant if a stable-baselines policy is used.
            Defaults to True.

    Returns:
        PolicyCallable: a function that takes in an array of observations
            and returns an array of actions for those observations.
    """
    if isinstance(policy, VecEnv):

        def get_actions(states):
            acts = []
            for _ in range(len(states)):
                acts.append(policy.action_space.sample())
            return np.stack(acts, axis=0)

    elif isinstance(policy, (BaseAlgorithm, BasePolicy)):

        def get_actions(states):
            acts, _ = policy.predict(states, deterministic=deterministic_policy)
            return acts

    elif isinstance(policy, Callable):
        get_actions = policy

    else:
        raise TypeError(
            "Policy must be a VecEnv, a stable-baselines policy or algorithm, "
            f"or a Callable, got {type(policy)} instead"
        )

    return get_actions


def get_policy(
    random_prob: float = 1,
    expert_path: Optional[str] = None,
    venv: Optional[VecEnv] = None,
) -> PolicyCallable:
    """Create a policy (which can be passed to get_transitions etc.)
    based on config values.

    Args:
        random_prob (float, optional): Probability at each step of
            using a random action. Defaults to 1, i.e. return a completely
            random policy. Set to 0 to use an expert policy without noise.
        expert_path (str, optional): path to a trained agent model.
            Needs to be set if random_prob < 1.
        venv (VecEnv, optional): environment to use for sampling random
            actions, only needed if random_prob > 0.
    """
    if random_prob > 0 and venv is None:
        raise ValueError("random_prob is > 0, so venv must be set")
    if random_prob < 1 and expert_path is None:
        raise ValueError("random_prob is < 1, so expert_path must be set")

    # type checkers don't realize that venv and expert_path can't
    # be None in the relevant branches, which is why we have
    # the # type: ignore comments

    if random_prob == 1:
        return policy_to_callable(venv)  # type: ignore

    agent = PPO.load(expert_path)  # type: ignore

    if random_prob == 0:
        return policy_to_callable(agent)

    return mix_policies(venv, agent, random_prob)  # type: ignore


def mix_policies(
    policy1: Policy, policy2: Policy, proportion: float, seed: int = 0
) -> Callable:
    """Create a policy that chooses randomly at each step between
    two policies.

    Args:
        policy1: first policy (can be a Callable, a stable baselines policy/algorithm
            or a VecEnv for random actions)
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
