"""Module implementing transitions, i.e. (s, a, s') tuples.

Transition is the class representing transitions throughout
this project.

get_transitions is a helper function to generate a set
of transition-reward pairs from an environment and a policy.
"""
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv
import torch


@dataclass
class Transition:
    """A transition in an MDP, consisting of a state, an action and a following state
    (the reward is not part of this class).

    This class is mostly agnostic to the type of the state, action and next state.
    However, it does provide .to(device) as a convenience method that makes
    working with Transitions consisting of tensors easier, and similar methods may
    be added in the future.
    """

    state: Any
    action: Any
    next_state: Any
    # can be a tensor of booleans if we have a batch of transitions
    done: Union[None, bool, torch.Tensor]

    def apply(self, fn):
        state = None if self.state is None else fn(self.state)
        action = None if self.action is None else fn(self.action)
        next_state = None if self.next_state is None else fn(self.next_state)
        done = None if self.done is None else fn(self.done)
        return Transition(state, action, next_state, done)

    def to(self, device):
        return self.apply(lambda x: x.to(device))


def get_transitions(
    venv: VecEnv,
    policy=None,
    deterministic_policy: bool = True,
    num: Optional[int] = None,
) -> Iterator[Tuple[Transition, float]]:
    """Generate transitions using a given environment and policy.

    Note that higher-level functions (such as inside Ingredients)
    should usually use DynamicRewardData instead, created using
    datasets.get_dynamic_dataset() or even with the utils.use_rollouts() helper.

    The reason to implement this as an iterator (rather than just returning
    a list of transitions) is mainly that this allows doing stuff immediately
    after each transition, such as rendering the environment.

    Args:
        venv (VecEnv): a vectorized gym environment
        policy (optional): how to get actions. If None (default), actions
            are sampled randomly. Alternatively, can be a stable-baselines BasePolicy or
            BaseAlgorithm instance. Finally, this can be any function that takes
            in an array of observations and returns an array or list of actions
            for those observations.
        deterministic_policy (bool, optional): Whether to use deterministic actions
            or stochastic ones. Only relevant if a stable-baselines policy is used.
            Defaults to True.
        num (int, optional): Number of transitions to return. If None (default),
            an infinite generator is returned.

    Raises:
        TypeError: if policy has an unexpected type

    Returns:
        Iterator[Tuple[Transition, float]]: an iterator over pairs of Transitions
            and corresponding rewards
    """
    if policy is None:

        def get_actions(states):
            acts = []
            for _ in range(len(states)):
                acts.append(venv.action_space.sample())
            return np.stack(acts, axis=0)

    elif isinstance(policy, (BaseAlgorithm, BasePolicy)):

        def get_actions(states):
            acts, _ = policy.predict(states, deterministic=deterministic_policy)
            return acts

    elif isinstance(policy, Callable):
        get_actions = policy
    else:
        raise TypeError(
            "Policy must be None, a stable-baselines policy or algorithm, "
            f"or a Callable, got {type(policy)} instead"
        )

    if isinstance(policy, BaseAlgorithm):
        # check that the observation and action spaces of policy and environment match
        check_for_correct_spaces(venv, policy.observation_space, policy.action_space)

    states = venv.reset()

    num_sampled = 0
    while num is None or num_sampled < num:
        acts = get_actions(states)
        next_states, rews, dones, infos = venv.step(acts)

        for state, act, next_state, rew, done, info in zip(
            states, acts, next_states, rews, dones, infos
        ):
            if done:
                # actual obs is inaccurate, so we use the one
                # inserted into step info by stable baselines wrapper
                real_ob = info["terminal_observation"]
            else:
                real_ob = next_state

            yield Transition(state, act, real_ob, done), rew
            num_sampled += 1

            if num is not None and num_sampled >= num:
                break

        states = next_states
