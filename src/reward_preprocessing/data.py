"""Module for generating trajectories and transition distributions."""
import math
from typing import Callable, List, Mapping, NamedTuple, Optional, Sequence

from imitation.data.rollout import (
    flatten_trajectories_with_rew,
    generate_trajectories,
    make_sample_until,
)
from imitation.data.types import AnyPath, TrajectoryWithRew, TransitionsWithRew
from imitation.data.types import load as load_trajectories
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
import torch

from reward_preprocessing.policy import get_policy
from reward_preprocessing.types import VenvFactory


class RolloutConfig(NamedTuple):
    """Specification of a source of trajectories.

    If `rollout_path` is set (pointing to a pickled Sequence[TrajectoryWithRew]),
    then `random_prob` and `agent_path` must be their defaults.
    In this case, the trajectories come directly from this static dataset.

    Alternatively, trajectories can be generated dynamically (which happens
    if `rollout_path` is `None`). In that case, each action is chosen uniformly
    at random with probability `random_prob` and from the given policy otherwise.
    """

    random_prob: float = 0.0
    agent_path: Optional[str] = None
    name: str = "<unnamed>"
    rollout_path: Optional[str] = None
    weight: float = 1.0


def get_trajectories(
    cfg: RolloutConfig,
    venv_factory: VenvFactory,
    min_timesteps: Optional[int] = None,
    min_episodes: Optional[int] = None,
    seed: int = 0,
) -> Sequence[TrajectoryWithRew]:
    if cfg.rollout_path is not None:
        if cfg.agent_path is not None:
            raise ValueError("Both rollout_path and agent_path are set")
        if cfg.random_prob > 0:
            raise ValueError("random actions not possible with static dataset")

        return _get_static_trajectories(cfg.rollout_path, min_timesteps, min_episodes)

    else:
        venv = venv_factory(_seed=seed)
        return _get_dynamic_trajectories(
            venv, cfg.random_prob, cfg.agent_path, min_timesteps, min_episodes
        )


def _get_static_trajectories(
    path: AnyPath, min_timesteps: Optional[int], min_episodes: Optional[int]
) -> Sequence[TrajectoryWithRew]:
    trajectories = load_trajectories(path)
    if min_episodes is not None and len(trajectories) < min_episodes:
        raise RuntimeError("Not enough trajectories in pickled file")

    if min_timesteps is not None:
        out_trajectories = _get_enough_trajectories(trajectories, min_timesteps)
        if min_episodes is None or len(out_trajectories) >= min_episodes:
            return out_trajectories
    return trajectories[:min_episodes]


def _get_dynamic_trajectories(
    venv: VecEnv,
    random_prob: float,
    agent_path: Optional[AnyPath],
    min_timesteps: Optional[int],
    min_episodes: Optional[int],
) -> Sequence[TrajectoryWithRew]:
    policy = get_policy(random_prob, agent_path, venv.action_space)

    return generate_trajectories(
        policy,
        venv,
        sample_until=make_sample_until(min_timesteps, min_episodes),
    )


def get_transition_dataset(
    rollouts: Sequence[RolloutConfig],
    venv_factory: VenvFactory,
    num: Optional[int] = None,
    seed: int = 0,
) -> TransitionsWithRew:
    """Helper function to create a transition dataset based on RolloutConfigs."""
    all_trajectories: List[Sequence[TrajectoryWithRew]] = []
    weights = [cfg.weight for cfg in rollouts]
    # Normalize:
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # TODO: This check is very hacky. We need it because we don't want
    # to have to specify a number of trajectories for a fixed dataset.
    # But there's probably a better way to design this.
    if num is None:
        if len(rollouts) == 1 and rollouts[0].rollout_path is not None:
            all_trajectories = [get_trajectories(rollouts[0], venv_factory)]
            num_transitions = [None]
        else:
            raise ValueError(
                "number of timesteps must be given unless "
                "the trajectory source is a single file"
            )
    else:
        num_transitions = [math.ceil(num * w) for w in weights]
        for i, (min_timesteps, cfg) in enumerate(zip(num_transitions, rollouts)):
            all_trajectories.append(
                get_trajectories(cfg, venv_factory, min_timesteps, seed=seed + i)
            )

    # we don't want to flatten all trajectories together since we want to
    # get the proportions right on the level of transitions and we might
    # have oversampled before.
    all_transitions = [
        flatten_trajectories_with_rew(trajectories) for trajectories in all_trajectories
    ]

    # For n=None (in the special case from above), this won't remove any
    # transitions, i.e. return the entire dataset. Otherwise, get only
    # as many transitions as we asked for.
    all_transitions = [
        transitions[:n] for n, transitions in zip(num_transitions, all_transitions)
    ]

    # Finally, stack all the transitions together
    out = TransitionsWithRew(
        obs=np.concatenate([t.obs for t in all_transitions]),
        acts=np.concatenate([t.acts for t in all_transitions]),
        next_obs=np.concatenate([t.next_obs for t in all_transitions]),
        rews=np.concatenate([t.rews for t in all_transitions]),
        dones=np.concatenate([t.dones for t in all_transitions]),
        infos=np.concatenate([t.infos for t in all_transitions]),
    )

    # make sure we return exactly num transitions
    # TODO: there is a bias here by discarding transitions at the end
    # but it should be small because the number of transitions for each
    # rollout config is off by at most 1.
    return out[:num]


# TODO: this is similar to the version from imitation,
# we just return Transitions instead of dicts and therefore also
# don't convert to pytorch. The entire data
# processing in reward_preprocessing is a bit of a mess and should
# be refactored again
def transitions_collate_fn(
    batch: Sequence[Mapping[str, np.ndarray]],
) -> TransitionsWithRew:
    return TransitionsWithRew(
        obs=np.array([sample["obs"] for sample in batch]),
        acts=np.array([sample["acts"] for sample in batch]),
        infos=np.array([sample["infos"] for sample in batch]),
        next_obs=np.array([sample["next_obs"] for sample in batch]),
        dones=np.array([sample["dones"] for sample in batch]),
        rews=np.array([sample["rews"] for sample in batch]),
    )


def get_dataloader(
    batch_size: int,
    rollouts: Sequence[RolloutConfig],
    venv_factory: Callable[..., VecEnv],
    num_workers: int = 0,
    seed: int = 0,
    num: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Create a dataloader for a dataset of transitions.

    Args:
        rollouts: a list of RolloutConfigs describing
            the different trajectory sources to use and how to weight them.
        batch_size (int): batch size for the dataloaders
        venv_factory: Should be a Callable that returns a new instance of the env
            to be used whenever it is called and accepts a `_seed` argument.
        num_workers (int, optional): number of dataloader workers. Defaults to 0.
        seed (int, optional): random seed for env and policy, only relevant if
            dynamic data is used.
        num (int, optional): number of samples.

    Returns:
        torch.utils.data.DataLoader: a dataloader which returns TransitionsWithRew
    """
    dataset = get_transition_dataset(rollouts, venv_factory, num, seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=transitions_collate_fn,
        num_workers=num_workers,
        shuffle=True,
    )


# TODO: this function will also be in imitation soon,
# should reuse from there probably
def _get_enough_trajectories(
    trajectories: Sequence[TrajectoryWithRew], steps: int
) -> Sequence[TrajectoryWithRew]:
    """Get enough trajectories to have at least `steps` transitions in total."""
    available_steps = sum(len(traj) for traj in trajectories)
    if available_steps < steps:
        raise RuntimeError(
            f"Asked for {steps} transitions but only {available_steps} available"
        )
    # We need the cumulative sum of trajectory lengths
    # to determine how many trajectories to return:
    steps_cumsum = np.cumsum([len(traj) for traj in trajectories])
    # Now we find the first index that gives us enough
    # total steps:
    idx = (steps_cumsum >= steps).argmax()
    # we need to include the element at position idx
    trajectories = trajectories[: idx + 1]
    # sanity check
    assert sum(len(traj) for traj in trajectories) >= steps
    return trajectories
