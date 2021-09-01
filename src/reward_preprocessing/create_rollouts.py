"""Experiment that creates a dataset of transition-reward pairs,
which can then be used to train a reward model in a supervised fashion.
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
from sacred import Experiment
from tqdm import tqdm

from reward_preprocessing.datasets import DynamicRewardData, RolloutConfig
from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.policy import get_policy
from reward_preprocessing.utils import add_observers, use_rollouts

ex = Experiment("create_rollouts", ingredients=[env_ingredient])
add_observers(ex)
_, get_dataset = use_rollouts(ex)


@ex.config
def config():
    # path where the dataset should be saved to (without .npz extension)
    save_path = ""
    run_dir = "runs/rollouts"
    rollout_steps = 1000  # steps per continuous rollout
    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(
    save_path: str,
    steps: int,
    test_steps: int,
    rollouts: Optional[List[RolloutConfig]],
    rollout_steps: int,
    _seed,
):
    if rollouts is None:
        raise ValueError(
            "For dataset creation, 'rollouts' must be set. "
            "Using a stored dataset is not possible."
        )

    states = {}
    actions = {}
    next_states = {}
    rewards = {}
    dones = {}

    for mode, num_samples in zip(["train", "test"], [steps, test_steps]):
        states[mode] = []
        actions[mode] = []
        next_states[mode] = []
        rewards[mode] = []
        dones[mode] = []
        dataset = get_dataset(venv_factory=create_env, steps=num_samples)
        for transition, reward in tqdm(dataset):
            states[mode].append(transition.state)
            actions[mode].append(transition.action)
            next_states[mode].append(transition.next_state)
            rewards[mode].append(reward)
            dones[mode].append(transition.done)

    # The previous datasets won't necessarily be in order,
    # first because of parallel environments, but also because we
    # might use multiple different rollout strategies and mix them.
    # Now we create additional dataset consisting of consecutive
    # transitions using a fixed policy.

    # turn the rollout configs from ReadOnlyLists into RolloutConfigs
    # (Sacred turns the namedtuples into lists)
    rollouts = [RolloutConfig(*x) for x in rollouts]

    for i, cfg in enumerate(rollouts):
        # environments are seeded by DynamicRewardData, this will already
        # ensure they all have different seeds
        venv = create_env(n_envs=1)
        policy = get_policy(cfg.random_prob, cfg.agent_path, venv)
        dataset = DynamicRewardData(venv, policy, seed=_seed + i, num=rollout_steps)
        mode = f"rollout_{cfg.name}"

        states[mode] = []
        actions[mode] = []
        next_states[mode] = []
        rewards[mode] = []
        dones[mode] = []
        for transition, reward in tqdm(dataset):
            states[mode].append(transition.state)
            actions[mode].append(transition.action)
            next_states[mode].append(transition.next_state)
            rewards[mode].append(reward)
            dones[mode].append(transition.done)

    data = {}
    for mode in states:
        data[f"{mode}_states"] = np.stack(states[mode], axis=0)
        data[f"{mode}_actions"] = np.array(actions[mode])
        data[f"{mode}_next_states"] = np.stack(next_states[mode], axis=0)
        data[f"{mode}_rewards"] = np.array(rewards[mode])
        data[f"{mode}_dones"] = np.array(dones[mode])

    path = Path(save_path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(str(path), **data)
    ex.add_artifact(path)
