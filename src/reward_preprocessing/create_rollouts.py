"""Experiment that creates a dataset of transition-reward pairs,
which can then be used to train a reward model in a supervised fashion.
"""
from pathlib import Path

import numpy as np
from sacred import Experiment
from tqdm import tqdm

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.utils import add_observers, use_rollouts

ex = Experiment("create_rollouts", ingredients=[env_ingredient])
add_observers(ex)
_, get_dataset = use_rollouts(ex)


@ex.config
def config():
    # path where the dataset should be saved to (without .npz extension)
    save_path = ""
    run_dir = "runs/rollouts"
    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(save_path: str, steps: int, test_steps: int):
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
        dataset = get_dataset(create_env, steps=num_samples)
        for transition, reward in tqdm(dataset):
            states[mode].append(transition.state)
            actions[mode].append(transition.action)
            next_states[mode].append(transition.next_state)
            rewards[mode].append(reward)
            dones[mode].append(transition.done)

    path = Path(save_path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(path),
        train_states=np.stack(states["train"], axis=0),
        train_actions=np.array(actions["train"]),
        train_next_states=np.stack(next_states["train"], axis=0),
        train_rewards=np.array(rewards["train"]),
        train_dones=np.array(dones["train"]),
        test_states=np.stack(states["test"], axis=0),
        test_actions=np.array(actions["test"]),
        test_next_states=np.stack(next_states["test"], axis=0),
        test_rewards=np.array(rewards["test"]),
        test_dones=np.array(dones["test"]),
    )
    ex.add_artifact(path)
