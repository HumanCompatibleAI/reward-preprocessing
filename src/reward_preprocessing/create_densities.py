import math
import os
from typing import Sequence, Union

import gym
from imitation.data.rollout import flatten_trajectories
from matplotlib import colors
import numpy as np
from sacred import Experiment

from reward_preprocessing.data import RolloutConfig, get_trajectories
from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.interp.gridworld_plot import plot_gridworld_rewards

ex = Experiment(
    "create_densities",
    ingredients=[
        env_ingredient,
    ],
)


@ex.config
def config():
    rollouts = []  # list of RolloutConfigs
    min_timesteps = []  # list of timesteps for the corresponding RolloutConfigs
    out_dir = None  # output directory for the densities

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(
    rollouts: Sequence[RolloutConfig],
    min_timesteps: Union[int, Sequence[int]],
    out_dir: str,
    _seed,
):
    env = create_env()
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 5
    size = int(math.sqrt(env.observation_space.n))

    if isinstance(min_timesteps, (int, float)):
        min_timesteps = [min_timesteps] * len(rollouts)
    assert len(min_timesteps) == len(rollouts)
    assert out_dir is not None
    rollouts = [RolloutConfig(*x) for x in rollouts]
    densities = {}
    for steps, cfg in zip(min_timesteps, rollouts):
        print(f"Calculating density for config '{cfg.name}'")
        trajectories = get_trajectories(
            cfg, create_env, min_timesteps=steps, seed=_seed
        )
        transitions = flatten_trajectories(trajectories)
        counts = np.zeros((size ** 2, 5))
        for t in transitions:
            counts[t["obs"], t["acts"]] += 1
        density = counts.reshape(size, size, 5) / len(transitions)
        np.save(os.path.join(out_dir, f"{cfg.name}_densities.npy"), density)
        densities[cfg.name] = density

    ncols = min(3, len(rollouts))
    fig = plot_gridworld_rewards(densities, ncols=ncols, discount=0.99, normalizer=colors.LogNorm, vmin=1e-5)
    fig.savefig(os.path.join(out_dir, "densities.pdf"))

