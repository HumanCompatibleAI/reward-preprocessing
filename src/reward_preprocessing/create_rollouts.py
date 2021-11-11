import os
from typing import Sequence, Union

from imitation.data.types import save as save_trajectories
from sacred import Experiment

from reward_preprocessing.data import RolloutConfig, get_trajectories
from reward_preprocessing.env import create_env, env_ingredient

ex = Experiment(
    "create_rollouts",
    ingredients=[
        env_ingredient,
    ],
)


@ex.config
def config():
    rollouts = []  # list of RolloutConfigs
    min_timesteps = []  # list of timesteps for the corresponding RolloutConfigs
    out_dir = None  # output directory for the trajectories

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(
    rollouts: Sequence[RolloutConfig],
    min_timesteps: Union[int, Sequence[int]],
    out_dir: str,
    _seed,
):
    if isinstance(min_timesteps, (int, float)):
        min_timesteps = [min_timesteps] * len(rollouts)
    assert len(min_timesteps) == len(rollouts)
    assert out_dir is not None
    rollouts = [RolloutConfig(*x) for x in rollouts]
    for steps, cfg in zip(min_timesteps, rollouts):
        print(f"Sampling for config '{cfg.name}'")
        trajectories = get_trajectories(
            cfg, create_env, min_timesteps=steps, seed=_seed
        )
        save_trajectories(os.path.join(out_dir, f"{cfg.name}.pkl"), trajectories)
