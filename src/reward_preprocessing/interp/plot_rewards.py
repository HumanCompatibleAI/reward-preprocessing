from typing import Mapping, Sequence

from imitation.data.rollout import flatten_trajectories_with_rew
from imitation.rewards.reward_nets import RewardNet
import matplotlib.pyplot as plt
import numpy as np
from sacred import Ingredient
import torch

from reward_preprocessing.data import (
    RolloutConfig,
    get_trajectories,
    transitions_collate_fn,
)
from reward_preprocessing.env.env_ingredient import create_env, env_ingredient
from reward_preprocessing.utils import sacred_save_fig

reward_ingredient = Ingredient("rewards", ingredients=[env_ingredient])


@reward_ingredient.config
def config():
    enabled = True
    # TODO: would be nice to determine these automatically
    # but that would require first collecting all rewards.
    # Should be fine though in terms of performance.
    min_reward = -10  # lower bound for the histogram
    max_reward = 10  # upper bound for the histogram
    # overrides the default from use_rollouts:
    rollout_steps = 1000  # length of the plotted rollouts
    _ = locals()  # make flake8 happy
    del _


@reward_ingredient.capture
def plot_rewards(
    models: Mapping[str, RewardNet],
    enabled: bool,
    min_reward: float,
    max_reward: float,
    rollout_steps: int,
    rollouts: Sequence[RolloutConfig],
    _seed,
    _run,
) -> None:
    """Visualizes a reward model by rendering the distribution and history
    of rewards."""
    if not enabled:
        return

    n_rows = len(rollouts)
    fig, ax = plt.subplots(n_rows, 1, figsize=(4, 4 * n_rows))

    rollouts = [RolloutConfig(*x) for x in rollouts]

    for i, cfg in enumerate(rollouts):
        trajectories = get_trajectories(
            cfg, create_env, min_timesteps=rollout_steps, seed=_seed
        )
        transitions = flatten_trajectories_with_rew(trajectories)
        actual_rewards = transitions.rews
        ax[i].plot(
            actual_rewards,
            label="Ground truth",
            linewidth=1,
        )

        dataloader = torch.utils.data.DataLoader(
            transitions,
            batch_size=32,
            collate_fn=transitions_collate_fn,
            num_workers=0,
            shuffle=False,
        )

        for objective, model in models.items():
            predicted_rewards = np.array([])
            for transitions_batch in dataloader:
                with torch.no_grad():
                    predicted_rewards_batch = model(
                        *model.preprocess(transitions_batch)
                    )
                    predicted_rewards = np.concatenate(
                        [predicted_rewards, predicted_rewards_batch.cpu().numpy()]
                    )

            ax[i].plot(
                predicted_rewards,
                label=f"{objective}",
                linewidth=1,
            )

        ax[i].axhline(0, linewidth=1, color="black")
        ax[i].set_ylim(top=max_reward, bottom=min_reward)
        ax[i].set_xlabel("time step")
        ax[i].set_ylabel("reward")
        ax[i].set(title=cfg.name)
        ax[i].legend()

    sacred_save_fig(fig, _run, "rewards_over_time")
