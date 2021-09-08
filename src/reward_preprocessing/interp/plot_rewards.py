from typing import Sequence

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
from reward_preprocessing.env.env_ingredient import (
    create_env,
    create_visualization_env,
    env_ingredient,
)
from reward_preprocessing.utils import sacred_save_fig, use_rollouts

reward_ingredient = Ingredient("rewards", ingredients=[env_ingredient])
get_dataloader, _ = use_rollouts(reward_ingredient)


@reward_ingredient.config
def config():
    enabled = True
    # TODO: would be nice to determine these automatically
    # but that would require first collecting all rewards.
    # Should be fine though in terms of performance.
    min_reward = -10  # lower bound for the histogram
    max_reward = 10  # upper bound for the histogram
    bins = 20  # number of bins for the histogram
    rollout_steps = 1000  # length of the plotted rollouts
    _ = locals()  # make flake8 happy
    del _


@reward_ingredient.capture
def plot_rewards(
    model: RewardNet,
    enabled: bool,
    bins: int,
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
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    bin_edges = np.linspace(min_reward, max_reward, bins + 1)
    # will contain the counts for all histogram bins
    actual_hist = np.zeros(bins, dtype=int)
    predicted_hist = np.zeros(bins, dtype=int)

    dataloader = get_dataloader(create_env)
    for transitions in dataloader:
        actual_rewards = transitions.rews
        with torch.no_grad():
            predicted_rewards = model(*model.preprocess(transitions))
        predicted_hist += np.histogram(predicted_rewards.cpu().numpy(), bin_edges)[0]
        actual_hist += np.histogram(actual_rewards, bin_edges)[0]

    # we have constant width, all bins are the same:
    width = bin_edges[1] - bin_edges[0]
    ax.bar(
        bin_edges[:-1],
        actual_hist,
        width=width,
        align="edge",
        label="Original reward",
        alpha=0.6,
    )
    ax.bar(
        bin_edges[:-1],
        predicted_hist,
        width=width,
        align="edge",
        label="Model output",
        alpha=0.6,
    )
    ax.set(title="Reward distribution")
    ax.legend()

    sacred_save_fig(fig, _run, "reward_histogram")

    fig, ax = plt.subplots(3, 1, figsize=(4, 12))

    rollouts = [RolloutConfig(*x) for x in rollouts]

    for i, cfg in enumerate(rollouts):
        predicted_rewards = np.array([])
        trajectories = get_trajectories(
            cfg, create_visualization_env, min_timesteps=rollout_steps, seed=_seed
        )
        transitions = flatten_trajectories_with_rew(trajectories)
        actual_rewards = transitions.rews

        dataloader = torch.utils.data.DataLoader(
            transitions,
            batch_size=32,
            collate_fn=transitions_collate_fn,
            num_workers=0,
            shuffle=False,
        )

        for transitions_batch in dataloader:
            with torch.no_grad():
                predicted_rewards_batch = model(*model.preprocess(transitions_batch))
            predicted_rewards = np.concatenate(
                [predicted_rewards, predicted_rewards_batch.cpu().numpy()]
            )

        ax[i].plot(
            actual_rewards,
            label="Original reward",
        )
        ax[i].plot(
            predicted_rewards,
            label="Model output",
        )
        ax[i].set_ylim(top=max_reward, bottom=min_reward)
        ax[i].set_xlabel("time step")
        ax[i].set_ylabel("reward")
        ax[i].set(title=cfg.name)
        ax[i].legend()

    sacred_save_fig(fig, _run, "rewards_over_time")
