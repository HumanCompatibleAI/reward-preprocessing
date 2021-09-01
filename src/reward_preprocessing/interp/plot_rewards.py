from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from sacred import Ingredient
import torch

from reward_preprocessing.env.env_ingredient import create_env, env_ingredient
from reward_preprocessing.models import RewardModel
from reward_preprocessing.transition import get_transitions
from reward_preprocessing.utils import sacred_save_fig, use_rollouts

reward_ingredient = Ingredient("rewards", ingredients=[env_ingredient])
get_dataloaders, _ = use_rollouts(reward_ingredient)


@reward_ingredient.config
def config():
    enabled = True
    # TODO: would be nice to determine these automatically
    # but that would require first collecting all rewards.
    # Should be fine though in terms of performance.
    min_reward = -10  # lower bound for the histogram
    max_reward = 10  # upper bound for the histogram
    bins = 20  # number of bins for the histogram
    _ = locals()  # make flake8 happy
    del _


@reward_ingredient.capture
def plot_rewards(
    model: RewardModel,
    device,
    enabled: bool,
    bins: int,
    min_reward: float,
    max_reward: float,
    _run,
) -> None:
    """Visualizes a reward model by rendering the distribution and history
    of rewards."""
    if not enabled:
        return
    # we plot a histogram and rewards over time
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    bin_edges = np.linspace(min_reward, max_reward, bins + 1)
    # will contain the counts for all histogram bins
    actual_hist = np.zeros(bins, dtype=int)
    predicted_hist = np.zeros(bins, dtype=int)

    dataloader, _ = get_dataloaders(create_env)
    for transitions, actual_rewards in dataloader:
        with torch.no_grad():
            predicted_rewards = model(transitions.to(device))
        predicted_hist += np.histogram(predicted_rewards.cpu().numpy(), bin_edges)[0]
        actual_hist += np.histogram(actual_rewards.cpu().numpy(), bin_edges)[0]

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

    sacred_save_fig(fig, _run, "rewards")
