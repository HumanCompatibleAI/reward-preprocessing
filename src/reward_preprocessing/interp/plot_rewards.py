import matplotlib.pyplot as plt
import numpy as np
from sacred import Ingredient
import torch

from reward_preprocessing.env.env_ingredient import create_env, env_ingredient
from reward_preprocessing.models import RewardModel
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
    model: RewardModel,
    device,
    enabled: bool,
    bins: int,
    min_reward: float,
    max_reward: float,
    rollout_steps: int,
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

    sacred_save_fig(fig, _run, "reward_histogram")

    fig, ax = plt.subplots(3, 1, figsize=(4, 12))

    for i, mode in enumerate(["expert", "mixed", "random"]):
        all_actual_rewards = np.array([])
        all_predicted_rewards = np.array([])
        dataloader = get_dataloader(
            create_env, mode=f"rollout_{mode}", steps=rollout_steps
        )

        for transitions, actual_rewards in dataloader:
            all_actual_rewards = np.concatenate(
                [all_actual_rewards, actual_rewards.cpu().numpy()]
            )
            with torch.no_grad():
                predicted_rewards = model(transitions.to(device))
            all_predicted_rewards = np.concatenate(
                [all_predicted_rewards, predicted_rewards.cpu().numpy()]
            )

        ax[i].plot(
            all_actual_rewards,
            label="Original reward",
        )
        ax[i].plot(
            all_predicted_rewards,
            label="Model output",
        )
        ax[i].set_ylim(top=max_reward, bottom=min_reward)
        ax[i].set_xlabel("time step")
        ax[i].set_ylabel("reward")
        ax[i].set(title=mode)
        ax[i].legend()

    sacred_save_fig(fig, _run, "rewards_over_time")
