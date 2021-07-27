from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from sacred import Ingredient
import torch

from reward_preprocessing.models import RewardModel
from reward_preprocessing.transition import get_transitions
from reward_preprocessing.utils import sacred_save_fig

rollout_ingredient = Ingredient("rollout_visualization")


@rollout_ingredient.config
def config():
    enabled = True
    plot_shape = (4, 4)
    _ = locals()  # make flake8 happy
    del _


@rollout_ingredient.capture
def visualize_rollout(
    model: RewardModel,
    env: gym.Env,
    device,
    plot_shape: Tuple[int, int],
    enabled: bool,
    _run,
    agent=None,
) -> None:
    """Visualizes a reward model by rendering a rollout together with the
    rewards predicted by the model."""
    if not enabled:
        return
    n_rows, n_cols = plot_shape
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_rows, 4 * n_cols))
    ax = ax.reshape(-1)
    for i, (transition, actual_reward) in enumerate(
        get_transitions(env, agent, num=n_rows * n_cols)
    ):
        done = transition.done

        # we add a batch singleton dimension to the front
        # use np.array because that works both if the field is already
        # an array (such as the state) and if it's a scalar (such as done)
        transition = transition.apply(lambda x: np.array([x]))
        transition = transition.apply(torch.from_numpy)
        transition = transition.apply(lambda x: x.float().to(device))
        predicted_reward = model(transition).item()

        ax[i].imshow(env.render(mode="rgb_array"))
        ax[i].set_axis_off()
        title = f"{predicted_reward:.2f} ({actual_reward:.2f})"
        if done:
            title += ", done"
        ax[i].set(title=title)

    sacred_save_fig(fig, _run, "rollout")
