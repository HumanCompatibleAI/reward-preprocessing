from pathlib import Path
import tempfile
from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from sacred import Ingredient
import torch

from reward_preprocessing.models import RewardModel
from reward_preprocessing.transition import get_transitions

rollout_ingredient = Ingredient("rollout_visualization")


@rollout_ingredient.config
def config():
    enabled = True
    plot_shape = (4, 4)
    save_path = None
    _ = locals()  # make flake8 happy
    del _


@rollout_ingredient.capture
def visualize_rollout(
    model: RewardModel,
    env: gym.Env,
    plot_shape: Tuple[int, int],
    save_path: str,
    enabled: bool,
    _run,
    agent=None,
) -> None:
    """Visualizes a reward model by rendering a rollout together with the
    rewards predicted by the model."""
    if not enabled:
        return
    n_rows, n_cols = plot_shape
    plt.figure(figsize=(4 * n_rows, 4 * n_cols))
    for i, (transition, actual_reward) in enumerate(
        get_transitions(env, agent, num=n_rows * n_cols)
    ):
        done = transition.done

        # we add a batch singleton dimension to the front
        # use np.array because that works both if the field is already
        # an array (such as the state) and if it's a scalar (such as done)
        transition = transition.apply(lambda x: np.array([x]))
        transition = transition.apply(torch.from_numpy)
        transition = transition.apply(lambda x: x.float())
        predicted_reward = model(transition).item()

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(env.render(mode="rgb_array"))
        plt.axis("off")
        title = f"{predicted_reward:.2f} ({actual_reward:.2f})"
        if done:
            title += ", done"
        plt.title(title)

    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname)
        # save the model
        if save_path:
            plot_path = Path(save_path)
        else:
            plot_path = path / "rollout.pdf"
        plt.savefig(plot_path)
        _run.add_artifact(plot_path)
