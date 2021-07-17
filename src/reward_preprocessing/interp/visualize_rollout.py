from pathlib import Path
import tempfile
from typing import Tuple

import gym
import matplotlib.pyplot as plt
from sacred import Ingredient
import torch

from reward_preprocessing.models import RewardModel
from reward_preprocessing.transition import Transition

rollout_ingredient = Ingredient("rollout_visualization")


@rollout_ingredient.config
def config():
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
    _run,
    agent=None,
):
    """Visualizes a reward model by rendering a rollout together with the
    rewards predicted by the model."""
    state = env.reset()
    n_rows, n_cols = plot_shape
    plt.figure(figsize=(4 * n_rows, 4 * n_cols))
    i = 0
    while i < n_rows * n_cols:
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(env.render(mode="rgb_array"))

        if agent:
            action, _ = agent.predict(state, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, actual_reward, done, info = env.step(action)

        if done[0]:
            next_state = info[0]["terminal_observation"][None]
        else:
            next_state = obs

        transition = Transition(state, action, next_state)
        transition = transition.apply(torch.from_numpy)
        transition = transition.apply(lambda x: x.float())
        predicted_reward = model(transition).item()

        state = obs

        plt.axis("off")
        title = f"{predicted_reward:.2f} ({actual_reward[0]:.2f})"
        if done[0]:
            title += ", done"
        plt.title(title)
        i += 1

    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname)
        # save the model
        if save_path:
            plot_path = Path(save_path)
        else:
            plot_path = path / "rollout.pdf"
        plt.savefig(plot_path)
        _run.add_artifact(plot_path)
