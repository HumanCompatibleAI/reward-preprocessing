from typing import Tuple

from imitation.data.types import Transitions
from imitation.rewards.reward_nets import RewardNet
import matplotlib.pyplot as plt
import numpy as np
from sacred import Ingredient

from reward_preprocessing.env.env_ingredient import (
    create_visualization_env,
    env_ingredient,
)
from reward_preprocessing.utils import sacred_save_fig, use_rollouts

rollout_ingredient = Ingredient("rollout_visualization", ingredients=[env_ingredient])
_, get_dataset = use_rollouts(rollout_ingredient)


@rollout_ingredient.config
def config():
    enabled = True
    plot_shape = (4, 4)  # number of frames along x and y axis
    _ = locals()  # make flake8 happy
    del _


@rollout_ingredient.capture
def visualize_rollout(
    model: RewardNet,
    plot_shape: Tuple[int, int],
    enabled: bool,
    _run,
) -> None:
    """Visualizes a reward model by rendering a rollout together with the
    rewards predicted by the model."""
    if not enabled:
        return
    n_rows, n_cols = plot_shape
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_rows, 4 * n_cols))
    ax = ax.reshape(-1)

    for i, transition in enumerate(
        get_dataset(create_visualization_env, steps=n_rows * n_cols)
    ):
        done = transition["dones"]
        actual_reward = transition["rews"]

        # We add a batch singleton dimension to the front.
        # Use np.array([...]) because that works both if the field is already
        # an array and if it's a scalar.
        inputs = Transitions(
            obs=np.array([transition["obs"]]),
            acts=np.array([transition["acts"]]),
            next_obs=np.array([transition["next_obs"]]),
            infos=None,
            dones=np.array([transition["dones"]]),
        )
        predicted_reward = model(*model.preprocess(inputs)).item()

        ax[i].imshow(transition["infos"]["rendering"])
        ax[i].set_axis_off()
        title = f"{predicted_reward:.2f} ({actual_reward:.2f})"
        if done:
            title += ", done"
        ax[i].set(title=title)

    sacred_save_fig(fig, _run, "rollout")
