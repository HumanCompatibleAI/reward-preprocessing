from dataclasses import dataclass, field
import math
from queue import PriorityQueue
from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from sacred import Ingredient
import torch

from reward_preprocessing.models import RewardModel
from reward_preprocessing.transition import get_transitions
from reward_preprocessing.utils import sacred_save_fig


@dataclass(order=True)
class TransitionData:
    priority: float
    # We only want to compare based on priority.
    # See https://docs.python.org/3/library/queue.html#queue.PriorityQueue
    actual_reward: float = field(compare=False)
    predicted_reward: float = field(compare=False)
    img: Tuple[np.ndarray, np.ndarray] = field(compare=False)


transition_ingredient = Ingredient("transition_visualization")


@transition_ingredient.config
def config():
    enabled = True
    num = 6  # number of transitions to plot
    num_samples = 250  # number of transitions to sample (to select the max from)
    _ = locals()  # make flake8 happy
    del _


@transition_ingredient.capture
def visualize_transitions(
    model: RewardModel,
    env: gym.Env,
    device,
    num: int,
    num_samples: int,
    enabled: bool,
    _run,
    agent=None,
) -> None:
    """Visualizes a reward model by rendering certain interesting transitions
    together with the rewards predicted by the model."""
    if not enabled:
        return
    # we collect the transitions with the highest and lowest
    # rewards, as well as random ones
    highest = PriorityQueue()
    lowest = PriorityQueue()
    random = PriorityQueue()
    env.reset()
    img = env.render(mode="rgb_array")
    for i, (transition, actual_reward) in enumerate(
        get_transitions(env, agent, num=num_samples)
    ):
        # we add a batch singleton dimension to the front
        # use np.array because that works both if the field is already
        # an array (such as the state) and if it's a scalar (such as done)
        transition = transition.apply(lambda x: np.array([x]))
        transition = transition.apply(torch.from_numpy)
        transition = transition.apply(lambda x: x.float().to(device))
        predicted_reward = model(transition).item()

        next_img = env.render(mode="rgb_array")

        # TODO: storing all the images won't scale
        # to other environments. Since we only need `num`
        # entries anyway, we can limit the queue size to `num`
        # and drop later entries
        highest.put(
            TransitionData(
                # first field is the priority. Because Python's
                # PriorityQueue returns elements from lowest to
                # highest, we need a minus sign to get the
                # highest rewards
                -predicted_reward,
                actual_reward,
                predicted_reward,
                (img, next_img),
            )
        )
        lowest.put(
            TransitionData(
                predicted_reward,
                actual_reward,
                predicted_reward,
                (img, next_img),
            )
        )
        random.put(
            TransitionData(
                np.random.rand(),
                actual_reward,
                predicted_reward,
                (img, next_img),
            )
        )
        img = next_img

    n_cols = 2
    n_rows = math.ceil(num / n_cols)
    for queue, title, filename in zip(
        [highest, lowest, random],
        [
            "Highest reward transitions",
            "Lowest reward transitions",
            "Random transitions",
        ],
        ["highest_transitions", "lowest_transitions", "random_transitions"],
    ):
        fig, ax = plt.subplots(
            n_rows, 2 * n_cols, squeeze=False, figsize=(5 * n_rows, 5 * n_cols)
        )
        fig.suptitle(title)
        ax = ax.reshape(-1)
        for i in range(0, 2 * num, 2):
            data = queue.get()
            ax[i].imshow(data.img[0])
            ax[i].text(
                1.1,
                0.5,
                round(data.predicted_reward, 2),
                size=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax[i].transAxes,
            )
            ax[i + 1].imshow(data.img[1])
            ax[i].set_axis_off()
            ax[i + 1].set_axis_off()
        fig.tight_layout()

        sacred_save_fig(fig, _run, filename)
