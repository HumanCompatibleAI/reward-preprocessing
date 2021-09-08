from dataclasses import dataclass, field
from heapq import heappush, heappushpop
import math
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


@dataclass(order=True)
class TransitionData:
    priority: float
    # We only want to compare based on priority.
    # See https://docs.python.org/3/library/queue.html#queue.PriorityQueue
    actual_reward: float = field(compare=False)
    predicted_reward: float = field(compare=False)
    img: Tuple[np.ndarray, np.ndarray] = field(compare=False)


class LimitedHeap:
    """Maintain a list of the highest `size` values."""

    def __init__(self, size: int):
        self.size = size
        self.heap = []

    def push(self, x):
        if len(self.heap) >= self.size:
            # heappushpop will push x and then pop the item with
            # the lowest value
            heappushpop(self.heap, x)
        else:
            heappush(self.heap, x)

    def __iter__(self):
        return iter(self.heap)

    def __len__(self):
        return len(self.heap)


transition_ingredient = Ingredient(
    "transition_visualization", ingredients=[env_ingredient]
)
_, get_dataset = use_rollouts(transition_ingredient)


@transition_ingredient.config
def config():
    enabled = True
    num = 6  # number of transitions to plot
    # overrides the default from use_rollouts:
    steps = 1000  # number of transitions to sample
    _ = locals()  # make flake8 happy
    del _


@transition_ingredient.capture
def visualize_transitions(
    model: RewardNet,
    num: int,
    enabled: bool,
    _run,
) -> None:
    """Visualizes a reward model by rendering certain interesting transitions
    together with the rewards predicted by the model."""
    if not enabled:
        return
    # we collect the transitions with the highest and lowest
    # rewards, as well as random ones
    highest = LimitedHeap(num)
    lowest = LimitedHeap(num)
    random = LimitedHeap(num)
    prev_image = None
    for i, transition in enumerate(get_dataset(create_visualization_env)):
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

        image = transition["infos"]["rendering"]
        # skip the first transition, where the previous image
        # isn't set yet
        if prev_image is not None:
            highest.push(
                TransitionData(
                    predicted_reward,
                    actual_reward,
                    predicted_reward,
                    (prev_image, image),
                )
            )
            lowest.push(
                TransitionData(
                    -predicted_reward,
                    actual_reward,
                    predicted_reward,
                    (prev_image, image),
                )
            )
            random.push(
                TransitionData(
                    np.random.rand(),
                    actual_reward,
                    predicted_reward,
                    (prev_image, image),
                )
            )
        prev_image = image

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
        for i, data in zip(range(0, 2 * num, 2), queue):
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
