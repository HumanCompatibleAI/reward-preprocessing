import itertools
import os

import torch

from reward_preprocessing.models import MazeRewardNet, MountainCarRewardNet

if __name__ == "__main__":
    os.makedirs("results/ground_truth_models", exist_ok=True)
    sizes = [4, 10]
    rewards = ["goal", "path"]
    shapings = ["unshaped", "dense", "antidense", "random"]
    for size, reward, shaping in itertools.product(sizes, rewards, shapings):
        model = MazeRewardNet(
            size=size, maze_name="EmptyMaze", reward=reward, shaping=shaping
        )
        torch.save(
            model,
            f"results/ground_truth_models/empty_maze_{size}_{reward}_{shaping}.pt",
        )

    for reward, shaping in itertools.product(rewards, shapings):
        model = MazeRewardNet(
            size=6, maze_name="KeyMaze", reward=reward, shaping=shaping
        )
        torch.save(
            model,
            f"results/ground_truth_models/key_maze_6_{reward}_{shaping}.pt",
        )

    for shaping in shapings:
        model = MountainCarRewardNet(shaping=shaping)
        torch.save(
            model, f"results/ground_truth_models/shaped_mountain_car_{shaping}.pt"
        )
