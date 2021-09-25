import itertools

import torch

from reward_preprocessing.models import EmptyMazeRewardNet, MountainCarRewardNet

if __name__ == "__main__":
    sizes = [4, 10]
    rewards = ["goal", "path"]
    shapings = ["unshaped", "dense", "antidense", "random"]
    for size, reward, shaping in itertools.product(sizes, rewards, shapings):
        model = EmptyMazeRewardNet(size=size, reward=reward, shaping=shaping)
        torch.save(model, f"results/ground_truth_models/empty_maze_{size}_{reward}_{shaping}.pt")

    for shaping in shapings:
        model = MountainCarRewardNet(shaping=shaping)
        torch.save(model, f"results/ground_truth_models/shaped_mountain_car_{shaping}.pt")
