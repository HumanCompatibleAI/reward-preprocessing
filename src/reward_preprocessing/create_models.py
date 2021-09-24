import torch

from reward_preprocessing.models import EmptyMazeRewardNet

if __name__ == "__main__":
    model = EmptyMazeRewardNet()
    torch.save(model, "results/ground_truth_models/empty_maze.pt")

    model = EmptyMazeRewardNet(shaping="dense")
    torch.save(model, "results/ground_truth_models/empty_maze_dense.pt")

    model = EmptyMazeRewardNet(shaping="antidense")
    torch.save(model, "results/ground_truth_models/empty_maze_antidense.pt")

    model = EmptyMazeRewardNet(reward="path")
    torch.save(model, "results/ground_truth_models/empty_maze_path.pt")

    model = EmptyMazeRewardNet(reward="path", shaping="dense")
    torch.save(model, "results/ground_truth_models/empty_maze_path_dense.pt")
