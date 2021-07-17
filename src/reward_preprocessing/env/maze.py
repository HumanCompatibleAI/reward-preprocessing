import random
from typing import Union

from gym.spaces import Box, Discrete
from mazelab import BaseEnv, BaseMaze
from mazelab import DeepMindColor as color
from mazelab import Object, VonNeumannMotion
import numpy as np
import torch


class Maze(BaseMaze):
    def __init__(self, maze_array: np.ndarray):
        self.data = maze_array
        super().__init__()

    @property
    def size(self):
        return self.data.shape

    def make_objects(self):
        free = Object(
            "free", 0, color.free, False, np.stack(np.where(self.data == 0), axis=1)
        )
        obstacle = Object(
            "obstacle",
            1,
            color.obstacle,
            True,
            np.stack(np.where(self.data == 1), axis=1),
        )
        agent = Object("agent", 2, color.agent, False, [])
        goal = Object("goal", 3, color.goal, False, [])
        return free, obstacle, agent, goal


class MazeEnv(BaseEnv):
    def __init__(self, size: int = 5, random_start: bool = True):
        super().__init__()
        x = np.zeros((size + 2, size + 2))
        x[0] = 1
        x[-1] = 1
        x[:, 0] = 1
        x[:, -1] = 1
        self.start_idx = [[1, 1]]
        self.goal_idx = [[4, 4]]
        self.random_start = random_start

        self.maze = Maze(x)
        self.motions = VonNeumannMotion()

        self.observation_space = Box(
            low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8
        )
        self.action_space = Discrete(len(self.motions))

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [
            current_position[0] + motion[0],
            current_position[1] + motion[1],
        ]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = 1.0
            done = True
        elif not valid:
            reward = 0.0
            done = False
        else:
            reward = 0.0
            done = False
        return self.maze.to_value(), reward, done, {}

    def reset(self):
        if self.random_start:
            self.maze.objects.agent.positions = [
                list(random.choice(self.maze.objects.free.positions))
            ]
        else:
            self.maze.objects.agent.positions = [self.start_idx]
        self.maze.objects.goal.positions = self.goal_idx
        return self.maze.to_value()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = (
            position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        )
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


def get_agent_positions(obs: torch.Tensor) -> Union[torch.Tensor, int]:
    """Get the position(s) of an agent in an observation or a batch of observations."""
    x_size, y_size = obs.shape[-2:]
    batch_shape = obs.shape[:-2]

    goal_positions = (obs == 3).nonzero()
    agent_positions = (obs == 2).nonzero()

    x = torch.empty(batch_shape, dtype=torch.long)
    y = torch.empty(batch_shape, dtype=torch.long)

    # If the agent reaches the goal, it vanishes.
    # So we first fill out the position using the goal
    # positions. Then, we overwrite the positions for
    # those cases where the agent is visible
    if len(batch_shape) == 1:
        x[goal_positions[:, 0]] = goal_positions[:, -2]
        y[goal_positions[:, 0]] = goal_positions[:, -1]

        x[agent_positions[:, 0]] = agent_positions[:, -2]
        y[agent_positions[:, 0]] = agent_positions[:, -1]
    elif len(batch_shape) == 2:
        x[goal_positions[:, 0], goal_positions[:, 1]] = goal_positions[:, -2]
        y[goal_positions[:, 0], goal_positions[:, 1]] = goal_positions[:, -1]

        x[agent_positions[:, 0], agent_positions[:, 1]] = agent_positions[:, -2]
        y[agent_positions[:, 0], agent_positions[:, 1]] = agent_positions[:, -1]
    else:
        raise NotImplementedError(
            "only batch shapes of length 1 or 2 are currently implemented"
        )

    return x + y * x_size