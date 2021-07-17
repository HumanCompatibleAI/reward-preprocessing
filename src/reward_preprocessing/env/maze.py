import random
from typing import List

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
        self.start_idx = [[1, 1]]
        self.goal_idx = [[4, 4]]
        self.random_start = random_start

        self.maze = Maze(x)
        self.motions = VonNeumannMotion()

        self.observation_space = Box(
            low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8
        )
        self.action_space = Discrete(len(self.motions))

    def seed(self, seed: int = 0) -> List[int]:
        super().seed(seed)
        self.rng = random.Random(seed)
        return [seed]

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
        self.maze.objects.goal.positions = self.goal_idx
        if self.random_start:
            available_positions = [
                pos
                # free positions are stored as a numpy array, we need a list
                # to compare to goal position
                for pos in self.maze.objects.free.positions.tolist()
                # The "free" object positions are not all empty tiles!
                # Multiple objects can be at one position, and "free"
                # just means that there is no wall there, but the goal
                # might still be on this field. So we need to filter that
                # out because the agent shouldn't start on top of the goal.
                if pos not in self.maze.objects.goal.positions
            ]
            self.maze.objects.agent.positions = [
                list(self.rng.choice(available_positions))
            ]
        else:
            self.maze.objects.agent.positions = [self.start_idx]
        return self.maze.to_value()

    def _is_valid(self, position):
        # position indices must be non-negative
        if position[0] < 0 or position[1] < 0:
            return False
        # position indices must not be out of bounds
        if position[0] >= self.maze.size[0] or position[1] >= self.maze.size[1]:
            return False
        # position must be passable
        if self.maze.to_impassable()[position[0]][position[1]]:
            return False
        return True

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


def get_agent_positions(obs: torch.Tensor) -> torch.Tensor:
    """Get the positions of an agent in a batch of observations.

    If no agent is found, then the goal position is returned instead
    (that's because in the Mazelab environment, the goal hides the
    agent in the terminal state).

    Currently, no checks regarding the number of goals and agents happen
    in this function, make sure not to pass invalid input or the results
    could be extremely weird!
    """
    assert (
        obs.ndim == 3
    ), "observation must have shape (batch_size, x_size, y_size)"
    batch_size, x_size, y_size = obs.shape

    # {agent/goal}_positions have shape (num_{agents/goals}, 3)
    # where the second axis contains indices into the three axes
    # of obs.
    goal_positions = (obs == 3).nonzero()
    agent_positions = (obs == 2).nonzero()
    # TODO: It would be nice to validate here that there is at most one agent
    # in each observation and that if there is no agent, there is exactly
    # one goal. But doing that in vectorized form seems tricky.

    # these will contain the x and y agent positions for all the observations
    x = torch.empty(batch_size, dtype=torch.long)
    y = torch.empty(batch_size, dtype=torch.long)

    # If the agent reaches the goal, the agent vanishes.
    # So we first fill out the position using the goal
    # positions. Then, we overwrite the positions for
    # those cases where the agent is visible
    x[goal_positions[:, 0]] = goal_positions[:, 1]
    y[goal_positions[:, 0]] = goal_positions[:, 2]

    x[agent_positions[:, 0]] = agent_positions[:, 1]
    y[agent_positions[:, 0]] = agent_positions[:, 2]

    # finally, we encode each (x, y) position as a single integer
    return y + x * y_size


def is_terminal(states):
    batch_size = states.size(0)
    # a Mazelab state is terminal if no agent (encoded as 2)
    # is visible (because it is hidden by the goal)
    return (states != 2).view(batch_size, -1).all(dim=1)
