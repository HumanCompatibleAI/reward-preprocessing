import itertools
import random
from typing import List

import gym
from gym.spaces import Discrete
from mazelab import BaseEnv, BaseMaze
from mazelab import DeepMindColor as color
from mazelab import Object, VonNeumannNoOpMotion
import numpy as np
import sacred


class Maze(BaseMaze):
    def __init__(self, maze_array: np.ndarray):
        self.data = maze_array
        super().__init__()

    @property
    def size(self):
        return self.data.shape

    def make_objects(self):
        free = Object(
            "free",
            0,
            color.free,
            False,
            np.stack(np.where(self.data == 0), axis=1),
        )
        obstacle = Object(
            "obstacle",
            1,
            color.obstacle,
            True,
            np.stack(np.where(self.data == 1), axis=1),
        )
        key = Object("key", 2, color.button, False, [])
        agent = Object("agent", 3, color.agent, False, [])
        goal = Object("goal", 4, color.goal, False, [])
        return free, obstacle, key, agent, goal


class MazeEnv(BaseEnv):
    def __init__(
        self,
        size: int = 10,
        random_start: bool = True,
        reward: str = "goal",
        shaping: str = "unshaped",
        key: bool = False,
        gamma: float = 0.99,
    ):
        # among other things, this calls self.seed() so that the self.rng
        # object exists
        super().__init__()

        n_obs = size ** 2
        # if we are using a key, we have twice as many observations
        if key:
            n_obs *= 2

        x = np.zeros((size, size))
        self.size = size
        self.start_idx = [[size - 2, 1]]
        self.goal_idx = [[size - 2, size - 2]]
        self.using_key = key
        self.has_key = False
        self.key_idx = [[1, 1]]
        self.random_start = random_start
        self.gamma = gamma

        self.maze = Maze(x)
        self.motions = VonNeumannNoOpMotion()
        self.observation_space = Discrete(n_obs)
        self.action_space = Discrete(len(self.motions))

        # The remainder of this function computes a lookup table for rewards,
        # which makes certain things such as plotting easier than computing
        # them on the fly.
        self.rewards = np.zeros((self.observation_space.n, self.observation_space.n))

        if reward == "goal":
            pass
        elif reward == "path":
            self.rewards -= 0.4
            diagonal = size * np.arange(size) + np.arange(size)
            off_diagonal = diagonal[:-1] + 1
            self.rewards[diagonal] = 0
            self.rewards[off_diagonal] = 0
        else:
            raise ValueError(f"Unknown reward type {reward}")

        if shaping == "unshaped":
            pass
        elif shaping == "dense":
            for i, j in itertools.product(range(size), repeat=2):
                pos = self._to_idx((i, j))

                potential = -(
                    abs(i - self.goal_idx[0][0]) + abs(j - self.goal_idx[0][1])
                )

                self.rewards[pos] -= potential
                self.rewards[:, pos] += self.gamma * potential
        elif shaping == "antidense":
            for i, j in itertools.product(range(size), repeat=2):
                pos = self._to_idx((i, j))

                potential = abs(i - self.goal_idx[0][0]) + abs(j - self.goal_idx[0][1])

                self.rewards[pos] -= potential
                self.rewards[pos] += self.gamma * potential
        elif shaping == "random":
            # sample potential uniformly between [-1, 1]
            rng = np.random.default_rng(seed=0)
            potential = 2 * rng.random(size ** 2) - 1
            for i, j in itertools.product(range(size), repeat=2):
                pos = self._to_idx((i, j))

                self.rewards[pos] -= potential[pos]
                self.rewards[pos] += self.gamma * potential[pos]
        else:
            raise ValueError(f"Unknown shaping type {shaping}")

        if key:
            # Copy the reward values to the section with the key
            x_coords, y_coords = np.meshgrid(
                range(size ** 2), range(size ** 2), indexing="ij"
            )
            self.rewards[x_coords + size ** 2, y_coords] = self.rewards[
                x_coords, y_coords
            ]
            self.rewards[x_coords, y_coords + size ** 2] = self.rewards[
                x_coords, y_coords
            ]
            self.rewards[x_coords + size ** 2, y_coords + size ** 2] = self.rewards[
                x_coords, y_coords
            ]

        # Finally, add the actual sparse goal reward
        goal_pos = size * self.goal_idx[0][0] + self.goal_idx[0][1]
        if key:
            goal_pos += size ** 2
        self.rewards[goal_pos, :] = 1.0

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
        else:
            new_position = current_position

        if (
            len(self.maze.objects.key.positions) > 0
            and new_position == self.maze.objects.key.positions[0]
        ):
            self.has_key = True
            self.maze.objects.key.positions = []

        done = False
        reward = self._reward(current_position, action, new_position, self.has_key)
        return self._get_obs(), reward, done, {}

    def _step(self, state, action):
        """Compute a step from an arbitrary starting state."""
        motion = self.motions[action]

        x_pos, y_pos, has_key = self._to_coords(state)
        new_position = [
            x_pos + motion[0],
            y_pos + motion[1],
        ]
        valid = self._is_valid(new_position)
        if not valid:
            new_position = [x_pos, y_pos]

        new_has_key = has_key
        if (
            len(self.maze.objects.key.positions) > 0
            and new_position == self.maze.objects.key.positions[0]
        ):
            new_has_key = True

        new_state = self._to_idx(new_position) + new_has_key * self.size ** 2
        return new_state, valid

    def _reward(self, state, action, next_state, has_key) -> float:
        return self.rewards[
            self._to_idx(state) + has_key * self.size ** 2,
            self._to_idx(next_state) + has_key * self.size ** 2,
        ]

    def _to_idx(self, position):
        return self.size * position[0] + position[1]

    def _to_coords(self, idx):
        has_key = idx // self.size ** 2
        assert has_key in [0, 1]
        idx = idx % self.size ** 2
        return idx // self.size, idx % self.size, has_key

    def _get_obs(self):
        current_position = self.maze.objects.agent.positions[0]
        return self._to_idx(current_position) + self.has_key * self.size ** 2

    def reset(self):
        self.maze.objects.goal.positions = self.goal_idx
        if self.using_key:
            self.maze.objects.key.positions = self.key_idx
        self.has_key = False
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
                if (
                    pos not in self.maze.objects.goal.positions
                    and pos not in self.maze.objects.key.positions
                )
            ]
            self.maze.objects.agent.positions = [
                list(self.rng.choice(available_positions)),
            ]
        else:
            self.maze.objects.agent.positions = [self.start_idx]
        return self._get_obs()

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


gym.register(
    "reward_preprocessing/EmptyMaze10-v0",
    entry_point=MazeEnv,
    max_episode_steps=20,
    kwargs={"size": 10},
)

gym.register(
    "reward_preprocessing/KeyMaze10-v0",
    entry_point=MazeEnv,
    max_episode_steps=30,
    kwargs={"size": 10, "key": True},
)

gym.register(
    "reward_preprocessing/EmptyMaze4-v0",
    entry_point=MazeEnv,
    max_episode_steps=8,
    kwargs={"size": 4},
)

gym.register(
    "reward_preprocessing/KeyMaze6-v0",
    entry_point=MazeEnv,
    max_episode_steps=20,
    kwargs={"size": 6, "key": True},
)


def use_config(
    ex: sacred.Experiment,
) -> None:
    @ex.named_config
    def dense():
        env_make_kwargs = {"shaping": "dense"}
        locals()  # make flake8 happy

    @ex.named_config
    def antidense():
        env_make_kwargs = {"shaping": "antidense"}
        locals()  # make flake8 happy

    @ex.named_config
    def random():
        env_make_kwargs = {"shaping": "random"}
        locals()  # make flake8 happy

    @ex.named_config
    def unshaped():
        env_make_kwargs = {"shaping": "unshaped"}
        locals()  # make flake8 happy

    @ex.named_config
    def path():
        env_make_kwargs = {"reward": "path"}
        locals()  # make flake8 happy

    @ex.named_config
    def goal():
        env_make_kwargs = {"reward": "goal"}
        locals()  # make flake8 happy
