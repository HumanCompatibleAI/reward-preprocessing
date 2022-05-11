from imitation.scripts.config.train_rl import train_rl_ex
from imitation.scripts.train_rl import main_console

from reward_preprocessing.env.maze import use_config

use_config(train_rl_ex)


@train_rl_ex.named_config
def empty_maze_10():
    common = dict(env_name="reward_preprocessing/EmptyMaze10-v0")
    total_timesteps = int(5e5)
    normalize = False
    locals()  # make flake8 happy


@train_rl_ex.named_config
def empty_maze_4():
    common = dict(env_name="reward_preprocessing/EmptyMaze4-v0")
    total_timesteps = int(1e5)
    normalize = False
    locals()  # make flake8 happy


@train_rl_ex.named_config
def key_maze_10():
    common = dict(env_name="reward_preprocessing/KeyMaze10-v0")
    total_timesteps = int(5e5)
    normalize = False
    locals()  # make flake8 happy


@train_rl_ex.named_config
def key_maze_6():
    common = dict(env_name="reward_preprocessing/KeyMaze6-v0")
    total_timesteps = int(5e5)
    normalize = False
    locals()  # make flake8 happy


if __name__ == "__main__":  # pragma: no cover
    main_console()
