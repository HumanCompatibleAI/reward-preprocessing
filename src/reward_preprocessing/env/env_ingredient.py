from typing import Any, Iterable, Mapping, Optional

import gym
from imitation.envs import maze, sparse  # noqa: F401
from sacred import Ingredient
import seals  # noqa: F401
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from reward_preprocessing.utils import instantiate

env_ingredient = Ingredient("env")


@env_ingredient.config
def config():
    name = None  # gym environment id
    options = {}  # gym env kwargs
    stats_path = None  # path to stats file for normalization (incl. extension)
    # list of complete gym wrapper names (incl. module), from inner- to outermost
    wrappers = []
    n_envs = 1  # number of parallel environments
    normalize = False  # whether to normalize observations
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def empty_maze_10():
    name = "imitation/EmptyMaze10-v0"
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def empty_maze_4():
    name = "imitation/EmptyMaze4-v0"
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def unshaped():
    options = {"shaping": "unshaped"}
    _ = locals()  # make flake8 happy
    del _

@env_ingredient.named_config
def goal():
    options = {"reward": "goal"}
    _ = locals()  # make flake8 happy
    del _

@env_ingredient.named_config
def dense():
    options = {"shaping": "dense"}
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def antidense():
    options = {"shaping": "antidense"}
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def random():
    options = {"shaping": "random"}
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def path():
    options = {"reward": "path"}
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def mountain_car():
    name = "seals/MountainCar-v0"
    stats_path = "old_results/agents/mountain_car/vec_normalize.pkl"
    normalize = True
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def pendulum():
    name = "Pendulum-v0"
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def half_cheetah():
    name = "seals/HalfCheetah-v0"
    stats_path = "results/agents/half_cheetah/vec_normalize.pkl"
    normalize = True
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def hopper():
    name = "seals/Hopper-v0"
    stats_path = "results/agents/hopper/vec_normalize.pkl"
    normalize = True
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def sparse_reacher():
    name = "imitation/SparseReacher-v0"
    stats_path = "results/agents/sparse_reacher/vec_normalize.pkl"
    normalize = True
    _ = locals()  # make flake8 happy
    del _


def wrap_env(env: gym.Env, wrappers: Iterable[str]) -> gym.Env:
    """Wrap a gym env in multiple wrappers, given by their class name.

    `wrappers` must be an Iterable of complete paths to the
    class, including the package and module.
    """
    for wrapper_name in wrappers:
        # wrapper_name is a complete class, like
        # sb3_contrib.common.wrappers.TimeFeatureWrapper.
        # We want to split that into module and class name:
        module_name, _, cls_name = wrapper_name.rpartition(".")
        env = instantiate(module_name, cls_name, env=env)
    return env


@env_ingredient.capture
def create_env(
    name: str,
    _seed: int,
    options: Mapping[str, Any] = {},
    stats_path: Optional[str] = None,
    n_envs: int = 1,
    normalize: bool = False,
    wrappers: Iterable[str] = [],
):
    # always wrap in a Monitor wrapper, which collects the returns and
    # episode lengths so that the original ones are available if they
    # are modified by other wrappers. This also enables automatic logging
    # of returns.
    wrappers = ["stable_baselines3.common.monitor.Monitor"] + list(wrappers)
    env = DummyVecEnv([lambda: wrap_env(gym.make(name, **options), wrappers)] * n_envs)
    env.seed(_seed)
    # the action space uses a distinct random seed from the environment
    # itself, which is important if we use randomly sampled actions
    env.action_space.np_random.seed(_seed)

    if normalize:
        if stats_path:
            env = VecNormalize.load(stats_path, env)
            # Deactivate training and reward normalization
            env.training = False
            env.norm_reward = False
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=False)

    return env


def create_visualization_env(_seed: int):
    return create_env(
        n_envs=1, wrappers=["reward_preprocessing.utils.RenderWrapper"], _seed=_seed
    )
