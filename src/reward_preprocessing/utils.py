import importlib
from pathlib import Path
import tempfile
from typing import Callable, Tuple

import gym
import matplotlib.pyplot as plt
import sacred
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs

from reward_preprocessing.data import (
    RolloutConfig,
    get_dataloader,
    get_transition_dataset,
)


class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        infos["rendering"] = self.env.render(mode="rgb_array")
        return (observations, rewards, dones, infos)


class ContinuousVideoRecorder(VecVideoRecorder):
    """Modification of the VecVideoRecorder that doesn't restart
    the video when an episode ends.
    """

    def reset(self, start_video=False) -> VecEnvObs:
        obs = self.venv.reset()
        if start_video:
            self.start_video_recorder()
        return obs


def add_observers(ex: sacred.Experiment) -> None:
    """Add a config hook to a Sacred Experiment which will add configurable observers.

    A 'run_dir' config field must exist for the Experiment.
    """

    def helper(config, command_name, logger):
        # Just to be safe, we check whether an observer already exists,
        # to avoid adding multiple copies of the same observer
        # (see https://github.com/IDSIA/sacred/issues/300)
        if len(ex.observers) == 0:
            ex.observers.append(sacred.observers.FileStorageObserver(config["run_dir"]))

    ex.config_hook(helper)


def use_rollouts(
    ing: sacred.Ingredient,
) -> Tuple[Callable, Callable]:
    """Add a config scope to a Sacred Experiment which will add configs
    needed for using rollouts.

    Returns a two capture functions that only need to be passed a venv factory.
    The first returns a dataloader, the second an instance of TransitionsWithRew.
    """

    def config():
        rollouts = None  # each element should be a RolloutConfig instance
        num_workers = 0  # number of workers for the Dataloader
        steps = None  # number of train transitions
        test_steps = None  # number of test transitions
        batch_size = 32  # how many transitions per batch

        _ = locals()  # make flake8 happy
        del _

    ing.config(config)

    def random_rollouts():
        rollouts = [RolloutConfig(random_prob=1, name="random")]
        _ = locals()  # make flake8 happy
        del _

    ing.named_config(random_rollouts)

    def _get_dataloader(
        venv_factory,
        batch_size,
        num_workers,
        _seed,
        rollouts,
        steps,
        test_steps,
        train=True,
    ):
        # turn the rollout configs from ReadOnlyLists into RolloutConfigs
        # (Sacred turns the namedtuples into lists)
        if rollouts is not None:
            rollouts = [RolloutConfig(*x) for x in rollouts]
        return get_dataloader(
            batch_size,
            rollouts,
            venv_factory,
            num_workers,
            # use different seed for testing
            _seed + int(train),
            steps if train else test_steps,
        )

    def _get_dataset(
        venv_factory,
        _seed,
        rollouts,
        steps,
        test_steps,
        train=True,
    ):
        # turn the rollout configs from ReadOnlyLists into RolloutConfigs
        # (Sacred turns the namedtuples into lists)
        if rollouts is not None:
            rollouts = [RolloutConfig(*x) for x in rollouts]

        return get_transition_dataset(
            rollouts=rollouts,
            venv_factory=venv_factory,
            num=steps if train else test_steps,
            # use different seed for testing
            seed=_seed + int(train),
        )

    return ing.capture(_get_dataloader), ing.capture(_get_dataset)


def sacred_save_fig(fig: plt.Figure, run, filename: str) -> None:
    """Save a matplotlib figure as a Sacred artifact.

    Args:
        fig (plt.Figure): the Figure to be saved
        run: the Sacred run instance (can be obtained via _run
            in captured functions)
        filename (str): the filename for the figure (without extension).
            May also consist of folders, then this hierarchy
            will be respected in the run directory for the Experiment.
    """
    with tempfile.TemporaryDirectory() as dirname:
        plot_path = Path(dirname) / f"{filename}.pdf"
        fig.set_tight_layout(True)
        fig.savefig(plot_path)
        run.add_artifact(plot_path)


def instantiate(module_name: str, class_name: str, **kwargs):
    """Instantiate a class from a string.

    Args:
        module_name (str): the full name of the module the class is located in
        class_name (str): the name of the class to instantiate
        **kwargs: kwargs to pass on to the class constructor

    Returns:
        an instance of the given class
    """
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def get_env_name(env: VecEnv) -> str:
    """Return the name of a vectorized environment (such as 'MountainCar-v0')."""
    # It's only one line but it's a somewhat hard to read and write one
    specs = env.get_attr("spec", indices=[0])
    return specs[0].id
