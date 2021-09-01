import importlib
from pathlib import Path
import tempfile
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
import sacred
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs

from reward_preprocessing.datasets import (
    RolloutConfig,
    get_data_loaders,
    get_dynamic_dataset,
    to_torch,
)

EnvFactory = Callable[[], VecEnv]


class ContinuousVideoRecorder(VecVideoRecorder):
    """Modification of the VecVideoRecorder that doesn't restart
    the video when an episode ends.
    """

    def reset(self, start_video=False) -> VecEnvObs:
        obs = self.venv.reset()
        if start_video:
            self.start_video_recorder()
        return obs


class ComposeTransforms:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for trafo in self.transforms:
            x = trafo(x)
        return x


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

    Returns a capture function that only needs to be passed a venv factory
    function and returns train and test dataloaders.
    """

    def config():
        # path to a dataset to load transitions from (without extension)
        data_path = None
        rollouts = None  # each element should be a RolloutConfig instance
        # number of workers for the Dataloader (only for static dataset)
        num_workers = 0
        steps = 10000  # number of train transitions
        test_steps = 5000  # number of test transitions
        batch_size = 32  # how many transitions per batch

        _ = locals()  # make flake8 happy
        del _

    ing.config(config)

    def random_rollouts():
        rollouts = [RolloutConfig(random_prob=1)]
        _ = locals()  # make flake8 happy
        del _

    ing.named_config(random_rollouts)

    def _get_data_loaders(
        env,
        batch_size,
        num_workers,
        _seed,
        data_path,
        rollouts,
        steps,
        test_steps,
    ):
        # turn the rollout configs from ReadOnlyLists into RolloutConfigs
        # (Sacred turns the namedtuples into lists)
        if rollouts is not None:
            rollouts = [RolloutConfig(*x) for x in rollouts]
        return get_data_loaders(
            batch_size,
            num_workers,
            _seed,
            data_path,
            env,
            rollouts,
            steps,
            test_steps,
            transform=to_torch,
        )

    def _get_dataset(
        venv_factory,
        _seed,
        rollouts,
        steps,
        transform=None,
    ):
        # turn the rollout configs from ReadOnlyLists into RolloutConfigs
        # (Sacred turns the namedtuples into lists)
        if rollouts is not None:
            rollouts = [RolloutConfig(*x) for x in rollouts]
        return get_dynamic_dataset(
            venv_factory=venv_factory,
            rollouts=rollouts,
            seed=_seed,
            num=steps,
            transform=transform,
        )

    return ing.capture(_get_data_loaders), ing.capture(_get_dataset)


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
