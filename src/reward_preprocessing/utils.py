from typing import Callable, List

import sacred
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


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
    def __init__(self, transforms: List[Callable]):
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
