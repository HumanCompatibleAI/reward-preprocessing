from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch


class RewardData(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        train: bool = True,
        transform: Callable = None,
        load_to_memory: bool = True,
    ):
        # If you change the data loading code here, also change it
        # in get_worker_init_fn
        mmap_mode = "r" if load_to_memory else None
        self.data = np.load(path.with_suffix(".npz"), mmap_mode=mmap_mode)
        self.mode = "train" if train else "test"
        self.transform = transform
        self.observation_size = np.product(self.data[f"{self.mode}_states"].shape[1:])

    def __getitem__(self, k):
        state = self.data[f"{self.mode}_states"][k]
        action = self.data[f"{self.mode}_actions"][k]
        next_state = self.data[f"{self.mode}_next_states"][k]
        reward = self.data[f"{self.mode}_rewards"][k]

        out = (state, action, next_state), reward

        if self.transform:
            out = self.transform(out)

        return out

    def __len__(self):
        return self.data[f"{self.mode}_states"].shape[0]

    def close(self):
        self.data.close()


class FilterObservations:
    def __init__(self, keep: Iterable[str] = set()):
        self.fields = ["state", "action", "next_state"]
        keep = set(keep)
        if not keep.issubset(self.fields):
            raise ValueError(
                f"Unknown field names: {keep.difference(self.fields)}. "
                f"Valid field names are {self.fields}"
            )
        self.keep = keep

    def __call__(self, x):
        obs, reward = x
        return (
            tuple(obs[i] for i, field in enumerate(self.fields) if field in self.keep),
            reward,
        )


def stack_observations(x):
    obs, reward = x
    return np.stack(obs, axis=0), reward


def to_torch(x):
    obs, reward = x
    return torch.from_numpy(obs).float(), reward


def get_worker_init_fn(path: Path, load_to_memory: bool = True):
    """Initialize the dataset with its own copy of the NpzFile in each worker.
    There is a bug in Numpy (https://github.com/numpy/numpy/issues/18124)
    which leads to errors when a .npz file is loaded in the main process
    and then the NpzFile object created in the main process is accessed in each worker.
    Pytorch creates shallow copies of datasets for each worker, so the NpzFile
    would usually be shared between workers, so this error would occur.
    As a workaround, we reload the .npz file in each worker.

    This is a bit hacky because we duplicate code from the RewardData class
    here (since RewardData should also work on its own, without any DataLoader).
    But so far, I haven't found a better solution.
    """
    mmap_mode = "r" if load_to_memory else None

    def worker_init_fn(worker_id):
        info = torch.utils.data.get_worker_info()
        info.dataset.data = np.load(path.with_suffix(".npz"), mmap_mode=mmap_mode)

    return worker_init_fn
