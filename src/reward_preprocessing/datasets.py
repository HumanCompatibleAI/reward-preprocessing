"""Module for datasets consisting of transition-reward pairs."""
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch

from reward_preprocessing.transition import Transition


class RewardData(torch.utils.data.Dataset):
    """Dataset of transition-reward pairs.

    Without any transforms, the outputs are Tuples consisting
    of a Transition instance and a scalar reward value for that
    transition.

    Args:
        path: path to the stored dataset (in .npz format).
            Doesn't need to contain the .npz extension.
        train: whether to use the training or test set.
        transform: transforms to apply
        load_to_memory: whether to load the dataset to memory
            or leave it on disk and read it from there on demand.
            Default is to load to memory (which makes sense for small datasets).
            If you use very large datasets, you may have to change that.
    """

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
        self.state_shape = self.data[f"{self.mode}_states"].shape[1:]

    def __getitem__(self, k):
        state = self.data[f"{self.mode}_states"][k]
        action = self.data[f"{self.mode}_actions"][k]
        next_state = self.data[f"{self.mode}_next_states"][k]
        reward = self.data[f"{self.mode}_rewards"][k]

        out = Transition(state, action, next_state), reward

        if self.transform:
            out = self.transform(out)

        return out

    def __len__(self):
        return self.data[f"{self.mode}_states"].shape[0]

    def close(self):
        self.data.close()


def to_torch(x: Tuple[Transition, float]) -> Tuple[Transition, torch.Tensor]:
    transition, reward = x
    transition = transition.apply(
        lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x)
    )
    # when we want pytorch tensors, we'll almost always want float as the dtype
    # to pass it into our models
    transition = transition.apply(lambda x: x.float())
    return transition, torch.tensor(reward).float()


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


def collate_fn(
    data: List[Tuple[Transition, torch.Tensor]]
) -> Tuple[Transition, torch.Tensor]:
    """Custom collate function for RewardData.

    Since RewardData returns Transition instances, the default Pytorch
    collate function doesn't work for it. Use this one instead:
    >>> dl = DataLoader(dataset, collate_fn=collate_fn, ...)
    """
    return (
        Transition(
            torch.stack([t.state for t, r in data]),
            torch.stack([t.action for t, r in data]),
            torch.stack([t.next_state for t, r in data]),
        ),
        torch.stack([r for t, r in data]),
    )
