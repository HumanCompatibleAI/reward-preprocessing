"""Module for datasets consisting of transition-reward pairs."""
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import warnings

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv
import torch

from reward_preprocessing.transition import Transition, get_transitions


class StoredRewardData(torch.utils.data.Dataset):
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
        transform: Optional[Callable] = None,
    ):
        # If you change the data loading code here, also change it
        # in get_worker_init_fn
        npz = np.load(path.with_suffix(".npz"))
        self.mode = "train" if train else "test"
        self.data = {
            "states": npz[f"{self.mode}_states"],
            "actions": npz[f"{self.mode}_actions"],
            "next_states": npz[f"{self.mode}_next_states"],
            "rewards": npz[f"{self.mode}_rewards"],
            "dones": npz[f"{self.mode}_dones"],
        }
        self.transform = transform
        self.state_shape = self.data["states"].shape[1:]
        self.action_shape = self.data["actions"].shape[1:]

    def __getitem__(self, k):
        state = self.data["states"][k]
        action = self.data["actions"][k]
        next_state = self.data["next_states"][k]
        reward = self.data["rewards"][k]
        done = self.data["dones"][k]

        out = Transition(state, action, next_state, done), reward

        if self.transform:
            out = self.transform(out)

        return out

    def __len__(self):
        return self.data["states"].shape[0]


class DynamicRewardData(torch.utils.data.IterableDataset):
    """Dataset of transition-reward pairs.

    In contrast to StoredRewardData, transitions are generated
    on the fly using a trained agent or by sampling actions
    randomly.

    Args:
        venv (VecEnv): a vectorized gym environment
        policy (optional): how to get actions. If None (default), actions
            are sampled randomly. Alternatively, can be a stable-baselines BasePolicy or
            BaseAlgorithm instance. Finally, this can be any function that takes
            in an array of observations and returns an array or list of actions
            for those observations.
        deterministic_policy (bool, optional): Whether to use deterministic actions
            or stochastic ones. Only relevant if a stable-baselines policy is used.
            Defaults to True.
        num (int, optional): Number of transitions to return. If None (default),
            an infinite generator is returned.
        transform: transforms to apply
    """

    def __init__(
        self,
        venv: VecEnv,
        policy=None,
        deterministic_policy: bool = True,
        num: Optional[int] = None,
        transform: Optional[Callable] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.transform = transform
        self.venv = venv
        self.policy = policy
        self.deterministic_policy = deterministic_policy
        self.num = num
        self.state_shape = venv.observation_space.shape

        # seed environment and policy
        self.venv.seed(seed)
        # the action space uses a distinct random seed from the environment
        # itself, which is important if we use randomly sampled actions
        self.venv.action_space.np_random.seed(seed)
        if isinstance(self.policy, (BasePolicy, BaseAlgorithm)):
            self.policy.set_random_seed(seed)

    def __iter__(self):
        for out in get_transitions(
            self.venv, self.policy, self.deterministic_policy, self.num
        ):
            if self.transform:
                out = self.transform(out)
            yield out


def to_torch(x: Tuple[Transition, float]) -> Tuple[Transition, torch.Tensor]:
    transition, reward = x
    # when we want pytorch tensors, we'll almost always want float as the dtype
    # to pass it into our models
    transition = transition.apply(lambda x: torch.as_tensor(x, dtype=torch.float32))
    reward = torch.tensor(reward).float()

    return transition, reward


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
            torch.stack([t.done for t, r in data]),
        ),
        torch.stack([r for t, r in data]),
    )


def _get_stored_data_loaders(
    batch_size: int,
    num_workers: int,
    data_path: str,
    num_train: Optional[int] = None,
    num_test: Optional[int] = None,
    transform: Optional[Callable] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    path = Path(data_path)
    train_data = StoredRewardData(path, transform=transform, train=True)
    test_data = StoredRewardData(path, transform=transform, train=False)
    if num_train is not None and num_train > len(train_data):
        warnings.warn("Fewer training samples than asked for are available.")
    if num_test is not None and num_test > len(test_data):
        warnings.warn("Fewer test samples than asked for are available.")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader


def _get_dynamic_data_loaders(
    batch_size: int,
    seed: int,
    venv: VecEnv,
    policy=None,
    num_train: Optional[int] = None,
    num_test: Optional[int] = None,
    transform: Optional[Callable] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if num_train is None or num_test is None:
        warnings.warn("No number of samples given, will return an infinite DataLoader.")
    train_data = DynamicRewardData(
        venv, policy, num=num_train, transform=transform, seed=seed
    )
    test_data = DynamicRewardData(
        venv, policy, num=num_test, transform=transform, seed=seed
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader


def get_data_loaders(
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    data_path: Optional[str] = None,
    venv: Optional[VecEnv] = None,
    policy=None,
    num_train: Optional[int] = None,
    num_test: Optional[int] = None,
    transform: Optional[Callable] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test Pytorch dataloaders for transitions.

    The underlying Dataset will be either StoredRewardData or DynamicRewardData,
    depending on the arguments passed to this function.

    If you want to use a stored dataset, pass a `data_path`. Otherwise,
    pass a `venv` and optionally a `policy` and `num_train`, `num_test`
    (to get a finite dataloader).

    Args:
        batch_size (int): batch size for the dataloaders
        num_workers (int, optional): number of dataloader workers. Defaults to 0.
        seed (int, optional): random seed for env and policy, only relevant if
            dynamic data is used.
        data_path (str, optional): path to a stored dataset for StoredRewardData.
        venv (VecEnv, optional): environment to use for rollouts, only required if
            `data_path` is None.
        policy (optional): policy, see `get_transitions` documentation for possible
            values. If left to None, random actions are chosen.
        num_train (int, optional): number of training samples.
        num_test (int, optional): number of test samples.
        transform (Callable, optional): transforms to pass on to the Dataset.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: a train and
            a test dataloader which return batches of (Transition, reward) tuples
    """
    if data_path is not None:
        # if a path to a dataset is given, we use that
        if policy is not None:
            raise ValueError(
                "Both policy and path to a dataset were given, can only use one."
            )

        return _get_stored_data_loaders(
            batch_size, num_workers, data_path, num_train, num_test, transform
        )

    # if no path is given, we return a dataloader that generates samples
    # dynamically
    if venv is None:
        raise ValueError("Path to dataset or an environment are required.")
    if num_workers > 0:
        raise ValueError(
            "Multiple workers are currently not supported for dynamic datasets."
        )
    return _get_dynamic_data_loaders(
        batch_size, seed, venv, policy, num_train, num_test, transform
    )
