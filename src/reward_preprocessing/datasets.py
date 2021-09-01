"""Module for datasets consisting of transition-reward pairs."""
from pathlib import Path
import random
from typing import Callable, NamedTuple, Optional, Sequence, Tuple
import warnings

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv
import torch

from reward_preprocessing.policy import get_policy
from reward_preprocessing.transition import Transition, get_transitions


class RolloutConfig(NamedTuple):
    random_prob: float
    agent_path: Optional[str] = None
    weight: float = 1.0


class MixedDataset(torch.utils.data.IterableDataset):
    """Combine different IterableDatasets in a weighted mix.

    This is somewhat similar to Pytorch's ChainDataset, but it will
    mix the order of datasets randomly and it will stop as soon as
    one dataset is exhausted (so it's not guaranteed to return all
    items in all datasets). This is done to avoid biasing the results
    towards larger datasets.

    The recommended way of using this class together with DynamicRewardData
    is to set num=None in each of the DynamicRewardData instances
    and then set num only for this class (if desired at all).
    This will give a dataset of a fixed size, which can
    also be accessed with len(...).

    Args:
        datasets: list of iterable-style datasets
        weights: weights for each of the datasets. If None (default),
            all datasets are weighted equally.
        num: maximum number of elements to return. If None (default),
            don't restrict the dataset size. The iteration will then
            either stop if a dataset is exhausted or never (if all
            datasets are infinite).
        seed: seed for the internal RNG
    """

    def __init__(
        self,
        datasets: Sequence[torch.utils.data.IterableDataset],
        weights: Optional[Sequence[float]] = None,
        num: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.datasets = datasets
        self.state_shape = datasets[0].state_shape
        for dataset in datasets:
            if dataset.state_shape != self.state_shape:
                raise ValueError("All datasets must have matching state spaces.")
        self.weights = weights
        self.num = num
        self.yielded = 0
        self.rng = random.Random(seed)

    def __iter__(self):
        # if self.num is None, then we just go until a dataset
        # is empty (or forever if they are all infinite).
        # Otherwise, we stop after returning self.num results
        iters = [iter(d) for d in self.datasets]
        while self.num is None or self.yielded < self.num:
            # pick a dataset at random
            iterator = self.rng.choices(iters, self.weights)[0]
            # This may raise a StopIteration exception if the dataset
            # is emptied. But that's what we want: as soon as one
            # of the datasets is empty and is selected,
            # this MixedDataset should also stop the iteration
            # in order to avoid bias.
            yield next(iterator)
            self.yielded += 1

    def __len__(self):
        """Maximum number of transitions in the dataset or None if infinite.

        Warning: the actual number of transitions might be lower
        if one of the constituent datasets is too small!"""
        return self.num


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
        done = self.data[f"{self.mode}_dones"][k]

        out = Transition(state, action, next_state, done), reward

        if self.transform:
            out = self.transform(out)

        return out

    def __len__(self):
        return self.data[f"{self.mode}_states"].shape[0]

    def close(self):
        self.data.close()


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
            self.policy.set_random_seed(seed)  # type: ignore

    def __iter__(self):
        for out in get_transitions(
            self.venv, self.policy, self.deterministic_policy, self.num
        ):
            if self.transform:
                out = self.transform(out)
            yield out

    def __len__(self):
        """Number of transitions in the dataset or None if infinite."""
        return self.num


def to_torch(x: Tuple[Transition, float]) -> Tuple[Transition, torch.Tensor]:
    transition, reward = x
    # when we want pytorch tensors, we'll almost always want float as the dtype
    # to pass it into our models
    transition = transition.apply(lambda x: torch.as_tensor(x, dtype=torch.float32))
    reward = torch.tensor(reward).float()

    return transition, reward


def get_worker_init_fn(path: Path, load_to_memory: bool = True):
    """Initialize the dataset with its own copy of the NpzFile in each worker.
    There is a bug in Numpy (https://github.com/numpy/numpy/issues/18124)
    which leads to errors when a .npz file is loaded in the main process
    and then the NpzFile object created in the main process is accessed in each worker.
    Pytorch creates shallow copies of datasets for each worker, so the NpzFile
    would usually be shared between workers, so this error would occur.
    As a workaround, we reload the .npz file in each worker.

    This is a bit hacky because we duplicate code from the StoredRewardData class
    here (since StoredRewardData should also work on its own, without any DataLoader).
    But so far, I haven't found a better solution.
    """
    mmap_mode = "r" if load_to_memory else None

    def worker_init_fn(worker_id):
        info = torch.utils.data.get_worker_info()
        info.dataset.data = np.load(path.with_suffix(".npz"), mmap_mode=mmap_mode)

    return worker_init_fn


def collate_fn(
    data: Sequence[Tuple[Transition, torch.Tensor]]
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


def get_dynamic_dataset(
    rollouts: Sequence[RolloutConfig],
    venv_factory: Callable[[], VecEnv],
    transform: Optional[Callable] = None,
    seed: int = 0,
    num: Optional[int] = None,
):
    datasets = []
    weights = []
    for i, cfg in enumerate(rollouts):
        # environments are seeded by DynamicRewardData, this will already
        # ensure they all have different seeds
        venv = venv_factory()
        policy = get_policy(cfg.random_prob, cfg.agent_path, venv.action_space)
        train_data = DynamicRewardData(venv, policy, transform=transform, seed=seed + i)
        datasets.append(train_data)
        weights.append(cfg.weight)

    # maybe this is too paranoid but we've already used `seed` for the
    # dynamic reward data, so we use `seed - 1` here
    return MixedDataset(datasets, weights, num, seed - 1)


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
        # workaround for https://github.com/numpy/numpy/issues/18124
        # see docstring of get_worker_init_fn for details
        worker_init_fn=get_worker_init_fn(path),
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # workaround for https://github.com/numpy/numpy/issues/18124
        # see docstring of get_worker_init_fn for details
        worker_init_fn=get_worker_init_fn(path),
        collate_fn=collate_fn,
    )
    return train_loader, test_loader


def _get_dynamic_data_loaders(
    batch_size: int,
    seed: int,
    venv_factory: Callable[[], VecEnv],
    rollouts: Sequence[RolloutConfig],
    num_train: Optional[int] = None,
    num_test: Optional[int] = None,
    transform: Optional[Callable] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if num_train is None or num_test is None:
        warnings.warn("No number of samples given, will return an infinite DataLoader.")

    train_data = get_dynamic_dataset(rollouts, venv_factory, transform, seed, num_train)
    test_data = get_dynamic_dataset(
        rollouts, venv_factory, transform, seed + 1, num_test
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
    venv_factory: Optional[Callable[[], VecEnv]] = None,
    rollouts: Optional[Sequence[RolloutConfig]] = None,
    num_train: Optional[int] = None,
    num_test: Optional[int] = None,
    transform: Optional[Callable] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test Pytorch dataloaders for transitions.

    The underlying Dataset will be either StoredRewardData or DynamicRewardData,
    depending on the arguments passed to this function.

    If you want to use a stored dataset, pass a `data_path`. Otherwise,
    pass a `venv`, a `rollouts` list of configurations
    and optionally `num_train` and `num_test`
    (to get a finite dataloader).

    Args:
        batch_size (int): batch size for the dataloaders
        num_workers (int, optional): number of dataloader workers. Defaults to 0.
        seed (int, optional): random seed for env and policy, only relevant if
            dynamic data is used.
        data_path (str, optional): path to a stored dataset for StoredRewardData.
        venv_factory (optional): only required if `data_path` is None.
            Should be a Callable that returns a new instance of the environment
            to be used whenever it is called. Seeding will happen automatically.
        rollouts (list, optional): a list of RolloutConfigs describing
            the different policies to use and how to weight them.
        num_train (int, optional): number of training samples.
        num_test (int, optional): number of test samples.
        transform (Callable, optional): transforms to pass on to the Dataset.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: a train and
            a test dataloader which return batches of (Transition, reward) tuples
    """
    if data_path is not None:
        # if a path to a dataset is given, we use that
        if rollouts is not None:
            raise ValueError(
                "Both policies and path to a dataset were given, can only use one."
            )

        return _get_stored_data_loaders(
            batch_size, num_workers, data_path, num_train, num_test, transform
        )

    # if no path is given, we return a dataloader that generates samples
    # dynamically
    if venv_factory is None:
        raise ValueError("Path to dataset or an environment are required.")
    if num_workers > 0:
        raise ValueError(
            "Multiple workers are currently not supported for dynamic datasets."
        )
    if rollouts is None:
        raise ValueError(
            "Neither rollout configuration nor dataset path are given, "
            "need one of them."
        )
    return _get_dynamic_data_loaders(
        batch_size, seed, venv_factory, rollouts, num_train, num_test, transform
    )
