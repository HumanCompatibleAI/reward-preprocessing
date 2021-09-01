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
    name: str = "<unnamed>"
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
        self.action_shape = datasets[0].action_shape
        for dataset in datasets:
            if dataset.state_shape != self.state_shape:
                raise ValueError("All datasets must have matching state spaces.")
            if dataset.action_shape != self.action_shape:
                raise ValueError("All datasets must have matching action spaces.")
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
        mode: the mode of the dataset to load. Can be "train" (default),
            "test" or "rollout_{i}" where i is one of the available
            contiguous rollouts
        num: if set, restrict the number of samples taken from
            the dataset to this number
        transform: transforms to apply
    """

    def __init__(
        self,
        path: Path,
        mode: str = "train",
        num: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        npz = np.load(path.with_suffix(".npz"))
        self.mode = mode
        self.data = {
            "states": npz[f"{self.mode}_states"][:num],
            "actions": npz[f"{self.mode}_actions"][:num],
            "next_states": npz[f"{self.mode}_next_states"][:num],
            "rewards": npz[f"{self.mode}_rewards"][:num],
            "dones": npz[f"{self.mode}_dones"][:num],
        }
        assert self.data["states"].shape == self.data["next_states"].shape
        assert (
            len(self.data["states"])
            == len(self.data["actions"])
            == len(self.data["rewards"])
            == len(self.data["dones"])
        )

        if num is not None and num > len(self):
            warnings.warn(
                f"Fewer samples than asked for are available (mode: {self.mode})."
            )

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
        self.action_shape = venv.action_space.shape

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


def _get_stored_dataloader(
    batch_size: int,
    num_workers: int,
    data_path: str,
    num: Optional[int] = None,
    transform: Optional[Callable] = None,
    mode: str = "train",
) -> torch.utils.data.DataLoader:
    path = Path(data_path)
    dataset = StoredRewardData(path, transform=transform, mode=mode, num=num)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader


def _get_dynamic_dataloader(
    batch_size: int,
    seed: int,
    venv_factory: Callable[[], VecEnv],
    rollouts: Sequence[RolloutConfig],
    num: Optional[int] = None,
    transform: Optional[Callable] = None,
    mode: str = "train",
) -> torch.utils.data.DataLoader:
    if num is None:
        warnings.warn("No number of samples given, will return an infinite DataLoader.")

    # ensure that the test dataloader uses a different seed
    if mode == "test":
        seed += 1

    train_data = get_dynamic_dataset(rollouts, venv_factory, transform, seed, num)

    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return dataloader


def get_dataloader(
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    data_path: Optional[str] = None,
    venv_factory: Optional[Callable[[], VecEnv]] = None,
    rollouts: Optional[Sequence[RolloutConfig]] = None,
    num: Optional[int] = None,
    transform: Optional[Callable] = None,
    mode: str = "train",
) -> torch.utils.data.DataLoader:
    """Create train and test Pytorch dataloaders for transitions.

    The underlying Dataset will be either StoredRewardData or DynamicRewardData,
    depending on the arguments passed to this function.

    If you want to use a stored dataset, pass a `data_path`. Otherwise,
    pass a `venv`, a `rollouts` list of configurations
    and optionally `num` (to get a finite dataloader).

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
        num (int, optional): number of samples.
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

        return _get_stored_dataloader(
            batch_size, num_workers, data_path, num, transform, mode
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
    return _get_dynamic_dataloader(
        batch_size, seed, venv_factory, rollouts, num, transform, mode
    )
