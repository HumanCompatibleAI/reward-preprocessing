from typing import Callable, List, Tuple, Type

from imitation.algorithms.preference_comparisons import SyntheticGatherer
from imitation.data import rollout
from imitation.data.types import AnyPath, path_to_str
from imitation.scripts.common.reward import reward_ingredient
from imitation.scripts.config.train_preference_comparisons import (
    train_preference_comparisons_ex,
)
from imitation.scripts.train_preference_comparisons import main_console
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
import torch

from reward_preprocessing.env.maze import use_config
from reward_preprocessing.tabular_reward_net import TabularRewardNet

use_config(train_preference_comparisons_ex)


class ShapedPreferenceGatherer(SyntheticGatherer):
    def __init__(
        self,
        potential: Callable[[np.ndarray], np.ndarray],
        **kwargs,
    ):
        """Initialize the reward model trainer.

        Args:
            potential: function mapping batch of states to potentials
            **kwargs: passed on to SyntheticGatherer
        """
        super().__init__(**kwargs)
        self._potential = potential

    def _reward_sums(self, fragment_pairs) -> Tuple[np.ndarray, np.ndarray]:
        # fragment_lists has length 2 and each item is a sequence of trajectories
        fragment_lists = zip(*fragment_pairs)
        # shaped_return_lists will have two items, each one an array with
        # size (number_of_fragments, )
        shaped_return_lists: List[np.ndarray] = []

        # In the outer loop, we iterate over the two lists of fragments
        for fragments in fragment_lists:
            shaped_returns = np.empty(len(fragments))
            # for each such list, we iterate over the fragments and compute
            # the returns of each fragment
            for i, fragment in enumerate(fragments):
                transitions = rollout.flatten_trajectories_with_rew([fragment])
                potential = self._potential(transitions.obs).flatten()
                next_potential = (
                    1 - transitions.dones.astype(np.float32)
                ) * self._potential(
                    transitions.next_obs,
                ).flatten()

                final_rew = (
                    transitions.rews + self.discount_factor * next_potential - potential
                )
                assert final_rew.shape == (transitions,)
                shaped_returns[i] = rollout.discounted_sum(
                    final_rew,
                    self.discount_factor,
                )

            shaped_return_lists.append(shaped_returns)

        return tuple(shaped_return_lists)


class ValueNetPreferenceGatherer(ShapedPreferenceGatherer):
    def __init__(
        self,
        path: AnyPath,
        algorithm_cls: Type[OnPolicyAlgorithm] = PPO,
        **kwargs,
    ):
        """Initialize the reward model trainer.

        Args:
            path: path to an OnPolicyAlgorithm's .zip file. This algorithms
                critic will be used for shaping.
            algorithm_cls: the type of OnPolicyAlgorithm used to load
                the file from `path`.
            **kwargs: passed on to SyntheticGatherer
        """
        algorithm = algorithm_cls.load(path_to_str(path))
        policy = algorithm.policy
        assert isinstance(policy, ActorCriticPolicy)

        # the value function isn't meant to be trained:
        for p in policy.parameters():
            p.requires_grad = False
        policy.eval()
        self._policy = policy

        def _potential(obs):
            th_obs = torch.as_tensor(obs, device=self._policy.device)
            th_obs = preprocess_obs(
                th_obs,
                self._policy.observation_space,
                self._policy.normalize_images,
            )

            # This function is equivalent to how policy.forward() computes
            # state values but we do only the computations necessary for the
            # value function (ignoring the action probabilities).
            with torch.no_grad():
                features = self._policy.extract_features(th_obs)
                shared_latent = self._policy.mlp_extractor.shared_net(features)
                latent_vf = self._policy.mlp_extractor.value_net(shared_latent)
                potential = self._policy.value_net(latent_vf)
            return potential.detach().cpu().numpy()

        super().__init__(potential=_potential, **kwargs)


@train_preference_comparisons_ex.named_config
def shaped():
    value_net_path = None
    gatherer_cls = ValueNetPreferenceGatherer
    gatherer_kwargs = {"path": value_net_path}
    locals()  # make flake8 happy


@train_preference_comparisons_ex.named_config
def empty_maze_10():
    common = dict(env_name="reward_preprocessing/EmptyMaze10-v0")
    fragment_length = 4
    random_frac = 0.5
    total_timesteps = int(5e5)
    total_comparisons = 5000
    comparisons_per_iteration = 300
    normalize = False
    locals()  # make flake8 happy


@train_preference_comparisons_ex.named_config
def empty_maze_4():
    common = dict(env_name="reward_preprocessing/EmptyMaze4-v0")
    fragment_length = 3
    random_frac = 0.5
    total_timesteps = int(1e5)
    total_comparisons = 1000
    comparisons_per_iteration = 100
    normalize = False
    locals()  # make flake8 happy


@train_preference_comparisons_ex.named_config
def key_maze_10():
    common = dict(env_name="reward_preprocessing/KeyMaze10-v0")
    fragment_length = 4
    random_frac = 0.5
    total_timesteps = int(5e5)
    total_comparisons = 5000
    comparisons_per_iteration = 300
    normalize = False
    locals()  # make flake8 happy


@reward_ingredient.named_config
def tabular():
    net_cls = TabularRewardNet
    locals()  # make flake8 happy


@train_preference_comparisons_ex.named_config
def key_maze_6():
    common = dict(env_name="reward_preprocessing/KeyMaze6-v0")
    agent_path = "output/key_maze_6_agent"
    fragment_length = 4
    random_frac = 0.5
    total_timesteps = int(5e5)
    total_comparisons = 5000
    comparisons_per_iteration = 300
    normalize = False
    locals()  # make flake8 happy


if __name__ == "__main__":  # pragma: no cover
    main_console()
