import itertools
from typing import Mapping

import gym
from imitation.rewards.reward_nets import RewardNet
import matplotlib.pyplot as plt
import numpy as np
from sacred import Ingredient
import torch

from reward_preprocessing.env.env_ingredient import create_env, env_ingredient
from reward_preprocessing.interp.gridworld_plot import (
    ACTION_DELTA,
    Actions,
    plot_gridworld_rewards,
    prepare_rewards,
)
from reward_preprocessing.utils import sacred_save_fig

heatmap_ingredient = Ingredient("heatmaps", ingredients=[env_ingredient])


@heatmap_ingredient.config
def config():
    enabled = True
    _ = locals()  # make flake8 happy
    del _


@heatmap_ingredient.capture
def plot_heatmaps(
    models: Mapping[str, Mapping[str, RewardNet]],
    gamma: float,
    enabled: bool,
    _seed,
    _run,
) -> None:
    if not enabled:
        return

    venv = create_env()
    # TODO: this is very brittle
    env = venv.envs[0].unwrapped
    env.reset()
    if isinstance(env.observation_space, gym.spaces.Discrete):
        rewards = {}

        states = []
        next_states = []
        actions = []
        invalid_indices = []
        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                delta = ACTION_DELTA[Actions(action)]
                x, y = divmod(state, env.size)
                next_x = x + delta[0]
                next_y = y + delta[1]
                if env._is_valid((next_x, next_y)):
                    next_state = env.size * next_x + next_y
                else:
                    next_state = state
                    invalid_indices.append(5 * state + action)
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
        # positions = torch.arange(space.n, device=model.device)
        # states = positions.unsqueeze(1).repeat(1, space.n).flatten()
        # next_states = positions.unsqueeze(0).repeat(space.n, 1).flatten()
        invalid_indices = np.array(invalid_indices)

        states = torch.tensor(states, dtype=torch.long)
        next_states = torch.tensor(next_states, dtype=torch.long)
        dones = torch.zeros(env.observation_space.n * 5, dtype=torch.bool)

        for model_name, model_versions in models.items():
            rewards[model_name] = {}
            for objective, model in model_versions.items():
                with torch.no_grad():
                    out = model(states, None, next_states, dones)

                np_out = out.detach().cpu().numpy()
                np_out[invalid_indices] = np.nan
                np_out = np_out.reshape(env.size, env.size, 5)
                rewards[model_name][objective] = np_out

        # reward_array = env.rewards
        # rewards["ground_truth"] = {
        #     "unmodified": prepare_rewards(torch.as_tensor(reward_array))
        # }

        ncols = min(3, len(rewards))
        # flatten the reward dict
        rewards = {
            f"{name} / {objective}": reward
            for name, reward_versions in rewards.items()
            for objective, reward in reward_versions.items()
        }
        fig = plot_gridworld_rewards(rewards, ncols=ncols, discount=gamma)
        sacred_save_fig(fig, _run, "gridworld_rewards")

    if isinstance(
        env.observation_space, gym.spaces.Box
    ) and env.observation_space.shape != (2,):
        objective = "unmodified"
        model = models[objective]
        grid_size = 100

        low = venv.normalize_obs(env.observation_space.low)
        high = venv.normalize_obs(env.observation_space.high)

        xs = np.linspace(low[0], high[0], grid_size, dtype=np.float32)
        ys = np.linspace(low[1], high[1], grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        values = np.stack([xx, yy], axis=2).reshape(-1, 2)

        # HACK: This assumes that all parameters are on the same device
        # (and that the module expects input to be on that device).
        # But I think that's always going to be the case for us,
        # and this is less hassle than passing around device arguments all the time
        device = next(model.parameters()).device  # pytype: disable=attribute-error
        action = (
            torch.tensor((0, 0, 1), device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(len(values), 1)
        )
        out = (
            model(torch.as_tensor(values, device=device), action, None, None)
            .detach()
            .cpu()
            .numpy()
        )

        out = out.reshape(grid_size, grid_size)
        fig, ax = plt.subplots()
        im = ax.imshow(
            out,
            extent=(low[0], high[0], low[1], high[1]),
            aspect="auto",
        )
        # ax.set_axis_off()
        fig.colorbar(im, ax=ax)

        # expert_cfg = rollouts[0]
        # assert expert_cfg.name == "expert"
        # trajectories = get_trajectories(
        #     expert_cfg, create_env, min_timesteps=rollout_steps, seed=_seed
        # )
        # transitions = flatten_trajectories_with_rew(trajectories)
        # states = transitions.obs[transitions.acts == 2]
        # ax.scatter(states[:, 0], states[:, 1])
        sacred_save_fig(fig, _run, f"reward_model_{objective}")

    venv.close()
