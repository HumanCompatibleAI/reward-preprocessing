import math
from typing import Mapping, Sequence

import gym
from imitation.data.rollout import flatten_trajectories_with_rew
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import RewardNet
import matplotlib.pyplot as plt
import numpy as np
from sacred import Ingredient
import torch

from reward_preprocessing.data import (
    RolloutConfig,
    get_trajectories,
    transitions_collate_fn,
)
from reward_preprocessing.env.env_ingredient import create_env, env_ingredient
from reward_preprocessing.interp.gridworld_plot import (
    plot_gridworld_rewards,
    prepare_rewards,
)
from reward_preprocessing.utils import sacred_save_fig

reward_ingredient = Ingredient("rewards", ingredients=[env_ingredient])


@reward_ingredient.config
def config():
    enabled = True
    rollout_steps = 1000  # length of the plotted rollouts
    _ = locals()  # make flake8 happy
    del _


@reward_ingredient.capture
def plot_rewards(
    models: Mapping[str, RewardNet],
    gamma: float,
    enabled: bool,
    rollout_steps: int,
    rollouts: Sequence[RolloutConfig],
    _seed,
    _run,
) -> None:
    """Visualizes a reward model by rendering the distribution and history
    of rewards."""
    if not enabled:
        return

    n_rows = len(rollouts)
    fig, ax = plt.subplots(n_rows, 1, figsize=(4, 4 * n_rows), squeeze=False)
    ax = ax[:, 0]

    rollouts = [RolloutConfig(*x) for x in rollouts]

    for i, cfg in enumerate(rollouts):
        trajectories = get_trajectories(
            cfg, create_env, min_timesteps=rollout_steps, seed=_seed
        )
        transitions = flatten_trajectories_with_rew(trajectories)
        actual_rewards = transitions.rews
        ax[i].plot(
            actual_rewards,
            label="Ground truth",
            linewidth=1,
        )

        dataloader = torch.utils.data.DataLoader(
            transitions,
            batch_size=32,
            collate_fn=transitions_collate_fn,
            num_workers=0,
            shuffle=False,
        )

        for objective, model in models.items():
            predicted_rewards = np.array([])
            for transitions_batch in dataloader:
                with torch.no_grad():
                    predicted_rewards_batch = model(
                        *model.preprocess(transitions_batch)
                    )
                    predicted_rewards = np.concatenate(
                        [predicted_rewards, predicted_rewards_batch.cpu().numpy()]
                    )

            ax[i].plot(
                predicted_rewards,
                label=f"{objective}",
                linewidth=1,
            )

        ax[i].axhline(0, linewidth=1, color="black")
        # ax[i].plot(
        #     transitions.acts,
        #     label="Action",
        #     linewidth=1,
        # )
        ax[i].set_xlabel("time step")
        ax[i].set_ylabel("reward")
        ax[i].set(title=cfg.name)
        ax[i].legend()

    sacred_save_fig(fig, _run, "rewards_over_time")

    space = models["unmodified"].observation_space
    if isinstance(space, gym.spaces.Discrete):
        rewards = {}
        for objective, model in models.items():
            positions = torch.arange(space.n, device=model.device)
            repeated_positions = positions.unsqueeze(1).repeat(1, space.n).flatten()
            next_positions = positions.unsqueeze(0).repeat(space.n, 1).flatten()

            states = torch.nn.functional.one_hot(repeated_positions, num_classes=space.n).float()
            next_states = torch.nn.functional.one_hot(next_positions, num_classes=space.n).float()
            dones = torch.zeros(space.n ** 2, dtype=torch.bool)

            with torch.no_grad():
                out = model(states, None, next_states, dones)
            rewards[objective] = prepare_rewards(out.view(space.n, space.n))
        ncols = min(3, len(rewards))
        fig = plot_gridworld_rewards(rewards, ncols=ncols, discount=gamma)
        sacred_save_fig(fig, _run, "gridworld_rewards")

    objective = "unmodified"
    model = models[objective]

    if isinstance(space, gym.spaces.Box) and space.shape != (2,):
        grid_size = 100

        venv = create_env()

        low = venv.normalize_obs(space.low)
        high = venv.normalize_obs(space.high)

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

        expert_cfg = rollouts[0]
        assert expert_cfg.name == "expert"
        trajectories = get_trajectories(
            expert_cfg, create_env, min_timesteps=rollout_steps, seed=_seed
        )
        transitions = flatten_trajectories_with_rew(trajectories)
        states = transitions.obs[transitions.acts == 2]
        ax.scatter(states[:, 0], states[:, 1])
        sacred_save_fig(fig, _run, f"reward_model_{objective}")

    elif isinstance(space, gym.spaces.Discrete):
        fig, ax = plt.subplots()

        # TODO: this is not very robust, only works for square
        # mazes
        size = int(math.sqrt(space.n))

        positions = torch.arange(space.n, device=model.device)
        states = torch.nn.functional.one_hot(positions, num_classes=space.n).float()

        out = (
            model(states, None, None, None)
            .detach()
            .cpu()
            .numpy()
        )

        im = ax.imshow(out.reshape(size, size))
        ax.set_axis_off()
        fig.colorbar(im, ax=ax)
        sacred_save_fig(fig, _run, f"reward_model_{objective}")
