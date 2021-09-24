from typing import Any, Mapping, Sequence

from imitation.data.types import Transitions
from imitation.rewards.reward_nets import RewardNet
import numpy as np
from sacred import Experiment
import torch

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.interp.gridworld_plot import ACTION_DELTA, Actions
from reward_preprocessing.preprocessing.potential_shaping import instantiate_potential
from reward_preprocessing.preprocessing.preprocessor import ScaleShift
from reward_preprocessing.utils import get_env_name, sacred_save_fig, use_rollouts

optimize_tabular_ex = Experiment("optimize_tabular", ingredients=[env_ingredient])


@optimize_tabular_ex.config
def config():
    gamma = 0.99  # discount rate
    steps = 10000  # number of steps to train for
    potential = None  # class name of the potential
    potential_options = {}  # kwargs for the potential (other than gamma)
    lr = 0.01  # learning rate
    log_every = 100  # log every n batches
    lr_decay_rate = None  # factor to multiply by on each LR decay
    lr_decay_every = 100  # decay the learning rate every n batches
    objectives = ["l1", "smooth"]  # names of the objectives to optimize

    _ = locals()  # make flake8 happy
    del _


def _local_mean_dist(x):
    dists = x[None] - x[:, None]
    means = (torch.exp(-dists.abs()) * x[None]).sum(1)
    return ((x - means) ** 2).sum()


def log_abs(x):
    return (1 + x.abs()).log()


OBJECTIVES = {
    "l1": lambda x: x.abs().mean(),
    "l_half": lambda x: x.abs().sqrt().mean(),
    "local_mean": _local_mean_dist,
    "log": lambda x: (1 + x.abs()).log().mean(),
    "smooth": lambda x: log_abs(x[1:] - x[:-1]).mean(),
}


@optimize_tabular_ex.capture
def optimize_tabular(
    model: RewardNet,
    device,
    gamma: float,
    enabled: bool,
    steps: int,
    lr: float,
    lr_decay_rate: float,
    lr_decay_every: int,
    potential_options: Mapping[str, Any],
    objectives: Sequence[str],
    log_every: int,
    potential: str,
    _run,
) -> Mapping[str, RewardNet]:
    models = {"unmodified": model}

    if not enabled:
        return models

    venv = create_env()

    env_name = get_env_name(venv)

    # TODO: this is very brittle
    env = venv.envs[0].unwrapped
    env.reset()

    states = []
    next_states = []
    actions = []
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
            states.append(state)
            next_states.append(next_state)
            actions.append(action)

    transitions = Transitions(
        obs=np.array(states),
        acts=np.array(actions),
        next_obs=np.array(next_states),
        dones=np.zeros(len(states), dtype=bool),
        infos=np.array([{}] * len(states)),
    )

    env.close()

    for objective in objectives:
        print(f"Optimizing {objective} objective")
        loss_fn = OBJECTIVES[objective]
        wrapped_model = ScaleShift(model, scale=False)
        wrapped_model = instantiate_potential(
            env_name,
            potential,
            model=wrapped_model,
            gamma=gamma,
            freeze_model=False,
            **potential_options,
        ).to(device)

        # the weights of the original model are automatically frozen,
        # we only train the final potential shaping
        optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=lr)
        scheduler = None
        if lr_decay_rate is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=lr_decay_rate
            )

        running_loss = 0.0

        for i in range(steps):
            optimizer.zero_grad()
            # use the original model for preprocessing, the wrappers
            # don't know about the tabular setting
            loss = loss_fn(
                wrapped_model(
                    *model.preprocess(
                        transitions.obs,
                        transitions.acts,
                        transitions.next_obs,
                        transitions.dones,
                    )
                )
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % log_every == 0:
                print(f"Step {i}, loss: ", running_loss)
                running_loss = 0.0

        models[objective] = wrapped_model.eval()

        try:
            fig = wrapped_model.plot()
            fig.suptitle(f"Learned potential with {objective} objective")
            sacred_save_fig(fig, _run, f"optimize_potential_{objective}")
        except NotImplementedError:
            print("Potential can't be plotted, skipping")

    return models


@optimize_tabular_ex.automain
def main(
    model_paths: Sequence[str],
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for path in model_paths:
        print(f"Loading model from {path}")
        model = torch.load(path, map_location=device)
        preprocessed_models = optimize_tabular(model, device=device, gamma=gamma)
        for objective, preprocessed_model in preprocessed_models.items():
            torch.save(preprocessed_model, path + f".{objective}")
