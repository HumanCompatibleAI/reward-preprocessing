from typing import Any, Mapping, Optional, Sequence

from gym.spaces import Discrete
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import RewardNet
import numpy as np
from sacred import Experiment
from stable_baselines3.common.vec_env import VecEnv
import torch

from reward_preprocessing.data import RolloutConfig
from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.preprocessing.potential_shaping import instantiate_potential
from reward_preprocessing.preprocessing.preprocessor import ScaleShift
from reward_preprocessing.utils import get_env_name, sacred_save_fig, use_rollouts

preprocess_ex = Experiment("preprocess", ingredients=[env_ingredient])
get_dataloader, _ = use_rollouts(preprocess_ex)


@preprocess_ex.config
def config():
    gamma = 0.99  # discount rate
    epochs = 3  # number of epochs to train for
    potential = None  # class name of the potential
    potential_options = {}  # kwargs for the potential (other than gamma)
    lr = 0.001  # learning rate
    batch_size = 200  # batch size for the dataloader
    objectives = [
        "sparse_l1",
        "smooth_l1",
        "sparse_log",
        "smooth_log",
    ]  # names of the objectives to optimize
    model_path = None
    save_path = None

    _ = locals()  # make flake8 happy
    del _


@preprocess_ex.named_config
def tabular():
    # This config isn't necessary to include, but it sets more reasonable defaults
    epochs = 10000  # an epoch in small gridworlds is very small
    locals()  # make flake8 happy


@preprocess_ex.named_config
def fast():
    epochs = 1
    _ = locals()  # make flake8 happy
    del _


@preprocess_ex.named_config
def linear():
    potential = "LinearPotentialShaping"  # class name of the potential
    locals()  # make flake8 happy


@preprocess_ex.named_config
def mlp():
    potential = "MlpPotentialShaping"  # class name of the potential
    potential_options = {
        "num_hidden": 4,
        "hidden_size": 128,
    }
    locals()  # make flake8 happy


def log_abs(x):
    return (1 + x.abs()).log()


def smoothness(rewards: torch.Tensor, idx1, idx2):
    # then for all such pairs, we want their rewards to be similar
    return rewards[idx1] - rewards[idx2]


OBJECTIVES = {
    "sparse_l1": lambda x, **kwargs: x.abs().mean(),
    "smooth_l1": lambda x, **kwargs: smoothness(x, **kwargs).abs().mean(),
    "sparse_log": lambda x, **kwargs: log_abs(x).mean(),
    "smooth_log": lambda x, **kwargs: log_abs(smoothness(x, **kwargs)).mean(),
}


def get_tabular_data(venv: VecEnv, device: torch.device):
    # TODO: this is very brittle
    env = venv.envs[0].unwrapped
    env.reset()

    states = []
    next_states = []
    actions = []
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            next_state, _ = env._step(state, action)
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

    # We return everything as a single batch
    return [transitions]


@preprocess_ex.capture
def preprocess(
    rollouts: Optional[Sequence[RolloutConfig]],
    model: RewardNet,
    device,
    gamma: float,
    epochs: int,
    lr: float,
    batch_size: int,
    potential_options: Mapping[str, Any],
    objectives: Sequence[str],
    potential: str,
    _run,
) -> Mapping[str, RewardNet]:

    venv = create_env()

    env_name = get_env_name(venv)

    if rollouts is not None:
        dataloader = get_dataloader(create_env, batch_size=batch_size)
    elif isinstance(venv.observation_space, Discrete):
        dataloader = get_tabular_data(venv, device)
    else:
        raise ValueError("Rollouts must be specified for non-tabular environments")

    models = {"unmodified": model}

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

        for e in range(epochs):
            epoch_loss = 0.0
            for transitions in dataloader:
                optimizer.zero_grad()

                if isinstance(venv.observation_space, Discrete):
                    # find all pairs of transitions that are next to each other
                    neighbors = (
                        transitions.obs[:, None] == transitions.next_obs[None, :]
                    )
                    neighbors = neighbors | (
                        transitions.obs[:, None] == transitions.obs[None, :]
                    )

                    idx1, idx2 = neighbors.nonzero()
                else:
                    idx1, idx2 = np.arange(len(transitions) - 1), np.arange(
                        1, len(transitions)
                    )
                idx1 = torch.as_tensor(idx1, device=device, dtype=torch.long)
                idx2 = torch.as_tensor(idx2, device=device, dtype=torch.long)

                loss = loss_fn(
                    wrapped_model(
                        *model.preprocess(
                            transitions.obs,
                            transitions.acts,
                            transitions.next_obs,
                            transitions.dones,
                        ),
                    ),
                    idx1=idx1,
                    idx2=idx2,
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {e} loss: ", epoch_loss)

        models[objective] = wrapped_model.eval()

        try:
            fig = wrapped_model.plot()
            fig.suptitle(f"Learned potential with {objective} objective")
            sacred_save_fig(fig, _run, f"preprocess_potential_{objective}")
        except NotImplementedError:
            print("Potential can't be plotted, skipping")

    return models


@preprocess_ex.automain
def main(
    model_path: str,
    save_path: str,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    preprocessed_models = preprocess(model=model, device=device)
    for objective, preprocessed_model in preprocessed_models.items():
        path = f"{save_path}.{objective}.pt"
        torch.save(preprocessed_model, path)
