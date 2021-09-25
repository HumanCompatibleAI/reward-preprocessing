from typing import Any, Mapping, Sequence

from imitation.rewards.reward_nets import RewardNet
from sacred import Experiment
import torch
from tqdm import tqdm

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.preprocessing.potential_shaping import instantiate_potential
from reward_preprocessing.preprocessing.preprocessor import ScaleShift
from reward_preprocessing.utils import get_env_name, sacred_save_fig, use_rollouts

optimize_continuous_ex = Experiment("optimize_continuous", ingredients=[env_ingredient])
get_dataloader, _ = use_rollouts(optimize_continuous_ex)

torch.set_num_threads(8)


@optimize_continuous_ex.config
def config():
    gamma = 0.99  # discount rate
    epochs = 3  # number of epochs to train for
    potential = None  # class name of the potential
    steps = 100000
    potential_options = {
        "num_hidden": 4,
        "hidden_size": 128,
    }  # kwargs for the potential (other than gamma)
    lr = 0.005  # learning rate
    log_every = 10  # log every n batches
    lr_decay_rate = None  # factor to multiply by on each LR decay
    lr_decay_every = 100  # decay the learning rate every n batches
    objectives = ["sparse_l1", "smooth_l1", "sparse_log", "smooth_log"]  # names of the objectives to optimize
    batch_size = 256  # batch size for the dataloader
    model_path = None
    save_path = None

    _ = locals()  # make flake8 happy
    del _


@optimize_continuous_ex.named_config
def fast():
    steps = 1
    batch_size = 2
    epochs = 1
    _ = locals()  # make flake8 happy
    del _


def _local_mean_dist(x, **kwargs):
    dists = x[None] - x[:, None]
    means = (torch.exp(-dists.abs()) * x[None]).sum(1)
    return ((x - means) ** 2).sum()


def log_abs(x):
    return (1 + x.abs()).log()


OBJECTIVES = {
    "sparse_l1": lambda x: x.abs().mean(),
    "smooth_l1": lambda x: (x[1:] - x[:-1]).abs().mean(),
    "l_half": lambda x: x.abs().sqrt().mean(),
    "local_mean": _local_mean_dist,
    "sparse_log": lambda x: (1 + x.abs()).log().mean(),
    "smooth_log": lambda x: log_abs(x[1:] - x[:-1]).mean(),
}


@optimize_continuous_ex.capture
def optimize_continuous(
    model: RewardNet,
    device,
    gamma: float,
    lr: float,
    lr_decay_rate: float,
    lr_decay_every: int,
    potential_options: Mapping[str, Any],
    objectives: Sequence[str],
    log_every: int,
    potential: str,
    epochs: int,
    batch_size: int,
    save_path: str,
) -> Mapping[str, RewardNet]:
    models = {"unmodified": model}

    venv = create_env()

    env_name = get_env_name(venv)

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

        train_loader = get_dataloader(create_env, batch_size=batch_size)

        # the weights of the original model are automatically frozen,
        # we only train the final potential shaping
        optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=lr)
        scheduler = None
        if lr_decay_rate is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=lr_decay_rate
            )

        running_loss = 0.0

        for e in range(epochs):
            for i, transitions in enumerate(train_loader):
                optimizer.zero_grad()
                loss = loss_fn(
                    wrapped_model(
                        *wrapped_model.preprocess(
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
                    print(f"Epoch {e}, step {i}, loss: ", running_loss)
                    running_loss = 0.0

        models[objective] = wrapped_model.eval()

        try:
            fig = wrapped_model.plot()
            fig.suptitle(f"Learned potential with {objective} objective")
            fig.savefig(f"{save_path}.{objective}.pdf")
        except NotImplementedError:
            print("Potential can't be plotted, skipping")

    return models


@optimize_continuous_ex.automain
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
    preprocessed_models = optimize_continuous(model, device=device)
    for objective, preprocessed_model in preprocessed_models.items():
        path = f"{save_path}.{objective}.pt"
        torch.save(preprocessed_model, path)
