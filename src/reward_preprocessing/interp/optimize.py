from typing import Any, Mapping, Sequence

from imitation.rewards.reward_nets import RewardNet
from sacred import Ingredient
import torch
from tqdm import tqdm
import wandb

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.preprocessing.potential_shaping import instantiate_potential
from reward_preprocessing.preprocessing.preprocessor import ScaleShift
from reward_preprocessing.utils import get_env_name, sacred_save_fig, use_rollouts

optimize_ingredient = Ingredient("optimize", ingredients=[env_ingredient])
get_dataloader, _ = use_rollouts(optimize_ingredient)


@optimize_ingredient.config
def config():
    enabled = False
    epochs = 1  # number of epochs to train for
    potential = None  # class name of the potential
    potential_options = {}  # kwargs for the potential (other than gamma)
    lr = 0.01  # learning rate
    log_every = 100  # log every n batches
    lr_decay_rate = None  # factor to multiply by on each LR decay
    lr_decay_every = 100  # decay the learning rate every n batches
    objectives = ["l1"]  # names of the objectives to optimize
    batch_size = 256  # batch size for the dataloader

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


@optimize_ingredient.capture
def optimize(
    model: RewardNet,
    device,
    gamma: float,
    use_wandb: bool,
    enabled: bool,
    lr: float,
    lr_decay_rate: float,
    lr_decay_every: int,
    potential_options: Mapping[str, Any],
    objectives: Sequence[str],
    log_every: int,
    potential: str,
    epochs: int,
    batch_size: int,
    _run,
) -> Mapping[str, RewardNet]:
    models = {"unmodified": model}

    if not enabled:
        return models

    env = create_env()

    env_name = get_env_name(env)
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
        step = 0
        for e in range(epochs):
            for i, inputs in tqdm(enumerate(train_loader)):
                step += 1
                optimizer.zero_grad()
                loss = loss_fn(
                    wrapped_model(
                        *wrapped_model.preprocess(
                            inputs.obs, inputs.acts, inputs.next_obs, inputs.dones
                        )
                    )
                )
                # loss = (
                #     (
                #         wrapped_model(*wrapped_model.preprocess(inputs))
                #         - torch.as_tensor(
                #             inputs.rews,
                #             dtype=torch.float32,
                #             device=wrapped_model.device,
                #         )
                #     )
                #     .abs()
                #     .mean()
                # )
                loss.backward()
                optimizer.step()
                if scheduler and i % lr_decay_every == lr_decay_every - 1:
                    scheduler.step()
                    if use_wandb:
                        wandb.log(
                            {"lr": scheduler.get_last_lr(), "epoch": e + 1}, step=step
                        )
                    else:
                        print(f"LR: {scheduler.get_last_lr()[0]:.2E}")
                running_loss += loss.item()
                if i % log_every == log_every - 1:
                    if use_wandb:
                        wandb.log(
                            {
                                "loss/train": running_loss / log_every,
                                "epoch": e + 1,
                            },
                            step=step,
                        )
                    else:
                        print(f"Loss: {running_loss / log_every:.3E}")
                    running_loss = 0.0
        models[objective] = wrapped_model.eval()

        try:
            fig = wrapped_model.plot()
            fig.suptitle(f"Learned potential with {objective} objective")
            sacred_save_fig(fig, _run, f"optimize_potential_{objective}")
        except NotImplementedError:
            print("Potential can't be plotted, skipping")

    return models
