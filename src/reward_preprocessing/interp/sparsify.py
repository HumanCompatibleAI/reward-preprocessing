from typing import Any, Mapping

from sacred import Ingredient
import torch

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.models import RewardModel
from reward_preprocessing.preprocessing.potential_shaping import instantiate_potential
from reward_preprocessing.utils import get_env_name, sacred_save_fig, use_rollouts

sparsify_ingredient = Ingredient("sparsify", ingredients=[env_ingredient])
get_data_loaders, _ = use_rollouts(sparsify_ingredient)


@sparsify_ingredient.config
def config():
    enabled = True
    potential = None  # class name of the potential
    potential_options = {}  # kwargs for the potential (other than gamma)
    lr = 0.01  # learning rate
    log_every = 100  # log every n batches
    lr_decay_rate = None  # factor to multiply by on each LR decay
    lr_decay_every = 100  # decay the learning rate every n batches

    _ = locals()  # make flake8 happy
    del _


@sparsify_ingredient.capture
def sparsify(
    model: RewardModel,
    device,
    gamma: float,
    enabled: bool,
    batch_size: int,
    lr: float,
    lr_decay_rate: float,
    lr_decay_every: int,
    potential_options: Mapping[str, Any],
    log_every: int,
    potential: str,
    _run,
) -> RewardModel:
    if not enabled:
        return model

    env = create_env()

    env_name = get_env_name(env)
    model = instantiate_potential(
        env_name, potential, model=model, gamma=gamma, **potential_options
    ).to(device)

    train_loader, _ = get_data_loaders(create_env, num_workers=0, test_steps=0)

    # the weights of the original model are automatically frozen,
    # we only train the final potential shaping
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_decay_rate is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_decay_rate
        )

    def loss_fn(x):
        return x.abs().mean()

    running_loss = 0.0
    num_episodes = 0.0
    for i, (inputs, rewards) in enumerate(train_loader):
        optimizer.zero_grad()
        num_episodes += torch.sum(inputs.done)
        loss = loss_fn(model(inputs.to(device)))
        if i == 0:
            print("Initial loss: ", loss.item())
            print(model(inputs.to(device)))
        loss.backward()
        optimizer.step()
        if scheduler and i % lr_decay_every == lr_decay_every - 1:
            scheduler.step()
            print(f"LR: {scheduler.get_last_lr()[0]:.2E}")
        running_loss += loss.item()
        if i % log_every == log_every - 1:
            print(f"Loss: {running_loss / log_every:2E}")
            running_loss = 0.0
            print(f"Avg. episode length: {i * batch_size / num_episodes.item():.1f}")

    try:
        fig = model.plot(env)
        fig.suptitle("Learned potential")
        sacred_save_fig(fig, _run, "potential")
    except NotImplementedError:
        print("Potential can't be plotted, skipping")

    env.close()
    return model
