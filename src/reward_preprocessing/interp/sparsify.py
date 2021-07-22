import warnings

from sacred import Ingredient
import torch

from reward_preprocessing.datasets import get_data_loaders, to_torch
from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.models import RewardModel
from reward_preprocessing.preprocessing.potential_shaping import instantiate_potential
from reward_preprocessing.utils import sacred_save_fig

sparsify_ingredient = Ingredient("sparsify", ingredients=[env_ingredient])


@sparsify_ingredient.config
def config():
    enabled = True
    steps = 100000
    batch_size = 32
    rollouts = "random"
    potential = None
    lr = 0.01
    log_every = 100

    _ = locals()  # make flake8 happy
    del _


@sparsify_ingredient.capture
def sparsify(
    model: RewardModel,
    gamma: float,
    enabled: bool,
    steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    rollouts: str,
    potential: str,
    _run,
    agent=None,
) -> RewardModel:
    if not enabled:
        return model

    env = create_env()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if rollouts == "random":
        agent = None
    elif rollouts == "expert":
        if agent is None:
            raise ValueError(
                "sparsify didn't receive an agent, expert rollouts can't be used"
            )
        if agent.gamma != gamma:
            # We want to allow setting a different gamma value
            # because that can be useful for quick experimentation.
            # But the user should be aware of that.
            warnings.warn(
                "Agent was trained with different gamma value "
                "than the one used for potential shaping."
            )
    else:
        raise ValueError(
            f"Invalid value {rollouts} for sparsify.rollouts. "
            "Valid options are 'random' and 'expert'."
        )

    env_name = env.envs[0].spec.id
    model = instantiate_potential(env_name, potential, model=model, gamma=gamma)

    train_loader, _ = get_data_loaders(
        batch_size=batch_size,
        num_workers=0,
        venv=env,
        policy=agent,
        num_train=steps,
        num_test=0,
        transform=to_torch,
    )

    # the weights of the original model are automatically frozen,
    # we only train the final potential shaping
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def loss_fn(x):
        return x.abs().mean()

    running_loss = 0.0
    num_episodes = 0.0
    for i, (inputs, rewards) in enumerate(train_loader):
        optimizer.zero_grad()
        num_episodes += torch.sum(inputs.done)
        loss = loss_fn(model(inputs.to(device)))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % log_every == log_every - 1:
            print("Loss: ", running_loss / log_every)
            running_loss = 0.0
            print("Avg. episode length: ", i * batch_size / num_episodes.item())

    try:
        fig = model.plot(env)
        fig.suptitle("Learned potential")
        sacred_save_fig(fig, _run, "potential")
    except NotImplementedError:
        print("Potential can't be plotted, skipping")

    env.close()
    return model
