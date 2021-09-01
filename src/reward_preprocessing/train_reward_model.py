from pathlib import Path
import tempfile
from typing import Any, Mapping

from sacred import Experiment
import torch
from tqdm import tqdm
import wandb

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.models import MlpRewardModel, SasRewardModel
from reward_preprocessing.utils import add_observers, use_rollouts

ex = Experiment("train_reward_model", ingredients=[env_ingredient])
add_observers(ex)
get_dataloader, _ = use_rollouts(ex)


@ex.config
def config():
    # 1 epoch is a good default because when we use dynamically
    # generated data, that's probably what we want
    epochs = 1
    # If empty, the trained model is only saved via Sacred observers
    # (you can still extract it manually later).
    # But if you already know you will need the trained model, then
    # set this to a filepath where you want the model to be stored,
    # without an extension (but including a filename).
    save_path = None  # path to save the model to (without extension)
    run_dir = "runs/reward_model"
    model_type = "ss"  # type of reward model, either 'ss' or 'sas'
    lr = 0.001  # learning rate
    lr_decay_rate = None  # factor to multiply by on each LR decay
    lr_decay_every = 100  # decay the learning rate every n batches
    wb = {}  # kwargs for wandb.init()
    eval_every = 5  # compute test loss every n episodes
    log_every = 20  # how many batches to aggregate before logging to wandb

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(
    epochs: int,
    save_path: str,
    model_type: str,
    lr: float,
    lr_decay_rate: float,
    lr_decay_every: int,
    wb: Mapping[str, Any],
    log_every: int,
    eval_every: int,
    _config,
):
    train_loader = get_dataloader(create_env)
    test_loader = get_dataloader(create_env, mode="test")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_type == "ss":
        model = MlpRewardModel(train_loader.dataset.state_shape).to(device)
    elif model_type == "sas":
        model = SasRewardModel(
            train_loader.dataset.state_shape, train_loader.dataset.action_shape
        ).to(device)
    else:
        raise ValueError(f"Unknown model type '{model_type}', expected 'ss' or 'sas'.")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_decay_rate is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_decay_rate
        )
    loss_fn = torch.nn.MSELoss()

    # this is what we'll return if no epochs are run
    # (mainly here to make pylance happy, shouldn't actually be needed)
    test_loss = None

    if wb:
        wandb.init(
            project="reward_preprocessing",
            job_type="reward_model",
            config=_config,
            **wb,
        )

    step = 0
    running_loss = 0.0
    for e in range(epochs):
        for inputs, rewards in tqdm(train_loader):
            optimizer.zero_grad()
            loss = loss_fn(model(inputs.to(device)), rewards.to(device))
            loss.backward()
            optimizer.step()
            step += 1
            running_loss += loss.item()
            if scheduler and step % lr_decay_every == 0:
                scheduler.step()
                if wb:
                    wandb.log(
                        {"lr": scheduler.get_last_lr(), "epoch": e + 1}, step=step
                    )
                else:
                    print(f"LR: {scheduler.get_last_lr()[0]:.2E}")
            if step % log_every == 0:
                if wb:
                    wandb.log(
                        {"loss/train": running_loss / log_every, "epoch": e + 1},
                        step=step,
                    )
                    running_loss = 0.0
                else:
                    print(f"Loss: {running_loss:.3E}")

        # eval every eval_every epochs, but also after the final epoch
        if (e + 1) % eval_every == 0 or e == epochs - 1:
            test_loss = torch.tensor(0.0, device=device)
            with torch.no_grad():
                i = 0
                for inputs, rewards in test_loader:
                    i += 1
                    test_loss += loss_fn(model(inputs.to(device)), rewards.to(device))
            print("Epoch {:3d} | Test Loss: {:.6f}".format(e, test_loss.item() / i))
            if wb:
                wandb.log(
                    {"loss/test": test_loss.item() / i, "epoch": e + 1}, step=step
                )

    # if save_path is set, we don't actually need the temporary directory
    # but just always creating it makes the code simpler
    with tempfile.TemporaryDirectory() as dirname:
        tmp_path = Path(dirname)
        # save the model
        if save_path:
            model_path = Path(save_path)
        else:
            model_path = tmp_path / "trained_model"

        model_path = model_path.with_suffix(".pt")
        torch.save(model.state_dict(), model_path)
        ex.add_artifact(model_path)

    return None if test_loss is None else test_loss.item()
