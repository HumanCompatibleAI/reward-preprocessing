from pathlib import Path
import tempfile

from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch
from tqdm import tqdm

from reward_preprocessing.datasets import (
    RewardData,
    collate_fn,
    get_worker_init_fn,
    to_torch,
)
from reward_preprocessing.models import MlpRewardModel

ex = Experiment("train_reward_model")


@ex.config
def config():
    epochs = 10
    # If empty, the trained model is only saved via Sacred observers
    # (you can still extract it manually later).
    # But if you already know you will need the trained model, then
    # set this to a filepath where you want the model to be stored,
    # without an extension (but including a filename).
    save_path = None
    run_dir = "runs/reward_model"
    data_path = None
    num_workers = 4
    batch_size = 32

    # Just to be save, we check whether an observer already exists,
    # to avoid adding multiple copies of the same observer
    # (see https://github.com/IDSIA/sacred/issues/300)
    if len(ex.observers) == 0:
        ex.observers.append(FileStorageObserver(run_dir))

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(
    epochs: int, num_workers: int, batch_size: int, data_path: str, save_path: str
):
    path = Path(data_path)

    transform = to_torch

    train_data = RewardData(path, transform=transform, train=True)
    test_data = RewardData(path, transform=transform, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # workaround for https://github.com/numpy/numpy/issues/18124
        # see docstring of get_worker_init_fn for details
        worker_init_fn=get_worker_init_fn(path),
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # workaround for https://github.com/numpy/numpy/issues/18124
        # see docstring of get_worker_init_fn for details
        worker_init_fn=get_worker_init_fn(path),
        collate_fn=collate_fn,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = MlpRewardModel(train_data.state_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    # this is what we'll return if no epochs are run
    # (mainly here to make pylance happy, shouldn't actually be needed)
    test_loss = None

    for e in range(epochs):
        for inputs, rewards in tqdm(train_loader):
            optimizer.zero_grad()
            loss = loss_fn(model(inputs.to(device)), rewards.to(device))
            loss.backward()
            optimizer.step()

        test_loss = torch.tensor(0.0)
        with torch.no_grad():
            for inputs, rewards in test_loader:
                test_loss += loss_fn(model(inputs.to(device)), rewards.to(device))
        print(
            "Epoch {:3d} | Test Loss: {:.6f}".format(
                e, float(test_loss) / len(test_data)
            )
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
