from pathlib import Path
import tempfile

from sacred import Experiment
from stable_baselines3 import PPO
import torch
from tqdm import tqdm

from reward_preprocessing.datasets import get_data_loaders, to_torch
from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.models import MlpRewardModel
from reward_preprocessing.utils import add_observers

ex = Experiment("train_reward_model", ingredients=[env_ingredient])
add_observers(ex)


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
    save_path = None
    run_dir = "runs/reward_model"
    data_path = None
    agent_path = None
    num_workers = 4
    steps = 10000
    test_steps = 5000
    batch_size = 32

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(
    epochs: int,
    num_workers: int,
    batch_size: int,
    data_path: str,
    save_path: str,
    agent_path: str,
    steps: int,
    test_steps: int,
):

    transform = to_torch
    if data_path:
        # if a path to a dataset is given, we don't need an environment
        env = None
    else:
        env = create_env()
    if agent_path:
        agent = PPO.load(agent_path)
    else:
        agent = None

    train_loader, test_loader = get_data_loaders(
        batch_size, num_workers, data_path, env, agent, steps, test_steps, transform
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = MlpRewardModel(train_loader.dataset.state_shape).to(device)
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
            i = 0
            for inputs, rewards in test_loader:
                i += 1
                test_loss += loss_fn(model(inputs.to(device)), rewards.to(device))
        print("Epoch {:3d} | Test Loss: {:.6f}".format(e, float(test_loss) / i))

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
