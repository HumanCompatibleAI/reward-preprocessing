from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
from sacred import Ingredient

from reward_preprocessing.models import RewardModel
from reward_preprocessing.preprocessing.potential_shaping import RandomPotentialShaping

noise_ingredient = Ingredient("noise")


@noise_ingredient.config
def config():
    enabled = True
    std = 1.0
    mean = 0.0

    _ = locals()  # make flake8 happy
    del _


@noise_ingredient.capture
def add_noise_potential(
    model: RewardModel, gamma: float, enabled: bool, std: float, mean: float, _run
) -> RewardModel:
    if not enabled:
        return model

    model = RandomPotentialShaping(model, gamma=gamma, mean=mean, std=std)

    fig, ax = plt.subplots()

    im = ax.imshow(
        model.potential_data.detach().cpu().numpy().reshape(*model.state_shape)
    )
    ax.set_axis_off()
    ax.set(title="Noise potential")
    fig.colorbar(im, ax=ax)

    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname) / "noise_potential.pdf"
        fig.savefig(path)
        _run.add_artifact(path)

    return model
