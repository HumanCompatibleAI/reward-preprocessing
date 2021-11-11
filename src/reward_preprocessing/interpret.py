from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from sacred import Experiment
import torch
import wandb

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.interp import (
    add_fixed_potential,
    add_noise_potential,
    fixed_ingredient,
    heatmap_ingredient,
    noise_ingredient,
    optimize,
    optimize_ingredient,
    plot_heatmaps,
    plot_rewards,
    reward_ingredient,
    value_net_ingredient,
    value_net_potential,
)
from reward_preprocessing.utils import add_observers

ex = Experiment(
    "interpret",
    ingredients=[
        env_ingredient,
        optimize_ingredient,
        noise_ingredient,
        fixed_ingredient,
        reward_ingredient,
        heatmap_ingredient,
        value_net_ingredient,
    ],
)
add_observers(ex)


@ex.config
def config():
    run_dir = "runs/interpret"
    model_paths = []  # paths to the models to be interpreted (with extension)
    model_names = None  # names for the models (to display in plots)
    gamma = 0.99  # discount rate (used for all potential shapings)
    wb = {}  # kwargs for wandb.init()

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(
    model_paths: Sequence[str],
    model_names: Optional[Sequence[str]],
    gamma: float,
    wb: Mapping[str, Any],
    _config,
):
    env = create_env()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_wandb = False
    if wb:
        wandb.init(
            project="reward_preprocessing",
            job_type="interpret",
            config=_config,
            **wb,
        )
        use_wandb = True

    models = {}
    if model_names is None:
        model_names = [Path(path).stem for path in model_paths]

    for path, name in zip(model_paths, model_names):
        models[name] = torch.load(path, map_location=device)

    # first key is the base model name, second key is the objective
    preprocessed_models: Dict[str, Dict[str, Any]] = {}
    for name, model in models.items():
        preprocessed_models[name] = optimize_tabular(model, device=device, gamma=gamma)

    plot_heatmaps(preprocessed_models, gamma=gamma)

    # model = add_fixed_potential(model, gamma)
    # model = add_noise_potential(model, gamma)
    # shaping_model = value_net_potential(model, gamma=gamma)
    # models = optimize(model, device=device, gamma=gamma, use_wandb=use_wandb)
    # if shaping_model is not None:
    #     models["Value net shaping"] = shaping_model
    # plot_rewards(models, gamma=gamma)
    env.close()
