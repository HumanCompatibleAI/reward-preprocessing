from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

from sacred import Experiment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from torch import nn

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.utils import (
    ContinuousVideoRecorder,
    add_observers,
    get_env_name,
)
from reward_preprocessing.wandb_logger import WandbOutputFormat

ex = Experiment("train_agent", ingredients=[env_ingredient])
add_observers(ex)


@ex.config
def config():
    steps = 100000
    # If empty, the trained agent is only saved via Sacred observers
    # (you can still extract it manually later).
    # But if you already know you will need the trained model, then
    # set this to a filepath where you want the model to be stored,
    # without an extension (but including a filename).
    save_path = ""
    num_frames = 100
    run_dir = "runs/agent"
    eval_episodes = 0
    ppo_options = {}
    wb = {}
    eval_every = 10000  # eval every n steps. Set to 0 to disable.

    _ = locals()  # make flake8 happy
    del _


DEFAULT_PPO_OPTIONS = {
    "MountainCar-v0": {"n_steps": 256, "gae_lambda": 0.98, "n_epochs": 4},
    # Converted from
    # https://github.com/DLR-RM/rl-trained-agents/blob/28bc94a4a39a7845bb9541796af2fb077a47673d/ppo/HalfCheetahBulletEnv-v0_1/HalfCheetahBulletEnv-v0/config.yml
    "HalfCheetah-v2": {
        "batch_size": 128,
        "clip_range": 0.4,
        "gamma": 0.99,
        "learning_rate": 3e-5,
        "n_steps": 512,
        "gae_lambda": 0.9,
        "n_epochs": 20,
        "use_sde": True,
        "sde_sample_freq": 4,
        "policy_kwargs": {
            "log_std_init": -2,
            "ortho_init": False,
            "activation_fn": nn.ReLU,
            "net_arch": [{"pi": [256, 256], "vf": [256, 256]}],
        },
    },
}


@ex.automain
def main(
    steps: int,
    save_path: str,
    num_frames: int,
    eval_episodes: int,
    ppo_options: Mapping[str, Any],
    wb: Mapping[str, Any],
    eval_every: int,
    _config,
):
    env = create_env()
    env_name = get_env_name(env)
    if env_name in DEFAULT_PPO_OPTIONS:
        # merge the default and custom options
        # custom options come second so they take precedence
        ppo_options = {**DEFAULT_PPO_OPTIONS[env_name], **ppo_options}

    model = PPO("MlpPolicy", env, verbose=1, **ppo_options)

    eval_callback = None
    if eval_every > 0:
        eval_callback = EvalCallback(
            create_env(),
            eval_freq=eval_every,
            deterministic=True,
            render=False,
        )

    # If any weights & biases options are set, we use that for logging.
    if wb:
        writer = WandbOutputFormat(wb, _config)
        logger = Logger(
            folder=None, output_formats=[writer, HumanOutputFormat(sys.stdout)]
        )
        model.set_logger(logger)
    model.learn(total_timesteps=steps, callback=eval_callback)

    if eval_episodes:
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=eval_episodes
        )
    else:
        mean_reward, std_reward = None, None

    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname)
        # save the model
        if save_path == "":
            model_path = path / "trained_agent"
        else:
            model_path = Path(save_path)
        model.save(model_path)
        ex.add_artifact(model_path.with_suffix(".zip"))

        if isinstance(env, VecNormalize):
            # if we used normalization, then store the final statistics
            # so we can reuse them later
            stats_path = path / "vec_normalize.pkl"
            env.save(str(stats_path))
            ex.add_artifact(stats_path)

        # record a video of the trained agent
        env = ContinuousVideoRecorder(
            env,
            str(path),
            record_video_trigger=lambda x: x == 0,
            video_length=num_frames,
            name_prefix="trained_agent",
        )
        obs = env.reset(start_video=True)
        # The sequence of observations can be used as a check in the
        # determinism test (see tests/test_determinism.py).
        # The rewards are not enough for this, because for untrained
        # agents they are often just all 0.
        observations = []
        for _ in range(num_frames + 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            observations.append(obs)

        env.close()
        video_path = Path(env.video_recorder.path)
        ex.add_artifact(video_path)

    return mean_reward, std_reward, observations
