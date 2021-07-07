from pathlib import Path
import tempfile

from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3 import PPO

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.utils import ContinuousVideoRecorder

ex = Experiment("train_agent", ingredients=[env_ingredient])


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
    run_dir = "runs"

    # Just to be save, we check whether an observer already exists,
    # to avoid adding multiple copies of the same observer
    # (see https://github.com/IDSIA/sacred/issues/300)
    if len(ex.observers) == 0:
        ex.observers.append(FileStorageObserver(run_dir))

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(steps: int, save_path: str, num_frames: int):
    env = create_env()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)

    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname)
        # save the model
        if save_path == "":
            model_path = path / "trained_agent"
        else:
            model_path = Path(save_path)
        model.save(model_path)
        ex.add_artifact(model_path.with_suffix(".zip"))

        # record a video of the trained agent
        env = ContinuousVideoRecorder(
            env,
            str(path),
            record_video_trigger=lambda x: x == 0,
            video_length=num_frames,
            name_prefix="trained_agent",
        )
        obs = env.reset(start_video=True)
        for _ in range(num_frames + 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()

        env.close()
        video_path = Path(env.video_recorder.path)
        ex.add_artifact(video_path)
