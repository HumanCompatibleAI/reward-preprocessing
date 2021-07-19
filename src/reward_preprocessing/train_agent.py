from pathlib import Path
import tempfile

from sacred import Experiment
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.utils import ContinuousVideoRecorder, add_observers

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

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(steps: int, save_path: str, num_frames: int, eval_episodes: int):
    env = create_env()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)

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
