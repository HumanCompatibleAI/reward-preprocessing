from pathlib import Path

import numpy as np
from sacred import Experiment
from stable_baselines3 import PPO

from reward_preprocessing.env import create_env, env_ingredient

ex = Experiment("create_rollouts", ingredients=[env_ingredient])


@ex.config
def config():
    model_path = ""
    save_path = ""
    num_samples = 10000
    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(model_path: str, save_path: str, num_samples: int):
    env = create_env()
    model = PPO.load(model_path)

    obs = env.reset()
    states = []
    actions = []
    next_states = []
    rewards = []
    for _ in range(num_samples):
        states.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        next_states.append(obs)
        rewards.append(reward)
        if done:
            obs = env.reset()

    env.close()

    path = Path(save_path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(path),
        states=np.stack(states, axis=0),
        actions=np.array(actions),
        next_states=np.stack(next_states, axis=0),
        rewards=np.array(rewards),
    )
