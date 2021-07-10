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
    train_samples = 10000
    test_samples = 10000
    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(model_path: str, save_path: str, train_samples: int, test_samples: int):
    env = create_env()
    model = PPO.load(model_path)

    states = {}
    actions = {}
    next_states = {}
    rewards = {}

    for mode, num_samples in zip(["train", "test"], [train_samples, test_samples]):
        obs = env.reset()
        states[mode] = []
        actions[mode] = []
        next_states[mode] = []
        rewards[mode] = []
        for _ in range(num_samples):
            states[mode].append(obs)
            action, _ = model.predict(obs, deterministic=True)
            actions[mode].append(action)
            obs, reward, done, info = env.step(action)
            # the environment automatically resets (because it's vectorized),
            # so if the episode was finished, then obs is already the observation
            # for the start of the next episode. So in this case,
            # we make sure to use the actual final state as the next_state.
            if done[0]:
                next_state = info[0]["terminal_observation"][None]
            else:
                next_state = obs
            next_states[mode].append(next_state)
            rewards[mode].append(reward)

    env.close()

    path = Path(save_path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(path),
        train_states=np.stack(states["train"], axis=0),
        train_actions=np.array(actions["train"]),
        train_next_states=np.stack(next_states["train"], axis=0),
        train_rewards=np.array(rewards["train"]),
        test_states=np.stack(states["test"], axis=0),
        test_actions=np.array(actions["test"]),
        test_next_states=np.stack(next_states["test"], axis=0),
        test_rewards=np.array(rewards["test"]),
    )
