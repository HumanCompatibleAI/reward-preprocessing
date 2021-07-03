import gym
import gym_minigrid  # noqa: F401
from gym_minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO


def test_ppo_on_minigrid():
    env = gym.make("MiniGrid-Empty-Random-6x6-v0")
    env = FlatObsWrapper(env)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10)

    obs = env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    env.close()
