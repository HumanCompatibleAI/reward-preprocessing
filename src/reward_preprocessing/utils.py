from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


class ContinuousVideoRecorder(VecVideoRecorder):
    def reset(self, start_video=False) -> VecEnvObs:
        obs = self.venv.reset()
        if start_video:
            self.start_video_recorder()
        return obs
