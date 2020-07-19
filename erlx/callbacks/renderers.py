from erlx.callbacks.base import BaseCallback

from time import sleep


class GymRenderer(BaseCallback):
    def __init__(self, env, frames_per_second):
        self._env = env
        self._seconds_per_frame = 1/float(frames_per_second)

    def on_step_end(self, action, observation, reward, done):
        self._env.gym_env.render()
        sleep(self._seconds_per_frame)

    def on_episode_end(self):
        self._env.gym_env.close()
