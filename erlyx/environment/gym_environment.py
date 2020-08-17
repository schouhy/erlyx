from erlyx.environment.base import BaseEnvironment, Episode
from erlyx import types
import gym

from time import sleep
import cv2


class GymEnvironment(BaseEnvironment, Episode):
    def __init__(self, env_name):
        self.gym_env = gym.make(env_name)

    def new_episode(self) -> (Episode, types.ObservationType):
        observation = self.gym_env.reset()
        return self, observation

    def step(self, action: types.ActionType) -> types.EpisodeStatus:
        observation, reward, done, _ = self.gym_env.step(action)
        return types.EpisodeStatus(observation, reward, done)


class GymAtariBWEnvironment(GymEnvironment):
    def __init__(self, env_name, repeat=4, simplified_reward=True, render=False, fps=30, img_hw=(105, 80)):
        super(GymAtariBWEnvironment, self).__init__(env_name)
        self._repeat = repeat
        self._simplified_reward = simplified_reward
        self._render = render
        self._render_speed = 1/float(fps)
        self._img_hw = img_hw

    def new_episode(self) -> (Episode, types.ObservationType):
        observation = self.gym_env.reset()
        return self, self._resize_bw(observation)

    def _resize_bw(self, observation):
        return cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY), (self._img_hw[1], self._img_hw[0]))

    def step(self, action):
        reward = 0
        for _ in range(self._repeat):
            observation, _reward, done, _ = self.gym_env.step(action)
            reward += _reward
            if self._render:
                self.gym_env.render()
                sleep(self._render_speed)
        if self._simplified_reward:
            if reward > 0:
                reward = 1
            elif reward < 0:
                reward = -1
            else:
                reward = 0
        return types.EpisodeStatus(self._resize_bw(observation), reward, done)
