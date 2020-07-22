from erlyx.environment.base import BaseEnvironment, Episode
from erlyx import types
import gym


class GymEnvironment(BaseEnvironment, Episode):
    def __init__(self, env_name):
        self.gym_env = gym.make(env_name)

    def new_episode(self) -> (Episode, types.ObservationType):
        observation = self.gym_env.reset()
        return self, observation

    def step(self, action: types.ActionType) -> types.EpisodeStatus:
        observation, reward, done, _ = self.gym_env.step(action)
        return types.EpisodeStatus(observation, reward, done)
