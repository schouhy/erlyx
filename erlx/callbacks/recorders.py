from erlx.callbacks import BaseCallback
from erlx.datasets import BaseDataset
from erlx import types
import erlx

import numpy as np
import pandas as pd
from collections import deque
from typing import NamedTuple
import copy


class Transition(NamedTuple):
    observation: types.ObservationType
    action: types.ActionType
    next_observation: types.ObservationType
    reward: types.RewardType
    done: bool


class TransitionRecorder(BaseCallback):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset
        self._prev_observation = None

    def on_episode_begin(self, initial_observation: types.ObservationType):
        self._prev_observation = initial_observation

    def on_step_end(
            self,
            action: types.ActionType,
            observation: types.ObservationType,
            reward: types.RewardType,
            done: bool
    ):
        self.dataset.push(Transition(self._prev_observation, action, observation, reward, done))
        self._prev_observation = observation


class TransitionSequenceRecorder(BaseCallback):
    def __init__(self, dataset: BaseDataset, sequence_length: int):
        super(TransitionSequenceRecorder, self).__init__()
        self.dataset = dataset
        self._buffer = deque(maxlen=sequence_length)
        self._prev_observation = None

    def on_episode_begin(self, initial_observation: types.ObservationType):
        self._prev_observation = initial_observation

    def on_step_end(
            self,
            action: types.ActionType,
            observation: types.ObservationType,
            reward: types.RewardType,
            done: bool
    ):
        self._buffer.append(Transition(self._prev_observation, action, observation, reward, done))
        self.dataset.push(list(self._buffer))
        self._prev_observation = observation

    def on_episode_end(self):
        # Drain buffer
        self._buffer.popleft()
        while self._buffer:
            self.dataset.push(list(self._buffer))
            self._buffer.popleft()


class RewardRecorder(BaseCallback):
    def __init__(self):
        self.rewards = []
        self.counter = 0
        self.total_reward = None

    def on_episode_begin(self, initial_observation):
        self.total_reward = 0

    def on_step_end(self, action, observation, reward, done):
        self.total_reward += reward

    def on_episode_end(self):
        self.rewards.append(self.total_reward)
        self.counter += 1

    def plot_rewards(self):
        pd.Series(self.rewards).plot()


class AgentVersionRecorder(BaseCallback):
    def __init__(self, agent):
        self.agent = agent
        self.best_agent = None
        self.best_reward = -float('inf')
        self._total_reward = None

    def on_episode_begin(self, initial_observation: types.ObservationType) -> types.Optional[bool]:
        self._total_reward = 0.
        return

    def on_step_end(
            self,
            action: types.ActionType,
            observation: types.ObservationType,
            reward: types.RewardType,
            done: bool
    ) -> types.Optional[bool]:
        self._total_reward += reward
        return

    def on_episode_end(self) -> types.Optional[bool]:
        if self._total_reward > self.best_reward:
            self.best_agent = copy.deepcopy(self.agent)
            self.best_reward = self._total_reward
            print(f'new best record: {self._total_reward}')
        return

