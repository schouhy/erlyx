from erlyx.callbacks import BaseCallback
from erlyx.datasets import BaseDataset
from erlyx import types

import pandas as pd
from collections import deque
from typing import NamedTuple
from pathlib import Path


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
    def __init__(self, log_filepath=None):
        self.rewards = []
        self.total_reward = None
        self._log_filepath = Path(log_filepath) if log_filepath is not None else None

    def on_episode_begin(self, initial_observation):
        self.total_reward = 0

    def on_step_end(self, action, observation, reward, done):
        self.total_reward += reward

    def on_episode_end(self):
        self.rewards.append(self.total_reward)
        if self._log_filepath is not None:
            with open(self._log_filepath, 'a') as file:
                file.write(str(self.total_reward))
                file.write('\n')

    def plot_rewards(self):
        pd.Series(self.rewards).plot()
