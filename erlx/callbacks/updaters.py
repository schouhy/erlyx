from erlx.callbacks.base import BaseCallback
from erlx.learners import BaseLearner
from erlx.datasets import BaseDataset
from erlx.agents import EpsilonGreedyAgent, EpsilonStochasticAgent
from erlx import types
from typing import Union


class OnlineUpdater(BaseCallback):
    def __init__(self, learner: BaseLearner, dataset: BaseDataset, min_observations: int, update_frequency: int):
        self._learner = learner
        self._dataset = dataset
        self._num_observations = -min_observations
        self._update_frequency = update_frequency

    def on_step_end(
            self,
            action: types.ActionType,
            observation: types.ObservationType,
            reward: types.RewardType,
            done: bool
    ):
        self._num_observations += 1
        if self._num_observations < 0:
            return
        if self._num_observations % self._update_frequency != 0:
            return
        self._learner.update(self._dataset)


class EpsilonDecay(BaseCallback):
    def __init__(self, agent: Union[EpsilonGreedyAgent, EpsilonStochasticAgent]):
        self.agent = agent

    def on_train_begin(self, *args):
        self.agent.epsilon = 2.

    def on_step_end(self, action, observation, reward, done):
        self.agent.epsilon = max(0.05, self.agent.epsilon * 0.99)
