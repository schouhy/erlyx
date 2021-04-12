from erlyx.callbacks.base import BaseCallback
from erlyx.learners import BaseLearner
from erlyx.datasets import BaseDataset
from erlyx.agents import EpsilonGreedyAgent, EpsilonStochasticAgent
from erlyx import types
from typing import Union


class OnlineUpdater(BaseCallback):
    def __init__(self, learner: BaseLearner, dataset: BaseDataset, min_observations: int, update_frequency: int):
        self._learner = learner
        self._dataset = dataset
        self._num_observations = -min_observations
        self._update_frequency = update_frequency

    def on_step_end(
            self,
            action: types.ActionData,
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


class ExponentialEpsilonDecay(BaseCallback):
    def __init__(
            self,
            agent: Union[EpsilonGreedyAgent, EpsilonStochasticAgent],
            initial_epsilon=1., min_epsilon=0.05, decay_factor=0.99):
        self._agent = agent
        self._initial_epsilon = initial_epsilon
        self._min_epsilon = min_epsilon
        self._decay_factor = decay_factor
        self._original_epsilon = None

    def on_train_begin(self, *args):
        self._original_epsilon = self._agent.epsilon
        self._agent.epsilon = self._initial_epsilon

    def on_step_end(self, action, observation, reward, done):
        self._agent.epsilon = max(self._min_epsilon, self._agent.epsilon * self._decay_factor)

    def on_train_end(self):
        self._agent.epsilon = self._original_epsilon


class LinearEpsilonDecay(BaseCallback):
    def __init__(
            self,
            agent: Union[EpsilonGreedyAgent, EpsilonStochasticAgent],
            initial_epsilon=1., min_epsilon=0.05, decay_step=1e-5):
        self._agent = agent
        self._initial_epsilon = initial_epsilon
        self._min_epsilon = min_epsilon
        self._decay_step = decay_step
        self._original_epsilon = None

    def on_train_begin(self, *args):
        self._original_epsilon = self._agent.epsilon
        self._agent.epsilon = self._initial_epsilon

    def on_step_end(self, action, observation, reward, done):
        self._agent.epsilon = max(self._min_epsilon, self._agent.epsilon - self._decay_step)

    def on_train_end(self):
        self._agent.epsilon = self._original_epsilon
