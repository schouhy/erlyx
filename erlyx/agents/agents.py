from erlyx.agents.base import PolicyAgent
from erlyx.policies import Policy
from erlyx import types

import numpy as np

from collections import deque


class EpsilonGreedyAgent(PolicyAgent):
    def __init__(self, policy: Policy, epsilon):
        super(EpsilonGreedyAgent, self).__init__(policy=policy)
        self.epsilon = epsilon

    def select_action(self, observation: types.ObservationType) -> types.ActionData:
        if np.random.uniform() < self.epsilon:
            return types.ActionData(action=np.random.choice(self.action_space))
        distribution = self.policy.get_distribution(observation)
        return types.ActionData(action=np.argmax(distribution))


class EpsilonGreedyHistoryAgent(PolicyAgent):
    def __init__(self, policy: Policy, epsilon, obs_shape):
        super(EpsilonGreedyHistoryAgent, self).__init__(policy=policy)
        self.epsilon = epsilon
        self._obs_shape = obs_shape
        self._memory_buffer = deque(maxlen=4)

    def reset_memory(self):
        self._memory_buffer = deque([np.zeros(shape=self._obs_shape)] * 4, maxlen=4)

    def select_action(self, observation: types.ObservationType) -> types.ActionData:
        self._memory_buffer.append(observation)
        if np.random.uniform() < self.epsilon:
            return types.ActionData(action=np.random.choice(self.action_space))
        observation = np.asarray(list(self._memory_buffer))
        distribution = self.policy.get_distribution(observation)
        return types.ActionData(action=np.argmax(distribution))


class EpsilonStochasticAgent(PolicyAgent):
    def __init__(self, policy: Policy, epsilon):
        super(EpsilonStochasticAgent, self).__init__(policy=policy)
        self.epsilon = epsilon

    def select_action(self, observation: types.ObservationType) -> types.ActionData:
        if np.random.uniform() < self.epsilon:
            return types.ActionData(action=np.random.choice(self.action_space))
        distribution = self.policy.get_distribution(observation)
        return types.ActionData(action=np.random.choice(self.action_space, p=distribution))


class NormalNoiseGreedyAgent(PolicyAgent):
    def __init__(self, policy: Policy, sigma):
        super(NormalNoiseGreedyAgent, self).__init__(policy=policy)
        self.sigma = sigma

    def select_action(self, observation: types.ObservationType) -> types.ActionData:
        distribution = self.policy.get_distribution(observation)
        noise = np.random.normal(loc=0., scale=self.sigma, size=len(self.action_space))
        distribution += noise
        return types.ActionData(action=np.argmax(distribution))


class NormalNoiseStochasticAgent(PolicyAgent):
    def __init__(self, policy: Policy, sigma):
        super(NormalNoiseStochasticAgent, self).__init__(policy=policy)
        self.sigma = sigma

    def select_action(self, observation: types.ObservationType) -> types.ActionData:
        distribution = self.policy.get_distribution(observation)
        noise = np.random.normal(loc=0., scale=self.sigma, size=len(self.action_space))
        distribution += noise
        return types.ActionData(action=np.random.choice(self.action_space, p=distribution))
