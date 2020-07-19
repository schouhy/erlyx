from erlx.policies import Policy
from erlx import types
import abc

import numpy as np


## Assumes discrete set of actions equal to range(N) for some N


class BaseAgent(abc.ABC):
    def __init__(self, policy: Policy):
        self.policy = policy
        self.action_space = list(range(policy.num_actions()))

    @abc.abstractmethod
    def select_action(self, observation: types.ObservationType) -> types.ActionType:
        pass
