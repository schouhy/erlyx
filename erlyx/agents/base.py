from erlyx.policies import Policy
from erlyx import types
import abc

import numpy as np

## Assumes discrete set of actions equal to range(N) for some N


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def select_action(self,
                      observation: types.ObservationType) -> types.ActionType:
        pass


class PolicyAgent(BaseAgent):
    def __init__(self, policy: Policy):
        self.policy = policy
        self.action_space = list(range(policy.num_actions()))
