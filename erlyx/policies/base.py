from erlyx import types

import abc


class Policy(abc.ABC):
    @abc.abstractmethod
    def get_distribution(self, observation: types.ObservationType) -> types.DiscreteDistributionType:
        """returns a distribution over the space of actions"""

    @abc.abstractmethod
    def num_actions(self) -> int:
        pass
