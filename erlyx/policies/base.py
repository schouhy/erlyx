from erlyx import types

import torch

import abc


class Policy(abc.ABC):
    @abc.abstractmethod
    def get_distribution(self, observation: types.ObservationType) -> types.DiscreteDistributionType:
        """returns a distribution over the space of actions"""

    @abc.abstractmethod
    def num_actions(self) -> int:
        pass


class PytorchPolicy(Policy):
    @property
    @abc.abstractmethod
    def model(self) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def process_observation(self, observation: types.ObservationType) -> torch.Tensor:
        pass
