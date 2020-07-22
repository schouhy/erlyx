import abc
from erlyx import types


class Episode(abc.ABC):
    @abc.abstractmethod
    def step(self, action: types.ActionType) -> types.EpisodeStatus:
        pass


class BaseEnvironment(abc.ABC):
    @abc.abstractmethod
    def new_episode(self) -> (Episode, types.ObservationType):
        pass
