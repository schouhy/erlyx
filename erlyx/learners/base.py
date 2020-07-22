from erlyx.datasets import BaseDataset

import abc


class BaseLearner(abc.ABC):
    @abc.abstractmethod
    def update(self, dataset: BaseDataset):
        """Performs update of the weights using `learners`"""


