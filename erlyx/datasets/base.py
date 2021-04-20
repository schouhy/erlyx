from collections import deque
import random
import abc


class BaseDataset(abc.ABC):
    @abc.abstractmethod
    def push(self, data):
        pass

    @abc.abstractmethod
    def sample(self, batch_size):
        pass


class DequeDataset(BaseDataset):
    def __init__(self, max_length):
        self._memory = deque(maxlen=max_length)

    def __len__(self):
        return len(self._memory)

    def push(self, data, batch=False):
        if batch:
            self._memory.extend(data)
        else:
            self._memory.append(data)

    def sample(self, batch_size):
        return random.sample(self._memory, min(batch_size, len(self)))
