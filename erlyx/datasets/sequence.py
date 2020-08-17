from erlyx.datasets import BaseDataset

import numpy as np


class SequenceDataset(BaseDataset):
    def __init__(self, input_shape, max_length, history_length=4, dtypes=None):
        if dtypes is None:
            dtypes = {o: None for o in ['state', 'action', 'reward', 'done']}
        self._states = np.zeros(shape=(max_length, *input_shape), dtype=dtypes['state'])
        self._actions = np.zeros(shape=max_length, dtype=dtypes['action'])
        self._rewards = np.zeros(shape=max_length, dtype=dtypes['reward'])
        self._dones = np.zeros(shape=max_length, dtype=dtypes['done'])
        self._position = 0
        self._max_length = max_length
        self._history_length = history_length
        self._rng = np.random.default_rng()

    def push(self, transition):
        state, action, next_state, reward, done = transition
        next_position = (self._position + 1) % self._max_length
        self._states[[self._position, next_position], :] = [state, next_state]
        self._actions[self._position] = action
        self._rewards[self._position] = reward
        self._dones[next_position] = done
        self._position = next_position

    def __getitem__(self, index):
        if self._dones[index]:
            return self[index - 1]
        indexes = np.arange(index - self._history_length + 1, index + 2) % self._max_length
        transition_slice = self._states[indexes, :]
        # Transforma los estados anteriores que tengan done=True en ceros
        previous_done_index = np.insert(self._dones[indexes[:-2]], 0, 0).cumsum().argmax()
        transition_slice[:previous_done_index, :] = 0
        # builds transition
        state = transition_slice[:self._history_length]
        next_state = transition_slice[1:self._history_length + 1]
        action = self._actions[index]
        reward = self._rewards[index]
        done = self._dones[indexes[-1]]
        return (state, action, next_state, reward, done)

    def sample(self, batch_size):
        return [self[index] for index in self._rng.choice(self._max_length, batch_size)]
