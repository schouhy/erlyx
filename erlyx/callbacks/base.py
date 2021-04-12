from erlyx import types


class BaseCallback:
    def call(self, event: str, **params: dict) -> types.Optional[bool]:
        return getattr(self, event)(**params)

    def __repr__(self):
        return type(self).__name__

    ## events

    def on_train_begin(self) -> types.Optional[bool]:
        pass

    def on_episode_begin(self, initial_observation: types.ObservationType) -> types.Optional[bool]:
        pass

    def on_step_begin(self):
        pass

    def on_step_end(
            self,
            action: types.ActionData,
            observation: types.ObservationType,
            reward: types.RewardType,
            done: bool
    ) -> types.Optional[bool]:
        pass

    def on_episode_end(self) -> types.Optional[bool]:
        pass

    def on_train_end(self) -> types.Optional[bool]:
        pass


class CallbackTuple:
    def __init__(self, callbacks=None):
        self._callbacks = tuple(callbacks) if callbacks is not None else ()
        self._state = None

    @property
    def callbacks(self):
        return self._callbacks

    def call(self, event, **params):
        outputs = []
        for cb in self._callbacks:
            outputs.append(cb.call(event, **params))
        self._state = outputs
        return any(outputs)

    def add_callback(self, new_callback):
        self._callbacks = (*self._callbacks, new_callback)

    def __getitem__(self, index):
        return self._callbacks[index]

    def __repr__(self):
        return f'CallbackTuple[{self._callbacks.__repr__()}]'
