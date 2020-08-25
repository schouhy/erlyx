from erlyx.callbacks import BaseCallback
from erlyx.callbacks.recorders import RewardRecorder
from erlyx import run_episodes

import numpy as np
import torch
from datetime import datetime
from pathlib import Path


class PytorchCheckPointer(BaseCallback):
    def __init__(
            self,
            environment,
            agent,
            learner,
            evaluation_frecuency,
            min_steps,
            checkpoint_folder,
            log_filepath=None
    ):
        self._environment = environment
        self._agent = agent
        self._learner = learner
        self._counter = -min_steps
        self._stats = [-float('inf'), None]
        self._evaluate = True
        self._evaluation_frequency = evaluation_frecuency
        self._log_filepath = Path(log_filepath) if log_filepath is not None else None
        self._checkpoint_folder = Path(checkpoint_folder)

    def on_step_end(self, action, observation, reward, done):
        self._counter += 1
        if self._counter % self._evaluation_frequency == 0:
            self._evaluate = True
        if self._counter < 0:
            self._evaluate = False

    def on_episode_end(self):
        if not self._evaluate:
            return
        # Evaluate
        self._evaluate = False
        reward_recorder = RewardRecorder()
        original_epsilon = self._agent.epsilon
        self._agent.epsilon = 0.01
        run_episodes(
            self._environment,
            self._agent,
            5,
            [reward_recorder],
            use_tqdm=False
        )
        self._agent.epsilon = original_epsilon
        new_mean = np.mean(reward_recorder.rewards)

        # Save
        self._checkpoint('latest_checkpoint.sd')
        if new_mean > self._stats[0]:
            print('new best wachin!')
            self._stats[0] = new_mean
            self._stats[1] = reward_recorder.rewards
            self._checkpoint('best_checkpoint.sd')
        print(f'[Evaluation] mean: {new_mean}, rewards: {reward_recorder.rewards}')

        if self._log_filepath is not None:
            # Log to file
            with open(self._log_filepath, 'a') as file:
                file.write(datetime.now().strftime("%d-%m-%Y %H:%M"))
                file.write('\n')
                if self._stats[0] == new_mean:
                    file.write('¡Sungaratungaratun! Tenemos un nuevo record:')
                    file.write('\n')
                    file.write(f'mean: {self._stats[0]}, rewards: {self._stats[1]}')
                    file.write('\n')
                else:
                    file.write(f'new mean: {new_mean}, rewards: {reward_recorder.rewards}')
                    file.write('\n')
                    file.write(f'best mean: {self._stats[0]}, rewards: {self._stats[1]}')
                    file.write('\n')

    def _checkpoint(self, name):
        state = {
            'model': self._agent.policy.model.state_dict(),
            'optimizer': self._learner.optimizer.state_dict()
        }
        torch.save(state, self._checkpoint_folder / name)
