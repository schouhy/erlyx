import erlyx

from erlyx.environment import GymAtariBWEnvironment

from erlyx.policies import PytorchPolicy
from erlyx.agents import BaseAgent
from erlyx.datasets import SequenceDataset
from erlyx.algorithms.ddqn import DoubleDeepQLearner

from erlyx.callbacks import BaseCallback
from erlyx.callbacks.recorders import TransitionRecorder, RewardRecorder
from erlyx.callbacks.updaters import OnlineUpdater, LinearEpsilonDecay
from erlyx.callbacks.checkpoint import PytorchCheckPointer

from gym import wrappers

import numpy as np
import pandas as pd
import torch
from datetime import datetime

from collections import deque
from pathlib import Path
import os
import h5py

import fire


## Agent

class EpsilonGreedyHistoryAgent(BaseAgent):
    def __init__(self, policy, epsilon, obs_shape):
        super(EpsilonGreedyHistoryAgent, self).__init__(policy=policy)
        self.epsilon = epsilon
        self._obs_shape = obs_shape
        self._memory_buffer = deque(maxlen=4)

    def reset_memory(self):
        self._memory_buffer = deque([np.zeros(shape=self._obs_shape)] * 4, maxlen=4)

    def select_action(self, observation):
        self._memory_buffer.append(observation)
        observation = np.asarray(list(self._memory_buffer))
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.action_space)
        distribution = self.policy.get_distribution(observation)
        return np.argmax(distribution)


class AgentMemoryReseter(BaseCallback):
    def __init__(self, agent):
        self._agent = agent

    def on_episode_begin(self, initial_observation):
        self._agent.reset_memory()


## Policy

class DuelingNetwork(torch.nn.Module):
    def __init__(self):
        super(DuelingNetwork, self).__init__()
        self._conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )
        self._linear_advantage = torch.nn.Sequential(
            torch.nn.Linear(64 * 49, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 18)
        )
        self._linear_value = torch.nn.Sequential(
            torch.nn.Linear(64 * 49, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        output = self._conv1(x)
        output = output.view(x.shape[0], -1)
        advantage = self._linear_advantage(output)
        value = self._linear_value(output)
        output = value + advantage - advantage.mean()
        return output


class CNNPolicy(PytorchPolicy):
    def __init__(self):
        self._model = DuelingNetwork()

    @property
    def model(self):
        return self._model

    def process_observation(self, observation):
        tensor = torch.Tensor(observation / 128. - 1.) / 0.35  # divided by 0.35 because of standard deviation
        return tensor

    def get_distribution(self, state):
        with torch.no_grad():
            self.model.eval()
            q_values = self.model(self.process_observation(state))
            distribution = torch.nn.functional.softmax(q_values, dim=1)
        return distribution.data.cpu().numpy().reshape(-1)

    def num_actions(self):
        return 18


def train(
        checkpoint_filepath,
        dataset_filepath,
        dataset_position,
        initial_epsilon,
        min_epsilon,
        decay_step,
        train_steps,
        log_folder
):
    # Maybe load torch `state_dict` of model and optimizer
    if checkpoint_filepath is not None:
        checkpoint = torch.load(checkpoint_filepath)

    # policy
    policy = CNNPolicy()
    if checkpoint_filepath is not None:
        print('loading model weights')
        policy.model.load_state_dict(checkpoint['model'])

    # agent
    img_hw = (84, 84)
    agent = EpsilonGreedyHistoryAgent(policy=policy, epsilon=0.01, obs_shape=img_hw)

    # dataset
    dataset_maxlen = 1_000_000
    dtypes = {o: np.uint8 for o in ['state', 'action', 'done']}
    dtypes['reward'] = np.float32
    dataset = SequenceDataset(input_shape=img_hw, max_length=dataset_maxlen, history_length=4, dtypes=dtypes)
    # maybe load dataset
    if dataset_filepath is not None:
        print('loading dataset')
        hf = h5py.File(dataset_filepath, 'r')
        dataset._actions = np.array(hf.get('actions'))
        dataset._rewards = np.array(hf.get('rewards'))
        dataset._states = np.array(hf.get('states'))
        dataset._dones = np.array(hf.get('dones'))
        if dataset_position is not None:
            dataset._position = dataset_position

    # learner
    learner = DoubleDeepQLearner(32, policy, 1e-4 / 4, 30_000, loss_func=torch.nn.SmoothL1Loss())
    if checkpoint_filepath is not None:
        print('loading optimizer')
        learner.optimizer.load_state_dict(checkpoint['optimizer'])

    # Train
    log_folder = log_folder or datetime.now().strftime("%Y%m%d%H%M%S")
    log_folder = Path(log_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    min_observations = 0 if dataset_filepath is not None else dataset_maxlen

    train_callbacks = [
        # base
        AgentMemoryReseter(agent=agent),
        TransitionRecorder(dataset=dataset),
        OnlineUpdater(learner=learner,
                      dataset=dataset,
                      min_observations=min_observations,
                      update_frequency=1),
        LinearEpsilonDecay(agent=agent,
                           initial_epsilon=initial_epsilon,
                           min_epsilon=min_epsilon,
                           decay_step=decay_step),
        # logging
        RewardRecorder(log_filepath=log_folder / 'reward_log'),
        PytorchCheckPointer(environment=GymAtariBWEnvironment('Seaquest-v0', simplified_reward=False, img_hw=img_hw),
                            agent=agent,
                            learner=learner,
                            evaluation_frecuency=25_000,
                            min_steps=min_observations,
                            checkpoint_folder=log_folder,
                            log_filepath=log_folder / 'checkpoint_log')
    ]

    print(f"""
    Training with following parameters:
        checkpoint_filepath={checkpoint_filepath},
        dataset_filepath={dataset_filepath},
        dataset_position={dataset_position},
        initial_epsilon={initial_epsilon},
        min_epsilon={min_epsilon},
        decay_step={decay_step},
        min_observations={min_observations},
        train_steps={train_steps},
        log_folder={log_folder},
    """)

    try:
        erlyx.run_steps(
            environment=GymAtariBWEnvironment('Seaquest-v0', img_hw=(84, 84)),
            agent=agent,
            n_steps=train_steps,
            callbacks=train_callbacks
        )
    finally:
        # persist dataset
        print('persist dataset..')
        hf = h5py.File(log_folder / 'dataset.h5', 'w')
        hf.create_dataset('states', data=dataset._states, dtype=np.uint8)
        hf.create_dataset('actions', data=dataset._actions, dtype=np.uint8)
        hf.create_dataset('rewards', data=dataset._rewards, dtype=np.float32)
        hf.create_dataset('dones', data=dataset._dones, dtype=np.uint8)
        print(f'latest dataset position: {dataset._position}')
        hf.close()

def evaluate(
        checkpoint_filepath,
        num_evaluations,
        evaluation_video_filepath
):
    if evaluation_video_filepath is not None and num_evaluations != 1:
        raise Exception('num_evaluations must be 1 if `evaluation_video_filepath` is provided')
    img_hw = (84, 84)
    # policy
    policy = CNNPolicy()
    policy.model.load_state_dict(torch.load(checkpoint_filepath)['model'])
    # agent
    agent = EpsilonGreedyHistoryAgent(policy=policy, epsilon=0.0, obs_shape=img_hw)
    env = GymAtariBWEnvironment('Seaquest-v0', simplified_reward=False, img_hw=img_hw)
    if evaluation_video_filepath is not None:
        env.gym_env = wrappers.Monitor(env.gym_env, evaluation_video_filepath, force=True)
    rr = RewardRecorder()
    erlyx.run_episodes(env, agent, num_evaluations, [AgentMemoryReseter(agent), rr])
    env.gym_env.close()
    print(pd.Series(rr.rewards).describe())


def main(train_mode=False,
         checkpoint_filepath=None,
         dataset_filepath=None,
         dataset_position=None,
         initial_epsilon=2,
         min_epsilon=0.01,
         decay_step=1e-6,
         train_steps=50_000_000,
         log_folder=None,
         evaluation_mode=False,
         num_evaluations=1,
         evaluation_video_filepath=None):
    if not train_mode != evaluation_mode:
        raise Exception('exactly one of `train_mode` and `evaluation_mode` must be True')
    if train_mode:
        train(
            checkpoint_filepath=checkpoint_filepath,
            dataset_filepath=dataset_filepath,
            dataset_position=dataset_position,
            initial_epsilon=initial_epsilon,
            min_epsilon=min_epsilon,
            decay_step=decay_step,
            train_steps=train_steps,
            log_folder=log_folder
        )
    elif evaluation_mode:
        evaluate(
            checkpoint_filepath=checkpoint_filepath,
            num_evaluations=num_evaluations,
            evaluation_video_filepath=evaluation_video_filepath
        )


if __name__ == '__main__':
    fire.Fire(main)
