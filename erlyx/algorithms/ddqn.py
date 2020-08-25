from erlyx.learners import BaseLearner
from erlyx.policies import PytorchPolicy
from typing import Callable
import torch
import copy


class DoubleDeepQLearner(BaseLearner):
    def __init__(
            self,
            batch_size: int,
            policy: PytorchPolicy,
            learning_rate: float,
            sync_frequency: int,
            gamma: float = 0.99,
            loss_func: Callable = torch.nn.MSELoss(),
    ):
        self.policy = policy
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(policy.model.parameters(), lr=learning_rate)
        self.sync_frequency = sync_frequency
        #self.losses = []
        self.update_counter = 0
        self.gamma = gamma
        self.loss_func = loss_func
        self.target_policy = self._make_target_policy()

    def _make_target_policy(self):
        target_policy = copy.deepcopy(self.policy)
        for parameter in target_policy.model.parameters():
            parameter.requires_grad = False
        target_policy.model.eval()
        return target_policy

    def _update_target_policy(self):
        self.target_policy.model.load_state_dict(self.policy.model.state_dict())

    def update(self, dataset):
        self.update_counter += 1
        self.policy.model.train()
        # Fetch minibatch
        mini_batch = dataset.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        # Convert types
        states = torch.stack(list(map(self.policy.process_observation, states)))
        next_states = torch.stack(list(map(self.target_policy.process_observation, next_states)))
        rewards = torch.stack(list(map(torch.tensor, rewards)))
        actions = torch.LongTensor(actions)
        dones = torch.FloatTensor(dones)

        # Compute Q values
        q_values = self.policy.model(states)
        q_value = q_values.gather(1, actions.view(-1, 1)).view(-1)

        # Compute target
        next_action = self.policy.model(next_states).argmax(1, keepdim=True)
        next_q_values_target = self.target_policy.model(next_states)
        next_q_value = next_q_values_target.gather(1, next_action).view(-1)
        target = (rewards + self.gamma * next_q_value * (1. - dones)).detach()

        # Loss
        loss = self.loss_func(q_value, target)
        #self.losses.append(loss.data.cpu().numpy())

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 10)
        self.optimizer.step()

        # Update target policy
        if self.update_counter % self.sync_frequency == 0:
            self._update_target_policy()