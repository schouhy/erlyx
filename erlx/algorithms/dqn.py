from erlx.learners import BaseLearner

import torch
import copy


class DeepQLearner(BaseLearner):
    def __init__(self, batch_size, policy, learning_rate, sync_frequency):
        self.policy = policy
        self.target_policy = copy.deepcopy(policy)
        self.target_policy.model.eval()
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(policy.model.parameters(), lr=learning_rate)
        self.sync_frequency = sync_frequency
        self.losses = []
        self.update_counter = 0

    def update(self, dataset):
        self.update_counter += 1
        self.policy.model.train()
        mini_batch = dataset.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*mini_batch)
        x_batch = torch.stack(list(map(torch.Tensor, states)))
        q_values = self.policy.model(x_batch)
        rewards = torch.stack(list(map(torch.tensor, rewards)))
        next_states = torch.stack(list(map(torch.Tensor, next_states)))
        next_q_values = self.target_policy.model(next_states)

        # esto tiene que quedarse con los valores de la columna ganadora para cada fila
        q_value = q_values.gather(1, torch.LongTensor(actions).view(-1, 1)).view(-1)
        next_q_value = next_q_values.max(1).values
        expected_q_value = rewards + 0.99 * next_q_value * (1. - torch.FloatTensor(dones))

        expected_q_value = expected_q_value.detach().view(-1)
        loss = torch.nn.MSELoss()(q_value, expected_q_value)
        # loss = torch.nn.SmoothL1Loss()(q_value, expected_q_value)
        self.losses.append(loss.data.cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 0.5)
        self.optimizer.step()

        if self.update_counter % self.sync_frequency == 0:
            self._update_target_policy()

    def _update_target_policy(self):
        self.target_policy = copy.deepcopy(self.policy)
        self.target_policy.model.eval()
