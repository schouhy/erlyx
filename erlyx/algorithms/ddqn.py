from erlyx.learners import BaseLearner

import torch
import copy


class DoubleDeepQLearner(BaseLearner):
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
        # build minibatch
        mini_batch = dataset.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*mini_batch)
        x_batch = torch.stack(list(map(torch.Tensor, states)))
        rewards = torch.stack(list(map(torch.tensor, rewards)))
        next_states = torch.stack(list(map(torch.Tensor, next_states)))

        # compute target
        next_action = self.policy.model(next_states).argmax(1)
        next_q_values_target = self.target_policy.model(next_states)
        next_q_value = next_q_values_target.gather(1, torch.LongTensor(next_action).view(-1, 1)).view(-1)
        expected_q_value = rewards + 0.99 * next_q_value * (1. - torch.FloatTensor(dones))
        expected_q_value = expected_q_value.detach().view(-1)

        # compute ind variable
        q_values = self.policy.model(x_batch)
        q_value = q_values.gather(1, torch.LongTensor(actions).view(-1, 1)).view(-1)

        loss = torch.nn.MSELoss()(q_value, expected_q_value)
        # loss = torch.nn.SmoothL1Loss()(q_value, expected_q_value)
        self.losses.append(loss.data.cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 0.5)
        self.optimizer.step()

        if self.update_counter % self.sync_frequency == 0:
            self.target_policy = copy.deepcopy(self.policy)
            self.target_policy.model.eval()
