from erlyx.policies.base import Policy

import torch
from typing import List


def linear_layer(input_dim, output_dim, bias, dropout_probability, batchnorm):
    layers = []
    if batchnorm:
        layers.append(torch.nn.BatchNorm1d(input_dim))
    if dropout_probability > 0.:
        layers.append(torch.nn.Dropout(dropout_probability))
    layers.append(torch.nn.Linear(input_dim, output_dim, bias))
    return torch.nn.Sequential(*layers)


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, layer_dimensions: List[int], dropout_ps, batchnorm):
        super(FeedForwardNetwork, self).__init__()
        layers = []
        input_hidden = layer_dimensions[:-1]
        dropout_ps = dropout_ps or [0.]*(len(input_hidden))
        assert len(dropout_ps) == len(layer_dimensions)-1

        for n_in, n_out, p in zip(input_hidden[:-1], input_hidden[1:], dropout_ps[:-1]):
            layers.append(linear_layer(n_in, n_out, True, p, batchnorm))
            layers.append(torch.nn.ReLU())
        layers.append(linear_layer(layer_dimensions[-2], layer_dimensions[-1], True, dropout_ps[-1], batchnorm))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(1, -1)
        return self.layers(x)


class FeedForwardNetworkPolicy(Policy):
    def __init__(self, layer_dimensions: List[int], dropout_ps=None, batchnorm=False):
        self.model = FeedForwardNetwork(layer_dimensions, dropout_ps, batchnorm)
        self._num_actions = layer_dimensions[-1]

    def num_actions(self):
        return self._num_actions

    def process_state(self, state):
        return torch.Tensor(state)

    def get_distribution(self, state):
        with torch.no_grad():
            self.model.eval()
            q_values = self.model(self.process_state(state))
            distribution = torch.nn.functional.softmax(q_values, dim=1)
        return distribution.data.cpu().numpy().reshape(-1)
