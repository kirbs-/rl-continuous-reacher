import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model for continuous control."""

    # def __init__(self, state_dim, action_cnt, num_of_neurons, seed=None):
    #     """Initialize parameters and build model.
    #     Params
    #     ======
    #         state_size (int): Dimension of each state
    #         action_size (int): Dimension of each action
    #         num_of_neurons (int): Number of neurons in each fully connected layer
    #     """
    #     super(Actor, self).__init__()
    #     if seed:
    #         self.seed = torch.manual_seed(seed)
    #     self.linear_relu_stack = nn.Sequential(
    #         nn.Linear(state_dim, num_of_neurons),
    #         nn.BatchNorm1d(num_of_neurons),
    #         nn.ReLU(),
    #         nn.Linear(num_of_neurons, num_of_neurons),
    #         nn.BatchNorm1d(num_of_neurons),
    #         nn.ReLU(),
    #         nn.Linear(num_of_neurons, num_of_neurons),
    #         nn.Tanh(),
    #         nn.Linear(num_of_neurons, action_cnt)
    #     )
    #     self.initialize_weights()

    # def initialize_weights(self):
    #     for layer in self._modules['linear_relu_stack']:
    #         if type(layer) == nn.Linear:
    #             layer.weight.data.uniform_(*hidden_init(layer))

    # def forward(self, state):
    #     """Build a network that maps state -> action values."""
    #     return self.linear_relu_stack(state.float())

    def __init__(self, state_size, action_size, num_of_neurons, seed=6, fc1_units=400, fc2_units=300, use_bn=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_bn = use_bn
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)            
        self.fc3 = nn.Linear(fc2_units, action_size)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(state_size)
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.use_bn:
            x = self.fc1(self.bn1(state))
        else:            
            x = self.fc1(state)
            
        x = F.relu(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.fc2(x)
        x = F.relu(x)
        if self.use_bn:
            x = self.bn3(x)
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) model."""

    def __init__(self, state_dim, action_cnt, num_of_neurons, seed=None):
        super(Critic, self).__init__()
        if seed:
            self.seed = torch.manual_seed(seed)

        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(state_dim, num_of_neurons),
        #     nn.BatchNorm1d(num_of_neurons),
        #     nn.ReLU(),
        #     nn.Linear(num_of_neurons + action_cnt, num_of_neurons),
        #     nn.BatchNorm1d(num_of_neurons),
        #     nn.ReLU(),
        #     nn.Linear(num_of_neurons, num_of_neurons),
        #     nn.ReLU(),
        #     nn.Linear(num_of_neurons, action_cnt)
        # )

        self.fcs1 = nn.Linear(state_dim, num_of_neurons)
        self.fc2 = nn.Linear(num_of_neurons+action_cnt, num_of_neurons)
        self.fc3 = nn.Linear(num_of_neurons, action_cnt)
        # self.initialize_weights()
        self.reset_parameters()

    def initialize_weights(self):
        for layer in self._modules['linear_relu_stack']:
            if type(layer) == nn.Linear:
                layer.weight.data.uniform_(*hidden_init(layer))

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        # x = torch.concat((state, action), dim=1)
        # return self.linear_relu_stack(x)
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)