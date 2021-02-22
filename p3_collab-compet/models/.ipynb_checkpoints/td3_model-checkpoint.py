import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
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
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


# Code inspired from the work of the original autor:
# https://github.com/sfujim/TD3
# and https://www.manning.com/books/grokking-deep-reinforcement-learning
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Stream 1
        self.s1fc1 = nn.Linear(state_size, fc1_units)
        self.s1fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.s1fc3 = nn.Linear(fc2_units, 1)
        
        self.s2fc1 = nn.Linear(state_size, fc1_units)
        self.s2fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.s2fc3 = nn.Linear(fc2_units, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.s1fc1.weight.data.uniform_(*hidden_init(self.s1fc1))
        self.s1fc2.weight.data.uniform_(*hidden_init(self.s1fc2))
        self.s1fc3.weight.data.uniform_(-3e-3, 3e-3)

        self.s2fc1.weight.data.uniform_(*hidden_init(self.s2fc1))
        self.s2fc2.weight.data.uniform_(*hidden_init(self.s2fc2))
        self.s2fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def Q1(self, state, action):
        """Forward pass for the first architecture"""
        # Q1
        xs = F.relu(self.s1fc1(state))
        x1 = torch.cat((xs, action), dim=1)
        x1 = F.relu(self.s1fc2(x1))
        q1 = self.s1fc3(x1)
        
        return q1
    
    
    def Q2(self, state, action):
        """Forward pass for the second architecture"""
        xs = F.relu(self.s2fc1(state))
        x2 = torch.cat((xs, action), dim=1)
        x2 = F.relu(self.s2fc2(x2))
        q2 = self.s2fc3(x2)
        
        return q2

    def forward(self, state, action):
        return self.Q1(state,action), self.Q2(state, action)
    