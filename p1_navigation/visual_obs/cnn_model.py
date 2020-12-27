import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Inspired by: https://github.com/BY571/DQN-Atari-Agents

class Dueling_QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,layer_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state 4 frames
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.conv_1 = nn.Conv2d(state_size, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.ff_A = nn.Linear(64*7*7, layer_size)
        self.ff_V = nn.Linear(64*7*7, layer_size)
        self.advantage = nn.Linear(layer_size,action_size)
        self.value = nn.Linear(layer_size,1)
        
    def forward(self, state):
        """
        Forward pass through the network
         Params
        ======
        
            state: state
        """
        x = torch.relu(self.conv_1(state))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = x.view(state.size(0), -1)
  
        x_A = torch.relu(self.ff_A(x))
        x_V = torch.relu(self.ff_V(x))   
      
        value = self.value(x_V)
        value = value.expand(state.size(0), self.action_size)
        advantage = self.advantage(x_A)
        Q = value + advantage - advantage.mean()
        return Q