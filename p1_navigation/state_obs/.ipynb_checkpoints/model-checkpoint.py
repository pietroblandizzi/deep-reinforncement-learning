import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_dims=(64, 32), activation_fc=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            device(string): Train on cpu or gpu
            hidden_dims (int, int): Number of nodes in each hidden layers
            activation_fc (torch.nn.functional): Type of activation of hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(state_size,hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        
        self.output_layer = nn.Linear(hidden_dims[-1], action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device, dtype=torch.float32)
            x = x.unsqueeze(0)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        
        x = self.output_layer(x)
        
        return x
    

class QDuelingNetwork(nn.Module):
    """Actor (Policy) Model using a dueling architecture"""

    def __init__(self, state_size, action_size, seed, hidden_dims=(64,32), activation_fc=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            device(string): Train on cpu or gpu
            hidden_dims (int, int): Number of nodes in each hidden layers
            activation_fc (torch.nn.functional): Type of activation of hidden layers
        """
        super(QDuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(state_size,hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            
        self.value_output = nn.Linear(hidden_dims[-1], 1)
        self.advantage_output = nn.Linear(hidden_dims[-1], action_size)
    
    def forward(self, state):
        """Build a network that maps state -> action values
        Using combination of value function and advantage function.
        """
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device, dtype=torch.float32)
            x = x.unsqueeze(0)
        x = self.activation_fc(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        
        a = self.advantage_output(x)
        v = self.value_output(x)
        v = v.expand_as(a)
        
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        
        return q


