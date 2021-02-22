import numpy as np
import random
import copy

from models.ddpg_model import Actor, Critic
from utils.noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim


LR_ACTOR = 5e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, num_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR, weight_decay=0)

        # Critic Network / Target Network)
        self.critic = Critic(state_size*num_agents, action_size*num_agents, random_seed).to(device)
        self.critic_target = Critic(state_size*num_agents, action_size*num_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=0)

        # Noise process
        self.noise = OUNoise(action_size, scale=1.0, sigma=0.4)
    
    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs)
        action += noise*self.noise.noise()
        action = torch.clamp(action,-1,1)
        return action
    
    # target actor is only used for updates not exploration and so has no noise
    def target_act(self, obs):
        obs = obs.to(device)
        action = self.actor_target(obs)
        return action
    
    def reset(self):
        self.noise.reset()
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)