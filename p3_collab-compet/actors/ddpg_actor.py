import numpy as np
import random
import copy

from models.ddpg_model import Actor, Critic
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
EPS_DECAY = 1e-6

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
        self.noise = OUNoise(action_size, scale=1.0)
    
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