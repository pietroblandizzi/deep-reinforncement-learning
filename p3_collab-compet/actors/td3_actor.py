import numpy as np
import random
import copy

from models.td3_model import Actor, Critic
from utils.noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99             # discount factor
TAU = 1e-3               # for soft update of target parameters
LR_ACTOR = 5e-4          # learning rate of the actor 
LR_CRITIC = 1e-3         # learning rate of the critic
WEIGHT_DECAY = 0         # L2 weight decay

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, num_agents, policy_noise=0.1):
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
        self.policy_noise = policy_noise

        # Actor Network (w/ Target Network)
        self.actor = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic = Critic(state_size*num_agents, action_size*num_agents, random_seed).to(device)
        self.critic_target = Critic(state_size*num_agents, action_size*num_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, scale=1.0, sigma=0.2)

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs)
        action += noise*self.noise.noise()
        action = torch.clamp(action,-1,1)
        return action
    

    def target_act(self, obs):
        obs = obs.to(device)
        action = self.actor_target(obs)
        return action

    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.total_it += 1
        
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        # add noise to the action used to compute the targets        
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-1, 1)
            actions_next = (self.actor_target(next_states) + noise).clamp(-1, 1)
            
            # Compute the target Q value
            Q1_targets_next, Q2_targets_next = self.critic_target(next_states, actions_next)
            Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q1_expected, Q2_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q1_expected, Q_targets) + F.mse_loss(Q2_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # As suggested from Udacity add grad clipping to the critic
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # Delayed Policy Update
        if self.total_it % POLICY_FREQ == 0:
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local.Q1(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)                     

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