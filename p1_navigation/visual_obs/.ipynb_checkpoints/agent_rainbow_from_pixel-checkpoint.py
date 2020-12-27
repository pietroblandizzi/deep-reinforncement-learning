import numpy as np
import random
from collections import namedtuple, deque

from visual_obs.replay_buffer_frames import ReplayBuffer, PrioritizedReplayBuffer
from visual_obs.cnn_model import Dueling_QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
MAX_GRANDIENT_NORM = float('inf')
ALPHA = 0.6             # how much prioritization is used

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_Rainbow():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, layer_size, ddqn):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ddqn = ddqn

        # Q-Network
        self.qnetwork_local = Dueling_QNetwork(state_size, action_size,layer_size, seed).to(device)
        self.qnetwork_target = Dueling_QNetwork(state_size, action_size, layer_size, seed).to(device)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory = PrioritizedReplayBuffer(max_samples=BUFFER_SIZE, batch_size=BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        experience = (state, action, reward, next_state, float(done))
        self.memory.add(experience)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Performing Double DQN

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        idxs, weights, \
        (states, actions, rewards, next_states, dones) = experiences
        weights = torch.from_numpy(weights).float().to(device)
        
        states = states.reshape(BATCH_SIZE, 12, 84, 84)
        next_states = next_states.reshape(BATCH_SIZE, 12, 84, 84)
            
        if self.ddqn:
            # Get the argmax action of the next state. From the online network
            argmax_a_q_next = self.qnetwork_local(next_states).max(1)[1]
            # Get the Q-values of the next state as before.
            Q_next = self.qnetwork_target(next_states).detach()
            # Now, we use the indices to get the max values of the next states.     
        
            Q_targets_next = Q_next[
                np.arange(BATCH_SIZE), argmax_a_q_next].unsqueeze(1)
        
        else:
            # Get max predicted Q values (for next states) from target model  
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
  
          
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))      
        # Get the current estimates. From local Model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
         # Compute loss
        td_error = Q_expected - Q_targets
        loss = (weights * td_error).pow(2).mul(0.5).mean()
        
         # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Ready for implementing Huber loss
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 
                                       MAX_GRANDIENT_NORM) 
        
        self.optimizer.step()
        
        # calculate priorities and update them
        priorities = np.abs(td_error.detach().cpu().numpy())
        self.memory.update(idxs, priorities)

      # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self):
        """ 
        Save the current paramters of the network.
        """
        torch.save(self.qnetwork_local.state_dict(), 'checkpoint_rainbow.pth')