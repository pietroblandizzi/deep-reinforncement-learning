# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
import numpy as np

from actors.ddpg_actor import Agent
from utils.replay_buffer import ReplayBuffer
from utils.utilities import soft_update, transpose_to_tensor, transpose_list

import torch
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = 'cpu'


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

LEARNING_STEPS = 1
LEARN_EACH = 1

class MADDPG:
    def __init__(self, state_size, action_size, random_seed, num_agents):
        super(MADDPG, self).__init__()

        # Generate a num_agents amount of players
        self.maddpg_agent = [Agent(state_size, action_size, random_seed, num_agents) for _ in range(num_agents)]
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        
        #1 global replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [maddpg_agent.actor_local for agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [agent.actor_target for agent in self.maddpg_agent]
        return target_actors
    
    def reset(self):
        for a in self.maddpg_agent:
            a.reset()
    
    def save(self, agent_ckp_prefix, critic_ckp_prefix):
        for i, agent in enumerate(self.maddpg_agent):
            torch.save(agent.actor_local.state_dict(), agent_ckp_prefix +'_'+ i + '_ckpt_local.pth')                 # save local actor
            torch.save(agent.actor_target.state_dict(), agent_ckp_prefix+'_'+ i + '_ckpt_target.pth')               # save target actor 
            torch.save(agent.critic_local.state_dict(), critic_ckp_prefix+'_'+ i + '_ckpt_local.pth')               # save local critic
            torch.save(agent.critic_target.state_dict(), critic_ckp_prefix +'_'+ i + '_ckpt_target.pth')   

    def target_act(self, obs):
        """get target network actions from all the agents in the MADDPG object 
        
        thr obs here are in shape (batch_size, state_size*2)
        
        """
        target_actions_0 = self.maddpg_agent[0].actor_target(obs[:,:24])
        target_actions_1 = self.maddpg_agent[1].actor_target(obs[:,24:])
        
        return torch.cat((target_actions_0, target_actions_1), dim=1)

    def local_act(self, obs):
        """get target network actions from all the agents in the MADDPG object 
        
        thr obs here are in shape (batch_size, state_size*2)
        
        """
        local_actions_0 = self.maddpg_agent[0].actor_local(obs[:,:24])
        local_actions_1 = self.maddpg_agent[1].actor_local(obs[:,24:])
        
        return torch.cat((local_actions_0, local_actions_1), dim=1)
  
    def act(self, obs, add_noise=True):
        """get actions from all agents in the MADDPG object
        
        note obs_all_agents has shape num_agents, state size (2, 24)
        
        """
        actions = np.zeros((self.num_agents, self.action_size))
        for i, agent in enumerate(self.maddpg_agent):
            actions[i,:] = agent.act(obs[i,:], add_noise)     
        return actions


    def step(self, state, action, reward, next_state, done):
        """update the critics and actors of all the agents """
        self.iter += 1   
        # Before adding it we make it full
        full_state = np.concatenate((state), axis=None)
        full_action = np.concatenate((action), axis=None)
        full_next_state = np.concatenate((next_state), axis=None)
        
        # Save experience / reward
        self.memory.add(full_state, full_action, reward, full_next_state, done)
    
        # Learn, if enough samples are available in memory and at interval settings
        if len(self.memory) > BATCH_SIZE:
            if self.iter % LEARN_EACH == 0:
                for _ in range(LEARNING_STEPS):
                    experiences = self.memory.sample()
                    for i in range(self.maddpg_agent):
                        self.learn(experiences, GAMMA, i)
                        


    def learn(self, experiences, gamma, agent_num):
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
        actor = self.maddpg_agent[agent_num]
        
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_act(next_states)
        Q_targets_next = actor.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards[agent_num] + (gamma * Q_targets_next * (1 - dones[agent_num]))
        # Compute critic loss
        Q_expected = actor.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        actor.critic_optimizer.zero_grad()
        critic_loss.backward()
        # As suggested from Udacity add grad clipping to the critic
        torch.nn.utils.clip_grad_norm_(actor.critic_local.parameters(), 1.0)
        actor.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # get actions of all agents        
        # combine all the actions and observations for input to critic
        actors_actions = self.local_act(states) 
        # Compute actor loss
        actor_loss = -actor.critic_local(states, actors_actions).mean()
        # Minimize the loss
        actor.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        actor.soft_update(actor.critic_local, actor.critic_target, TAU)
        actor.soft_update(actor.actor_local, actor.actor_target, TAU)    
                

            
            




