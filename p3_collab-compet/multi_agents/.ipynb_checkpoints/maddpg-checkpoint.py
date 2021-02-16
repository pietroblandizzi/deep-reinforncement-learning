# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
import numpy as np

from actors.ddpg_actor import Agent
from utils.utilities import soft_update, transpose_to_tensor, transpose_list

import torch
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = 'cpu'


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters 

class MADDPG:
    def __init__(self, state_size, action_size, random_seed, num_agents):
        super(MADDPG, self).__init__()

        # Generate a num_agents amount of players
        self.maddpg_agent = [Agent(state_size, action_size, random_seed, num_agents) for _ in range(num_agents)]
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [maddpg_agent.actor_local for agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [agent.actor_target for agent in self.maddpg_agent]
        return target_actors
    
    def save(self, agent_ckp_prefix, critic_ckp_prefix):
        for i, agent in enumerate(self.maddpg_agent):
            torch.save(agent.actor.state_dict(), agent_ckp_prefix +'_'+ i + '_ckpt_local.pth')                 # save local actor
            torch.save(agent.actor_target.state_dict(), agent_ckp_prefix+'_'+ i + '_ckpt_target.pth')               # save target actor 
            torch.save(agent.critic.state_dict(), critic_ckp_prefix+'_'+ i + '_ckpt_local.pth')               # save local critic
            torch.save(agent.critic_target.state_dict(), critic_ckp_prefix +'_'+ i + '_ckpt_target.pth')   

    
    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [agent.target_act(obs) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, action, reward, next_obs, done = map(transpose_to_tensor, samples) # get data

        # full versions of obs and actions are needed for the critics
        obs_full = torch.cat(obs, 1)
        next_obs_full = torch.cat(next_obs, 1)
        action_full = torch.cat(action, 1)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        q_next = agent.critic_target(next_obs_full, target_actions).detach()
                
        y = reward[agent_number].unsqueeze(-1) + (GAMMA * q_next * (1 - done[agent_number].unsqueeze(-1)))
        q = agent.critic(obs_full, action_full)

        loss = torch.nn.MSELoss()
        critic_loss = loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(obs_full, q_input).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()


    def update_targets(self):
        """soft update targets"""
        for agent in self.maddpg_agent:
            soft_update(agent.actor_target, agent.actor, TAU)
            soft_update(agent.critic_target, agent.critic, TAU)
            
            
            






  