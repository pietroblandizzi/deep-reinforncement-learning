import numpy as np

from actors.td3_actor import Agent
from utils.utilities import soft_update, transpose_to_tensor, transpose_list

import torch
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters 
POLICY_FREQ = 1        # Delay steps to update the policy

class MATD3:
    def __init__(self, state_size, action_size, random_seed, num_agents, policy_noise=0.1):
        super(MATD3, self).__init__()

        # Generate a num_agents amount of players
        self.matd3_agent = [Agent(state_size, action_size, random_seed, num_agents, policy_noise) for _ in range(num_agents)]
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.policy_noise = policy_noise
        
        self.iteration = np.zeros(num_agents)

    def get_actors(self):
        """get actors of all the agents in the MATD3 object"""
        actors = [agent.actor for agent in self.matd3_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MATD3 object"""
        target_actors = [agent.actor_target for agent in self.matd3_agent]
        return target_actors
    
    def save(self, agent_ckp_prefix, critic_ckp_prefix):
        for i, agent in enumerate(self.matd3_agent):
            torch.save(agent.actor.state_dict(), agent_ckp_prefix +'_'+ str(i) + '_ckpt_local.pth')                 # save local actor
            torch.save(agent.actor_target.state_dict(), agent_ckp_prefix+'_'+ str(i) + '_ckpt_target.pth')               # save target actor 
            torch.save(agent.critic.state_dict(), critic_ckp_prefix+'_'+ str(i) + '_ckpt_local.pth')               # save local critic
            torch.save(agent.critic_target.state_dict(), critic_ckp_prefix +'_'+ str(i) + '_ckpt_target.pth')   

    def load(self, agent_ckp_prefix, critic_ckp_prefix):
        for i, agent in enumerate(self.matd3_agent):
                     
            checkpoint_local_actor =  torch.load(agent_ckp_prefix +'_'+ str(i) + '_ckpt_local.pth')              
            checkpoint_target_actor = torch.load(agent_ckp_prefix+'_'+ str(i) + '_ckpt_target.pth')              
            checkpoint_local_critic = torch.load(critic_ckp_prefix+'_'+ str(i) + '_ckpt_local.pth')              
            checkpoint_target_critic = torch.load(critic_ckp_prefix +'_'+ str(i) + '_ckpt_target.pth')  
    
        
            agent.actor.load_state_dict(checkpoint_local_actor)
            agent.actor_target.load_state_dict(checkpoint_target_actor)
            agent.critic.load_state_dict(checkpoint_local_critic)
            agent.critic_target.load_state_dict(checkpoint_target_critic)

    
    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MATD3 object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.matd3_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MATD3 object """
        target_actions = [agent.target_act(obs) for agent, obs in zip(self.matd3_agent, obs_all_agents)]
        return target_actions
    
    def reset(self):
        for a in self.matd3_agent:
            a.reset()

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        obs, action, reward, next_obs, done = map(transpose_to_tensor, samples) # get data

        # full versions of obs and actions are needed for the critics
        obs_full = torch.cat(obs, 1)
        next_obs_full = torch.cat(next_obs, 1)
        action_full = torch.cat(action, 1)
        
        agent = self.matd3_agent[agent_number]
        agent.critic_optimizer.zero_grad()
        
        self.iteration[agent_number] +=1
        
        with torch.no_grad():
            target_actions = self.target_act(next_obs)
            target_actions = torch.cat(target_actions, dim=1)
            noise = (torch.randn_like(target_actions) * self.policy_noise).clamp(-1, 1)   
            target_actions = torch.clamp(target_actions + noise,-1,1)
            
        
            q1_next, q2_next = agent.critic_target(next_obs_full, target_actions)
            q_next = torch.min(q1_next, q2_next)
                
            y = reward[agent_number].unsqueeze(-1) + (GAMMA * q_next * (1 - done[agent_number].unsqueeze(-1)))
                                                                                                                                            
        q1, q2 = agent.critic(obs_full, action_full)

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()
                
        #### Delay update actor
        if self.iteration[agent_number] % POLICY_FREQ == 0:
            #update actor network using policy gradient
            agent.actor_optimizer.zero_grad()
            # make input to agent
            # detach the other agents to save computation
            # saves some time for computing derivative
            q_input = [ self.matd3_agent[i].actor(ob) if i == agent_number \
                       else self.matd3_agent[i].actor(ob).detach()
                       for i, ob in enumerate(obs) ]
                
            q_input = torch.cat(q_input, dim=1)
        
            # get the policy gradient 
            actor_loss = -agent.critic.Q1(obs_full, q_input).mean()
            actor_loss.backward()
            agent.actor_optimizer.step()
        
            # ----------------------- update target networks ----------------------- #
            agent.soft_update(agent.critic, agent.critic_target, TAU)
            agent.soft_update(agent.actor, agent.actor_target, TAU)





  