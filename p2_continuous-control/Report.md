[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "DDPG without noise"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "TD3 without noise"
[image3]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "TD3 with noise"




# Project 2: Continuous Control

## Introduction

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## What is implemented - step by step
Here i describe the single version of DDPG and TD3 but similar considerations can be made for the environment with 20 agents.
For each agent you can find the checkpoints in the checkpoints folder.


### DDPG
I started with a simple implementation of DDPG. I slightly modified the code from the Udacity example in:
[ddpg-bipedal](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal)
and [OpenAi baselines](https://github.com/openai/baselines/tree/master/baselines/ddpg)

And i followed the paper [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
LEARNING](https://arxiv.org/pdf/1804.00361.pdf) and some online threard for the parameters.
I started with batch normalization in all layers and noice process added to the actions.
It did not learn at all. 
I removed batch normalization and it still did not learn.
Then i applied batch normalization only on the first layer and removed the noise process from the actions.
In this way i started to get good results as shown in the report in reports/Continous_control_single_DDPG_no_noise.pdf
I also tryed to reduce slowly the noise after each episode but the learning happened slowly.
I tried to reduce the noise the more we reach the target score even with that it did not converge fast enough.

Network architecture: really similar to the paper:
Both actor and critic have 2 hidden layers with 400 and 300 units each.
The activation is a Relu for all the layers except the last.
The last layer of the actor has a tanh to squeeze the output between -1 and 1
while the critic has not activation.

We use batch normalization after the input layer

Those are the parameters used:

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
EPS_DECAY = 1e-6

LEARNING_STEPS = 10
LEARN_EACH = 20

MIN_EPS_NOISE = 0.01

The agent solved the environment in 827 episodes with an average score of 30.14

![DDPG][image1]

### TD3
I then wrote the TD3 algorithm following the author implementation (TD3)[https://github.com/sfujim/TD3/blob/master/TD3.py]
TD3 basically brings 3 improvements to the standard DDPG.
1. We introduce a network with 2 separate streams for the Q network which leads on two separate estimates of the state-action pair
2. Add noise also to the actions used for target computation
3. We delay the update to the policy and target network so that the Q network updates more frequently.

TD3 without noisy actions completed the environment much faster than DDPG and i was able to complete the training also adding noisy actions(which decay linearly) as we can see from the 2 reports 
1. reports/Continous_control_single_TD3_no_noise.pdf
2. reports/Continous_control_single_TD3_noise.pdf

There the networks are slightly more complicated:
Actor: is the same as a normal DDPG
Critic: it is made from 2 DDPG Critic networks

How do we use them?
We get the min values out of the 2 streams from the target critic and use them to compute the Q targets.
Then we get the expected values from the online critic networks stream and we sum the 2 losses.
Then we backpropagate using this sum of losses.

We also delay the update of the policy.
Already in DDPG we used a soft update. 
Here we perform the soft update each other step. This makes the learning more stable because we allow the value function to stabilize before let it guide the policy.

Parameters:

BUFFER_SIZE = int(1e6)   # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1e-3               # for soft update of target parameters
LR_ACTOR = 1e-3          # learning rate of the actor 
LR_CRITIC = 1e-3         # learning rate of the critic
WEIGHT_DECAY = 0         # L2 weight decay
EPS_DECAY = 1e-6

LEARNING_STEPS = 10
LEARN_EACH = 20

POLICY_FREQ = 2        # Delay steps to update the policy

The TD3 without noise solved the environment in 670 episodes with an average score of 30.15
![TD3_without_noise][image2]

The TD3 with noise solved the environment in 1357 episodes with an average score of 30.05
![TD3_with_noise][image3]


Note that i have included checkpoints and report of the TD3 and DDPG agents trained with the 20 agents as well.
The only difference is that they share the same replay buffer and the same noise.
The step is done 20 times(one for each agent)

### D4PG

(D4PG)[https://openreview.net/forum?id=SyZipzbCb]

I tried to implement the d4pg but it was not converging. So i still need to double check it.

### Improvements

There are plenty of papers and improvements one can pick.
We could spend months tuning parameters get the best network architecture for the task.

One idea is to use PER instead of a normal replay buffer.
Another idea is to try different noises and decay of the noise


