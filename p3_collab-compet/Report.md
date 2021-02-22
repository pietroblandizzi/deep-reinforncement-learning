[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://github.com/pietroblandizzi/deep-reinforncement-learning/blob/main/p3_collab-compet/DDPG_1 "DDPG 1"
[image3]: https://github.com/pietroblandizzi/deep-reinforncement-learning/blob/main/p3_collab-compet/DDPG_2 "DDPG 2"
[image4]: https://github.com/pietroblandizzi/deep-reinforncement-learning/blob/main/p3_collab-compet/TD3 "TD3"




# Project 3: Multi agent system

## Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## What is implemented - step by step
Here i describe the Multi agent DDPG i implemented along with the pain i went through to reach a state where it learn something and my fail in TD3

### DDPG
I started with implementing Multi agent ddpg following the code in the laboratory and the DDPG i implemented in the 
[Continuous control project](https://github.com/pietroblandizzi/deep-reinforncement-learning/tree/main/p2_continuous-control)

It did not seem to learn at all. 

Network architecture: really similar to the paper:
Both actor and critic have 2 hidden layers with 400 and 300 units each.
The activation is a Relu for all the layers except the last.
The last layer of the actor has a tanh to squeeze the output between -1 and 1
while the critic has not activation.

Batch normalization is applied on the first layer.


Initial parameters:

| Param                       | Value         |
|-----------------------------|--------------:|
| LR Actor                    |    1e-3       |
| LR Critic                   |    1e-3       |
| Batch size                  | 128           |
| Replay buffer size          |    1e6        |
| gamma                       |    0.99       |  
| tau                         |    1e-3       |
| Noise sigma                 | 0.2           |
| Start noise                 | 1.0           |
| Noise decay                 |   0.998       |
| Num episodes                | 5000          |


Here i describe all the changes i tried until i reached a learning state.
Not to mention i re implemented the code 3 times looking for bugs

1. Removed batch normalization
2. Add droput -> Did not change anything so i removed it
3. Tried different hidden layer sizes (512,256) and (256,128)
4. Increase the amount of episodes
5. Learn every 20 episodes 15 times
6. Decrease tau to 1e-2 -> I saw something so it made me think it was not learning fast enough from the noise
7. Increased sigma to 0.4
8. Increase batch size to 256
9. Added 10 episodes of random actions
10. Noise decay rate 0.9995

Finally with this configuration it learned!

| Param                       | Value         |
|-----------------------------|--------------:|
| LR Actor                    |    5e-4       |
| LR Critic                   |    1e-3       |
| Batch size                  | 256           |
| Replay buffer size          |    1e6        |
| gamma                       |    0.99       |  
| tau                         |    1e-3       |
| Noise sigma                 | 0.4           |
| Start noise                 | 1.0           |
| Noise decay                 |   0.9995      |
| Num episodes                | 10000         |
| Experience gathering        | 10            |
| Steps before noise decay    | 1000          |




The agent solved the environment in 1958 episodes (see reports/Tennid.pdf)

![DDPG][image2]


It is worth to mention the learning is not stable as we can see from anther run where the agent solved the environment in 2871 episodes

![DDPG][image3]

### TD3
I then wrote the TD3 algorithm following the author implementation (TD3)[https://github.com/sfujim/TD3/blob/master/TD3.py]
Again following my previous implementation for the [Continuous control project](https://github.com/pietroblandizzi/deep-reinforncement-learning/tree/main/p2_continuous-control)
TD3 basically brings 3 improvements to the standard DDPG.
1. We introduce a network with 2 separate streams for the Q network which leads on two separate estimates of the state-action pair
2. Add noise also to the actions used for target computation
3. We delay the update to the policy and target network so that the Q network updates more frequently.

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

It did not learn. I tried
1. Remove the delayed update
2. Different amplitude for the policy noise
3. Remove the policy noise
4. Different sigma in the OU noise
5. Different values for noise decay

It just does not go over 0.1 average score.
Without noisy targets and without delayed policy the only difference is made by the 2 critic networks which seems to have 
a huge effect on the learning.


![TD3][image4]


### Improvements

There are plenty of papers and improvements one can pick.
We could spend months tuning parameters get the best network architecture for the task.

One idea is to use PER instead of a normal replay buffer.
Another definitely good idea to make the learning more stable is to use N step return.
Another idea is to try different noises and decay of the noise.


It is clearly an hard task to solve and the noise play an important role in the learning of this algorithm so i suggest to explore it.



