[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[image2]:

[image3]:
# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2.  Place the file in the PROJECT GitHub repository, in the `rainbow_navigation_project/` folder, and unzip (or decompress) the file.

3.  install environment

    1. pip install matplotlib
    2. pip install mlagents
    3. pip install numpy
    4. That should be it.

### Instructions

Follow the instructions in `Navigation.ipynb` either train or test the solution!
To test the trained model load the `checkpoint.pth` for a simple DQN or the `checkpoint_rainbow.pth`
to test the version with Dueling DDQN + PER
Note: I used the PER implementation from the Gokking Deep Reinforcement Learning book and adapted to the problem


### Results

I started with the simple implementation of the DQN provided from Udacity.
A simple NN with 2 hidden layers with (64,32) and the parameters from the book already solved the environment in
roughtly 500 episodes. So i did not see a reason to change parameters or increase the network size.

This project contain the code for the training using state
in the folder `state_obs` and the code to train from visual observation in the folder `visual_obs`.

It is implemented:

1. Standard DQN in  the file agent.py`
![DQN][image2]

2. Rainbow with:

    1. Double DQN
    2. Dueling DDQN
    3. PER

    in `agent_rainbow.py`

In the `model.py` the Standard NN and the Dueling architecture are implemented


Summary:

| Method  | Implementation | Paper |
| ------------- | ------------- | ------------- |
| DQN  | Yes| https://arxiv.org/pdf/1312.5602.pdf |
| Double DQN  | Yes| https://arxiv.org/pdf/1509.06461.pdf |
| Prioritised Experience Replay  | Used implementation from the book| https://arxiv.org/pdf/1511.05952.pdf |
| Duel DQN  | Yes | https://arxiv.org/pdf/1511.06581.pdf |
| Noisy DQN  | No | https://arxiv.org/pdf/1706.10295.pdf |
| Distributional Q-Learning  | No | https://arxiv.org/pdf/1707.06887.pdf |
| Asyncronous Learning  | No | https://arxiv.org/pdf/1602.01783.pdf |


### Notes.

I used inspiration from:
[rainbow is all you need]( https://github.com/Curt-Park/rainbow-is-all-you-need ),
[Gokking Deep Reinforcement Learning] (https://www.manning.com/books/grokking-deep-learning)
[Udacity Deep Reinforcement Learning] (https://github.com/udacity/deep-reinforcement-learning)

The code using visual observation is not tested due to lack of GPU
I tried to use google colab but so far not working.
