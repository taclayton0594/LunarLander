# Lunar Lander Problem
The problem consists of an 8-dimensional continuous state space and a discrete action space. The four discrete actions available are: do nothing, fire the left orientation engine, fire the main engine, fire the right orientation engine. The landing pad is always at coordinates (0,0). Coordinates consist of the first two numbers in the state vector. The total reward for moving from the top of the screen to the landing pad ranges from 100 - 140 points varying on the lander placement on the pad. If the lander moves away from the landing pad it is penalized the amount of reward that would be gained by moving towards the pad. An episode finishes if the lander crashes or comes to rest, receiving an additional -100 or +100 points respectively. Each leg ground contact is worth +10 points. Firing the main engine incurs a -0.3 point penalty for each occurrence. Landing outside of the landing pad is possible. Fuel is infinite, so, an agent could learn to fly and land on its first attempt. The problem is considered solved when achieving a score of 200 points or higher on average over 100 consecutive runs.

## States
At each time step, a tuple of size 8 is given representing the 8 states: 
$$\left( x,y,v_x,v_y,\theta,v_{\theta},leg_L,leg_R \right)$$

States in respective order:
- *x-coordinate* 
- *y-coordinate*
- *horizontal velocity* 
$$\left( v_x \right)$$
- *vertical velocity*
$$\left( v_y \right)$$
- *angle of lander with respect to verical access*
- *angular velocity of the lander*
- *boolean for if left leg is touching ground*
- *boolean for if right leg is touching ground*

## Rewards
Reward for moving from the top of the screen to the landing pad and coming to rest is about 100-140 points. If the lander moves away from the landing pad, it loses reward. If the lander crashes, it receives an additional -100 points. If it comes to rest, it receives an additional +100 points. Each leg with ground contact is +10 points. Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame. Solved is 200 points.

## RL Agent
For the reinforcement learning (RL) agent, I went with **Double Q-Learning**, a version of Q-Learning. Being a "model-free" algorithm makes it suitable for solving the Lunar Lander problem. A model-free algorithm does not require additional information about the environment, for example the approximate physics models of firing one of the engines, which can lead to longer learning times, while also reducing the effort needed to estimate the dynamics of a given environment.

The goal of any RL agent is to determine optimal state-action pairs: **$$\left(Q: S x A -> \U+211D \right)$$**
Due to the continuous state space (i.e. infinite state space), Deep Q-Learning employs the use of a neural network to approximate this state-action mapping. There is the potential to utilize methods that discretize the state space to make the state space finite; however, I did not consider these methodologies in my solution. In the future, I hope to continue this work to compare other RL agents such as Deterministic Policy Gradient (DDPG), Proximal Policy Optimization (PPO), & actor-critic algorithms to say a few. 

Below is a screenshot of the Double Q-Learning algorithm. Note the use of a target network in this algorithm, which is the difference between Double Q and simple Deep Q-Learning (DQN - Deep Q Network). 
<p align="center">
    <img src="./results_figures/Double_Q_Learning.png">
<p>
The purpose of the target network is to utilize one function to evaluate the best action from the current state and a second, target network, to approximate the best next action from a following state, creating an off-polocy agent that reduces overestimation bias over simple DQN. This target network is frozen and slowly updated, or weights copied, from the evaluation network every N episodes or training steps, which can stabilize training in highly stochastic environment. Again, in the future, I hope to compare RL agents such as standard DQN with the results I obtained here, as well as improving the learning speed of my Double Q Agent.

## Hyperparameter Tuning and Results
There are many hyperparameters to tune when using Deep Q-Learning. This can often make it challenging to find a successful solution due to the stochastic environments agents are often placed in, which leads to small changes of key parameters having large impacts on learning.  To show the impact of a few of tuning parameters, I selected a set of parameters that successfully solved the Lunar Lander problem and then changed a single parameter at a time to see the effect. Below are the hyperparameters I used as a reference:

- **Number of neurons in first hidden layer - 64**
- **Number of neurons in second hidden layer - 32**
- **Learning rate ($$\alpha$$) - 0.00015**
- **Learning rate decay rate - 1 (no decay)**
- **$$\epsilon$$ Decay Rate - 0.999**
- **Buffer size - 100000**
- **Batch size - 64**
- **Target network update steps - 4**
- **Evaluation network training steps - 4**

In order to find the best possible set of parameters, I would need to try an exhaustive combination of parameters that would not be feasible given local computing resources. Therefore, these results cannot give definitive conclusions on whether a larger batch size is better for learning for example. It is only meant to demonstrate the importance of dedicating time to hyperparameter tuning. 

The two images below show the impact when making small steps in the learning rate and epsilon decay rate. We see that a higher learning rate (0.001) shows quicker learning initially, before slowing down and not reaching the goal. On the other hand, a learning rate that is too low (0.0001) might not allow an agent to quickly find an optimal policy. This is often caused by the agent getting stuck hovering above the platform and not learning the maximal episode reward obtained with a successful landing. This phenomena is also very important when thinking about the best epsilon decay rate. 
<p align="center" float="left">
    <img src="./results_figures/alpha.png" width="49%" />
    <img src="./results_figures/eps_decay.png" width="49%" />
</p>
The epsilon decay rate determines how much exploration the learner performs to obtain the optimal policy. We see with a low decay rate (i.e. reduce the number of random steps faster, less exploration) of 0.99 that the agent's average reward has a steeper slope initially before falling off around 1500 episodes. Again, this is due to the agent getting stuck hovering. Increasing the decay rate value is often the best solution to get around this problem at the cost of additional learning time. 
<p></p>

The next two plots show the impact of changing the batch size and number of neurons in the first hidden layer. All my networks used 2 hidden layers, but I only changed the size of the first layer to reduce the number of experiments to run. We notice similar behavior to learning rate where a batch size that is too small (8) can learn slower and not reach the goal quickly. However, a batch size that is too large may reduce the impact of singlular data points (i.e. high reward actions like landing) that could cause slower in some cases.
<p align="center" float="left">
    <img src="./results_figures/batch_size.png" width="49%" />
    <img src="./results_figures/layer_1_neurons.png" width="49%" />
</p>
The image showing the results with different neuron counts seems to show that having more neurons can potentially have a positive impact. However, there is not a large difference between 64 and 128 neurons in this evaluation so more experiments are required. 
<p></p>

In order to improve the conclusions that could be derived from these results, I would need to do a more exhaustive DOE of hyperparameters to find the best combinations to solve the problem as quickly as possible. I would also need to try different seeds in order to show consistency of the results when considering random noise factors. The main conclusion that can be drawn from these results is that Deep Q-Learning is a powerful tool that can learn complex models with little to no information on the environment and actions that can be taken. Additionally, due to various tuning parameters to be selected, there is a need to spend adequate time experimenting with different combinations of parameters in order to find the best policies. 

Future extentions of this work would also include experimenting with methods like a prioritized replay buffer, decaying learning rate, and other popular methods. 