import sys
import numpy as np
import gym
from torch import nn
from src.exception import CustomException
from src.logger import logging
from src.components.double_q_learner import DoubleQLearner

class LunarLander():
    def __init__(self,alpha=0.01,alpha_decay=0.99,alpha_min=1e-4,learn_rate=0.001,learn_rate_decay=1,gamma=0.99,eps=1,eps_decay=0.992,
                 buf_size=2048,min_buf_size=256,batch_size=32,learn_rate_min=1e-6,eps_min=0.0001,num_states=8,
                 num_actions=4):
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.learn_rate = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.buf_size = buf_size
        self.min_buf_size = min_buf_size
        self.batch_size = batch_size
        self.learn_rate_min = learn_rate_min
        self.eps_min = eps_min        
        self.num_states = num_states
        self.num_actions = num_actions
        self.reward = 0
        self.step_count = 0
        self.env = np.empty(dtype=object)
        self.curr_state = np.zeros((num_states,))
        self.DoubleQLearner = np.empty(dtype=object)
        logging.info(f"New Lunar Lander object has been created.")

    def __str__(self):
        n1='\n'
        return (
            f'Lunar Lander object with the following parameters:{n1}'
            f'# of states: {self.num_states}{n1}'
            f'# of actions: {self.num_actions}{n1}'
            f'alpha: {self.alpha}{n1}'
            f'gamma: {self.gamma}{n1}'
            f'eps: {self.eps}{n1}'
            f'batch_size: {self.batch_size}{n1}'
            )
    
    def CreateQLearner(self,num_layers,neurons,num_inputs=8,num_outputs=4,loss=nn.MSELoss):
        self.DoubleQLearner = DoubleQLearner(num_layers,neurons,num_inputs,loss,self.learn_rate,
                            self.num_actions,self.buf_size,self.batch_size,self.alpha,self.gamma,self.eps)
        
        return self.DoubleQLearner

    def CreateEnvironment(self,seed=222980):
        try:
            # initialize environment
            self.env = gym.make('LunarLander-v2',render_mode="human")
            self.env.action_space.seed(seed)

            # get info on environment and seed
            self.curr_state, _ = self.env.reset(seed=seed, options={})

            logging.info(f"Lunar Lander environment has been created.")

        except Exception as e:
            raise CustomException(e,sys)
        
    def EnvironmentStep(self,action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        # Store new experience
        experience = self.curr_state, next_state, action, reward, done
        self.DoubleQLearner.replay_buffer.store(experience)

        # Update state
        self.curr_state = next_state
        
    def UpdateAlpha(self):
        alpha_new = self.alpha * self.alpha_decay

        if alpha_new > self.alpha_min:
            self.alpha = alpha_new

        logging.info("Alpha learning rate has been decremented.")

    def UpdateANNLearnRate(self):
        learn_rate_new = self.learn_rate * self.learn_rate_decay

        if learn_rate_new > self.alpha_min:
            self.learn_rate = learn_rate_new

        logging.info("ANN learning rate has been decremented.")

    def UpdateEpsilon(self):
        eps_new = self.eps * self.eps_decay

        if eps_new > self.eps_min:
            self.eps = eps_new

        logging.info("Epsilon decay factor has been decremented.")
        