import sys
import numpy as np
import gym
from torch import nn
from src.exception import CustomException
from src.logger import logging
from src.components.double_q_learner import DoubleQLearner

class LunarLander():
    def __init__(self,alpha=0.01,alpha_decay=0.99,alpha_min=1e-6,gamma=0.99,eps=1,eps_decay=0.992,
                 buf_size=2048,min_buf_size=256,batch_size=32,eps_min=0.01,num_states=8,
                 num_actions=4):
        self.alpha = alpha
        self.alpha_init = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.eps = eps
        self.eps_init = eps
        self.eps_decay = eps_decay
        self.buf_size = buf_size
        self.min_buf_size = min_buf_size
        self.batch_size = batch_size
        self.eps_min = eps_min        
        self.num_states = num_states
        self.num_actions = num_actions
        self.reward = 0
        self.eps_step_count = 0
        self.tot_step_count = 0
        self.env = np.empty((1,),dtype=object)
        self.curr_state = np.zeros((num_states,))
        self.DoubleQLearner = np.empty((1,),dtype=object)
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
        try:
            self.DoubleQLearner = DoubleQLearner(num_layers,neurons,num_inputs,loss,self.num_actions,self.buf_size,
                                                 self.batch_size,self.alpha,self.alpha_decay,self.alpha_min,self.gamma)
        except Exception as e:
            raise CustomException(e,sys)

        return self.DoubleQLearner

    def CreateEnvironment(self,seed=222980):
        try:
            # initialize environment
            self.env = gym.make('LunarLander-v2',render_mode="human")
            self.env.action_space.seed(seed)            

            # get info on environment and seed
            self.curr_state, _ = self.env.reset(seed=seed, options={})
            self.curr_state = np.squeeze(self.curr_state)

            logging.info(f"Lunar Lander environment has been created.")

        except Exception as e:
            raise CustomException(e,sys)
        
    def ResetEnvironment(self):
        self.curr_state = self.env.reset()
        self.reward = 0
        self.eps_step_count = 0
        
    def EnvironmentStep(self,action):
        try:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store new experience
            experience = self.curr_state, next_state, action, reward, done
            self.DoubleQLearner.replay_buffer.store(experience)

            # Update state
            self.curr_state = np.reshape(next_state,(1,8))
            self.eps_step_count = self.eps_step_count + 1
            self.tot_step_count = self.tot_step_count + 1

            return experience
        
        except Exception as e:
            raise CustomException(e,sys)

    def UpdateEpsilon(self):
        eps_new = self.eps * self.eps_decay

        if eps_new > self.eps_min:
            self.eps = eps_new
        else:
            self.eps = self.eps_min

            logging.info("Epsilon decay factor has been decremented.")
        