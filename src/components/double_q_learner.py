import sys
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.logger import logging
from src.components.ann_model import DoubleQLearnerANN,device
from src.components.replay_buffer import ReplayBuffer

'''
This main class for the RL Double Q-Learner. On initialization, 4 neural networks will be created, 2 for 2 nets used in double Q
and then another 2 for target nets for each of the previous nets that will be frozen for a tunable amount of time.
This freezing of target nets allows for more stable training.
'''
class DoubleQLearner():
    def __init__(self,num_layers,neurons,num_inputs=8,num_outputs=4,loss=nn.MSELoss,learn_rate=0.0001,
                size=2048,batch_size=32,alpha=0.01,gamma=0.99):
        self.Q_a = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_outputs,loss,learn_rate)
        self.Q_b = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_outputs,loss,learn_rate)
        self.Q_a_target = self.Q_a
        self.Q_b_target = self.Q_b
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer(size,batch_size)
        logging.info(f"Double Q-Learner has been created.")

    def get_targets(self,update_var):
        try:
            # use input value to determine which net to use for target calcs
            if update_var < 0.5:
                Q1 = self.Q_a
                Q2 = self.Q_b
            else:
                Q1 = self.Q_b
                Q2 = self.Q_a



        except Exception as e:
            raise CustomException(e,sys)

        