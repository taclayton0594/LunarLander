import sys
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.logger import logging
from src.components.ann_model import DoubleQLearnerANN
from src.components.replay_buffer import ReplayBuffer

'''
This main class for the RL Double Q-Learner. On initialization, 4 neural networks will be created, 2 for 2 nets used in double Q
and then another 2 for target nets for each of the previous nets that will be frozen for a tunable amount of time.
This freezing of target nets allows for more stable training.
'''
class DoubleQLearner():
    def __init__(self,num_layers,neurons,num_inputs=8,loss=nn.MSELoss,learn_rate=0.0001,
                num_actions=4,buf_size=2048,batch_size=32,alpha=0.01,gamma=0.99,eps=0):
        self.Q_a = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,learn_rate)
        self.Q_b = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,learn_rate)
        self.Q_a_target = self.Q_a
        self.Q_b_target = self.Q_b
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buf_size,batch_size)
        logging.info(f"Double Q-Learner has been created.")

    def getBestAction(self,Q):
        eps_check = np.random.random()
            
        if eps_check <= self.eps:
            rand_a = np.random.randint(0,self.num_actions,self.batch_size)
            a = Q[:][rand_a]
        else:
            a = np.argmax(Q,axis=0)
            
        return a

    def get_targets(self,update_var,batch):
        try:
            # use input value to determine which net to use for target calcs
            if update_var < 0.5:
                Q_1 = self.Q_a_target
                Q_2 = self.Q_b_target
            else:
                Q_1 = self.Q_b_target
                Q_2 = self.Q_a_target

            # Get the model outputs for each batch sample
            s_1_mat = batch[:][0]
            s_2_mat = batch[:][1]
            Q_1_preds = Q_1(s_1_mat) # Predictions using state 1 (previous state)
            Q_2_preds = Q_2(s_2_mat) # Predictions for state 2 (next state)

            # Get best/random actions based on epsilon greedy policy
            a_1 = self.getBestAction(Q_1_preds)
            a_2 = self.getBestAction(Q_2_preds)

            # Extract other useful info from batch data
            rews = batch[:][3]
            done = batch[:][4] #boolean array specifying whether the state was a terminal one

            # Calculate targets for each batch sampe
            # NOTE: sample in batch = x_1,x_2,a_1,reward,buf_done (rows)
            targets = Q_1_preds[:][a_1] # initialize equal to outputs and then add second term

            # Add second term to targets
            targets[not done] += self.alpha * (rews[done] + self.gamma * Q_2_preds[:][a_2]
                                               - targets)
            
            return targets
        
        except Exception as e:
            raise CustomException(e,sys)

        