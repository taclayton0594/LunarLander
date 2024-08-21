import sys
import numpy as np
import torch
from torch import nn
from src.exception import CustomException
from src.logger import logging
from src.components.ann_model import DoubleQLearnerANN
from src.components.replay_buffer import ReplayBuffer
from src.components.custom_data import LunarLanderDataset

'''
This main class for the RL Double Q-Learner. On initialization, 4 neural networks will be created, 2 for 2 nets used in double Q
and then another 2 for target nets for each of the previous nets that will be frozen for a tunable amount of time.
This freezing of target nets allows for more stable training.
'''
class DoubleQLearner():
    def __init__(self,num_layers,neurons,num_inputs=8,loss=nn.MSELoss(),learn_rate=0.0001,
                num_actions=4,buf_size=2048,batch_size=32,alpha=0.01,gamma=0.99,eps=0):
        self.Q_a_obj = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,learn_rate)
        self.Q_b_obj = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,learn_rate)
        self.Q_a = self.Q_a_obj.ANN_relu
        self.Q_b = self.Q_b_obj.ANN_relu
        self.Q_a_target = self.Q_a
        self.Q_b_target = self.Q_b
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buf_size,batch_size)
        logging.info(f"Double Q-Learner has been created.")

    def __str__(self):
        n1='\n'
        return (
            f'Double Q-Learner with the following parameters:{n1}'
            f'alpha = {self.alpha}{n1}'
            f'gamma = {self.gamma}{n1}'
            f'eps = {self.eps}{n1}'
            f'batch_size = {self.batch_size}{n1}'
            )
    
    def getBestAction(self,Q):
        a = torch.argmax(Q)
            
        return a

    def getBestActionEps(self,Q):
        eps_check = np.random.random()
            
        if eps_check <= self.eps:
            a = np.random.randint(0,self.num_actions)
        else:
            a = np.argmax(Q)
            
        return a
    
    def updateTargetANNs(self):
        self.Q_a = self.Q_a_target
        self.Q_b = self.Q_b_target

    def get_targets(self,update_var,states,next_states,actions,rewards,done_bools):
        try:
            # use input value to determine which net to use for target calcs
            if update_var < 0.5:
                Q_1 = self.Q_a_target
                Q_2 = self.Q_b_target
            else:
                Q_1 = self.Q_b_target
                Q_2 = self.Q_a_target

            # Get the model outputs for each batch sample
            s_1_mat = states
            s_2_mat = next_states
            Q_1_preds = Q_1(s_1_mat) # Predictions using state 1 (previous state)
            Q_2_preds = Q_2(s_2_mat) # Predictions for state 2 (next state)

            # Get best actions 
            a_1 = torch.argmax(Q_1_preds,dim=1)
            a_2 = torch.argmax(Q_2_preds,dim=1)

            # Extract other useful info from batch data
            rews = torch.unsqueeze(rewards,1)
            done = torch.unsqueeze(done_bools,1) #boolean array specifying whether the state was a terminal one

            # Calculate targets for each batch sampe
            # NOTE: sample in batch = x_1,x_2,a_1,reward,buf_done (rows)
            targets = Q_1_preds # initialize equal to outputs and then add second term

            # Add second term to targets
            done_inds = done.squeeze()
            not_done_inds = torch.logical_not(done).squeeze()
            updates = self.alpha * (rews + self.gamma * Q_2_preds[:].gather(0,torch.unsqueeze(a_1,1))
                                               - targets[:].gather(0,torch.unsqueeze(a_1,1)))
            targets[not_done_inds,a_1[not_done_inds]]  = torch.squeeze(targets[not_done_inds].gather(0,torch.unsqueeze(a_1[not_done_inds],1)) + updates[not_done_inds])
            targets[done_inds,a_1[done_inds]] = torch.squeeze(targets[done_inds].gather(0,torch.unsqueeze(a_1[done_inds],1)) + self.alpha * rews[done_inds])
            
            return targets,a_1,updates
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_ANNs(self,update_var,epochs=1):
        try:
            # Get batch data for training
            states,next_states,actions,rewards,done_bools = self.replay_buffer.sample()

            # Get target matrix
            targets,_,_ = self.get_targets(update_var,states,next_states,actions,rewards,done_bools)

            # Convert data to Torch Dataset
            train_data = LunarLanderDataset(states,targets)

            # Train ANNs
            if update_var < 0.5:
                # Set the module into training mode
                self.Q_a.train()
                self.Q_a_obj.train_q_learner(train_data)
            else:
                # Set the module into training mode
                self.Q_b.train()
                self.Q_b_obj.train_q_learner(train_data)

        except Exception as e:
            raise CustomException(e,sys)

        