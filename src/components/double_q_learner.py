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
    def __init__(self,num_layers,neurons,num_inputs=8,loss=nn.MSELoss(),num_actions=4,buf_size=50000,batch_size=32,
                 alpha=0.0001,alpha_decay=1.0,alpha_min=1e-6,gamma=0.99,eps=1.0,steps_to_update=10000):
        self.Q_a_obj = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min,steps_to_update)
        self.Q_b_obj = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min,steps_to_update)
        self.Q_a_obj_target = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min,steps_to_update)
        self.Q_b_obj_target = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min,steps_to_update)
        self.Q_a = self.Q_a_obj.ANN_relu
        self.Q_b = self.Q_b_obj.ANN_relu
        self.Q_a_target = self.Q_a_obj_target.ANN_relu
        self.Q_b_target = self.Q_b_obj_target.ANN_relu
        self.num_actions = num_actions
        self.gamma = gamma
        self.eps = eps
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buf_size,batch_size)
        logging.info(f"Double Q-Learner has been created.")

    def __str__(self):
        n1='\n'
        return (
            f'Double Q-Learner with the following parameters:{n1}'
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
        self.Q_a_target.load_state_dict(self.Q_a.state_dict())
        self.Q_b_target.load_state_dict(self.Q_b.state_dict())

        logging.info("Target networks have been updated.")

    def get_targets(self,update_var,states,next_states,actions,rewards,done_bools):
        try:
            # use input value to determine which net to use for target calcs
            if update_var < 0.5:
                Q_1 = self.Q_a
                Q_2 = self.Q_b_target
            else:
                Q_1 = self.Q_b
                Q_2 = self.Q_a_target

            # Get the model outputs for each batch sample
            s_1_mat = torch.tensor(states).clone() 
            s_2_mat = next_states
            Q_1_preds = torch.tensor(Q_1(s_1_mat)).clone() # Predictions using state 1 (previous state)
            Q_2_preds = Q_2(s_2_mat) # Predictions for state 2 (next state)

            # Get best actions 
            a_1 = actions
            a_2 = torch.argmax(torch.tensor(Q_1(s_2_mat)).clone(),dim=1)

            # Extract other useful info from batch data
            rews = rewards.view(self.batch_size,1)
            done = done_bools.view(self.batch_size,1)

            # Calculate targets for each batch sampe
            # NOTE: sample in batch = x_1,x_2,a_1,reward,buf_done (rows)
            targets = Q_1_preds # initialize equal to outputs 
        
            targets[torch.arange(self.batch_size).long(),a_1] = (rews + self.gamma * Q_2_preds[:].gather(0,a_2.view(self.batch_size,1)) * (1 - done.int())).view(self.batch_size)

            return targets,a_1
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_ANNs(self,update_var,epochs=1):
        try:
            # Get batch data for training
            states,next_states,actions,rewards,done_bools = self.replay_buffer.sample()

            # Get target matrix
            targets,_ = self.get_targets(update_var,states,next_states,actions,rewards,done_bools)

            # Convert data to Torch Dataset
            train_data = LunarLanderDataset(states,targets)

            # Train ANNs
            if update_var < 0.5:
                # Set the module into training mode
                self.Q_a_obj.train_q_learner(train_data,self.batch_size)
            else:
                # Set the module into training mode
                self.Q_b_obj.train_q_learner(train_data,self.batch_size)


        except Exception as e:
            raise CustomException(e,sys)

        