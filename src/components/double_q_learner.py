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
                 alpha=0.0001,alpha_decay=1.0,alpha_min=1e-6,gamma=0.99):
        self.Q_a_obj = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min)
        self.Q_b_obj = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min)
        self.Q_a_obj_target = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min)
        self.Q_b_obj_target = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min)
        
        # initialize weights
        self.Q_a_obj.apply(self.Q_a_obj.initialize_weights)
        self.Q_b_obj.apply(self.Q_b_obj.initialize_weights)
        self.updateTargetANNs()
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buf_size,batch_size)
        logging.info(f"Double Q-Learner has been created.")

    def __str__(self):
        n1='\n'
        return (
            f'Double Q-Learner with the following parameters:{n1}'
            f'gamma = {self.gamma}{n1}'
            f'batch_size = {self.batch_size}{n1}'
            )
    
    def updateTargetANNs(self):
        self.Q_a_obj_target.load_state_dict(self.Q_a_obj.state_dict())
        self.Q_b_obj_target.load_state_dict(self.Q_b_obj.state_dict())

        logging.info("Target networks have been updated.")

    def get_targets(self,update_var,states,next_states,actions,rewards,done_bools):
        try:
            # use input value to determine which net to use for target calcs
            if update_var < 0.5:
                Q_1 = self.Q_a_obj
                Q_2 = self.Q_a_obj_target #self.Q_b_obj_target
            else:
                print("should never get here")
                Q_1 = self.Q_b_obj
                Q_2 = self.Q_a_obj_target

            # Get the model outputs for each batch sample
            s_1_mat = states.clone().detach()
            s_2_mat = next_states.clone().detach()
            Q_1_preds  = Q_1(s_1_mat).clone().detach()
            Q_2_preds = Q_2(s_2_mat).clone() # Predictions for state 2 (next state)

            # Get best actions 
            a_1 = actions.clone().detach()
            a_2 = torch.argmax(Q_2(s_2_mat).clone(),dim=1)

            # Get prediction            
            preds = Q_1_preds.gather(1,a_1.view(self.batch_size,1)).view(self.batch_size) # Predictions using state 1 (previous state)
            
            # Extract other useful info from batch data
            rews = rewards.view(self.batch_size,1)
            done = done_bools.view(self.batch_size,1)

            # Calculate targets for each batch sample
            targets = (rews + self.gamma * Q_2_preds[:].gather(1,a_2.view(self.batch_size,1)) * (1 - done.int())).view(self.batch_size)

            return preds,targets,a_1
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_ANNs(self,update_var,epochs=1):
        try:
            # Get batch data for training
            states,next_states,actions,rewards,done_bools = self.replay_buffer.sample()

            # Set to training mode
            if update_var < 0.5:
                self.Q_a_obj.train()
            else:
                self.Q_b_obj.train()

            # Get target matrix
            preds,targets,_ = self.get_targets(update_var,states,next_states,actions,rewards,done_bools)
            
            # Convert data to Torch Dataset
            train_data = LunarLanderDataset(preds,targets)

            # Train ANNs
            if update_var < 0.5:
                # Set the module into training mode
                self.Q_a_obj.train_q_learner(train_data,self.batch_size)
            else:
                # Set the module into training mode
                self.Q_b_obj.train_q_learner(train_data,self.batch_size)


        except Exception as e:
            raise CustomException(e,sys)

        