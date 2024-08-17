import numpy as np
import torch
import sys
from src.exception import CustomException
from src.logger import logging

'''
Replay buffer that can store a specified number of samples and create minibatch samples for training.
'''
class ReplayBuffer:
    def __init__(self,size,batch_size,num_inputs=8):
        self.data = np.zeros(size, dtype=object)
        self.size = 0
        self.pointer = 0
        self.max_size = size
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        logging.info(f"Created new replay buffer of max size: {size} and batch size: {batch_size}.")
            
    def store(self, experience):
        try:
            # Append data frame
            self.data[self.pointer] = experience
            
            # Update pointer/size
            if self.pointer == self.max_size-1:
                logging.info(f"Moving pointer to beginning of replay buffer.")
                self.pointer = 0
            else:
                self.pointer += 1
                
            if self.size != self.max_size:
                self.size += 1
        except Exception as e:
            raise CustomException(e,sys)

    def sample(self):
        try:
            # Initialize batch output variables
            states = torch.zeros(self.batch_size, self.num_inputs)
            next_states = torch.zeros(self.batch_size, self.num_inputs)
            actions = torch.zeros(self.batch_size)
            rewards = torch.zeros(self.batch_size)
            done_bools = torch.zeros(self.batch_size,dtype=bool)

            # Get random batch sample indeces
            inds = np.random.choice(self.size,size=self.batch_size) 

            # populate minibatch
            for i in range(self.batch_size):
                curr_ind = inds[i]
                print(f"test={np.shape(self.data[curr_ind][0])}")
                print(f"test_2={self.data[curr_ind][0]}")
                print(np.array(self.data[curr_ind][0],dtype=float))
                states[i] = torch.from_numpy(np.array(self.data[curr_ind][0],dtype=float)).float()
                next_states[i] = torch.from_numpy(np.array(self.data[curr_ind][1],dtype=float)).float()
                actions[i] = self.data[curr_ind][2]
                rewards[i] = self.data[curr_ind][3]
                done_bools[i] = self.data[curr_ind][4]

            '''
            print(f"states batch = {states}")
            print(f"next_states batch = {next_states}")
            print(f"actions batch = {actions}")
            print(f"rewards batch = {rewards}")
            print(f"done_bools batch = {done_bools}")
            '''

            return states,next_states,actions,rewards,done_bools
        except Exception as e:
            logging.info(f"Error occurred during replay buffer sampling.")
            raise CustomException(e,sys)