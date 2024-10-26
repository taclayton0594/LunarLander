import numpy as np
import torch
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.ann_model import device

'''
Replay buffer that can store a specified number of samples and create minibatch samples for training.
'''
class ReplayBuffer:
    def __init__(self,size,batch_size,num_inputs=8):
        self.data = np.zeros((size,5), dtype="object")
        self.size = 0
        self.pointer = 0
        self.max_size = size
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        # initialize batch matrices/arrays to avoide memory leak
        self.states = torch.zeros(self.batch_size, self.num_inputs,device=device)
        self.next_states = torch.zeros(self.batch_size, self.num_inputs,device=device)
        self.actions = torch.zeros(self.batch_size,device=device,dtype=torch.int64)
        self.rewards = torch.zeros(self.batch_size,device=device)
        self.done_bools = torch.zeros(self.batch_size,dtype=bool,device=device)
        logging.info(f"Created new replay buffer of max size: {size} and batch size: {batch_size}.")
            
    '''
    Store current info of Lunar Lander state. Update the size of position of the oldest sample. 
    '''
    def store(self, experience):
        try:
            # Append data frame
            self.data[self.pointer][:] = experience
            
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

    '''
    Get a random batch sample from replay buffer. 
    '''
    def sample(self):
        try:
            # Get random batch sample indeces
            inds = np.random.choice(self.size,size=self.batch_size)

            # populate minibatch
            for i in range(self.batch_size):
                curr_ind = inds[i]
                try:
                    self.states[i] = torch.from_numpy(np.array(self.data[curr_ind][0],dtype=float)).float().to(device)
                except:
                    try:
                        self.states[i] = torch.from_numpy(np.array(self.data[curr_ind][0][0],dtype=float)).float().to(device)
                    except Exception as e:
                        raise CustomException(e,sys)
                self.next_states[i] = torch.from_numpy(np.array(self.data[curr_ind][1],dtype=float)).float().to(device)
                self.actions[i] = self.data[curr_ind][2]
                self.rewards[i] = self.data[curr_ind][3]
                self.done_bools[i] = torch.tensor(self.data[curr_ind][4]).to(device)

            return self.states,self.next_states,self.actions,self.rewards,self.done_bools
        except Exception as e:
            logging.info(f"Error occurred during replay buffer sampling.")
            raise CustomException(e,sys)