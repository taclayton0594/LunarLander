import numpy as np
import sys
from src.exception import CustomException
from src.logger import logging

'''
Replay buffer that can store a specified number of samples and create minibatch samples for training.
'''
class ReplayBuffer:
    def __init__(self,size,batch_size):
        self.data = np.zeros(size, dtype=object)
        self.size = 0
        self.pointer = 0
        self.max_size = size
        self.batch_size = batch_size
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
            # Create a minibatch array that will contains the minibatch
            inds = np.random.choice(self.size,size=self.batch_size) 

            #print(f"replay sample = {self.data[inds][0][0]}")
            
            return self.data[inds][:]
        except Exception as e:
            logging.info(f"Error occurred during replay buffer sampling.")
            raise CustomException(e,sys)