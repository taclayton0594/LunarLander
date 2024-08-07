import os
import sys
import numpy as np
from dataclasses import dataclass
from torch import nn
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,setup_lunar_lander_grid
from src.components.lunar_lander import LunarLander

@dataclass
class RLModelTrainerConfig:
    best_model_file_path = os.path.join('artifacts',"best_model.pkl")

class RLModelTrainer:
    def __init__(self):
        self.model_trainer_config = RLModelTrainerConfig
        self.LunarLander = LunarLander()
        self.DOE_num = 0

    def set_model_hyperparameters():
        params = {
            "layer_1_neurons": [32,64,128],
            "layer_2_neurons": [32,64,128],
            "layer_3_neurons": [0,32,64],
            "alpha": [0.001,0.01,0.1],
            "alpha_decay": [0.98,0.99,0.992],
            "learn_rate": [0.0001,0.001,0.01], #will not decay NN learning rate for now to reduce DOE
            "eps_decay": [0.98,0.992], # epsilon will always start at 1
            "buf_size": [2048,8192], # minimum buffer size will always be 2*batch_size
            "batch_size": [16,32,64,128],
            "target_update_epochs": [256,1024],
            "batch_update_epochs": [4,32]
        }

        return params
    
    def get_hyperparameter_grid(self):
        self.grid,self.num_experiments = setup_lunar_lander_grid(self.set_model_hyperparameters())

    def initialize_Q_Learner(self,num_layers,neurons,num_inputs=8,num_outputs=4,loss=nn.MSELoss):
        self.LunarLander.CreateQLearner(num_layers,neurons)
    
    def set_parameters(self):
        # Get the current hyperparameter values
        curr_params = self.grid[self.DOE_num]
        layer_1_neurons = curr_params["layer_1_neurons"]
        layer_2_neurons = curr_params["layer_2_neurons"]
        layer_3_neurons = curr_params["layer_3_neurons"]
        neurons = [layer_1_neurons,layer_2_neurons,layer_3_neurons]
        alpha = curr_params["alpha"]
        alpha_decay = curr_params["alpha_decay"]
        learn_rate = curr_params["learn_rate"]
        eps_decay = curr_params["eps_decay"]
        buf_size = curr_params["buf_size"]
        batch_size = curr_params["batch_size"]
        self.target_update_epochs = curr_params["target_update_epochs"]

        # Set corresponding values in LunarLander object (NOTE: other parameters are constant for each experiment)
        self.LunarLander.alpha = alpha
        self.LunarLander.alpha_decay = alpha_decay
        self.LunarLander.learn_rate = learn_rate
        self.LunarLander.eps_decay = eps_decay
        self.LunarLander.buf_size = buf_size
        self.LunarLander.batch_size = batch_size

        # Get the number of layers
        if layer_3_neurons == 0:
            num_layers = 2
        else: 
            num_layers = 3

        # Create Double Q learner
        self.initialize_Q_Learner(num_layers,neurons)

    def start_RL_training(self):
        # Initialize the grid of hyperparameters for each experiment
        self.get_hyperparameter_grid()

        # Loop through and run each experiment
        for i in range(self.num_experiments):
            # Set necessary class elements for the current experiment
            self.set_parameters()

            # Initalize the gym environment
            self.LunarLander.CreateEnvironment()

            # Run until the experiment ends or max steps hit
            #while True:
            #   self.LunarLander.step(action)
             
                  
            # Update learning rates
            self.LunarLander.UpdateAlpha()
            self.LunarLander.UpdateANNLearnRate()
            self.LunarLander.UpdateEpsilon()

            # Update experiment number at end of experiment
            #self.DOE_num = self.DOE_num+1 