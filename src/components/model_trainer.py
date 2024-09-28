import os
import sys
import numpy as np
from dataclasses import dataclass
import torch
from torch import nn
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,setup_lunar_lander_grid
from src.components.lunar_lander import LunarLander
from src.components.ann_model import device

@dataclass
class RLModelTrainerConfig:
    best_model_file_path = os.path.join('artifacts',"best_model.pkl")

class RLModelTrainer:
    def __init__(self):
        self.model_trainer_config = RLModelTrainerConfig()
        self.LunarLander = LunarLander()
        self.max_steps = 1000
        self.max_trials = 2000 
        self.rewards = np.empty((0,),dtype=float)
        self.experiment_num = 0
        self.trial_num = 0

    def set_model_hyperparameters(self):
        params = {
            "layer_1_neurons": [32],
            "layer_2_neurons": [32],
            "layer_3_neurons": [16],
            "alpha": [0.01,0.001],
            "alpha_decay": [0.9995],
            "eps_decay": [0.993], # epsilon will always start at 1
            "buf_size": [100000], # minimum buffer size will always be 2000
            "batch_size": [64],
            "target_update_steps": [10],
            "batch_update_steps": [1]
        }

        return params
    
    def get_hyperparameter_grid(self):
        self.grid,self.num_experiments = setup_lunar_lander_grid(self.set_model_hyperparameters())

    def initialize_Q_Learner(self,num_layers,neurons,num_inputs=8,num_outputs=4,loss=nn.MSELoss):
        self.LunarLander.CreateQLearner(num_layers,neurons)
    
    def set_parameters(self):
        # Get the current hyperparameter values
        curr_params = self.grid[self.experiment_num]
        layer_1_neurons = curr_params["layer_1_neurons"]
        layer_2_neurons = curr_params["layer_2_neurons"]
        layer_3_neurons = curr_params["layer_3_neurons"]
        neurons = [layer_1_neurons,layer_2_neurons,layer_3_neurons]
        alpha = curr_params["alpha"]
        alpha_decay = curr_params["alpha_decay"]
        eps_decay = curr_params["eps_decay"]
        buf_size = curr_params["buf_size"]
        batch_size = curr_params["batch_size"]
        self.target_update_steps = curr_params["target_update_steps"]
        self.batch_update_steps = curr_params["batch_update_steps"]

        # Set corresponding values in LunarLander object (NOTE: other parameters are constant for each experiment)
        self.LunarLander.alpha = alpha
        self.LunarLander.alpha_decay = alpha_decay
        self.LunarLander.eps_decay = eps_decay
        self.LunarLander.buf_size = buf_size
        self.LunarLander.batch_size = batch_size
        self.trial_num = 1
        self.LunarLander.tot_step_count = 0 # needed when starting next hyperparamter experiment

        # Get the number of layers
        if layer_3_neurons == 0:
            num_layers = 2
        else: 
            num_layers = 3

        # Create Double Q learner
        self.initialize_Q_Learner(num_layers,neurons)

    def printPerformance(self):
        # calculate 100pt moving average of rewards
        if self.trial_num <= 100:
            mov_avg = np.sum(self.rewards) / self.trial_num
        else:
            mov_avg = np.sum(self.rewards[-100:]) / 100.0

        str_out = f"The 100 trial moving average is {mov_avg} at trial {self.trial_num}."
        print(str_out)
        logging.info(str_out)


    def start_RL_training(self):
        # Initialize the grid of hyperparameters for each experiment
        self.get_hyperparameter_grid()

        # Create array to hold rewards
        num_trials = self.num_experiments * self.max_trials

        # Initalize the gym environment
        self.LunarLander.CreateEnvironment()

        # Loop through and run each experiment
        for i in range(self.num_experiments):
            # Set necessary class elements for the current experiment
            self.set_parameters()

            # Continue running experiments until max number is exceeded
            for j in range(self.max_trials):
                # Reset the environment
                self.LunarLander.ResetEnvironment()

                # Run until the experiment ends or max steps hit
                while True:
                    # Determine which matrix to update
                    update_var = np.random.random()

                    # Get Q values array for state by which model will be updated (chosen at random)
                    if update_var < 0.5:
                        Q = self.LunarLander.DoubleQLearner.Q_a(torch.tensor(self.LunarLander.curr_state[0]).to(device))
                    else:
                        Q = self.LunarLander.DoubleQLearner.Q_b(torch.tensor(self.LunarLander.curr_state[0]).to(device))

                    # Get the next action (using and Epsilon Greedy policy)
                    a = self.LunarLander.DoubleQLearner.getBestActionEps(Q)

                    # Take a step and store relevant information
                    _, _, _, reward, done = self.LunarLander.EnvironmentStep(a)

                    # Add reward to current episode total
                    self.LunarLander.reward = self.LunarLander.reward + reward

                    # Train model if counter has been reached and the buffer has enough elements or the episode has ended
                    curr_buf_size = self.LunarLander.DoubleQLearner.replay_buffer.size
                    if ((self.LunarLander.tot_step_count % self.batch_update_steps == 0) and 
                        curr_buf_size >= self.LunarLander.min_buf_size):
                        self.LunarLander.DoubleQLearner.train_ANNs(update_var)

                    # End the episode if the epoch is done or the max number of steps have been taken
                    if (self.LunarLander.eps_step_count >= self.max_steps) or done:
                        logging.info(f"Trial {j+1}/{self.max_trials} of experiment {i+1}/{self.num_experiments} ended with a reward of: {self.LunarLander.reward}")
                        print(f"The final reward of trial {j+1}/{self.max_trials}: {self.LunarLander.reward}.")  
                        self.rewards = np.append(self.rewards,np.expand_dims(self.LunarLander.reward,axis=0),axis=0)
                        break

                    # Update target ANNs if counter hit
                    if ((self.LunarLander.tot_step_count % self.target_update_steps == 0) and
                        (curr_buf_size >= self.LunarLander.min_buf_size)):
                        self.LunarLander.DoubleQLearner.updateTargetANNs()

                # Update Epsilon
                if (curr_buf_size >= self.LunarLander.min_buf_size):
                    self.LunarLander.UpdateEpsilon()  

                if (j+1) % 10 == 0:
                    print(f"alpha_a = {self.LunarLander.DoubleQLearner.Q_a_obj.get_lr()}")
                    print(f"alpha_b = {self.LunarLander.DoubleQLearner.Q_b_obj.get_lr()}")
                    print(f"epsilon = {self.LunarLander.eps}") 

                    self.printPerformance()

                # Increment trial number
                self.trial_num = self.trial_num + 1

            # Increment experiment number
            self.experiment_num = self.experiment_num + 1