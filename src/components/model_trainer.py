import os
import numpy as np
from dataclasses import dataclass
import torch
from torch import nn
from src.logger import logging
from src.utils import save_object,setup_lunar_lander_grid
from src.components.lunar_lander import LunarLander
from src.components.ann_model import device
from datetime import datetime

class RLModelTrainer:
    def __init__(self):
        self.max_steps = 1000
        self.max_trials = 2000 
        self.print_perf_cnt = 10
        self.experiment_num = 0
        self.trial_num = 0
        self.LunarLander = LunarLander(max_steps=self.max_steps)

    def set_model_hyperparameters(self):
        params = {
            "layer_1_neurons": [64], # 64
            "layer_2_neurons": [32], #32
            "layer_3_neurons": [0],
            "alpha": [0.00015], #0.00015 sucess 11-1
            "alpha_decay": [1], # 1
            "eps_decay": [0.999], # 0.999 
            "buf_size": [100000], # 100000
            "batch_size": [64], #64
            "target_update_steps": [4], # 4
            "batch_update_steps": [4] # 4
        }

        return params
    
    def get_hyperparameter_grid(self):
        self.grid,self.num_experiments = setup_lunar_lander_grid(self.set_model_hyperparameters())

        # Initialize rewards mat now that know the number of experiments
        self.rewards = np.zeros((self.num_experiments,self.max_trials),dtype=float)

        # Get date string 
        current_datetime = datetime.now()
        self.file_date_str = current_datetime.strftime("%d_%m_%Y_%H_%M")

        # Save experiment info
        file_name = f'Experiment_Info_{self.file_date_str}.pkl'
        file_path = os.path.join("artifacts",file_name)
        obj = (self.grid,self.num_experiments)
        save_object(file_path,obj)
        logging.info('Experiment hyperparameter tuning grid info has been saved.')

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
        self.LunarLander.eps = 1.0
        self.LunarLander.eps_decay = eps_decay
        self.LunarLander.buf_size = buf_size
        self.LunarLander.min_buf_size = batch_size
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

    def print_performance(self,i,j):
        # calculate 100pt moving average of rewards
        if self.trial_num <= 100:
            mov_avg = np.sum(self.rewards[self.experiment_num][0:self.trial_num-1]) / self.trial_num
        else:
            mov_avg = np.sum(self.rewards[self.experiment_num][(self.trial_num-100):self.trial_num-1]) / 100.0

        str_out = f"The 100 trial moving average is {mov_avg} at trial {self.trial_num} of experiment {self.experiment_num+1}."
        print(str_out)
        print(f"alpha_a = {self.LunarLander.DoubleQLearner.get_lr()}")
        print(f"epsilon = {self.LunarLander.eps}") 
        logging.info(str_out)

    def save_experiment_rewards(self):
        file_name = f"exp_rewards_{self.file_date_str}.pkl"
        file_path = os.path.join("artifacts",file_name)
        obj = self.rewards
        save_object(file_path,obj)

        logging.info(f'Experiment reward info has been saved.')

    def start_RL_training(self):
        # Initialize the grid of hyperparameters for each experiment
        self.get_hyperparameter_grid()

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
                    # Get Q values array for state 
                    with torch.no_grad():
                        Q = self.LunarLander.DoubleQLearner.Q_a_obj(torch.tensor(self.LunarLander.curr_state[0]).to(device))

                    # Get the next action (using and Epsilon Greedy policy)
                    a = self.LunarLander.getBestActionEps(Q)

                    # Take a step and store relevant information
                    (_, _, _, reward, done), truncated = self.LunarLander.EnvironmentStep(a)

                    # Add reward to current episode total
                    self.LunarLander.reward = self.LunarLander.reward + reward

                    # Train model if counter has been reached and the buffer has enough elements or the episode has ended
                    curr_buf_size = self.LunarLander.DoubleQLearner.replay_buffer.size
                    if ((self.LunarLander.tot_step_count % self.batch_update_steps == 0) and 
                        curr_buf_size >= self.LunarLander.min_buf_size):
                        self.LunarLander.DoubleQLearner.train_ANNs()

                    # End the episode if the epoch is done or the max number of steps have been taken
                    if (self.LunarLander.eps_step_count >= self.max_steps) or done or truncated: 
                        self.rewards[i][j] = self.LunarLander.reward

                        break

                    # Update target ANNs if counter hit
                    if ((self.LunarLander.tot_step_count % self.target_update_steps == 0) and
                        (curr_buf_size >= self.LunarLander.min_buf_size)):
                        # self.LunarLander.DoubleQLearner.updateTargetANNs()
                        self.LunarLander.DoubleQLearner.updateTargetANNs(self.LunarLander.DoubleQLearner.Q_a_obj,
                                                                            self.LunarLander.DoubleQLearner.Q_a_obj_target)

                # Update Epsilon
                if (curr_buf_size >= self.LunarLander.min_buf_size):
                    self.LunarLander.UpdateEpsilon()  

                if (j+1) % self.print_perf_cnt == 0:
                    self.print_performance(i,j)

                # Increment trial number
                self.trial_num = self.trial_num + 1

            # Increment experiment number
            self.experiment_num = self.experiment_num + 1

            # Reset seeds after each hyperparameter tuning experiment
            self.LunarLander.ReseedAll()

            # Log final performance
            logging.info(f"Experiment {i+1}/{self.num_experiments} had final average reward of: {np.sum(self.rewards[i][(self.trial_num-100):self.trial_num-1]) / 100.0}")

        # Save the results array 
        self.save_experiment_rewards()