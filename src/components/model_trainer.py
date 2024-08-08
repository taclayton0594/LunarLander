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
        self.max_steps = 1000
        self.max_trials = 500 
        self.rewards = np.empty(float)

    def set_model_hyperparameters():
        params = {
            "layer_1_neurons": [32,64,128],
            "layer_2_neurons": [32,64,128],
            "layer_3_neurons": [0,32,64],
            "alpha": [0.001,0.01,0.1],
            "alpha_decay": [0.98,0.992],
            "learn_rate": [0.0001,0.001], #will not decay NN learning rate for now to reduce DOE
            "eps_decay": [0.98,0.992], # epsilon will always start at 1
            "buf_size": [2048,8192], # minimum buffer size will always be 2*batch_size
            "batch_size": [32,64,128],
            "target_update_steps": [32,128],
            "batch_update_steps": [1,8]
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
        self.target_update_steps = curr_params["target_update_steps"]
        self.batch_update_steps = curr_params["batch_update_steps"]

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
                        Q = self.LunarLander.DoubleQLearner.Q_a(self.LunarLander.curr_state)
                    else:
                        Q = self.LunarLander.DoubleQLearner.Q_b(self.LunarLander.curr_state)

                    # Get the next action (using and Epsilon Greedy policy)
                    a = self.LunarLander.DoubleQLearner.getBestActionEps(Q)

                    # Take a step and store relevant information
                    _, _, _, reward, done = self.LunarLander.EnvironmentStep(a)

                    # Add reward to current episode total
                    self.LunarLander.reward = self.LunarLander.reward + reward

                    # Train model if counter has been reached and the buffer has enough elements or the episode has ended
                    curr_buf_size = self.LunarLander.DoubleQLearner.replay_buffer.size
                    if ((self.LunarLander.step_count % self.batch_update_steps == 0) and 
                        curr_buf_size >= self.LunarLander.min_buf_size) or done:
                        self.LunarLander.DoubleQLearner.train_ANNs(update_var)

                    # End the episode if the epoch is done or the max number of steps have been taken
                    if (self.LunarLander.step_count >= self.max_steps) or done:
                        logging.info(f"Trial {j}/{self.max_trials} of experiment {i}/{self.num_experiments} 
                                     ended with a reward of: {self.LunarLander.reward}")
                        print(f"The final reward of trial {j}/{self.max_trials}: {self.LunarLander.reward}.")  
                        self.rewards = np.append(self.rewards,self.LunarLander.reward,axis=0)
                        break

                    # Update target ANNs if counter hit
                    if self.LunarLander.step_count % self.target_update_steps == 0:
                        self.LunarLander.DoubleQLearner.updateTargetANNs()

                    # Write performance to terminal for tracking
                    print(f"The current reward is: {self.LunarLander.reward}.")    

                # Update learning rates
                self.LunarLander.UpdateAlpha()
                self.LunarLander.UpdateANNLearnRate()
                self.LunarLander.UpdateEpsilon()      