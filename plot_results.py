import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from src.utils import load_object
from src.logger import logging

# Define folder location of hyperparameter tuning pickle files
pickle_data_path = os.path.join(os.getcwd(),"artifacts")

# Define folder location for plots
plots_data_path = os.path.join(os.getcwd(),"results_figures")

# Define string array for accessing each file
hyp_strs = ["First_Success","Eps_Decay","Neuron_Count","Learn_Rate","Batch_Size"]
num_exp = len(hyp_strs)

# Misc strings
info_str = "Experiment_Info_"
rew_str = "exp_rewards_"
ttl_str = "Hyperparameter Experiment for "

# File names for saved pickle results 
FS_info_fn = info_str+hyp_strs[0]+".pkl"
FS_rews_fn = rew_str+hyp_strs[0]+".pkl"
ED_info_fn = info_str+hyp_strs[1]+".pkl"
NC_info_fn = info_str+hyp_strs[2]+".pkl"
LR_info_fn = info_str+hyp_strs[3]+".pkl"
BS_info_fn = info_str+hyp_strs[4]+".pkl"
ED_rews_fn = rew_str+hyp_strs[1]+".pkl"
NC_rews_fn = rew_str+hyp_strs[2]+".pkl"
LR_rews_fn = rew_str+hyp_strs[3]+".pkl"
BS_rews_fn = rew_str+hyp_strs[4]+".pkl"

# Load the first success results separately
fs_grid,_ = load_object(os.path.join(pickle_data_path,FS_info_fn))
fs_grid = fs_grid[0] # Only one experiment in this file
fs_rews = load_object(os.path.join(pickle_data_path,FS_rews_fn))
fs_rews = pd.DataFrame(np.squeeze(fs_rews))

def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

def plot_results(rew_mat,grid,grid_param,legend_str,title_str,save_plots):
    num_plots,num_trials = np.shape(rew_mat)
    x = np.arange(0,num_trials)

    # First plot the results from the first success
    fs_val = fs_grid[grid_param]
    label = legend_str + str(fs_val)
    plt.figure()
    plt.plot(x,fs_rews.rolling(window=100).mean(),label=label)

    # Loop through each experiment and plot
    for i in range(num_plots):
        param_vals = grid[i]
        curr_val = param_vals[grid_param]
        label = legend_str + str(curr_val)
        curr_rews = rew_mat[i][:]
        curr_rews = pd.DataFrame(curr_rews)

        # Plot 100-pt moving average in same color
        plt.plot(x,curr_rews.rolling(window=100).mean(),label=label)

    # Add axis labels and legend
    plt.xlabel('Episode Number')
    plt.ylabel('100-Episode Average Reward')
    plt.title(title_str)
    plt.legend()

    # Show and save plot if specified
    if save_plots:
        plt.savefig(os.path.join(plots_data_path,grid_param+".png"))
        logging.info('Image saved.')
    
def unpack_and_plot(cwd_path,info_file_name,rews_file_name,grid_param,legend_str,title_str,save_plots):
    # Unpack data
    grid,_ = load_object(os.path.join(cwd_path,info_file_name))
    rews = load_object(os.path.join(cwd_path,rews_file_name))

    # Plot
    plot_results(rews,grid,grid_param,legend_str,title_str,save_plots)

    return 

# Unpack stored information and plot results
unpack_and_plot(pickle_data_path,BS_info_fn,BS_rews_fn,"batch_size","Batch Size = ",ttl_str+"Batch Size",True)
unpack_and_plot(pickle_data_path,ED_info_fn,ED_rews_fn,"eps_decay","Decay Rate = ",ttl_str+"Epsilon Decay Rate",True)
unpack_and_plot(pickle_data_path,NC_info_fn,NC_rews_fn,"layer_1_neurons","Neuron Count = ",ttl_str+"Neuron Count",True)
unpack_and_plot(pickle_data_path,LR_info_fn,LR_rews_fn,"alpha","Learn Rate = ",ttl_str+"Learn Rate",True)
plt.show()
logging.info('Displaying plots into individual windows now.')



