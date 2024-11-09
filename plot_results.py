import matplotlib.pyplot as plt
import os
import numpy as np
from src.utils import load_object

# Define folder location of hyperparameter tuning pickle files
pickle_data_path = os.path.join(os.getcwd(),"artifacts")

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
fs_rews = np.squeeze(fs_rews) # squeeze to avoid errors plotting

def plot_results(rew_mat,grid,grid_param,legend_str,title_str):
    num_plots,num_trials = np.shape(rew_mat)
    x = np.arange(0,num_trials)

    # First plot the results from the first success
    fs_val = fs_grid[grid_param]
    label = legend_str + str(fs_val)
    plt.plot(x,fs_rews,label=label)

    # Loop through each experiment and plot
    for i in range(num_plots):
        param_vals = grid[i]
        curr_val = param_vals[grid_param]
        label = legend_str + str(curr_val)
        plt.plot(x,rew_mat[i][:],label=label)

    # Add axis labels and legend and show plot
    plt.xlabel('Episode Number')
    plt.ylabel('100-Episode Average Reward')
    plt.title(title_str)
    plt.legend()
    plt.show()
    
def unpack_and_plot(cwd_path,info_file_name,rews_file_name,grid_param,legend_str,title_str):
    # Unpack data
    grid,_ = load_object(os.path.join(cwd_path,info_file_name))
    rews = load_object(os.path.join(cwd_path,rews_file_name))

    # Plot
    plot_results(rews,grid,grid_param,legend_str,title_str)


    return 

# Unpack stored information and plot results
unpack_and_plot(pickle_data_path,BS_info_fn,BS_rews_fn,"batch_size","Batch Size = ",ttl_str+"Batch Size")
unpack_and_plot(pickle_data_path,ED_info_fn,ED_rews_fn,"eps_decay","Decay Rate = ",ttl_str+"Epsilon Decay Rate")
unpack_and_plot(pickle_data_path,NC_info_fn,NC_rews_fn,"layer_1_neurons","Neuron Count = ",ttl_str+"Neuron Count")
unpack_and_plot(pickle_data_path,LR_info_fn,LR_rews_fn,"alpha","Learn Rate = ",ttl_str+"Learn Rate")



