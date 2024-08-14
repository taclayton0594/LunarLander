import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.model_selection import ParameterGrid

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
def setup_lunar_lander_grid(params):
    try:
        # Get list of hyperparameters
        grid = ParameterGrid(params)

        # Get number of different experiments to run (i.e. number of hyperparameter combs)
        num_experiments = len(grid)
        
        logging.info(f"Hyperparameter tuning parameters have been loaded.")

        return grid, num_experiments
    
    except Exception as e:
        raise CustomException(e,sys)