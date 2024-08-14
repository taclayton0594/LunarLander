from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.components.replay_buffer import ReplayBuffer
import gym
from src.components.ann_model import DoubleQLearnerANN,device
from src.components.lunar_lander import LunarLander
from src.components.model_trainer import RLModelTrainer,RLModelTrainerConfig
#from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

#@app.route('/LunarRover',methods=['GET','POST'])
#def run_lunar_rover():


if __name__=="__main__":
    #app.run(host='0.0.0.0', port=8080)      

    '''
    buf = ReplayBuffer(100,10)

    for i in range(100):
        buf.store(np.random.choice(10,(8,)))

    minibatch = buf.sample()
    print(f"minibatch[0][:]={minibatch[0][:]}")
    print(f"minibatch[9][:]={minibatch[9][:]}")

    qLearner = DoubleQLearnerANN(3,np.array([32,32,32])).to(device)
    print(qLearner)

    LunarLander = LunarLander()
    print(LunarLander)
    '''
    LunarLanderMdl = RLModelTrainer()
    LunarLanderMdl.start_RL_training()
