import torch
from torch import nn
from src.logger import logging
import torch.nn.functional as F

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# device = "cpu"
logging.info(f"Using {device} device for neural network training.")

torch.autograd.set_detect_anomaly(True)

'''
ANN class for double Q learner model. All neural networks will be fully connected, use
Relu as activation function, and Adam optimizer with definable learning rate. NOTE: all models will have at least 2 layers.
'''
class DoubleQLearnerANN(nn.Module):
    def __init__(self,num_layers,neurons,num_inputs=8,num_outputs=4,loss=nn.MSELoss(),learn_rate=0.0001,
                 learn_rate_decay=1.0,learn_rate_min=1e-5,steps_to_update=100,max_steps=10000):
        super(DoubleQLearnerANN,self).__init__()
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.neurons = neurons
        self.num_outputs = num_outputs
        self.loss_fcn = loss
        self.learn_rate = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.learn_rate_min = learn_rate_min
        self.train_step_count = 0
        self.running_loss = 0
        self.lin1 = nn.Linear(num_inputs,neurons[0])
        self.lin2 = nn.Linear(neurons[0],neurons[1])
        if num_layers == 3:
            self.lin3 = nn.Linear(neurons[1],neurons[2])
        self.out = nn.Linear(neurons[num_layers-1],num_outputs)
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=learn_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=steps_to_update,
                                                            gamma=learn_rate_decay)
        logging.info(f"Neural network initialized.")
        
    def __str__(self):
        n1='\n'
        return (
            f'ANN for Q-Learner with the following parameters:{n1}'
            f'# of layers = {self.num_layers}{n1}'
            f'# of neurons = {self.neurons}{n1}'
            f'# of inputs = {self.num_inputs}{n1}'
            f'# of outputs = {self.num_outputs}{n1}'
            f'loss function = {self.loss_fcn}{n1}'
            f'learn_rate = {self.learn_rate}{n1}'
            )
    
    def forward(self,x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        if self.num_layers == 3:
           x = F.relu(self.lin3(x)) 
        
        return self.out(x)

