import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.logger import logging
from torch.utils.data import DataLoader
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
        super().__init__()
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
    
    def initialize_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self,x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        if self.num_layers == 3:
           x = F.relu(self.lin3(x)) 
        
        return x
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
        
    def train_q_learner(self,batch_data,batch_size,epochs=1):
        try:
            batch_dataloader = DataLoader(batch_data,batch_size=batch_size)

            # Increase count of train steps and perform model training
            self.train_step_count += 1
            for _ in range(epochs):
                for batch,(X,y) in enumerate(batch_dataloader):
                    # Compute prediction error
                    loss = self.loss_fcn()(X,y)
                    self.running_loss += loss

                    # Clear gradients before each step
                    self.optimizer.zero_grad()

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()                 

                if self.get_lr() > self.learn_rate_min:
                    self.scheduler.step()

            # turn off training mode
            self.eval()

            if (self.train_step_count % 3000 == 0):
                avg_err = self.running_loss / self.train_step_count / epochs / batch_size
                # print(f"X={X}")
                # print(f"y={y}")
                print(f"Average batch error is = {avg_err} on train step #{self.train_step_count}.")

        except Exception as e:
            raise CustomException(e,sys)

