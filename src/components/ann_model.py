import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.logger import logging
from torch.utils.data import DataLoader

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
                 learn_rate_decay=1.0,learn_rate_min=1e-6,steps_to_update=10000):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.neurons = neurons
        self.num_outputs = num_outputs
        self.loss_fcn = loss
        self.learn_rate = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.learn_rate_min = learn_rate_min
        try:
            # Create model structure and add input layer
            layers = []
            layers.append(nn.Linear(num_inputs,neurons[0]))
            layers.append(nn.ReLU())

            # Add additional layers except last
            for i in range(num_layers-1):
                layers.append(nn.Linear(neurons[i],neurons[i+1]))
                layers.append(nn.ReLU())

            # Add output layer
            layers.append(nn.Linear(neurons[i+1],num_outputs))
            layers.append(nn.Softmax(dim=-1))

            # Create sequential model
            self.ANN_relu = nn.Sequential(*layers).to(device)
                
        except Exception as e:
            raise CustomException(e,sys)
        
        self.optimizer = torch.optim.Adam(self.ANN_relu.parameters(),lr=learn_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[steps_to_update], 
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
        out = self.ANN_relu(DataLoader(x))
        return out
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
        
    def train_q_learner(self,batch_data,batch_size,epochs=1):
        try:
            batch_dataloader = DataLoader(batch_data,batch_size=batch_size)

            # Set the module into training mode
            self.ANN_relu.train()
            for _ in range(epochs):
                for batch, (X, y) in enumerate(batch_dataloader):
                    # Compute prediction error
                    pred = self.ANN_relu(X)
                    loss = self.loss_fcn()(pred,y)
                    #print(f"Loss pre = {loss}")

                    # Backpropagation
                    loss.backward(retain_graph=True)
                    self.optimizer.step()                    
                    self.optimizer.zero_grad()

                    if self.get_lr() > self.learn_rate_min:
                        self.scheduler.step()
                    '''
                    pred = self.ANN_relu(X)
                    loss = self.loss_fcn()(pred,y)
                    print(f"Loss post = {loss}")
                    '''

        except Exception as e:
            raise CustomException(e,sys)

