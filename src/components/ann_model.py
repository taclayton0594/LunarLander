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
logging.info(f"Using {device} device for neural network training.")

'''
ANN class for double Q learner model. All neural networks will be fully connected, use
Relu as activation function, and Adam optimizer with definable learning rate. NOTE: all models will have at least 2 layers.
'''
class DoubleQLearnerANN(nn.Module):
    def __init__(self,num_layers,neurons,num_inputs=8,num_outputs=4,loss=nn.MSELoss(),learn_rate=0.0001):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.neurons = neurons
        self.num_outputs = num_outputs
        self.loss_fcn = loss
        self.learn_rate = learn_rate
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
        return self.ANN_relu(DataLoader(x))
        
    def train_q_learner(self,batch_data,epochs=1):
        try:
            batch_dataloader = DataLoader(batch_data)

            # Set the module into training mode
            self.ANN_relu.train()
            for _ in range(epochs):
                for batch, (X, y) in enumerate(batch_dataloader):
                    X, y = X.to(device), y.to(device)

                    # Compute prediction error
                    pred = self.ANN_relu(X)
                    loss = self.loss_fcn()(pred,y)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        except Exception as e:
            raise CustomException(e,sys)

