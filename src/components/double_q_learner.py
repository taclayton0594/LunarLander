import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.logger import logging
from src.components.ann_model import DoubleQLearnerANN
from src.components.replay_buffer import ReplayBuffer
from src.components.custom_data import LunarLanderDataset
from src.components.ann_model import device

'''
This main class for the RL Double Q-Learner. On initialization, 2 neural networks will be created, 1 for evaluation and the other
as a target net that will be frozen for a tunable amount of time. This freezing of target nets allows for more stable training using 
an off-policy methodology.
'''
class DoubleQLearner():
    def __init__(self,num_layers,neurons,num_inputs=8,loss=nn.MSELoss(),num_actions=4,buf_size=50000,batch_size=32,
                 alpha=0.0001,alpha_decay=1.0,alpha_min=1e-6,gamma=0.99,tau=0.001):
        self.Q_a_obj = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min).to(device)
        self.Q_a_obj_target = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min).to(device)
        self.no_gradient_model() # tell Pytorch not to compute gradients for target model

        # initialize weights
        self.Q_a_obj.apply(self.initialize_weights)\
        self.Q_a_obj_target.apply(self.initialize_weights)
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buf_size,batch_size)
        self.tau = tau
        logging.info(f"Double Q-Learner has been created.")

    def __str__(self):
        n1='\n'
        return (
            f'Double Q-Learner with the following parameters:{n1}'
            f'gamma = {self.gamma}{n1}'
            f'batch_size = {self.batch_size}{n1}'
            )
    
    def initialize_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def no_gradient_model(self):
        for p in self.Q_a_obj_target.parameters():
            p.requires_grad_(False)
    
    def updateTargetANNs(self,main,target,copy_weights=False):
        if copy_weights:
            self.Q_a_obj_target.load_state_dict(self.Q_a_obj.state_dict())
            logging.info("Target networks have been updated.")
        else:
            for target_weights, main_weights in zip(target.parameters(),main.parameters()):
                target_weights.data.copy_(self.tau*main_weights + (1-self.tau)*target_weights)

    def get_targets(self,next_states,rewards,done_bools):
        try:
            # Get the model outputs for each batch sample
            with torch.no_grad(): # save memory usage and time by not saving gradient info
                Q_2_preds = self.Q_a_obj_target(next_states).detach().to(device) # Predictions for state 2 (next state)

            # Get best actions 
            a_2 = torch.argmax(Q_2_preds,dim=1).to(device)

            # Calculate targets for each batch sample
            targets = rewards + self.gamma * Q_2_preds[:].gather(1,a_2.view(self.batch_size,1)).view(self.batch_size).detach() * (1 - done_bools.int())

            return targets
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_ANNs(self,epochs=1):
        try:
            # Get batch data for training
            states,next_states,actions,rewards,done_bools = self.replay_buffer.sample()

            # Get target matrix
            targets = self.get_targets(next_states,rewards,done_bools)

            # Convert data to Torch Dataset
            train_data = LunarLanderDataset(states,targets)

            # Train ANN
            self.train_q_learner(actions,train_data,self.batch_size,epochs)
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_lr(self):
        return self.Q_a_obj.optimizer.param_groups[0]['lr']
        
    def train_q_learner(self,actions,batch_data,batch_size,epochs=1,output_step_count=5000,clip_gradients=False):
        try:
            # Set to training mode
            self.Q_a_obj.train() # not necessary but good practice when adding batch norm and other layers

            # Load data into custom DataLoader
            batch_dataloader = DataLoader(batch_data,batch_size=batch_size,num_workers=0)

            # Increase count of train steps and perform model training
            self.Q_a_obj.train_step_count += 1
            for _ in range(epochs):
                for batch,(X,y) in enumerate(batch_dataloader):
                    X = X.to(device)
                    y = y.to(device)
                    # Get predictions
                    preds = self.Q_a_obj(X).gather(1,actions.view(batch_size,1)).view(batch_size)

                    # Compute prediction error
                    loss = self.Q_a_obj.loss_fcn(preds,y)
                    self.Q_a_obj.running_loss += loss.detach()

                    # Clear gradients before each step
                    self.Q_a_obj.optimizer.zero_grad()

                    # Backpropagation
                    loss.backward()

                    # Clip the gradients
                    if clip_gradients:
                        for p in self.Q_a_obj.parameters():
                            p.grad.data.clamp_(-1,1)

                    self.Q_a_obj.optimizer.step()     

                if self.get_lr() > self.Q_a_obj.learn_rate_min:
                    self.Q_a_obj.scheduler.step()

            # turn off training mode - not necessary but good practice
            # Required for batch normalization and other layers with different behavior during training and eval
            self.Q_a_obj.eval()

            if (self.Q_a_obj.train_step_count % output_step_count == 0):
                avg_err = self.Q_a_obj.running_loss / self.Q_a_obj.train_step_count / epochs / batch_size
                print(f"Average batch error is = {avg_err} on train step #{self.Q_a_obj.train_step_count}.")

        except Exception as e:
            raise CustomException(e,sys)

        