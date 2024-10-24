import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.logger import logging
from src.components.ann_model import DoubleQLearnerANN
from src.components.replay_buffer import ReplayBuffer
from src.components.custom_data import LunarLanderDataset

'''
This main class for the RL Double Q-Learner. On initialization, 4 neural networks will be created, 2 for 2 nets used in double Q
and then another 2 for target nets for each of the previous nets that will be frozen for a tunable amount of time.
This freezing of target nets allows for more stable training.
'''
class DoubleQLearner():
    def __init__(self,num_layers,neurons,num_inputs=8,loss=nn.MSELoss(),num_actions=4,buf_size=50000,batch_size=32,
                 alpha=0.0001,alpha_decay=1.0,alpha_min=1e-6,gamma=0.99,tau=0.01):
        self.Q_a_obj = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min)
        self.Q_a_obj_target = DoubleQLearnerANN(num_layers,neurons,num_inputs,num_actions,loss,alpha,alpha_decay,alpha_min)
        
        # # initialize weights
        # self.Q_a_obj.apply(self.initialize_weights)
        # # self.updateTargetANNs()
        # self.Q_a_obj_target.apply(self.initialize_weights)
        
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
    
    def updateTargetANNs(self,main,target,copy_weights=False):
        if copy_weights:
            self.Q_a_obj_target.load_state_dict(self.Q_a_obj.state_dict())
            logging.info("Target networks have been updated.")
        else:
            for target_weights, main_weights in zip(target.parameters(),main.parameters()):
                target_weights.data.copy_(self.tau*main_weights + (1-self.tau)*target_weights)

    def get_targets(self,next_states,rewards,done_bools):
        try:
            # Get main and target nets
            Q_2 = self.Q_a_obj_target

            # Get the model outputs for each batch sample
            s_2_mat = next_states
            Q_2_preds = Q_2(s_2_mat) # Predictions for state 2 (next state)
            # Q_2_next = Q_1(s_2_mat).clone()
            # Q_2_next_targ = Q_2(s_2_mat).clone()

            # Get best actions 
            a_2 = torch.argmax(Q_2_preds,dim=1)
            # a_2 = torch.argmax(Q_2_next,dim=1)
            # a_2_targ = torch.argmax(Q_2_next_targ,dim=1)

            # Predictions using state 1 (previous state)           
            # preds = Q_1_preds

            # Get min Q value for clipped double q learning
            # Q_next = Q_2_next.gather(1,a_2.view(self.batch_size,1)).view(self.batch_size)
            # Q_next_targ = Q_2_next_targ.gather(1,a_2_targ.view(self.batch_size,1)).view(self.batch_size)
            # Q_min = torch.min(Q_next,Q_next_targ)

            # Extract other useful info from batch data
            rews = rewards.view(self.batch_size)
            done = done_bools.int().view(self.batch_size)

            # Calculate targets for each batch sample
            targets = rews + self.gamma * Q_2_preds[:].gather(1,a_2.view(self.batch_size,1)).view(self.batch_size) * (1 - done)
            # targets = rews + self.gamma * Q_min * (1 - done)

            return targets
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_ANNs(self,epochs=1):
        try:
            # Get batch data for training
            states,next_states,actions,rewards,done_bools = self.replay_buffer.sample()

            # Set to training mode
            self.Q_a_obj.train() # not necessary but good practice when adding batch norm and other layers

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
        
    def train_q_learner(self,actions,batch_data,batch_size,epochs=1):
        try:
            batch_dataloader = DataLoader(batch_data,batch_size=batch_size)

            # Increase count of train steps and perform model training
            self.Q_a_obj.train_step_count += 1
            for _ in range(epochs):
                for batch,(X,y) in enumerate(batch_dataloader):
                    # Get predictions
                    preds = self.Q_a_obj(X).gather(1,actions.view(batch_size,1)).view(batch_size)

                    # Compute prediction error
                    loss = self.Q_a_obj.loss_fcn(preds,y)
                    self.Q_a_obj.running_loss += loss

                    # Clear gradients before each step
                    self.Q_a_obj.optimizer.zero_grad()

                    # Backpropagation
                    loss.backward()
                    # loss.backward(retain_graph=True)
                    self.Q_a_obj.optimizer.step()     

                if self.get_lr() > self.Q_a_obj.learn_rate_min:
                    self.Q_a_obj.scheduler.step()

            # turn off training mode - not necessary
            self.Q_a_obj.eval()

            if (self.Q_a_obj.train_step_count % 5000 == 0):
                avg_err = self.Q_a_obj.running_loss / self.Q_a_obj.train_step_count / epochs / batch_size
                print(f"X={X}")
                print(f"y={y}")
                print(f"Average batch error is = {avg_err} on train step #{self.Q_a_obj.train_step_count}.")

        except Exception as e:
            raise CustomException(e,sys)

        