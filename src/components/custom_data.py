import torch
from torch.utils.data import Dataset

class LunarLanderDataset(Dataset):
    def __init__(self,batch_inputs,batch_outputs,num_inputs=8,transform=None,target_transform=None):
        self.batch_inputs = batch_inputs
        self.batch_outputs = batch_outputs
        self.num_inputs = num_inputs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.batch_inputs)
    
    def __getitem__(self,idx):
        inputs = self.batch_inputs
        outputs = self.batch_outputs
        if self.transform:
            inputs = self.transform(self.batch_inputs)
        if self.target_transform:
            outputs = self.target_transform(self.batch_outputs)

        return inputs[idx,:], outputs[idx,:]
    
    def pin_memory(self):
        self.batch_inputs = self.batch_inputs.pin_memory()
        self.batch_outputs = self.batch_outputs.pin_memory()