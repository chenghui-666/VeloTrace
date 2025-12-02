import numpy as np
from torch import nn
import torch
from torchdiffeq import odeint
import pandas as pd
from .utils import set_seed

set_seed(4)

# ---- Model Definition: ODEFunc, ODEBlock, LinearWeightedMSE----
class ODEFunc(nn.Module):
    def __init__(self, input_dim):
        super(ODEFunc, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1000)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, input_dim)

    def forward(self, t, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y) + y
        y = self.relu(y)
        return self.linear3(y)
       
options = {
    'step_size': 0.001
}

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
    
    def forward(self, y0, t):
        out = odeint(self.odefunc, y0, t, rtol=1e-4, atol=1e-6)
        return out
    
class LinearWeightedMSE(nn.Module):
    def __init__(self, min_weight=0.5):
        super().__init__()
        self.min_weight = min_weight  
    
    def forward(self, input, target):
        squared_diff = torch.log1p((input - target) ** 2 + 1).sum(dim=-1)
        
        batch_size = input.size(0)
        # 线性从1.0衰减到min_weight
        weights = torch.linspace(1.0, self.min_weight, steps=batch_size, device=input.device)
        
        weighted_loss = weights * squared_diff
        
        return weighted_loss.mean()
