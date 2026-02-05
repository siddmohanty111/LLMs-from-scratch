"""
A file containing all stochastic activation functions to be used in thesis experimentation
"""

import torch
import torch.nn as nn
import random

r = random.Random()

class StReLU(nn.Module):
    """
    Stochastic ReLU function that switches between standard ReLU and negative ReLU 
        (x, x < 0 ; 
         0, x >= 0)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, switch_proba = 0.5):
        if r.uniform(0, 1) < switch_proba:
            return nn.ReLU(x)
        else:
            return x * (x < 0)
        
